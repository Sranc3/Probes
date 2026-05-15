"""Lightweight probe trainers + selective-prediction metrics.

Three probe families:

- ``LogRegProbe``       : sklearn-style logistic regression on hidden state.
- ``RidgeProbe``        : sklearn-style ridge regression for SEPs / ARD.
- ``MLPProbe``          : 2-layer MLP (PyTorch) for DCP and ARD.

We keep the API deliberately uniform: every probe exposes
``fit(X, y) -> self`` and ``predict_score(X) -> np.ndarray``. For classifiers
``predict_score`` returns calibrated probabilities; for regressors it returns
the raw prediction (caller is responsible for sign-flipping when used as a
confidence score).

We also provide ``GroupKFold`` runners so probes are always evaluated with
question-level splits to avoid same-question leakage between train/test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def sanitize(X: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with zero (post-standardisation this is the safe default)."""
    arr = np.asarray(X, dtype=np.float64)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


class LogRegProbe:
    """Logistic regression on standardised hidden-state features (SEPs-style)."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 0):
        self.scaler: StandardScaler | None = None
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogRegProbe":
        Xc = sanitize(X)
        self.scaler = StandardScaler().fit(Xc)
        Xs = sanitize(self.scaler.transform(Xc))
        self.model.fit(Xs, y.astype(int))
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        assert self.scaler is not None
        Xs = sanitize(self.scaler.transform(sanitize(X)))
        return self.model.predict_proba(Xs)[:, 1]


class RidgeProbe:
    """Ridge regression baseline (used for SEPs target and ARD per-dim)."""

    def __init__(self, alpha: float = 1.0, seed: int = 0):
        self.scaler: StandardScaler | None = None
        self.model = Ridge(alpha=alpha, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeProbe":
        Xc = sanitize(X)
        self.scaler = StandardScaler().fit(Xc)
        Xs = sanitize(self.scaler.transform(Xc))
        self.model.fit(Xs, y)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        assert self.scaler is not None
        Xs = sanitize(self.scaler.transform(sanitize(X)))
        return self.model.predict(Xs)


class MLPProbe:
    """2-layer MLP, PyTorch, supports binary classification or regression."""

    def __init__(
        self,
        hidden: int = 128,
        out_dim: int = 1,
        loss: str = "bce",
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
        device: str | None = None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for MLPProbe")
        self.hidden = hidden
        self.out_dim = out_dim
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler: StandardScaler | None = None
        self.model: nn.Module | None = None

    def _build(self, in_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden, self.hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden // 2, self.out_dim),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPProbe":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        Xc = sanitize(X)
        self.scaler = StandardScaler().fit(Xc)
        Xs = sanitize(self.scaler.transform(Xc)).astype(np.float32)
        if y.ndim == 1:
            y_arr = y.astype(np.float32).reshape(-1, 1)
        else:
            y_arr = y.astype(np.float32)
        if y_arr.shape[1] != self.out_dim:
            raise ValueError(f"y has {y_arr.shape[1]} cols but probe expects {self.out_dim}")
        self.model = self._build(Xs.shape[1]).to(self.device)
        criterion = nn.BCEWithLogitsLoss() if self.loss == "bce" else nn.MSELoss()
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n = Xs.shape[0]
        Xt = torch.from_numpy(Xs).to(self.device)
        yt = torch.from_numpy(y_arr).to(self.device)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                logits = self.model(Xt[idx])
                loss = criterion(logits, yt[idx])
                optim.zero_grad()
                loss.backward()
                optim.step()
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        assert self.scaler is not None and self.model is not None
        self.model.eval()
        Xs = sanitize(self.scaler.transform(sanitize(X))).astype(np.float32)
        with torch.no_grad():
            out = self.model(torch.from_numpy(Xs).to(self.device))
            if self.loss == "bce":
                out = torch.sigmoid(out)
            arr = out.cpu().numpy()
        if arr.shape[1] == 1:
            return arr[:, 0]
        return arr


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------


@dataclass
class CVPrediction:
    """OOF (out-of-fold) prediction container."""

    y_true: np.ndarray
    y_score: np.ndarray  # probability of correct (or predicted target)
    fold_id: np.ndarray
    extra: dict[str, np.ndarray]


def group_kfold_predict(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    probe_factory,
    n_splits: int = 5,
    seed: int = 0,
) -> CVPrediction:
    """Run GroupKFold and return out-of-fold predictions for the *full* dataset."""
    kf = GroupKFold(n_splits=n_splits)
    y_score = np.zeros(X.shape[0], dtype=np.float64)
    fold_id = np.full(X.shape[0], -1, dtype=np.int32)
    for fid, (tr, te) in enumerate(kf.split(X, y, groups=groups)):
        probe = probe_factory()
        probe.fit(X[tr], y[tr])
        scores = probe.predict_score(X[te])
        if scores.ndim > 1:
            raise ValueError("group_kfold_predict expects 1-D scores; use raw_predict for vectors")
        y_score[te] = scores
        fold_id[te] = fid
    return CVPrediction(y_true=y.copy(), y_score=y_score, fold_id=fold_id, extra={})


def group_kfold_raw_predict(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    probe_factory,
    n_splits: int = 5,
) -> np.ndarray:
    """Return OOF predictions when the probe outputs vectors (e.g., ARD)."""
    kf = GroupKFold(n_splits=n_splits)
    y_pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in kf.split(X, y, groups=groups):
        probe = probe_factory()
        probe.fit(X[tr], y[tr])
        out = probe.predict_score(X[te])
        if out.ndim == 1:
            y_pred[te] = out
        else:
            y_pred[te] = out
    return y_pred


# ---------------------------------------------------------------------------
# Selective-prediction metrics (small re-implementation to avoid cross-deps)
# ---------------------------------------------------------------------------


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC with proper tie handling.

    NOTE: an earlier hand-rolled implementation used ``argsort().argsort()``
    which gives strict ordering and therefore systematically inflated AUROC
    when many scores are tied (e.g. an MLP saturating to 1.0 on OOD inputs).
    We now defer to sklearn's ``roc_auc_score`` which uses average ranks
    (Mann-Whitney U with tie correction), matching standard practice.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0:
        return float("nan")
    pos_mask = y_true > 0.5
    if pos_mask.all() or (~pos_mask).all():
        return float("nan")
    if not np.isfinite(y_score).all():
        y_score = np.nan_to_num(y_score, nan=0.0, posinf=0.0, neginf=0.0)
    return float(roc_auc_score(pos_mask.astype(int), y_score))


def selective_accuracy_at_coverage(y_true: np.ndarray, y_score: np.ndarray, coverage: float) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = max(1, int(round(coverage * n)))
    order = np.argsort(-y_score)
    top = order[:k]
    return float(np.mean(y_true[top]))


def aurc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    n = len(y_true)
    order = np.argsort(-y_score)
    sorted_correct = y_true[order]
    cum = np.cumsum(sorted_correct)
    coverage = np.arange(1, n + 1) / n
    accuracy = cum / np.arange(1, n + 1)
    risk = 1.0 - accuracy
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(risk, coverage))


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(np.mean((y_true - y_score) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        in_bin = (y_score >= lo) & (y_score < hi if i < n_bins - 1 else y_score <= hi)
        if not in_bin.any():
            continue
        confidence = float(np.mean(y_score[in_bin]))
        accuracy = float(np.mean(y_true[in_bin]))
        ece += (in_bin.sum() / n) * abs(confidence - accuracy)
    return float(ece)


def summarize_metrics(y_true: np.ndarray, y_score: np.ndarray, calibrated: bool) -> dict[str, float]:
    out = {
        "auroc": safe_auroc(y_true, y_score),
        "aurc": aurc(y_true, y_score),
        "sel_acc@0.25": selective_accuracy_at_coverage(y_true, y_score, 0.25),
        "sel_acc@0.50": selective_accuracy_at_coverage(y_true, y_score, 0.50),
        "sel_acc@0.75": selective_accuracy_at_coverage(y_true, y_score, 0.75),
    }
    if calibrated:
        out["brier"] = brier_score(y_true, y_score)
        out["ece"] = expected_calibration_error(y_true, y_score)
    else:
        out["brier"] = float("nan")
        out["ece"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Extended selective-prediction metrics (added 2026-05-14)
# ---------------------------------------------------------------------------


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the Precision-Recall curve (a.k.a. average precision).

    More sensitive than AUROC under class imbalance (e.g. NQ-Open where
    base_acc < 0.5, so the *correct* class is the minority and AUROC can mask
    differences in precision at high-confidence operating points).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0:
        return float("nan")
    pos_mask = y_true > 0.5
    if pos_mask.all() or (~pos_mask).all():
        return float("nan")
    if not np.isfinite(y_score).all():
        y_score = np.nan_to_num(y_score, nan=0.0, posinf=0.0, neginf=0.0)
    return float(average_precision_score(pos_mask.astype(int), y_score))


def risk_coverage_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (coverage, risk) arrays describing the full RC curve.

    Coverage[i] = (i+1)/n, Risk[i] = 1 - cumulative_accuracy of top-(i+1) most
    confident predictions.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(-y_score)
    sorted_correct = y_true[order]
    cum = np.cumsum(sorted_correct)
    coverage = np.arange(1, n + 1) / n
    accuracy = cum / np.arange(1, n + 1)
    risk = 1.0 - accuracy
    return coverage, risk


def auarc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the Accuracy-Rejection (i.e. accuracy-coverage) curve.

    Equivalent to ``1 - AURC`` numerically (since accuracy = 1 - risk integrated
    over the same x-axis), but reported separately because the literature cites
    both. Higher is better.
    """
    cov, risk = risk_coverage_curve(y_true, y_score)
    if cov.size == 0:
        return float("nan")
    accuracy = 1.0 - risk
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(accuracy, cov))


def coverage_at_risk(y_true: np.ndarray, y_score: np.ndarray, max_risk: float) -> float:
    """Largest coverage we can attain while keeping risk <= ``max_risk``.

    Useful operating-point metric: "if we want error rate <= X%, what fraction
    of questions can we still answer?"  Returns 0.0 if no operating point
    satisfies the constraint.
    """
    cov, risk = risk_coverage_curve(y_true, y_score)
    if cov.size == 0:
        return float("nan")
    valid = risk <= max_risk
    if not valid.any():
        return 0.0
    return float(cov[valid].max())


def reliability_diagram_data(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_confidence, bin_accuracy, bin_count) for a reliability plot."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    conf = np.zeros(n_bins)
    acc = np.zeros(n_bins)
    cnt = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_score >= lo) & (y_score < hi if i < n_bins - 1 else y_score <= hi)
        if mask.any():
            conf[i] = float(np.mean(y_score[mask]))
            acc[i] = float(np.mean(y_true[mask]))
            cnt[i] = int(mask.sum())
    return conf, acc, cnt


def brier_decomposition(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """Murphy (1973) decomposition: Brier = Reliability - Resolution + Uncertainty.

    - Reliability ↓ : how far bin confidences are from bin accuracies (calibration)
    - Resolution ↑ : how far bin accuracies are from the marginal base rate
                     (discrimination — useful spread)
    - Uncertainty :  intrinsic difficulty of the dataset (= base_acc * (1 - base_acc))
    """
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    if n == 0:
        return {"reliability": float("nan"), "resolution": float("nan"),
                "uncertainty": float("nan"), "brier": float("nan")}
    base_rate = float(np.mean(y_true))
    uncertainty = base_rate * (1.0 - base_rate)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_score >= lo) & (y_score < hi if i < n_bins - 1 else y_score <= hi)
        nk = int(mask.sum())
        if nk == 0:
            continue
        conf_k = float(np.mean(y_score[mask]))
        acc_k = float(np.mean(y_true[mask]))
        reliability += (nk / n) * (conf_k - acc_k) ** 2
        resolution += (nk / n) * (acc_k - base_rate) ** 2

    brier = float(np.mean((y_true - y_score) ** 2))
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier": brier,
    }


def extended_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    calibrated: bool,
    risk_thresholds: tuple[float, ...] = (0.05, 0.10, 0.20),
    sel_acc_coverages: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> dict[str, float]:
    """Comprehensive selective-prediction metric pack used by the extended
    dashboard. Includes everything in ``summarize_metrics`` plus AUPRC, AUARC,
    Coverage@Risk thresholds, and (when calibrated) Brier decomposition."""
    out: dict[str, float] = {
        "auroc": safe_auroc(y_true, y_score),
        "auprc": auprc(y_true, y_score),
        "aurc": aurc(y_true, y_score),
        "auarc": auarc(y_true, y_score),
    }
    for c in sel_acc_coverages:
        out[f"sel_acc@{c:.2f}"] = selective_accuracy_at_coverage(y_true, y_score, c)
    for r in risk_thresholds:
        out[f"cov@risk<={r:.2f}"] = coverage_at_risk(y_true, y_score, r)
    if calibrated:
        decomp = brier_decomposition(y_true, y_score)
        out["brier"] = decomp["brier"]
        out["brier_reliability"] = decomp["reliability"]
        out["brier_resolution"] = decomp["resolution"]
        out["brier_uncertainty"] = decomp["uncertainty"]
        out["ece"] = expected_calibration_error(y_true, y_score)
    else:
        for k in ("brier", "brier_reliability", "brier_resolution",
                  "brier_uncertainty", "ece"):
            out[k] = float("nan")
    return out


def negate_for_confidence(score: np.ndarray) -> np.ndarray:
    """SEPs predicts entropy; we want a confidence score so we flip the sign."""
    return -score


def minmax_normalize(score: np.ndarray) -> np.ndarray:
    lo = float(np.min(score))
    hi = float(np.max(score))
    if math.isclose(hi, lo):
        return np.zeros_like(score)
    return (score - lo) / (hi - lo)
