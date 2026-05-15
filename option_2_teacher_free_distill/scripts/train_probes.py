"""Train SEPs / DCP / ARD probes across all extracted layers and save OOF preds.

For every layer in the cached hidden state file we train three probes via
GroupKFold (5 folds, grouped on question_id):

- SEPs       : Ridge regression -> semantic_entropy_weighted_set
               (confidence = -prediction)
- SEPs-cls   : Logistic regression directly on hidden state -> P(correct)
               (this is the *strong* SEPs variant used in Kossen et al. 2024)
- DCP-MLP    : 2-layer MLP -> P(correct)
- ARD-Ridge  : Ridge regression -> 7-dim teacher anchor; then logistic regression
               on the predicted vector -> P(correct)
- ARD-MLP    : 2-layer MLP regressing teacher anchor; then logistic regression
               head on top.

We also train ``cross_seed`` variants: train on rows with seed=42 only, evaluate
on rows with seed=43, and vice versa, then average.

Outputs:
- ``runs/probe_predictions.csv`` : long format with columns
  ``probe, layer, regime, question_id, seed, y_true, y_score``
- ``runs/probe_predictions.npz`` : same but per (probe, layer, regime) dense
  arrays for quick loading by evaluate_probes.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR / "shared"))

from data_utils import (  # noqa: E402
    TEACHER_ANCHOR_COLUMNS,
    attach_targets,
    expand_to_rows,
    load_anchor_rows,
    question_index_map,
    select_sample0_rows,
)
from probe_utils import (  # noqa: E402
    LogRegProbe,
    MLPProbe,
    RidgeProbe,
    group_kfold_predict,
    group_kfold_raw_predict,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--questions",
        default="/zhutingqi/song/Plan_gpt55/cross_model_anchor/runs/run_20260513_012413_gptoss_anchor_triviaqa_full500_k4_tok256/qwen_candidate_anchor_rows_final_only.csv",
    )
    p.add_argument(
        "--hidden-states",
        default=str(THIS_DIR / "runs" / "hidden_states.npz"),
    )
    p.add_argument(
        "--out-dir",
        default=str(THIS_DIR / "runs"),
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mlp-hidden", type=int, default=128)
    p.add_argument("--mlp-epochs", type=int, default=200)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--logreg-c", type=float, default=1.0)
    p.add_argument(
        "--max-questions", type=int, default=None, help="Truncate for smoke run"
    )
    return p.parse_args()


def stack_anchor_oof(
    X: np.ndarray,
    y_anchor: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Train one ridge per anchor dim under the same fold structure and return predicted matrix."""
    kf = GroupKFold(n_splits=args.n_splits)
    n, d = y_anchor.shape
    pred = np.zeros_like(y_anchor, dtype=np.float64)
    for tr, te in kf.split(X, y_anchor[:, 0], groups=groups):
        for j in range(d):
            probe = RidgeProbe(alpha=args.ridge_alpha, seed=args.seed)
            probe.fit(X[tr], y_anchor[tr, j])
            pred[te, j] = probe.predict_score(X[te])
    return pred


def stack_anchor_oof_mlp(
    X: np.ndarray,
    y_anchor: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Same as ``stack_anchor_oof`` but uses a single MLP with 7 outputs (MSE)."""
    kf = GroupKFold(n_splits=args.n_splits)
    pred = np.zeros_like(y_anchor, dtype=np.float64)
    for tr, te in kf.split(X, y_anchor[:, 0], groups=groups):
        probe = MLPProbe(
            hidden=args.mlp_hidden,
            out_dim=y_anchor.shape[1],
            loss="mse",
            epochs=args.mlp_epochs,
            seed=args.seed,
        )
        probe.fit(X[tr], y_anchor[tr])
        pred[te] = probe.predict_score(X[te])
    return pred


def fit_logreg_head(
    pred_anchor: np.ndarray, y_corr: np.ndarray, groups: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    """Fit a 2nd-stage logistic regression on predicted anchor features."""
    kf = GroupKFold(n_splits=args.n_splits)
    out = np.zeros(pred_anchor.shape[0], dtype=np.float64)
    for tr, te in kf.split(pred_anchor, y_corr, groups=groups):
        scaler = StandardScaler().fit(pred_anchor[tr])
        Xs_tr = scaler.transform(pred_anchor[tr])
        Xs_te = scaler.transform(pred_anchor[te])
        head = LogisticRegression(C=args.logreg_c, max_iter=1000, random_state=args.seed)
        head.fit(Xs_tr, y_corr[tr].astype(int))
        out[te] = head.predict_proba(Xs_te)[:, 1]
    return out


def cross_seed_predict(
    X: np.ndarray,
    y: np.ndarray,
    seeds: np.ndarray,
    probe_factory,
) -> np.ndarray:
    """Train on seed=42 -> predict seed=43 and vice versa, return concatenated OOF."""
    out = np.zeros_like(y, dtype=np.float64)
    for held in (42, 43):
        tr = seeds != held
        te = seeds == held
        probe = probe_factory()
        probe.fit(X[tr], y[tr])
        score = probe.predict_score(X[te])
        if score.ndim > 1:
            raise ValueError("cross_seed_predict expects 1-D scores")
        out[te] = score
    return out


def cross_seed_anchor_then_head(
    X: np.ndarray,
    y_anchor: np.ndarray,
    y_corr: np.ndarray,
    seeds: np.ndarray,
    args: argparse.Namespace,
    backbone: str,
) -> np.ndarray:
    out = np.zeros_like(y_corr, dtype=np.float64)
    for held in (42, 43):
        tr = seeds != held
        te = seeds == held
        if backbone == "ridge":
            preds_tr = np.zeros_like(y_anchor[tr], dtype=np.float64)
            preds_te = np.zeros_like(y_anchor[te], dtype=np.float64)
            for j in range(y_anchor.shape[1]):
                probe = RidgeProbe(alpha=args.ridge_alpha, seed=args.seed)
                probe.fit(X[tr], y_anchor[tr, j])
                preds_tr[:, j] = probe.predict_score(X[tr])
                preds_te[:, j] = probe.predict_score(X[te])
        else:
            probe = MLPProbe(
                hidden=args.mlp_hidden,
                out_dim=y_anchor.shape[1],
                loss="mse",
                epochs=args.mlp_epochs,
                seed=args.seed,
            )
            probe.fit(X[tr], y_anchor[tr])
            preds_tr = probe.predict_score(X[tr])
            preds_te = probe.predict_score(X[te])
        scaler = StandardScaler().fit(preds_tr)
        Xs_tr = scaler.transform(preds_tr)
        Xs_te = scaler.transform(preds_te)
        head = LogisticRegression(C=args.logreg_c, max_iter=1000, random_state=args.seed)
        head.fit(Xs_tr, y_corr[tr].astype(int))
        out[te] = head.predict_proba(Xs_te)[:, 1]
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] anchor rows: {args.questions}")
    df = load_anchor_rows(args.questions)
    sample0 = select_sample0_rows(df)

    if args.max_questions is not None:
        keep_qids = set(sample0["question_id"].drop_duplicates().head(args.max_questions))
        sample0 = sample0[sample0["question_id"].isin(keep_qids)].reset_index(drop=True)

    targets = attach_targets(sample0)
    qid_to_idx = question_index_map(sample0)

    print(f"[load] hidden states: {args.hidden_states}")
    blob = np.load(args.hidden_states, allow_pickle=True)
    qids_cached = list(blob["question_ids"])
    h_all = blob["hidden_prompt_last"]  # (Q, L, H)
    layers = list(blob["layer_indices"])
    print(f"[load] cached questions: {len(qids_cached)} layers: {layers}")

    cache_idx = {qid: i for i, qid in enumerate(qids_cached)}
    if not set(qid_to_idx).issubset(cache_idx):
        missing = sorted(set(qid_to_idx) - set(cache_idx))
        raise KeyError(f"Hidden state cache missing questions (showing 5): {missing[:5]}")

    qid_order = list(qid_to_idx.keys())
    h_per_question = np.stack([h_all[cache_idx[q]] for q in qid_order], axis=0)
    qid_to_local = {q: i for i, q in enumerate(qid_order)}

    y_corr = targets["y_corr"]
    y_se = targets["y_se"]
    y_anchor = targets["y_anchor"]
    seeds = sample0["seed"].astype(int).to_numpy()
    qids = sample0["question_id"].astype(str).to_numpy()
    groups = np.array([qid_to_local[q] for q in qids], dtype=np.int32)
    rows_by_question = sample0["question_id"].map(qid_to_local).to_numpy()
    print(f"[shape] rows={len(y_corr)} positives={int(y_corr.sum())} questions={len(qid_order)}")

    out_records: list[dict[str, Any]] = []

    for li, layer in enumerate(layers):
        H = h_per_question[:, li, :].astype(np.float32)
        X_rows = H[rows_by_question]
        print(f"[layer {layer}] hidden_dim={H.shape[1]} rows={X_rows.shape[0]}")

        # ---------- SEPs (regression baseline) ----------
        cv_pred = group_kfold_predict(
            X_rows, y_se, groups,
            probe_factory=lambda: RidgeProbe(alpha=args.ridge_alpha, seed=args.seed),
            n_splits=args.n_splits, seed=args.seed,
        )
        score_conf = -cv_pred.y_score
        for r, q in enumerate(qids):
            out_records.append({
                "probe": "SEPs_ridge", "layer": int(layer), "regime": "cv_5fold",
                "question_id": q, "seed": int(seeds[r]),
                "y_true": float(y_corr[r]), "y_score": float(score_conf[r]),
            })
        cs_pred = cross_seed_predict(
            X_rows, y_se, seeds,
            probe_factory=lambda: RidgeProbe(alpha=args.ridge_alpha, seed=args.seed),
        )
        cs_conf = -cs_pred
        for r, q in enumerate(qids):
            out_records.append({
                "probe": "SEPs_ridge", "layer": int(layer), "regime": "cross_seed",
                "question_id": q, "seed": int(seeds[r]),
                "y_true": float(y_corr[r]), "y_score": float(cs_conf[r]),
            })

        # ---------- SEPs strong variant (logreg directly on h -> P(correct)) ----------
        cv_pred_lr = group_kfold_predict(
            X_rows, y_corr, groups,
            probe_factory=lambda: LogRegProbe(C=args.logreg_c, seed=args.seed),
            n_splits=args.n_splits, seed=args.seed,
        )
        for r, q in enumerate(qids):
            out_records.append({
                "probe": "SEPs_logreg", "layer": int(layer), "regime": "cv_5fold",
                "question_id": q, "seed": int(seeds[r]),
                "y_true": float(y_corr[r]), "y_score": float(cv_pred_lr.y_score[r]),
            })
        cs_pred_lr = cross_seed_predict(
            X_rows, y_corr, seeds,
            probe_factory=lambda: LogRegProbe(C=args.logreg_c, seed=args.seed),
        )
        for r, q in enumerate(qids):
            out_records.append({
                "probe": "SEPs_logreg", "layer": int(layer), "regime": "cross_seed",
                "question_id": q, "seed": int(seeds[r]),
                "y_true": float(y_corr[r]), "y_score": float(cs_pred_lr[r]),
            })

        # ---------- DCP-MLP (ours, direct) ----------
        cv_pred_dcp = group_kfold_predict(
            X_rows, y_corr, groups,
            probe_factory=lambda: MLPProbe(
                hidden=args.mlp_hidden, out_dim=1, loss="bce",
                epochs=args.mlp_epochs, seed=args.seed,
            ),
            n_splits=args.n_splits, seed=args.seed,
        )
        for r, q in enumerate(qids):
            out_records.append({
                "probe": "DCP_mlp", "layer": int(layer), "regime": "cv_5fold",
                "question_id": q, "seed": int(seeds[r]),
                "y_true": float(y_corr[r]), "y_score": float(cv_pred_dcp.y_score[r]),
            })
        cs_pred_dcp = cross_seed_predict(
            X_rows, y_corr, seeds,
            probe_factory=lambda: MLPProbe(
                hidden=args.mlp_hidden, out_dim=1, loss="bce",
                epochs=args.mlp_epochs, seed=args.seed,
            ),
        )
        for r, q in enumerate(qids):
            out_records.append({
                "probe": "DCP_mlp", "layer": int(layer), "regime": "cross_seed",
                "question_id": q, "seed": int(seeds[r]),
                "y_true": float(y_corr[r]), "y_score": float(cs_pred_dcp[r]),
            })

        # ---------- ARD (anchor regression -> logreg head) ----------
        anchor_oof = stack_anchor_oof(X_rows, y_anchor, groups, args)
        ard_pred = fit_logreg_head(anchor_oof, y_corr, groups, args)
        for r, q in enumerate(qids):
            out_records.append({
                "probe": "ARD_ridge", "layer": int(layer), "regime": "cv_5fold",
                "question_id": q, "seed": int(seeds[r]),
                "y_true": float(y_corr[r]), "y_score": float(ard_pred[r]),
            })
        ard_cs = cross_seed_anchor_then_head(X_rows, y_anchor, y_corr, seeds, args, backbone="ridge")
        for r, q in enumerate(qids):
            out_records.append({
                "probe": "ARD_ridge", "layer": int(layer), "regime": "cross_seed",
                "question_id": q, "seed": int(seeds[r]),
                "y_true": float(y_corr[r]), "y_score": float(ard_cs[r]),
            })

        anchor_oof_mlp = stack_anchor_oof_mlp(X_rows, y_anchor, groups, args)
        ard_pred_mlp = fit_logreg_head(anchor_oof_mlp, y_corr, groups, args)
        for r, q in enumerate(qids):
            out_records.append({
                "probe": "ARD_mlp", "layer": int(layer), "regime": "cv_5fold",
                "question_id": q, "seed": int(seeds[r]),
                "y_true": float(y_corr[r]), "y_score": float(ard_pred_mlp[r]),
            })
        ard_cs_mlp = cross_seed_anchor_then_head(X_rows, y_anchor, y_corr, seeds, args, backbone="mlp")
        for r, q in enumerate(qids):
            out_records.append({
                "probe": "ARD_mlp", "layer": int(layer), "regime": "cross_seed",
                "question_id": q, "seed": int(seeds[r]),
                "y_true": float(y_corr[r]), "y_score": float(ard_cs_mlp[r]),
            })

    out_df = pd.DataFrame(out_records)
    out_csv = out_dir / "probe_predictions.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
