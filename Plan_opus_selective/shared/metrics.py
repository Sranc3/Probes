"""Selective-prediction metrics.

Following Geifman & El-Yaniv (2017) and Ding et al. (2020):

- ``risk_coverage_curve``: sort items by confidence (descending), accept the
  top ``coverage`` fraction, return (coverage, risk) pairs.
- ``aurc``: area under that curve - lower is better, ``aurc=0`` means a
  perfect ranker that defers all wrong items first.
- ``selective_accuracy_at_coverage``: 1 - risk at fixed coverage levels.
- ``brier_score`` and ``ece``: standard calibration metrics.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


def risk_coverage_curve(confidences: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = correct.size
    order = np.argsort(-confidences, kind="stable")
    correct_sorted = correct[order]
    cumulative_correct = np.cumsum(correct_sorted)
    coverages = np.arange(1, n + 1, dtype=np.float64) / float(n)
    accuracies = cumulative_correct / np.arange(1, n + 1, dtype=np.float64)
    risks = 1.0 - accuracies
    return coverages, risks


def aurc(confidences: np.ndarray, correct: np.ndarray) -> float:
    coverages, risks = risk_coverage_curve(confidences, correct)
    return float(np.trapz(risks, coverages))


def selective_accuracy_at_coverage(
    confidences: np.ndarray, correct: np.ndarray, coverage: float
) -> float:
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = correct.size
    if n == 0:
        return float("nan")
    k = max(1, int(round(coverage * n)))
    order = np.argsort(-confidences, kind="stable")
    accepted = correct[order[:k]]
    return float(accepted.mean())


def brier_score(probs: np.ndarray, correct: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    return float(np.mean((probs - correct) ** 2))


def ece(probs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    correct = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = correct.size
    total = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        conf = float(probs[mask].mean())
        acc = float(correct[mask].mean())
        total += (mask.sum() / n) * abs(conf - acc)
    return float(total)


def safe_auroc(scores: np.ndarray, correct: np.ndarray) -> float:
    correct = np.asarray(correct, dtype=np.float64)
    if correct.min() == correct.max():
        return float("nan")
    return float(roc_auc_score(correct, scores))


def summarize(confidences: np.ndarray, correct: np.ndarray, probs: np.ndarray | None = None,
              coverages: Iterable[float] = (0.25, 0.5, 0.75, 1.0)) -> dict[str, float]:
    out: dict[str, float] = {
        "n": int(correct.size),
        "base_acc": float(np.mean(correct)),
        "auroc": safe_auroc(confidences, correct),
        "aurc": aurc(confidences, correct),
    }
    for c in coverages:
        out[f"sel_acc@{c:.2f}"] = selective_accuracy_at_coverage(confidences, correct, c)
    if probs is not None:
        out["brier"] = brier_score(probs, correct)
        out["ece"] = ece(probs, correct)
    return out
