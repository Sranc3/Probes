"""Load anchor candidate rows and aggregate them into per-question features.

Two prediction *settings* are supported:

- **sample0**: predict ``P(strict_correct=1)`` for the first sample (k=1, no
  rerank). Selected answer = the row with ``is_sample0 == 1`` for that question.
- **fixed_k_majority_basin**: predict ``P(strict_correct=1)`` for the answer
  picked by the majority-basin rule over the first k samples. Selected answer
  = the shortest text in the most populous canonical basin among the first k
  samples (matches the no-leak fixed-k protocol used everywhere else).

For each (question, seed) we produce a single row with:

1. **Selected-answer features**: features of the candidate that the policy
   actually returns (so that the predictor sees what the user sees).
2. **Group-level features**: aggregations over the K samples that summarise the
   answer-basin geometry for the question (mean entropy, num basins, top
   cluster mass, teacher consensus, ...).

This file deliberately makes no assumption about ``strict_correct`` being
available at inference time - that label is only used for training the
predictor and for evaluation.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


STOPWORDS = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with", "by"}


def normalize_text(text: Any) -> str:
    text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(cleaned.split())


def canonical_answer(text: Any, max_tokens: int = 12) -> str:
    normalized = normalize_text(text)
    tokens = [token for token in normalized.split() if token not in STOPWORDS]
    return " ".join(tokens[:max_tokens])


def majority_basin_choice(canonicals: list[str], completions: list[str]) -> int:
    """Same rule as the eval pipeline: most populous basin, then shortest text, then earliest."""
    counts = Counter(canonicals)
    first_seen = {key: idx for idx, key in reversed(list(enumerate(canonicals)))}
    best_basin = max(counts.keys(), key=lambda b: (counts[b], -first_seen[b]))
    candidates = [idx for idx, key in enumerate(canonicals) if key == best_basin]
    return min(candidates, key=lambda idx: (len(completions[idx].split()), idx))


SELF_NUMERIC_FEATURES = [
    "logprob_avg",
    "token_mean_entropy",
    "token_max_entropy",
    "token_count",
    "completion_tokens",
    "cluster_size",
    "cluster_weight_mass",
    "semantic_entropy_weighted_set",
    "semantic_clusters_set",
    "logprob_avg_z",
    "token_mean_entropy_z",
    "token_max_entropy_z",
    "token_count_z",
    "cluster_size_z",
    "cluster_weight_mass_z",
    "logprob_rank",
    "low_entropy_rank",
    "cluster_size_rank",
    "verifier_score_v05",
]

TEACHER_NUMERIC_FEATURES = [
    "teacher_best_similarity",
    "teacher_best_basin_mass",
    "teacher_best_basin_strict_any",
    "teacher_support_mass",
    "teacher_correct_support_mass",
    "qwen_only_stable_mass",
    "anchor_score_noleak",
]


def _safe_float(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _agg_basin_geometry(rows: pd.DataFrame) -> dict[str, float]:
    """Group-level features that don't depend on which candidate was selected."""
    canonicals = [canonical_answer(text) for text in rows["answer_text"].tolist()]
    counts = Counter(canonicals)
    sizes = sorted(counts.values(), reverse=True)
    n = len(rows)
    top1 = sizes[0] if sizes else 0
    top2 = sizes[1] if len(sizes) > 1 else 0
    return {
        "group_size": float(n),
        "group_num_basins": float(len(counts)),
        "group_top1_basin_share": float(top1) / max(1.0, n),
        "group_top2_basin_share": float(top2) / max(1.0, n),
        "group_top1_minus_top2": float(top1 - top2) / max(1.0, n),
        "group_basin_entropy": float(
            -sum((c / n) * np.log((c / n) + 1e-12) for c in counts.values())
        ),
    }


def _agg_features(rows: pd.DataFrame, columns: Iterable[str], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in columns:
        if col not in rows.columns:
            continue
        values = rows[col].apply(_safe_float).to_numpy()
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            mu = mn = mx = std = float("nan")
        else:
            mu = float(np.mean(finite))
            mn = float(np.min(finite))
            mx = float(np.max(finite))
            std = float(np.std(finite, ddof=0))
        out[f"{prefix}_{col}_mean"] = mu
        out[f"{prefix}_{col}_min"] = mn
        out[f"{prefix}_{col}_max"] = mx
        out[f"{prefix}_{col}_std"] = std
    return out


def _selected_features(selected: pd.Series, columns: Iterable[str], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{col}": _safe_float(selected.get(col)) for col in columns if col in selected.index}


def build_question_features(
    df: pd.DataFrame,
    setting: str,
    fixed_k: int = 8,
) -> pd.DataFrame:
    """For each (question_id, seed) build one feature row + label.

    Args:
        df: candidate-level dataframe, expected columns include
            ``question_id``, ``seed``, ``sample_index``, ``is_sample0``,
            ``strict_correct``, ``answer_text`` plus the SELF/TEACHER feature
            columns from ``qwen_candidate_anchor_rows_final_only.csv``.
        setting: ``"sample0"`` or ``"fixed_k"`` (uses ``fixed_k`` samples).
        fixed_k: number of samples to consider when ``setting == "fixed_k"``.

    Returns:
        DataFrame with one row per (question_id, seed). Columns:
            - ``question_id``, ``seed``
            - ``label`` (1 if selected answer is strict_correct)
            - ``selected_strict``, ``selected_token_count`` (sanity / cost)
            - ``sel_*`` features from the selected row
            - ``self_*`` and ``teacher_*`` group-level aggregations
            - ``group_*`` basin geometry
    """
    if setting not in {"sample0", "fixed_k"}:
        raise ValueError(f"Unknown setting: {setting}")
    rows_out: list[dict[str, Any]] = []
    for (qid, seed), group in df.groupby(["question_id", "seed"], sort=False):
        group = group.sort_values("sample_index").reset_index(drop=True)
        if setting == "sample0":
            sel_idx = int(group.index[group["is_sample0"] == 1][0]) if (group["is_sample0"] == 1).any() else 0
            window = group.iloc[: 1]
        else:
            window = group.iloc[: fixed_k]
            if window.shape[0] == 0:
                continue
            canonicals = [canonical_answer(text) for text in window["answer_text"].tolist()]
            completions = window["answer_text"].fillna("").tolist()
            sel_idx_local = majority_basin_choice(canonicals, completions)
            sel_idx = int(window.index[sel_idx_local])
        selected = group.loc[sel_idx]
        out_row: dict[str, Any] = {
            "question_id": str(qid),
            "seed": int(seed),
            "label": float(selected["strict_correct"]),
            "selected_token_count": _safe_float(selected.get("token_count")),
            "selected_completion_tokens": _safe_float(selected.get("completion_tokens")),
            "selected_answer_text": str(selected.get("answer_text", "")),
            "any_correct_in_window": float((window["strict_correct"] > 0.5).any()),
            "any_correct_in_full_group": float((group["strict_correct"] > 0.5).any()),
        }
        out_row.update(_selected_features(selected, SELF_NUMERIC_FEATURES, "sel_self"))
        out_row.update(_selected_features(selected, TEACHER_NUMERIC_FEATURES, "sel_teacher"))
        out_row.update(_agg_features(window, SELF_NUMERIC_FEATURES, "win_self"))
        out_row.update(_agg_features(window, TEACHER_NUMERIC_FEATURES, "win_teacher"))
        out_row.update(_agg_basin_geometry(window))
        rows_out.append(out_row)
    return pd.DataFrame(rows_out)


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {
        "question_id",
        "seed",
        "label",
        "selected_token_count",
        "selected_completion_tokens",
        "selected_answer_text",
        "any_correct_in_window",
        "any_correct_in_full_group",
    }
    return [col for col in df.columns if col not in skip]


def fill_missing(matrix: np.ndarray, fill: str = "median") -> np.ndarray:
    out = matrix.copy().astype(np.float64)
    for col in range(out.shape[1]):
        column = out[:, col]
        finite_mask = np.isfinite(column)
        if not finite_mask.any():
            out[:, col] = 0.0
            continue
        if fill == "median":
            value = float(np.median(column[finite_mask]))
        elif fill == "mean":
            value = float(np.mean(column[finite_mask]))
        else:
            value = 0.0
        column[~finite_mask] = value
        out[:, col] = column
    return out


def load_anchor_rows(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
