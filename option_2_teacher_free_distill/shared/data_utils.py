"""Per-question label and target construction for teacher-free probes.

We reuse the anchor candidate rows from Plan_gpt55 (the same source
Plan_opus_selective uses) and build:

- ``y_corr``       : ``strict_correct`` of sample0 (per (question, seed) row)
- ``y_se``         : ``semantic_entropy_weighted_set`` (SEPs target, per row)
- ``y_anchor``     : 7-dim teacher feature vector (ARD target, per row)
- ``meta``         : question_id / seed / answer_text bookkeeping

Notes:
- The probes operate on Qwen hidden states extracted from the prompt only.
  There is one hidden state per *question*; both seeds re-use that state but
  carry different stochastic-sample0 labels. Training therefore happens at
  the row level (1000 rows, 500 unique x), and *all CV splits group on
  question_id* to avoid leakage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TEACHER_ANCHOR_COLUMNS = [
    "teacher_best_similarity",
    "teacher_best_basin_mass",
    "teacher_best_basin_strict_any",
    "teacher_support_mass",
    "teacher_correct_support_mass",
    "qwen_only_stable_mass",
    "anchor_score_noleak",
]

SEPs_TARGET_COLUMN = "semantic_entropy_weighted_set"


def load_anchor_rows(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def select_sample0_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (question_id, seed) corresponding to sample0 (sample_index == 0)."""
    sample0 = df[df["sample_index"] == 0].copy()
    sample0 = sample0.sort_values(["question_id", "seed"]).reset_index(drop=True)
    expected_cols = {
        "question_id",
        "seed",
        "question",
        "strict_correct",
        SEPs_TARGET_COLUMN,
        *TEACHER_ANCHOR_COLUMNS,
    }
    missing = expected_cols - set(sample0.columns)
    if missing:
        raise ValueError(f"Missing required columns in anchor rows: {sorted(missing)}")
    return sample0


def attach_targets(sample0: pd.DataFrame) -> dict[str, Any]:
    """Build numeric targets aligned to ``sample0`` row order."""
    y_corr = sample0["strict_correct"].astype(np.float32).to_numpy()
    y_se = sample0[SEPs_TARGET_COLUMN].astype(np.float32).to_numpy()
    y_anchor = sample0[TEACHER_ANCHOR_COLUMNS].astype(np.float32).to_numpy()
    meta = sample0[["question_id", "seed", "question", "answer_text"]].copy()
    return {"y_corr": y_corr, "y_se": y_se, "y_anchor": y_anchor, "meta": meta}


def question_index_map(sample0: pd.DataFrame) -> dict[str, int]:
    """Stable question_id -> integer index used to look up cached hidden states."""
    qids = sample0["question_id"].drop_duplicates().tolist()
    return {qid: i for i, qid in enumerate(qids)}


def expand_to_rows(
    h_per_question: np.ndarray,
    qid_to_idx: dict[str, int],
    sample0: pd.DataFrame,
) -> np.ndarray:
    """Replicate per-question hidden states across both seed rows."""
    idx = sample0["question_id"].map(qid_to_idx).to_numpy()
    if (idx < 0).any() or (idx >= h_per_question.shape[0]).any():
        bad = sample0.loc[(idx < 0) | (idx >= h_per_question.shape[0]), "question_id"].tolist()
        raise KeyError(f"Hidden state cache missing question ids: {bad[:5]}...")
    return h_per_question[idx]
