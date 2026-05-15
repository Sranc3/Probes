"""End-to-end selective-prediction experiments for Plan_opus_selective.

For each setting in {sample0, fixed_8} we evaluate multiple confidence
predictors:

- Single-feature heuristics (negative entropy, logprob, semantic entropy,
  cluster mass, verifier_score_v05, teacher_support_mass, ...)
- Logistic regression on three feature subsets:
    * self-only  (no teacher columns)
    * teacher-only
    * all features (self + teacher + group geometry)
- Small MLP on all features

Two evaluation regimes:
- 5-fold CV at *question* level (seed-balanced, no leakage)
- cross-seed: train on seed 42, test on seed 43 (and reverse, then average)

For every (setting, predictor, regime) we compute selective-prediction
metrics (AURC, AUROC, Brier, ECE, selective accuracy at coverage levels)
and write them to ``results/selective_metrics_long.csv`` and
``results/selective_metrics_pivot.csv``.

We also dump the per-item predictions used by the best predictor in each
setting so that the routing analysis script can reuse them without re-training.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
sys.path.insert(0, str(SHARED_DIR))

from data_utils import (  # noqa: E402
    SELF_NUMERIC_FEATURES,
    TEACHER_NUMERIC_FEATURES,
    build_question_features,
    feature_columns,
    fill_missing,
    load_anchor_rows,
)
from metrics import safe_auroc, summarize  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--anchor-rows",
        default="/zhutingqi/song/Plan_gpt55/cross_model_anchor/runs/run_20260513_012413_gptoss_anchor_triviaqa_full500_k4_tok256/qwen_candidate_anchor_rows_final_only.csv",
    )
    p.add_argument("--output-dir", default="/zhutingqi/song/Plan_opus_selective/results")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--mlp-hidden", type=int, default=64)
    p.add_argument("--mlp-max-iter", type=int, default=400)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


SINGLE_FEATURE_BASELINES: list[dict[str, str]] = [
    {"label": "neg_token_mean_entropy_sel", "column": "sel_self_token_mean_entropy", "direction": "neg"},
    {"label": "logprob_avg_sel", "column": "sel_self_logprob_avg", "direction": "pos"},
    {"label": "neg_semantic_entropy_set_sel", "column": "sel_self_semantic_entropy_weighted_set", "direction": "neg"},
    {"label": "cluster_size_sel", "column": "sel_self_cluster_size", "direction": "pos"},
    {"label": "verifier_score_v05_sel", "column": "sel_self_verifier_score_v05", "direction": "pos"},
    {"label": "teacher_best_similarity_sel", "column": "sel_teacher_teacher_best_similarity", "direction": "pos"},
    {"label": "teacher_support_mass_sel", "column": "sel_teacher_teacher_support_mass", "direction": "pos"},
    {"label": "anchor_score_noleak_sel", "column": "sel_teacher_anchor_score_noleak", "direction": "pos"},
    {"label": "neg_qwen_only_stable_mass_sel", "column": "sel_teacher_qwen_only_stable_mass", "direction": "neg"},
    {"label": "group_top1_basin_share", "column": "group_top1_basin_share", "direction": "pos"},
    {"label": "neg_group_basin_entropy", "column": "group_basin_entropy", "direction": "neg"},
]


def feature_subset(columns: list[str], subset: str) -> list[str]:
    if subset == "all":
        return columns
    if subset == "self":
        return [
            c
            for c in columns
            if (c.startswith("sel_self_") or c.startswith("win_self_") or c.startswith("group_"))
            and not c.startswith("sel_teacher_")
            and not c.startswith("win_teacher_")
        ]
    if subset == "teacher":
        return [c for c in columns if c.startswith("sel_teacher_") or c.startswith("win_teacher_")]
    raise ValueError(subset)


def fit_logreg(X_train, y_train, X_test):
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)
    if len(np.unique(y_train)) < 2:
        # Trivial fallback if a fold is degenerate.
        return np.full(X_test.shape[0], y_train.mean()), np.full(X_test.shape[0], y_train.mean())
    clf = LogisticRegressionCV(
        Cs=10,
        cv=3,
        scoring="roc_auc",
        max_iter=2000,
        class_weight="balanced",
        n_jobs=1,
    ).fit(Xtr, y_train)
    probs = clf.predict_proba(Xte)[:, 1]
    return probs, probs


def fit_mlp(X_train, y_train, X_test, hidden, max_iter, seed):
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)
    if len(np.unique(y_train)) < 2:
        return np.full(X_test.shape[0], y_train.mean()), np.full(X_test.shape[0], y_train.mean())
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden, hidden // 2),
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    ).fit(Xtr, y_train)
    probs = clf.predict_proba(Xte)[:, 1]
    return probs, probs


def evaluate_single_feature(features_df: pd.DataFrame, baseline: dict[str, str]) -> dict[str, Any] | None:
    column = baseline["column"]
    if column not in features_df.columns:
        return None
    raw = features_df[column].astype(np.float64).to_numpy()
    finite_mask = np.isfinite(raw)
    if finite_mask.sum() == 0:
        return None
    median = float(np.median(raw[finite_mask]))
    raw = np.where(finite_mask, raw, median)
    direction = baseline["direction"]
    if direction == "neg":
        scores = -raw
    elif direction == "pos":
        scores = raw
    else:
        raise ValueError(direction)
    correct = features_df["label"].astype(np.float64).to_numpy()
    return {
        "label": baseline["label"],
        "regime": "single_feature",
        "feature_subset": "single",
        "metrics": summarize(scores, correct),
        "scores": scores,
    }


def cv_predictions(
    features_df: pd.DataFrame,
    columns: list[str],
    fit_fn,
    n_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = features_df[columns].to_numpy(dtype=np.float64)
    X = fill_missing(X)
    y = features_df["label"].astype(np.float64).to_numpy()
    qids = features_df["question_id"].astype(str).to_numpy()
    kf = GroupKFold(n_splits=n_folds)
    probs_out = np.zeros_like(y, dtype=np.float64)
    fold_assignments = np.zeros_like(y, dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    inv_perm = np.argsort(perm)
    Xp, yp, qidsp = X[perm], y[perm], qids[perm]
    probs_perm = np.zeros_like(yp, dtype=np.float64)
    fold_perm = np.zeros_like(yp, dtype=np.int64)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(Xp, yp, groups=qidsp)):
        probs, _ = fit_fn(Xp[train_idx], yp[train_idx], Xp[test_idx])
        probs_perm[test_idx] = probs
        fold_perm[test_idx] = fold_idx
    probs_out = probs_perm[inv_perm]
    fold_assignments = fold_perm[inv_perm]
    return probs_out, y, fold_assignments


def cross_seed_predictions(
    features_df: pd.DataFrame,
    columns: list[str],
    fit_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = features_df[columns].to_numpy(dtype=np.float64)
    X = fill_missing(X)
    y = features_df["label"].astype(np.float64).to_numpy()
    seeds = features_df["seed"].astype(int).to_numpy()
    unique_seeds = sorted(set(seeds.tolist()))
    if len(unique_seeds) < 2:
        return cv_predictions(features_df, columns, fit_fn, 5, 0)[:3]
    probs_out = np.full_like(y, fill_value=np.nan)
    fold_assignments = np.zeros_like(y, dtype=np.int64)
    for fold_idx, test_seed in enumerate(unique_seeds):
        train_mask = seeds != test_seed
        test_mask = ~train_mask
        probs, _ = fit_fn(X[train_mask], y[train_mask], X[test_mask])
        probs_out[test_mask] = probs
        fold_assignments[test_mask] = fold_idx
    return probs_out, y, fold_assignments


def evaluate_predictor(
    features_df: pd.DataFrame,
    columns: list[str],
    label: str,
    fit_fn,
    folds: int,
    seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    probs, y, _ = cv_predictions(features_df, columns, fit_fn, folds, seed)
    out.append(
        {
            "label": label,
            "regime": "cv_5fold",
            "feature_subset": label.split(":")[-1] if ":" in label else label,
            "metrics": summarize(probs, y, probs=probs),
            "scores": probs,
            "probs": probs,
        }
    )
    probs_cs, y_cs, _ = cross_seed_predictions(features_df, columns, fit_fn)
    out.append(
        {
            "label": label,
            "regime": "cross_seed",
            "feature_subset": label.split(":")[-1] if ":" in label else label,
            "metrics": summarize(probs_cs, y_cs, probs=probs_cs),
            "scores": probs_cs,
            "probs": probs_cs,
        }
    )
    return out


def run_setting(
    df_full: pd.DataFrame,
    setting: str,
    args: argparse.Namespace,
    output_dir: Path,
    fixed_k: int = 8,
) -> dict[str, Any]:
    print(f"\n=== Setting: {setting} (fixed_k={fixed_k if setting!='sample0' else 1}) ===", flush=True)
    features_df = build_question_features(df_full, setting=setting, fixed_k=fixed_k)
    n_correct = int(features_df["label"].sum())
    print(f"  n={len(features_df)}, base_acc={n_correct / max(1, len(features_df)):.4f}", flush=True)

    rows: list[dict[str, Any]] = []
    full_columns = feature_columns(features_df)

    # Single-feature baselines (deterministic, no fitting).
    single_predictions: dict[str, np.ndarray] = {}
    for baseline in SINGLE_FEATURE_BASELINES:
        result = evaluate_single_feature(features_df, baseline)
        if not result:
            continue
        rows.append(
            {
                "setting": setting,
                "predictor": result["label"],
                "regime": result["regime"],
                "feature_subset": "single",
                **result["metrics"],
            }
        )
        single_predictions[result["label"]] = result["scores"]

    # Trained predictors on three feature subsets.
    trained_predictions: dict[str, np.ndarray] = {}
    for subset in ("self", "teacher", "all"):
        cols = feature_subset(full_columns, subset)
        if not cols:
            continue
        predictor_label_logreg = f"logreg:{subset}"
        for entry in evaluate_predictor(features_df, cols, predictor_label_logreg, fit_logreg, args.folds, args.seed):
            rows.append(
                {
                    "setting": setting,
                    "predictor": entry["label"],
                    "regime": entry["regime"],
                    "feature_subset": subset,
                    **entry["metrics"],
                }
            )
            if entry["regime"] == "cv_5fold":
                trained_predictions[predictor_label_logreg] = entry["probs"]
        if subset == "all":
            mlp_label = "mlp:all"
            for entry in evaluate_predictor(
                features_df,
                cols,
                mlp_label,
                lambda Xtr, ytr, Xte: fit_mlp(Xtr, ytr, Xte, args.mlp_hidden, args.mlp_max_iter, args.seed),
                args.folds,
                args.seed,
            ):
                rows.append(
                    {
                        "setting": setting,
                        "predictor": entry["label"],
                        "regime": entry["regime"],
                        "feature_subset": "all",
                        **entry["metrics"],
                    }
                )
                if entry["regime"] == "cv_5fold":
                    trained_predictions[mlp_label] = entry["probs"]

    # Persist per-item predictions for the best CV predictor.
    output_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_dir / f"question_features_{setting}.csv", index=False)
    cv_rows = [r for r in rows if r["regime"] == "cv_5fold"]
    if cv_rows:
        best = max(cv_rows, key=lambda r: r.get("auroc", 0.0))
        best_label = best["predictor"]
        if best_label in trained_predictions:
            best_scores = trained_predictions[best_label]
        elif best_label in single_predictions:
            best_scores = single_predictions[best_label]
        else:
            best_scores = None
        if best_scores is not None:
            out_df = features_df[["question_id", "seed", "label", "selected_token_count", "selected_answer_text"]].copy()
            out_df["confidence"] = best_scores
            if "sel_teacher_teacher_best_basin_strict_any" in features_df.columns:
                out_df["teacher_correct"] = features_df["sel_teacher_teacher_best_basin_strict_any"].fillna(0.0).astype(float)
            else:
                out_df["teacher_correct"] = np.nan
            out_df.to_csv(output_dir / f"best_predictions_{setting}.csv", index=False)
        # Also save runner-up predictors so routing can compare.
        for label, scores in trained_predictions.items():
            if label == best_label:
                continue
            out_df = features_df[["question_id", "seed", "label"]].copy()
            out_df["confidence"] = scores
            out_df.to_csv(output_dir / f"predictions_{setting}_{label.replace(':', '_')}.csv", index=False)

    return {"setting": setting, "rows": rows, "n": len(features_df)}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    df = load_anchor_rows(args.anchor_rows)
    print(f"loaded {len(df)} candidate rows from {args.anchor_rows}")

    all_rows: list[dict[str, Any]] = []
    for setting, fixed_k in (("sample0", 1), ("fixed_k", 8)):
        result = run_setting(df, setting, args, output_dir, fixed_k=fixed_k)
        all_rows.extend(result["rows"])

    long_df = pd.DataFrame(all_rows)
    long_path = output_dir / "selective_metrics_long.csv"
    long_df.to_csv(long_path, index=False)

    pivot = long_df.pivot_table(
        index=["setting", "predictor", "feature_subset"],
        columns="regime",
        values=["auroc", "aurc", "sel_acc@0.25", "sel_acc@0.50", "sel_acc@0.75", "sel_acc@1.00", "brier", "ece"],
        aggfunc="first",
    )
    pivot_path = output_dir / "selective_metrics_pivot.csv"
    pivot.to_csv(pivot_path)
    print(f"\nWrote {long_path} and {pivot_path}")
    print(long_df.to_string(index=False, float_format=lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)))


if __name__ == "__main__":
    main()
