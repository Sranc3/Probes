#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from run_entropy_anatomy import auc_score, ensure_dir, mean, read_csv, safe_float, stdev, write_csv, write_json
from run_learned_basin_verifier import LogisticModel, question_folds


DEFAULT_WAVEFORM_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260430_021911_entropy_waveform_analysis_v2"
DEFAULT_CANDIDATE_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"


AGGREGATE_FEATURES = [
    "token_mean_entropy_mean",
    "token_mean_entropy_std",
    "token_max_entropy_mean",
    "token_max_entropy_std",
    "logprob_avg_mean",
    "logprob_avg_std",
    "cluster_weight_mass",
    "cluster_size",
    "centroid_entropy_z",
    "centroid_max_entropy_z",
    "centroid_logprob_z",
    "centroid_len_z",
    "within_basin_lexical_entropy",
    "within_basin_jaccard",
    "semantic_entropy_weighted_set",
    "fragmentation_entropy",
    "normalized_fragmentation_entropy",
    "top2_weight_margin",
    "top2_logprob_margin",
    "top2_low_entropy_margin",
    "distance_to_sample0_entropy_logprob",
    "stable_score",
    "low_entropy_score",
    "low_entropy_basin_rank",
    "logprob_basin_rank",
    "weight_basin_rank",
    "stable_basin_rank",
    "delta_entropy_vs_sample0",
    "delta_logprob_vs_sample0",
    "delta_weight_vs_sample0",
    "delta_stable_score_vs_sample0",
    "sample0_basin_entropy_z",
    "sample0_basin_logprob_z",
    "sample0_basin_weight",
]

WAVEFORM_FEATURES = [
    "wf_entropy_mean_mean",
    "wf_entropy_mean_std",
    "wf_entropy_std_mean",
    "wf_entropy_early_mean_mean",
    "wf_entropy_mid_mean_mean",
    "wf_entropy_late_mean_mean",
    "wf_entropy_late_minus_early_mean",
    "wf_entropy_slope_mean",
    "wf_entropy_auc_mean",
    "wf_entropy_max_mean",
    "wf_entropy_max_pos_norm_mean",
    "wf_entropy_spike_count_mean",
    "wf_entropy_spike_frac_mean",
    "wf_entropy_early_collapse_mean",
    "wf_entropy_late_hesitation_mean",
    "wf_prob_mean_mean",
    "wf_prob_min_mean",
    "wf_prob_early_mean_mean",
    "wf_prob_late_mean_mean",
    "wf_basin_entropy_curve_var",
    "wf_basin_prob_curve_var",
]

FEATURE_SETS = {
    "aggregate_only": AGGREGATE_FEATURES,
    "waveform_only": WAVEFORM_FEATURES,
    "aggregate_plus_waveform": AGGREGATE_FEATURES + WAVEFORM_FEATURES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run entropy waveform hallucination detector and damage-veto controller.")
    parser.add_argument("--waveform-run", default=DEFAULT_WAVEFORM_RUN)
    parser.add_argument("--candidate-run", default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="waveform_hallucination_veto")
    return parser.parse_args()


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def representative_candidates(candidate_rows: list[dict[str, Any]]) -> dict[tuple[int, str, int], dict[str, Any]]:
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[(int(row["seed"]), str(row["question_id"]), int(row["cluster_id"]))].append(row)
    reps: dict[tuple[int, str, int], dict[str, Any]] = {}
    for key, rows in grouped.items():
        reps[key] = max(
            rows,
            key=lambda row: (
                safe_float(row["logprob_avg_z"]) - 0.35 * safe_float(row["token_mean_entropy_z"]) - 0.1 * safe_float(row["token_count_z"]),
                -int(row["sample_index"]),
            ),
        )
    return reps


def build_rows(waveform_run: Path, candidate_run: Path) -> tuple[list[dict[str, Any]], dict[tuple[int, str], list[dict[str, Any]]]]:
    rows = read_csv(waveform_run / "basin_waveform_table.csv")
    candidates = read_csv(candidate_run / "candidate_features.csv")
    reps = representative_candidates(candidates)
    by_pair: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (int(row["seed"]), str(row["question_id"]), int(row["cluster_id"]))
        rep = reps[key]
        row["representative_sample_index"] = int(rep["sample_index"])
        row["representative_strict_correct"] = safe_float(rep["strict_correct"])
        row["representative_preview"] = rep["answer_preview"]
        row["stable_hallucination_label"] = float(row.get("basin_regime") == "stable_hallucination")
        row["stable_detector_row"] = float(row.get("basin_regime") in {"stable_correct", "stable_hallucination"})
        by_pair[(int(row["seed"]), str(row["question_id"]))].append(row)
    for group in by_pair.values():
        group.sort(key=lambda row: int(row["cluster_id"]))
        sample0 = next(row for row in group if safe_float(row["contains_sample0"]) > 0)
        for row in group:
            row["delta_entropy_vs_sample0"] = safe_float(row["centroid_entropy_z"]) - safe_float(sample0["centroid_entropy_z"])
            row["delta_logprob_vs_sample0"] = safe_float(row["centroid_logprob_z"]) - safe_float(sample0["centroid_logprob_z"])
            row["delta_weight_vs_sample0"] = safe_float(row["cluster_weight_mass"]) - safe_float(sample0["cluster_weight_mass"])
            row["delta_stable_score_vs_sample0"] = safe_float(row["stable_score"]) - safe_float(sample0["stable_score"])
            row["sample0_basin_entropy_z"] = safe_float(sample0["centroid_entropy_z"])
            row["sample0_basin_logprob_z"] = safe_float(sample0["centroid_logprob_z"])
            row["sample0_basin_weight"] = safe_float(sample0["cluster_weight_mass"])
            row["switch_gain_label"] = float(
                safe_float(sample0["representative_strict_correct"]) <= 0 and safe_float(row["representative_strict_correct"]) > 0 and safe_float(row["contains_sample0"]) <= 0
            )
    return [row for group in by_pair.values() for row in group], dict(by_pair)


def classification_metrics(rows: list[dict[str, Any]], scores: list[float], threshold: float, split: str, method: str) -> dict[str, Any]:
    labels = [int(safe_float(row["stable_hallucination_label"]) > 0) for row in rows]
    preds = [int(score >= threshold) for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "split": split,
        "method": method,
        "rows": len(rows),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "threshold": threshold,
        "auc_hallucination_high": auc_score([score for score, label in zip(scores, labels) if label], [score for score, label in zip(scores, labels) if not label]),
        "accuracy": mean([float(y == p) for y, p in zip(labels, preds)]),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def tune_detector_threshold(rows: list[dict[str, Any]], scores: list[float]) -> float:
    best = (-1.0, 0.5)
    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        metrics = classification_metrics(rows, scores, threshold, "train", "train")
        score = safe_float(metrics["f1"]) + 0.25 * safe_float(metrics["balanced_accuracy"])
        if score > best[0]:
            best = (score, threshold)
    return best[1]


def evaluate_selection(groups: list[list[dict[str, Any]]], selected: list[dict[str, Any]], method: str, split: str) -> dict[str, Any]:
    sample0s = [next(row for row in group if safe_float(row["contains_sample0"]) > 0) for group in groups]
    deltas = [safe_float(sel["representative_strict_correct"]) - safe_float(s0["representative_strict_correct"]) for sel, s0 in zip(selected, sample0s)]
    rescue_groups = [
        (group, sel)
        for group, sel in zip(groups, selected)
        if safe_float(next(row for row in group if safe_float(row["contains_sample0"]) > 0)["representative_strict_correct"]) <= 0
        and any(safe_float(row["representative_strict_correct"]) > 0 for row in group)
    ]
    initially_correct = [(group, sel) for group, sel in zip(groups, selected) if safe_float(next(row for row in group if safe_float(row["contains_sample0"]) > 0)["representative_strict_correct"]) > 0]
    return {
        "split": split,
        "method": method,
        "pairs": len(groups),
        "strict_correct_rate": mean([safe_float(row["representative_strict_correct"]) for row in selected]),
        "sample0_strict_correct_rate": mean([safe_float(row["representative_strict_correct"]) for row in sample0s]),
        "delta_vs_sample0": mean(deltas),
        "improved_count": sum(1 for value in deltas if value > 0),
        "damaged_count": sum(1 for value in deltas if value < 0),
        "net_gain_count": sum(1 for value in deltas if value > 0) - sum(1 for value in deltas if value < 0),
        "answer_changed_rate": mean([float(int(sel["cluster_id"]) != int(s0["cluster_id"])) for sel, s0 in zip(selected, sample0s)]),
        "rescue_recall": mean([safe_float(sel["representative_strict_correct"]) for _group, sel in rescue_groups]) if rescue_groups else 0.0,
        "damage_avoidance": mean([safe_float(sel["representative_strict_correct"]) for _group, sel in initially_correct]) if initially_correct else 0.0,
    }


def choose_switch(group: list[dict[str, Any]], model: LogisticModel, threshold: float) -> dict[str, Any]:
    sample0 = next(row for row in group if safe_float(row["contains_sample0"]) > 0)
    alternatives = [row for row in group if safe_float(row["contains_sample0"]) <= 0]
    if not alternatives:
        return sample0
    alt = max(alternatives, key=model.predict)
    return alt if model.predict(alt) >= threshold else sample0


def tune_switch(groups: list[list[dict[str, Any]]], model: LogisticModel) -> float:
    best = (-1e9, 0.5)
    for threshold in [0.2, 0.35, 0.5, 0.65, 0.8]:
        selected = [choose_switch(group, model, threshold) for group in groups]
        metrics = evaluate_selection(groups, selected, "train", "train")
        score = 100 * safe_float(metrics["delta_vs_sample0"]) + int(metrics["net_gain_count"]) - 2.0 * int(metrics["damaged_count"])
        if score > best[0]:
            best = (score, threshold)
    return best[1]


def choose_with_veto(
    group: list[dict[str, Any]],
    switch_model: LogisticModel,
    switch_threshold: float,
    risk_model: LogisticModel | None,
    risk_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample0 = next(row for row in group if safe_float(row["contains_sample0"]) > 0)
    proposed = choose_switch(group, switch_model, switch_threshold)
    switched = int(proposed["cluster_id"]) != int(sample0["cluster_id"])
    risk_score = risk_model.predict(proposed) if risk_model is not None else 0.0
    vetoed = switched and risk_model is not None and risk_score >= risk_threshold
    selected = sample0 if vetoed else proposed
    return selected, {
        "proposed_cluster_id": proposed["cluster_id"],
        "selected_cluster_id": selected["cluster_id"],
        "proposed_correct": proposed["representative_strict_correct"],
        "selected_correct": selected["representative_strict_correct"],
        "sample0_correct": sample0["representative_strict_correct"],
        "switched": float(switched),
        "vetoed": float(vetoed),
        "risk_score": risk_score,
        "risk_threshold": risk_threshold if risk_model is not None else "",
    }


def tune_veto_threshold(
    groups: list[list[dict[str, Any]]],
    switch_model: LogisticModel,
    switch_threshold: float,
    risk_model: LogisticModel,
) -> float:
    best = (-1e9, 1.01)
    for threshold in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.01]:
        selected = [choose_with_veto(group, switch_model, switch_threshold, risk_model, threshold)[0] for group in groups]
        metrics = evaluate_selection(groups, selected, "train", "train")
        score = 100 * safe_float(metrics["delta_vs_sample0"]) + int(metrics["net_gain_count"]) - 3.0 * int(metrics["damaged_count"])
        if score > best[0]:
            best = (score, threshold)
    return best[1]


def summarize(rows: list[dict[str, Any]], method_key: str = "method") -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "question_grouped_cv":
            continue
        grouped[str(row[method_key])].append(row)
    summary = []
    for method, items in sorted(grouped.items()):
        out = {"split": "question_grouped_cv", method_key: method, "folds": len(items)}
        for key in items[0]:
            if key in {"split", method_key, "feature_set"}:
                continue
            values = [safe_float(item.get(key, 0.0)) for item in items]
            if key.endswith("_count") or key in {"tp", "tn", "fp", "fn", "pairs", "rows", "improved_count", "damaged_count", "net_gain_count"}:
                out[key] = sum(int(round(value)) for value in values)
            else:
                out[key] = mean(values)
        summary.append(out)
    return summary


def run_detector_cv(rows: list[dict[str, Any]], by_pair: dict[tuple[int, str], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, str], dict[str, LogisticModel]]]:
    keys = sorted(by_pair)
    fold_metrics: list[dict[str, Any]] = []
    weights: list[dict[str, Any]] = []
    risk_models: dict[tuple[int, str], dict[str, LogisticModel]] = {}
    detector_rows = [row for row in rows if safe_float(row.get("stable_detector_row")) > 0]
    for fold_idx, (train_qids, test_qids) in enumerate(question_folds(keys)):
        train = [row for row in detector_rows if row["question_id"] in train_qids]
        test = [row for row in detector_rows if row["question_id"] in test_qids]
        risk_models[(fold_idx, "models")] = {}
        for name, features in FEATURE_SETS.items():
            model = LogisticModel(features)
            model.fit(train, "stable_hallucination_label", epochs=650, lr=0.07, l2=0.02)
            train_scores = [model.predict(row) for row in train]
            threshold = tune_detector_threshold(train, train_scores)
            test_scores = [model.predict(row) for row in test]
            metrics = classification_metrics(test, test_scores, threshold, f"fold_{fold_idx}", name)
            fold_metrics.append(metrics)
            weights.extend(model.weight_rows(f"hallucination_detector_{name}", f"fold_{fold_idx}", top_k=20))
            risk_models[(fold_idx, "models")][name] = model
    return summarize(fold_metrics), fold_metrics, risk_models


def run_veto_cv(
    by_pair: dict[tuple[int, str], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    keys = sorted(by_pair)
    fold_metrics: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    weights: list[dict[str, Any]] = []
    for fold_idx, (train_qids, test_qids) in enumerate(question_folds(keys)):
        train_groups = [group for (_seed, qid), group in by_pair.items() if qid in train_qids]
        test_groups = [group for (_seed, qid), group in by_pair.items() if qid in test_qids]
        train_rows = [row for group in train_groups for row in group]
        train_alt = [row for group in train_groups for row in group if safe_float(row["contains_sample0"]) <= 0]
        stable_train = [row for row in train_rows if safe_float(row.get("stable_detector_row")) > 0]

        switch_model = LogisticModel(AGGREGATE_FEATURES)
        switch_model.fit(train_alt, "switch_gain_label", epochs=700, lr=0.07, l2=0.01)
        switch_threshold = tune_switch(train_groups, switch_model)
        weights.extend(switch_model.weight_rows("baseline_switch_model", f"fold_{fold_idx}", top_k=20))

        baseline_selected = [choose_with_veto(group, switch_model, switch_threshold, None, 1.01)[0] for group in test_groups]
        baseline_metrics = evaluate_selection(test_groups, baseline_selected, "baseline_switch_no_veto", f"fold_{fold_idx}")
        baseline_metrics.update({"risk_feature_set": "none", "switch_threshold": switch_threshold, "risk_threshold": ""})
        fold_metrics.append(baseline_metrics)

        for risk_name, risk_features in FEATURE_SETS.items():
            risk_model = LogisticModel(risk_features)
            risk_model.fit(stable_train, "stable_hallucination_label", epochs=650, lr=0.07, l2=0.02)
            risk_threshold = tune_veto_threshold(train_groups, switch_model, switch_threshold, risk_model)
            weights.extend(risk_model.weight_rows(f"risk_model_{risk_name}", f"fold_{fold_idx}", top_k=20))

            selected: list[dict[str, Any]] = []
            detail_rows: list[dict[str, Any]] = []
            for group in test_groups:
                chosen, detail = choose_with_veto(group, switch_model, switch_threshold, risk_model, risk_threshold)
                sample0 = next(row for row in group if safe_float(row["contains_sample0"]) > 0)
                proposed = next(row for row in group if int(row["cluster_id"]) == int(detail["proposed_cluster_id"]))
                selected.append(chosen)
                detail_rows.append(
                    {
                        "split": f"fold_{fold_idx}",
                        "method": f"{risk_name}_veto",
                        "seed": chosen["seed"],
                        "question_id": chosen["question_id"],
                        "question": chosen["question"],
                        "sample0_cluster_id": sample0["cluster_id"],
                        "proposed_cluster_id": detail["proposed_cluster_id"],
                        "selected_cluster_id": detail["selected_cluster_id"],
                        "sample0_correct": detail["sample0_correct"],
                        "proposed_correct": detail["proposed_correct"],
                        "selected_correct": detail["selected_correct"],
                        "delta_if_no_veto": safe_float(proposed["representative_strict_correct"]) - safe_float(sample0["representative_strict_correct"]),
                        "delta_after_veto": safe_float(chosen["representative_strict_correct"]) - safe_float(sample0["representative_strict_correct"]),
                        "switched": detail["switched"],
                        "vetoed": detail["vetoed"],
                        "risk_score": detail["risk_score"],
                        "risk_threshold": detail["risk_threshold"],
                        "proposed_regime": proposed["basin_regime"],
                        "selected_regime": chosen["basin_regime"],
                        "sample0_preview": sample0.get("representative_preview", sample0.get("answer_preview", "")),
                        "proposed_preview": proposed.get("representative_preview", proposed.get("answer_preview", "")),
                        "selected_preview": chosen.get("representative_preview", chosen.get("answer_preview", "")),
                    }
                )
            metrics = evaluate_selection(test_groups, selected, f"{risk_name}_veto", f"fold_{fold_idx}")
            metrics.update(
                {
                    "risk_feature_set": risk_name,
                    "switch_threshold": switch_threshold,
                    "risk_threshold": risk_threshold,
                    "veto_count": sum(int(safe_float(row["vetoed"]) > 0) for row in detail_rows),
                    "prevented_damage_count": sum(1 for row in detail_rows if safe_float(row["vetoed"]) > 0 and safe_float(row["delta_if_no_veto"]) < 0 and safe_float(row["delta_after_veto"]) >= 0),
                    "blocked_rescue_count": sum(1 for row in detail_rows if safe_float(row["vetoed"]) > 0 and safe_float(row["delta_if_no_veto"]) > 0 and safe_float(row["delta_after_veto"]) <= 0),
                }
            )
            fold_metrics.append(metrics)
            selections.extend(detail_rows)
    return summarize(fold_metrics), fold_metrics, selections, weights


def make_plots(output_dir: Path, detector_summary: list[dict[str, Any]], veto_summary: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = ensure_dir(output_dir / "plots")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    methods = [row["method"] for row in detector_summary]
    aucs = [safe_float(row["auc_hallucination_high"]) for row in detector_summary]
    f1s = [safe_float(row["f1"]) for row in detector_summary]
    x = list(range(len(methods)))
    ax.bar([i - 0.18 for i in x], aucs, width=0.36, label="AUC")
    ax.bar([i + 0.18 for i in x], f1s, width=0.36, label="F1")
    ax.set_xticks(x, methods, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Stable Hallucination Detector")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "01_hallucination_detector_auc_f1.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    methods = [row["method"] for row in veto_summary]
    deltas = [100 * safe_float(row["delta_vs_sample0"]) for row in veto_summary]
    damages = [safe_float(row.get("damaged_count", 0)) for row in veto_summary]
    ax.bar(methods, deltas, label="Delta vs sample0 (%)")
    ax2 = ax.twinx()
    ax2.plot(methods, damages, color="#d62728", marker="o", label="Damaged count")
    ax.set_xticks(list(range(len(methods))), methods, rotation=20, ha="right")
    ax.set_title("Damage Veto Controller Tradeoff")
    ax.set_ylabel("Delta vs sample0 (%)")
    ax2.set_ylabel("Damaged count")
    fig.tight_layout()
    fig.savefig(plot_dir / "02_veto_controller_tradeoff.png")
    plt.close(fig)


def write_report(output_dir: Path, detector_summary: list[dict[str, Any]], veto_summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Entropy Waveform Hallucination Detector and Veto Controller",
        "",
        "## 0. 重要审计说明",
        "",
        "这份结果使用的是 **no-leak** 特征版本：`stable_hallucination_score` 以及由它派生的 delta 特征没有进入 detector / risk model / switch model。包含该字段会把 correctness-derived 标签信息泄漏进模型，不适合作为正式实验结论。",
        "",
        "## 1. Stable Hallucination Detector",
        "",
        "| Method | AUC | F1 | Balanced Acc | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in detector_summary:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['auc_hallucination_high']):.3f}` | `{safe_float(row['f1']):.3f}` | `{safe_float(row['balanced_accuracy']):.3f}` | `{safe_float(row['precision']):.3f}` | `{safe_float(row['recall']):.3f}` |"
        )
    lines.extend(["", "## 2. Damage Veto Controller", "", "| Method | Delta | Improved | Damaged | Net | Veto | Prevented Damage | Blocked Rescue |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in veto_summary:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['delta_vs_sample0']):.2%}` | `{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{int(row['net_gain_count'])}` | `{int(safe_float(row.get('veto_count', 0)))}` | `{int(safe_float(row.get('prevented_damage_count', 0)))}` | `{int(safe_float(row.get('blocked_rescue_count', 0)))}` |"
        )
    best_detector = max(detector_summary, key=lambda row: safe_float(row["auc_hallucination_high"]))
    best_veto = max(veto_summary, key=lambda row: (safe_float(row["delta_vs_sample0"]), -safe_float(row.get("damaged_count", 0))))
    lines.extend(
        [
            "",
            "## 3. 初步结论",
            "",
            f"- 最强 hallucination detector 是 `{best_detector['method']}`，AUC `{safe_float(best_detector['auc_hallucination_high']):.3f}`，F1 `{safe_float(best_detector['f1']):.3f}`。",
            f"- 当前最优 controller 变体是 `{best_veto['method']}`，delta `{safe_float(best_veto['delta_vs_sample0']):.2%}`，damage `{int(best_veto['damaged_count'])}`。",
            "- 如果 veto 明显降低 damage 且没有吃掉 rescue，可继续接入 adaptive controller；否则 waveform 更适合作为机制分析、case-level 解释，或后续更细粒度 prefix-token 模型的输入。",
        ]
    )
    (output_dir / "waveform_hallucination_veto_findings_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    rows, by_pair = build_rows(Path(args.waveform_run), Path(args.candidate_run))
    detector_summary, detector_folds, _risk_models = run_detector_cv(rows, by_pair)
    veto_summary, veto_folds, selections, weights = run_veto_cv(by_pair)
    write_csv(output_dir / "hallucination_detector_cv_summary.csv", detector_summary)
    write_csv(output_dir / "hallucination_detector_cv_folds.csv", detector_folds)
    write_csv(output_dir / "veto_controller_cv_summary.csv", veto_summary)
    write_csv(output_dir / "veto_controller_cv_folds.csv", veto_folds)
    write_csv(output_dir / "veto_case_audit.csv", selections)
    write_csv(output_dir / "learned_weights.csv", weights)
    make_plots(output_dir, detector_summary, veto_summary)
    write_report(output_dir, detector_summary, veto_summary)
    write_json(
        output_dir / "run_metadata.json",
        {
            "waveform_run": args.waveform_run,
            "candidate_run": args.candidate_run,
            "basin_rows": len(rows),
            "pairs": len(by_pair),
            "feature_sets": {key: len(value) for key, value in FEATURE_SETS.items()},
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows), "pairs": len(by_pair)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
