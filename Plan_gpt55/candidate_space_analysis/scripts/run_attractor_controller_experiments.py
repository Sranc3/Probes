#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attractor-controller offline experiments.")
    parser.add_argument("--input-run", default=DEFAULT_INPUT_RUN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="attractor_controller")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: flatten(row.get(key)) for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def flatten(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def cohen_d(pos: list[float], neg: list[float]) -> float:
    if len(pos) < 2 or len(neg) < 2:
        return 0.0
    pooled = ((len(pos) - 1) * stdev(pos) ** 2 + (len(neg) - 1) * stdev(neg) ** 2) / (len(pos) + len(neg) - 2)
    return (mean(pos) - mean(neg)) / math.sqrt(pooled) if pooled > 0 else 0.0


def auc_score(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.5
    wins = 0.0
    total = 0
    for p_value in pos:
        for n_value in neg:
            if p_value > n_value:
                wins += 1.0
            elif p_value == n_value:
                wins += 0.5
            total += 1
    return wins / total if total else 0.5


def entropy(values: list[float]) -> float:
    total = sum(max(value, 0.0) for value in values)
    if total <= 0:
        return 0.0
    probs = [max(value, 0.0) / total for value in values if value > 0]
    return -sum(prob * math.log(prob) for prob in probs)


def group_candidates(rows: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["question_id"]))].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row["sample_index"]))
    return dict(grouped)


def add_candidate_features(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["logprob_z"] = safe_float(row["logprob_avg_z"])
        row["neg_entropy_z"] = -safe_float(row["token_mean_entropy_z"])
        row["neg_max_entropy_z"] = -safe_float(row["token_max_entropy_z"])
        row["neg_len_z"] = -safe_float(row["token_count_z"])
        row["cluster_size_z_feature"] = safe_float(row["cluster_size_z"])
        row["cluster_weight_z_feature"] = safe_float(row["cluster_weight_mass_z"])
        row["neg_cluster_rank"] = -safe_float(row["cluster_size_rank"])
        row["neg_logprob_rank"] = -safe_float(row["logprob_rank"])
        row["neg_low_entropy_rank"] = -safe_float(row["low_entropy_rank"])


def build_attractor_tables(candidate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[int, str, int], dict[str, Any]]]:
    grouped = group_candidates(candidate_rows)
    attractor_rows: list[dict[str, Any]] = []
    attractor_lookup: dict[tuple[int, str, int], dict[str, Any]] = {}
    for (seed, question_id), group in grouped.items():
        sample0 = next(row for row in group if int(row["sample_index"]) == 0)
        sample0_cluster_id = int(sample0["cluster_id"])
        cluster_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            cluster_groups[int(row["cluster_id"])].append(row)
        cluster_weights = [safe_float(rows[0]["cluster_weight_mass"]) for rows in cluster_groups.values()]
        fragmentation_entropy = entropy(cluster_weights)
        sample0_cluster = cluster_groups[sample0_cluster_id]
        sample0_centroid = {
            "logprob_z": mean([safe_float(row["logprob_avg_z"]) for row in sample0_cluster]),
            "entropy_z": mean([safe_float(row["token_mean_entropy_z"]) for row in sample0_cluster]),
            "cluster_size_z": mean([safe_float(row["cluster_size_z"]) for row in sample0_cluster]),
            "cluster_weight_z": mean([safe_float(row["cluster_weight_mass_z"]) for row in sample0_cluster]),
        }
        sample0_correct = safe_float(sample0["strict_correct"]) > 0.0
        for cluster_id, cluster_rows in sorted(cluster_groups.items()):
            correct_count = sum(1 for row in cluster_rows if safe_float(row["strict_correct"]) > 0.0)
            wrong_count = len(cluster_rows) - correct_count
            centroid_logprob_z = mean([safe_float(row["logprob_avg_z"]) for row in cluster_rows])
            centroid_entropy_z = mean([safe_float(row["token_mean_entropy_z"]) for row in cluster_rows])
            centroid_cluster_size_z = mean([safe_float(row["cluster_size_z"]) for row in cluster_rows])
            centroid_cluster_weight_z = mean([safe_float(row["cluster_weight_mass_z"]) for row in cluster_rows])
            distance_to_sample0_basin = math.sqrt(
                (centroid_logprob_z - sample0_centroid["logprob_z"]) ** 2
                + (centroid_entropy_z - sample0_centroid["entropy_z"]) ** 2
                + 0.5 * (centroid_cluster_size_z - sample0_centroid["cluster_size_z"]) ** 2
                + 0.5 * (centroid_cluster_weight_z - sample0_centroid["cluster_weight_z"]) ** 2
            )
            contains_sample0 = cluster_id == sample0_cluster_id
            row = {
                "seed": seed,
                "question_id": question_id,
                "question_index": int(float(cluster_rows[0]["question_index"])),
                "question": cluster_rows[0]["question"],
                "cluster_id": cluster_id,
                "cluster_size": len(cluster_rows),
                "cluster_weight_mass": safe_float(cluster_rows[0]["cluster_weight_mass"]),
                "semantic_clusters_set": int(float(cluster_rows[0]["semantic_clusters_set"])),
                "semantic_entropy_weighted_set": safe_float(cluster_rows[0]["semantic_entropy_weighted_set"]),
                "fragmentation_entropy": fragmentation_entropy,
                "centroid_logprob_z": centroid_logprob_z,
                "centroid_entropy_z": centroid_entropy_z,
                "centroid_max_entropy_z": mean([safe_float(row["token_max_entropy_z"]) for row in cluster_rows]),
                "centroid_len_z": mean([safe_float(row["token_count_z"]) for row in cluster_rows]),
                "centroid_cluster_size_z": centroid_cluster_size_z,
                "centroid_cluster_weight_z": centroid_cluster_weight_z,
                "distance_to_sample0_basin": distance_to_sample0_basin,
                "contains_sample0": float(contains_sample0),
                "sample0_cluster_size": len(sample0_cluster),
                "sample0_cluster_weight_mass": safe_float(sample0_cluster[0]["cluster_weight_mass"]),
                "sample0_cluster_correct_rate": mean([safe_float(row["strict_correct"]) for row in sample0_cluster]),
                "sample0_strict_correct": safe_float(sample0["strict_correct"]),
                "correct_count": correct_count,
                "wrong_count": wrong_count,
                "attractor_correct_rate": correct_count / len(cluster_rows),
                "is_pure_correct": float(correct_count == len(cluster_rows)),
                "is_pure_wrong": float(correct_count == 0),
                "is_rescue_attractor": float((not sample0_correct) and correct_count > 0),
                "is_damage_attractor": float(sample0_correct and wrong_count > 0),
                "best_member_logprob_z": max(safe_float(row["logprob_avg_z"]) for row in cluster_rows),
                "best_member_neg_entropy_z": max(-safe_float(row["token_mean_entropy_z"]) for row in cluster_rows),
                "best_member_sample_index": min(cluster_rows, key=lambda row: int(row["logprob_rank"]))["sample_index"],
            }
            attractor_rows.append(row)
            attractor_lookup[(seed, question_id, cluster_id)] = row
    return attractor_rows, attractor_lookup


def attach_attractor_features(candidate_rows: list[dict[str, Any]], attractor_lookup: dict[tuple[int, str, int], dict[str, Any]]) -> None:
    for row in candidate_rows:
        attractor = attractor_lookup[(int(row["seed"]), str(row["question_id"]), int(row["cluster_id"]))]
        row["attractor_centroid_logprob_z"] = attractor["centroid_logprob_z"]
        row["attractor_centroid_entropy_z"] = attractor["centroid_entropy_z"]
        row["attractor_centroid_len_z"] = attractor["centroid_len_z"]
        row["attractor_cluster_size"] = attractor["cluster_size"]
        row["attractor_weight_mass"] = attractor["cluster_weight_mass"]
        row["attractor_cluster_size_z"] = attractor["centroid_cluster_size_z"]
        row["attractor_cluster_weight_z"] = attractor["centroid_cluster_weight_z"]
        row["distance_to_sample0_basin"] = attractor["distance_to_sample0_basin"]
        row["neg_distance_to_sample0_basin"] = -attractor["distance_to_sample0_basin"]
        row["same_sample0_basin"] = attractor["contains_sample0"]
        row["fragmentation_entropy"] = attractor["fragmentation_entropy"]


def summarize_attractor_features(attractor_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = [
        ("is_rescue_attractor", "rescue_vs_nonrescue"),
        ("is_damage_attractor", "damage_vs_nondamage"),
        ("is_pure_correct", "pure_correct_vs_other"),
        ("contains_sample0", "sample0_basin_vs_other"),
    ]
    features = [
        "cluster_size",
        "cluster_weight_mass",
        "fragmentation_entropy",
        "semantic_clusters_set",
        "semantic_entropy_weighted_set",
        "centroid_logprob_z",
        "centroid_entropy_z",
        "centroid_len_z",
        "centroid_cluster_size_z",
        "centroid_cluster_weight_z",
        "distance_to_sample0_basin",
        "sample0_cluster_size",
        "sample0_cluster_weight_mass",
        "sample0_cluster_correct_rate",
        "attractor_correct_rate",
    ]
    summary: list[dict[str, Any]] = []
    for label, comparison in labels:
        pos_rows = [row for row in attractor_rows if safe_float(row[label]) > 0.0]
        neg_rows = [row for row in attractor_rows if safe_float(row[label]) <= 0.0]
        for feature in features:
            pos = [safe_float(row[feature]) for row in pos_rows]
            neg = [safe_float(row[feature]) for row in neg_rows]
            summary.append(
                {
                    "comparison": comparison,
                    "label": label,
                    "feature": feature,
                    "positive_count": len(pos),
                    "negative_count": len(neg),
                    "positive_mean": mean(pos),
                    "negative_mean": mean(neg),
                    "mean_diff": mean(pos) - mean(neg),
                    "cohen_d": cohen_d(pos, neg),
                    "auc_positive_high": auc_score(pos, neg),
                }
            )
    summary.sort(key=lambda row: abs(row["cohen_d"]), reverse=True)
    return summary


def feature_value(row: dict[str, Any], feature: str) -> float:
    if feature.startswith("-"):
        return -safe_float(row[feature[1:]])
    return safe_float(row.get(feature, 0.0))


Formula = dict[str, Any]


def score_row(row: dict[str, Any], formula: Formula) -> float:
    return sum(float(weight) * feature_value(row, feature) for feature, weight in formula["weights"].items())


def choose_candidate(group: list[dict[str, Any]], formula: Formula, prefix_k: int | None = None) -> dict[str, Any]:
    candidates = group[:prefix_k] if prefix_k is not None else group
    return max(candidates, key=lambda row: (score_row(row, formula), -int(row["sample_index"])))


def formulas() -> dict[str, list[Formula]]:
    point = [
        {"name": "point_logprob", "weights": {"logprob_z": 1.0}},
        {"name": "point_logprob_entropy", "weights": {"logprob_z": 1.0, "neg_entropy_z": 0.75}},
        {"name": "point_entropy_len", "weights": {"neg_entropy_z": 1.0, "neg_len_z": 0.25}},
        {"name": "point_balanced", "weights": {"logprob_z": 1.0, "neg_entropy_z": 0.75, "neg_len_z": 0.25}},
        {"name": "point_ranked", "weights": {"neg_logprob_rank": 0.6, "neg_low_entropy_rank": 0.4}},
    ]
    cluster = [
        {"name": "cluster_size", "weights": {"cluster_size_z_feature": 1.0, "logprob_z": 0.05}},
        {"name": "cluster_weight", "weights": {"cluster_weight_z_feature": 1.0, "logprob_z": 0.05}},
        {"name": "cluster_size_entropy", "weights": {"cluster_size_z_feature": 0.8, "cluster_weight_z_feature": 0.8, "neg_entropy_z": 0.2}},
        {"name": "cluster_rank_logprob", "weights": {"neg_cluster_rank": 0.7, "neg_logprob_rank": 0.3}},
    ]
    attractor = [
        {
            "name": "attractor_basin_quality",
            "weights": {
                "attractor_cluster_size_z": 0.75,
                "attractor_cluster_weight_z": 0.75,
                "attractor_centroid_logprob_z": 0.8,
                "attractor_centroid_entropy_z": -0.4,
                "logprob_z": 0.4,
                "neg_len_z": 0.15,
            },
        },
        {
            "name": "attractor_safe_switch",
            "weights": {
                "attractor_cluster_weight_z": 0.9,
                "attractor_centroid_logprob_z": 0.9,
                "attractor_centroid_entropy_z": -0.7,
                "distance_to_sample0_basin": 0.15,
                "neg_len_z": 0.2,
            },
        },
        {
            "name": "attractor_local_basin",
            "weights": {
                "attractor_cluster_size_z": 0.6,
                "attractor_centroid_logprob_z": 0.9,
                "neg_distance_to_sample0_basin": 0.15,
                "neg_entropy_z": 0.3,
            },
        },
        {
            "name": "attractor_point_cluster",
            "weights": {
                "logprob_z": 0.8,
                "neg_entropy_z": 0.5,
                "cluster_size_z_feature": 0.5,
                "cluster_weight_z_feature": 0.5,
                "neg_len_z": 0.15,
            },
        },
    ]
    return {"point": point, "cluster": cluster, "attractor": attractor}


def evaluate_selected(groups: list[list[dict[str, Any]]], selected_rows: list[dict[str, Any]], method: str, split: str) -> dict[str, Any]:
    sample0_rows = [next(row for row in group if int(row["sample_index"]) == 0) for group in groups]
    deltas = [safe_float(sel["strict_correct"]) - safe_float(base["strict_correct"]) for sel, base in zip(selected_rows, sample0_rows)]
    return {
        "split": split,
        "method": method,
        "pairs": len(groups),
        "strict_correct_rate": mean([safe_float(row["strict_correct"]) for row in selected_rows]),
        "sample0_strict_correct_rate": mean([safe_float(row["strict_correct"]) for row in sample0_rows]),
        "delta_vs_sample0": mean(deltas),
        "improved_count": sum(1 for value in deltas if value > 0.0),
        "damaged_count": sum(1 for value in deltas if value < 0.0),
        "net_gain_count": sum(1 for value in deltas if value > 0.0) - sum(1 for value in deltas if value < 0.0),
        "answer_changed_rate": mean([float(sel["answer_text"] != base["answer_text"]) for sel, base in zip(selected_rows, sample0_rows)]),
        "rescue_recall": rescue_recall(groups, selected_rows),
        "damage_avoidance": damage_avoidance(groups, selected_rows),
        "avg_selected_sample_index": mean([float(row["sample_index"]) for row in selected_rows]),
    }


def rescue_recall(groups: list[list[dict[str, Any]]], selected_rows: list[dict[str, Any]]) -> float:
    rescue_groups = 0
    captured = 0
    for group, selected in zip(groups, selected_rows):
        sample0 = next(row for row in group if int(row["sample_index"]) == 0)
        if safe_float(sample0["strict_correct"]) > 0.0:
            continue
        if any(safe_float(row["strict_correct"]) > 0.0 for row in group):
            rescue_groups += 1
            captured += int(safe_float(selected["strict_correct"]) > 0.0)
    return captured / rescue_groups if rescue_groups else 0.0


def damage_avoidance(groups: list[list[dict[str, Any]]], selected_rows: list[dict[str, Any]]) -> float:
    initially_correct = 0
    avoided = 0
    for group, selected in zip(groups, selected_rows):
        sample0 = next(row for row in group if int(row["sample_index"]) == 0)
        if safe_float(sample0["strict_correct"]) <= 0.0:
            continue
        initially_correct += 1
        avoided += int(safe_float(selected["strict_correct"]) > 0.0)
    return avoided / initially_correct if initially_correct else 0.0


def grouped_folds(grouped: dict[tuple[int, str], list[dict[str, Any]]], fold_count: int = 5) -> list[tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]]:
    question_ids = sorted({question_id for _seed, question_id in grouped})
    folds: list[tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]] = []
    for fold_idx in range(fold_count):
        test_questions = {qid for idx, qid in enumerate(question_ids) if idx % fold_count == fold_idx}
        train = [group for (_seed, qid), group in grouped.items() if qid not in test_questions]
        test = [group for (_seed, qid), group in grouped.items() if qid in test_questions]
        folds.append((train, test))
    return folds


def train_best_formula(groups: list[list[dict[str, Any]]], candidate_formulas: list[Formula]) -> Formula:
    best_formula = candidate_formulas[0]
    best_score = -1e9
    for formula in candidate_formulas:
        selected = [choose_candidate(group, formula) for group in groups]
        metrics = evaluate_selected(groups, selected, formula["name"], "train")
        objective = metrics["delta_vs_sample0"] + 0.01 * metrics["net_gain_count"] - 0.005 * metrics["damaged_count"]
        if objective > best_score:
            best_formula = formula
            best_score = objective
    return best_formula


def run_feature_comparison(grouped: dict[tuple[int, str], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    formula_sets = formulas()
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for fold_idx, (train_groups, test_groups) in enumerate(grouped_folds(grouped)):
        for family, family_formulas in formula_sets.items():
            best_formula = train_best_formula(train_groups, family_formulas)
            selected = [choose_candidate(group, best_formula) for group in test_groups]
            metrics = evaluate_selected(test_groups, selected, family, f"question_fold_{fold_idx}")
            metrics["selected_formula"] = best_formula["name"]
            fold_rows.append(metrics)
            for group, selected_row in zip(test_groups, selected):
                sample0 = next(row for row in group if int(row["sample_index"]) == 0)
                selection_rows.append(
                    {
                        "split": f"question_fold_{fold_idx}",
                        "family": family,
                        "formula": best_formula["name"],
                        "seed": selected_row["seed"],
                        "question_id": selected_row["question_id"],
                        "selected_sample_index": selected_row["sample_index"],
                        "sample0_strict_correct": sample0["strict_correct"],
                        "selected_strict_correct": selected_row["strict_correct"],
                        "delta_strict_correct": safe_float(selected_row["strict_correct"]) - safe_float(sample0["strict_correct"]),
                        "selected_cluster_id": selected_row["cluster_id"],
                        "sample0_cluster_id": sample0["cluster_id"],
                        "selected_preview": selected_row["answer_preview"],
                    }
                )
    summary_rows = summarize_metric_rows(fold_rows, key="method", split_name="question_grouped_cv")
    return summary_rows + fold_rows, selection_rows


def summarize_metric_rows(rows: list[dict[str, Any]], key: str, split_name: str) -> list[dict[str, Any]]:
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row[key])].append(row)
    summary: list[dict[str, Any]] = []
    for name, items in sorted(grouped_rows.items()):
        summary.append(
            {
                "split": split_name,
                key: name,
                "pairs": sum(int(item["pairs"]) for item in items),
                "strict_correct_rate": mean([safe_float(item["strict_correct_rate"]) for item in items]),
                "sample0_strict_correct_rate": mean([safe_float(item["sample0_strict_correct_rate"]) for item in items]),
                "delta_vs_sample0": mean([safe_float(item["delta_vs_sample0"]) for item in items]),
                "improved_count": sum(int(item["improved_count"]) for item in items),
                "damaged_count": sum(int(item["damaged_count"]) for item in items),
                "net_gain_count": sum(int(item["net_gain_count"]) for item in items),
                "answer_changed_rate": mean([safe_float(item["answer_changed_rate"]) for item in items]),
                "rescue_recall": mean([safe_float(item["rescue_recall"]) for item in items]),
                "damage_avoidance": mean([safe_float(item["damage_avoidance"]) for item in items]),
                "avg_selected_sample_index": mean([safe_float(item["avg_selected_sample_index"]) for item in items]),
                "selected_formula": most_common([str(item.get("selected_formula", "")) for item in items]),
            }
        )
    return summary


def most_common(values: list[str]) -> str:
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]


RiskFormula = dict[str, Any]


def risk_formulas() -> list[RiskFormula]:
    return [
        {
            "name": "sample0_weak_or_fragmented",
            "weights": {
                "sample0_entropy_z": 0.8,
                "sample0_neg_logprob_z": 0.8,
                "sample0_neg_cluster_size_z": 0.7,
                "fragmentation_entropy": 0.5,
            },
        },
        {
            "name": "basin_instability",
            "weights": {
                "sample0_neg_cluster_weight_z": 0.9,
                "fragmentation_entropy": 0.8,
                "semantic_entropy_weighted_set": 0.3,
                "sample0_entropy_z": 0.4,
            },
        },
        {
            "name": "conservative_switch",
            "weights": {
                "sample0_neg_logprob_z": 0.6,
                "sample0_entropy_z": 0.6,
                "sample0_neg_cluster_size_z": 0.4,
            },
        },
    ]


def pair_features(group: list[dict[str, Any]]) -> dict[str, float]:
    sample0 = next(row for row in group if int(row["sample_index"]) == 0)
    return {
        "sample0_entropy_z": safe_float(sample0["token_mean_entropy_z"]),
        "sample0_neg_logprob_z": -safe_float(sample0["logprob_avg_z"]),
        "sample0_neg_cluster_size_z": -safe_float(sample0["cluster_size_z"]),
        "sample0_neg_cluster_weight_z": -safe_float(sample0["cluster_weight_mass_z"]),
        "fragmentation_entropy": safe_float(sample0["fragmentation_entropy"]),
        "semantic_entropy_weighted_set": safe_float(sample0["semantic_entropy_weighted_set"]),
    }


def risk_score(group: list[dict[str, Any]], formula: RiskFormula) -> float:
    features = pair_features(group)
    return sum(float(weight) * features.get(feature, 0.0) for feature, weight in formula["weights"].items())


def quantiles(values: list[float]) -> list[float]:
    if not values:
        return [0.0]
    sorted_values = sorted(values)
    qs = [0.25, 0.4, 0.5, 0.6, 0.75]
    return [sorted_values[min(len(sorted_values) - 1, max(0, int(q * (len(sorted_values) - 1))))] for q in qs]


def choose_two_stage(
    group: list[dict[str, Any]],
    risk_formula: RiskFormula,
    risk_threshold: float,
    selector_formula: Formula,
    alt_margin: float,
    prefix_k: int | None = None,
) -> dict[str, Any]:
    candidates = group[:prefix_k] if prefix_k is not None else group
    sample0 = next(row for row in candidates if int(row["sample_index"]) == 0)
    if risk_score(candidates, risk_formula) < risk_threshold:
        return sample0
    alternatives = [row for row in candidates if int(row["cluster_id"]) != int(sample0["cluster_id"])]
    if not alternatives:
        return sample0
    best_alt = max(alternatives, key=lambda row: (score_row(row, selector_formula), -int(row["sample_index"])))
    if score_row(best_alt, selector_formula) < score_row(sample0, selector_formula) + alt_margin:
        return sample0
    return best_alt


def train_two_stage(groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
    selector_candidates = formulas()["attractor"] + formulas()["cluster"]
    best_config: dict[str, Any] = {}
    best_objective = -1e9
    for risk_formula in risk_formulas():
        risks = [risk_score(group, risk_formula) for group in groups]
        for risk_threshold in quantiles(risks):
            for selector_formula in selector_candidates:
                for alt_margin in [-0.5, -0.25, 0.0, 0.25, 0.5]:
                    selected = [choose_two_stage(group, risk_formula, risk_threshold, selector_formula, alt_margin) for group in groups]
                    metrics = evaluate_selected(groups, selected, "two_stage", "train")
                    objective = (
                        metrics["delta_vs_sample0"]
                        + 0.012 * metrics["net_gain_count"]
                        - 0.02 * metrics["damaged_count"]
                        + 0.01 * metrics["damage_avoidance"]
                    )
                    if objective > best_objective:
                        best_objective = objective
                        best_config = {
                            "risk_formula": risk_formula,
                            "risk_threshold": risk_threshold,
                            "selector_formula": selector_formula,
                            "alt_margin": alt_margin,
                            "train_objective": best_objective,
                        }
    return best_config


def run_two_stage_cv(grouped: dict[tuple[int, str], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    selected_rows_detail: list[dict[str, Any]] = []
    for fold_idx, (train_groups, test_groups) in enumerate(grouped_folds(grouped)):
        config = train_two_stage(train_groups)
        selected = [
            choose_two_stage(
                group,
                config["risk_formula"],
                config["risk_threshold"],
                config["selector_formula"],
                config["alt_margin"],
            )
            for group in test_groups
        ]
        metrics = evaluate_selected(test_groups, selected, "two_stage_controller", f"question_fold_{fold_idx}")
        metrics["risk_formula"] = config["risk_formula"]["name"]
        metrics["selector_formula"] = config["selector_formula"]["name"]
        metrics["risk_threshold"] = config["risk_threshold"]
        metrics["alt_margin"] = config["alt_margin"]
        fold_rows.append(metrics)
        for group, chosen in zip(test_groups, selected):
            sample0 = next(row for row in group if int(row["sample_index"]) == 0)
            selected_rows_detail.append(
                {
                    "split": f"question_fold_{fold_idx}",
                    "seed": chosen["seed"],
                    "question_id": chosen["question_id"],
                    "selected_sample_index": chosen["sample_index"],
                    "sample0_strict_correct": sample0["strict_correct"],
                    "selected_strict_correct": chosen["strict_correct"],
                    "delta_strict_correct": safe_float(chosen["strict_correct"]) - safe_float(sample0["strict_correct"]),
                    "selected_cluster_id": chosen["cluster_id"],
                    "sample0_cluster_id": sample0["cluster_id"],
                    "risk_formula": config["risk_formula"]["name"],
                    "selector_formula": config["selector_formula"]["name"],
                    "risk_score": risk_score(group, config["risk_formula"]),
                    "answer_changed": float(chosen["answer_text"] != sample0["answer_text"]),
                    "selected_preview": chosen["answer_preview"],
                }
            )
    return summarize_metric_rows(fold_rows, key="method", split_name="question_grouped_cv") + fold_rows, selected_rows_detail


def run_sequential_simulation(grouped: dict[tuple[int, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kmax in [2, 3, 4, 8]:
        fold_metrics: list[dict[str, Any]] = []
        for fold_idx, (train_groups, test_groups) in enumerate(grouped_folds(grouped)):
            config = train_two_stage(train_groups)
            selected: list[dict[str, Any]] = []
            costs: list[float] = []
            for group in test_groups:
                chosen, cost = adaptive_select(group, config, kmax)
                selected.append(chosen)
                costs.append(cost)
            metrics = evaluate_selected(test_groups, selected, f"adaptive_k{kmax}", f"question_fold_{fold_idx}")
            metrics["avg_generated_candidates"] = mean(costs)
            metrics["max_candidates"] = kmax
            metrics["risk_formula"] = config["risk_formula"]["name"]
            metrics["selector_formula"] = config["selector_formula"]["name"]
            fold_metrics.append(metrics)
        summary = summarize_metric_rows(fold_metrics, key="method", split_name="question_grouped_cv")
        for summary_row in summary:
            matching = [row for row in fold_metrics if row["method"] == summary_row["method"]]
            summary_row["avg_generated_candidates"] = mean([safe_float(row["avg_generated_candidates"]) for row in matching])
            summary_row["max_candidates"] = kmax
        rows.extend(summary)
        rows.extend(fold_metrics)
    return rows


def adaptive_select(group: list[dict[str, Any]], config: dict[str, Any], kmax: int) -> tuple[dict[str, Any], int]:
    sample0 = group[0]
    for current_k in range(1, kmax + 1):
        prefix = group[:current_k]
        sample0_prefix_cluster_size = sum(1 for row in prefix if int(row["cluster_id"]) == int(sample0["cluster_id"]))
        stable_sample0 = (
            current_k >= 2
            and sample0_prefix_cluster_size >= min(2, current_k)
            and safe_float(sample0["token_mean_entropy_z"]) <= 0.0
            and safe_float(sample0["logprob_avg_z"]) >= -0.25
        )
        if stable_sample0:
            return sample0, current_k
        if current_k >= 2:
            chosen = choose_two_stage(
                prefix,
                config["risk_formula"],
                config["risk_threshold"],
                config["selector_formula"],
                config["alt_margin"],
                prefix_k=None,
            )
            if int(chosen["sample_index"]) != 0:
                return chosen, current_k
    return choose_two_stage(
        group[:kmax],
        config["risk_formula"],
        config["risk_threshold"],
        config["selector_formula"],
        config["alt_margin"],
        prefix_k=None,
    ), kmax


def run_seed_split(grouped: dict[tuple[int, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for train_seed, test_seed in [(42, 43), (43, 42)]:
        train_groups = [group for (seed, _qid), group in grouped.items() if seed == train_seed]
        test_groups = [group for (seed, _qid), group in grouped.items() if seed == test_seed]
        for family, family_formulas in formulas().items():
            formula = train_best_formula(train_groups, family_formulas)
            selected = [choose_candidate(group, formula) for group in test_groups]
            metrics = evaluate_selected(test_groups, selected, family, f"train_seed_{train_seed}_test_seed_{test_seed}")
            metrics["selected_formula"] = formula["name"]
            rows.append(metrics)
        config = train_two_stage(train_groups)
        selected = [
            choose_two_stage(group, config["risk_formula"], config["risk_threshold"], config["selector_formula"], config["alt_margin"])
            for group in test_groups
        ]
        metrics = evaluate_selected(test_groups, selected, "two_stage_controller", f"train_seed_{train_seed}_test_seed_{test_seed}")
        metrics["risk_formula"] = config["risk_formula"]["name"]
        metrics["selector_formula"] = config["selector_formula"]["name"]
        rows.append(metrics)
    return rows


def run_pilot_comparison(input_run: Path, current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pilot_path = input_run.parent / "run_20260427_091924_candidate_space_v1" / "candidate_features.csv"
    rows: list[dict[str, Any]] = []
    for label, path_or_rows in [("all200", current_rows), ("pilot_test31", pilot_path)]:
        if isinstance(path_or_rows, Path):
            if not path_or_rows.exists():
                continue
            dataset_rows = read_csv(path_or_rows)
            add_candidate_features(dataset_rows)
        else:
            dataset_rows = path_or_rows
        grouped = group_candidates(dataset_rows)
        groups = list(grouped.values())
        sample0_selected = [group[0] for group in groups]
        rows.append(evaluate_selected(groups, sample0_selected, "sample0", label))
        for family, family_formulas in formulas().items():
            formula = train_best_formula(groups, family_formulas)
            selected = [choose_candidate(group, formula) for group in groups]
            metrics = evaluate_selected(groups, selected, family, label)
            metrics["selected_formula"] = formula["name"]
            rows.append(metrics)
    return rows


def make_plots(output_dir: Path, feature_summary: list[dict[str, Any]], controller_rows: list[dict[str, Any]], sequential_rows: list[dict[str, Any]], robustness_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = ensure_dir(output_dir / "plots")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})

    top = feature_summary[:14]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    labels = [f"{row['comparison']}\n{row['feature']}" for row in top]
    values = [safe_float(row["cohen_d"]) for row in top]
    ax.barh(labels[::-1], values[::-1], color=["#3182bd" if value > 0 else "#de2d26" for value in values[::-1]])
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_xlabel("Cohen's d")
    ax.set_title("Attractor-Level Feature Separability")
    fig.tight_layout()
    fig.savefig(plot_dir / "01_attractor_separability.png")
    plt.close(fig)

    summary_controller = [row for row in controller_rows if row["split"] == "question_grouped_cv"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    names = [row["method"] for row in summary_controller]
    values = [safe_float(row["strict_correct_rate"]) for row in summary_controller]
    ax.bar(names, values, color="#4c78a8", alpha=0.82)
    ax.set_ylim(0, max(values + [0.5]) + 0.08)
    ax.set_ylabel("Strict correctness")
    ax.set_title("Point vs Cluster vs Attractor Selectors")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plot_dir / "02_controller_cv_results.png")
    plt.close(fig)

    seq_summary = [row for row in sequential_rows if row["split"] == "question_grouped_cv"]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot([safe_float(row["avg_generated_candidates"]) for row in seq_summary], [safe_float(row["strict_correct_rate"]) for row in seq_summary], marker="o", lw=1.8)
    for row in seq_summary:
        ax.text(safe_float(row["avg_generated_candidates"]), safe_float(row["strict_correct_rate"]), row["method"], fontsize=8)
    ax.set_xlabel("Average generated candidates")
    ax.set_ylabel("Strict correctness")
    ax.set_title("Adaptive Sampling Cost Curve")
    fig.tight_layout()
    fig.savefig(plot_dir / "03_sequential_cost_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = [f"{row['split']}\n{row['method']}" for row in robustness_rows]
    values = [safe_float(row["delta_vs_sample0"]) for row in robustness_rows]
    ax.barh(labels[::-1], values[::-1], color=["#31a354" if value >= 0 else "#de2d26" for value in values[::-1]])
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_xlabel("Delta vs sample0")
    ax.set_title("Robustness Checks")
    fig.tight_layout()
    fig.savefig(plot_dir / "04_robustness_summary.png")
    plt.close(fig)


def build_report(
    output_dir: Path,
    attractor_rows: list[dict[str, Any]],
    feature_summary: list[dict[str, Any]],
    controller_rows: list[dict[str, Any]],
    two_stage_rows: list[dict[str, Any]],
    sequential_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
) -> str:
    cv_summary = [row for row in controller_rows if row["split"] == "question_grouped_cv"]
    two_stage_summary = [row for row in two_stage_rows if row["split"] == "question_grouped_cv"]
    seq_summary = [row for row in sequential_rows if row["split"] == "question_grouped_cv"]
    top_features = feature_summary[:8]
    lines = [
        "# Attractor Controller Experiments",
        "",
        "## Scale",
        "",
        f"- Semantic attractors: `{len(attractor_rows)}`",
        f"- Rescue attractors: `{sum(1 for row in attractor_rows if safe_float(row['is_rescue_attractor']) > 0.0)}`",
        f"- Damage attractors: `{sum(1 for row in attractor_rows if safe_float(row['is_damage_attractor']) > 0.0)}`",
        "",
        "## Strongest Attractor-Level Signals",
        "",
        "| Comparison | Feature | Cohen d | AUC | Pos mean | Neg mean |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in top_features:
        lines.append(
            f"| `{row['comparison']}` | `{row['feature']}` | `{safe_float(row['cohen_d']):.3f}` | "
            f"`{safe_float(row['auc_positive_high']):.3f}` | `{safe_float(row['positive_mean']):.4f}` | `{safe_float(row['negative_mean']):.4f}` |"
        )
    lines.extend(["", "## Point vs Cluster vs Attractor Selectors", "", "| Method | Strict | Delta | Improved | Damaged | Rescue Recall | Damage Avoid |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in cv_summary:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{safe_float(row['rescue_recall']):.2%}` | `{safe_float(row['damage_avoidance']):.2%}` |"
        )
    lines.extend(["", "## Two-Stage Controller", "", "| Method | Strict | Delta | Improved | Damaged | Rescue Recall | Damage Avoid |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in two_stage_summary:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{safe_float(row['rescue_recall']):.2%}` | `{safe_float(row['damage_avoidance']):.2%}` |"
        )
    lines.extend(["", "## Adaptive Sequential Sampling", "", "| Policy | Avg candidates | Strict | Delta | Improved | Damaged |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in seq_summary:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row.get('avg_generated_candidates', 0.0)):.2f}` | `{safe_float(row['strict_correct_rate']):.2%}` | "
            f"`{safe_float(row['delta_vs_sample0']):.2%}` | `{int(row['improved_count'])}` | `{int(row['damaged_count'])}` |"
        )
    lines.extend(["", "## Robustness", "", "| Split | Method | Strict | Delta | Improved | Damaged |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in robustness_rows:
        lines.append(
            f"| `{row['split']}` | `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Attractor-level features are useful if they improve rescue recall without increasing damaged cases.",
            "- The two-stage controller is the publishable direction only if it beats point-only selection under grouped validation.",
            "- The sequential policy is the practical direction only if it recovers some gains at average cost below blind best-of-8.",
            "",
            f"Output directory: `{output_dir}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (output_dir / "controller_experiment_report_zh.md").write_text(report, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    input_run = Path(args.input_run)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(args.output_root) / f"run_{timestamp}_{args.tag}")
    candidate_rows = read_csv(input_run / "candidate_features.csv")
    add_candidate_features(candidate_rows)

    attractor_rows, attractor_lookup = build_attractor_tables(candidate_rows)
    attach_attractor_features(candidate_rows, attractor_lookup)
    grouped = group_candidates(candidate_rows)

    feature_summary = summarize_attractor_features(attractor_rows)
    controller_rows, controller_selection_rows = run_feature_comparison(grouped)
    two_stage_rows, two_stage_selection_rows = run_two_stage_cv(grouped)
    sequential_rows = run_sequential_simulation(grouped)
    robustness_rows = run_seed_split(grouped) + run_pilot_comparison(input_run, candidate_rows)

    write_csv(output_dir / "attractor_cluster_table.csv", attractor_rows)
    write_csv(output_dir / "attractor_feature_summary.csv", feature_summary)
    write_csv(output_dir / "controller_cv_results.csv", controller_rows + two_stage_rows)
    write_csv(output_dir / "controller_selection_rows.csv", controller_selection_rows + two_stage_selection_rows)
    write_csv(output_dir / "sequential_cost_curve.csv", sequential_rows)
    write_csv(output_dir / "robustness_checks.csv", robustness_rows)
    write_json(
        output_dir / "run_metadata.json",
        {
            "input_run": str(input_run),
            "candidate_count": len(candidate_rows),
            "attractor_count": len(attractor_rows),
            "group_count": len(grouped),
            "tag": args.tag,
        },
    )
    make_plots(output_dir, feature_summary, controller_rows + two_stage_rows, sequential_rows, robustness_rows)
    build_report(output_dir, attractor_rows, feature_summary, controller_rows, two_stage_rows, sequential_rows, robustness_rows)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "candidate_count": len(candidate_rows),
                "attractor_count": len(attractor_rows),
                "group_count": len(grouped),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
