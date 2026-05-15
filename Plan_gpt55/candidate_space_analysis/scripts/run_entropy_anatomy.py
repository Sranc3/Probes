#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "the",
    "their",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze entropy anatomy of answer basins.")
    parser.add_argument("--input-run", default=DEFAULT_INPUT_RUN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="entropy_anatomy")
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
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: flatten(row.get(key, "")) for key in fieldnames})


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


def entropy_from_weights(values: list[float]) -> float:
    total = sum(max(value, 0.0) for value in values)
    if total <= 0:
        return 0.0
    probs = [max(value, 0.0) / total for value in values if value > 0]
    return -sum(prob * math.log(prob) for prob in probs)


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


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def content_tokens(text: str) -> list[str]:
    return [token for token in normalize(text).split() if token not in STOPWORDS and len(token) > 2]


def token_set(text: str) -> set[str]:
    return set(content_tokens(text))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def lexical_entropy(texts: list[str]) -> float:
    counts = Counter(token for text in texts for token in content_tokens(text))
    return entropy_from_weights([float(value) for value in counts.values()])


def pairwise_jaccard(texts: list[str]) -> float:
    sets = [token_set(text) for text in texts]
    if len(sets) < 2:
        return 1.0
    return mean([jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))])


def group_candidates(rows: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["question_id"]))].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row["sample_index"]))
    return dict(grouped)


def rank_values(values: list[float], reverse: bool) -> list[int]:
    ordered = sorted(range(len(values)), key=lambda idx: values[idx], reverse=reverse)
    ranks = [0] * len(values)
    for rank, idx in enumerate(ordered, start=1):
        ranks[idx] = rank
    return ranks


def top2_margin(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    ordered = sorted(values, reverse=True)
    return ordered[0] - ordered[1]


def build_basin_tables(candidate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = group_candidates(candidate_rows)
    basin_rows: list[dict[str, Any]] = []
    enriched_candidates: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    for (seed, question_id), group in sorted(grouped.items()):
        sample0 = next(row for row in group if int(row["sample_index"]) == 0)
        sample0_cluster_id = int(sample0["cluster_id"])
        cluster_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            cluster_groups[int(row["cluster_id"])].append(row)

        cluster_weights = [safe_float(rows[0]["cluster_weight_mass"]) for rows in cluster_groups.values()]
        cluster_sizes = [float(len(rows)) for rows in cluster_groups.values()]
        cluster_mean_logprobs = [mean([safe_float(row["logprob_avg_z"]) for row in rows]) for rows in cluster_groups.values()]
        cluster_mean_entropies = [mean([safe_float(row["token_mean_entropy_z"]) for row in rows]) for rows in cluster_groups.values()]
        fragmentation_entropy = entropy_from_weights(cluster_weights)
        normalized_fragmentation_entropy = fragmentation_entropy / math.log(len(cluster_weights)) if len(cluster_weights) > 1 else 0.0
        top2_weight_margin = top2_margin(cluster_weights)
        top2_logprob_margin = top2_margin(cluster_mean_logprobs)
        top2_low_entropy_margin = top2_margin([-value for value in cluster_mean_entropies])

        sample0_cluster = cluster_groups[sample0_cluster_id]
        sample0_centroid_entropy_z = mean([safe_float(row["token_mean_entropy_z"]) for row in sample0_cluster])
        sample0_centroid_logprob_z = mean([safe_float(row["logprob_avg_z"]) for row in sample0_cluster])

        for cluster_id, members in sorted(cluster_groups.items()):
            correct_count = sum(1 for row in members if safe_float(row["strict_correct"]) > 0)
            wrong_count = len(members) - correct_count
            correct_rate = correct_count / len(members)
            texts = [str(row["answer_text"]) for row in members]
            token_mean_values = [safe_float(row["token_mean_entropy"]) for row in members]
            token_max_values = [safe_float(row["token_max_entropy"]) for row in members]
            logprob_values = [safe_float(row["logprob_avg"]) for row in members]
            token_count_values = [safe_float(row["token_count"]) for row in members]
            entropy_z_values = [safe_float(row["token_mean_entropy_z"]) for row in members]
            logprob_z_values = [safe_float(row["logprob_avg_z"]) for row in members]
            weight = safe_float(members[0]["cluster_weight_mass"])
            size = len(members)
            centroid_entropy_z = mean(entropy_z_values)
            centroid_logprob_z = mean(logprob_z_values)
            lexical_ent = lexical_entropy(texts)
            within_jaccard = pairwise_jaccard(texts)
            internal_token_entropy_std = stdev(token_mean_values)
            stable_score = (
                0.45 * weight
                + 0.25 * (size / 8.0)
                + 0.15 * max(0.0, centroid_logprob_z)
                + 0.15 * max(0.0, -centroid_entropy_z)
            )
            low_entropy_score = max(0.0, -centroid_entropy_z) + max(0.0, -mean(token_max_values) + 2.0) * 0.05
            stable_hallucination_score = stable_score * float(correct_rate == 0.0) * (1.0 + low_entropy_score)
            distance_to_sample0 = math.sqrt(
                (centroid_entropy_z - sample0_centroid_entropy_z) ** 2
                + (centroid_logprob_z - sample0_centroid_logprob_z) ** 2
            )
            is_rescue = float(safe_float(sample0["strict_correct"]) == 0.0 and correct_count > 0)
            is_damage = float(safe_float(sample0["strict_correct"]) > 0.0 and wrong_count > 0)
            is_mixed = float(correct_count > 0 and wrong_count > 0)
            is_stable = float((size >= 2 or weight >= 0.25) and centroid_entropy_z <= 0.25 and centroid_logprob_z >= -0.35)
            is_stable_correct = float(correct_rate >= 0.8 and is_stable > 0.0)
            is_stable_hallucination = float(correct_rate == 0.0 and is_stable > 0.0)
            is_unstable_wrong = float(correct_rate == 0.0 and is_stable <= 0.0)
            regime = regime_label(
                is_stable_correct=is_stable_correct,
                is_stable_hallucination=is_stable_hallucination,
                is_unstable_wrong=is_unstable_wrong,
                is_rescue=is_rescue,
                is_damage=is_damage,
                is_mixed=is_mixed,
            )
            basin_row = {
                "seed": seed,
                "question_id": question_id,
                "question_index": int(float(sample0["question_index"])),
                "question": sample0["question"],
                "cluster_id": cluster_id,
                "contains_sample0": float(cluster_id == sample0_cluster_id),
                "sample0_strict_correct": safe_float(sample0["strict_correct"]),
                "sample0_cluster_id": sample0_cluster_id,
                "cluster_size": size,
                "cluster_weight_mass": weight,
                "correct_count": correct_count,
                "wrong_count": wrong_count,
                "basin_correct_rate": correct_rate,
                "is_pure_correct": float(correct_rate == 1.0),
                "is_pure_wrong": float(correct_rate == 0.0),
                "is_mixed_basin": is_mixed,
                "is_rescue_basin": is_rescue,
                "is_damage_basin": is_damage,
                "is_stable_correct_basin": is_stable_correct,
                "is_stable_hallucination_basin": is_stable_hallucination,
                "is_unstable_wrong_basin": is_unstable_wrong,
                "basin_regime": regime,
                "stable_score": stable_score,
                "low_entropy_score": low_entropy_score,
                "stable_hallucination_score": stable_hallucination_score,
                "token_mean_entropy_mean": mean(token_mean_values),
                "token_mean_entropy_std": stdev(token_mean_values),
                "token_max_entropy_mean": mean(token_max_values),
                "token_max_entropy_std": stdev(token_max_values),
                "logprob_avg_mean": mean(logprob_values),
                "logprob_avg_std": stdev(logprob_values),
                "token_count_mean": mean(token_count_values),
                "centroid_entropy_z": centroid_entropy_z,
                "centroid_max_entropy_z": mean([safe_float(row["token_max_entropy_z"]) for row in members]),
                "centroid_logprob_z": centroid_logprob_z,
                "centroid_len_z": mean([safe_float(row["token_count_z"]) for row in members]),
                "internal_token_entropy_std": internal_token_entropy_std,
                "within_basin_lexical_entropy": lexical_ent,
                "within_basin_jaccard": within_jaccard,
                "semantic_entropy_weighted_set": safe_float(sample0["semantic_entropy_weighted_set"]),
                "semantic_clusters_set": int(float(sample0["semantic_clusters_set"])),
                "fragmentation_entropy": fragmentation_entropy,
                "normalized_fragmentation_entropy": normalized_fragmentation_entropy,
                "top2_weight_margin": top2_weight_margin,
                "top2_logprob_margin": top2_logprob_margin,
                "top2_low_entropy_margin": top2_low_entropy_margin,
                "distance_to_sample0_entropy_logprob": distance_to_sample0,
                "answer_preview": members[0]["answer_preview"],
            }
            basin_rows.append(basin_row)
            for row in members:
                candidate = dict(row)
                candidate.update(
                    {
                        "basin_regime": regime,
                        "basin_correct_rate": correct_rate,
                        "basin_stable_score": stable_score,
                        "basin_stable_hallucination_score": stable_hallucination_score,
                        "basin_centroid_entropy_z": centroid_entropy_z,
                        "basin_fragmentation_entropy": fragmentation_entropy,
                    }
                )
                enriched_candidates.append(candidate)
            if is_rescue > 0.0 or is_damage > 0.0:
                switch_rows.append(
                    {
                        "seed": seed,
                        "question_id": question_id,
                        "cluster_id": cluster_id,
                        "switch_type": "rescue" if is_rescue > 0.0 else "damage",
                        "sample0_strict_correct": safe_float(sample0["strict_correct"]),
                        "basin_correct_rate": correct_rate,
                        "stable_score": stable_score,
                        "stable_hallucination_score": stable_hallucination_score,
                        "token_mean_entropy_mean": mean(token_mean_values),
                        "centroid_entropy_z": centroid_entropy_z,
                        "centroid_logprob_z": centroid_logprob_z,
                        "cluster_weight_mass": weight,
                        "fragmentation_entropy": fragmentation_entropy,
                        "top2_weight_margin": top2_weight_margin,
                        "distance_to_sample0_entropy_logprob": distance_to_sample0,
                        "answer_preview": members[0]["answer_preview"],
                    }
                )
    add_ranks(basin_rows)
    return basin_rows, enriched_candidates, switch_rows


def regime_label(**flags: float) -> str:
    if flags["is_stable_correct"] > 0:
        return "stable_correct"
    if flags["is_stable_hallucination"] > 0:
        return "stable_hallucination"
    if flags["is_mixed"] > 0:
        return "mixed"
    if flags["is_rescue"] > 0:
        return "rescue"
    if flags["is_damage"] > 0:
        return "damage"
    if flags["is_unstable_wrong"] > 0:
        return "unstable_wrong"
    return "other"


def add_ranks(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["question_id"]))].append(row)
    for group in grouped.values():
        entropy_ranks = rank_values([safe_float(row["centroid_entropy_z"]) for row in group], reverse=False)
        logprob_ranks = rank_values([safe_float(row["centroid_logprob_z"]) for row in group], reverse=True)
        weight_ranks = rank_values([safe_float(row["cluster_weight_mass"]) for row in group], reverse=True)
        stable_ranks = rank_values([safe_float(row["stable_score"]) for row in group], reverse=True)
        for row, entropy_rank, logprob_rank, weight_rank, stable_rank in zip(
            group,
            entropy_ranks,
            logprob_ranks,
            weight_ranks,
            stable_ranks,
        ):
            row["low_entropy_basin_rank"] = entropy_rank
            row["logprob_basin_rank"] = logprob_rank
            row["weight_basin_rank"] = weight_rank
            row["stable_basin_rank"] = stable_rank


def feature_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = [
        ("is_stable_hallucination_basin", "stable_hallucination_vs_other"),
        ("is_stable_correct_basin", "stable_correct_vs_other"),
        ("is_rescue_basin", "rescue_vs_nonrescue"),
        ("is_damage_basin", "damage_vs_nondamage"),
        ("is_pure_wrong", "pure_wrong_vs_not"),
        ("is_pure_correct", "pure_correct_vs_not"),
    ]
    features = entropy_features()
    out: list[dict[str, Any]] = []
    for label, comparison in comparisons:
        pos_rows = [row for row in rows if safe_float(row[label]) > 0]
        neg_rows = [row for row in rows if safe_float(row[label]) <= 0]
        for feature in features:
            pos = [safe_float(row[feature]) for row in pos_rows]
            neg = [safe_float(row[feature]) for row in neg_rows]
            out.append(
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
                    "auc_positive_low": 1.0 - auc_score(pos, neg),
                }
            )
    out.sort(key=lambda row: abs(row["cohen_d"]), reverse=True)
    return out


def entropy_features() -> list[str]:
    return [
        "token_mean_entropy_mean",
        "token_mean_entropy_std",
        "token_max_entropy_mean",
        "logprob_avg_mean",
        "cluster_weight_mass",
        "cluster_size",
        "centroid_entropy_z",
        "centroid_max_entropy_z",
        "centroid_logprob_z",
        "internal_token_entropy_std",
        "within_basin_lexical_entropy",
        "within_basin_jaccard",
        "semantic_entropy_weighted_set",
        "semantic_clusters_set",
        "fragmentation_entropy",
        "normalized_fragmentation_entropy",
        "top2_weight_margin",
        "top2_logprob_margin",
        "top2_low_entropy_margin",
        "distance_to_sample0_entropy_logprob",
        "stable_score",
        "low_entropy_score",
        "stable_hallucination_score",
        "low_entropy_basin_rank",
        "logprob_basin_rank",
        "weight_basin_rank",
        "stable_basin_rank",
    ]


def regime_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["basin_regime"])].append(row)
    out: list[dict[str, Any]] = []
    for regime, items in sorted(grouped.items()):
        out.append(
            {
                "basin_regime": regime,
                "count": len(items),
                "mean_correct_rate": mean([safe_float(row["basin_correct_rate"]) for row in items]),
                "mean_token_entropy": mean([safe_float(row["token_mean_entropy_mean"]) for row in items]),
                "mean_centroid_entropy_z": mean([safe_float(row["centroid_entropy_z"]) for row in items]),
                "mean_logprob": mean([safe_float(row["logprob_avg_mean"]) for row in items]),
                "mean_centroid_logprob_z": mean([safe_float(row["centroid_logprob_z"]) for row in items]),
                "mean_cluster_weight": mean([safe_float(row["cluster_weight_mass"]) for row in items]),
                "mean_fragmentation_entropy": mean([safe_float(row["fragmentation_entropy"]) for row in items]),
                "mean_within_jaccard": mean([safe_float(row["within_basin_jaccard"]) for row in items]),
                "mean_stable_score": mean([safe_float(row["stable_score"]) for row in items]),
                "mean_stable_hallucination_score": mean([safe_float(row["stable_hallucination_score"]) for row in items]),
            }
        )
    out.sort(key=lambda row: row["count"], reverse=True)
    return out


def entropy_filter_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    thresholds = [
        ("low_entropy_wrong_candidate", lambda row: safe_float(row["basin_correct_rate"]) == 0.0 and safe_float(row["centroid_entropy_z"]) <= 0.0),
        ("low_entropy_high_weight_wrong", lambda row: safe_float(row["basin_correct_rate"]) == 0.0 and safe_float(row["centroid_entropy_z"]) <= 0.0 and safe_float(row["cluster_weight_mass"]) >= 0.25),
        ("confident_low_entropy_wrong", lambda row: safe_float(row["basin_correct_rate"]) == 0.0 and safe_float(row["centroid_entropy_z"]) <= 0.0 and safe_float(row["centroid_logprob_z"]) >= 0.0),
        ("high_entropy_wrong", lambda row: safe_float(row["basin_correct_rate"]) == 0.0 and safe_float(row["centroid_entropy_z"]) > 0.5),
    ]
    out: list[dict[str, Any]] = []
    for name, predicate in thresholds:
        selected = [row for row in rows if predicate(row)]
        out.append(
            {
                "filter": name,
                "basin_count": len(selected),
                "share_of_all_basins": len(selected) / max(1, len(rows)),
                "mean_cluster_weight": mean([safe_float(row["cluster_weight_mass"]) for row in selected]),
                "mean_token_entropy": mean([safe_float(row["token_mean_entropy_mean"]) for row in selected]),
                "mean_centroid_logprob_z": mean([safe_float(row["centroid_logprob_z"]) for row in selected]),
                "rescue_basin_count": sum(1 for row in selected if safe_float(row["is_rescue_basin"]) > 0),
                "damage_basin_count": sum(1 for row in selected if safe_float(row["is_damage_basin"]) > 0),
                "sample0_basin_count": sum(1 for row in selected if safe_float(row["contains_sample0"]) > 0),
            }
        )
    return out


def case_examples(rows: list[dict[str, Any]], limit: int = 16) -> dict[str, list[dict[str, Any]]]:
    stable_hallucination = [
        row for row in rows if safe_float(row["is_stable_hallucination_basin"]) > 0
    ]
    stable_correct = [row for row in rows if safe_float(row["is_stable_correct_basin"]) > 0]
    high_entropy_wrong = [
        row for row in rows if safe_float(row["is_unstable_wrong_basin"]) > 0 and safe_float(row["centroid_entropy_z"]) > 0.5
    ]
    rescue = [row for row in rows if safe_float(row["is_rescue_basin"]) > 0]
    damage = [row for row in rows if safe_float(row["is_damage_basin"]) > 0]

    def payload(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "seed": row["seed"],
            "question_id": row["question_id"],
            "question": row["question"],
            "cluster_id": row["cluster_id"],
            "basin_regime": row["basin_regime"],
            "cluster_size": row["cluster_size"],
            "cluster_weight_mass": row["cluster_weight_mass"],
            "basin_correct_rate": row["basin_correct_rate"],
            "token_mean_entropy_mean": row["token_mean_entropy_mean"],
            "centroid_entropy_z": row["centroid_entropy_z"],
            "centroid_logprob_z": row["centroid_logprob_z"],
            "stable_hallucination_score": row["stable_hallucination_score"],
            "answer_preview": row["answer_preview"],
        }

    return {
        "stable_hallucination": [
            payload(row)
            for row in sorted(
                stable_hallucination,
                key=lambda item: (-safe_float(item["stable_hallucination_score"]), safe_float(item["centroid_entropy_z"])),
            )[:limit]
        ],
        "stable_correct": [
            payload(row)
            for row in sorted(stable_correct, key=lambda item: (-safe_float(item["stable_score"]), safe_float(item["centroid_entropy_z"])))[:limit]
        ],
        "high_entropy_wrong": [
            payload(row)
            for row in sorted(high_entropy_wrong, key=lambda item: -safe_float(item["centroid_entropy_z"]))[:limit]
        ],
        "rescue": [
            payload(row)
            for row in sorted(rescue, key=lambda item: (safe_float(item["centroid_entropy_z"]), -safe_float(item["cluster_weight_mass"])))[:limit]
        ],
        "damage": [
            payload(row)
            for row in sorted(damage, key=lambda item: (safe_float(item["centroid_entropy_z"]), -safe_float(item["cluster_weight_mass"])))[:limit]
        ],
    }


def make_plots(output_dir: Path, basin_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]], regime_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = ensure_dir(output_dir / "plots")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 9,
        }
    )

    colors = {
        "stable_correct": "#2ca25f",
        "stable_hallucination": "#de2d26",
        "unstable_wrong": "#9e9e9e",
        "rescue": "#3182bd",
        "damage": "#f16913",
        "mixed": "#756bb1",
        "other": "#cccccc",
    }
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    for regime, group in sorted(group_by_regime(basin_rows).items()):
        ax.scatter(
            [safe_float(row["centroid_logprob_z"]) for row in group],
            [safe_float(row["centroid_entropy_z"]) for row in group],
            s=[20 + 60 * safe_float(row["cluster_weight_mass"]) for row in group],
            c=colors.get(regime, "#cccccc"),
            alpha=0.68,
            label=regime,
            edgecolors="none",
        )
    ax.axhline(0, color="#777777", lw=0.8)
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_xlabel("Basin centroid logprob z")
    ax.set_ylabel("Basin centroid entropy z")
    ax.set_title("Entropy Regime Map of Answer Basins")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "01_entropy_regime_map.png")
    plt.close(fig)

    stable_correct = [row for row in basin_rows if safe_float(row["is_stable_correct_basin"]) > 0]
    stable_hallu = [row for row in basin_rows if safe_float(row["is_stable_hallucination_basin"]) > 0]
    metrics = ["token_mean_entropy_mean", "centroid_entropy_z", "centroid_logprob_z", "cluster_weight_mass", "within_basin_jaccard", "fragmentation_entropy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.6))
    for ax, metric in zip(axes, metrics):
        ax.bar(
            ["stable\ncorrect", "stable\nhallucination"],
            [mean([safe_float(row[metric]) for row in stable_correct]), mean([safe_float(row[metric]) for row in stable_hallu])],
            color=["#2ca25f", "#de2d26"],
            alpha=0.82,
        )
        ax.set_title(metric)
    fig.suptitle("Stable Correct vs Stable Hallucination Basins", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(plot_dir / "02_stable_correct_vs_hallucination.png")
    plt.close(fig)

    rescue = [row for row in basin_rows if safe_float(row["is_rescue_basin"]) > 0]
    damage = [row for row in basin_rows if safe_float(row["is_damage_basin"]) > 0]
    metrics = ["token_mean_entropy_mean", "centroid_entropy_z", "centroid_logprob_z", "cluster_weight_mass", "distance_to_sample0_entropy_logprob", "fragmentation_entropy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.6))
    for ax, metric in zip(axes, metrics):
        ax.bar(
            ["rescue", "damage"],
            [mean([safe_float(row[metric]) for row in rescue]), mean([safe_float(row[metric]) for row in damage])],
            color=["#3182bd", "#f16913"],
            alpha=0.82,
        )
        ax.set_title(metric)
    fig.suptitle("Rescue vs Damage Entropy Anatomy", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(plot_dir / "03_rescue_damage_entropy_anatomy.png")
    plt.close(fig)

    top_features = feature_rows[:18]
    fig, ax = plt.subplots(figsize=(9, 6.4))
    labels = [f"{row['comparison']}\n{row['feature']}" for row in top_features]
    values = [safe_float(row["cohen_d"]) for row in top_features]
    ax.barh(labels[::-1], values[::-1], color=["#3182bd" if value >= 0 else "#de2d26" for value in values[::-1]])
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_xlabel("Cohen's d")
    ax.set_title("Entropy Feature Separation")
    fig.tight_layout()
    fig.savefig(plot_dir / "04_entropy_feature_separation.png")
    plt.close(fig)

    cases = [
        row for row in sorted(basin_rows, key=lambda item: -safe_float(item["stable_hallucination_score"]))[:12]
    ]
    fig, ax = plt.subplots(figsize=(9, 5.0))
    names = [f"{row['question_id']}\nc{row['cluster_id']}" for row in cases]
    values = [safe_float(row["stable_hallucination_score"]) for row in cases]
    ax.barh(names[::-1], values[::-1], color="#de2d26", alpha=0.82)
    ax.set_xlabel("Stable hallucination score")
    ax.set_title("Top Low-Entropy Hallucination-Like Basins")
    fig.tight_layout()
    fig.savefig(plot_dir / "05_low_entropy_hallucination_cases.png")
    plt.close(fig)


def group_by_regime(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["basin_regime"])].append(row)
    return dict(grouped)


def build_report(
    output_dir: Path,
    basin_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
    filter_rows: list[dict[str, Any]],
    examples: dict[str, list[dict[str, Any]]],
) -> None:
    stable_hallu = [row for row in basin_rows if safe_float(row["is_stable_hallucination_basin"]) > 0]
    stable_correct = [row for row in basin_rows if safe_float(row["is_stable_correct_basin"]) > 0]
    rescue = [row for row in basin_rows if safe_float(row["is_rescue_basin"]) > 0]
    damage = [row for row in basin_rows if safe_float(row["is_damage_basin"]) > 0]
    lines = [
        "# Entropy Anatomy of Answer Basins",
        "",
        "## Core Question",
        "",
        "我们要检验：稳定幻觉是不是一种低熵错误吸引盆，而不是普通高不确定性错误。",
        "",
        "## Scale",
        "",
        f"- Basins: `{len(basin_rows)}`",
        f"- Stable correct basins: `{len(stable_correct)}`",
        f"- Stable hallucination basins: `{len(stable_hallu)}`",
        f"- Rescue basins: `{len(rescue)}`",
        f"- Damage basins: `{len(damage)}`",
        "",
        "## Entropy Regimes",
        "",
        "| Regime | Count | Correct Rate | Token Ent | Entropy z | Logprob z | Weight | Frag Ent | Jaccard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in regime_rows:
        lines.append(
            f"| `{row['basin_regime']}` | `{int(row['count'])}` | `{safe_float(row['mean_correct_rate']):.2%}` | "
            f"`{safe_float(row['mean_token_entropy']):.4f}` | `{safe_float(row['mean_centroid_entropy_z']):.3f}` | "
            f"`{safe_float(row['mean_centroid_logprob_z']):.3f}` | `{safe_float(row['mean_cluster_weight']):.3f}` | "
            f"`{safe_float(row['mean_fragmentation_entropy']):.3f}` | `{safe_float(row['mean_within_jaccard']):.3f}` |"
        )
    lines.extend(["", "## Strongest Entropy Separations", "", "| Comparison | Feature | Pos Mean | Neg Mean | Cohen d | AUC high | AUC low |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in feature_rows[:18]:
        lines.append(
            f"| `{row['comparison']}` | `{row['feature']}` | `{safe_float(row['positive_mean']):.4f}` | "
            f"`{safe_float(row['negative_mean']):.4f}` | `{safe_float(row['cohen_d']):.3f}` | "
            f"`{safe_float(row['auc_positive_high']):.3f}` | `{safe_float(row['auc_positive_low']):.3f}` |"
        )
    lines.extend(["", "## Low-Entropy Wrong Filters", "", "| Filter | Basins | Share | Rescue | Damage | Sample0 | Token Ent | Logprob z |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in filter_rows:
        lines.append(
            f"| `{row['filter']}` | `{int(row['basin_count'])}` | `{safe_float(row['share_of_all_basins']):.2%}` | "
            f"`{int(row['rescue_basin_count'])}` | `{int(row['damage_basin_count'])}` | `{int(row['sample0_basin_count'])}` | "
            f"`{safe_float(row['mean_token_entropy']):.4f}` | `{safe_float(row['mean_centroid_logprob_z']):.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Representative Stable Hallucination-Like Basins",
            "",
        ]
    )
    for item in examples["stable_hallucination"][:8]:
        lines.append(
            f"- `{item['question_id']}` c`{item['cluster_id']}`: weight `{safe_float(item['cluster_weight_mass']):.3f}`, "
            f"entropy_z `{safe_float(item['centroid_entropy_z']):.3f}`, logprob_z `{safe_float(item['centroid_logprob_z']):.3f}` — {item['answer_preview']}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 如果 stable hallucination 同时表现为低 entropy、高 weight、高 logprob，它支持“低熵错误吸引盆”假说。",
            "- 如果 stable hallucination 和 stable correct 高度重叠，则 entropy 可以解释稳定性，但不能单独判断事实正确性。",
            "- 如果 damage basins 更接近 stable hallucination，则 controller 的失败主要来自稳定幻觉，而不是发散噪声。",
            "",
            "## Paper-Facing Claim",
            "",
            "> Candidate-space entropy has two failure modes: high-entropy fragmentation and low-entropy false attraction. The latter explains why confidence or entropy minimization alone can select stable hallucinations.",
        ]
    )
    (output_dir / "entropy_anatomy_findings_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_run = Path(args.input_run)
    output_dir = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    candidate_rows = read_csv(input_run / "candidate_features.csv")
    basin_rows, enriched_candidates, switch_rows = build_basin_tables(candidate_rows)
    features = feature_summary(basin_rows)
    regimes = regime_summary(basin_rows)
    filters = entropy_filter_diagnostics(basin_rows)
    examples = case_examples(basin_rows)

    write_csv(output_dir / "entropy_basin_table.csv", basin_rows)
    write_csv(output_dir / "entropy_candidate_table.csv", enriched_candidates)
    write_csv(output_dir / "entropy_feature_summary.csv", features)
    write_csv(output_dir / "entropy_regime_summary.csv", regimes)
    write_csv(output_dir / "entropy_switch_diagnostics.csv", switch_rows)
    write_csv(output_dir / "entropy_filter_diagnostics.csv", filters)
    write_json(output_dir / "entropy_case_examples.json", examples)
    write_json(
        output_dir / "run_metadata.json",
        {
            "input_run": str(input_run),
            "candidate_count": len(candidate_rows),
            "basin_count": len(basin_rows),
            "features": entropy_features(),
        },
    )
    make_plots(output_dir, basin_rows, features, regimes)
    build_report(output_dir, basin_rows, features, regimes, filters, examples)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "candidate_count": len(candidate_rows),
                "basin_count": len(basin_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
