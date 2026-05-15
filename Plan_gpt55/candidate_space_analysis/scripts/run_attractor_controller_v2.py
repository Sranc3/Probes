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

from run_attractor_controller_experiments import (
    add_candidate_features,
    attach_attractor_features,
    build_attractor_tables,
    ensure_dir,
    evaluate_selected,
    flatten,
    group_candidates,
    grouped_folds,
    mean,
    read_csv,
    safe_float,
    write_csv,
    write_json,
)


DEFAULT_INPUT_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run basin-aware attractor controller v2.")
    parser.add_argument("--input-run", default=DEFAULT_INPUT_RUN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="attractor_controller_v2")
    return parser.parse_args()


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: flatten(row.get(key, "")) for key in fieldnames})


def cluster_groups(group: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in group:
        grouped[int(row["cluster_id"])].append(row)
    return dict(grouped)


def best_member(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            safe_float(row["logprob_avg_z"])
            - 0.35 * safe_float(row["token_mean_entropy_z"])
            - 0.15 * safe_float(row["token_count_z"]),
            -int(row["sample_index"]),
        ),
    )


def basin_record(group: list[dict[str, Any]], cluster_id: int) -> dict[str, Any]:
    members = cluster_groups(group)[cluster_id]
    first = members[0]
    sample0 = group[0]
    return {
        "cluster_id": cluster_id,
        "members": members,
        "representative": best_member(members),
        "contains_sample0": float(cluster_id == int(sample0["cluster_id"])),
        "size": len(members),
        "weight_mass": safe_float(first["attractor_weight_mass"]),
        "centroid_logprob_z": safe_float(first["attractor_centroid_logprob_z"]),
        "centroid_entropy_z": safe_float(first["attractor_centroid_entropy_z"]),
        "centroid_len_z": safe_float(first["attractor_centroid_len_z"]),
        "cluster_size_z": safe_float(first["attractor_cluster_size_z"]),
        "cluster_weight_z": safe_float(first["attractor_cluster_weight_z"]),
        "distance_to_sample0_basin": safe_float(first["distance_to_sample0_basin"]),
        "correct_count": sum(1 for row in members if safe_float(row["strict_correct"]) > 0.0),
    }


def pair_basins(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [basin_record(group, cluster_id) for cluster_id in sorted(cluster_groups(group))]


def pair_features(group: list[dict[str, Any]]) -> dict[str, float]:
    sample0 = group[0]
    basins = pair_basins(group)
    sample0_basin = next(basin for basin in basins if basin["contains_sample0"] > 0.0)
    alternatives = [basin for basin in basins if basin["contains_sample0"] <= 0.0]
    top_alt_weight = max([safe_float(basin["weight_mass"]) for basin in alternatives], default=0.0)
    top_alt_logprob = max([safe_float(basin["centroid_logprob_z"]) for basin in alternatives], default=-4.0)
    top_alt_size_z = max([safe_float(basin["cluster_size_z"]) for basin in alternatives], default=-4.0)
    return {
        "sample0_logprob_z": safe_float(sample0["logprob_avg_z"]),
        "sample0_entropy_z": safe_float(sample0["token_mean_entropy_z"]),
        "sample0_len_z": safe_float(sample0["token_count_z"]),
        "sample0_basin_weight": safe_float(sample0_basin["weight_mass"]),
        "sample0_basin_size": safe_float(sample0_basin["size"]),
        "sample0_basin_weight_z": safe_float(sample0_basin["cluster_weight_z"]),
        "sample0_basin_size_z": safe_float(sample0_basin["cluster_size_z"]),
        "fragmentation_entropy": safe_float(sample0["fragmentation_entropy"]),
        "semantic_clusters": safe_float(sample0["semantic_clusters_set"]),
        "semantic_entropy": safe_float(sample0["semantic_entropy_weighted_set"]),
        "top_alt_weight": top_alt_weight,
        "top_alt_logprob_z": top_alt_logprob,
        "top_alt_size_z": top_alt_size_z,
        "alt_weight_margin": top_alt_weight - safe_float(sample0_basin["weight_mass"]),
        "alt_logprob_margin": top_alt_logprob - safe_float(sample0_basin["centroid_logprob_z"]),
        "alt_size_margin": top_alt_size_z - safe_float(sample0_basin["cluster_size_z"]),
    }


RiskFormula = dict[str, Any]
BasinFormula = dict[str, Any]


def risk_formulas() -> list[RiskFormula]:
    return [
        {
            "name": "weak_sample0_basin",
            "weights": {
                "sample0_logprob_z": -0.9,
                "sample0_entropy_z": 0.7,
                "sample0_basin_weight_z": -0.8,
                "sample0_basin_size_z": -0.6,
                "fragmentation_entropy": 0.35,
                "alt_logprob_margin": 0.25,
            },
        },
        {
            "name": "fragmented_with_competitor",
            "weights": {
                "fragmentation_entropy": 0.85,
                "semantic_clusters": 0.16,
                "top_alt_weight": 0.75,
                "alt_weight_margin": 0.45,
                "sample0_entropy_z": 0.25,
            },
        },
        {
            "name": "sample0_uncertain",
            "weights": {
                "sample0_logprob_z": -1.0,
                "sample0_entropy_z": 1.0,
                "sample0_len_z": 0.25,
                "semantic_entropy": 0.2,
            },
        },
        {
            "name": "basin_margin_risk",
            "weights": {
                "alt_logprob_margin": 0.7,
                "alt_size_margin": 0.35,
                "alt_weight_margin": 0.6,
                "sample0_basin_weight_z": -0.35,
            },
        },
    ]


def basin_formulas() -> list[BasinFormula]:
    return [
        {
            "name": "basin_balanced",
            "weights": {
                "centroid_logprob_z": 1.0,
                "centroid_entropy_z": -0.55,
                "cluster_weight_z": 0.7,
                "cluster_size_z": 0.55,
                "centroid_len_z": -0.15,
                "distance_to_sample0_basin": 0.08,
            },
        },
        {
            "name": "basin_rescue_lean",
            "weights": {
                "centroid_logprob_z": 1.05,
                "centroid_entropy_z": -0.35,
                "cluster_weight_z": 0.45,
                "cluster_size_z": 0.45,
                "distance_to_sample0_basin": 0.2,
            },
        },
        {
            "name": "basin_safe",
            "weights": {
                "centroid_logprob_z": 0.75,
                "centroid_entropy_z": -0.85,
                "cluster_weight_z": 1.0,
                "cluster_size_z": 0.85,
                "centroid_len_z": -0.35,
                "distance_to_sample0_basin": -0.1,
            },
        },
        {
            "name": "basin_antidamage",
            "weights": {
                "centroid_logprob_z": 1.0,
                "centroid_entropy_z": -1.2,
                "cluster_weight_z": 0.55,
                "cluster_size_z": 0.5,
                "centroid_len_z": -0.5,
            },
        },
    ]


def linear_score(features: dict[str, float], formula: dict[str, Any]) -> float:
    return sum(float(weight) * safe_float(features.get(name, 0.0)) for name, weight in formula["weights"].items())


def risk_score(group: list[dict[str, Any]], formula: RiskFormula) -> float:
    return linear_score(pair_features(group), formula)


def basin_score(basin: dict[str, Any], formula: BasinFormula) -> float:
    features = {key: safe_float(value) for key, value in basin.items() if not isinstance(value, list)}
    return linear_score(features, formula)


def quantile_thresholds(values: list[float]) -> list[float]:
    if not values:
        return [0.0]
    sorted_values = sorted(values)
    thresholds = []
    for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        thresholds.append(sorted_values[min(len(sorted_values) - 1, int(q * (len(sorted_values) - 1)))])
    return sorted(set(thresholds))


def choose_v2(group: list[dict[str, Any]], config: dict[str, Any], prefix_k: int | None = None) -> dict[str, Any]:
    current_group = group[:prefix_k] if prefix_k is not None else group
    sample0 = current_group[0]
    if risk_score(current_group, config["risk_formula"]) < config["risk_threshold"]:
        return sample0
    basins = pair_basins(current_group)
    sample0_basin = next(basin for basin in basins if basin["contains_sample0"] > 0.0)
    alternatives = [basin for basin in basins if basin["contains_sample0"] <= 0.0]
    if not alternatives:
        return sample0
    best_alt = max(alternatives, key=lambda basin: (basin_score(basin, config["basin_formula"]), -safe_float(basin["representative"]["sample_index"])))
    margin = basin_score(best_alt, config["basin_formula"]) - basin_score(sample0_basin, config["basin_formula"])
    if margin < config["switch_margin"]:
        return sample0
    return best_alt["representative"]


def metric_objective(metrics: dict[str, Any], damage_penalty: float, rescue_weight: float, switch_penalty: float = 0.0) -> float:
    return (
        100.0 * safe_float(metrics["delta_vs_sample0"])
        + rescue_weight * int(metrics["improved_count"])
        - damage_penalty * int(metrics["damaged_count"])
        - switch_penalty * safe_float(metrics["answer_changed_rate"])
    )


def evaluate_config(groups: list[list[dict[str, Any]]], config: dict[str, Any], method: str, split: str) -> dict[str, Any]:
    selected = [choose_v2(group, config) for group in groups]
    metrics = evaluate_selected(groups, selected, method, split)
    metrics.update(
        {
            "risk_formula": config["risk_formula"]["name"],
            "basin_formula": config["basin_formula"]["name"],
            "risk_threshold": config["risk_threshold"],
            "switch_margin": config["switch_margin"],
            "damage_penalty": config.get("damage_penalty", 0.0),
            "rescue_weight": config.get("rescue_weight", 1.0),
        }
    )
    return metrics


def train_config(groups: list[list[dict[str, Any]]], damage_penalty: float, rescue_weight: float) -> dict[str, Any]:
    best_config: dict[str, Any] | None = None
    best_objective = -1e18
    for risk_formula in risk_formulas():
        thresholds = quantile_thresholds([risk_score(group, risk_formula) for group in groups])
        for risk_threshold in thresholds:
            for basin_formula in basin_formulas():
                for switch_margin in [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]:
                    config = {
                        "risk_formula": risk_formula,
                        "risk_threshold": risk_threshold,
                        "basin_formula": basin_formula,
                        "switch_margin": switch_margin,
                        "damage_penalty": damage_penalty,
                        "rescue_weight": rescue_weight,
                    }
                    metrics = evaluate_config(groups, config, "train", "train")
                    objective = metric_objective(metrics, damage_penalty=damage_penalty, rescue_weight=rescue_weight, switch_penalty=0.25)
                    if objective > best_objective:
                        best_objective = objective
                        best_config = config
    assert best_config is not None
    best_config["train_objective"] = best_objective
    return best_config


def pareto_settings() -> list[dict[str, float]]:
    return [
        {"name": "rescue_heavy", "damage_penalty": 0.25, "rescue_weight": 1.5},
        {"name": "balanced", "damage_penalty": 1.0, "rescue_weight": 1.0},
        {"name": "safe", "damage_penalty": 2.5, "rescue_weight": 0.75},
        {"name": "very_safe", "damage_penalty": 5.0, "rescue_weight": 0.5},
    ]


def summarize(rows: list[dict[str, Any]], group_key: str, split_name: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        out.append(
            {
                "split": split_name,
                group_key: key,
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
            }
        )
    return out


def run_pareto_cv(grouped: dict[tuple[int, str], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    for fold_idx, (train_groups, test_groups) in enumerate(grouped_folds(grouped)):
        for setting in pareto_settings():
            config = train_config(train_groups, damage_penalty=setting["damage_penalty"], rescue_weight=setting["rescue_weight"])
            method = f"v2_{setting['name']}"
            metrics = evaluate_config(test_groups, config, method, f"question_fold_{fold_idx}")
            metrics["pareto_setting"] = setting["name"]
            fold_rows.append(metrics)
            config_rows.append(
                {
                    "split": f"question_fold_{fold_idx}",
                    "pareto_setting": setting["name"],
                    "risk_formula": config["risk_formula"]["name"],
                    "basin_formula": config["basin_formula"]["name"],
                    "risk_threshold": config["risk_threshold"],
                    "switch_margin": config["switch_margin"],
                    "damage_penalty": setting["damage_penalty"],
                    "rescue_weight": setting["rescue_weight"],
                    "train_objective": config["train_objective"],
                }
            )
            for group in test_groups:
                selected = choose_v2(group, config)
                sample0 = group[0]
                selection_rows.append(
                    {
                        "split": f"question_fold_{fold_idx}",
                        "pareto_setting": setting["name"],
                        "seed": selected["seed"],
                        "question_id": selected["question_id"],
                        "selected_sample_index": selected["sample_index"],
                        "sample0_strict_correct": sample0["strict_correct"],
                        "selected_strict_correct": selected["strict_correct"],
                        "delta_strict_correct": safe_float(selected["strict_correct"]) - safe_float(sample0["strict_correct"]),
                        "selected_cluster_id": selected["cluster_id"],
                        "sample0_cluster_id": sample0["cluster_id"],
                        "risk_score": risk_score(group, config["risk_formula"]),
                        "selected_preview": selected["answer_preview"],
                    }
                )
    return summarize(fold_rows, "method", "question_grouped_cv") + fold_rows, selection_rows, config_rows


def adaptive_select(group: list[dict[str, Any]], config: dict[str, Any], max_k: int) -> tuple[dict[str, Any], int]:
    sample0 = group[0]
    for k in range(1, max_k + 1):
        prefix = group[:k]
        if k >= 2:
            pf = pair_features(prefix)
            stable = (
                pf["sample0_basin_size"] >= 2
                and pf["sample0_logprob_z"] >= -0.2
                and pf["sample0_entropy_z"] <= 0.15
                and pf["fragmentation_entropy"] <= 0.8
            )
            if stable:
                return sample0, k
            selected = choose_v2(prefix, config)
            if int(selected["sample_index"]) != 0:
                return selected, k
    return choose_v2(group[:max_k], config), max_k


def run_sequential_cv(grouped: dict[tuple[int, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for max_k in [2, 3, 4, 8]:
        fold_rows: list[dict[str, Any]] = []
        for fold_idx, (train_groups, test_groups) in enumerate(grouped_folds(grouped)):
            config = train_config(train_groups, damage_penalty=1.0, rescue_weight=1.0)
            selected: list[dict[str, Any]] = []
            costs: list[float] = []
            for group in test_groups:
                row, cost = adaptive_select(group, config, max_k)
                selected.append(row)
                costs.append(float(cost))
            metrics = evaluate_selected(test_groups, selected, f"v2_adaptive_k{max_k}", f"question_fold_{fold_idx}")
            metrics["avg_generated_candidates"] = mean(costs)
            metrics["max_candidates"] = max_k
            fold_rows.append(metrics)
        summary = summarize(fold_rows, "method", "question_grouped_cv")
        for row in summary:
            matching = [item for item in fold_rows if item["method"] == row["method"]]
            row["avg_generated_candidates"] = mean([safe_float(item["avg_generated_candidates"]) for item in matching])
            row["max_candidates"] = max_k
        rows.extend(summary)
        rows.extend(fold_rows)
    return rows


def run_seed_split(grouped: dict[tuple[int, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for train_seed, test_seed in [(42, 43), (43, 42)]:
        train_groups = [group for (seed, _qid), group in grouped.items() if seed == train_seed]
        test_groups = [group for (seed, _qid), group in grouped.items() if seed == test_seed]
        for setting in pareto_settings():
            config = train_config(train_groups, damage_penalty=setting["damage_penalty"], rescue_weight=setting["rescue_weight"])
            metrics = evaluate_config(test_groups, config, f"v2_{setting['name']}", f"train_seed_{train_seed}_test_seed_{test_seed}")
            metrics["pareto_setting"] = setting["name"]
            rows.append(metrics)
    return rows


def failure_cases(selection_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in selection_rows if row["pareto_setting"] == "balanced"]
    interesting = [row for row in rows if safe_float(row["delta_strict_correct"]) != 0.0]
    interesting.sort(key=lambda row: (safe_float(row["delta_strict_correct"]), -safe_float(row["risk_score"])))
    return interesting[:20] + interesting[-20:]


def make_plots(output_dir: Path, pareto_rows: list[dict[str, Any]], sequential_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = ensure_dir(output_dir / "plots")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})

    summary = [row for row in pareto_rows if row["split"] == "question_grouped_cv"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter([safe_float(row["damaged_count"]) for row in summary], [safe_float(row["improved_count"]) for row in summary], s=90)
    for row in summary:
        ax.text(safe_float(row["damaged_count"]) + 0.05, safe_float(row["improved_count"]), row["method"], fontsize=8)
    ax.set_xlabel("Damaged cases")
    ax.set_ylabel("Improved cases")
    ax.set_title("Rescue-Damage Pareto")
    fig.tight_layout()
    fig.savefig(plot_dir / "01_rescue_damage_pareto.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar([row["method"] for row in summary], [safe_float(row["strict_correct_rate"]) for row in summary], color="#4c78a8", alpha=0.82)
    ax.set_ylabel("Strict correctness")
    ax.set_title("Basin-Aware Controller v2")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(plot_dir / "02_v2_cv_accuracy.png")
    plt.close(fig)

    seq = [row for row in sequential_rows if row["split"] == "question_grouped_cv"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot([safe_float(row["avg_generated_candidates"]) for row in seq], [safe_float(row["strict_correct_rate"]) for row in seq], marker="o")
    for row in seq:
        ax.text(safe_float(row["avg_generated_candidates"]), safe_float(row["strict_correct_rate"]), row["method"], fontsize=8)
    ax.set_xlabel("Average generated candidates")
    ax.set_ylabel("Strict correctness")
    ax.set_title("v2 Adaptive Sampling")
    fig.tight_layout()
    fig.savefig(plot_dir / "03_v2_cost_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    labels = [f"{row['split']}\n{row['method']}" for row in seed_rows]
    values = [safe_float(row["delta_vs_sample0"]) for row in seed_rows]
    ax.barh(labels[::-1], values[::-1], color=["#31a354" if value >= 0 else "#de2d26" for value in values[::-1]])
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_xlabel("Delta vs sample0")
    ax.set_title("v2 Seed-Split Robustness")
    fig.tight_layout()
    fig.savefig(plot_dir / "04_v2_seed_robustness.png")
    plt.close(fig)


def build_report(output_dir: Path, pareto_rows: list[dict[str, Any]], sequential_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]], config_rows: list[dict[str, Any]]) -> None:
    summary = [row for row in pareto_rows if row["split"] == "question_grouped_cv"]
    seq = [row for row in sequential_rows if row["split"] == "question_grouped_cv"]
    lines = [
        "# Basin-Aware Controller v2",
        "",
        "## What Changed From v1",
        "",
        "- v1 used a coarse two-stage rule. v2 separates sample0 risk gating from alternative-basin selection.",
        "- v2 trains several rescue-damage objectives and reports a Pareto frontier instead of a single hand-picked threshold.",
        "- v2 uses pair-level features such as sample0 basin dominance, alternative-basin margins, fragmentation, and basin distance.",
        "",
        "## Rescue-Damage Pareto",
        "",
        "| Method | Strict | Delta | Improved | Damaged | Net | Rescue Recall | Damage Avoid | Changed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{int(row['net_gain_count'])}` | "
            f"`{safe_float(row['rescue_recall']):.2%}` | `{safe_float(row['damage_avoidance']):.2%}` | `{safe_float(row['answer_changed_rate']):.2%}` |"
        )
    lines.extend(["", "## Adaptive Cost Curve", "", "| Policy | Avg candidates | Strict | Delta | Improved | Damaged |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in seq:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['avg_generated_candidates']):.2f}` | `{safe_float(row['strict_correct_rate']):.2%}` | "
            f"`{safe_float(row['delta_vs_sample0']):.2%}` | `{int(row['improved_count'])}` | `{int(row['damaged_count'])}` |"
        )
    lines.extend(["", "## Seed Split Robustness", "", "| Split | Method | Strict | Delta | Improved | Damaged |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in seed_rows:
        lines.append(
            f"| `{row['split']}` | `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` |"
        )
    common_configs = defaultdict(int)
    for row in config_rows:
        key = (row["pareto_setting"], row["risk_formula"], row["basin_formula"], row["switch_margin"])
        common_configs[key] += 1
    lines.extend(["", "## Frequently Selected Configs", "", "| Setting | Risk gate | Basin selector | Margin | Count |", "| --- | --- | --- | ---: | ---: |"])
    for (setting, risk_name, basin_name, margin), count in sorted(common_configs.items(), key=lambda item: item[1], reverse=True)[:12]:
        lines.append(f"| `{setting}` | `{risk_name}` | `{basin_name}` | `{safe_float(margin):.2f}` | `{count}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "v2 should be judged by the Pareto curve, not by a single number. A good result is not merely higher accuracy; it should expose a controllable tradeoff between rescue and damage.",
            "",
            "If the rescue-heavy setting captures many more rescue cases with tolerable damage, it becomes the discovery-oriented controller. If the safe setting preserves most gains with lower damage, it becomes the deployment-oriented controller.",
        ]
    )
    (output_dir / "basin_aware_controller_v2_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_run = Path(args.input_run)
    output_dir = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    candidate_rows = read_csv(input_run / "candidate_features.csv")
    add_candidate_features(candidate_rows)
    _attractor_rows, lookup = build_attractor_tables(candidate_rows)
    attach_attractor_features(candidate_rows, lookup)
    grouped = group_candidates(candidate_rows)

    pareto_rows, selection_rows, config_rows = run_pareto_cv(grouped)
    sequential_rows = run_sequential_cv(grouped)
    seed_rows = run_seed_split(grouped)
    failures = failure_cases(selection_rows)

    write_dict_csv(output_dir / "v2_pareto_cv_results.csv", pareto_rows)
    write_dict_csv(output_dir / "v2_selection_rows.csv", selection_rows)
    write_dict_csv(output_dir / "v2_trained_configs.csv", config_rows)
    write_dict_csv(output_dir / "v2_sequential_cost_curve.csv", sequential_rows)
    write_dict_csv(output_dir / "v2_seed_split_robustness.csv", seed_rows)
    write_dict_csv(output_dir / "v2_failure_cases.csv", failures)
    write_json(
        output_dir / "run_metadata.json",
        {
            "input_run": str(input_run),
            "candidate_count": len(candidate_rows),
            "group_count": len(grouped),
            "pareto_settings": pareto_settings(),
        },
    )
    make_plots(output_dir, pareto_rows, sequential_rows, seed_rows)
    build_report(output_dir, pareto_rows, sequential_rows, seed_rows, config_rows)
    print(json.dumps({"output_dir": str(output_dir), "group_count": len(grouped)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
