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

from run_entropy_anatomy import ensure_dir, mean, read_csv, safe_float, write_csv, write_json


DEFAULT_CANDIDATE_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25"
DEFAULT_ENTROPY_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260429_063326_entropy_anatomy"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"


FEATURE_SETS = {
    "entropy_only": [
        "token_mean_entropy_mean",
        "token_mean_entropy_std",
        "token_max_entropy_mean",
        "centroid_entropy_z",
        "centroid_max_entropy_z",
        "internal_token_entropy_std",
        "semantic_entropy_weighted_set",
        "fragmentation_entropy",
        "normalized_fragmentation_entropy",
        "low_entropy_score",
        "delta_entropy_vs_sample0",
        "sample0_basin_entropy_z",
    ],
    "geometry_entropy": [
        "token_mean_entropy_mean",
        "token_max_entropy_mean",
        "logprob_avg_mean",
        "cluster_weight_mass",
        "cluster_size",
        "centroid_entropy_z",
        "centroid_logprob_z",
        "fragmentation_entropy",
        "top2_weight_margin",
        "distance_to_sample0_entropy_logprob",
        "stable_score",
        "stable_hallucination_score",
        "delta_entropy_vs_sample0",
        "delta_logprob_vs_sample0",
        "delta_weight_vs_sample0",
    ],
    "full_basin": [
        "token_mean_entropy_mean",
        "token_mean_entropy_std",
        "token_max_entropy_mean",
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
        "delta_entropy_vs_sample0",
        "delta_logprob_vs_sample0",
        "delta_weight_vs_sample0",
        "delta_stable_score_vs_sample0",
        "delta_hallucination_score_vs_sample0",
        "sample0_basin_entropy_z",
        "sample0_basin_logprob_z",
        "sample0_basin_weight",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight learned basin verifier.")
    parser.add_argument("--candidate-run", default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--entropy-run", default=DEFAULT_ENTROPY_RUN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="learned_basin_verifier")
    return parser.parse_args()


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


class LogisticModel:
    def __init__(self, features: list[str]):
        self.features = features
        self.means = {feature: 0.0 for feature in features}
        self.stds = {feature: 1.0 for feature in features}
        self.weights = {feature: 0.0 for feature in features}
        self.bias = 0.0

    def fit(self, rows: list[dict[str, Any]], label: str, epochs: int = 700, lr: float = 0.07, l2: float = 0.01) -> None:
        for feature in self.features:
            values = [safe_float(row.get(feature, 0.0)) for row in rows]
            self.means[feature] = mean(values)
            self.stds[feature] = stdev(values) or 1.0
        pos = sum(1 for row in rows if safe_float(row[label]) > 0.0)
        neg = max(0, len(rows) - pos)
        pos_w = len(rows) / max(1, 2 * pos)
        neg_w = len(rows) / max(1, 2 * neg)
        for _ in range(epochs):
            grad = {feature: 0.0 for feature in self.features}
            grad_b = 0.0
            for row in rows:
                y = 1.0 if safe_float(row[label]) > 0.0 else 0.0
                p = self.predict(row)
                err = (p - y) * (pos_w if y else neg_w)
                for feature in self.features:
                    grad[feature] += err * self.z(row, feature)
                grad_b += err
            scale = 1.0 / max(1, len(rows))
            for feature in self.features:
                self.weights[feature] -= lr * (grad[feature] * scale + l2 * self.weights[feature])
            self.bias -= lr * grad_b * scale

    def z(self, row: dict[str, Any], feature: str) -> float:
        return (safe_float(row.get(feature, 0.0)) - self.means[feature]) / self.stds[feature]

    def predict(self, row: dict[str, Any]) -> float:
        value = self.bias
        for feature in self.features:
            value += self.weights[feature] * self.z(row, feature)
        return sigmoid(value)

    def weight_rows(self, model_name: str, split: str, top_k: int = 16) -> list[dict[str, Any]]:
        rows = [{"split": split, "model": model_name, "feature": "bias", "weight": self.bias}]
        for feature in sorted(self.features, key=lambda item: abs(self.weights[item]), reverse=True)[:top_k]:
            rows.append({"split": split, "model": model_name, "feature": feature, "weight": self.weights[feature]})
        return rows


def representative_candidates(candidate_rows: list[dict[str, Any]]) -> dict[tuple[int, str, int], dict[str, Any]]:
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[(int(row["seed"]), str(row["question_id"]), int(row["cluster_id"]))].append(row)
    reps = {}
    for key, rows in grouped.items():
        reps[key] = max(
            rows,
            key=lambda row: (
                safe_float(row["logprob_avg_z"]) - 0.35 * safe_float(row["token_mean_entropy_z"]) - 0.1 * safe_float(row["token_count_z"]),
                -int(row["sample_index"]),
            ),
        )
    return reps


def build_tables(candidate_run: Path, entropy_run: Path) -> tuple[list[dict[str, Any]], dict[tuple[int, str], list[dict[str, Any]]]]:
    basin_rows = read_csv(entropy_run / "entropy_basin_table.csv")
    candidate_rows = read_csv(candidate_run / "candidate_features.csv")
    reps = representative_candidates(candidate_rows)
    by_pair: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in basin_rows:
        key = (int(row["seed"]), str(row["question_id"]), int(row["cluster_id"]))
        rep = reps[key]
        row["representative_sample_index"] = int(rep["sample_index"])
        row["representative_strict_correct"] = safe_float(rep["strict_correct"])
        row["representative_preview"] = rep["answer_preview"]
        pair_key = (int(row["seed"]), str(row["question_id"]))
        by_pair[pair_key].append(row)
    for group in by_pair.values():
        group.sort(key=lambda row: int(row["cluster_id"]))
        sample0 = next(row for row in group if safe_float(row["contains_sample0"]) > 0)
        for row in group:
            row["delta_entropy_vs_sample0"] = safe_float(row["centroid_entropy_z"]) - safe_float(sample0["centroid_entropy_z"])
            row["delta_logprob_vs_sample0"] = safe_float(row["centroid_logprob_z"]) - safe_float(sample0["centroid_logprob_z"])
            row["delta_weight_vs_sample0"] = safe_float(row["cluster_weight_mass"]) - safe_float(sample0["cluster_weight_mass"])
            row["delta_stable_score_vs_sample0"] = safe_float(row["stable_score"]) - safe_float(sample0["stable_score"])
            row["delta_hallucination_score_vs_sample0"] = safe_float(row["stable_hallucination_score"]) - safe_float(sample0["stable_hallucination_score"])
            row["sample0_basin_entropy_z"] = safe_float(sample0["centroid_entropy_z"])
            row["sample0_basin_logprob_z"] = safe_float(sample0["centroid_logprob_z"])
            row["sample0_basin_weight"] = safe_float(sample0["cluster_weight_mass"])
            row["switch_gain_label"] = float(
                safe_float(sample0["representative_strict_correct"]) <= 0 and safe_float(row["representative_strict_correct"]) > 0 and safe_float(row["contains_sample0"]) <= 0
            )
    return [row for group in by_pair.values() for row in group], dict(by_pair)


def question_folds(keys: list[tuple[int, str]], k: int = 5) -> list[tuple[set[str], set[str]]]:
    qids = sorted({qid for _seed, qid in keys})
    folds = []
    for i in range(k):
        test = {qid for idx, qid in enumerate(qids) if idx % k == i}
        folds.append((set(qids) - test, test))
    return folds


def choose_basin(group: list[dict[str, Any]], model: LogisticModel, threshold: float, margin: float) -> dict[str, Any]:
    sample0 = next(row for row in group if safe_float(row["contains_sample0"]) > 0)
    alternatives = [row for row in group if safe_float(row["contains_sample0"]) <= 0]
    if not alternatives:
        return sample0
    sample0_p = model.predict(sample0)
    alt = max(alternatives, key=model.predict)
    alt_p = model.predict(alt)
    if sample0_p < threshold and alt_p - sample0_p >= margin:
        return alt
    return sample0


def choose_switch(group: list[dict[str, Any]], model: LogisticModel, threshold: float) -> dict[str, Any]:
    sample0 = next(row for row in group if safe_float(row["contains_sample0"]) > 0)
    alternatives = [row for row in group if safe_float(row["contains_sample0"]) <= 0]
    if not alternatives:
        return sample0
    alt = max(alternatives, key=model.predict)
    return alt if model.predict(alt) >= threshold else sample0


def evaluate(groups: list[list[dict[str, Any]]], selected: list[dict[str, Any]], method: str, split: str) -> dict[str, Any]:
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


def tune_basin(groups: list[list[dict[str, Any]]], model: LogisticModel) -> tuple[float, float]:
    best = (-1e9, 0.5, 0.0)
    for threshold in [0.2, 0.35, 0.5, 0.65, 0.8]:
        for margin in [-0.2, 0.0, 0.2, 0.4]:
            selected = [choose_basin(group, model, threshold, margin) for group in groups]
            metrics = evaluate(groups, selected, "train", "train")
            score = 100 * safe_float(metrics["delta_vs_sample0"]) + int(metrics["net_gain_count"]) - 1.5 * int(metrics["damaged_count"])
            if score > best[0]:
                best = (score, threshold, margin)
    return best[1], best[2]


def tune_switch(groups: list[list[dict[str, Any]]], model: LogisticModel) -> float:
    best = (-1e9, 0.5)
    for threshold in [0.2, 0.35, 0.5, 0.65, 0.8]:
        selected = [choose_switch(group, model, threshold) for group in groups]
        metrics = evaluate(groups, selected, "train", "train")
        score = 100 * safe_float(metrics["delta_vs_sample0"]) + int(metrics["net_gain_count"]) - 2.0 * int(metrics["damaged_count"])
        if score > best[0]:
            best = (score, threshold)
    return best[1]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summary = []
    for method, items in sorted(grouped.items()):
        summary.append(
            {
                "split": "question_grouped_cv",
                "method": method,
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
            }
        )
    return summary


def run_cv(rows: list[dict[str, Any]], by_pair: dict[tuple[int, str], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    weights: list[dict[str, Any]] = []
    keys = sorted(by_pair)
    for fold_idx, (train_q, test_q) in enumerate(question_folds(keys)):
        train_groups = [group for (_seed, qid), group in by_pair.items() if qid in train_q]
        test_groups = [group for (_seed, qid), group in by_pair.items() if qid in test_q]
        train_rows = [row for group in train_groups for row in group]
        train_alt = [row for group in train_groups for row in group if safe_float(row["contains_sample0"]) <= 0]
        for name, features in FEATURE_SETS.items():
            basin_model = LogisticModel(features)
            basin_model.fit(train_rows, "representative_strict_correct")
            threshold, margin = tune_basin(train_groups, basin_model)
            selected = [choose_basin(group, basin_model, threshold, margin) for group in test_groups]
            method = f"basin_model_{name}"
            metrics = evaluate(test_groups, selected, method, f"fold_{fold_idx}")
            metrics.update({"feature_set": name, "threshold": threshold, "margin": margin})
            fold_rows.append(metrics)
            weights.extend(basin_model.weight_rows(method, f"fold_{fold_idx}"))

            switch_model = LogisticModel(features)
            switch_model.fit(train_alt, "switch_gain_label")
            switch_threshold = tune_switch(train_groups, switch_model)
            selected_switch = [choose_switch(group, switch_model, switch_threshold) for group in test_groups]
            switch_method = f"switch_model_{name}"
            switch_metrics = evaluate(test_groups, selected_switch, switch_method, f"fold_{fold_idx}")
            switch_metrics.update({"feature_set": name, "threshold": switch_threshold, "margin": ""})
            fold_rows.append(switch_metrics)
            weights.extend(switch_model.weight_rows(switch_method, f"fold_{fold_idx}"))

            for method_name, chosen in [(method, selected), (switch_method, selected_switch)]:
                for group, row in zip(test_groups, chosen):
                    sample0 = next(item for item in group if safe_float(item["contains_sample0"]) > 0)
                    selections.append(
                        {
                            "split": f"fold_{fold_idx}",
                            "method": method_name,
                            "seed": row["seed"],
                            "question_id": row["question_id"],
                            "sample0_correct": sample0["representative_strict_correct"],
                            "selected_correct": row["representative_strict_correct"],
                            "delta_correct": safe_float(row["representative_strict_correct"]) - safe_float(sample0["representative_strict_correct"]),
                            "sample0_cluster_id": sample0["cluster_id"],
                            "selected_cluster_id": row["cluster_id"],
                            "selected_regime": row["basin_regime"],
                            "selected_entropy_z": row["centroid_entropy_z"],
                            "selected_logprob_z": row["centroid_logprob_z"],
                            "selected_weight": row["cluster_weight_mass"],
                            "selected_preview": row["representative_preview"],
                        }
                    )
    return summarize(fold_rows) + fold_rows, selections, weights


def seed_split(by_pair: dict[tuple[int, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for train_seed, test_seed in [(42, 43), (43, 42)]:
        train_groups = [group for (seed, _qid), group in by_pair.items() if seed == train_seed]
        test_groups = [group for (seed, _qid), group in by_pair.items() if seed == test_seed]
        train_rows = [row for group in train_groups for row in group]
        for name, features in FEATURE_SETS.items():
            model = LogisticModel(features)
            model.fit(train_rows, "representative_strict_correct")
            threshold, margin = tune_basin(train_groups, model)
            selected = [choose_basin(group, model, threshold, margin) for group in test_groups]
            metrics = evaluate(test_groups, selected, f"basin_model_{name}", f"train_seed_{train_seed}_test_seed_{test_seed}")
            metrics.update({"feature_set": name, "threshold": threshold, "margin": margin})
            rows.append(metrics)
    return rows


def interesting_cases(selections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in selections if safe_float(row["delta_correct"]) != 0]
    rows.sort(key=lambda row: (safe_float(row["delta_correct"]), safe_float(row["selected_entropy_z"])))
    return rows[:25] + rows[-25:]


def make_plots(output_dir: Path, cv_rows: list[dict[str, Any]], selections: list[dict[str, Any]], weights: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = ensure_dir(output_dir / "plots")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    summary = [row for row in cv_rows if row["split"] == "question_grouped_cv"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar([row["method"].replace("_", "\n") for row in summary], [safe_float(row["strict_correct_rate"]) for row in summary], color="#4c78a8", alpha=0.82)
    ax.set_ylabel("Strict correctness")
    ax.set_title("Learned Basin Verifier CV")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plot_dir / "01_learned_verifier_cv.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter([safe_float(row["damaged_count"]) for row in summary], [safe_float(row["improved_count"]) for row in summary], s=90)
    for row in summary:
        ax.text(safe_float(row["damaged_count"]) + 0.04, safe_float(row["improved_count"]), row["method"].replace("basin_model_", "b_").replace("switch_model_", "s_"), fontsize=8)
    ax.set_xlabel("Damaged")
    ax.set_ylabel("Improved")
    ax.set_title("Learned Verifier Rescue-Damage")
    fig.tight_layout()
    fig.savefig(plot_dir / "02_rescue_damage_tradeoff.png")
    plt.close(fig)

    case_rows = interesting_cases(selections)
    improved = [row for row in case_rows if safe_float(row["delta_correct"]) > 0]
    damaged = [row for row in case_rows if safe_float(row["delta_correct"]) < 0]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter([safe_float(row["selected_logprob_z"]) for row in improved], [safe_float(row["selected_entropy_z"]) for row in improved], label="improved", c="#3182bd")
    ax.scatter([safe_float(row["selected_logprob_z"]) for row in damaged], [safe_float(row["selected_entropy_z"]) for row in damaged], label="damaged", c="#de2d26")
    ax.axhline(0, color="#777777", lw=0.8)
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_xlabel("Selected basin logprob z")
    ax.set_ylabel("Selected basin entropy z")
    ax.set_title("Learned Switches in Entropy Space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "03_switch_entropy_space.png")
    plt.close(fig)

    first_weights = [row for row in weights if row["split"] == "fold_0" and row["feature"] != "bias"][:24]
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [f"{row['model']}\n{row['feature']}" for row in first_weights]
    values = [safe_float(row["weight"]) for row in first_weights]
    ax.barh(labels[::-1], values[::-1], color=["#31a354" if value >= 0 else "#de2d26" for value in values[::-1]])
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_xlabel("Standardized weight")
    ax.set_title("Example Learned Weights")
    fig.tight_layout()
    fig.savefig(plot_dir / "04_learned_weights.png")
    plt.close(fig)


def report(output_dir: Path, cv_rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]], weights: list[dict[str, Any]]) -> None:
    summary = [row for row in cv_rows if row["split"] == "question_grouped_cv"]
    lines = [
        "# Learned Basin Verifier v1",
        "",
        "## Setup",
        "",
        "This is a lightweight logistic verifier over basin-level numerical features. It learns basin correctness and switch-gain policies under question-heldout validation.",
        "",
        "## Grouped CV",
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
    lines.extend(["", "## Seed Split", "", "| Split | Method | Strict | Delta | Improved | Damaged |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in seed_rows:
        lines.append(f"| `{row['split']}` | `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | `{int(row['improved_count'])}` | `{int(row['damaged_count'])}` |")
    agg: dict[str, list[float]] = defaultdict(list)
    for row in weights:
        if row["feature"] != "bias":
            agg[f"{row['model']}::{row['feature']}"].append(safe_float(row["weight"]))
    lines.extend(["", "## Strongest Average Weights", "", "| Model / Feature | Mean Weight |", "| --- | ---: |"])
    for name, values in sorted(agg.items(), key=lambda item: abs(mean(item[1])), reverse=True)[:24]:
        lines.append(f"| `{name}` | `{mean(values):.4f}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If full-basin models beat entropy-only, the verifier is learning more than a low-entropy heuristic.",
            "- If switch models are safer but less effective, they are closer to deployment controllers.",
            "- If learned numeric features still damage stable hallucinations, the next step needs text/semantic supervision.",
        ]
    )
    (output_dir / "learned_basin_verifier_findings_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    basin_rows, by_pair = build_tables(Path(args.candidate_run), Path(args.entropy_run))
    cv_rows, selections, weights = run_cv(basin_rows, by_pair)
    seed_rows = seed_split(by_pair)
    cases = interesting_cases(selections)
    write_csv(output_dir / "learned_basin_table.csv", basin_rows)
    write_csv(output_dir / "learned_verifier_cv_results.csv", cv_rows)
    write_csv(output_dir / "learned_verifier_selection_rows.csv", selections)
    write_csv(output_dir / "learned_verifier_weights.csv", weights)
    write_csv(output_dir / "learned_verifier_seed_split.csv", seed_rows)
    write_csv(output_dir / "learned_verifier_interesting_cases.csv", cases)
    write_json(output_dir / "run_metadata.json", {"basin_count": len(basin_rows), "pair_count": len(by_pair), "feature_sets": FEATURE_SETS})
    make_plots(output_dir, cv_rows, selections, weights)
    report(output_dir, cv_rows, seed_rows, weights)
    print(json.dumps({"output_dir": str(output_dir), "basin_count": len(basin_rows), "pair_count": len(by_pair)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
