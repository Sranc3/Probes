#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/configs/candidate_space_config.json"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze local answer-candidate feature space.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: flatten(row.get(key)) for key in fieldnames})


def flatten(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def normalize_text(text: str) -> str:
    lowered = str(text).lower()
    cleaned = "".join(character if character.isalnum() else " " for character in lowered)
    return " ".join(cleaned.split())


def exact_match(answer: str, ideal_answers: list[str]) -> bool:
    normalized_answer = normalize_text(answer)
    return any(normalized_answer == normalize_text(ideal) for ideal in ideal_answers if normalize_text(ideal))


def contains_match(answer: str, ideal_answers: list[str]) -> bool:
    normalized_answer = normalize_text(answer)
    if not normalized_answer:
        return False
    for ideal in ideal_answers:
        normalized_ideal = normalize_text(ideal)
        if normalized_ideal and normalized_ideal in normalized_answer:
            return True
    return False


def strict_correct(answer: str, ideal_answers: list[str]) -> float:
    return float(exact_match(answer, ideal_answers) or contains_match(answer, ideal_answers))


def truncate(text: str, limit: int = 220) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def load_metadata(run_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    metadata: dict[tuple[int, str], dict[str, Any]] = {}
    with (run_dir / "method_rows.csv").open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            if row["method"] != "single_sample_baseline":
                continue
            key = (int(row["seed"]), row["question_id"])
            metadata[key] = {
                "question_index": int(row["question_index"]),
                "question": row["question"],
                "ideal_answers": json.loads(row["ideal_answers"]),
                "sample0_text": row["answer_text"],
                "sample0_strict_correct": safe_float(row["final_strict_correct"]),
            }
    return metadata


def cluster_lookup(sample_set: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for cluster in sample_set["clusters"]:
        for sample_index in cluster["sample_indices"]:
            lookup[int(sample_index)] = cluster
    return lookup


def rank_values(values: list[float], reverse: bool) -> list[int]:
    ordered = sorted(range(len(values)), key=lambda idx: values[idx], reverse=reverse)
    ranks = [0] * len(values)
    for rank, idx in enumerate(ordered, start=1):
        ranks[idx] = rank
    return ranks


def add_within_question_coordinates(rows: list[dict[str, Any]], features: list[str]) -> None:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), row["question_id"])].append(row)
    for group in grouped.values():
        for feature in features:
            values = [safe_float(row[feature]) for row in group]
            mu = mean(values)
            sigma = stdev(values)
            min_value = min(values) if values else 0.0
            max_value = max(values) if values else 0.0
            for row, value in zip(group, values):
                row[f"{feature}_z"] = (value - mu) / sigma if sigma > 0.0 else 0.0
                row[f"{feature}_minmax"] = (value - min_value) / (max_value - min_value) if max_value > min_value else 0.0
        logprob_ranks = rank_values([safe_float(row["logprob_avg"]) for row in group], reverse=True)
        entropy_ranks = rank_values([safe_float(row["token_mean_entropy"]) for row in group], reverse=False)
        cluster_ranks = rank_values([safe_float(row["cluster_size"]) for row in group], reverse=True)
        for row, logprob_rank, entropy_rank, cluster_rank in zip(group, logprob_ranks, entropy_ranks, cluster_ranks):
            row["logprob_rank"] = logprob_rank
            row["low_entropy_rank"] = entropy_rank
            row["cluster_size_rank"] = cluster_rank


def extract_candidate_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = Path(config["phase2a_run_dir"])
    metadata = load_metadata(run_dir)
    sample_sets = read_json(run_dir / "sample_sets.json")
    rows: list[dict[str, Any]] = []
    for sample_set in sample_sets:
        seed = int(sample_set["seed"])
        question_id = sample_set["question_id"]
        meta = metadata[(seed, question_id)]
        clusters = cluster_lookup(sample_set)
        labels = [
            strict_correct(sample["text"], meta["ideal_answers"])
            for sample in sorted(sample_set["generations"], key=lambda item: int(item["sample_index"]))
        ]
        sample0_correct = float(meta["sample0_strict_correct"])
        oracle_available = float(any(label > sample0_correct for label in labels))
        any_correct = float(any(label > 0.0 for label in labels))
        for sample in sample_set["generations"]:
            sample_index = int(sample["sample_index"])
            cluster = clusters[sample_index]
            is_correct = strict_correct(sample["text"], meta["ideal_answers"])
            rows.append(
                {
                    "seed": seed,
                    "question_index": int(meta["question_index"]),
                    "question_id": question_id,
                    "question": meta["question"],
                    "sample_index": sample_index,
                    "is_sample0": float(sample_index == 0),
                    "strict_correct": is_correct,
                    "sample0_strict_correct": sample0_correct,
                    "rescue_candidate": float(sample0_correct == 0.0 and is_correct > 0.0),
                    "damage_candidate": float(sample0_correct > 0.0 and is_correct == 0.0),
                    "oracle_available_for_question": oracle_available,
                    "any_correct_in_question": any_correct,
                    "cluster_id": int(sample["cluster_id"]),
                    "cluster_size": int(cluster["size"]),
                    "cluster_weight_mass": safe_float(cluster["weight_mass"]),
                    "semantic_entropy_weighted_set": safe_float(sample_set["semantic_entropy_weighted"]),
                    "semantic_clusters_set": int(sample_set["semantic_clusters"]),
                    "logprob_avg": safe_float(sample["logprob_avg"]),
                    "token_mean_entropy": safe_float(sample["token_mean_entropy"]),
                    "token_max_entropy": safe_float(sample["token_max_entropy"]),
                    "token_count": int(sample["token_count"]),
                    "prompt_tokens": int(safe_float(sample.get("prompt_tokens", 0))),
                    "completion_tokens": int(safe_float(sample.get("completion_tokens", sample.get("token_count", 0)))),
                    "total_tokens": int(safe_float(sample.get("total_tokens", safe_float(sample.get("prompt_tokens", 0)) + safe_float(sample.get("token_count", 0))))),
                    "forward_calls": int(safe_float(sample.get("forward_calls", sample.get("token_count", 0)))),
                    "latency_ms": safe_float(sample.get("latency_ms", sample.get("elapsed_ms", 0.0))),
                    "cuda_peak_memory_bytes": int(safe_float(sample.get("cuda_peak_memory_bytes", 0))),
                    "answer_text": sample["text"],
                    "answer_preview": truncate(sample["text"]),
                }
            )
    add_within_question_coordinates(rows, config["features"])
    return rows


def cohen_d(pos: list[float], neg: list[float]) -> float:
    if len(pos) < 2 or len(neg) < 2:
        return 0.0
    pooled_var = ((len(pos) - 1) * stdev(pos) ** 2 + (len(neg) - 1) * stdev(neg) ** 2) / (len(pos) + len(neg) - 2)
    if pooled_var <= 0.0:
        return 0.0
    return float((mean(pos) - mean(neg)) / math.sqrt(pooled_var))


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
    return float(wins / total) if total else 0.5


def feature_label_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    labels = [
        ("strict_correct", "correct_vs_wrong"),
        ("rescue_candidate", "rescue_vs_nonrescue"),
        ("damage_candidate", "damage_vs_nondamage"),
    ]
    feature_names = list(config["features"])
    expanded_features = feature_names + [f"{name}_z" for name in feature_names] + [
        "logprob_rank",
        "low_entropy_rank",
        "cluster_size_rank",
    ]
    summary: list[dict[str, Any]] = []
    for label_key, comparison in labels:
        pos_rows = [row for row in rows if safe_float(row[label_key]) > 0.0]
        neg_rows = [row for row in rows if safe_float(row[label_key]) <= 0.0]
        for feature in expanded_features:
            pos = [safe_float(row[feature]) for row in pos_rows]
            neg = [safe_float(row[feature]) for row in neg_rows]
            summary.append(
                {
                    "comparison": comparison,
                    "label": label_key,
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
    summary.sort(key=lambda row: (row["comparison"], abs(row["cohen_d"])), reverse=True)
    return summary


def apply_score(row: dict[str, Any], weights: dict[str, float]) -> float:
    return float(sum(safe_float(row.get(feature, 0.0)) * float(weight) for feature, weight in weights.items()))


def score_formula_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), row["question_id"])].append(row)

    selection_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for formula_name, weights in config["score_formulas"].items():
        selected_rows: list[dict[str, Any]] = []
        for (seed, question_id), group in sorted(grouped.items()):
            sample0 = next(row for row in group if int(row["sample_index"]) == 0)
            selected = max(group, key=lambda row: apply_score(row, weights))
            selected_rows.append(selected)
            selection_rows.append(
                {
                    "formula": formula_name,
                    "seed": seed,
                    "question_id": question_id,
                    "selected_sample_index": selected["sample_index"],
                    "sample0_strict_correct": sample0["strict_correct"],
                    "selected_strict_correct": selected["strict_correct"],
                    "delta_strict_correct": safe_float(selected["strict_correct"]) - safe_float(sample0["strict_correct"]),
                    "answer_changed": float(selected["answer_text"] != sample0["answer_text"]),
                    "score": apply_score(selected, weights),
                    "selected_preview": selected["answer_preview"],
                    "sample0_preview": sample0["answer_preview"],
                }
            )
        deltas = [
            safe_float(selection["delta_strict_correct"])
            for selection in selection_rows
            if selection["formula"] == formula_name
        ]
        summary_rows.append(
            {
                "formula": formula_name,
                "pairs": len(selected_rows),
                "strict_correct_rate": mean([safe_float(row["strict_correct"]) for row in selected_rows]),
                "delta_strict_correct_vs_sample0": mean(deltas),
                "improved_count": sum(1 for value in deltas if value > 0.0),
                "damaged_count": sum(1 for value in deltas if value < 0.0),
                "answer_changed_rate": mean(
                    [safe_float(selection["answer_changed"]) for selection in selection_rows if selection["formula"] == formula_name]
                ),
                "token_mean_entropy_mean": mean([safe_float(row["token_mean_entropy"]) for row in selected_rows]),
                "token_count_mean": mean([safe_float(row["token_count"]) for row in selected_rows]),
            }
        )
    return summary_rows, selection_rows


def question_geometry_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), row["question_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (seed, question_id), group in sorted(grouped.items()):
        correct = [row for row in group if safe_float(row["strict_correct"]) > 0.0]
        wrong = [row for row in group if safe_float(row["strict_correct"]) <= 0.0]
        sample0 = next(row for row in group if int(row["sample_index"]) == 0)
        best_correct_logprob_rank = min([int(row["logprob_rank"]) for row in correct], default=0)
        best_correct_entropy_rank = min([int(row["low_entropy_rank"]) for row in correct], default=0)
        summaries.append(
            {
                "seed": seed,
                "question_id": question_id,
                "question": sample0["question"],
                "sample0_strict_correct": sample0["strict_correct"],
                "correct_candidates": len(correct),
                "wrong_candidates": len(wrong),
                "oracle_available": float(safe_float(sample0["strict_correct"]) == 0.0 and len(correct) > 0),
                "best_correct_logprob_rank": best_correct_logprob_rank,
                "best_correct_low_entropy_rank": best_correct_entropy_rank,
                "correct_logprob_mean": mean([safe_float(row["logprob_avg"]) for row in correct]),
                "wrong_logprob_mean": mean([safe_float(row["logprob_avg"]) for row in wrong]),
                "correct_entropy_mean": mean([safe_float(row["token_mean_entropy"]) for row in correct]),
                "wrong_entropy_mean": mean([safe_float(row["token_mean_entropy"]) for row in wrong]),
                "semantic_entropy_weighted_set": sample0["semantic_entropy_weighted_set"],
                "semantic_clusters_set": sample0["semantic_clusters_set"],
            }
        )
    summaries.sort(key=lambda row: (row["oracle_available"], row["correct_candidates"]), reverse=True)
    return summaries


def case_examples(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    limit = int(config.get("top_case_count", 12))
    rescue = [row for row in rows if safe_float(row["rescue_candidate"]) > 0.0]
    damage = [row for row in rows if safe_float(row["damage_candidate"]) > 0.0]
    high_conf_wrong = [
        row
        for row in rows
        if safe_float(row["strict_correct"]) <= 0.0 and int(row.get("logprob_rank", 99)) <= 2
    ]
    low_entropy_wrong = [
        row
        for row in rows
        if safe_float(row["strict_correct"]) <= 0.0 and int(row.get("low_entropy_rank", 99)) <= 2
    ]

    def payload(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "seed": row["seed"],
            "question_id": row["question_id"],
            "question": truncate(row["question"]),
            "sample_index": row["sample_index"],
            "strict_correct": row["strict_correct"],
            "logprob_rank": row["logprob_rank"],
            "low_entropy_rank": row["low_entropy_rank"],
            "cluster_size": row["cluster_size"],
            "logprob_avg": row["logprob_avg"],
            "token_mean_entropy": row["token_mean_entropy"],
            "answer_preview": row["answer_preview"],
        }

    return {
        "rescue_candidates": [
            payload(row)
            for row in sorted(rescue, key=lambda item: (int(item["logprob_rank"]), safe_float(item["token_mean_entropy"])))[:limit]
        ],
        "damage_candidates": [
            payload(row)
            for row in sorted(damage, key=lambda item: (int(item["logprob_rank"]), safe_float(item["token_mean_entropy"])))[:limit]
        ],
        "high_confidence_wrong": [
            payload(row)
            for row in sorted(high_conf_wrong, key=lambda item: (int(item["logprob_rank"]), safe_float(item["token_mean_entropy"])))[:limit]
        ],
        "low_entropy_wrong": [
            payload(row)
            for row in sorted(low_entropy_wrong, key=lambda item: (int(item["low_entropy_rank"]), -safe_float(item["logprob_avg"])))[:limit]
        ],
    }


def build_markdown(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    feature_summary: list[dict[str, Any]],
    score_summary: list[dict[str, Any]],
    question_summary: list[dict[str, Any]],
    examples: dict[str, list[dict[str, Any]]],
) -> str:
    total = len(rows)
    correct = sum(1 for row in rows if safe_float(row["strict_correct"]) > 0.0)
    rescue = sum(1 for row in rows if safe_float(row["rescue_candidate"]) > 0.0)
    damage = sum(1 for row in rows if safe_float(row["damage_candidate"]) > 0.0)
    oracle_questions = sum(1 for row in question_summary if safe_float(row["oracle_available"]) > 0.0)
    lines = [
        "# Candidate Space Analysis",
        "",
        f"- Description: {config['description']}",
        f"- Candidate points: `{total}`",
        f"- Strict-correct points: `{correct}` (`{correct / max(total, 1):.2%}`)",
        f"- Rescue candidates: `{rescue}`",
        f"- Damage candidates: `{damage}`",
        f"- Questions where sample0 is wrong but a correct candidate exists: `{oracle_questions}`",
        "",
        "## Strongest Feature Separations",
        "",
        "| Comparison | Feature | Pos Mean | Neg Mean | Cohen d | AUC high=positive |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in feature_summary[:18]:
        lines.append(
            f"| `{row['comparison']}` | `{row['feature']}` | `{row['positive_mean']:.4f}` | "
            f"`{row['negative_mean']:.4f}` | `{row['cohen_d']:.3f}` | `{row['auc_positive_high']:.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Simple Geometric Scores",
            "",
            "| Formula | Strict Correct | Δ vs sample0 | Improved | Damaged | Changed | Token Mean Ent |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in score_summary:
        lines.append(
            f"| `{row['formula']}` | `{row['strict_correct_rate']:.2%}` | "
            f"`{row['delta_strict_correct_vs_sample0']:.4f}` | `{row['improved_count']}` | "
            f"`{row['damaged_count']}` | `{row['answer_changed_rate']:.2%}` | `{row['token_mean_entropy_mean']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Important Cases",
            "",
            f"- Rescue examples shown: `{len(examples['rescue_candidates'])}`",
            f"- Damage examples shown: `{len(examples['damage_candidates'])}`",
            f"- High-confidence wrong examples shown: `{len(examples['high_confidence_wrong'])}`",
            f"- Low-entropy wrong examples shown: `{len(examples['low_entropy_wrong'])}`",
            "",
            "## Interpretation",
            "",
            "- This is the first chart of the candidate space, not a final geometric model.",
            "- If correct candidates consistently occupy a feature region, a cheap controller may be learnable.",
            "- If high-confidence or low-entropy wrong answers are common, entropy/logprob alone cannot be trusted.",
            "- A future manifold/Riemannian version should first learn a better answer embedding; the current coordinates are a practical local chart.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    tag = args.tag or config.get("tag", Path(args.config).stem)
    run_root = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}")
    write_json(run_root / "config_snapshot.json", config)

    rows = extract_candidate_rows(config)
    feature_summary = feature_label_summary(rows, config)
    score_summary, score_rows = score_formula_summary(rows, config)
    question_summary = question_geometry_summary(rows)
    examples = case_examples(rows, config)

    write_csv(run_root / "candidate_features.csv", rows)
    write_csv(run_root / "feature_label_summary.csv", feature_summary)
    write_csv(run_root / "score_formula_summary.csv", score_summary)
    write_csv(run_root / "score_formula_rows.csv", score_rows)
    write_csv(run_root / "question_geometry_summary.csv", question_summary)
    write_json(run_root / "case_examples.json", examples)
    write_json(
        run_root / "candidate_space_summary.json",
        {
            "candidate_count": len(rows),
            "feature_label_summary": feature_summary,
            "score_formula_summary": score_summary,
            "top_questions": question_summary[:20],
            "case_examples": examples,
        },
    )
    (run_root / "summary.md").write_text(
        build_markdown(config, rows, feature_summary, score_summary, question_summary, examples),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "candidate_count": len(rows),
                "feature_summary_count": len(feature_summary),
                "score_formula_count": len(score_summary),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
