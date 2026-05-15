#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "/zhutingqi/song/Plan_gpt55/configs/phase2a_v2_cost_curve.json"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2A-v2 post-hoc reranking cost curve.")
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


def truncate(text: str, limit: int = 220) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def normalize_text(text: str) -> str:
    lowered = str(text).lower()
    cleaned = "".join(character if character.isalnum() else " " for character in lowered)
    return " ".join(cleaned.split())


def compute_exact_match(final_answer: str, ideal_answers: list[str]) -> bool:
    normalized_answer = normalize_text(final_answer)
    return any(normalized_answer == normalize_text(ideal) for ideal in ideal_answers if normalize_text(ideal))


def compute_contains_match(final_answer: str, ideal_answers: list[str]) -> bool:
    normalized_answer = normalize_text(final_answer)
    if not normalized_answer:
        return False
    for ideal in ideal_answers:
        normalized_ideal = normalize_text(ideal)
        if normalized_ideal and normalized_ideal in normalized_answer:
            return True
    return False


def load_method_metadata(run_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    metadata: dict[tuple[int, str], dict[str, Any]] = {}
    with (run_dir / "method_rows.csv").open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            if row["method"] != "single_sample_baseline":
                continue
            key = (int(row["seed"]), row["question_id"])
            ideal_answers = json.loads(row["ideal_answers"])
            metadata[key] = {
                "question_index": int(row["question_index"]),
                "question": row["question"],
                "ideal_answers": ideal_answers,
                "baseline_answer_text": row["answer_text"],
                "baseline_strict_correct": float(row["final_strict_correct"]),
            }
    return metadata


def strict_correct(text: str, ideal_answers: list[str]) -> float:
    return float(bool(compute_exact_match(text, ideal_answers)) or bool(compute_contains_match(text, ideal_answers)))


def prefix_cluster_members(prefix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(int(sample["cluster_id"]) for sample in prefix)
    best_cluster_id, best_count = max(
        counts.items(),
        key=lambda item: (
            item[1],
            max(float(sample["logprob_avg"]) for sample in prefix if int(sample["cluster_id"]) == item[0]),
            -item[0],
        ),
    )
    return [sample for sample in prefix if int(sample["cluster_id"]) == best_cluster_id and best_count > 0]


def pick_sample(method: str, prefix: list[dict[str, Any]], labels: dict[int, float]) -> dict[str, Any]:
    sample0 = prefix[0]
    if method == "sample0":
        return sample0
    if method == "best_logprob":
        return max(prefix, key=lambda sample: float(sample["logprob_avg"]))
    if method == "low_token_entropy":
        return min(prefix, key=lambda sample: (float(sample["token_mean_entropy"]), -float(sample["logprob_avg"])))
    if method == "oracle_strict":
        correct = [sample for sample in prefix if labels[int(sample["sample_index"])] > 0.0]
        if correct:
            return max(correct, key=lambda sample: (float(sample["logprob_avg"]), -float(sample["token_mean_entropy"])))
        return max(prefix, key=lambda sample: float(sample["logprob_avg"]))

    members = prefix_cluster_members(prefix)
    if method == "majority_logprob":
        return max(members, key=lambda sample: float(sample["logprob_avg"]))
    if method == "majority_low_entropy":
        return min(members, key=lambda sample: (float(sample["token_mean_entropy"]), -float(sample["logprob_avg"])))
    if method == "cautious_majority_logprob":
        if len(members) < 2:
            return sample0
        return max(members, key=lambda sample: float(sample["logprob_avg"]))
    raise ValueError(f"Unsupported method: {method}")


def build_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = Path(config["phase2a_run_dir"])
    sample_sets = read_json(run_dir / "sample_sets.json")
    metadata = load_method_metadata(run_dir)
    rows: list[dict[str, Any]] = []
    for sample_set in sample_sets:
        seed = int(sample_set["seed"])
        question_id = sample_set["question_id"]
        meta = metadata[(seed, question_id)]
        samples = sorted(sample_set["generations"], key=lambda sample: int(sample["sample_index"]))
        labels = {int(sample["sample_index"]): strict_correct(sample["text"], meta["ideal_answers"]) for sample in samples}
        for k in config["k_values"]:
            prefix = samples[: int(k)]
            if not prefix:
                continue
            for method in config["methods"]:
                selected = pick_sample(method, prefix, labels)
                selected_index = int(selected["sample_index"])
                baseline_correct = float(meta["baseline_strict_correct"])
                selected_correct = float(labels[selected_index])
                cluster_size_in_prefix = sum(1 for sample in prefix if int(sample["cluster_id"]) == int(selected["cluster_id"]))
                rows.append(
                    {
                        "seed": seed,
                        "question_index": int(meta["question_index"]),
                        "question_id": question_id,
                        "question": meta["question"],
                        "k": int(k),
                        "method": method,
                        "selected_sample_index": selected_index,
                        "selected_cluster_id": int(selected["cluster_id"]),
                        "selected_cluster_size_in_prefix": cluster_size_in_prefix,
                        "answer_text": selected["text"],
                        "answer_preview": truncate(selected["text"]),
                        "baseline_answer_preview": truncate(meta["baseline_answer_text"]),
                        "answer_changed_vs_sample0": float(selected["text"] != meta["baseline_answer_text"]),
                        "strict_correct": selected_correct,
                        "baseline_strict_correct": baseline_correct,
                        "delta_strict_correct_vs_sample0": selected_correct - baseline_correct,
                        "any_strict_correct_in_prefix": float(any(labels[int(sample["sample_index"])] > 0.0 for sample in prefix)),
                        "strict_oracle_available": float(any(labels[int(sample["sample_index"])] > baseline_correct for sample in prefix)),
                        "token_count": int(selected["token_count"]),
                        "logprob_avg": float(selected["logprob_avg"]),
                        "token_mean_entropy": float(selected["token_mean_entropy"]),
                        "token_max_entropy": float(selected["token_max_entropy"]),
                    }
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["k"]), row["method"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (k, method), group in sorted(grouped.items()):
        deltas = [float(row["delta_strict_correct_vs_sample0"]) for row in group]
        summaries.append(
            {
                "k": k,
                "cost_multiplier": k,
                "method": method,
                "pairs": len(group),
                "strict_correct_rate": mean([float(row["strict_correct"]) for row in group]),
                "delta_strict_correct_vs_sample0": mean(deltas),
                "improved_count": sum(1 for value in deltas if value > 0.0),
                "damaged_count": sum(1 for value in deltas if value < 0.0),
                "same_count": sum(1 for value in deltas if value == 0.0),
                "answer_changed_vs_sample0_rate": mean([float(row["answer_changed_vs_sample0"]) for row in group]),
                "token_mean_entropy_mean": mean([float(row["token_mean_entropy"]) for row in group]),
                "token_mean_entropy_std": stdev([float(row["token_mean_entropy"]) for row in group]),
                "token_count_mean": mean([float(row["token_count"]) for row in group]),
                "selected_cluster_size_mean": mean([float(row["selected_cluster_size_in_prefix"]) for row in group]),
                "oracle_available_rate": mean([float(row["strict_oracle_available"]) for row in group]),
                "any_correct_in_prefix_rate": mean([float(row["any_strict_correct_in_prefix"]) for row in group]),
            }
        )
    return summaries


def build_markdown(config: dict[str, Any], summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 2A-v2 Cost Curve",
        "",
        f"- Description: {config['description']}",
        f"- Source run: `{config['phase2a_run_dir']}`",
        "- All correctness numbers below use strict exact/contains matching only.",
        "",
        "## Cost Curve",
        "",
        "| K | Method | Strict Correct | Δ Strict | Improved | Damaged | Changed | Token Mean Ent | Oracle Available |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    preferred_order = {
        "sample0": 0,
        "cautious_majority_logprob": 1,
        "majority_logprob": 2,
        "majority_low_entropy": 3,
        "best_logprob": 4,
        "low_token_entropy": 5,
        "oracle_strict": 6,
    }
    for item in sorted(summaries, key=lambda row: (row["k"], preferred_order.get(row["method"], 99))):
        lines.append(
            f"| `{item['k']}` | `{item['method']}` | `{item['strict_correct_rate']:.2%}` | "
            f"`{item['delta_strict_correct_vs_sample0']:.4f}` | `{item['improved_count']}` | "
            f"`{item['damaged_count']}` | `{item['answer_changed_vs_sample0_rate']:.2%}` | "
            f"`{item['token_mean_entropy_mean']:.4f}` | `{item['oracle_available_rate']:.2%}` |"
        )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `oracle_strict` is not deployable because it uses labels; it is only an upper bound showing whether better answers exist among the first K samples.",
            "- `cautious_majority_logprob` is the low-risk candidate: it only switches away from sample0 if at least two sampled answers land in the same semantic cluster.",
            "- A practical direction needs useful gains at K=2 or K=3; K=8 should be treated as an expensive oracle baseline.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    tag = args.tag or config.get("tag", Path(args.config).stem)
    run_root = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}")
    write_json(run_root / "config_snapshot.json", config)

    rows = build_rows(config)
    summaries = summarize(rows)
    write_csv(run_root / "cost_curve_rows.csv", rows)
    write_csv(run_root / "cost_curve_summary.csv", summaries)
    write_json(run_root / "phase2a_v2_summary.json", {"cost_curve_summary": summaries, "row_count": len(rows)})
    (run_root / "summary.md").write_text(build_markdown(config, summaries), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "row_count": len(rows),
                "summary_count": len(summaries),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
