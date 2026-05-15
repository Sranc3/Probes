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


DEFAULT_CONFIG = "/zhutingqi/song/Plan_gpt55/configs/phase1c_case_diagnosis.json"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-hoc Phase 1C case-level ITI diagnosis.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
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
            writer.writerow({key: flatten_value(row.get(key)) for key in fieldnames})


def flatten_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def parse_number(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def parse_int(value: Any, default: int = 0) -> int:
    return int(round(parse_number(value, float(default))))


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return float(cov / math.sqrt(x_var * y_var))


def truncate(text: str, limit: int = 220) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def discover_question_csvs(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("candidates/*/question_rows.csv"))


def split_candidate_name(candidate_name: str) -> tuple[str, str]:
    if "__" in candidate_name:
        base, gate = candidate_name.split("__", 1)
        return base, gate
    return candidate_name, "default"


def load_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_spec in config["inputs"]:
        experiment = input_spec["experiment"]
        run_dir = Path(input_spec["run_dir"])
        csv_paths = discover_question_csvs(run_dir)
        if not csv_paths:
            raise FileNotFoundError(f"No candidate question_rows.csv files under {run_dir}")
        for csv_path in csv_paths:
            candidate_name = csv_path.parent.name
            base_candidate, gate_name = split_candidate_name(candidate_name)
            with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
                for raw in csv.DictReader(file_obj):
                    row = dict(raw)
                    row["experiment"] = experiment
                    row["run_dir"] = str(run_dir)
                    row["candidate"] = candidate_name
                    row["base_candidate"] = base_candidate
                    row["gate"] = gate_name
                    normalize_row(row)
                    rows.append(row)
    return rows


def normalize_row(row: dict[str, Any]) -> None:
    numeric_fields = [
        "seed",
        "question_index",
        "answer_changed",
        "baseline_semantic_entropy_weighted",
        "intervention_semantic_entropy_weighted",
        "delta_semantic_entropy_weighted",
        "baseline_token_mean_entropy",
        "intervention_token_mean_entropy",
        "delta_token_mean_entropy",
        "baseline_token_max_entropy",
        "intervention_token_max_entropy",
        "delta_token_max_entropy",
        "baseline_num_generated_tokens",
        "intervention_num_generated_tokens",
        "delta_num_generated_tokens",
        "baseline_elapsed_ms",
        "intervention_elapsed_ms",
        "delta_elapsed_ms",
        "baseline_final_semantic_correct",
        "intervention_final_semantic_correct",
        "delta_final_semantic_correct",
        "baseline_final_exact_match",
        "intervention_final_exact_match",
        "delta_final_exact_match",
        "baseline_final_contains_match",
        "intervention_final_contains_match",
        "delta_final_contains_match",
        "intervention_total_events",
        "intervention_steps_with_events",
        "intervention_event_density",
    ]
    for field in numeric_fields:
        row[field] = parse_number(row.get(field))
    row["seed"] = parse_int(row["seed"])
    row["question_index"] = parse_int(row["question_index"])
    row["answer_changed"] = parse_int(row["answer_changed"])


def transition_label(row: dict[str, Any]) -> str:
    baseline_correct = bool(row["baseline_final_semantic_correct"])
    intervention_correct = bool(row["intervention_final_semantic_correct"])
    changed = bool(row["answer_changed"])
    if intervention_correct and not baseline_correct:
        return "corrected"
    if baseline_correct and not intervention_correct:
        return "damaged"
    if baseline_correct and intervention_correct and changed:
        return "changed_but_correct"
    if baseline_correct and intervention_correct:
        return "stable_correct"
    if changed:
        return "wrong_to_wrong_changed"
    return "stable_wrong"


def candidate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["experiment"], row["candidate"], row["gate"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (experiment, candidate, gate), group in sorted(grouped.items()):
        transitions = [transition_label(row) for row in group]
        events = [row["intervention_total_events"] for row in group]
        semantic_deltas = [row["delta_semantic_entropy_weighted"] for row in group]
        token_mean_deltas = [row["delta_token_mean_entropy"] for row in group]
        token_max_deltas = [row["delta_token_max_entropy"] for row in group]
        elapsed_deltas = [row["delta_elapsed_ms"] for row in group]
        correct_deltas = [row["delta_final_semantic_correct"] for row in group]
        answer_changed = [row["answer_changed"] for row in group]
        summary = {
            "experiment": experiment,
            "candidate": candidate,
            "base_candidate": group[0]["base_candidate"],
            "gate": gate,
            "pairs": len(group),
            "unique_questions": len({row["question_id"] for row in group}),
            "seeds": len({row["seed"] for row in group}),
            "answer_changed_count": sum(answer_changed),
            "answer_changed_rate": rate(sum(answer_changed), len(group)),
            "corrected_count": transitions.count("corrected"),
            "corrected_without_contains_count": sum(
                1
                for row in group
                if transition_label(row) == "corrected" and row["intervention_final_contains_match"] <= 0.0
            ),
            "damaged_count": transitions.count("damaged"),
            "changed_but_correct_count": transitions.count("changed_but_correct"),
            "wrong_to_wrong_changed_count": transitions.count("wrong_to_wrong_changed"),
            "stable_correct_count": transitions.count("stable_correct"),
            "stable_wrong_count": transitions.count("stable_wrong"),
            "correct_delta_mean": mean(correct_deltas),
            "semantic_delta_mean": mean(semantic_deltas),
            "semantic_delta_std": stdev(semantic_deltas),
            "semantic_nonpositive_rate": mean([float(value <= 0.0) for value in semantic_deltas]),
            "token_mean_delta_mean": mean(token_mean_deltas),
            "token_mean_delta_std": stdev(token_mean_deltas),
            "token_mean_nonpositive_rate": mean([float(value <= 0.0) for value in token_mean_deltas]),
            "token_max_delta_mean": mean(token_max_deltas),
            "token_max_delta_std": stdev(token_max_deltas),
            "token_max_nonpositive_rate": mean([float(value <= 0.0) for value in token_max_deltas]),
            "elapsed_delta_mean": mean(elapsed_deltas),
            "elapsed_nonpositive_rate": mean([float(value <= 0.0) for value in elapsed_deltas]),
            "events_mean": mean(events),
            "events_std": stdev(events),
            "event_answer_changed_corr": pearson(events, answer_changed),
            "event_semantic_delta_corr": pearson(events, semantic_deltas),
            "event_token_mean_delta_corr": pearson(events, token_mean_deltas),
            "event_correct_delta_corr": pearson(events, correct_deltas),
        }
        summaries.append(summary)
    return summaries


def question_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["experiment"], row["question_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (experiment, question_id), group in sorted(grouped.items()):
        transitions = [transition_label(row) for row in group]
        changed_count = sum(row["answer_changed"] for row in group)
        semantic_deltas = [row["delta_semantic_entropy_weighted"] for row in group]
        token_mean_deltas = [row["delta_token_mean_entropy"] for row in group]
        summaries.append(
            {
                "experiment": experiment,
                "question_id": question_id,
                "question": truncate(group[0]["question"], 180),
                "pairs": len(group),
                "candidates": len({row["candidate"] for row in group}),
                "seeds": len({row["seed"] for row in group}),
                "baseline_correct_rate": mean([row["baseline_final_semantic_correct"] for row in group]),
                "answer_changed_count": changed_count,
                "answer_changed_rate": rate(changed_count, len(group)),
                "corrected_count": transitions.count("corrected"),
                "damaged_count": transitions.count("damaged"),
                "wrong_to_wrong_changed_count": transitions.count("wrong_to_wrong_changed"),
                "changed_but_correct_count": transitions.count("changed_but_correct"),
                "mean_abs_semantic_delta": mean([abs(value) for value in semantic_deltas]),
                "mean_semantic_delta": mean(semantic_deltas),
                "mean_abs_token_mean_delta": mean([abs(value) for value in token_mean_deltas]),
                "mean_events": mean([row["intervention_total_events"] for row in group]),
            }
        )
    summaries.sort(key=lambda row: (row["answer_changed_count"], row["mean_abs_semantic_delta"]), reverse=True)
    return summaries


def case_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": row["experiment"],
        "candidate": row["candidate"],
        "gate": row["gate"],
        "seed": row["seed"],
        "question_id": row["question_id"],
        "question": truncate(row["question"], 220),
        "ideal_answers": truncate(row["ideal_answers"], 220),
        "baseline_answer_text": truncate(row["baseline_answer_text"], 260),
        "intervention_answer_text": truncate(row["intervention_answer_text"], 260),
        "transition": transition_label(row),
        "answer_changed": row["answer_changed"],
        "baseline_final_semantic_correct": row["baseline_final_semantic_correct"],
        "intervention_final_semantic_correct": row["intervention_final_semantic_correct"],
        "baseline_final_contains_match": row["baseline_final_contains_match"],
        "intervention_final_contains_match": row["intervention_final_contains_match"],
        "baseline_final_exact_match": row["baseline_final_exact_match"],
        "intervention_final_exact_match": row["intervention_final_exact_match"],
        "delta_final_semantic_correct": row["delta_final_semantic_correct"],
        "delta_semantic_entropy_weighted": row["delta_semantic_entropy_weighted"],
        "delta_token_mean_entropy": row["delta_token_mean_entropy"],
        "delta_token_max_entropy": row["delta_token_max_entropy"],
        "delta_num_generated_tokens": row["delta_num_generated_tokens"],
        "delta_elapsed_ms": row["delta_elapsed_ms"],
        "intervention_total_events": row["intervention_total_events"],
        "intervention_event_density": row["intervention_event_density"],
    }


def build_case_examples(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    limit = int(thresholds.get("top_case_count", 12))
    changed = [row for row in rows if row["answer_changed"]]
    corrected = [row for row in rows if transition_label(row) == "corrected"]
    damaged = [row for row in rows if transition_label(row) == "damaged"]
    wrong_changed = [row for row in rows if transition_label(row) == "wrong_to_wrong_changed"]
    changed_correct = [row for row in rows if transition_label(row) == "changed_but_correct"]
    entropy_down_no_change = [
        row
        for row in rows
        if not row["answer_changed"]
        and row["delta_semantic_entropy_weighted"] <= -float(thresholds["large_abs_semantic_delta"])
    ]
    entropy_up = [row for row in rows if row["delta_semantic_entropy_weighted"] >= float(thresholds["large_abs_semantic_delta"])]
    high_event_no_change = [
        row
        for row in rows
        if not row["answer_changed"]
        and row["intervention_event_density"] >= float(thresholds["meaningful_event_density"])
    ]
    return {
        "corrected": [case_payload(row) for row in sorted(corrected, key=lambda r: abs(r["delta_semantic_entropy_weighted"]), reverse=True)[:limit]],
        "damaged": [case_payload(row) for row in sorted(damaged, key=lambda r: abs(r["delta_semantic_entropy_weighted"]), reverse=True)[:limit]],
        "wrong_to_wrong_changed": [case_payload(row) for row in sorted(wrong_changed, key=lambda r: abs(r["delta_semantic_entropy_weighted"]), reverse=True)[:limit]],
        "changed_but_correct": [case_payload(row) for row in sorted(changed_correct, key=lambda r: abs(r["delta_semantic_entropy_weighted"]), reverse=True)[:limit]],
        "largest_answer_changes": [case_payload(row) for row in sorted(changed, key=lambda r: r["intervention_total_events"], reverse=True)[:limit]],
        "large_entropy_down_no_answer_change": [
            case_payload(row)
            for row in sorted(entropy_down_no_change, key=lambda r: r["delta_semantic_entropy_weighted"])[:limit]
        ],
        "large_entropy_up": [case_payload(row) for row in sorted(entropy_up, key=lambda r: r["delta_semantic_entropy_weighted"], reverse=True)[:limit]],
        "high_event_no_answer_change": [
            case_payload(row) for row in sorted(high_event_no_change, key=lambda r: r["intervention_total_events"], reverse=True)[:limit]
        ],
    }


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{parsed:.{digits}f}"


def build_markdown(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    examples: dict[str, list[dict[str, Any]]],
) -> str:
    transition_counts = {
        "corrected": sum(1 for row in rows if transition_label(row) == "corrected"),
        "damaged": sum(1 for row in rows if transition_label(row) == "damaged"),
        "changed_but_correct": sum(1 for row in rows if transition_label(row) == "changed_but_correct"),
        "wrong_to_wrong_changed": sum(1 for row in rows if transition_label(row) == "wrong_to_wrong_changed"),
    }
    corrected_without_contains = sum(
        1
        for row in rows
        if transition_label(row) == "corrected" and row["intervention_final_contains_match"] <= 0.0
    )
    changed_total = sum(transition_counts.values())
    lines = [
        "# Phase 1C Case-Level Diagnosis",
        "",
        f"- Description: {config['description']}",
        f"- Total paired rows: `{len(rows)}`",
        f"- Experiments: `{', '.join(sorted({row['experiment'] for row in rows}))}`",
        f"- Answer-changing rows: `{changed_total}` (`{rate(changed_total, len(rows)):.2%}`)",
        f"- Changed-case composition: corrected `{transition_counts['corrected']}`, damaged `{transition_counts['damaged']}`, "
        f"changed-but-correct `{transition_counts['changed_but_correct']}`, wrong-to-wrong `{transition_counts['wrong_to_wrong_changed']}`",
        f"- Evaluator warning: `{corrected_without_contains}/{transition_counts['corrected']}` corrected cases have no contains-match and need manual review.",
        "",
        "## Candidate-Level Diagnosis",
        "",
        "| Experiment | Candidate | Gate | Pairs | Changed | Corrected | Corr. no contains | Damaged | Wrong->Wrong | Events | Semantic Δ | Token Mean Δ | Event->Changed r |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| `{item['experiment']}` | `{item['base_candidate']}` | `{item['gate']}` | `{item['pairs']}` | "
            f"`{item['answer_changed_rate']:.2%}` | `{item['corrected_count']}` | `{item['corrected_without_contains_count']}` | `{item['damaged_count']}` | "
            f"`{item['wrong_to_wrong_changed_count']}` | `{item['events_mean']:.2f}` | "
            f"`{item['semantic_delta_mean']:.4f}` | `{item['token_mean_delta_mean']:.4f}` | "
            f"`{format_float(item['event_answer_changed_corr'], 3)}` |"
        )
    lines.extend(
        [
            "",
            "## Most Perturbed Questions",
            "",
            "| Experiment | Question | Changed | Damaged | Corrected | Mean abs semantic Δ |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in questions[:12]:
        lines.append(
            f"| `{item['experiment']}` | `{item['question_id']}` | `{item['answer_changed_count']}/{item['pairs']}` | "
            f"`{item['damaged_count']}` | `{item['corrected_count']}` | `{item['mean_abs_semantic_delta']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Case Interpretation",
            "",
            f"- Corrected cases: `{len(examples['corrected'])}` shown, from total `{transition_counts['corrected']}`.",
            f"- Damaged cases: `{len(examples['damaged'])}` shown, from total `{transition_counts['damaged']}`.",
            f"- Wrong-to-wrong changed cases shown: `{len(examples['wrong_to_wrong_changed'])}`. These are important because they look like activity but do not improve correctness.",
            f"- High-event no-answer-change cases shown: `{len(examples['high_event_no_answer_change'])}`. These indicate hidden-state perturbation can be absorbed without changing final behavior.",
            "",
            "## Bottom Line",
            "",
            "- If most changed cases are wrong-to-wrong rather than corrected, fixed ITI is acting more like a perturbation probe than a reliable controller.",
            "- Treat NLI-only corrected cases without contains-match as suspicious until manually checked.",
            "- If high-event no-change cases are common, event count alone is not a useful success metric.",
            "- If corrected and damaged cases are sparse and asymmetric across seeds/questions, the next step should be an offline controller/reranker, not broader fixed-vector tuning.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    tag = args.tag or config.get("tag", Path(args.config).stem)
    run_root = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}")
    write_json(run_root / "config_snapshot.json", config)

    rows = load_rows(config)
    summaries = candidate_summary(rows)
    questions = question_summary(rows)
    examples = build_case_examples(rows, config["thresholds"])

    write_csv(run_root / "candidate_diagnosis.csv", summaries)
    write_csv(run_root / "question_diagnosis.csv", questions)
    write_json(run_root / "case_examples.json", examples)
    write_json(
        run_root / "phase1c_summary.json",
        {
            "row_count": len(rows),
            "candidate_count": len(summaries),
            "question_count": len(questions),
            "candidate_diagnosis": summaries,
            "top_questions": questions[:20],
            "case_examples": examples,
        },
    )
    (run_root / "summary.md").write_text(build_markdown(config, rows, summaries, questions, examples), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "row_count": len(rows),
                "candidate_count": len(summaries),
                "question_count": len(questions),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
