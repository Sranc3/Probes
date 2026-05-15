#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_adaptive_semantic_basin_verifier import FEATURE_SETS as ADAPTIVE_FEATURE_SETS  # noqa: E402
from run_prefix_aware_student_policy import STATE_FEATURES  # noqa: E402


DEFAULT_CANDIDATE_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"

BANNED_FRAGMENTS = [
    "ideal",
    "answer_label",
    "correct",
    "strict_correct",
    "representative_strict_correct",
    "sample0_strict_correct",
    "correct_rate",
    "basin_correct_rate",
    "hallucination_score",
    "rescue",
    "damage",
    "oracle",
    "gold",
    "matched_ideal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cost, CI, no-leak, and decision-gate reports for adaptive scaling.")
    parser.add_argument("--candidate-run", default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--adaptive-selection-rows", default="")
    parser.add_argument("--prefix-selection-rows", default="")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="adaptive_scaling_cost_study")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


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
    return sum(values) / len(values) if values else 0.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def group_candidates(rows: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["question_id"]))].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(row["sample_index"]))
    return dict(grouped)


def sample_cost(samples: list[dict[str, Any]], k: int, metric: str) -> float:
    prefix = samples[: max(1, min(k, len(samples)))]
    if metric == "candidate_count":
        return float(len(prefix))
    if metric == "completion_tokens":
        return sum(safe_float(row.get("completion_tokens", row.get("token_count", 0))) for row in prefix)
    if metric == "total_tokens":
        return sum(safe_float(row.get("total_tokens", safe_float(row.get("prompt_tokens", 0)) + safe_float(row.get("token_count", 0)))) for row in prefix)
    if metric == "latency_ms":
        return sum(safe_float(row.get("latency_ms", 0.0)) for row in prefix)
    if metric == "forward_calls":
        return sum(safe_float(row.get("forward_calls", row.get("token_count", 0))) for row in prefix)
    raise ValueError(f"Unsupported cost metric: {metric}")


def pair_costs(samples: list[dict[str, Any]], generated: int) -> dict[str, float]:
    costs: dict[str, float] = {}
    for metric in ["candidate_count", "completion_tokens", "total_tokens", "latency_ms", "forward_calls"]:
        value = sample_cost(samples, generated, metric)
        sample0 = max(1.0, sample_cost(samples, 1, metric))
        full = max(1.0, sample_cost(samples, len(samples), metric))
        costs[f"{metric}_cost"] = value
        costs[f"{metric}_cost_vs_sample0"] = value / sample0
        costs[f"{metric}_cost_vs_full"] = value / full
    return costs


def majority_logprob(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_cluster[int(sample["cluster_id"])].append(sample)
    best_cluster = max(
        by_cluster.values(),
        key=lambda members: (
            len(members),
            safe_float(members[0].get("cluster_weight_mass", 0.0)),
            max(safe_float(row["logprob_avg"]) for row in members),
        ),
    )
    return max(best_cluster, key=lambda row: safe_float(row["logprob_avg"]))


def build_baseline_rows(grouped: dict[tuple[int, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (seed, qid), samples in sorted(grouped.items()):
        sample0 = samples[0]
        full_choice = majority_logprob(samples)
        any_correct = any(safe_float(row["strict_correct"]) > 0 for row in samples)
        for method, selected, generated, stop_reason in [
            ("sample0", sample0, 1, "sample0"),
            ("full8_majority_logprob", full_choice, len(samples), "full_budget"),
        ]:
            delta = safe_float(selected["strict_correct"]) - safe_float(sample0["strict_correct"])
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "question_id": qid,
                    "generated_candidates": generated,
                    "stop_reason": stop_reason,
                    "sample0_correct": safe_float(sample0["strict_correct"]),
                    "selected_correct": safe_float(selected["strict_correct"]),
                    "delta_correct": delta,
                    "answer_changed": float(int(selected["cluster_id"]) != int(sample0["cluster_id"])),
                    "full8_any_correct": float(any_correct),
                    **pair_costs(samples, generated),
                }
            )
    return rows


def rows_from_selection_file(path: Path, grouped: dict[tuple[int, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    raw_rows = read_csv(path)
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        key = (int(row["seed"]), str(row["question_id"]))
        samples = grouped[key]
        generated = int(safe_float(row.get("generated_candidates", len(samples))))
        sample0 = samples[0]
        any_correct = any(safe_float(sample["strict_correct"]) > 0 for sample in samples)
        selected_cluster = int(safe_float(row.get("selected_cluster_id", sample0["cluster_id"])))
        rows.append(
            {
                "method": row["method"],
                "seed": key[0],
                "question_id": key[1],
                "generated_candidates": generated,
                "stop_reason": row.get("stop_reason", ""),
                "sample0_correct": safe_float(row.get("sample0_correct", sample0["strict_correct"])),
                "selected_correct": safe_float(row.get("selected_correct", 0.0)),
                "delta_correct": safe_float(row.get("delta_correct", 0.0)),
                "answer_changed": float(selected_cluster != int(sample0["cluster_id"])),
                "full8_any_correct": float(any_correct),
                **pair_costs(samples, generated),
            }
        )
    return rows


def summarize_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    group = [row for row in rows if row["method"] == method]
    deltas = [safe_float(row["delta_correct"]) for row in group]
    changed = [safe_float(row["answer_changed"]) for row in group]
    return {
        "method": method,
        "pairs": len(group),
        "strict_correct_rate": mean([safe_float(row["selected_correct"]) for row in group]),
        "sample0_strict_correct_rate": mean([safe_float(row["sample0_correct"]) for row in group]),
        "delta_vs_sample0": mean(deltas),
        "improved_count": sum(1 for value in deltas if value > 0),
        "damaged_count": sum(1 for value in deltas if value < 0),
        "net_gain_count": sum(1 for value in deltas if value > 0) - sum(1 for value in deltas if value < 0),
        "answer_changed_rate": mean(changed),
        "avg_generated_candidates": mean([safe_float(row["generated_candidates"]) for row in group]),
        "candidate_cost_vs_sample0": mean([safe_float(row["candidate_count_cost_vs_sample0"]) for row in group]),
        "completion_token_cost_vs_sample0": mean([safe_float(row["completion_tokens_cost_vs_sample0"]) for row in group]),
        "total_token_cost_vs_sample0": mean([safe_float(row["total_tokens_cost_vs_sample0"]) for row in group]),
        "latency_cost_vs_sample0": mean([safe_float(row["latency_ms_cost_vs_sample0"]) for row in group]),
        "completion_token_cost_vs_full8": mean([safe_float(row["completion_tokens_cost_vs_full"]) for row in group]),
        "total_token_cost_vs_full8": mean([safe_float(row["total_tokens_cost_vs_full"]) for row in group]),
        "latency_cost_vs_full8": mean([safe_float(row["latency_ms_cost_vs_full"]) for row in group]),
        "trust_stop_rate": mean([float(row["stop_reason"] == "trust_sample0") for row in group]),
        "switch_stop_rate": mean([float(row["stop_reason"] == "switch") for row in group]),
        "budget_stop_rate": mean([float(row["stop_reason"] in {"budget", "budget_best", "full_budget"}) for row in group]),
    }


def bootstrap_summary(rows: list[dict[str, Any]], method: str, iters: int, rng: random.Random) -> dict[str, Any]:
    group = [row for row in rows if row["method"] == method]
    if not group:
        return {}
    deltas: list[float] = []
    damages: list[float] = []
    stricts: list[float] = []
    token_costs: list[float] = []
    for _ in range(iters):
        sample = [group[rng.randrange(len(group))] for _ in range(len(group))]
        delta_values = [safe_float(row["delta_correct"]) for row in sample]
        deltas.append(mean(delta_values))
        damages.append(sum(1 for value in delta_values if value < 0) / max(1, len(sample)))
        stricts.append(mean([safe_float(row["selected_correct"]) for row in sample]))
        token_costs.append(mean([safe_float(row["total_tokens_cost_vs_sample0"]) for row in sample]))
    return {
        "method": method,
        "delta_ci_low": quantile(deltas, 0.025),
        "delta_ci_high": quantile(deltas, 0.975),
        "damage_rate_ci_low": quantile(damages, 0.025),
        "damage_rate_ci_high": quantile(damages, 0.975),
        "strict_ci_low": quantile(stricts, 0.025),
        "strict_ci_high": quantile(stricts, 0.975),
        "total_token_cost_vs_sample0_ci_low": quantile(token_costs, 0.025),
        "total_token_cost_vs_sample0_ci_high": quantile(token_costs, 0.975),
    }


def stop_reason_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    for method, group in sorted(by_method.items()):
        counts = Counter(str(row["stop_reason"]) for row in group)
        for reason, count in sorted(counts.items()):
            output.append({"method": method, "stop_reason": reason, "count": count, "rate": count / max(1, len(group))})
    return output


def failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    for method, group in sorted(by_method.items()):
        counts = Counter()
        for row in group:
            sample0_correct = safe_float(row["sample0_correct"]) > 0
            selected_correct = safe_float(row["selected_correct"]) > 0
            changed = safe_float(row["answer_changed"]) > 0
            if sample0_correct and not selected_correct:
                counts["sample0_correct_controller_damaged"] += 1
            if not sample0_correct and selected_correct:
                counts["sample0_wrong_controller_rescued"] += 1
            if safe_float(row["generated_candidates"]) > 1 and not changed:
                counts["spent_cost_no_answer_change"] += 1
            if safe_float(row["generated_candidates"]) < 8 and not selected_correct and safe_float(row["full8_any_correct"]) > 0:
                counts["stopped_early_missed_rescue_candidate"] += 1
        for category, count in sorted(counts.items()):
            output.append({"method": method, "failure_category": category, "count": count, "rate": count / max(1, len(group))})
    return output


def no_leak_audit_rows() -> list[dict[str, Any]]:
    feature_sets: dict[str, list[str]] = {f"adaptive_{name}": list(features) for name, features in ADAPTIVE_FEATURE_SETS.items()}
    feature_sets["prefix_state_features"] = list(STATE_FEATURES)
    rows: list[dict[str, Any]] = []
    for set_name, features in sorted(feature_sets.items()):
        for feature in features:
            lower = feature.lower()
            hits = [fragment for fragment in BANNED_FRAGMENTS if fragment in lower]
            rows.append(
                {
                    "feature_set": set_name,
                    "feature": feature,
                    "status": "fail" if hits else "pass",
                    "banned_fragments": ",".join(hits),
                    "note": "feature name suggests gold/label leakage" if hits else "name-only audit passed",
                }
            )
    return rows


def decision_gate(summary_rows: list[dict[str, Any]], pair_count: int) -> dict[str, Any]:
    adaptive_rows = [row for row in summary_rows if str(row["method"]).startswith(("adaptive_", "teacher_value_"))]
    best = max(adaptive_rows, key=lambda row: (safe_float(row["delta_vs_sample0"]), -safe_float(row["total_token_cost_vs_sample0"])), default=None)
    if best is None:
        return {"recommendation": "run_generation_time_controllers", "reason": "No adaptive/value-policy rows were provided."}
    delta = safe_float(best["delta_vs_sample0"])
    cost_vs_full = safe_float(best["total_token_cost_vs_full8"])
    damage = int(best["damaged_count"])
    if pair_count < 1000:
        scale_note = "Current evidence is below the planned >=1000 seed-question pair scale."
    else:
        scale_note = "Current evidence reaches the planned TriviaQA scale."
    if delta >= 0.02 and damage <= max(2, int(0.01 * pair_count)) and cost_vs_full < 0.75:
        recommendation = "continue_adaptive_inference"
        reason = f"{best['method']} reaches >=2pp gain with controlled damage and material token savings vs full-8. {scale_note}"
    else:
        recommendation = "prepare_cross_dataset_then_consider_grpo"
        reason = f"Best generation-time row is {best['method']} with delta={delta:.2%}, damage={damage}, total-token cost vs full8={cost_vs_full:.2f}. {scale_note}"
    return {"recommendation": recommendation, "best_method": best["method"], "reason": reason}


def write_report(output_dir: Path, summary_rows: list[dict[str, Any]], ci_rows: list[dict[str, Any]], decision: dict[str, Any], pair_count: int) -> None:
    ci_by_method = {row["method"]: row for row in ci_rows}
    lines = [
        "# Adaptive Controller Scaling and Real-Cost Report",
        "",
        f"- Seed-question pairs: `{pair_count}`",
        "- Correctness labels are used only for final evaluation and bootstrap intervals.",
        "- Cost views include candidate count, completion tokens, total prompt+completion tokens, latency, and forward-call proxies.",
        "",
        "## Controller Benchmark",
        "",
        "| Method | Strict | Delta | 95% CI Delta | Improved | Damaged | Avg Gen | Total Token Cost vs 1x | Total Token Cost vs Full8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary_rows, key=lambda item: (-safe_float(item["delta_vs_sample0"]), safe_float(item["total_token_cost_vs_sample0"]))):
        ci = ci_by_method.get(row["method"], {})
        ci_text = f"[{safe_float(ci.get('delta_ci_low')):.2%}, {safe_float(ci.get('delta_ci_high')):.2%}]" if ci else ""
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | `{ci_text}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{safe_float(row['avg_generated_candidates']):.2f}` | "
            f"`{safe_float(row['total_token_cost_vs_sample0']):.2f}x` | `{safe_float(row['total_token_cost_vs_full8']):.2f}x` |"
        )
    lines.extend(
        [
            "",
            "## Decision Gate",
            "",
            f"- Recommendation: `{decision['recommendation']}`",
            f"- Reason: {decision['reason']}",
            "",
            "## No-Leak Rule",
            "",
            "The controller feature audit is name-based and intentionally conservative. Any failed row in `no_leak_feature_audit.csv` must be inspected before a feature set is allowed into a controller.",
        ]
    )
    (output_dir / "adaptive_scaling_cost_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidate_run = Path(args.candidate_run)
    output_dir = Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=False)

    grouped = group_candidates(read_csv(candidate_run / "candidate_features.csv"))
    controller_rows = build_baseline_rows(grouped)
    if args.adaptive_selection_rows:
        controller_rows.extend(rows_from_selection_file(Path(args.adaptive_selection_rows), grouped))
    if args.prefix_selection_rows:
        controller_rows.extend(rows_from_selection_file(Path(args.prefix_selection_rows), grouped))

    methods = sorted({row["method"] for row in controller_rows})
    summary_rows = [summarize_method(controller_rows, method) for method in methods]
    rng = random.Random(args.seed)
    ci_rows = [bootstrap_summary(controller_rows, method, args.bootstrap_iters, rng) for method in methods]
    audit_rows = no_leak_audit_rows()
    failures = failure_rows(controller_rows)
    decision = decision_gate(summary_rows, len(grouped))

    write_csv(output_dir / "controller_pair_cost_rows.csv", controller_rows)
    write_csv(output_dir / "controller_cost_accuracy_summary.csv", summary_rows)
    write_csv(output_dir / "controller_bootstrap_ci.csv", ci_rows)
    write_csv(output_dir / "stop_reason_distribution.csv", stop_reason_rows(controller_rows))
    write_csv(output_dir / "failure_analysis.csv", failures)
    write_csv(output_dir / "no_leak_feature_audit.csv", audit_rows)
    write_json(
        output_dir / "run_metadata.json",
        {
            "candidate_run": str(candidate_run),
            "pair_count": len(grouped),
            "adaptive_selection_rows": args.adaptive_selection_rows,
            "prefix_selection_rows": args.prefix_selection_rows,
            "bootstrap_iters": args.bootstrap_iters,
            "decision_gate": decision,
            "audit_fail_count": sum(1 for row in audit_rows if row["status"] == "fail"),
        },
    )
    write_report(output_dir, summary_rows, ci_rows, decision, len(grouped))
    print(json.dumps({"output_dir": str(output_dir), "pair_count": len(grouped), "decision": decision}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
