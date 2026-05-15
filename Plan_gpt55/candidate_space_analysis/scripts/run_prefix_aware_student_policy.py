#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from run_adaptive_semantic_basin_verifier import (
    FEATURE_SETS,
    TorchLogisticModel,
    build_prefix_dataset,
    choose_best_basin,
    choose_prefix_basin,
    ensure_dir,
    mean,
    read_csv,
    safe_float,
    sample0_at,
    write_csv,
    write_json,
)
from run_learned_basin_verifier import question_folds


DEFAULT_CANDIDATE_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"


STATE_FEATURES = [
    "state_prefix_size",
    "state_prefix_frac",
    "state_remaining_budget",
    "state_semantic_clusters",
    "state_fragmentation_entropy",
    "state_top_cluster_margin",
    "state_sample0_mass",
    "state_sample0_entropy",
    "state_sample0_logprob",
    "state_sample0_len",
    "state_sample0_type_match",
    "state_sample0_bad_marker",
    "state_has_alt",
    "state_best_alt_mass",
    "state_best_alt_entropy",
    "state_best_alt_logprob",
    "state_best_alt_len",
    "state_best_alt_type_match",
    "state_best_alt_bad_marker",
    "state_best_alt_overlap",
    "state_best_alt_core_size",
    "state_alt_minus_sample0_mass",
    "state_alt_minus_sample0_entropy",
    "state_alt_minus_sample0_logprob",
    "state_alt_heuristic_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prefix-aware student / continue-value policy.")
    parser.add_argument("--candidate-run", default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="prefix_aware_student_policy")
    parser.add_argument("--max-candidates", type=int, default=8)
    return parser.parse_args()


def basin_heuristic(row: dict[str, Any]) -> float:
    return (
        safe_float(row.get("obs_cluster_mass", 0.0))
        + 0.35 * safe_float(row.get("rep_logprob_z", 0.0))
        - 0.25 * safe_float(row.get("rep_entropy_z", 0.0))
        + 0.15 * safe_float(row.get("type_match_score", 0.0))
        - 0.25 * safe_float(row.get("bad_marker_score", 0.0))
    )


def best_alt(prefix: dict[str, Any]) -> dict[str, Any] | None:
    alternatives = [row for row in prefix["basins"] if safe_float(row["contains_sample0"]) <= 0]
    if not alternatives:
        return None
    return max(alternatives, key=basin_heuristic)


def state_row(prefixes: list[dict[str, Any]], prefix_idx: int, max_candidates: int) -> dict[str, Any]:
    prefix = prefixes[prefix_idx]
    sample0 = sample0_at(prefix)
    alt = best_alt(prefix)
    full_oracle_correct = float(any(safe_float(row["representative_strict_correct"]) > 0 for row in prefixes[-1]["basins"]))
    current_oracle_correct = float(any(safe_float(row["representative_strict_correct"]) > 0 for row in prefix["basins"]))
    continue_label = float(current_oracle_correct <= 0 and full_oracle_correct > 0 and int(prefix["prefix_size"]) < max_candidates)
    row: dict[str, Any] = {
        "seed": sample0["seed"],
        "question_id": sample0["question_id"],
        "prefix_size": int(prefix["prefix_size"]),
        "state_prefix_size": int(prefix["prefix_size"]),
        "state_prefix_frac": int(prefix["prefix_size"]) / max_candidates,
        "state_remaining_budget": (max_candidates - int(prefix["prefix_size"])) / max_candidates,
        "state_semantic_clusters": sample0["obs_semantic_clusters"],
        "state_fragmentation_entropy": sample0["obs_fragmentation_entropy"],
        "state_top_cluster_margin": sample0["obs_top_cluster_margin"],
        "state_sample0_mass": sample0["obs_cluster_mass"],
        "state_sample0_entropy": sample0["rep_entropy_z"],
        "state_sample0_logprob": sample0["rep_logprob_z"],
        "state_sample0_len": sample0["rep_len_z"],
        "state_sample0_type_match": sample0["type_match_score"],
        "state_sample0_bad_marker": sample0["bad_marker_score"],
        "state_has_alt": float(alt is not None),
        "current_oracle_correct": current_oracle_correct,
        "full_oracle_correct": full_oracle_correct,
        "continue_label": continue_label,
    }
    if alt is None:
        row.update(
            {
                "state_best_alt_mass": 0.0,
                "state_best_alt_entropy": 0.0,
                "state_best_alt_logprob": 0.0,
                "state_best_alt_len": 0.0,
                "state_best_alt_type_match": 0.0,
                "state_best_alt_bad_marker": 0.0,
                "state_best_alt_overlap": 0.0,
                "state_best_alt_core_size": 0.0,
                "state_alt_minus_sample0_mass": 0.0,
                "state_alt_minus_sample0_entropy": 0.0,
                "state_alt_minus_sample0_logprob": 0.0,
                "state_alt_heuristic_gap": 0.0,
            }
        )
    else:
        row.update(
            {
                "state_best_alt_mass": alt["obs_cluster_mass"],
                "state_best_alt_entropy": alt["rep_entropy_z"],
                "state_best_alt_logprob": alt["rep_logprob_z"],
                "state_best_alt_len": alt["rep_len_z"],
                "state_best_alt_type_match": alt["type_match_score"],
                "state_best_alt_bad_marker": alt["bad_marker_score"],
                "state_best_alt_overlap": alt["question_answer_content_overlap"],
                "state_best_alt_core_size": alt["consensus_core_size"],
                "state_alt_minus_sample0_mass": safe_float(alt["obs_cluster_mass"]) - safe_float(sample0["obs_cluster_mass"]),
                "state_alt_minus_sample0_entropy": safe_float(alt["rep_entropy_z"]) - safe_float(sample0["rep_entropy_z"]),
                "state_alt_minus_sample0_logprob": safe_float(alt["rep_logprob_z"]) - safe_float(sample0["rep_logprob_z"]),
                "state_alt_heuristic_gap": basin_heuristic(alt) - basin_heuristic(sample0),
            }
        )
    return row


def build_state_rows(by_pair_prefixes: dict[tuple[int, str], list[dict[str, Any]]], max_candidates: int) -> dict[tuple[int, str], list[dict[str, Any]]]:
    rows = {}
    for key, prefixes in by_pair_prefixes.items():
        rows[key] = [state_row(prefixes, idx, max_candidates) for idx in range(len(prefixes))]
    return rows


def generated_token_cost(prefixes: list[dict[str, Any]], generated: int) -> float:
    prefix = prefixes[min(generated, len(prefixes)) - 1]
    return sum(safe_float(sample["token_count"]) for sample in prefix["samples"])


def simulate_pair(
    prefixes: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    trust_model: TorchLogisticModel,
    switch_model: TorchLogisticModel,
    basin_model: TorchLogisticModel,
    continue_model: TorchLogisticModel,
    trust_threshold: float,
    switch_threshold: float,
    continue_threshold: float,
    basin_margin: float,
    min_candidates: int,
    terminal_best: bool,
    max_candidates: int,
) -> tuple[dict[str, Any], int, str]:
    for idx, prefix in enumerate(prefixes[:max_candidates]):
        sample0 = sample0_at(prefix)
        choice = choose_prefix_basin(prefix, switch_model, switch_threshold, basin_model, basin_margin)
        if int(prefix["prefix_size"]) >= min_candidates and int(choice["cluster_id"]) != int(sample0["cluster_id"]):
            return choice, int(prefix["prefix_size"]), "switch"
        if trust_model.predict(sample0) >= trust_threshold:
            return sample0, int(prefix["prefix_size"]), "trust_sample0"
        if int(prefix["prefix_size"]) < min_candidates:
            continue
        if int(prefix["prefix_size"]) >= max_candidates:
            break
        if continue_model.predict(state_rows[idx]) >= continue_threshold:
            continue
        return choice, int(prefix["prefix_size"]), "stop_low_continue"
    final_prefix = prefixes[min(max_candidates, len(prefixes)) - 1]
    if terminal_best:
        return choose_best_basin(final_prefix, basin_model, basin_margin), int(final_prefix["prefix_size"]), "terminal_best"
    return choose_prefix_basin(final_prefix, switch_model, switch_threshold, basin_model, basin_margin), int(final_prefix["prefix_size"]), "budget"


def evaluate(
    items: list[tuple[tuple[int, str], list[dict[str, Any]], list[dict[str, Any]]]],
    selections: list[tuple[dict[str, Any], int, str]],
    method: str,
    split: str,
) -> dict[str, Any]:
    sample0s = [sample0_at(prefixes[0]) for _key, prefixes, _states in items]
    selected = [row for row, _generated, _reason in selections]
    generated = [generated for _row, generated, _reason in selections]
    reasons = Counter(reason for _row, _generated, reason in selections)
    generated_tokens = [generated_token_cost(prefixes, gen) for (_key, prefixes, _states), gen in zip(items, generated)]
    sample0_tokens = [safe_float(prefixes[0]["samples"][0]["token_count"]) for _key, prefixes, _states in items]
    full_tokens = [generated_token_cost(prefixes, len(prefixes)) for _key, prefixes, _states in items]
    deltas = [safe_float(row["representative_strict_correct"]) - safe_float(s0["representative_strict_correct"]) for row, s0 in zip(selected, sample0s)]
    return {
        "split": split,
        "method": method,
        "pairs": len(items),
        "strict_correct_rate": mean([safe_float(row["representative_strict_correct"]) for row in selected]),
        "sample0_strict_correct_rate": mean([safe_float(row["representative_strict_correct"]) for row in sample0s]),
        "delta_vs_sample0": mean(deltas),
        "improved_count": sum(1 for value in deltas if value > 0),
        "damaged_count": sum(1 for value in deltas if value < 0),
        "net_gain_count": sum(1 for value in deltas if value > 0) - sum(1 for value in deltas if value < 0),
        "answer_changed_rate": mean([float(int(row["cluster_id"]) != int(s0["cluster_id"])) for row, s0 in zip(selected, sample0s)]),
        "avg_generated_candidates": mean(generated),
        "avg_generated_tokens": mean(generated_tokens),
        "token_cost_vs_sample0": mean([gen / max(1.0, base) for gen, base in zip(generated_tokens, sample0_tokens)]),
        "token_savings_vs_full8": 1.0 - mean([gen / max(1.0, full) for gen, full in zip(generated_tokens, full_tokens)]),
        "trust_stop_rate": reasons["trust_sample0"] / max(1, len(items)),
        "switch_stop_rate": reasons["switch"] / max(1, len(items)),
        "continue_stop_rate": reasons["stop_low_continue"] / max(1, len(items)),
        "terminal_best_rate": reasons["terminal_best"] / max(1, len(items)),
        "budget_stop_rate": reasons["budget"] / max(1, len(items)),
    }


def tune_policy(
    train_items: list[tuple[tuple[int, str], list[dict[str, Any]], list[dict[str, Any]]]],
    trust_model: TorchLogisticModel,
    switch_model: TorchLogisticModel,
    basin_model: TorchLogisticModel,
    continue_model: TorchLogisticModel,
    profile: str,
    max_candidates: int,
) -> dict[str, Any]:
    penalties = {"quality": 0.2, "balanced": 0.8, "production": 1.8}
    damage_penalties = {"quality": 2.5, "balanced": 4.0, "production": 6.0}
    best: tuple[float, dict[str, Any]] = (-1e9, {})
    for trust_threshold in [0.45, 0.6, 0.75, 0.88]:
        for switch_threshold in [0.45, 0.6, 0.75]:
            for continue_threshold in [0.25, 0.4, 0.55, 0.7]:
                for margin in [-0.2, 0.0, 0.2]:
                    for min_candidates in [2, 3, 4]:
                        for terminal_best in [False, True]:
                            selections = [
                                simulate_pair(prefixes, states, trust_model, switch_model, basin_model, continue_model, trust_threshold, switch_threshold, continue_threshold, margin, min_candidates, terminal_best, max_candidates)
                                for _key, prefixes, states in train_items
                            ]
                            metrics = evaluate(train_items, selections, "train", "train")
                            score = (
                                100 * safe_float(metrics["delta_vs_sample0"])
                                + int(metrics["net_gain_count"])
                                - damage_penalties[profile] * int(metrics["damaged_count"])
                                - penalties[profile] * safe_float(metrics["avg_generated_candidates"])
                            )
                            if score > best[0]:
                                best = (
                                    score,
                                    {
                                        "trust_threshold": trust_threshold,
                                        "switch_threshold": switch_threshold,
                                        "continue_threshold": continue_threshold,
                                        "basin_margin": margin,
                                        "min_candidates": min_candidates,
                                        "terminal_best": terminal_best,
                                    },
                                )
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
                "avg_generated_candidates": mean([safe_float(item["avg_generated_candidates"]) for item in items]),
                "token_cost_vs_sample0": mean([safe_float(item["token_cost_vs_sample0"]) for item in items]),
                "token_savings_vs_full8": mean([safe_float(item["token_savings_vs_full8"]) for item in items]),
                "trust_stop_rate": mean([safe_float(item["trust_stop_rate"]) for item in items]),
                "switch_stop_rate": mean([safe_float(item["switch_stop_rate"]) for item in items]),
                "continue_stop_rate": mean([safe_float(item["continue_stop_rate"]) for item in items]),
                "terminal_best_rate": mean([safe_float(item["terminal_best_rate"]) for item in items]),
                "budget_stop_rate": mean([safe_float(item["budget_stop_rate"]) for item in items]),
            }
        )
    return summary


def run_cv(by_pair_prefixes: dict[tuple[int, str], list[dict[str, Any]]], by_pair_states: dict[tuple[int, str], list[dict[str, Any]]], max_candidates: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    keys = sorted(by_pair_prefixes)
    fold_rows = []
    selection_rows = []
    state_weight_rows = []
    for fold_idx, (train_qids, test_qids) in enumerate(question_folds(keys)):
        train_items = [(key, by_pair_prefixes[key], by_pair_states[key]) for key in keys if key[1] in train_qids]
        test_items = [(key, by_pair_prefixes[key], by_pair_states[key]) for key in keys if key[1] in test_qids]
        train_sample0_rows = [sample0_at(prefixes[0]) for _key, prefixes, _states in train_items]
        train_alt_rows = [row for _key, prefixes, _states in train_items for prefix in prefixes for row in prefix["basins"] if safe_float(row["contains_sample0"]) <= 0]
        train_all_rows = [row for _key, prefixes, _states in train_items for prefix in prefixes for row in prefix["basins"]]
        train_state_rows = [row for _key, _prefixes, states in train_items for row in states]
        for feature_name, basin_features in [("numeric", FEATURE_SETS["numeric_adaptive"]), ("full", FEATURE_SETS["full_adaptive"])]:
            trust_model = TorchLogisticModel(basin_features)
            trust_model.fit(train_sample0_rows, "representative_strict_correct", epochs=220, lr=0.08, l2=0.02)
            switch_model = TorchLogisticModel(basin_features)
            switch_model.fit(train_alt_rows, "switch_gain_label", epochs=220, lr=0.08, l2=0.02)
            basin_model = TorchLogisticModel(basin_features)
            basin_model.fit(train_all_rows, "representative_strict_correct", epochs=220, lr=0.08, l2=0.02)
            continue_model = TorchLogisticModel(STATE_FEATURES)
            continue_model.fit(train_state_rows, "continue_label", epochs=220, lr=0.08, l2=0.02)
            for profile in ["quality", "balanced", "production"]:
                params = tune_policy(train_items, trust_model, switch_model, basin_model, continue_model, profile, max_candidates)
                selections = [
                    simulate_pair(prefixes, states, trust_model, switch_model, basin_model, continue_model, params["trust_threshold"], params["switch_threshold"], params["continue_threshold"], params["basin_margin"], params["min_candidates"], params["terminal_best"], max_candidates)
                    for _key, prefixes, states in test_items
                ]
                method = f"prefix_student_{profile}_{feature_name}"
                metrics = evaluate(test_items, selections, method, f"fold_{fold_idx}")
                metrics.update({"feature_set": feature_name, **params})
                fold_rows.append(metrics)
                state_weight_rows.extend(continue_model.weight_rows(f"{method}_continue", f"fold_{fold_idx}", top_k=14))
                for (key, _prefixes, _states), (row, generated, reason) in zip(test_items, selections):
                    s0 = sample0_at(by_pair_prefixes[key][0])
                    selection_rows.append(
                        {
                            "split": f"fold_{fold_idx}",
                            "method": method,
                            "seed": key[0],
                            "question_id": key[1],
                            "generated_candidates": generated,
                            "stop_reason": reason,
                            "sample0_correct": s0["representative_strict_correct"],
                            "selected_correct": row["representative_strict_correct"],
                            "delta_correct": safe_float(row["representative_strict_correct"]) - safe_float(s0["representative_strict_correct"]),
                            "sample0_cluster_id": s0["cluster_id"],
                            "selected_cluster_id": row["cluster_id"],
                            "selected_preview": row["representative_preview"],
                        }
                    )
    return summarize(fold_rows) + fold_rows, selection_rows, state_weight_rows


def write_report(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary = [row for row in rows if row["split"] == "question_grouped_cv"]
    lines = [
        "# Prefix-Aware Student / Continue-Value Policy",
        "",
        "## Summary",
        "",
        "| Method | Strict | Delta | Improved | Damaged | Net | Avg Gen | Cost vs 1x | Save vs 8x | Trust | Switch | Continue Stop | Terminal Best |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{int(row['net_gain_count'])}` | "
            f"`{safe_float(row['avg_generated_candidates']):.2f}` | `{safe_float(row['token_cost_vs_sample0']):.2f}x` | `{safe_float(row['token_savings_vs_full8']):.2%}` | "
            f"`{safe_float(row['trust_stop_rate']):.2%}` | `{safe_float(row['switch_stop_rate']):.2%}` | `{safe_float(row['continue_stop_rate']):.2%}` | `{safe_float(row['terminal_best_rate']):.2%}` |"
        )
    best = max(summary, key=lambda row: (safe_float(row["delta_vs_sample0"]) - 0.01 * safe_float(row["avg_generated_candidates"]) - 0.02 * int(row["damaged_count"]))) if summary else None
    if best:
        lines.extend(
            [
                "",
                "## 初步判断",
                "",
                f"当前综合成本和收益最值得关注的是 `{best['method']}`：strict `{safe_float(best['strict_correct_rate']):.2%}`，delta `{safe_float(best['delta_vs_sample0']):.2%}`，avg generated `{safe_float(best['avg_generated_candidates']):.2f}`，damage `{int(best['damaged_count'])}`。",
                "",
                "这轮实验的关键不是固定阈值搜索，而是显式学习 continue value：当 prefix 还没有可靠答案但未来可能出现 rescue basin 时，策略应该继续采样；否则应尽早停止。",
            ]
        )
    (output_dir / "prefix_aware_student_findings_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    candidate_rows = read_csv(Path(args.candidate_run) / "candidate_features.csv")
    by_pair_prefixes, _sample0_rows, _alt_rows = build_prefix_dataset(candidate_rows, args.max_candidates)
    by_pair_states = build_state_rows(by_pair_prefixes, args.max_candidates)
    cv_rows, selection_rows, state_weight_rows = run_cv(by_pair_prefixes, by_pair_states, args.max_candidates)
    write_csv(output_dir / "prefix_student_cv_results.csv", cv_rows)
    write_csv(output_dir / "prefix_student_selection_rows.csv", selection_rows)
    write_csv(output_dir / "prefix_student_continue_weights.csv", state_weight_rows)
    write_json(output_dir / "run_metadata.json", {"pair_count": len(by_pair_prefixes), "max_candidates": args.max_candidates, "state_features": STATE_FEATURES})
    write_report(output_dir, cv_rows)
    print(json.dumps({"output_dir": str(output_dir), "pair_count": len(by_pair_prefixes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
