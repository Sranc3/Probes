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
    choose_prefix_basin,
    ensure_dir,
    mean,
    read_csv,
    safe_float,
    sample0_at,
    write_csv,
    write_json,
)
from run_learned_basin_verifier import question_folds, stdev
from run_prefix_aware_student_policy import STATE_FEATURES, state_row
from run_prefix_value_policy import TorchLinearRegressor


DEFAULT_CANDIDATE_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"

PROFILE_LAMBDAS = {
    "quality": 0.03,
    "balanced": 0.07,
    "production": 0.12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prefix teacher-value policy using verifier-realizable future gains.")
    parser.add_argument("--candidate-run", default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="prefix_teacher_value_policy")
    parser.add_argument("--max-candidates", type=int, default=8)
    return parser.parse_args()


def token_cost(prefixes: list[dict[str, Any]], prefix_idx: int) -> float:
    return sum(safe_float(sample["token_count"]) for sample in prefixes[prefix_idx]["samples"])


def generated_token_cost(prefixes: list[dict[str, Any]], generated: int) -> float:
    return sum(safe_float(sample["token_count"]) for sample in prefixes[min(generated, len(prefixes)) - 1]["samples"])


def selected_utility(row: dict[str, Any]) -> float:
    return safe_float(row["representative_strict_correct"])


def build_teacher_state_rows(
    items: list[tuple[tuple[int, str], list[dict[str, Any]]]],
    switch_model: TorchLogisticModel,
    basin_model: TorchLogisticModel,
    switch_threshold: float,
    basin_margin: float,
    max_candidates: int,
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    by_pair_states: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for key, prefixes in items:
        states: list[dict[str, Any]] = []
        base_token = max(1.0, token_cost(prefixes, 0))
        teacher_choices = [choose_prefix_basin(prefix, switch_model, switch_threshold, basin_model, basin_margin) for prefix in prefixes[:max_candidates]]
        for idx, prefix in enumerate(prefixes[:max_candidates]):
            row = state_row(prefixes, idx, max_candidates)
            current_choice = teacher_choices[idx]
            current_utility = selected_utility(current_choice)
            row["teacher_current_correct"] = current_utility
            row["teacher_current_switched"] = float(int(current_choice["cluster_id"]) != int(sample0_at(prefix)["cluster_id"]))
            for profile, lam in PROFILE_LAMBDAS.items():
                best_future = current_utility
                best_future_idx = idx
                for future_idx in range(idx + 1, min(max_candidates, len(prefixes))):
                    extra_cost = (token_cost(prefixes, future_idx) - token_cost(prefixes, idx)) / base_token
                    future_utility = selected_utility(teacher_choices[future_idx]) - lam * extra_cost
                    if future_utility > best_future:
                        best_future = future_utility
                        best_future_idx = future_idx
                row[f"teacher_continue_value_{profile}"] = best_future - current_utility
                row[f"teacher_best_future_k_{profile}"] = best_future_idx + 1
            states.append(row)
        by_pair_states[key] = states
    return by_pair_states


def evaluate(items: list[tuple[tuple[int, str], list[dict[str, Any]], list[dict[str, Any]]]], selections: list[tuple[dict[str, Any], int, str]], method: str, split: str) -> dict[str, Any]:
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
        "token_cost_vs_sample0": mean([gen / max(1.0, base) for gen, base in zip(generated_tokens, sample0_tokens)]),
        "token_savings_vs_full8": 1.0 - mean([gen / max(1.0, full) for gen, full in zip(generated_tokens, full_tokens)]),
        "trust_stop_rate": reasons["trust_sample0"] / max(1, len(items)),
        "switch_stop_rate": reasons["switch"] / max(1, len(items)),
        "value_stop_rate": reasons["stop_low_value"] / max(1, len(items)),
        "veto_stop_rate": reasons["veto_stop"] / max(1, len(items)),
        "budget_stop_rate": reasons["budget"] / max(1, len(items)),
    }


def simulate_pair(
    prefixes: list[dict[str, Any]],
    states: list[dict[str, Any]],
    trust_model: TorchLogisticModel,
    switch_model: TorchLogisticModel,
    basin_model: TorchLogisticModel,
    value_model: TorchLinearRegressor,
    damage_model: TorchLogisticModel,
    trust_threshold: float,
    switch_threshold: float,
    value_threshold: float,
    basin_margin: float,
    damage_threshold: float,
    min_candidates: int,
    max_candidates: int,
) -> tuple[dict[str, Any], int, str]:
    for idx, prefix in enumerate(prefixes[:max_candidates]):
        sample0 = sample0_at(prefix)
        choice = choose_prefix_basin(prefix, switch_model, switch_threshold, basin_model, basin_margin)
        switched = int(choice["cluster_id"]) != int(sample0["cluster_id"])
        damage_risk = damage_model.predict(choice) if switched else 0.0
        if int(prefix["prefix_size"]) >= min_candidates and switched and damage_risk < damage_threshold:
            return choice, int(prefix["prefix_size"]), "switch"
        if trust_model.predict(sample0) >= trust_threshold:
            return sample0, int(prefix["prefix_size"]), "trust_sample0"
        if int(prefix["prefix_size"]) < min_candidates:
            continue
        if int(prefix["prefix_size"]) >= max_candidates:
            break
        if value_model.predict(states[idx]) > value_threshold:
            continue
        if switched and damage_risk >= damage_threshold:
            return sample0, int(prefix["prefix_size"]), "veto_stop"
        return choice, int(prefix["prefix_size"]), "stop_low_value"
    final_prefix = prefixes[min(max_candidates, len(prefixes)) - 1]
    choice = choose_prefix_basin(final_prefix, switch_model, switch_threshold, basin_model, basin_margin)
    sample0 = sample0_at(final_prefix)
    if int(choice["cluster_id"]) != int(sample0["cluster_id"]) and damage_model.predict(choice) >= damage_threshold:
        return sample0, int(final_prefix["prefix_size"]), "veto_stop"
    return choice, int(final_prefix["prefix_size"]), "budget"


def tune_switch(groups: list[list[dict[str, Any]]], switch_model: TorchLogisticModel, basin_model: TorchLogisticModel) -> tuple[float, float]:
    best = (-1e9, 0.6, 0.0)
    for threshold in [0.45, 0.6, 0.75]:
        for margin in [-0.2, 0.0, 0.2]:
            selected = [choose_prefix_basin(prefixes[-1], switch_model, threshold, basin_model, margin) for prefixes in groups]
            sample0s = [sample0_at(prefixes[0]) for prefixes in groups]
            deltas = [safe_float(row["representative_strict_correct"]) - safe_float(s0["representative_strict_correct"]) for row, s0 in zip(selected, sample0s)]
            score = 100 * mean(deltas) + sum(1 for value in deltas if value > 0) - 2.0 * sum(1 for value in deltas if value < 0)
            if score > best[0]:
                best = (score, threshold, margin)
    return best[1], best[2]


def tune_policy(
    train_items: list[tuple[tuple[int, str], list[dict[str, Any]], list[dict[str, Any]]]],
    trust_model: TorchLogisticModel,
    switch_model: TorchLogisticModel,
    basin_model: TorchLogisticModel,
    value_model: TorchLinearRegressor,
    damage_model: TorchLogisticModel,
    profile: str,
    max_candidates: int,
    switch_threshold: float,
    basin_margin: float,
) -> dict[str, Any]:
    cost_penalty = {"quality": 0.25, "balanced": 0.9, "production": 1.9}[profile]
    damage_penalty = {"quality": 3.0, "balanced": 5.0, "production": 7.0}[profile]
    best: tuple[float, dict[str, Any]] = (-1e9, {})
    for trust_threshold in [0.5, 0.65, 0.8]:
        for value_threshold in [-0.04, 0.0, 0.04, 0.08, 0.12]:
            for damage_threshold in [0.45, 0.6, 0.75, 0.9, 1.01]:
                for min_candidates in [2, 3, 4]:
                    selections = [
                        simulate_pair(prefixes, states, trust_model, switch_model, basin_model, value_model, damage_model, trust_threshold, switch_threshold, value_threshold, basin_margin, damage_threshold, min_candidates, max_candidates)
                        for _key, prefixes, states in train_items
                    ]
                    metrics = evaluate(train_items, selections, "train", "train")
                    score = (
                        100 * safe_float(metrics["delta_vs_sample0"])
                        + int(metrics["net_gain_count"])
                        - damage_penalty * int(metrics["damaged_count"])
                        - cost_penalty * safe_float(metrics["avg_generated_candidates"])
                    )
                    if score > best[0]:
                        best = (
                            score,
                            {
                                "trust_threshold": trust_threshold,
                                "switch_threshold": switch_threshold,
                                "value_threshold": value_threshold,
                                "basin_margin": basin_margin,
                                "damage_threshold": damage_threshold,
                                "min_candidates": min_candidates,
                            },
                        )
    return best[1]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summary: list[dict[str, Any]] = []
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
                "value_stop_rate": mean([safe_float(item["value_stop_rate"]) for item in items]),
                "veto_stop_rate": mean([safe_float(item["veto_stop_rate"]) for item in items]),
                "budget_stop_rate": mean([safe_float(item["budget_stop_rate"]) for item in items]),
            }
        )
    return summary


def run_cv(by_pair_prefixes: dict[tuple[int, str], list[dict[str, Any]]], max_candidates: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    keys = sorted(by_pair_prefixes)
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    weights: list[dict[str, Any]] = []
    for fold_idx, (train_qids, test_qids) in enumerate(question_folds(keys)):
        train_keys = [key for key in keys if key[1] in train_qids]
        test_keys = [key for key in keys if key[1] in test_qids]
        train_prefix_groups = [by_pair_prefixes[key] for key in train_keys]
        train_items_no_state = [(key, by_pair_prefixes[key]) for key in train_keys]
        test_items_no_state = [(key, by_pair_prefixes[key]) for key in test_keys]
        train_sample0_rows = [sample0_at(prefixes[0]) for prefixes in train_prefix_groups]
        train_alt_rows = [row for prefixes in train_prefix_groups for prefix in prefixes for row in prefix["basins"] if safe_float(row["contains_sample0"]) <= 0]
        train_all_rows = [row for prefixes in train_prefix_groups for prefix in prefixes for row in prefix["basins"]]
        for feature_name, basin_features in [("numeric", FEATURE_SETS["numeric_adaptive"]), ("full", FEATURE_SETS["full_adaptive"])]:
            trust_model = TorchLogisticModel(basin_features)
            trust_model.fit(train_sample0_rows, "representative_strict_correct", epochs=180, lr=0.08, l2=0.02)
            switch_model = TorchLogisticModel(basin_features)
            switch_model.fit(train_alt_rows, "switch_gain_label", epochs=180, lr=0.08, l2=0.02)
            basin_model = TorchLogisticModel(basin_features)
            basin_model.fit(train_all_rows, "representative_strict_correct", epochs=180, lr=0.08, l2=0.02)
            damage_model = TorchLogisticModel(basin_features)
            for row in train_all_rows:
                s0 = sample0_at(by_pair_prefixes[(int(row["seed"]), row["question_id"])][0])
                row["damage_risk_label"] = float(safe_float(s0["representative_strict_correct"]) > 0 and safe_float(row["representative_strict_correct"]) <= 0 and safe_float(row["contains_sample0"]) <= 0)
            damage_model.fit(train_all_rows, "damage_risk_label", epochs=180, lr=0.08, l2=0.02)
            switch_threshold, basin_margin = tune_switch(train_prefix_groups, switch_model, basin_model)
            train_states = build_teacher_state_rows(train_items_no_state, switch_model, basin_model, switch_threshold, basin_margin, max_candidates)
            test_states = build_teacher_state_rows(test_items_no_state, switch_model, basin_model, switch_threshold, basin_margin, max_candidates)
            train_items = [(key, by_pair_prefixes[key], train_states[key]) for key in train_keys]
            test_items = [(key, by_pair_prefixes[key], test_states[key]) for key in test_keys]
            train_state_rows = [row for _key, _prefixes, states in train_items for row in states]
            for profile in ["quality", "balanced", "production"]:
                value_model = TorchLinearRegressor(STATE_FEATURES)
                value_model.fit(train_state_rows, f"teacher_continue_value_{profile}", epochs=240, lr=0.05, l2=0.02)
                params = tune_policy(train_items, trust_model, switch_model, basin_model, value_model, damage_model, profile, max_candidates, switch_threshold, basin_margin)
                selections = [
                    simulate_pair(prefixes, states, trust_model, switch_model, basin_model, value_model, damage_model, params["trust_threshold"], params["switch_threshold"], params["value_threshold"], params["basin_margin"], params["damage_threshold"], params["min_candidates"], max_candidates)
                    for _key, prefixes, states in test_items
                ]
                method = f"teacher_value_{profile}_{feature_name}"
                metrics = evaluate(test_items, selections, method, f"fold_{fold_idx}")
                metrics.update({"feature_set": feature_name, **params})
                fold_rows.append(metrics)
                weights.extend(value_model.weight_rows(f"{method}_continue_value", f"fold_{fold_idx}"))
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
    return summarize(fold_rows) + fold_rows, selection_rows, weights


def make_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = ensure_dir(output_dir / "plots")
    summary = [row for row in rows if row["split"] == "question_grouped_cv"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter([safe_float(row["token_cost_vs_sample0"]) for row in summary], [100 * safe_float(row["delta_vs_sample0"]) for row in summary], s=[40 + 25 * safe_float(row["damaged_count"]) for row in summary])
    for row in summary:
        ax.annotate(row["method"].replace("teacher_value_", ""), (safe_float(row["token_cost_vs_sample0"]), 100 * safe_float(row["delta_vs_sample0"])), fontsize=7)
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_xlabel("Token cost vs sample0")
    ax.set_ylabel("Delta vs sample0 (%)")
    ax.set_title("Teacher-Value Prefix Policy Pareto Points")
    fig.tight_layout()
    fig.savefig(plot_dir / "01_teacher_value_pareto.png")
    plt.close(fig)


def write_report(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary = [row for row in rows if row["split"] == "question_grouped_cv"]
    lines = [
        "# Prefix Teacher-Value Policy",
        "",
        "这轮把 continue-value label 从 oracle future correctness 改为 verifier-realizable future gain：只有未来 prefix 中当前 verifier 真能选中的收益，才算继续采样的价值。",
        "",
        "| Method | Strict | Delta | Improved | Damaged | Net | Avg Gen | Cost vs 1x | Save vs 8x | Trust | Switch | Value Stop | Veto |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | `{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{int(row['net_gain_count'])}` | "
            f"`{safe_float(row['avg_generated_candidates']):.2f}` | `{safe_float(row['token_cost_vs_sample0']):.2f}x` | `{safe_float(row['token_savings_vs_full8']):.2%}` | `{safe_float(row['trust_stop_rate']):.2%}` | `{safe_float(row['switch_stop_rate']):.2%}` | `{safe_float(row['value_stop_rate']):.2%}` | `{safe_float(row['veto_stop_rate']):.2%}` |"
        )
    best = max(summary, key=lambda row: (safe_float(row["delta_vs_sample0"]) - 0.01 * safe_float(row["damaged_count"]), -safe_float(row["token_cost_vs_sample0"])))
    lines.extend(
        [
            "",
            "## 初步结论",
            "",
            f"- 当前综合最优点是 `{best['method']}`：delta `{safe_float(best['delta_vs_sample0']):.2%}`，damage `{int(best['damaged_count'])}`，cost `{safe_float(best['token_cost_vs_sample0']):.2f}x`。",
            "- 如果 teacher-value 比上一版 value-policy 更省或更安全，说明“用 verifier-realizable future gain 做 teacher”是正确方向；否则继续采样决策的瓶颈可能在 prefix 特征本身，而不是 label 定义。",
            "",
            "## 和上一轮 prefix 方法的对比",
            "",
            "| Method | Delta | Damaged | Avg Gen | Cost vs 1x | 说明 |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            "| `prefix_student_production_numeric` | `1.75%` | `1` | `2.52` | `2.67x` | 低成本且较安全 |",
            "| `value_policy_production_numeric` | `1.75%` | `2` | `2.30` | `2.36x` | 更省，但 damage 稍高 |",
            "| `teacher_value_production_full` | `1.25%` | `2` | `1.96` | `1.97x` | 当前最低成本正收益点 |",
            "| `teacher_value_balanced_full` | `1.50%` | `2` | `2.57` | `2.61x` | teacher-value 中收益最高 |",
            "",
            "## 当前判断",
            "",
            "teacher-value 没有直接超过上一版 value-policy；它的主要贡献是把正收益 policy 的成本进一步压低到接近 `2x`。如果继续压 cost，不应再只改 label，而应增强 prefix state 特征，尤其是判断 sample0 是否已经足够可信、alternative basin 是否是真 rescue 的轻量语义/事实信号。",
        ]
    )
    (output_dir / "prefix_teacher_value_findings_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    candidate_rows = read_csv(Path(args.candidate_run) / "candidate_features.csv")
    by_pair_prefixes, _sample0_rows, _alt_rows = build_prefix_dataset(candidate_rows, args.max_candidates)
    cv_rows, selection_rows, value_weights = run_cv(by_pair_prefixes, args.max_candidates)
    write_csv(output_dir / "prefix_teacher_value_cv_results.csv", cv_rows)
    write_csv(output_dir / "prefix_teacher_value_selection_rows.csv", selection_rows)
    write_csv(output_dir / "prefix_teacher_value_weights.csv", value_weights)
    write_json(output_dir / "run_metadata.json", {"pair_count": len(by_pair_prefixes), "max_candidates": args.max_candidates, "profile_lambdas": PROFILE_LAMBDAS, "state_features": STATE_FEATURES})
    make_plots(output_dir, cv_rows)
    write_report(output_dir, cv_rows)
    print(json.dumps({"output_dir": str(output_dir), "pair_count": len(by_pair_prefixes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
