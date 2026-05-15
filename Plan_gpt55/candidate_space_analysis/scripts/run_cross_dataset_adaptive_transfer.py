#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_adaptive_semantic_basin_verifier import (  # noqa: E402
    FEATURE_SETS,
    TorchLogisticModel,
    build_prefix_dataset,
    ensure_dir,
    evaluate,
    read_csv,
    safe_float,
    sample0_at,
    simulate_pair,
    tune_policy,
    write_csv,
    write_json,
)


DEFAULT_SOURCE_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260511_100502_candidate_space_triviaqa_scale500_qwen25"
DEFAULT_TARGET_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_hotpotqa_candidate_placeholder"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adaptive controller on source candidates and evaluate transfer on target candidates.")
    parser.add_argument("--source-candidate-run", default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--target-candidate-run", default=DEFAULT_TARGET_RUN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="cross_dataset_adaptive_transfer")
    parser.add_argument("--max-candidates", type=int, default=8)
    return parser.parse_args()


def source_label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "rows": len(rows),
        "correct_rows": sum(1 for row in rows if safe_float(row.get("strict_correct")) > 0),
        "rescue_rows": sum(1 for row in rows if safe_float(row.get("rescue_candidate")) > 0),
        "damage_rows": sum(1 for row in rows if safe_float(row.get("damage_candidate")) > 0),
    }


def weight_rows(model: TorchLogisticModel, model_name: str, feature_set: str, top_k: int = 12) -> list[dict[str, Any]]:
    rows = [{"model": model_name, "feature_set": feature_set, "feature": "bias", "weight": model.bias}]
    for feature in sorted(model.features, key=lambda item: abs(model.weights[item]), reverse=True)[:top_k]:
        rows.append({"model": model_name, "feature_set": feature_set, "feature": feature, "weight": model.weights[feature]})
    return rows


def run_transfer(
    source_prefixes: dict[tuple[int, str], list[dict[str, Any]]],
    target_prefixes: dict[tuple[int, str], list[dict[str, Any]]],
    max_candidates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_groups = list(source_prefixes.values())
    target_items = sorted(target_prefixes.items())
    target_groups = [prefixes for _key, prefixes in target_items]
    source_sample0_rows = [sample0_at(prefixes[0]) for prefixes in source_groups]
    source_alt_rows = [row for prefixes in source_groups for prefix in prefixes for row in prefix["basins"] if safe_float(row["contains_sample0"]) <= 0]
    source_all_rows = [row for prefixes in source_groups for prefix in prefixes for row in prefix["basins"]]

    summary_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []

    for feature_name, features in FEATURE_SETS.items():
        trust_model = TorchLogisticModel(features)
        trust_model.fit(source_sample0_rows, "representative_strict_correct", epochs=180, lr=0.1, l2=0.02)
        switch_model = TorchLogisticModel(features)
        switch_model.fit(source_alt_rows, "switch_gain_label", epochs=180, lr=0.1, l2=0.02)
        basin_model = TorchLogisticModel(features)
        basin_model.fit(source_all_rows, "representative_strict_correct", epochs=180, lr=0.1, l2=0.02)

        model_rows.extend(weight_rows(trust_model, f"transfer_{feature_name}_trust", feature_name))
        model_rows.extend(weight_rows(switch_model, f"transfer_{feature_name}_switch", feature_name))
        model_rows.extend(weight_rows(basin_model, f"transfer_{feature_name}_basin", feature_name))

        for profile, efficiency_penalty in [("quality", 0.0), ("balanced", 0.6), ("production", 1.8)]:
            params = tune_policy(source_groups, trust_model, switch_model, basin_model, max_candidates, efficiency_penalty)
            selections = [
                simulate_pair(
                    prefixes,
                    trust_model,
                    switch_model,
                    params["trust_threshold"],
                    params["switch_threshold"],
                    params["min_candidates"],
                    max_candidates,
                    basin_model,
                    params["basin_margin"],
                    params["budget_best"],
                )
                for prefixes in target_groups
            ]
            method = f"transfer_adaptive_{profile}_{feature_name}"
            metrics = evaluate(target_groups, selections, method, "hotpotqa_target")
            metrics.update({"source": "triviaqa_scale500", "target": "hotpotqa", "feature_set": feature_name, **params})
            summary_rows.append(metrics)

            for (seed, qid), (row, generated, reason) in zip([key for key, _prefixes in target_items], selections):
                s0 = sample0_at(target_prefixes[(seed, qid)][0])
                selection_rows.append(
                    {
                        "split": "hotpotqa_target",
                        "method": method,
                        "seed": seed,
                        "question_id": qid,
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
    return summary_rows, selection_rows, model_rows


def write_report(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Cross-Dataset Adaptive Transfer",
        "",
        "Source: TriviaQA scale500. Target: HotpotQA. Target labels are used only for final evaluation, not tuning.",
        "",
        "| Method | Strict | Delta vs sample0 | Improved | Damaged | Avg Gen | Token Cost vs sample0 | Token Cost vs full8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (-safe_float(item["delta_vs_sample0"]), safe_float(item["token_cost_vs_sample0"]))):
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{safe_float(row['avg_generated_candidates']):.2f}` | "
            f"`{safe_float(row['token_cost_vs_sample0']):.2f}x` | `{safe_float(row['token_cost_vs_full8']):.2f}x` |"
        )
    best = max(rows, key=lambda item: safe_float(item["delta_vs_sample0"]), default=None)
    if best:
        lines.extend(
            [
                "",
                "## Initial Read",
                "",
                f"Best transfer method: `{best['method']}`, delta `{safe_float(best['delta_vs_sample0']):.2%}`, damage `{int(best['damaged_count'])}`, average generated candidates `{safe_float(best['avg_generated_candidates']):.2f}`.",
            ]
        )
    (output_dir / "cross_dataset_transfer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    source_rows = read_csv(Path(args.source_candidate_run) / "candidate_features.csv")
    target_rows = read_csv(Path(args.target_candidate_run) / "candidate_features.csv")
    source_prefixes, _source_sample0, _source_alt = build_prefix_dataset(source_rows, args.max_candidates)
    target_prefixes, _target_sample0, _target_alt = build_prefix_dataset(target_rows, args.max_candidates)
    summary_rows, selection_rows, model_rows = run_transfer(source_prefixes, target_prefixes, args.max_candidates)

    write_csv(output_dir / "cross_dataset_transfer_summary.csv", summary_rows)
    write_csv(output_dir / "cross_dataset_transfer_selection_rows.csv", selection_rows)
    write_csv(output_dir / "cross_dataset_transfer_model_weights.csv", model_rows)
    write_json(
        output_dir / "run_metadata.json",
        {
            "source_candidate_run": str(Path(args.source_candidate_run)),
            "target_candidate_run": str(Path(args.target_candidate_run)),
            "source_pair_count": len(source_prefixes),
            "target_pair_count": len(target_prefixes),
            "max_candidates": args.max_candidates,
            "source_label_counts": source_label_counts(source_rows),
            "target_label_counts": source_label_counts(target_rows),
            "feature_sets": FEATURE_SETS,
        },
    )
    write_report(output_dir, summary_rows)
    print(json.dumps({"output_dir": str(output_dir), "source_pairs": len(source_prefixes), "target_pairs": len(target_prefixes)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
