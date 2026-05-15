#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from run_adaptive_semantic_basin_verifier import (
    FEATURE_SETS,
    TorchLogisticModel,
    build_prefix_dataset,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-k prefix geometry verifier cost curve.")
    parser.add_argument("--candidate-run", default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="fixedk_geometry_verifier")
    parser.add_argument("--max-candidates", type=int, default=8)
    return parser.parse_args()


def generated_token_cost(prefix: dict[str, Any]) -> float:
    return sum(safe_float(sample["token_count"]) for sample in prefix["samples"])


def choose_best(prefix: dict[str, Any], model: TorchLogisticModel, threshold: float, margin: float) -> dict[str, Any]:
    sample0 = sample0_at(prefix)
    best = max(prefix["basins"], key=model.predict)
    if int(best["cluster_id"]) != int(sample0["cluster_id"]) and model.predict(best) - model.predict(sample0) >= margin and model.predict(best) >= threshold:
        return best
    return sample0


def evaluate(items: list[tuple[tuple[int, str], list[dict[str, Any]]]], selected: list[dict[str, Any]], k: int, method: str, split: str) -> dict[str, Any]:
    sample0s = [sample0_at(prefixes[0]) for _key, prefixes in items]
    deltas = [safe_float(row["representative_strict_correct"]) - safe_float(s0["representative_strict_correct"]) for row, s0 in zip(selected, sample0s)]
    prefix_tokens = [generated_token_cost(prefixes[k - 1]) for _key, prefixes in items]
    sample0_tokens = [generated_token_cost(prefixes[0]) for _key, prefixes in items]
    full8_tokens = [generated_token_cost(prefixes[-1]) for _key, prefixes in items]
    return {
        "split": split,
        "method": method,
        "k": k,
        "pairs": len(items),
        "strict_correct_rate": mean([safe_float(row["representative_strict_correct"]) for row in selected]),
        "sample0_strict_correct_rate": mean([safe_float(row["representative_strict_correct"]) for row in sample0s]),
        "delta_vs_sample0": mean(deltas),
        "improved_count": sum(1 for value in deltas if value > 0),
        "damaged_count": sum(1 for value in deltas if value < 0),
        "net_gain_count": sum(1 for value in deltas if value > 0) - sum(1 for value in deltas if value < 0),
        "answer_changed_rate": mean([float(int(row["cluster_id"]) != int(s0["cluster_id"])) for row, s0 in zip(selected, sample0s)]),
        "avg_generated_candidates": float(k),
        "token_cost_vs_sample0": mean([cost / max(1.0, base) for cost, base in zip(prefix_tokens, sample0_tokens)]),
        "token_savings_vs_full8": 1.0 - mean([cost / max(1.0, full) for cost, full in zip(prefix_tokens, full8_tokens)]),
    }


def tune(train_items: list[tuple[tuple[int, str], list[dict[str, Any]]]], model: TorchLogisticModel, k: int) -> dict[str, float]:
    best: tuple[float, dict[str, float]] = (-1e9, {})
    prefix_items = [(key, prefixes) for key, prefixes in train_items if len(prefixes) >= k]
    for threshold in [0.25, 0.4, 0.55, 0.7, 0.85]:
        for margin in [-0.3, -0.1, 0.0, 0.1, 0.25, 0.4]:
            selected = [choose_best(prefixes[k - 1], model, threshold, margin) for _key, prefixes in prefix_items]
            metrics = evaluate(prefix_items, selected, k, "train", "train")
            score = 100 * safe_float(metrics["delta_vs_sample0"]) + int(metrics["net_gain_count"]) - 3.0 * int(metrics["damaged_count"])
            if score > best[0]:
                best = (score, {"threshold": threshold, "margin": margin})
    return best[1]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["k"]))].append(row)
    summary = []
    for (method, k), items in sorted(grouped.items()):
        summary.append(
            {
                "split": "question_grouped_cv",
                "method": method,
                "k": k,
                "pairs": sum(int(item["pairs"]) for item in items),
                "strict_correct_rate": mean([safe_float(item["strict_correct_rate"]) for item in items]),
                "sample0_strict_correct_rate": mean([safe_float(item["sample0_strict_correct_rate"]) for item in items]),
                "delta_vs_sample0": mean([safe_float(item["delta_vs_sample0"]) for item in items]),
                "improved_count": sum(int(item["improved_count"]) for item in items),
                "damaged_count": sum(int(item["damaged_count"]) for item in items),
                "net_gain_count": sum(int(item["net_gain_count"]) for item in items),
                "answer_changed_rate": mean([safe_float(item["answer_changed_rate"]) for item in items]),
                "avg_generated_candidates": float(k),
                "token_cost_vs_sample0": mean([safe_float(item["token_cost_vs_sample0"]) for item in items]),
                "token_savings_vs_full8": mean([safe_float(item["token_savings_vs_full8"]) for item in items]),
            }
        )
    return summary


def run_cv(by_pair_prefixes: dict[tuple[int, str], list[dict[str, Any]]], max_candidates: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keys = sorted(by_pair_prefixes)
    fold_rows = []
    selection_rows = []
    features = FEATURE_SETS["numeric_adaptive"]
    for fold_idx, (train_qids, test_qids) in enumerate(question_folds(keys)):
        train_items = [(key, by_pair_prefixes[key]) for key in keys if key[1] in train_qids]
        test_items = [(key, by_pair_prefixes[key]) for key in keys if key[1] in test_qids]
        for k in range(1, max_candidates + 1):
            train_rows = [row for _key, prefixes in train_items for row in prefixes[k - 1]["basins"]]
            model = TorchLogisticModel(features)
            model.fit(train_rows, "representative_strict_correct", epochs=220, lr=0.08, l2=0.02)
            params = {"threshold": 0.0, "margin": 0.0} if k == 1 else tune(train_items, model, k)
            selected = [sample0_at(prefixes[0]) if k == 1 else choose_best(prefixes[k - 1], model, params["threshold"], params["margin"]) for _key, prefixes in test_items]
            method = "fixedk_geometry"
            metrics = evaluate(test_items, selected, k, method, f"fold_{fold_idx}")
            metrics.update(params)
            fold_rows.append(metrics)
            for (key, prefixes), row in zip(test_items, selected):
                s0 = sample0_at(prefixes[0])
                selection_rows.append(
                    {
                        "split": f"fold_{fold_idx}",
                        "method": method,
                        "k": k,
                        "seed": key[0],
                        "question_id": key[1],
                        "sample0_correct": s0["representative_strict_correct"],
                        "selected_correct": row["representative_strict_correct"],
                        "delta_correct": safe_float(row["representative_strict_correct"]) - safe_float(s0["representative_strict_correct"]),
                        "sample0_cluster_id": s0["cluster_id"],
                        "selected_cluster_id": row["cluster_id"],
                        "selected_preview": row["representative_preview"],
                    }
                )
    return summarize(fold_rows) + fold_rows, selection_rows


def write_report(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary = [row for row in rows if row["split"] == "question_grouped_cv"]
    lines = [
        "# Fixed-k Geometry Verifier Cost Curve",
        "",
        "| k | Strict | Delta | Improved | Damaged | Net | Cost vs 1x | Save vs 8x | Changed |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| `{int(row['k'])}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{int(row['net_gain_count'])}` | "
            f"`{safe_float(row['token_cost_vs_sample0']):.2f}x` | `{safe_float(row['token_savings_vs_full8']):.2%}` | `{safe_float(row['answer_changed_rate']):.2%}` |"
        )
    (output_dir / "fixedk_geometry_findings_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    candidate_rows = read_csv(Path(args.candidate_run) / "candidate_features.csv")
    by_pair_prefixes, _sample0_rows, _alt_rows = build_prefix_dataset(candidate_rows, args.max_candidates)
    cv_rows, selection_rows = run_cv(by_pair_prefixes, args.max_candidates)
    write_csv(output_dir / "fixedk_geometry_cv_results.csv", cv_rows)
    write_csv(output_dir / "fixedk_geometry_selection_rows.csv", selection_rows)
    write_json(output_dir / "run_metadata.json", {"pair_count": len(by_pair_prefixes), "max_candidates": args.max_candidates, "feature_set": "numeric_adaptive"})
    write_report(output_dir, cv_rows)
    print(json.dumps({"output_dir": str(output_dir), "pair_count": len(by_pair_prefixes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
