#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import run_learned_basin_verifier as learned
from run_entropy_anatomy import ensure_dir, safe_float, write_csv, write_json


DEFAULT_CANDIDATE_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25"
DEFAULT_ENTROPY_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260429_063326_entropy_anatomy"
DEFAULT_ORIGINAL_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260429_065555_learned_basin_verifier"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"


BANNED_FEATURES = {
    "stable_hallucination_score",
    "delta_hallucination_score_vs_sample0",
    "basin_correct_rate",
    "correct_count",
    "wrong_count",
    "is_pure_correct",
    "is_pure_wrong",
    "is_mixed_basin",
    "is_rescue_basin",
    "is_damage_basin",
    "is_stable_correct_basin",
    "is_stable_hallucination_basin",
    "is_unstable_wrong_basin",
    "basin_regime",
    "sample0_strict_correct",
    "representative_strict_correct",
    "switch_gain_label",
}


NOLEAK_FEATURE_SETS = {
    "entropy_only_noleak": [
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
    "geometry_entropy_noleak": [
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
        "delta_entropy_vs_sample0",
        "delta_logprob_vs_sample0",
        "delta_weight_vs_sample0",
    ],
    "full_basin_noleak": [
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
        "low_entropy_basin_rank",
        "logprob_basin_rank",
        "weight_basin_rank",
        "stable_basin_rank",
        "delta_entropy_vs_sample0",
        "delta_logprob_vs_sample0",
        "delta_weight_vs_sample0",
        "delta_stable_score_vs_sample0",
        "sample0_basin_entropy_z",
        "sample0_basin_logprob_z",
        "sample0_basin_weight",
    ],
}


ORIGINAL_SUMMARY = {
    "basin_model_entropy_only": {"strict_correct_rate": 0.5025, "delta_vs_sample0": -0.0050, "improved_count": 0, "damaged_count": 2},
    "basin_model_geometry_entropy": {"strict_correct_rate": 0.5775, "delta_vs_sample0": 0.0700, "improved_count": 29, "damaged_count": 1},
    "basin_model_full_basin": {"strict_correct_rate": 0.5675, "delta_vs_sample0": 0.0600, "improved_count": 26, "damaged_count": 2},
    "switch_model_entropy_only": {"strict_correct_rate": 0.5000, "delta_vs_sample0": -0.0075, "improved_count": 1, "damaged_count": 4},
    "switch_model_geometry_entropy": {"strict_correct_rate": 0.5575, "delta_vs_sample0": 0.0500, "improved_count": 21, "damaged_count": 1},
    "switch_model_full_basin": {"strict_correct_rate": 0.5700, "delta_vs_sample0": 0.0625, "improved_count": 25, "damaged_count": 0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-8 learned basin verifier after removing label-derived leakage features.")
    parser.add_argument("--candidate-run", default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--entropy-run", default=DEFAULT_ENTROPY_RUN)
    parser.add_argument("--original-run", default=DEFAULT_ORIGINAL_RUN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="fixed8_noleak_basin_verifier")
    return parser.parse_args()


def audit_feature_sets() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_set, features in learned.FEATURE_SETS.items():
        for feature in features:
            rows.append(
                {
                    "source": "original",
                    "feature_set": feature_set,
                    "feature": feature,
                    "is_banned": float(feature in BANNED_FEATURES),
                    "reason": "label_or_regime_derived" if feature in BANNED_FEATURES else "",
                }
            )
    for feature_set, features in NOLEAK_FEATURE_SETS.items():
        for feature in features:
            rows.append(
                {
                    "source": "noleak",
                    "feature_set": feature_set,
                    "feature": feature,
                    "is_banned": float(feature in BANNED_FEATURES),
                    "reason": "ERROR_banned_feature_retained" if feature in BANNED_FEATURES else "",
                }
            )
    return rows


def compare_with_original(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparison: list[dict[str, Any]] = []
    method_map = {
        "basin_model_entropy_only_noleak": "basin_model_entropy_only",
        "basin_model_geometry_entropy_noleak": "basin_model_geometry_entropy",
        "basin_model_full_basin_noleak": "basin_model_full_basin",
        "switch_model_entropy_only_noleak": "switch_model_entropy_only",
        "switch_model_geometry_entropy_noleak": "switch_model_geometry_entropy",
        "switch_model_full_basin_noleak": "switch_model_full_basin",
    }
    for row in summary_rows:
        if row["split"] != "question_grouped_cv":
            continue
        method = str(row["method"])
        original_method = method_map.get(method)
        if not original_method:
            continue
        original = ORIGINAL_SUMMARY[original_method]
        comparison.append(
            {
                "method": method,
                "original_method": original_method,
                "noleak_strict": safe_float(row["strict_correct_rate"]),
                "original_strict": original["strict_correct_rate"],
                "strict_drop": safe_float(row["strict_correct_rate"]) - original["strict_correct_rate"],
                "noleak_delta": safe_float(row["delta_vs_sample0"]),
                "original_delta": original["delta_vs_sample0"],
                "delta_drop": safe_float(row["delta_vs_sample0"]) - original["delta_vs_sample0"],
                "noleak_improved": int(row["improved_count"]),
                "original_improved": original["improved_count"],
                "noleak_damaged": int(row["damaged_count"]),
                "original_damaged": original["damaged_count"],
            }
        )
    return comparison


def make_plots(output_dir: Path, comparison_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = ensure_dir(output_dir / "plots")
    labels = [row["method"].replace("_noleak", "").replace("_model_", "\n") for row in comparison_rows]
    original = [100 * safe_float(row["original_delta"]) for row in comparison_rows]
    noleak = [100 * safe_float(row["noleak_delta"]) for row in comparison_rows]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar([idx - 0.18 for idx in x], original, width=0.36, label="original")
    ax.bar([idx + 0.18 for idx in x], noleak, width=0.36, label="no-leak")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Delta vs sample0 (percentage points)")
    ax.set_title("Fixed-8 Learned Verifier: Original vs No-Leak")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "01_original_vs_noleak_delta.png")
    plt.close(fig)


def write_report(output_dir: Path, summary_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> None:
    original_banned = [row for row in audit_rows if row["source"] == "original" and safe_float(row["is_banned"]) > 0]
    lines = [
        "# Fixed-8 No-Leak Basin Verifier Audit",
        "",
        "## 1. 为什么做这个实验",
        "",
        "原始 fixed-8 learned verifier 中包含 `stable_hallucination_score` 和 `delta_hallucination_score_vs_sample0`。其中 `stable_hallucination_score` 在 `run_entropy_anatomy.py` 中显式使用了 `correct_rate == 0.0`，属于 gold-label-derived feature。这个实验移除这类字段，重新估计 fixed-8 上限。",
        "",
        "## 2. 原始特征泄漏审计",
        "",
        "| Feature Set | Banned Feature | Reason |",
        "| --- | --- | --- |",
    ]
    for row in original_banned:
        lines.append(f"| `{row['feature_set']}` | `{row['feature']}` | {row['reason']} |")
    lines.extend(
        [
            "",
            "## 3. No-Leak CV 结果",
            "",
            "| Method | Strict | Delta | Improved | Damaged | Net | Changed | Rescue Recall | Damage Avoid |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in [item for item in summary_rows if item["split"] == "question_grouped_cv"]:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{int(row['net_gain_count'])}` | "
            f"`{safe_float(row['answer_changed_rate']):.2%}` | `{safe_float(row['rescue_recall']):.2%}` | `{safe_float(row['damage_avoidance']):.2%}` |"
        )
    lines.extend(
        [
            "",
            "## 4. Original vs No-Leak 差距",
            "",
            "| Method | Original Delta | No-Leak Delta | Drop | Original Damage | No-Leak Damage |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['original_delta']):.2%}` | `{safe_float(row['noleak_delta']):.2%}` | "
            f"`{safe_float(row['delta_drop']):.2%}` | `{int(row['original_damaged'])}` | `{int(row['noleak_damaged'])}` |"
        )
    best = max([item for item in summary_rows if item["split"] == "question_grouped_cv"], key=lambda item: safe_float(item["delta_vs_sample0"]))
    lines.extend(
        [
            "",
            "## 5. 初步结论",
            "",
            f"- No-leak 后最强方法是 `{best['method']}`：delta `{safe_float(best['delta_vs_sample0']):.2%}`，damage `{int(best['damaged_count'])}`。",
            "- 如果 no-leak 后仍有显著正收益，说明 answer-basin structure 本身确实可用；如果大幅下降，则原始 fixed-8 上限应改写为 oracle/diagnostic upper bound。",
            "- 论文中的 Pareto 图应区分 deployable no-leak frontier 与包含 label-derived features 的 diagnostic upper bound。",
        ]
    )
    (output_dir / "fixed8_noleak_findings_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    audit_rows = audit_feature_sets()
    retained_banned = [row for row in audit_rows if row["source"] == "noleak" and safe_float(row["is_banned"]) > 0]
    if retained_banned:
        raise RuntimeError(f"No-leak feature set still contains banned features: {retained_banned}")
    rows, by_pair = learned.build_tables(Path(args.candidate_run), Path(args.entropy_run))
    original_feature_sets = learned.FEATURE_SETS
    learned.FEATURE_SETS = NOLEAK_FEATURE_SETS
    try:
        cv_rows, selection_rows, weights = learned.run_cv(rows, by_pair)
    finally:
        learned.FEATURE_SETS = original_feature_sets
    summary_rows = [row for row in cv_rows if row["split"] == "question_grouped_cv"]
    all_cv_rows = cv_rows
    comparison_rows = compare_with_original(summary_rows)
    write_csv(output_dir / "feature_leakage_audit.csv", audit_rows)
    write_csv(output_dir / "fixed8_noleak_cv_results.csv", all_cv_rows)
    write_csv(output_dir / "fixed8_noleak_selection_rows.csv", selection_rows)
    write_csv(output_dir / "fixed8_noleak_weights.csv", weights)
    write_csv(output_dir / "fixed8_original_vs_noleak.csv", comparison_rows)
    make_plots(output_dir, comparison_rows)
    write_report(output_dir, summary_rows, comparison_rows, audit_rows)
    write_json(
        output_dir / "run_metadata.json",
        {
            "candidate_run": args.candidate_run,
            "entropy_run": args.entropy_run,
            "original_run": args.original_run,
            "banned_features": sorted(BANNED_FEATURES),
            "noleak_feature_sets": NOLEAK_FEATURE_SETS,
            "pair_count": len(by_pair),
            "row_count": len(rows),
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "pairs": len(by_pair), "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
