#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import math
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path("/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260430_021911_entropy_waveform_analysis_v2")

PALETTE = {
    "blue": "#069DFF",
    "gray": "#808080",
    "green": "#A4E048",
    "black": "#010101",
    "light_red": "#FFA0A0",
}

KEY_FEATURES = [
    ("wf_entropy_mean_mean", "Mean entropy"),
    ("wf_entropy_std_mean", "Entropy volatility"),
    ("wf_entropy_max_mean", "Max entropy spike"),
    ("wf_entropy_late_mean_mean", "Late entropy"),
    ("wf_entropy_late_minus_early_mean", "Late - early entropy"),
    ("wf_prob_min_mean", "Minimum chosen-token prob."),
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def safe_float(value: Any) -> float:
    try:
        if value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def cohen_d(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.0
    pooled = math.sqrt(((len(pos) - 1) * stdev(pos) ** 2 + (len(neg) - 1) * stdev(neg) ** 2) / max(1, len(pos) + len(neg) - 2))
    return (mean(pos) - mean(neg)) / pooled if pooled else 0.0


def parse_curve(value: str) -> list[float]:
    try:
        parsed = ast.literal_eval(value)
        return [float(item) for item in parsed]
    except Exception:
        return []


def setup_plotting() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )
    return plt


def groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "pure_correct": [row for row in rows if safe_float(row.get("is_pure_correct")) > 0],
        "pure_wrong": [row for row in rows if safe_float(row.get("is_pure_wrong")) > 0],
        "stable_correct": [row for row in rows if row.get("basin_regime") == "stable_correct"],
        "stable_hallucination": [row for row in rows if row.get("basin_regime") == "stable_hallucination"],
        "rescue": [row for row in rows if safe_float(row.get("is_rescue_basin")) > 0],
        "damage": [row for row in rows if safe_float(row.get("is_damage_basin")) > 0],
    }


def plot_waveform_bands(plt: Any, plot_dir: Path, group_rows: dict[str, list[dict[str, Any]]]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    specs = [
        ("pure_correct", PALETTE["green"], "Pure correct"),
        ("pure_wrong", PALETTE["gray"], "Pure wrong"),
        ("stable_correct", PALETTE["blue"], "Stable correct"),
        ("stable_hallucination", PALETTE["light_red"], "Stable hallucination"),
    ]
    xs = [idx / 19 for idx in range(20)]
    for key, color, label in specs:
        curves = [parse_curve(row.get("wf_basin_entropy_curve", "")) for row in group_rows[key]]
        curves = [curve for curve in curves if len(curve) == 20]
        if not curves:
            continue
        means = [mean([curve[idx] for curve in curves]) for idx in range(20)]
        sds = [stdev([curve[idx] for curve in curves]) for idx in range(20)]
        lower = [m - 0.25 * s for m, s in zip(means, sds)]
        upper = [m + 0.25 * s for m, s in zip(means, sds)]
        ax.plot(xs, means, color=color, linewidth=1.6, label=f"{label} (n={len(curves)})")
        ax.fill_between(xs, lower, upper, color=color, alpha=0.18, linewidth=0)
    ax.set_title("Correct vs Wrong Basin Entropy Waveforms")
    ax.set_xlabel("Normalized answer position")
    ax.set_ylabel("Shannon entropy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "06_correct_vs_wrong_waveform_bands.png")
    plt.close(fig)


def plot_feature_distributions(plt: Any, plot_dir: Path, group_rows: dict[str, list[dict[str, Any]]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    labels = ["stable_correct", "stable_hallucination", "pure_wrong"]
    colors = [PALETTE["green"], PALETTE["light_red"], PALETTE["gray"]]
    for ax, (feature, title) in zip(axes.ravel(), KEY_FEATURES):
        data = [[safe_float(row.get(feature)) for row in group_rows[label]] for label in labels]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.35)
        parts["cmeans"].set_color(PALETTE["black"])
        ax.set_xticks([1, 2, 3], ["stable\ncorrect", "stable\nhalluc.", "pure\nwrong"])
        ax.set_title(title)
    fig.suptitle("Entropy Feature Distributions: Correct vs Hallucination/Wrong Basins", y=0.99)
    fig.tight_layout()
    fig.savefig(plot_dir / "07_entropy_feature_distributions.png")
    plt.close(fig)


def plot_effect_bars(plt: Any, plot_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_rows = groups(rows)
    comparisons = [
        ("Stable correct - stable hallucination", group_rows["stable_correct"], group_rows["stable_hallucination"]),
        ("Pure correct - pure wrong", group_rows["pure_correct"], group_rows["pure_wrong"]),
        ("Rescue - damage", group_rows["rescue"], group_rows["damage"]),
    ]
    effect_rows: list[dict[str, Any]] = []
    for comparison, pos_rows, neg_rows in comparisons:
        for feature, title in KEY_FEATURES:
            pos = [safe_float(row.get(feature)) for row in pos_rows]
            neg = [safe_float(row.get(feature)) for row in neg_rows]
            effect_rows.append(
                {
                    "comparison": comparison,
                    "feature": feature,
                    "label": title,
                    "positive_mean": mean(pos),
                    "negative_mean": mean(neg),
                    "cohen_d": cohen_d(pos, neg),
                    "positive_count": len(pos),
                    "negative_count": len(neg),
                }
            )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True)
    for ax, (comparison, _pos, _neg) in zip(axes, comparisons):
        subset = [row for row in effect_rows if row["comparison"] == comparison]
        subset.sort(key=lambda row: abs(float(row["cohen_d"])))
        ax.barh([row["label"] for row in subset], [float(row["cohen_d"]) for row in subset], color="#4c78a8")
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_title(comparison)
        ax.set_xlabel("Cohen d")
    fig.suptitle("Entropy Feature Effect Sizes", y=0.99)
    fig.tight_layout()
    fig.savefig(plot_dir / "08_entropy_feature_effect_sizes.png")
    plt.close(fig)
    return effect_rows


def plot_scatter(plt: Any, plot_dir: Path, group_rows: dict[str, list[dict[str, Any]]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    specs = [
        ("stable_correct", PALETTE["green"], "Stable correct"),
        ("stable_hallucination", PALETTE["gray"], "Stable hallucination"),
        ("rescue", PALETTE["blue"], "Rescue"),
        ("damage", PALETTE["light_red"], "Damage"),
    ]
    for key, color, label in specs:
        rows = group_rows[key]
        ax.scatter(
            [safe_float(row.get("wf_entropy_max_mean")) for row in rows],
            [safe_float(row.get("wf_prob_min_mean")) for row in rows],
            s=[22 + 45 * safe_float(row.get("cluster_weight_mass")) for row in rows],
            alpha=0.58,
            color=color,
            label=f"{label} (n={len(rows)})",
            edgecolors="none",
        )
    ax.set_title("Max Entropy Spike vs Minimum Chosen-Token Probability")
    ax.set_xlabel("Max entropy spike")
    ax.set_ylabel("Minimum chosen-token probability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "09_entropy_spike_vs_minprob_scatter.png")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(run_dir: Path, effect_rows: list[dict[str, Any]], group_rows: dict[str, list[dict[str, Any]]]) -> None:
    def line_for(comparison: str) -> list[dict[str, Any]]:
        return sorted([row for row in effect_rows if row["comparison"] == comparison], key=lambda row: abs(float(row["cohen_d"])), reverse=True)

    stable = line_for("Stable correct - stable hallucination")
    pure = line_for("Pure correct - pure wrong")
    rescue = line_for("Rescue - damage")
    lines = [
        "# Entropy Waveform Visual Comparison",
        "",
        "## 图像输出",
        "",
        "- `plots/06_correct_vs_wrong_waveform_bands.png`: 正确/错误 basin 的平均 entropy waveform band。",
        "- `plots/07_entropy_feature_distributions.png`: 关键 entropy 特征在 stable correct、stable hallucination、pure wrong 间的分布。",
        "- `plots/08_entropy_feature_effect_sizes.png`: 三组对比的 Cohen's d 效应量。",
        "- `plots/09_entropy_spike_vs_minprob_scatter.png`: 最大 entropy spike 与最小 chosen-token probability 的二维散点图。",
        "",
        "## 样本规模",
        "",
        f"- Pure correct basin: `{len(group_rows['pure_correct'])}`",
        f"- Pure wrong basin: `{len(group_rows['pure_wrong'])}`",
        f"- Stable correct basin: `{len(group_rows['stable_correct'])}`",
        f"- Stable hallucination basin: `{len(group_rows['stable_hallucination'])}`",
        f"- Rescue basin: `{len(group_rows['rescue'])}`",
        f"- Damage basin: `{len(group_rows['damage'])}`",
        "",
        "## Stable Correct vs Stable Hallucination",
        "",
        "| Feature | Correct Mean | Hallucination Mean | Cohen d | Interpretation |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in stable:
        direction = "correct 更高" if float(row["cohen_d"]) > 0 else "hallucination 更高"
        lines.append(f"| `{row['feature']}` | `{row['positive_mean']:.4f}` | `{row['negative_mean']:.4f}` | `{row['cohen_d']:.3f}` | {direction} |")
    lines.extend(["", "## Pure Correct vs Pure Wrong", "", "| Feature | Correct Mean | Wrong Mean | Cohen d | Interpretation |", "| --- | ---: | ---: | ---: | --- |"])
    for row in pure:
        direction = "correct 更高" if float(row["cohen_d"]) > 0 else "wrong 更高"
        lines.append(f"| `{row['feature']}` | `{row['positive_mean']:.4f}` | `{row['negative_mean']:.4f}` | `{row['cohen_d']:.3f}` | {direction} |")
    lines.extend(["", "## Rescue vs Damage", "", "| Feature | Rescue Mean | Damage Mean | Cohen d | Interpretation |", "| --- | ---: | ---: | ---: | --- |"])
    for row in rescue:
        direction = "rescue 更高" if float(row["cohen_d"]) > 0 else "damage 更高"
        lines.append(f"| `{row['feature']}` | `{row['positive_mean']:.4f}` | `{row['negative_mean']:.4f}` | `{row['cohen_d']:.3f}` | {direction} |")
    lines.extend(
        [
            "",
            "## 直观解释",
            "",
            "正确 basin 的 token 轨迹通常更低、更平，尤其最大 entropy spike 和 entropy volatility 更小；stable hallucination 虽然在 basin 层面很稳定，但内部 token 轨迹并不完全平滑，常出现更大的局部犹豫峰。",
            "",
            "这说明 entropy waveform 最适合回答“这个 basin 内部有没有局部不稳的生成痕迹”。它不是单独的答案选择器，但可以成为 damage-risk veto 或 prefix early-warning 的输入。",
        ]
    )
    (run_dir / "entropy_waveform_visual_comparison_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    run_dir = DEFAULT_RUN_DIR
    rows = read_csv(run_dir / "basin_waveform_table.csv")
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_plotting()
    group_rows = groups(rows)
    plot_waveform_bands(plt, plot_dir, group_rows)
    plot_feature_distributions(plt, plot_dir, group_rows)
    effect_rows = plot_effect_bars(plt, plot_dir, rows)
    plot_scatter(plt, plot_dir, group_rows)
    write_csv(run_dir / "entropy_waveform_visual_effect_sizes.csv", effect_rows)
    write_report(run_dir, effect_rows, group_rows)
    print(f"Wrote visual comparison outputs to {run_dir}")


if __name__ == "__main__":
    main()
