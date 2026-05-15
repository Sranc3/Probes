#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


OUTPUT_DIR = Path("/zhutingqi/song/memo_for_paper/figures")
CSV_PATH = OUTPUT_DIR / "cost_accuracy_pareto_points.csv"
MD_PATH = OUTPUT_DIR / "cost_accuracy_pareto_explanation_zh.md"

PALETTE = {
    "blue": "#069DFF",
    "gray": "#808080",
    "green": "#A4E048",
    "black": "#010101",
}


POINTS: list[dict[str, Any]] = [
    {
        "regime": "baseline",
        "method": "normal_sample0",
        "label": "sample0",
        "strict": 0.5000,
        "delta": 0.0000,
        "damage": 0,
        "cost": 1.00,
        "frontier": 1,
    },
    {
        "regime": "fixed_k_baseline",
        "method": "fixed2_geometry",
        "label": "fixed-2",
        "strict": 0.5075,
        "delta": 0.0075,
        "damage": 1,
        "cost": 2.06,
        "frontier": 0,
    },
    {
        "regime": "fixed_k_baseline",
        "method": "fixed4_geometry",
        "label": "fixed-4",
        "strict": 0.5100,
        "delta": 0.0100,
        "damage": 1,
        "cost": 4.09,
        "frontier": 0,
    },
    {
        "regime": "extreme_low_cost",
        "method": "teacher_value_production_full",
        "label": "teacher-value",
        "strict": 0.5125,
        "delta": 0.0125,
        "damage": 2,
        "cost": 1.97,
        "frontier": 1,
    },
    {
        "regime": "low_cost",
        "method": "value_policy_production_numeric",
        "label": "value-policy",
        "strict": 0.5175,
        "delta": 0.0175,
        "damage": 2,
        "cost": 2.36,
        "frontier": 1,
    },
    {
        "regime": "low_cost_safe",
        "method": "prefix_student_production_numeric",
        "label": "prefix-student",
        "strict": 0.5175,
        "delta": 0.0175,
        "damage": 1,
        "cost": 2.67,
        "frontier": 1,
    },
    {
        "regime": "production",
        "method": "adaptive_v1_production_numeric",
        "label": "adaptive-v1",
        "strict": 0.5225,
        "delta": 0.0225,
        "damage": 0,
        "cost": 4.48,
        "frontier": 1,
    },
    {
        "regime": "high_compute_noleak",
        "method": "fixed8_noleak_basin_geometry_entropy",
        "label": "fixed-8 no-leak",
        "strict": 0.5150,
        "delta": 0.0075,
        "damage": 0,
        "cost": 7.78,
        "frontier": 0,
    },
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 10,
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
        }
    )
    fig, ax = plt.subplots(figsize=(9, 5.6))
    colors = {
        "baseline": PALETTE["gray"],
        "fixed_k_baseline": "#b6b6b6",
        "extreme_low_cost": "#7ecbff",
        "low_cost": PALETTE["blue"],
        "low_cost_safe": "#6fc4ff",
        "production": PALETTE["green"],
        "high_compute_noleak": "#9a9a9a",
    }
    for row in rows:
        size = 75 + 38 * int(row["damage"])
        marker = "o" if int(row["frontier"]) else "x"
        scatter_kwargs = {
            "s": size,
            "color": colors[str(row["regime"])],
            "marker": marker,
            "alpha": 0.88,
        }
        if marker == "o":
            scatter_kwargs.update({"edgecolors": PALETTE["black"], "linewidths": 0.6})
        ax.scatter(float(row["cost"]), 100 * float(row["delta"]), **scatter_kwargs)
        ax.annotate(str(row["label"]), (float(row["cost"]), 100 * float(row["delta"])), xytext=(4, 4), textcoords="offset points", fontsize=8)
    frontier = sorted([row for row in rows if int(row["frontier"])], key=lambda item: (float(item["cost"]), float(item["delta"])))
    ax.plot([float(row["cost"]) for row in frontier], [100 * float(row["delta"]) for row in frontier], color=PALETTE["black"], linewidth=1.4, alpha=0.72, label="Pareto path")
    ax.axhline(0, color=PALETTE["gray"], linewidth=0.8)
    ax.set_xlabel("Relative token generation cost vs sample0")
    ax.set_ylabel("Strict correctness gain over sample0 (percentage points)")
    ax.set_title("Cost-Accuracy Pareto Frontier of Answer-Basin Control")
    ax.text(1.05, 2.65, "Only no-leak / deployable-compliant methods are shown", fontsize=8, color=PALETTE["gray"])
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cost_accuracy_pareto_frontier.png")
    fig.savefig(OUTPUT_DIR / "cost_accuracy_pareto_frontier.pdf")
    plt.close(fig)


def write_explanation(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Cost-Accuracy Pareto Frontier 图解",
        "",
        "这张图把目前主要 inference regimes 放在同一坐标系中：横轴是相对正常 `sample0` 的 token generation cost，纵轴是 strict correctness 相对 `sample0` 的绝对提升。",
        "",
        "## 图中点的含义",
        "",
        "| Regime | Method | Delta | Damage | Cost | Role |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    role = {
        "normal_sample0": "最低成本 baseline",
        "fixed2_geometry": "简单 fixed-k baseline",
        "fixed4_geometry": "中成本 fixed-k baseline",
        "teacher_value_production_full": "约 2x 成本的新正收益点",
        "value_policy_production_numeric": "低成本 adaptive 点",
        "prefix_student_production_numeric": "较安全的低成本 adaptive 点",
        "adaptive_v1_production_numeric": "当前 production-style 主结果",
        "fixed8_noleak_basin_geometry_entropy": "no-leak fixed-8 复验点，优势大幅收缩",
    }
    for row in rows:
        lines.append(f"| `{row['regime']}` | `{row['method']}` | `{100 * float(row['delta']):.2f}` | `{row['damage']}` | `{float(row['cost']):.2f}x` | {role[row['method']]} |")
    lines.extend(
        [
            "",
            "## 论文中可用的主张",
            "",
            "> Answer-basin control exposes a cost-accuracy Pareto frontier: deployable no-leak controllers currently achieve modest but real gains, with adaptive policies outperforming fixed high-compute numeric basin verification under compliant features.",
            "",
            "## 重要边界",
            "",
            "- 本图只展示 no-leak / 合规方法；已排除包含 label-derived feature 的原始 fixed-8 结果。",
            "- no-leak fixed-8 最强点约为 `+0.75%`、damage `0`，明显低于原始 `+6%~+7%`。",
            "- `entropy waveform` 的作用目前应定位为机制分析，而不是 Pareto frontier 上的主 controller 点。",
            "- 当前 production-style 首选仍是 `adaptive_v1_production_numeric`：`+2.25%`、damage `0`、约 `4.48x` cost。",
        ]
    )
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(CSV_PATH, POINTS)
    make_plot(POINTS)
    write_explanation(POINTS)
    print(f"Wrote Pareto outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
