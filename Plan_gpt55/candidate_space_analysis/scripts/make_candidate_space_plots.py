#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_091924_candidate_space_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create plots for candidate-space analysis.")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def f(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def ensure_plot_dir(run_dir: Path) -> Path:
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def setup_matplotlib():
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


def plot_pipeline(plot_dir: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.axis("off")
    boxes = [
        ("Question", "same prompt"),
        ("8 Candidates", "sampled answers"),
        ("Candidate Space", "each answer is a point"),
        ("Feature Signals", "logprob / entropy / cluster / length"),
        ("Controller", "choose cheaply and safely"),
    ]
    x_positions = [0.08, 0.29, 0.50, 0.71, 0.90]
    for idx, ((title, subtitle), x) in enumerate(zip(boxes, x_positions)):
        ax.text(
            x,
            0.58,
            f"{title}\n{subtitle}",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef5ff", edgecolor="#4a7ebb", linewidth=1.4),
            transform=ax.transAxes,
        )
        if idx < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.085, 0.58),
                xytext=(x + 0.085, 0.58),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.6),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )
    ax.set_title("From Repeated Answers to Candidate-Space Control", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(plot_dir / "01_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def plot_space_scatter(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    wrong = [row for row in rows if f(row, "strict_correct") <= 0.0 and f(row, "rescue_candidate") <= 0.0 and f(row, "damage_candidate") <= 0.0]
    correct = [row for row in rows if f(row, "strict_correct") > 0.0 and f(row, "rescue_candidate") <= 0.0]
    rescue = [row for row in rows if f(row, "rescue_candidate") > 0.0]
    damage = [row for row in rows if f(row, "damage_candidate") > 0.0]

    def scatter(group: list[dict[str, Any]], label: str, color: str, marker: str, size: int, alpha: float) -> None:
        ax.scatter(
            [f(row, "logprob_avg_z") for row in group],
            [f(row, "token_mean_entropy_z") for row in group],
            s=size,
            c=color,
            marker=marker,
            alpha=alpha,
            label=label,
            edgecolors="white" if marker != "x" else None,
            linewidths=0.5,
        )

    scatter(wrong, "wrong", "#b8b8b8", "o", 28, 0.42)
    scatter(correct, "correct", "#2ca25f", "o", 35, 0.62)
    scatter(rescue, "rescue", "#1f78b4", "*", 120, 0.95)
    scatter(damage, "damage", "#d7301f", "X", 80, 0.95)
    ax.axvline(0.0, color="#555555", lw=0.8, alpha=0.5)
    ax.axhline(0.0, color="#555555", lw=0.8, alpha=0.5)
    ax.set_xlabel("Relative logprob within question (z-score, higher = model more confident)")
    ax.set_ylabel("Relative token mean entropy within question (z-score, lower = less hesitant)")
    ax.set_title("Candidate Space: Confidence vs Hesitation")
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "02_candidate_space_scatter.png")
    plt.close(fig)


def plot_rescue_damage_bars(rows: list[dict[str, Any]], plot_dir: Path) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    rescue = [row for row in rows if f(row, "rescue_candidate") > 0.0]
    damage = [row for row in rows if f(row, "damage_candidate") > 0.0]
    other = [row for row in rows if f(row, "rescue_candidate") <= 0.0 and f(row, "damage_candidate") <= 0.0]
    groups = [("rescue", rescue, "#1f78b4"), ("damage", damage, "#d7301f"), ("other", other, "#999999")]
    metrics = [
        ("logprob_avg_z", "Relative logprob"),
        ("token_mean_entropy_z", "Relative mean entropy"),
        ("token_count_z", "Relative answer length"),
    ]
    for ax, (metric, title) in zip(axes, metrics):
        means = [sum(f(row, metric) for row in group) / max(len(group), 1) for _, group, _ in groups]
        colors = [color for _, _, color in groups]
        ax.bar([name for name, _, _ in groups], means, color=colors, alpha=0.82)
        ax.axhline(0.0, color="#333333", lw=0.9)
        ax.set_title(title)
        ax.set_ylabel("mean z-score")
    fig.suptitle("What Makes Rescue/Damage Candidates Different?", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(plot_dir / "03_rescue_damage_feature_bars.png", bbox_inches="tight")
    plt.close(fig)


def plot_feature_separation(feature_rows: list[dict[str, Any]], plot_dir: Path) -> None:
    plt = setup_matplotlib()
    selected = [row for row in feature_rows if row["comparison"] in {"rescue_vs_nonrescue", "damage_vs_nondamage"}]
    selected = sorted(selected, key=lambda row: abs(f(row, "cohen_d")), reverse=True)[:14]
    labels = [f"{row['comparison'].replace('_vs_', ' vs ')}\n{row['feature']}" for row in selected]
    values = [f(row, "cohen_d") for row in selected]
    colors = ["#1f78b4" if row["comparison"] == "rescue_vs_nonrescue" else "#d7301f" for row in selected]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    y = list(range(len(selected)))
    ax.barh(y, values, color=colors, alpha=0.85)
    ax.axvline(0.0, color="#333333", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d (effect size)")
    ax.set_title("Which Features Separate Rescue/Damage Candidates?")
    fig.tight_layout()
    fig.savefig(plot_dir / "04_feature_separation.png")
    plt.close(fig)


def plot_score_summary(score_rows: list[dict[str, Any]], plot_dir: Path) -> None:
    plt = setup_matplotlib()
    formulas = [row["formula"] for row in score_rows]
    correct = [f(row, "strict_correct_rate") * 100.0 for row in score_rows]
    changed = [f(row, "answer_changed_rate") * 100.0 for row in score_rows]
    damaged = [f(row, "damaged_count") for row in score_rows]
    baseline = 41.94

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    x = list(range(len(formulas)))
    ax1.bar([pos - 0.18 for pos in x], correct, width=0.36, color="#2ca25f", label="Strict correct %")
    ax1.axhline(baseline, color="#555555", linestyle="--", lw=1.2, label="sample0 baseline")
    ax1.set_ylabel("Strict correct (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(formulas, rotation=18, ha="right")
    ax1.set_ylim(35, max(correct) + 8)

    ax2 = ax1.twinx()
    ax2.plot(x, changed, color="#756bb1", marker="o", label="Changed %")
    ax2.scatter(x, damaged, color="#d7301f", marker="x", s=80, label="Damaged count")
    ax2.set_ylabel("Changed (%) / Damaged count")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Simple Geometric Scores vs Baseline")
    fig.tight_layout()
    fig.savefig(plot_dir / "05_score_summary.png")
    plt.close(fig)


def plot_oracle_gap(question_rows: list[dict[str, Any]], plot_dir: Path) -> None:
    plt = setup_matplotlib()
    buckets = {
        "sample0 correct": 0,
        "sample0 wrong\nbut rescue exists": 0,
        "sample0 wrong\nno rescue": 0,
    }
    for row in question_rows:
        if f(row, "sample0_strict_correct") > 0.0:
            buckets["sample0 correct"] += 1
        elif f(row, "oracle_available") > 0.0:
            buckets["sample0 wrong\nbut rescue exists"] += 1
        else:
            buckets["sample0 wrong\nno rescue"] += 1
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    names = list(buckets.keys())
    values = list(buckets.values())
    ax.bar(names, values, color=["#2ca25f", "#1f78b4", "#b8b8b8"], alpha=0.86)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.8, str(value), ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Question-seed pairs")
    ax.set_title("Where Can Candidate-Space Selection Help?")
    fig.tight_layout()
    fig.savefig(plot_dir / "06_oracle_gap.png")
    plt.close(fig)


def write_visual_explanation(
    run_dir: Path,
    plot_dir: Path,
    rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    question_rows: list[dict[str, Any]],
) -> None:
    pair_count = len(question_rows)
    candidate_count = len(rows)
    rescue_count = sum(1 for row in rows if f(row, "rescue_candidate") > 0.0)
    damage_count = sum(1 for row in rows if f(row, "damage_candidate") > 0.0)
    sample0_correct = sum(1 for row in question_rows if f(row, "sample0_strict_correct") > 0.0)
    rescue_pair_count = sum(
        1
        for row in question_rows
        if f(row, "sample0_strict_correct") <= 0.0 and f(row, "oracle_available") > 0.0
    )
    sample0_wrong_no_rescue = pair_count - sample0_correct - rescue_pair_count

    best_score = max(score_rows, key=lambda row: f(row, "strict_correct_rate")) if score_rows else {}
    best_formula = best_score.get("formula", "N/A")
    best_rate = f(best_score, "strict_correct_rate")
    best_delta = f(best_score, "delta_strict_correct_vs_sample0")

    text = f"""# 候选空间可视化解读

## 01_pipeline.png

这张图讲的是整体思路：

同一个问题先生成多个候选答案，然后把每个候选答案变成一个特征点。我们不再直接改模型内部，而是学习如何在候选空间里识别更靠谱的点。

本轮数据规模：`{pair_count}` 个 question-seed pair，`{candidate_count}` 个候选答案点，包含 `{rescue_count}` 个 rescue candidate 和 `{damage_count}` 个 damage candidate。

## 02_candidate_space_scatter.png

横轴是相对 logprob，越往右表示模型在同一问题的候选里越“自信”。

纵轴是相对 token entropy，越往下表示生成时越“不犹豫”。

蓝色星星是 rescue candidate：sample0 错，但这个候选是对的。红色叉是 damage candidate：sample0 对，但这个候选会把答案换错。

最重要的直觉：

> rescue 点不是随机噪声，但在更大样本里不应被简单概括成“高置信、低犹豫必然可救”。更稳健的说法是：候选空间存在可学习结构，但高置信/低犹豫只是其中一部分信号。

所以 logprob/entropy 有用，但不能单独信。

## 03_rescue_damage_feature_bars.png

这张图把 rescue、damage、普通候选放在一起比较。

可以看到 damage 候选的答案长度明显偏长，而且相对 entropy 更高、相对 logprob 更低。

通俗说：

> 有些坏答案的特征像“啰嗦、绕、没把握但硬说”。

## 04_feature_separation.png

这张图显示哪些特征最能区分 rescue 或 damage。

当前最有价值的信号包括：

- rescue 候选和语义簇结构有关，不能只看单条答案；
- rescue 候选的相对 logprob/entropy 信号会随样本规模变弱；
- damage 候选通常更长；
- damage 候选通常更高 entropy、更低 logprob、更小语义簇。

这解释了为什么简单多数投票不够：正确答案有时是少数派。

## 05_score_summary.png

这张图比较几个简单组合分数。

这些分数把 logprob、entropy、cluster size、cluster weight 合在一起。本轮最好的简单公式是 `{best_formula}`，严格正确率为 `{best_rate:.2%}`，相对 sample0 提升 `{best_delta:.2%}`。

它说明：

> 多维组合信号比单独看一个指标更有希望。

## 06_oracle_gap.png

这张图回答“候选空间到底有没有可救的空间”。

在 `{pair_count}` 个 question-seed pair 里：

- `{sample0_correct}` 个 sample0 本来就是对的；
- `{rescue_pair_count}` 个 sample0 错，但候选里存在严格正确答案；
- `{sample0_wrong_no_rescue}` 个 sample0 错，候选里也没找到严格正确答案。

中间这类就是未来 controller 最值得学习的区域。

## 一句话总结

候选空间不是一团随机噪声。里面确实有结构，但更大样本说明这个结构不是一句“最高置信就是对”或“最低熵就是对”能概括的。更合理的方向是学习一个轻量 verifier/controller，综合多个特征判断什么时候应该保留 sample0，什么时候值得额外采样并切换答案。
"""
    (run_dir / "visual_explanation_zh.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    plot_dir = ensure_plot_dir(run_dir)
    rows = read_csv(run_dir / "candidate_features.csv")
    feature_rows = read_csv(run_dir / "feature_label_summary.csv")
    score_rows = read_csv(run_dir / "score_formula_summary.csv")
    question_rows = read_csv(run_dir / "question_geometry_summary.csv")

    plot_pipeline(plot_dir)
    plot_space_scatter(rows, plot_dir)
    plot_rescue_damage_bars(rows, plot_dir)
    plot_feature_separation(feature_rows, plot_dir)
    plot_score_summary(score_rows, plot_dir)
    plot_oracle_gap(question_rows, plot_dir)
    write_visual_explanation(run_dir, plot_dir, rows, score_rows, question_rows)
    print(f"Wrote plots to {plot_dir}")


if __name__ == "__main__":
    main()
