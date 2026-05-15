#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create attractor-level candidate-space plots.")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--max-panels", type=int, default=6)
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
            "grid.alpha": 0.22,
            "font.size": 9,
        }
    )
    return plt


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def grouped_by_pair(rows: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), row["question_id"])].append(row)
    return grouped


def build_attractors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attractors: list[dict[str, Any]] = []
    for (seed, question_id), pair_rows in grouped_by_pair(rows).items():
        sample0 = next(row for row in pair_rows if int(row["sample_index"]) == 0)
        sample0_correct = f(sample0, "strict_correct") > 0.0
        cluster_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in pair_rows:
            cluster_groups[int(row["cluster_id"])].append(row)
        for cluster_id, cluster_rows in cluster_groups.items():
            correct_count = sum(1 for row in cluster_rows if f(row, "strict_correct") > 0.0)
            wrong_count = len(cluster_rows) - correct_count
            contains_sample0 = any(int(row["sample_index"]) == 0 for row in cluster_rows)
            contains_rescue = (not sample0_correct) and correct_count > 0
            contains_damage = sample0_correct and wrong_count > 0
            attractors.append(
                {
                    "seed": seed,
                    "question_id": question_id,
                    "question": cluster_rows[0]["question"],
                    "cluster_id": cluster_id,
                    "x": mean([f(row, "logprob_avg_z") for row in cluster_rows]),
                    "y": mean([f(row, "token_mean_entropy_z") for row in cluster_rows]),
                    "size": len(cluster_rows),
                    "weight_mass": mean([f(row, "cluster_weight_mass") for row in cluster_rows]),
                    "correct_rate": correct_count / len(cluster_rows),
                    "correct_count": correct_count,
                    "wrong_count": wrong_count,
                    "contains_sample0": float(contains_sample0),
                    "contains_rescue": float(contains_rescue),
                    "contains_damage": float(contains_damage),
                    "sample0_correct": float(sample0_correct),
                    "semantic_clusters_set": int(float(cluster_rows[0]["semantic_clusters_set"])),
                    "semantic_entropy_weighted_set": f(cluster_rows[0], "semantic_entropy_weighted_set"),
                }
            )
    return attractors


def plot_attractor_basin_map(rows: list[dict[str, Any]], attractors: list[dict[str, Any]], plot_dir: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    ax.scatter(
        [f(row, "logprob_avg_z") for row in rows],
        [f(row, "token_mean_entropy_z") for row in rows],
        s=12,
        c="#d9d9d9",
        alpha=0.28,
        linewidths=0,
        label="candidate",
    )
    xs = [f(row, "x") for row in attractors]
    ys = [f(row, "y") for row in attractors]
    sizes = [26 + 34 * f(row, "size") for row in attractors]
    colors = [f(row, "correct_rate") for row in attractors]
    scatter = ax.scatter(
        xs,
        ys,
        s=sizes,
        c=colors,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        alpha=0.72,
        edgecolors="#333333",
        linewidths=0.45,
        label="semantic attractor",
    )
    rescue = [row for row in attractors if f(row, "contains_rescue") > 0.0]
    damage = [row for row in attractors if f(row, "contains_damage") > 0.0]
    ax.scatter(
        [f(row, "x") for row in rescue],
        [f(row, "y") for row in rescue],
        s=[70 + 38 * f(row, "size") for row in rescue],
        marker="*",
        facecolors="none",
        edgecolors="#1f78b4",
        linewidths=1.4,
        label="rescue attractor",
    )
    ax.scatter(
        [f(row, "x") for row in damage],
        [f(row, "y") for row in damage],
        s=[58 + 30 * f(row, "size") for row in damage],
        marker="x",
        c="#e34a33",
        linewidths=1.5,
        label="damage attractor",
    )
    ax.axhline(0, color="#999999", lw=0.8)
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_xlabel("Attractor centroid: relative logprob")
    ax.set_ylabel("Attractor centroid: relative mean entropy")
    ax.set_title("Semantic Attractors in Candidate Space")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    cbar.set_label("attractor strict-correct rate")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(plot_dir / "07_attractor_basin_map.png")
    plt.close(fig)


def pick_rescue_pairs(rows: list[dict[str, Any]], max_panels: int) -> list[tuple[tuple[int, str], list[dict[str, Any]]]]:
    pairs: list[tuple[tuple[int, str], list[dict[str, Any]], float]] = []
    for key, pair_rows in grouped_by_pair(rows).items():
        sample0 = next(row for row in pair_rows if int(row["sample_index"]) == 0)
        if f(sample0, "strict_correct") > 0.0:
            continue
        rescue_rows = [row for row in pair_rows if f(row, "strict_correct") > 0.0]
        if not rescue_rows:
            continue
        semantic_entropy = f(pair_rows[0], "semantic_entropy_weighted_set")
        cluster_count = f(pair_rows[0], "semantic_clusters_set")
        pairs.append((key, pair_rows, semantic_entropy + 0.15 * cluster_count))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return [(key, pair_rows) for key, pair_rows, _score in pairs[:max_panels]]


def plot_rescue_micrographs(rows: list[dict[str, Any]], plot_dir: Path, max_panels: int) -> None:
    plt = setup_matplotlib()
    selected = pick_rescue_pairs(rows, max_panels)
    if not selected:
        return
    cols = 3
    rows_count = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(12.0, 3.7 * rows_count), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, ((seed, question_id), pair_rows) in zip(axes.flat, selected):
        ax.axis("on")
        sample0 = next(row for row in pair_rows if int(row["sample_index"]) == 0)
        rescue_rows = [row for row in pair_rows if f(row, "strict_correct") > 0.0]
        cluster_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in pair_rows:
            cluster_groups[int(row["cluster_id"])].append(row)
        for cluster_id, cluster_rows in cluster_groups.items():
            cx = mean([f(row, "logprob_avg_z") for row in cluster_rows])
            cy = mean([f(row, "token_mean_entropy_z") for row in cluster_rows])
            ax.scatter(
                [f(row, "logprob_avg_z") for row in cluster_rows],
                [f(row, "token_mean_entropy_z") for row in cluster_rows],
                s=52,
                alpha=0.75,
                label=f"c{cluster_id}",
            )
            ax.scatter([cx], [cy], s=130 + 28 * len(cluster_rows), facecolors="none", edgecolors="#333333", linewidths=1.1)
            ax.text(cx, cy, f"C{cluster_id}\n{len(cluster_rows)}", ha="center", va="center", fontsize=7)
        for row in rescue_rows:
            ax.plot(
                [f(sample0, "logprob_avg_z"), f(row, "logprob_avg_z")],
                [f(sample0, "token_mean_entropy_z"), f(row, "token_mean_entropy_z")],
                color="#1f78b4",
                lw=1.2,
                alpha=0.65,
            )
        wrong_rows = [row for row in pair_rows if f(row, "strict_correct") <= 0.0 and int(row["sample_index"]) != 0]
        ax.scatter([f(row, "logprob_avg_z") for row in wrong_rows], [f(row, "token_mean_entropy_z") for row in wrong_rows], s=42, c="#bdbdbd", alpha=0.8)
        ax.scatter([f(row, "logprob_avg_z") for row in rescue_rows], [f(row, "token_mean_entropy_z") for row in rescue_rows], s=115, marker="*", c="#1f78b4", edgecolors="white", linewidths=0.5)
        ax.scatter([f(sample0, "logprob_avg_z")], [f(sample0, "token_mean_entropy_z")], s=90, marker="s", c="#fdae6b", edgecolors="#6b4c1d", linewidths=0.8)
        ax.axhline(0, color="#999999", lw=0.7)
        ax.axvline(0, color="#999999", lw=0.7)
        short_qid = question_id.replace("trivia_qa_", "q")
        ax.set_title(f"{short_qid}, seed={seed}: sample0 -> rescue attractor", fontsize=9)
        ax.set_xlabel("relative logprob")
        ax.set_ylabel("relative entropy")
    fig.suptitle("Rescue Cases as Local Semantic Attractors", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(plot_dir / "08_rescue_attractor_micrographs.png")
    plt.close(fig)


def plot_attractor_type_bars(attractors: list[dict[str, Any]], plot_dir: Path) -> None:
    plt = setup_matplotlib()
    groups = {
        "rescue": [row for row in attractors if f(row, "contains_rescue") > 0.0],
        "damage": [row for row in attractors if f(row, "contains_damage") > 0.0],
        "other": [row for row in attractors if f(row, "contains_rescue") <= 0.0 and f(row, "contains_damage") <= 0.0],
    }
    metrics = [
        ("centroid logprob", "x"),
        ("centroid entropy", "y"),
        ("basin size", "size"),
        ("correct rate", "correct_rate"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13.2, 3.4))
    colors = {"rescue": "#3182bd", "damage": "#de2d26", "other": "#9e9e9e"}
    for ax, (title, key) in zip(axes, metrics):
        names = list(groups.keys())
        values = [mean([f(row, key) for row in groups[name]]) for name in names]
        ax.bar(names, values, color=[colors[name] for name in names], alpha=0.82)
        ax.axhline(0, color="#777777", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel("mean")
    fig.suptitle("What Kind of Attractors Are Rescue/Damage?", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(plot_dir / "09_attractor_type_bars.png")
    plt.close(fig)


def write_attractor_notes(run_dir: Path, rows: list[dict[str, Any]], attractors: list[dict[str, Any]]) -> None:
    rescue = [row for row in attractors if f(row, "contains_rescue") > 0.0]
    damage = [row for row in attractors if f(row, "contains_damage") > 0.0]
    other = [row for row in attractors if f(row, "contains_rescue") <= 0.0 and f(row, "contains_damage") <= 0.0]
    text = f"""# 吸引子视角：图 2 的细粒度解释

## 什么是“吸引子”？

这里的吸引子不是额外假设出来的神秘对象，而是一个可观测定义：

> 在同一个问题的 8 个候选答案中，语义聚类形成的每个 cluster 就是一个局部答案吸引子。

对应关系：

- cluster centroid：这个吸引子在 logprob-entropy 平面里的位置。
- cluster size / weight mass：吸引盆大小，表示多少候选落进这个语义答案方向。
- cluster strict-correct rate：这个吸引子的质量。
- rescue attractor：sample0 错，但某个 cluster 中包含严格正确答案。
- damage attractor：sample0 对，但某个 cluster 中包含严格错误答案。

## 本轮 all-200 统计

- 候选答案点：`{len(rows)}`
- 语义吸引子数量：`{len(attractors)}`
- rescue attractor：`{len(rescue)}`
- damage attractor：`{len(damage)}`
- other attractor：`{len(other)}`

## 三张新图

- `07_attractor_basin_map.png`：把图 2 从“候选点”提升到“语义吸引子”。圆越大表示吸引盆越大，颜色越绿表示该吸引子越正确。
- `08_rescue_attractor_micrographs.png`：挑出若干 rescue case，展示 sample0 如何从错误点连到正确吸引子。
- `09_attractor_type_bars.png`：比较 rescue / damage / other 三类吸引子的平均位置、大小和正确率。

## 当前最重要的解释

图 2 的深层结构不是“单个点是否高置信”，而是“候选点如何落入不同语义吸引盆”。

damage attractor 更像发散尾部：低 logprob、高 entropy、小 basin、低正确率。

rescue attractor 更像困难问题里的局部正确盆：它未必是全局最大盆，也未必总在右下角，但在该问题内部通常不是孤立噪声，而是一个可以被识别的语义方向。

这说明下一步 controller 不应该只看单点分数，而应该使用 cluster-level 特征：sample0 所在盆的大小、候选空间分裂程度、备选盆的正确性线索、以及从 sample0 盆到备选盆的几何距离。
"""
    (run_dir / "attractor_visual_explanation_zh.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    plot_dir = ensure_plot_dir(run_dir)
    rows = read_csv(run_dir / "candidate_features.csv")
    attractors = build_attractors(rows)
    plot_attractor_basin_map(rows, attractors, plot_dir)
    plot_rescue_micrographs(rows, plot_dir, args.max_panels)
    plot_attractor_type_bars(attractors, plot_dir)
    write_attractor_notes(run_dir, rows, attractors)
    print(
        {
            "run_dir": str(run_dir),
            "candidate_count": len(rows),
            "attractor_count": len(attractors),
            "plots": [
                str(plot_dir / "07_attractor_basin_map.png"),
                str(plot_dir / "08_rescue_attractor_micrographs.png"),
                str(plot_dir / "09_attractor_type_bars.png"),
            ],
        }
    )


if __name__ == "__main__":
    main()
