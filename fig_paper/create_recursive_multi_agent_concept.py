#!/usr/bin/env python3
"""Create a clean conceptual figure for recursive strong-to-weak multi-agent guidance."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("/zhutingqi/song/fig_paper")


COLORS = {
    "teacher": "#2F5D9B",
    "agent": "#E8F1FF",
    "agent_edge": "#6E95C8",
    "aggregator": "#F3F4F6",
    "aggregator_edge": "#8A8F98",
    "anchor": "#E8F6EE",
    "anchor_edge": "#2A9D62",
    "output": "#FFF4E6",
    "output_edge": "#D98C2B",
    "text": "#222222",
    "muted": "#5F6673",
}


def rounded_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, label: str, fc: str, ec: str, fontsize: int = 12) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.6,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=COLORS["text"],
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#4B5563",
    rad: float = 0.0,
    lw: float = 1.7,
    style: str = "-|>",
    dashed: bool = False,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.94,
        "Recursive Strong-to-Weak Multi-Agent Guidance",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        0.5,
        0.895,
        "A strong model supplies anchors and feedback; weak agents explore in parallel; aggregation feeds the next round.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    rounded_box(ax, (0.39, 0.75), 0.22, 0.09, "Strong model\nteacher / anchor", "#EAF1FF", COLORS["teacher"], 12)
    rounded_box(ax, (0.08, 0.50), 0.19, 0.08, "Weak agent 1", COLORS["agent"], COLORS["agent_edge"], 11)
    rounded_box(ax, (0.31, 0.50), 0.19, 0.08, "Weak agent 2", COLORS["agent"], COLORS["agent_edge"], 11)
    rounded_box(ax, (0.54, 0.50), 0.19, 0.08, "Weak agent 3", COLORS["agent"], COLORS["agent_edge"], 11)
    rounded_box(ax, (0.77, 0.50), 0.15, 0.08, "Agent n", COLORS["agent"], COLORS["agent_edge"], 11)

    rounded_box(ax, (0.31, 0.30), 0.18, 0.085, "Basin / answer\naggregation", COLORS["aggregator"], COLORS["aggregator_edge"], 10.5)
    rounded_box(ax, (0.53, 0.30), 0.18, 0.085, "Verifier\nor judge", COLORS["anchor"], COLORS["anchor_edge"], 10.5)
    rounded_box(ax, (0.39, 0.12), 0.22, 0.085, "Selected answer\n+ next prompt state", COLORS["output"], COLORS["output_edge"], 10.5)

    teacher_bottom = (0.50, 0.75)
    for x in [0.175, 0.405, 0.635, 0.845]:
        arrow(ax, teacher_bottom, (x, 0.585), color=COLORS["teacher"], rad=0.0, lw=1.45)

    for x in [0.175, 0.405, 0.635, 0.845]:
        arrow(ax, (x, 0.50), (0.40, 0.385), color="#6B7280", rad=0.0, lw=1.35)

    arrow(ax, (0.49, 0.342), (0.53, 0.342), color="#6B7280", lw=1.6)
    arrow(ax, (0.62, 0.30), (0.50, 0.205), color=COLORS["anchor_edge"], lw=1.7)
    arrow(ax, (0.40, 0.30), (0.48, 0.205), color=COLORS["aggregator_edge"], lw=1.7)

    arrow(ax, (0.61, 0.165), (0.61, 0.795), color="#7C3AED", rad=0.62, lw=2.0, dashed=True)
    ax.text(
        0.80,
        0.34,
        "recursive feedback:\nupdate anchors,\nprompts, or roles",
        ha="left",
        va="center",
        fontsize=9.5,
        color="#6D28D9",
        fontweight="bold",
    )

    ax.text(0.12, 0.43, "parallel exploration", ha="left", va="center", fontsize=9.2, color=COLORS["muted"])
    ax.text(0.41, 0.43, "candidate basins", ha="left", va="center", fontsize=9.2, color=COLORS["muted"])
    ax.text(0.67, 0.43, "cross-checking", ha="left", va="center", fontsize=9.2, color=COLORS["muted"])

    fig.text(
        0.5,
        0.035,
        "Conceptual diagram: the strong model need not generate the final answer directly; it can guide, verify, and recursively reshape a weak-agent search process.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLORS["muted"],
    )

    png_path = OUT_DIR / "recursive_multi_agent_concept.png"
    svg_path = OUT_DIR / "recursive_multi_agent_concept.svg"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
