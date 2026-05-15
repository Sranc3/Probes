#!/usr/bin/env python3
"""Create a conceptual cross-model answer-basin figure for paper drafts.

This is an illustrative diagram, not a plot of measured embeddings. It shows
how a weak model can have diffuse answer basins while a stronger model may form
sharper basins, including both factually anchored and stable-hallucination modes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource


OUT_DIR = Path("/zhutingqi/song/fig_paper")


def gaussian_basin(x_grid: np.ndarray, y_grid: np.ndarray, center: tuple[float, float], depth: float, width: float) -> np.ndarray:
    x0, y0 = center
    dist2 = (x_grid - x0) ** 2 + (y_grid - y0) ** 2
    return -depth * np.exp(-dist2 / (2 * width**2))


def make_landscape(kind: str, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    base = 0.18 * np.sin(1.4 * x_grid) * np.cos(1.1 * y_grid)
    base += 0.05 * (x_grid**2 + y_grid**2)

    if kind == "weak":
        wells = [
            ((-1.8, 0.9), 1.05, 0.90),
            ((0.2, -0.2), 0.85, 1.05),
            ((1.55, 1.35), 0.70, 0.85),
            ((-0.8, -1.55), 0.60, 0.75),
        ]
        ripple = 0.10 * np.sin(3.2 * x_grid + 0.5) * np.sin(2.7 * y_grid)
    else:
        wells = [
            ((-1.35, 0.55), 1.90, 0.46),  # anchored correct basin
            ((1.35, 0.95), 1.55, 0.40),  # stable hallucination basin
            ((0.15, -1.45), 0.75, 0.55),
        ]
        ripple = 0.04 * np.sin(3.4 * x_grid) * np.sin(2.8 * y_grid + 0.8)

    z_grid = base + ripple
    for center, depth, width in wells:
        z_grid += gaussian_basin(x_grid, y_grid, center, depth, width)
    return z_grid


def plot_panel(ax: plt.Axes, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray, title: str) -> None:
    light = LightSource(azdeg=320, altdeg=35)
    facecolors = light.shade(z_grid, cmap=cm.turbo, vert_exag=0.8, blend_mode="soft")
    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        facecolors=facecolors,
        rstride=2,
        cstride=2,
        linewidth=0,
        antialiased=True,
        alpha=0.96,
    )
    ax.plot_wireframe(x_grid, y_grid, z_grid + 0.012, rstride=7, cstride=7, color="white", linewidth=0.35, alpha=0.55)
    ax.contour(x_grid, y_grid, z_grid, zdir="z", offset=z_grid.min() - 0.28, levels=12, cmap="turbo", linewidths=0.55, alpha=0.8)
    ax.set_title(title, pad=10, fontsize=13, fontweight="bold")
    ax.set_xlabel("semantic axis 1", labelpad=4)
    ax.set_ylabel("semantic axis 2", labelpad=4)
    ax.set_zlabel("attraction energy", labelpad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=36, azim=-58)
    ax.set_box_aspect((1.0, 1.0, 0.55))
    ax.set_zlim(z_grid.min() - 0.30, z_grid.max() + 0.10)


def annotate_basin(ax: plt.Axes, x: float, y: float, z_grid: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, color: str) -> None:
    idx = np.unravel_index(np.argmin((x_grid - x) ** 2 + (y_grid - y) ** 2), x_grid.shape)
    z = float(z_grid[idx])
    ax.scatter([x], [y], [z - 0.04], s=62, color=color, edgecolor="black", linewidth=0.8, depthshade=False, zorder=10)


def value_at(x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray, x: float, y: float) -> float:
    idx = np.unravel_index(np.argmin((x_grid - x) ** 2 + (y_grid - y) ** 2), x_grid.shape)
    return float(z_grid[idx])


def plot_anchor_influence_panel(fig: plt.Figure) -> None:
    ax = fig.add_axes([0.045, 0.105, 0.91, 0.425], projection="3d")
    x = np.linspace(-2.8, 2.8, 220)
    y = np.linspace(-1.7, 1.7, 150)
    x_grid, y_grid = np.meshgrid(x, y)
    z = (
        gaussian_basin(x_grid, y_grid, (-1.25, 0.05), 1.0, 0.62)
        + gaussian_basin(x_grid, y_grid, (0.55, 0.32), 1.7, 0.42)
        + gaussian_basin(x_grid, y_grid, (1.75, -0.45), 1.15, 0.45)
        + 0.05 * (x_grid**2 + y_grid**2)
    )
    light = LightSource(azdeg=320, altdeg=36)
    facecolors = light.shade(z, cmap=cm.turbo, vert_exag=0.8, blend_mode="soft")
    ax.plot_surface(
        x_grid,
        y_grid,
        z,
        facecolors=facecolors,
        rstride=2,
        cstride=2,
        linewidth=0,
        antialiased=True,
        alpha=0.96,
    )
    ax.plot_wireframe(x_grid, y_grid, z + 0.01, rstride=8, cstride=8, color="white", linewidth=0.25, alpha=0.45)
    ax.contour(x_grid, y_grid, z, zdir="z", offset=z.min() - 0.18, levels=10, cmap="turbo", linewidths=0.45, alpha=0.72)

    weak = (-1.25, 0.05)
    anchor = (0.55, 0.32)
    unsupported = (1.75, -0.45)
    lift = 0.46
    weak_z = value_at(x_grid, y_grid, z, *weak) + lift
    anchor_z = value_at(x_grid, y_grid, z, *anchor) + lift
    unsupported_z = value_at(x_grid, y_grid, z, *unsupported) + lift
    ax.scatter(*weak, weak_z, s=88, c="#0B63CE", edgecolor="black", linewidth=0.8, depthshade=False, zorder=8)
    ax.scatter(*anchor, anchor_z, s=140, c="#0A8F3C", edgecolor="black", linewidth=0.8, marker="*", depthshade=False, zorder=9)
    ax.scatter(*unsupported, unsupported_z, s=88, c="#C92828", edgecolor="black", linewidth=0.8, depthshade=False, zorder=8)
    ax.plot([weak[0], weak[0]], [weak[1], weak[1]], [value_at(x_grid, y_grid, z, *weak), weak_z], color="#0B63CE", lw=1.1, alpha=0.9)
    ax.plot([anchor[0], anchor[0]], [anchor[1], anchor[1]], [value_at(x_grid, y_grid, z, *anchor), anchor_z], color="#0A8F3C", lw=1.1, alpha=0.9)
    ax.plot(
        [unsupported[0], unsupported[0]],
        [unsupported[1], unsupported[1]],
        [value_at(x_grid, y_grid, z, *unsupported), unsupported_z],
        color="#C92828",
        lw=1.1,
        alpha=0.9,
    )
    ax.quiver(
        weak[0] + 0.08,
        weak[1] + 0.04,
        weak_z + 0.18,
        anchor[0] - weak[0] - 0.22,
        anchor[1] - weak[1] - 0.05,
        anchor_z - weak_z,
        color="#0A8F3C",
        linewidth=3.4,
        arrow_length_ratio=0.20,
        normalize=False,
    )
    ax.quiver(
        anchor[0] + 0.20,
        anchor[1] - 0.08,
        anchor_z + 0.13,
        unsupported[0] - anchor[0] - 0.40,
        unsupported[1] - anchor[1] + 0.15,
        unsupported_z - anchor_z,
        color="#C92828",
        linewidth=2.2,
        arrow_length_ratio=0.18,
        linestyle="--",
        normalize=False,
    )
    ax.set_title("Anchor-guided basin shift", fontsize=11, fontweight="bold", pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=41, azim=-58)
    ax.set_box_aspect((2.05, 1.0, 0.52))
    ax.set_zlim(z.min() - 0.18, z.max() + 0.95)

    for x_pos, y_pos, label, color in [
        (0.39, 0.29, "[1]", "#0B63CE"),
        (0.52, 0.36, "[2]", "#0A8F3C"),
        (0.63, 0.25, "[3]", "#C92828"),
    ]:
        fig.text(
            x_pos,
            y_pos,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": color, "edgecolor": "black", "alpha": 0.96},
        )
    ax.annotate(
        "",
        xy=(0.52, 0.36),
        xytext=(0.40, 0.29),
        xycoords=fig.transFigure,
        textcoords=fig.transFigure,
        arrowprops={"arrowstyle": "->", "lw": 2.6, "color": "#0A8F3C"},
    )
    ax.annotate(
        "",
        xy=(0.63, 0.25),
        xytext=(0.54, 0.35),
        xycoords=fig.transFigure,
        textcoords=fig.transFigure,
        arrowprops={"arrowstyle": "-|>", "lw": 2.0, "linestyle": "--", "color": "#C92828"},
    )
    fig.text(0.20, 0.445, "[1] student basin", ha="center", va="center", color="#0B63CE", fontsize=9.5, fontweight="bold")
    fig.text(0.50, 0.445, "[2] teacher factual anchor pulls probability mass", ha="center", va="center", color="#0A8F3C", fontsize=9.5, fontweight="bold")
    fig.text(0.80, 0.445, "[3] unsupported stable basin is penalized", ha="center", va="center", color="#C92828", fontsize=9.5, fontweight="bold")


def figure_legend(fig: plt.Figure, entries: list[tuple[str, str]]) -> None:
    x_positions = [0.18, 0.385, 0.61, 0.82]
    for x_pos, (label, color) in zip(x_positions, entries):
        fig.text(
            x_pos,
            0.064,
            f"● {label}",
            ha="center",
            va="center",
            color=color,
            fontsize=9.5,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": color, "alpha": 0.90},
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-3, 3, 150)
    y = np.linspace(-3, 3, 150)
    x_grid, y_grid = np.meshgrid(x, y)

    weak_z = make_landscape("weak", x_grid, y_grid)
    strong_z = make_landscape("strong", x_grid, y_grid)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelsize": 8,
        }
    )

    fig = plt.figure(figsize=(13.5, 6.4), constrained_layout=False)
    ax1 = fig.add_axes([0.015, 0.17, 0.47, 0.71], projection="3d")
    ax2 = fig.add_axes([0.515, 0.17, 0.47, 0.71], projection="3d")

    plot_panel(ax1, x_grid, y_grid, weak_z, "Smaller model: diffuse competing answer basins")
    plot_panel(ax2, x_grid, y_grid, strong_z, "Larger model: sharper basins, not always factual")

    annotate_basin(ax1, -1.8, 0.9, weak_z, x_grid, y_grid, "#0B63CE")
    annotate_basin(ax1, 0.2, -0.2, weak_z, x_grid, y_grid, "#7B3FC8")
    annotate_basin(ax2, -1.35, 0.55, strong_z, x_grid, y_grid, "#0A8F3C")
    annotate_basin(ax2, 1.35, 0.95, strong_z, x_grid, y_grid, "#C92828")
    figure_legend(
        fig,
        [
            ("diffuse candidate basin", "#0B63CE"),
            ("ambiguous competitor", "#7B3FC8"),
            ("factual anchor basin", "#0A8F3C"),
            ("stable hallucination basin", "#C92828"),
        ],
    )

    fig.text(
        0.5,
        0.945,
        "Cross-Model Answer Basin Geometry",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.027,
        "Conceptual diagram: stronger models may sharpen basin geometry; correctness depends on factual anchoring, not stability alone.",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
    )

    png_path = OUT_DIR / "cross_model_answer_basin_concept.png"
    svg_path = OUT_DIR / "cross_model_answer_basin_concept.svg"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    caption = OUT_DIR / "cross_model_answer_basin_concept_caption.md"
    caption.write_text(
        "# Cross-Model Answer Basin Concept Figure\n\n"
        "This is a conceptual figure for paper/abstract use, not a measured embedding plot. "
        "It illustrates the hypothesis that larger models may form sharper answer basins, "
        "while basin sharpness alone does not imply factual correctness. A basin becomes a "
        "factual anchor when it is supported by cross-model agreement or evidence, whereas "
        "a sharp unsupported basin can represent stable hallucination.\n\n"
        "Generated files:\n\n"
        "- `cross_model_answer_basin_concept.png`\n"
        "- `cross_model_answer_basin_concept.svg`\n",
        encoding="utf-8",
    )

    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
