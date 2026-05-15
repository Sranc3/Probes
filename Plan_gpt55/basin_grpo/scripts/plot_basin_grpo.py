#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Basin-GRPO training curves.")
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    rows = read_jsonl(run_dir / "train_metrics.jsonl")
    if not rows:
        raise SystemExit(f"No train_metrics.jsonl found in {run_dir}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    steps = [row["step"] for row in rows]

    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})

    panels = [
        ("reward_loss", [("mean_reward", "Reward"), ("loss", "Loss")], "Reward and Loss"),
        ("correctness", [("mean_strict", "Train Strict"), ("dev_mean_strict", "Dev Greedy Strict")], "Correctness"),
        ("reward_terms", [("mean_f1", "F1"), ("mean_length_cost", "Length"), ("mean_stable_wrong_basin", "Wrong Basin"), ("mean_damage", "Damage"), ("mean_correct_consensus", "Correct Consensus")], "Reward Terms"),
        ("optimization", [("mean_kl", "KL proxy"), ("mean_ratio", "Ratio"), ("grad_norm", "Grad Norm")], "Optimization Stability"),
    ]
    for filename, series, title in panels:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for key, label in series:
            values = [row.get(key) for row in rows]
            if any(value is not None for value in values):
                ax.plot(steps, [float(value) if value is not None else float("nan") for value in values], marker="o", markersize=3, label=label)
        ax.set_xlabel("Step")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{filename}.png")
        plt.close(fig)

    # Combined dashboard.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    dashboard = [
        (axes[0, 0], [("mean_reward", "Reward"), ("loss", "Loss")], "Reward/Loss"),
        (axes[0, 1], [("mean_strict", "Train Strict"), ("dev_mean_strict", "Dev Strict")], "Correctness"),
        (axes[1, 0], [("mean_stable_wrong_basin", "Wrong Basin"), ("mean_damage", "Damage"), ("mean_correct_consensus", "Consensus")], "Basin Terms"),
        (axes[1, 1], [("mean_kl", "KL"), ("grad_norm", "Grad Norm")], "Stability"),
    ]
    for ax, series, title in dashboard:
        for key, label in series:
            values = [row.get(key) for row in rows]
            if any(value is not None for value in values):
                ax.plot(steps, [float(value) if value is not None else float("nan") for value in values], marker="o", markersize=3, label=label)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "training_dashboard.png")
    plt.close(fig)
    print(json.dumps({"plot_dir": str(plot_dir), "plots": sorted(path.name for path in plot_dir.glob("*.png"))}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
