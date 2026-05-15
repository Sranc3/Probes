#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot VBPO training metrics.")
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    rows = read_metrics(run_dir / "train_metrics.jsonl")
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(json.dumps({"plot_dir": str(plot_dir), "plots": []}, indent=2))
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(steps, [row["loss"] for row in rows], color="#4C78A8")
    axes[0, 0].set_title("Preference Loss")
    axes[0, 1].plot(steps, [row["mean_margin_delta"] for row in rows], color="#59A14F", label="train")
    dev_rows = [row for row in rows if "dev_eval_margin_delta" in row]
    if dev_rows:
        axes[0, 1].plot([row["step"] for row in dev_rows], [row["dev_eval_margin_delta"] for row in dev_rows], color="#F28E2B", label="dev")
    axes[0, 1].axhline(0, color="#888888", linewidth=0.8)
    axes[0, 1].set_title("DPO Margin Delta")
    axes[0, 1].legend()
    axes[1, 0].plot(steps, [row["pair_accuracy"] for row in rows], color="#B07AA1", label="train")
    if dev_rows:
        axes[1, 0].plot([row["step"] for row in dev_rows], [row["dev_eval_pair_accuracy"] for row in dev_rows], color="#E15759", label="dev")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Pair Accuracy")
    axes[1, 0].legend()
    axes[1, 1].plot(steps, [row["grad_norm"] for row in rows], color="#79706E")
    axes[1, 1].set_title("Grad Norm")
    for ax in axes.ravel():
        ax.set_xlabel("step")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    out = plot_dir / "vbpo_training_dashboard.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(json.dumps({"plot_dir": str(plot_dir), "plots": [out.name]}, indent=2))


if __name__ == "__main__":
    main()
