"""Phase-5 unified dashboard combining cascade / difficulty / latency / geometry.

A single 2x3 grid that captures the four new analyses for a paper figure:

  (0,0) Multi-agent cascade Pareto frontier (HotpotQA OOD)
  (0,1) DCP score distribution (reveals bimodality limiting cascade)
  (0,2) Latency comparison (K=1 prompt forward, K=1 greedy, K=8 sampling)

  (1,0) Per-difficulty AUROC (probe x difficulty bucket bar chart)
  (1,1) Linear separability vs normalised depth (Qwen-7B/Llama-3B/Qwen-72B)
  (1,2) Adjacent-layer CKA vs depth — "where does the representation lock in?"

Output:  reports/dashboard_phase5.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR / "shared"))
sys.path.insert(0, str(THIS_DIR / "scripts"))
from probe_utils import risk_coverage_curve  # noqa: E402,F401  (kept for future use)
from run_all_models import BASES  # noqa: E402

BASE_TAGS = ["qwen7b", "llama3b", "qwen72b"]
BASE_PRETTY = {"qwen7b": "Qwen2.5-7B", "llama3b": "Llama-3.2-3B",
               "qwen72b": "Qwen2.5-72B"}
BASE_SHORT = {"qwen7b": "Qwen-7B", "llama3b": "Llama-3B", "qwen72b": "Qwen-72B"}
BASE_COLORS_DEEP = {"qwen7b": "#3182bd", "llama3b": "#9ecae1",
                    "qwen72b": "#08519c"}
BASE_MARKERS = {"qwen7b": "o", "llama3b": "s", "qwen72b": "^"}
BASE_N_LAYERS = {b.tag: b.num_hidden_layers for b in BASES}

PROBE_PRETTY = {
    "DCP_mlp": "DCP-MLP (ours)",
    "SEPs_logreg": "SEPs-LR",
    "SEPs_ridge": "SEPs-Ridge",
    "ARD_mlp": "ARD-MLP (ours)",
    "ARD_ridge": "ARD-Ridge (ours)",
}
PROBE_COLORS = {
    "DCP_mlp": "#d62728", "SEPs_logreg": "#1f77b4", "SEPs_ridge": "#7f7f7f",
    "ARD_mlp": "#2ca02c", "ARD_ridge": "#9467bd",
}


# ---------------------------------------------------------------------------
# Panel (0,0) Cascade Pareto
# ---------------------------------------------------------------------------


def panel_cascade(ax) -> None:
    df = pd.read_csv(THIS_DIR / "results" / "cascade_per_threshold_hotpotqa.csv")
    # Always-baselines
    for s, color, marker, lbl in [
        ("always_Llama-3B", "#9ecae1", "^", "Always Llama-3B"),
        ("always_Qwen-7B",  "#6baed6", "s", "Always Qwen-7B"),
        ("always_Qwen-72B", "#3182bd", "D", "Always Qwen-72B"),
        ("always_72B_K8",   "#08519c", "p", "Always Qwen-72B K=8"),
        ("always_teacher",  "#000000", "*", "Always Teacher"),
    ]:
        sub = df[df["strategy"] == s]
        if sub.empty:
            continue
        r = sub.iloc[0]
        ax.scatter(r["cost"], r["accuracy"], s=140, marker=marker, color=color,
                   edgecolors="black", linewidths=0.6, zorder=4)
        ax.annotate(lbl, (r["cost"], r["accuracy"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=7.5,
                    color=color)

    # Random cascade
    sub = df[df["strategy"] == "random_cascade"].sort_values("cost")
    ax.plot(sub["cost"], sub["accuracy"], color="#bcbcbc", linewidth=1.0,
            marker="x", markersize=3, alpha=0.6, label="Random cascade",
            zorder=2)

    # DCP cascade
    sub = df[df["strategy"] == "dcp_cascade"].sort_values("cost")
    ax.plot(sub["cost"], sub["accuracy"], color="#d62728", linewidth=1.8,
            marker="o", markersize=4, alpha=0.95,
            label="DCP cascade (ours)", zorder=5)

    # Oracle
    sub = df[df["strategy"] == "oracle_cascade"]
    if not sub.empty:
        r = sub.iloc[0]
        ax.scatter(r["cost"], r["accuracy"], s=160, marker="v",
                   color="#2ca02c", edgecolors="black", linewidths=0.7,
                   zorder=6, label="Oracle cascade")

    ax.set_xlabel("Avg cost per question (Qwen-7B forward units)")
    ax.set_ylabel("Avg accuracy")
    ax.set_xscale("log")
    ax.set_title("Multi-agent cascade — HotpotQA OOD\n"
                 "Cost-accuracy Pareto: cascade dominated by Always-72B in this regime",
                 fontsize=9.5)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax.set_ylim(0.2, 1.05)


# ---------------------------------------------------------------------------
# Panel (0,1) DCP score bimodality
# ---------------------------------------------------------------------------


def panel_score_bimodality(ax) -> None:
    df = pd.read_csv(THIS_DIR / "results" / "per_item_predictions_all.csv")
    df = df[(df["dataset"] == "hotpotqa") & (df["probe"] == "DCP_mlp")]
    for tag in BASE_TAGS:
        sub = df[df["base"] == tag]
        ax.hist(sub["y_score"], bins=30, alpha=0.55,
                color=BASE_COLORS_DEEP[tag], label=BASE_SHORT[tag],
                edgecolor="black", linewidth=0.3)
    ax.set_xlabel("DCP-MLP confidence score")
    ax.set_ylabel("Question count")
    ax.set_title("DCP score distribution (HotpotQA OOD)\n"
                 "Heavy bimodality (~50% at 0, ~30% at 1) → "
                 "cascade has discrete operating points, not a smooth curve.",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


# ---------------------------------------------------------------------------
# Panel (0,2) Latency comparison
# ---------------------------------------------------------------------------


def panel_latency(ax) -> None:
    df = pd.read_csv(THIS_DIR / "results" / "latency_measurements.csv")
    modes = ["prompt_only_forward", "k1_greedy", "k8_sampling"]
    mode_pretty = {
        "prompt_only_forward": "Prompt forward (probe input)",
        "k1_greedy": "K=1 greedy generation",
        "k8_sampling": "K=8 self-consistency",
    }
    mode_colors = {"prompt_only_forward": "#2ca02c",
                   "k1_greedy": "#1f77b4",
                   "k8_sampling": "#d62728"}
    n_bases = len(BASE_TAGS)
    n_modes = len(modes)
    width = 0.25
    x = np.arange(n_bases)
    for i, mode in enumerate(modes):
        vals = []
        for tag in BASE_TAGS:
            sub = df[(df["base"] == tag) & (df["mode"] == mode)]
            vals.append(float(sub.iloc[0]["median_ms"]) if not sub.empty else 0.0)
        offsets = (i - (n_modes - 1) / 2) * width
        bars = ax.bar(x + offsets, vals, width,
                      color=mode_colors[mode], label=mode_pretty[mode],
                      alpha=0.9, edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + max(vals)*0.01,
                    f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([BASE_SHORT[t] for t in BASE_TAGS], fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Median wall-clock (ms, log scale)")
    ax.set_title("Inference latency (8× H200, fp16, batch=1)\n"
                 "K=8 ≈ K=1 latency on GPU (batched sampling); compute cost still 8×",
                 fontsize=9)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.4)


# ---------------------------------------------------------------------------
# Panel (1,0) Per-difficulty AUROC
# ---------------------------------------------------------------------------


def panel_difficulty_buckets(ax) -> None:
    df = pd.read_csv(THIS_DIR / "results" / "difficulty_buckets.csv")
    bucket_order = ["easy", "hard_solvable", "saturated_wrong"]
    probes_show = ["DCP_mlp", "SEPs_logreg", "SEPs_ridge", "ARD_mlp"]
    n_buckets = len(bucket_order)
    n_probes = len(probes_show)
    width = 0.8 / n_probes
    x = np.arange(n_buckets)
    for pi, probe in enumerate(probes_show):
        vals, ns = [], []
        for b in bucket_order:
            sub = df[(df["bucket"] == b) & (df["probe"] == probe)]
            if sub.empty:
                vals.append(0)
                ns.append(0)
            else:
                vals.append(float(sub.iloc[0]["auroc"]))
                ns.append(int(sub.iloc[0]["n_questions"]))
        offsets = (pi - (n_probes - 1)/2) * width
        bars = ax.bar(x + offsets, vals, width,
                      color=PROBE_COLORS[probe],
                      label=PROBE_PRETTY[probe].split(" (")[0],
                      alpha=0.9, edgecolor="black", linewidth=0.3)
        for b, v in zip(bars, vals):
            if v > 0.005:
                ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                        ha="center", fontsize=6.5)
    # baseline for AUROC = 0.5
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(2.4, 0.51, " AUROC = 0.5 (random)", fontsize=7, color="black",
            alpha=0.7)
    # annotate bucket sizes
    bucket_n = {b: int(df[df["bucket"] == b]["n_questions"].iloc[0])
                if not df[df["bucket"] == b].empty else 0
                for b in bucket_order}
    bucket_labels = ["Easy\n(≥7/8 K=8 correct,\n"
                     f"n={bucket_n['easy']})",
                     "Hard-but-solvable\n(majority correct, <7/8,\n"
                     f"n={bucket_n['hard_solvable']})",
                     "Saturated wrong\n(majority wrong = hallucination zone,\n"
                     f"n={bucket_n['saturated_wrong']})"]
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, fontsize=7.5)
    ax.set_ylabel("AUROC within bucket")
    ax.set_title("Where does each probe win? — AUROC stratified by question difficulty\n"
                 "DCP wins on hallucination-prone questions; SEPs-LR on consistent-correct",
                 fontsize=9)
    ax.set_ylim(0.3, 0.85)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


# ---------------------------------------------------------------------------
# Panel (1,1) Linear separability vs normalised depth
# ---------------------------------------------------------------------------


def panel_linear_sep(ax) -> None:
    df = pd.read_csv(THIS_DIR / "results" / "layer_geometry.csv")
    for tag in BASE_TAGS:
        sub = df[df["base"] == tag].sort_values("norm_depth")
        ax.plot(sub["norm_depth"], sub["linear_sep_auroc"],
                color=BASE_COLORS_DEEP[tag], marker=BASE_MARKERS[tag],
                markersize=7, linewidth=2.0, label=BASE_PRETTY[tag])
    ax.axvline(0.71, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.text(0.715, 0.55, "  ref 0.71", fontsize=7, alpha=0.7)
    ax.set_xlabel("Normalised depth = layer / N")
    ax.set_ylabel("AUROC of single-layer logreg")
    ax.set_title("Linear separability vs depth\n"
                 "Where does P(correct) become linearly readable?",
                 fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.5, 0.86)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)


# ---------------------------------------------------------------------------
# Panel (1,2) Adjacent-layer CKA vs depth
# ---------------------------------------------------------------------------


def panel_cka(ax) -> None:
    df = pd.read_csv(THIS_DIR / "results" / "layer_geometry.csv")
    for tag in BASE_TAGS:
        sub = df[df["base"] == tag].dropna(subset=["cka_to_next"]).sort_values("norm_depth")
        # midpoint depth = depth of L (point shows CKA(L, L+1))
        ax.plot(sub["norm_depth"], sub["cka_to_next"],
                color=BASE_COLORS_DEEP[tag], marker=BASE_MARKERS[tag],
                markersize=7, linewidth=2.0, label=BASE_PRETTY[tag])
    ax.axhline(0.95, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(0.02, 0.96, "high similarity (locked in)", fontsize=7, color="gray",
            alpha=0.7)
    ax.axvline(0.71, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Normalised depth (start of L → L+1 step)")
    ax.set_ylabel("Linear CKA(L, L+1)")
    ax.set_title("Adjacent-layer CKA — where does the representation lock in?\n"
                 "High CKA = layer doesn't change much from previous (answer fixed)",
                 fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def main() -> None:
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.30,
                  left=0.05, right=0.97, top=0.92, bottom=0.06)

    fig.suptitle(
        "Phase-5 Unified Dashboard — Cascade · Difficulty · Latency · Geometry\n"
        "Going beyond AUROC: deployment-relevant cost, structural insights, mechanistic interpretation",
        fontsize=13, fontweight="bold", y=0.985,
    )

    panel_cascade(fig.add_subplot(gs[0, 0]))
    panel_score_bimodality(fig.add_subplot(gs[0, 1]))
    panel_latency(fig.add_subplot(gs[0, 2]))

    panel_difficulty_buckets(fig.add_subplot(gs[1, 0]))
    panel_linear_sep(fig.add_subplot(gs[1, 1]))
    panel_cka(fig.add_subplot(gs[1, 2]))

    out = THIS_DIR / "reports" / "dashboard_phase5.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[done] dashboard_phase5 -> {out}")
    print(f"[size] {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
