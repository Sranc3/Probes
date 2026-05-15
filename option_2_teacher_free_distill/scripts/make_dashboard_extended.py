"""Extended cross-base dashboard focused on metrics beyond AUROC.

3 x 3 grid:
  Row 1 (Risk-Coverage trade-off curves, the main visual story):
    (0,0) ID  RC curves
    (0,1) HotpotQA OOD RC curves
    (0,2) NQ-Open OOD RC curves
  Row 2 (Operating-point views):
    (1,0) Coverage @ Risk <= 10%   bar matrix (3 datasets x 3 bases x 3 probes)
    (1,1) AUPRC heatmap (probes x base*dataset)
    (1,2) sel_acc @ 50% coverage   bar matrix
  Row 3 (Calibration story):
    (2,0) Brier vs AUROC scatter — ARD is brutally well-calibrated
    (2,1) Reliability diagram (Qwen-72B ID, DCP vs SEPs-LR vs ARD-MLP)
    (2,2) Brier decomposition stacked bars (Reliability + Resolution)

Output: ``reports/dashboard_extended.png``.
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
from probe_utils import (  # noqa: E402
    reliability_diagram_data,
    risk_coverage_curve,
)
from run_all_models import BASES  # noqa: E402

BASE_TAGS = ["qwen7b", "llama3b", "qwen72b"]
BASE_PRETTY = {"qwen7b": "Qwen2.5-7B", "llama3b": "Llama-3.2-3B", "qwen72b": "Qwen2.5-72B"}
BASE_SHORT = {"qwen7b": "Qwen-7B", "llama3b": "Llama-3B", "qwen72b": "Qwen-72B"}
BASE_N_LAYERS = {b.tag: b.num_hidden_layers for b in BASES}

PROBE_PRETTY = {
    "DCP_mlp": "DCP-MLP (ours)",
    "SEPs_logreg": "SEPs-LR",
    "SEPs_ridge": "SEPs-Ridge (Kossen 2024)",
    "ARD_mlp": "ARD-MLP (ours)",
    "ARD_ridge": "ARD-Ridge (ours)",
}
PROBE_COLORS = {
    "DCP_mlp": "#d62728",
    "SEPs_logreg": "#1f77b4",
    "SEPs_ridge": "#7f7f7f",
    "ARD_mlp": "#2ca02c",
    "ARD_ridge": "#9467bd",
}
BASE_LINE = {"qwen7b": "-", "llama3b": "--", "qwen72b": ":"}
BASE_MARKERS = {"qwen7b": "o", "llama3b": "s", "qwen72b": "^"}

DATASET_PRETTY = {"id": "TriviaQA  ID", "hotpotqa": "HotpotQA  OOD",
                  "nq": "NQ-Open  OOD"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_metrics() -> pd.DataFrame:
    return pd.read_csv(THIS_DIR / "results" / "extended_metrics_long.csv")


def load_per_item() -> pd.DataFrame:
    return pd.read_csv(THIS_DIR / "results" / "per_item_predictions_all.csv")


# ---------------------------------------------------------------------------
# Row 1: Risk-Coverage curves
# ---------------------------------------------------------------------------


def panel_rc_curves(ax, per_item: pd.DataFrame, dataset: str) -> None:
    probes_main = ["DCP_mlp", "SEPs_ridge"]
    probes_ref = ["SEPs_logreg"]

    for tag in BASE_TAGS:
        for probe, lw, alpha in [(p, 1.8, 0.95) for p in probes_main] + \
                                [(p, 1.0, 0.5) for p in probes_ref]:
            sub = per_item[(per_item["base"] == tag) &
                           (per_item["dataset"] == dataset) &
                           (per_item["probe"] == probe)]
            if sub.empty:
                continue
            cov, risk = risk_coverage_curve(
                sub["y_true"].to_numpy(), sub["y_score"].to_numpy()
            )
            ax.plot(cov, risk,
                    color=PROBE_COLORS[probe], linestyle=BASE_LINE[tag],
                    linewidth=lw, alpha=alpha)

    # 10% / 20% risk reference horizontal lines
    ax.axhline(0.10, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.axhline(0.20, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.text(0.02, 0.105, "risk = 10%", fontsize=6.5, color="gray", alpha=0.7)
    ax.text(0.02, 0.205, "risk = 20%", fontsize=6.5, color="gray", alpha=0.7)

    ax.set_xlabel("Coverage  (fraction answered)")
    ax.set_ylabel("Risk  (1 − selective accuracy)")
    ax.set_title(f"{DATASET_PRETTY[dataset]} — Risk-Coverage trade-off",
                 fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.85 if dataset != "id" else 0.55)
    ax.grid(linestyle=":", alpha=0.4)

    if dataset == "id":
        probe_handles = [
            Line2D([0], [0], color=PROBE_COLORS["DCP_mlp"], linewidth=2.2,
                   label="DCP-MLP (ours)"),
            Line2D([0], [0], color=PROBE_COLORS["SEPs_ridge"], linewidth=2.2,
                   label="SEPs-Ridge"),
            Line2D([0], [0], color=PROBE_COLORS["SEPs_logreg"], linewidth=1.5,
                   alpha=0.5, label="SEPs-LR (ref)"),
        ]
        base_handles = [
            Line2D([0], [0], color="black", linestyle=BASE_LINE[t],
                   marker=BASE_MARKERS[t], markersize=4, label=BASE_PRETTY[t])
            for t in BASE_TAGS
        ]
        leg1 = ax.legend(handles=probe_handles, loc="upper left", fontsize=7,
                         title="Probe", title_fontsize=7, framealpha=0.9)
        ax.add_artist(leg1)
        ax.legend(handles=base_handles, loc="lower right", fontsize=7,
                  title="Base", title_fontsize=7, framealpha=0.9)


# ---------------------------------------------------------------------------
# Row 2 (1,0): Coverage @ Risk <= 10% bar matrix
# ---------------------------------------------------------------------------


def panel_cov_at_risk(ax, metrics: pd.DataFrame, max_risk: float = 0.10) -> None:
    probes = ["DCP_mlp", "SEPs_logreg", "SEPs_ridge"]
    datasets = ["id", "hotpotqa", "nq"]
    col = f"cov@risk<={max_risk:.2f}"

    n_groups = len(BASE_TAGS) * len(datasets)
    n_probes = len(probes)
    width = 0.8 / n_probes
    x = np.arange(n_groups)

    for pi, probe in enumerate(probes):
        vals = []
        for tag in BASE_TAGS:
            for ds in datasets:
                sub = metrics[(metrics["base"] == tag) &
                              (metrics["dataset"] == ds) &
                              (metrics["probe"] == probe)]
                vals.append(float(sub.iloc[0][col]) if not sub.empty else 0.0)
        offsets = (pi - (n_probes - 1) / 2) * width
        bars = ax.bar(x + offsets, vals, width,
                      color=PROBE_COLORS[probe],
                      label=PROBE_PRETTY[probe].split(" (")[0],
                      alpha=0.85)
        for b, v in zip(bars, vals):
            if v > 0.005:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5)

    labels = []
    for tag in BASE_TAGS:
        for ds in datasets:
            labels.append(f"{BASE_SHORT[tag]}\n{ds.upper()}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    # add base separators
    for i in range(1, len(BASE_TAGS)):
        ax.axvline(i * len(datasets) - 0.5, color="gray",
                   linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_ylabel(f"Coverage @ Risk ≤ {int(max_risk*100)}%")
    ax.set_title(f"Coverage @ Risk ≤ {int(max_risk*100)}%  —  deployment-actionable\n"
                 f"\"If we accept ≤{int(max_risk*100)}% error, what fraction can we still answer?\"",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(0, max(0.40, ax.get_ylim()[1]))


# ---------------------------------------------------------------------------
# Row 2 (1,1): AUPRC heatmap
# ---------------------------------------------------------------------------


def panel_auprc_heatmap(ax, metrics: pd.DataFrame) -> None:
    probes = ["DCP_mlp", "SEPs_logreg", "SEPs_ridge", "ARD_mlp", "ARD_ridge"]
    datasets = ["id", "hotpotqa", "nq"]
    col_pairs = [(t, ds) for t in BASE_TAGS for ds in datasets]

    M = np.full((len(probes), len(col_pairs)), np.nan)
    for ri, probe in enumerate(probes):
        for ci, (tag, ds) in enumerate(col_pairs):
            sub = metrics[(metrics["base"] == tag) &
                          (metrics["dataset"] == ds) &
                          (metrics["probe"] == probe)]
            if not sub.empty:
                M[ri, ci] = float(sub.iloc[0]["auprc"])

    # base_acc reference per dataset (lower bound for AUPRC = base rate)
    base_rates = []
    for tag, ds in col_pairs:
        sub = metrics[(metrics["base"] == tag) & (metrics["dataset"] == ds)]
        base_rates.append(float(sub.iloc[0]["base_acc"]) if not sub.empty else np.nan)

    im = ax.imshow(M, aspect="auto", cmap="YlGnBu", vmin=0.4, vmax=0.9)
    ax.set_xticks(range(len(col_pairs)))
    ax.set_xticklabels([f"{BASE_SHORT[t]}\n{ds.upper()}\n(p={br:.2f})"
                        for (t, ds), br in zip(col_pairs, base_rates)],
                       fontsize=7)
    ax.set_yticks(range(len(probes)))
    ax.set_yticklabels([PROBE_PRETTY[p].split(" (")[0] for p in probes], fontsize=8)
    ax.set_title("AUPRC (Average Precision)\n"
                 "Higher = better at ranking correct answers above incorrect.\n"
                 "p = base rate (random AUPRC).", fontsize=9)
    for ri in range(M.shape[0]):
        for ci in range(M.shape[1]):
            v = M[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=("black" if v < 0.7 else "white"))
    for i in range(1, len(BASE_TAGS)):
        ax.axvline(i * len(datasets) - 0.5, color="white",
                   linestyle="-", linewidth=1.5)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="AUPRC")


# ---------------------------------------------------------------------------
# Row 2 (1,2): sel_acc @ 50% coverage matrix
# ---------------------------------------------------------------------------


def panel_sel_acc_at_50(ax, metrics: pd.DataFrame) -> None:
    probes = ["DCP_mlp", "SEPs_logreg", "SEPs_ridge"]
    datasets = ["id", "hotpotqa", "nq"]
    n_groups = len(BASE_TAGS) * len(datasets)
    n_probes = len(probes)
    width = 0.8 / n_probes
    x = np.arange(n_groups)

    for pi, probe in enumerate(probes):
        vals = []
        for tag in BASE_TAGS:
            for ds in datasets:
                sub = metrics[(metrics["base"] == tag) &
                              (metrics["dataset"] == ds) &
                              (metrics["probe"] == probe)]
                vals.append(float(sub.iloc[0]["sel_acc@0.50"]) if not sub.empty else 0.0)
        offsets = (pi - (n_probes - 1) / 2) * width
        bars = ax.bar(x + offsets, vals, width,
                      color=PROBE_COLORS[probe],
                      label=PROBE_PRETTY[probe].split(" (")[0],
                      alpha=0.85)
        for b, v in zip(bars, vals):
            if v > 0.005:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5)

    # base accuracy reference dotted lines per group
    for gi, (tag, ds) in enumerate([(t, ds) for t in BASE_TAGS for ds in ["id","hotpotqa","nq"]]):
        sub = metrics[(metrics["base"] == tag) & (metrics["dataset"] == ds)]
        if not sub.empty:
            ba = float(sub.iloc[0]["base_acc"])
            ax.hlines(ba, gi - 0.4, gi + 0.4, color="black",
                      linestyle=":", linewidth=1.0, alpha=0.6)

    labels = [f"{BASE_SHORT[t]}\n{ds.upper()}"
              for t in BASE_TAGS for ds in datasets]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    for i in range(1, len(BASE_TAGS)):
        ax.axvline(i * len(datasets) - 0.5, color="gray",
                   linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_ylabel("Selective accuracy @ 50% coverage")
    ax.set_title("Sel-Acc @ 50% coverage\n"
                 "Dotted black = base accuracy (no abstention).", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(0, 1.0)


# ---------------------------------------------------------------------------
# Row 3 (2,0): Brier vs AUROC scatter
# ---------------------------------------------------------------------------


def panel_brier_auroc_scatter(ax, metrics: pd.DataFrame) -> None:
    probes = ["DCP_mlp", "SEPs_logreg", "ARD_mlp", "ARD_ridge"]  # SEPs-Ridge has NaN brier
    datasets = ["id", "hotpotqa", "nq"]
    base_marker = {"qwen7b": "o", "llama3b": "s", "qwen72b": "^"}
    ds_size = {"id": 110, "hotpotqa": 70, "nq": 45}

    for tag in BASE_TAGS:
        for probe in probes:
            for ds in datasets:
                sub = metrics[(metrics["base"] == tag) &
                              (metrics["dataset"] == ds) &
                              (metrics["probe"] == probe)]
                if sub.empty:
                    continue
                au = float(sub.iloc[0]["auroc"])
                br = float(sub.iloc[0]["brier"])
                if np.isnan(br):
                    continue
                ax.scatter(br, au, s=ds_size[ds],
                           color=PROBE_COLORS[probe], marker=base_marker[tag],
                           edgecolors="black", linewidths=0.6, alpha=0.85)

    # legend
    probe_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PROBE_COLORS[p],
               markersize=8, label=PROBE_PRETTY[p].split(" (")[0])
        for p in probes
    ]
    base_handles = [
        Line2D([0], [0], marker=base_marker[t], color="w",
               markerfacecolor="lightgray", markersize=8,
               markeredgecolor="black", label=BASE_PRETTY[t])
        for t in BASE_TAGS
    ]
    ax.set_xlabel("Brier score  (lower = better calibration+discrimination)")
    ax.set_ylabel("AUROC")
    ax.set_title("Calibration vs Discrimination\n"
                 "ARD probes: lowest Brier (best-calibrated). "
                 "DCP/SEPs-LR: highest AUROC but worst calibration.",
                 fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)
    leg1 = ax.legend(handles=probe_handles, loc="lower right",
                     fontsize=7, title="Probe", title_fontsize=7)
    ax.add_artist(leg1)
    ax.legend(handles=base_handles, loc="upper right",
              fontsize=7, title="Base (marker)", title_fontsize=7)


# ---------------------------------------------------------------------------
# Row 3 (2,1): Reliability diagram (Qwen-72B ID, multi-probe)
# ---------------------------------------------------------------------------


def panel_reliability_diagram(ax, per_item: pd.DataFrame,
                              tag: str = "qwen72b", dataset: str = "id") -> None:
    probes = ["DCP_mlp", "SEPs_logreg", "ARD_mlp", "ARD_ridge"]
    n_bins = 10

    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8,
            alpha=0.5, label="perfect calibration")

    for probe in probes:
        sub = per_item[(per_item["base"] == tag) &
                       (per_item["dataset"] == dataset) &
                       (per_item["probe"] == probe)]
        if sub.empty:
            continue
        y = sub["y_true"].to_numpy().astype(float)
        s = sub["y_score"].to_numpy().astype(float)
        # ARD outputs a regression-style score; clip to [0,1] for visualisation
        if probe.startswith("ARD"):
            s = np.clip(s, 0.0, 1.0)
        conf, acc, cnt = reliability_diagram_data(y, s, n_bins=n_bins)
        mask = cnt > 0
        ax.plot(conf[mask], acc[mask],
                marker="o", color=PROBE_COLORS[probe], linewidth=1.5,
                label=PROBE_PRETTY[probe].split(" (")[0], alpha=0.9)

    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability diagram — {BASE_PRETTY[tag]} {dataset.upper()}\n"
                 "Closer to diagonal = better calibrated.", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)


# ---------------------------------------------------------------------------
# Row 3 (2,2): Brier decomposition stacked bars
# ---------------------------------------------------------------------------


def panel_brier_decomposition(ax, metrics: pd.DataFrame) -> None:
    """Stacked bar of Reliability + (Uncertainty − Resolution) for ID only.

    Brier = Reliability − Resolution + Uncertainty
    Lower is better; Reliability bad = miscalibration; Resolution good = useful spread.
    """
    probes = ["DCP_mlp", "SEPs_logreg", "ARD_mlp", "ARD_ridge"]
    rows = []
    for tag in BASE_TAGS:
        for probe in probes:
            sub = metrics[(metrics["base"] == tag) &
                          (metrics["dataset"] == "id") &
                          (metrics["probe"] == probe)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            if np.isnan(r["brier_reliability"]):
                continue
            rows.append({
                "label": f"{BASE_SHORT[tag]} | {PROBE_PRETTY[probe].split(' (')[0]}",
                "tag": tag, "probe": probe,
                "reliability": float(r["brier_reliability"]),
                "resolution": float(r["brier_resolution"]),
                "uncertainty": float(r["brier_uncertainty"]),
                "brier": float(r["brier"]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        ax.set_title("(no data)")
        return

    x = np.arange(len(df))
    # Plot reliability (worse = more miscal) and resolution (good = more useful spread)
    rel = df["reliability"].to_numpy()
    res = df["resolution"].to_numpy()
    bri = df["brier"].to_numpy()

    ax.bar(x - 0.18, rel, 0.35, color="#d62728", alpha=0.8, label="Reliability ↓ (miscal)")
    ax.bar(x + 0.18, res, 0.35, color="#2ca02c", alpha=0.8, label="Resolution ↑ (useful spread)")

    # annotate Brier on top
    for i, b in enumerate(bri):
        ax.text(x[i], max(rel[i], res[i]) + 0.005,
                f"Brier={b:.2f}", ha="center", fontsize=6.5, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"].tolist(), rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Brier component (per ID dataset)")
    ax.set_title("Brier decomposition (ID, lower-bin avg per probe)\n"
                 "Reliability = miscalibration penalty. Resolution = discrimination credit.",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def main() -> None:
    metrics = load_metrics()
    per_item = load_per_item()

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.32,
                  left=0.06, right=0.97, top=0.93, bottom=0.07)

    fig.suptitle(
        "Teacher-Free Probes — Extended Selective-Prediction Dashboard\n"
        "Beyond AUROC: Risk-Coverage curves, Coverage @ Risk, AUPRC, "
        "Sel-Acc, Calibration (Brier / ECE / Reliability)",
        fontsize=14, fontweight="bold", y=0.985,
    )

    # Row 1: RC curves
    panel_rc_curves(fig.add_subplot(gs[0, 0]), per_item, "id")
    panel_rc_curves(fig.add_subplot(gs[0, 1]), per_item, "hotpotqa")
    panel_rc_curves(fig.add_subplot(gs[0, 2]), per_item, "nq")

    # Row 2: operating points
    panel_cov_at_risk(fig.add_subplot(gs[1, 0]), metrics, max_risk=0.10)
    panel_auprc_heatmap(fig.add_subplot(gs[1, 1]), metrics)
    panel_sel_acc_at_50(fig.add_subplot(gs[1, 2]), metrics)

    # Row 3: calibration story
    panel_brier_auroc_scatter(fig.add_subplot(gs[2, 0]), metrics)
    panel_reliability_diagram(fig.add_subplot(gs[2, 1]), per_item,
                              tag="qwen72b", dataset="id")
    panel_brier_decomposition(fig.add_subplot(gs[2, 2]), metrics)

    out = THIS_DIR / "reports" / "dashboard_extended.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[done] dashboard_extended -> {out}")
    print(f"[size] {out.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
