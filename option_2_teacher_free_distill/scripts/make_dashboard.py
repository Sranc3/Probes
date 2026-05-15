"""Render a single dashboard image summarising the cross-base experiments.

Panels (3 x 3 grid):
  Row 1 (headline AUROC bar charts, one per dataset):
    (0,0) TriviaQA ID         -- 5 probes × 3 bases
    (0,1) HotpotQA OOD        -- 3 probes × 3 bases (DCP, SEPs-LR, SEPs-Ridge)
    (0,2) NQ-Open OOD         -- 3 probes × 3 bases
  Row 2 (layer sweep curves, x = normalised depth):
    (1,0) ID layer sweep      -- DCP-MLP, SEPs-LR, SEPs-Ridge for all 3 bases
    (1,1) HotpotQA layer sweep
    (1,2) NQ-Open layer sweep
  Row 3 (analytical):
    (2,0) OOD AUROC drop heatmap (probe x (base, OOD set))
    (2,1) Bootstrap forest plot: DCP vs SEPs-Ridge across 9 cells
    (2,2) Scaling curve: best-layer AUROC vs base model size

Output: ``reports/dashboard.png`` (high DPI, ~3000x2400 px).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR / "scripts"))
from run_all_models import BASES  # noqa: E402

BASE_TAGS = ["qwen7b", "llama3b", "qwen72b"]   # canonical order
BASE_PRETTY = {
    "qwen7b": "Qwen2.5-7B",
    "llama3b": "Llama-3.2-3B",
    "qwen72b": "Qwen2.5-72B",
}
BASE_PARAMS_B = {"qwen7b": 7.6, "llama3b": 3.2, "qwen72b": 72.7}  # for log-scale x
BASE_N_LAYERS = {b.tag: b.num_hidden_layers for b in BASES}

PROBE_PRETTY = {
    "DCP_mlp": "DCP-MLP (ours)",
    "SEPs_logreg": "SEPs-LR",
    "SEPs_ridge": "SEPs-Ridge (Kossen 2024)",
    "ARD_mlp": "ARD-MLP (ours)",
    "ARD_ridge": "ARD-Ridge (ours)",
}
PROBE_COLORS = {
    "DCP_mlp": "#d62728",        # red — ours headline
    "SEPs_logreg": "#1f77b4",    # blue — strong SEPs variant
    "SEPs_ridge": "#7f7f7f",     # gray — original SEPs
    "ARD_mlp": "#2ca02c",        # green — ours secondary
    "ARD_ridge": "#9467bd",      # purple — ours secondary
}
BASE_MARKERS = {"qwen7b": "o", "llama3b": "s", "qwen72b": "^"}
BASE_LINE = {"qwen7b": "-", "llama3b": "--", "qwen72b": ":"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_id_long(tag: str) -> pd.DataFrame:
    df = pd.read_csv(THIS_DIR / "results" / tag / "all_metrics_long.csv")
    df = df[df["regime"] == "cv_5fold"].copy()
    df["base"] = tag
    df["norm_depth"] = df["layer"] / BASE_N_LAYERS[tag]
    df["dataset"] = "id"
    return df


def load_ood_long(tag: str, dataset: str) -> pd.DataFrame:
    p = THIS_DIR / "results" / tag / f"ood_{dataset}_metrics_long.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["base"] = tag
    df["norm_depth"] = df["layer"] / BASE_N_LAYERS[tag]
    df["dataset"] = dataset
    return df


def load_id_best(tag: str) -> pd.DataFrame:
    df = pd.read_csv(THIS_DIR / "results" / tag / "best_per_probe.csv")
    df = df[df["regime"] == "cv_5fold"].copy()
    df["base"] = tag
    df["dataset"] = "id"
    return df


def load_ood_best(tag: str, dataset: str) -> pd.DataFrame:
    p = THIS_DIR / "results" / tag / f"ood_{dataset}_best_per_probe.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["base"] = tag
    df["dataset"] = dataset
    return df


def load_id_boot(tag: str) -> pd.DataFrame:
    df = pd.read_csv(THIS_DIR / "results" / tag / "bootstrap_pairs.csv")
    df = df[df["regime"] == "cv_5fold"].copy()
    df["base"] = tag
    df["dataset"] = "id"
    return df


def load_ood_boot(tag: str, dataset: str) -> pd.DataFrame:
    p = THIS_DIR / "results" / tag / f"ood_{dataset}_bootstrap_pairs.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["base"] = tag
    df["dataset"] = dataset
    return df


# ---------------------------------------------------------------------------
# Panel: bar charts (Row 1)
# ---------------------------------------------------------------------------


def panel_bar_auroc(ax, dataset: str, probes: list[str]) -> None:
    rows = []
    for tag in BASE_TAGS:
        df = load_id_best(tag) if dataset == "id" else load_ood_best(tag, dataset)
        if df.empty:
            continue
        for probe in probes:
            sub = df[df["probe"] == probe]
            if sub.empty:
                continue
            rows.append({
                "base": tag, "probe": probe,
                "auroc": float(sub.iloc[0]["auroc"]),
                "layer": int(sub.iloc[0]["layer"]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        ax.set_title(f"(no data: {dataset})")
        return
    n_p = len(probes)
    n_b = len(BASE_TAGS)
    width = 0.8 / n_b
    x = np.arange(n_p)
    for i, tag in enumerate(BASE_TAGS):
        sub = df[df["base"] == tag].set_index("probe").reindex(probes)
        vals = sub["auroc"].to_numpy()
        offsets = (i - (n_b - 1) / 2) * width
        bars = ax.bar(x + offsets, vals, width, label=BASE_PRETTY[tag], alpha=0.9)
        for j, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(x[j] + offsets, v + 0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7, rotation=0)
    title_map = {
        "id": "TriviaQA  (ID, n=1000, base_acc=0.566)",
        "hotpotqa": ("HotpotQA  (OOD multi-hop, with context, n=500)\n"
                     "base_acc: Qwen-7B 0.330 / Llama-3B 0.326 / Qwen-72B 0.450"),
        "nq": ("NQ-Open  (OOD single-hop, no context, n=500)\n"
               "base_acc: Qwen-7B 0.328 / Llama-3B 0.476 / Qwen-72B 0.506"),
    }
    ax.set_xticks(x)
    ax.set_xticklabels([PROBE_PRETTY[p].split(" (")[0] for p in probes], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_title(title_map.get(dataset, dataset), fontsize=10)
    ax.set_ylim(0.5, 0.92)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    if dataset == "id":
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


# ---------------------------------------------------------------------------
# Panel: layer sweep (Row 2)
# ---------------------------------------------------------------------------


def panel_layer_sweep(ax, dataset: str) -> None:
    """Cleaner version: emphasise DCP-MLP (red, thick) vs SEPs-Ridge (gray, dashed),
    with SEPs-LR shown as a thin blue reference. Linestyle encodes base model."""
    probes_main = ["DCP_mlp", "SEPs_ridge"]
    probes_ref = ["SEPs_logreg"]
    for tag in BASE_TAGS:
        df = load_id_long(tag) if dataset == "id" else load_ood_long(tag, dataset)
        if df.empty:
            continue
        for probe in probes_main:
            sub = df[df["probe"] == probe].sort_values("norm_depth")
            if sub.empty:
                continue
            ax.plot(
                sub["norm_depth"], sub["auroc"],
                color=PROBE_COLORS[probe],
                linestyle=BASE_LINE[tag],
                marker=BASE_MARKERS[tag], markersize=5,
                alpha=0.95, linewidth=1.8,
            )
        for probe in probes_ref:
            sub = df[df["probe"] == probe].sort_values("norm_depth")
            if sub.empty:
                continue
            ax.plot(
                sub["norm_depth"], sub["auroc"],
                color=PROBE_COLORS[probe],
                linestyle=BASE_LINE[tag],
                marker=BASE_MARKERS[tag], markersize=3,
                alpha=0.4, linewidth=0.9,
            )

    title_map = {"id": "ID layer sweep — TriviaQA (cv_5fold)",
                 "hotpotqa": "HotpotQA OOD layer sweep",
                 "nq": "NQ-Open OOD layer sweep"}
    ax.set_xlabel("Normalised depth = layer / N")
    ax.set_ylabel("AUROC")
    ax.set_title(title_map.get(dataset, dataset), fontsize=10)
    ax.grid(linestyle=":", alpha=0.4)
    ax.set_xlim(0, 1.05)
    if dataset == "id":
        ax.set_ylim(0.55, 0.86)
    else:
        ax.set_ylim(0.45, 0.82)
    ax.axvline(0.71, color="black", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.text(0.715, ax.get_ylim()[0] + 0.005, " ref 0.71",
            fontsize=7, color="black", alpha=0.6)

    # Two-axis legend (probe family + base) — drawn only on ID panel
    if dataset == "id":
        from matplotlib.lines import Line2D
        probe_handles = [
            Line2D([0], [0], color=PROBE_COLORS["DCP_mlp"], linewidth=2.5, label="DCP-MLP (ours)"),
            Line2D([0], [0], color=PROBE_COLORS["SEPs_ridge"], linewidth=2.5, label="SEPs-Ridge"),
            Line2D([0], [0], color=PROBE_COLORS["SEPs_logreg"], linewidth=1.5, alpha=0.5, label="SEPs-LR (ref)"),
        ]
        base_handles = [
            Line2D([0], [0], color="black", linestyle=BASE_LINE[t],
                   marker=BASE_MARKERS[t], markersize=5, label=BASE_PRETTY[t])
            for t in BASE_TAGS
        ]
        leg1 = ax.legend(handles=probe_handles, loc="upper left", fontsize=7,
                         framealpha=0.9, title="Probe", title_fontsize=7)
        ax.add_artist(leg1)
        ax.legend(handles=base_handles, loc="lower right", fontsize=7,
                  framealpha=0.9, title="Base", title_fontsize=7)


# ---------------------------------------------------------------------------
# Panel: OOD drop heatmap (3,0)
# ---------------------------------------------------------------------------


def panel_ood_drop_heatmap(ax) -> None:
    probes = ["DCP_mlp", "SEPs_logreg", "SEPs_ridge"]
    cols: list[str] = []
    matrix = np.full((len(probes), 0), np.nan)
    col_pairs: list[tuple[str, str]] = []
    for tag in BASE_TAGS:
        for ood in ("hotpotqa", "nq"):
            col_pairs.append((tag, ood))
    matrix = np.full((len(probes), len(col_pairs)), np.nan)
    for ci, (tag, ood) in enumerate(col_pairs):
        id_best = load_id_best(tag).set_index("probe")
        ood_best = load_ood_best(tag, ood).set_index("probe")
        for ri, probe in enumerate(probes):
            if probe in id_best.index and probe in ood_best.index:
                matrix[ri, ci] = float(ood_best.loc[probe, "auroc"]) - float(id_best.loc[probe, "auroc"])

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-0.22, vmax=0.0)
    ax.set_xticks(range(len(col_pairs)))
    ax.set_xticklabels([f"{BASE_PRETTY[t]}\n{ood.upper()}" for t, ood in col_pairs],
                       fontsize=7.5)
    ax.set_yticks(range(len(probes)))
    ax.set_yticklabels([PROBE_PRETTY[p].split(" (")[0] for p in probes], fontsize=8)
    ax.set_title("ID → OOD AUROC drop\n(green = small drop = robust)", fontsize=9)
    for ri in range(matrix.shape[0]):
        for ci in range(matrix.shape[1]):
            v = matrix[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:+.3f}", ha="center", va="center",
                        fontsize=8.5,
                        color="black" if v > -0.10 else "white")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="ΔAUROC")


# ---------------------------------------------------------------------------
# Panel: Bootstrap forest plot (3,1)
# ---------------------------------------------------------------------------


def panel_forest_dcp_vs_sepsridge(ax) -> None:
    rows: list[dict] = []
    for tag in BASE_TAGS:
        # ID
        df = load_id_boot(tag)
        if not df.empty:
            sub = df[df["probe_b"].str.startswith("SEPs_ridge")]
            if not sub.empty:
                r = sub.iloc[0]
                rows.append({
                    "label": f"{BASE_PRETTY[tag]} | TriviaQA ID",
                    "diff": float(r["mean_diff"]),
                    "lo": float(r["ci_low"]),
                    "hi": float(r["ci_high"]),
                    "p": float(r["p_value"]),
                })
        # OOD
        for ood in ("hotpotqa", "nq"):
            df = load_ood_boot(tag, ood)
            if df.empty:
                continue
            sub = df[df["probe_b"].str.startswith("SEPs_ridge")]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append({
                "label": f"{BASE_PRETTY[tag]} | {ood.upper()} OOD",
                "diff": float(r["mean_diff"]),
                "lo": float(r["ci_low"]),
                "hi": float(r["ci_high"]),
                "p": float(r["p_value"]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        ax.set_title("(no bootstrap data)")
        return

    y_pos = np.arange(len(df))[::-1]
    for yi, (_, r) in zip(y_pos, df.iterrows()):
        sig_color = "#d62728" if r["p"] < 0.05 else ("#ff7f0e" if r["p"] < 0.10 else "#7f7f7f")
        ax.errorbar(r["diff"], yi,
                    xerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]],
                    fmt="o", color=sig_color, ecolor=sig_color,
                    elinewidth=1.5, capsize=3, markersize=6)
        if r["p"] < 0.001:
            star = "***"
        elif r["p"] < 0.01:
            star = "**"
        elif r["p"] < 0.05:
            star = "*"
        elif r["p"] < 0.10:
            star = "·"
        else:
            star = ""
        ax.text(r["hi"] + 0.012, yi, f"{star}  p={r['p']:.3f}",
                va="center", fontsize=7.5, color=sig_color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"].tolist(), fontsize=8)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("ΔAUROC (DCP-MLP − SEPs-Ridge)")
    ax.set_title("DCP-MLP > SEPs-Ridge across 9 base × dataset cells\n"
                 "(red ● p<0.05; orange p<0.10; gray n.s.)", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_xlim(-0.04, 0.30)


# ---------------------------------------------------------------------------
# Panel: Scaling curve (3,2)
# ---------------------------------------------------------------------------


def panel_scaling(ax) -> None:
    probes = ["DCP_mlp", "SEPs_logreg", "SEPs_ridge", "ARD_mlp", "ARD_ridge"]
    rows = []
    for tag in BASE_TAGS:
        df = load_id_best(tag)
        for probe in probes:
            sub = df[df["probe"] == probe]
            if sub.empty:
                continue
            rows.append({"base": tag, "probe": probe, "auroc": float(sub.iloc[0]["auroc"])})
    df = pd.DataFrame(rows)
    # plot ordered by params
    order = sorted(BASE_TAGS, key=lambda t: BASE_PARAMS_B[t])
    x_params = [BASE_PARAMS_B[t] for t in order]
    for probe in probes:
        ys = []
        for tag in order:
            sub = df[(df["base"] == tag) & (df["probe"] == probe)]
            ys.append(float(sub.iloc[0]["auroc"]) if not sub.empty else np.nan)
        ax.plot(x_params, ys,
                marker="o", color=PROBE_COLORS[probe], linewidth=1.6, markersize=6,
                label=PROBE_PRETTY[probe].split(" (")[0])
    # Annotate only the 72B endpoint to avoid overlap
    for probe in probes:
        sub = df[(df["base"] == "qwen72b") & (df["probe"] == probe)]
        if not sub.empty:
            v = float(sub.iloc[0]["auroc"])
            ax.text(BASE_PARAMS_B["qwen72b"] * 1.08, v, f"{v:.3f}",
                    fontsize=7, color=PROBE_COLORS[probe], va="center")
    ax.set_xscale("log")
    ax.set_xticks(x_params)
    ax.set_xticklabels([BASE_PRETTY[t] for t in order], fontsize=8, rotation=10)
    ax.set_xlabel("Base model (log scale by params)")
    ax.set_ylabel("ID best-layer AUROC")
    ax.set_title("Scaling curve: K=1 probe ID-AUROC vs base model", fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7)


# ---------------------------------------------------------------------------
# Compose the dashboard
# ---------------------------------------------------------------------------


def main() -> None:
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.32, left=0.06, right=0.97, top=0.94, bottom=0.07)

    title = ("Teacher-Free K=1 Selective-Prediction Probes  —  "
             "Cross-base Dashboard\n"
             "3 base models × 3 datasets × 5 probes × 8–10 layers × 2000 paired bootstraps")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.985)

    # Row 1
    panel_bar_auroc(fig.add_subplot(gs[0, 0]), "id",
                    ["DCP_mlp", "SEPs_logreg", "SEPs_ridge", "ARD_mlp", "ARD_ridge"])
    panel_bar_auroc(fig.add_subplot(gs[0, 1]), "hotpotqa",
                    ["DCP_mlp", "SEPs_logreg", "SEPs_ridge"])
    panel_bar_auroc(fig.add_subplot(gs[0, 2]), "nq",
                    ["DCP_mlp", "SEPs_logreg", "SEPs_ridge"])

    # Row 2
    panel_layer_sweep(fig.add_subplot(gs[1, 0]), "id")
    panel_layer_sweep(fig.add_subplot(gs[1, 1]), "hotpotqa")
    panel_layer_sweep(fig.add_subplot(gs[1, 2]), "nq")

    # Row 3
    panel_ood_drop_heatmap(fig.add_subplot(gs[2, 0]))
    panel_forest_dcp_vs_sepsridge(fig.add_subplot(gs[2, 1]))
    panel_scaling(fig.add_subplot(gs[2, 2]))

    out_path = THIS_DIR / "reports" / "dashboard.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"[done] dashboard -> {out_path}")
    print(f"[size] {out_path.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
