"""Render a markdown comparison table:

- This work's probes (best layer per probe per regime) on K=1 hidden state
- Plan_opus_selective baselines (single-feature heuristics + logreg over K=8
  features) re-formatted to the same columns
- A 'cost tier' column making the deployment cost gap explicit:
    * K=1 prompt-only        : SEPs, DCP, ARD (ours and SEPs baseline)
    * K=8 self-introspection : logreg:self
    * K=8 + teacher API call : logreg:teacher, logreg:all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = "/zhutingqi/song/Plan_opus_selective/results/selective_metrics_long.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--probe-metrics",
        default=str(THIS_DIR / "results" / "all_metrics_long.csv"),
    )
    p.add_argument(
        "--baselines",
        default=DEFAULT_BASELINES,
    )
    p.add_argument(
        "--output",
        default=str(THIS_DIR / "results" / "comparison_table.md"),
    )
    return p.parse_args()


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return "—"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


COST_TIER = {
    # K=1 prompt only (no answer generation, no teacher)
    "SEPs_ridge": "K=1 prompt only",
    "SEPs_logreg": "K=1 prompt only",
    "DCP_mlp": "K=1 prompt only",
    "ARD_ridge": "K=1 prompt only",
    "ARD_mlp": "K=1 prompt only",
    # K=8 sampling required to produce these features
    "single_feature": "K=8 features",
    "logreg:self": "K=8 self only (no teacher)",
    "logreg:teacher": "K=8 + teacher API call",
    "logreg:all": "K=8 + teacher API call",
    "mlp:all": "K=8 + teacher API call",
}


def assign_cost_tier(probe: str) -> str:
    if probe in COST_TIER:
        return COST_TIER[probe]
    if probe.startswith("logreg:"):
        return COST_TIER.get(probe, "K=8 features")
    if probe.startswith("mlp:"):
        return "K=8 + teacher API call"
    return "single feature (K=8)"


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] our probes: {args.probe_metrics}")
    probes = pd.read_csv(args.probe_metrics)
    # Keep best layer per (probe, regime) by AUROC
    best_rows: list[dict] = []
    for (probe, regime), grp in probes.groupby(["probe", "regime"], sort=False):
        best = grp.sort_values("auroc", ascending=False).iloc[0]
        d = best.to_dict()
        d["best_layer"] = int(best["layer"])
        d["cost_tier"] = assign_cost_tier(probe)
        best_rows.append(d)
    best_df = pd.DataFrame(best_rows)

    baseline_rows: list[dict] = []
    if args.baselines and Path(args.baselines).exists():
        print(f"[load] baselines: {args.baselines}")
        base = pd.read_csv(args.baselines)
        # Only keep sample0 rows for fair comparison (our hidden states are sample0-aligned)
        if "setting" in base.columns:
            base = base[base["setting"] == "sample0"]
        for _, row in base.iterrows():
            d = {
                "probe": row.get("predictor", row.get("probe", "?")),
                "regime": row.get("regime", "cv_5fold"),
                "n": row.get("n", float("nan")),
                "base_acc": row.get("base_acc", float("nan")),
                "auroc": row.get("auroc", float("nan")),
                "aurc": row.get("aurc", float("nan")),
                "sel_acc@0.25": row.get("sel_acc@0.25", float("nan")),
                "sel_acc@0.50": row.get("sel_acc@0.50", float("nan")),
                "sel_acc@0.75": row.get("sel_acc@0.75", float("nan")),
                "brier": row.get("brier", float("nan")),
                "ece": row.get("ece", float("nan")),
                "best_layer": "—",
                "cost_tier": assign_cost_tier(str(row.get("predictor", "?"))),
            }
            baseline_rows.append(d)
    else:
        print(f"[warn] baselines path missing: {args.baselines}")

    base_df = pd.DataFrame(baseline_rows)
    full = pd.concat([best_df, base_df], ignore_index=True, sort=False)

    cost_order = {
        "K=1 prompt only": 0,
        "single feature (K=8)": 1,
        "K=8 features": 1,
        "K=8 self only (no teacher)": 2,
        "K=8 + teacher API call": 3,
    }
    full["cost_rank"] = full["cost_tier"].map(lambda t: cost_order.get(t, 99))
    full = full.sort_values(["regime", "cost_rank", "auroc"], ascending=[True, True, False]).reset_index(drop=True)

    columns = [
        "regime", "cost_tier", "probe", "best_layer",
        "n", "base_acc", "auroc", "aurc",
        "sel_acc@0.25", "sel_acc@0.50", "sel_acc@0.75",
        "brier", "ece",
    ]
    full = full[columns + [c for c in full.columns if c not in columns and c != "cost_rank"]]

    # Render markdown
    md_lines: list[str] = []
    md_lines.append("# Teacher-Free Distillation: Head-to-Head Comparison\n")
    md_lines.append(
        "Our probes operate on a **single forward pass** of Qwen on the prompt "
        "(no generation, no teacher). The Plan_opus_selective baselines instead "
        "consume features computed from K=8 stochastic samples (and teacher "
        "calls for the *teacher*/*all* variants).\n"
    )
    for regime, grp in full.groupby("regime", sort=False):
        md_lines.append(f"\n## Regime: {regime}\n")
        header = "| " + " | ".join(columns[1:]) + " |"
        sep = "| " + " | ".join(["---"] * (len(columns) - 1)) + " |"
        md_lines.append(header)
        md_lines.append(sep)
        for _, row in grp.iterrows():
            md_lines.append("| " + " | ".join(fmt(row[c]) for c in columns[1:]) + " |")

    out_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
