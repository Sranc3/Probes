"""Evaluate probe predictions and assemble selective-prediction metrics.

Inputs:
- ``runs/probe_predictions.csv`` from train_probes.py
- (optional) Plan_opus_selective single-feature & logreg baselines for a
  side-by-side comparison column.

Outputs:
- ``results/all_metrics_long.csv``
- ``results/best_per_probe.csv``  (best layer per probe per regime, selected
  by AUROC on cv_5fold)
- ``results/per_item_predictions_best.csv`` (predictions of the best probe
  for downstream routing reuse)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR / "shared"))

from probe_utils import summarize_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--predictions",
        default=str(THIS_DIR / "runs" / "probe_predictions.csv"),
    )
    p.add_argument(
        "--out-dir",
        default=str(THIS_DIR / "results"),
    )
    p.add_argument(
        "--baselines-csv",
        default=None,
        help="Optional Plan_opus_selective metrics CSV to merge as comparison",
    )
    return p.parse_args()


CALIBRATED_PROBES = {"SEPs_logreg", "DCP_mlp", "ARD_ridge", "ARD_mlp"}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] predictions: {args.predictions}")
    df = pd.read_csv(args.predictions)
    print(f"[load] {len(df)} rows  probes={sorted(df['probe'].unique())} layers={sorted(df['layer'].unique())}")

    rows: list[dict[str, Any]] = []
    for (probe, layer, regime), grp in df.groupby(["probe", "layer", "regime"], sort=False):
        if grp["y_score"].isna().any():
            print(f"[warn] NaN scores in {probe}/{layer}/{regime} - skipping")
            continue
        y = grp["y_true"].to_numpy()
        s = grp["y_score"].to_numpy()
        calibrated = probe in CALIBRATED_PROBES
        metrics = summarize_metrics(y, s, calibrated=calibrated)
        rows.append({
            "probe": probe,
            "layer": int(layer),
            "regime": regime,
            "n": int(len(grp)),
            "base_acc": float(np.mean(y)),
            **metrics,
        })

    metrics_df = pd.DataFrame(rows)
    metrics_long = out_dir / "all_metrics_long.csv"
    metrics_df.to_csv(metrics_long, index=False)
    print(f"[done] wrote {metrics_long} ({len(metrics_df)} rows)")

    # Pick best layer per (probe, regime) by AUROC
    best_rows: list[dict[str, Any]] = []
    for (probe, regime), grp in metrics_df.groupby(["probe", "regime"], sort=False):
        best = grp.sort_values("auroc", ascending=False).iloc[0]
        best_rows.append(best.to_dict())
    best_df = pd.DataFrame(best_rows)
    best_csv = out_dir / "best_per_probe.csv"
    best_df.to_csv(best_csv, index=False)
    print(f"[done] wrote {best_csv}")

    # Dump per-item predictions for the best probe (cv_5fold) so routing can re-use
    cv_best = (
        metrics_df[metrics_df["regime"] == "cv_5fold"]
        .sort_values("auroc", ascending=False)
        .iloc[0]
    )
    print(f"[best cv_5fold] {cv_best['probe']} layer={cv_best['layer']} AUROC={cv_best['auroc']:.4f}")
    cv_best_pred = df[
        (df["probe"] == cv_best["probe"]) & (df["layer"] == cv_best["layer"]) & (df["regime"] == "cv_5fold")
    ]
    cv_best_csv = out_dir / "per_item_predictions_best_cv.csv"
    cv_best_pred.to_csv(cv_best_csv, index=False)
    print(f"[done] wrote {cv_best_csv}")


if __name__ == "__main__":
    main()
