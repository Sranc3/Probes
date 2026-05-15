"""Paired bootstrap comparison between any two probe predictions.

For each pair (A, B) we resample question_ids with replacement, recompute
AUROC for both predictors, and report the difference distribution.

Outputs:
- ``results/bootstrap_pairs.csv`` with columns
  ``regime, probe_a, probe_b, mean_diff, ci_low, ci_high, p_value``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR / "shared"))

from probe_utils import safe_auroc  # noqa: E402

DEFAULT_BASELINE_PRED_DIR = "/zhutingqi/song/Plan_opus_selective/results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--probe-predictions",
        default=str(THIS_DIR / "runs" / "probe_predictions.csv"),
    )
    p.add_argument(
        "--baseline-preds-dir",
        default=DEFAULT_BASELINE_PRED_DIR,
        help="Where Plan_opus_selective dumped its per-item predictions",
    )
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output",
        default=str(THIS_DIR / "results" / "bootstrap_pairs.csv"),
    )
    return p.parse_args()


def load_our_best(predictions: pd.DataFrame, regime: str) -> dict[str, pd.DataFrame]:
    """Pick the best layer per probe (by full-data AUROC) and return (qid, seed, score, label)."""
    sub = predictions[predictions["regime"] == regime].copy()
    chosen: dict[str, pd.DataFrame] = {}
    for probe, grp in sub.groupby("probe", sort=False):
        # AUROC over the full pool selects best layer (proxy)
        best_layer = -1
        best_auroc = -1.0
        for layer, lg in grp.groupby("layer", sort=False):
            au = safe_auroc(lg["y_true"].to_numpy(), lg["y_score"].to_numpy())
            if not np.isnan(au) and au > best_auroc:
                best_auroc = au
                best_layer = int(layer)
        if best_layer < 0:
            continue
        chosen[f"{probe}@L{best_layer}"] = (
            grp[grp["layer"] == best_layer][["question_id", "seed", "y_true", "y_score"]]
            .copy()
            .reset_index(drop=True)
        )
    return chosen


def load_baseline_predictions(baseline_dir: Path) -> dict[str, pd.DataFrame]:
    """Load Plan_opus_selective per-item predictions if available."""
    out: dict[str, pd.DataFrame] = {}
    csv = baseline_dir / "selective_per_item_predictions.csv"
    if not csv.exists():
        print(f"[warn] baseline per-item predictions not found at {csv}")
        return out
    df = pd.read_csv(csv)
    if "setting" in df.columns:
        df = df[df["setting"] == "sample0"]
    for predictor, grp in df.groupby("predictor", sort=False):
        out[f"baseline:{predictor}"] = (
            grp.rename(columns={"label": "y_true", "score": "y_score"})[
                ["question_id", "seed", "y_true", "y_score"]
            ]
            .copy()
            .reset_index(drop=True)
        )
    return out


def paired_bootstrap_auroc_diff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Resample question_ids with replacement and report diff = AUROC(A) - AUROC(B)."""
    merged = df_a.merge(df_b, on=["question_id", "seed"], suffixes=("_a", "_b"))
    if merged.empty:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if not (merged["y_true_a"] == merged["y_true_b"]).all():
        raise ValueError("Label mismatch between predictors - cannot pair-bootstrap")
    qids = merged["question_id"].unique()
    qid_to_rows: dict[str, np.ndarray] = {q: merged.index[merged["question_id"] == q].to_numpy() for q in qids}
    diffs = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample_qids = rng.choice(qids, size=len(qids), replace=True)
        idx = np.concatenate([qid_to_rows[q] for q in sample_qids])
        y_true = merged.loc[idx, "y_true_a"].to_numpy()
        au_a = safe_auroc(y_true, merged.loc[idx, "y_score_a"].to_numpy())
        au_b = safe_auroc(y_true, merged.loc[idx, "y_score_b"].to_numpy())
        diffs[b] = au_a - au_b
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(diffs))
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    pval = float(2 * min(np.mean(diffs > 0), np.mean(diffs < 0)))
    return mean, lo, hi, pval


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] our predictions: {args.probe_predictions}")
    preds = pd.read_csv(args.probe_predictions)
    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    for regime in ["cv_5fold", "cross_seed"]:
        ours = load_our_best(preds, regime)
        if not ours:
            continue
        # Compare DCP_mlp vs SEPs_ridge, SEPs_logreg, ARD_mlp, ARD_ridge
        keys = list(ours.keys())
        ref_key = next(k for k in keys if k.startswith("DCP_mlp"))
        for other in keys:
            if other == ref_key:
                continue
            mean, lo, hi, pval = paired_bootstrap_auroc_diff(
                ours[ref_key], ours[other], n_boot=args.n_boot, rng=rng
            )
            rows.append({
                "regime": regime,
                "probe_a": ref_key,
                "probe_b": other,
                "mean_diff": mean,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": pval,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path} ({len(df)} comparisons)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
