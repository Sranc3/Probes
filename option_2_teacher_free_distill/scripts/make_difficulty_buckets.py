"""Risk-Coverage curves segmented by question difficulty.

Question difficulty is defined from the K=8 anchor data (Plan_gpt55):

  - easy            : K=8 majority correct AND >=7 of 8 samples correct
                      (the model is consistently right)
  - hard_solvable   : K=8 majority correct BUT <7 of 8 samples correct
                      (right answer reachable but probe needs to spot it)
  - saturated_wrong : majority *wrong*  (the canonical "stable wrong basin")
  - mixed           : split (no majority)

For each bucket we plot a separate Risk-Coverage curve for our K=1 probes
on the ID dataset (TriviaQA). This reveals where the probe wins / loses.

Output:
  results/difficulty_buckets.csv    — per (probe, bucket) AUROC + count
  reports/dashboard_difficulty.png  — RC curves split by bucket
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR / "shared"))
from probe_utils import risk_coverage_curve, safe_auroc  # noqa: E402

ANCHOR_CSV = (
    "/zhutingqi/song/Plan_gpt55/cross_model_anchor/runs/"
    "run_20260513_012413_gptoss_anchor_triviaqa_full500_k4_tok256/"
    "qwen_candidate_anchor_rows_final_only.csv"
)


def assign_buckets(anchor: pd.DataFrame) -> pd.DataFrame:
    """Compute difficulty bucket per question_id from K=8 strict_correct counts."""
    g = anchor.groupby("question_id")["strict_correct"].agg(["sum", "count"]).reset_index()
    g.rename(columns={"sum": "n_correct", "count": "n_samples"}, inplace=True)
    g["frac_correct"] = g["n_correct"] / g["n_samples"]

    def label(row) -> str:
        f = row["frac_correct"]
        if f >= 7 / 8:
            return "easy"
        elif f > 0.5:
            return "hard_solvable"
        elif f == 0.5:
            return "mixed"
        else:
            return "saturated_wrong"
    g["bucket"] = g.apply(label, axis=1)
    return g


def load_probe_predictions(tag: str = "qwen7b") -> pd.DataFrame:
    df = pd.read_csv(THIS_DIR / "results" / tag / "per_item_predictions_id_best.csv")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PROBE_COLORS = {
    "DCP_mlp": "#d62728",
    "SEPs_logreg": "#1f77b4",
    "SEPs_ridge": "#7f7f7f",
    "ARD_mlp": "#2ca02c",
    "ARD_ridge": "#9467bd",
}
PROBE_PRETTY = {
    "DCP_mlp": "DCP-MLP (ours)",
    "SEPs_logreg": "SEPs-LR",
    "SEPs_ridge": "SEPs-Ridge",
    "ARD_mlp": "ARD-MLP",
    "ARD_ridge": "ARD-Ridge",
}
BUCKET_TITLES = {
    "easy": "Easy  (≥ 7/8 samples correct)",
    "hard_solvable": "Hard-but-solvable  (majority correct, < 7/8)",
    "mixed": "Mixed  (4/8 correct, the boundary)",
    "saturated_wrong": "Saturated wrong  (majority wrong, the hallucination zone)",
}


def panel_rc_curve(ax, predictions: pd.DataFrame, qid_in_bucket: set,
                   bucket_name: str, base_acc: float, n_in_bucket: int,
                   probes: list[str]) -> None:
    if n_in_bucket == 0:
        ax.set_title(f"{BUCKET_TITLES[bucket_name]}\n(no questions in bucket)",
                     fontsize=9)
        return
    sub = predictions[predictions["question_id"].isin(qid_in_bucket)]
    for probe in probes:
        psub = sub[sub["probe"] == probe]
        if psub.empty:
            continue
        y = psub["y_true"].to_numpy()
        s = psub["y_score"].to_numpy()
        if len(np.unique(y)) < 2:
            # all correct or all wrong → RC degenerate (constant risk)
            cov = np.linspace(1/len(y), 1.0, len(y))
            risk = np.ones_like(cov) * (1.0 - base_acc)
            ax.plot(cov, risk, color=PROBE_COLORS[probe], linewidth=1.2,
                    alpha=0.7)
            continue
        cov, risk = risk_coverage_curve(y, s)
        au = safe_auroc(y, s)
        ax.plot(cov, risk, color=PROBE_COLORS[probe], linewidth=1.6,
                label=f"{PROBE_PRETTY[probe]} (AUROC={au:.3f})")
    # Reference: baseline risk = 1 - base_acc
    ax.axhline(1.0 - base_acc, color="gray", linestyle=":", linewidth=0.8,
               label=f"baseline risk = {1.0 - base_acc:.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk = 1 - selective acc")
    ax.set_title(f"{BUCKET_TITLES[bucket_name]}\n"
                 f"n={n_in_bucket}, base_acc={base_acc:.2f}",
                 fontsize=9)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)


def main() -> None:
    print(f"[load] anchor CSV: {ANCHOR_CSV}")
    anchor = pd.read_csv(ANCHOR_CSV)
    bucket_df = assign_buckets(anchor)
    print("\n[bucket counts]")
    print(bucket_df["bucket"].value_counts())

    # Load DCP predictions for Qwen-7B (we have 5 probes)
    preds = load_probe_predictions("qwen7b")
    print(f"\n[load] {len(preds)} per-item predictions, "
          f"probes={sorted(preds['probe'].unique())}")

    # Build summary table per (probe, bucket)
    summary_rows: list[dict] = []
    for bucket_name, bg in bucket_df.groupby("bucket"):
        qids = set(bg["question_id"].tolist())
        n = len(qids)
        base_acc_in_bucket = float(bg["frac_correct"].mean())
        for probe, psub in preds[preds["question_id"].isin(qids)].groupby("probe"):
            y = psub["y_true"].to_numpy()
            s = psub["y_score"].to_numpy()
            if len(np.unique(y)) < 2:
                au = float("nan")
            else:
                au = safe_auroc(y, s)
            summary_rows.append({
                "bucket": bucket_name, "n_questions": n,
                "base_acc_within_bucket": base_acc_in_bucket,
                "probe": probe, "auroc": au,
            })
    summary = pd.DataFrame(summary_rows)
    out_csv = THIS_DIR / "results" / "difficulty_buckets.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\n[done] {out_csv}")
    print(summary.pivot(index="probe", columns="bucket", values="auroc"))

    # Compose dashboard: 2x2 RC curves, one per bucket
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Risk-Coverage Curves by Question Difficulty (Qwen-7B, TriviaQA ID)\n"
        "Where does each K=1 probe win? — Easy/Hard-solvable/Mixed/Saturated-wrong buckets",
        fontsize=13, fontweight="bold", y=0.995,
    )
    bucket_order = ["easy", "hard_solvable", "mixed", "saturated_wrong"]
    probes = ["DCP_mlp", "SEPs_logreg", "SEPs_ridge", "ARD_mlp", "ARD_ridge"]
    for ax, bucket_name in zip(axes.flatten(), bucket_order):
        bg = bucket_df[bucket_df["bucket"] == bucket_name]
        qids = set(bg["question_id"].tolist())
        n = len(qids)
        if n == 0:
            ax.set_title(f"{BUCKET_TITLES[bucket_name]}\n(no questions)")
            ax.axis("off")
            continue
        base_acc = float(bg["frac_correct"].mean())
        panel_rc_curve(ax, preds, qids, bucket_name, base_acc, n, probes)

    out = THIS_DIR / "reports" / "dashboard_difficulty.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[done] dashboard_difficulty -> {out}")


if __name__ == "__main__":
    main()
