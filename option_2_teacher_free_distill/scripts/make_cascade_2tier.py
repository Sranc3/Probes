"""2-tier multi-agent cascade analysis (Qwen-7B → GPT-OSS-120B).

Simplified, controllable cascade with real measured numbers:

  Tier 0  Qwen2.5-7B K=1 + DCP-MLP probe
          - Real wall-clock: 324 ms (greedy gen on H200)
          - Sample0 accuracy: 0.566 on TriviaQA
          - DCP-MLP confidence in [0, 1]

  Tier 1  GPT-OSS-120B K=1 greedy (the original anchor teacher)
          - Real wall-clock: 4562 ms (median, on the anchor run hardware)
          - Greedy accuracy: 0.764 on TriviaQA

Routing rule: at Tier 0, if probe_score >= threshold commit Qwen answer;
otherwise pay Tier 1 cost, commit teacher answer.

We compare *different probe routing signals* (all single-forward, K=1
prompt-only):
  - DCP-MLP   (ours, the headline probe)
  - SEPs-LR   (the strong SEPs variant)
  - SEPs-Ridge (Kossen 2024 main)

…against three baselines:
  - Always-Qwen-7B  (cost=1.0, acc=0.566)
  - Always-Teacher  (cost=14.1, acc=0.764)
  - Oracle cascade  (cheat: escalate iff Qwen-7B wrong)

Outputs:
  results/cascade_2tier_results.csv
  reports/dashboard_cascade_2tier.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

THIS_DIR = Path(__file__).resolve().parents[1]

ANCHOR_DIR = Path("/zhutingqi/song/Plan_gpt55/cross_model_anchor/runs/"
                  "run_20260513_012413_gptoss_anchor_triviaqa_full500_k4_tok256")

# Cost units (seconds-of-Qwen-7B-K=1, real measured on H200 fp16)
QWEN_LATENCY_MS = 324       # measured in measure_latency.py
TEACHER_LATENCY_MS = 4562   # median over 500 GPT-OSS-120B greedy generations
COST_QWEN_K1 = 1.0
COST_TEACHER = TEACHER_LATENCY_MS / QWEN_LATENCY_MS   # 14.08

# Probe candidates
PROBE_NAMES = ["DCP_mlp", "SEPs_logreg", "SEPs_ridge"]
PROBE_PRETTY = {
    "DCP_mlp": "DCP-MLP (ours)",
    "SEPs_logreg": "SEPs-LR",
    "SEPs_ridge": "SEPs-Ridge (Kossen 2024)",
}
PROBE_COLORS = {
    "DCP_mlp": "#d62728",
    "SEPs_logreg": "#1f77b4",
    "SEPs_ridge": "#7f7f7f",
}
PROBE_MARKERS = {
    "DCP_mlp": "o", "SEPs_logreg": "s", "SEPs_ridge": "D",
}


# ---------------------------------------------------------------------------
# Build aligned per-question table (one row per question_id)
# ---------------------------------------------------------------------------


def build_aligned_table() -> pd.DataFrame:
    # 1) Qwen-7B sample0 strict_correct (averaged across seeds)
    qwen = pd.read_csv(ANCHOR_DIR / "qwen_candidate_anchor_rows_final_only.csv")
    qwen0 = qwen[qwen["sample_index"] == 0]
    qwen_per_q = qwen0.groupby("question_id").agg(
        qwen_sample0_correct=("strict_correct", "mean"),  # avg over 2 seeds
    ).reset_index()

    # 2) Teacher greedy strict_correct + real latency (per question)
    teacher_records = []
    with open(ANCHOR_DIR / "teacher_generations.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            teacher_records.append({
                "question_id": rec["question_id"],
                "decode_mode": rec.get("teacher_decode_mode", ""),
                "teacher_strict_correct": rec.get("teacher_strict_correct", 0.0),
                "latency_ms": rec.get("latency_ms", 0.0),
                "completion_tokens": rec.get("completion_tokens", 0),
            })
    teacher_df = pd.DataFrame(teacher_records)
    greedy = teacher_df[teacher_df["decode_mode"] == "greedy"]
    teacher_per_q = greedy.groupby("question_id").agg(
        teacher_greedy_correct=("teacher_strict_correct", "max"),
        teacher_latency_ms=("latency_ms", "median"),
    ).reset_index()

    # 3) DCP-MLP / SEPs-LR / SEPs-Ridge confidences for Qwen-7B
    pred_path = THIS_DIR / "results" / "qwen7b" / "per_item_predictions_id_best.csv"
    pred = pd.read_csv(pred_path)

    table = qwen_per_q.merge(teacher_per_q, on="question_id", how="inner")
    for probe in PROBE_NAMES:
        psub = pred[pred["probe"] == probe][["question_id", "y_score"]]
        psub = psub.rename(columns={"y_score": f"score_{probe}"})
        table = table.merge(psub, on="question_id", how="left")

    return table


# ---------------------------------------------------------------------------
# Cascade simulation
# ---------------------------------------------------------------------------


def simulate_cascade(
    table: pd.DataFrame, probe: str, escalation_rate: float,
) -> tuple[float, float, float, float]:
    """Return (avg_cost, avg_accuracy, fraction_escalated, threshold_used).

    We use *percentile-based* thresholding so all probes are comparable
    despite having different raw score ranges (DCP/SEPs-LR in [0,1],
    SEPs-Ridge in approximately [-2.24, 0.82]).

    The cascade escalates the LOWEST-scoring `escalation_rate` fraction of
    questions to the teacher and commits the rest with Qwen-7B.
    """
    score = table[f"score_{probe}"].to_numpy()
    qwen_correct = table["qwen_sample0_correct"].to_numpy()
    teacher_correct = table["teacher_greedy_correct"].to_numpy()
    if escalation_rate <= 0:
        threshold = float(np.min(score)) - 1.0
        commit_qwen = np.ones(len(score), dtype=bool)
    elif escalation_rate >= 1:
        threshold = float(np.max(score)) + 1.0
        commit_qwen = np.zeros(len(score), dtype=bool)
    else:
        threshold = float(np.quantile(score, escalation_rate))
        commit_qwen = score > threshold
    correct = np.where(commit_qwen, qwen_correct, teacher_correct)
    cost = np.where(commit_qwen, COST_QWEN_K1, COST_QWEN_K1 + COST_TEACHER)
    actual_frac = float((~commit_qwen).mean())
    return float(cost.mean()), float(correct.mean()), actual_frac, threshold


def simulate_random_cascade(
    table: pd.DataFrame, p_keep: float, n_trials: int = 100, seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    qwen_correct = table["qwen_sample0_correct"].to_numpy()
    teacher_correct = table["teacher_greedy_correct"].to_numpy()
    avg_costs, avg_accs = [], []
    for _ in range(n_trials):
        commit_qwen = rng.random(len(table)) < p_keep
        correct = np.where(commit_qwen, qwen_correct, teacher_correct)
        cost = np.where(commit_qwen, COST_QWEN_K1, COST_QWEN_K1 + COST_TEACHER)
        avg_costs.append(cost.mean())
        avg_accs.append(correct.mean())
    return float(np.mean(avg_costs)), float(np.mean(avg_accs))


def simulate_oracle_cascade(table: pd.DataFrame) -> tuple[float, float]:
    qwen_correct = table["qwen_sample0_correct"].to_numpy()
    teacher_correct = table["teacher_greedy_correct"].to_numpy()
    # Escalate iff Qwen wrong; if both wrong, take teacher (still wrong).
    commit_qwen = qwen_correct > 0.5
    correct = np.where(commit_qwen, qwen_correct, teacher_correct)
    cost = np.where(commit_qwen, COST_QWEN_K1, COST_QWEN_K1 + COST_TEACHER)
    return float(cost.mean()), float(correct.mean())


# ---------------------------------------------------------------------------
# Sweep + assemble
# ---------------------------------------------------------------------------


def sweep_all_strategies(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    # Always-Qwen
    qa = float(table["qwen_sample0_correct"].mean())
    rows.append({"strategy": "always_qwen", "probe": None, "threshold": np.nan,
                 "cost": COST_QWEN_K1, "accuracy": qa, "frac_escalated": 0.0})
    # Always-Teacher (still pay Qwen K=0 because we go straight to teacher)
    ta = float(table["teacher_greedy_correct"].mean())
    rows.append({"strategy": "always_teacher", "probe": None, "threshold": np.nan,
                 "cost": COST_TEACHER, "accuracy": ta, "frac_escalated": 1.0})
    # Probe cascades — sweep escalation rate (percentile-based, fair across probes)
    rates = np.linspace(0.0, 1.0, 101)
    for probe in PROBE_NAMES:
        for r in rates:
            cost, acc, actual_frac, th_used = simulate_cascade(table, probe, r)
            rows.append({"strategy": "probe_cascade", "probe": probe,
                         "threshold": th_used, "cost": cost, "accuracy": acc,
                         "frac_escalated": actual_frac,
                         "target_escalation_rate": r})
    # Random cascade
    for p in np.linspace(0.0, 1.0, 41):
        cost, acc = simulate_random_cascade(table, p, n_trials=50)
        rows.append({"strategy": "random_cascade", "probe": None,
                     "threshold": p, "cost": cost, "accuracy": acc,
                     "frac_escalated": 1.0 - p})
    # Oracle cascade
    cost, acc = simulate_oracle_cascade(table)
    rows.append({"strategy": "oracle_cascade", "probe": None, "threshold": np.nan,
                 "cost": cost, "accuracy": acc,
                 "frac_escalated": float((table["qwen_sample0_correct"] < 0.5).mean())})
    return pd.DataFrame(rows)


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values("cost").reset_index(drop=True)
    keep = []
    best_acc = -1.0
    for _, r in df_sorted.iterrows():
        if r["accuracy"] > best_acc:
            keep.append(True)
            best_acc = r["accuracy"]
        else:
            keep.append(False)
    return df_sorted[keep]


def cost_savings_at_acc_target(
    df: pd.DataFrame, targets: list[float],
) -> pd.DataFrame:
    rows = []
    for tgt in targets:
        always_t_cost = float(df[df["strategy"] == "always_teacher"]["cost"].iloc[0])
        always_q = df[df["strategy"] == "always_qwen"].iloc[0]
        always_q_meets = float(always_q["accuracy"]) >= tgt
        for probe in PROBE_NAMES:
            sub = df[(df["strategy"] == "probe_cascade") & (df["probe"] == probe)]
            meet = sub[sub["accuracy"] >= tgt].sort_values("cost")
            if meet.empty:
                cheapest = "—"
                savings_vs_teacher = "—"
            else:
                r = meet.iloc[0]
                cheapest = f"{r['cost']:.2f} (esc={r['frac_escalated']:.0%})"
                savings_vs_teacher = f"{(always_t_cost - r['cost']) / always_t_cost * 100:+.0f}%"
            rows.append({"target_acc": tgt,
                         "probe": PROBE_PRETTY[probe],
                         "cheapest_cost": cheapest,
                         "savings_vs_always_teacher": savings_vs_teacher,
                         "always_qwen_meets": "✓" if always_q_meets else "✗"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def panel_pareto(ax, df: pd.DataFrame) -> None:
    # Always-baselines
    aq = df[df["strategy"] == "always_qwen"].iloc[0]
    at = df[df["strategy"] == "always_teacher"].iloc[0]
    ax.scatter(aq["cost"], aq["accuracy"], s=180, marker="s", color="#3182bd",
               edgecolors="black", linewidths=0.7, zorder=4)
    ax.annotate("Always Qwen-7B (K=1)\n(no cascade)",
                (aq["cost"], aq["accuracy"]),
                textcoords="offset points", xytext=(8, -20), fontsize=8.5,
                color="#3182bd")
    ax.scatter(at["cost"], at["accuracy"], s=200, marker="*", color="#000000",
               edgecolors="black", linewidths=0.7, zorder=4)
    ax.annotate("Always GPT-OSS-120B\n(no cascade)",
                (at["cost"], at["accuracy"]),
                textcoords="offset points", xytext=(8, -3), fontsize=8.5)

    # Random cascade cloud (light grey)
    rc = df[df["strategy"] == "random_cascade"].sort_values("cost")
    ax.plot(rc["cost"], rc["accuracy"], color="#bcbcbc", linewidth=1.0,
            marker="x", markersize=4, alpha=0.6,
            label="Random cascade (uniform escalation)", zorder=2)

    # Probe cascades
    for probe in PROBE_NAMES:
        sub = df[(df["strategy"] == "probe_cascade") & (df["probe"] == probe)]
        sub = sub.sort_values("cost")
        ax.plot(sub["cost"], sub["accuracy"],
                color=PROBE_COLORS[probe], marker=PROBE_MARKERS[probe],
                markersize=4, linewidth=1.8, alpha=0.95,
                label=f"{PROBE_PRETTY[probe]} cascade", zorder=5)

    # Oracle cascade
    oc = df[df["strategy"] == "oracle_cascade"]
    if not oc.empty:
        r = oc.iloc[0]
        ax.scatter(r["cost"], r["accuracy"], s=220, marker="v",
                   color="#2ca02c", edgecolors="black", linewidths=0.8, zorder=6)
        ax.annotate(f"Oracle cascade\n(escalate iff Qwen wrong)\ncost={r['cost']:.2f}, acc={r['accuracy']:.2f}",
                    (r["cost"], r["accuracy"]),
                    textcoords="offset points", xytext=(-10, -45), fontsize=8.5,
                    color="#2ca02c")

    ax.set_xlabel("Avg cost per question  (Qwen-7B forward units, latency-based)")
    ax.set_ylabel("Avg accuracy")
    ax.set_xscale("log")
    ax.set_title(
        "2-Tier Cascade — Qwen-7B → GPT-OSS-120B on TriviaQA (n=500)\n"
        "Cost = real measured wall-clock (Qwen-7B 324 ms vs GPT-OSS 4562 ms = 14.1×)",
        fontsize=10,
    )
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.set_ylim(0.55, 0.85)


def panel_acc_vs_escalation(ax, df: pd.DataFrame, table: pd.DataFrame) -> None:
    """Accuracy as a function of escalation rate (deployment-relevant view)."""
    for probe in PROBE_NAMES:
        sub = df[(df["strategy"] == "probe_cascade") & (df["probe"] == probe)]
        sub = sub.sort_values("frac_escalated")
        ax.plot(sub["frac_escalated"] * 100, sub["accuracy"],
                color=PROBE_COLORS[probe], marker=PROBE_MARKERS[probe],
                markersize=3, linewidth=1.8,
                label=f"{PROBE_PRETTY[probe]}")
    aq = float(table["qwen_sample0_correct"].mean())
    at = float(table["teacher_greedy_correct"].mean())
    ax.axhline(aq, color="#3182bd", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(at, color="#000000", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(2, aq + 0.005, f"Always Qwen: {aq:.3f}", fontsize=8, color="#3182bd")
    ax.text(2, at - 0.012, f"Always Teacher: {at:.3f}", fontsize=8)
    ax.set_xlabel("% questions escalated to teacher")
    ax.set_ylabel("Avg accuracy")
    ax.set_title("Accuracy vs escalation budget\n"
                 "DCP-MLP gives the steepest gain at low escalation rates",
                 fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.55, 0.82)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=8.5, loc="lower right")


def panel_savings_table(ax, savings: pd.DataFrame, table: pd.DataFrame) -> None:
    ax.axis("off")
    cell = []
    for _, r in savings.iterrows():
        cell.append([
            f"≥ {r['target_acc']:.2f}",
            r["probe"], r["cheapest_cost"], r["savings_vs_always_teacher"],
            r["always_qwen_meets"],
        ])
    col_labels = ["target_acc", "probe", "min cost (th, %escalate)",
                  "saves vs always-teacher", "qwen alone enough?"]
    tbl = ax.table(cellText=cell, colLabels=col_labels,
                   loc="upper center", cellLoc="center", colLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)
    tbl.scale(1, 1.55)
    for j, _ in enumerate(col_labels):
        tbl[0, j].set_facecolor("#cccccc")
        tbl[0, j].set_text_props(weight="bold")
    # color savings cell
    for i, r in enumerate(cell, start=1):
        if r[3] != "—" and r[3].endswith("%"):
            try:
                v = int(r[3].rstrip("%"))
                color = "#2ca02c" if v < 0 else ("#d62728" if v > 0 else "black")
                tbl[i, 3].set_text_props(color=color, weight="bold")
            except ValueError:
                pass
    ax.set_title(
        "Cost savings at fixed accuracy targets\n"
        f"(always_qwen acc={table['qwen_sample0_correct'].mean():.3f}, "
        f"always_teacher acc={table['teacher_greedy_correct'].mean():.3f}, "
        f"teacher cost={COST_TEACHER:.2f})",
        fontsize=9.5, y=0.98,
    )


def panel_routing_at_threshold(ax, table: pd.DataFrame, probe: str = "DCP_mlp",
                               threshold: float = 0.5) -> None:
    """Quadrant chart: Qwen-correct × DCP-confident, with sizes."""
    qc = table["qwen_sample0_correct"].to_numpy()
    sc = table[f"score_{probe}"].to_numpy()
    confident = sc >= threshold
    correct = qc > 0.5
    quads = {
        "TT": int(((confident) & (correct)).sum()),    # confident & correct: keep Qwen
        "TF": int(((confident) & (~correct)).sum()),   # confident but wrong (BAD)
        "FT": int(((~confident) & (correct)).sum()),   # unsure but correct (escalation waste)
        "FF": int(((~confident) & (~correct)).sum()),  # unsure & wrong (escalate, hopefully teacher saves)
    }
    qwen_acc_when_committed = quads["TT"] / max(quads["TT"] + quads["TF"], 1)
    n = len(table)
    teacher_corr_when_escalated = (
        table[f"score_{probe}"] < threshold) & (table["teacher_greedy_correct"] > 0.5)
    n_escalated_corr = int(teacher_corr_when_escalated.sum())
    n_escalated = int((table[f"score_{probe}"] < threshold).sum())
    teacher_acc_on_escalated = n_escalated_corr / max(n_escalated, 1)

    cats = ["Confident +\nCorrect (commit ✓)",
            "Confident +\nWRONG (commit ✗)",
            "Unsure +\nCorrect (escalate, waste)",
            "Unsure +\nWrong (escalate → teacher)"]
    counts = [quads["TT"], quads["TF"], quads["FT"], quads["FF"]]
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#7f7f7f"]
    ax.bar(range(4), counts, color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.5)
    for i, c in enumerate(counts):
        ax.text(i, c + 5, f"{c}\n({c/n*100:.0f}%)", ha="center", fontsize=8.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("Question count")
    ax.set_title(
        f"Routing breakdown @ {PROBE_PRETTY[probe]} threshold = {threshold:.2f}\n"
        f"Qwen acc when committed: {qwen_acc_when_committed:.2%}  |  "
        f"Teacher acc on escalated: {teacher_acc_on_escalated:.2%}",
        fontsize=9.5,
    )
    ax.set_ylim(0, max(counts) * 1.25)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = THIS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building aligned table...")
    table = build_aligned_table()
    print(f"   n={len(table)} questions aligned across Qwen-7B / GPT-OSS / 3 probes")
    print(f"   Qwen-7B sample0 acc: {table['qwen_sample0_correct'].mean():.3f}")
    print(f"   GPT-OSS greedy acc:  {table['teacher_greedy_correct'].mean():.3f}")
    print(f"   Teacher latency med: {table['teacher_latency_ms'].median():.0f} ms")
    print(f"   Cost ratio: {COST_TEACHER:.2f}")

    print("\n[2/4] Sweeping all strategies (3 probes × 101 thresholds + baselines + oracle)...")
    df = sweep_all_strategies(table)
    df.to_csv(out_dir / "cascade_2tier_results.csv", index=False)
    print(f"   wrote {len(df)} rows -> results/cascade_2tier_results.csv")

    print("\n[3/4] Computing cost savings at accuracy targets...")
    targets = [0.60, 0.65, 0.70, 0.72, 0.74, 0.76]
    savings = cost_savings_at_acc_target(df, targets)
    savings.to_csv(out_dir / "cascade_2tier_savings.csv", index=False)
    print(savings.to_string(index=False))

    print("\n[4/4] Composing dashboard...")
    fig = plt.figure(figsize=(20, 13))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30,
                  left=0.06, right=0.97, top=0.92, bottom=0.06,
                  height_ratios=[1.2, 1.0])
    fig.suptitle(
        "2-Tier Multi-Agent Cascade — Qwen2.5-7B (student) → GPT-OSS-120B (teacher)\n"
        "TriviaQA ID, n=500, real measured latencies on H200; routing via 3 probes",
        fontsize=13, fontweight="bold", y=0.985,
    )
    panel_pareto(fig.add_subplot(gs[0, :]), df)
    panel_acc_vs_escalation(fig.add_subplot(gs[1, 0]), df, table)
    panel_routing_at_threshold(fig.add_subplot(gs[1, 1]), table,
                               probe="DCP_mlp", threshold=0.5)
    out = THIS_DIR / "reports" / "dashboard_cascade_2tier.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\n[done] dashboard_cascade_2tier -> {out}")
    print(f"[size] {out.stat().st_size / 1e6:.2f} MB")

    # Print headline numbers
    print("\n=== HEADLINE NUMBERS ===")
    aq = df[df["strategy"] == "always_qwen"].iloc[0]
    at = df[df["strategy"] == "always_teacher"].iloc[0]
    oc = df[df["strategy"] == "oracle_cascade"].iloc[0]
    print(f"  Always Qwen-7B:    cost={aq['cost']:.2f}  acc={aq['accuracy']:.3f}")
    print(f"  Always Teacher:    cost={at['cost']:.2f}  acc={at['accuracy']:.3f}")
    print(f"  Oracle cascade:    cost={oc['cost']:.2f}  acc={oc['accuracy']:.3f}")
    for probe in PROBE_NAMES:
        sub = df[(df["strategy"] == "probe_cascade") & (df["probe"] == probe)]
        # Find the threshold that maximises (acc - cost*lambda); also find Pareto
        pareto = pareto_front(sub.assign(strategy="probe_cascade"))
        # A "balanced" point: minimum cost reaching teacher's accuracy minus 0.02
        target_acc = at["accuracy"] - 0.02
        meet = sub[sub["accuracy"] >= target_acc].sort_values("cost")
        if meet.empty:
            print(f"  {PROBE_PRETTY[probe]:<26} -- can't reach acc={target_acc:.3f}")
        else:
            r = meet.iloc[0]
            saving = (at["cost"] - r["cost"]) / at["cost"] * 100
            print(f"  {PROBE_PRETTY[probe]:<26} target acc={target_acc:.3f}: "
                  f"cost={r['cost']:.2f} "
                  f"(escalate {r['frac_escalated']:.0%}), "
                  f"saves {saving:+.0f}% vs always-teacher")


if __name__ == "__main__":
    main()
