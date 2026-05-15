"""Multi-Agent Cascade Analysis (cost-aware utility + Pareto frontier).

We simulate a deployment where a question is sequentially processed by
increasingly expensive tiers. At each tier the local DCP-MLP probe gives a
confidence; if confidence >= threshold the cascade commits with that tier's
answer, otherwise escalates.

Tiers (for OOD analysis where each base has its own strict_correct):
  Tier 0  Llama-3.2-3B  K=1            cost ~ 3.2 GFLOPs/token  -> 0.4
  Tier 1  Qwen2.5-7B    K=1            cost ~ 7.6 GFLOPs/token  -> 1.0
  Tier 2  Qwen2.5-72B   K=1            cost ~ 72.7 GFLOPs/token -> 9.6
  Tier 3  Qwen2.5-72B   K=8 (self-cons) cost ~ 8 * Tier 2       -> 76.7
  Tier 4  External teacher API (always-correct oracle)          -> 50.0
          (priced as ~5x Qwen-72B-K=1 in deployed-cost units)

Confidence signal at each tier = best-layer DCP-MLP probe trained on that
base's hidden states (we have these in
``results/<base>/per_item_predictions_<dataset>_best.csv``).

We compare 6 strategies:
  1. always_tier_N (5 baselines: 3B, 7B, 72B, 72B-K8, teacher)
  2. dcp_cascade(threshold)   — sweep threshold in [0,1]
  3. random_cascade(p_keep)   — at each tier random commit/escalate
  4. oracle_cascade           — cheat: route to cheapest tier that gets it right
  5. random+oracle bounds visualised together for context

Outputs:
  reports/dashboard_cascade.png   — Pareto plot + companion panels
  results/cascade_analysis.csv    — per-strategy (cost, acc) table
  results/cascade_per_threshold_*.csv — full DCP threshold sweep per dataset
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

THIS_DIR = Path(__file__).resolve().parents[1]

# Cost model in "Qwen-7B forward unit" (FLOP proxy)
TIER_NAMES = ["Llama-3B", "Qwen-7B", "Qwen-72B", "Qwen-72B K=8", "Teacher API"]
TIER_TAGS  = ["llama3b",   "qwen7b",  "qwen72b",  None,             None]
TIER_COSTS = [0.42,        1.0,       9.57,        76.6,             50.0]
# Tier 3 (K=8) and Tier 4 (teacher) use Tier 2's probe to make the
# escalation decision (since we have no separate K=8 / teacher confidence).
# For the *cumulative* cost we sum every tier visited.

# For Tier 3 (K=8) accuracy: estimate as K=1 + a small empirical uplift.
# We use 1.05x as a conservative scaling factor (TriviaQA anchor data
# shows K=8 majority-voting accuracy ~= K=1 acc + 0.05 on 7B).
K8_UPLIFT = 0.05
# For Tier 4 (teacher) accuracy: oracle = 1.0
TEACHER_ACC = 1.0


# ---------------------------------------------------------------------------
# Build aligned per-question table for one OOD dataset
# ---------------------------------------------------------------------------


def build_cascade_table(dataset: str) -> pd.DataFrame:
    """Return one row per question with per-tier (correct, dcp_score)."""
    per_item = pd.read_csv(THIS_DIR / "results" / "per_item_predictions_all.csv")
    per_item = per_item[(per_item["dataset"] == dataset) &
                        (per_item["probe"] == "DCP_mlp")]
    pivot = per_item.pivot_table(
        index="question_id", columns="base",
        values=["y_true", "y_score"], aggfunc="first"
    )
    pivot.columns = [f"{m}_{b}" for m, b in pivot.columns]
    pivot = pivot.reset_index()
    cols = ["question_id"]
    for tag in ["llama3b", "qwen7b", "qwen72b"]:
        cols += [f"y_true_{tag}", f"y_score_{tag}"]
    return pivot[cols]


# ---------------------------------------------------------------------------
# Cascade simulation
# ---------------------------------------------------------------------------


def simulate_cascade(
    table: pd.DataFrame, threshold: float, use_teacher: bool = True,
    use_k8: bool = True,
) -> tuple[float, float]:
    """Return (avg_cost, avg_accuracy) for a single global threshold.

    Tier visit logic:
      - Run Tier 0 (Llama-3B) -> get confidence -> if >= threshold commit.
      - Else escalate to Tier 1 (Qwen-7B) -> repeat.
      - Else Tier 2 (Qwen-72B).
      - Else (use_k8) Tier 3 (Qwen-72B K=8): commit anyway, accuracy = 72B K=1 acc + K8_UPLIFT
        (clip to [0,1]).
      - Else (use_teacher) Tier 4: commit, accuracy = 1.0.
    """
    n = len(table)
    costs = np.zeros(n)
    correct = np.zeros(n)
    base_tags = ["llama3b", "qwen7b", "qwen72b"]
    base_costs = [TIER_COSTS[0], TIER_COSTS[1], TIER_COSTS[2]]
    for i, (_, row) in enumerate(table.iterrows()):
        committed = False
        cum_cost = 0.0
        for ti, tag in enumerate(base_tags):
            cum_cost += base_costs[ti]
            conf = float(row[f"y_score_{tag}"])
            if conf >= threshold:
                correct[i] = float(row[f"y_true_{tag}"])
                costs[i] = cum_cost
                committed = True
                break
        if committed:
            continue
        # No K=1 tier was confident enough.
        if use_k8:
            cum_cost += TIER_COSTS[3]
            base_acc_72b = float(row["y_true_qwen72b"])
            # K=8 majority voting accuracy upper-bounded at 1.0.
            correct[i] = min(1.0, base_acc_72b + K8_UPLIFT)
            costs[i] = cum_cost
            committed = True
            continue
        if use_teacher:
            cum_cost += TIER_COSTS[4]
            correct[i] = TEACHER_ACC
            costs[i] = cum_cost
            committed = True
            continue
        # If neither K=8 nor teacher, fall back to Qwen-72B's K=1 answer.
        correct[i] = float(row["y_true_qwen72b"])
        costs[i] = cum_cost
    return float(costs.mean()), float(correct.mean())


def simulate_random_cascade(
    table: pd.DataFrame, p_keep: float, n_trials: int = 50,
    use_teacher: bool = True, use_k8: bool = True, rng_seed: int = 0,
) -> tuple[float, float]:
    """At each tier, with prob p_keep commit; else escalate. Random Bernoulli."""
    rng = np.random.default_rng(rng_seed)
    avg_costs, avg_accs = [], []
    for _ in range(n_trials):
        n = len(table)
        costs = np.zeros(n)
        correct = np.zeros(n)
        base_tags = ["llama3b", "qwen7b", "qwen72b"]
        base_costs = [TIER_COSTS[0], TIER_COSTS[1], TIER_COSTS[2]]
        for i, (_, row) in enumerate(table.iterrows()):
            committed = False
            cum_cost = 0.0
            for ti, tag in enumerate(base_tags):
                cum_cost += base_costs[ti]
                if rng.random() < p_keep:
                    correct[i] = float(row[f"y_true_{tag}"])
                    costs[i] = cum_cost
                    committed = True
                    break
            if committed:
                continue
            if use_k8:
                cum_cost += TIER_COSTS[3]
                base_acc_72b = float(row["y_true_qwen72b"])
                correct[i] = min(1.0, base_acc_72b + K8_UPLIFT)
                costs[i] = cum_cost
                continue
            if use_teacher:
                cum_cost += TIER_COSTS[4]
                correct[i] = TEACHER_ACC
                costs[i] = cum_cost
                continue
            correct[i] = float(row["y_true_qwen72b"])
            costs[i] = cum_cost
        avg_costs.append(costs.mean())
        avg_accs.append(correct.mean())
    return float(np.mean(avg_costs)), float(np.mean(avg_accs))


def simulate_oracle_cascade(table: pd.DataFrame) -> tuple[float, float]:
    """Cheat: for each question, route to the cheapest tier that is correct.
    If no K=1 tier is correct, route to teacher (cost = sum of all tiers + teacher).
    """
    n = len(table)
    costs = np.zeros(n)
    correct = np.zeros(n)
    base_tags = ["llama3b", "qwen7b", "qwen72b"]
    base_costs = [TIER_COSTS[0], TIER_COSTS[1], TIER_COSTS[2]]
    for i, (_, row) in enumerate(table.iterrows()):
        committed = False
        cum_cost = 0.0
        for ti, tag in enumerate(base_tags):
            cum_cost += base_costs[ti]
            if float(row[f"y_true_{tag}"]) > 0.5:
                correct[i] = 1.0
                costs[i] = cum_cost
                committed = True
                break
        if not committed:
            # All K=1 wrong → go straight to teacher
            cum_cost += TIER_COSTS[4]
            correct[i] = TEACHER_ACC
            costs[i] = cum_cost
    return float(costs.mean()), float(correct.mean())


# ---------------------------------------------------------------------------
# Sweep + assemble per-strategy results
# ---------------------------------------------------------------------------


def sweep_strategies(table: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows: list[dict] = []
    # Always-Tier-N baselines (no cascade).
    base_tags = ["llama3b", "qwen7b", "qwen72b"]
    for ti, tag in enumerate(base_tags):
        acc = float(table[f"y_true_{tag}"].mean())
        rows.append({
            "strategy": f"always_{TIER_NAMES[ti]}",
            "threshold": np.nan, "cost": TIER_COSTS[ti],
            "accuracy": acc, "label": f"Always {TIER_NAMES[ti]} (K=1)",
        })
    # Always-K=8 on Qwen-72B.
    rows.append({
        "strategy": "always_72B_K8",
        "threshold": np.nan, "cost": TIER_COSTS[3],
        "accuracy": min(1.0, float(table["y_true_qwen72b"].mean()) + K8_UPLIFT),
        "label": "Always Qwen-72B K=8",
    })
    # Always teacher.
    rows.append({
        "strategy": "always_teacher",
        "threshold": np.nan, "cost": TIER_COSTS[4],
        "accuracy": TEACHER_ACC,
        "label": "Always Teacher API",
    })

    # DCP cascade threshold sweep.
    thresholds = np.linspace(0.05, 0.99, 50)
    for th in thresholds:
        cost, acc = simulate_cascade(table, th, use_teacher=True, use_k8=True)
        rows.append({
            "strategy": "dcp_cascade",
            "threshold": th, "cost": cost, "accuracy": acc,
            "label": f"DCP cascade (th={th:.2f})",
        })

    # Random cascade for context.
    for p in np.linspace(0.0, 1.0, 21):
        cost, acc = simulate_random_cascade(
            table, p, n_trials=20, use_teacher=True, use_k8=True
        )
        rows.append({
            "strategy": "random_cascade",
            "threshold": p, "cost": cost, "accuracy": acc,
            "label": f"Random cascade (p={p:.2f})",
        })

    # Oracle cascade.
    cost, acc = simulate_oracle_cascade(table)
    rows.append({
        "strategy": "oracle_cascade",
        "threshold": np.nan, "cost": cost, "accuracy": acc,
        "label": "Oracle cascade (cheat)",
    })

    df = pd.DataFrame(rows)
    df["dataset"] = dataset
    return df


# ---------------------------------------------------------------------------
# Pareto extraction
# ---------------------------------------------------------------------------


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that lie on the Pareto frontier of (low cost, high accuracy)."""
    df_sorted = df.sort_values("cost").reset_index(drop=True)
    keep = []
    best_acc = -1
    for _, r in df_sorted.iterrows():
        if r["accuracy"] > best_acc:
            keep.append(True)
            best_acc = r["accuracy"]
        else:
            keep.append(False)
    return df_sorted[keep]


# ---------------------------------------------------------------------------
# Cost-aware utility (cents per correct answer)
# ---------------------------------------------------------------------------


def utility_curve(df_cascade: pd.DataFrame, c_wrong: float) -> pd.DataFrame:
    """Compute utility = accuracy - c_wrong * (1-accuracy) - cost_factor.

    To make units comparable we report accuracy and cost separately and let
    the user pick a c_wrong. Here we define utility = (1 + c_wrong) * acc
    - c_wrong - lambda * cost, with lambda = 0.01 (cost in $0.01-equivalent
    per Qwen-7B forward).
    """
    cost_lambda = 0.01
    df = df_cascade.copy()
    df["utility"] = (1 + c_wrong) * df["accuracy"] - c_wrong - cost_lambda * df["cost"]
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

DATASET_PRETTY = {"hotpotqa": "HotpotQA OOD", "nq": "NQ-Open OOD"}
COLOR = {
    "always_Llama-3B":      "#9ecae1",
    "always_Qwen-7B":       "#6baed6",
    "always_Qwen-72B":      "#3182bd",
    "always_72B_K8":        "#08519c",
    "always_teacher":       "#000000",
    "dcp_cascade":          "#d62728",
    "random_cascade":       "#bcbcbc",
    "oracle_cascade":       "#2ca02c",
}
MARKER = {
    "always_Llama-3B":      "^",
    "always_Qwen-7B":       "s",
    "always_Qwen-72B":      "D",
    "always_72B_K8":        "p",
    "always_teacher":       "*",
    "dcp_cascade":          "o",
    "random_cascade":       "x",
    "oracle_cascade":       "v",
}


def panel_pareto(ax, df: pd.DataFrame, dataset: str) -> None:
    # Always-baselines
    for s in ["always_Llama-3B", "always_Qwen-7B", "always_Qwen-72B",
              "always_72B_K8", "always_teacher"]:
        sub = df[df["strategy"] == s]
        if sub.empty:
            continue
        r = sub.iloc[0]
        ax.scatter(r["cost"], r["accuracy"], s=140, marker=MARKER[s],
                   color=COLOR[s], edgecolors="black", linewidths=0.7,
                   zorder=4)
        ax.annotate(r["label"].replace("Always ", ""),
                    (r["cost"], r["accuracy"]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=7.5, color=COLOR[s])

    # Random cascade cloud
    sub = df[df["strategy"] == "random_cascade"].sort_values("cost")
    ax.plot(sub["cost"], sub["accuracy"],
            marker=MARKER["random_cascade"], color=COLOR["random_cascade"],
            linewidth=1.0, alpha=0.6, label="Random cascade", zorder=2)

    # DCP cascade
    sub = df[df["strategy"] == "dcp_cascade"].sort_values("cost")
    ax.plot(sub["cost"], sub["accuracy"],
            marker=MARKER["dcp_cascade"], color=COLOR["dcp_cascade"],
            linewidth=1.8, markersize=4, alpha=0.95,
            label="DCP cascade (ours)", zorder=5)
    # Annotate threshold endpoints on DCP curve
    if not sub.empty:
        sub_sorted = sub.sort_values("cost")
        first = sub_sorted.iloc[0]
        last = sub_sorted.iloc[-1]
        ax.annotate(f"th={first['threshold']:.2f}",
                    (first["cost"], first["accuracy"]),
                    textcoords="offset points", xytext=(-30, -14),
                    fontsize=6.5, color=COLOR["dcp_cascade"])
        ax.annotate(f"th={last['threshold']:.2f}",
                    (last["cost"], last["accuracy"]),
                    textcoords="offset points", xytext=(8, -10),
                    fontsize=6.5, color=COLOR["dcp_cascade"])

    # Oracle cascade
    sub = df[df["strategy"] == "oracle_cascade"]
    if not sub.empty:
        r = sub.iloc[0]
        ax.scatter(r["cost"], r["accuracy"], s=160, marker=MARKER["oracle_cascade"],
                   color=COLOR["oracle_cascade"], edgecolors="black",
                   linewidths=0.8, zorder=6)
        ax.annotate("Oracle cascade\n(cheat lower bound)",
                    (r["cost"], r["accuracy"]),
                    textcoords="offset points", xytext=(10, -5),
                    fontsize=7.5, color=COLOR["oracle_cascade"])

    ax.set_xlabel("Avg cost per question (Qwen-7B forward units)")
    ax.set_ylabel("Avg accuracy")
    ax.set_xscale("log")
    ax.set_title(f"{DATASET_PRETTY[dataset]} — cost-vs-accuracy Pareto frontier",
                 fontsize=10)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_ylim(0.2, 1.05)


def panel_dcp_threshold(ax, df: pd.DataFrame, dataset: str) -> None:
    sub = df[df["strategy"] == "dcp_cascade"].sort_values("threshold")
    if sub.empty:
        return
    ax2 = ax.twinx()
    ax.plot(sub["threshold"], sub["accuracy"],
            color="#d62728", linewidth=1.8, label="Accuracy")
    ax2.plot(sub["threshold"], sub["cost"],
             color="#1f77b4", linewidth=1.8, linestyle="--", label="Cost")
    ax.set_xlabel("Confidence threshold (commit if probe ≥ th)")
    ax.set_ylabel("Avg accuracy", color="#d62728")
    ax2.set_ylabel("Avg cost (Qwen-7B units)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#d62728")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_title(f"{DATASET_PRETTY[dataset]} — DCP cascade threshold sweep",
                 fontsize=10)
    ax.grid(linestyle=":", alpha=0.4)


def panel_routing_breakdown(ax, table: pd.DataFrame, dataset: str,
                            threshold: float = 0.5) -> None:
    """Stacked bar showing fraction routed to each tier at one threshold."""
    base_tags = ["llama3b", "qwen7b", "qwen72b"]
    n = len(table)
    counts = np.zeros(5, dtype=int)  # tier 0-4
    correct_per_tier = np.zeros(5)
    for _, row in table.iterrows():
        committed = False
        for ti, tag in enumerate(base_tags):
            conf = float(row[f"y_score_{tag}"])
            if conf >= threshold:
                counts[ti] += 1
                correct_per_tier[ti] += float(row[f"y_true_{tag}"])
                committed = True
                break
        if not committed:
            # K=8 tier
            counts[3] += 1
            base_acc_72b = float(row["y_true_qwen72b"])
            correct_per_tier[3] += min(1.0, base_acc_72b + K8_UPLIFT)
    fractions = counts / n
    accs_per = np.where(counts > 0, correct_per_tier / np.maximum(counts, 1), np.nan)

    bars = ax.bar(range(5), fractions,
                  color=[COLOR.get(f"always_{TIER_NAMES[i]}",
                                   COLOR.get("always_72B_K8") if i == 3 else COLOR.get("always_teacher"))
                         for i in range(5)],
                  edgecolor="black", linewidth=0.5)
    for i, (b, f, a) in enumerate(zip(bars, fractions, accs_per)):
        if f > 0:
            label = f"{f*100:.0f}%"
            if not np.isnan(a):
                label += f"\nacc={a:.2f}"
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    label, ha="center", fontsize=7)
    ax.set_xticks(range(5))
    ax.set_xticklabels([TIER_NAMES[i] for i in range(5)],
                       fontsize=7, rotation=12)
    ax.set_ylabel("Fraction of questions routed to tier")
    ax.set_title(f"{DATASET_PRETTY[dataset]} — DCP cascade @ th={threshold:.2f}\n"
                 "Where do questions get answered?",
                 fontsize=10)
    ax.set_ylim(0, max(0.5, max(fractions) * 1.3))
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def panel_score_distribution(ax, dataset: str) -> None:
    """Histogram of DCP scores per base — reveals bimodality limiting cascade."""
    per_item = pd.read_csv(THIS_DIR / "results" / "per_item_predictions_all.csv")
    per_item = per_item[(per_item["dataset"] == dataset) &
                        (per_item["probe"] == "DCP_mlp")]
    bases = ["llama3b", "qwen7b", "qwen72b"]
    base_pretty = {"llama3b": "Llama-3B", "qwen7b": "Qwen-7B", "qwen72b": "Qwen-72B"}
    base_color = {"llama3b": "#9ecae1", "qwen7b": "#3182bd", "qwen72b": "#08519c"}
    for tag in bases:
        sub = per_item[per_item["base"] == tag]
        ax.hist(sub["y_score"], bins=30, alpha=0.55,
                color=base_color[tag], label=base_pretty[tag],
                edgecolor="black", linewidth=0.3)
    ax.set_xlabel("DCP-MLP confidence score")
    ax.set_ylabel("Count of questions")
    ax.set_title(f"{DATASET_PRETTY[dataset]} — DCP score distribution per base\n"
                 "Bimodality (peaks at 0 / 1) limits smooth threshold tuning.",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def panel_cost_acc_lift(ax, df: pd.DataFrame, dataset: str) -> None:
    """For each cost level achievable by DCP cascade, what's the accuracy lift
    over the cheapest baseline at that cost?"""
    dcp = df[df["strategy"] == "dcp_cascade"].sort_values("cost")
    bls = df[df["strategy"].str.startswith("always_")].sort_values("cost")
    if dcp.empty or bls.empty:
        ax.set_title("(no data)")
        return

    # For each DCP point, find the best baseline at the same OR lower cost.
    lifts = []
    for _, r in dcp.iterrows():
        bl_at_or_below = bls[bls["cost"] <= r["cost"]]
        if bl_at_or_below.empty:
            continue
        best_bl_acc = bl_at_or_below["accuracy"].max()
        lifts.append({
            "cost": r["cost"],
            "dcp_acc": r["accuracy"],
            "bl_acc": best_bl_acc,
            "lift": r["accuracy"] - best_bl_acc,
        })
    lift_df = pd.DataFrame(lifts)
    if lift_df.empty:
        ax.set_title("(no data)")
        return

    colors = ["#2ca02c" if v > 0 else "#d62728" for v in lift_df["lift"]]
    ax.bar(np.arange(len(lift_df)), lift_df["lift"], color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(np.arange(len(lift_df))[::3])
    ax.set_xticklabels([f"{c:.1f}" for c in lift_df["cost"][::3]],
                       fontsize=7, rotation=30)
    ax.set_xlabel("DCP cascade cost (Qwen-7B units)")
    ax.set_ylabel("Accuracy lift over best baseline\n(at same or lower cost)")
    ax.set_title(f"{DATASET_PRETTY[dataset]} — Does DCP cascade beat baselines?\n"
                 "Green = DCP wins; Red = always-baseline wins.",
                 fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = THIS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    tables = {}
    for dataset in ["hotpotqa", "nq"]:
        print(f"\n[cascade] dataset = {dataset}")
        table = build_cascade_table(dataset)
        tables[dataset] = table
        print(f"  n={len(table)}")
        df = sweep_strategies(table, dataset)
        all_results.append(df)
        df_path = out_dir / f"cascade_per_threshold_{dataset}.csv"
        df.to_csv(df_path, index=False)
        print(f"  wrote {df_path}")

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(out_dir / "cascade_analysis.csv", index=False)
    print(f"\n[done] cascade_analysis.csv ({len(df_all)} rows)")

    # Compose dashboard
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.30,
                  left=0.06, right=0.97, top=0.93, bottom=0.06,
                  height_ratios=[1.2, 1, 1])

    fig.suptitle(
        "Multi-Agent Cascade Analysis — Cost-aware Selective Prediction\n"
        "Tiers: Llama-3B → Qwen-7B → Qwen-72B → Qwen-72B K=8 → Teacher API\n"
        "Each tier uses its own DCP-MLP probe to decide commit vs escalate.",
        fontsize=13, fontweight="bold", y=0.985,
    )

    # Row 1: Pareto frontiers (the money figure)
    panel_pareto(fig.add_subplot(gs[0, 0]), all_results[0], "hotpotqa")
    panel_pareto(fig.add_subplot(gs[0, 1]), all_results[1], "nq")

    # Row 2: threshold sweep
    panel_dcp_threshold(fig.add_subplot(gs[1, 0]), all_results[0], "hotpotqa")
    panel_dcp_threshold(fig.add_subplot(gs[1, 1]), all_results[1], "nq")

    # Row 3: score distribution (reveals bimodality) + lift over baselines
    panel_score_distribution(fig.add_subplot(gs[2, 0]), "hotpotqa")
    panel_cost_acc_lift(fig.add_subplot(gs[2, 1]), all_results[1], "nq")

    out = THIS_DIR / "reports" / "dashboard_cascade.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[done] dashboard_cascade -> {out}")
    print(f"[size] {out.stat().st_size / 1e6:.2f} MB")

    # Print headline numbers for quick check
    print("\n=== Headline numbers ===")
    for ds, dfg in [("hotpotqa", all_results[0]), ("nq", all_results[1])]:
        print(f"\n{ds}:")
        for s in ["always_Llama-3B", "always_Qwen-7B", "always_Qwen-72B",
                  "always_72B_K8", "always_teacher"]:
            sub = dfg[dfg["strategy"] == s]
            if not sub.empty:
                r = sub.iloc[0]
                print(f"  {r['label']:<28} cost={r['cost']:6.2f}  acc={r['accuracy']:.3f}")
        # DCP at threshold = 0.5
        dcp = dfg[(dfg["strategy"] == "dcp_cascade")]
        if not dcp.empty:
            # closest threshold to 0.5
            closest = dcp.iloc[(dcp["threshold"] - 0.5).abs().idxmin() if "threshold" in dcp else 0]
            print(f"  DCP cascade (th=0.50)        cost={closest['cost']:6.2f}  acc={closest['accuracy']:.3f}")
        ora = dfg[dfg["strategy"] == "oracle_cascade"]
        if not ora.empty:
            r = ora.iloc[0]
            print(f"  Oracle cascade               cost={r['cost']:6.2f}  acc={r['accuracy']:.3f}")


if __name__ == "__main__":
    main()
