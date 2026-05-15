"""Render compact comparison tables for the SELECTIVE_PREDICTION_REPORT."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("/zhutingqi/song/Plan_opus_selective/results")


def fmt(value, places: int = 4) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except TypeError:
        pass
    if isinstance(value, str):
        return value
    if isinstance(value, (int,)):
        return str(value)
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    metrics = pd.read_csv(RESULTS_DIR / "selective_metrics_long.csv")
    routing = pd.read_csv(RESULTS_DIR / "routing_summary.csv")

    headers = ["setting", "predictor", "regime", "n", "base_acc", "auroc", "aurc",
               "sel_acc@0.25", "sel_acc@0.50", "sel_acc@0.75", "brier", "ece"]
    available = [c for c in headers if c in metrics.columns]
    md_lines: list[str] = []
    for setting in ["sample0", "fixed_k"]:
        md_lines.append(f"\n### Setting: {setting}\n")
        md_lines.append("| " + " | ".join(available) + " |")
        md_lines.append("|" + "|".join(["---"] * len(available)) + "|")
        sub = metrics[metrics["setting"] == setting].copy()
        # Order: single-feature first (sorted by auroc desc), then trained predictors.
        sub_single = sub[sub["regime"] == "single_feature"].sort_values("auroc", ascending=False)
        sub_trained = sub[sub["regime"] != "single_feature"].sort_values(["regime", "auroc"], ascending=[True, False])
        for _, row in pd.concat([sub_single, sub_trained]).iterrows():
            md_lines.append("| " + " | ".join(fmt(row.get(col)) for col in available) + " |")
    metrics_md = "\n".join(md_lines)

    routing_md_lines: list[str] = []
    for setting in ["sample0", "fixed_k"]:
        routing_md_lines.append(f"\n### Setting: {setting}\n")
        cols = ["cost_model", "method", "utility_per_item", "answer_count", "teacher_count", "abstain_count",
                "answer_accuracy", "teacher_accuracy", "answer_threshold", "defer_threshold"]
        cols = [c for c in cols if c in routing.columns]
        routing_md_lines.append("| " + " | ".join(cols) + " |")
        routing_md_lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        sub = routing[routing["setting"] == setting].sort_values(["cost_model", "method"]).reset_index(drop=True)
        for _, row in sub.iterrows():
            routing_md_lines.append("| " + " | ".join(fmt(row.get(col)) for col in cols) + " |")
    routing_md = "\n".join(routing_md_lines)

    (RESULTS_DIR / "metrics_table.md").write_text(metrics_md + "\n", encoding="utf-8")
    (RESULTS_DIR / "routing_table.md").write_text(routing_md + "\n", encoding="utf-8")
    print(metrics_md)
    print()
    print(routing_md)


if __name__ == "__main__":
    main()
