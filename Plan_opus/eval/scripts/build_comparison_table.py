"""Aggregate every eval run under ``Plan_opus/eval/results`` into one report.

Reads ``eval_summary_avg.csv`` from each run subfolder and produces:
- ``comparison_long.csv`` with every (model, method) row
- ``comparison_pivot.csv`` with model rows and method columns (strict_rate)
- ``comparison_summary.md`` with a markdown table for the report
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path("/zhutingqi/song/Plan_opus/eval/results")
SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
sys.path.insert(0, str(SHARED_DIR))
from text_utils import write_csv  # noqa: E402


def collect_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for child in sorted(ROOT.iterdir()):
        csv_path = child / "eval_summary_avg.csv"
        if not csv_path.exists():
            continue
        label = child.name.split("_", 1)[1] if "_" in child.name else child.name
        with csv_path.open("r", encoding="utf-8") as file_obj:
            for row in csv.DictReader(file_obj):
                row["model"] = label
                rows.append(row)
    return rows


def pivot(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    methods = ["sample0", "fixed_2_majority_basin", "fixed_4_majority_basin", "fixed_8_majority_basin"]
    by_model: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["method"]] = row
    pivot_rows: list[dict[str, str]] = []
    for model, methods_dict in sorted(by_model.items()):
        out: dict[str, str] = {"model": model}
        for method in methods:
            data = methods_dict.get(method)
            if data:
                out[f"{method}_strict"] = data.get("strict_rate_mean", "")
                out[f"{method}_min"] = data.get("strict_rate_min", "")
                out[f"{method}_max"] = data.get("strict_rate_max", "")
                out[f"{method}_f1"] = data.get("mean_f1_mean", "")
                out[f"{method}_tokens"] = data.get("avg_selected_tokens_mean", "")
        pivot_rows.append(out)
    return pivot_rows


def fmt(value: str | float | None, places: int = 3) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return str(value)


def render_md(pivot_rows: list[dict[str, str]]) -> str:
    headers = [
        "Model",
        "sample0",
        "[min,max]",
        "fixed_2",
        "fixed_4",
        "fixed_8",
        "f1@s0",
        "tok@s0",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in pivot_rows:
        s0 = fmt(row.get("sample0_strict"))
        lo = fmt(row.get("sample0_min"))
        hi = fmt(row.get("sample0_max"))
        f2 = fmt(row.get("fixed_2_majority_basin_strict"))
        f4 = fmt(row.get("fixed_4_majority_basin_strict"))
        f8 = fmt(row.get("fixed_8_majority_basin_strict"))
        f1 = fmt(row.get("sample0_f1"), 4)
        tk = fmt(row.get("sample0_tokens"), 2)
        lines.append(f"| {row['model']} | {s0} | [{lo}, {hi}] | {f2} | {f4} | {f8} | {f1} | {tk} |")
    return "\n".join(lines)


def main() -> None:
    rows = collect_rows()
    if not rows:
        print("No eval_summary_avg.csv files found.")
        return
    pivot_rows = pivot(rows)
    write_csv(ROOT / "comparison_long.csv", rows)
    write_csv(ROOT / "comparison_pivot.csv", pivot_rows)
    md = render_md(pivot_rows)
    (ROOT / "comparison_summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
