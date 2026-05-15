#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel a Phase2A run with strict exact/contains labels from a normalized QA jsonl.")
    parser.add_argument("--phase2a-run", required=True)
    parser.add_argument("--label-jsonl", required=True)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="phase2a_relabel_strict")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    lowered = str(text).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(cleaned.split())


def exact_match(answer: str, ideals: list[str]) -> bool:
    normalized = normalize_text(answer)
    return any(normalized == normalize_text(ideal) for ideal in ideals if normalize_text(ideal))


def contains_match(answer: str, ideals: list[str]) -> bool:
    normalized = normalize_text(answer)
    if not normalized:
        return False
    return any(normalize_text(ideal) and normalize_text(ideal) in normalized for ideal in ideals)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_labels(path: Path) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            row = json.loads(line)
            ideals = row.get("ideal_answers") or row.get("ideal") or row.get("answers") or row.get("answer") or []
            if isinstance(ideals, str):
                ideals = [ideals]
            labels[str(row["id"])] = [str(item) for item in ideals]
    return labels


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    baseline = {(row["seed"], row["question_id"]): row for row in rows if row["method"] == "single_sample_baseline"}
    for row in rows:
        grouped[row["method"]].append(row)
    summaries: list[dict[str, Any]] = []
    for method, group in sorted(grouped.items()):
        deltas = []
        changed = []
        for row in group:
            base = baseline.get((row["seed"], row["question_id"]))
            if base:
                deltas.append(float(row["final_strict_correct"]) - float(base["final_strict_correct"]))
                changed.append(float(row["answer_text"] != base["answer_text"]))
        summaries.append(
            {
                "method": method,
                "pairs": len(group),
                "strict_correct_rate": sum(float(row["final_strict_correct"]) for row in group) / max(1, len(group)),
                "semantic_correct_rate": sum(float(row.get("final_semantic_correct", row["final_strict_correct"])) for row in group) / max(1, len(group)),
                "nli_only_correct_rate": sum(float(row.get("nli_only_correct", 0.0)) for row in group) / max(1, len(group)),
                "answer_changed_vs_sample0_rate": sum(changed) / max(1, len(changed)),
                "delta_strict_correct_vs_sample0": sum(deltas) / max(1, len(deltas)),
                "delta_semantic_correct_vs_sample0": sum(deltas) / max(1, len(deltas)),
                "token_mean_entropy_mean": sum(float(row["token_mean_entropy"]) for row in group) / max(1, len(group)),
                "token_max_entropy_mean": sum(float(row["token_max_entropy"]) for row in group) / max(1, len(group)),
                "token_count_mean": sum(float(row["token_count"]) for row in group) / max(1, len(group)),
                "selected_cluster_size_mean": sum(float(row["selected_cluster_size"]) for row in group) / max(1, len(group)),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    source_dir = Path(args.phase2a_run)
    output_dir = Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=False)
    labels = load_labels(Path(args.label_jsonl))
    rows = read_csv(source_dir / "method_rows.csv")
    missing: set[str] = set()
    for row in rows:
        ideals = labels.get(str(row["question_id"]), [])
        if not ideals:
            missing.add(str(row["question_id"]))
        exact = exact_match(str(row["answer_text"]), ideals)
        contains = contains_match(str(row["answer_text"]), ideals)
        row["ideal_answers"] = json.dumps(ideals, ensure_ascii=False)
        row["final_exact_match"] = float(exact)
        row["final_contains_match"] = float(contains)
        row["final_strict_correct"] = float(exact or contains)
        row["final_semantic_correct"] = float(exact or contains)
        row["nli_only_correct"] = 0.0
        row["matched_ideal_answer"] = next((ideal for ideal in ideals if normalize_text(ideal) and normalize_text(ideal) in normalize_text(str(row["answer_text"]))), "")
        row["match_source"] = "strict_relabel"

    shutil.copy2(source_dir / "sample_sets.json", output_dir / "sample_sets.json")
    shutil.copy2(source_dir / "config_snapshot.json", output_dir / "config_snapshot.json")
    write_csv(output_dir / "method_rows.csv", rows)
    method_summary = summarize(rows)
    write_csv(output_dir / "method_summary.csv", method_summary)
    (output_dir / "summary.md").write_text(
        "# Phase2A Strict Relabel\n\n"
        f"- Source run: `{source_dir}`\n"
        f"- Label jsonl: `{Path(args.label_jsonl)}`\n"
        f"- Missing label ids: `{len(missing)}`\n",
        encoding="utf-8",
    )
    (output_dir / "phase2a_summary.json").write_text(
        json.dumps({"source_run": str(source_dir), "label_jsonl": args.label_jsonl, "method_summary": method_summary, "missing_label_ids": sorted(missing)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows), "missing_label_ids": len(missing)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
