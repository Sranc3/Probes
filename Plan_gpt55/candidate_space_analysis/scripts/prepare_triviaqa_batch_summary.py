#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT_JSONL = "/zhutingqi/song/datasets/trivia_qa/processed/test.full.jsonl"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a lightweight batch_summary.json from normalized TriviaQA jsonl.")
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="triviaqa_scale_batch")
    parser.add_argument("--num-questions", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def read_rows(path: Path, offset: int, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_index, line in enumerate(file_obj):
            if line_index < offset:
                continue
            if len(rows) >= limit:
                break
            payload = json.loads(line)
            ideals = payload.get("ideal") or payload.get("ideal_answers") or []
            rows.append(
                {
                    "id": str(payload["id"]),
                    "question": str(payload["question"]),
                    "ideal_answers": list(ideals),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    rows = read_rows(Path(args.input_jsonl), args.offset, args.num_questions)
    summary = {
        "config": {
            "input_jsonl": str(Path(args.input_jsonl).resolve()),
            "num_questions": int(args.num_questions),
            "offset": int(args.offset),
            "created_for": "adaptive_controller_scaling_study",
        },
        "rows": rows,
    }
    (run_dir / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "row_count": len(rows)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
