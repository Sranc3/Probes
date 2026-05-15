#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize QA datasets into the lightweight batch_summary.json format used by Phase2A.")
    parser.add_argument("--dataset", choices=["triviaqa", "hotpotqa", "longbench"], required=True)
    parser.add_argument("--input", required=True, help="Input json/jsonl file.")
    parser.add_argument("--output-dir", required=True, help="Directory where batch_summary.json will be written.")
    parser.add_argument("--num-questions", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--include-context", action="store_true", help="Include context/evidence in the prompt when available.")
    return parser.parse_args()


def iter_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file_obj:
            return [json.loads(line) for line in file_obj if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ["data", "examples", "rows"]:
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"Cannot find records in {path}")


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def context_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, list):
                if len(item) >= 2 and isinstance(item[1], list):
                    title = str(item[0])
                    sentences = " ".join(str(sentence) for sentence in item[1])
                    chunks.append(f"{title}: {sentences}")
                else:
                    chunks.append(" ".join(str(part) for part in item))
            elif isinstance(item, dict):
                chunks.append(" ".join(str(part) for part in item.values()))
            else:
                chunks.append(str(item))
        return "\n".join(chunk for chunk in chunks if chunk.strip())
    if isinstance(value, dict):
        return "\n".join(str(part) for part in value.values())
    return str(value)


def normalize_record(dataset: str, record: dict[str, Any], index: int, include_context: bool) -> dict[str, Any]:
    if dataset == "triviaqa":
        question = str(record.get("question", ""))
        ideals = as_list(record.get("ideal") or record.get("ideal_answers") or record.get("answer"))
        row_id = str(record.get("id", f"triviaqa_{index}"))
        prompt_question = question
    elif dataset == "hotpotqa":
        question = str(record.get("question", ""))
        ideals = as_list(record.get("answer") or record.get("answers") or record.get("ideal") or record.get("ideal_answers"))
        row_id = str(record.get("_id", record.get("id", f"hotpotqa_{index}")))
        context = context_to_text(record.get("context") or record.get("supporting_facts"))
        prompt_question = f"Context:\n{context}\n\nQuestion: {question}" if include_context and context else question
    else:
        question = str(record.get("question") or record.get("input") or record.get("query") or "")
        ideals = as_list(record.get("answers") or record.get("answer") or record.get("outputs"))
        row_id = str(record.get("id", record.get("_id", f"longbench_{index}")))
        context = context_to_text(record.get("context") or record.get("passage") or record.get("document"))
        prompt_question = f"Context:\n{context}\n\nQuestion: {question}" if include_context and context else question

    return {
        "id": row_id,
        "question": prompt_question,
        "raw_question": question,
        "ideal_answers": ideals,
        "dataset": dataset,
        "strict_evaluator": "normalized_exact_or_contains",
        "prompt_includes_context": bool(include_context and prompt_question != question),
    }


def main() -> None:
    args = parse_args()
    records = iter_records(Path(args.input))
    selected = records[args.offset : args.offset + args.num_questions]
    rows = [normalize_record(args.dataset, record, args.offset + idx, args.include_context) for idx, record in enumerate(selected)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "dataset": args.dataset,
            "input": str(Path(args.input).resolve()),
            "num_questions": args.num_questions,
            "offset": args.offset,
            "include_context": args.include_context,
            "split_policy": "train/tune thresholds separately; evaluate transfer separately",
        },
        "rows": rows,
    }
    (output_dir / "batch_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(rows)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
