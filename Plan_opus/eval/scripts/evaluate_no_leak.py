"""No-leak fixed-k evaluator usable across Plan_gpt55 / Plan_opus checkpoints.

Mirrors the protocol in ``Plan_gpt55/basin_grpo/scripts/evaluate_basin_grpo.py``:
generate ``max_k`` samples per question, then for each ``k`` in ``--fixed-k``
pick by majority canonical basin (shortest text within the majority). The
``sample0`` slot is the first sampled completion (k=1).

Adds:
- multi-seed support in one invocation (``--seeds 2026 42 17 1729``)
- supports both LoRA-adapter and base-model evaluation (``--adapter-dir`` empty
  to evaluate the base model)
- writes consolidated CSV ``eval_summary_per_seed.csv`` and aggregated
  ``eval_summary_avg.csv``
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
sys.path.insert(0, str(SHARED_DIR))

from text_utils import (  # noqa: E402
    canonical_answer,
    load_triviaqa_records,
    mean,
    read_json,
    strict_correct,
    best_f1,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed no-leak fixed-k evaluator.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--offset", type=int, default=500)
    parser.add_argument("--num-questions", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 42])
    parser.add_argument("--fixed-k", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="run")
    return parser.parse_args()


def load_model(model_dir: str, adapter_dir: str, device: str):
    tokenizer_dir = adapter_dir or model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map={"": device} if device.startswith("cuda") else None,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        model = base
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer: Any, question: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\nQuestion: {question}\nAnswer:"


def majority_basin_choice(canonicals: list[str], completions: list[str]) -> int:
    counts = Counter(canonicals)
    first_seen = {key: idx for idx, key in reversed(list(enumerate(canonicals)))}
    best_basin = max(counts.keys(), key=lambda b: (counts[b], -first_seen[b]))
    candidates = [idx for idx, key in enumerate(canonicals) if key == best_basin]
    return min(candidates, key=lambda idx: (len(completions[idx].split()), idx))


def evaluate_record(model, tokenizer, record: dict[str, Any], args, device: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt = build_prompt(tokenizer, record["question"], record["system_prompt"])
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = int(encoded["input_ids"].shape[-1])
    max_k = max(args.fixed_k)
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=max_k,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completions = [text.strip() for text in tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)]
    token_counts = [int((row[prompt_len:] != tokenizer.pad_token_id).sum().item()) for row in outputs]
    canonicals = [canonical_answer(text) for text in completions]
    method_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for idx, completion in enumerate(completions):
        sample_rows.append(
            {
                "question_id": record["id"],
                "sample_index": idx,
                "completion": completion,
                "strict": float(strict_correct(completion, record["ideal_answers"])),
                "f1": best_f1(completion, record["ideal_answers"]),
                "token_count": token_counts[idx],
                "canonical_basin": canonicals[idx],
            }
        )
    for k in args.fixed_k:
        prefix_canonicals = canonicals[:k]
        prefix_completions = completions[:k]
        if k == 1:
            chosen_idx = 0
            method = "sample0"
        else:
            chosen_idx = majority_basin_choice(prefix_canonicals, prefix_completions)
            method = f"fixed_{k}_majority_basin"
        chosen_completion = completions[chosen_idx]
        method_rows.append(
            {
                "question_id": record["id"],
                "method": method,
                "k": k,
                "strict": float(strict_correct(chosen_completion, record["ideal_answers"])),
                "f1": best_f1(chosen_completion, record["ideal_answers"]),
                "token_count": token_counts[chosen_idx],
                "selected_sample_index": chosen_idx,
                "answer": chosen_completion,
            }
        )
    return method_rows, sample_rows


def summarize(method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in method_rows:
        by_method.setdefault(row["method"], []).append(row)
    sample0 = {row["question_id"]: row for row in method_rows if row["method"] == "sample0"}
    rows: list[dict[str, Any]] = []
    for method, group in sorted(by_method.items()):
        deltas = [float(row["strict"]) - float(sample0[row["question_id"]]["strict"]) for row in group if row["question_id"] in sample0]
        rows.append(
            {
                "method": method,
                "items": len(group),
                "strict_rate": mean(float(row["strict"]) for row in group),
                "mean_f1": mean(float(row["f1"]) for row in group),
                "delta_vs_sample0": mean(deltas),
                "improved_count": sum(1 for v in deltas if v > 0),
                "damaged_count": sum(1 for v in deltas if v < 0),
                "avg_selected_tokens": mean(float(row["token_count"]) for row in group),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{timestamp}_{args.label}"
    output_dir.mkdir(parents=True, exist_ok=False)

    records = load_triviaqa_records(args.input_jsonl, args.offset, args.num_questions)
    model, tokenizer = load_model(args.model_dir, args.adapter_dir, args.device)

    per_seed_summaries: list[dict[str, Any]] = []
    all_method_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seed_methods: list[dict[str, Any]] = []
        seed_samples: list[dict[str, Any]] = []
        for idx, record in enumerate(records):
            method_rows, sample_rows = evaluate_record(model, tokenizer, record, args, args.device)
            seed_methods.extend(method_rows)
            seed_samples.extend(sample_rows)
            if (idx + 1) % 50 == 0:
                print(f"[seed {seed}] {idx+1}/{len(records)}", flush=True)
        for row in seed_methods:
            row["seed"] = seed
            all_method_rows.append(row)
        seed_summary = summarize(seed_methods)
        for row in seed_summary:
            row["seed"] = seed
            per_seed_summaries.append(row)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        write_csv(seed_dir / "eval_method_rows.csv", seed_methods)
        write_csv(seed_dir / "eval_sample_rows.csv", seed_samples)
        write_csv(seed_dir / "eval_summary.csv", seed_summary)
        print(json.dumps({"seed": seed, "summary": seed_summary}, ensure_ascii=False, indent=2), flush=True)

    write_csv(output_dir / "eval_summary_per_seed.csv", per_seed_summaries)

    avg_rows: dict[str, dict[str, Any]] = {}
    for row in per_seed_summaries:
        method = row["method"]
        bucket = avg_rows.setdefault(method, {"method": method, "items": 0, "values": {"strict_rate": [], "mean_f1": [], "delta_vs_sample0": [], "improved_count": [], "damaged_count": [], "avg_selected_tokens": []}})
        bucket["items"] = max(bucket["items"], int(row.get("items", 0)))
        for key in bucket["values"]:
            bucket["values"][key].append(float(row[key]))
    avg_summary: list[dict[str, Any]] = []
    for method, payload in avg_rows.items():
        avg_summary.append(
            {
                "method": method,
                "items": payload["items"],
                "seeds_evaluated": len(payload["values"]["strict_rate"]),
                "strict_rate_mean": mean(payload["values"]["strict_rate"]),
                "strict_rate_min": min(payload["values"]["strict_rate"]),
                "strict_rate_max": max(payload["values"]["strict_rate"]),
                "mean_f1_mean": mean(payload["values"]["mean_f1"]),
                "delta_vs_sample0_mean": mean(payload["values"]["delta_vs_sample0"]),
                "improved_count_mean": mean(payload["values"]["improved_count"]),
                "damaged_count_mean": mean(payload["values"]["damaged_count"]),
                "avg_selected_tokens_mean": mean(payload["values"]["avg_selected_tokens"]),
            }
        )
    write_csv(output_dir / "eval_summary_avg.csv", avg_summary)
    write_json(
        output_dir / "eval_metadata.json",
        {
            "model_dir": args.model_dir,
            "adapter_dir": args.adapter_dir,
            "input_jsonl": args.input_jsonl,
            "offset": args.offset,
            "num_questions": args.num_questions,
            "seeds": args.seeds,
            "fixed_k": args.fixed_k,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "label": args.label,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "avg_summary": avg_summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
