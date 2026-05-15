#!/usr/bin/env python3
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

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from grpo_utils import (  # noqa: E402
    build_prompt,
    compute_group_rewards,
    load_triviaqa_records,
    mean,
    read_json,
    split_records,
    summarize_reward_rows,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Basin-GRPO LoRA checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--fixed-k", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seed", type=int, default=-1)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model(config: dict[str, Any], adapter_dir: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or config["model_dir"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        config["model_dir"],
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


def majority_basin_choice(reward_rows: list[dict[str, Any]], completions: list[str]) -> int:
    counts = Counter(row["canonical_basin"] for row in reward_rows)
    first_seen = {row["canonical_basin"]: idx for idx, row in reversed(list(enumerate(reward_rows)))}
    best_basin = max(counts, key=lambda basin: (counts[basin], -first_seen[basin]))
    candidates = [idx for idx, row in enumerate(reward_rows) if row["canonical_basin"] == best_basin]
    return min(candidates, key=lambda idx: (len(completions[idx].split()), idx))


def evaluate_record(model, tokenizer, record: dict[str, Any], config: dict[str, Any], fixed_k_values: list[int], device: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt = build_prompt(tokenizer, record["question"], record["system_prompt"])
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = int(encoded["input_ids"].shape[-1])
    max_k = max(fixed_k_values)
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            do_sample=True,
            temperature=float(config["training"]["temperature"]),
            top_p=float(config["training"]["top_p"]),
            max_new_tokens=int(config["training"]["max_new_tokens"]),
            num_return_sequences=max_k,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completions = [text.strip() for text in tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)]
    token_counts = [int((row[prompt_len:] != tokenizer.pad_token_id).sum().item()) for row in outputs]
    reward_rows = compute_group_rewards(completions, record["ideal_answers"], 0.0, token_counts, config["reward"])
    sample_rows = [
        {
            "question_id": record["id"],
            "sample_index": idx,
            "completion": completion,
            **reward,
        }
        for idx, (completion, reward) in enumerate(zip(completions, reward_rows))
    ]
    method_rows: list[dict[str, Any]] = []
    for k in fixed_k_values:
        prefix = reward_rows[:k]
        selected_idx = 0 if k == 1 else majority_basin_choice(prefix, completions[:k])
        selected = reward_rows[selected_idx]
        method_rows.append(
            {
                "question_id": record["id"],
                "method": f"fixed_{k}_majority_basin" if k > 1 else "sample0",
                "k": k,
                "strict": selected["strict"],
                "f1": selected["f1"],
                "reward": selected["reward"],
                "token_count": token_counts[selected_idx],
                "selected_sample_index": selected_idx,
                "answer": completions[selected_idx],
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
                "strict_rate": mean([float(row["strict"]) for row in group]),
                "mean_f1": mean([float(row["f1"]) for row in group]),
                "mean_reward": mean([float(row["reward"]) for row in group]),
                "delta_vs_sample0": mean(deltas),
                "improved_count": sum(1 for value in deltas if value > 0),
                "damaged_count": sum(1 for value in deltas if value < 0),
                "avg_selected_tokens": mean([float(row["token_count"]) for row in group]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    seed = int(config["seed"] if args.seed < 0 else args.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    records = load_triviaqa_records(
        config["input_jsonl"],
        int(config["split"].get("offset", 0)),
        int(config["split"]["train_size"]) + int(config["split"]["dev_size"]) + int(config["split"]["test_size"]),
        int(config["seed"]),
    )
    split_rows = split_records(records, config["split"])[args.split][: args.max_items]
    model, tokenizer = load_model(config, args.adapter_dir, args.device)
    output_root = Path(args.output_root or Path(args.adapter_dir or config["output_root"]) / "eval")
    output_dir = output_root / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    method_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for record in split_rows:
        rows, samples = evaluate_record(model, tokenizer, record, config, args.fixed_k, args.device)
        method_rows.extend(rows)
        sample_rows.extend(samples)
    summary_rows = summarize(method_rows)
    write_csv(output_dir / "eval_method_rows.csv", method_rows)
    write_csv(output_dir / "eval_sample_rows.csv", sample_rows)
    write_csv(output_dir / "eval_summary.csv", summary_rows)
    write_json(output_dir / "eval_metadata.json", {"adapter_dir": args.adapter_dir, "split": args.split, "items": len(split_rows), "fixed_k": args.fixed_k, "seed": seed})
    print(json.dumps({"output_dir": str(output_dir), "summary": summary_rows}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
