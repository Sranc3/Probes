#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from grpo_utils import (  # noqa: E402
    append_jsonl,
    baseline_coverage,
    build_prompt,
    compute_group_rewards,
    load_baseline_correctness,
    load_triviaqa_records,
    mean,
    read_json,
    split_records,
    stdev,
    strict_correct,
    summarize_reward_rows,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Basin-GRPO LoRA on TriviaQA.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def setup_run(config: dict[str, Any], config_path: str) -> Path:
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['tag']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config_snapshot.json")
    (run_dir / "checkpoints").mkdir()
    (run_dir / "plots").mkdir()
    return run_dir


def load_model_and_tokenizer(config: dict[str, Any], device: str):
    tokenizer = AutoTokenizer.from_pretrained(config["model_dir"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if config["training"].get("bf16", True) and torch.cuda.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        config["model_dir"],
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map={"": device} if device.startswith("cuda") else None,
    )
    model.config.use_cache = False
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


def generate_group(model, tokenizer, prompt: str, train_cfg: dict[str, Any], device: str) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = int(encoded["input_ids"].shape[-1])
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            do_sample=True,
            temperature=float(train_cfg["temperature"]),
            top_p=float(train_cfg["top_p"]),
            max_new_tokens=int(train_cfg["max_new_tokens"]),
            num_return_sequences=int(train_cfg["group_size"]),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completions = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
    completion_token_counts = [int((row[prompt_len:] != tokenizer.pad_token_id).sum().item()) for row in outputs]
    return {"input_ids": outputs, "prompt_len": prompt_len, "completions": [text.strip() for text in completions], "token_counts": completion_token_counts}


def sequence_logprobs(model, input_ids: torch.Tensor, prompt_len: int, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    # Generated tokens start at original input position prompt_len, predicted by logits index prompt_len-1.
    mask = torch.zeros_like(token_logprobs)
    mask[:, max(prompt_len - 1, 0) :] = 1.0
    mask = mask * attention_mask[:, 1:].float()
    summed = (token_logprobs * mask).sum(dim=-1)
    counts = mask.sum(dim=-1).clamp_min(1.0)
    return summed / counts


def ref_sequence_logprobs(model, input_ids: torch.Tensor, prompt_len: int, attention_mask: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            return sequence_logprobs(model, input_ids, prompt_len, attention_mask).detach()
    return sequence_logprobs(model, input_ids, prompt_len, attention_mask).detach()


def fill_missing_greedy_baseline(
    model,
    tokenizer,
    splits: dict[str, list[dict[str, Any]]],
    baseline_correct: dict[str, float],
    config: dict[str, Any],
    device: str,
    run_dir: Path,
) -> dict[str, float]:
    cache_path_value = str(config.get("baseline_cache_path", "")).strip()
    cache_path = Path(cache_path_value) if cache_path_value else None
    if cache_path is not None:
        cached = load_baseline_correctness(cache_path)
        baseline_correct.update(cached)

    if not config.get("build_missing_baseline", False):
        return baseline_correct

    all_records = [record for records in splits.values() for record in records]
    missing = [record for record in all_records if record["id"] not in baseline_correct]
    if not missing:
        return baseline_correct

    model.eval()
    max_new_tokens = int(config["training"].get("baseline_max_new_tokens", config["training"].get("max_new_tokens", 48)))
    baseline_rows_path = run_dir / "baseline_greedy_rows.jsonl"
    context = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    with context:
        for index, record in enumerate(missing, start=1):
            prompt = build_prompt(tokenizer, record["question"], record["system_prompt"])
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = int(encoded["input_ids"].shape[-1])
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True).strip()
            strict = float(strict_correct(completion, record["ideal_answers"]))
            baseline_correct[record["id"]] = strict
            append_jsonl(
                baseline_rows_path,
                {
                    "question_id": record["id"],
                    "strict_correct": strict,
                    "completion": completion,
                    "missing_index": index,
                    "missing_total": len(missing),
                },
            )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            cache_path,
            {
                "baseline_correct": baseline_correct,
                "metadata": {
                    "source": "candidate_features_plus_greedy_fill",
                    "items": len(baseline_correct),
                    "max_new_tokens": max_new_tokens,
                },
            },
        )
    model.train()
    return baseline_correct


def evaluate_sample0(model, tokenizer, records: list[dict[str, Any]], reward_cfg: dict[str, Any], max_items: int, device: str) -> dict[str, Any]:
    model.eval()
    reward_rows: list[dict[str, Any]] = []
    for record in records[:max_items]:
        prompt = build_prompt(tokenizer, record["question"], record["system_prompt"])
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = int(encoded["input_ids"].shape[-1])
        with torch.no_grad():
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=48,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True).strip()
        rows = compute_group_rewards([completion], record["ideal_answers"], 0.0, [max(1, output.shape[-1] - prompt_len)], reward_cfg)
        reward_rows.extend(rows)
    model.train()
    summary = summarize_reward_rows(reward_rows)
    summary["eval_items"] = len(reward_rows)
    return summary


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    train_cfg = config["training"]
    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))

    run_dir = setup_run(config, args.config)
    records = load_triviaqa_records(
        config["input_jsonl"],
        int(config["split"].get("offset", 0)),
        int(config["split"]["train_size"]) + int(config["split"]["dev_size"]) + int(config["split"]["test_size"]),
        int(config["seed"]),
    )
    splits = split_records(records, config["split"])
    write_json(run_dir / "split_manifest.json", {name: [row["id"] for row in rows] for name, rows in splits.items()})

    model, tokenizer = load_model_and_tokenizer(config, args.device)
    baseline_correct = load_baseline_correctness(config.get("baseline_candidate_features"))
    baseline_correct = fill_missing_greedy_baseline(model, tokenizer, splits, baseline_correct, config, args.device, run_dir)
    write_json(run_dir / "baseline_coverage.json", {name: baseline_coverage(rows, baseline_correct) for name, rows in splits.items()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]))
    model.train()

    metrics_path = run_dir / "train_metrics.jsonl"
    samples_path = run_dir / "train_samples.jsonl"
    train_records = splits["train"]
    for step in range(1, int(train_cfg["max_steps"]) + 1):
        start_index = (step - 1) * int(train_cfg["questions_per_step"])
        step_records = [train_records[(start_index + idx) % len(train_records)] for idx in range(int(train_cfg["questions_per_step"]))]
        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        all_reward_rows: list[dict[str, Any]] = []
        all_advantages: list[float] = []
        all_kl: list[float] = []
        all_ratios: list[float] = []

        for record in step_records:
            prompt = build_prompt(tokenizer, record["question"], record["system_prompt"])
            generated = generate_group(model, tokenizer, prompt, train_cfg, args.device)
            input_ids = generated["input_ids"].to(args.device)
            attention_mask = torch.ones_like(input_ids)
            prompt_len = int(generated["prompt_len"])
            with torch.no_grad():
                old_logp = sequence_logprobs(model, input_ids, prompt_len, attention_mask).detach()
                ref_logp = ref_sequence_logprobs(model, input_ids, prompt_len, attention_mask)
            reward_rows = compute_group_rewards(
                generated["completions"],
                record["ideal_answers"],
                baseline_correct.get(record["id"], 0.0),
                generated["token_counts"],
                config["reward"],
            )
            rewards = [float(row["reward"]) for row in reward_rows]
            mu = mean(rewards)
            sigma = stdev(rewards) or 1.0
            advantages = torch.tensor([(reward - mu) / (sigma + 1e-6) for reward in rewards], dtype=torch.float32, device=args.device)

            logp = sequence_logprobs(model, input_ids, prompt_len, attention_mask)
            ratios = torch.exp(logp - old_logp)
            clipped = torch.clamp(ratios, 1.0 - float(train_cfg["clip_epsilon"]), 1.0 + float(train_cfg["clip_epsilon"]))
            policy_loss = -torch.mean(torch.minimum(ratios * advantages, clipped * advantages))
            kl = torch.mean(torch.square(logp - ref_logp))
            loss = policy_loss + float(train_cfg["beta_kl"]) * kl
            losses.append(loss)

            all_reward_rows.extend(reward_rows)
            all_advantages.extend([float(item) for item in advantages.detach().cpu().tolist()])
            all_kl.append(float(kl.detach().cpu().item()))
            all_ratios.extend([float(item) for item in ratios.detach().cpu().tolist()])
            append_jsonl(
                samples_path,
                {
                    "step": step,
                    "question_id": record["id"],
                    "question": record["question"],
                    "ideal_answers": record["ideal_answers"],
                    "baseline_sample0_correct": baseline_correct.get(record["id"], 0.0),
                    "completions": generated["completions"],
                    "reward_rows": reward_rows,
                },
            )

        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
        optimizer.step()

        summary = summarize_reward_rows(all_reward_rows)
        metrics = {
            "step": step,
            "loss": float(total_loss.detach().cpu().item()),
            "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
            "mean_advantage": mean(all_advantages),
            "mean_kl": mean(all_kl),
            "mean_ratio": mean(all_ratios),
            "max_ratio": max(all_ratios) if all_ratios else 0.0,
            "questions": len(step_records),
            **summary,
        }
        if step % int(train_cfg["eval_every_steps"]) == 0:
            eval_items = min(int(train_cfg.get("eval_max_items", 16)), len(splits["dev"]))
            dev_summary = evaluate_sample0(model, tokenizer, splits["dev"], config["reward"], max_items=eval_items, device=args.device)
            metrics.update({f"dev_{key}": value for key, value in dev_summary.items()})
        append_jsonl(metrics_path, metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

        if step % int(train_cfg["save_every_steps"]) == 0 or step == int(train_cfg["max_steps"]):
            ckpt_dir = run_dir / "checkpoints" / f"step_{step:04d}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    write_json(run_dir / "run_complete.json", {"status": "complete", "run_dir": str(run_dir), "steps": int(train_cfg["max_steps"])})
    print(json.dumps({"run_dir": str(run_dir), "status": "complete"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
