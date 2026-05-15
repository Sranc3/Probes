#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

GRPO_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "basin_grpo" / "scripts"
sys.path.insert(0, str(GRPO_SCRIPT_DIR))

from grpo_utils import (  # noqa: E402
    append_jsonl,
    best_f1,
    build_prompt,
    canonical_answer,
    mean,
    read_json,
    stdev,
    strict_correct,
    token_f1,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Anchor-aware GRPO with cross-model teacher basins.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def setup_run(config: dict[str, Any], config_path: str) -> Path:
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['tag']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config_snapshot.json")
    (run_dir / "checkpoints").mkdir()
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
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    return model, tokenizer


def load_records(path: str | Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            payload = json.loads(line)
            ideals = payload.get("ideal") or payload.get("ideal_answers") or []
            rows[str(payload["id"])] = {
                "id": str(payload["id"]),
                "question": str(payload["question"]),
                "ideal_answers": [str(item) for item in ideals],
                "system_prompt": payload.get("system_prompt", "Answer the question briefly and factually."),
            }
    return rows


def load_teacher_basins(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    basins: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(path):
        basins[str(row["question_id"])].append(
            {
                "answer": str(row.get("teacher_representative_answer", "")).strip(),
                "mass": safe_float(row.get("teacher_basin_mass")),
                "strict_any": safe_float(row.get("teacher_basin_strict_any")),
                "greedy_member": safe_float(row.get("teacher_greedy_member")),
            }
        )
    for group in basins.values():
        group.sort(key=lambda row: (row["mass"], row["greedy_member"]), reverse=True)
    return dict(basins)


def split_anchor_records(records: dict[str, dict[str, Any]], teacher_basins: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    qids = [qid for qid, basins in teacher_basins.items() if qid in records and basins]
    random.Random(int(config["seed"])).shuffle(qids)
    split_cfg = config["split"]
    train_size = int(split_cfg["train_size"])
    dev_size = int(split_cfg["dev_size"])
    test_size = int(split_cfg.get("test_size", 0))
    return {
        "train": [records[qid] for qid in qids[:train_size]],
        "dev": [records[qid] for qid in qids[train_size : train_size + dev_size]],
        "test": [records[qid] for qid in qids[train_size + dev_size : train_size + dev_size + test_size]],
    }


def answer_similarity(left: str, right: str) -> float:
    left_norm = " ".join(str(left).lower().split())
    right_norm = " ".join(str(right).lower().split())
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        return 0.95
    return token_f1(left, right)


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


def clean_guided_answer(text: str, max_words: int) -> bool:
    text = str(text).strip()
    if not text or len(text.split()) > max_words:
        return False
    lowered = text.lower()
    bad_markers = ["analysis", "however", "let's", "we need", "assistant", "final", "not sure"]
    return not any(marker in lowered for marker in bad_markers)


def guided_teacher_completions(teacher_basins: list[dict[str, Any]], train_cfg: dict[str, Any]) -> list[str]:
    guided_cfg = train_cfg.get("guided_candidates", {})
    if not guided_cfg.get("enabled", False):
        return []
    max_candidates = int(guided_cfg.get("max_teacher_candidates", 1))
    max_words = int(guided_cfg.get("max_words", 16))
    min_mass = float(guided_cfg.get("min_basin_mass", 0.0))
    require_greedy = bool(guided_cfg.get("require_greedy_member", False))
    answers: list[str] = []
    seen: set[str] = set()
    sorted_basins = sorted(
        teacher_basins,
        key=lambda row: (
            safe_float(row.get("mass")),
            safe_float(row.get("greedy_member")),
            answer_similarity(str(row.get("answer", "")), str(row.get("answer", ""))),
        ),
        reverse=True,
    )
    for basin in sorted_basins:
        if safe_float(basin.get("mass")) < min_mass:
            continue
        if require_greedy and safe_float(basin.get("greedy_member")) <= 0.0:
            continue
        answer = str(basin.get("answer", "")).strip()
        key = " ".join(answer.lower().split())
        if key in seen or not clean_guided_answer(answer, max_words):
            continue
        answers.append(answer)
        seen.add(key)
        if len(answers) >= max_candidates:
            break
    return answers


def build_candidate_batch(tokenizer, prompt: str, completions: list[str], device: str) -> dict[str, Any]:
    prompt_ids = tokenizer(prompt, return_tensors="pt")
    prompt_len = int(prompt_ids["input_ids"].shape[-1])
    full_texts = [prompt + completion for completion in completions]
    encoded = tokenizer(full_texts, return_tensors="pt", padding=True).to(device)
    token_counts = [max(1, int(tokenizer(completion, return_tensors="pt")["input_ids"].shape[-1])) for completion in completions]
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])),
        "prompt_len": prompt_len,
        "token_counts": token_counts,
    }


def sequence_logprobs(model, input_ids: torch.Tensor, prompt_len: int, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(token_logprobs)
    mask[:, max(prompt_len - 1, 0) :] = 1.0
    mask = mask * attention_mask[:, 1:].float()
    return (token_logprobs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def ref_sequence_logprobs(model, input_ids: torch.Tensor, prompt_len: int, attention_mask: torch.Tensor) -> torch.Tensor:
    context = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    with context:
        return sequence_logprobs(model, input_ids, prompt_len, attention_mask).detach()


def compute_anchor_rewards(
    completions: list[str],
    token_counts: list[int],
    teacher_basins: list[dict[str, Any]],
    ideals: list[str],
    reward_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    canonicals = [canonical_answer(text) for text in completions]
    basin_counts = Counter(canonicals)
    group_size = max(1, len(completions))
    rows: list[dict[str, Any]] = []
    for text, canonical, token_count in zip(completions, canonicals, token_counts):
        best_similarity = 0.0
        support_mass = 0.0
        correct_support_mass = 0.0
        greedy_support = 0.0
        for basin in teacher_basins:
            similarity = answer_similarity(text, str(basin["answer"]))
            best_similarity = max(best_similarity, similarity)
            if similarity >= float(reward_cfg.get("alignment_threshold", 0.80)):
                support_mass += safe_float(basin["mass"])
                correct_support_mass += safe_float(basin["mass"]) * safe_float(basin["strict_any"])
                greedy_support = max(greedy_support, safe_float(basin["greedy_member"]))
        support_mass = min(support_mass, 1.0)
        correct_support_mass = min(correct_support_mass, 1.0)
        qwen_basin_mass = basin_counts[canonical] / group_size
        qwen_only_stable = qwen_basin_mass * (1.0 - support_mass)
        anchored_consensus = qwen_basin_mass * support_mass
        length_cost = min(float(token_count) / max(1.0, float(reward_cfg.get("length_max_tokens", 48))), 1.0)
        reward = (
            float(reward_cfg.get("anchor_support", 1.0)) * support_mass
            + float(reward_cfg.get("teacher_similarity", 0.2)) * best_similarity
            + float(reward_cfg.get("teacher_correct_support", 0.0)) * correct_support_mass
            + float(reward_cfg.get("anchored_consensus", 0.2)) * anchored_consensus
            + float(reward_cfg.get("teacher_greedy_support", 0.0)) * greedy_support
            - float(reward_cfg.get("qwen_only_stable", 0.7)) * qwen_only_stable
            - float(reward_cfg.get("length", 0.04)) * length_cost
        )
        strict_value = float(strict_correct(text, ideals))
        rows.append(
            {
                "completion": text,
                "canonical_basin": canonical,
                "reward": reward,
                "anchor_support": support_mass,
                "teacher_similarity": best_similarity,
                "teacher_correct_support": correct_support_mass,
                "anchored_consensus": anchored_consensus,
                "qwen_basin_mass": qwen_basin_mass,
                "qwen_only_stable": qwen_only_stable,
                "teacher_greedy_support": greedy_support,
                "length_cost": length_cost,
                "strict": strict_value,
                "f1": best_f1(text, ideals),
            }
        )
    return rows


def summarize_reward_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "reward",
        "anchor_support",
        "teacher_similarity",
        "teacher_correct_support",
        "anchored_consensus",
        "qwen_basin_mass",
        "qwen_only_stable",
        "length_cost",
        "strict",
        "f1",
    ]
    return {f"mean_{key}": mean([float(row[key]) for row in rows]) for key in keys}


def evaluate_sample0(model, tokenizer, records: list[dict[str, Any]], teacher_basins: dict[str, list[dict[str, Any]]], reward_cfg: dict[str, Any], max_items: int, device: str) -> dict[str, Any]:
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
                max_new_tokens=int(reward_cfg.get("eval_max_new_tokens", 48)),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True).strip()
        reward_rows.extend(
            compute_anchor_rewards(
                [completion],
                [max(1, output.shape[-1] - prompt_len)],
                teacher_basins.get(record["id"], []),
                record["ideal_answers"],
                reward_cfg,
            )
        )
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
    records = load_records(config["input_jsonl"])
    teacher_basins = load_teacher_basins(config["teacher_basin_rows"])
    splits = split_anchor_records(records, teacher_basins, config)
    write_json(run_dir / "split_manifest.json", {name: [row["id"] for row in rows] for name, rows in splits.items()})

    model, tokenizer = load_model_and_tokenizer(config, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]))
    model.train()

    metrics_path = run_dir / "train_metrics.jsonl"
    samples_path = run_dir / "train_samples.jsonl"
    train_records = splits["train"]
    if not train_records:
        raise RuntimeError("No anchor training records found.")

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
            teacher_guided = guided_teacher_completions(teacher_basins.get(record["id"], []), train_cfg)
            completions = list(generated["completions"]) + teacher_guided
            candidate_batch = build_candidate_batch(tokenizer, prompt, completions, args.device)
            input_ids = candidate_batch["input_ids"]
            attention_mask = candidate_batch["attention_mask"]
            prompt_len = int(candidate_batch["prompt_len"])
            with torch.no_grad():
                old_logp = sequence_logprobs(model, input_ids, prompt_len, attention_mask).detach()
                ref_logp = ref_sequence_logprobs(model, input_ids, prompt_len, attention_mask)
            reward_rows = compute_anchor_rewards(
                completions,
                candidate_batch["token_counts"],
                teacher_basins.get(record["id"], []),
                record["ideal_answers"],
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
            losses.append(policy_loss + float(train_cfg["beta_kl"]) * kl)

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
                    "generated_completions": generated["completions"],
                    "teacher_guided_completions": teacher_guided,
                    "completions": completions,
                    "reward_rows": reward_rows,
                },
            )

        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
        optimizer.step()

        metrics = {
            "step": step,
            "loss": float(total_loss.detach().cpu().item()),
            "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
            "mean_advantage": mean(all_advantages),
            "mean_kl": mean(all_kl),
            "mean_ratio": mean(all_ratios),
            "max_ratio": max(all_ratios) if all_ratios else 0.0,
            "questions": len(step_records),
            **summarize_reward_rows(all_reward_rows),
        }
        if step % int(train_cfg["eval_every_steps"]) == 0:
            eval_items = min(int(train_cfg.get("eval_max_items", 32)), len(splits["dev"]))
            dev_summary = evaluate_sample0(model, tokenizer, splits["dev"], teacher_basins, config["reward"], max_items=eval_items, device=args.device)
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
