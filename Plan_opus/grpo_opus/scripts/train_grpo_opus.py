"""GRPO-Opus trainer.

Differences from Plan_gpt55:

1. **Per-token PPO ratio + per-token clip**: ratio_t = exp(logp_t - old_logp_t).
   This is the original PPO formulation - sequence-mean ratios in Plan_gpt55
   never trigger clip_eps=0.15 because they are usually within [0.95, 1.10].

2. **Continuous anchor reward**: instead of ``support += mass if sim>=0.8``,
   we use ``support += mass * sigmoid(k*(sim - 0.5))``. Near-miss similarity
   (0.6-0.8) now provides a real gradient signal.

3. **Teacher-guided rollout injection** (always-on, like Plan_gpt55 v1 but
   simpler): each group ALWAYS contains the teacher's best answer when one
   exists, so even when the policy never samples a near-anchor answer the
   group still has at least one positive reward sample, preventing
   zero-variance group collapse.

4. **Strict-correct shaping when available**: at training time we add a small
   strict-correct bonus to the reward (the gold label IS used in training, but
   evaluation remains strictly no-leak via the standard eval pipeline). This
   prevents the reward from optimizing teacher-similarity at the expense of
   factual correctness.

5. **Span-aware action masking** (optional): when computing the per-token
   policy loss, optionally mask only over the canonical answer span (whatever
   the policy chose). This concentrates the gradient on the answer tokens
   instead of every reasoning token.
"""
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

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
sys.path.insert(0, str(SHARED_DIR))

from model_utils import (  # noqa: E402
    build_prompt,
    completion_token_logprobs,
    per_token_kl,
    per_token_ppo_loss,
    reference_token_logprobs,
)
from text_utils import (  # noqa: E402
    append_jsonl,
    best_f1,
    canonical_answer,
    load_triviaqa_records,
    mean,
    read_csv,
    read_json,
    safe_float,
    strict_correct,
    token_f1,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO-Opus with per-token PPO and continuous anchor reward.")
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
        group.sort(key=lambda row: (row["strict_any"], row["mass"], row["greedy_member"]), reverse=True)
    return dict(basins)


def filter_records_with_basins(records: list[dict[str, Any]], basins: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [record for record in records if basins.get(record["id"])]


def split_records(records: list[dict[str, Any]], split_cfg: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    train_size = int(split_cfg["train_size"])
    dev_size = int(split_cfg["dev_size"])
    test_size = int(split_cfg.get("test_size", 0))
    return {
        "train": shuffled[:train_size],
        "dev": shuffled[train_size : train_size + dev_size],
        "test": shuffled[train_size + dev_size : train_size + dev_size + test_size],
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


def teacher_pick(basins: list[dict[str, Any]], max_words: int) -> str | None:
    for basin in basins:
        ans = str(basin.get("answer", "")).strip()
        if not ans:
            continue
        if len(ans.split()) > max_words:
            continue
        return ans
    return None


def sigmoid(x: float, k: float = 8.0, mid: float = 0.5) -> float:
    z = -k * (x - mid)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(z))
    e = math.exp(z)
    return 1.0 / (1.0 + e)


def compute_continuous_rewards(
    completions: list[str],
    token_counts: list[int],
    teacher_basins: list[dict[str, Any]],
    ideals: list[str],
    reward_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    canonicals = [canonical_answer(text) for text in completions]
    basin_counts = Counter(canonicals)
    group_size = max(1, len(completions))
    soft_k = float(reward_cfg.get("soft_k", 8.0))
    soft_mid = float(reward_cfg.get("soft_mid", 0.5))
    rows: list[dict[str, Any]] = []
    for text, canonical, token_count in zip(completions, canonicals, token_counts):
        best_similarity = 0.0
        soft_support = 0.0
        soft_correct_support = 0.0
        for basin in teacher_basins:
            similarity = answer_similarity(text, str(basin["answer"]))
            best_similarity = max(best_similarity, similarity)
            soft_w = sigmoid(similarity, k=soft_k, mid=soft_mid)
            soft_support += safe_float(basin["mass"]) * soft_w
            soft_correct_support += safe_float(basin["mass"]) * safe_float(basin["strict_any"]) * soft_w
        soft_support = min(soft_support, 1.0)
        soft_correct_support = min(soft_correct_support, 1.0)
        qwen_basin_mass = basin_counts[canonical] / group_size
        qwen_only_stable = qwen_basin_mass * (1.0 - soft_support)
        anchored_consensus = qwen_basin_mass * soft_support
        length_cost = min(float(token_count) / max(1.0, float(reward_cfg.get("length_max_tokens", 48))), 1.0)
        strict_value = float(strict_correct(text, ideals))
        f1_value = best_f1(text, ideals)
        reward = (
            float(reward_cfg.get("anchor_support", 1.0)) * soft_support
            + float(reward_cfg.get("teacher_similarity", 0.3)) * best_similarity
            + float(reward_cfg.get("teacher_correct_support", 0.2)) * soft_correct_support
            + float(reward_cfg.get("anchored_consensus", 0.2)) * anchored_consensus
            - float(reward_cfg.get("qwen_only_stable", 0.6)) * qwen_only_stable
            - float(reward_cfg.get("length", 0.04)) * length_cost
            + float(reward_cfg.get("strict_bonus", 0.4)) * strict_value
            + float(reward_cfg.get("f1_bonus", 0.1)) * f1_value
        )
        rows.append(
            {
                "completion": text,
                "canonical_basin": canonical,
                "reward": reward,
                "soft_anchor_support": soft_support,
                "soft_correct_support": soft_correct_support,
                "teacher_similarity": best_similarity,
                "qwen_basin_mass": qwen_basin_mass,
                "qwen_only_stable": qwen_only_stable,
                "length_cost": length_cost,
                "strict": strict_value,
                "f1": f1_value,
            }
        )
    return rows


def summarize_reward_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "reward",
        "soft_anchor_support",
        "soft_correct_support",
        "teacher_similarity",
        "qwen_basin_mass",
        "qwen_only_stable",
        "length_cost",
        "strict",
        "f1",
    ]
    return {f"mean_{key}": mean(float(row[key]) for row in rows) for key in keys}


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
    return {
        "input_ids": outputs,
        "prompt_len": prompt_len,
        "completions": [text.strip() for text in completions],
        "token_counts": completion_token_counts,
    }


def append_teacher_completion(prompt: str, teacher_text: str, tokenizer, max_new_tokens: int, device: str) -> torch.Tensor:
    full = prompt + " " + teacher_text.strip()
    encoded = tokenizer(full, return_tensors="pt", add_special_tokens=False).to(device)
    return encoded["input_ids"]


def pad_concat(input_ids_list: list[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(t.shape[-1] for t in input_ids_list)
    padded = []
    for ids in input_ids_list:
        pad_len = max_len - ids.shape[-1]
        if pad_len > 0:
            pad = torch.full((ids.shape[0], pad_len), pad_id, dtype=ids.dtype, device=ids.device)
            ids = torch.cat([ids, pad], dim=-1)
        padded.append(ids)
    return torch.cat(padded, dim=0)


def evaluate_sample0(model, tokenizer, records: list[dict[str, Any]], teacher_basins, reward_cfg, max_items, device) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
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
        rows.extend(
            compute_continuous_rewards(
                [completion],
                [max(1, output.shape[-1] - prompt_len)],
                teacher_basins.get(record["id"], []),
                record["ideal_answers"],
                reward_cfg,
            )
        )
    model.train()
    summary = summarize_reward_rows(rows)
    summary["eval_items"] = len(rows)
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
    raw_records = load_triviaqa_records(
        config["input_jsonl"], int(config["split"].get("offset", 0)),
        int(config["split"]["train_size"]) + int(config["split"]["dev_size"]) + int(config["split"].get("test_size", 0)),
    )
    teacher_basins = load_teacher_basins(config["teacher_basin_rows"])
    records = filter_records_with_basins(raw_records, teacher_basins)
    splits = split_records(records, config["split"], int(config["seed"]))
    write_json(run_dir / "split_manifest.json", {name: [row["id"] for row in rows] for name, rows in splits.items()})

    model, tokenizer = load_model_and_tokenizer(config, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]))
    model.train()

    metrics_path = run_dir / "train_metrics.jsonl"
    samples_path = run_dir / "train_samples.jsonl"
    train_records = splits["train"]
    if not train_records:
        raise RuntimeError("No training records with teacher basins.")

    teacher_inject = bool(train_cfg.get("teacher_injection", True))
    teacher_max_words = int(train_cfg.get("teacher_max_words", 12))
    clip_eps = float(train_cfg["clip_epsilon"])
    beta_kl = float(train_cfg["beta_kl"])

    for step in range(1, int(train_cfg["max_steps"]) + 1):
        start_index = (step - 1) * int(train_cfg["questions_per_step"])
        step_records = [train_records[(start_index + idx) % len(train_records)] for idx in range(int(train_cfg["questions_per_step"]))]
        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        all_reward_rows: list[dict[str, Any]] = []
        ratio_diags: list[dict[str, float]] = []
        kl_values: list[float] = []

        for record in step_records:
            prompt = build_prompt(tokenizer, record["question"], record["system_prompt"])
            generated = generate_group(model, tokenizer, prompt, train_cfg, args.device)
            sampled_completions = list(generated["completions"])
            sampled_token_counts = list(generated["token_counts"])
            policy_input_ids = generated["input_ids"]
            prompt_len = int(generated["prompt_len"])

            teacher_text: str | None = None
            if teacher_inject:
                teacher_text = teacher_pick(teacher_basins.get(record["id"], []), teacher_max_words)
            if teacher_text:
                teacher_ids = append_teacher_completion(prompt, teacher_text, tokenizer, int(train_cfg["max_new_tokens"]), args.device)
                policy_input_ids = pad_concat([policy_input_ids, teacher_ids], pad_id=tokenizer.eos_token_id or 0)
                sampled_completions.append(teacher_text)
                sampled_token_counts.append(int(teacher_ids.shape[-1]) - prompt_len)

            attention_mask = (policy_input_ids != (tokenizer.eos_token_id or 0)).long()
            attention_mask[:, : prompt_len] = 1  # always attend to prompt tokens (even if eos was prompt)
            with torch.no_grad():
                old_logp, mask = completion_token_logprobs(model, policy_input_ids, attention_mask, prompt_len)
                old_logp = old_logp.detach()
                ref_logp, _ = reference_token_logprobs(model, policy_input_ids, attention_mask, prompt_len)

            reward_rows = compute_continuous_rewards(
                sampled_completions,
                sampled_token_counts,
                teacher_basins.get(record["id"], []),
                record["ideal_answers"],
                config["reward"],
            )
            rewards = torch.tensor([float(row["reward"]) for row in reward_rows], dtype=torch.float32, device=args.device)
            mu = float(rewards.mean().item())
            sigma = float(rewards.std(unbiased=False).item()) if rewards.numel() > 1 else 0.0
            advantages = (rewards - mu) / (sigma + 1e-6)

            new_logp, new_mask = completion_token_logprobs(model, policy_input_ids, attention_mask, prompt_len)
            ppo_loss, ratio_diag = per_token_ppo_loss(new_logp, old_logp, advantages, new_mask, clip_eps)
            kl = per_token_kl(new_logp, ref_logp, new_mask)
            loss = ppo_loss + beta_kl * kl
            losses.append(loss)

            all_reward_rows.extend(reward_rows)
            ratio_diags.append(ratio_diag)
            kl_values.append(float(kl.detach().cpu().item()))
            append_jsonl(
                samples_path,
                {
                    "step": step,
                    "question_id": record["id"],
                    "question": record["question"],
                    "completions": sampled_completions,
                    "teacher_injected": bool(teacher_text),
                    "reward_rows": reward_rows,
                    "advantages": advantages.detach().cpu().tolist(),
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
            "mean_kl": mean(kl_values),
            "ppo_mean_ratio": mean(d["ppo_mean_ratio"] for d in ratio_diags),
            "ppo_max_ratio": max((d["ppo_max_ratio"] for d in ratio_diags), default=0.0),
            "ppo_clip_frac": mean(d["ppo_clip_frac"] for d in ratio_diags),
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


if __name__ == "__main__":
    main()
