#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

GRPO_SCRIPT_DIR = Path(__file__).resolve().parents[1].parent / "basin_grpo" / "scripts"
sys.path.insert(0, str(GRPO_SCRIPT_DIR))

from grpo_utils import build_prompt, load_triviaqa_records, mean, read_json, split_records, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train verifier-guided basin preference optimization.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


DEFAULT_SCORE_TERMS = [
    {"feature": "logprob_avg_z", "weight": 0.45},
    {"feature": "token_mean_entropy_z", "weight": -0.35},
    {"feature": "token_count_z", "weight": -0.10},
    {"feature": "cluster_weight_mass_z", "weight": 0.35},
    {"feature": "cluster_size_z", "weight": 0.15},
]


def feature_score(row: dict[str, Any], terms: list[dict[str, Any]]) -> float:
    return sum(float(term["weight"]) * safe_float(row.get(str(term["feature"]), 0.0)) for term in terms)


def verifier_score(row: dict[str, Any], config: dict[str, Any] | None = None) -> float:
    config = config or {}
    return feature_score(row, list(config.get("score_terms", DEFAULT_SCORE_TERMS)))


def risk_score(row: dict[str, Any], config: dict[str, Any] | None = None) -> float:
    config = config or {}
    terms = list(
        config.get(
            "risk_terms",
            [
                {"feature": "cluster_weight_mass_z", "weight": 0.40},
                {"feature": "cluster_size_z", "weight": 0.30},
                {"feature": "logprob_avg_z", "weight": 0.25},
                {"feature": "token_mean_entropy_z", "weight": -0.20},
            ],
        )
    )
    return feature_score(row, terms)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def group_candidates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(float(row["sample_index"])))
    return dict(grouped)


def make_pair(
    record: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    pair_type: str,
    weight: float,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    return {
        "question_id": record["id"],
        "question": record["question"],
        "system_prompt": record["system_prompt"],
        "chosen": str(chosen["answer_text"]).strip(),
        "rejected": str(rejected["answer_text"]).strip(),
        "chosen_sample_index": int(float(chosen["sample_index"])),
        "rejected_sample_index": int(float(rejected["sample_index"])),
        "chosen_cluster_id": int(float(chosen["cluster_id"])),
        "rejected_cluster_id": int(float(rejected["cluster_id"])),
        "pair_type": pair_type,
        "weight": weight,
        "chosen_verifier_score": verifier_score(chosen, config),
        "rejected_verifier_score": verifier_score(rejected, config),
        "chosen_risk_score": risk_score(chosen, config),
        "rejected_risk_score": risk_score(rejected, config),
    }


def pair_weight(config: dict[str, Any], pair_type: str, default: float) -> float:
    weights = config.get("pair_weights", {})
    return float(weights.get(pair_type, default))


def dynamic_pair_weight(config: dict[str, Any], pair_type: str, chosen: dict[str, Any], rejected: dict[str, Any], default: float) -> float:
    weight = pair_weight(config, pair_type, default)
    dynamic_cfg = config.get("dynamic_weighting", {})
    if not dynamic_cfg.get("enabled", False):
        return weight
    hard_alpha = float(dynamic_cfg.get("hard_negative_alpha", 0.0))
    rescue_alpha = float(dynamic_cfg.get("rescue_confidence_alpha", 0.0))
    risk = sigmoid(risk_score(rejected, config))
    confidence_gap = sigmoid(verifier_score(chosen, config) - verifier_score(rejected, config))
    if pair_type in {"damage_guard", "correct_vs_stable_wrong"}:
        weight *= 1.0 + hard_alpha * risk
    if pair_type == "rescue_from_sample0":
        weight *= 1.0 + rescue_alpha * confidence_gap
    max_weight = float(dynamic_cfg.get("max_weight", weight))
    return min(weight, max_weight)


def build_pairs(
    records: list[dict[str, Any]],
    candidate_groups: dict[str, list[dict[str, Any]]],
    max_pairs_per_question: int,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config = config or {}
    pairs: list[dict[str, Any]] = []
    for record in records:
        group = candidate_groups.get(record["id"], [])
        if not group:
            continue
        correct = [row for row in group if safe_float(row.get("strict_correct")) > 0.0]
        wrong = [row for row in group if safe_float(row.get("strict_correct")) <= 0.0]
        if not correct or not wrong:
            continue
        sample0 = next((row for row in group if int(float(row["sample_index"])) == 0), group[0])
        correct_sorted = sorted(correct, key=lambda row: verifier_score(row, config), reverse=True)
        wrong_sorted = sorted(wrong, key=lambda row: (risk_score(row, config), verifier_score(row, config)), reverse=True)
        local_pairs: list[dict[str, Any]] = []
        if safe_float(sample0.get("strict_correct")) > 0.0:
            for rejected in wrong_sorted[:max_pairs_per_question]:
                weight = dynamic_pair_weight(config, "damage_guard", sample0, rejected, 1.5)
                local_pairs.append(make_pair(record, sample0, rejected, "damage_guard", weight, config))
        else:
            correct_limit = int(config.get("max_correct_chosen_per_question", 2))
            for best_correct in correct_sorted[:correct_limit]:
                weight = dynamic_pair_weight(config, "rescue_from_sample0", best_correct, sample0, 1.7)
                local_pairs.append(make_pair(record, best_correct, sample0, "rescue_from_sample0", weight, config))
                for rejected in wrong_sorted:
                    if int(float(rejected["sample_index"])) != 0:
                        weight = dynamic_pair_weight(config, "correct_vs_stable_wrong", best_correct, rejected, 1.2)
                        local_pairs.append(
                            make_pair(record, best_correct, rejected, "correct_vs_stable_wrong", weight, config)
                        )
                    if len(local_pairs) >= max_pairs_per_question:
                        break
                if len(local_pairs) >= max_pairs_per_question:
                    break
        pairs.extend(local_pairs[:max_pairs_per_question])
    return pairs


def sequence_logprob(model, tokenizer, prompt: str, completion: str, device: str) -> torch.Tensor:
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)
    full_text = prompt + completion
    encoded = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    prompt_len = int(prompt_ids["input_ids"].shape[-1])
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(token_logprobs)
    mask[:, max(prompt_len - 1, 0) :] = 1.0
    mask = mask * attention_mask[:, 1:].float()
    return (token_logprobs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def ref_logprob(model, tokenizer, prompt: str, completion: str, device: str) -> torch.Tensor:
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            return sequence_logprob(model, tokenizer, prompt, completion, device).detach()
    return sequence_logprob(model, tokenizer, prompt, completion, device).detach()


def pair_loss(model, tokenizer, pair: dict[str, Any], beta: float, device: str) -> tuple[torch.Tensor, dict[str, float]]:
    prompt = build_prompt(tokenizer, pair["question"], pair["system_prompt"])
    chosen_logp = sequence_logprob(model, tokenizer, prompt, pair["chosen"], device)
    rejected_logp = sequence_logprob(model, tokenizer, prompt, pair["rejected"], device)
    with torch.no_grad():
        ref_chosen = ref_logprob(model, tokenizer, prompt, pair["chosen"], device)
        ref_rejected = ref_logprob(model, tokenizer, prompt, pair["rejected"], device)
    pi_margin = chosen_logp - rejected_logp
    ref_margin = ref_chosen - ref_rejected
    logits = float(beta) * (pi_margin - ref_margin)
    loss = -F.logsigmoid(logits).mean() * float(pair["weight"])
    return loss, {
        "pi_margin": float(pi_margin.detach().cpu().item()),
        "ref_margin": float(ref_margin.detach().cpu().item()),
        "margin_delta": float((pi_margin - ref_margin).detach().cpu().item()),
    }


def evaluate_pairs(model, tokenizer, pairs: list[dict[str, Any]], beta: float, device: str, max_pairs: int = 64) -> dict[str, float]:
    model.eval()
    rows = []
    for pair in pairs[:max_pairs]:
        with torch.no_grad():
            loss, metrics = pair_loss(model, tokenizer, pair, beta, device)
        rows.append({"loss": float(loss.detach().cpu().item()), **metrics})
    model.train()
    return {
        "eval_pairs": len(rows),
        "eval_loss": mean([row["loss"] for row in rows]),
        "eval_margin_delta": mean([row["margin_delta"] for row in rows]),
        "eval_pair_accuracy": mean([float(row["margin_delta"] > 0.0) for row in rows]),
    }


def count_by_type(pairs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for pair in pairs:
        counts[str(pair["pair_type"])] += 1
    return dict(counts)


def make_balanced_batch(
    buckets: dict[str, list[dict[str, Any]]],
    step: int,
    batch_size: int,
    pair_type_order: list[str],
) -> list[dict[str, Any]]:
    available = [pair_type for pair_type in pair_type_order if buckets.get(pair_type)]
    if not available:
        return []
    batch: list[dict[str, Any]] = []
    cursor = step - 1
    while len(batch) < batch_size:
        pair_type = available[(cursor + len(batch)) % len(available)]
        bucket = buckets[pair_type]
        item_idx = ((step - 1) * batch_size + len(batch)) // len(available)
        batch.append(bucket[item_idx % len(bucket)])
    return batch


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    train_cfg = config["training"]
    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))
    run_dir = setup_run(config, args.config)

    total_records = int(config["split"]["train_size"]) + int(config["split"]["dev_size"]) + int(config["split"]["test_size"])
    records = load_triviaqa_records(config["input_jsonl"], int(config["split"].get("offset", 0)), total_records, int(config["seed"]))
    splits = split_records(records, config["split"])
    write_json(run_dir / "split_manifest.json", {name: [row["id"] for row in rows] for name, rows in splits.items()})

    candidate_groups = group_candidates(read_csv(config["candidate_features"]))
    max_pairs = int(train_cfg.get("max_pairs_per_question", 3))
    pair_build_cfg = config.get("pair_building", {})
    pair_splits = {name: build_pairs(rows, candidate_groups, max_pairs, pair_build_cfg) for name, rows in splits.items()}
    for pairs in pair_splits.values():
        random.shuffle(pairs)
    write_json(
        run_dir / "pair_manifest.json",
        {
            name: {"pairs": len(pairs), "pair_types": count_by_type(pairs)}
            for name, pairs in pair_splits.items()
        },
    )
    if not pair_splits["train"]:
        raise RuntimeError("No preference pairs were built for training.")

    model, tokenizer = load_model_and_tokenizer(config, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]))
    model.train()

    metrics_path = run_dir / "train_metrics.jsonl"
    train_pairs = pair_splits["train"]
    dev_pairs = pair_splits["dev"]
    pair_type_order = list(train_cfg.get("pair_type_order", ["damage_guard", "rescue_from_sample0", "correct_vs_stable_wrong"]))
    train_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in train_pairs:
        train_buckets[str(pair["pair_type"])].append(pair)
    for step in range(1, int(train_cfg["max_steps"]) + 1):
        if train_cfg.get("balanced_pair_batches", False):
            batch = make_balanced_batch(train_buckets, step, int(train_cfg["batch_size"]), pair_type_order)
        else:
            batch = [train_pairs[((step - 1) * int(train_cfg["batch_size"]) + idx) % len(train_pairs)] for idx in range(int(train_cfg["batch_size"]))]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        per_pair_metrics = []
        for pair in batch:
            loss, metrics = pair_loss(model, tokenizer, pair, float(train_cfg["beta"]), args.device)
            losses.append(loss)
            per_pair_metrics.append(metrics)
        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
        optimizer.step()

        metrics_row = {
            "step": step,
            "loss": float(total_loss.detach().cpu().item()),
            "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
            "mean_pi_margin": mean([row["pi_margin"] for row in per_pair_metrics]),
            "mean_ref_margin": mean([row["ref_margin"] for row in per_pair_metrics]),
            "mean_margin_delta": mean([row["margin_delta"] for row in per_pair_metrics]),
            "pair_accuracy": mean([float(row["margin_delta"] > 0.0) for row in per_pair_metrics]),
            "batch_pairs": len(batch),
        }
        if step % int(train_cfg["eval_every_steps"]) == 0:
            metrics_row.update({f"dev_{key}": value for key, value in evaluate_pairs(model, tokenizer, dev_pairs, float(train_cfg["beta"]), args.device).items()})
        append_jsonl(metrics_path, metrics_row)
        print(json.dumps(metrics_row, ensure_ascii=False), flush=True)

        if step % int(train_cfg["save_every_steps"]) == 0 or step == int(train_cfg["max_steps"]):
            ckpt_dir = run_dir / "checkpoints" / f"step_{step:04d}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    write_json(run_dir / "run_complete.json", {"status": "complete", "run_dir": str(run_dir), "steps": int(train_cfg["max_steps"])})
    print(json.dumps({"run_dir": str(run_dir), "status": "complete"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
