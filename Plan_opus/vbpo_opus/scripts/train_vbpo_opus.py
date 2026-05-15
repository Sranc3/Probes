"""VBPO-Opus trainer.

Differences from Plan_gpt55:

- DPO uses **summed** logprob over the chosen-answer span (or full completion
  if span not located), not length-averaged. β is therefore re-tuned upward.
- Pairs are built externally by ``build_pairs.py`` with strict label hygiene.
- Reference logprobs use disable_adapter and are computed once per sample
  inside the same forward call structure.
- Adds an explicit ``span_only_loss`` mode (default) that limits DPO loss to
  the canonical answer span tokens, concentrating gradient on the factual
  answer rather than verbose surrounding text.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
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
    dpo_loss,
    dpo_pair_logits,
    encode_pair,
    locate_answer_span,
    reference_token_logprobs,
    sequence_logprob_sum,
)
from text_utils import append_jsonl, mean, read_csv, read_json, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VBPO-Opus DPO with span-only sum-logprob loss.")
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


def pair_dpo_loss(
    model,
    tokenizer,
    pair: dict[str, Any],
    beta: float,
    device: str,
    span_only: bool,
    use_mean: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    prompt = build_prompt(tokenizer, pair["question"], pair["system_prompt"])

    def encode_and_logp(completion: str, canonical_text: str | None) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        encoded = encode_pair(tokenizer, prompt, completion, device)
        token_logp, mask = completion_token_logprobs(model, encoded["input_ids"], encoded["attention_mask"], encoded["prompt_len"])
        ref_logp, ref_mask = reference_token_logprobs(model, encoded["input_ids"], encoded["attention_mask"], encoded["prompt_len"])
        span = None
        if span_only and canonical_text:
            span = locate_answer_span(
                tokenizer,
                prompt,
                completion,
                canonical_text,
                encoded["completion_len"],
                encoded["prompt_len"],
            )
        seq_logp = sequence_logprob_sum(token_logp, mask, span=span, use_mean=use_mean)
        ref_seq_logp = sequence_logprob_sum(ref_logp, ref_mask, span=span, use_mean=use_mean)
        return seq_logp, ref_seq_logp, {"span": span, "completion_len": encoded["completion_len"]}

    chosen_logp, ref_chosen_logp, chosen_meta = encode_and_logp(pair["chosen"], pair.get("chosen_canonical"))
    rejected_logp, ref_rejected_logp, rejected_meta = encode_and_logp(pair["rejected"], pair.get("rejected_canonical"))

    logits = dpo_pair_logits(chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp, beta)
    loss = dpo_loss(logits, weight=float(pair.get("weight", 1.0)))
    with torch.no_grad():
        margin = (chosen_logp - rejected_logp) - (ref_chosen_logp - ref_rejected_logp)
        diag = {
            "pi_chosen_logp": float(chosen_logp.detach().mean().item()),
            "pi_rejected_logp": float(rejected_logp.detach().mean().item()),
            "ref_chosen_logp": float(ref_chosen_logp.mean().item()),
            "ref_rejected_logp": float(ref_rejected_logp.mean().item()),
            "margin_delta": float(margin.detach().mean().item()),
            "logit": float(logits.detach().mean().item()),
            "chosen_span_len": float((chosen_meta["span"][1] - chosen_meta["span"][0]) if chosen_meta["span"] else chosen_meta["completion_len"]),
            "rejected_span_len": float((rejected_meta["span"][1] - rejected_meta["span"][0]) if rejected_meta["span"] else rejected_meta["completion_len"]),
        }
    return loss, diag


def evaluate_pairs(model, tokenizer, pairs: list[dict[str, Any]], beta: float, device: str, max_pairs: int, span_only: bool, use_mean: bool = False) -> dict[str, float]:
    model.eval()
    rows = []
    for pair in pairs[:max_pairs]:
        with torch.no_grad():
            loss, metrics = pair_dpo_loss(model, tokenizer, pair, beta, device, span_only, use_mean=use_mean)
        rows.append({"loss": float(loss.detach().cpu().item()), **metrics})
    model.train()
    return {
        "eval_pairs": len(rows),
        "eval_loss": mean([row["loss"] for row in rows]),
        "eval_margin_delta": mean([row["margin_delta"] for row in rows]),
        "eval_pair_accuracy": mean([float(row["margin_delta"] > 0.0) for row in rows]),
        "eval_logit": mean([row["logit"] for row in rows]),
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    train_cfg = config["training"]
    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))

    run_dir = setup_run(config, args.config)
    pair_dir = Path(config["pair_dir"])
    train_pairs = read_csv(pair_dir / "pairs_train.csv")
    dev_pairs = read_csv(pair_dir / "pairs_dev.csv")
    if not train_pairs:
        raise RuntimeError(f"No training pairs in {pair_dir}")

    # Coerce numeric fields back to floats
    for split in (train_pairs, dev_pairs):
        for pair in split:
            pair["weight"] = float(pair.get("weight", 1.0))

    random.Random(int(config["seed"])).shuffle(train_pairs)
    write_json(run_dir / "pair_manifest.json", {
        "train_pairs": len(train_pairs),
        "dev_pairs": len(dev_pairs),
        "train_questions": len({p["question_id"] for p in train_pairs}),
        "dev_questions": len({p["question_id"] for p in dev_pairs}),
        "mean_weight_train": mean(float(p["weight"]) for p in train_pairs),
        "mean_chosen_words": mean(len(str(p["chosen"]).split()) for p in train_pairs),
        "mean_rejected_words": mean(len(str(p["rejected"]).split()) for p in train_pairs),
    })

    model, tokenizer = load_model_and_tokenizer(config, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]))
    model.train()

    metrics_path = run_dir / "train_metrics.jsonl"
    span_only = bool(train_cfg.get("span_only_loss", True))
    use_mean = bool(train_cfg.get("use_mean_logprob", False))
    beta = float(train_cfg["beta"])

    for step in range(1, int(train_cfg["max_steps"]) + 1):
        batch = [
            train_pairs[((step - 1) * int(train_cfg["batch_size"]) + idx) % len(train_pairs)]
            for idx in range(int(train_cfg["batch_size"]))
        ]
        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        per_pair_metrics: list[dict[str, float]] = []
        for pair in batch:
            loss, diag = pair_dpo_loss(model, tokenizer, pair, beta, args.device, span_only, use_mean=use_mean)
            losses.append(loss)
            per_pair_metrics.append(diag)
        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
        optimizer.step()

        metrics = {
            "step": step,
            "loss": float(total_loss.detach().cpu().item()),
            "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
            "mean_margin_delta": mean(row["margin_delta"] for row in per_pair_metrics),
            "pair_accuracy": mean(float(row["margin_delta"] > 0.0) for row in per_pair_metrics),
            "mean_logit": mean(row["logit"] for row in per_pair_metrics),
            "mean_chosen_logp": mean(row["pi_chosen_logp"] for row in per_pair_metrics),
            "mean_rejected_logp": mean(row["pi_rejected_logp"] for row in per_pair_metrics),
            "mean_chosen_span_len": mean(row["chosen_span_len"] for row in per_pair_metrics),
            "mean_rejected_span_len": mean(row["rejected_span_len"] for row in per_pair_metrics),
            "batch_pairs": len(batch),
        }
        if dev_pairs and step % int(train_cfg["eval_every_steps"]) == 0:
            metrics.update({
                f"dev_{key}": value
                for key, value in evaluate_pairs(
                    model, tokenizer, dev_pairs, beta, args.device,
                    max_pairs=int(train_cfg.get("dev_eval_max_pairs", 64)),
                    span_only=span_only,
                    use_mean=use_mean,
                ).items()
            })
        append_jsonl(metrics_path, metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

        if step % int(train_cfg["save_every_steps"]) == 0 or step == int(train_cfg["max_steps"]):
            ckpt_dir = run_dir / "checkpoints" / f"step_{step:04d}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    write_json(run_dir / "run_complete.json", {"status": "complete", "run_dir": str(run_dir), "steps": int(train_cfg["max_steps"])})


if __name__ == "__main__":
    main()
