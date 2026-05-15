#!/usr/bin/env python3
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

VBPO_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "verifier_bpo" / "scripts"
sys.path.insert(0, str(VBPO_SCRIPT_DIR))

from train_verifier_bpo import (  # noqa: E402
    append_jsonl,
    count_by_type,
    evaluate_pairs,
    load_model_and_tokenizer,
    make_balanced_batch,
    pair_loss,
    read_csv,
    safe_float,
)

GRPO_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "basin_grpo" / "scripts"
sys.path.insert(0, str(GRPO_SCRIPT_DIR))

from grpo_utils import mean, read_json, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Anchor-aware VBPO from cross-model anchor rows.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def setup_run(config: dict[str, Any], config_path: str) -> Path:
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['tag']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config_snapshot.json")
    (run_dir / "checkpoints").mkdir()
    return run_dir


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("question_id", "")), " ".join(str(row.get("answer_text", "")).strip().lower().split()))


def anchor_chosen_score(row: dict[str, Any], config: dict[str, Any]) -> float:
    score_cfg = config.get("anchor_scoring", {})
    return (
        float(score_cfg.get("anchor_score_weight", 0.55)) * safe_float(row.get("anchor_score_noleak"))
        + float(score_cfg.get("teacher_support_weight", 0.55)) * safe_float(row.get("teacher_support_mass"))
        + float(score_cfg.get("teacher_similarity_weight", 0.25)) * safe_float(row.get("teacher_best_similarity"))
        + float(score_cfg.get("student_verifier_weight", 0.10)) * safe_float(row.get("verifier_score_v05"))
    )


def stable_wrong_risk(row: dict[str, Any], config: dict[str, Any]) -> float:
    score_cfg = config.get("anchor_scoring", {})
    return (
        float(score_cfg.get("qwen_only_stable_weight", 0.60)) * safe_float(row.get("qwen_only_stable_mass"))
        + float(score_cfg.get("student_cluster_mass_weight", 0.35)) * safe_float(row.get("cluster_weight_mass_minmax"))
        + float(score_cfg.get("student_verifier_risk_weight", 0.15)) * safe_float(row.get("verifier_score_v05"))
        + float(score_cfg.get("unsupported_penalty_weight", 0.35)) * (1.0 - safe_float(row.get("teacher_support_mass")))
    )


def dedupe_best(rows: list[dict[str, Any]], score_fn, reverse: bool = True) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = row_key(row)
        if key not in best or score_fn(row) > score_fn(best[key]):
            best[key] = row
    return sorted(best.values(), key=score_fn, reverse=reverse)


def is_anchor_supported(row: dict[str, Any], config: dict[str, Any]) -> bool:
    selection = config.get("pair_building", {})
    return (
        safe_float(row.get("teacher_support_mass")) >= float(selection.get("min_teacher_support_mass", 0.10))
        or safe_float(row.get("teacher_best_similarity")) >= float(selection.get("min_teacher_similarity", 0.55))
        or safe_float(row.get("anchor_score_noleak")) >= float(selection.get("min_anchor_score_noleak", 0.15))
    )


def dynamic_anchor_weight(pair_type: str, chosen: dict[str, Any], rejected: dict[str, Any], config: dict[str, Any]) -> float:
    pair_cfg = config.get("pair_building", {})
    weights = pair_cfg.get("pair_weights", {})
    base = float(weights.get(pair_type, 1.0))
    support = safe_float(chosen.get("teacher_support_mass"))
    rejected_stability = safe_float(rejected.get("qwen_only_stable_mass"))
    multiplier = 1.0 + float(pair_cfg.get("chosen_support_alpha", 0.40)) * support
    multiplier += float(pair_cfg.get("rejected_stability_alpha", 0.30)) * rejected_stability
    return min(base * multiplier, float(pair_cfg.get("max_pair_weight", 2.8)))


def word_count(text: str) -> int:
    return len(str(text).strip().split())


def normalize_for_pair(text: str) -> list[str]:
    import re

    cleaned = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    return cleaned.split()


def answer_similarity_for_pair(left: str, right: str) -> float:
    left_tokens = normalize_for_pair(left)
    right_tokens = normalize_for_pair(right)
    if not left_tokens or not right_tokens:
        return 0.0
    left_norm = " ".join(left_tokens)
    right_norm = " ".join(right_tokens)
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        return 0.95
    left_counts: dict[str, int] = defaultdict(int)
    right_counts: dict[str, int] = defaultdict(int)
    for token in left_tokens:
        left_counts[token] += 1
    for token in right_tokens:
        right_counts[token] += 1
    overlap = sum(min(left_counts[token], right_counts[token]) for token in left_counts)
    if overlap <= 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2.0 * precision * recall / max(precision + recall, 1e-8)


def is_clean_completion(text: str, config: dict[str, Any], side: str = "chosen") -> bool:
    pair_cfg = config.get("pair_building", {})
    text = str(text).strip()
    if not text:
        return False
    max_words_key = "max_chosen_words" if side == "chosen" else "max_rejected_words"
    if word_count(text) > int(pair_cfg.get(max_words_key, 32 if side == "chosen" else 80)):
        return False
    lowered = text.lower()
    for marker in pair_cfg.get("bad_completion_markers", ["analysis", "however", "let's", "we need", "assistant", "final"]):
        if str(marker).lower() in lowered:
            return False
    if pair_cfg.get("reject_unfinished_completions", True) and text.rstrip().endswith((",", "and", "or", "th", "the")):
        return False
    return True


def chosen_completion_text(row: dict[str, Any], config: dict[str, Any]) -> tuple[str, str]:
    pair_cfg = config.get("pair_building", {})
    raw = str(row.get("answer_text", "")).strip()
    teacher = str(row.get("teacher_best_answer", "")).strip()
    if is_clean_completion(raw, config, "chosen"):
        return raw, "student_candidate"
    if pair_cfg.get("use_teacher_answer_for_chosen", True) and is_clean_completion(teacher, config, "chosen"):
        return teacher, "teacher_anchor"
    return raw, "student_candidate_unclean"


def teacher_completion_text(row: dict[str, Any], config: dict[str, Any]) -> tuple[str, str]:
    teacher = str(row.get("teacher_best_answer", "")).strip()
    if is_clean_completion(teacher, config, "chosen"):
        return teacher, "teacher_anchor"
    return chosen_completion_text(row, config)


def usable_chosen(row: dict[str, Any], config: dict[str, Any]) -> bool:
    text, source = chosen_completion_text(row, config)
    if source.endswith("unclean"):
        return False
    return is_clean_completion(text, config, "chosen")


def usable_rejected(row: dict[str, Any], config: dict[str, Any]) -> bool:
    return is_clean_completion(str(row.get("answer_text", "")).strip(), config, "rejected")


def make_anchor_pair(
    question_id: str,
    question: str,
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    pair_type: str,
    weight: float,
    config: dict[str, Any],
    chosen_text: str | None = None,
    rejected_text: str | None = None,
    chosen_source: str = "student_candidate",
    rejected_source: str = "student_candidate",
) -> dict[str, Any]:
    chosen_completion = str(chosen_text if chosen_text is not None else chosen["answer_text"]).strip()
    rejected_completion = str(rejected_text if rejected_text is not None else rejected["answer_text"]).strip()
    return {
        "question_id": question_id,
        "question": question,
        "system_prompt": config.get("system_prompt", "Answer the question briefly and factually."),
        "chosen": chosen_completion,
        "rejected": rejected_completion,
        "chosen_source": chosen_source,
        "rejected_source": rejected_source,
        "chosen_sample_index": int(float(chosen.get("sample_index", -1))),
        "rejected_sample_index": int(float(rejected.get("sample_index", -1))),
        "chosen_seed": int(float(chosen.get("seed", -1))),
        "rejected_seed": int(float(rejected.get("seed", -1))),
        "chosen_cluster_id": int(float(chosen.get("cluster_id", -1))),
        "rejected_cluster_id": int(float(rejected.get("cluster_id", -1))),
        "pair_type": pair_type,
        "weight": weight,
        "chosen_anchor_score": anchor_chosen_score(chosen, config),
        "rejected_anchor_score": anchor_chosen_score(rejected, config),
        "chosen_teacher_support_mass": safe_float(chosen.get("teacher_support_mass")),
        "rejected_teacher_support_mass": safe_float(rejected.get("teacher_support_mass")),
        "chosen_qwen_only_stable_mass": safe_float(chosen.get("qwen_only_stable_mass")),
        "rejected_qwen_only_stable_mass": safe_float(rejected.get("qwen_only_stable_mass")),
        "rejected_stable_wrong_risk": stable_wrong_risk(rejected, config),
        "chosen_strict_correct": safe_float(chosen.get("strict_correct")),
        "rejected_strict_correct": safe_float(rejected.get("strict_correct")),
        "chosen_words": word_count(chosen_completion),
        "rejected_words": word_count(rejected_completion),
    }


def group_anchor_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    return dict(grouped)


def split_question_ids(question_ids: list[str], config: dict[str, Any]) -> dict[str, list[str]]:
    split_cfg = config["split"]
    shuffled = list(question_ids)
    random.Random(int(config["seed"])).shuffle(shuffled)
    train_size = int(split_cfg["train_questions"])
    dev_size = int(split_cfg["dev_questions"])
    return {
        "train": shuffled[:train_size],
        "dev": shuffled[train_size : train_size + dev_size],
    }


def build_anchor_pairs_for_group(question_id: str, group: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    pair_cfg = config.get("pair_building", {})
    max_pairs = int(pair_cfg.get("max_pairs_per_question", 4))
    question = str(group[0]["question"])
    correct = [row for row in group if safe_float(row.get("strict_correct")) > 0.0]
    wrong = [row for row in group if safe_float(row.get("strict_correct")) <= 0.0]
    if not group:
        return []

    supported_correct = [row for row in correct if is_anchor_supported(row, config) and usable_chosen(row, config)]
    if not supported_correct and pair_cfg.get("allow_best_correct_fallback", True):
        supported_correct = [row for row in correct if usable_chosen(row, config)]

    correct_sorted = dedupe_best(supported_correct, lambda row: anchor_chosen_score(row, config))
    wrong_sorted = dedupe_best([row for row in wrong if usable_rejected(row, config)], lambda row: stable_wrong_risk(row, config))
    sample0_correct = [row for row in correct_sorted if safe_float(row.get("is_sample0")) > 0.0]
    sample0_wrong = dedupe_best(
        [row for row in wrong if safe_float(row.get("is_sample0")) > 0.0],
        lambda row: stable_wrong_risk(row, config),
    )
    anchored_candidates = dedupe_best(
        [row for row in group if is_anchor_supported(row, config) and usable_chosen(row, config)],
        lambda row: anchor_chosen_score(row, config),
    )
    correct_teacher_rows = dedupe_best(
        [
            row
            for row in group
            if safe_float(row.get("teacher_best_basin_strict_any")) > 0.0
            and safe_float(row.get("teacher_support_mass")) > 0.0
            and is_clean_completion(str(row.get("teacher_best_answer", "")).strip(), config, "chosen")
        ],
        lambda row: anchor_chosen_score(row, config),
    )
    student_only_candidates = dedupe_best(
        [
            row
            for row in group
            if safe_float(row.get("teacher_support_mass")) <= float(pair_cfg.get("max_rejected_teacher_support_mass", 0.0))
            and safe_float(row.get("qwen_only_stable_mass")) >= float(pair_cfg.get("min_rejected_qwen_only_stable_mass", 0.10))
            and usable_rejected(row, config)
        ],
        lambda row: stable_wrong_risk(row, config),
    )

    pairs: list[dict[str, Any]] = []

    if correct_sorted:
        for rejected in sample0_wrong[: int(pair_cfg.get("max_sample0_rescues", 2))]:
            chosen = correct_sorted[0]
            chosen_text, chosen_source = chosen_completion_text(chosen, config)
            weight = dynamic_anchor_weight("anchor_rescue", chosen, rejected, config)
            pairs.append(
                make_anchor_pair(
                    question_id,
                    question,
                    chosen,
                    rejected,
                    "anchor_rescue",
                    weight,
                    config,
                    chosen_text=chosen_text,
                    chosen_source=chosen_source,
                )
            )

    if wrong_sorted:
        for chosen in sample0_correct[: int(pair_cfg.get("max_damage_guard_chosen", 1))]:
            for rejected in wrong_sorted[: int(pair_cfg.get("max_damage_guard_rejected", 2))]:
                chosen_text, chosen_source = chosen_completion_text(chosen, config)
                weight = dynamic_anchor_weight("anchor_damage_guard", chosen, rejected, config)
                pairs.append(
                    make_anchor_pair(
                        question_id,
                        question,
                        chosen,
                        rejected,
                        "anchor_damage_guard",
                        weight,
                        config,
                        chosen_text=chosen_text,
                        chosen_source=chosen_source,
                    )
                )

    if correct_sorted and wrong_sorted:
        for chosen in correct_sorted[: int(pair_cfg.get("max_correct_chosen", 1))]:
            for rejected in wrong_sorted:
                if row_key(chosen) == row_key(rejected):
                    continue
                chosen_text, chosen_source = chosen_completion_text(chosen, config)
                weight = dynamic_anchor_weight("anchor_correct_vs_qwen_only_stable_wrong", chosen, rejected, config)
                pairs.append(
                    make_anchor_pair(
                        question_id,
                        question,
                        chosen,
                        rejected,
                        "anchor_correct_vs_qwen_only_stable_wrong",
                        weight,
                        config,
                        chosen_text=chosen_text,
                        chosen_source=chosen_source,
                    )
                )
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break

    if pair_cfg.get("enable_teacher_anchor_pairs", True):
        min_gap = float(pair_cfg.get("min_teacher_anchor_pair_score_gap", 0.25))
        for chosen in anchored_candidates[: int(pair_cfg.get("max_teacher_anchor_chosen", 2))]:
            for rejected in student_only_candidates[: int(pair_cfg.get("max_teacher_anchor_rejected", 3))]:
                if len(pairs) >= max_pairs:
                    break
                if row_key(chosen) == row_key(rejected):
                    continue
                if safe_float(chosen.get("strict_correct")) <= 0.0 and safe_float(rejected.get("strict_correct")) > 0.0:
                    continue
                if pair_cfg.get("teacher_anchor_pairs_require_strict", False) and (
                    safe_float(chosen.get("strict_correct")) <= 0.0 or safe_float(rejected.get("strict_correct")) > 0.0
                ):
                    continue
                if anchor_chosen_score(chosen, config) - anchor_chosen_score(rejected, config) < min_gap:
                    continue
                chosen_text, chosen_source = chosen_completion_text(chosen, config)
                weight = dynamic_anchor_weight("teacher_anchor_vs_student_only", chosen, rejected, config)
                pairs.append(
                    make_anchor_pair(
                        question_id,
                        question,
                        chosen,
                        rejected,
                        "teacher_anchor_vs_student_only",
                        weight,
                        config,
                        chosen_text=chosen_text,
                        chosen_source=chosen_source,
                    )
                )

    if pair_cfg.get("enable_teacher_correct_rescue", False):
        for chosen in correct_teacher_rows[: int(pair_cfg.get("max_teacher_correct_chosen", 2))]:
            chosen_text, chosen_source = teacher_completion_text(chosen, config)
            for rejected in student_only_candidates[: int(pair_cfg.get("max_teacher_correct_rejected", 3))]:
                if len(pairs) >= max_pairs:
                    break
                if safe_float(rejected.get("strict_correct")) > 0.0:
                    continue
                if answer_similarity_for_pair(chosen_text, str(rejected.get("answer_text", ""))) >= float(
                    pair_cfg.get("max_teacher_rejected_similarity", 0.80)
                ):
                    continue
                weight = dynamic_anchor_weight("teacher_correct_anchor_rescue", chosen, rejected, config)
                pairs.append(
                    make_anchor_pair(
                        question_id,
                        question,
                        chosen,
                        rejected,
                        "teacher_correct_anchor_rescue",
                        weight,
                        config,
                        chosen_text=chosen_text,
                        chosen_source=chosen_source,
                    )
                )

    return pairs[:max_pairs]


def build_pair_splits(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    groups = group_anchor_rows(rows)
    split_ids = split_question_ids(sorted(groups), config)
    pair_splits = {
        split_name: [
            pair
            for question_id in question_ids
            for pair in build_anchor_pairs_for_group(question_id, groups[question_id], config)
        ]
        for split_name, question_ids in split_ids.items()
    }
    for pairs in pair_splits.values():
        random.shuffle(pairs)
    return pair_splits, split_ids


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pairs": len(pairs),
        "questions": len({pair["question_id"] for pair in pairs}),
        "pair_types": count_by_type(pairs),
        "mean_weight": mean([float(pair["weight"]) for pair in pairs]),
        "mean_chosen_teacher_support": mean([float(pair["chosen_teacher_support_mass"]) for pair in pairs]),
        "mean_rejected_teacher_support": mean([float(pair["rejected_teacher_support_mass"]) for pair in pairs]),
        "mean_rejected_qwen_only_stable": mean([float(pair["rejected_qwen_only_stable_mass"]) for pair in pairs]),
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
    rows = read_csv(config["anchor_rows"])
    pair_splits, split_ids = build_pair_splits(rows, config)
    write_json(run_dir / "split_manifest.json", split_ids)
    write_json(run_dir / "pair_manifest.json", {name: summarize_pairs(pairs) for name, pairs in pair_splits.items()})
    write_json(run_dir / "train_pairs_preview.json", pair_splits["train"][:20])
    print(json.dumps({"run_dir": str(run_dir), "pair_manifest": {name: summarize_pairs(pairs) for name, pairs in pair_splits.items()}}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        write_json(run_dir / "run_complete.json", {"status": "dry_run", "run_dir": str(run_dir)})
        return
    if not pair_splits["train"]:
        raise RuntimeError("No Anchor-aware preference pairs were built for training.")

    model, tokenizer = load_model_and_tokenizer(config, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]))
    model.train()

    train_pairs = pair_splits["train"]
    dev_pairs = pair_splits["dev"]
    train_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in train_pairs:
        train_buckets[str(pair["pair_type"])].append(pair)

    pair_type_order = list(
        train_cfg.get(
            "pair_type_order",
            ["anchor_rescue", "anchor_damage_guard", "anchor_correct_vs_qwen_only_stable_wrong"],
        )
    )
    metrics_path = run_dir / "train_metrics.jsonl"
    for step in range(1, int(train_cfg["max_steps"]) + 1):
        if train_cfg.get("balanced_pair_batches", True):
            batch = make_balanced_batch(train_buckets, step, int(train_cfg["batch_size"]), pair_type_order)
        else:
            batch = [
                train_pairs[((step - 1) * int(train_cfg["batch_size"]) + idx) % len(train_pairs)]
                for idx in range(int(train_cfg["batch_size"]))
            ]
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
            metrics_row.update(
                {
                    f"dev_{key}": value
                    for key, value in evaluate_pairs(
                        model,
                        tokenizer,
                        dev_pairs,
                        float(train_cfg["beta"]),
                        args.device,
                    ).items()
                }
            )
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
