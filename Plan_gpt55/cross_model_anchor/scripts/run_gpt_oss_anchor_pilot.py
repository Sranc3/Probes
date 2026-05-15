#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_DIR = "/zhutingqi/gpt-oss-120b"
DEFAULT_INPUT_JSONL = "/zhutingqi/song/datasets/trivia_qa/processed/test.full.jsonl"
DEFAULT_CANDIDATE_FEATURES = (
    "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/"
    "run_20260511_100502_candidate_space_triviaqa_scale500_qwen25/candidate_features.csv"
)
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/cross_model_anchor/runs"
STOPWORDS = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with", "by"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-model factual anchor pilot with gpt-oss-120b.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--candidate-features", default=DEFAULT_CANDIDATE_FEATURES)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="gptoss_anchor_triviaqa_pilot100_k4")
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--samples-per-question", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--alignment-threshold", type=float, default=0.8)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: flatten(row.get(key, "")) for key in fieldnames})


def flatten(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def normalize_text(text: str) -> str:
    lowered = str(text).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(cleaned.split())


def canonical_answer(text: str) -> str:
    normalized = normalize_text(text)
    tokens = [token for token in normalized.split() if token not in STOPWORDS]
    return " ".join(tokens[:12])


def exact_match(answer: str, ideals: list[str]) -> bool:
    normalized = normalize_text(answer)
    return any(normalized and normalized == normalize_text(ideal) for ideal in ideals)


def contains_match(answer: str, ideals: list[str]) -> bool:
    normalized = normalize_text(answer)
    return any(normalize_text(ideal) and normalize_text(ideal) in normalized for ideal in ideals)


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def best_f1(answer: str, ideals: list[str]) -> float:
    return max([token_f1(answer, ideal) for ideal in ideals] or [0.0])


def answer_similarity(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        return 0.95
    return token_f1(left, right)


def verifier_score(row: dict[str, Any]) -> float:
    return (
        0.32 * safe_float(row.get("logprob_avg_z"))
        - 0.22 * safe_float(row.get("token_mean_entropy_z"))
        - 0.08 * safe_float(row.get("token_max_entropy_z"))
        - 0.06 * safe_float(row.get("token_count_z"))
        + 0.28 * safe_float(row.get("cluster_weight_mass_z"))
        + 0.12 * safe_float(row.get("cluster_size_z"))
        + 0.08 * safe_float(row.get("cluster_size_minmax"))
        + 0.08 * safe_float(row.get("cluster_weight_mass_minmax"))
    )


def auc_positive_high(rows: list[dict[str, Any]], score_key: str, label_key: str) -> float:
    positives = [safe_float(row[score_key]) for row in rows if safe_float(row[label_key]) > 0.0]
    negatives = [safe_float(row[score_key]) for row in rows if safe_float(row[label_key]) <= 0.0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            wins += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
            total += 1
    return wins / max(1, total)


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for index, line in enumerate(file_obj):
            payload = json.loads(line)
            ideals = payload.get("ideal") or payload.get("ideal_answers") or []
            records[str(payload["id"])] = {
                "question_index": index,
                "question_id": str(payload["id"]),
                "question": str(payload["question"]),
                "ideal_answers": [str(item) for item in ideals],
            }
    return records


def load_candidate_groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            row["verifier_score_v05"] = verifier_score(row)
            groups[str(row["question_id"])].append(row)
    for group in groups.values():
        group.sort(key=lambda row: int(float(row["sample_index"])))
    return dict(groups)


def select_question_ids(groups: dict[str, list[dict[str, Any]]], records: dict[str, dict[str, Any]], count: int, seed: int) -> list[str]:
    qids = [qid for qid in groups if qid in records]
    qids.sort(key=lambda qid: int(groups[qid][0].get("question_index", 0)))
    rng = random.Random(seed)
    rng.shuffle(qids)
    return qids[:count]


def parse_harmony(raw_text: str) -> dict[str, str]:
    analysis = ""
    final = ""
    analysis_match = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)", raw_text, flags=re.S)
    final_match = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", raw_text, flags=re.S)
    if analysis_match:
        analysis = analysis_match.group(1).strip()
    if final_match:
        final = final_match.group(1).strip()
    fallback = final or re.sub(r"<\|[^>]+?\|>", " ", raw_text).strip()
    fallback = " ".join(fallback.split())
    return {"analysis": analysis, "final": final, "answer": fallback, "has_final": str(bool(final))}


def build_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Reasoning: low\nAnswer with only the short answer, no explanation.",
        },
        {"role": "user", "content": question},
    ]


def existing_generation_keys(path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((str(row["question_id"]), int(row["teacher_sample_index"])))
    return keys


def load_generation_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def generate_one(model: Any, tokenizer: Any, messages: list[dict[str, str]], args: argparse.Namespace, sample_index: int) -> dict[str, Any]:
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    do_sample = sample_index > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    latency_ms = (time.time() - started) * 1000.0
    generated = output_ids[0, prompt_tokens:]
    raw_text = tokenizer.decode(generated, skip_special_tokens=False)
    return {
        "raw_text": raw_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(generated.shape[-1]),
        "total_tokens": int(output_ids.shape[-1]),
        "latency_ms": latency_ms,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "decode_mode": "sample" if do_sample else "greedy",
    }


def build_teacher_basins(generation_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        grouped[str(row["question_id"])].append(row)
    basin_rows: list[dict[str, Any]] = []
    basin_by_question: dict[str, list[dict[str, Any]]] = {}
    for qid, rows in grouped.items():
        rows.sort(key=lambda row: int(row["teacher_sample_index"]))
        basin_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            basin_groups[str(row["teacher_canonical_basin"])].append(row)
        question_basins: list[dict[str, Any]] = []
        for basin_id, (canonical, members) in enumerate(sorted(basin_groups.items(), key=lambda item: (-len(item[1]), item[0]))):
            representative = members[0]["teacher_final_answer"]
            basin = {
                "question_id": qid,
                "teacher_basin_id": basin_id,
                "teacher_canonical_basin": canonical,
                "teacher_representative_answer": representative,
                "teacher_basin_size": len(members),
                "teacher_basin_mass": len(members) / max(1, len(rows)),
                "teacher_member_indices": [int(member["teacher_sample_index"]) for member in members],
                "teacher_basin_strict_any": max(safe_float(member["teacher_strict_correct"]) for member in members),
                "teacher_basin_f1_max": max(safe_float(member["teacher_best_f1"]) for member in members),
                "teacher_basin_f1_mean": sum(safe_float(member["teacher_best_f1"]) for member in members) / max(1, len(members)),
                "teacher_greedy_member": float(any(int(member["teacher_sample_index"]) == 0 for member in members)),
            }
            basin_rows.append(basin)
            question_basins.append(basin)
        basin_by_question[qid] = question_basins
    return basin_rows, basin_by_question


def build_anchor_rows(
    candidate_groups: dict[str, list[dict[str, Any]]],
    teacher_basins: dict[str, list[dict[str, Any]]],
    alignment_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid, candidates in candidate_groups.items():
        basins = teacher_basins.get(qid, [])
        for candidate in candidates:
            best_basin: dict[str, Any] | None = None
            best_similarity = 0.0
            support_mass = 0.0
            correct_support_mass = 0.0
            for basin in basins:
                similarity = answer_similarity(str(candidate["answer_text"]), str(basin["teacher_representative_answer"]))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_basin = basin
                if similarity >= alignment_threshold:
                    support_mass += safe_float(basin["teacher_basin_mass"])
                    correct_support_mass += safe_float(basin["teacher_basin_mass"]) * safe_float(basin["teacher_basin_strict_any"])
            qwen_stability = safe_float(candidate.get("cluster_weight_mass"))
            noleak_anchor_score = (
                0.50 * support_mass
                + 0.20 * best_similarity
                + 0.20 * safe_float(candidate.get("verifier_score_v05"))
                - 0.20 * qwen_stability * (1.0 - support_mass)
            )
            oracle_anchor_score = noleak_anchor_score + 0.50 * correct_support_mass
            rows.append(
                {
                    **candidate,
                    "teacher_best_similarity": best_similarity,
                    "teacher_best_basin_id": "" if best_basin is None else best_basin["teacher_basin_id"],
                    "teacher_best_answer": "" if best_basin is None else best_basin["teacher_representative_answer"],
                    "teacher_best_basin_mass": 0.0 if best_basin is None else best_basin["teacher_basin_mass"],
                    "teacher_best_basin_strict_any": 0.0 if best_basin is None else best_basin["teacher_basin_strict_any"],
                    "teacher_support_mass": min(support_mass, 1.0),
                    "teacher_correct_support_mass": min(correct_support_mass, 1.0),
                    "qwen_only_stable_mass": qwen_stability * (1.0 - min(support_mass, 1.0)),
                    "anchor_score_noleak": noleak_anchor_score,
                    "anchor_score_oracle_analysis_only": oracle_anchor_score,
                }
            )
    return rows


def build_question_summary(
    selected_qids: list[str],
    records: dict[str, dict[str, Any]],
    candidate_groups: dict[str, list[dict[str, Any]]],
    generation_rows: list[dict[str, Any]],
    teacher_basins: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    generation_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        generation_by_qid[str(row["question_id"])].append(row)
    summaries: list[dict[str, Any]] = []
    for qid in selected_qids:
        gens = generation_by_qid.get(qid, [])
        candidates = candidate_groups.get(qid, [])
        sample0 = next((row for row in candidates if int(float(row["sample_index"])) == 0), candidates[0] if candidates else {})
        basins = teacher_basins.get(qid, [])
        majority_basin = basins[0] if basins else {}
        sample0_similarity = answer_similarity(str(sample0.get("answer_text", "")), str(majority_basin.get("teacher_representative_answer", "")))
        summaries.append(
            {
                "question_id": qid,
                "question_index": records[qid]["question_index"],
                "question": records[qid]["question"],
                "teacher_generations": len(gens),
                "teacher_strict_rate": sum(safe_float(row["teacher_strict_correct"]) for row in gens) / max(1, len(gens)),
                "teacher_any_correct": float(any(safe_float(row["teacher_strict_correct"]) > 0.0 for row in gens)),
                "teacher_majority_answer": majority_basin.get("teacher_representative_answer", ""),
                "teacher_majority_mass": majority_basin.get("teacher_basin_mass", 0.0),
                "teacher_majority_strict": majority_basin.get("teacher_basin_strict_any", 0.0),
                "teacher_semantic_clusters": len(basins),
                "qwen_sample0_answer": sample0.get("answer_text", ""),
                "qwen_sample0_strict": sample0.get("sample0_strict_correct", sample0.get("strict_correct", 0.0)),
                "qwen_any_correct": max([safe_float(row.get("strict_correct")) for row in candidates] or [0.0]),
                "qwen_semantic_entropy": sample0.get("semantic_entropy_weighted_set", ""),
                "qwen_semantic_clusters": sample0.get("semantic_clusters_set", ""),
                "sample0_teacher_majority_similarity": sample0_similarity,
                "teacher_corrects_qwen_sample0": float(
                    safe_float(sample0.get("strict_correct")) <= 0.0
                    and safe_float(majority_basin.get("teacher_basin_strict_any", 0.0)) > 0.0
                ),
                "teacher_conflicts_with_correct_sample0": float(
                    safe_float(sample0.get("strict_correct")) > 0.0
                    and sample0_similarity < 0.8
                    and safe_float(majority_basin.get("teacher_basin_strict_any", 0.0)) <= 0.0
                ),
            }
        )
    return summaries


def metric_summary(anchor_rows: list[dict[str, Any]], question_rows: list[dict[str, Any]], generation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_metrics = {
        "candidate_auc_verifier_score_v05": auc_positive_high(anchor_rows, "verifier_score_v05", "strict_correct"),
        "candidate_auc_teacher_support_mass": auc_positive_high(anchor_rows, "teacher_support_mass", "strict_correct"),
        "candidate_auc_anchor_score_noleak": auc_positive_high(anchor_rows, "anchor_score_noleak", "strict_correct"),
        "candidate_auc_anchor_score_oracle_analysis_only": auc_positive_high(
            anchor_rows, "anchor_score_oracle_analysis_only", "strict_correct"
        ),
        "candidate_auc_cluster_weight_mass": auc_positive_high(anchor_rows, "cluster_weight_mass", "strict_correct"),
    }
    return {
        "items": {
            "questions": len(question_rows),
            "teacher_generations": len(generation_rows),
            "qwen_candidate_rows": len(anchor_rows),
        },
        "teacher": {
            "generation_strict_rate": sum(safe_float(row["teacher_strict_correct"]) for row in generation_rows) / max(1, len(generation_rows)),
            "question_any_correct_rate": sum(safe_float(row["teacher_any_correct"]) for row in question_rows) / max(1, len(question_rows)),
            "question_majority_strict_rate": sum(safe_float(row["teacher_majority_strict"]) for row in question_rows) / max(1, len(question_rows)),
        },
        "cross_model": {
            "teacher_corrects_qwen_sample0_rate": sum(safe_float(row["teacher_corrects_qwen_sample0"]) for row in question_rows)
            / max(1, len(question_rows)),
            "teacher_conflicts_with_correct_sample0_rate": sum(
                safe_float(row["teacher_conflicts_with_correct_sample0"]) for row in question_rows
            )
            / max(1, len(question_rows)),
        },
        "candidate_auc": candidate_metrics,
    }


def final_only_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if safe_float(row.get("teacher_has_final")) > 0.0]


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(args.output_root) / f"run_{timestamp}_{args.tag}")
    generation_path = run_dir / "teacher_generations.jsonl"
    records = load_records(Path(args.input_jsonl))
    candidate_groups_all = load_candidate_groups(Path(args.candidate_features))
    selected_qids = select_question_ids(candidate_groups_all, records, args.num_questions, args.seed)
    candidate_groups = {qid: candidate_groups_all[qid] for qid in selected_qids}

    metadata = {
        "run_dir": str(run_dir),
        "started_at": timestamp,
        "script": __file__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers_env": os.environ.get("TRANSFORMERS_CACHE", ""),
        "model_dir": args.model_dir,
        "input_jsonl": args.input_jsonl,
        "candidate_features": args.candidate_features,
        "num_questions": len(selected_qids),
        "samples_per_question": args.samples_per_question,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "alignment_threshold": args.alignment_threshold,
        "selected_question_ids": selected_qids,
        "no_leak_statement": "Gold answers are not placed in prompts; they are used only after generation for labels and analysis.",
    }
    write_json(run_dir / "run_metadata.json", metadata)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
        device_map=args.device_map,
        trust_remote_code=True,
    )
    existing = existing_generation_keys(generation_path)
    for question_offset, qid in enumerate(selected_qids):
        record = records[qid]
        messages = build_messages(record["question"])
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        for sample_index in range(args.samples_per_question):
            if (qid, sample_index) in existing:
                continue
            generated = generate_one(model, tokenizer, messages, args, sample_index)
            parsed = parse_harmony(generated["raw_text"])
            answer = parsed["answer"]
            row = {
                "question_offset": question_offset,
                "question_index": record["question_index"],
                "question_id": qid,
                "question": record["question"],
                "ideal_answers": record["ideal_answers"],
                "teacher_model": args.model_dir,
                "teacher_sample_index": sample_index,
                "teacher_decode_mode": generated["decode_mode"],
                "teacher_prompt": prompt_text,
                "teacher_raw_text": generated["raw_text"],
                "teacher_analysis_text": parsed["analysis"],
                "teacher_has_final": float(parsed["has_final"] == "True"),
                "teacher_final_answer": answer,
                "teacher_normalized_answer": normalize_text(answer),
                "teacher_canonical_basin": canonical_answer(answer),
                "teacher_exact_match": float(exact_match(answer, record["ideal_answers"])),
                "teacher_contains_match": float(contains_match(answer, record["ideal_answers"])),
                "teacher_strict_correct": float(exact_match(answer, record["ideal_answers"]) or contains_match(answer, record["ideal_answers"])),
                "teacher_best_f1": best_f1(answer, record["ideal_answers"]),
                "prompt_tokens": generated["prompt_tokens"],
                "completion_tokens": generated["completion_tokens"],
                "total_tokens": generated["total_tokens"],
                "latency_ms": generated["latency_ms"],
                "cuda_peak_memory_bytes": generated["cuda_peak_memory_bytes"],
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0.0 if sample_index == 0 else args.temperature,
                "top_p": 1.0 if sample_index == 0 else args.top_p,
            }
            append_jsonl(generation_path, row)
            existing.add((qid, sample_index))
            print(json.dumps({"generated": len(existing), "question_id": qid, "sample_index": sample_index, "answer": answer}, ensure_ascii=False), flush=True)

    generation_rows = load_generation_rows(generation_path)
    teacher_basin_rows, teacher_basins = build_teacher_basins(generation_rows)
    anchor_rows = build_anchor_rows(candidate_groups, teacher_basins, args.alignment_threshold)
    question_rows = build_question_summary(selected_qids, records, candidate_groups, generation_rows, teacher_basins)
    metrics = metric_summary(anchor_rows, question_rows, generation_rows)

    final_generation_rows = final_only_rows(generation_rows)
    final_teacher_basin_rows, final_teacher_basins = build_teacher_basins(final_generation_rows)
    final_anchor_rows = build_anchor_rows(candidate_groups, final_teacher_basins, args.alignment_threshold)
    final_question_rows = build_question_summary(selected_qids, records, candidate_groups, final_generation_rows, final_teacher_basins)
    final_metrics = metric_summary(final_anchor_rows, final_question_rows, final_generation_rows)

    write_csv(run_dir / "teacher_basin_rows.csv", teacher_basin_rows)
    write_csv(run_dir / "qwen_candidate_anchor_rows.csv", anchor_rows)
    write_csv(run_dir / "question_anchor_summary.csv", question_rows)
    write_json(run_dir / "anchor_metric_summary.json", metrics)
    write_csv(run_dir / "teacher_basin_rows_final_only.csv", final_teacher_basin_rows)
    write_csv(run_dir / "qwen_candidate_anchor_rows_final_only.csv", final_anchor_rows)
    write_csv(run_dir / "question_anchor_summary_final_only.csv", final_question_rows)
    write_json(run_dir / "anchor_metric_summary_final_only.json", final_metrics)
    report = [
        "# GPT-OSS Cross-Model Anchor Pilot",
        "",
        f"- Questions: `{len(question_rows)}`",
        f"- Teacher generations: `{len(generation_rows)}`",
        f"- Teacher generations with final channel: `{len(final_generation_rows)}` (`{len(final_generation_rows) / max(1, len(generation_rows)):.2%}`)",
        f"- Teacher generation strict rate: `{metrics['teacher']['generation_strict_rate']:.3f}`",
        f"- Teacher strict rate among final-channel generations: `{final_metrics['teacher']['generation_strict_rate']:.3f}`",
        f"- Teacher any-correct question rate: `{metrics['teacher']['question_any_correct_rate']:.3f}`",
        f"- AUC verifier score: `{metrics['candidate_auc']['candidate_auc_verifier_score_v05']:.3f}`",
        f"- AUC teacher support mass: `{metrics['candidate_auc']['candidate_auc_teacher_support_mass']:.3f}`",
        f"- AUC anchor score no-leak: `{metrics['candidate_auc']['candidate_auc_anchor_score_noleak']:.3f}`",
        f"- Final-only AUC teacher support mass: `{final_metrics['candidate_auc']['candidate_auc_teacher_support_mass']:.3f}`",
        f"- Final-only AUC anchor score no-leak: `{final_metrics['candidate_auc']['candidate_auc_anchor_score_noleak']:.3f}`",
        "",
        "Gold labels were used only for post-generation analysis.",
    ]
    (run_dir / "ANCHOR_PILOT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"run_dir": str(run_dir), "status": "complete", "metrics": metrics, "final_only_metrics": final_metrics},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
