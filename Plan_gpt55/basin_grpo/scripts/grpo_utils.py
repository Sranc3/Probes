#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STOPWORDS = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with", "by"}


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    lowered = str(text).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(cleaned.split())


def canonical_answer(text: str) -> str:
    normalized = normalize_text(text)
    tokens = [token for token in normalized.split() if token not in STOPWORDS]
    return " ".join(tokens[:12])


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


def strict_correct(answer: str, ideals: list[str]) -> bool:
    normalized = normalize_text(answer)
    if not normalized:
        return False
    for ideal in ideals:
        normalized_ideal = normalize_text(ideal)
        if normalized_ideal and (normalized == normalized_ideal or normalized_ideal in normalized):
            return True
    return False


def best_f1(answer: str, ideals: list[str]) -> float:
    return max([token_f1(answer, ideal) for ideal in ideals] or [0.0])


def load_triviaqa_records(path: str | Path, offset: int, total: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line_index, line in enumerate(file_obj):
            if line_index < offset:
                continue
            if len(rows) >= total:
                break
            payload = json.loads(line)
            rows.append(
                {
                    "id": str(payload["id"]),
                    "question": str(payload["question"]),
                    "ideal_answers": [str(item) for item in payload.get("ideal", payload.get("ideal_answers", []))],
                    "system_prompt": payload.get("system_prompt", "Answer the question briefly and factually."),
                }
            )
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def split_records(records: list[dict[str, Any]], split_cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    train_size = int(split_cfg["train_size"])
    dev_size = int(split_cfg["dev_size"])
    test_size = int(split_cfg["test_size"])
    return {
        "train": records[:train_size],
        "dev": records[train_size : train_size + dev_size],
        "test": records[train_size + dev_size : train_size + dev_size + test_size],
    }


def load_baseline_correctness(path: str | Path | None) -> dict[str, float]:
    if not path:
        return {}
    candidate_path = Path(path)
    if not candidate_path.exists():
        return {}
    if candidate_path.suffix == ".json":
        payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "baseline_correct" in payload:
            payload = payload["baseline_correct"]
        return {str(key): float(value) for key, value in dict(payload).items()}
    if candidate_path.suffix == ".jsonl":
        baseline_jsonl: dict[str, float] = {}
        with candidate_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if not line.strip():
                    continue
                row = json.loads(line)
                baseline_jsonl[str(row["question_id"])] = float(row["strict_correct"])
        return baseline_jsonl
    baseline: dict[str, list[float]] = defaultdict(list)
    with candidate_path.open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            if int(float(row.get("sample_index", 1))) == 0:
                baseline[str(row["question_id"])].append(float(row.get("strict_correct", 0.0)))
    return {qid: sum(values) / len(values) for qid, values in baseline.items() if values}


def baseline_coverage(records: list[dict[str, Any]], baseline_correct: dict[str, float]) -> dict[str, Any]:
    covered = [record["id"] for record in records if record["id"] in baseline_correct]
    values = [baseline_correct[qid] for qid in covered]
    return {
        "items": len(records),
        "covered": len(covered),
        "coverage_rate": len(covered) / max(1, len(records)),
        "sample0_correct_rate": mean([float(value) for value in values]) if values else 0.0,
        "missing_examples": [record["id"] for record in records if record["id"] not in baseline_correct][:10],
    }


def build_prompt(tokenizer: Any, question: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\nQuestion: {question}\nAnswer:"


def compute_group_rewards(
    completions: list[str],
    ideals: list[str],
    baseline_sample0_correct: float,
    token_counts: list[int],
    reward_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    canonicals = [canonical_answer(text) for text in completions]
    basin_counts = Counter(canonicals)
    group_size = max(1, len(completions))
    strict_values = [float(strict_correct(text, ideals)) for text in completions]
    f1_values = [best_f1(text, ideals) for text in completions]
    basin_correct: dict[str, float] = {}
    for canonical in basin_counts:
        members = [idx for idx, item in enumerate(canonicals) if item == canonical]
        basin_correct[canonical] = max(strict_values[idx] for idx in members)

    rows: list[dict[str, Any]] = []
    for text, canonical, token_count, strict_value, f1_value in zip(completions, canonicals, token_counts, strict_values, f1_values):
        mass = basin_counts[canonical] / group_size
        length_cost = min(float(token_count) / max(1.0, float(reward_cfg["length_max_tokens"])), 1.0)
        stable_wrong = mass if basin_correct[canonical] <= 0.0 else 0.0
        damage = 1.0 if baseline_sample0_correct > 0.5 and strict_value <= 0.0 else 0.0
        consensus = mass if strict_value > 0.0 and basin_correct[canonical] > 0.0 else 0.0
        reward = (
            float(reward_cfg["strict"]) * strict_value
            + float(reward_cfg["f1"]) * f1_value
            - float(reward_cfg["length"]) * length_cost
            - float(reward_cfg["stable_wrong_basin"]) * stable_wrong
            - float(reward_cfg["damage"]) * damage
            + float(reward_cfg["correct_consensus"]) * consensus
        )
        rows.append(
            {
                "completion": text,
                "canonical_basin": canonical,
                "basin_mass": mass,
                "strict": strict_value,
                "f1": f1_value,
                "length_cost": length_cost,
                "stable_wrong_basin": stable_wrong,
                "damage": damage,
                "correct_consensus": consensus,
                "reward": reward,
            }
        )
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def summarize_reward_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ["reward", "strict", "f1", "length_cost", "stable_wrong_basin", "damage", "correct_consensus", "basin_mass"]
    return {f"mean_{key}": mean([float(row[key]) for row in rows]) for key in keys}
