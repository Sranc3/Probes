"""Text utilities shared across Plan_opus trainers and evaluators."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

STOPWORDS = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with", "by"}


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


def best_ideal_alias(answer: str, ideals: list[str]) -> str | None:
    """Return the alias actually contained in the answer (the canonical span)."""
    normalized = normalize_text(answer)
    best: str | None = None
    best_len = -1
    for ideal in ideals:
        normalized_ideal = normalize_text(ideal)
        if not normalized_ideal:
            continue
        if normalized == normalized_ideal or normalized_ideal in normalized:
            if len(normalized_ideal) > best_len:
                best = normalized_ideal
                best_len = len(normalized_ideal)
    return best


def shortest_correct_span(answer: str, ideals: list[str]) -> str | None:
    """Return shortest ideal alias contained in answer (used as canonical chosen)."""
    normalized = normalize_text(answer)
    candidates: list[str] = []
    for ideal in ideals:
        normalized_ideal = normalize_text(ideal)
        if normalized_ideal and (normalized == normalized_ideal or normalized_ideal in normalized):
            candidates.append(str(ideal).strip())
    if not candidates:
        return None
    return min(candidates, key=lambda text: (len(text.split()), len(text)))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_triviaqa_records(path: str | Path, offset: int, total: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line_index, line in enumerate(file_obj):
            if line_index < offset:
                continue
            if total > 0 and len(rows) >= total:
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
    return rows


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed == parsed else 0.0  # NaN guard
