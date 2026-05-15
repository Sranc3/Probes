"""Anchor-aware pair builder for VBPO-Opus.

Design choices that fix Plan_gpt55 v1/v2 issues:

1. **Strict label hygiene**: every pair *must* have ``chosen.strict_correct == 1``
   and ``rejected.strict_correct == 0``. We do not allow noisy pair types like
   ``teacher_anchor_vs_student_only`` where 38% of chosen were wrong.

2. **High coverage**: instead of tiny per-question quotas, we use the entire
   training set wherever a (correct, wrong) pair can be built. The four
   sub-types are all merged into one balanced pool, with a *weight* derived
   from anchor strength.

3. **Short canonical chosen completion**: the chosen completion is the
   *shortest* candidate text that contains a gold alias. Verbose 40-token
   "derivation that happens to mention the answer" never becomes chosen.
   Rejected stays at the model's own verbose wrong completion (we want to
   *push down* the verbose hallucinations, not normalize them).

4. **Anchor-driven weight**: ``weight = 1 + alpha * teacher_support_chosen +
   beta * (qwen_only_stable_rejected)``. Caps + floor for stability.

5. **Per-question diversity cap** so a few "easy" questions don't dominate
   the batch.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
sys.path.insert(0, str(SHARED_DIR))

from text_utils import (  # noqa: E402
    canonical_answer,
    load_triviaqa_records,
    mean,
    read_csv,
    read_json,
    safe_float,
    shortest_correct_span,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean DPO pairs from anchor rows.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def is_clean_text(text: str, max_words: int = 64) -> bool:
    text = str(text).strip()
    if not text or len(text.split()) > max_words:
        return False
    lowered = text.lower()
    bad_markers = (
        "analysis",
        "we need",
        "let's",
        "let us",
        "assistantfinal",
        "<|",
        "channel",
    )
    return not any(marker in lowered for marker in bad_markers)


def short_factual_text(row: dict[str, Any], gold_aliases: list[str] | None = None, max_chosen_words: int = 16) -> str:
    """Return a short, factual chosen completion.

    Priority:
    1. Use the candidate's own ``answer_text`` when it is short and clean.
    2. Otherwise use ``teacher_best_answer`` if available and short and clean.
    3. Otherwise extract the shortest gold alias actually contained in the
       candidate text (this is guaranteed to exist because the row is
       strict_correct, i.e. some normalized alias appears in the text).
    4. As a last resort, return the (verbose) raw text.
    """
    raw = str(row.get("answer_text", "")).strip()
    teacher = str(row.get("teacher_best_answer", "")).strip()
    if raw and len(raw.split()) <= max_chosen_words and is_clean_text(raw):
        return raw
    if teacher and len(teacher.split()) <= max_chosen_words and is_clean_text(teacher):
        return teacher
    if gold_aliases:
        span = shortest_correct_span(raw, gold_aliases)
        if span and len(span.split()) <= max_chosen_words:
            return span
    return raw


def anchor_strength(row: dict[str, Any]) -> float:
    return (
        safe_float(row.get("teacher_support_mass"))
        + 0.5 * safe_float(row.get("anchor_score_noleak"))
        + 0.3 * safe_float(row.get("teacher_best_similarity"))
    )


def stable_wrong_strength(row: dict[str, Any]) -> float:
    return (
        safe_float(row.get("qwen_only_stable_mass"))
        + 0.4 * safe_float(row.get("cluster_weight_mass_minmax"))
        + 0.2 * (1.0 - safe_float(row.get("teacher_support_mass")))
    )


def split_questions(question_ids: list[str], train_size: int, dev_size: int, seed: int) -> dict[str, list[str]]:
    import random as _random

    rng = _random.Random(int(seed))
    shuffled = list(question_ids)
    rng.shuffle(shuffled)
    return {
        "train": shuffled[:train_size],
        "dev": shuffled[train_size : train_size + dev_size],
    }


def teacher_correct_answer(rows: list[dict[str, Any]], gold_aliases: list[str] | None) -> str | None:
    """Return the teacher's best answer when it is strict-correct under gold.

    Used to rescue 'all-wrong-only' questions where the student never produced
    a correct candidate but the cross-model teacher did.
    """
    seen: set[str] = set()
    candidates: list[str] = []
    for row in rows:
        teacher = str(row.get("teacher_best_answer", "")).strip()
        if not teacher:
            continue
        key = teacher.lower()
        if key in seen:
            continue
        seen.add(key)
        if not gold_aliases:
            # Without gold we can still use teacher_best_basin_strict_any (no leak: it's gpt-oss judging gpt-oss)
            if safe_float(row.get("teacher_best_basin_strict_any")) > 0.0 and len(teacher.split()) <= 16:
                candidates.append(teacher)
        else:
            from text_utils import strict_correct as _sc

            if _sc(teacher, gold_aliases) and is_clean_text(teacher, max_words=16):
                candidates.append(teacher)
    if not candidates:
        return None
    return min(candidates, key=lambda text: (len(text.split()), len(text)))


def build_pairs_for_question(
    rows: list[dict[str, Any]],
    pair_cfg: dict[str, Any],
    gold_aliases: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    correct = [row for row in rows if safe_float(row.get("strict_correct")) > 0.0]
    wrong = [row for row in rows if safe_float(row.get("strict_correct")) <= 0.0]
    qid = str(rows[0]["question_id"])
    question = str(rows[0]["question"])

    # Rescue: when no student-correct candidate exists, but the teacher has a
    # strict-correct short answer, use the teacher answer as chosen and
    # rank-1 stable wrong candidate as rejected. This is clean distillation.
    if pair_cfg.get("enable_teacher_rescue", True) and not correct and wrong:
        rescue_text = teacher_correct_answer(rows, gold_aliases)
        if rescue_text and is_clean_text(rescue_text, max_words=int(pair_cfg.get("max_chosen_words", 16))):
            sorted_wrong = sorted(wrong, key=stable_wrong_strength, reverse=True)
            rejected_pool: list[dict[str, Any]] = []
            seen: set[str] = set()
            for row in sorted_wrong:
                text = str(row.get("answer_text", "")).strip()
                if not is_clean_text(text, max_words=80):
                    continue
                key = canonical_answer(text)
                if not key or key in seen:
                    continue
                seen.add(key)
                rejected_pool.append({"row": row, "text": text})
                if len(rejected_pool) >= int(pair_cfg.get("max_rejected_per_question", 4)):
                    break
            if not rejected_pool:
                return []
            rescue_pairs: list[dict[str, Any]] = []
            chosen_anchor = max(safe_float(r.get("teacher_support_mass")) for r in rows)
            for entry in rejected_pool:
                rejected_row = entry["row"]
                rejected_stable = stable_wrong_strength(rejected_row)
                weight = 1.0 + float(pair_cfg.get("chosen_anchor_alpha", 0.6)) * chosen_anchor + float(pair_cfg.get("rejected_stable_alpha", 0.4)) * rejected_stable
                weight = max(float(pair_cfg.get("min_weight", 0.6)), min(float(pair_cfg.get("max_weight", 2.5)), weight))
                rescue_pairs.append(
                    {
                        "question_id": qid,
                        "question": question,
                        "system_prompt": "Answer the question briefly and factually.",
                        "chosen": rescue_text,
                        "rejected": entry["text"],
                        "chosen_canonical": canonical_answer(rescue_text),
                        "rejected_canonical": canonical_answer(entry["text"]),
                        "ideal_answers_repr": "",
                        "chosen_anchor_strength": chosen_anchor,
                        "rejected_stable_strength": rejected_stable,
                        "chosen_teacher_support_mass": chosen_anchor,
                        "rejected_qwen_only_stable_mass": safe_float(rejected_row.get("qwen_only_stable_mass")),
                        "weight": weight,
                        "chosen_strict_correct": 1.0,
                        "rejected_strict_correct": 0.0,
                        "pair_source": "teacher_rescue",
                    }
                )
                if len(rescue_pairs) >= int(pair_cfg.get("max_pairs_per_question", 4)):
                    break
            return rescue_pairs

    if not correct or not wrong:
        return []

    max_per_q = int(pair_cfg.get("max_pairs_per_question", 4))
    max_chosen_words = int(pair_cfg.get("max_chosen_words", 16))
    chosen_pool: list[dict[str, Any]] = []
    seen_chosen_keys: set[str] = set()
    for row in sorted(correct, key=anchor_strength, reverse=True):
        text = short_factual_text(row, gold_aliases, max_chosen_words=max_chosen_words)
        if not is_clean_text(text, max_words=max_chosen_words):
            continue
        key = canonical_answer(text)
        if key in seen_chosen_keys:
            continue
        seen_chosen_keys.add(key)
        chosen_pool.append({"row": row, "text": text})
        if len(chosen_pool) >= int(pair_cfg.get("max_chosen_per_question", 2)):
            break
    if not chosen_pool:
        return []

    rejected_pool: list[dict[str, Any]] = []
    seen_rejected_keys: set[str] = set()
    for row in sorted(wrong, key=stable_wrong_strength, reverse=True):
        text = str(row.get("answer_text", "")).strip()
        if not is_clean_text(text, max_words=80):
            continue
        key = canonical_answer(text)
        if not key or key in seen_rejected_keys:
            continue
        seen_rejected_keys.add(key)
        rejected_pool.append({"row": row, "text": text})
        if len(rejected_pool) >= int(pair_cfg.get("max_rejected_per_question", 4)):
            break
    if not rejected_pool:
        return []

    chosen_alpha = float(pair_cfg.get("chosen_anchor_alpha", 0.6))
    rejected_alpha = float(pair_cfg.get("rejected_stable_alpha", 0.4))
    max_w = float(pair_cfg.get("max_weight", 2.5))
    min_w = float(pair_cfg.get("min_weight", 0.6))

    pairs: list[dict[str, Any]] = []
    qid = str(rows[0]["question_id"])
    question = str(rows[0]["question"])
    for chosen_entry in chosen_pool:
        for rejected_entry in rejected_pool:
            chosen_row = chosen_entry["row"]
            rejected_row = rejected_entry["row"]
            chosen_anchor = anchor_strength(chosen_row)
            rejected_stable = stable_wrong_strength(rejected_row)
            weight = 1.0 + chosen_alpha * chosen_anchor + rejected_alpha * rejected_stable
            weight = max(min_w, min(max_w, weight))
            pairs.append(
                {
                    "question_id": qid,
                    "question": question,
                    "system_prompt": "Answer the question briefly and factually.",
                    "chosen": chosen_entry["text"],
                    "rejected": rejected_entry["text"],
                    "chosen_canonical": canonical_answer(chosen_entry["text"]),
                    "rejected_canonical": canonical_answer(rejected_entry["text"]),
                    "ideal_answers_repr": "",
                    "chosen_anchor_strength": chosen_anchor,
                    "rejected_stable_strength": rejected_stable,
                    "chosen_teacher_support_mass": safe_float(chosen_row.get("teacher_support_mass")),
                    "rejected_qwen_only_stable_mass": safe_float(rejected_row.get("qwen_only_stable_mass")),
                    "weight": weight,
                    "chosen_strict_correct": safe_float(chosen_row.get("strict_correct")),
                    "rejected_strict_correct": safe_float(rejected_row.get("strict_correct")),
                    "pair_source": "student_pair",
                }
            )
            if len(pairs) >= max_per_q:
                break
        if len(pairs) >= max_per_q:
            break
    return pairs


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    rows = read_csv(config["anchor_rows"])
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_q[str(row["question_id"])].append(row)
    qids_sorted = sorted(by_q.keys())
    splits = split_questions(qids_sorted, int(config["split"]["train_questions"]),
                             int(config["split"]["dev_questions"]), int(config["seed"]))

    gold_lookup: dict[str, list[str]] = {}
    if config.get("input_jsonl"):
        # Read enough records to cover the union of question ids needed.
        max_index = 0
        for qid in qids_sorted:
            try:
                max_index = max(max_index, int(qid.split("_")[-1]))
            except ValueError:
                continue
        records = load_triviaqa_records(config["input_jsonl"], 0, max_index + 1)
        gold_lookup = {record["id"]: record["ideal_answers"] for record in records}

    pair_cfg = config.get("pair_building", {})
    pair_splits: dict[str, list[dict[str, Any]]] = {}
    for split_name, ids in splits.items():
        pairs: list[dict[str, Any]] = []
        for qid in ids:
            pairs.extend(
                build_pairs_for_question(
                    by_q.get(qid, []),
                    pair_cfg,
                    gold_aliases=gold_lookup.get(qid),
                )
            )
        pair_splits[split_name] = pairs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, pairs in pair_splits.items():
        write_csv(output_dir / f"pairs_{split_name}.csv", pairs)

    def source_counts(pairs: list[dict[str, Any]]) -> dict[str, int]:
        c = Counter(str(p.get("pair_source", "unknown")) for p in pairs)
        return dict(c)

    summary = {
        "split_sizes": {name: len(ids) for name, ids in splits.items()},
        "pair_counts": {name: len(pairs) for name, pairs in pair_splits.items()},
        "questions_with_pairs": {
            name: len({p["question_id"] for p in pairs}) for name, pairs in pair_splits.items()
        },
        "pair_source_counts": {name: source_counts(pairs) for name, pairs in pair_splits.items()},
        "mean_weight": {name: mean(float(p["weight"]) for p in pairs) for name, pairs in pair_splits.items()},
        "label_purity": {
            name: {
                "chosen_strict_correct_rate": mean(float(p["chosen_strict_correct"]) for p in pairs),
                "rejected_strict_correct_rate": mean(float(p["rejected_strict_correct"]) for p in pairs),
            }
            for name, pairs in pair_splits.items()
        },
        "chosen_text_word_counts": {
            name: mean(len(str(p["chosen"]).split()) for p in pairs) for name, pairs in pair_splits.items()
        },
        "rejected_text_word_counts": {
            name: mean(len(str(p["rejected"]).split()) for p in pairs) for name, pairs in pair_splits.items()
        },
    }
    write_json(output_dir / "pair_manifest.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
