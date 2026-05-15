#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from run_attractor_controller_experiments import (
    add_candidate_features,
    attach_attractor_features,
    build_attractor_tables,
    ensure_dir,
    evaluate_selected,
    flatten,
    group_candidates,
    grouped_folds,
    mean,
    read_csv,
    safe_float,
    write_json,
)
from run_attractor_controller_v2 import (
    basin_formulas,
    basin_score,
    cluster_groups,
    pair_basins,
    pareto_settings,
    risk_formulas,
    risk_score,
)


DEFAULT_INPUT_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"
VERIFIER_CACHE: dict[tuple[int, str, int], dict[str, float]] = {}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "the",
    "their",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
}
MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
NUMBER_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "hundred",
    "thousand",
    "million",
    "billion",
}
LOCATION_WORDS = {
    "city",
    "country",
    "state",
    "county",
    "river",
    "sea",
    "lake",
    "mount",
    "mountain",
    "stadium",
    "airport",
    "island",
    "england",
    "france",
    "germany",
    "italy",
    "spain",
    "china",
    "india",
    "america",
    "canada",
    "australia",
    "philadelphia",
    "london",
    "paris",
}
BAD_MARKERS = {
    "however",
    "actually",
    "although",
    "but",
    "cannot",
    "can't",
    "not",
    "unknown",
    "unclear",
    "rather",
    "instead",
    "appears",
    "possibly",
    "probably",
    "i ",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run basin verifier v3 experiments.")
    parser.add_argument("--input-run", default=DEFAULT_INPUT_RUN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="basin_verifier_v3")
    return parser.parse_args()


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: flatten(row.get(key, "")) for key in fieldnames})


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


def content_tokens(text: str) -> list[str]:
    return [tok for tok in normalize(text).split() if tok not in STOPWORDS and len(tok) > 2]


def token_set(text: str) -> set[str]:
    return set(content_tokens(text))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def question_type(question: str) -> str:
    q = normalize(question)
    if q.startswith("who") or " whose " in f" {q} ":
        return "person"
    if q.startswith("where") or " in which country" in q or " in which city" in q or "which country" in q:
        return "place"
    if q.startswith("when") or "which year" in q or "which decade" in q or "on which date" in q:
        return "date"
    if q.startswith("how many") or "to the nearest" in q:
        return "number"
    if q.startswith("which") and any(word in q for word in ["movie", "film", "novel", "book", "song", "musical", "play"]):
        return "work"
    if q.startswith("which") and any(word in q for word in ["state", "country", "city", "river", "sea"]):
        return "place"
    return "entity"


def capitalized_spans(text: str) -> list[str]:
    return re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", str(text))


def answer_type_score(question: str, answer: str) -> float:
    qtype = question_type(question)
    ans_norm = normalize(answer)
    spans = capitalized_spans(answer)
    has_year = bool(re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", answer))
    has_number = bool(re.search(r"\b\d+\b", answer)) or any(word in ans_norm.split() for word in NUMBER_WORDS)
    has_month = any(month in ans_norm.split() for month in MONTHS)
    has_location = bool(token_set(answer) & LOCATION_WORDS) or any(word in ans_norm for word in [" in ", " at ", " near "])
    if qtype == "person":
        return 1.0 if spans else 0.15
    if qtype == "place":
        return 1.0 if spans or has_location else 0.2
    if qtype == "date":
        return 1.0 if has_year or has_month else 0.1
    if qtype == "number":
        return 1.0 if has_number else 0.05
    if qtype == "work":
        return 1.0 if "\"" in answer or spans else 0.25
    return 0.65 if answer.strip() else 0.0


def bad_marker_score(answer: str) -> float:
    ans = f" {normalize(answer)} "
    count = sum(1 for marker in BAD_MARKERS if marker in ans)
    return min(1.0, count / 2.0)


def answer_focus_score(question: str, answer: str) -> float:
    q_tokens = token_set(question)
    a_tokens = token_set(answer)
    if not a_tokens:
        return 0.0
    # Good answers often avoid long digressions while preserving some relation to the question.
    overlap = len(q_tokens & a_tokens) / max(1, len(q_tokens))
    length = len(content_tokens(answer))
    concise = 1.0 / (1.0 + max(0, length - 12) / 16.0)
    return 0.65 * concise + 0.35 * min(1.0, overlap * 2.0)


def consensus_features(members: list[dict[str, Any]]) -> dict[str, float]:
    texts = [str(row["answer_text"]) for row in members]
    sets = [token_set(text) for text in texts]
    if len(sets) <= 1:
        return {"consensus_jaccard": 1.0, "core_token_fraction": 1.0, "entity_consensus": 1.0}
    pairs = [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    token_counts = Counter(token for tokens in sets for token in tokens)
    core_tokens = [token for token, count in token_counts.items() if count >= math.ceil(len(sets) / 2)]
    spans = [set(capitalized_spans(text)) for text in texts]
    span_counts = Counter(span for span_set in spans for span in span_set)
    shared_spans = [span for span, count in span_counts.items() if count >= math.ceil(len(spans) / 2)]
    return {
        "consensus_jaccard": mean(pairs),
        "core_token_fraction": len(core_tokens) / max(1, len(token_counts)),
        "entity_consensus": len(shared_spans) / max(1, len(span_counts)),
    }


def verifier_features(basin: dict[str, Any], question: str) -> dict[str, float]:
    rep = basin["representative"]
    cache_key = (int(rep["seed"]), str(rep["question_id"]), int(rep["cluster_id"]))
    if cache_key in VERIFIER_CACHE:
        return VERIFIER_CACHE[cache_key]
    members = basin["members"]
    rep_answer = str(rep["answer_text"])
    consensus = consensus_features(members)
    type_scores = [answer_type_score(question, str(row["answer_text"])) for row in members]
    focus_scores = [answer_focus_score(question, str(row["answer_text"])) for row in members]
    bad_scores = [bad_marker_score(str(row["answer_text"])) for row in members]
    avg_len = mean([float(len(content_tokens(str(row["answer_text"])))) for row in members])
    features = {
        "answer_type_match": mean(type_scores),
        "rep_answer_type_match": answer_type_score(question, rep_answer),
        "focus_score": mean(focus_scores),
        "rep_focus_score": answer_focus_score(question, rep_answer),
        "bad_marker_rate": mean(bad_scores),
        "rep_bad_marker": bad_marker_score(rep_answer),
        "consensus_jaccard": consensus["consensus_jaccard"],
        "core_token_fraction": consensus["core_token_fraction"],
        "entity_consensus": consensus["entity_consensus"],
        "avg_content_len": avg_len,
        "len_penalty": min(1.0, max(0.0, avg_len - 14.0) / 24.0),
    }
    VERIFIER_CACHE[cache_key] = features
    return features


VerifierFormula = dict[str, Any]


def verifier_formulas() -> list[VerifierFormula]:
    return [
        {
            "name": "geometry_only",
            "geometry_weight": 1.0,
            "verifier_weight": 0.0,
            "weights": {},
        },
        {
            "name": "verifier_only",
            "geometry_weight": 0.0,
            "verifier_weight": 1.0,
            "weights": {
                "answer_type_match": 0.9,
                "focus_score": 0.6,
                "bad_marker_rate": -0.9,
                "len_penalty": -0.5,
                "consensus_jaccard": 0.4,
            },
        },
        {
            "name": "geometry_type",
            "geometry_weight": 1.0,
            "verifier_weight": 1.0,
            "weights": {
                "answer_type_match": 1.0,
                "rep_answer_type_match": 0.5,
                "bad_marker_rate": -0.4,
            },
        },
        {
            "name": "geometry_consensus",
            "geometry_weight": 1.0,
            "verifier_weight": 1.0,
            "weights": {
                "consensus_jaccard": 0.7,
                "core_token_fraction": 0.6,
                "entity_consensus": 0.4,
                "bad_marker_rate": -0.4,
            },
        },
        {
            "name": "geometry_grounding",
            "geometry_weight": 1.0,
            "verifier_weight": 1.0,
            "weights": {
                "focus_score": 0.8,
                "rep_focus_score": 0.4,
                "bad_marker_rate": -0.8,
                "rep_bad_marker": -0.5,
                "len_penalty": -0.6,
            },
        },
        {
            "name": "geometry_all",
            "geometry_weight": 1.0,
            "verifier_weight": 1.0,
            "weights": {
                "answer_type_match": 0.8,
                "focus_score": 0.55,
                "consensus_jaccard": 0.45,
                "core_token_fraction": 0.35,
                "entity_consensus": 0.25,
                "bad_marker_rate": -0.9,
                "rep_bad_marker": -0.5,
                "len_penalty": -0.55,
            },
        },
        {
            "name": "geometry_all_safe",
            "geometry_weight": 1.0,
            "verifier_weight": 1.2,
            "weights": {
                "answer_type_match": 0.7,
                "focus_score": 0.5,
                "consensus_jaccard": 0.35,
                "bad_marker_rate": -1.3,
                "rep_bad_marker": -0.8,
                "len_penalty": -0.9,
            },
        },
    ]


def weighted_verifier_score(features: dict[str, float], formula: VerifierFormula) -> float:
    return sum(float(weight) * safe_float(features.get(name, 0.0)) for name, weight in formula["weights"].items())


def compact_thresholds(values: list[float]) -> list[float]:
    if not values:
        return [0.0]
    sorted_values = sorted(values)
    thresholds = []
    for q in [0.25, 0.5, 0.75]:
        thresholds.append(sorted_values[min(len(sorted_values) - 1, int(q * (len(sorted_values) - 1)))])
    return sorted(set(thresholds))


def compact_risk_formulas() -> list[dict[str, Any]]:
    names = {"weak_sample0_basin", "fragmented_with_competitor", "sample0_uncertain"}
    return [formula for formula in risk_formulas() if formula["name"] in names]


def compact_basin_formulas() -> list[dict[str, Any]]:
    names = {"basin_balanced", "basin_safe", "basin_antidamage"}
    return [formula for formula in basin_formulas() if formula["name"] in names]


def basin_score_v3(basin: dict[str, Any], question: str, basin_formula: dict[str, Any], verifier_formula: VerifierFormula) -> float:
    geometry = basin_score(basin, basin_formula)
    verifier = weighted_verifier_score(verifier_features(basin, question), verifier_formula)
    return safe_float(verifier_formula["geometry_weight"]) * geometry + safe_float(verifier_formula["verifier_weight"]) * verifier


def choose_v3(group: list[dict[str, Any]], config: dict[str, Any], prefix_k: int | None = None) -> dict[str, Any]:
    current_group = group[:prefix_k] if prefix_k is not None else group
    sample0 = current_group[0]
    if risk_score(current_group, config["risk_formula"]) < config["risk_threshold"]:
        return sample0
    basins = pair_basins(current_group)
    sample0_basin = next(basin for basin in basins if basin["contains_sample0"] > 0.0)
    alternatives = [basin for basin in basins if basin["contains_sample0"] <= 0.0]
    if not alternatives:
        return sample0
    question = str(sample0["question"])
    best_alt = max(
        alternatives,
        key=lambda basin: (
            basin_score_v3(basin, question, config["basin_formula"], config["verifier_formula"]),
            -safe_float(basin["representative"]["sample_index"]),
        ),
    )
    margin = basin_score_v3(best_alt, question, config["basin_formula"], config["verifier_formula"]) - basin_score_v3(
        sample0_basin,
        question,
        config["basin_formula"],
        config["verifier_formula"],
    )
    if margin < config["switch_margin"]:
        return sample0
    return best_alt["representative"]


def metric_objective(metrics: dict[str, Any], damage_penalty: float, rescue_weight: float, switch_penalty: float) -> float:
    return (
        100.0 * safe_float(metrics["delta_vs_sample0"])
        + rescue_weight * int(metrics["improved_count"])
        - damage_penalty * int(metrics["damaged_count"])
        - switch_penalty * safe_float(metrics["answer_changed_rate"])
    )


def evaluate_config(groups: list[list[dict[str, Any]]], config: dict[str, Any], method: str, split: str) -> dict[str, Any]:
    selected = [choose_v3(group, config) for group in groups]
    metrics = evaluate_selected(groups, selected, method, split)
    metrics.update(
        {
            "risk_formula": config["risk_formula"]["name"],
            "basin_formula": config["basin_formula"]["name"],
            "verifier_formula": config["verifier_formula"]["name"],
            "risk_threshold": config["risk_threshold"],
            "switch_margin": config["switch_margin"],
            "damage_penalty": config["damage_penalty"],
            "rescue_weight": config["rescue_weight"],
        }
    )
    return metrics


def train_config(
    groups: list[list[dict[str, Any]]],
    damage_penalty: float,
    rescue_weight: float,
    allowed_verifiers: list[str] | None = None,
) -> dict[str, Any]:
    best_config: dict[str, Any] | None = None
    best_objective = -1e18
    verifiers = [vf for vf in verifier_formulas() if allowed_verifiers is None or vf["name"] in allowed_verifiers]
    for risk_formula in compact_risk_formulas():
        thresholds = compact_thresholds([risk_score(group, risk_formula) for group in groups])
        for risk_threshold in thresholds:
            for basin_formula in compact_basin_formulas():
                for verifier_formula in verifiers:
                    for switch_margin in [-0.25, 0.25, 1.0]:
                        config = {
                            "risk_formula": risk_formula,
                            "risk_threshold": risk_threshold,
                            "basin_formula": basin_formula,
                            "verifier_formula": verifier_formula,
                            "switch_margin": switch_margin,
                            "damage_penalty": damage_penalty,
                            "rescue_weight": rescue_weight,
                        }
                        metrics = evaluate_config(groups, config, "train", "train")
                        objective = metric_objective(metrics, damage_penalty, rescue_weight, switch_penalty=0.35)
                        if objective > best_objective:
                            best_objective = objective
                            best_config = config
    assert best_config is not None
    best_config["train_objective"] = best_objective
    return best_config


def summarize(rows: list[dict[str, Any]], group_key: str, split_name: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        out.append(
            {
                "split": split_name,
                group_key: key,
                "pairs": sum(int(item["pairs"]) for item in items),
                "strict_correct_rate": mean([safe_float(item["strict_correct_rate"]) for item in items]),
                "sample0_strict_correct_rate": mean([safe_float(item["sample0_strict_correct_rate"]) for item in items]),
                "delta_vs_sample0": mean([safe_float(item["delta_vs_sample0"]) for item in items]),
                "improved_count": sum(int(item["improved_count"]) for item in items),
                "damaged_count": sum(int(item["damaged_count"]) for item in items),
                "net_gain_count": sum(int(item["net_gain_count"]) for item in items),
                "answer_changed_rate": mean([safe_float(item["answer_changed_rate"]) for item in items]),
                "rescue_recall": mean([safe_float(item["rescue_recall"]) for item in items]),
                "damage_avoidance": mean([safe_float(item["damage_avoidance"]) for item in items]),
                "avg_selected_sample_index": mean([safe_float(item["avg_selected_sample_index"]) for item in items]),
            }
        )
    return out


def run_pareto_cv(grouped: dict[tuple[int, str], list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    for fold_idx, (train_groups, test_groups) in enumerate(grouped_folds(grouped)):
        for setting in pareto_settings():
            config = train_config(train_groups, setting["damage_penalty"], setting["rescue_weight"])
            method = f"v3_{setting['name']}"
            metrics = evaluate_config(test_groups, config, method, f"question_fold_{fold_idx}")
            metrics["pareto_setting"] = setting["name"]
            fold_rows.append(metrics)
            configs.append(
                {
                    "split": f"question_fold_{fold_idx}",
                    "pareto_setting": setting["name"],
                    "risk_formula": config["risk_formula"]["name"],
                    "basin_formula": config["basin_formula"]["name"],
                    "verifier_formula": config["verifier_formula"]["name"],
                    "risk_threshold": config["risk_threshold"],
                    "switch_margin": config["switch_margin"],
                    "damage_penalty": setting["damage_penalty"],
                    "rescue_weight": setting["rescue_weight"],
                    "train_objective": config["train_objective"],
                }
            )
            for group in test_groups:
                selected = choose_v3(group, config)
                sample0 = group[0]
                basin = next(item for item in pair_basins(group) if int(item["cluster_id"]) == int(selected["cluster_id"]))
                vf = verifier_features(basin, str(sample0["question"]))
                selections.append(
                    {
                        "split": f"question_fold_{fold_idx}",
                        "pareto_setting": setting["name"],
                        "seed": selected["seed"],
                        "question_id": selected["question_id"],
                        "selected_sample_index": selected["sample_index"],
                        "sample0_strict_correct": sample0["strict_correct"],
                        "selected_strict_correct": selected["strict_correct"],
                        "delta_strict_correct": safe_float(selected["strict_correct"]) - safe_float(sample0["strict_correct"]),
                        "selected_cluster_id": selected["cluster_id"],
                        "sample0_cluster_id": sample0["cluster_id"],
                        "risk_score": risk_score(group, config["risk_formula"]),
                        "verifier_formula": config["verifier_formula"]["name"],
                        "answer_type_match": vf["answer_type_match"],
                        "focus_score": vf["focus_score"],
                        "consensus_jaccard": vf["consensus_jaccard"],
                        "bad_marker_rate": vf["bad_marker_rate"],
                        "len_penalty": vf["len_penalty"],
                        "selected_preview": selected["answer_preview"],
                    }
                )
    return summarize(fold_rows, "method", "question_grouped_cv") + fold_rows, selections, configs


def run_ablation_cv(grouped: dict[tuple[int, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for verifier in [vf["name"] for vf in verifier_formulas()]:
        fold_rows: list[dict[str, Any]] = []
        for fold_idx, (train_groups, test_groups) in enumerate(grouped_folds(grouped)):
            config = train_config(train_groups, damage_penalty=1.0, rescue_weight=1.0, allowed_verifiers=[verifier])
            metrics = evaluate_config(test_groups, config, f"ablate_{verifier}", f"question_fold_{fold_idx}")
            metrics["verifier_ablation"] = verifier
            fold_rows.append(metrics)
        rows.extend(summarize(fold_rows, "method", "question_grouped_cv"))
        rows.extend(fold_rows)
    return rows


def adaptive_select(group: list[dict[str, Any]], config: dict[str, Any], max_k: int) -> tuple[dict[str, Any], int]:
    sample0 = group[0]
    for k in range(1, max_k + 1):
        prefix = group[:k]
        if k >= 2:
            sample0_cluster_size = sum(1 for row in prefix if int(row["cluster_id"]) == int(sample0["cluster_id"]))
            stable = (
                sample0_cluster_size >= 2
                and safe_float(sample0["logprob_avg_z"]) >= -0.2
                and safe_float(sample0["token_mean_entropy_z"]) <= 0.15
            )
            if stable:
                return sample0, k
            selected = choose_v3(prefix, config)
            if int(selected["sample_index"]) != 0:
                return selected, k
    return choose_v3(group[:max_k], config), max_k


def run_sequential_cv(grouped: dict[tuple[int, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for max_k in [2, 3, 4, 8]:
        fold_rows: list[dict[str, Any]] = []
        for fold_idx, (train_groups, test_groups) in enumerate(grouped_folds(grouped)):
            config = train_config(train_groups, damage_penalty=1.0, rescue_weight=1.0)
            selected: list[dict[str, Any]] = []
            costs: list[float] = []
            for group in test_groups:
                row, cost = adaptive_select(group, config, max_k)
                selected.append(row)
                costs.append(float(cost))
            metrics = evaluate_selected(test_groups, selected, f"v3_adaptive_k{max_k}", f"question_fold_{fold_idx}")
            metrics["avg_generated_candidates"] = mean(costs)
            metrics["max_candidates"] = max_k
            fold_rows.append(metrics)
        summary = summarize(fold_rows, "method", "question_grouped_cv")
        for row in summary:
            matching = [item for item in fold_rows if item["method"] == row["method"]]
            row["avg_generated_candidates"] = mean([safe_float(item["avg_generated_candidates"]) for item in matching])
            row["max_candidates"] = max_k
        rows.extend(summary)
        rows.extend(fold_rows)
    return rows


def run_seed_split(grouped: dict[tuple[int, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for train_seed, test_seed in [(42, 43), (43, 42)]:
        train_groups = [group for (seed, _qid), group in grouped.items() if seed == train_seed]
        test_groups = [group for (seed, _qid), group in grouped.items() if seed == test_seed]
        for setting in pareto_settings():
            config = train_config(train_groups, setting["damage_penalty"], setting["rescue_weight"])
            metrics = evaluate_config(test_groups, config, f"v3_{setting['name']}", f"train_seed_{train_seed}_test_seed_{test_seed}")
            metrics["pareto_setting"] = setting["name"]
            rows.append(metrics)
    return rows


def feature_diagnostics(selection_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in selection_rows if row["pareto_setting"] == "balanced" and int(float(row["selected_sample_index"])) != 0]
    labels = [("rescued", lambda row: safe_float(row["delta_strict_correct"]) > 0), ("damaged", lambda row: safe_float(row["delta_strict_correct"]) < 0)]
    features = ["answer_type_match", "focus_score", "consensus_jaccard", "bad_marker_rate", "len_penalty", "risk_score"]
    out: list[dict[str, Any]] = []
    for label_name, predicate in labels:
        pos = [row for row in rows if predicate(row)]
        neg = [row for row in rows if not predicate(row)]
        for feature in features:
            out.append(
                {
                    "comparison": f"{label_name}_vs_other_switched",
                    "feature": feature,
                    "positive_count": len(pos),
                    "negative_count": len(neg),
                    "positive_mean": mean([safe_float(row[feature]) for row in pos]),
                    "negative_mean": mean([safe_float(row[feature]) for row in neg]),
                    "mean_diff": mean([safe_float(row[feature]) for row in pos]) - mean([safe_float(row[feature]) for row in neg]),
                }
            )
    return out


def interesting_cases(selection_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in selection_rows if row["pareto_setting"] == "balanced" and safe_float(row["delta_strict_correct"]) != 0.0]
    rows.sort(key=lambda row: (safe_float(row["delta_strict_correct"]), -safe_float(row["risk_score"])))
    return rows[:20] + rows[-20:]


def make_plots(output_dir: Path, pareto_rows: list[dict[str, Any]], ablation_rows: list[dict[str, Any]], seq_rows: list[dict[str, Any]], diag_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = ensure_dir(output_dir / "plots")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    summary = [row for row in pareto_rows if row["split"] == "question_grouped_cv"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter([safe_float(row["damaged_count"]) for row in summary], [safe_float(row["improved_count"]) for row in summary], s=90)
    for row in summary:
        ax.text(safe_float(row["damaged_count"]) + 0.05, safe_float(row["improved_count"]), row["method"], fontsize=8)
    ax.set_xlabel("Damaged cases")
    ax.set_ylabel("Improved cases")
    ax.set_title("v3 Rescue-Damage Pareto")
    fig.tight_layout()
    fig.savefig(plot_dir / "01_v3_rescue_damage_pareto.png")
    plt.close(fig)

    ab_summary = [row for row in ablation_rows if row["split"] == "question_grouped_cv"]
    fig, ax = plt.subplots(figsize=(10.0, 4.7))
    ax.bar([row["method"].replace("ablate_", "") for row in ab_summary], [safe_float(row["strict_correct_rate"]) for row in ab_summary], color="#4c78a8", alpha=0.82)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Strict correctness")
    ax.set_title("Verifier Feature Ablations")
    fig.tight_layout()
    fig.savefig(plot_dir / "02_v3_ablation_accuracy.png")
    plt.close(fig)

    seq = [row for row in seq_rows if row["split"] == "question_grouped_cv"]
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.plot([safe_float(row["avg_generated_candidates"]) for row in seq], [safe_float(row["strict_correct_rate"]) for row in seq], marker="o")
    for row in seq:
        ax.text(safe_float(row["avg_generated_candidates"]), safe_float(row["strict_correct_rate"]), row["method"], fontsize=8)
    ax.set_xlabel("Average generated candidates")
    ax.set_ylabel("Strict correctness")
    ax.set_title("v3 Adaptive Sampling")
    fig.tight_layout()
    fig.savefig(plot_dir / "03_v3_cost_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = [f"{row['comparison']}\n{row['feature']}" for row in diag_rows]
    values = [safe_float(row["mean_diff"]) for row in diag_rows]
    ax.barh(labels[::-1], values[::-1], color=["#31a354" if value >= 0 else "#de2d26" for value in values[::-1]])
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_xlabel("Mean difference")
    ax.set_title("Verifier Signals on Switched Cases")
    fig.tight_layout()
    fig.savefig(plot_dir / "04_v3_verifier_diagnostics.png")
    plt.close(fig)


def build_report(
    output_dir: Path,
    pareto_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    sequential_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    diag_rows: list[dict[str, Any]],
    config_rows: list[dict[str, Any]],
) -> None:
    summary = [row for row in pareto_rows if row["split"] == "question_grouped_cv"]
    ab_summary = [row for row in ablation_rows if row["split"] == "question_grouped_cv"]
    seq = [row for row in sequential_rows if row["split"] == "question_grouped_cv"]
    lines = [
        "# Basin Verifier v3",
        "",
        "## What v3 Adds",
        "",
        "v3 keeps the v2 basin geometry, but adds cheap verifier features: answer-type match, basin consensus, lexical focus, bad-marker penalties, and length penalties.",
        "",
        "## Pareto Results",
        "",
        "| Method | Strict | Delta | Improved | Damaged | Net | Rescue Recall | Damage Avoid | Changed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{int(row['net_gain_count'])}` | "
            f"`{safe_float(row['rescue_recall']):.2%}` | `{safe_float(row['damage_avoidance']):.2%}` | `{safe_float(row['answer_changed_rate']):.2%}` |"
        )
    lines.extend(["", "## Verifier Ablations", "", "| Ablation | Strict | Delta | Improved | Damaged | Rescue Recall |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in ab_summary:
        lines.append(
            f"| `{row['method'].replace('ablate_', '')}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{safe_float(row['rescue_recall']):.2%}` |"
        )
    lines.extend(["", "## Adaptive Cost", "", "| Policy | Avg candidates | Strict | Delta | Improved | Damaged |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in seq:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['avg_generated_candidates']):.2f}` | `{safe_float(row['strict_correct_rate']):.2%}` | "
            f"`{safe_float(row['delta_vs_sample0']):.2%}` | `{int(row['improved_count'])}` | `{int(row['damaged_count'])}` |"
        )
    lines.extend(["", "## Seed Robustness", "", "| Split | Method | Strict | Delta | Improved | Damaged |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in seed_rows:
        lines.append(
            f"| `{row['split']}` | `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` |"
        )
    lines.extend(["", "## Verifier Diagnostics On Switched Cases", "", "| Comparison | Feature | Pos mean | Neg mean | Diff |", "| --- | --- | ---: | ---: | ---: |"])
    for row in diag_rows:
        lines.append(
            f"| `{row['comparison']}` | `{row['feature']}` | `{safe_float(row['positive_mean']):.4f}` | `{safe_float(row['negative_mean']):.4f}` | `{safe_float(row['mean_diff']):.4f}` |"
        )
    config_counter = Counter((row["pareto_setting"], row["verifier_formula"]) for row in config_rows)
    lines.extend(["", "## Frequently Selected Verifiers", "", "| Setting | Verifier | Count |", "| --- | --- | ---: |"])
    for (setting, verifier), count in config_counter.most_common(12):
        lines.append(f"| `{setting}` | `{verifier}` | `{count}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If verifier features reduce damaged cases without losing too many rescue cases, they are useful as safety checks. If they mainly reduce switching and do not improve the rescue-damage frontier, then cheap lexical verifier features are insufficient and a learned verifier/NLI-style scorer is needed.",
        ]
    )
    (output_dir / "basin_verifier_v3_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_run = Path(args.input_run)
    output_dir = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    candidate_rows = read_csv(input_run / "candidate_features.csv")
    add_candidate_features(candidate_rows)
    _attractors, lookup = build_attractor_tables(candidate_rows)
    attach_attractor_features(candidate_rows, lookup)
    grouped = group_candidates(candidate_rows)

    pareto_rows, selection_rows, config_rows = run_pareto_cv(grouped)
    ablation_rows = run_ablation_cv(grouped)
    sequential_rows = run_sequential_cv(grouped)
    seed_rows = run_seed_split(grouped)
    diag_rows = feature_diagnostics(selection_rows)
    cases = interesting_cases(selection_rows)

    write_dict_csv(output_dir / "v3_pareto_cv_results.csv", pareto_rows)
    write_dict_csv(output_dir / "v3_selection_rows.csv", selection_rows)
    write_dict_csv(output_dir / "v3_trained_configs.csv", config_rows)
    write_dict_csv(output_dir / "v3_ablation_cv_results.csv", ablation_rows)
    write_dict_csv(output_dir / "v3_sequential_cost_curve.csv", sequential_rows)
    write_dict_csv(output_dir / "v3_seed_split_robustness.csv", seed_rows)
    write_dict_csv(output_dir / "v3_verifier_diagnostics.csv", diag_rows)
    write_dict_csv(output_dir / "v3_interesting_cases.csv", cases)
    write_json(
        output_dir / "run_metadata.json",
        {
            "input_run": str(input_run),
            "candidate_count": len(candidate_rows),
            "group_count": len(grouped),
            "verifier_formulas": [formula["name"] for formula in verifier_formulas()],
        },
    )
    make_plots(output_dir, pareto_rows, ablation_rows, sequential_rows, diag_rows)
    build_report(output_dir, pareto_rows, ablation_rows, sequential_rows, seed_rows, diag_rows, config_rows)
    print(json.dumps({"output_dir": str(output_dir), "group_count": len(grouped)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
