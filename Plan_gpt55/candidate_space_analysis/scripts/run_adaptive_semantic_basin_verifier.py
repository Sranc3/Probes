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

from run_entropy_anatomy import ensure_dir, mean, safe_float, write_csv, write_json
from run_learned_basin_verifier import LogisticModel, question_folds, stdev

try:
    import torch
except Exception:  # pragma: no cover - CPU fallback for environments without torch.
    torch = None


DEFAULT_CANDIDATE_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"


CONTENT_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "was",
    "were",
    "is",
    "are",
    "that",
    "which",
    "who",
    "what",
    "when",
    "where",
    "did",
    "does",
    "had",
    "has",
    "its",
    "it",
    "as",
    "at",
    "from",
    "this",
    "his",
    "her",
    "their",
    "be",
}

BAD_MARKERS = [
    "there is no",
    "there are no",
    "not enough information",
    "cannot determine",
    "can't determine",
    "unknown",
    "unclear",
    "no definitive",
    "i don't know",
    "confusion",
    "however",
    "not ",
]


NUMERIC_FEATURES = [
    "obs_cluster_size",
    "obs_cluster_mass",
    "obs_semantic_clusters",
    "obs_fragmentation_entropy",
    "obs_top_cluster_margin",
    "token_mean_entropy_mean",
    "token_max_entropy_mean",
    "logprob_avg_mean",
    "token_count_mean",
    "rep_logprob_z",
    "rep_entropy_z",
    "rep_len_z",
    "delta_entropy_vs_sample0",
    "delta_logprob_vs_sample0",
    "delta_mass_vs_sample0",
]

TEXT_FEATURES = [
    "expected_person",
    "expected_place",
    "expected_time",
    "expected_number",
    "answer_has_year",
    "answer_has_number",
    "answer_has_person_shape",
    "answer_has_place_marker",
    "type_match_score",
    "bad_marker_score",
    "uncertainty_marker_score",
    "question_answer_content_overlap",
    "consensus_core_size",
    "consensus_core_coverage",
    "answer_len_tokens",
    "answer_quote_count",
]

FEATURE_SETS = {
    "numeric_adaptive": NUMERIC_FEATURES,
    "semantic_factual": TEXT_FEATURES,
    "full_adaptive": NUMERIC_FEATURES + TEXT_FEATURES,
}


class TorchLogisticModel(LogisticModel):
    def fit(self, rows: list[dict[str, Any]], label: str, epochs: int = 180, lr: float = 0.1, l2: float = 0.02) -> None:
        if torch is None or not torch.cuda.is_available():
            super().fit(rows, label, epochs=epochs, lr=lr, l2=l2)
            return
        if not rows:
            return
        device = torch.device("cuda:0")
        values_by_feature = []
        for feature in self.features:
            values = [safe_float(row.get(feature, 0.0)) for row in rows]
            self.means[feature] = mean(values)
            self.stds[feature] = stdev(values) or 1.0
            values_by_feature.append([(value - self.means[feature]) / self.stds[feature] for value in values])
        x = torch.tensor(list(zip(*values_by_feature)), dtype=torch.float32, device=device)
        y = torch.tensor([1.0 if safe_float(row[label]) > 0.0 else 0.0 for row in rows], dtype=torch.float32, device=device)
        pos = torch.clamp(y.sum(), min=1.0)
        neg = torch.clamp(torch.tensor(float(len(rows)), device=device) - y.sum(), min=1.0)
        sample_w = torch.where(y > 0.5, len(rows) / (2.0 * pos), len(rows) / (2.0 * neg))
        w = torch.zeros(x.shape[1], dtype=torch.float32, device=device, requires_grad=True)
        b = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([w, b], lr=lr, weight_decay=l2)
        for _ in range(epochs):
            logits = x @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, weight=sample_w)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        weights = w.detach().cpu().tolist()
        self.bias = float(b.detach().cpu().item())
        for feature, weight in zip(self.features, weights):
            self.weights[feature] = float(weight)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run production-style adaptive semantic basin verifier.")
    parser.add_argument("--candidate-run", default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="adaptive_semantic_basin_verifier")
    parser.add_argument("--max-candidates", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9']+", text)]


def content_tokens(text: str) -> list[str]:
    return [token for token in tokens(text) if len(token) > 2 and token not in CONTENT_STOPWORDS]


def entropy_from_counts(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        value -= p * math.log(p)
    return value


def expected_type_features(question: str) -> dict[str, float]:
    q = question.lower()
    expected_person = float(q.startswith("who") or " who " in q or "whose" in q)
    expected_place = float(q.startswith("where") or " in which country" in q or " in which city" in q or "which country" in q or "which city" in q)
    expected_time = float(q.startswith("when") or "which year" in q or "what year" in q or "which decade" in q or "what decade" in q or "date" in q)
    expected_number = float(q.startswith("how many") or q.startswith("how much") or "number of" in q)
    return {
        "expected_person": expected_person,
        "expected_place": expected_place,
        "expected_time": expected_time,
        "expected_number": expected_number,
    }


def answer_shape_features(question: str, answer: str, basin_answers: list[str]) -> dict[str, float]:
    lower = answer.lower()
    toks = tokens(answer)
    q_toks = set(content_tokens(question))
    a_toks = set(content_tokens(answer))
    expected = expected_type_features(question)
    has_year = float(bool(re.search(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2}|\d{2}0s|\d{2}ies)\b", lower)))
    has_number = float(bool(re.search(r"\b\d+(\.\d+)?\b", lower)) or any(token in {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"} for token in toks))
    has_person_shape = float(bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", answer)))
    has_place_marker = float(any(marker in lower for marker in ["city", "country", "state", "island", "river", "mount", "lake", "province", "county"]))
    type_targets = [
        (expected["expected_time"], has_year or has_number),
        (expected["expected_number"], has_number),
        (expected["expected_person"], has_person_shape),
        (expected["expected_place"], has_place_marker or has_person_shape),
    ]
    active = [match for is_expected, match in type_targets if is_expected > 0]
    type_match = mean(active) if active else 0.5
    bad_count = sum(1 for marker in BAD_MARKERS if marker in lower)
    uncertainty_count = sum(1 for marker in ["maybe", "possibly", "probably", "appears", "seems", "unclear", "unknown"] if marker in lower)
    overlap = len(q_toks & a_toks) / max(1, len(q_toks | a_toks))
    answer_token_sets = [set(content_tokens(item)) for item in basin_answers]
    if answer_token_sets:
        core = set.intersection(*answer_token_sets) if len(answer_token_sets) > 1 else answer_token_sets[0]
        core = {token for token in core if token not in q_toks}
    else:
        core = set()
    return {
        **expected,
        "answer_has_year": has_year,
        "answer_has_number": has_number,
        "answer_has_person_shape": has_person_shape,
        "answer_has_place_marker": has_place_marker,
        "type_match_score": type_match,
        "bad_marker_score": min(1.0, bad_count / 2),
        "uncertainty_marker_score": min(1.0, uncertainty_count / 2),
        "question_answer_content_overlap": overlap,
        "consensus_core_size": float(len(core)),
        "consensus_core_coverage": len(core & a_toks) / max(1, len(a_toks)),
        "answer_len_tokens": float(len(toks)),
        "answer_quote_count": float(answer.count('"')),
    }


def representative(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            safe_float(row["logprob_avg_z"]) - 0.35 * safe_float(row["token_mean_entropy_z"]) - 0.1 * safe_float(row["token_count_z"]),
            -int(row["sample_index"]),
        ),
    )


def build_observed_basins(prefix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in prefix_rows:
        by_cluster[int(row["cluster_id"])].append(row)
    cluster_counts = [len(rows) for rows in by_cluster.values()]
    total = len(prefix_rows)
    frag_entropy = entropy_from_counts(cluster_counts)
    top_counts = sorted(cluster_counts, reverse=True)
    top_margin = (top_counts[0] - top_counts[1]) / total if len(top_counts) > 1 else 1.0
    sample0_rows = by_cluster[int(prefix_rows[0]["cluster_id"])]
    sample0_rep = representative(sample0_rows)
    sample0_entropy = safe_float(sample0_rep["token_mean_entropy_z"])
    sample0_logprob = safe_float(sample0_rep["logprob_avg_z"])
    sample0_mass = len(sample0_rows) / total
    basins = []
    for cluster_id, rows in by_cluster.items():
        rep = representative(rows)
        row = {
            "seed": int(rep["seed"]),
            "question_id": rep["question_id"],
            "question_index": int(rep["question_index"]),
            "question": rep["question"],
            "cluster_id": cluster_id,
            "contains_sample0": float(cluster_id == int(prefix_rows[0]["cluster_id"])),
            "sample0_cluster_id": int(prefix_rows[0]["cluster_id"]),
            "sample0_strict_correct": safe_float(prefix_rows[0]["strict_correct"]),
            "representative_sample_index": int(rep["sample_index"]),
            "representative_strict_correct": safe_float(rep["strict_correct"]),
            "representative_preview": rep["answer_preview"],
            "obs_prefix_size": total,
            "obs_cluster_size": len(rows),
            "obs_cluster_mass": len(rows) / total,
            "obs_semantic_clusters": len(by_cluster),
            "obs_fragmentation_entropy": frag_entropy,
            "obs_top_cluster_margin": top_margin,
            "token_mean_entropy_mean": mean([safe_float(item["token_mean_entropy"]) for item in rows]),
            "token_max_entropy_mean": mean([safe_float(item["token_max_entropy"]) for item in rows]),
            "logprob_avg_mean": mean([safe_float(item["logprob_avg"]) for item in rows]),
            "token_count_mean": mean([safe_float(item["token_count"]) for item in rows]),
            "rep_logprob_z": safe_float(rep["logprob_avg_z"]),
            "rep_entropy_z": safe_float(rep["token_mean_entropy_z"]),
            "rep_len_z": safe_float(rep["token_count_z"]),
        }
        row["delta_entropy_vs_sample0"] = safe_float(row["rep_entropy_z"]) - sample0_entropy
        row["delta_logprob_vs_sample0"] = safe_float(row["rep_logprob_z"]) - sample0_logprob
        row["delta_mass_vs_sample0"] = safe_float(row["obs_cluster_mass"]) - sample0_mass
        row.update(answer_shape_features(str(rep["question"]), str(rep["answer_text"]), [str(item["answer_text"]) for item in rows]))
        row["switch_gain_label"] = float(
            safe_float(prefix_rows[0]["strict_correct"]) <= 0 and safe_float(rep["strict_correct"]) > 0 and cluster_id != int(prefix_rows[0]["cluster_id"])
        )
        basins.append(row)
    basins.sort(key=lambda row: int(row["cluster_id"]))
    return basins


def build_prefix_dataset(candidate_rows: list[dict[str, Any]], max_candidates: int) -> tuple[dict[tuple[int, str], list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_pair_samples: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_pair_samples[(int(row["seed"]), str(row["question_id"]))].append(row)
    by_pair_prefixes: dict[tuple[int, str], list[dict[str, Any]]] = {}
    sample0_rows = []
    alt_rows = []
    for key, rows in by_pair_samples.items():
        rows.sort(key=lambda row: int(row["sample_index"]))
        prefixes = []
        for k in range(1, min(max_candidates, len(rows)) + 1):
            basins = build_observed_basins(rows[:k])
            prefixes.append({"prefix_size": k, "samples": rows[:k], "basins": basins})
            sample0_rows.append(next(row for row in basins if safe_float(row["contains_sample0"]) > 0))
            alt_rows.extend([row for row in basins if safe_float(row["contains_sample0"]) <= 0])
        by_pair_prefixes[key] = prefixes
    return by_pair_prefixes, sample0_rows, alt_rows


def sample0_at(prefix: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in prefix["basins"] if safe_float(row["contains_sample0"]) > 0)


def choose_prefix_basin(prefix: dict[str, Any], switch_model: LogisticModel, switch_threshold: float, basin_model: LogisticModel | None = None, basin_margin: float = 0.0) -> dict[str, Any]:
    sample0 = sample0_at(prefix)
    alternatives = [row for row in prefix["basins"] if safe_float(row["contains_sample0"]) <= 0]
    if not alternatives:
        return sample0
    alt = max(alternatives, key=switch_model.predict)
    if switch_model.predict(alt) < switch_threshold:
        return sample0
    if basin_model is not None and basin_model.predict(alt) - basin_model.predict(sample0) < basin_margin:
        return sample0
    return alt


def choose_best_basin(prefix: dict[str, Any], basin_model: LogisticModel, margin: float) -> dict[str, Any]:
    sample0 = sample0_at(prefix)
    best = max(prefix["basins"], key=basin_model.predict)
    if int(best["cluster_id"]) != int(sample0["cluster_id"]) and basin_model.predict(best) - basin_model.predict(sample0) >= margin:
        return best
    return sample0


def simulate_pair(
    prefixes: list[dict[str, Any]],
    trust_model: LogisticModel,
    switch_model: LogisticModel,
    trust_threshold: float,
    switch_threshold: float,
    min_candidates: int,
    max_candidates: int,
    basin_model: LogisticModel | None = None,
    basin_margin: float = 0.0,
    budget_best: bool = False,
) -> tuple[dict[str, Any], int, str]:
    first = prefixes[0]
    sample0 = sample0_at(first)
    if trust_model.predict(sample0) >= trust_threshold:
        return sample0, 1, "trust_sample0"
    last_choice = sample0
    for prefix in prefixes[1:max_candidates]:
        choice = choose_prefix_basin(prefix, switch_model, switch_threshold, basin_model, basin_margin)
        last_choice = choice
        if int(prefix["prefix_size"]) >= min_candidates and int(choice["cluster_id"]) != int(sample0["cluster_id"]):
            return choice, int(prefix["prefix_size"]), "switch"
    final_prefix = prefixes[min(max_candidates, len(prefixes)) - 1]
    if budget_best and basin_model is not None:
        return choose_best_basin(final_prefix, basin_model, basin_margin), int(final_prefix["prefix_size"]), "budget_best"
    return choose_prefix_basin(final_prefix, switch_model, switch_threshold, basin_model, basin_margin), int(final_prefix["prefix_size"]), "budget"


def evaluate(prefix_groups: list[list[dict[str, Any]]], selections: list[tuple[dict[str, Any], int, str]], method: str, split: str) -> dict[str, Any]:
    sample0s = [sample0_at(prefixes[0]) for prefixes in prefix_groups]
    selected_rows = [item[0] for item in selections]
    generated = [item[1] for item in selections]
    stop_reasons = Counter(item[2] for item in selections)
    generated_tokens = [
        sum(safe_float(sample["token_count"]) for sample in prefixes[min(gen, len(prefixes)) - 1]["samples"])
        for prefixes, gen in zip(prefix_groups, generated)
    ]
    sample0_tokens = [safe_float(prefixes[0]["samples"][0]["token_count"]) for prefixes in prefix_groups]
    full_tokens = [sum(safe_float(sample["token_count"]) for sample in prefixes[-1]["samples"]) for prefixes in prefix_groups]
    deltas = [safe_float(sel["representative_strict_correct"]) - safe_float(s0["representative_strict_correct"]) for sel, s0 in zip(selected_rows, sample0s)]
    return {
        "split": split,
        "method": method,
        "pairs": len(prefix_groups),
        "strict_correct_rate": mean([safe_float(row["representative_strict_correct"]) for row in selected_rows]),
        "sample0_strict_correct_rate": mean([safe_float(row["representative_strict_correct"]) for row in sample0s]),
        "delta_vs_sample0": mean(deltas),
        "improved_count": sum(1 for value in deltas if value > 0),
        "damaged_count": sum(1 for value in deltas if value < 0),
        "net_gain_count": sum(1 for value in deltas if value > 0) - sum(1 for value in deltas if value < 0),
        "answer_changed_rate": mean([float(int(sel["cluster_id"]) != int(s0["cluster_id"])) for sel, s0 in zip(selected_rows, sample0s)]),
        "avg_generated_candidates": mean(generated),
        "avg_generated_tokens": mean(generated_tokens),
        "avg_sample0_tokens": mean(sample0_tokens),
        "avg_full8_tokens": mean(full_tokens),
        "token_cost_vs_sample0": mean([gen / max(1.0, base) for gen, base in zip(generated_tokens, sample0_tokens)]),
        "token_cost_vs_full8": mean([gen / max(1.0, full) for gen, full in zip(generated_tokens, full_tokens)]),
        "token_savings_vs_full8": 1.0 - mean([gen / max(1.0, full) for gen, full in zip(generated_tokens, full_tokens)]),
        "median_generated_candidates": sorted(generated)[len(generated) // 2] if generated else 0,
        "max_generated_candidates": max(generated) if generated else 0,
        "trust_stop_rate": stop_reasons["trust_sample0"] / max(1, len(selections)),
        "switch_stop_rate": stop_reasons["switch"] / max(1, len(selections)),
        "budget_stop_rate": stop_reasons["budget"] / max(1, len(selections)),
        "budget_best_stop_rate": stop_reasons["budget_best"] / max(1, len(selections)),
    }


def tune_policy(
    train_prefix_groups: list[list[dict[str, Any]]],
    trust_model: LogisticModel,
    switch_model: LogisticModel,
    basin_model: LogisticModel | None,
    max_candidates: int,
    favor_efficiency: float,
) -> dict[str, Any]:
    best: tuple[float, dict[str, Any]] = (-1e9, {})
    trust_thresholds = [0.35, 0.5, 0.7, 0.85]
    switch_thresholds = [0.35, 0.5, 0.65, 0.8]
    min_candidates_options = [2, 3, 4]
    margins = [-0.2, 0.0, 0.2] if basin_model else [0.0]
    budget_options = [False, True] if basin_model else [False]
    for trust_threshold in trust_thresholds:
        for switch_threshold in switch_thresholds:
            for min_candidates in min_candidates_options:
                for margin in margins:
                    for budget_best in budget_options:
                        selections = [
                            simulate_pair(prefixes, trust_model, switch_model, trust_threshold, switch_threshold, min_candidates, max_candidates, basin_model, margin, budget_best)
                            for prefixes in train_prefix_groups
                        ]
                        metrics = evaluate(train_prefix_groups, selections, "train", "train")
                        score = (
                            100 * safe_float(metrics["delta_vs_sample0"])
                            + int(metrics["net_gain_count"])
                            - 3.0 * int(metrics["damaged_count"])
                            - favor_efficiency * safe_float(metrics["avg_generated_candidates"])
                        )
                        if score > best[0]:
                            best = (
                                score,
                                {
                                    "trust_threshold": trust_threshold,
                                    "switch_threshold": switch_threshold,
                                    "min_candidates": min_candidates,
                                    "basin_margin": margin,
                                    "budget_best": budget_best,
                                },
                            )
    return best[1]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summary = []
    for method, items in sorted(grouped.items()):
        summary.append(
            {
                "split": "question_grouped_cv",
                "method": method,
                "pairs": sum(int(item["pairs"]) for item in items),
                "strict_correct_rate": mean([safe_float(item["strict_correct_rate"]) for item in items]),
                "sample0_strict_correct_rate": mean([safe_float(item["sample0_strict_correct_rate"]) for item in items]),
                "delta_vs_sample0": mean([safe_float(item["delta_vs_sample0"]) for item in items]),
                "improved_count": sum(int(item["improved_count"]) for item in items),
                "damaged_count": sum(int(item["damaged_count"]) for item in items),
                "net_gain_count": sum(int(item["net_gain_count"]) for item in items),
                "answer_changed_rate": mean([safe_float(item["answer_changed_rate"]) for item in items]),
                "avg_generated_candidates": mean([safe_float(item["avg_generated_candidates"]) for item in items]),
                "avg_generated_tokens": mean([safe_float(item["avg_generated_tokens"]) for item in items]),
                "avg_sample0_tokens": mean([safe_float(item["avg_sample0_tokens"]) for item in items]),
                "avg_full8_tokens": mean([safe_float(item["avg_full8_tokens"]) for item in items]),
                "token_cost_vs_sample0": mean([safe_float(item["token_cost_vs_sample0"]) for item in items]),
                "token_cost_vs_full8": mean([safe_float(item["token_cost_vs_full8"]) for item in items]),
                "token_savings_vs_full8": mean([safe_float(item["token_savings_vs_full8"]) for item in items]),
                "trust_stop_rate": mean([safe_float(item["trust_stop_rate"]) for item in items]),
                "switch_stop_rate": mean([safe_float(item["switch_stop_rate"]) for item in items]),
                "budget_stop_rate": mean([safe_float(item["budget_stop_rate"]) for item in items]),
                "budget_best_stop_rate": mean([safe_float(item["budget_best_stop_rate"]) for item in items]),
            }
        )
    return summary


def run_grouped_cv(by_pair_prefixes: dict[tuple[int, str], list[dict[str, Any]]], max_candidates: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    keys = sorted(by_pair_prefixes)
    fold_rows = []
    selection_rows = []
    weight_rows = []
    for fold_idx, (train_qids, test_qids) in enumerate(question_folds(keys)):
        train_groups = [prefixes for (_seed, qid), prefixes in by_pair_prefixes.items() if qid in train_qids]
        test_items = [((seed, qid), prefixes) for (seed, qid), prefixes in by_pair_prefixes.items() if qid in test_qids]
        train_sample0_rows = [sample0_at(prefixes[0]) for prefixes in train_groups]
        train_alt_rows = [row for prefixes in train_groups for prefix in prefixes for row in prefix["basins"] if safe_float(row["contains_sample0"]) <= 0]
        train_all_rows = [row for prefixes in train_groups for prefix in prefixes for row in prefix["basins"]]
        for feature_name, features in FEATURE_SETS.items():
            trust_model = TorchLogisticModel(features)
            trust_model.fit(train_sample0_rows, "representative_strict_correct", epochs=180, lr=0.1, l2=0.02)
            switch_model = TorchLogisticModel(features)
            switch_model.fit(train_alt_rows, "switch_gain_label", epochs=180, lr=0.1, l2=0.02)
            basin_model = TorchLogisticModel(features)
            basin_model.fit(train_all_rows, "representative_strict_correct", epochs=180, lr=0.1, l2=0.02)
            for profile, efficiency_penalty in [("quality", 0.0), ("balanced", 0.6), ("production", 1.8)]:
                params = tune_policy(train_groups, trust_model, switch_model, basin_model, max_candidates, efficiency_penalty)
                selections = [
                    simulate_pair(
                        prefixes,
                        trust_model,
                        switch_model,
                        params["trust_threshold"],
                        params["switch_threshold"],
                        params["min_candidates"],
                        max_candidates,
                        basin_model,
                        params["basin_margin"],
                        params["budget_best"],
                    )
                    for _key, prefixes in test_items
                ]
                method = f"adaptive_{profile}_{feature_name}"
                metrics = evaluate([prefixes for _key, prefixes in test_items], selections, method, f"fold_{fold_idx}")
                metrics.update({"feature_set": feature_name, **params})
                fold_rows.append(metrics)
                weight_rows.extend(trust_model.weight_rows(f"{method}_trust", f"fold_{fold_idx}", top_k=10))
                weight_rows.extend(switch_model.weight_rows(f"{method}_switch", f"fold_{fold_idx}", top_k=10))
                for (seed, qid), (row, generated, reason) in zip([key for key, _prefixes in test_items], selections):
                    s0 = sample0_at(by_pair_prefixes[(seed, qid)][0])
                    selection_rows.append(
                        {
                            "split": f"fold_{fold_idx}",
                            "method": method,
                            "seed": seed,
                            "question_id": qid,
                            "generated_candidates": generated,
                            "stop_reason": reason,
                            "sample0_correct": s0["representative_strict_correct"],
                            "selected_correct": row["representative_strict_correct"],
                            "delta_correct": safe_float(row["representative_strict_correct"]) - safe_float(s0["representative_strict_correct"]),
                            "sample0_cluster_id": s0["cluster_id"],
                            "selected_cluster_id": row["cluster_id"],
                            "selected_preview": row["representative_preview"],
                        }
                    )
    return summarize(fold_rows) + fold_rows, selection_rows, weight_rows


def make_plots(output_dir: Path, cv_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = ensure_dir(output_dir / "plots")
    summary = [row for row in cv_rows if row["split"] == "question_grouped_cv"]
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter([safe_float(row["avg_generated_candidates"]) for row in summary], [safe_float(row["strict_correct_rate"]) for row in summary], s=90)
    for row in summary:
        ax.text(safe_float(row["avg_generated_candidates"]) + 0.02, safe_float(row["strict_correct_rate"]), row["method"].replace("adaptive_", "").replace("_", "\n"), fontsize=7)
    ax.set_xlabel("Average generated candidates")
    ax.set_ylabel("Strict correctness")
    ax.set_title("Adaptive Basin Verifier: Quality vs Cost")
    fig.tight_layout()
    fig.savefig(plot_dir / "01_quality_vs_cost.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    methods = [row["method"].replace("adaptive_", "").replace("_", "\n") for row in summary]
    ax.bar(methods, [safe_float(row["trust_stop_rate"]) for row in summary], label="trust sample0")
    ax.bar(methods, [safe_float(row["switch_stop_rate"]) for row in summary], bottom=[safe_float(row["trust_stop_rate"]) for row in summary], label="switch")
    bottoms = [safe_float(row["trust_stop_rate"]) + safe_float(row["switch_stop_rate"]) for row in summary]
    ax.bar(methods, [safe_float(row["budget_stop_rate"]) for row in summary], bottom=bottoms, label="budget")
    bottoms = [bottom + safe_float(row["budget_stop_rate"]) for bottom, row in zip(bottoms, summary)]
    ax.bar(methods, [safe_float(row["budget_best_stop_rate"]) for row in summary], bottom=bottoms, label="budget best")
    ax.set_ylabel("Stop reason rate")
    ax.set_title("Adaptive Stop Reasons")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "02_stop_reasons.png")
    plt.close(fig)


def write_report(output_dir: Path, cv_rows: list[dict[str, Any]]) -> None:
    summary = [row for row in cv_rows if row["split"] == "question_grouped_cv"]
    lines = [
        "# Adaptive Semantic/Factual Basin Verifier",
        "",
        "## 核心问题",
        "",
        "这轮实验模拟更接近生产环境的策略：先只生成 sample0；如果 sample0 basin 看起来可信，就直接停止，不再强行生成其他 basin。只有 risk gate 认为 sample0 不够可信时，才继续逐步暴露后续候选并动态更新 answer basins。",
        "",
        "## Grouped CV Summary",
        "",
        "| Method | Strict | Delta | Improved | Damaged | Net | Avg Gen | Cost vs 1x | Save vs 8x | Changed | Trust Stop | Switch Stop | Budget Stop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{int(row['net_gain_count'])}` | "
            f"`{safe_float(row['avg_generated_candidates']):.2f}` | `{safe_float(row['token_cost_vs_sample0']):.2f}x` | `{safe_float(row['token_savings_vs_full8']):.2%}` | `{safe_float(row['answer_changed_rate']):.2%}` | "
            f"`{safe_float(row['trust_stop_rate']):.2%}` | `{safe_float(row['switch_stop_rate']):.2%}` | `{safe_float(row['budget_stop_rate']):.2%}` |"
        )
    best = max(summary, key=lambda row: (safe_float(row["delta_vs_sample0"]) - 0.01 * safe_float(row["avg_generated_candidates"]), -int(row["damaged_count"]))) if summary else None
    if best:
        lines.extend(
            [
                "",
                "## 初步解读",
                "",
                f"综合正确率提升和生成成本，当前最值得关注的是 `{best['method']}`：strict correctness `{safe_float(best['strict_correct_rate']):.2%}`，相对 sample0 `{safe_float(best['delta_vs_sample0']):.2%}`，平均生成 `{safe_float(best['avg_generated_candidates']):.2f}` 个候选。",
                "",
                "如果 production profile 能在较低平均生成数下维持正收益并控制 damage，说明 basin verifier 可以从固定 best-of-N 走向 adaptive inference。",
                "",
                "如果 semantic/factual 特征优于 numeric-only，说明对 `question + basin representative + consensus core` 的轻量一致性判断确实能帮助区分 stable correct 与 stable hallucination。",
            ]
        )
    (output_dir / "adaptive_semantic_basin_findings_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    candidate_rows = read_csv(Path(args.candidate_run) / "candidate_features.csv")
    by_pair_prefixes, sample0_rows, alt_rows = build_prefix_dataset(candidate_rows, args.max_candidates)
    cv_rows, selection_rows, weight_rows = run_grouped_cv(by_pair_prefixes, args.max_candidates)
    write_csv(output_dir / "adaptive_cv_results.csv", cv_rows)
    write_csv(output_dir / "adaptive_selection_rows.csv", selection_rows)
    write_csv(output_dir / "adaptive_model_weights.csv", weight_rows)
    write_json(
        output_dir / "run_metadata.json",
        {
            "pair_count": len(by_pair_prefixes),
            "sample0_training_rows": len(sample0_rows),
            "alternative_training_rows": len(alt_rows),
            "max_candidates": args.max_candidates,
            "feature_sets": FEATURE_SETS,
            "torch_cuda_available": bool(torch is not None and torch.cuda.is_available()),
        },
    )
    make_plots(output_dir, cv_rows)
    write_report(output_dir, cv_rows)
    print(json.dumps({"output_dir": str(output_dir), "pair_count": len(by_pair_prefixes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
