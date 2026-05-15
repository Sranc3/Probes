#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_basin_theory_deep_audit import LogisticModel, mean, question_folds, safe_float, write_csv, write_json  # noqa: E402


DEFAULT_FACTUAL_RUN = Path("/zhutingqi/song/Plan_gpt55/basin_theory_analysis/runs/run_20260430_094647_factual_riemann_strong_verifier_compact")
DEFAULT_CANDIDATE_RUN = Path("/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25")
DEFAULT_OUTPUT_ROOT = Path("/zhutingqi/song/Plan_gpt55/basin_theory_analysis/runs")

BANNED_AS_PREDICTOR = {
    "strict_correct",
    "sample0_strict_correct",
    "representative_strict_correct",
    "basin_correct_rate",
    "correct_count",
    "wrong_count",
    "is_pure_correct",
    "is_pure_wrong",
    "is_mixed_basin",
    "is_rescue_basin",
    "is_damage_basin",
    "is_stable_correct_basin",
    "is_stable_hallucination_basin",
    "is_unstable_wrong_basin",
    "basin_regime",
    "stable_hallucination_score",
}

FEATURE_SETS = {
    "theory_core": [
        "cluster_size",
        "cluster_weight_mass",
        "stable_score",
        "fragmentation_entropy",
        "top2_weight_margin",
        "distance_to_sample0_entropy_logprob",
        "wf_entropy_max_mean",
        "wf_entropy_roughness",
        "wf_prob_min_mean",
        "delta_cluster_weight_mass_vs_sample0",
        "delta_wf_prob_min_mean_vs_sample0",
        "riemann_anisotropy",
    ],
    "qwen_structured": [
        "qwen_structured_acceptability_score",
        "qwen_structured_factual_residual",
        "qwen_answer_responsiveness_score",
        "qwen_constraint_satisfaction_score",
        "qwen_entity_number_consistency_score",
        "qwen_basin_consensus_support_score",
        "qwen_overclaim_risk",
        "qwen_world_knowledge_conflict_risk",
        "delta_qwen_structured_acceptability_score_vs_sample0",
        "delta_qwen_world_knowledge_conflict_risk_vs_sample0",
    ],
    "theory_plus_structured": [
        "cluster_size",
        "cluster_weight_mass",
        "stable_score",
        "fragmentation_entropy",
        "top2_weight_margin",
        "distance_to_sample0_entropy_logprob",
        "wf_entropy_max_mean",
        "wf_entropy_roughness",
        "wf_prob_min_mean",
        "delta_cluster_weight_mass_vs_sample0",
        "delta_wf_prob_min_mean_vs_sample0",
        "riemann_anisotropy",
        "qwen_structured_acceptability_score",
        "qwen_structured_factual_residual",
        "qwen_answer_responsiveness_score",
        "qwen_constraint_satisfaction_score",
        "qwen_entity_number_consistency_score",
        "qwen_basin_consensus_support_score",
        "qwen_overclaim_risk",
        "qwen_world_knowledge_conflict_risk",
        "delta_qwen_structured_acceptability_score_vs_sample0",
        "delta_qwen_world_knowledge_conflict_risk_vs_sample0",
    ],
    "veto_bonus_compact": [
        "delta_cluster_weight_mass_vs_sample0",
        "delta_wf_prob_min_mean_vs_sample0",
        "delta_qwen_structured_acceptability_score_vs_sample0",
        "delta_qwen_world_knowledge_conflict_risk_vs_sample0",
        "qwen_structured_acceptability_score",
        "qwen_world_knowledge_conflict_risk",
        "qwen_answer_responsiveness_score",
        "qwen_constraint_satisfaction_score",
        "riemann_anisotropy",
    ],
}

GATE_FEATURES = ["logprob_avg", "token_mean_entropy", "token_max_entropy", "token_count"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed-8 structured factual basin controllers.")
    parser.add_argument("--factual-run", type=Path, default=DEFAULT_FACTUAL_RUN)
    parser.add_argument("--candidate-run", type=Path, default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="structured_factual_controller_eval")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def group_by_pair(rows: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["question_id"]))].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: int(float(row["cluster_id"])))
    return grouped


def sample0_basin(group: list[dict[str, Any]]) -> dict[str, Any]:
    return next(row for row in group if safe_float(row.get("contains_sample0")) > 0)


def candidate_costs(candidate_rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, float]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[(int(row["seed"]), str(row["question_id"]))].append(row)
    costs = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(float(row["sample_index"])))
        sample0_tokens = safe_float(rows[0].get("token_count")) if rows else 0.0
        full_tokens = sum(safe_float(row.get("token_count")) for row in rows)
        costs[key] = {
            "candidate_count": float(len(rows)),
            "sample0_tokens": sample0_tokens,
            "full_tokens": full_tokens,
            "token_cost_vs_sample0": full_tokens / max(1.0, sample0_tokens),
        }
    return costs


def sample0_candidate_rows(candidate_rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    rows = {}
    for row in candidate_rows:
        if int(float(row.get("sample_index", 0))) == 0:
            rows[(int(row["seed"]), str(row["question_id"]))] = row
    return rows


def row_label(row: dict[str, Any]) -> float:
    return 1.0 if safe_float(row.get("representative_strict_correct")) > 0 else 0.0


def evaluate_selections(
    groups: list[tuple[tuple[int, str], list[dict[str, Any]]]],
    selections: list[dict[str, Any]],
    method: str,
    split: str,
    cost_by_pair: dict[tuple[int, str], dict[str, float]],
    uses_structured_verifier: bool,
    full_generation: bool,
) -> dict[str, Any]:
    sample0s = [sample0_basin(group) for _key, group in groups]
    deltas = [row_label(sel) - row_label(s0) for sel, s0 in zip(selections, sample0s)]
    basin_counts = [len(group) for _key, group in groups]
    candidate_counts = [cost_by_pair.get(key, {}).get("candidate_count", 8.0) if full_generation else 1.0 for key, _group in groups]
    token_costs = [cost_by_pair.get(key, {}).get("token_cost_vs_sample0", 8.0) if full_generation else 1.0 for key, _group in groups]
    verifier_calls = [7.0 * basin_count if uses_structured_verifier else 0.0 for basin_count in basin_counts]
    selected_changed = [float(int(sel["cluster_id"]) != int(s0["cluster_id"])) for sel, s0 in zip(selections, sample0s)]
    return {
        "split": split,
        "method": method,
        "pairs": len(groups),
        "strict_correct_rate": mean([row_label(row) for row in selections]),
        "sample0_strict_correct_rate": mean([row_label(row) for row in sample0s]),
        "delta_vs_sample0": mean(deltas),
        "improved_count": sum(1 for delta in deltas if delta > 0),
        "damaged_count": sum(1 for delta in deltas if delta < 0),
        "net_gain_count": sum(1 for delta in deltas if delta > 0) - sum(1 for delta in deltas if delta < 0),
        "answer_changed_rate": mean(selected_changed),
        "avg_generated_candidates": mean(candidate_counts),
        "candidate_cost_vs_sample0": mean(candidate_counts),
        "token_cost_vs_sample0": mean(token_costs),
        "avg_basin_count": mean([float(item) for item in basin_counts]),
        "avg_structured_verifier_prompt_calls": mean(verifier_calls),
        "uses_structured_verifier": float(uses_structured_verifier),
        "full_generation": float(full_generation),
    }


def evaluate_variable_cost_selections(
    groups: list[tuple[tuple[int, str], list[dict[str, Any]]]],
    selections: list[dict[str, Any]],
    escalated: list[bool],
    method: str,
    split: str,
    cost_by_pair: dict[tuple[int, str], dict[str, float]],
) -> dict[str, Any]:
    sample0s = [sample0_basin(group) for _key, group in groups]
    deltas = [row_label(sel) - row_label(s0) for sel, s0 in zip(selections, sample0s)]
    basin_counts = [len(group) for _key, group in groups]
    candidate_counts = [cost_by_pair.get(key, {}).get("candidate_count", 8.0) if is_full else 1.0 for (key, _group), is_full in zip(groups, escalated)]
    token_costs = [cost_by_pair.get(key, {}).get("token_cost_vs_sample0", 8.0) if is_full else 1.0 for (key, _group), is_full in zip(groups, escalated)]
    verifier_calls = [7.0 * basin_count if is_full else 0.0 for basin_count, is_full in zip(basin_counts, escalated)]
    selected_changed = [float(int(sel["cluster_id"]) != int(s0["cluster_id"])) for sel, s0 in zip(selections, sample0s)]
    return {
        "split": split,
        "method": method,
        "pairs": len(groups),
        "strict_correct_rate": mean([row_label(row) for row in selections]),
        "sample0_strict_correct_rate": mean([row_label(row) for row in sample0s]),
        "delta_vs_sample0": mean(deltas),
        "improved_count": sum(1 for delta in deltas if delta > 0),
        "damaged_count": sum(1 for delta in deltas if delta < 0),
        "net_gain_count": sum(1 for delta in deltas if delta > 0) - sum(1 for delta in deltas if delta < 0),
        "answer_changed_rate": mean(selected_changed),
        "avg_generated_candidates": mean(candidate_counts),
        "candidate_cost_vs_sample0": mean(candidate_counts),
        "token_cost_vs_sample0": mean(token_costs),
        "avg_basin_count": mean([float(item) for item in basin_counts]),
        "avg_structured_verifier_prompt_calls": mean(verifier_calls),
        "uses_structured_verifier": 1.0,
        "full_generation": mean([float(item) for item in escalated]),
        "escalation_rate": mean([float(item) for item in escalated]),
    }


def select_by_model(group: list[dict[str, Any]], model: LogisticModel, margin: float) -> dict[str, Any]:
    sample0 = sample0_basin(group)
    best = max(group, key=model.predict)
    if int(best["cluster_id"]) == int(sample0["cluster_id"]):
        return sample0
    if model.predict(best) - model.predict(sample0) >= margin:
        return best
    return sample0


def tune_margin(
    train_groups: list[tuple[tuple[int, str], list[dict[str, Any]]]],
    model: LogisticModel,
    cost_by_pair: dict[tuple[int, str], dict[str, float]],
    uses_structured_verifier: bool,
) -> float:
    best_score = -1e9
    best_margin = 0.0
    for margin in [-0.15, -0.05, 0.0, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50]:
        selections = [select_by_model(group, model, margin) for _key, group in train_groups]
        metrics = evaluate_selections(train_groups, selections, "train", "train", cost_by_pair, uses_structured_verifier, True)
        score = (
            100.0 * safe_float(metrics["delta_vs_sample0"])
            + safe_float(metrics["net_gain_count"])
            - 4.0 * safe_float(metrics["damaged_count"])
            - 0.4 * safe_float(metrics["answer_changed_rate"])
        )
        if score > best_score:
            best_score = score
            best_margin = margin
    return best_margin


def select_cluster_weight_structured_veto(group: list[dict[str, Any]], params: dict[str, float]) -> dict[str, Any]:
    sample0 = sample0_basin(group)
    best = max(group, key=lambda row: safe_float(row.get("cluster_weight_mass")))
    if int(best["cluster_id"]) == int(sample0["cluster_id"]):
        return sample0
    if safe_float(best.get("delta_qwen_structured_acceptability_score_vs_sample0")) < params["min_delta_accept"]:
        return sample0
    if safe_float(best.get("delta_qwen_world_knowledge_conflict_risk_vs_sample0")) > params["max_delta_conflict"]:
        return sample0
    if safe_float(best.get("qwen_answer_responsiveness_score")) < params["min_responsive"]:
        return sample0
    if safe_float(best.get("qwen_constraint_satisfaction_score")) < params["min_constraint"]:
        return sample0
    return best


def tune_structured_veto(
    train_groups: list[tuple[tuple[int, str], list[dict[str, Any]]]],
    cost_by_pair: dict[tuple[int, str], dict[str, float]],
) -> dict[str, float]:
    best_score = -1e9
    best_params = {"min_delta_accept": -1.0, "max_delta_conflict": 1.0, "min_responsive": 0.0, "min_constraint": 0.0}
    for min_delta_accept in [-0.50, -0.25, -0.10, 0.0, 0.10, 0.20]:
        for max_delta_conflict in [-0.20, 0.0, 0.15, 0.30, 0.60, 1.0]:
            for min_responsive in [0.0, 0.50, 0.70, 0.85]:
                for min_constraint in [0.0, 0.50, 0.70, 0.85]:
                    params = {
                        "min_delta_accept": min_delta_accept,
                        "max_delta_conflict": max_delta_conflict,
                        "min_responsive": min_responsive,
                        "min_constraint": min_constraint,
                    }
                    selections = [select_cluster_weight_structured_veto(group, params) for _key, group in train_groups]
                    metrics = evaluate_selections(train_groups, selections, "train", "train", cost_by_pair, True, True)
                    score = (
                        100.0 * safe_float(metrics["delta_vs_sample0"])
                        + safe_float(metrics["net_gain_count"])
                        - 4.0 * safe_float(metrics["damaged_count"])
                        - 0.3 * safe_float(metrics["answer_changed_rate"])
                    )
                    if score > best_score:
                        best_score = score
                        best_params = params
    return best_params


def two_stage_select(
    key: tuple[int, str],
    group: list[dict[str, Any]],
    gate_model: LogisticModel,
    sample0_candidates: dict[tuple[int, str], dict[str, Any]],
    trust_threshold: float,
    veto_params: dict[str, float],
) -> tuple[dict[str, Any], bool]:
    s0 = sample0_basin(group)
    sample0_candidate = sample0_candidates.get(key)
    if sample0_candidate is None:
        return s0, False
    if gate_model.predict(sample0_candidate) >= trust_threshold:
        return s0, False
    return select_cluster_weight_structured_veto(group, veto_params), True


def tune_two_stage_threshold(
    train_groups: list[tuple[tuple[int, str], list[dict[str, Any]]]],
    gate_model: LogisticModel,
    sample0_candidates: dict[tuple[int, str], dict[str, Any]],
    veto_params: dict[str, float],
    cost_by_pair: dict[tuple[int, str], dict[str, float]],
    cost_penalty: float,
) -> float:
    best_score = -1e9
    best_threshold = 0.5
    for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        selected_pairs = [two_stage_select(key, group, gate_model, sample0_candidates, threshold, veto_params) for key, group in train_groups]
        selections = [item[0] for item in selected_pairs]
        escalated = [item[1] for item in selected_pairs]
        metrics = evaluate_variable_cost_selections(train_groups, selections, escalated, "train", "train", cost_by_pair)
        score = (
            100.0 * safe_float(metrics["delta_vs_sample0"])
            + safe_float(metrics["net_gain_count"])
            - 4.0 * safe_float(metrics["damaged_count"])
            - cost_penalty * safe_float(metrics["avg_generated_candidates"])
        )
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def question_fold_keys(keys: list[tuple[int, str]], k: int = 5) -> list[tuple[set[str], set[str]]]:
    qids = sorted({qid for _seed, qid in keys})
    folds = []
    for idx in range(k):
        test = {qid for pos, qid in enumerate(qids) if pos % k == idx}
        folds.append((set(qids) - test, test))
    return folds


def run_cv(
    groups_by_pair: dict[tuple[int, str], list[dict[str, Any]]],
    cost_by_pair: dict[tuple[int, str], dict[str, float]],
    sample0_candidates: dict[tuple[int, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    keys = sorted(groups_by_pair)
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for fold_idx, (train_qids, test_qids) in enumerate(question_fold_keys(keys)):
        train_groups = [(key, groups_by_pair[key]) for key in keys if key[1] in train_qids]
        test_groups = [(key, groups_by_pair[key]) for key in keys if key[1] in test_qids]
        train_rows = [row for _key, group in train_groups for row in group]
        for method, features in FEATURE_SETS.items():
            model = LogisticModel(features)
            model.fit(train_rows, "representative_strict_correct")
            uses_structured = any(feature.startswith("qwen_") or "qwen_" in feature for feature in features)
            margin = tune_margin(train_groups, model, cost_by_pair, uses_structured)
            selections = [select_by_model(group, model, margin) for _key, group in test_groups]
            metrics = evaluate_selections(test_groups, selections, method, f"fold_{fold_idx}", cost_by_pair, uses_structured, True)
            metrics["margin"] = margin
            fold_rows.append(metrics)
            for feature, weight in sorted(model.weights.items(), key=lambda item: abs(item[1]), reverse=True)[:12]:
                weight_rows.append({"split": f"fold_{fold_idx}", "method": method, "feature": feature, "weight": weight})
            for (key, group), selected in zip(test_groups, selections):
                s0 = sample0_basin(group)
                selection_rows.append(
                    {
                        "split": f"fold_{fold_idx}",
                        "method": method,
                        "seed": key[0],
                        "question_id": key[1],
                        "sample0_cluster_id": s0["cluster_id"],
                        "selected_cluster_id": selected["cluster_id"],
                        "sample0_correct": row_label(s0),
                        "selected_correct": row_label(selected),
                        "delta_correct": row_label(selected) - row_label(s0),
                        "selected_preview": selected.get("answer_preview", ""),
                    }
                )
        params = tune_structured_veto(train_groups, cost_by_pair)
        selections = [select_cluster_weight_structured_veto(group, params) for _key, group in test_groups]
        metrics = evaluate_selections(test_groups, selections, "cluster_weight_structured_veto", f"fold_{fold_idx}", cost_by_pair, True, True)
        metrics.update(params)
        fold_rows.append(metrics)
        for (key, group), selected in zip(test_groups, selections):
            s0 = sample0_basin(group)
            selection_rows.append(
                {
                    "split": f"fold_{fold_idx}",
                    "method": "cluster_weight_structured_veto",
                    "seed": key[0],
                    "question_id": key[1],
                    "sample0_cluster_id": s0["cluster_id"],
                    "selected_cluster_id": selected["cluster_id"],
                    "sample0_correct": row_label(s0),
                    "selected_correct": row_label(selected),
                    "delta_correct": row_label(selected) - row_label(s0),
                    "selected_preview": selected.get("answer_preview", ""),
                }
            )
        train_sample0_candidates = [sample0_candidates[key] for key, _group in train_groups if key in sample0_candidates]
        gate_model = LogisticModel(GATE_FEATURES)
        gate_model.fit(train_sample0_candidates, "strict_correct")
        for profile, cost_penalty in [("balanced", 0.35), ("production", 1.25)]:
            threshold = tune_two_stage_threshold(train_groups, gate_model, sample0_candidates, params, cost_by_pair, cost_penalty)
            selected_pairs = [two_stage_select(key, group, gate_model, sample0_candidates, threshold, params) for key, group in test_groups]
            selections = [item[0] for item in selected_pairs]
            escalated = [item[1] for item in selected_pairs]
            method = f"two_stage_structured_{profile}"
            metrics = evaluate_variable_cost_selections(test_groups, selections, escalated, method, f"fold_{fold_idx}", cost_by_pair)
            metrics["trust_threshold"] = threshold
            metrics.update(params)
            fold_rows.append(metrics)
            for (key, group), selected, is_escalated in zip(test_groups, selections, escalated):
                s0 = sample0_basin(group)
                selection_rows.append(
                    {
                        "split": f"fold_{fold_idx}",
                        "method": method,
                        "seed": key[0],
                        "question_id": key[1],
                        "escalated_to_full8": float(is_escalated),
                        "sample0_cluster_id": s0["cluster_id"],
                        "selected_cluster_id": selected["cluster_id"],
                        "sample0_correct": row_label(s0),
                        "selected_correct": row_label(selected),
                        "delta_correct": row_label(selected) - row_label(s0),
                        "selected_preview": selected.get("answer_preview", ""),
                    }
                )
    return summarize(fold_rows), fold_rows, selection_rows + weight_rows


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
                "pairs": sum(int(row["pairs"]) for row in items),
                "strict_correct_rate": mean([safe_float(row["strict_correct_rate"]) for row in items]),
                "sample0_strict_correct_rate": mean([safe_float(row["sample0_strict_correct_rate"]) for row in items]),
                "delta_vs_sample0": mean([safe_float(row["delta_vs_sample0"]) for row in items]),
                "improved_count": sum(int(row["improved_count"]) for row in items),
                "damaged_count": sum(int(row["damaged_count"]) for row in items),
                "net_gain_count": sum(int(row["net_gain_count"]) for row in items),
                "answer_changed_rate": mean([safe_float(row["answer_changed_rate"]) for row in items]),
                "avg_generated_candidates": mean([safe_float(row["avg_generated_candidates"]) for row in items]),
                "candidate_cost_vs_sample0": mean([safe_float(row["candidate_cost_vs_sample0"]) for row in items]),
                "token_cost_vs_sample0": mean([safe_float(row["token_cost_vs_sample0"]) for row in items]),
                "avg_basin_count": mean([safe_float(row["avg_basin_count"]) for row in items]),
                "avg_structured_verifier_prompt_calls": mean([safe_float(row["avg_structured_verifier_prompt_calls"]) for row in items]),
                "uses_structured_verifier": mean([safe_float(row["uses_structured_verifier"]) for row in items]),
                "full_generation": mean([safe_float(row["full_generation"]) for row in items]),
                "escalation_rate": mean([safe_float(row.get("escalation_rate", row.get("full_generation"))) for row in items]),
                "mean_margin": mean([safe_float(row.get("margin")) for row in items]),
            }
        )
    return summary


def baseline_rows(groups_by_pair: dict[tuple[int, str], list[dict[str, Any]]], cost_by_pair: dict[tuple[int, str], dict[str, float]]) -> list[dict[str, Any]]:
    groups = sorted(groups_by_pair.items())
    sample0_selected = [sample0_basin(group) for _key, group in groups]
    rows = [evaluate_selections(groups, sample0_selected, "sample0_baseline", "question_grouped_cv", cost_by_pair, False, False)]
    for method, chooser in [
        ("fixed8_cluster_weight", lambda group: max(group, key=lambda row: safe_float(row.get("cluster_weight_mass")))),
        ("fixed8_low_entropy_weight", lambda group: max(group, key=lambda row: safe_float(row.get("stable_score")))),
        ("fixed8_structured_acceptability", lambda group: max(group, key=lambda row: safe_float(row.get("qwen_structured_acceptability_score")))),
    ]:
        selected = []
        for _key, group in groups:
            s0 = sample0_basin(group)
            best = chooser(group)
            selected.append(best if int(best["cluster_id"]) != int(s0["cluster_id"]) else s0)
        rows.append(evaluate_selections(groups, selected, method, "question_grouped_cv", cost_by_pair, "structured" in method, True))
    return rows


def no_leak_audit(features: list[str], available: set[str]) -> list[dict[str, Any]]:
    rows = []
    for feature in sorted(set(features) | available):
        used = feature in features
        status = "ok"
        if used and feature in BANNED_AS_PREDICTOR:
            status = "ERROR_used_banned"
        if used and feature not in available:
            status = "ERROR_missing"
        rows.append({"feature": feature, "used_as_predictor": float(used), "is_banned": float(feature in BANNED_AS_PREDICTOR), "status": status})
    errors = [row for row in rows if row["status"] != "ok"]
    if errors:
        raise RuntimeError(f"No-leak audit failed: {errors[:8]}")
    return rows


def make_plots(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(summary, key=lambda row: safe_float(row["candidate_cost_vs_sample0"]))
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.scatter([safe_float(row["candidate_cost_vs_sample0"]) for row in rows], [safe_float(row["strict_correct_rate"]) for row in rows], s=90)
    for row in rows:
        ax.text(safe_float(row["candidate_cost_vs_sample0"]) + 0.03, safe_float(row["strict_correct_rate"]), row["method"].replace("_", "\n"), fontsize=7)
    ax.set_xlabel("Candidate generation cost vs sample0")
    ax.set_ylabel("Strict correctness")
    ax.set_title("Structured Factual Controller: Accuracy vs Candidate Cost")
    fig.tight_layout()
    fig.savefig(plot_dir / "01_accuracy_vs_candidate_cost.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    labels = [row["method"].replace("_", "\n") for row in rows]
    ax.bar(labels, [100 * safe_float(row["delta_vs_sample0"]) for row in rows], color="#069DFF", alpha=0.82)
    ax.axhline(0, color="#808080", lw=0.9)
    ax.set_ylabel("Delta vs sample0 (percentage points)")
    ax.set_title("Controller Net Accuracy Change")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plot_dir / "02_delta_vs_sample0.png")
    plt.close(fig)


def write_report(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Structured Factual Controller Evaluation",
        "",
        "## Summary",
        "",
        "| Method | Strict | Delta | Improved | Damaged | Net | Changed | Escalate | Cand Cost | Token Cost | Basin Count | Verifier Calls |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary, key=lambda item: (-safe_float(item["strict_correct_rate"]), safe_float(item["avg_structured_verifier_prompt_calls"]))):
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{int(row['net_gain_count'])}` | `{safe_float(row['answer_changed_rate']):.2%}` | "
            f"`{safe_float(row.get('escalation_rate', row.get('full_generation'))):.2%}` | "
            f"`{safe_float(row['candidate_cost_vs_sample0']):.2f}x` | `{safe_float(row['token_cost_vs_sample0']):.2f}x` | `{safe_float(row['avg_basin_count']):.2f}` | `{safe_float(row['avg_structured_verifier_prompt_calls']):.2f}` |"
        )
    best = max(summary, key=lambda row: (safe_float(row["delta_vs_sample0"]), -safe_float(row["avg_structured_verifier_prompt_calls"])))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"Best accuracy method: `{best['method']}` with strict `{safe_float(best['strict_correct_rate']):.2%}` and delta `{safe_float(best['delta_vs_sample0']):.2%}`.",
            "",
            "Fixed-8 methods pay full-8 candidate cost for every question. Two-stage methods first run a cheap sample0 gate and only pay full-8 plus structured verifier calls for escalated questions.",
            "",
            "If structured features improve accuracy at fixed-8 but do not beat adaptive-v1's low-cost frontier, the next step should be a two-stage controller: cheap theory-core gate first, structured verifier only for ambiguous high-value switches.",
        ]
    )
    (output_dir / "structured_factual_controller_findings_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    basin_rows = read_csv(args.factual_run / "factual_riemann_table.csv")
    candidate_rows = read_csv(args.candidate_run / "candidate_features.csv")
    groups_by_pair = group_by_pair(basin_rows)
    cost_by_pair = candidate_costs(candidate_rows)
    sample0_candidates = sample0_candidate_rows(candidate_rows)
    used_features = sorted({feature for features in FEATURE_SETS.values() for feature in features} | set(GATE_FEATURES))
    audit_rows = no_leak_audit(used_features, {key for row in basin_rows for key in row} | {key for row in sample0_candidates.values() for key in row})
    summary_rows, fold_rows, detail_rows = run_cv(groups_by_pair, cost_by_pair, sample0_candidates)
    all_summary = baseline_rows(groups_by_pair, cost_by_pair) + summary_rows
    write_csv(output_dir / "structured_controller_summary.csv", all_summary)
    write_csv(output_dir / "structured_controller_fold_results.csv", fold_rows)
    write_csv(output_dir / "structured_controller_details.csv", detail_rows)
    write_csv(output_dir / "no_leak_audit_structured_controller.csv", audit_rows)
    write_json(output_dir / "run_metadata.json", {"factual_run": str(args.factual_run), "candidate_run": str(args.candidate_run), "feature_sets": FEATURE_SETS, "pair_count": len(groups_by_pair)})
    make_plots(output_dir, all_summary)
    write_report(output_dir, all_summary)
    print(json.dumps({"output_dir": str(output_dir), "pair_count": len(groups_by_pair)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
