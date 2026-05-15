#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from run_adaptive_semantic_basin_verifier import (
    FEATURE_SETS,
    TorchLogisticModel,
    build_prefix_dataset,
    choose_prefix_basin,
    ensure_dir,
    mean,
    read_csv,
    safe_float,
    sample0_at,
    write_csv,
    write_json,
)
from run_learned_basin_verifier import question_folds, stdev
from run_prefix_aware_student_policy import STATE_FEATURES, state_row

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


DEFAULT_CANDIDATE_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"


PROFILE_LAMBDAS = {
    "quality": 0.04,
    "balanced": 0.08,
    "production": 0.14,
}


class TorchLinearRegressor:
    def __init__(self, features: list[str]):
        self.features = features
        self.means = {feature: 0.0 for feature in features}
        self.stds = {feature: 1.0 for feature in features}
        self.weights = {feature: 0.0 for feature in features}
        self.bias = 0.0

    def fit(self, rows: list[dict[str, Any]], label: str, epochs: int = 260, lr: float = 0.05, l2: float = 0.02) -> None:
        if not rows:
            return
        if torch is None or not torch.cuda.is_available():
            self._fit_cpu(rows, label, epochs, lr, l2)
            return
        device = torch.device("cuda:0")
        values_by_feature = []
        for feature in self.features:
            values = [safe_float(row.get(feature, 0.0)) for row in rows]
            self.means[feature] = mean(values)
            self.stds[feature] = stdev(values) or 1.0
            values_by_feature.append([(value - self.means[feature]) / self.stds[feature] for value in values])
        x = torch.tensor(list(zip(*values_by_feature)), dtype=torch.float32, device=device)
        y = torch.tensor([safe_float(row[label]) for row in rows], dtype=torch.float32, device=device)
        w = torch.zeros(x.shape[1], dtype=torch.float32, device=device, requires_grad=True)
        b = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([w, b], lr=lr, weight_decay=l2)
        for _ in range(epochs):
            pred = x @ w + b
            loss = torch.mean((pred - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.bias = float(b.detach().cpu().item())
        for feature, weight in zip(self.features, w.detach().cpu().tolist()):
            self.weights[feature] = float(weight)

    def _fit_cpu(self, rows: list[dict[str, Any]], label: str, epochs: int, lr: float, l2: float) -> None:
        for feature in self.features:
            values = [safe_float(row.get(feature, 0.0)) for row in rows]
            self.means[feature] = mean(values)
            self.stds[feature] = stdev(values) or 1.0
        for _ in range(epochs):
            grad = {feature: 0.0 for feature in self.features}
            grad_b = 0.0
            for row in rows:
                err = self.predict(row) - safe_float(row[label])
                for feature in self.features:
                    grad[feature] += err * self.z(row, feature)
                grad_b += err
            scale = 1.0 / max(1, len(rows))
            for feature in self.features:
                self.weights[feature] -= lr * (grad[feature] * scale + l2 * self.weights[feature])
            self.bias -= lr * grad_b * scale

    def z(self, row: dict[str, Any], feature: str) -> float:
        return (safe_float(row.get(feature, 0.0)) - self.means[feature]) / self.stds[feature]

    def predict(self, row: dict[str, Any]) -> float:
        return self.bias + sum(self.weights[feature] * self.z(row, feature) for feature in self.features)

    def weight_rows(self, model_name: str, split: str, top_k: int = 14) -> list[dict[str, Any]]:
        rows = [{"split": split, "model": model_name, "feature": "bias", "weight": self.bias}]
        for feature in sorted(self.features, key=lambda item: abs(self.weights[item]), reverse=True)[:top_k]:
            rows.append({"split": split, "model": model_name, "feature": feature, "weight": self.weights[feature]})
        return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run marginal continue-value prefix policy.")
    parser.add_argument("--candidate-run", default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="prefix_value_policy")
    parser.add_argument("--max-candidates", type=int, default=8)
    return parser.parse_args()


def token_cost(prefixes: list[dict[str, Any]], prefix_idx: int) -> float:
    return sum(safe_float(sample["token_count"]) for sample in prefixes[prefix_idx]["samples"])


def oracle_correct(prefix: dict[str, Any]) -> float:
    return float(any(safe_float(row["representative_strict_correct"]) > 0 for row in prefix["basins"]))


def build_value_state_rows(by_pair_prefixes: dict[tuple[int, str], list[dict[str, Any]]], max_candidates: int) -> dict[tuple[int, str], list[dict[str, Any]]]:
    by_pair_states: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for key, prefixes in by_pair_prefixes.items():
        base_token = max(1.0, token_cost(prefixes, 0))
        states = []
        for idx, prefix in enumerate(prefixes):
            row = state_row(prefixes, idx, max_candidates)
            current_utility = oracle_correct(prefix) - 0.0
            for profile, lam in PROFILE_LAMBDAS.items():
                best_future = current_utility
                for future_idx in range(idx + 1, len(prefixes)):
                    extra_cost = (token_cost(prefixes, future_idx) - token_cost(prefixes, idx)) / base_token
                    future_utility = oracle_correct(prefixes[future_idx]) - lam * extra_cost
                    best_future = max(best_future, future_utility)
                row[f"continue_value_{profile}"] = best_future - current_utility
            states.append(row)
        by_pair_states[key] = states
    return by_pair_states


def generated_token_cost(prefixes: list[dict[str, Any]], generated: int) -> float:
    return sum(safe_float(sample["token_count"]) for sample in prefixes[min(generated, len(prefixes)) - 1]["samples"])


def simulate_pair(
    prefixes: list[dict[str, Any]],
    states: list[dict[str, Any]],
    trust_model: TorchLogisticModel,
    switch_model: TorchLogisticModel,
    basin_model: TorchLogisticModel,
    value_model: TorchLinearRegressor,
    trust_threshold: float,
    switch_threshold: float,
    value_threshold: float,
    basin_margin: float,
    min_candidates: int,
    max_candidates: int,
) -> tuple[dict[str, Any], int, str]:
    for idx, prefix in enumerate(prefixes[:max_candidates]):
        sample0 = sample0_at(prefix)
        choice = choose_prefix_basin(prefix, switch_model, switch_threshold, basin_model, basin_margin)
        if int(prefix["prefix_size"]) >= min_candidates and int(choice["cluster_id"]) != int(sample0["cluster_id"]):
            return choice, int(prefix["prefix_size"]), "switch"
        if trust_model.predict(sample0) >= trust_threshold:
            return sample0, int(prefix["prefix_size"]), "trust_sample0"
        if int(prefix["prefix_size"]) < min_candidates:
            continue
        if int(prefix["prefix_size"]) >= max_candidates:
            break
        if value_model.predict(states[idx]) > value_threshold:
            continue
        return choice, int(prefix["prefix_size"]), "stop_low_value"
    final_prefix = prefixes[min(max_candidates, len(prefixes)) - 1]
    return choose_prefix_basin(final_prefix, switch_model, switch_threshold, basin_model, basin_margin), int(final_prefix["prefix_size"]), "budget"


def evaluate(items: list[tuple[tuple[int, str], list[dict[str, Any]], list[dict[str, Any]]]], selections: list[tuple[dict[str, Any], int, str]], method: str, split: str) -> dict[str, Any]:
    sample0s = [sample0_at(prefixes[0]) for _key, prefixes, _states in items]
    selected = [row for row, _generated, _reason in selections]
    generated = [generated for _row, generated, _reason in selections]
    reasons = Counter(reason for _row, _generated, reason in selections)
    generated_tokens = [generated_token_cost(prefixes, gen) for (_key, prefixes, _states), gen in zip(items, generated)]
    sample0_tokens = [safe_float(prefixes[0]["samples"][0]["token_count"]) for _key, prefixes, _states in items]
    full_tokens = [generated_token_cost(prefixes, len(prefixes)) for _key, prefixes, _states in items]
    deltas = [safe_float(row["representative_strict_correct"]) - safe_float(s0["representative_strict_correct"]) for row, s0 in zip(selected, sample0s)]
    return {
        "split": split,
        "method": method,
        "pairs": len(items),
        "strict_correct_rate": mean([safe_float(row["representative_strict_correct"]) for row in selected]),
        "sample0_strict_correct_rate": mean([safe_float(row["representative_strict_correct"]) for row in sample0s]),
        "delta_vs_sample0": mean(deltas),
        "improved_count": sum(1 for value in deltas if value > 0),
        "damaged_count": sum(1 for value in deltas if value < 0),
        "net_gain_count": sum(1 for value in deltas if value > 0) - sum(1 for value in deltas if value < 0),
        "answer_changed_rate": mean([float(int(row["cluster_id"]) != int(s0["cluster_id"])) for row, s0 in zip(selected, sample0s)]),
        "avg_generated_candidates": mean(generated),
        "token_cost_vs_sample0": mean([gen / max(1.0, base) for gen, base in zip(generated_tokens, sample0_tokens)]),
        "token_savings_vs_full8": 1.0 - mean([gen / max(1.0, full) for gen, full in zip(generated_tokens, full_tokens)]),
        "trust_stop_rate": reasons["trust_sample0"] / max(1, len(items)),
        "switch_stop_rate": reasons["switch"] / max(1, len(items)),
        "value_stop_rate": reasons["stop_low_value"] / max(1, len(items)),
        "budget_stop_rate": reasons["budget"] / max(1, len(items)),
    }


def tune_policy(
    train_items: list[tuple[tuple[int, str], list[dict[str, Any]], list[dict[str, Any]]]],
    trust_model: TorchLogisticModel,
    switch_model: TorchLogisticModel,
    basin_model: TorchLogisticModel,
    value_model: TorchLinearRegressor,
    profile: str,
    max_candidates: int,
) -> dict[str, Any]:
    cost_penalty = {"quality": 0.2, "balanced": 0.8, "production": 1.7}[profile]
    damage_penalty = {"quality": 2.5, "balanced": 4.0, "production": 6.0}[profile]
    best: tuple[float, dict[str, Any]] = (-1e9, {})
    for trust_threshold in [0.5, 0.65, 0.8]:
        for switch_threshold in [0.45, 0.6, 0.75]:
            for value_threshold in [-0.04, 0.0, 0.04, 0.08]:
                for margin in [-0.2, 0.0, 0.2]:
                    for min_candidates in [2, 3, 4]:
                        selections = [
                            simulate_pair(prefixes, states, trust_model, switch_model, basin_model, value_model, trust_threshold, switch_threshold, value_threshold, margin, min_candidates, max_candidates)
                            for _key, prefixes, states in train_items
                        ]
                        metrics = evaluate(train_items, selections, "train", "train")
                        score = (
                            100 * safe_float(metrics["delta_vs_sample0"])
                            + int(metrics["net_gain_count"])
                            - damage_penalty * int(metrics["damaged_count"])
                            - cost_penalty * safe_float(metrics["avg_generated_candidates"])
                        )
                        if score > best[0]:
                            best = (
                                score,
                                {
                                    "trust_threshold": trust_threshold,
                                    "switch_threshold": switch_threshold,
                                    "value_threshold": value_threshold,
                                    "basin_margin": margin,
                                    "min_candidates": min_candidates,
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
                "token_cost_vs_sample0": mean([safe_float(item["token_cost_vs_sample0"]) for item in items]),
                "token_savings_vs_full8": mean([safe_float(item["token_savings_vs_full8"]) for item in items]),
                "trust_stop_rate": mean([safe_float(item["trust_stop_rate"]) for item in items]),
                "switch_stop_rate": mean([safe_float(item["switch_stop_rate"]) for item in items]),
                "value_stop_rate": mean([safe_float(item["value_stop_rate"]) for item in items]),
                "budget_stop_rate": mean([safe_float(item["budget_stop_rate"]) for item in items]),
            }
        )
    return summary


def run_cv(by_pair_prefixes: dict[tuple[int, str], list[dict[str, Any]]], by_pair_states: dict[tuple[int, str], list[dict[str, Any]]], max_candidates: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    keys = sorted(by_pair_prefixes)
    fold_rows = []
    selection_rows = []
    value_weights = []
    for fold_idx, (train_qids, test_qids) in enumerate(question_folds(keys)):
        train_items = [(key, by_pair_prefixes[key], by_pair_states[key]) for key in keys if key[1] in train_qids]
        test_items = [(key, by_pair_prefixes[key], by_pair_states[key]) for key in keys if key[1] in test_qids]
        train_sample0_rows = [sample0_at(prefixes[0]) for _key, prefixes, _states in train_items]
        train_alt_rows = [row for _key, prefixes, _states in train_items for prefix in prefixes for row in prefix["basins"] if safe_float(row["contains_sample0"]) <= 0]
        train_all_rows = [row for _key, prefixes, _states in train_items for prefix in prefixes for row in prefix["basins"]]
        train_state_rows = [row for _key, _prefixes, states in train_items for row in states]
        for feature_name, basin_features in [("numeric", FEATURE_SETS["numeric_adaptive"]), ("full", FEATURE_SETS["full_adaptive"])]:
            trust_model = TorchLogisticModel(basin_features)
            trust_model.fit(train_sample0_rows, "representative_strict_correct", epochs=200, lr=0.08, l2=0.02)
            switch_model = TorchLogisticModel(basin_features)
            switch_model.fit(train_alt_rows, "switch_gain_label", epochs=200, lr=0.08, l2=0.02)
            basin_model = TorchLogisticModel(basin_features)
            basin_model.fit(train_all_rows, "representative_strict_correct", epochs=200, lr=0.08, l2=0.02)
            for profile in ["quality", "balanced", "production"]:
                value_model = TorchLinearRegressor(STATE_FEATURES)
                value_model.fit(train_state_rows, f"continue_value_{profile}", epochs=260, lr=0.05, l2=0.02)
                params = tune_policy(train_items, trust_model, switch_model, basin_model, value_model, profile, max_candidates)
                selections = [
                    simulate_pair(prefixes, states, trust_model, switch_model, basin_model, value_model, params["trust_threshold"], params["switch_threshold"], params["value_threshold"], params["basin_margin"], params["min_candidates"], max_candidates)
                    for _key, prefixes, states in test_items
                ]
                method = f"value_policy_{profile}_{feature_name}"
                metrics = evaluate(test_items, selections, method, f"fold_{fold_idx}")
                metrics.update({"feature_set": feature_name, **params})
                fold_rows.append(metrics)
                value_weights.extend(value_model.weight_rows(f"{method}_continue_value", f"fold_{fold_idx}"))
                for (key, _prefixes, _states), (row, generated, reason) in zip(test_items, selections):
                    s0 = sample0_at(by_pair_prefixes[key][0])
                    selection_rows.append(
                        {
                            "split": f"fold_{fold_idx}",
                            "method": method,
                            "seed": key[0],
                            "question_id": key[1],
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
    return summarize(fold_rows) + fold_rows, selection_rows, value_weights


def write_report(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary = [row for row in rows if row["split"] == "question_grouped_cv"]
    lines = [
        "# Prefix Marginal Continue-Value Policy",
        "",
        "| Method | Strict | Delta | Improved | Damaged | Net | Avg Gen | Cost vs 1x | Save vs 8x | Trust | Switch | Value Stop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | "
            f"`{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{int(row['net_gain_count'])}` | `{safe_float(row['avg_generated_candidates']):.2f}` | "
            f"`{safe_float(row['token_cost_vs_sample0']):.2f}x` | `{safe_float(row['token_savings_vs_full8']):.2%}` | `{safe_float(row['trust_stop_rate']):.2%}` | "
            f"`{safe_float(row['switch_stop_rate']):.2%}` | `{safe_float(row['value_stop_rate']):.2%}` |"
        )
    (output_dir / "prefix_value_policy_findings_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    candidate_rows = read_csv(Path(args.candidate_run) / "candidate_features.csv")
    by_pair_prefixes, _sample0_rows, _alt_rows = build_prefix_dataset(candidate_rows, args.max_candidates)
    by_pair_states = build_value_state_rows(by_pair_prefixes, args.max_candidates)
    cv_rows, selection_rows, value_weights = run_cv(by_pair_prefixes, by_pair_states, args.max_candidates)
    write_csv(output_dir / "prefix_value_cv_results.csv", cv_rows)
    write_csv(output_dir / "prefix_value_selection_rows.csv", selection_rows)
    write_csv(output_dir / "prefix_value_weights.csv", value_weights)
    write_json(output_dir / "run_metadata.json", {"pair_count": len(by_pair_prefixes), "max_candidates": args.max_candidates, "profile_lambdas": PROFILE_LAMBDAS, "state_features": STATE_FEATURES})
    write_report(output_dir, cv_rows)
    print(json.dumps({"output_dir": str(output_dir), "pair_count": len(by_pair_prefixes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
