#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path("/zhutingqi/song")
DEPLOY_EVAL_DIR = ROOT_DIR / "cand008_deploy_eval_v1"
INTERVENTION_A_DIR = ROOT_DIR / "ITI_for_entropy" / "Intervention_A"
LAYER_LEVEL_DIR = ROOT_DIR / "layer_level"
SEMANTIC_DIR = ROOT_DIR / "semantic_entropy_calculate"
sys.path.insert(0, str(DEPLOY_EVAL_DIR))
sys.path.insert(0, str(INTERVENTION_A_DIR))
sys.path.insert(0, str(LAYER_LEVEL_DIR))
sys.path.insert(0, str(SEMANTIC_DIR))

from intervention_utils import (  # noqa: E402
    evaluate_final_semantic_correct,
    evaluate_semantic_entropy_for_generations,
    sample_condition_generations,
)
from run_deploy_eval import build_shared_context, ensure_dir, safe_write_json, select_eval_rows  # noqa: E402
from run_layer_entropy_validation import compute_contains_match, compute_exact_match  # noqa: E402


DEFAULT_CONFIG = "/zhutingqi/song/Plan_gpt55/configs/phase2a_reranking.json"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2A output-level reranking without hidden-state intervention.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: flatten(row.get(key)) for key in fieldnames})


def flatten(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) >= 2 else 0.0


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def text_preview(text: str, limit: int = 220) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


class ProgressReporter:
    def __init__(self, total_units: int) -> None:
        self.total_units = max(1, total_units)
        self.completed = 0
        self.started = time.time()

    def advance(self, label: str) -> None:
        self.completed += 1
        elapsed = max(time.time() - self.started, 1e-6)
        remaining = max(self.total_units - self.completed, 0)
        eta = remaining / max(self.completed / elapsed, 1e-6)
        print(
            f"[PROGRESS] {self.completed}/{self.total_units} "
            f"({self.completed / self.total_units * 100:5.1f}%) "
            f"elapsed_min={elapsed / 60:.1f} eta_min={eta / 60:.1f} | {label}",
            flush=True,
        )


def validate_config(config: dict[str, Any]) -> None:
    if int(config["evaluation"]["num_samples"]) < 2:
        raise ValueError("Phase 2A reranking needs at least 2 samples per question.")
    supported = {
        "single_sample_baseline",
        "random_of_n",
        "best_of_n_logprob",
        "semantic_cluster_majority",
        "low_token_mean_entropy",
        "cluster_then_low_entropy",
    }
    unknown = sorted(set(config["methods"]) - supported)
    if unknown:
        raise ValueError(f"Unsupported reranking methods: {unknown}")


def trace_by_sample(sample_traces: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(trace.get("sample_index", index)): trace for index, trace in enumerate(sample_traces)}


def pick_generation(
    method: str,
    generations: list[Any],
    sample_traces: list[dict[str, Any]],
    semantic_metrics: dict[str, Any],
    seed: int,
    question_offset: int,
) -> Any:
    traces = trace_by_sample(sample_traces)
    if method == "single_sample_baseline":
        return generations[0]
    if method == "random_of_n":
        return generations[(int(seed) + int(question_offset)) % len(generations)]
    if method == "best_of_n_logprob":
        return max(generations, key=lambda generation: float(generation.logprob_avg))
    if method == "low_token_mean_entropy":
        return min(generations, key=lambda generation: float(traces[generation.sample_index]["token_mean_entropy"]))

    clusters = list(semantic_metrics["clusters"])
    if not clusters:
        return generations[0]
    largest_cluster = max(
        clusters,
        key=lambda cluster: (int(cluster["size"]), float(cluster["weight_mass"]), -int(cluster["cluster_id"])),
    )
    members = [generations[int(index)] for index in largest_cluster["sample_indices"]]
    if method == "semantic_cluster_majority":
        return max(members, key=lambda generation: float(generation.logprob_avg))
    if method == "cluster_then_low_entropy":
        return min(members, key=lambda generation: float(traces[generation.sample_index]["token_mean_entropy"]))
    raise ValueError(f"Unsupported method: {method}")


def strict_correctness(answer: str, ideal_answers: list[str]) -> dict[str, Any]:
    exact = bool(compute_exact_match(answer, ideal_answers))
    contains = bool(compute_contains_match(answer, ideal_answers))
    return {
        "final_exact_match": float(exact),
        "final_contains_match": float(contains),
        "final_strict_correct": float(exact or contains),
    }


def evaluate_selected_answer(
    *,
    question: str,
    answer: str,
    ideal_answers: list[str],
    nli_tokenizer,
    nli_model,
    device: str,
) -> dict[str, Any]:
    strict = strict_correctness(answer, ideal_answers)
    nli = evaluate_final_semantic_correct(
        question=question,
        final_answer=answer,
        ideal_answers=ideal_answers,
        tokenizer=nli_tokenizer,
        model=nli_model,
        device=device,
    )
    return {
        **strict,
        "final_semantic_correct": float(nli["final_semantic_correct"]),
        "nli_only_correct": float(bool(nli["final_semantic_correct"]) and not bool(strict["final_strict_correct"])),
        "matched_ideal_answer": nli.get("matched_ideal_answer", ""),
        "match_source": nli.get("match_source", ""),
    }


def build_method_row(
    *,
    experiment_row: dict[str, Any],
    seed: int,
    question_offset: int,
    method: str,
    selected: Any,
    sample_trace: dict[str, Any],
    semantic_metrics: dict[str, Any],
    correctness: dict[str, Any],
) -> dict[str, Any]:
    clusters = semantic_metrics["clusters"]
    selected_cluster = next(
        (cluster for cluster in clusters if int(selected.sample_index) in {int(index) for index in cluster["sample_indices"]}),
        None,
    )
    return {
        "seed": int(seed),
        "question_index": int(question_offset + 1),
        "question_id": experiment_row["id"],
        "question": experiment_row["question"],
        "ideal_answers": experiment_row["ideal_answers"],
        "method": method,
        "selected_sample_index": int(selected.sample_index),
        "selected_cluster_id": int(selected.cluster_id) if selected.cluster_id is not None else -1,
        "selected_cluster_size": int(selected_cluster["size"]) if selected_cluster else 0,
        "selected_cluster_weight_mass": float(selected_cluster["weight_mass"]) if selected_cluster else 0.0,
        "answer_text": selected.text,
        "answer_preview": text_preview(selected.text),
        "token_count": int(selected.token_count),
        "logprob_sum": float(selected.logprob_sum),
        "logprob_avg": float(selected.logprob_avg),
        "token_mean_entropy": float(sample_trace["token_mean_entropy"]),
        "token_max_entropy": float(sample_trace["token_max_entropy"]),
        "elapsed_ms": float(sample_trace["elapsed_ms"]),
        "latency_ms": float(sample_trace.get("latency_ms", sample_trace["elapsed_ms"])),
        "prompt_tokens": int(sample_trace.get("prompt_tokens", 0)),
        "completion_tokens": int(sample_trace.get("completion_tokens", sample_trace.get("num_generated_tokens", selected.token_count))),
        "total_tokens": int(sample_trace.get("total_tokens", sample_trace.get("prompt_tokens", 0) + sample_trace.get("num_generated_tokens", selected.token_count))),
        "forward_calls": int(sample_trace.get("forward_calls", sample_trace.get("num_generated_tokens", selected.token_count))),
        "cuda_peak_memory_bytes": int(sample_trace.get("cuda_peak_memory_bytes", 0)),
        "semantic_entropy_weighted_set": float(semantic_metrics["semantic_entropy_weighted"]),
        "semantic_entropy_uniform_set": float(semantic_metrics["semantic_entropy_uniform"]),
        "semantic_clusters": int(semantic_metrics["semantic_clusters"]),
        **correctness,
    }


def summarize_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    summaries: list[dict[str, Any]] = []
    baseline_by_pair = {
        (row["seed"], row["question_id"]): row for row in rows if row["method"] == "single_sample_baseline"
    }
    for method, group in sorted(grouped.items()):
        strict = [float(row["final_strict_correct"]) for row in group]
        semantic = [float(row["final_semantic_correct"]) for row in group]
        nli_only = [float(row["nli_only_correct"]) for row in group]
        token_mean = [float(row["token_mean_entropy"]) for row in group]
        token_max = [float(row["token_max_entropy"]) for row in group]
        token_count = [float(row["token_count"]) for row in group]
        cluster_sizes = [float(row["selected_cluster_size"]) for row in group]
        answer_changed = []
        strict_deltas = []
        semantic_deltas = []
        for row in group:
            baseline = baseline_by_pair.get((row["seed"], row["question_id"]))
            if not baseline:
                continue
            answer_changed.append(float(row["answer_text"] != baseline["answer_text"]))
            strict_deltas.append(float(row["final_strict_correct"]) - float(baseline["final_strict_correct"]))
            semantic_deltas.append(float(row["final_semantic_correct"]) - float(baseline["final_semantic_correct"]))
        summaries.append(
            {
                "method": method,
                "pairs": len(group),
                "strict_correct_rate": mean(strict),
                "semantic_correct_rate": mean(semantic),
                "nli_only_correct_rate": mean(nli_only),
                "answer_changed_vs_sample0_rate": mean(answer_changed),
                "delta_strict_correct_vs_sample0": mean(strict_deltas),
                "delta_semantic_correct_vs_sample0": mean(semantic_deltas),
                "token_mean_entropy_mean": mean(token_mean),
                "token_mean_entropy_std": stdev(token_mean),
                "token_max_entropy_mean": mean(token_max),
                "token_count_mean": mean(token_count),
                "selected_cluster_size_mean": mean(cluster_sizes),
            }
        )
    return summaries


def build_markdown(config: dict[str, Any], summaries: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 2A Output-Level Reranking",
        "",
        f"- Description: {config['description']}",
        f"- Pairs per method: `{len(rows) // max(len(config['methods']), 1)}`",
        f"- Methods: `{', '.join(config['methods'])}`",
        "",
        "## Method Summary",
        "",
        "| Method | Strict Correct | NLI Correct | NLI-only | Δ Strict vs sample0 | Changed vs sample0 | Token Mean Ent | Token Count | Cluster Size |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| `{summary['method']}` | `{summary['strict_correct_rate']:.2%}` | "
            f"`{summary['semantic_correct_rate']:.2%}` | `{summary['nli_only_correct_rate']:.2%}` | "
            f"`{summary['delta_strict_correct_vs_sample0']:.4f}` | "
            f"`{summary['answer_changed_vs_sample0_rate']:.2%}` | "
            f"`{summary['token_mean_entropy_mean']:.4f}` | `{summary['token_count_mean']:.2f}` | "
            f"`{summary['selected_cluster_size_mean']:.2f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Prefer strict correctness over NLI correctness when they disagree.",
            "- A useful reranker should improve or preserve strict correctness while reducing token entropy or answer length.",
            "- A high NLI-only rate is an evaluator warning, not evidence of real improvement.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    validate_config(config)
    tag = args.tag or config.get("tag", Path(args.config).stem)
    run_root = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}")
    safe_write_json(run_root / "config_snapshot.json", config)

    eval_rows, selection_label = select_eval_rows(config)
    shared = build_shared_context(config)
    tokenizer = shared["generation_tokenizer"]
    model = shared["generation_model"]
    nli_tokenizer = shared["nli_tokenizer"]
    nli_model = shared["nli_model"]
    device = shared["device"]
    evaluation = config["evaluation"]
    system_prompt = evaluation.get("system_prompt", "Answer the question briefly and factually.")
    progress = ProgressReporter(len(eval_rows) * len(evaluation["seeds"]))

    method_rows: list[dict[str, Any]] = []
    sample_sets: list[dict[str, Any]] = []
    correctness_cache: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    for seed in evaluation["seeds"]:
        seed_int = int(seed)
        for question_offset, row in enumerate(eval_rows):
            seed_base = seed_int + question_offset * 1000
            generations, sample_traces = sample_condition_generations(
                model=model,
                tokenizer=tokenizer,
                question=row["question"],
                system_prompt=system_prompt,
                num_samples=int(evaluation["num_samples"]),
                max_new_tokens=int(evaluation["max_new_tokens"]),
                temperature=float(evaluation["temperature"]),
                top_p=float(evaluation["top_p"]),
                device=device,
                seed_base=seed_base,
                condition_name="phase2a_baseline_samples",
                site_payloads=None,
                alpha_scale=0.0,
                target_mode="increase_semantic_high",
                trigger_mode="disabled",
                gate_entropy_threshold=0.0,
                enabled_sites=None,
            )
            semantic_metrics = evaluate_semantic_entropy_for_generations(
                question=row["question"],
                generations=generations,
                nli_tokenizer=nli_tokenizer,
                nli_model=nli_model,
                device=device,
            )
            traces = trace_by_sample(sample_traces)
            for method in config["methods"]:
                selected = pick_generation(method, generations, sample_traces, semantic_metrics, seed_int, question_offset)
                cache_key = (row["question"], selected.text, tuple(row["ideal_answers"]))
                if cache_key not in correctness_cache:
                    correctness_cache[cache_key] = evaluate_selected_answer(
                        question=row["question"],
                        answer=selected.text,
                        ideal_answers=row["ideal_answers"],
                        nli_tokenizer=nli_tokenizer,
                        nli_model=nli_model,
                        device=device,
                    )
                method_rows.append(
                    build_method_row(
                        experiment_row=row,
                        seed=seed_int,
                        question_offset=question_offset,
                        method=method,
                        selected=selected,
                        sample_trace=traces[selected.sample_index],
                        semantic_metrics=semantic_metrics,
                        correctness=correctness_cache[cache_key],
                    )
                )
            sample_sets.append(
                {
                    "seed": seed_int,
                    "question_id": row["id"],
                    "semantic_entropy_weighted": semantic_metrics["semantic_entropy_weighted"],
                    "semantic_clusters": semantic_metrics["semantic_clusters"],
                    "generations": [
                        {
                            "sample_index": int(generation.sample_index),
                            "cluster_id": int(generation.cluster_id) if generation.cluster_id is not None else -1,
                            "text": generation.text,
                            "token_count": int(generation.token_count),
                            "logprob_avg": float(generation.logprob_avg),
                            "token_mean_entropy": float(traces[generation.sample_index]["token_mean_entropy"]),
                            "token_max_entropy": float(traces[generation.sample_index]["token_max_entropy"]),
                            "elapsed_ms": float(traces[generation.sample_index]["elapsed_ms"]),
                            "latency_ms": float(traces[generation.sample_index].get("latency_ms", traces[generation.sample_index]["elapsed_ms"])),
                            "prompt_tokens": int(traces[generation.sample_index].get("prompt_tokens", 0)),
                            "completion_tokens": int(traces[generation.sample_index].get("completion_tokens", traces[generation.sample_index].get("num_generated_tokens", generation.token_count))),
                            "total_tokens": int(
                                traces[generation.sample_index].get(
                                    "total_tokens",
                                    traces[generation.sample_index].get("prompt_tokens", 0)
                                    + traces[generation.sample_index].get("num_generated_tokens", generation.token_count),
                                )
                            ),
                            "forward_calls": int(traces[generation.sample_index].get("forward_calls", traces[generation.sample_index].get("num_generated_tokens", generation.token_count))),
                            "cuda_peak_memory_bytes": int(traces[generation.sample_index].get("cuda_peak_memory_bytes", 0)),
                        }
                        for generation in generations
                    ],
                    "clusters": semantic_metrics["clusters"],
                }
            )
            progress.advance(f"seed={seed_int} q={question_offset + 1}/{len(eval_rows)} id={row['id']}")

    summaries = summarize_methods(method_rows)
    write_csv(run_root / "method_rows.csv", method_rows)
    write_csv(run_root / "method_summary.csv", summaries)
    safe_write_json(run_root / "sample_sets.json", sample_sets)
    safe_write_json(
        run_root / "phase2a_summary.json",
        {
            "selection_label": selection_label,
            "method_summary": summaries,
            "row_count": len(method_rows),
            "sample_set_count": len(sample_sets),
        },
    )
    (run_root / "summary.md").write_text(build_markdown(config, summaries, method_rows), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "selection_label": selection_label,
                "question_count": len(eval_rows),
                "seed_count": len(evaluation["seeds"]),
                "method_count": len(config["methods"]),
                "method_row_count": len(method_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
