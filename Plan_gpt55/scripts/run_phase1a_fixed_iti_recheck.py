#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path("/zhutingqi/song")
DEPLOY_EVAL_DIR = ROOT_DIR / "cand008_deploy_eval_v1"
INTERVENTION_A_DIR = ROOT_DIR / "ITI_for_entropy" / "Intervention_A"
INTERVENTION_C_SCRIPTS_DIR = ROOT_DIR / "ITI_for_entropy" / "Intervention_C" / "scripts"
SEMANTIC_DIR = ROOT_DIR / "semantic_entropy_calculate"
sys.path.insert(0, str(DEPLOY_EVAL_DIR))
sys.path.insert(0, str(INTERVENTION_A_DIR))
sys.path.insert(0, str(INTERVENTION_C_SCRIPTS_DIR))
sys.path.insert(0, str(SEMANTIC_DIR))

from intervention_utils import (  # noqa: E402
    build_prompt,
    evaluate_final_semantic_correct,
    evaluate_semantic_entropy_for_generations,
    sample_condition_generations,
    summarize_condition_metrics,
    trace_generation_with_optional_intervention,
)
from run_deploy_eval import (  # noqa: E402
    build_overall_summary,
    build_question_row,
    build_seed_summary,
    build_shared_context,
    ensure_dir,
    prepare_assets,
    safe_write_csv,
    safe_write_json,
    select_eval_rows,
)


DEFAULT_CONFIG = "/zhutingqi/song/Plan_gpt55/configs/phase1a_fixed_iti_recheck.json"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1A pure held-out fixed ITI recheck.")
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
            flat: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key)
                flat[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            writer.writerow(flat)


def candidate_site_payloads(candidate: dict[str, Any], asset_info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base_sites = asset_info["assets"]["sites"]
    weights = candidate.get("site_weights", {})
    payloads: dict[str, dict[str, Any]] = {}
    for site_key in candidate["site_keys"]:
        payload = base_sites[site_key]
        copied = dict(payload)
        copied["unit_direction"] = payload["unit_direction"].clone()
        copied["raw_direction"] = payload["raw_direction"].clone()
        copied["median_step_norm"] = float(payload["median_step_norm"]) * abs(float(weights.get(site_key, 1.0)))
        payloads[site_key] = copied
    return payloads


class ProgressReporter:
    def __init__(self, total_units: int) -> None:
        self.total_units = max(1, int(total_units))
        self.completed = 0
        self.started = time.time()

    def advance(self, label: str) -> None:
        self.completed += 1
        elapsed = max(time.time() - self.started, 1e-6)
        rate = self.completed / elapsed
        remaining = max(self.total_units - self.completed, 0)
        eta = remaining / max(rate, 1e-6)
        print(
            f"[PROGRESS] {self.completed}/{self.total_units} "
            f"({self.completed / self.total_units * 100:5.1f}%) "
            f"elapsed_min={elapsed / 60:.1f} eta_min={eta / 60:.1f} | {label}",
            flush=True,
        )


def evaluate_baseline_for_seed(
    *,
    seed: int,
    eval_rows: list[dict[str, Any]],
    config: dict[str, Any],
    shared: dict[str, Any],
    progress: ProgressReporter,
) -> list[dict[str, Any]]:
    evaluation = config["evaluation"]
    tokenizer = shared["generation_tokenizer"]
    model = shared["generation_model"]
    nli_tokenizer = shared["nli_tokenizer"]
    nli_model = shared["nli_model"]
    device = shared["device"]
    system_prompt = evaluation.get("system_prompt", "Answer the question briefly and factually.")
    baseline_rows: list[dict[str, Any]] = []
    for question_offset, row in enumerate(eval_rows):
        prompt = build_prompt(tokenizer, row["question"], system_prompt)
        seed_base = int(seed) + question_offset * 1000
        trace = trace_generation_with_optional_intervention(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=int(evaluation["trace_max_new_tokens"]),
            temperature=float(evaluation["temperature"]),
            top_p=float(evaluation["top_p"]),
            device=device,
            seed=seed_base,
            condition_name="baseline",
            capture_step_trace=False,
        )
        generations, _ = sample_condition_generations(
            model=model,
            tokenizer=tokenizer,
            question=row["question"],
            system_prompt=system_prompt,
            num_samples=int(evaluation["semantic_num_samples"]),
            max_new_tokens=int(evaluation["semantic_max_new_tokens"]),
            temperature=float(evaluation["temperature"]),
            top_p=float(evaluation["top_p"]),
            device=device,
            seed_base=seed_base + 100,
            condition_name="baseline",
            site_payloads=None,
            alpha_scale=0.0,
            target_mode="increase_semantic_high",
            trigger_mode="disabled",
            gate_entropy_threshold=0.0,
            enabled_sites=None,
        )
        semantic = evaluate_semantic_entropy_for_generations(
            question=row["question"],
            generations=generations,
            nli_tokenizer=nli_tokenizer,
            nli_model=nli_model,
            device=device,
        )
        correctness = evaluate_final_semantic_correct(
            question=row["question"],
            final_answer=trace["generated_answer_text"],
            ideal_answers=row["ideal_answers"],
            tokenizer=nli_tokenizer,
            model=nli_model,
            device=device,
        )
        baseline_rows.append(
            summarize_condition_metrics(
                row=row,
                condition_name="baseline",
                primary_trace=trace,
                semantic_metrics=semantic,
                correctness_info=correctness,
            )
        )
        progress.advance(f"baseline seed={seed} q={question_offset + 1}/{len(eval_rows)} id={row['id']}")
    return baseline_rows


def evaluate_candidate_for_seed(
    *,
    seed: int,
    candidate: dict[str, Any],
    eval_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    config: dict[str, Any],
    shared: dict[str, Any],
    asset_info: dict[str, Any],
    progress: ProgressReporter,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation = config["evaluation"]
    tokenizer = shared["generation_tokenizer"]
    model = shared["generation_model"]
    nli_tokenizer = shared["nli_tokenizer"]
    nli_model = shared["nli_model"]
    device = shared["device"]
    system_prompt = evaluation.get("system_prompt", "Answer the question briefly and factually.")
    site_payloads = candidate_site_payloads(candidate, asset_info)
    enabled_sites = set(candidate["site_keys"])
    intervention_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    for question_offset, row in enumerate(eval_rows):
        prompt = build_prompt(tokenizer, row["question"], system_prompt)
        seed_base = int(seed) + question_offset * 1000
        trace = trace_generation_with_optional_intervention(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=int(evaluation["trace_max_new_tokens"]),
            temperature=float(evaluation["temperature"]),
            top_p=float(evaluation["top_p"]),
            device=device,
            seed=seed_base,
            condition_name=candidate["name"],
            site_payloads=site_payloads,
            alpha_scale=float(candidate["alpha_scale"]),
            target_mode=candidate["target_mode"],
            trigger_mode=candidate.get("trigger_mode", "prev_entropy_quantile"),
            gate_entropy_threshold=float(asset_info["metadata"]["gate_entropy_threshold"]),
            enabled_sites=enabled_sites,
            capture_step_trace=False,
        )
        generations, _ = sample_condition_generations(
            model=model,
            tokenizer=tokenizer,
            question=row["question"],
            system_prompt=system_prompt,
            num_samples=int(evaluation["semantic_num_samples"]),
            max_new_tokens=int(evaluation["semantic_max_new_tokens"]),
            temperature=float(evaluation["temperature"]),
            top_p=float(evaluation["top_p"]),
            device=device,
            seed_base=seed_base + 100,
            condition_name=candidate["name"],
            site_payloads=site_payloads,
            alpha_scale=float(candidate["alpha_scale"]),
            target_mode=candidate["target_mode"],
            trigger_mode=candidate.get("trigger_mode", "prev_entropy_quantile"),
            gate_entropy_threshold=float(asset_info["metadata"]["gate_entropy_threshold"]),
            enabled_sites=enabled_sites,
        )
        semantic = evaluate_semantic_entropy_for_generations(
            question=row["question"],
            generations=generations,
            nli_tokenizer=nli_tokenizer,
            nli_model=nli_model,
            device=device,
        )
        correctness = evaluate_final_semantic_correct(
            question=row["question"],
            final_answer=trace["generated_answer_text"],
            ideal_answers=row["ideal_answers"],
            tokenizer=nli_tokenizer,
            model=nli_model,
            device=device,
        )
        metrics = summarize_condition_metrics(
            row=row,
            condition_name=candidate["name"],
            primary_trace=trace,
            semantic_metrics=semantic,
            correctness_info=correctness,
        )
        intervention_rows.append(metrics)
        question_rows.append(build_question_row(row, int(seed), question_offset + 1, baseline_rows[question_offset], metrics))
        progress.advance(f"{candidate['name']} seed={seed} q={question_offset + 1}/{len(eval_rows)} id={row['id']}")
    return intervention_rows, question_rows


def summarize_candidate(
    *,
    config: dict[str, Any],
    candidate: dict[str, Any],
    baseline_by_seed: dict[int, list[dict[str, Any]]],
    intervention_by_seed: dict[int, list[dict[str, Any]]],
    question_rows_by_seed: dict[int, list[dict[str, Any]]],
    asset_info: dict[str, Any],
    selection_label: str,
) -> dict[str, Any]:
    candidate_config = copy.deepcopy(config)
    candidate_config["candidate"] = candidate
    seed_summaries = [
        build_seed_summary(seed, baseline_by_seed[seed], intervention_by_seed[seed], question_rows_by_seed[seed])
        for seed in config["evaluation"]["seeds"]
    ]
    all_question_rows = [row for seed in config["evaluation"]["seeds"] for row in question_rows_by_seed[int(seed)]]
    overall = build_overall_summary(candidate_config, all_question_rows, seed_summaries, asset_info, selection_label)
    return {
        "candidate": candidate,
        "overall_summary": overall,
        "seed_summaries": seed_summaries,
        "question_rows": all_question_rows,
    }


def build_markdown(config: dict[str, Any], candidate_summaries: list[dict[str, Any]], selection_label: str) -> str:
    lines = [
        "# Phase 1A Fixed ITI Recheck",
        "",
        f"- Description: {config['description']}",
        f"- Selection: `{selection_label}`",
        f"- Seeds: `{config['evaluation']['seeds']}`",
        f"- Semantic samples: `{config['evaluation']['semantic_num_samples']}`",
        "",
        "## Candidate Summary",
        "",
        "| Candidate | Mode | Alpha | Answer Changed | Correct Delta | Token Mean Delta | Token Max Delta | Semantic Delta | Elapsed Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in candidate_summaries:
        candidate = item["candidate"]
        overall = item["overall_summary"]
        pairs = max(int(overall["question_pairs"]), 1)
        answer_rate = int(overall["answer_changed_count"]) / pairs
        lines.append(
            f"| `{candidate['name']}` | `{candidate['target_mode']}` | `{float(candidate['alpha_scale']):.6f}` | "
            f"`{answer_rate:.2%}` | `{overall['delta_final_semantic_correct_rate']:.6f}` | "
            f"`{overall['delta_token_mean_entropy_mean']:.6f}` | `{overall['delta_token_max_entropy_mean']:.6f}` | "
            f"`{overall['delta_semantic_entropy_weighted_mean']:.6f}` | `{overall['delta_elapsed_ms_mean']:.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- If answer changed rate remains below 10%, fixed ITI is too weak as a deployment route.",
            "- If correctness drops, fixed ITI is unsafe.",
            "- If token/semantic effects are smaller than seed/question variance, treat the result as diagnostic rather than deployable.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    tag = args.tag or config.get("tag", Path(args.config).stem)
    run_root = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}")
    safe_write_json(run_root / "config_snapshot.json", config)

    eval_rows, selection_label = select_eval_rows(config)
    asset_info = prepare_assets(config, run_root)
    shared = build_shared_context(config)
    total_units = len(eval_rows) * len(config["evaluation"]["seeds"]) * (1 + len(config["candidates"]))
    progress = ProgressReporter(total_units)

    baseline_by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed in config["evaluation"]["seeds"]:
        seed_int = int(seed)
        baseline_by_seed[seed_int] = evaluate_baseline_for_seed(
            seed=seed_int,
            eval_rows=eval_rows,
            config=config,
            shared=shared,
            progress=progress,
        )
    safe_write_json(run_root / "baseline_rows_by_seed.json", baseline_by_seed)

    candidate_summaries: list[dict[str, Any]] = []
    for candidate in config["candidates"]:
        candidate_dir = ensure_dir(run_root / "candidates" / candidate["name"])
        intervention_by_seed: dict[int, list[dict[str, Any]]] = {}
        question_rows_by_seed: dict[int, list[dict[str, Any]]] = {}
        for seed in config["evaluation"]["seeds"]:
            seed_int = int(seed)
            intervention_rows, question_rows = evaluate_candidate_for_seed(
                seed=seed_int,
                candidate=candidate,
                eval_rows=eval_rows,
                baseline_rows=baseline_by_seed[seed_int],
                config=config,
                shared=shared,
                asset_info=asset_info,
                progress=progress,
            )
            intervention_by_seed[seed_int] = intervention_rows
            question_rows_by_seed[seed_int] = question_rows
        summary = summarize_candidate(
            config=config,
            candidate=candidate,
            baseline_by_seed=baseline_by_seed,
            intervention_by_seed=intervention_by_seed,
            question_rows_by_seed=question_rows_by_seed,
            asset_info=asset_info,
            selection_label=selection_label,
        )
        candidate_summaries.append(summary)
        safe_write_json(candidate_dir / "summary.json", summary)
        safe_write_csv(candidate_dir / "seed_summaries.csv", summary["seed_summaries"])
        write_csv(candidate_dir / "question_rows.csv", summary["question_rows"])

    compact = [
        {
            "candidate": item["candidate"],
            "overall_summary": item["overall_summary"],
            "seed_summaries": item["seed_summaries"],
        }
        for item in candidate_summaries
    ]
    safe_write_json(run_root / "phase1a_summary.json", {"selection_label": selection_label, "candidates": compact})
    (run_root / "summary.md").write_text(build_markdown(config, candidate_summaries, selection_label), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "selection_label": selection_label,
                "question_count": len(eval_rows),
                "seed_count": len(config["evaluation"]["seeds"]),
                "candidate_count": len(config["candidates"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
