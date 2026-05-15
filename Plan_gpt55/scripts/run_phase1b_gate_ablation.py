#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_phase1a_fixed_iti_recheck import (  # noqa: E402
    ProgressReporter,
    build_overall_summary,
    build_seed_summary,
    build_shared_context,
    candidate_site_payloads,
    ensure_dir,
    evaluate_baseline_for_seed,
    evaluate_candidate_for_seed,
    read_json,
    safe_write_csv,
    safe_write_json,
    select_eval_rows,
)
from run_deploy_eval import prepare_assets  # noqa: E402


DEFAULT_CONFIG = "/zhutingqi/song/Plan_gpt55/configs/phase1b_gate_ablation.json"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1B gate ablation for fixed ITI candidates.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {}
            for key in fieldnames:
                value = row.get(key)
                flat[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            writer.writerow(flat)


def gate_asset_quantile(gate: dict[str, Any], fallback_quantile: float) -> float:
    if gate["trigger_mode"] == "always":
        return float(gate.get("asset_gate_entropy_quantile", fallback_quantile))
    return float(gate["gate_entropy_quantile"])


def validate_config(config: dict[str, Any]) -> None:
    if not config.get("base_candidates"):
        raise ValueError("Phase 1B config requires at least one base candidate.")
    if not config.get("gate_variants"):
        raise ValueError("Phase 1B config requires at least one gate variant.")
    supported_modes = {"always", "prev_entropy_quantile"}
    names: set[str] = set()
    for gate in config["gate_variants"]:
        mode = gate["trigger_mode"]
        if mode not in supported_modes:
            raise ValueError(f"Unsupported gate trigger_mode={mode!r}; supported={sorted(supported_modes)}")
        if mode == "prev_entropy_quantile":
            quantile = float(gate["gate_entropy_quantile"])
            if not 0.0 <= quantile <= 1.0:
                raise ValueError(f"Invalid gate quantile {quantile}; expected [0, 1].")
        if gate["name"] in names:
            raise ValueError(f"Duplicate gate name: {gate['name']}")
        names.add(gate["name"])


def expand_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    fallback_quantile = float(config["asset_prep"].get("gate_entropy_quantile", 0.67))
    candidates: list[dict[str, Any]] = []
    for base in config["base_candidates"]:
        for gate in config["gate_variants"]:
            candidate = copy.deepcopy(base)
            candidate["base_candidate_name"] = base["name"]
            candidate["gate_name"] = gate["name"]
            candidate["trigger_mode"] = gate["trigger_mode"]
            candidate["gate_entropy_quantile"] = gate.get("gate_entropy_quantile")
            candidate["asset_gate_entropy_quantile"] = gate_asset_quantile(gate, fallback_quantile)
            candidate["name"] = f"{base['name']}__{gate['name']}"
            candidates.append(candidate)
    return candidates


def prepare_assets_by_quantile(config: dict[str, Any], run_root: Path, candidates: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    assets_by_quantile: dict[float, dict[str, Any]] = {}
    for quantile in sorted({float(candidate["asset_gate_entropy_quantile"]) for candidate in candidates}):
        asset_config = copy.deepcopy(config)
        asset_config["asset_prep"]["gate_entropy_quantile"] = quantile
        print(f"[CHECK] preparing assets for gate_quantile={quantile:.2f}", flush=True)
        asset_info = prepare_assets(asset_config, run_root / f"assets_q{str(quantile).replace('.', 'p')}")
        threshold = float(asset_info["metadata"]["gate_entropy_threshold"])
        if not math.isfinite(threshold):
            raise RuntimeError(f"Non-finite gate threshold for quantile {quantile}: {threshold}")
        print(f"[CHECK] asset gate_quantile={quantile:.2f} threshold={threshold:.6f}", flush=True)
        assets_by_quantile[quantile] = asset_info
    return assets_by_quantile


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
    candidate_config["asset_prep"]["gate_entropy_quantile"] = float(candidate["asset_gate_entropy_quantile"])
    candidate_config["candidate"] = candidate
    seed_summaries = [
        build_seed_summary(seed, baseline_by_seed[int(seed)], intervention_by_seed[int(seed)], question_rows_by_seed[int(seed)])
        for seed in config["evaluation"]["seeds"]
    ]
    all_question_rows = [row for seed in config["evaluation"]["seeds"] for row in question_rows_by_seed[int(seed)]]
    overall = build_overall_summary(candidate_config, all_question_rows, seed_summaries, asset_info, selection_label)
    overall["base_candidate_name"] = candidate["base_candidate_name"]
    overall["gate_name"] = candidate["gate_name"]
    overall["trigger_mode"] = candidate["trigger_mode"]
    return {
        "candidate": candidate,
        "overall_summary": overall,
        "seed_summaries": seed_summaries,
        "question_rows": all_question_rows,
    }


def build_markdown(config: dict[str, Any], candidate_summaries: list[dict[str, Any]], selection_label: str) -> str:
    lines = [
        "# Phase 1B Gate Ablation",
        "",
        f"- Description: {config['description']}",
        f"- Selection: `{selection_label}`",
        f"- Seeds: `{config['evaluation']['seeds']}`",
        f"- Semantic samples: `{config['evaluation']['semantic_num_samples']}`",
        "",
        "## Gate Summary",
        "",
        "| Candidate | Gate | Trigger | Gate q | Events | Answer Changed | Correct Delta | Token Mean Delta | Semantic Delta | Elapsed Delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in candidate_summaries:
        candidate = item["candidate"]
        overall = item["overall_summary"]
        pairs = max(int(overall["question_pairs"]), 1)
        answer_rate = int(overall["answer_changed_count"]) / pairs
        lines.append(
            f"| `{candidate['base_candidate_name']}` | `{candidate['gate_name']}` | `{candidate['trigger_mode']}` | "
            f"`{float(candidate['asset_gate_entropy_quantile']):.2f}` | `{overall['mean_intervention_total_events']:.3f}` | "
            f"`{answer_rate:.2%}` | `{overall['delta_final_semantic_correct_rate']:.6f}` | "
            f"`{overall['delta_token_mean_entropy_mean']:.6f}` | `{overall['delta_semantic_entropy_weighted_mean']:.6f}` | "
            f"`{overall['delta_elapsed_ms_mean']:.3f}` |"
        )
    lines.extend(
        [
            "",
            "## Reasonableness Checks",
            "",
            "- `always` should usually have the largest event count; if it does not, inspect generation-length drift and event traces.",
            "- Lower previous-entropy quantiles should usually trigger at least as often as higher quantiles, but exact monotonicity is not guaranteed because the intervention can change the continuation.",
            "- A gate is not deployable if it only increases event count while correctness drops or answer changes remain negligible.",
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
    candidates = expand_candidates(config)
    print(
        f"[CHECK] selected_questions={len(eval_rows)} seeds={len(config['evaluation']['seeds'])} "
        f"base_candidates={len(config['base_candidates'])} gate_variants={len(config['gate_variants'])} "
        f"expanded_candidates={len(candidates)}",
        flush=True,
    )
    assets_by_quantile = prepare_assets_by_quantile(config, run_root, candidates)
    shared = build_shared_context(config)
    total_units = len(eval_rows) * len(config["evaluation"]["seeds"]) * (1 + len(candidates))
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
        if len(baseline_by_seed[seed_int]) != len(eval_rows):
            raise RuntimeError(f"Baseline row count mismatch for seed={seed_int}")
    safe_write_json(run_root / "baseline_rows_by_seed.json", baseline_by_seed)
    print("[CHECK] baseline complete; row counts match selection", flush=True)

    candidate_summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_dir = ensure_dir(run_root / "candidates" / candidate["name"])
        asset_info = assets_by_quantile[float(candidate["asset_gate_entropy_quantile"])]
        _ = candidate_site_payloads(candidate, asset_info)
        intervention_by_seed: dict[int, list[dict[str, Any]]] = {}
        question_rows_by_seed: dict[int, list[dict[str, Any]]] = {}
        print(
            f"[CHECK] candidate={candidate['name']} trigger={candidate['trigger_mode']} "
            f"gate_q={candidate['asset_gate_entropy_quantile']:.2f}",
            flush=True,
        )
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
            if len(question_rows) != len(eval_rows):
                raise RuntimeError(f"Question row count mismatch for {candidate['name']} seed={seed_int}")
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
        overall = summary["overall_summary"]
        print(
            f"[CHECK] done {candidate['name']} events={overall['mean_intervention_total_events']:.3f} "
            f"answer_changed={int(overall['answer_changed_count'])}/{int(overall['question_pairs'])} "
            f"correct_delta={overall['delta_final_semantic_correct_rate']:.6f}",
            flush=True,
        )

    compact = [
        {
            "candidate": item["candidate"],
            "overall_summary": item["overall_summary"],
            "seed_summaries": item["seed_summaries"],
        }
        for item in candidate_summaries
    ]
    safe_write_json(run_root / "phase1b_summary.json", {"selection_label": selection_label, "candidates": compact})
    (run_root / "summary.md").write_text(build_markdown(config, candidate_summaries, selection_label), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "selection_label": selection_label,
                "question_count": len(eval_rows),
                "seed_count": len(config["evaluation"]["seeds"]),
                "candidate_count": len(candidates),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
