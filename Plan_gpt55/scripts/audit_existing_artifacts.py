#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "/zhutingqi/song/Plan_gpt55/configs/artifact_audit_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit existing ITI artifacts and summarize evidence strength.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ratio(mean_value: float, std_value: float) -> float:
    if abs(std_value) <= 1e-12:
        return float("inf") if abs(mean_value) > 0 else 0.0
    return abs(mean_value) / abs(std_value)


def classify_effect(mean_value: float, std_value: float, small_effect_snr: float) -> str:
    snr = ratio(mean_value, std_value)
    if snr < small_effect_snr:
        return "noise_scale"
    if mean_value < 0:
        return "directionally_beneficial"
    if mean_value > 0:
        return "directionally_harmful"
    return "neutral"


def audit_intervention_d(config: dict[str, Any]) -> dict[str, Any]:
    section = config["intervention_d"]
    run_cfg = read_json(section["config_snapshot"])
    pareto_payload = read_json(section["final_pareto"])
    candidate_rows = read_csv_rows(section["final_candidates_csv"])
    entries = pareto_payload.get("entries", [])
    final_entry = entries[0] if entries else None
    overall = final_entry["evaluation"]["overall"] if final_entry else {}
    candidate = final_entry["candidate"] if final_entry else {}
    heldout_thresholds = run_cfg.get("robustness", {}).get("heldout", {})

    failed_gates: list[str] = []
    for key, metric in [
        ("semantic_band_min_rate", "semantic_band_success_rate"),
        ("token_mean_nonpositive_min_rate", "token_mean_nonpositive_rate"),
        ("token_max_nonpositive_min_rate", "token_max_nonpositive_rate"),
        ("correct_nonnegative_min_rate", "correct_nonnegative_rate"),
        ("exact_nonnegative_min_rate", "exact_nonnegative_rate"),
        ("safety_success_min_rate", "safety_success_rate"),
    ]:
        threshold = safe_float(heldout_thresholds.get(key, 0.0))
        value = safe_float(overall.get(metric, 0.0))
        if value < threshold:
            failed_gates.append(f"{metric}={value:.3f} < {threshold:.3f}")
    max_stability = safe_float(heldout_thresholds.get("max_stability_penalty", float("inf")))
    stability = safe_float(overall.get("stability_penalty", 0.0))
    if stability > max_stability:
        failed_gates.append(f"stability_penalty={stability:.3f} > {max_stability:.3f}")

    return {
        "run_root": section["run_root"],
        "final_pareto_size": len(entries),
        "final_candidate_count": len(candidate_rows),
        "candidate_id": candidate.get("individual_id", ""),
        "candidate_signature": candidate.get("signature", ""),
        "target_mode": candidate.get("target_mode", ""),
        "alpha_scale": safe_float(candidate.get("alpha_scale")),
        "gate_entropy_quantile": safe_float(candidate.get("gate_entropy_quantile")),
        "heldout_overall": overall,
        "heldout_gate_failed": bool(failed_gates),
        "failed_gates": failed_gates,
        "interpretation": (
            "fallback_pareto_representative_not_robust_solution"
            if failed_gates
            else "passed_configured_heldout_gate"
        ),
    }


def audit_deploy_eval(config: dict[str, Any]) -> dict[str, Any]:
    section = config["deploy_eval"]
    thresholds = config["interpretation_thresholds"]
    summary = read_json(section["overall_summary"])
    small_effect_snr = safe_float(thresholds.get("small_effect_snr", 0.2))
    question_pairs = max(int(summary.get("question_pairs", 0)), 1)
    answer_change_rate = safe_float(summary.get("answer_changed_count")) / float(question_pairs)

    metric_judgments = {}
    for metric in [
        "delta_semantic_entropy_weighted",
        "delta_token_mean_entropy",
        "delta_token_max_entropy",
        "delta_elapsed_ms",
    ]:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        metric_judgments[metric] = {
            "mean": safe_float(summary.get(mean_key)),
            "std": safe_float(summary.get(std_key)),
            "abs_mean_over_std": ratio(safe_float(summary.get(mean_key)), safe_float(summary.get(std_key))),
            "classification": classify_effect(
                safe_float(summary.get(mean_key)),
                safe_float(summary.get(std_key)),
                small_effect_snr,
            ),
        }

    return {
        "run_root": section["run_root"],
        "question_pairs": question_pairs,
        "num_unique_questions": int(summary.get("num_unique_questions", 0)),
        "num_seeds": int(summary.get("num_seeds", 0)),
        "answer_changed_count": int(summary.get("answer_changed_count", 0)),
        "answer_change_rate": answer_change_rate,
        "correctness_delta": safe_float(summary.get("delta_final_semantic_correct_rate")),
        "exact_delta": safe_float(summary.get("delta_final_exact_match_rate")),
        "metric_judgments": metric_judgments,
        "interpretation": (
            "weak_behavioral_perturbation"
            if answer_change_rate < safe_float(thresholds.get("min_answer_change_rate_for_strong_claim", 0.1))
            else "substantial_behavioral_perturbation"
        ),
    }


def build_markdown(payload: dict[str, Any]) -> str:
    d = payload["intervention_d"]
    deploy = payload["deploy_eval"]
    lines = [
        "# Existing Artifact Audit",
        "",
        "## Intervention_D",
        "",
        f"- Run: `{d['run_root']}`",
        f"- Final Pareto size: `{d['final_pareto_size']}`",
        f"- Final candidate count: `{d['final_candidate_count']}`",
        f"- Candidate: `{d['candidate_id']}`",
        f"- Mode / alpha: `{d['target_mode']} @ {d['alpha_scale']:.6f}`",
        f"- Interpretation: `{d['interpretation']}`",
        "",
        "### Held-Out Gate Failures",
        "",
    ]
    if d["failed_gates"]:
        lines.extend([f"- `{item}`" for item in d["failed_gates"]])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## cand008 Deploy Eval",
            "",
            f"- Run: `{deploy['run_root']}`",
            f"- Question pairs: `{deploy['question_pairs']}`",
            f"- Unique questions: `{deploy['num_unique_questions']}`",
            f"- Seeds: `{deploy['num_seeds']}`",
            f"- Answer changed: `{deploy['answer_changed_count']}` ({deploy['answer_change_rate']:.2%})",
            f"- Correctness delta: `{deploy['correctness_delta']:.6f}`",
            f"- Exact delta: `{deploy['exact_delta']:.6f}`",
            f"- Interpretation: `{deploy['interpretation']}`",
            "",
            "### Effect Strength",
            "",
            "| Metric | Mean | Std | abs(mean)/std | Classification |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for metric, row in deploy["metric_judgments"].items():
        lines.append(
            f"| `{metric}` | `{row['mean']:.6f}` | `{row['std']:.6f}` | "
            f"`{row['abs_mean_over_std']:.3f}` | `{row['classification']}` |"
        )
    lines.extend(
        [
            "",
            "## Overall Judgment",
            "",
            "- The current evidence supports entropy-related mechanism discovery.",
            "- It does not support claiming a robust deployable fixed-ITI solution.",
            "- The strongest next step is to treat ITI as a diagnostic probe and move optimization to learned or decoding-level control.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "intervention_d": audit_intervention_d(config),
        "deploy_eval": audit_deploy_eval(config),
    }
    (output_dir / "artifact_audit_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "artifact_audit_summary.md").write_text(
        build_markdown(payload),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "status": "ok"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
