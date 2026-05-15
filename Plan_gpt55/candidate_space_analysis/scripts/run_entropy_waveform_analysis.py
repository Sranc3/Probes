#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_entropy_anatomy import auc_score, cohen_d, ensure_dir, mean, read_csv, safe_float, stdev, write_csv, write_json
from run_learned_basin_verifier import LogisticModel, question_folds


DEFAULT_CANDIDATE_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25"
DEFAULT_ENTROPY_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260429_063326_entropy_anatomy"
DEFAULT_PHASE2A_RUN = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_100339_phase2a_reranking_all200_qwen25"
DEFAULT_OUTPUT_ROOT = "/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs"


AGGREGATE_FEATURES = [
    "token_mean_entropy_mean",
    "token_mean_entropy_std",
    "token_max_entropy_mean",
    "logprob_avg_mean",
    "cluster_weight_mass",
    "cluster_size",
    "centroid_entropy_z",
    "centroid_logprob_z",
    "fragmentation_entropy",
    "top2_weight_margin",
    "stable_score",
    "stable_hallucination_score",
]

WAVEFORM_FEATURES = [
    "wf_entropy_mean_mean",
    "wf_entropy_std_mean",
    "wf_entropy_early_mean_mean",
    "wf_entropy_mid_mean_mean",
    "wf_entropy_late_mean_mean",
    "wf_entropy_late_minus_early_mean",
    "wf_entropy_slope_mean",
    "wf_entropy_auc_mean",
    "wf_entropy_max_mean",
    "wf_entropy_max_pos_norm_mean",
    "wf_entropy_spike_count_mean",
    "wf_entropy_spike_frac_mean",
    "wf_entropy_early_collapse_mean",
    "wf_entropy_late_hesitation_mean",
    "wf_prob_mean_mean",
    "wf_prob_min_mean",
    "wf_basin_entropy_curve_var",
    "wf_basin_prob_curve_var",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze token-level entropy waveforms across answer basins.")
    parser.add_argument("--candidate-run", default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--entropy-run", default=DEFAULT_ENTROPY_RUN)
    parser.add_argument("--phase2a-run", default=DEFAULT_PHASE2A_RUN)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default="entropy_waveform_analysis")
    parser.add_argument("--limit", type=int, default=0, help="Optional candidate limit for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=8, help="Teacher-forcing batch size.")
    return parser.parse_args()


def load_config(phase2a_run: Path) -> dict[str, Any]:
    return json.loads((phase2a_run / "config_snapshot.json").read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_text(tokenizer: Any, system_prompt: str, question: str) -> str:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{system_prompt}\n\nQuestion: {question}\nAnswer:"


def token_entropy_trace(
    tokenizer: Any,
    model: Any,
    prompt: str,
    answer: str,
    device: torch.device,
) -> dict[str, Any]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    if not answer_ids:
        return {"tokens": [], "entropies": [], "chosen_probs": [], "token_count": 0}
    input_ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]
    entropies: list[float] = []
    chosen_probs: list[float] = []
    tokens: list[str] = []
    for idx, token_id in enumerate(answer_ids):
        pred_pos = len(prompt_ids) + idx - 1
        if pred_pos < 0 or pred_pos >= logits.shape[0]:
            continue
        step_logits = logits[pred_pos].float()
        log_probs = torch.log_softmax(step_logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs).item()
        entropies.append(float(entropy))
        chosen_probs.append(float(probs[token_id].item()))
        tokens.append(tokenizer.decode([token_id], skip_special_tokens=False))
    return {"tokens": tokens, "entropies": entropies, "chosen_probs": chosen_probs, "token_count": len(tokens)}


def batch_token_entropy_traces(
    tokenizer: Any,
    model: Any,
    items: list[tuple[str, str]],
    device: torch.device,
) -> list[dict[str, Any]]:
    encoded_items = []
    for prompt, answer in items:
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
        encoded_items.append({"prompt_ids": prompt_ids, "answer_ids": answer_ids, "input_ids": prompt_ids + answer_ids})
    if not encoded_items:
        return []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_len = max(len(item["input_ids"]) for item in encoded_items)
    input_ids = []
    attention_mask = []
    for item in encoded_items:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_len)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids=input_tensor, attention_mask=mask_tensor).logits
    traces: list[dict[str, Any]] = []
    for batch_idx, item in enumerate(encoded_items):
        answer_ids = item["answer_ids"]
        prompt_len = len(item["prompt_ids"])
        entropies: list[float] = []
        chosen_probs: list[float] = []
        tokens: list[str] = []
        for idx, token_id in enumerate(answer_ids):
            pred_pos = prompt_len + idx - 1
            if pred_pos < 0 or pred_pos >= logits.shape[1]:
                continue
            step_logits = logits[batch_idx, pred_pos].float()
            log_probs = torch.log_softmax(step_logits, dim=-1)
            probs = torch.exp(log_probs)
            entropies.append(float(-torch.sum(probs * log_probs).item()))
            chosen_probs.append(float(probs[token_id].item()))
            tokens.append(tokenizer.decode([token_id], skip_special_tokens=False))
        traces.append({"tokens": tokens, "entropies": entropies, "chosen_probs": chosen_probs, "token_count": len(tokens)})
    return traces


def thirds(values: list[float]) -> tuple[list[float], list[float], list[float]]:
    if not values:
        return [], [], []
    n = len(values)
    a = max(1, n // 3)
    b = max(a + 1, (2 * n) // 3) if n >= 3 else n
    return values[:a], values[a:b], values[b:]


def slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    xs = list(range(len(values)))
    x_mu = mean([float(x) for x in xs])
    y_mu = mean(values)
    denom = sum((x - x_mu) ** 2 for x in xs)
    if denom <= 0:
        return 0.0
    return sum((x - x_mu) * (y - y_mu) for x, y in zip(xs, values)) / denom


def normalized_curve(values: list[float], points: int = 20) -> list[float]:
    if not values:
        return [0.0] * points
    if len(values) == 1:
        return [values[0]] * points
    output = []
    for i in range(points):
        pos = i * (len(values) - 1) / (points - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            output.append(values[lo])
        else:
            weight = pos - lo
            output.append(values[lo] * (1 - weight) + values[hi] * weight)
    return output


def waveform_features(entropies: list[float], probs: list[float]) -> dict[str, Any]:
    early, mid, late = thirds(entropies)
    mu = mean(entropies)
    sd = stdev(entropies)
    threshold = mu + sd
    spikes = [value for value in entropies if value > threshold and value > 0.25]
    max_entropy = max(entropies) if entropies else 0.0
    max_pos = entropies.index(max_entropy) / max(1, len(entropies) - 1) if entropies else 0.0
    early_mean = mean(early)
    late_mean = mean(late)
    prob_early, _prob_mid, prob_late = thirds(probs)
    early_collapse = mean(prob_early) - mean(probs) if probs else 0.0
    late_hesitation = late_mean - mu
    return {
        "wf_token_count": len(entropies),
        "wf_entropy_mean": mu,
        "wf_entropy_std": sd,
        "wf_entropy_early_mean": early_mean,
        "wf_entropy_mid_mean": mean(mid),
        "wf_entropy_late_mean": late_mean,
        "wf_entropy_late_minus_early": late_mean - early_mean,
        "wf_entropy_slope": slope(entropies),
        "wf_entropy_auc": sum(entropies) / max(1, len(entropies)),
        "wf_entropy_max": max_entropy,
        "wf_entropy_max_pos_norm": max_pos,
        "wf_entropy_spike_count": len(spikes),
        "wf_entropy_spike_frac": len(spikes) / max(1, len(entropies)),
        "wf_entropy_early_collapse": early_collapse,
        "wf_entropy_late_hesitation": late_hesitation,
        "wf_prob_mean": mean(probs),
        "wf_prob_min": min(probs) if probs else 0.0,
        "wf_prob_early_mean": mean(prob_early),
        "wf_prob_late_mean": mean(prob_late),
        "wf_entropy_curve": normalized_curve(entropies),
        "wf_prob_curve": normalized_curve(probs),
    }


def curve_variance(curves: list[list[float]]) -> float:
    if len(curves) < 2:
        return 0.0
    width = len(curves[0])
    variances = []
    for idx in range(width):
        values = [curve[idx] for curve in curves]
        variances.append(stdev(values) ** 2)
    return mean(variances)


def build_waveforms(
    candidate_rows: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_dir = config["model_dir"]
    system_prompt = config["evaluation"]["system_prompt"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map={"": 0} if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    unique_items: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in candidate_rows:
        key = (row["question"], row["answer_text"])
        if key in seen:
            continue
        seen.add(key)
        unique_items.append((row["question"], row["answer_text"], prompt_text(tokenizer, system_prompt, row["question"])))
    for start in range(0, len(unique_items), batch_size):
        batch = unique_items[start : start + batch_size]
        traces = batch_token_entropy_traces(tokenizer, model, [(prompt, answer) for question, answer, prompt in batch], device)
        for (question, answer, _prompt), trace in zip(batch, traces):
            cache[(question, answer)] = trace
        print(json.dumps({"teacher_forced_unique": min(start + batch_size, len(unique_items)), "unique_total": len(unique_items)}, ensure_ascii=False))
    trace_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    total = len(candidate_rows)
    for row_idx, row in enumerate(candidate_rows, start=1):
        key = (row["question"], row["answer_text"])
        trace = cache[key]
        features = waveform_features(trace["entropies"], trace["chosen_probs"])
        trace_rows.append(
            {
                "seed": row["seed"],
                "question_id": row["question_id"],
                "sample_index": row["sample_index"],
                "cluster_id": row["cluster_id"],
                "tokens": trace["tokens"],
                "entropies": trace["entropies"],
                "chosen_probs": trace["chosen_probs"],
            }
        )
        feature_row = {
            "seed": row["seed"],
            "question_id": row["question_id"],
            "question_index": row["question_index"],
            "question": row["question"],
            "sample_index": row["sample_index"],
            "cluster_id": row["cluster_id"],
            "strict_correct": row["strict_correct"],
            "sample0_strict_correct": row["sample0_strict_correct"],
            "answer_preview": row["answer_preview"],
            **{key: value for key, value in features.items() if not isinstance(value, list)},
            "wf_entropy_curve": features["wf_entropy_curve"],
            "wf_prob_curve": features["wf_prob_curve"],
        }
        feature_rows.append(feature_row)
        if row_idx % 200 == 0:
            print(json.dumps({"processed": row_idx, "total": total, "unique_traces": len(cache)}, ensure_ascii=False))
    write_jsonl(output_dir / "candidate_entropy_waveforms.jsonl", trace_rows)
    return trace_rows, feature_rows


def build_basin_waveforms(feature_rows: list[dict[str, Any]], basin_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features_by_basin: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        features_by_basin[(int(row["seed"]), row["question_id"], int(row["cluster_id"]))].append(row)
    basin_output: list[dict[str, Any]] = []
    for basin in basin_rows:
        key = (int(basin["seed"]), basin["question_id"], int(basin["cluster_id"]))
        rows = features_by_basin.get(key, [])
        output = dict(basin)
        output["representative_strict_correct"] = safe_float(output.get("basin_correct_rate", 0.0))
        if rows:
            for feature in [name for name in rows[0] if name.startswith("wf_") and not name.endswith("_curve")]:
                output[f"{feature}_mean"] = mean([safe_float(row.get(feature, 0.0)) for row in rows])
                output[f"{feature}_std"] = stdev([safe_float(row.get(feature, 0.0)) for row in rows])
            entropy_curves = [row["wf_entropy_curve"] for row in rows]
            prob_curves = [row["wf_prob_curve"] for row in rows]
            output["wf_basin_entropy_curve"] = [mean([curve[idx] for curve in entropy_curves]) for idx in range(20)]
            output["wf_basin_prob_curve"] = [mean([curve[idx] for curve in prob_curves]) for idx in range(20)]
            output["wf_basin_entropy_curve_var"] = curve_variance(entropy_curves)
            output["wf_basin_prob_curve_var"] = curve_variance(prob_curves)
        basin_output.append(output)
    return basin_output


def feature_summary(rows: list[dict[str, Any]], pos_filter: Any, neg_filter: Any, features: list[str], comparison: str) -> list[dict[str, Any]]:
    pos_rows = [row for row in rows if pos_filter(row)]
    neg_rows = [row for row in rows if neg_filter(row)]
    summary = []
    for feature in features:
        pos = [safe_float(row.get(feature, 0.0)) for row in pos_rows]
        neg = [safe_float(row.get(feature, 0.0)) for row in neg_rows]
        summary.append(
            {
                "comparison": comparison,
                "feature": feature,
                "positive_count": len(pos),
                "negative_count": len(neg),
                "positive_mean": mean(pos),
                "negative_mean": mean(neg),
                "mean_diff": mean(pos) - mean(neg),
                "cohen_d": cohen_d(pos, neg),
                "auc_positive_high": auc_score(pos, neg),
            }
        )
    return summary


def grouped_cv_ablation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[(int(row["seed"]), row["question_id"])].append(row)
    keys = sorted(by_pair)
    sets = {
        "aggregate_only": AGGREGATE_FEATURES,
        "waveform_only": WAVEFORM_FEATURES,
        "aggregate_plus_waveform": AGGREGATE_FEATURES + WAVEFORM_FEATURES,
    }
    fold_rows = []
    for fold_idx, (train_qids, test_qids) in enumerate(question_folds(keys)):
        train_rows = [row for (seed, qid), group in by_pair.items() if qid in train_qids for row in group]
        test_groups = [group for (seed, qid), group in by_pair.items() if qid in test_qids]
        for name, features in sets.items():
            model = LogisticModel(features)
            model.fit(train_rows, "representative_strict_correct", epochs=500, lr=0.08, l2=0.02)
            selected = [max(group, key=model.predict) for group in test_groups]
            sample0s = [next(row for row in group if safe_float(row["contains_sample0"]) > 0) for group in test_groups]
            deltas = [safe_float(sel["representative_strict_correct"]) - safe_float(s0["representative_strict_correct"]) for sel, s0 in zip(selected, sample0s)]
            fold_rows.append(
                {
                    "split": f"fold_{fold_idx}",
                    "method": name,
                    "pairs": len(test_groups),
                    "strict_correct_rate": mean([safe_float(row["representative_strict_correct"]) for row in selected]),
                    "sample0_strict_correct_rate": mean([safe_float(row["representative_strict_correct"]) for row in sample0s]),
                    "delta_vs_sample0": mean(deltas),
                    "improved_count": sum(1 for value in deltas if value > 0),
                    "damaged_count": sum(1 for value in deltas if value < 0),
                    "net_gain_count": sum(1 for value in deltas if value > 0) - sum(1 for value in deltas if value < 0),
                }
            )
    summary = []
    for method in sets:
        items = [row for row in fold_rows if row["method"] == method]
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
            }
        )
    return summary + fold_rows


def make_plots(output_dir: Path, basin_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = ensure_dir(output_dir / "plots")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
    regimes = ["stable_correct", "stable_hallucination", "rescue", "damage", "unstable_wrong"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for regime in regimes:
        rows = [row for row in basin_rows if row.get("basin_regime") == regime and isinstance(row.get("wf_basin_entropy_curve"), list)]
        if not rows:
            continue
        curve = [mean([row["wf_basin_entropy_curve"][idx] for row in rows]) for idx in range(20)]
        ax.plot([idx / 19 for idx in range(20)], curve, label=f"{regime} (n={len(rows)})")
    ax.set_xlabel("Normalized answer position")
    ax.set_ylabel("Shannon entropy")
    ax.set_title("Average Entropy Waveforms by Basin Regime")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "01_regime_entropy_waveforms.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for regime, color in [("stable_correct", "#2ca02c"), ("stable_hallucination", "#d62728")]:
        rows = [row for row in basin_rows if row.get("basin_regime") == regime and isinstance(row.get("wf_basin_entropy_curve"), list)]
        if not rows:
            continue
        xs = [idx / 19 for idx in range(20)]
        means = [mean([row["wf_basin_entropy_curve"][idx] for row in rows]) for idx in range(20)]
        sds = [stdev([row["wf_basin_entropy_curve"][idx] for row in rows]) for idx in range(20)]
        ax.plot(xs, means, label=f"{regime} mean", color=color)
        ax.fill_between(xs, [m - 0.35 * s for m, s in zip(means, sds)], [m + 0.35 * s for m, s in zip(means, sds)], color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("Normalized answer position")
    ax.set_ylabel("Shannon entropy")
    ax.set_title("Stable Correct vs Stable Hallucination")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "02_stable_correct_vs_hallucination_bands.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for regime, alpha in [("stable_correct", 0.65), ("stable_hallucination", 0.65), ("rescue", 0.5), ("damage", 0.5)]:
        values = [safe_float(row.get("wf_entropy_max_pos_norm_mean")) for row in basin_rows if row.get("basin_regime") == regime]
        if values:
            ax.hist(values, bins=12, alpha=alpha, label=regime, density=True)
    ax.set_xlabel("Position of maximum entropy spike")
    ax.set_ylabel("Density")
    ax.set_title("Spike Position Histograms")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "03_spike_position_histograms.png")
    plt.close(fig)

    top = sorted(summary_rows, key=lambda row: abs(safe_float(row["cohen_d"])), reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [row["feature"] for row in top]
    values = [safe_float(row["cohen_d"]) for row in top]
    ax.barh(labels[::-1], values[::-1])
    ax.axvline(0, color="#666666", lw=0.8)
    ax.set_xlabel("Cohen d")
    ax.set_title("Top Waveform Separations")
    fig.tight_layout()
    fig.savefig(plot_dir / "04_waveform_feature_separation.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    examples = [
        ("stable_correct", max([row for row in basin_rows if row.get("basin_regime") == "stable_correct" and isinstance(row.get("wf_basin_entropy_curve"), list)], key=lambda row: safe_float(row.get("cluster_weight_mass")), default=None)),
        ("stable_hallucination", max([row for row in basin_rows if row.get("basin_regime") == "stable_hallucination" and isinstance(row.get("wf_basin_entropy_curve"), list)], key=lambda row: safe_float(row.get("wf_entropy_max_mean")), default=None)),
        ("rescue", max([row for row in basin_rows if safe_float(row.get("is_rescue_basin")) > 0 and isinstance(row.get("wf_basin_entropy_curve"), list)], key=lambda row: safe_float(row.get("wf_prob_min_mean")), default=None)),
        ("damage", max([row for row in basin_rows if safe_float(row.get("is_damage_basin")) > 0 and isinstance(row.get("wf_basin_entropy_curve"), list)], key=lambda row: safe_float(row.get("wf_entropy_max_mean")), default=None)),
    ]
    xs = [idx / 19 for idx in range(20)]
    for ax, (label, row) in zip(axes.ravel(), examples):
        if row is None:
            continue
        ax.plot(xs, row["wf_basin_entropy_curve"], color="#1f77b4")
        ax.set_title(f"{label}: {str(row.get('answer_preview', ''))[:48]}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Entropy")
    fig.suptitle("Representative Entropy Waveform Micrographs", y=0.98)
    fig.tight_layout()
    fig.savefig(plot_dir / "05_representative_waveform_micrographs.png")
    plt.close(fig)


def write_report(output_dir: Path, feature_summary_rows: list[dict[str, Any]], cv_rows: list[dict[str, Any]]) -> None:
    stable = [row for row in feature_summary_rows if row["comparison"] == "stable_correct_vs_stable_hallucination"]
    top = sorted(stable, key=lambda row: abs(safe_float(row["cohen_d"])), reverse=True)[:12]
    cv_summary = [row for row in cv_rows if row["split"] == "question_grouped_cv"]
    aggregate = next((row for row in cv_summary if row["method"] == "aggregate_only"), {})
    waveform = next((row for row in cv_summary if row["method"] == "waveform_only"), {})
    combined = next((row for row in cv_summary if row["method"] == "aggregate_plus_waveform"), {})
    lines = [
        "# Entropy Waveform Basin Analysis",
        "",
        "## 核心问题",
        "",
        "这轮实验重建了每个候选答案的逐 token Shannon entropy waveform，并检查正确 basin 与幻觉 basin 是否存在可见和可学习的轨迹差异。",
        "",
        "## 主要发现",
        "",
        "1. `stable_hallucination` 不是完全“平滑自信”的：在逐 token 轨迹上，它比 `stable_correct` 有更高的平均熵、更高的最大熵、更大的波动，以及更低的最小 chosen-token probability。这说明宏观 basin 很稳定，不代表生成过程内部每一步都稳定。",
        "2. waveform 单独有信号，但还不够强：它能带来小幅正收益，不过 damage 也增加，说明它更适合作为风险特征或 early-warning veto，而不是单独替代现有 basin geometry verifier。",
        "3. aggregate + waveform 在当前简单线性选择器中没有超过 aggregate-only，说明下一步若集成 waveform，应该偏向 damage-risk veto / prefix continue-value，而不是直接把全部 waveform 特征塞进最终 basin selector。",
        "",
        "## Stable Correct vs Stable Hallucination: Top Waveform Features",
        "",
        "| Feature | Correct Mean | Hallucination Mean | Cohen d | AUC Correct High |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in top:
        lines.append(
            f"| `{row['feature']}` | `{safe_float(row['positive_mean']):.4f}` | `{safe_float(row['negative_mean']):.4f}` | `{safe_float(row['cohen_d']):.3f}` | `{safe_float(row['auc_positive_high']):.3f}` |"
        )
    lines.extend(["", "## Controller Ablation", "", "| Method | Strict | Delta | Improved | Damaged | Net |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in cv_summary:
        lines.append(
            f"| `{row['method']}` | `{safe_float(row['strict_correct_rate']):.2%}` | `{safe_float(row['delta_vs_sample0']):.2%}` | `{int(row['improved_count'])}` | `{int(row['damaged_count'])}` | `{int(row['net_gain_count'])}` |"
        )
    lines.extend(
        [
            "",
            "## Controller Recommendation",
            "",
            f"- `aggregate_only` 当前仍是最强选择器：delta `{safe_float(aggregate.get('delta_vs_sample0')):.2%}`，damage `{int(aggregate.get('damaged_count', 0))}`。",
            f"- `waveform_only` 有弱正信号：delta `{safe_float(waveform.get('delta_vs_sample0')):.2%}`，但 damage `{int(waveform.get('damaged_count', 0))}`，不适合作为独立 selector。",
            f"- `aggregate_plus_waveform` delta `{safe_float(combined.get('delta_vs_sample0')):.2%}`，与 aggregate-only 接近但略低；当前证据支持把 waveform 作为早期风险探针，而不是主 verifier。",
            "",
            "论文表述上，这个实验可以作为机制分析：basin-level entropy 的“两面性”之外，token-level waveform 揭示了 stable hallucination 的微观犹豫峰值。工程上，建议下一步只把 `wf_entropy_max`、`wf_entropy_std`、`wf_prob_min`、`late-minus-early` 这类高分离特征接入 prefix continue-value / damage veto。",
        ]
    )
    (output_dir / "entropy_waveform_findings_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}")
    candidate_rows = read_csv(Path(args.candidate_run) / "candidate_features.csv")
    if args.limit:
        candidate_rows = candidate_rows[: args.limit]
    basin_rows = read_csv(Path(args.entropy_run) / "entropy_basin_table.csv")
    config = load_config(Path(args.phase2a_run))
    trace_rows, feature_rows = build_waveforms(candidate_rows, config, output_dir, args.batch_size)
    write_csv(output_dir / "candidate_waveform_features.csv", feature_rows)
    basin_waveform_rows = build_basin_waveforms(feature_rows, basin_rows)
    write_csv(output_dir / "basin_waveform_table.csv", basin_waveform_rows)
    summary_rows = []
    summary_rows.extend(
        feature_summary(
            basin_waveform_rows,
            lambda row: row.get("basin_regime") == "stable_correct",
            lambda row: row.get("basin_regime") == "stable_hallucination",
            WAVEFORM_FEATURES,
            "stable_correct_vs_stable_hallucination",
        )
    )
    summary_rows.extend(
        feature_summary(
            basin_waveform_rows,
            lambda row: safe_float(row.get("is_rescue_basin")) > 0,
            lambda row: safe_float(row.get("is_damage_basin")) > 0,
            WAVEFORM_FEATURES,
            "rescue_vs_damage",
        )
    )
    write_csv(output_dir / "waveform_feature_summary.csv", summary_rows)
    cv_rows = grouped_cv_ablation(basin_waveform_rows)
    write_csv(output_dir / "waveform_ablation_cv.csv", cv_rows)
    make_plots(output_dir, basin_waveform_rows, summary_rows)
    write_report(output_dir, summary_rows, cv_rows)
    write_json(
        output_dir / "run_metadata.json",
        {
            "candidate_count": len(candidate_rows),
            "trace_count": len(trace_rows),
            "basin_count": len(basin_waveform_rows),
            "model_dir": config["model_dir"],
            "system_prompt": config["evaluation"]["system_prompt"],
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "candidate_count": len(candidate_rows), "basin_count": len(basin_waveform_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
