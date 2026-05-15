#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_CANDIDATE_RUN = Path("/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25")
DEFAULT_WAVEFORM_RUN = Path("/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260430_021911_entropy_waveform_analysis_v2")
DEFAULT_OUTPUT_ROOT = Path("/zhutingqi/song/Plan_gpt55/basin_theory_analysis/runs")

PALETTE = {
    "blue": "#069DFF",
    "gray": "#808080",
    "green": "#A4E048",
    "black": "#010101",
    "light_red": "#FFA0A0",
}

BANNED_AS_FEATURE = {
    "strict_correct",
    "sample0_strict_correct",
    "correct_count",
    "wrong_count",
    "basin_correct_rate",
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
    "representative_strict_correct",
}

LABEL_COLUMNS = sorted(BANNED_AS_FEATURE | {"seed", "question_id", "question", "question_index", "cluster_id", "answer_preview"})

FEATURE_FAMILIES = {
    "stability": [
        "cluster_size",
        "cluster_weight_mass",
        "stable_score",
        "low_entropy_score",
        "fragmentation_entropy",
        "normalized_fragmentation_entropy",
        "semantic_entropy_weighted_set",
        "semantic_clusters_set",
    ],
    "geometry": [
        "centroid_entropy_z",
        "centroid_max_entropy_z",
        "centroid_logprob_z",
        "centroid_len_z",
        "distance_to_sample0_entropy_logprob",
        "top2_weight_margin",
        "top2_logprob_margin",
        "top2_low_entropy_margin",
        "low_entropy_basin_rank",
        "logprob_basin_rank",
        "weight_basin_rank",
        "stable_basin_rank",
    ],
    "lexical_shape": [
        "within_basin_lexical_entropy",
        "within_basin_jaccard",
        "internal_token_entropy_std",
        "token_mean_entropy_std",
        "logprob_avg_std",
        "lexical_pca_anisotropy",
        "lexical_effective_dim",
        "lexical_radius",
    ],
    "waveform_functional": [
        "wf_entropy_auc_mean",
        "wf_entropy_max_mean",
        "wf_entropy_std_mean",
        "wf_entropy_roughness",
        "wf_entropy_late_minus_early_mean",
        "wf_entropy_spike_count_mean",
        "wf_entropy_spike_frac_mean",
        "wf_entropy_max_pos_norm_mean",
        "wf_basin_entropy_curve_var",
        "wf_prob_min_mean",
        "wf_prob_drop",
        "wf_basin_prob_curve_var",
        "wf_local_instability_index",
    ],
    "sample0_delta": [
        "delta_cluster_weight_mass_vs_sample0",
        "delta_stable_score_vs_sample0",
        "delta_low_entropy_score_vs_sample0",
        "delta_centroid_entropy_z_vs_sample0",
        "delta_centroid_logprob_z_vs_sample0",
        "delta_wf_entropy_max_mean_vs_sample0",
        "delta_wf_entropy_roughness_vs_sample0",
        "delta_wf_prob_min_mean_vs_sample0",
        "delta_distance_to_sample0_entropy_logprob_vs_sample0",
    ],
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value: Any) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def cohen_d(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.0
    pooled = math.sqrt(((len(pos) - 1) * stdev(pos) ** 2 + (len(neg) - 1) * stdev(neg) ** 2) / max(1, len(pos) + len(neg) - 2))
    return (mean(pos) - mean(neg)) / pooled if pooled else 0.0


def auc_score(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def parse_curve(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    try:
        return [float(item) for item in ast.literal_eval(str(value))]
    except Exception:
        return []


def curve_roughness(curve: list[float]) -> float:
    if len(curve) < 3:
        return 0.0
    diffs = [curve[idx + 1] - curve[idx] for idx in range(len(curve) - 1)]
    return mean([value * value for value in diffs])


def curve_auc(curve: list[float]) -> float:
    return mean(curve)


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 2]


def lexical_pca_features(texts: list[str]) -> dict[str, float]:
    docs = [tokenize(text) for text in texts]
    vocab_counts = Counter(tok for doc in docs for tok in doc)
    vocab = [tok for tok, count in vocab_counts.most_common(80) if count >= 1]
    if len(docs) < 2 or len(vocab) < 2:
        return {"lexical_pca_anisotropy": 0.0, "lexical_effective_dim": 0.0, "lexical_radius": 0.0}
    index = {tok: idx for idx, tok in enumerate(vocab)}
    mat = np.zeros((len(docs), len(vocab)), dtype=np.float64)
    df = np.zeros(len(vocab), dtype=np.float64)
    for row_idx, doc in enumerate(docs):
        counts = Counter(tok for tok in doc if tok in index)
        for tok, count in counts.items():
            mat[row_idx, index[tok]] = count
        for tok in counts:
            df[index[tok]] += 1
    idf = np.log((1 + len(docs)) / (1 + df)) + 1
    mat *= idf
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.maximum(norms, 1e-9)
    centered = mat - mat.mean(axis=0, keepdims=True)
    try:
        _u, svals, _vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"lexical_pca_anisotropy": 0.0, "lexical_effective_dim": 0.0, "lexical_radius": 0.0}
    eig = svals**2
    total = float(eig.sum())
    anisotropy = float(eig[0] / total) if total > 0 else 0.0
    effective_dim = float((total * total) / np.maximum((eig * eig).sum(), 1e-9)) if total > 0 else 0.0
    radius = float(np.mean(np.linalg.norm(centered, axis=1)))
    return {"lexical_pca_anisotropy": anisotropy, "lexical_effective_dim": effective_dim, "lexical_radius": radius}


def build_theory_table(candidate_rows: list[dict[str, Any]], basin_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates_by_basin: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidates_by_basin[(int(row["seed"]), str(row["question_id"]), int(row["cluster_id"]))].append(row)

    by_pair: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    output_rows: list[dict[str, Any]] = []
    for row in basin_rows:
        item = dict(row)
        key = (int(item["seed"]), str(item["question_id"]), int(item["cluster_id"]))
        cand_rows = candidates_by_basin.get(key, [])
        item.update(lexical_pca_features([str(cand.get("answer_text", "")) for cand in cand_rows]))

        entropy_curve = parse_curve(item.get("wf_basin_entropy_curve", ""))
        prob_curve = parse_curve(item.get("wf_basin_prob_curve", ""))
        item["wf_entropy_roughness"] = curve_roughness(entropy_curve)
        item["wf_entropy_auc_recomputed"] = curve_auc(entropy_curve)
        item["wf_prob_drop"] = (mean(prob_curve[:5]) - mean(prob_curve[-5:])) if len(prob_curve) >= 10 else 0.0
        item["wf_local_instability_index"] = safe_float(item.get("wf_entropy_max_mean")) / (safe_float(item.get("wf_entropy_mean_mean")) + 1e-6)
        output_rows.append(item)
        by_pair[(int(item["seed"]), str(item["question_id"]))].append(item)

    delta_features = [
        "cluster_weight_mass",
        "stable_score",
        "low_entropy_score",
        "centroid_entropy_z",
        "centroid_logprob_z",
        "wf_entropy_max_mean",
        "wf_entropy_roughness",
        "wf_prob_min_mean",
        "distance_to_sample0_entropy_logprob",
    ]
    for group in by_pair.values():
        sample0 = next((row for row in group if safe_float(row.get("contains_sample0")) > 0), None)
        if sample0 is None:
            continue
        for row in group:
            for feature in delta_features:
                row[f"delta_{feature}_vs_sample0"] = safe_float(row.get(feature)) - safe_float(sample0.get(feature))
    return output_rows


def feature_audit(all_features: list[str]) -> list[dict[str, Any]]:
    rows = []
    used = set(feature for features in FEATURE_FAMILIES.values() for feature in features)
    available = set(all_features)
    for feature in sorted(all_features):
        rows.append(
            {
                "feature": feature,
                "used_as_predictor": float(feature in used),
                "is_banned_as_feature": float(feature in BANNED_AS_FEATURE),
                "status": "ERROR_used_banned" if feature in used and feature in BANNED_AS_FEATURE else "ok",
            }
        )
    for feature in sorted(used - available):
        rows.append(
            {
                "feature": feature,
                "used_as_predictor": 1.0,
                "is_banned_as_feature": float(feature in BANNED_AS_FEATURE),
                "status": "ERROR_missing_predictor",
            }
        )
    errors = [row for row in rows if row["status"] != "ok"]
    if errors:
        raise RuntimeError(f"Banned features were included as predictors: {errors}")
    return rows


def feature_family(feature: str) -> str:
    for family, features in FEATURE_FAMILIES.items():
        if feature in features:
            return family
    return "other"


def all_predictor_features() -> list[str]:
    return [feature for features in FEATURE_FAMILIES.values() for feature in features]


def compare_groups(rows: list[dict[str, Any]], pos_filter: Callable[[dict[str, Any]], bool], neg_filter: Callable[[dict[str, Any]], bool], comparison: str) -> list[dict[str, Any]]:
    pos_rows = [row for row in rows if pos_filter(row)]
    neg_rows = [row for row in rows if neg_filter(row)]
    out = []
    for feature in all_predictor_features():
        pos = [safe_float(row.get(feature)) for row in pos_rows]
        neg = [safe_float(row.get(feature)) for row in neg_rows]
        out.append(
            {
                "comparison": comparison,
                "feature": feature,
                "family": feature_family(feature),
                "positive_count": len(pos),
                "negative_count": len(neg),
                "positive_mean": mean(pos),
                "negative_mean": mean(neg),
                "mean_diff": mean(pos) - mean(neg),
                "cohen_d": cohen_d(pos, neg),
                "auc_positive_high": auc_score(pos, neg),
            }
        )
    return out


class LogisticModel:
    def __init__(self, features: list[str]):
        self.features = features
        self.means = {feature: 0.0 for feature in features}
        self.stds = {feature: 1.0 for feature in features}
        self.weights = {feature: 0.0 for feature in features}
        self.bias = 0.0

    def z(self, row: dict[str, Any], feature: str) -> float:
        return (safe_float(row.get(feature)) - self.means[feature]) / self.stds[feature]

    def predict(self, row: dict[str, Any]) -> float:
        value = self.bias + sum(self.weights[feature] * self.z(row, feature) for feature in self.features)
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))

    def fit(self, rows: list[dict[str, Any]], label: str, epochs: int = 500, lr: float = 0.06, l2: float = 0.02) -> None:
        for feature in self.features:
            values = [safe_float(row.get(feature)) for row in rows]
            self.means[feature] = mean(values)
            self.stds[feature] = stdev(values) or 1.0
        pos = sum(1 for row in rows if safe_float(row[label]) > 0)
        neg = max(1, len(rows) - pos)
        pos_w = len(rows) / max(1, 2 * pos)
        neg_w = len(rows) / max(1, 2 * neg)
        for _ in range(epochs):
            grad = {feature: 0.0 for feature in self.features}
            grad_b = 0.0
            for row in rows:
                y = 1.0 if safe_float(row[label]) > 0 else 0.0
                p = self.predict(row)
                err = (p - y) * (pos_w if y else neg_w)
                for feature in self.features:
                    grad[feature] += err * self.z(row, feature)
                grad_b += err
            scale = 1.0 / max(1, len(rows))
            for feature in self.features:
                self.weights[feature] -= lr * (grad[feature] * scale + l2 * self.weights[feature])
            self.bias -= lr * grad_b * scale


def question_folds(rows: list[dict[str, Any]], k: int = 5) -> list[tuple[set[str], set[str]]]:
    qids = sorted({str(row["question_id"]) for row in rows})
    folds = []
    for idx in range(k):
        test = {qid for pos, qid in enumerate(qids) if pos % k == idx}
        folds.append((set(qids) - test, test))
    return folds


def classification_metrics(labels: list[float], probs: list[float]) -> dict[str, float]:
    if not labels:
        return {"auc": 0.5, "balanced_acc": 0.0, "accuracy": 0.0}
    preds = [1.0 if prob >= 0.5 else 0.0 for prob in probs]
    tp = sum(1 for y, p in zip(labels, preds) if y > 0 and p > 0)
    tn = sum(1 for y, p in zip(labels, preds) if y <= 0 and p <= 0)
    fp = sum(1 for y, p in zip(labels, preds) if y <= 0 and p > 0)
    fn = sum(1 for y, p in zip(labels, preds) if y > 0 and p <= 0)
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    pos = [prob for y, prob in zip(labels, probs) if y > 0]
    neg = [prob for y, prob in zip(labels, probs) if y <= 0]
    return {"auc": auc_score(pos, neg), "balanced_acc": 0.5 * (tpr + tnr), "accuracy": (tp + tn) / max(1, len(labels))}


def run_diagnostic_cv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = {
        "stable_correct_vs_hallucination": (
            lambda row: safe_float(row.get("is_stable_correct_basin")) > 0 or safe_float(row.get("is_stable_hallucination_basin")) > 0,
            lambda row: float(safe_float(row.get("is_stable_correct_basin")) > 0),
        ),
        "pure_correct_vs_wrong": (
            lambda row: safe_float(row.get("is_pure_correct")) > 0 or safe_float(row.get("is_pure_wrong")) > 0,
            lambda row: float(safe_float(row.get("is_pure_correct")) > 0),
        ),
        "rescue_vs_damage": (
            lambda row: safe_float(row.get("is_rescue_basin")) > 0 or safe_float(row.get("is_damage_basin")) > 0,
            lambda row: float(safe_float(row.get("is_rescue_basin")) > 0),
        ),
    }
    feature_sets = {**FEATURE_FAMILIES, "all_noleak": all_predictor_features()}
    out: list[dict[str, Any]] = []
    for task_name, (row_filter, label_func) in tasks.items():
        task_rows = [dict(row, diagnostic_label=label_func(row)) for row in rows if row_filter(row)]
        if len({safe_float(row["diagnostic_label"]) for row in task_rows}) < 2:
            continue
        for fold_idx, (train_qids, test_qids) in enumerate(question_folds(task_rows)):
            train = [row for row in task_rows if str(row["question_id"]) in train_qids]
            test = [row for row in task_rows if str(row["question_id"]) in test_qids]
            for set_name, features in feature_sets.items():
                model = LogisticModel(features)
                model.fit(train, "diagnostic_label")
                probs = [model.predict(row) for row in test]
                labels = [safe_float(row["diagnostic_label"]) for row in test]
                metrics = classification_metrics(labels, probs)
                out.append(
                    {
                        "split": f"fold_{fold_idx}",
                        "task": task_name,
                        "feature_set": set_name,
                        "rows": len(test),
                        "positive_rate": mean(labels),
                        **metrics,
                    }
                )
    summary = []
    for key, group in defaultdict(list, {k: [row for row in out if (row["task"], row["feature_set"]) == k] for k in sorted({(row["task"], row["feature_set"]) for row in out})}).items():
        task, feature_set = key
        summary.append(
            {
                "split": "question_grouped_cv",
                "task": task,
                "feature_set": feature_set,
                "rows": sum(int(row["rows"]) for row in group),
                "positive_rate": mean([safe_float(row["positive_rate"]) for row in group]),
                "auc": mean([safe_float(row["auc"]) for row in group]),
                "balanced_acc": mean([safe_float(row["balanced_acc"]) for row in group]),
                "accuracy": mean([safe_float(row["accuracy"]) for row in group]),
            }
        )
    return summary + out


def summarize_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for comparison, pos_filter, neg_filter in [
        ("rescue_vs_damage", lambda r: safe_float(r.get("is_rescue_basin")) > 0, lambda r: safe_float(r.get("is_damage_basin")) > 0),
        ("non_sample0_rescue_vs_other_alt", lambda r: safe_float(r.get("is_rescue_basin")) > 0, lambda r: safe_float(r.get("contains_sample0")) <= 0 and safe_float(r.get("is_rescue_basin")) <= 0),
    ]:
        out.extend(compare_groups(rows, pos_filter, neg_filter, comparison))
    return out


def setup_plotting() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def make_plots(output_dir: Path, rows: list[dict[str, Any]], effect_rows: list[dict[str, Any]], cv_rows: list[dict[str, Any]]) -> None:
    plt = setup_plotting()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    top = sorted(effect_rows, key=lambda row: abs(safe_float(row["cohen_d"])), reverse=True)[:28]
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    labels = [f"{row['comparison']}\n{row['feature']}" for row in top[::-1]]
    colors = [PALETTE["green"] if safe_float(row["cohen_d"]) > 0 else PALETTE["blue"] for row in top[::-1]]
    ax.barh(labels, [safe_float(row["cohen_d"]) for row in top[::-1]], color=colors, alpha=0.88)
    ax.axvline(0, color=PALETTE["black"], lw=0.8)
    ax.set_title("Top No-Leak Theory Feature Effect Sizes")
    ax.set_xlabel("Cohen d")
    fig.tight_layout()
    fig.savefig(plot_dir / "01_theory_feature_effect_sizes.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    regime_colors = {
        "stable_correct": PALETTE["green"],
        "stable_hallucination": PALETTE["light_red"],
        "rescue": PALETTE["blue"],
        "damage": "#ffb8b8",
        "unstable_wrong": PALETTE["gray"],
        "mixed": "#b6b6b6",
    }
    for regime, group in sorted(group_by(rows, "basin_regime").items()):
        ax.scatter(
            [safe_float(row.get("stable_score")) for row in group],
            [safe_float(row.get("wf_local_instability_index")) for row in group],
            s=[22 + 70 * safe_float(row.get("cluster_weight_mass")) for row in group],
            color=regime_colors.get(regime, "#d0d0d0"),
            alpha=0.62,
            label=f"{regime} (n={len(group)})",
            edgecolors="none",
        )
    ax.set_xlabel("Basin stability score")
    ax.set_ylabel("Waveform local instability index")
    ax.set_title("Stability vs Token-Level Instability")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(plot_dir / "02_stability_vs_waveform_phase_map.png")
    plt.close(fig)

    radar_features = ["cluster_weight_mass", "stable_score", "centroid_logprob_z", "wf_entropy_max_mean", "wf_entropy_roughness", "wf_prob_min_mean"]
    sc = [row for row in rows if safe_float(row.get("is_stable_correct_basin")) > 0]
    sh = [row for row in rows if safe_float(row.get("is_stable_hallucination_basin")) > 0]
    values = []
    for group in [sc, sh]:
        values.append([mean([safe_float(row.get(feature)) for row in group]) for feature in radar_features])
    mins = [min(values[0][idx], values[1][idx]) for idx in range(len(radar_features))]
    maxs = [max(values[0][idx], values[1][idx]) for idx in range(len(radar_features))]
    norm = [[(vals[idx] - mins[idx]) / (maxs[idx] - mins[idx] + 1e-9) for idx in range(len(radar_features))] for vals in values]
    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    fig = plt.figure(figsize=(6.5, 6.0))
    ax = fig.add_subplot(111, polar=True)
    for vals, color, label in [(norm[0], PALETTE["green"], "stable correct"), (norm[1], PALETTE["light_red"], "stable hallucination")]:
        loop_vals = vals + vals[:1]
        loop_angles = angles + angles[:1]
        ax.plot(loop_angles, loop_vals, color=color, linewidth=1.8, label=label)
        ax.fill(loop_angles, loop_vals, color=color, alpha=0.18)
    ax.set_xticks(angles, radar_features, fontsize=7)
    ax.set_yticklabels([])
    ax.set_title("Stable Correct vs Stable Hallucination Profile")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "03_stable_correct_vs_hallucination_radar.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for label, filt, color in [
        ("rescue", lambda r: safe_float(r.get("is_rescue_basin")) > 0, PALETTE["green"]),
        ("damage", lambda r: safe_float(r.get("is_damage_basin")) > 0, PALETTE["light_red"]),
    ]:
        group = [row for row in rows if filt(row)]
        ax.scatter(
            [safe_float(row.get("delta_stable_score_vs_sample0")) for row in group],
            [safe_float(row.get("delta_wf_entropy_roughness_vs_sample0")) for row in group],
            s=[26 + 55 * safe_float(row.get("cluster_weight_mass")) for row in group],
            color=color,
            alpha=0.68,
            label=f"{label} (n={len(group)})",
            edgecolors="none",
        )
    ax.axhline(0, color=PALETTE["gray"], lw=0.8)
    ax.axvline(0, color=PALETTE["gray"], lw=0.8)
    ax.set_xlabel("Delta stability vs sample0")
    ax.set_ylabel("Delta waveform roughness vs sample0")
    ax.set_title("Rescue/Damage Relative to sample0 Basin")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "04_rescue_damage_delta_map.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    groups = [
        ("stable_correct", [row for row in rows if safe_float(row.get("is_stable_correct_basin")) > 0], PALETTE["green"]),
        ("stable_hallucination", [row for row in rows if safe_float(row.get("is_stable_hallucination_basin")) > 0], PALETTE["light_red"]),
        ("pure_wrong", [row for row in rows if safe_float(row.get("is_pure_wrong")) > 0], PALETTE["gray"]),
    ]
    wf_features = ["wf_entropy_auc_mean", "wf_entropy_max_mean", "wf_entropy_roughness", "wf_entropy_late_minus_early_mean", "wf_prob_min_mean", "wf_local_instability_index"]
    for ax, feature in zip(axes.ravel(), wf_features):
        data = [[safe_float(row.get(feature)) for row in group] for _name, group, _color in groups]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for body, (_name, _group, color) in zip(parts["bodies"], groups):
            body.set_facecolor(color)
            body.set_alpha(0.38)
        parts["cmeans"].set_color(PALETTE["black"])
        ax.set_xticks([1, 2, 3], ["stable\ncorrect", "stable\nhallu.", "pure\nwrong"])
        ax.set_title(feature)
    fig.suptitle("Waveform Functional Distributions", y=0.99)
    fig.tight_layout()
    fig.savefig(plot_dir / "05_waveform_functionals_grid.png")
    plt.close(fig)

    corr_features = []
    for feature in all_predictor_features():
        values = [safe_float(row.get(feature)) for row in rows]
        if stdev(values) > 1e-12:
            corr_features.append(feature)
    matrix = np.array([[safe_float(row.get(feature)) for feature in corr_features] for row in rows], dtype=np.float64)
    if matrix.shape[0] > 1:
        corr = np.corrcoef(matrix, rowvar=False)
        corr = np.nan_to_num(corr)
    else:
        corr = np.zeros((len(corr_features), len(corr_features)))
    fig, ax = plt.subplots(figsize=(11.5, 10.0))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr_features)), corr_features, rotation=90, fontsize=5)
    ax.set_yticks(range(len(corr_features)), corr_features, fontsize=5)
    ax.set_title("No-Leak Feature Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.035)
    fig.tight_layout()
    fig.savefig(plot_dir / "06_feature_family_correlation_heatmap.png")
    plt.close(fig)

    cv_summary = [row for row in cv_rows if row["split"] == "question_grouped_cv"]
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    labels = [f"{row['task']}\n{row['feature_set']}" for row in cv_summary]
    ax.barh(labels[::-1], [safe_float(row["auc"]) for row in cv_summary[::-1]], color=PALETTE["blue"], alpha=0.82)
    ax.axvline(0.5, color=PALETTE["gray"], lw=0.9, linestyle="--")
    ax.set_xlabel("Question-heldout diagnostic AUC")
    ax.set_title("Diagnostic Separability by No-Leak Feature Family")
    fig.tight_layout()
    fig.savefig(plot_dir / "07_diagnostic_classifier_auc.png")
    plt.close(fig)

    make_dashboard(plt, output_dir, plot_dir)


def make_dashboard(plt: Any, output_dir: Path, plot_dir: Path) -> None:
    import matplotlib.image as mpimg

    panels = [
        ("A. Effect Sizes", plot_dir / "01_theory_feature_effect_sizes.png"),
        ("B. Stability vs Waveform", plot_dir / "02_stability_vs_waveform_phase_map.png"),
        ("C. Stable Basin Radar", plot_dir / "03_stable_correct_vs_hallucination_radar.png"),
        ("D. Rescue/Damage Deltas", plot_dir / "04_rescue_damage_delta_map.png"),
        ("E. Waveform Functionals", plot_dir / "05_waveform_functionals_grid.png"),
        ("F. Feature Correlations", plot_dir / "06_feature_family_correlation_heatmap.png"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.suptitle("Basin Theory Deep Audit Dashboard", fontsize=16, fontweight="bold", color=PALETTE["black"])
    for ax, (title, path) in zip(axes.flat, panels):
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", color=PALETTE["black"])
        ax.imshow(mpimg.imread(path))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_dir / "07_theory_dashboard.png")
    fig.savefig(output_dir / "theory_dashboard.png")
    plt.close(fig)


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return dict(grouped)


def write_reports(output_dir: Path, rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]], effect_rows: list[dict[str, Any]], cv_rows: list[dict[str, Any]]) -> None:
    top_by_comparison = defaultdict(list)
    for row in sorted(effect_rows, key=lambda item: abs(safe_float(item["cohen_d"])), reverse=True):
        if len(top_by_comparison[row["comparison"]]) < 8:
            top_by_comparison[row["comparison"]].append(row)
    cv_summary = [row for row in cv_rows if row["split"] == "question_grouped_cv"]
    lines = [
        "# Basin Theory Deep Audit Findings",
        "",
        "## 0. No-Leak 审计",
        "",
        "本实验把 correctness-derived 字段只作为 label / grouping / post-hoc diagnostics，不作为 predictor feature。脚本会在发现 banned feature 被纳入特征集时直接报错。",
        "",
        f"- Basin rows: `{len(rows)}`",
        f"- Predictor features: `{len(all_predictor_features())}`",
        f"- Banned columns checked: `{len(BANNED_AS_FEATURE)}`",
        "",
        "## 1. 数学特征族",
        "",
        "| Family | Features |",
        "| --- | --- |",
    ]
    for family, features in FEATURE_FAMILIES.items():
        lines.append(f"| `{family}` | `{', '.join(features)}` |")
    lines.extend(["", "## 2. Top Decomposition Signals", ""])
    for comparison, items in top_by_comparison.items():
        lines.extend([f"### {comparison}", "", "| Feature | Family | Pos Mean | Neg Mean | Cohen d | AUC pos high |", "| --- | --- | ---: | ---: | ---: | ---: |"])
        for row in items:
            lines.append(
                f"| `{row['feature']}` | `{row['family']}` | `{safe_float(row['positive_mean']):.4f}` | `{safe_float(row['negative_mean']):.4f}` | `{safe_float(row['cohen_d']):.3f}` | `{safe_float(row['auc_positive_high']):.3f}` |"
            )
        lines.append("")
    lines.extend(["## 3. Question-Heldout Diagnostic Separability", "", "| Task | Feature Set | AUC | Balanced Acc | Rows |", "| --- | --- | ---: | ---: | ---: |"])
    for row in sorted(cv_summary, key=lambda item: (item["task"], -safe_float(item["auc"]))):
        lines.append(f"| `{row['task']}` | `{row['feature_set']}` | `{safe_float(row['auc']):.3f}` | `{safe_float(row['balanced_acc']):.3f}` | `{int(row['rows'])}` |")
    lines.extend(
        [
            "",
            "## 4. 初步解释",
            "",
            "- 如果 stability 特征强但不能区分 stable correct / stable hallucination，说明“稳定性”主要刻画 basin attractor，而不是 factual correctness。",
            "- 如果 waveform functional 特征在 stable hallucination 上仍有较大 effect size，说明宏观稳定和微观 token 不稳定可以同时存在。",
            "- 如果 sample0_delta 特征对 rescue/damage 更强，说明 controller 应该关注 alternative basin 相对 sample0 的变化，而不是只看单个 basin 的绝对分数。",
            "- 这些结果仍然是机制分析；除 question-heldout diagnostic classifier 外，不应被表述为 deployable controller 效果。",
        ]
    )
    (output_dir / "basin_theory_deep_findings_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    controller_lines = [
        "# Controller Implications from Basin Theory Audit",
        "",
        "## 核心结论",
        "",
        "当前最值得投入的不是更重的 RL，而是把 controller state 从浅层 geometry 扩展为 stability + waveform dynamics + sample0-relative contrast + future factual residual。",
        "",
        "建议的 state 形式：",
        "",
        "\\[",
        "s_k = [S_k, G_k, U_k, \\Delta_{0\\rightarrow k}, R_{fact}]",
        "\\]",
        "",
        "其中本实验覆盖了 `S_k`、`G_k`、`U_k` 和 `Delta`，尚未覆盖真正的 `R_fact`。",
        "",
        "## 下一步优先级",
        "",
        "1. 若 waveform functional 在 stable hallucination 上稳定分离，应把 `wf_entropy_max`、`roughness`、`prob_min` 接入 prefix risk model。",
        "2. 若 sample0_delta 对 rescue/damage 更强，应把 controller 改成 pairwise sample0-vs-alternative scoring，而不是 absolute basin scoring。",
        "3. 若所有 no-leak 数值特征仍弱，则下一步应做 semantic/factual basin verifier pilot，而不是继续调数值 controller。",
    ]
    (output_dir / "controller_implications_zh.md").write_text("\n".join(controller_lines) + "\n", encoding="utf-8")


def main() -> None:
    output_dir = DEFAULT_OUTPUT_ROOT / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_basin_theory_deep_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = read_csv(DEFAULT_CANDIDATE_RUN / "candidate_features.csv")
    basin_rows = read_csv(DEFAULT_WAVEFORM_RUN / "basin_waveform_table.csv")
    rows = build_theory_table(candidate_rows, basin_rows)
    audit_rows = feature_audit(sorted({key for row in rows for key in row}))
    effect_rows: list[dict[str, Any]] = []
    effect_rows.extend(
        compare_groups(
            rows,
            lambda row: safe_float(row.get("is_stable_correct_basin")) > 0,
            lambda row: safe_float(row.get("is_stable_hallucination_basin")) > 0,
            "stable_correct_vs_stable_hallucination",
        )
    )
    effect_rows.extend(compare_groups(rows, lambda row: safe_float(row.get("is_pure_correct")) > 0, lambda row: safe_float(row.get("is_pure_wrong")) > 0, "pure_correct_vs_pure_wrong"))
    effect_rows.extend(summarize_deltas(rows))
    cv_rows = run_diagnostic_cv(rows)
    write_csv(output_dir / "basin_theory_table.csv", rows)
    write_csv(output_dir / "feature_leakage_audit.csv", audit_rows)
    write_csv(output_dir / "theory_feature_effect_sizes.csv", effect_rows)
    write_csv(output_dir / "diagnostic_classifier_cv.csv", cv_rows)
    make_plots(output_dir, rows, effect_rows, cv_rows)
    write_reports(output_dir, rows, audit_rows, effect_rows, cv_rows)
    write_json(
        output_dir / "run_metadata.json",
        {
            "candidate_run": str(DEFAULT_CANDIDATE_RUN),
            "waveform_run": str(DEFAULT_WAVEFORM_RUN),
            "basin_rows": len(rows),
            "predictor_features": all_predictor_features(),
            "banned_as_feature": sorted(BANNED_AS_FEATURE),
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "basin_rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
