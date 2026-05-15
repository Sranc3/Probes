#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_basin_theory_deep_audit import (  # noqa: E402
    BANNED_AS_FEATURE,
    DEFAULT_CANDIDATE_RUN,
    DEFAULT_OUTPUT_ROOT,
    LogisticModel,
    PALETTE,
    auc_score,
    classification_metrics,
    cohen_d,
    mean,
    question_folds,
    read_csv,
    safe_float,
    setup_plotting,
    stdev,
    tokenize,
    write_csv,
    write_json,
)


DEFAULT_THEORY_RUN = Path("/zhutingqi/song/Plan_gpt55/basin_theory_analysis/runs/run_20260430_084251_basin_theory_deep_audit")
DEFAULT_PHASE2A_RUN = Path("/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_100339_phase2a_reranking_all200_qwen25")
DEFAULT_CACHE_DIR = Path("/zhutingqi/song/Plan_gpt55/basin_theory_analysis/cache")

FACT_PROXY_FEATURES = [
    "fact_question_entity_coverage",
    "fact_numeric_alignment",
    "fact_answer_specificity",
    "fact_consensus_overlap",
    "fact_unsupported_entity_density_proxy",
    "fact_internal_dispersion_proxy",
    "fact_proxy_residual",
]

QWEN_FACT_FEATURES = [
    "qwen_supported_score",
    "qwen_contradiction_risk",
    "qwen_missing_constraint_risk",
    "qwen_factual_residual",
]

QWEN_STRUCTURED_FACT_FEATURES = [
    "qwen_answer_responsiveness_score",
    "qwen_constraint_satisfaction_score",
    "qwen_entity_number_consistency_score",
    "qwen_temporal_consistency_score",
    "qwen_basin_consensus_support_score",
    "qwen_overclaim_risk",
    "qwen_world_knowledge_conflict_risk",
    "qwen_structured_factual_residual",
    "qwen_structured_verifier_confidence",
    "qwen_structured_acceptability_score",
]

RIEMANN_FEATURES = [
    "riemann_anisotropy",
    "riemann_effective_dim",
    "riemann_local_radius",
    "riemann_knn_density",
    "riemann_geodesic_distortion",
    "riemann_curvature_proxy",
]

HIDDEN_RIEMANN_FEATURES = [
    "hidden_riemann_anisotropy",
    "hidden_riemann_effective_dim",
    "hidden_riemann_local_radius",
    "hidden_riemann_knn_density",
    "hidden_riemann_geodesic_distortion",
    "hidden_riemann_curvature_proxy",
]

BASELINE_FEATURE_SETS = {
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
    ],
    "factual_proxy": FACT_PROXY_FEATURES,
    "qwen_factual": QWEN_FACT_FEATURES,
    "qwen_structured_factual": QWEN_STRUCTURED_FACT_FEATURES,
    "riemann_geometry": RIEMANN_FEATURES,
    "hidden_riemann_geometry": HIDDEN_RIEMANN_FEATURES,
    "factual_plus_riemann": FACT_PROXY_FEATURES + QWEN_FACT_FEATURES + QWEN_STRUCTURED_FACT_FEATURES + RIEMANN_FEATURES + HIDDEN_RIEMANN_FEATURES,
    "factual_riemann_delta": [
        "delta_qwen_factual_residual_vs_sample0",
        "delta_qwen_structured_factual_residual_vs_sample0",
        "delta_qwen_answer_responsiveness_score_vs_sample0",
        "delta_qwen_constraint_satisfaction_score_vs_sample0",
        "delta_qwen_world_knowledge_conflict_risk_vs_sample0",
        "delta_fact_proxy_residual_vs_sample0",
        "delta_riemann_curvature_proxy_vs_sample0",
        "delta_hidden_riemann_curvature_proxy_vs_sample0",
    ],
    "strong_verifier_controller": [
        "qwen_structured_factual_residual",
        "qwen_structured_acceptability_score",
        "qwen_answer_responsiveness_score",
        "qwen_constraint_satisfaction_score",
        "qwen_basin_consensus_support_score",
        "qwen_world_knowledge_conflict_risk",
        "delta_qwen_structured_factual_residual_vs_sample0",
        "delta_qwen_structured_acceptability_score_vs_sample0",
        "delta_qwen_answer_responsiveness_score_vs_sample0",
        "delta_qwen_constraint_satisfaction_score_vs_sample0",
        "delta_qwen_world_knowledge_conflict_risk_vs_sample0",
        "delta_wf_prob_min_mean_vs_sample0",
        "delta_cluster_weight_mass_vs_sample0",
        "riemann_anisotropy",
    ],
    "structured_rescue_veto_compact": [
        "qwen_structured_acceptability_score",
        "qwen_answer_responsiveness_score",
        "qwen_constraint_satisfaction_score",
        "qwen_entity_number_consistency_score",
        "qwen_overclaim_risk",
        "qwen_world_knowledge_conflict_risk",
        "delta_qwen_structured_acceptability_score_vs_sample0",
        "delta_qwen_world_knowledge_conflict_risk_vs_sample0",
    ],
    "theory_plus_structured_compact": [
        "cluster_size",
        "cluster_weight_mass",
        "stable_score",
        "fragmentation_entropy",
        "top2_weight_margin",
        "wf_entropy_max_mean",
        "wf_entropy_roughness",
        "wf_prob_min_mean",
        "delta_cluster_weight_mass_vs_sample0",
        "delta_wf_prob_min_mean_vs_sample0",
        "qwen_structured_acceptability_score",
        "qwen_world_knowledge_conflict_risk",
        "delta_qwen_structured_acceptability_score_vs_sample0",
        "delta_qwen_world_knowledge_conflict_risk_vs_sample0",
        "riemann_anisotropy",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-leak factual residual and Riemann geometry follow-up audit.")
    parser.add_argument("--theory-run", type=Path, default=DEFAULT_THEORY_RUN)
    parser.add_argument("--candidate-run", type=Path, default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--phase2a-run", type=Path, default=DEFAULT_PHASE2A_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--tag", default="factual_riemann_audit")
    parser.add_argument("--qwen-batch-size", type=int, default=8)
    parser.add_argument("--hidden-batch-size", type=int, default=8)
    parser.add_argument("--max-hidden-texts", type=int, default=4096)
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-hidden", action="store_true")
    return parser.parse_args()


def text_hash(*parts: str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8", errors="ignore"))
        hasher.update(b"\0")
    return hasher.hexdigest()[:24]


def extract_entities(text: str) -> set[str]:
    chunks: set[str] = set()
    for match in re_find_entities(text):
        norm = " ".join(tokenize(match))
        if norm:
            chunks.add(norm)
    return chunks


def re_find_entities(text: str) -> list[str]:
    import re

    pattern = r"(?:[A-Z][a-zA-Z0-9'&.-]+(?:\s+|$)){1,5}"
    return [item.strip() for item in re.findall(pattern, text) if len(item.strip()) > 1]


def extract_numbers(text: str) -> set[str]:
    import re

    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def candidate_groups(candidate_rows: list[dict[str, Any]]) -> dict[tuple[int, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[(int(row["seed"]), str(row["question_id"]), int(row["cluster_id"]))].append(row)
    return grouped


def consensus_texts(rows: list[dict[str, Any]], max_items: int = 4) -> list[str]:
    unique = []
    seen = set()
    for row in sorted(rows, key=lambda item: int(float(item.get("sample_index", 0)))):
        text = str(row.get("answer_text", "")).strip()
        if text and text not in seen:
            unique.append(text)
            seen.add(text)
        if len(unique) >= max_items:
            break
    return unique


def factual_proxy_features(row: dict[str, Any], cands: list[dict[str, Any]]) -> dict[str, float]:
    question = str(row.get("question", ""))
    answer = str(row.get("answer_preview", ""))
    snippets = consensus_texts(cands)
    q_tokens = set(tokenize(question))
    a_tokens = set(tokenize(answer))
    q_entities = extract_entities(question)
    a_entities = extract_entities(answer)
    q_numbers = extract_numbers(question)
    a_numbers = extract_numbers(answer)

    entity_coverage = len(q_entities & a_entities) / max(1, len(q_entities)) if q_entities else 1.0
    numeric_alignment = 1.0 if not q_numbers else len(q_numbers & a_numbers) / max(1, len(q_numbers))
    specificity = min(1.0, (len(a_entities) + len(a_numbers) + 0.2 * len(a_tokens)) / 16.0)
    unsupported_density = len(a_entities - q_entities) / max(1, len(a_entities))
    question_coverage = jaccard(q_tokens, a_tokens)

    snippet_sets = [set(tokenize(text)) for text in snippets]
    if len(snippet_sets) >= 2:
        overlaps = [jaccard(snippet_sets[i], snippet_sets[j]) for i in range(len(snippet_sets)) for j in range(i + 1, len(snippet_sets))]
        consensus_overlap = mean(overlaps)
        internal_dispersion = 1.0 - consensus_overlap
    else:
        consensus_overlap = 1.0
        internal_dispersion = 0.0

    residual = (
        0.28 * (1.0 - entity_coverage)
        + 0.22 * (1.0 - numeric_alignment)
        + 0.18 * unsupported_density
        + 0.16 * (1.0 - consensus_overlap)
        + 0.10 * (1.0 - question_coverage)
        + 0.06 * (1.0 - specificity)
    )
    return {
        "fact_question_entity_coverage": entity_coverage,
        "fact_numeric_alignment": numeric_alignment,
        "fact_answer_specificity": specificity,
        "fact_consensus_overlap": consensus_overlap,
        "fact_unsupported_entity_density_proxy": unsupported_density,
        "fact_internal_dispersion_proxy": internal_dispersion,
        "fact_proxy_residual": residual,
    }


def build_tfidf_vectors(texts: list[str], max_vocab: int = 512) -> tuple[np.ndarray, dict[str, int]]:
    docs = [tokenize(text) for text in texts]
    vocab_counts = Counter(tok for doc in docs for tok in doc)
    vocab = [tok for tok, count in vocab_counts.most_common(max_vocab) if count >= 1]
    index = {tok: idx for idx, tok in enumerate(vocab)}
    mat = np.zeros((len(docs), len(vocab)), dtype=np.float64)
    df = np.zeros(len(vocab), dtype=np.float64)
    for row_idx, doc in enumerate(docs):
        counts = Counter(tok for tok in doc if tok in index)
        for tok, count in counts.items():
            mat[row_idx, index[tok]] = count
        for tok in counts:
            df[index[tok]] += 1
    if len(vocab):
        mat *= np.log((1 + len(docs)) / (1 + df)) + 1
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.maximum(norms, 1e-9)
    return mat, index


def geometry_features(vectors: np.ndarray, prefix: str) -> dict[str, float]:
    keys = {
        f"{prefix}_anisotropy": 0.0,
        f"{prefix}_effective_dim": 0.0,
        f"{prefix}_local_radius": 0.0,
        f"{prefix}_knn_density": 0.0,
        f"{prefix}_geodesic_distortion": 0.0,
        f"{prefix}_curvature_proxy": 0.0,
    }
    if vectors.shape[0] < 2 or vectors.shape[1] < 1:
        return keys
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    try:
        _u, svals, _vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return keys
    eig = svals**2
    total = float(eig.sum())
    if total > 0:
        keys[f"{prefix}_anisotropy"] = float(eig[0] / total)
        keys[f"{prefix}_effective_dim"] = float((total * total) / max(float((eig * eig).sum()), 1e-9))
    dist = pairwise_distances(vectors)
    upper = dist[np.triu_indices(vectors.shape[0], k=1)]
    keys[f"{prefix}_local_radius"] = float(np.mean(upper)) if upper.size else 0.0
    nonzero = upper[upper > 1e-9]
    threshold = float(np.percentile(nonzero, 40)) if nonzero.size else 0.0
    adjacency = (dist <= threshold) & (dist > 1e-9) if threshold > 0 else np.zeros_like(dist, dtype=bool)
    possible = vectors.shape[0] * (vectors.shape[0] - 1)
    keys[f"{prefix}_knn_density"] = float(adjacency.sum() / max(1, possible))
    keys[f"{prefix}_geodesic_distortion"] = geodesic_distortion(dist, adjacency)
    keys[f"{prefix}_curvature_proxy"] = curvature_proxy(vectors)
    return keys


def pairwise_distances(vectors: np.ndarray) -> np.ndarray:
    diff = vectors[:, None, :] - vectors[None, :, :]
    return np.sqrt(np.maximum((diff * diff).sum(axis=-1), 0.0))


def geodesic_distortion(dist: np.ndarray, adjacency: np.ndarray) -> float:
    n = dist.shape[0]
    if n < 3 or not adjacency.any():
        return 0.0
    graph = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(graph, 0.0)
    graph[adjacency] = dist[adjacency]
    for k in range(n):
        graph = np.minimum(graph, graph[:, [k]] + graph[[k], :])
    mask = np.isfinite(graph) & (dist > 1e-9)
    if not mask.any():
        return 0.0
    ratios = graph[mask] / np.maximum(dist[mask], 1e-9)
    ratios = ratios[np.isfinite(ratios)]
    return float(np.mean(ratios)) if ratios.size else 0.0


def curvature_proxy(vectors: np.ndarray) -> float:
    if vectors.shape[0] < 3:
        return 0.0
    center = vectors.mean(axis=0)
    dirs = vectors - center
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(norms, 1e-9)
    cos = dirs @ dirs.T
    upper = cos[np.triu_indices(vectors.shape[0], k=1)]
    return float(np.var(upper)) if upper.size else 0.0


def add_semantic_riemann(rows: list[dict[str, Any]], cands_by_basin: dict[tuple[int, str, int], list[dict[str, Any]]]) -> None:
    all_texts = []
    row_keys = []
    for key, cands in cands_by_basin.items():
        for cand in cands:
            all_texts.append(str(cand.get("answer_text", "")))
            row_keys.append(key)
    vectors, _index = build_tfidf_vectors(all_texts)
    vectors_by_basin: dict[tuple[int, str, int], list[np.ndarray]] = defaultdict(list)
    for key, vector in zip(row_keys, vectors):
        vectors_by_basin[key].append(vector)
    for row in rows:
        key = (int(row["seed"]), str(row["question_id"]), int(row["cluster_id"]))
        mat = np.array(vectors_by_basin.get(key, []), dtype=np.float64)
        row.update(geometry_features(mat, "riemann"))


def load_jsonl_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            item = json.loads(line)
            cache[str(item["cache_key"])] = item
    return cache


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_model_and_tokenizer(model_dir: Path) -> tuple[Any, Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map={"": 0} if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, device, str(device)


def verifier_prompts(row: dict[str, Any], snippets: list[str]) -> dict[str, str]:
    consensus = "\n".join(f"- {text}" for text in snippets[:4])
    base = (
        "You are a strict factual verifier. Do not answer the question anew.\n"
        f"Question: {row.get('question', '')}\n"
        f"Candidate basin answer: {row.get('answer_preview', '')}\n"
        f"Consensus snippets from the same answer basin:\n{consensus}\n"
        "Judge only the candidate basin answer.\n"
    )
    return {
        "supported": base + "Is the candidate answer factually supported and responsive? Answer Yes or No:",
        "contradiction": base + "Is there an internal contradiction or likely factual conflict? Answer Yes or No:",
        "missing": base + "Does the answer miss a key constraint in the question? Answer Yes or No:",
    }


def structured_verifier_prompts(row: dict[str, Any], snippets: list[str]) -> dict[str, str]:
    consensus = "\n".join(f"- {text}" for text in snippets[:5])
    base = (
        "You are a conservative factual verifier for a basin of model answers.\n"
        "Do not solve the task from scratch and do not reveal a new answer.\n"
        "Use the question, the candidate basin answer, the basin snippets, and your factual knowledge only to judge the candidate answer.\n"
        f"Question: {row.get('question', '')}\n"
        f"Candidate basin answer: {row.get('answer_preview', '')}\n"
        f"Consensus snippets from the same answer basin:\n{consensus}\n"
    )
    return {
        "answer_responsive": base + "Does the candidate answer directly fill the answer slot asked by the question? Answer Yes or No:",
        "constraints_satisfied": base + "Does the candidate respect all explicit constraints in the question, such as date, decade, title, person, place, or relation? Answer Yes or No:",
        "entity_number_consistent": base + "Are the named entities and numbers in the candidate consistent with the question and not obviously mismatched? Answer Yes or No:",
        "temporal_consistent": base + "If the question contains time constraints, does the candidate respect them; if no time constraint exists, answer Yes. Answer Yes or No:",
        "basin_consensus_supports": base + "Do the basin snippets support the same core claim as the candidate answer? Answer Yes or No:",
        "overclaim": base + "Does the candidate add unsupported specific claims beyond what is needed to answer the question? Answer Yes or No:",
        "world_conflict": base + "Based on common factual knowledge, is the candidate likely factually wrong or conflicting with known facts? Answer Yes or No:",
    }


def yes_no_scores(model: Any, tokenizer: Any, device: Any, prompts: list[str], batch_size: int) -> list[float]:
    import torch

    yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode(" No", add_special_tokens=False)[0]
    scores: list[float] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=768).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        last_indices = enc["attention_mask"].sum(dim=1) - 1
        next_logits = logits[torch.arange(len(batch), device=device), last_indices]
        pair_logits = torch.stack([next_logits[:, yes_id], next_logits[:, no_id]], dim=-1).float()
        probs = torch.softmax(pair_logits, dim=-1)[:, 0]
        scores.extend([float(item) for item in probs.detach().cpu()])
    return scores


def add_qwen_factual_scores(
    rows: list[dict[str, Any]],
    cands_by_basin: dict[tuple[int, str, int], list[dict[str, Any]]],
    model_dir: Path,
    cache_dir: Path,
    batch_size: int,
    skip: bool,
) -> dict[str, Any]:
    for row in rows:
        for feature in QWEN_FACT_FEATURES:
            row[feature] = 0.0
    metadata = {"qwen_verifier": "skipped" if skip else "attempted", "qwen_rows_scored": 0}
    if skip:
        return metadata
    cache_path = cache_dir / "qwen_factual_scores.jsonl"
    cache = load_jsonl_cache(cache_path)
    pending: list[dict[str, Any]] = []
    prompts: list[str] = []
    prompt_meta: list[tuple[str, str]] = []
    for row in rows:
        key = (int(row["seed"]), str(row["question_id"]), int(row["cluster_id"]))
        snippets = consensus_texts(cands_by_basin.get(key, []))
        cache_key = text_hash(str(row.get("question", "")), str(row.get("answer_preview", "")), "\n".join(snippets))
        row["_qwen_cache_key"] = cache_key
        if cache_key in cache:
            continue
        prompt_map = verifier_prompts(row, snippets)
        pending.append({"cache_key": cache_key})
        for kind in ["supported", "contradiction", "missing"]:
            prompts.append(prompt_map[kind])
            prompt_meta.append((cache_key, kind))
    if pending:
        try:
            config = json.loads((DEFAULT_PHASE2A_RUN / "config_snapshot.json").read_text(encoding="utf-8"))
            model, tokenizer, device, device_name = load_model_and_tokenizer(Path(config["model_dir"]))
            scores = yes_no_scores(model, tokenizer, device, prompts, batch_size)
            grouped_scores: dict[str, dict[str, float]] = defaultdict(dict)
            for (cache_key, kind), score in zip(prompt_meta, scores):
                grouped_scores[cache_key][kind] = score
            new_rows = []
            for cache_key, item in grouped_scores.items():
                supported = item.get("supported", 0.5)
                contradiction = item.get("contradiction", 0.5)
                missing = item.get("missing", 0.5)
                new_rows.append(
                    {
                        "cache_key": cache_key,
                        "qwen_supported_score": supported,
                        "qwen_contradiction_risk": contradiction,
                        "qwen_missing_constraint_risk": missing,
                        "qwen_factual_residual": 1.0 - supported + 0.5 * contradiction + 0.5 * missing,
                    }
                )
            append_jsonl(cache_path, new_rows)
            cache.update({str(row["cache_key"]): row for row in new_rows})
            metadata.update({"qwen_device": device_name, "qwen_new_scores": len(new_rows)})
        except Exception as exc:
            metadata.update({"qwen_verifier": "failed", "qwen_error": repr(exc)})
    for row in rows:
        item = cache.get(str(row.get("_qwen_cache_key")), {})
        for feature in QWEN_FACT_FEATURES:
            row[feature] = safe_float(item.get(feature))
    metadata["qwen_rows_scored"] = sum(1 for row in rows if safe_float(row.get("qwen_supported_score")) > 0)
    return metadata


def add_qwen_structured_factual_scores(
    rows: list[dict[str, Any]],
    cands_by_basin: dict[tuple[int, str, int], list[dict[str, Any]]],
    model_dir: Path,
    cache_dir: Path,
    batch_size: int,
    skip: bool,
) -> dict[str, Any]:
    for row in rows:
        for feature in QWEN_STRUCTURED_FACT_FEATURES:
            row[feature] = 0.0
    metadata = {"qwen_structured_verifier": "skipped" if skip else "attempted", "qwen_structured_rows_scored": 0}
    if skip:
        return metadata
    cache_path = cache_dir / "qwen_structured_factual_scores.jsonl"
    cache = load_jsonl_cache(cache_path)
    prompts: list[str] = []
    prompt_meta: list[tuple[str, str]] = []
    for row in rows:
        key = (int(row["seed"]), str(row["question_id"]), int(row["cluster_id"]))
        snippets = consensus_texts(cands_by_basin.get(key, []), max_items=5)
        cache_key = text_hash("structured_v1", str(row.get("question", "")), str(row.get("answer_preview", "")), "\n".join(snippets))
        row["_qwen_structured_cache_key"] = cache_key
        if cache_key in cache:
            continue
        prompt_map = structured_verifier_prompts(row, snippets)
        for kind in [
            "answer_responsive",
            "constraints_satisfied",
            "entity_number_consistent",
            "temporal_consistent",
            "basin_consensus_supports",
            "overclaim",
            "world_conflict",
        ]:
            prompts.append(prompt_map[kind])
            prompt_meta.append((cache_key, kind))
    if prompts:
        try:
            model, tokenizer, device, device_name = load_model_and_tokenizer(model_dir)
            scores = yes_no_scores(model, tokenizer, device, prompts, batch_size)
            grouped_scores: dict[str, dict[str, float]] = defaultdict(dict)
            for (cache_key, kind), score in zip(prompt_meta, scores):
                grouped_scores[cache_key][kind] = score
            new_rows = []
            for cache_key, item in grouped_scores.items():
                responsiveness = item.get("answer_responsive", 0.5)
                constraint = item.get("constraints_satisfied", 0.5)
                entity_number = item.get("entity_number_consistent", 0.5)
                temporal = item.get("temporal_consistent", 0.5)
                consensus = item.get("basin_consensus_supports", 0.5)
                overclaim = item.get("overclaim", 0.5)
                world_conflict = item.get("world_conflict", 0.5)
                positive_mean = mean([responsiveness, constraint, entity_number, temporal, consensus])
                risk_mean = mean([overclaim, world_conflict])
                acceptability = positive_mean - risk_mean
                residual = (
                    0.22 * (1.0 - responsiveness)
                    + 0.24 * (1.0 - constraint)
                    + 0.14 * (1.0 - entity_number)
                    + 0.10 * (1.0 - temporal)
                    + 0.10 * (1.0 - consensus)
                    + 0.08 * overclaim
                    + 0.12 * world_conflict
                )
                new_rows.append(
                    {
                        "cache_key": cache_key,
                        "qwen_answer_responsiveness_score": responsiveness,
                        "qwen_constraint_satisfaction_score": constraint,
                        "qwen_entity_number_consistency_score": entity_number,
                        "qwen_temporal_consistency_score": temporal,
                        "qwen_basin_consensus_support_score": consensus,
                        "qwen_overclaim_risk": overclaim,
                        "qwen_world_knowledge_conflict_risk": world_conflict,
                        "qwen_structured_factual_residual": residual,
                        "qwen_structured_verifier_confidence": abs(positive_mean - risk_mean),
                        "qwen_structured_acceptability_score": acceptability,
                    }
                )
            append_jsonl(cache_path, new_rows)
            cache.update({str(row["cache_key"]): row for row in new_rows})
            metadata.update({"qwen_structured_device": device_name, "qwen_structured_new_scores": len(new_rows)})
        except Exception as exc:
            metadata.update({"qwen_structured_verifier": "failed", "qwen_structured_error": repr(exc)})
    for row in rows:
        item = cache.get(str(row.get("_qwen_structured_cache_key")), {})
        for feature in QWEN_STRUCTURED_FACT_FEATURES:
            row[feature] = safe_float(item.get(feature))
        if "qwen_structured_acceptability_score" not in item:
            positive_mean = mean(
                [
                    safe_float(row.get("qwen_answer_responsiveness_score")),
                    safe_float(row.get("qwen_constraint_satisfaction_score")),
                    safe_float(row.get("qwen_entity_number_consistency_score")),
                    safe_float(row.get("qwen_temporal_consistency_score")),
                    safe_float(row.get("qwen_basin_consensus_support_score")),
                ]
            )
            risk_mean = mean([safe_float(row.get("qwen_overclaim_risk")), safe_float(row.get("qwen_world_knowledge_conflict_risk"))])
            row["qwen_structured_acceptability_score"] = positive_mean - risk_mean
    metadata["qwen_structured_rows_scored"] = sum(1 for row in rows if safe_float(row.get("qwen_answer_responsiveness_score")) > 0)
    return metadata


def hidden_cache_key(text: str) -> str:
    return text_hash("hidden", text)


def add_hidden_riemann(
    rows: list[dict[str, Any]],
    cands_by_basin: dict[tuple[int, str, int], list[dict[str, Any]]],
    model_dir: Path,
    cache_dir: Path,
    batch_size: int,
    max_texts: int,
    skip: bool,
) -> dict[str, Any]:
    for row in rows:
        for feature in HIDDEN_RIEMANN_FEATURES:
            row[feature] = 0.0
    metadata = {"hidden_riemann": "skipped" if skip else "attempted"}
    if skip:
        return metadata
    texts = []
    for cands in cands_by_basin.values():
        for cand in cands:
            text = str(cand.get("answer_text", ""))
            if text:
                texts.append(text)
    unique = list(dict.fromkeys(texts))[:max_texts]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = cache_dir / "hidden_answer_embeddings.npz"
    embeddings: dict[str, np.ndarray] = {}
    if cache_npz.exists():
        loaded = np.load(cache_npz)
        embeddings = {key: loaded[key] for key in loaded.files}
    missing = [text for text in unique if hidden_cache_key(text) not in embeddings]
    if missing:
        try:
            import torch

            config = json.loads((DEFAULT_PHASE2A_RUN / "config_snapshot.json").read_text(encoding="utf-8"))
            model, tokenizer, device, device_name = load_model_and_tokenizer(Path(config["model_dir"]))
            for start in range(0, len(missing), batch_size):
                batch = missing[start : start + batch_size]
                enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True)
                hidden = out.hidden_states[-1].float()
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                for text, vector in zip(batch, pooled.detach().cpu().numpy()):
                    embeddings[hidden_cache_key(text)] = vector.astype(np.float32)
            np.savez_compressed(cache_npz, **embeddings)
            metadata.update({"hidden_device": device_name, "hidden_new_embeddings": len(missing)})
        except Exception as exc:
            metadata.update({"hidden_riemann": "failed", "hidden_error": repr(exc)})
            return metadata
    for row in rows:
        key = (int(row["seed"]), str(row["question_id"]), int(row["cluster_id"]))
        vectors = []
        for cand in cands_by_basin.get(key, []):
            text = str(cand.get("answer_text", ""))
            vector = embeddings.get(hidden_cache_key(text))
            if vector is not None:
                vectors.append(vector)
        if vectors:
            feats = geometry_features(np.array(vectors, dtype=np.float64), "hidden_riemann")
            row.update(feats)
    metadata.update({"hidden_riemann": metadata.get("hidden_riemann", "completed"), "hidden_embeddings_total": len(embeddings)})
    return metadata


def add_factual_proxy_and_riemann(rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> dict[tuple[int, str, int], list[dict[str, Any]]]:
    cands_by_basin = candidate_groups(candidate_rows)
    for row in rows:
        key = (int(row["seed"]), str(row["question_id"]), int(row["cluster_id"]))
        cands = cands_by_basin.get(key, [])
        row.update(factual_proxy_features(row, cands))
    add_semantic_riemann(rows, cands_by_basin)
    return cands_by_basin


def no_leak_audit(all_features: list[str], used_features: list[str]) -> list[dict[str, Any]]:
    rows = []
    available = set(all_features)
    used = set(used_features)
    extra_banned = BANNED_AS_FEATURE | {
        "strict_correct",
        "basin_correct_rate",
        "is_rescue_basin",
        "is_stable_correct_basin",
        "is_stable_hallucination_basin",
        "is_damage_basin",
    }
    for feature in sorted(available | used):
        status = "ok"
        if feature in used and feature in extra_banned:
            status = "ERROR_used_banned"
        if feature in used and feature not in available:
            status = "ERROR_missing_predictor"
        rows.append(
            {
                "feature": feature,
                "used_as_predictor": float(feature in used),
                "is_banned_as_feature": float(feature in extra_banned),
                "status": status,
            }
        )
    errors = [row for row in rows if row["status"] != "ok"]
    if errors:
        raise RuntimeError(f"No-leak audit failed: {errors[:8]}")
    return rows


def all_feature_sets() -> dict[str, list[str]]:
    sets = dict(BASELINE_FEATURE_SETS)
    sets["all_no_leak"] = sorted(set(sum(sets.values(), [])))
    return sets


def compare_groups(rows: list[dict[str, Any]], pos_filter: Any, neg_filter: Any, comparison: str, features: list[str]) -> list[dict[str, Any]]:
    pos_rows = [row for row in rows if pos_filter(row)]
    neg_rows = [row for row in rows if neg_filter(row)]
    out = []
    for feature in features:
        pos = [safe_float(row.get(feature)) for row in pos_rows]
        neg = [safe_float(row.get(feature)) for row in neg_rows]
        out.append(
            {
                "comparison": comparison,
                "feature": feature,
                "positive_count": len(pos),
                "negative_count": len(neg),
                "positive_mean": mean(pos),
                "negative_mean": mean(neg),
                "cohen_d": cohen_d(pos, neg),
                "auc_positive_high": auc_score(pos, neg),
            }
        )
    return out


def run_cv(rows: list[dict[str, Any]], feature_sets: dict[str, list[str]]) -> list[dict[str, Any]]:
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
    out = []
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
                out.append({"split": f"fold_{fold_idx}", "task": task_name, "feature_set": set_name, "rows": len(test), **classification_metrics(labels, probs)})
    summary = []
    keys = sorted({(row["task"], row["feature_set"]) for row in out})
    for task, feature_set in keys:
        group = [row for row in out if row["task"] == task and row["feature_set"] == feature_set]
        summary.append(
            {
                "split": "question_grouped_cv",
                "task": task,
                "feature_set": feature_set,
                "rows": sum(int(row["rows"]) for row in group),
                "auc": mean([safe_float(row["auc"]) for row in group]),
                "balanced_acc": mean([safe_float(row["balanced_acc"]) for row in group]),
                "accuracy": mean([safe_float(row["accuracy"]) for row in group]),
            }
        )
    return summary + out


def make_plots(output_dir: Path, rows: list[dict[str, Any]], effects: list[dict[str, Any]], cv_rows: list[dict[str, Any]]) -> None:
    plt = setup_plotting()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    specs = [
        ("qwen_factual_residual", "Qwen factual residual"),
        ("fact_proxy_residual", "Proxy factual residual"),
        ("qwen_supported_score", "Qwen supported score"),
        ("fact_consensus_overlap", "Consensus overlap"),
    ]
    groups = [
        ("stable_correct", lambda r: safe_float(r.get("is_stable_correct_basin")) > 0, PALETTE["green"]),
        ("stable_hallu", lambda r: safe_float(r.get("is_stable_hallucination_basin")) > 0, PALETTE["light_red"]),
        ("pure_wrong", lambda r: safe_float(r.get("is_pure_wrong")) > 0, PALETTE["gray"]),
    ]
    for ax, (feature, title) in zip(axes.ravel(), specs):
        data = [[safe_float(row.get(feature)) for row in rows if filt(row)] for _name, filt, _color in groups]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for body, (_name, _filt, color) in zip(parts["bodies"], groups):
            body.set_facecolor(color)
            body.set_alpha(0.38)
        parts["cmeans"].set_color(PALETTE["black"])
        ax.set_xticks([1, 2, 3], [name for name, _filt, _color in groups])
        ax.set_title(title)
    fig.suptitle("Factual Residual Distributions", y=0.99)
    fig.tight_layout()
    fig.savefig(plot_dir / "01_factual_residual_distributions.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    for label, filt, color in [
        ("stable correct", lambda r: safe_float(r.get("is_stable_correct_basin")) > 0, PALETTE["green"]),
        ("stable hallucination", lambda r: safe_float(r.get("is_stable_hallucination_basin")) > 0, PALETTE["light_red"]),
        ("rescue", lambda r: safe_float(r.get("is_rescue_basin")) > 0, PALETTE["blue"]),
        ("damage", lambda r: safe_float(r.get("is_damage_basin")) > 0, "#ffb8b8"),
    ]:
        group = [row for row in rows if filt(row)]
        ax.scatter(
            [safe_float(row.get("riemann_curvature_proxy")) for row in group],
            [safe_float(row.get("riemann_anisotropy")) for row in group],
            color=color,
            alpha=0.62,
            s=[25 + 55 * safe_float(row.get("cluster_weight_mass")) for row in group],
            label=f"{label} (n={len(group)})",
            edgecolors="none",
        )
    ax.set_xlabel("Riemann curvature proxy")
    ax.set_ylabel("Riemann anisotropy")
    ax.set_title("Semantic / Riemann Geometry Phase Map")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(plot_dir / "02_riemann_geometry_phase_map.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    for label, filt, color in [
        ("stable correct", lambda r: safe_float(r.get("is_stable_correct_basin")) > 0, PALETTE["green"]),
        ("stable hallucination", lambda r: safe_float(r.get("is_stable_hallucination_basin")) > 0, PALETTE["light_red"]),
    ]:
        group = [row for row in rows if filt(row)]
        ax.scatter(
            [safe_float(row.get("stable_score")) for row in group],
            [safe_float(row.get("qwen_factual_residual")) for row in group],
            color=color,
            alpha=0.68,
            s=[25 + 55 * safe_float(row.get("cluster_weight_mass")) for row in group],
            label=f"{label} (n={len(group)})",
            edgecolors="none",
        )
    ax.set_xlabel("Basin stability")
    ax.set_ylabel("Factual residual")
    ax.set_title("Stable Hallucination: Factual Residual vs Stability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "03_stable_hallucination_factual_vs_stability.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for label, filt, color in [
        ("rescue", lambda r: safe_float(r.get("is_rescue_basin")) > 0, PALETTE["green"]),
        ("damage", lambda r: safe_float(r.get("is_damage_basin")) > 0, PALETTE["light_red"]),
    ]:
        group = [row for row in rows if filt(row)]
        ax.scatter(
            [safe_float(row.get("delta_qwen_factual_residual_vs_sample0")) for row in group],
            [safe_float(row.get("delta_riemann_curvature_proxy_vs_sample0")) for row in group],
            color=color,
            alpha=0.7,
            s=[28 + 55 * safe_float(row.get("cluster_weight_mass")) for row in group],
            label=f"{label} (n={len(group)})",
            edgecolors="none",
        )
    ax.axhline(0, color=PALETTE["gray"], lw=0.8)
    ax.axvline(0, color=PALETTE["gray"], lw=0.8)
    ax.set_xlabel("Delta factual residual vs sample0")
    ax.set_ylabel("Delta curvature proxy vs sample0")
    ax.set_title("Rescue/Damage Factual and Geometry Deltas")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "04_rescue_damage_factual_delta_map.png")
    plt.close(fig)

    summary = [row for row in cv_rows if row["split"] == "question_grouped_cv"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    labels = [f"{row['task']}\n{row['feature_set']}" for row in summary]
    ax.barh(labels[::-1], [safe_float(row["auc"]) for row in summary[::-1]], color=PALETTE["blue"], alpha=0.86)
    ax.axvline(0.5, color=PALETTE["gray"], lw=0.9, linestyle="--")
    ax.set_xlabel("Question-heldout AUC")
    ax.set_title("Feature Family AUC Comparison")
    fig.tight_layout()
    fig.savefig(plot_dir / "05_feature_family_auc_comparison.png")
    plt.close(fig)

    top_curv = sorted(rows, key=lambda r: safe_float(r.get("riemann_curvature_proxy")), reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh([f"{r['question_id']}\nc{r['cluster_id']}" for r in top_curv[::-1]], [safe_float(r.get("riemann_curvature_proxy")) for r in top_curv[::-1]], color=PALETTE["green"], alpha=0.82)
    ax.set_xlabel("Curvature proxy")
    ax.set_title("High-Curvature Basin Examples")
    fig.tight_layout()
    fig.savefig(plot_dir / "06_riemann_curvature_examples.png")
    plt.close(fig)

    make_dashboard(plt, plot_dir)


def make_dashboard(plt: Any, plot_dir: Path) -> None:
    import matplotlib.image as mpimg

    panels = [
        ("A. Factual Residuals", plot_dir / "01_factual_residual_distributions.png"),
        ("B. Riemann Map", plot_dir / "02_riemann_geometry_phase_map.png"),
        ("C. Factual vs Stability", plot_dir / "03_stable_hallucination_factual_vs_stability.png"),
        ("D. Rescue/Damage Deltas", plot_dir / "04_rescue_damage_factual_delta_map.png"),
        ("E. Family AUC", plot_dir / "05_feature_family_auc_comparison.png"),
        ("F. Curvature Examples", plot_dir / "06_riemann_curvature_examples.png"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.suptitle("Factual Residual and Riemann Geometry Dashboard", fontsize=16, fontweight="bold", color=PALETTE["black"])
    for ax, (title, path) in zip(axes.flat, panels):
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.imshow(mpimg.imread(path))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_dir / "07_factual_riemann_dashboard.png")
    plt.close(fig)


def add_sample0_deltas(rows: list[dict[str, Any]], features: list[str]) -> None:
    by_pair: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[(int(row["seed"]), str(row["question_id"]))].append(row)
    for group in by_pair.values():
        sample0 = next((row for row in group if safe_float(row.get("contains_sample0")) > 0), None)
        if sample0 is None:
            continue
        for row in group:
            for feature in features:
                row[f"delta_{feature}_vs_sample0"] = safe_float(row.get(feature)) - safe_float(sample0.get(feature))


def write_reports(output_dir: Path, rows: list[dict[str, Any]], effects: list[dict[str, Any]], cv_rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    summary = [row for row in cv_rows if row["split"] == "question_grouped_cv"]
    lines = [
        "# Factual Residual and Riemann Geometry Deep Findings",
        "",
        "## 0. No-Leak Audit",
        "",
        "本实验不使用 benchmark gold answer 或 correctness-derived 字段作为 feature。`no_leak_audit_factual_riemann.csv` 中所有 predictor 均需为 `ok`。",
        "",
        f"- Rows: `{len(rows)}`",
        f"- Qwen verifier status: `{metadata.get('qwen_verifier')}`",
        f"- Qwen rows scored: `{metadata.get('qwen_rows_scored', 0)}`",
        f"- Structured Qwen verifier status: `{metadata.get('qwen_structured_verifier')}`",
        f"- Structured Qwen rows scored: `{metadata.get('qwen_structured_rows_scored', 0)}`",
        f"- Hidden geometry status: `{metadata.get('hidden_riemann')}`",
        "",
        "## 1. Feature Family AUC",
        "",
        "| Task | Feature Set | AUC | Balanced Acc | Rows |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in sorted(summary, key=lambda r: (r["task"], -safe_float(r["auc"]))):
        lines.append(f"| `{row['task']}` | `{row['feature_set']}` | `{safe_float(row['auc']):.3f}` | `{safe_float(row['balanced_acc']):.3f}` | `{int(row['rows'])}` |")
    lines.extend(["", "## 2. Top Factual / Riemann Signals", ""])
    for comparison in sorted({row["comparison"] for row in effects}):
        subset = sorted([row for row in effects if row["comparison"] == comparison], key=lambda r: abs(safe_float(r["cohen_d"])), reverse=True)[:10]
        lines.extend([f"### {comparison}", "", "| Feature | Pos Mean | Neg Mean | Cohen d | AUC pos high |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in subset:
            lines.append(f"| `{row['feature']}` | `{safe_float(row['positive_mean']):.4f}` | `{safe_float(row['negative_mean']):.4f}` | `{safe_float(row['cohen_d']):.3f}` | `{safe_float(row['auc_positive_high']):.3f}` |")
        lines.append("")
    lines.extend(
        [
            "## 3. Interpretation",
            "",
            "- 如果 factual residual 在 `stable_correct_vs_hallucination` 上超过 stability/waveform，说明稳定幻觉的关键缺口确实是事实约束残差。",
            "- 如果 Riemann geometry 在 rescue/damage 或 stable hallucination 上提供额外 AUC，说明 basin manifold shape 有独立理论价值。",
            "- 新增 structured Qwen verifier 显式检查 answer slot、question constraints、entity/number/time consistency、basin consensus、overclaim 和 world-knowledge conflict。",
            "- `strong_verifier_controller` 用 structured residual 的 sample0 delta 连接已有 basin dynamics，主要面向 rescue/damage 风险控制，而不是替代所有 theory-core 特征。",
        ]
    )
    (output_dir / "factual_riemann_deep_findings_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    controller = [
        "# Controller Implications: Factual + Riemann",
        "",
        "建议下一版 controller 不再只做 absolute basin scoring，而是使用 pairwise sample0-vs-alternative state:",
        "",
        "\\[",
        "s_{0,k}=[S_k-S_0, U_k-U_0, R_{fact,k}-R_{fact,0}, K_k-K_0]",
        "\\]",
        "",
        "其中 `R_fact` 来自 factual residual，`K` 来自 Riemann/curvature features。",
        "",
        "增强版 factual verifier 的用法：",
        "",
        "- `qwen_structured_factual_residual` 作为 absolute factual risk。",
        "- `delta_qwen_structured_factual_residual_vs_sample0` 作为 alternative basin 是否比 sample0 更可信的核心信号。",
        "- `qwen_world_knowledge_conflict_risk` 适合做 damage veto。",
        "- `qwen_structured_acceptability_score` 是 positive factual checks 减去 risk checks 的 directed score，比 confidence 更适合排序。",
        "- `qwen_answer_responsiveness_score` 和 `qwen_constraint_satisfaction_score` 适合做 rescue bonus 的必要条件。",
        "- `strong_verifier_controller` feature family 将这些 verifier 特征和 `delta_wf_prob_min_mean_vs_sample0`、`delta_cluster_weight_mass_vs_sample0`、`riemann_anisotropy` 合并，用来检验是否能比单独 verifier 更强。",
        "- `structured_rescue_veto_compact` 是只用 verifier 的轻量 veto/bonus 组合；`theory_plus_structured_compact` 是推荐的下一版 controller 候选。",
        "",
        "不要把本实验中的 label/grouping 字段放进 controller。",
    ]
    (output_dir / "controller_implications_factual_riemann_zh.md").write_text("\n".join(controller) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.theory_run / "basin_theory_table.csv")
    candidate_rows = read_csv(args.candidate_run / "candidate_features.csv")
    cands_by_basin = add_factual_proxy_and_riemann(rows, candidate_rows)
    config = json.loads((args.phase2a_run / "config_snapshot.json").read_text(encoding="utf-8"))
    metadata: dict[str, Any] = {
        "theory_run": str(args.theory_run),
        "candidate_run": str(args.candidate_run),
        "phase2a_run": str(args.phase2a_run),
        "model_dir": config.get("model_dir", ""),
    }
    metadata.update(add_qwen_factual_scores(rows, cands_by_basin, Path(config["model_dir"]), args.cache_dir, args.qwen_batch_size, args.skip_qwen))
    metadata.update(add_qwen_structured_factual_scores(rows, cands_by_basin, Path(config["model_dir"]), args.cache_dir, args.qwen_batch_size, args.skip_qwen))
    metadata.update(add_hidden_riemann(rows, cands_by_basin, Path(config["model_dir"]), args.cache_dir, args.hidden_batch_size, args.max_hidden_texts, args.skip_hidden))
    add_sample0_deltas(
        rows,
        [
            "qwen_factual_residual",
            "qwen_structured_factual_residual",
            "qwen_structured_acceptability_score",
            "qwen_answer_responsiveness_score",
            "qwen_constraint_satisfaction_score",
            "qwen_world_knowledge_conflict_risk",
            "fact_proxy_residual",
            "riemann_curvature_proxy",
            "hidden_riemann_curvature_proxy",
        ],
    )
    feature_sets = all_feature_sets()
    used_features = sorted(set(sum(feature_sets.values(), [])))
    audit_rows = no_leak_audit(sorted({key for row in rows for key in row}), used_features)
    effects = []
    effects.extend(compare_groups(rows, lambda r: safe_float(r.get("is_stable_correct_basin")) > 0, lambda r: safe_float(r.get("is_stable_hallucination_basin")) > 0, "stable_correct_vs_stable_hallucination", used_features))
    effects.extend(compare_groups(rows, lambda r: safe_float(r.get("is_pure_correct")) > 0, lambda r: safe_float(r.get("is_pure_wrong")) > 0, "pure_correct_vs_pure_wrong", used_features))
    effects.extend(compare_groups(rows, lambda r: safe_float(r.get("is_rescue_basin")) > 0, lambda r: safe_float(r.get("is_damage_basin")) > 0, "rescue_vs_damage", used_features))
    cv_rows = run_cv(rows, feature_sets)
    write_csv(output_dir / "factual_riemann_table.csv", rows)
    write_csv(output_dir / "no_leak_audit_factual_riemann.csv", audit_rows)
    write_csv(output_dir / "factual_riemann_effect_sizes.csv", effects)
    write_csv(output_dir / "factual_riemann_diagnostic_cv.csv", cv_rows)
    make_plots(output_dir, rows, effects, cv_rows)
    write_reports(output_dir, rows, effects, cv_rows, metadata)
    write_json(output_dir / "run_metadata.json", metadata)
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
