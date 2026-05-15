#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "semantic_entropy_calculate"))
sys.path.insert(0, str(ROOT_DIR / "token_entropy_calculate"))

from entropy_utils import maybe_save_plot, write_summary as write_token_summary, write_trace_csv, write_trace_json
from run_entropy_trace import build_summary_lines as build_token_summary_lines
from run_entropy_trace import trace_generation
from run_semantic_entropy import (
    build_pairwise_results,
    build_prompt,
    build_summary_lines as build_semantic_summary_lines,
    cluster_generations,
    load_causal_model,
    load_nli_model,
    sample_answers,
    summarize_clusters,
)
from semantic_utils import create_output_dir, shannon_entropy, write_generations_csv, write_json, write_pairs_csv, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run token entropy tracing and semantic entropy on TriviaQA samples.")
    parser.add_argument(
        "--input-jsonl",
        default="/public_zhutingqi/song/datasets/trivia_qa/processed/dev.preview.jsonl",
        help="Path to a normalized TriviaQA jsonl file.",
    )
    parser.add_argument("--num-questions", type=int, default=3, help="Number of TriviaQA questions to run.")
    parser.add_argument("--offset", type=int, default=0, help="Start index within the jsonl file.")
    parser.add_argument(
        "--model-dir",
        default="/public_zhutingqi/song/qwen_model/model",
        help="Path to the local causal LM directory.",
    )
    parser.add_argument(
        "--deberta-model-dir",
        default="/public_zhutingqi/song/DeBERTa_model/model",
        help="Path to the local DeBERTa MNLI model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="/public_zhutingqi/song/output",
        help="Base directory for timestamped outputs.",
    )
    parser.add_argument("--token-max-new-tokens", type=int, default=96, help="Token trace generation length.")
    parser.add_argument("--semantic-max-new-tokens", type=int, default=72, help="Semantic sampling generation length.")
    parser.add_argument("--semantic-num-samples", type=int, default=6, help="Number of semantic samples per question.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k tokens to save per token-trace step.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling threshold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for both models.",
    )
    return parser.parse_args()


def load_records(path: str, offset: int, limit: int) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line_index, line in enumerate(file_obj):
            if line_index < offset:
                continue
            if len(records) >= limit:
                break
            records.append(json.loads(line))
    return records


def sanitize_name(text: str) -> str:
    filtered = "".join(character if character.isalnum() else "_" for character in text.lower())
    return filtered[:48].strip("_") or "sample"


def maybe_save_semantic_plot(path: Path, cluster_summaries: list[dict]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    cluster_ids = [str(cluster["cluster_id"]) for cluster in cluster_summaries]
    masses = [cluster["weight_mass"] for cluster in cluster_summaries]
    sizes = [cluster["size"] for cluster in cluster_summaries]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(cluster_ids, masses, color="#7c4dff", alpha=0.85)
    plt.title("Semantic Cluster Probability Mass")
    plt.xlabel("Semantic Cluster")
    plt.ylabel("Normalized Mass")
    plt.ylim(0, max(masses) * 1.2 if masses else 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, size, mass in zip(bars, sizes, masses):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"n={size}\n{mass:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def maybe_save_overview_plot(path: Path, rows: list[dict]) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    labels = [f"Q{i}" for i in range(len(rows))]
    token_means = [row["token_mean_entropy"] for row in rows]
    token_maxes = [row["token_max_entropy"] for row in rows]
    semantic_weighted = [row["semantic_entropy_weighted"] for row in rows]
    semantic_uniform = [row["semantic_entropy_uniform"] for row in rows]

    x = np.arange(len(rows))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x - width / 2, token_means, width, label="token mean", color="#1565c0")
    axes[0].bar(x + width / 2, token_maxes, width, label="token max", color="#64b5f6")
    axes[0].set_title("Token Entropy Summary")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Entropy")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].bar(x - width / 2, semantic_uniform, width, label="semantic uniform", color="#6a1b9a")
    axes[1].bar(x + width / 2, semantic_weighted, width, label="semantic weighted", color="#ba68c8")
    axes[1].set_title("Semantic Entropy Summary")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Entropy")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def compute_token_stats(steps: list) -> dict[str, float]:
    entropies = [step.entropy for step in steps]
    if not entropies:
        return {"token_mean_entropy": 0.0, "token_max_entropy": 0.0, "token_min_entropy": 0.0}
    return {
        "token_mean_entropy": sum(entropies) / len(entropies),
        "token_max_entropy": max(entropies),
        "token_min_entropy": min(entropies),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = load_records(args.input_jsonl, args.offset, args.num_questions)
    batch_dir = create_output_dir(args.output_dir)

    generation_tokenizer, generation_model = load_causal_model(args.model_dir, args.device)
    nli_tokenizer, nli_model = load_nli_model(args.deberta_model_dir, args.device)

    overview_rows: list[dict] = []
    overview_lines = [
        "TriviaQA joint entropy batch",
        f"input_jsonl: {args.input_jsonl}",
        f"offset: {args.offset}",
        f"num_questions: {len(records)}",
        f"token_max_new_tokens: {args.token_max_new_tokens}",
        f"semantic_max_new_tokens: {args.semantic_max_new_tokens}",
        f"semantic_num_samples: {args.semantic_num_samples}",
        f"temperature: {args.temperature}",
        f"top_p: {args.top_p}",
        "",
    ]

    for local_index, record in enumerate(records):
        question = record["question"]
        system_prompt = record.get("system_prompt", "")
        question_dir = batch_dir / f"{local_index:02d}_{sanitize_name(record['id'])}"
        question_dir.mkdir(parents=True, exist_ok=False)

        prompt = build_prompt(generation_tokenizer, question, system_prompt)
        token_steps, token_full_text = trace_generation(
            model=generation_model,
            tokenizer=generation_tokenizer,
            prompt=prompt,
            max_new_tokens=args.token_max_new_tokens,
            top_k=args.top_k,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )
        token_metadata = {
            "timestamp": question_dir.parent.name.replace("run_", ""),
            "question_id": record["id"],
            "model_dir": str(Path(args.model_dir).resolve()),
            "device": args.device,
            "prompt": prompt,
            "max_new_tokens": args.token_max_new_tokens,
            "top_k": args.top_k,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "output_dir": str(question_dir.resolve()),
        }
        token_plot_created = maybe_save_plot(question_dir / "token_entropy_plot.png", token_steps)
        write_trace_json(question_dir / "token_entropy_trace.json", token_metadata, token_steps)
        write_trace_csv(question_dir / "token_entropy_trace.csv", token_steps)
        (question_dir / "token_generated_text.txt").write_text(token_full_text, encoding="utf-8")
        write_token_summary(
            question_dir / "token_summary.txt",
            build_token_summary_lines(token_metadata, token_steps, token_full_text, token_plot_created),
        )
        token_stats = compute_token_stats(token_steps)

        generations = sample_answers(
            tokenizer=generation_tokenizer,
            model=generation_model,
            question=question,
            system_prompt=system_prompt,
            num_samples=args.semantic_num_samples,
            max_new_tokens=args.semantic_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )
        pairwise_results = build_pairwise_results(
            question=question,
            generations=generations,
            tokenizer=nli_tokenizer,
            model=nli_model,
            device=args.device,
        )
        clusters, _ = cluster_generations(generations, pairwise_results)
        cluster_summaries = summarize_clusters(generations, clusters)
        uniform_cluster_probs = [cluster["size"] / len(generations) for cluster in cluster_summaries]
        weighted_cluster_probs = [cluster["weight_mass"] for cluster in cluster_summaries]
        uniform_entropy = shannon_entropy(uniform_cluster_probs)
        weighted_entropy = shannon_entropy(weighted_cluster_probs)

        semantic_run_args = argparse.Namespace(
            question=question,
            system_prompt=system_prompt,
            model_dir=args.model_dir,
            deberta_model_dir=args.deberta_model_dir,
            num_samples=args.semantic_num_samples,
            max_new_tokens=args.semantic_max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        semantic_summary_lines = build_semantic_summary_lines(
            semantic_run_args,
            generations,
            cluster_summaries,
            uniform_entropy,
            weighted_entropy,
        )
        semantic_summary_lines.extend(["", f"id: {record['id']}", f"ideal_answers: {record.get('ideal', [])}"])

        semantic_payload = {
            "record": record,
            "uniform_entropy": uniform_entropy,
            "weighted_entropy": weighted_entropy,
            "clusters": cluster_summaries,
            "generations": [asdict(generation) for generation in generations],
            "pairwise_results": [asdict(result) for result in pairwise_results],
        }
        semantic_plot_created = maybe_save_semantic_plot(question_dir / "semantic_entropy_plot.png", cluster_summaries)

        write_json(question_dir / "semantic_entropy.json", semantic_payload)
        write_generations_csv(question_dir / "semantic_generations.csv", generations)
        write_pairs_csv(question_dir / "semantic_pairwise_nli.csv", pairwise_results)
        write_summary(question_dir / "semantic_summary.txt", semantic_summary_lines)

        overview_row = {
            "id": record["id"],
            "question": question,
            "ideal_answers": record.get("ideal", []),
            "result_dir": str(question_dir),
            "semantic_clusters": len(cluster_summaries),
            "semantic_entropy_uniform": uniform_entropy,
            "semantic_entropy_weighted": weighted_entropy,
            "semantic_plot_created": semantic_plot_created,
            **token_stats,
        }
        overview_rows.append(overview_row)

        overview_lines.extend(
            [
                f"[{local_index}] id={record['id']}",
                f"question: {question}",
                f"ideal_answers: {record.get('ideal', [])}",
                f"token_mean_entropy: {token_stats['token_mean_entropy']:.6f}",
                f"token_max_entropy: {token_stats['token_max_entropy']:.6f}",
                f"semantic_clusters: {len(cluster_summaries)}",
                f"semantic_entropy_uniform: {uniform_entropy:.6f}",
                f"semantic_entropy_weighted: {weighted_entropy:.6f}",
                f"result_dir: {question_dir}",
                "",
            ]
        )

    overview_plot_created = maybe_save_overview_plot(batch_dir / "joint_entropy_overview.png", overview_rows)
    overview_payload = {
        "config": vars(args),
        "overview_plot_created": overview_plot_created,
        "runs": overview_rows,
    }
    write_summary(batch_dir / "batch_summary.txt", overview_lines)
    write_json(batch_dir / "batch_summary.json", overview_payload)

    print("\n".join(overview_lines))
    print(f"overview_plot_created: {overview_plot_created}")
    print(f"Saved batch results to: {batch_dir}")


if __name__ == "__main__":
    main()
