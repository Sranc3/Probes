"""Extract Qwen2.5-7B hidden states for each TriviaQA question.

For every unique question_id present in the anchor CSV we run a single forward
pass with ``output_hidden_states=True`` on the prompt only (no generation).

Outputs (one ``.npz`` file):

- ``question_ids``        : (Q,) string array
- ``layer_indices``       : (L,) int array (layer ids we kept; includes the
  embedding layer at index 0 and the final layer ``num_layers``).
- ``hidden_prompt_last``  : (Q, L, hidden_dim) fp16
- ``prompt_lengths``      : (Q,) int array (#tokens including chat template)

Note: We deliberately use *prompt-only* hidden states to match the SEPs setup
(single forward pass, no generation cost). This makes the comparison apples to
apples with SEPs at deployment time.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR / "shared"))

from data_utils import load_anchor_rows, select_sample0_rows  # noqa: E402

DEFAULT_LAYERS = [4, 8, 12, 16, 20, 24, 27, 28]  # for 28-layer Qwen2.5-7B (+1 final)
SYSTEM_PROMPT = "Answer the question briefly and factually."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--questions",
        default="/zhutingqi/song/Plan_gpt55/cross_model_anchor/runs/run_20260513_012413_gptoss_anchor_triviaqa_full500_k4_tok256/qwen_candidate_anchor_rows_final_only.csv",
    )
    p.add_argument("--model-dir", default="/zhutingqi/song/qwen_model/model")
    p.add_argument(
        "--output",
        default=str(THIS_DIR / "runs" / "hidden_states.npz"),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-questions", type=int, default=None, help="If set, truncate for smoke test")
    p.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Layer indices to keep (defaults to a spread across depth + final layer)",
    )
    return p.parse_args()


def to_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] anchor rows: {args.questions}")
    df = load_anchor_rows(args.questions)
    sample0 = select_sample0_rows(df)
    qid_order = sample0["question_id"].drop_duplicates().tolist()
    if args.max_questions is not None:
        qid_order = qid_order[: args.max_questions]
    print(f"[load] unique questions: {len(qid_order)}")

    qid_to_question: dict[str, str] = {}
    for qid in qid_order:
        rows = sample0[sample0["question_id"] == qid]
        qid_to_question[qid] = str(rows["question"].iloc[0])

    print(f"[load] tokenizer + model from {args.model_dir}")
    from transformers import AutoModelForCausalLM, AutoTokenizer  # local import to keep deps light

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=to_dtype(args.dtype),
        trust_remote_code=True,
        device_map=args.device,
        attn_implementation="eager",
    )
    model.eval()
    config = model.config
    n_layers = int(getattr(config, "num_hidden_layers"))
    hidden_dim = int(getattr(config, "hidden_size"))
    print(f"[load] num_hidden_layers={n_layers} hidden_size={hidden_dim}")

    layers = sorted(set(args.layers if args.layers else DEFAULT_LAYERS))
    layers = [layer for layer in layers if 0 <= layer <= n_layers]
    print(f"[layers] keeping {layers}")

    # output buffers
    Q = len(qid_order)
    L = len(layers)
    hidden_prompt_last = np.zeros((Q, L, hidden_dim), dtype=np.float32)
    prompt_lengths = np.zeros((Q,), dtype=np.int32)

    t0 = time.time()
    for i, qid in enumerate(qid_order):
        question = qid_to_question[qid]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoding = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
        prompt_lengths[i] = int(encoding["input_ids"].shape[-1])
        with torch.inference_mode():
            outputs = model(
                **encoding,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        # outputs.hidden_states: tuple of (n_layers + 1) tensors of shape [1, T, H]
        last_pos = encoding["input_ids"].shape[-1] - 1
        for li, layer in enumerate(layers):
            h = outputs.hidden_states[layer][0, last_pos, :]
            hidden_prompt_last[i, li] = h.detach().float().cpu().numpy()
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            eta = avg * (Q - i - 1)
            print(f"[extract] {i+1}/{Q} avg={avg*1000:.1f}ms eta={eta:.0f}s")

    np.savez_compressed(
        out_path,
        question_ids=np.array(qid_order, dtype=object),
        layer_indices=np.array(layers, dtype=np.int32),
        hidden_prompt_last=hidden_prompt_last,
        prompt_lengths=prompt_lengths,
    )
    print(f"[done] wrote {out_path} ({hidden_prompt_last.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
