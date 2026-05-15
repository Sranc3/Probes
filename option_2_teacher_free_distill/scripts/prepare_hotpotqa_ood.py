"""Prepare HotpotQA OOD evaluation data for the teacher-free probes.

For each of N HotpotQA dev_distractor questions we:

1. Build the chat-template prompt with the same system prompt used for TriviaQA
   (``"Answer the question briefly and factually."``).
2. Run a single forward pass + greedy generation on Qwen2.5-7B-Instruct.
3. Cache the last prompt-token hidden state at the same layer set used in
   TriviaQA so the trained probes apply unchanged.
4. Decode the greedy answer, compute ``strict_correct`` against ``ideal``.

Outputs (single ``.npz``):
- ``question_ids``         : (Q,) string array
- ``layer_indices``        : (L,) int array (same as TriviaQA cache)
- ``hidden_prompt_last``   : (Q, L, hidden_dim) float32
- ``prompt_lengths``       : (Q,) int
- ``answers``              : (Q,) string array (decoded greedy answer)
- ``strict_correct``       : (Q,) int array (1 if greedy answer matched any ideal)
- ``ideals``               : (Q,) object array (list[str] of acceptable answers)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR / "shared"))
sys.path.insert(0, "/zhutingqi/song/Plan_opus/shared")

from text_utils import strict_correct  # noqa: E402

DEFAULT_LAYERS = [4, 8, 12, 16, 20, 24, 27, 28]
SYSTEM_PROMPT = "Answer the question briefly and factually."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-jsonl",
        default="/zhutingqi/song/datasets/HotpotQA/processed/hotpotqa.dev_distractor.context1200.jsonl",
    )
    p.add_argument("--model-dir", default="/zhutingqi/song/qwen_model/model")
    p.add_argument(
        "--output",
        default=str(THIS_DIR / "runs" / "hotpotqa_ood.npz"),
    )
    p.add_argument("--max-questions", type=int, default=500)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Layer indices to keep (defaults to TriviaQA spread)",
    )
    return p.parse_args()


def to_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_records(path: Path, max_n: int) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_n:
                break
            obj = json.loads(line)
            records.append(obj)
    return records


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(Path(args.input_jsonl), args.max_questions)
    print(f"[load] {len(records)} HotpotQA records from {args.input_jsonl}")

    print(f"[load] tokenizer + model from {args.model_dir}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    n_layers = int(model.config.num_hidden_layers)
    hidden_dim = int(model.config.hidden_size)
    print(f"[load] num_hidden_layers={n_layers} hidden_size={hidden_dim}")

    layers = sorted(set(args.layers if args.layers else DEFAULT_LAYERS))
    layers = [layer for layer in layers if 0 <= layer <= n_layers]
    print(f"[layers] {layers}")

    Q = len(records)
    L = len(layers)
    hidden_prompt_last = np.zeros((Q, L, hidden_dim), dtype=np.float32)
    prompt_lengths = np.zeros((Q,), dtype=np.int32)
    qids = np.empty((Q,), dtype=object)
    answers = np.empty((Q,), dtype=object)
    strict = np.zeros((Q,), dtype=np.int32)
    ideals_arr = np.empty((Q,), dtype=object)

    t0 = time.time()
    n_match = 0
    for i, rec in enumerate(records):
        question = str(rec["question"])
        ideals = list(rec.get("ideal") or rec.get("ideal_answers") or [])
        qid = str(rec.get("id", f"hotpot_{i}"))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoding = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
        prompt_len = int(encoding["input_ids"].shape[-1])
        prompt_lengths[i] = prompt_len

        # 1) one prompt-only forward pass for hidden states (matches TriviaQA cache exactly)
        with torch.inference_mode():
            outputs = model(
                **encoding,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        last_pos = prompt_len - 1
        for li, layer in enumerate(layers):
            h = outputs.hidden_states[layer][0, last_pos, :]
            hidden_prompt_last[i, li] = h.detach().float().cpu().numpy()

        # 2) greedy generation for sample0 + strict correctness label
        with torch.inference_mode():
            gen_ids = model.generate(
                **encoding,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        new_tokens = gen_ids[0, prompt_len:].tolist()
        answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        sc = bool(strict_correct(answer_text, ideals))
        qids[i] = qid
        answers[i] = answer_text
        strict[i] = int(sc)
        ideals_arr[i] = ideals
        n_match += int(sc)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            eta = avg * (Q - i - 1)
            acc = n_match / (i + 1)
            print(f"[gen] {i+1}/{Q} avg={avg:.2f}s eta={eta:.0f}s acc_so_far={acc:.3f}")

    base_acc = float(strict.mean())
    print(f"[done] greedy strict_correct = {n_match}/{Q} = {base_acc:.4f}")

    np.savez_compressed(
        out_path,
        question_ids=qids,
        layer_indices=np.array(layers, dtype=np.int32),
        hidden_prompt_last=hidden_prompt_last,
        prompt_lengths=prompt_lengths,
        answers=answers,
        strict_correct=strict,
        ideals=ideals_arr,
    )
    print(f"[done] wrote {out_path} ({hidden_prompt_last.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
