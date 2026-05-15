"""Measure end-to-end inference latency for K=1 and K=8 across base models.

For each base in {Qwen-7B, Llama-3B} (small enough to load quickly):

  - Prompt-only forward (1 pass, get prompt-last hidden state for DCP probe)
  - K=1 greedy generation (~25 completion tokens, the actual answer)
  - K=8 multinomial sampling (~25 completion tokens each, for self-consistency)

For Qwen-72B we also measure (slower but feasible on 1x H200/H100).

We then add the DCP probe forward latency (sklearn MLP, runs on CPU). The
purpose is to get the *deployment-relevant* numbers used in the cascade
analysis and the paper's cost-effectiveness section.

Output: results/latency_measurements.csv
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parents[1]

MODELS = [
    {"tag": "qwen7b", "name": "Qwen2.5-7B", "path": "/zhutingqi/song/qwen_model/model",
     "params_b": 7.6},
    {"tag": "llama3b", "name": "Llama-3.2-3B", "path": "/zhutingqi/Llama-3.2-3B-Instruct",
     "params_b": 3.2},
    {"tag": "qwen72b", "name": "Qwen2.5-72B", "path": "/zhutingqi/Qwen2.5-72B-Instruct",
     "params_b": 72.7},
]

# 30 prompts, similar style to TriviaQA short-answer questions
SAMPLE_PROMPTS = [
    "Q: Who wrote the novel '1984'?\nA:",
    "Q: What is the capital of France?\nA:",
    "Q: When did World War II end?\nA:",
    "Q: Who painted the Mona Lisa?\nA:",
    "Q: What is the largest planet in our solar system?\nA:",
    "Q: Who developed the theory of general relativity?\nA:",
    "Q: What is the chemical symbol for gold?\nA:",
    "Q: Which country has the largest population?\nA:",
    "Q: Who is the author of 'Pride and Prejudice'?\nA:",
    "Q: What year did the Berlin Wall fall?\nA:",
    "Q: Which mountain is the tallest on Earth?\nA:",
    "Q: Who composed the Symphony No. 9?\nA:",
    "Q: What is the smallest country by area?\nA:",
    "Q: Who founded Microsoft?\nA:",
    "Q: What is the speed of light in m/s?\nA:",
    "Q: What is the capital of Japan?\nA:",
    "Q: Who wrote Hamlet?\nA:",
    "Q: What's the boiling point of water in Fahrenheit?\nA:",
    "Q: Which ocean is the largest?\nA:",
    "Q: Who invented the telephone?\nA:",
    "Q: What is the longest river in Africa?\nA:",
    "Q: Who painted 'Starry Night'?\nA:",
    "Q: What is the currency of the United Kingdom?\nA:",
    "Q: Which element has the atomic number 1?\nA:",
    "Q: Who wrote the play 'A Midsummer Night's Dream'?\nA:",
    "Q: What is the deepest known point in the ocean?\nA:",
    "Q: Who was the first person to walk on the moon?\nA:",
    "Q: What is the chemical symbol for sodium?\nA:",
    "Q: Who is the author of 'The Great Gatsby'?\nA:",
    "Q: What is the largest desert on Earth?\nA:",
]

MAX_NEW_TOKENS = 25
WARMUP = 3
N_TRIALS = 30


def measure_one_model(cfg: dict, do_72b: bool = False) -> list[dict]:
    if cfg["tag"] == "qwen72b" and not do_72b:
        # Use scaling from Qwen-7B and Llama-3B (we'll fill in later).
        return []
    print(f"\n[load] {cfg['name']} from {cfg['path']}")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg["tag"] == "qwen72b":
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16,
            device_map="cuda:0", trust_remote_code=True,
        )
    model.eval()
    t_load = time.perf_counter() - t0
    print(f"[load] done in {t_load:.1f}s")

    rows: list[dict] = []
    device = next(model.parameters()).device

    # Pre-tokenize prompts.
    inputs = [tokenizer(p, return_tensors="pt").to(device) for p in SAMPLE_PROMPTS[:N_TRIALS + WARMUP]]

    # ----- 1. Prompt-only forward (for DCP probe input) -----
    print(f"[{cfg['tag']}] measuring prompt-only forward x{N_TRIALS}")
    times_pf: list[float] = []
    with torch.no_grad():
        for i, inp in enumerate(inputs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(**inp, output_hidden_states=True, use_cache=False)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            if i >= WARMUP:
                times_pf.append(elapsed)

    # ----- 2. K=1 greedy generation -----
    print(f"[{cfg['tag']}] measuring K=1 greedy gen x{N_TRIALS}")
    times_k1: list[float] = []
    with torch.no_grad():
        for i, inp in enumerate(inputs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.generate(
                **inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            if i >= WARMUP:
                times_k1.append(elapsed)

    # ----- 3. K=8 sampling -----
    print(f"[{cfg['tag']}] measuring K=8 sampling x{N_TRIALS}")
    times_k8: list[float] = []
    with torch.no_grad():
        for i, inp in enumerate(inputs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.generate(
                **inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                num_return_sequences=8, temperature=0.7, top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            if i >= WARMUP:
                times_k8.append(elapsed)

    rows.append({"base": cfg["tag"], "name": cfg["name"], "params_b": cfg["params_b"],
                 "mode": "prompt_only_forward", "n": len(times_pf),
                 "mean_ms": float(np.mean(times_pf)), "median_ms": float(np.median(times_pf)),
                 "p25_ms": float(np.percentile(times_pf, 25)),
                 "p75_ms": float(np.percentile(times_pf, 75)),
                 "min_ms": float(np.min(times_pf)), "max_ms": float(np.max(times_pf))})
    rows.append({"base": cfg["tag"], "name": cfg["name"], "params_b": cfg["params_b"],
                 "mode": "k1_greedy", "n": len(times_k1),
                 "mean_ms": float(np.mean(times_k1)), "median_ms": float(np.median(times_k1)),
                 "p25_ms": float(np.percentile(times_k1, 25)),
                 "p75_ms": float(np.percentile(times_k1, 75)),
                 "min_ms": float(np.min(times_k1)), "max_ms": float(np.max(times_k1))})
    rows.append({"base": cfg["tag"], "name": cfg["name"], "params_b": cfg["params_b"],
                 "mode": "k8_sampling", "n": len(times_k8),
                 "mean_ms": float(np.mean(times_k8)), "median_ms": float(np.median(times_k8)),
                 "p25_ms": float(np.percentile(times_k8, 25)),
                 "p75_ms": float(np.percentile(times_k8, 75)),
                 "min_ms": float(np.min(times_k8)), "max_ms": float(np.max(times_k8))})
    print(f"[{cfg['tag']}] DONE: prompt={np.mean(times_pf):.0f}ms, "
          f"k1={np.mean(times_k1):.0f}ms, k8={np.mean(times_k8):.0f}ms")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-72b", action="store_true",
                    help="Also measure Qwen2.5-72B (~10 min load + slow inference)")
    args = ap.parse_args()

    out_path = THIS_DIR / "results" / "latency_measurements.csv"
    rows: list[dict] = []
    for cfg in MODELS:
        try:
            rows += measure_one_model(cfg, do_72b=args.include_72b)
        except Exception as e:
            print(f"[error] {cfg['tag']}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n[done] {out_path} ({len(df)} rows)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
