"""Run GPT-OSS-120B (teacher) on the same OOD questions seen by Qwen2.5-7B.

This generates the teacher data needed to extend the 2-tier cascade analysis
to OOD distributions (HotpotQA dev_distractor + NQ-Open validation).

For each OOD dataset:
  1. Load Qwen-7B's cached OOD npz (gives us the *exact* question_ids and
     ideal answers that the student saw).
  2. Re-load the source JSONL/parquet to recover the actual question text
     (the npz only stores qids + ideals, not the prompt text).
  3. Build the same chat prompt used for Qwen ("Answer the question briefly
     and factually." system + user message), tokenise with GPT-OSS's chat
     template.
  4. Greedy-decode up to 128 tokens.
  5. Compute strict_correct vs ideals, record latency_ms.
  6. Save a JSONL with one record per question.

Outputs (per OOD dataset):
  runs/teacher_oss120b_<dataset>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# GPT-OSS-120B uses MXFP4 quantization which requires the
# kernels-community/gpt-oss-triton-kernels package from HF Hub. The kernels
# library tries to validate variants over the network even when the snapshot
# is already cached. We use LOCAL_KERNELS to bypass the network entirely.
_KERNEL_SNAPSHOT = (
    "/root/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels"
    "/snapshots/143c69c7b43f14f88c051fde44d0c89bed7aa813"
)
if Path(_KERNEL_SNAPSHOT).exists():
    os.environ.setdefault(
        "LOCAL_KERNELS",
        f"kernels-community/gpt-oss-triton-kernels={_KERNEL_SNAPSHOT}",
    )
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pyarrow.parquet as pq
import torch

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR / "shared"))
sys.path.insert(0, "/zhutingqi/song/Plan_opus/shared")
from text_utils import strict_correct  # noqa: E402

# Use the same "Reasoning: low + short answer" system prompt as the original
# Plan_gpt55 cross-model anchor experiments. This skips GPT-OSS's analysis
# channel and produces short factual answers we can score with strict_correct.
SYSTEM_PROMPT = "Reasoning: low\nAnswer with only the short answer, no explanation."
TEACHER_MODEL_DIR = "/zhutingqi/gpt-oss-120b"


def parse_harmony(raw_text: str) -> tuple[str, str]:
    """Extract (analysis, final_answer) from GPT-OSS Harmony channel format."""
    analysis = ""
    final = ""
    m_a = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)",
                    raw_text, flags=re.S)
    m_f = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
                    raw_text, flags=re.S)
    if m_a:
        analysis = m_a.group(1).strip()
    if m_f:
        final = m_f.group(1).strip()
    if not final:
        # Fallback: strip control tokens, return whatever's left
        final = re.sub(r"<\|[^>]+?\|>", " ", raw_text).strip()
        final = " ".join(final.split())
    return analysis, final
HOTPOTQA_JSONL = "/zhutingqi/song/datasets/HotpotQA/processed/hotpotqa.dev_distractor.context1200.jsonl"
NQ_PARQUET_DIR = Path("/zhutingqi/song/datasets/nq_open/nq_open")


def load_questions_for_dataset(dataset: str) -> dict[str, dict]:
    """Return mapping qid -> {question, ideal} matching the npz qids."""
    if dataset == "hotpotqa":
        out: dict[str, dict] = {}
        with open(HOTPOTQA_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                qid = str(rec.get("id"))
                q = rec.get("question") or rec.get("raw_question")
                ideals = rec.get("ideal") or rec.get("ideal_answers") or []
                out[qid] = {"question": q, "ideals": list(ideals)}
        return out

    if dataset == "nq":
        out = {}
        files = sorted(NQ_PARQUET_DIR.glob("*.parquet"))
        # Use validation split (matches what prepare_nq_ood.py used)
        files = [p for p in files if "validation" in p.stem]
        if not files:
            files = sorted(NQ_PARQUET_DIR.glob("*.parquet"))
        for fpath in files:
            table = pq.read_table(fpath, columns=["question", "answer"])
            questions = table.column("question").to_pylist()
            answers = table.column("answer").to_pylist()
            for i, (q, a) in enumerate(zip(questions, answers)):
                if isinstance(a, str):
                    a = [a]
                if not (isinstance(q, str) and q.strip() and isinstance(a, list)):
                    continue
                ideals = [str(s).strip() for s in a if str(s).strip()]
                if not ideals:
                    continue
                qid = f"nq_open_pq_{fpath.stem}_{i}"
                out[qid] = {"question": q.strip(), "ideals": ideals}
        return out

    raise ValueError(dataset)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "nq"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-questions", type=int, default=500)
    p.add_argument("--qwen-tag", default="qwen7b",
                   help="Use this tag's npz to align question IDs.")
    args = p.parse_args()

    print(f"[load] tokenizer + GPT-OSS-120B from {TEACHER_MODEL_DIR}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    t_load = time.time() - t_load_start
    print(f"[load] done in {t_load:.1f}s, device map: {set(model.hf_device_map.values()) if hasattr(model, 'hf_device_map') else 'single'}")

    for dataset in args.datasets:
        print(f"\n=== dataset = {dataset} ===")
        # 1) Get the qids the student saw, in the same order
        npz_path = THIS_DIR / "runs" / args.qwen_tag / f"{dataset}_ood.npz"
        if not npz_path.exists():
            print(f"[skip] no student cache at {npz_path}")
            continue
        student_blob = np.load(npz_path, allow_pickle=True)
        target_qids = list(student_blob["question_ids"])[:args.max_questions]
        student_correct = list(student_blob["strict_correct"][:args.max_questions])
        student_answers = list(student_blob["answers"][:args.max_questions])
        print(f"[load] {len(target_qids)} student qids from {npz_path}")

        # 2) Re-load source to recover question text
        qmap = load_questions_for_dataset(dataset)
        missing = [q for q in target_qids if q not in qmap]
        if missing:
            print(f"[warn] {len(missing)} qids missing from source dataset; first 3: {missing[:3]}")

        out_path = THIS_DIR / "runs" / f"teacher_oss120b_{dataset}.jsonl"
        n_correct = 0
        t0 = time.time()
        with out_path.open("w", encoding="utf-8") as out_f:
            for i, qid in enumerate(target_qids):
                if qid not in qmap:
                    rec = {"question_id": qid, "missing": True}
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                    continue
                question = qmap[qid]["question"]
                ideals = qmap[qid]["ideals"]
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                encoding = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                # Move to first device of the model
                device = next(model.parameters()).device
                encoding = {k: v.to(device) for k, v in encoding.items()}
                prompt_len = int(encoding["input_ids"].shape[-1])

                t_gen_start = time.time()
                with torch.inference_mode():
                    gen_ids = model.generate(
                        **encoding,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latency_ms = (time.time() - t_gen_start) * 1000

                new_tokens = gen_ids[0, prompt_len:].tolist()
                # Decode WITH special tokens so we can locate Harmony channels
                raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
                analysis, final = parse_harmony(raw_text)

                sc = bool(strict_correct(final, ideals))
                n_correct += int(sc)

                rec = {
                    "question_id": qid,
                    "question": question[:500],
                    "ideals": ideals,
                    "teacher_raw_text": raw_text[:1500],
                    "teacher_analysis": analysis[:500],
                    "teacher_final_answer": final[:500],
                    "teacher_strict_correct": int(sc),
                    "latency_ms": latency_ms,
                    "completion_tokens": len(new_tokens),
                    "prompt_tokens": prompt_len,
                    "student_strict_correct": int(student_correct[i]),
                    "student_answer": str(student_answers[i])[:500],
                }
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    avg = elapsed / (i + 1)
                    eta = avg * (len(target_qids) - i - 1)
                    acc = n_correct / (i + 1)
                    print(f"[{dataset}] {i+1}/{len(target_qids)} avg={avg:.2f}s "
                          f"eta={eta:.0f}s teacher_acc_so_far={acc:.3f} "
                          f"(latest latency={latency_ms:.0f}ms, {len(new_tokens)} toks)")

        elapsed = time.time() - t0
        print(f"[done] {dataset}: teacher acc = {n_correct}/{len(target_qids)} "
              f"= {n_correct/len(target_qids):.3f} (total {elapsed/60:.1f} min)")
        print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
