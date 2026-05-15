"""Prepare NaturalQuestions OOD evaluation data for the teacher-free probes.

Auto-detects three NQ formats and converts them to the same
``{question, ideal}`` structure used elsewhere in this project, then runs the
exact same pipeline as ``prepare_hotpotqa_ood.py``:

1. Single Qwen forward pass on the prompt -> hidden state at layer set
2. Greedy generation of sample0 -> ``strict_correct`` against ``ideal``

Supported formats (auto-detected by file extension and column names):

A. **NQ-Open jsonl** (Lee et al. 2019, recommended):
   one record per line with ``question`` (str) and ``answer`` (list[str]).
B. **NQ-Simplified validation parquet** (subset of raw NQ kept simple):
   columns include ``question_text`` and ``annotations.short_answers``.
C. **NQ-raw parquet** (full Wikipedia HTML, what huggingface ships):
   columns ``question`` (struct), ``annotations`` (sequence with
   ``short_answers`` and ``yes_no_answer``). We extract
   ``question.text`` and ``annotations[0].short_answers[0].text`` and skip
   yes/no questions and questions without any short answer.

Outputs an ``.npz`` with the same schema as ``hotpotqa_ood.npz`` so
``evaluate_ood.py`` can consume it directly via ``--hotpotqa-cache``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

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
        "--input",
        required=True,
        help=(
            "Path to NQ data: either a single .jsonl (NQ-Open style), a single "
            ".parquet, or a directory containing one or more .parquet files."
        ),
    )
    p.add_argument(
        "--format",
        choices=["auto", "nq_open", "nq_open_parquet", "nq_simplified", "nq_raw"],
        default="auto",
    )
    p.add_argument("--model-dir", default="/zhutingqi/song/qwen_model/model")
    p.add_argument(
        "--output",
        default=str(THIS_DIR / "runs" / "nq_ood.npz"),
    )
    p.add_argument("--max-questions", type=int, default=500)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument(
        "--filter-yesno",
        action="store_true",
        default=True,
        help="(NQ-raw) Skip questions where yes_no_answer != -1",
    )
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed for raw-NQ subset")
    p.add_argument(
        "--split",
        default="validation",
        help="(parquet only) Substring filter on parquet filenames, e.g. 'validation' or 'train'.",
    )
    return p.parse_args()


def to_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def detect_format(input_path: Path, hint: str) -> str:
    if hint != "auto":
        return hint
    if input_path.is_file() and input_path.suffix == ".jsonl":
        return "nq_open"
    # parquet path / directory: peek at a single file's columns
    files = list_parquets(input_path)
    if not files:
        # Maybe input is a parent dir; try one level deeper
        if input_path.is_dir():
            for sub in input_path.iterdir():
                if sub.is_dir():
                    inner = list_parquets(sub)
                    if inner:
                        files = inner
                        break
        if not files:
            raise ValueError(f"No parquet files found at {input_path}")
    import pyarrow.parquet as pq
    first = pq.read_table(files[0])
    cols = set(first.column_names)
    if {"question", "annotations", "document"}.issubset(cols):
        return "nq_raw"
    if "question_text" in cols or "short_answers" in cols:
        return "nq_simplified"
    if {"question", "answer"}.issubset(cols):
        # NQ-Open in parquet form (question: str, answer: list[str])
        return "nq_open_parquet"
    raise ValueError(f"Cannot auto-detect NQ format from columns {cols}")


def list_parquets(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".parquet" else []
    return sorted(path.glob("*.parquet"))


def load_nq_open(path: Path, max_n: int) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get("question") or obj.get("question_text")
            a = obj.get("answer") or obj.get("answers") or obj.get("ideal")
            if isinstance(a, str):
                a = [a]
            if not (isinstance(q, str) and q.strip() and isinstance(a, list) and any(s for s in a if str(s).strip())):
                continue
            out.append({
                "id": obj.get("id", f"nq_open_{i}"),
                "question": q.strip(),
                "ideal": [str(s).strip() for s in a if str(s).strip()],
            })
            if len(out) >= max_n:
                break
    return out


def load_nq_open_parquet(parquet_files: list[Path], max_n: int) -> list[dict]:
    """NQ-Open distributed as parquet with columns {question: str, answer: list[str]}."""
    import pyarrow.parquet as pq
    out: list[dict] = []
    for fpath in parquet_files:
        if len(out) >= max_n:
            break
        df = pq.read_table(fpath, columns=["question", "answer"]).to_pandas()
        for i, row in df.iterrows():
            if len(out) >= max_n:
                break
            q = row.get("question")
            a = row.get("answer")
            if hasattr(a, "tolist"):
                a = a.tolist()
            if isinstance(a, str):
                a = [a]
            if not (isinstance(q, str) and q.strip() and isinstance(a, list)):
                continue
            ideals = [str(s).strip() for s in a if str(s).strip()]
            if not ideals:
                continue
            out.append({
                "id": f"nq_open_pq_{fpath.stem}_{i}",
                "question": q.strip(),
                "ideal": ideals,
            })
    return out


def load_nq_raw(parquet_files: list[Path], max_n: int, filter_yesno: bool, seed: int) -> list[dict]:
    """Extract {question, ideal} from raw NQ parquet (Wikipedia + bytes + annotations)."""
    import pyarrow.parquet as pq
    out: list[dict] = []
    rng = np.random.default_rng(seed)
    for fpath in parquet_files:
        if len(out) >= max_n:
            break
        df = pq.read_table(fpath, columns=["id", "question", "annotations"]).to_pandas()
        # Shuffle within file so we don't take only the first questions of each file
        idx = rng.permutation(len(df))
        for i in idx:
            row = df.iloc[i]
            qstruct = row["question"]
            ann = row["annotations"]
            if isinstance(qstruct, dict):
                question = str(qstruct.get("text", "")).strip()
            else:
                question = str(qstruct).strip()
            if not question:
                continue
            ideals = _extract_ideals_from_raw_annotations(ann, filter_yesno=filter_yesno)
            if not ideals:
                continue
            out.append({
                "id": str(row.get("id", f"nq_raw_{fpath.stem}_{i}")),
                "question": question,
                "ideal": ideals,
            })
            if len(out) >= max_n:
                break
    return out


def _extract_ideals_from_raw_annotations(ann: Any, filter_yesno: bool) -> list[str]:
    """Pull the first non-empty short_answer text from the (sequence-of-struct) annotations."""
    if ann is None:
        return []
    # NQ raw: annotations is a sequence with one entry per annotator
    # In pyarrow → pandas land it becomes a dict-of-lists with parallel arrays
    if isinstance(ann, dict):
        yn = ann.get("yes_no_answer", None)
        sa = ann.get("short_answers", None)
        if filter_yesno and yn is not None:
            try:
                yn_arr = np.asarray(yn).reshape(-1)
                if yn_arr.size > 0 and int(yn_arr[0]) != -1:
                    return []
            except (TypeError, ValueError):
                pass
        if sa is None:
            return []
        # sa is itself a list-of-dicts (one per annotator), each dict with parallel arrays
        ideals: list[str] = []
        # Try to handle both "list of dicts" and "single dict with parallel lists"
        if isinstance(sa, list):
            for entry in sa:
                if isinstance(entry, dict):
                    texts = entry.get("text", None)
                    if texts is not None:
                        for t in np.asarray(texts).reshape(-1).tolist():
                            t = str(t).strip()
                            if t:
                                ideals.append(t)
        elif isinstance(sa, dict):
            texts = sa.get("text", None)
            if texts is not None:
                for t in np.asarray(texts).reshape(-1).tolist():
                    t = str(t).strip()
                    if t:
                        ideals.append(t)
        # Dedup while preserving order
        seen, dedup = set(), []
        for t in ideals:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                dedup.append(t)
        return dedup
    # Fallback: list of dicts directly
    if isinstance(ann, list):
        ideals: list[str] = []
        for entry in ann:
            if not isinstance(entry, dict):
                continue
            yn = entry.get("yes_no_answer", -1)
            if filter_yesno and int(yn) != -1:
                continue
            sa = entry.get("short_answers") or []
            for s in sa:
                if isinstance(s, dict):
                    texts = s.get("text", None)
                    if texts is not None:
                        for t in np.asarray(texts).reshape(-1).tolist():
                            t = str(t).strip()
                            if t:
                                ideals.append(t)
        return list(dict.fromkeys(ideals))
    return []


def load_nq_simplified(parquet_files: list[Path], max_n: int) -> list[dict]:
    """NQ simplified: question_text + short_answers (already de-Wikipediafied)."""
    import pyarrow.parquet as pq
    out: list[dict] = []
    for fpath in parquet_files:
        if len(out) >= max_n:
            break
        df = pq.read_table(fpath).to_pandas()
        for _, row in df.iterrows():
            q = str(row.get("question_text", row.get("question", ""))).strip()
            sa = row.get("short_answers", row.get("annotations", None))
            if not q:
                continue
            ideals: list[str] = []
            if sa is None:
                continue
            if isinstance(sa, dict):
                texts = sa.get("text", None) or sa.get("texts", None)
                if texts is not None:
                    for t in np.asarray(texts).reshape(-1).tolist():
                        t = str(t).strip()
                        if t:
                            ideals.append(t)
            elif isinstance(sa, list):
                for entry in sa:
                    if isinstance(entry, str):
                        if entry.strip():
                            ideals.append(entry.strip())
                    elif isinstance(entry, dict):
                        texts = entry.get("text", None) or entry.get("texts", None)
                        if texts is not None:
                            for t in np.asarray(texts).reshape(-1).tolist():
                                t = str(t).strip()
                                if t:
                                    ideals.append(t)
            if not ideals:
                continue
            out.append({
                "id": str(row.get("id", f"nq_simp_{fpath.stem}_{len(out)}")),
                "question": q,
                "ideal": list(dict.fromkeys(ideals)),
            })
            if len(out) >= max_n:
                break
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    fmt = detect_format(input_path, args.format)
    print(f"[load] detected format = {fmt}")

    if fmt == "nq_open":
        records = load_nq_open(input_path, args.max_questions)
    elif fmt == "nq_open_parquet":
        # Resolve parquet files; allow input_path to be either the file, the
        # subfolder, or the parent folder containing the subfolder.
        files = list_parquets(input_path)
        if not files and input_path.is_dir():
            for sub in sorted(input_path.iterdir()):
                if sub.is_dir():
                    inner = list_parquets(sub)
                    if inner:
                        files = inner
                        break
        if args.split:
            files = [f for f in files if args.split in f.name]
        print(f"[load] found {len(files)} parquet files (split filter='{args.split or '*'}')")
        records = load_nq_open_parquet(files, args.max_questions)
    elif fmt == "nq_raw":
        files = list_parquets(input_path)
        print(f"[load] found {len(files)} parquet files")
        records = load_nq_raw(files, args.max_questions, args.filter_yesno, args.seed)
    elif fmt == "nq_simplified":
        files = list_parquets(input_path)
        print(f"[load] found {len(files)} parquet files")
        records = load_nq_simplified(files, args.max_questions)
    else:
        raise ValueError(f"unknown format {fmt}")

    print(f"[load] kept {len(records)} usable NQ records")
    if not records:
        raise SystemExit("No usable NQ records after filtering.")

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
        question = rec["question"]
        ideals = list(rec["ideal"])
        qid = str(rec.get("id", f"nq_{i}"))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoding = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
        prompt_len = int(encoding["input_ids"].shape[-1])
        prompt_lengths[i] = prompt_len

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
