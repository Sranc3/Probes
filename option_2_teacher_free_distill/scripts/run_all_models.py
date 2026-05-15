"""Multi-base-model orchestrator for teacher-free probe experiments.

Runs the full pipeline for each base model in BASES:

  1. Extract Qwen prompt-only hidden states on TriviaQA  (ID training cache)
  2. Train probes (DCP / SEPs-LR / SEPs-Ridge / ARD-Ridge / ARD-MLP) across layers
  3. Evaluate ID metrics + bootstrap CIs vs Plan_opus_selective baselines
  4. Build HotpotQA OOD cache (hidden + greedy + strict_correct)
  5. Build NQ-Open OOD cache  (hidden + greedy + strict_correct)
  6. Multi-OOD evaluation (HotpotQA + NQ together) with paired bootstraps

All paths are namespaced under ``runs/<tag>/`` and ``results/<tag>/`` so
multiple bases coexist. Each step is *resumable*: if the output file
already exists and ``--force`` is not passed, that step is skipped.

Per-step stdout/stderr is tee'd to ``runs/<tag>/run_master.log`` so the
final artefact has a complete audit trail.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

THIS_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = THIS_DIR / "scripts"
RUNS = THIS_DIR / "runs"
RESULTS = THIS_DIR / "results"

PYTHON = sys.executable  # use whichever Python is running this orchestrator (conda env-aware)

NQ_DATASET = "/zhutingqi/song/datasets/nq_open"


@dataclass
class BaseConfig:
    tag: str
    model_dir: str
    layers: list[int]
    hidden_size: int
    num_hidden_layers: int
    notes: str = ""
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base-model registry
# ---------------------------------------------------------------------------
# Layer choice strategy: 8 layers spanning normalised depth ~0.05 .. 1.0,
# anchoring on Qwen-7B's known sweet spot of L20/28 = 0.71. For 80-layer
# Qwen-72B, the same normalised depth corresponds to L57; we therefore
# include {48, 56, 64} densely around the predicted sweet spot.

BASES: list[BaseConfig] = [
    BaseConfig(
        tag="qwen7b",
        model_dir="/zhutingqi/song/qwen_model/model",
        layers=[4, 8, 12, 16, 20, 24, 27, 28],
        hidden_size=3584,
        num_hidden_layers=28,
        notes="ID baseline; results pre-existing, only re-run if --force",
    ),
    BaseConfig(
        tag="llama3b",
        model_dir="/zhutingqi/Llama-3.2-3B-Instruct",
        layers=[4, 8, 12, 16, 20, 24, 27, 28],
        hidden_size=3072,
        num_hidden_layers=28,
        notes="28 layers — same indices as Qwen-7B for direct comparison",
    ),
    BaseConfig(
        tag="qwen72b",
        model_dir="/zhutingqi/Qwen2.5-72B-Instruct",
        # Equivalent to Qwen-7B layers {4,8,12,16,20,24,27,28} after rescaling
        # to 80 layers: {11, 23, 34, 46, 57, 69, 77, 80} -> we pick a smooth
        # 8-point sweep that always includes the predicted sweet spot ~57.
        layers=[8, 16, 24, 32, 40, 48, 56, 64, 72, 80],
        hidden_size=8192,
        num_hidden_layers=80,
        notes="10 layers (vs 8 for smaller bases) so we can locate the sweet spot",
    ),
]


# ---------------------------------------------------------------------------
# Step runner with logging
# ---------------------------------------------------------------------------


class StepFailed(RuntimeError):
    pass


def tee_run(name: str, cmd: str, log_path: Path, master_log: Path, force: bool, output_check: Path | None) -> bool:
    """Run a shell command, tee output to two log files. Returns True if executed, False if skipped."""
    if output_check is not None and output_check.exists() and not force:
        msg = f"[skip] {name}: output exists ({output_check}) — pass --force to overwrite"
        print(msg, flush=True)
        with master_log.open("a") as f:
            f.write(f"\n{msg}\n")
        return False
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[run ] {name}\n       cmd: {cmd}\n       log: {log_path}", flush=True)
    with master_log.open("a") as f:
        f.write(f"\n{'='*72}\n[run ] {name}\n[time] {time.strftime('%Y-%m-%d %H:%M:%S')}\n[cmd ] {cmd}\n[log ] {log_path}\n{'='*72}\n")
    t0 = time.time()
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    tail = log_path.read_text().splitlines()[-15:]
    summary = f"[done] {name}: exit={proc.returncode}, elapsed={dt:.1f}s"
    print(summary, flush=True)
    for line in tail:
        print(f"       | {line}", flush=True)
    with master_log.open("a") as f:
        f.write(summary + "\n")
        for line in tail:
            f.write(f"  | {line}\n")
    if proc.returncode != 0:
        raise StepFailed(f"{name} failed (exit {proc.returncode}); see {log_path}")
    return True


# ---------------------------------------------------------------------------
# Per-base pipeline
# ---------------------------------------------------------------------------


def run_for_base(b: BaseConfig, force: bool, max_questions: int) -> None:
    runs_dir = RUNS / b.tag
    res_dir = RESULTS / b.tag
    runs_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    master = runs_dir / "run_master.log"
    master.write_text(
        f"# Master log for base = {b.tag}\n"
        f"model_dir         : {b.model_dir}\n"
        f"num_hidden_layers : {b.num_hidden_layers}\n"
        f"hidden_size       : {b.hidden_size}\n"
        f"layers            : {b.layers}\n"
        f"max_questions     : {max_questions}\n"
        f"started_at        : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    layers_arg = " ".join(map(str, b.layers))
    n_extra = f" --max-questions {max_questions}" if max_questions != 500 else ""

    # 1. ID hidden states
    hidden_path = runs_dir / "hidden_states.npz"
    cmd1 = (
        f"{PYTHON} {SCRIPTS / 'extract_hidden_states.py'}"
        f" --model-dir {shlex.quote(b.model_dir)}"
        f" --output {hidden_path}"
        f" --layers {layers_arg}"
    )
    tee_run(f"[{b.tag}] 1/6 extract TriviaQA hidden states", cmd1, runs_dir / "extract.log", master, force, hidden_path)

    # 2. Train probes
    pred_path = runs_dir / "probe_predictions.csv"
    cmd2 = (
        f"{PYTHON} {SCRIPTS / 'train_probes.py'}"
        f" --hidden-states {hidden_path}"
        f" --out-dir {runs_dir}"
    )
    tee_run(f"[{b.tag}] 2/6 train probes (DCP/SEPs/ARD across layers)", cmd2, runs_dir / "train.log", master, force, pred_path)

    # 3. Evaluate ID metrics
    best_path = res_dir / "best_per_probe.csv"
    cmd3 = (
        f"{PYTHON} {SCRIPTS / 'evaluate_probes.py'}"
        f" --predictions {pred_path}"
        f" --out-dir {res_dir}"
    )
    tee_run(f"[{b.tag}] 3/6 evaluate ID probes", cmd3, runs_dir / "eval_id.log", master, force, best_path)

    # 3b. ID bootstrap
    boot_path = res_dir / "bootstrap_pairs.csv"
    cmd3b = (
        f"{PYTHON} {SCRIPTS / 'bootstrap_compare.py'}"
        f" --probe-predictions {pred_path}"
        f" --output {boot_path}"
        f" --n-boot 2000"
    )
    tee_run(f"[{b.tag}] 3b/6 ID paired bootstrap", cmd3b, runs_dir / "boot_id.log", master, force, boot_path)

    # 4. HotpotQA OOD cache
    hpq_cache = runs_dir / "hotpotqa_ood.npz"
    cmd4 = (
        f"{PYTHON} {SCRIPTS / 'prepare_hotpotqa_ood.py'}"
        f" --model-dir {shlex.quote(b.model_dir)}"
        f" --output {hpq_cache}"
        f" --layers {layers_arg}"
    )
    tee_run(f"[{b.tag}] 4/6 prepare HotpotQA OOD cache", cmd4, runs_dir / "hotpotqa_prep.log", master, force, hpq_cache)

    # 5. NQ OOD cache
    nq_cache = runs_dir / "nq_ood.npz"
    cmd5 = (
        f"{PYTHON} {SCRIPTS / 'prepare_nq_ood.py'}"
        f" --input {NQ_DATASET}"
        f" --split validation"
        f" --max-questions 500"
        f" --model-dir {shlex.quote(b.model_dir)}"
        f" --output {nq_cache}"
        f" --layers {layers_arg}"
    )
    tee_run(f"[{b.tag}] 5/6 prepare NQ-Open OOD cache", cmd5, runs_dir / "nq_prep.log", master, force, nq_cache)

    # 6. Multi-OOD eval
    ood_combined = res_dir / "ood_combined_table.md"
    cmd6 = (
        f"{PYTHON} {SCRIPTS / 'evaluate_ood.py'}"
        f" --triviaqa-hidden {hidden_path}"
        f" --ood-cache hotpotqa={hpq_cache}"
        f" --ood-cache nq={nq_cache}"
        f" --out-dir {res_dir}"
        f" --n-boot 2000"
    )
    tee_run(f"[{b.tag}] 6/6 multi-OOD evaluation + bootstrap", cmd6, runs_dir / "eval_ood.log", master, force, ood_combined)

    with master.open("a") as f:
        f.write(f"\nfinished_at : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bases",
        nargs="*",
        default=None,
        help=f"Subset of base tags to run. Default: all of {[b.tag for b in BASES]}",
    )
    p.add_argument("--force", action="store_true", help="Re-run even if outputs already exist")
    p.add_argument("--max-questions", type=int, default=500, help="ID set size (TriviaQA cache builds 500x2 rows)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.bases or [b.tag for b in BASES]
    run_set = [b for b in BASES if b.tag in selected]
    print(f"=== Multi-base orchestrator ===")
    for b in run_set:
        print(f"  - {b.tag}: {b.model_dir} (layers={b.layers})")

    overall_t0 = time.time()
    failures: list[str] = []
    for b in run_set:
        print(f"\n\n#####  BASE = {b.tag}  #####")
        try:
            run_for_base(b, force=args.force, max_questions=args.max_questions)
        except StepFailed as e:
            print(f"!!! {b.tag} failed: {e}", flush=True)
            failures.append(b.tag)
        except Exception as e:
            print(f"!!! {b.tag} crashed: {e!r}", flush=True)
            failures.append(b.tag)

    dt = time.time() - overall_t0
    print(f"\n\n=== Done in {dt/60:.1f} min ===")
    if failures:
        print(f"Bases with failures: {failures}")
        sys.exit(1)
    print("All bases completed successfully.")


if __name__ == "__main__":
    main()
