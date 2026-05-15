"""Routing analysis: compare best Plan_opus_selective predictor against
single-feature baselines and trivial constant policies.

For each ``best_predictions_<setting>.csv`` produced by
``run_selective_experiments.py`` we sweep (answer_threshold, defer_threshold)
pairs over multiple confidence sources, compute the per-item utility under
several cost models, and report:

- Pareto frontier (utility vs total coverage)
- Best operating point for each method
- Comparison against trivial baselines (always answer / always teacher / always abstain)
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
sys.path.insert(0, str(SHARED_DIR))

from routing import CostModel, best_constant_baselines, utility_curve  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="/zhutingqi/song/Plan_opus_selective/results")
    p.add_argument(
        "--cost-models",
        default="default,strict,lenient",
        help="Comma-separated names of cost models to sweep.",
    )
    p.add_argument("--n-thresholds", type=int, default=21)
    return p.parse_args()


COST_MODELS: dict[str, CostModel] = {
    "default": CostModel(correct_reward=1.0, wrong_penalty=3.0, teacher_cost=0.3, abstain_value=0.0),
    "strict": CostModel(correct_reward=1.0, wrong_penalty=5.0, teacher_cost=0.4, abstain_value=0.0),
    "lenient": CostModel(correct_reward=1.0, wrong_penalty=1.5, teacher_cost=0.2, abstain_value=0.0),
}


def load_setting(results_dir: Path, setting: str) -> dict[str, Any] | None:
    base = results_dir / f"best_predictions_{setting}.csv"
    if not base.exists():
        return None
    df = pd.read_csv(base)
    extras: dict[str, np.ndarray] = {}
    for path in results_dir.glob(f"predictions_{setting}_*.csv"):
        label = path.stem.replace(f"predictions_{setting}_", "")
        other = pd.read_csv(path)
        merged = df.merge(other[["question_id", "seed", "confidence"]], on=["question_id", "seed"], suffixes=("", f"_{label}"))
        extras[label] = merged[f"confidence_{label}"].to_numpy(dtype=np.float64)
    return {"df": df, "extras": extras}


def best_operating_point(records: dict[str, np.ndarray]) -> dict[str, float]:
    idx = int(np.argmax(records["utility_per_item"]))
    return {key: float(value[idx]) for key, value in records.items()}


def run(results_dir: Path, args: argparse.Namespace) -> None:
    summary_rows: list[dict[str, Any]] = []
    pareto_rows: list[dict[str, Any]] = []
    cost_names = [name.strip() for name in args.cost_models.split(",") if name.strip()]
    for setting in ("sample0", "fixed_k"):
        loaded = load_setting(results_dir, setting)
        if loaded is None:
            continue
        df = loaded["df"]
        correct = df["label"].astype(np.float64).to_numpy()
        teacher_correct_raw = df["teacher_correct"].to_numpy()
        teacher_correct = (
            np.where(np.isfinite(teacher_correct_raw), teacher_correct_raw, 0.0)
            if "teacher_correct" in df.columns
            else None
        )
        candidate_scores = {"best": df["confidence"].to_numpy(dtype=np.float64), **loaded["extras"]}
        for cost_name in cost_names:
            cost = COST_MODELS[cost_name]
            constants = best_constant_baselines(correct, teacher_correct, cost)
            for const_name, value in constants.items():
                summary_rows.append(
                    {
                        "setting": setting,
                        "cost_model": cost_name,
                        "method": const_name,
                        "utility_per_item": value,
                        "answer_count": float(0 if "abstain" in const_name else (df.shape[0] if "answer" in const_name else 0)),
                        "teacher_count": float(df.shape[0] if "teacher" in const_name else 0),
                        "abstain_count": float(df.shape[0] if "abstain" in const_name else 0),
                        "coverage_self": 1.0 if "answer" in const_name else 0.0,
                        "coverage_total": 1.0 if "answer" in const_name or "teacher" in const_name else 0.0,
                        "answer_accuracy": float("nan"),
                        "teacher_accuracy": float("nan"),
                        "answer_threshold": float("nan"),
                        "defer_threshold": float("nan"),
                    }
                )

            for label, scores in candidate_scores.items():
                lo, hi = float(np.nanmin(scores)), float(np.nanmax(scores))
                if lo == hi:
                    continue
                grid_a = np.linspace(lo, hi, args.n_thresholds)
                grid_d = np.linspace(lo, hi, args.n_thresholds)
                records = utility_curve(scores, correct, teacher_correct, cost, grid_a, grid_d)
                best = best_operating_point(records)
                summary_rows.append(
                    {
                        "setting": setting,
                        "cost_model": cost_name,
                        "method": label,
                        **best,
                    }
                )
                # Build a Pareto frontier on (coverage_total, utility_per_item).
                order = np.argsort(records["coverage_total"])
                coverages = records["coverage_total"][order]
                utilities = records["utility_per_item"][order]
                best_so_far = -np.inf
                for c, u, a, d in zip(coverages, utilities, records["answer_threshold"][order], records["defer_threshold"][order]):
                    if u > best_so_far:
                        pareto_rows.append(
                            {
                                "setting": setting,
                                "cost_model": cost_name,
                                "method": label,
                                "coverage_total": float(c),
                                "utility_per_item": float(u),
                                "answer_threshold": float(a),
                                "defer_threshold": float(d),
                            }
                        )
                        best_so_far = u
    summary_df = pd.DataFrame(summary_rows)
    pareto_df = pd.DataFrame(pareto_rows)
    summary_df.to_csv(results_dir / "routing_summary.csv", index=False)
    pareto_df.to_csv(results_dir / "routing_pareto.csv", index=False)
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)))


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    run(results_dir, args)


if __name__ == "__main__":
    main()
