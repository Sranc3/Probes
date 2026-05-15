"""3-action routing analysis: {answer, call_teacher, abstain}.

For every candidate confidence score we sweep two thresholds:

- ``answer_threshold`` (``t_a``): if ``confidence >= t_a`` -> answer
- ``defer_threshold``  (``t_d``): if ``t_d <= confidence < t_a`` -> call_teacher
- otherwise -> abstain

Utility per item:
    answer-correct        : +1
    answer-wrong          : -alpha
    teacher-correct       : +1 - cost_teacher
    teacher-wrong         : -alpha - cost_teacher
    abstain               : 0

We assume ``teacher_correct`` is given by ``teacher_best_basin_strict_any``
which is the noiseless oracle on whether the teacher's basin is strict-correct
(captured from gpt-oss anchor table). When that's missing we treat teacher as
having accuracy ``teacher_acc_default``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CostModel:
    correct_reward: float = 1.0
    wrong_penalty: float = 3.0
    teacher_cost: float = 0.3
    abstain_value: float = 0.0
    teacher_acc_default: float = 0.7


def utility_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
    teacher_correct: np.ndarray | None,
    cost: CostModel,
    answer_thresholds: Iterable[float] | None = None,
    defer_thresholds: Iterable[float] | None = None,
) -> dict[str, np.ndarray]:
    """Sweep (answer_threshold, defer_threshold) pairs; return arrays."""
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if teacher_correct is None:
        teacher_correct_arr = np.full_like(correct, fill_value=cost.teacher_acc_default)
    else:
        teacher_correct_arr = np.asarray(teacher_correct, dtype=np.float64)
    if answer_thresholds is None:
        answer_thresholds = np.linspace(np.min(confidences), np.max(confidences), 21)
    if defer_thresholds is None:
        defer_thresholds = np.linspace(np.min(confidences), np.max(confidences), 21)
    answer_thresholds = np.asarray(list(answer_thresholds), dtype=np.float64)
    defer_thresholds = np.asarray(list(defer_thresholds), dtype=np.float64)

    records: list[dict[str, float]] = []
    for t_a in answer_thresholds:
        for t_d in defer_thresholds:
            if t_d > t_a:
                continue
            answer_mask = confidences >= t_a
            teacher_mask = (confidences < t_a) & (confidences >= t_d)
            abstain_mask = confidences < t_d
            n = correct.size
            answer_count = int(answer_mask.sum())
            teacher_count = int(teacher_mask.sum())
            abstain_count = int(abstain_mask.sum())
            answer_correct = float(correct[answer_mask].sum()) if answer_count else 0.0
            answer_wrong = answer_count - answer_correct
            teacher_correct_count = float(teacher_correct_arr[teacher_mask].sum()) if teacher_count else 0.0
            teacher_wrong = teacher_count - teacher_correct_count
            utility = (
                cost.correct_reward * answer_correct
                - cost.wrong_penalty * answer_wrong
                + (cost.correct_reward - cost.teacher_cost) * teacher_correct_count
                - (cost.wrong_penalty + cost.teacher_cost) * teacher_wrong
                + cost.abstain_value * abstain_count
            )
            coverage_self = answer_count / n
            coverage_total = (answer_count + teacher_count) / n
            answer_acc = answer_correct / max(1, answer_count)
            teacher_acc = teacher_correct_count / max(1, teacher_count)
            records.append(
                {
                    "answer_threshold": float(t_a),
                    "defer_threshold": float(t_d),
                    "answer_count": float(answer_count),
                    "teacher_count": float(teacher_count),
                    "abstain_count": float(abstain_count),
                    "coverage_self": float(coverage_self),
                    "coverage_total": float(coverage_total),
                    "answer_accuracy": float(answer_acc),
                    "teacher_accuracy": float(teacher_acc),
                    "utility_total": float(utility),
                    "utility_per_item": float(utility / n),
                }
            )
    return {key: np.array([r[key] for r in records]) for key in records[0].keys()}


def best_constant_baselines(
    correct: np.ndarray,
    teacher_correct: np.ndarray | None,
    cost: CostModel,
) -> dict[str, float]:
    """Return per-item utility of trivial policies (always answer / always teacher / always abstain)."""
    correct = np.asarray(correct, dtype=np.float64)
    if teacher_correct is None:
        teacher_correct_arr = np.full_like(correct, fill_value=cost.teacher_acc_default)
    else:
        teacher_correct_arr = np.asarray(teacher_correct, dtype=np.float64)
    n = correct.size
    answer_correct = float(correct.sum())
    answer_utility = (cost.correct_reward * answer_correct - cost.wrong_penalty * (n - answer_correct)) / n
    teacher_correct_count = float(teacher_correct_arr.sum())
    teacher_utility = (
        (cost.correct_reward - cost.teacher_cost) * teacher_correct_count
        - (cost.wrong_penalty + cost.teacher_cost) * (n - teacher_correct_count)
    ) / n
    return {
        "always_answer_utility": answer_utility,
        "always_teacher_utility": teacher_utility,
        "always_abstain_utility": float(cost.abstain_value),
    }
