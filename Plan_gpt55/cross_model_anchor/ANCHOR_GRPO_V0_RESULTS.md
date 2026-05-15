# Anchor-Aware GRPO v0 Results

## Motivation

VBPO/DPO trains on offline preference pairs and does not directly optimize the rollout candidate distribution. Anchor-aware GRPO was tested because basin behavior is naturally group-relative: generate several answers, score each answer/basin, normalize rewards within the group, and update online.

## Implementation

- Script: `scripts/train_anchor_grpo.py`
- Config: `configs/anchor_grpo_v0_triviaqa_full500.json`
- Run dir: `runs/run_20260513_071012_anchor_grpo_v0_triviaqa_full500`
- Teacher basins: `run_20260513_012413_gptoss_anchor_triviaqa_full500_k4_tok256/teacher_basin_rows_final_only.csv`

The reward is anchor-only for optimization:

```text
R(y) =
  anchor_support
  + 0.25 * teacher_similarity
  + 0.25 * anchored_consensus
  + 0.05 * teacher_greedy_support
  - 0.75 * qwen_only_stable
  - 0.04 * length_cost
```

Gold labels are logged as `strict` and `f1`, but are not used in the policy reward.

## Training Observations

Training ran for 100 steps, group size 6, 4 questions per step.

The online reward had real variation, but coverage was uneven:

- Some batches had high anchor support and high strict rate.
- Some batches had almost no teacher-supported rollout, leaving the update mostly driven by `qwen_only_stable` and length.
- KL stayed small, while gradients were sometimes large, suggesting noisy local updates rather than smooth distribution movement.

Representative dev sample0 metrics:

| step | dev anchor reward | dev anchor support | dev qwen-only stable | dev strict |
|---:|---:|---:|---:|---:|
| 50 | 0.404 | 0.508 | 0.492 | 0.594 |
| 100 | 0.405 | 0.508 | 0.492 | 0.594 |

The dev anchor reward did not improve after step50.

## Heldout Evaluation

Protocol: TriviaQA `offset500/test500`, no-leak fixed-k evaluation.

Average over seeds 2026 and 42 for step50:

| variant | sample0 | fixed2 | fixed4 | fixed8 |
|---|---:|---:|---:|---:|
| base | 0.661 | 0.663 | 0.664 | 0.677 |
| full Anchor-VBPO v1 step100 | **0.677** | **0.682** | **0.679** | **0.678** |
| Anchor-GRPO v0 step50 | 0.660 | 0.664 | 0.667 | 0.672 |

Seed2026 step100 was worse than step50:

| checkpoint | sample0 | fixed2 | fixed4 | fixed8 |
|---|---:|---:|---:|---:|
| step50 seed2026 | 0.656 | 0.662 | 0.668 | 0.676 |
| step100 seed2026 | 0.656 | 0.656 | 0.662 | 0.666 |

## Reflection

The method match is conceptually better than DPO, but v0 reward is not yet strong enough.

Main issues:

1. **Sparse positive anchor reward in online rollouts.** If the current policy does not sample a teacher-supported basin, the group-relative update cannot learn much positive direction.
2. **Reward is too binary around the alignment threshold.** Most useful AUC information is candidate-level continuous ranking, but online reward uses thresholded basin support. Near-miss answers receive weak or unstable gradients.
3. **No explicit rescue from teacher answer.** GRPO only updates on model-generated candidates. If the model never samples the anchor basin for a question, the teacher anchor cannot directly pull it there.
4. **Heldout reward mismatch remains.** The reward optimizes teacher-basin similarity on train-anchor questions, while heldout evaluation is exact-answer TriviaQA on offset500.

## Current Takeaway

Anchor-aware GRPO v0 does not beat base or Anchor-VBPO v1. The best current model remains full Anchor-VBPO v1 step100.

However, this result is informative: online GRPO needs either better exploration or a denser anchor reward. The next promising version should:

- use continuous similarity/support, not only thresholded support;
- add a teacher-guided sampled candidate into each group, or mix offline teacher-anchor distillation with online GRPO;
- train only on questions where the current policy occasionally samples near-anchor answers, at least for v1;
- report reward variance per question and skip zero-variance groups.
