# Teacher-Guided Anchor-GRPO v1 Results

## Motivation

Anchor-GRPO v0 was conceptually closer to the basin objective than DPO/VBPO, but its online reward was sparse: if Qwen did not sample a teacher-supported basin, the group had no positive anchor candidate. v1 tests a teacher-guided group construction:

- generate `5` Qwen completions online;
- add up to `1` GPT-OSS teacher-basin representative as an extra group candidate;
- score all candidates with the same anchor reward and group-normalized advantage;
- keep gold labels only for logging/evaluation, not for reward weights.

## Implementation

- Script: `scripts/train_anchor_grpo.py`
- Config: `configs/anchor_grpo_v1_teacher_guided_full500.json`
- Run dir: `runs/run_20260513_080622_anchor_grpo_v1_teacher_guided_full500`
- Checkpoints: `step_0025`, `step_0050`, `step_0075`

Important guardrail from the smoke test: naive teacher injection can reinforce teacher mistakes, because a teacher candidate is naturally self-supported by the teacher basin. Therefore v1 only injects teacher candidates from high-confidence teacher basins:

- `mass >= 0.5`
- `greedy_member == true`
- short/clean answer text

This gives `394 / 457` eligible teacher-basin questions (`86.2%`) while filtering at least some obviously risky teacher-only answers.

## Training Observations

Training ran for `75` steps with LoRA rank `8`, LR `1e-6`, KL beta `0.06`, and group size `5 + guided`.

Dev sample0 metrics:

| step | dev reward | dev anchor support | dev qwen-only stable | dev strict |
|---:|---:|---:|---:|---:|
| 25 | 0.457 | 0.508 | 0.492 | 0.625 |
| 75 | 0.457 | 0.508 | 0.492 | 0.594 |

The guided candidate fixed v0's missing-positive-sample problem in training batches, but dev reward did not improve after step25.

## Heldout Evaluation

Protocol: TriviaQA `offset500/test500`, no-leak fixed-k evaluation.

Average over seeds `2026` and `42` for v1 step75:

| variant | sample0 | fixed2 | fixed4 | fixed8 |
|---|---:|---:|---:|---:|
| base | 0.661 | 0.663 | 0.664 | 0.677 |
| full Anchor-VBPO v1 step100 | **0.677** | **0.682** | **0.679** | 0.678 |
| Anchor-GRPO v0 step50 | 0.660 | 0.664 | 0.667 | 0.672 |
| Teacher-guided Anchor-GRPO v1 step75 | 0.657 | 0.658 | 0.669 | **0.681** |

Per-seed strict accuracy for Teacher-guided Anchor-GRPO v1:

| checkpoint | seed | sample0 | fixed2 | fixed4 | fixed8 |
|---|---:|---:|---:|---:|---:|
| step25 | 2026 | 0.664 | 0.670 | 0.676 | 0.678 |
| step75 | 2026 | 0.656 | 0.656 | 0.674 | 0.678 |
| step75 | 42 | 0.658 | 0.660 | 0.664 | 0.684 |

## Reflection

Teacher-guided GRPO v1 is a useful negative/partial result.

What improved:

- Online training now often sees a high-anchor positive candidate, so the reward signal is less sparse than v0.
- `fixed8` average reached `0.681`, slightly above base and VBPO v1, but the gain is small and not accompanied by better `sample0`/`fixed2`.

What failed:

- The method still does not beat Anchor-VBPO v1 for single-sample and low-k generation.
- Teacher injection can become self-confirming: if the teacher basin is wrong, anchor reward will still score the injected answer highly unless filtered.
- The model appears to absorb teacher-anchor style locally without reliably moving the heldout generation distribution.

## Current Takeaway

The best overall model remains full Anchor-VBPO v1 step100. Teacher-guided Anchor-GRPO v1 slightly helps fixed8 but hurts sample0/fixed2, so it is not the new best checkpoint.

The next serious GRPO attempt should not directly inject teacher text as a trainable completion. It should instead use teacher anchors to build a denser reward over Qwen-generated candidates, and train only on groups with nonzero reward variance or near-anchor Qwen samples.
