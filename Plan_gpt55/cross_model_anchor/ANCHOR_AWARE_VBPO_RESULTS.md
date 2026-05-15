# Anchor-Aware VBPO v0 Results

## Run

- Trainer: `scripts/train_anchor_vbpo.py`
- Config: `configs/anchor_vbpo_v0_triviaqa_gptoss_pilot100.json`
- Run dir: `runs/run_20260513_005408_anchor_vbpo_v0_triviaqa_gptoss_pilot100`
- Checkpoint used for eval: `checkpoints/step_0090`
- Training data: GPT-OSS final-only anchor pilot, 100 TriviaQA questions

## Pair Manifest

| split | pairs | questions | mean weight | chosen teacher support | rejected teacher support | rejected qwen-only stable |
|---|---:|---:|---:|---:|---:|---:|
| train | 66 | 19 | 2.111 | 0.513 | 0.000 | 0.419 |
| dev | 19 | 5 | 2.428 | 0.947 | 0.000 | 0.177 |

Train pair types:

| pair type | count |
|---|---:|
| `teacher_anchor_vs_student_only` | 29 |
| `anchor_correct_vs_qwen_only_stable_wrong` | 17 |
| `anchor_damage_guard` | 13 |
| `anchor_rescue` | 7 |

## Training Signal

Training remained stable through 90 steps. Step90 dev pair accuracy was `0.632`, with dev margin delta `0.0037`.

This is a small-data experiment: only 19 train questions produce preference pairs under the conservative anchor rules. The positive heldout result is therefore meaningful, but should be treated as a pilot rather than a final conclusion.

## Heldout Evaluation

Protocol: TriviaQA `offset500/test500`, no-leak fixed-k evaluation, seeds `2026` and `42`.

### Average Across Seeds

| variant | sample0 strict | fixed8 strict | sample0 reward | fixed8 reward |
|---|---:|---:|---:|---:|
| base | 0.661 | 0.677 | 0.587 | 0.590 |
| VBPO v0 | 0.668 | 0.673 | 0.596 | 0.585 |
| VBPO v0.5 | 0.663 | 0.678 | 0.591 | 0.589 |
| Anchor-aware VBPO v0 | **0.670** | **0.683** | **0.600** | **0.596** |

### Per-Seed Strict Accuracy

| variant | seed | sample0 | fixed8 |
|---|---:|---:|---:|
| base | 2026 | 0.666 | 0.680 |
| base | 42 | 0.656 | 0.674 |
| VBPO v0 | 2026 | 0.674 | 0.670 |
| VBPO v0 | 42 | 0.662 | 0.676 |
| VBPO v0.5 | 2026 | 0.662 | 0.678 |
| VBPO v0.5 | 42 | 0.664 | 0.678 |
| Anchor-aware VBPO v0 | 2026 | 0.672 | 0.684 |
| Anchor-aware VBPO v0 | 42 | 0.668 | 0.682 |

For the previous `v0.5-long240` run, only seed2026 had been evaluated at 500 heldout items. It did not improve with longer training: step120 `sample0/fixed8 = 0.660/0.676`, step180 `0.654/0.670`, step240 `0.658/0.676`. Anchor-aware v0 on the same seed is `0.672/0.684`.

## Interpretation

Anchor-aware VBPO is the first VBPO variant here to improve both sample0 and fixed8 relative to base on the two heldout seeds:

- sample0: `+0.009` over base, `+0.007` over v0.5
- fixed8: `+0.006` over base, `+0.005` over v0.5
- reward also improves over prior VBPO variants for both sample0 and fixed8

The effect size is modest, but it is directionally consistent across both seeds and across both single-sample and basin-majority decoding. This supports the anchor hypothesis more strongly than the earlier Qwen-only verifier features.

## Next Iteration

The main bottleneck is anchor coverage, not optimization. The current pilot has strong pair purity but only 66 train pairs. The next version should scale teacher anchoring from 100 questions to at least the 425-question VBPO train split, then compare:

- Anchor-aware v0 scaled, same hyperparameters
- Anchor-aware with stricter teacher final parsing
- Anchor-aware with support-gap-only no-gold pairs versus strict-correct-only pairs
