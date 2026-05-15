# Anchor-Aware VBPO v2 Results and Reflection

## Goal

v2 was designed after auditing v1. The goal was to test whether the strong cross-model anchor AUC can transfer better if the DPO supervision is cleaner.

Changes from v1:

- filtered chosen completions to short, clean answers;
- allowed `teacher_best_answer` as chosen text when Qwen candidate text was verbose;
- required `teacher_anchor_vs_student_only` to be strict-correct vs strict-wrong;
- added `teacher_correct_anchor_rescue`;
- increased beta from `0.07` to `0.12`;
- trained for 100 steps with checkpoints every 25.

## Pair Quality

Run dir: `runs/run_20260513_062253_anchor_vbpo_v2_triviaqa_gptoss_full500_clean`

v2 dry-run pair manifest:

| split | pairs | questions | chosen teacher support | rejected teacher support | rejected qwen-only stable |
|---|---:|---:|---:|---:|---:|
| train | 264 | 58 | 0.862 | 0.045 | 0.260 |
| dev | 20 | 6 | 0.925 | 0.025 | 0.275 |

Pair correctness audit:

| pair type | n | chosen correct | rejected correct | chosen mean words |
|---|---:|---:|---:|---:|
| `anchor_correct_vs_qwen_only_stable_wrong` | 121 | 1.000 | 0.000 | 11.3 |
| `anchor_rescue` | 57 | 1.000 | 0.000 | 10.6 |
| `anchor_damage_guard` | 46 | 1.000 | 0.000 | 10.8 |
| `teacher_anchor_vs_student_only` | 20 | 1.000 | 0.000 | 11.2 |
| `teacher_correct_anchor_rescue` | 20 | 1.000 | 0.000 | 1.9 |

This fixed the most obvious v1 pair-noise issue. In v1, `teacher_anchor_vs_student_only` had only `60.9%` chosen correctness and `37.0%` rejected correctness.

## Training

Training pair metrics improved strongly:

| step | dev pair accuracy | dev margin delta |
|---:|---:|---:|
| 25 | 0.350 | -0.011 |
| 75 | 0.750 | 0.018 |
| 100 | 0.800 | 0.058 |

So v2 supervision was learnable by the model.

## Heldout Evaluation

Protocol: TriviaQA `offset500/test500`, no-leak fixed-k evaluation, seeds `2026` and `42`.

Average strict accuracy:

| variant | sample0 | fixed2 | fixed4 | fixed8 |
|---|---:|---:|---:|---:|
| base | 0.661 | 0.663 | 0.664 | 0.677 |
| full v1 step100 | **0.677** | **0.682** | **0.679** | **0.678** |
| v2 step75 | 0.662 | 0.667 | 0.672 | 0.677 |
| v2 step100 | 0.661 | 0.660 | 0.669 | 0.666 |

Per-seed strict accuracy for v2:

| checkpoint | seed | sample0 | fixed2 | fixed4 | fixed8 |
|---|---:|---:|---:|---:|---:|
| step75 | 2026 | 0.676 | 0.684 | 0.680 | 0.682 |
| step75 | 42 | 0.648 | 0.650 | 0.664 | 0.672 |
| step100 | 2026 | 0.674 | 0.676 | 0.684 | 0.670 |
| step100 | 42 | 0.648 | 0.644 | 0.654 | 0.662 |

## Reflection

v2 fixed pair noise but did not improve heldout performance. The core lesson is that pair cleanliness alone is not enough.

Likely reasons:

1. **Coverage collapsed further.** v2 covers only `58` train questions, down from v1's `98`. The clean signal is too narrow to reliably move heldout generation.
2. **Teacher-answer distillation creates distribution shift.** `teacher_correct_anchor_rescue` chosen answers are often 1-2 words. Training against long Qwen rejected answers teaches a strong margin, but may not teach the model how to produce stable, calibrated Qwen-style completions under sampling.
3. **Dev pair accuracy is not predictive enough.** v2 dev pairs are only 20 pairs from 6 questions. The model can fit these without improving heldout generation.
4. **Beta may now be too strong.** With cleaner but narrower pairs, `beta=0.12` plus teacher short answers may over-concentrate probability locally and hurt seed robustness.

## Current Best Variant

The best full500 result remains v1 step100:

| variant | sample0 | fixed2 | fixed8 |
|---|---:|---:|---:|
| base | 0.661 | 0.663 | 0.677 |
| full v1 step100 | **0.677** | **0.682** | 0.678 |
| v2 step75 | 0.662 | 0.667 | 0.677 |

## Next Hypothesis

The anchor signal should probably not be forced into raw DPO pairs alone. Better next directions:

- train a lightweight anchor reranker/verifier for generated candidates;
- use anchor score as a reward/advantage over candidate basins instead of short-answer DPO;
- construct more coverage-preserving pairs from all anchored questions, but keep chosen in the student's own answer style;
- use teacher answer only as a label/anchor feature, not necessarily as the direct completion target.

For VBPO specifically, the next v3 should preserve v1's coverage while only removing the clearly bad `teacher_anchor_vs_student_only` noise. It should not replace too many chosen completions with ultra-short teacher answers.
