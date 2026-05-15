# Anchor-Aware VBPO Full500 Results

## Full GPT-OSS Anchor Run

- Run dir: `runs/run_20260513_012413_gptoss_anchor_triviaqa_full500_k4_tok256`
- Questions: `500`
- Teacher generations: `2000`
- Final-channel teacher generations: `1691` (`84.55%`)
- Qwen candidate rows: `8000`

Final-only anchor signal:

| metric | value |
|---|---:|
| teacher generation strict rate | 0.838 |
| teacher any-correct question rate | 0.768 |
| candidate AUC: verifier score v0.5 | 0.491 |
| candidate AUC: teacher support mass | 0.895 |
| candidate AUC: anchor score no-leak | 0.890 |
| teacher corrects Qwen sample0 rate | 0.222 |
| teacher conflicts with correct Qwen sample0 rate | 0.036 |

The full run confirms the pilot finding: cross-model teacher support remains a strong correctness signal at 500 questions, while the Qwen-only verifier score is near random on this candidate-level discrimination task.

## Full Anchor-Aware VBPO v1 Training

- Config: `configs/anchor_vbpo_v1_triviaqa_gptoss_full500.json`
- Run dir: `runs/run_20260513_052518_anchor_vbpo_v1_triviaqa_gptoss_full500`
- Training checkpoint evaluated as best current candidate: `checkpoints/step_0100`
- Training schedule: LoRA rank 8, LR `1e-6`, beta `0.07`, batch size 4

Pair manifest:

| split | pairs | questions | chosen teacher support | rejected teacher support | rejected qwen-only stable |
|---|---:|---:|---:|---:|---:|
| train | 341 | 98 | 0.678 | 0.034 | 0.366 |
| dev | 30 | 10 | 0.900 | 0.017 | 0.238 |

Train pair types:

| pair type | count |
|---|---:|
| `anchor_correct_vs_qwen_only_stable_wrong` | 116 |
| `teacher_anchor_vs_student_only` | 92 |
| `anchor_rescue` | 70 |
| `anchor_damage_guard` | 63 |

Training was stable. Dev pair accuracy reached `0.633` by step75 and stayed at `0.633` through step150. Step150 degraded heldout sample0 on seed2026, so step100 is the better checkpoint so far.

## Heldout Evaluation

Protocol: TriviaQA `offset500/test500`, no-leak fixed-k evaluation, seeds `2026` and `42`.

| variant | sample0 strict | fixed2 strict | fixed8 strict | sample0 reward | fixed8 reward |
|---|---:|---:|---:|---:|---:|
| base | 0.661 | 0.663 | 0.677 | 0.587 | 0.590 |
| Anchor-aware VBPO pilot100 step90 | 0.670 | 0.671 | **0.683** | 0.600 | **0.596** |
| Anchor-aware VBPO full500 step100 | **0.677** | **0.682** | 0.678 | **0.610** | 0.592 |

Per-seed strict accuracy for full500 step100:

| seed | sample0 | fixed2 | fixed8 |
|---:|---:|---:|---:|
| 2026 | 0.682 | 0.686 | 0.668 |
| 42 | 0.672 | 0.678 | 0.688 |

## Interpretation

Full500 strengthens single-sample generation:

- `sample0` improves from base `0.661` to `0.677`, a `+1.6` point gain.
- `fixed2` improves from base `0.663` to `0.682`, a `+1.9` point gain.
- Rewards improve clearly for sample0 and fixed2.

But full500 does not improve fixed8 beyond the pilot:

- fixed8 is `0.678`, basically base-level and below pilot100's `0.683`.
- The seed split is asymmetric: seed2026 fixed8 drops to `0.668`, while seed42 rises to `0.688`.

This suggests that the current full pair builder pushes the first-answer distribution toward anchored answers, but may reduce the diversity/cluster structure needed for majority-basin decoding. The likely issue is not anchor signal quality, because full500 anchor AUC remains high; it is how the anchor signal is converted into preference pairs and weights.

## Next Fix

The next version should explicitly protect fixed-k behavior:

- Downweight or remove `teacher_anchor_vs_student_only` pairs where the chosen candidate is not strict-correct.
- Require rejected candidates in strict-correct pair types to have low teacher support, not just high Qwen-only stability.
- Add a diversity guard: do not over-penalize alternate teacher-supported basins.
- Evaluate step50/75/100 more cheaply on a smaller validation slice before full test500.
