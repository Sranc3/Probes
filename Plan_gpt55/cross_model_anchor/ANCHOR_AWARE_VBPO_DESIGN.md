# Anchor-Aware VBPO Design

## Core Hypothesis

Correct generation depends on whether an answer basin is factually anchored, not only whether it is stable inside the student model. A Qwen basin that is independently supported by `gpt-oss-120b` is treated as a cross-model factual anchor. A high-mass Qwen basin that receives little or no teacher support is treated as a likely stable hallucination basin.

## Training Data

Use the final-channel-only anchor table:

`runs/run_20260512_122630_gptoss_anchor_triviaqa_pilot100_k4_tok256/qwen_candidate_anchor_rows_final_only.csv`

This table has 100 TriviaQA questions and 1600 Qwen candidate rows, joined with GPT-OSS teacher support features.

## Pair Types

- `anchor_rescue`: sample0 is wrong; chosen is a correct Qwen candidate with high teacher support; rejected is sample0 or another high-mass unsupported wrong basin.
- `anchor_damage_guard`: sample0 is correct; chosen is sample0; rejected is a wrong candidate with high Qwen stability but weak teacher support.
- `anchor_correct_vs_qwen_only_stable_wrong`: chosen is a teacher-supported correct candidate; rejected is a high `qwen_only_stable_mass` wrong candidate.

## Pair Selection Rules

- Chosen answers must be strict-correct under TriviaQA labels.
- Preferred chosen candidates should have high `teacher_support_mass` or high `anchor_score_noleak`.
- Rejected answers must be strict-wrong.
- Preferred rejected candidates should have high `qwen_only_stable_mass`, high Qwen cluster mass, and low teacher support.
- Use final-only teacher features by default to avoid training on incomplete harmony analysis traces.

## Weighting

Weights are intentionally moderate because the anchor pilot has only 100 questions.

- `anchor_rescue`: base weight `2.2`
- `anchor_damage_guard`: base weight `1.4`
- `anchor_correct_vs_qwen_only_stable_wrong`: base weight `1.7`

Dynamic multiplier:

```text
1 + 0.4 * teacher_support_mass(chosen) + 0.3 * qwen_only_stable_mass(rejected)
```

Weights are capped at `2.8`.

## Training Schedule

- LoRA rank: 8
- Learning rate: `1e-6`
- DPO beta: `0.07`
- Max steps: `90`
- Batch size: `4`
- Save/evaluate every `15/30` steps

This matches the stable v0.5 regime and avoids the long-training degradation seen in `v0.5-long240`.

## Evaluation

Evaluate on the same no-leak heldout split used for VBPO v0/v1/v0.5:

`TriviaQA offset500 test500`, seeds `2026` and `42`, fixed-k `1/2/4/8`.

Primary comparison:

- base original model
- VBPO v0
- VBPO v0.5
- Anchor-aware VBPO

Primary success criterion:

Anchor-aware VBPO should improve `sample0` without collapsing `fixed8`. Because training uses only 100 anchored questions, even a small heldout gain would be meaningful.
