# Anchor-Aware VBPO Audit

## Why Strong AUC Did Not Transfer Cleanly

The full500 anchor AUC is real at the candidate-ranking level:

- final-only `teacher_support_mass` AUC: `0.895`
- final-only `anchor_score_noleak` AUC: `0.890`
- Qwen-only verifier AUC: `0.491`

However, the current VBPO conversion from anchor scores to DPO pairs is lossy and noisy.

## Finding 1: Pair Coverage Is Too Low

Full500 has 500 anchored questions and 8000 Qwen candidate rows, but the full v1 trainer built only:

- `341` train pairs
- covering `98` train questions
- out of `425` train question ids

Most high-AUC anchor rows never become training supervision.

Training split coverage pattern:

| condition | train questions |
|---|---:|
| any correct, no wrong, has anchor | 179 |
| no correct, wrong only, no anchor | 92 |
| any correct and wrong, has anchor/student-only, has pair | 54 |
| no correct, wrong only, has anchor/student-only, no pair | 25 |
| other pair-covered cases | 44 |

So the training signal is concentrated in a small subset of contrastive questions.

## Finding 2: `teacher_anchor_vs_student_only` Is Noisy

This pair type does not require gold-correct chosen answers. It was intended as a no-leak anchor pair, but in the current implementation it has high label noise:

| pair type | n | chosen strict-correct | rejected strict-correct |
|---|---:|---:|---:|
| `teacher_anchor_vs_student_only` | 92 | 0.609 | 0.370 |
| `anchor_correct_vs_qwen_only_stable_wrong` | 116 | 1.000 | 0.000 |
| `anchor_rescue` | 70 | 1.000 | 0.000 |
| `anchor_damage_guard` | 63 | 1.000 | 0.000 |

This means a quarter of balanced batches is spent on a pair type that often teaches weak or ambiguous preferences.

## Finding 3: Strict Labels Favor Verbose Candidate Text

The trainer uses raw `answer_text` as the DPO completion. For TriviaQA, `strict_correct` is based on normalized exact match or substring containment. This is useful for evaluation, but it can mark verbose completions as correct when they merely contain a gold alias.

Examples observed in training pairs:

- A chosen answer for a numeric question says a long, implausible derivation but contains the target number.
- A chosen answer for `None`-style aliases can be marked correct because the word appears inside a sentence.
- Many chosen answers are 40-60 tokens, while the target task asks for brief factual answers.

This creates a mismatch:

- AUC measures whether anchor features rank candidate rows.
- DPO trains the model to emit the full raw candidate text.

The latter can teach verbosity or spurious reasoning fragments instead of just moving probability mass toward the factual basin.

## Finding 4: DPO Signal Is Very Gentle

Current training uses:

- average sequence log-probability, not summed log-probability
- `beta = 0.07`
- `learning_rate = 1e-6`
- LoRA rank 8

Observed margin deltas stay near zero through training. This is not necessarily a bug, but with noisy pairs it means the model receives a weak and unstable preference signal.

## Most Likely Root Cause

The anchor signal is strong, but the current pair builder is not using it in the cleanest form.

The biggest issues are:

1. too few questions converted into training pairs;
2. noisy no-gold `teacher_anchor_vs_student_only` pairs;
3. raw verbose candidate completions used as chosen/rejected strings;
4. conservative DPO strength after the signal has already been diluted.

## Recommended v2 Fix

Do not simply increase steps or learning rate. First clean the supervision.

Recommended changes:

- Disable or heavily downweight `teacher_anchor_vs_student_only` unless chosen is strict-correct and rejected is strict-wrong.
- Use a stricter chosen filter:
  - max chosen length, e.g. `<= 24` words;
  - no `analysis`, `however`, unfinished reasoning, or multi-sentence derivation;
  - prefer sample0/shortest candidate within the same anchored basin.
- Convert chosen/rejected to canonical short answers where possible, instead of training on raw verbose completions.
- Increase coverage by creating within-anchor positive distillation pairs:
  - anchored short candidate vs unsupported sample0;
  - anchored short candidate vs high-mass unsupported basin;
  - skip questions with only all-correct or all-wrong candidates unless a clean teacher answer can be used.
- After cleaning, retune:
  - `beta`: try `0.10` or `0.15`;
  - `learning_rate`: keep `1e-6` first, then test `2e-6`;
  - max steps: early-stop around 60-100.
