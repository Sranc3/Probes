# Phase 1B/1C: Gate and Alpha Ablation

## Purpose

Determine whether fixed ITI failed because the direction is bad, or because the gate/alpha policy is too crude.

## Gate Ablation

| Gate ID | Rule |
| --- | --- |
| `always` | Apply intervention every generation step |
| `prev_entropy_q50` | Previous token entropy >= train q50 |
| `prev_entropy_q67` | Previous token entropy >= train q67 |
| `prev_entropy_q80` | Previous token entropy >= train q80 |
| `trend_up` | Entropy increases over recent local window |
| `position_late` | Apply only after minimum generated token count |

## Alpha/Sign Grid

| Parameter | Values |
| --- | --- |
| alpha | `[0.001, 0.0025, 0.005, 0.0075, 0.010, 0.015]` |
| target mode | `increase_semantic_high`, `reduce_semantic_high` |
| sites | `18`, `24`, `18+24` |

## Minimal Design

Run on pure test split first:

- seeds: `[42, 43, 52]`
- semantic samples: `8`
- max tokens: use same decoding budget as deployment eval or explicitly separate short/long settings

## Analysis Questions

1. Does increasing alpha produce monotonic or smooth changes?
2. Does polarity flip produce interpretable opposite effects?
3. Does any gate improve event-to-outcome alignment?
4. Does layer 18 or layer 24 dominate the effect?

## Pass Criteria

- At least one setting keeps correctness non-drop at `1.0`.
- Token hesitation improvement has abs(mean)/std >= `0.2`.
- Semantic breadth remains within band.
- Event density is neither near zero nor uncontrolled.

## Failure Interpretation

- No smooth alpha/sign trend: fixed direction is not a reliable control knob.
- Gate changes affect event count but not outcomes: gate is not aligned with useful intervention opportunities.
- Strong effects only at unsafe alpha: fixed ITI has narrow or unusable safety margin.
