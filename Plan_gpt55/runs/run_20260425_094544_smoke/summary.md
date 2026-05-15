# Phase 1A Fixed ITI Recheck

- Description: Smoke test for pure held-out fixed ITI recheck runner.
- Selection: `split:test[0:2]`
- Seeds: `[42]`
- Semantic samples: `2`

## Candidate Summary

| Candidate | Mode | Alpha | Answer Changed | Correct Delta | Token Mean Delta | Token Max Delta | Semantic Delta | Elapsed Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cand008_primary` | `increase_semantic_high` | `0.005000` | `50.00%` | `0.000000` | `-0.083904` | `-1.085399` | `-0.034721` | `-377.500` |
| `ind0067_safety_pareto` | `reduce_semantic_high` | `0.007030` | `50.00%` | `0.000000` | `-0.011554` | `-0.000847` | `-0.000047` | `-388.000` |

## Interpretation Rules

- If answer changed rate remains below 10%, fixed ITI is too weak as a deployment route.
- If correctness drops, fixed ITI is unsafe.
- If token/semantic effects are smaller than seed/question variance, treat the result as diagnostic rather than deployable.
