# Phase 1A Fixed ITI Recheck

- Description: Pure held-out test split recheck for fixed ITI candidates after Plan_gpt55 diagnosis.
- Selection: `split:test[0:31]`
- Seeds: `[42, 43, 52, 53, 54]`
- Semantic samples: `8`

## Candidate Summary

| Candidate | Mode | Alpha | Answer Changed | Correct Delta | Token Mean Delta | Token Max Delta | Semantic Delta | Elapsed Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cand008_primary` | `increase_semantic_high` | `0.005000` | `7.10%` | `-0.006452` | `-0.001592` | `-0.023523` | `-0.007767` | `2.374` |
| `ind0067_safety_pareto` | `reduce_semantic_high` | `0.007030` | `5.16%` | `0.006452` | `-0.001445` | `-0.015531` | `0.021798` | `-2.213` |

## Interpretation Rules

- If answer changed rate remains below 10%, fixed ITI is too weak as a deployment route.
- If correctness drops, fixed ITI is unsafe.
- If token/semantic effects are smaller than seed/question variance, treat the result as diagnostic rather than deployable.
