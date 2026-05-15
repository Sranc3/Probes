# Structured Factual Controller Evaluation

## Summary

| Method | Strict | Delta | Improved | Damaged | Net | Changed | Cand Cost | Token Cost | Basin Count | Verifier Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed8_cluster_weight` | `53.00%` | `2.25%` | `15` | `6` | `9` | `24.50%` | `8.00x` | `8.30x` | `2.73` | `0.00` |
| `cluster_weight_structured_veto` | `52.75%` | `2.00%` | `11` | `3` | `8` | `16.00%` | `8.00x` | `8.30x` | `2.74` | `19.14` |
| `fixed8_structured_acceptability` | `52.25%` | `1.50%` | `17` | `11` | `6` | `32.50%` | `8.00x` | `8.30x` | `2.73` | `19.14` |
| `fixed8_low_entropy_weight` | `52.00%` | `1.25%` | `12` | `7` | `5` | `25.25%` | `8.00x` | `8.30x` | `2.73` | `0.00` |
| `veto_bonus_compact` | `51.00%` | `0.25%` | `2` | `1` | `1` | `6.50%` | `8.00x` | `8.30x` | `2.74` | `19.14` |
| `theory_core` | `50.75%` | `0.00%` | `2` | `2` | `0` | `6.00%` | `8.00x` | `8.30x` | `2.74` | `0.00` |
| `theory_plus_structured` | `50.75%` | `0.00%` | `2` | `2` | `0` | `4.75%` | `8.00x` | `8.30x` | `2.74` | `19.14` |
| `sample0_baseline` | `50.75%` | `0.00%` | `0` | `0` | `0` | `0.00%` | `1.00x` | `1.00x` | `2.73` | `0.00` |
| `qwen_structured` | `50.25%` | `-0.50%` | `0` | `2` | `-2` | `5.75%` | `8.00x` | `8.30x` | `2.74` | `19.14` |

## Interpretation

Best accuracy method: `fixed8_cluster_weight` with strict `53.00%` and delta `2.25%`.

This is a fixed-8 post-verifier evaluation: candidate generation cost is full-8 for all non-sample0 methods. Structured verifier cost is reported separately as prompt calls per question, not hidden inside token cost.

If structured features improve accuracy at fixed-8 but do not beat adaptive-v1's low-cost frontier, the next step should be a two-stage controller: cheap theory-core gate first, structured verifier only for ambiguous high-value switches.
