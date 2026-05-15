# Structured Factual Controller Evaluation

## Summary

| Method | Strict | Delta | Improved | Damaged | Net | Changed | Escalate | Cand Cost | Token Cost | Basin Count | Verifier Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed8_cluster_weight` | `53.00%` | `2.25%` | `15` | `6` | `9` | `24.50%` | `100.00%` | `8.00x` | `8.30x` | `2.73` | `0.00` |
| `cluster_weight_structured_veto` | `52.75%` | `2.00%` | `11` | `3` | `8` | `16.00%` | `100.00%` | `8.00x` | `8.30x` | `2.74` | `19.14` |
| `fixed8_structured_acceptability` | `52.25%` | `1.50%` | `17` | `11` | `6` | `32.50%` | `100.00%` | `8.00x` | `8.30x` | `2.73` | `19.14` |
| `fixed8_low_entropy_weight` | `52.00%` | `1.25%` | `12` | `7` | `5` | `25.25%` | `100.00%` | `8.00x` | `8.30x` | `2.73` | `0.00` |
| `two_stage_structured_balanced` | `51.25%` | `0.50%` | `4` | `2` | `2` | `10.25%` | `30.50%` | `3.14x` | `3.19x` | `2.74` | `10.75` |
| `veto_bonus_compact` | `51.00%` | `0.25%` | `2` | `1` | `1` | `6.50%` | `100.00%` | `8.00x` | `8.30x` | `2.74` | `19.14` |
| `theory_core` | `50.75%` | `0.00%` | `2` | `2` | `0` | `6.00%` | `100.00%` | `8.00x` | `8.30x` | `2.74` | `0.00` |
| `two_stage_structured_production` | `50.75%` | `0.00%` | `2` | `2` | `0` | `8.00%` | `17.50%` | `2.23x` | `2.21x` | `2.74` | `6.97` |
| `theory_plus_structured` | `50.75%` | `0.00%` | `2` | `2` | `0` | `4.75%` | `100.00%` | `8.00x` | `8.30x` | `2.74` | `19.14` |
| `sample0_baseline` | `50.75%` | `0.00%` | `0` | `0` | `0` | `0.00%` | `0.00%` | `1.00x` | `1.00x` | `2.73` | `0.00` |
| `qwen_structured` | `50.25%` | `-0.50%` | `0` | `2` | `-2` | `5.75%` | `100.00%` | `8.00x` | `8.30x` | `2.74` | `19.14` |

## Interpretation

Best accuracy method: `fixed8_cluster_weight` with strict `53.00%` and delta `2.25%`.

Fixed-8 methods pay full-8 candidate cost for every question. Two-stage methods first run a cheap sample0 gate and only pay full-8 plus structured verifier calls for escalated questions.

If structured features improve accuracy at fixed-8 but do not beat adaptive-v1's low-cost frontier, the next step should be a two-stage controller: cheap theory-core gate first, structured verifier only for ambiguous high-value switches.
