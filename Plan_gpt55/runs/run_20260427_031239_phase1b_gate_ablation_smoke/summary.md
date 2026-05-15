# Phase 1B Gate Ablation

- Description: Smoke test for Phase 1B gate ablation runner.
- Selection: `split:test[0:2]`
- Seeds: `[42]`
- Semantic samples: `2`

## Gate Summary

| Candidate | Gate | Trigger | Gate q | Events | Answer Changed | Correct Delta | Token Mean Delta | Semantic Delta | Elapsed Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cand008_primary` | `always` | `always` | `0.67` | `53.000` | `50.00%` | `0.000000` | `-0.083723` | `-0.034716` | `-249.000` |
| `cand008_primary` | `prev_entropy_q50` | `prev_entropy_quantile` | `0.50` | `23.000` | `50.00%` | `0.000000` | `-0.083873` | `-0.034721` | `-260.500` |
| `cand008_primary` | `prev_entropy_q67` | `prev_entropy_quantile` | `0.67` | `13.000` | `50.00%` | `0.000000` | `-0.083893` | `-0.034721` | `-262.000` |
| `cand008_primary` | `prev_entropy_q80` | `prev_entropy_quantile` | `0.80` | `9.000` | `50.00%` | `0.000000` | `-0.083894` | `-0.034670` | `-266.000` |
| `ind0067_safety_pareto` | `always` | `always` | `0.67` | `53.000` | `50.00%` | `0.000000` | `-0.012729` | `-0.000050` | `-253.500` |
| `ind0067_safety_pareto` | `prev_entropy_q50` | `prev_entropy_quantile` | `0.50` | `24.000` | `50.00%` | `0.000000` | `-0.012324` | `-0.000047` | `-88.000` |
| `ind0067_safety_pareto` | `prev_entropy_q67` | `prev_entropy_quantile` | `0.67` | `15.000` | `50.00%` | `0.000000` | `-0.011550` | `-0.000047` | `-264.500` |
| `ind0067_safety_pareto` | `prev_entropy_q80` | `prev_entropy_quantile` | `0.80` | `10.000` | `50.00%` | `0.000000` | `-0.011550` | `-0.000047` | `-263.500` |

## Reasonableness Checks

- `always` should usually have the largest event count; if it does not, inspect generation-length drift and event traces.
- Lower previous-entropy quantiles should usually trigger at least as often as higher quantiles, but exact monotonicity is not guaranteed because the intervention can change the continuation.
- A gate is not deployable if it only increases event count while correctness drops or answer changes remain negligible.
