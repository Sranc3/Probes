# Phase 1B Gate Ablation

- Description: Phase 1B gate ablation on pure held-out test split: compare always-on vs previous-token entropy quantile gates for fixed ITI candidates.
- Selection: `split:test[0:31]`
- Seeds: `[42, 43, 52]`
- Semantic samples: `8`

## Gate Summary

| Candidate | Gate | Trigger | Gate q | Events | Answer Changed | Correct Delta | Token Mean Delta | Semantic Delta | Elapsed Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cand008_primary` | `always` | `always` | `0.67` | `45.376` | `8.60%` | `0.000000` | `0.003134` | `-0.007984` | `15.570` |
| `cand008_primary` | `prev_entropy_q50` | `prev_entropy_quantile` | `0.50` | `20.968` | `8.60%` | `0.000000` | `0.004531` | `-0.017488` | `6.258` |
| `cand008_primary` | `prev_entropy_q67` | `prev_entropy_quantile` | `0.67` | `13.204` | `7.53%` | `0.000000` | `0.000377` | `-0.023741` | `4.022` |
| `cand008_primary` | `prev_entropy_q80` | `prev_entropy_quantile` | `0.80` | `8.000` | `7.53%` | `0.000000` | `0.000372` | `-0.020486` | `-5.075` |
| `ind0067_safety_pareto` | `always` | `always` | `0.67` | `45.677` | `10.75%` | `0.010753` | `-0.001945` | `0.051513` | `17.065` |
| `ind0067_safety_pareto` | `prev_entropy_q50` | `prev_entropy_quantile` | `0.50` | `20.645` | `7.53%` | `0.010753` | `0.001096` | `0.020069` | `6.742` |
| `ind0067_safety_pareto` | `prev_entropy_q67` | `prev_entropy_quantile` | `0.67` | `13.011` | `5.38%` | `0.010753` | `-0.003478` | `0.028345` | `1.828` |
| `ind0067_safety_pareto` | `prev_entropy_q80` | `prev_entropy_quantile` | `0.80` | `7.742` | `4.30%` | `0.000000` | `-0.002688` | `0.033153` | `1.151` |

## Reasonableness Checks

- `always` should usually have the largest event count; if it does not, inspect generation-length drift and event traces.
- Lower previous-entropy quantiles should usually trigger at least as often as higher quantiles, but exact monotonicity is not guaranteed because the intervention can change the continuation.
- A gate is not deployable if it only increases event count while correctness drops or answer changes remain negligible.
