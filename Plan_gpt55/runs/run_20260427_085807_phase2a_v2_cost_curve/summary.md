# Phase 2A-v2 Cost Curve

- Description: Post-hoc cost curve for low-cost output-level reranking using existing Phase 2A sample sets.
- Source run: `/zhutingqi/song/Plan_gpt55/runs/run_20260427_074132_phase2a_reranking`
- All correctness numbers below use strict exact/contains matching only.

## Cost Curve

| K | Method | Strict Correct | Δ Strict | Improved | Damaged | Changed | Token Mean Ent | Oracle Available |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `sample0` | `41.94%` | `0.0000` | `0` | `0` | `0.00%` | `0.2352` | `0.00%` |
| `1` | `cautious_majority_logprob` | `41.94%` | `0.0000` | `0` | `0` | `0.00%` | `0.2352` | `0.00%` |
| `1` | `majority_logprob` | `41.94%` | `0.0000` | `0` | `0` | `0.00%` | `0.2352` | `0.00%` |
| `1` | `majority_low_entropy` | `41.94%` | `0.0000` | `0` | `0` | `0.00%` | `0.2352` | `0.00%` |
| `1` | `best_logprob` | `41.94%` | `0.0000` | `0` | `0` | `0.00%` | `0.2352` | `0.00%` |
| `1` | `low_token_entropy` | `41.94%` | `0.0000` | `0` | `0` | `0.00%` | `0.2352` | `0.00%` |
| `1` | `oracle_strict` | `41.94%` | `0.0000` | `0` | `0` | `0.00%` | `0.2352` | `0.00%` |
| `2` | `sample0` | `41.94%` | `0.0000` | `0` | `0` | `0.00%` | `0.2352` | `3.23%` |
| `2` | `cautious_majority_logprob` | `41.94%` | `0.0000` | `0` | `0` | `5.38%` | `0.2291` | `3.23%` |
| `2` | `majority_logprob` | `43.01%` | `0.0108` | `2` | `1` | `19.35%` | `0.2111` | `3.23%` |
| `2` | `majority_low_entropy` | `43.01%` | `0.0108` | `2` | `1` | `21.51%` | `0.2084` | `3.23%` |
| `2` | `best_logprob` | `43.01%` | `0.0108` | `2` | `1` | `19.35%` | `0.2111` | `3.23%` |
| `2` | `low_token_entropy` | `41.94%` | `0.0000` | `1` | `1` | `23.66%` | `0.2024` | `3.23%` |
| `2` | `oracle_strict` | `45.16%` | `0.0323` | `3` | `0` | `19.35%` | `0.2119` | `3.23%` |
| `3` | `sample0` | `41.94%` | `0.0000` | `0` | `0` | `0.00%` | `0.2352` | `5.38%` |
| `3` | `cautious_majority_logprob` | `41.94%` | `0.0000` | `1` | `1` | `15.05%` | `0.2327` | `5.38%` |
| `3` | `majority_logprob` | `44.09%` | `0.0215` | `3` | `1` | `29.03%` | `0.2121` | `5.38%` |
| `3` | `majority_low_entropy` | `44.09%` | `0.0215` | `3` | `1` | `27.96%` | `0.2087` | `5.38%` |
| `3` | `best_logprob` | `45.16%` | `0.0323` | `4` | `1` | `29.03%` | `0.2046` | `5.38%` |
| `3` | `low_token_entropy` | `43.01%` | `0.0108` | `2` | `1` | `29.03%` | `0.1918` | `5.38%` |
| `3` | `oracle_strict` | `47.31%` | `0.0538` | `5` | `0` | `29.03%` | `0.2054` | `5.38%` |
| `4` | `sample0` | `41.94%` | `0.0000` | `0` | `0` | `0.00%` | `0.2352` | `6.45%` |
| `4` | `cautious_majority_logprob` | `43.01%` | `0.0108` | `2` | `1` | `23.66%` | `0.2153` | `6.45%` |
| `4` | `majority_logprob` | `44.09%` | `0.0215` | `3` | `1` | `32.26%` | `0.2091` | `6.45%` |
| `4` | `majority_low_entropy` | `44.09%` | `0.0215` | `3` | `1` | `31.18%` | `0.2050` | `6.45%` |
| `4` | `best_logprob` | `46.24%` | `0.0430` | `5` | `1` | `31.18%` | `0.2036` | `6.45%` |
| `4` | `low_token_entropy` | `43.01%` | `0.0108` | `2` | `1` | `32.26%` | `0.1851` | `6.45%` |
| `4` | `oracle_strict` | `48.39%` | `0.0645` | `6` | `0` | `31.18%` | `0.2043` | `6.45%` |
| `8` | `sample0` | `41.94%` | `0.0000` | `0` | `0` | `0.00%` | `0.2352` | `12.90%` |
| `8` | `cautious_majority_logprob` | `44.09%` | `0.0215` | `3` | `1` | `30.11%` | `0.1990` | `12.90%` |
| `8` | `majority_logprob` | `47.31%` | `0.0538` | `6` | `1` | `35.48%` | `0.1925` | `12.90%` |
| `8` | `majority_low_entropy` | `47.31%` | `0.0538` | `6` | `1` | `35.48%` | `0.1845` | `12.90%` |
| `8` | `best_logprob` | `49.46%` | `0.0753` | `8` | `1` | `34.41%` | `0.1919` | `12.90%` |
| `8` | `low_token_entropy` | `45.16%` | `0.0323` | `4` | `1` | `38.71%` | `0.1718` | `12.90%` |
| `8` | `oracle_strict` | `54.84%` | `0.1290` | `12` | `0` | `35.48%` | `0.1920` | `12.90%` |

## Reading Guide

- `oracle_strict` is not deployable because it uses labels; it is only an upper bound showing whether better answers exist among the first K samples.
- `cautious_majority_logprob` is the low-risk candidate: it only switches away from sample0 if at least two sampled answers land in the same semantic cluster.
- A practical direction needs useful gains at K=2 or K=3; K=8 should be treated as an expensive oracle baseline.
