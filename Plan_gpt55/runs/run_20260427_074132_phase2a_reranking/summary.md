# Phase 2A Output-Level Reranking

- Description: Phase 2A output-level reranking on held-out test split without hidden-state intervention.
- Pairs per method: `93`
- Methods: `single_sample_baseline, random_of_n, best_of_n_logprob, semantic_cluster_majority, low_token_mean_entropy, cluster_then_low_entropy`

## Method Summary

| Method | Strict Correct | NLI Correct | NLI-only | Δ Strict vs sample0 | Changed vs sample0 | Token Mean Ent | Token Count | Cluster Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `best_of_n_logprob` | `49.46%` | `54.84%` | `5.38%` | `0.0753` | `34.41%` | `0.1919` | `20.98` | `6.23` |
| `cluster_then_low_entropy` | `47.31%` | `53.76%` | `6.45%` | `0.0538` | `36.56%` | `0.1846` | `21.61` | `6.38` |
| `low_token_mean_entropy` | `45.16%` | `50.54%` | `5.38%` | `0.0323` | `38.71%` | `0.1718` | `20.52` | `6.08` |
| `random_of_n` | `41.94%` | `47.31%` | `5.38%` | `0.0000` | `39.78%` | `0.2290` | `23.13` | `5.97` |
| `semantic_cluster_majority` | `47.31%` | `53.76%` | `6.45%` | `0.0538` | `36.56%` | `0.1934` | `21.76` | `6.38` |
| `single_sample_baseline` | `41.94%` | `45.16%` | `3.23%` | `0.0000` | `0.00%` | `0.2352` | `22.23` | `5.95` |

## Interpretation Rules

- Prefer strict correctness over NLI correctness when they disagree.
- A useful reranker should improve or preserve strict correctness while reducing token entropy or answer length.
- A high NLI-only rate is an evaluator warning, not evidence of real improvement.
