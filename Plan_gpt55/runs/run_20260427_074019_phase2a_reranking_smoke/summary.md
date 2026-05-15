# Phase 2A Output-Level Reranking

- Description: Smoke test for output-level reranking without hidden-state intervention.
- Pairs per method: `2`
- Methods: `single_sample_baseline, random_of_n, best_of_n_logprob, semantic_cluster_majority, low_token_mean_entropy, cluster_then_low_entropy`

## Method Summary

| Method | Strict Correct | NLI Correct | NLI-only | Δ Strict vs sample0 | Changed vs sample0 | Token Mean Ent | Token Count | Cluster Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `best_of_n_logprob` | `0.00%` | `0.00%` | `0.00%` | `-0.5000` | `100.00%` | `0.2885` | `31.50` | `2.00` |
| `cluster_then_low_entropy` | `0.00%` | `0.00%` | `0.00%` | `-0.5000` | `100.00%` | `0.2885` | `31.50` | `2.00` |
| `low_token_mean_entropy` | `0.00%` | `0.00%` | `0.00%` | `-0.5000` | `100.00%` | `0.2885` | `31.50` | `2.00` |
| `random_of_n` | `0.00%` | `0.00%` | `0.00%` | `-0.5000` | `100.00%` | `0.3540` | `32.00` | `2.00` |
| `semantic_cluster_majority` | `0.00%` | `0.00%` | `0.00%` | `-0.5000` | `100.00%` | `0.2885` | `31.50` | `2.00` |
| `single_sample_baseline` | `50.00%` | `50.00%` | `0.00%` | `0.0000` | `0.00%` | `0.3862` | `30.00` | `1.00` |

## Interpretation Rules

- Prefer strict correctness over NLI correctness when they disagree.
- A useful reranker should improve or preserve strict correctness while reducing token entropy or answer length.
- A high NLI-only rate is an evaluator warning, not evidence of real improvement.
