# VBPO v0.5 Feature-Aware Analysis

## Motivation

VBPO v0/v1 used only a coarse fixed verifier score for pair construction. The existing candidate feature table contains richer signals, including token entropy, max entropy, logprob, cluster mass, cluster size, rank/minmax variants, and question-level semantic entropy. v0.5 makes the pair construction finer-grained while keeping training gentle enough to avoid the v1 dense-pair collapse.

## Feature Usage

- Previously used directly: `logprob_avg_z`, `token_mean_entropy_z`, `token_count_z`, `cluster_weight_mass_z`, `cluster_size_z`.
- Added to configurable score terms: `token_max_entropy_z`, `cluster_size_minmax`, `cluster_weight_mass_minmax`.
- Added dynamic hard-negative risk score: stable high-mass, large-cluster, high-logprob, low-entropy wrong basins receive larger pair weights.
- Not directly used in candidate-level ranking: `semantic_entropy_weighted_set`, because it is question-level rather than candidate-level. It is better suited for curriculum gating or question difficulty weighting in the next iteration.

## Hyperparameter Rationale

- Learning rate reduced to `1e-6` to avoid v1-style distribution compression.
- DPO beta reduced to `0.07` for a softer update.
- Pair count kept moderate with `max_pairs_per_question=4`.
- Dynamic pair weights are capped at `2.3`, with observed weights in `1.155-2.060`.

## Pair Statistics

- Train records: `425`
- Train pairs: `238`
- Pair types: `damage_guard=125`, `rescue_from_sample0=31`, `correct_vs_stable_wrong=82`
- Dev pairs: `31`

## External No-Leak Test500 Results

| Run | Seed | sample0 | fixed4 | fixed8 |
|---|---:|---:|---:|---:|
| base | 2026 | 0.666 | 0.668 | 0.680 |
| VBPO v0 | 2026 | 0.674 | 0.680 | 0.670 |
| VBPO v1 | 2026 | 0.670 | 0.660 | 0.662 |
| VBPO v0.5 | 2026 | 0.662 | 0.678 | 0.678 |
| base | 42 | 0.656 | 0.660 | 0.674 |
| VBPO v0 | 42 | 0.662 | 0.668 | 0.676 |
| VBPO v1 | 42 | 0.650 | 0.662 | 0.666 |
| VBPO v0.5 | 42 | 0.664 | 0.668 | 0.678 |

Average over two seeds:

| Run | sample0 | fixed4 | fixed8 |
|---|---:|---:|---:|
| base | 0.661 | 0.664 | 0.677 |
| VBPO v0 | 0.668 | 0.674 | 0.673 |
| VBPO v1 | 0.660 | 0.661 | 0.664 |
| VBPO v0.5 | 0.663 | 0.673 | 0.678 |

## Interpretation

v0.5 fixes the biggest v1 failure: fixed-k performance no longer collapses. Its fixed8 average is the best among the tested VBPO variants and slightly above base, but its sample0 gain is weaker than v0. This suggests the feature-aware weighting helps preserve basin diversity, but does not yet strongly improve the first sampled answer distribution.

The next likely improvement is not more dense pairs. It should be question-level curriculum weighting using semantic entropy and oracle availability: high semantic-entropy questions should receive weaker or delayed updates, while low-to-medium entropy questions with clear rescue candidates should receive stronger rescue training.
