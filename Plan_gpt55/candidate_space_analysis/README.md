# Candidate Space Analysis

This workspace studies the answer-candidate space revealed by Phase 2A.

The core idea is simple:

> For some questions, the model's first answer is wrong, but a later sampled answer is correct. That means the correct answer already exists somewhere in the local candidate space.

This folder treats every sampled answer as a point with measurable coordinates:

- log probability
- token entropy
- answer length
- semantic-cluster size
- semantic-cluster weight
- rank inside the sampled set
- strict correctness label from exact/contains matching

The first goal is not to claim a sophisticated geometric method. Instead, it builds the empirical foundation needed for one:

1. Are correct candidates separable from wrong candidates in this feature space?
2. Are "rescue" candidates, where sample0 is wrong but another sample is correct, visibly different?
3. Which features are useful, harmful, or ambiguous?
4. Can a cheap controller approximate the expensive best-of-N oracle?

If these diagnostics show stable structure, later work can move toward learned scoring, local geometry, or manifold-style controllers.
