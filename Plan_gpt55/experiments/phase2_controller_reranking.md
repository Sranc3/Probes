# Phase 2: Learned Controller and Reranking

## Purpose

Move from hand-coded fixed ITI to adaptive control or output-level selection.

## Track C: Decoding / Reranking

### Hypothesis

Uncertainty signals may be more useful for choosing among generated answers than for directly steering hidden states.

### Candidate Methods

1. `best_of_n_logprob`
2. `semantic_cluster_majority`
3. `low_entropy_answer`
4. `cluster_then_entropy`
5. `correctness_proxy_rerank`

### Required Baselines

- single sample baseline
- fixed ITI baseline
- random choice from N samples
- best logprob among N samples

### Success Criteria

- correctness improves or remains non-negative;
- semantic diversity does not collapse;
- selected answer length remains reasonable;
- evaluation works on pure test split.

## Track B: Learned Controller

### Offline Prototype

The first controller should be offline and low-cost.

Input row = one question/seed/candidate evaluation.

Features:

- baseline token mean/max entropy;
- baseline semantic entropy;
- candidate alpha;
- target mode;
- active sites;
- event density;
- projection summaries if available;
- generated token count.

Label:

- positive if correctness does not drop and hesitation decreases within semantic band;
- negative otherwise.

Model:

- logistic regression first;
- small MLP only if logistic regression has signal.

### Online Controller Later

Only if offline controller has predictive power:

- step-level features;
- intervene / no-intervene decision;
- optional alpha bucket prediction.

## Stop Conditions

- If reranking beats fixed ITI, prioritize reranking.
- If controller cannot predict safe candidates better than random, do not train online controller.
- If both fail, write the result as a mechanistic negative finding.
