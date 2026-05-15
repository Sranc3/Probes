# Existing Artifact Audit

## Intervention_D

- Run: `/zhutingqi/song/ITI_for_entropy/Intervention_D/runs/run_20260416_085929_nsga2_safety_pareto_v1`
- Final Pareto size: `1`
- Final candidate count: `1`
- Candidate: `ind_0067`
- Mode / alpha: `reduce_semantic_high @ 0.007030`
- Interpretation: `fallback_pareto_representative_not_robust_solution`

### Held-Out Gate Failures

- `semantic_band_success_rate=0.667 < 0.670`
- `token_mean_nonpositive_rate=0.667 < 0.670`
- `token_max_nonpositive_rate=0.333 < 0.500`
- `safety_success_rate=0.000 < 0.670`
- `stability_penalty=6.541 > 4.500`

## cand008 Deploy Eval

- Run: `/zhutingqi/song/cand008_deploy_eval_v1/runs/run_20260416_034151_cand008_deploy_eval_v1`
- Question pairs: `400`
- Unique questions: `200`
- Seeds: `2`
- Answer changed: `13` (3.25%)
- Correctness delta: `0.000000`
- Exact delta: `0.000000`
- Interpretation: `weak_behavioral_perturbation`

### Effect Strength

| Metric | Mean | Std | abs(mean)/std | Classification |
| --- | ---: | ---: | ---: | --- |
| `delta_semantic_entropy_weighted` | `0.002383` | `0.070772` | `0.034` | `noise_scale` |
| `delta_token_mean_entropy` | `-0.001044` | `0.025901` | `0.040` | `noise_scale` |
| `delta_token_max_entropy` | `0.006795` | `0.119411` | `0.057` | `noise_scale` |
| `delta_elapsed_ms` | `1.947500` | `124.517393` | `0.016` | `noise_scale` |

## Overall Judgment

- The current evidence supports entropy-related mechanism discovery.
- It does not support claiming a robust deployable fixed-ITI solution.
- The strongest next step is to treat ITI as a diagnostic probe and move optimization to learned or decoding-level control.
