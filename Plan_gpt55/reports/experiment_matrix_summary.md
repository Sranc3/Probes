# Experiment Matrix Summary

- Project: `Plan_gpt55`
- Purpose: New experiment roadmap after fixed ITI diagnosis

## Global Principles

- `do_not_claim_fallback_as_success`: `True`
- `pure_heldout_first`: `True`
- `semantic_entropy_is_breadth_not_primary_minimize_target`: `True`
- `elapsed_ms_requires_cache_aware_validation`: `True`

## Success Criteria

| Criterion | Value |
| --- | ---: |
| `correctness_non_drop_rate` | `1.0` |
| `exact_non_drop_rate` | `1.0` |
| `answer_change_rate_min` | `0.1` |
| `semantic_band_success_rate_min` | `0.8` |
| `token_mean_nonpositive_rate_min` | `0.67` |
| `token_max_nonpositive_rate_min` | `0.5` |
| `safety_success_rate_min` | `0.67` |
| `effect_abs_mean_over_std_min` | `0.2` |

## Experiments

| Priority | ID | Status | Goal | Decision |
| ---: | --- | --- | --- | --- |
| `1` | `phase1a_pure_heldout_fixed_iti` | `planned` | Recheck fixed ITI candidates on pure test split. | If weak answer_change_rate persists, stop fixed ITI as deployment route. |
| `2` | `phase1b_gate_ablation` | `planned` | Test whether fixed ITI failure is caused by crude gate policy. | If any gate aligns event density with safe hesitation reduction, proceed to learned controller. |
| `3` | `phase1c_alpha_sign_stress` | `planned` | Check whether alpha/sign/site effects are smooth and interpretable. | No smooth trend means fixed direction is not a reliable control knob. |
| `4` | `phase2a_decoding_reranking` | `planned` | Use entropy signals without hidden-state intervention. | If reranking beats fixed ITI, prioritize output-level control. |
| `5` | `phase2b_offline_controller` | `planned` | Train offline classifier to predict safe intervention actions. | Only build online controller if offline prediction beats random and majority baselines. |
| `6` | `phase3_distillation_sft` | `conditional` | phase2a or phase2b must show stable improvement | Do not train unless prior phases produce stable correctness-preserving behavior. |
