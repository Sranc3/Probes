# Phase 1C Case-Level Diagnosis

- Description: Post-hoc case-level diagnosis for Phase 1A fixed ITI and Phase 1B gate ablation outputs.
- Total paired rows: `1054`
- Experiments: `phase1a_fixed_iti_recheck, phase1b_gate_ablation`

## Candidate-Level Diagnosis

| Experiment | Candidate | Gate | Pairs | Changed | Corrected | Damaged | Wrong->Wrong | Events | Semantic Δ | Token Mean Δ | Event->Changed r |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase1a_fixed_iti_recheck` | `cand008_primary` | `default` | `155` | `7.10%` | `0` | `1` | `9` | `13.56` | `-0.0078` | `-0.0016` | `0.262` |
| `phase1a_fixed_iti_recheck` | `ind0067_safety_pareto` | `default` | `155` | `5.16%` | `1` | `0` | `7` | `13.43` | `0.0218` | `-0.0014` | `0.183` |
| `phase1b_gate_ablation` | `cand008_primary` | `always` | `93` | `8.60%` | `0` | `0` | `8` | `45.38` | `-0.0080` | `0.0031` | `0.446` |
| `phase1b_gate_ablation` | `cand008_primary` | `prev_entropy_q50` | `93` | `8.60%` | `0` | `0` | `8` | `20.97` | `-0.0175` | `0.0045` | `0.490` |
| `phase1b_gate_ablation` | `cand008_primary` | `prev_entropy_q67` | `93` | `7.53%` | `0` | `0` | `7` | `13.20` | `-0.0237` | `0.0004` | `0.406` |
| `phase1b_gate_ablation` | `cand008_primary` | `prev_entropy_q80` | `93` | `7.53%` | `0` | `0` | `7` | `8.00` | `-0.0205` | `0.0004` | `0.427` |
| `phase1b_gate_ablation` | `ind0067_safety_pareto` | `always` | `93` | `10.75%` | `1` | `0` | `9` | `45.68` | `0.0515` | `-0.0019` | `0.298` |
| `phase1b_gate_ablation` | `ind0067_safety_pareto` | `prev_entropy_q50` | `93` | `7.53%` | `1` | `0` | `6` | `20.65` | `0.0201` | `0.0011` | `0.285` |
| `phase1b_gate_ablation` | `ind0067_safety_pareto` | `prev_entropy_q67` | `93` | `5.38%` | `1` | `0` | `4` | `13.01` | `0.0283` | `-0.0035` | `0.272` |
| `phase1b_gate_ablation` | `ind0067_safety_pareto` | `prev_entropy_q80` | `93` | `4.30%` | `0` | `0` | `4` | `7.74` | `0.0332` | `-0.0027` | `0.059` |

## Most Perturbed Questions

| Experiment | Question | Changed | Damaged | Corrected | Mean |abs semantic Δ| |
| --- | --- | ---: | ---: | ---: | ---: |
| `phase1b_gate_ablation` | `trivia_qa_104` | `20/24` | `0` | `0` | `0.1422` |
| `phase1b_gate_ablation` | `trivia_qa_138` | `10/24` | `0` | `0` | `0.0957` |
| `phase1b_gate_ablation` | `trivia_qa_61` | `9/24` | `0` | `0` | `0.1462` |
| `phase1b_gate_ablation` | `trivia_qa_178` | `7/24` | `0` | `3` | `0.1717` |
| `phase1a_fixed_iti_recheck` | `trivia_qa_104` | `7/10` | `0` | `0` | `0.1197` |
| `phase1b_gate_ablation` | `trivia_qa_127` | `6/24` | `0` | `0` | `0.0018` |
| `phase1b_gate_ablation` | `trivia_qa_135` | `4/24` | `0` | `0` | `0.3793` |
| `phase1a_fixed_iti_recheck` | `trivia_qa_138` | `4/10` | `0` | `0` | `0.0495` |
| `phase1a_fixed_iti_recheck` | `trivia_qa_178` | `2/10` | `0` | `1` | `0.3183` |
| `phase1a_fixed_iti_recheck` | `trivia_qa_127` | `2/10` | `0` | `0` | `0.0011` |
| `phase1a_fixed_iti_recheck` | `trivia_qa_135` | `1/10` | `0` | `0` | `0.2387` |
| `phase1a_fixed_iti_recheck` | `trivia_qa_156` | `1/10` | `1` | `0` | `0.0004` |

## Case Interpretation

- Corrected cases: `4` shown, from total `4`.
- Damaged cases: `1` shown, from total `1`.
- Wrong-to-wrong changed cases shown: `12`. These are important because they look like activity but do not improve correctness.
- High-event no-answer-change cases shown: `12`. These indicate hidden-state perturbation can be absorbed without changing final behavior.

## Bottom Line

- If most changed cases are wrong-to-wrong rather than corrected, fixed ITI is acting more like a perturbation probe than a reliable controller.
- If high-event no-change cases are common, event count alone is not a useful success metric.
- If corrected and damaged cases are sparse and asymmetric across seeds/questions, the next step should be an offline controller/reranker, not broader fixed-vector tuning.
