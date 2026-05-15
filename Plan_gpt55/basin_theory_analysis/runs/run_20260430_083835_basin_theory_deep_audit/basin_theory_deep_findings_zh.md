# Basin Theory Deep Audit Findings

## 0. No-Leak 审计

本实验把 correctness-derived 字段只作为 label / grouping / post-hoc diagnostics，不作为 predictor feature。脚本会在发现 banned feature 被纳入特征集时直接报错。

- Basin rows: `1094`
- Predictor features: `50`
- Banned columns checked: `16`

## 1. 数学特征族

| Family | Features |
| --- | --- |
| `stability` | `cluster_size, cluster_weight_mass, stable_score, low_entropy_score, fragmentation_entropy, normalized_fragmentation_entropy, semantic_entropy_weighted_set, semantic_clusters_set` |
| `geometry` | `centroid_entropy_z, centroid_max_entropy_z, centroid_logprob_z, centroid_len_z, distance_to_sample0_entropy_logprob, top2_weight_margin, top2_logprob_margin, top2_low_entropy_margin, low_entropy_basin_rank, logprob_basin_rank, weight_basin_rank, stable_basin_rank` |
| `lexical_shape` | `within_basin_lexical_entropy, within_basin_jaccard, internal_token_entropy_std, token_mean_entropy_std, logprob_avg_std, lexical_pca_anisotropy, lexical_effective_dim, lexical_radius` |
| `waveform_functional` | `wf_entropy_auc_mean, wf_entropy_max_mean, wf_entropy_std_mean, wf_entropy_roughness, wf_entropy_late_minus_early_mean, wf_entropy_spike_count_mean, wf_entropy_spike_frac_mean, wf_entropy_max_pos_norm_mean, wf_entropy_curve_var, wf_prob_min_mean, wf_prob_drop, wf_prob_curve_var, wf_local_instability_index` |
| `sample0_delta` | `delta_cluster_weight_mass_vs_sample0, delta_stable_score_vs_sample0, delta_low_entropy_score_vs_sample0, delta_centroid_entropy_z_vs_sample0, delta_centroid_logprob_z_vs_sample0, delta_wf_entropy_max_mean_vs_sample0, delta_wf_entropy_roughness_vs_sample0, delta_wf_prob_min_mean_vs_sample0, delta_distance_to_sample0_entropy_logprob_vs_sample0` |

## 2. Top Decomposition Signals

### pure_correct_vs_pure_wrong

| Feature | Family | Pos Mean | Neg Mean | Cohen d | AUC pos high |
| --- | --- | ---: | ---: | ---: | ---: |
| `cluster_weight_mass` | `stability` | `0.6920` | `0.2464` | `1.483` | `0.797` |
| `cluster_size` | `stability` | `5.5018` | `1.9826` | `1.482` | `0.797` |
| `normalized_fragmentation_entropy` | `stability` | `0.3809` | `0.8290` | `-1.430` | `0.218` |
| `wf_prob_min_mean` | `waveform_functional` | `0.4273` | `0.1237` | `1.368` | `0.822` |
| `fragmentation_entropy` | `stability` | `0.5646` | `1.4572` | `-1.329` | `0.206` |
| `semantic_entropy_weighted_set` | `stability` | `0.5646` | `1.4572` | `-1.329` | `0.207` |
| `top2_weight_margin` | `geometry` | `0.6979` | `0.2677` | `1.323` | `0.789` |
| `stable_score` | `stability` | `0.5333` | `0.2636` | `1.267` | `0.790` |

### non_sample0_rescue_vs_other_alt

| Feature | Family | Pos Mean | Neg Mean | Cohen d | AUC pos high |
| --- | --- | ---: | ---: | ---: | ---: |
| `cluster_size` | `stability` | `2.6154` | `1.2842` | `1.357` | `0.721` |
| `cluster_weight_mass` | `stability` | `0.3321` | `0.1581` | `1.347` | `0.727` |
| `lexical_effective_dim` | `lexical_shape` | `0.5849` | `0.0817` | `0.983` | `0.599` |
| `wf_prob_min_mean` | `waveform_functional` | `0.1723` | `0.0797` | `0.888` | `0.744` |
| `lexical_radius` | `lexical_shape` | `0.1106` | `0.0216` | `0.820` | `0.600` |
| `within_basin_jaccard` | `lexical_shape` | `0.8936` | `0.9810` | `-0.803` | `0.422` |
| `stable_score` | `stability` | `0.3290` | `0.2103` | `0.780` | `0.693` |
| `wf_entropy_roughness` | `waveform_functional` | `0.3550` | `0.9983` | `-0.750` | `0.246` |

### stable_correct_vs_stable_hallucination

| Feature | Family | Pos Mean | Neg Mean | Cohen d | AUC pos high |
| --- | --- | ---: | ---: | ---: | ---: |
| `top2_weight_margin` | `geometry` | `0.8919` | `0.5760` | `1.012` | `0.732` |
| `normalized_fragmentation_entropy` | `stability` | `0.1557` | `0.5180` | `-1.012` | `0.270` |
| `cluster_weight_mass` | `stability` | `0.9232` | `0.6888` | `0.975` | `0.725` |
| `cluster_size` | `stability` | `7.3229` | `5.3987` | `0.958` | `0.727` |
| `fragmentation_entropy` | `stability` | `0.1975` | `0.7225` | `-0.935` | `0.274` |
| `semantic_entropy_weighted_set` | `stability` | `0.1975` | `0.7225` | `-0.935` | `0.279` |
| `stable_score` | `stability` | `0.6760` | `0.5781` | `0.890` | `0.707` |
| `semantic_clusters_set` | `stability` | `1.5781` | `3.0127` | `-0.849` | `0.281` |

### rescue_vs_damage

| Feature | Family | Pos Mean | Neg Mean | Cohen d | AUC pos high |
| --- | --- | ---: | ---: | ---: | ---: |
| `wf_prob_min_mean` | `waveform_functional` | `0.1723` | `0.0620` | `0.968` | `0.800` |
| `delta_cluster_weight_mass_vs_sample0` | `sample0_delta` | `0.0313` | `-0.2825` | `0.948` | `0.761` |
| `wf_entropy_auc_mean` | `waveform_functional` | `0.3066` | `0.5172` | `-0.916` | `0.240` |
| `delta_stable_score_vs_sample0` | `sample0_delta` | `0.0215` | `-0.2512` | `0.906` | `0.745` |
| `delta_wf_prob_min_mean_vs_sample0` | `sample0_delta` | `0.0250` | `-0.1799` | `0.895` | `0.752` |
| `wf_entropy_std_mean` | `waveform_functional` | `0.6976` | `1.0078` | `-0.791` | `0.277` |
| `wf_local_instability_index` | `waveform_functional` | `11.0505` | `8.3812` | `0.710` | `0.655` |
| `wf_entropy_max_mean` | `waveform_functional` | `2.9147` | `3.9231` | `-0.690` | `0.305` |

## 3. Question-Heldout Diagnostic Separability

| Task | Feature Set | AUC | Balanced Acc | Rows |
| --- | --- | ---: | ---: | ---: |
| `pure_correct_vs_wrong` | `stability` | `0.830` | `0.771` | `1085` |
| `pure_correct_vs_wrong` | `geometry` | `0.823` | `0.773` | `1085` |
| `pure_correct_vs_wrong` | `waveform_functional` | `0.819` | `0.736` | `1085` |
| `pure_correct_vs_wrong` | `all_noleak` | `0.799` | `0.735` | `1085` |
| `pure_correct_vs_wrong` | `lexical_shape` | `0.732` | `0.667` | `1085` |
| `pure_correct_vs_wrong` | `sample0_delta` | `0.703` | `0.654` | `1085` |
| `rescue_vs_damage` | `all_noleak` | `0.787` | `0.665` | `123` |
| `rescue_vs_damage` | `waveform_functional` | `0.774` | `0.719` | `123` |
| `rescue_vs_damage` | `stability` | `0.711` | `0.644` | `123` |
| `rescue_vs_damage` | `sample0_delta` | `0.676` | `0.655` | `123` |
| `rescue_vs_damage` | `geometry` | `0.664` | `0.652` | `123` |
| `rescue_vs_damage` | `lexical_shape` | `0.639` | `0.480` | `123` |
| `stable_correct_vs_hallucination` | `all_noleak` | `0.724` | `0.658` | `350` |
| `stable_correct_vs_hallucination` | `stability` | `0.717` | `0.687` | `350` |
| `stable_correct_vs_hallucination` | `geometry` | `0.704` | `0.679` | `350` |
| `stable_correct_vs_hallucination` | `waveform_functional` | `0.698` | `0.602` | `350` |
| `stable_correct_vs_hallucination` | `sample0_delta` | `0.616` | `0.607` | `350` |
| `stable_correct_vs_hallucination` | `lexical_shape` | `0.548` | `0.499` | `350` |

## 4. 初步解释

- 如果 stability 特征强但不能区分 stable correct / stable hallucination，说明“稳定性”主要刻画 basin attractor，而不是 factual correctness。
- 如果 waveform functional 特征在 stable hallucination 上仍有较大 effect size，说明宏观稳定和微观 token 不稳定可以同时存在。
- 如果 sample0_delta 特征对 rescue/damage 更强，说明 controller 应该关注 alternative basin 相对 sample0 的变化，而不是只看单个 basin 的绝对分数。
- 这些结果仍然是机制分析；除 question-heldout diagnostic classifier 外，不应被表述为 deployable controller 效果。
