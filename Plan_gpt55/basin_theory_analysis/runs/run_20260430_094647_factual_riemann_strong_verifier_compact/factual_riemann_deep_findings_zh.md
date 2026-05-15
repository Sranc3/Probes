# Factual Residual and Riemann Geometry Deep Findings

## 0. No-Leak Audit

本实验不使用 benchmark gold answer 或 correctness-derived 字段作为 feature。`no_leak_audit_factual_riemann.csv` 中所有 predictor 均需为 `ok`。

- Rows: `1094`
- Qwen verifier status: `attempted`
- Qwen rows scored: `1094`
- Structured Qwen verifier status: `attempted`
- Structured Qwen rows scored: `1094`
- Hidden geometry status: `attempted`

## 1. Feature Family AUC

| Task | Feature Set | AUC | Balanced Acc | Rows |
| --- | --- | ---: | ---: | ---: |
| `pure_correct_vs_wrong` | `theory_core` | `0.828` | `0.760` | `1085` |
| `pure_correct_vs_wrong` | `theory_plus_structured_compact` | `0.826` | `0.760` | `1085` |
| `pure_correct_vs_wrong` | `all_no_leak` | `0.820` | `0.749` | `1085` |
| `pure_correct_vs_wrong` | `factual_plus_riemann` | `0.784` | `0.747` | `1085` |
| `pure_correct_vs_wrong` | `strong_verifier_controller` | `0.769` | `0.733` | `1085` |
| `pure_correct_vs_wrong` | `riemann_geometry` | `0.764` | `0.754` | `1085` |
| `pure_correct_vs_wrong` | `structured_rescue_veto_compact` | `0.718` | `0.635` | `1085` |
| `pure_correct_vs_wrong` | `qwen_structured_factual` | `0.702` | `0.620` | `1085` |
| `pure_correct_vs_wrong` | `qwen_factual` | `0.672` | `0.623` | `1085` |
| `pure_correct_vs_wrong` | `factual_proxy` | `0.616` | `0.625` | `1085` |
| `pure_correct_vs_wrong` | `hidden_riemann_geometry` | `0.585` | `0.580` | `1085` |
| `pure_correct_vs_wrong` | `factual_riemann_delta` | `0.466` | `0.561` | `1085` |
| `rescue_vs_damage` | `theory_core` | `0.766` | `0.731` | `123` |
| `rescue_vs_damage` | `theory_plus_structured_compact` | `0.754` | `0.696` | `123` |
| `rescue_vs_damage` | `all_no_leak` | `0.724` | `0.661` | `123` |
| `rescue_vs_damage` | `factual_riemann_delta` | `0.716` | `0.671` | `123` |
| `rescue_vs_damage` | `strong_verifier_controller` | `0.710` | `0.720` | `123` |
| `rescue_vs_damage` | `qwen_structured_factual` | `0.701` | `0.678` | `123` |
| `rescue_vs_damage` | `structured_rescue_veto_compact` | `0.690` | `0.666` | `123` |
| `rescue_vs_damage` | `qwen_factual` | `0.684` | `0.627` | `123` |
| `rescue_vs_damage` | `factual_plus_riemann` | `0.650` | `0.617` | `123` |
| `rescue_vs_damage` | `riemann_geometry` | `0.598` | `0.600` | `123` |
| `rescue_vs_damage` | `hidden_riemann_geometry` | `0.512` | `0.535` | `123` |
| `rescue_vs_damage` | `factual_proxy` | `0.492` | `0.460` | `123` |
| `stable_correct_vs_hallucination` | `theory_core` | `0.747` | `0.685` | `350` |
| `stable_correct_vs_hallucination` | `theory_plus_structured_compact` | `0.742` | `0.676` | `350` |
| `stable_correct_vs_hallucination` | `strong_verifier_controller` | `0.685` | `0.616` | `350` |
| `stable_correct_vs_hallucination` | `all_no_leak` | `0.674` | `0.655` | `350` |
| `stable_correct_vs_hallucination` | `riemann_geometry` | `0.662` | `0.648` | `350` |
| `stable_correct_vs_hallucination` | `structured_rescue_veto_compact` | `0.633` | `0.530` | `350` |
| `stable_correct_vs_hallucination` | `factual_plus_riemann` | `0.627` | `0.564` | `350` |
| `stable_correct_vs_hallucination` | `qwen_structured_factual` | `0.605` | `0.515` | `350` |
| `stable_correct_vs_hallucination` | `qwen_factual` | `0.601` | `0.554` | `350` |
| `stable_correct_vs_hallucination` | `factual_riemann_delta` | `0.505` | `0.554` | `350` |
| `stable_correct_vs_hallucination` | `hidden_riemann_geometry` | `0.498` | `0.511` | `350` |
| `stable_correct_vs_hallucination` | `factual_proxy` | `0.406` | `0.408` | `350` |

## 2. Top Factual / Riemann Signals

### pure_correct_vs_pure_wrong

| Feature | Pos Mean | Neg Mean | Cohen d | AUC pos high |
| --- | ---: | ---: | ---: | ---: |
| `cluster_weight_mass` | `0.6920` | `0.2464` | `1.483` | `0.797` |
| `cluster_size` | `5.5018` | `1.9826` | `1.482` | `0.797` |
| `wf_prob_min_mean` | `0.4273` | `0.1237` | `1.368` | `0.822` |
| `fragmentation_entropy` | `0.5646` | `1.4572` | `-1.329` | `0.206` |
| `top2_weight_margin` | `0.6979` | `0.2677` | `1.323` | `0.789` |
| `stable_score` | `0.5333` | `0.2636` | `1.267` | `0.790` |
| `wf_entropy_max_mean` | `1.8151` | `3.6915` | `-1.205` | `0.188` |
| `riemann_anisotropy` | `0.5828` | `0.1514` | `1.137` | `0.737` |
| `wf_entropy_roughness` | `0.2336` | `0.9226` | `-0.874` | `0.177` |
| `qwen_missing_constraint_risk` | `0.5808` | `0.7577` | `-0.717` | `0.320` |

### rescue_vs_damage

| Feature | Pos Mean | Neg Mean | Cohen d | AUC pos high |
| --- | ---: | ---: | ---: | ---: |
| `wf_prob_min_mean` | `0.1723` | `0.0620` | `0.968` | `0.800` |
| `delta_cluster_weight_mass_vs_sample0` | `0.0313` | `-0.2825` | `0.948` | `0.761` |
| `delta_wf_prob_min_mean_vs_sample0` | `0.0250` | `-0.1799` | `0.895` | `0.752` |
| `qwen_missing_constraint_risk` | `0.6247` | `0.8080` | `-0.842` | `0.288` |
| `qwen_overclaim_risk` | `0.2022` | `0.4843` | `-0.820` | `0.261` |
| `qwen_structured_verifier_confidence` | `0.8874` | `0.6825` | `0.810` | `0.757` |
| `qwen_contradiction_risk` | `0.2616` | `0.5267` | `-0.770` | `0.273` |
| `delta_qwen_factual_residual_vs_sample0` | `-0.2603` | `0.2524` | `-0.736` | `0.238` |
| `delta_qwen_world_knowledge_conflict_risk_vs_sample0` | `-0.1185` | `0.1675` | `-0.730` | `0.208` |
| `wf_entropy_max_mean` | `2.9147` | `3.9231` | `-0.690` | `0.305` |

### stable_correct_vs_stable_hallucination

| Feature | Pos Mean | Neg Mean | Cohen d | AUC pos high |
| --- | ---: | ---: | ---: | ---: |
| `top2_weight_margin` | `0.8919` | `0.5760` | `1.012` | `0.732` |
| `cluster_weight_mass` | `0.9232` | `0.6888` | `0.975` | `0.725` |
| `cluster_size` | `7.3229` | `5.3987` | `0.958` | `0.727` |
| `fragmentation_entropy` | `0.1975` | `0.7225` | `-0.935` | `0.274` |
| `stable_score` | `0.6760` | `0.5781` | `0.890` | `0.707` |
| `wf_prob_min_mean` | `0.5569` | `0.3616` | `0.728` | `0.697` |
| `wf_entropy_max_mean` | `1.3226` | `2.2604` | `-0.697` | `0.276` |
| `wf_entropy_roughness` | `0.0963` | `0.2718` | `-0.631` | `0.253` |
| `distance_to_sample0_entropy_logprob` | `0.1719` | `0.6461` | `-0.548` | `0.370` |
| `riemann_anisotropy` | `0.7949` | `0.6201` | `0.442` | `0.616` |

## 3. Interpretation

- 如果 factual residual 在 `stable_correct_vs_hallucination` 上超过 stability/waveform，说明稳定幻觉的关键缺口确实是事实约束残差。
- 如果 Riemann geometry 在 rescue/damage 或 stable hallucination 上提供额外 AUC，说明 basin manifold shape 有独立理论价值。
- 新增 structured Qwen verifier 显式检查 answer slot、question constraints、entity/number/time consistency、basin consensus、overclaim 和 world-knowledge conflict。
- `strong_verifier_controller` 用 structured residual 的 sample0 delta 连接已有 basin dynamics，主要面向 rescue/damage 风险控制，而不是替代所有 theory-core 特征。
