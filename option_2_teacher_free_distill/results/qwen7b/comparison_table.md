# Teacher-Free Distillation: Head-to-Head Comparison

Our probes operate on a **single forward pass** of Qwen on the prompt (no generation, no teacher). The Plan_opus_selective baselines instead consume features computed from K=8 stochastic samples (and teacher calls for the *teacher*/*all* variants).


## Regime: cross_seed

| cost_tier | probe | best_layer | n | base_acc | auroc | aurc | sel_acc@0.25 | sel_acc@0.50 | sel_acc@0.75 | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K=1 prompt only | SEPs_logreg | 20.0000 | 1000.0000 | 0.5660 | 0.9618 | 0.1325 | 0.9720 | 0.9600 | 0.7427 | 0.0553 | 0.0523 |
| K=1 prompt only | DCP_mlp | 20.0000 | 1000.0000 | 0.5660 | 0.9608 | 0.1333 | 0.9840 | 0.9520 | 0.7413 | 0.0560 | 0.0560 |
| K=1 prompt only | ARD_ridge | 20.0000 | 1000.0000 | 0.5660 | 0.9244 | 0.1533 | 0.9760 | 0.9540 | 0.7120 | 0.0888 | 0.0458 |
| K=1 prompt only | ARD_mlp | 24.0000 | 1000.0000 | 0.5660 | 0.9222 | 0.1504 | 0.9560 | 0.9540 | 0.7107 | 0.0889 | 0.0499 |
| K=1 prompt only | SEPs_ridge | 27.0000 | 1000.0000 | 0.5660 | 0.8022 | 0.2313 | 0.8360 | 0.7860 | 0.6987 | — | — |
| K=8 self only (no teacher) | logreg:self | — | 1000.0000 | 0.5660 | 0.8185 | 0.2129 | 0.8920 | 0.8000 | 0.6947 | 0.1826 | 0.1004 |
| K=8 + teacher API call | logreg:all | — | 1000.0000 | 0.5660 | 0.9697 | 0.1233 | 1.0000 | 0.9960 | 0.7360 | 0.0532 | 0.0199 |
| K=8 + teacher API call | logreg:teacher | — | 1000.0000 | 0.5660 | 0.9685 | 0.1238 | 1.0000 | 0.9940 | 0.7333 | 0.0527 | 0.0150 |
| K=8 + teacher API call | mlp:all | — | 1000.0000 | 0.5660 | 0.9471 | 0.1330 | 1.0000 | 0.9460 | 0.7267 | 0.1209 | 0.1902 |

## Regime: cv_5fold

| cost_tier | probe | best_layer | n | base_acc | auroc | aurc | sel_acc@0.25 | sel_acc@0.50 | sel_acc@0.75 | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K=1 prompt only | DCP_mlp | 20.0000 | 1000.0000 | 0.5660 | 0.7985 | 0.2143 | 0.8960 | 0.7880 | 0.6773 | 0.2355 | 0.2175 |
| K=1 prompt only | ARD_mlp | 20.0000 | 1000.0000 | 0.5660 | 0.7898 | 0.2279 | 0.8520 | 0.7780 | 0.6787 | 0.1850 | 0.0623 |
| K=1 prompt only | SEPs_logreg | 24.0000 | 1000.0000 | 0.5660 | 0.7894 | 0.2202 | 0.9000 | 0.7900 | 0.6600 | 0.2259 | 0.1943 |
| K=1 prompt only | ARD_ridge | 24.0000 | 1000.0000 | 0.5660 | 0.7743 | 0.2359 | 0.8600 | 0.7800 | 0.6693 | 0.1917 | 0.0560 |
| K=1 prompt only | SEPs_ridge | 27.0000 | 1000.0000 | 0.5660 | 0.7471 | 0.2543 | 0.8400 | 0.7360 | 0.6560 | — | — |
| K=8 self only (no teacher) | logreg:self | — | 1000.0000 | 0.5660 | 0.8082 | 0.2247 | 0.8360 | 0.8000 | 0.7040 | 0.1748 | 0.0645 |
| K=8 + teacher API call | logreg:all | — | 1000.0000 | 0.5660 | 0.9691 | 0.1236 | 1.0000 | 0.9940 | 0.7387 | 0.0539 | 0.0203 |
| K=8 + teacher API call | logreg:teacher | — | 1000.0000 | 0.5660 | 0.9642 | 0.1253 | 1.0000 | 0.9900 | 0.7280 | 0.0551 | 0.0195 |
| K=8 + teacher API call | mlp:all | — | 1000.0000 | 0.5660 | 0.9600 | 0.1270 | 1.0000 | 0.9820 | 0.7280 | 0.0638 | 0.0255 |

## Regime: single_feature

| cost_tier | probe | best_layer | n | base_acc | auroc | aurc | sel_acc@0.25 | sel_acc@0.50 | sel_acc@0.75 | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single feature (K=8) | teacher_best_similarity_sel | — | 1000.0000 | 0.5660 | 0.9058 | 0.1593 | 0.9440 | 0.9160 | 0.7160 | — | — |
| single feature (K=8) | teacher_support_mass_sel | — | 1000.0000 | 0.5660 | 0.8955 | 0.1655 | 0.9320 | 0.9360 | 0.7013 | — | — |
| single feature (K=8) | anchor_score_noleak_sel | — | 1000.0000 | 0.5660 | 0.8856 | 0.1846 | 0.9400 | 0.9320 | 0.6973 | — | — |
| single feature (K=8) | neg_qwen_only_stable_mass_sel | — | 1000.0000 | 0.5660 | 0.8297 | 0.1901 | 0.9320 | 0.8760 | 0.6640 | — | — |
| single feature (K=8) | neg_semantic_entropy_set_sel | — | 1000.0000 | 0.5660 | 0.7771 | 0.2705 | 0.7920 | 0.7740 | 0.6987 | — | — |
| single feature (K=8) | neg_token_mean_entropy_sel | — | 1000.0000 | 0.5660 | 0.7771 | 0.2370 | 0.8560 | 0.7700 | 0.6760 | — | — |
| single feature (K=8) | logprob_avg_sel | — | 1000.0000 | 0.5660 | 0.7713 | 0.2405 | 0.8640 | 0.7620 | 0.6720 | — | — |
| single feature (K=8) | cluster_size_sel | — | 1000.0000 | 0.5660 | 0.7639 | 0.2942 | 0.7560 | 0.7540 | 0.7040 | — | — |
| single feature (K=8) | group_top1_basin_share | — | 1000.0000 | 0.5660 | 0.5000 | 0.4459 | 0.5480 | 0.5800 | 0.5680 | — | — |
| single feature (K=8) | neg_group_basin_entropy | — | 1000.0000 | 0.5660 | 0.5000 | 0.4459 | 0.5480 | 0.5800 | 0.5680 | — | — |
| single feature (K=8) | verifier_score_v05_sel | — | 1000.0000 | 0.5660 | 0.4799 | 0.5157 | 0.3880 | 0.5420 | 0.6333 | — | — |
