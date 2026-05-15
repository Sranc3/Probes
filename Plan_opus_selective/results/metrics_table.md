
### Setting: sample0

| setting | predictor | regime | n | base_acc | auroc | aurc | sel_acc@0.25 | sel_acc@0.50 | sel_acc@0.75 | brier | ece |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sample0 | teacher_best_similarity_sel | single_feature | 1000 | 0.5660 | 0.9058 | 0.1593 | 0.9440 | 0.9160 | 0.7160 | — | — |
| sample0 | teacher_support_mass_sel | single_feature | 1000 | 0.5660 | 0.8955 | 0.1655 | 0.9320 | 0.9360 | 0.7013 | — | — |
| sample0 | anchor_score_noleak_sel | single_feature | 1000 | 0.5660 | 0.8856 | 0.1846 | 0.9400 | 0.9320 | 0.6973 | — | — |
| sample0 | neg_qwen_only_stable_mass_sel | single_feature | 1000 | 0.5660 | 0.8297 | 0.1901 | 0.9320 | 0.8760 | 0.6640 | — | — |
| sample0 | neg_semantic_entropy_set_sel | single_feature | 1000 | 0.5660 | 0.7771 | 0.2705 | 0.7920 | 0.7740 | 0.6987 | — | — |
| sample0 | neg_token_mean_entropy_sel | single_feature | 1000 | 0.5660 | 0.7771 | 0.2370 | 0.8560 | 0.7700 | 0.6760 | — | — |
| sample0 | logprob_avg_sel | single_feature | 1000 | 0.5660 | 0.7713 | 0.2405 | 0.8640 | 0.7620 | 0.6720 | — | — |
| sample0 | cluster_size_sel | single_feature | 1000 | 0.5660 | 0.7639 | 0.2942 | 0.7560 | 0.7540 | 0.7040 | — | — |
| sample0 | group_top1_basin_share | single_feature | 1000 | 0.5660 | 0.5000 | 0.4459 | 0.5480 | 0.5800 | 0.5680 | — | — |
| sample0 | neg_group_basin_entropy | single_feature | 1000 | 0.5660 | 0.5000 | 0.4459 | 0.5480 | 0.5800 | 0.5680 | — | — |
| sample0 | verifier_score_v05_sel | single_feature | 1000 | 0.5660 | 0.4799 | 0.5157 | 0.3880 | 0.5420 | 0.6333 | — | — |
| sample0 | logreg:all | cross_seed | 1000 | 0.5660 | 0.9697 | 0.1233 | 1.0000 | 0.9960 | 0.7360 | 0.0532 | 0.0199 |
| sample0 | logreg:teacher | cross_seed | 1000 | 0.5660 | 0.9685 | 0.1238 | 1.0000 | 0.9940 | 0.7333 | 0.0527 | 0.0150 |
| sample0 | mlp:all | cross_seed | 1000 | 0.5660 | 0.9471 | 0.1330 | 1.0000 | 0.9460 | 0.7267 | 0.1209 | 0.1902 |
| sample0 | logreg:self | cross_seed | 1000 | 0.5660 | 0.8185 | 0.2129 | 0.8920 | 0.8000 | 0.6947 | 0.1826 | 0.1004 |
| sample0 | logreg:all | cv_5fold | 1000 | 0.5660 | 0.9691 | 0.1236 | 1.0000 | 0.9940 | 0.7387 | 0.0539 | 0.0203 |
| sample0 | logreg:teacher | cv_5fold | 1000 | 0.5660 | 0.9642 | 0.1253 | 1.0000 | 0.9900 | 0.7280 | 0.0551 | 0.0195 |
| sample0 | mlp:all | cv_5fold | 1000 | 0.5660 | 0.9600 | 0.1270 | 1.0000 | 0.9820 | 0.7280 | 0.0638 | 0.0255 |
| sample0 | logreg:self | cv_5fold | 1000 | 0.5660 | 0.8082 | 0.2247 | 0.8360 | 0.8000 | 0.7040 | 0.1748 | 0.0645 |

### Setting: fixed_k

| setting | predictor | regime | n | base_acc | auroc | aurc | sel_acc@0.25 | sel_acc@0.50 | sel_acc@0.75 | brier | ece |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_k | teacher_best_similarity_sel | single_feature | 1000 | 0.5870 | 0.9000 | 0.1474 | 0.9480 | 0.9260 | 0.7387 | — | — |
| fixed_k | teacher_support_mass_sel | single_feature | 1000 | 0.5870 | 0.8847 | 0.1566 | 0.9400 | 0.9360 | 0.7240 | — | — |
| fixed_k | anchor_score_noleak_sel | single_feature | 1000 | 0.5870 | 0.8832 | 0.1644 | 0.9440 | 0.9360 | 0.7107 | — | — |
| fixed_k | neg_qwen_only_stable_mass_sel | single_feature | 1000 | 0.5870 | 0.8312 | 0.1770 | 0.9400 | 0.8800 | 0.6880 | — | — |
| fixed_k | neg_token_mean_entropy_sel | single_feature | 1000 | 0.5870 | 0.7725 | 0.2252 | 0.8560 | 0.7840 | 0.6920 | — | — |
| fixed_k | neg_semantic_entropy_set_sel | single_feature | 1000 | 0.5870 | 0.7575 | 0.2680 | 0.7880 | 0.7760 | 0.7120 | — | — |
| fixed_k | logprob_avg_sel | single_feature | 1000 | 0.5870 | 0.7553 | 0.2340 | 0.8640 | 0.7700 | 0.6907 | — | — |
| fixed_k | cluster_size_sel | single_feature | 1000 | 0.5870 | 0.7459 | 0.2879 | 0.7560 | 0.7520 | 0.7173 | — | — |
| fixed_k | neg_group_basin_entropy | single_feature | 1000 | 0.5870 | 0.7170 | 0.2677 | 0.7800 | 0.7520 | 0.6867 | — | — |
| fixed_k | group_top1_basin_share | single_feature | 1000 | 0.5870 | 0.7123 | 0.2696 | 0.7800 | 0.7520 | 0.6760 | — | — |
| fixed_k | verifier_score_v05_sel | single_feature | 1000 | 0.5870 | 0.3702 | 0.5448 | 0.3640 | 0.4580 | 0.5733 | — | — |
| fixed_k | logreg:all | cross_seed | 1000 | 0.5870 | 0.9685 | 0.1118 | 1.0000 | 1.0000 | 0.7600 | 0.0556 | 0.0287 |
| fixed_k | logreg:teacher | cross_seed | 1000 | 0.5870 | 0.9592 | 0.1150 | 1.0000 | 1.0000 | 0.7480 | 0.0607 | 0.0452 |
| fixed_k | mlp:all | cross_seed | 1000 | 0.5870 | 0.9544 | 0.1175 | 1.0000 | 0.9720 | 0.7520 | 0.0828 | 0.0677 |
| fixed_k | logreg:self | cross_seed | 1000 | 0.5870 | 0.7854 | 0.2197 | 0.8240 | 0.8000 | 0.7080 | 0.1993 | 0.1115 |
| fixed_k | mlp:all | cv_5fold | 1000 | 0.5870 | 0.9531 | 0.1171 | 1.0000 | 0.9900 | 0.7480 | 0.0684 | 0.0473 |
| fixed_k | logreg:all | cv_5fold | 1000 | 0.5870 | 0.9502 | 0.1182 | 1.0000 | 0.9960 | 0.7440 | 0.0765 | 0.0499 |
| fixed_k | logreg:teacher | cv_5fold | 1000 | 0.5870 | 0.9414 | 0.1231 | 0.9920 | 0.9960 | 0.7387 | 0.0705 | 0.0299 |
| fixed_k | logreg:self | cv_5fold | 1000 | 0.5870 | 0.7777 | 0.2277 | 0.8520 | 0.7780 | 0.7013 | 0.1913 | 0.0850 |
