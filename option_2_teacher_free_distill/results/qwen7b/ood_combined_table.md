# Teacher-Free Probes: TriviaQA -> Multi-OOD Evaluation

Trained on full TriviaQA (1000 rows). Each OOD dataset: greedy sample0 with `strict_correct` against ideal answers.


## OOD setup

| dataset | n | base_acc |
| --- | --- | --- |
| hotpotqa | 500 | 0.3300 |
| nq | 500 | 0.3280 |

## Best AUROC per probe (per OOD dataset)


### hotpotqa  (n=500, base_acc=0.3300)

| probe | layer | auroc | aurc | sel_acc@0.25 | sel_acc@0.50 | sel_acc@0.75 | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DCP_mlp | 16.0000 | 0.7302 | 0.5158 | 0.6080 | 0.4760 | 0.3920 | 0.2660 | 0.2687 |
| SEPs_logreg | 16.0000 | 0.7244 | 0.5184 | 0.5920 | 0.4920 | 0.3840 | 0.2940 | 0.2897 |
| SEPs_ridge | 16.0000 | 0.6838 | 0.5315 | 0.5360 | 0.4440 | 0.3840 | nan | nan |

### nq  (n=500, base_acc=0.3280)

| probe | layer | auroc | aurc | sel_acc@0.25 | sel_acc@0.50 | sel_acc@0.75 | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DCP_mlp | 20.0000 | 0.6673 | 0.5688 | 0.5120 | 0.4360 | 0.3867 | 0.3744 | 0.3690 |
| SEPs_logreg | 20.0000 | 0.6614 | 0.5768 | 0.4880 | 0.4320 | 0.3893 | 0.3558 | 0.3472 |
| SEPs_ridge | 16.0000 | 0.6190 | 0.6007 | 0.3760 | 0.4120 | 0.3787 | nan | nan |

## DCP-MLP vs SEPs (paired bootstrap, per OOD dataset)

| OOD | comparison | mean ΔAUROC | 95% CI | p-value |
| --- | --- | --- | --- | --- |
| hotpotqa | DCP_mlp@L16 vs SEPs_ridge@L16 | +0.0467 | [-0.0020, +0.0991] | 0.063 |
| hotpotqa | DCP_mlp@L16 vs SEPs_logreg@L16 | +0.0066 | [-0.0234, +0.0364] | 0.688 |
| nq | DCP_mlp@L20 vs SEPs_ridge@L16 | +0.0498 | [-0.0068, +0.1040] | 0.081 |
| nq | DCP_mlp@L20 vs SEPs_logreg@L20 | +0.0065 | [-0.0274, +0.0434] | 0.720 |

## Per-probe AUROC across OOD datasets (best-layer)

AUROC table:

| probe | hotpotqa | nq |
| --- | --- | --- |
| DCP_mlp | 0.7302 | 0.6673 |
| SEPs_logreg | 0.7244 | 0.6614 |
| SEPs_ridge | 0.6838 | 0.6190 |

Best-layer table:

| probe | hotpotqa | nq |
| --- | --- | --- |
| DCP_mlp | 16 | 20 |
| SEPs_logreg | 16 | 20 |
| SEPs_ridge | 16 | 16 |
