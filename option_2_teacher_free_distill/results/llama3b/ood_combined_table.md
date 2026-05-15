# Teacher-Free Probes: TriviaQA -> Multi-OOD Evaluation

Trained on full TriviaQA (1000 rows). Each OOD dataset: greedy sample0 with `strict_correct` against ideal answers.


## OOD setup

| dataset | n | base_acc |
| --- | --- | --- |
| hotpotqa | 500 | 0.3260 |
| nq | 500 | 0.4760 |

## Best AUROC per probe (per OOD dataset)


### hotpotqa  (n=500, base_acc=0.3260)

| probe | layer | auroc | aurc | sel_acc@0.25 | sel_acc@0.50 | sel_acc@0.75 | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEPs_logreg | 16.0000 | 0.6767 | 0.5377 | 0.4880 | 0.4400 | 0.3787 | 0.2690 | 0.2551 |
| SEPs_ridge | 16.0000 | 0.6644 | 0.5433 | 0.4960 | 0.4320 | 0.3627 | nan | nan |
| DCP_mlp | 16.0000 | 0.6571 | 0.5600 | 0.4320 | 0.4360 | 0.3787 | 0.3305 | 0.3137 |

### nq  (n=500, base_acc=0.4760)

| probe | layer | auroc | aurc | sel_acc@0.25 | sel_acc@0.50 | sel_acc@0.75 | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DCP_mlp | 24.0000 | 0.6127 | 0.4467 | 0.5680 | 0.5600 | 0.5280 | 0.3842 | 0.3774 |
| SEPs_logreg | 12.0000 | 0.6040 | 0.4562 | 0.6000 | 0.5560 | 0.5173 | 0.3658 | 0.3416 |
| SEPs_ridge | 12.0000 | 0.5379 | 0.5006 | 0.5120 | 0.5000 | 0.4933 | nan | nan |

## DCP-MLP vs SEPs (paired bootstrap, per OOD dataset)

| OOD | comparison | mean ΔAUROC | 95% CI | p-value |
| --- | --- | --- | --- | --- |
| hotpotqa | DCP_mlp@L16 vs SEPs_ridge@L16 | -0.0074 | [-0.0643, +0.0517] | 0.803 |
| hotpotqa | DCP_mlp@L16 vs SEPs_logreg@L16 | -0.0197 | [-0.0484, +0.0093] | 0.171 |
| nq | DCP_mlp@L24 vs SEPs_ridge@L12 | +0.0742 | [+0.0045, +0.1407] | 0.037 |
| nq | DCP_mlp@L24 vs SEPs_logreg@L12 | +0.0081 | [-0.0418, +0.0599] | 0.760 |

## Per-probe AUROC across OOD datasets (best-layer)

AUROC table:

| probe | hotpotqa | nq |
| --- | --- | --- |
| DCP_mlp | 0.6571 | 0.6127 |
| SEPs_logreg | 0.6767 | 0.6040 |
| SEPs_ridge | 0.6644 | 0.5379 |

Best-layer table:

| probe | hotpotqa | nq |
| --- | --- | --- |
| DCP_mlp | 16 | 24 |
| SEPs_logreg | 16 | 12 |
| SEPs_ridge | 16 | 12 |
