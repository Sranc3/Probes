# Teacher-Free Probes: TriviaQA -> Multi-OOD Evaluation

Trained on full TriviaQA (1000 rows). Each OOD dataset: greedy sample0 with `strict_correct` against ideal answers.


## OOD setup

| dataset | n | base_acc |
| --- | --- | --- |
| hotpotqa | 500 | 0.4500 |
| nq | 500 | 0.5060 |

## Best AUROC per probe (per OOD dataset)


### hotpotqa  (n=500, base_acc=0.4500)

| probe | layer | auroc | aurc | sel_acc@0.25 | sel_acc@0.50 | sel_acc@0.75 | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEPs_logreg | 56.0000 | 0.7910 | 0.3459 | 0.7520 | 0.6640 | 0.5547 | 0.2377 | 0.2184 |
| DCP_mlp | 64.0000 | 0.7715 | 0.3547 | 0.7680 | 0.6600 | 0.5467 | 0.2583 | 0.2535 |
| SEPs_ridge | 56.0000 | 0.5892 | 0.4871 | 0.5520 | 0.5240 | 0.4800 | nan | nan |

### nq  (n=500, base_acc=0.5060)

| probe | layer | auroc | aurc | sel_acc@0.25 | sel_acc@0.50 | sel_acc@0.75 | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEPs_logreg | 56.0000 | 0.7009 | 0.3529 | 0.6720 | 0.6520 | 0.6000 | 0.3009 | 0.2886 |
| DCP_mlp | 64.0000 | 0.6538 | 0.3973 | 0.6480 | 0.6120 | 0.5893 | 0.3367 | 0.3241 |
| SEPs_ridge | 48.0000 | 0.5970 | 0.4375 | 0.6000 | 0.5720 | 0.5520 | nan | nan |

## DCP-MLP vs SEPs (paired bootstrap, per OOD dataset)

| OOD | comparison | mean ΔAUROC | 95% CI | p-value |
| --- | --- | --- | --- | --- |
| hotpotqa | DCP_mlp@L64 vs SEPs_ridge@L56 | +0.1817 | [+0.1225, +0.2424] | 0.000 |
| hotpotqa | DCP_mlp@L64 vs SEPs_logreg@L56 | -0.0195 | [-0.0511, +0.0122] | 0.221 |
| nq | DCP_mlp@L64 vs SEPs_ridge@L48 | +0.0576 | [+0.0025, +0.1133] | 0.046 |
| nq | DCP_mlp@L64 vs SEPs_logreg@L56 | -0.0466 | [-0.0759, -0.0157] | 0.004 |

## Per-probe AUROC across OOD datasets (best-layer)

AUROC table:

| probe | hotpotqa | nq |
| --- | --- | --- |
| DCP_mlp | 0.7715 | 0.6538 |
| SEPs_logreg | 0.7910 | 0.7009 |
| SEPs_ridge | 0.5892 | 0.5970 |

Best-layer table:

| probe | hotpotqa | nq |
| --- | --- | --- |
| DCP_mlp | 64 | 64 |
| SEPs_logreg | 56 | 56 |
| SEPs_ridge | 56 | 48 |
