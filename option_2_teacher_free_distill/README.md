# Option 2: Teacher-Free Distillation for Selective Prediction

> Goal: At inference time, **predict whether the model's sample0 answer is
> correct using a single forward pass** of Qwen2.5-7B-Instruct on the prompt -
> no K-sampling, no external teacher, no answer generation needed.
>
> Headline comparison: **Our distilled probes vs. Semantic Entropy Probes
> (SEPs, Kossen et al. 2024 / NeurIPS)** under matched single-forward-pass
> deployment cost.

## Why this experiment

`Plan_opus_selective` showed that a logistic-regression with cross-model
anchor + self-introspection features reaches AUROC 0.97 / sel_acc@50% = 0.994.
But those features cost **8 sampling forward passes + a teacher API call** per
question. This is too expensive for production.

Teacher-free distillation tries to compress that confidence signal into a
small head on top of Qwen's own hidden states, so deployment is:

```
prompt -> Qwen forward (1 pass) -> hidden state -> 2-layer MLP -> P(correct)
```

If we can match SEPs (or beat it) with this setup, we have a methodological
contribution that directly defends a top-venue submission.

## Probes implemented

For every (layer, pooling) we train and evaluate three probes:

| Name | Target | Loss | Reference |
|---|---|---|---|
| **SEPs** | `semantic_entropy_weighted_set` (regression) | MSE | Kossen et al., NeurIPS 2024 |
| **DCP** (Direct Correctness Probe) | `strict_correct` (classification) | BCE | ours |
| **ARD** (Anchor Regression Distillation) | 7-dim teacher anchor features | MSE then logreg head | ours |

`pooling` ∈ {`prompt_last`, `answer_last`, `answer_mean`} - default
`prompt_last` for SEPs-style 1-pass deployment; the other two are ablations
that allow the probe to peek at the answer tokens.

## Layout

```
option_2_teacher_free_distill/
├── README.md
├── shared/
│   ├── data_utils.py          # link to anchor labels and per-question targets
│   └── probe_utils.py         # MLP / logreg trainers, calibration, GroupKFold
├── scripts/
│   ├── extract_hidden_states.py    # Qwen prompt-only forward pass -> .npz cache (TriviaQA)
│   ├── train_probes.py             # train SEPs/DCP/ARD across layers (ID)
│   ├── evaluate_probes.py          # selective-prediction metrics + per-item dumps (ID)
│   ├── bootstrap_compare.py        # paired bootstrap CIs vs Plan_opus_selective baselines (ID)
│   ├── build_results_table.py      # markdown ID comparison table
│   ├── prepare_hotpotqa_ood.py     # extract HotpotQA hidden + greedy + strict_correct
│   ├── prepare_nq_ood.py           # extract NQ-Open hidden + greedy + strict_correct
│   └── evaluate_ood.py             # multi-OOD evaluation + per-dataset bootstrap
├── runs/                       # cached hidden states, probe predictions, training logs
├── results/                    # eval CSVs + markdown tables (ID + ood_<name>_*)
└── reports/                    # TEACHER_FREE_REPORT.md (v2)
```

## Reproduce

See section 7 of `reports/TEACHER_FREE_REPORT.md` for the full pipeline. Quick
recipe:

```bash
cd /zhutingqi/song/option_2_teacher_free_distill

# In-distribution (TriviaQA)
conda run -n vllm python scripts/extract_hidden_states.py
conda run -n vllm python scripts/train_probes.py
conda run -n vllm python scripts/evaluate_probes.py
conda run -n vllm python scripts/bootstrap_compare.py --n-boot 2000

# OOD-1 (HotpotQA dev_distractor, with context)
conda run -n vllm python scripts/prepare_hotpotqa_ood.py

# OOD-2 (Natural Questions, NQ-Open validation, no context)
conda run -n vllm python scripts/prepare_nq_ood.py \
    --input /zhutingqi/song/datasets/nq_open --split validation \
    --max-questions 500 --output runs/nq_ood.npz

# Multi-OOD evaluation + bootstrap (HotpotQA + NQ together)
conda run -n vllm python scripts/evaluate_ood.py \
    --ood-cache hotpotqa=runs/hotpotqa_ood.npz \
    --ood-cache nq=runs/nq_ood.npz \
    --n-boot 2000
```
