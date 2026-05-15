# Plan_opus_selective — Selective Prediction & Routing over Basin/Anchor Features

**框架 A 的实证版本**：把 50+ basin/anchor 特征当 *risk score* 输入，做 selective prediction 与 3-action routing（answer / call_teacher / abstain），而不是再用它们去 reshape generation distribution。

完整结论与表格见 [`reports/SELECTIVE_PREDICTION_REPORT.md`](reports/SELECTIVE_PREDICTION_REPORT.md)。

## 一句话结论

| 框架 | 方法 | 数字 |
|---|---|---|
| C: generation shaping (Plan_opus) | 最佳 LoRA 微调 | sample0 strict 0.660 → 0.674 (+1.4 点，单次 lucky checkpoint) |
| **A: selective prediction (本目录)** | **logreg on basin+anchor features** | **sel_acc@50% coverage = 0.994** (50% 题答错率从 44% 降到 0.4%) |
| **A: routing** | **logreg:teacher 路由** | **risk-adjusted utility −0.74 → +0.51 / 题** |

同一份特征。不同 framing。两个数量级的差异。

## 核心数字（双 regime 一致 ⇒ 不是过拟合）

### Setting: sample0 (base_acc = 0.566)

| Predictor | AUROC | sel_acc@25% | sel_acc@50% | AURC | Brier | ECE |
|---|---|---|---|---|---|---|
| **logreg:all (5-fold CV)** | **0.969** | **1.000** | **0.994** | **0.124** | 0.054 | 0.020 |
| logreg:all (cross-seed) | 0.970 | 1.000 | 0.996 | 0.123 | 0.053 | 0.020 |
| logreg:teacher (cross-seed) | 0.969 | 1.000 | 0.994 | 0.124 | 0.053 | 0.015 |
| logreg:self only (no teacher call) | 0.808 | 0.836 | 0.800 | 0.225 | 0.175 | 0.065 |
| `teacher_best_similarity` 单特征 | 0.906 | 0.944 | 0.916 | 0.159 | — | — |
| `−token_mean_entropy` 单特征 | 0.777 | 0.856 | 0.770 | 0.237 | — | — |
| `verifier_score_v05` (Plan_gpt55 旧版) | 0.480 | 0.388 | 0.542 | 0.516 | — | — |

### Setting: fixed_8 (base_acc = 0.587)

| Predictor | AUROC | sel_acc@25% | sel_acc@50% | AURC |
|---|---|---|---|---|
| **logreg:all (cross-seed)** | **0.969** | **1.000** | **1.000** | **0.112** |
| logreg:all (5-fold CV) | 0.950 | 1.000 | 0.996 | 0.118 |
| `verifier_score_v05` (旧) | 0.370 | 0.364 | 0.458 | 0.545 |

### Routing (default cost: correct +1, wrong −3, teacher −0.3, abstain 0)

| Setting | Method | answer 数 | answer accuracy | utility/题 |
|---|---|---|---|---|
| sample0 | always_answer (baseline) | 1000 | 0.566 | **−0.736** |
| sample0 | always_abstain | 0 | — | 0.000 |
| **sample0** | **logreg:teacher routing** | **487** | **1.000** | **+0.513** |
| sample0 | logreg:self routing (no teacher needed) | 491 | 0.805 | +0.107 |

## 重要发现（独立于本框架）

1. **`verifier_score_v05` 在 anchor rows 上 AUROC = 0.48**——基本是反向的。Plan_gpt55 训出来的 learned verifier 不可直接 reuse 到 anchor 数据。
2. **`teacher_best_similarity` 单个 feature AUROC 0.906**，强于所有 self-introspection 类特征——印证"50 个特征里只有 teacher_* 真携带 ground-truth 信息"。
3. **5-fold CV 与 cross-seed 一致 (差异 < 0.01)**——logreg 在这份数据上不存在 question-level 过拟合。
4. **MLP 不如 LogReg**（cv: 0.960 vs 0.969），cross-seed ECE 飙到 0.19——非线性容易在小样本上过拟合校准。

## 复现命令

```bash
python scripts/run_selective_experiments.py    # ~30s
python scripts/run_routing_analysis.py         # ~5s
python scripts/build_report_tables.py          # ~1s
```

## 接下来必须做的三件事

1. **跨数据集**：在 NaturalQuestions / SciQ 上重做，验证不是 TriviaQA-overfit
2. **In test500 (offset=500)**：在 Plan_opus 评测的同一份 500 题上重新生成候选 + 抽特征，得到 selective vs generation 框架**完全可比**的数字
3. **去 teacher 依赖**：用 SelfCheckGPT 风格的二次问询丰富 self-features，把 self-only AUROC 从 0.81 推到 0.85+，使整套 pipeline 不依赖外部 teacher 调用

## 目录

```
shared/
  data_utils.py    # 候选 → 问题级特征聚合 (selected + window agg + basin geometry)
  metrics.py       # AURC, sel_acc@k, Brier, ECE
  routing.py       # 3-action utility curve
scripts/
  run_selective_experiments.py   # single-feature + LogReg + MLP，CV + cross-seed
  run_routing_analysis.py        # Pareto over (answer_threshold, defer_threshold)
  build_report_tables.py         # markdown 表
results/
  selective_metrics_long.csv    selective_metrics_pivot.csv
  question_features_<setting>.csv
  best_predictions_<setting>.csv  predictions_<setting>_<predictor>.csv
  routing_summary.csv  routing_pareto.csv
  metrics_table.md  routing_table.md
reports/SELECTIVE_PREDICTION_REPORT.md   # 完整报告
```
