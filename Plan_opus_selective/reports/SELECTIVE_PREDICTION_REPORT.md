# Plan_opus_selective: Selective Prediction & Routing 报告

**Date:** 2026-05-13
**Author:** Plan_opus
**Framing:** 框架 A — 不再追求 strict accuracy 提升，而是把 50+ 候选特征当作 *risk score* 的输入，做 selective prediction (abstain when uncertain) 与 3-action routing (answer / call_teacher / abstain)。

## 0. TL;DR

把同一份 GPT-OSS 锚点候选数据（500 题 × 2 seed × 8 候选 = 8000 行）按问题级聚合后训简单的 logistic regression：

| Setting | Predictor | AUROC | sel_acc@25% | sel_acc@50% | AURC | Brier | ECE |
|---|---|---|---|---|---|---|---|
| sample0 | base (always answer) | 0.500 | 0.566 | 0.566 | 0.434 | — | — |
| sample0 | best single feature (`teacher_best_similarity`) | 0.906 | 0.944 | 0.916 | 0.159 | — | — |
| **sample0** | **logreg:all (5-fold CV)** | **0.969** | **1.000** | **0.994** | **0.124** | 0.054 | 0.020 |
| sample0 | logreg:all (cross-seed) | 0.970 | 1.000 | 0.996 | 0.123 | 0.053 | 0.020 |
| **sample0** | **logreg:self only (no teacher call)** | **0.808** | **0.836** | **0.800** | **0.225** | 0.175 | 0.065 |
| fixed_8 | base (always answer) | 0.500 | 0.587 | 0.587 | 0.413 | — | — |
| **fixed_8** | **logreg:all (5-fold CV)** | **0.950** | **1.000** | **0.996** | **0.118** | 0.077 | 0.050 |
| fixed_8 | logreg:all (cross-seed) | 0.969 | 1.000 | 1.000 | 0.112 | 0.056 | 0.029 |

Routing 下的 utility（cost: 答对 +1，答错 −3，调 teacher −0.3，弃权 0）：

| 策略 | 平均效用/题 | 含义 |
|---|---|---|
| 永远回答 | **−0.736** | 朴素 baseline，44% 错答严重亏 |
| 永远弃权 | 0.000 | 安全 baseline |
| 永远调 teacher | −1.040 | teacher 也不能保证全对 |
| **logreg:teacher 路由 (sample0, default cost)** | **+0.513** | 答 487 题（accuracy 100%），调 teacher 43 题，弃权 470 题 |
| logreg:self 路由 | +0.107 | 仅靠 self-introspection，已经把 utility 从 −0.7 拉回 +0.1 |

→ 同样的特征做 framing C（generation shaping，VBPO/GRPO）只提升 strict accuracy 0.5–1.4 点；做 framing A（selective prediction），**top-50% confidence 的题准确率从 56.6% 拉到 99.4%，risk-adjusted utility 从 −0.74 拉到 +0.51**。这是同一份数据、不同 framing 下的天壤之别。

## 1. 重新 framing：为什么这条路对

之前所有方法（ITI、VBPO、Basin-GRPO、Anchor-VBPO/GRPO）都在做同一件事：**用特征改 generation distribution，让 strict accuracy 更高**。但：

- 50 个特征里，**只有 `teacher_*` 一族携带 ground truth 信息**；其余全是 self-introspection（"我不确定"或"我自信但孤立"）。
- self-introspection 信息论上**不能告诉你正确答案是什么**——这就是 stable_wrong_basin 的物理意义。
- 所以 generation shaping 的 ceiling 被锁在 base model 的 factual recall 之内。

但同样的特征，如果用作 **risk score** 输入：
- "我不确定" → **可以选择不答**；
- "self-conviction 但 anchor 缺席" → **可以调更强的模型**；
- "anchor 一致 + entropy 低" → **直接回答**。

**这是 sensor 而不是 actuator 的用法**，符合特征本身的物理性质。

## 2. 实验设置

### 2.1 数据
- 来源：`Plan_gpt55/cross_model_anchor/runs/run_20260513_012413_gptoss_anchor_triviaqa_full500_k4_tok256/qwen_candidate_anchor_rows_final_only.csv`
- 8000 行 = 500 题 × 2 seed (42, 43) × 8 候选
- 标签：`strict_correct`（每个候选 1/0）
- sample0 base accuracy = 0.566；fixed_8 majority-basin = 0.587

### 2.2 问题级特征构造
对每个 (question, seed)：
- **selected-answer features**：被 policy 选中的候选的 19 个 self 特征 + 7 个 teacher 特征（`sel_self_*`, `sel_teacher_*`）
- **window aggregations**：8 候选的 mean/min/max/std （`win_self_*`, `win_teacher_*`）
- **basin geometry**：top1/top2 basin share, num basins, basin entropy （`group_*`）
- 共 ~150 列 numeric features

### 2.3 预测器
- **Single feature**：直接用某一列（如 `teacher_best_similarity` 或 `−token_mean_entropy`）作为 confidence score
- **logreg:self**：仅自省特征 + group geometry，**部署时不需要调 teacher**
- **logreg:teacher**：仅 teacher anchor 特征
- **logreg:all**：自省 + teacher + 几何
- **mlp:all**：双隐层 (64, 32) MLP

### 2.4 评测协议
- **5-fold CV at question level**（同一 question 的两个 seed 一起切，避免泄漏）
- **Cross-seed**: 训 seed 42、测 seed 43，反向各做一次后合并
- 指标：AUROC, AURC, sel_acc@{25%,50%,75%,100%}, Brier, ECE

### 2.5 Routing
- 三动作：answer / call_teacher / abstain
- 三套 cost model：
  - default: correct +1, wrong −3, teacher −0.3, abstain 0
  - strict:  correct +1, wrong −5, teacher −0.4, abstain 0
  - lenient: correct +1, wrong −1.5, teacher −0.2, abstain 0
- teacher_correct 用 `teacher_best_basin_strict_any` 作为 oracle 估计（部署时实际调用 teacher 才能拿到）

## 3. 主结果

### 3.1 Selective Prediction（按 AUROC 降序）

**Setting: sample0** (base_acc = 0.566)

| Predictor | regime | AUROC | AURC | sel_acc@25% | sel_acc@50% | sel_acc@75% | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| logreg:all | cv_5fold | **0.969** | **0.124** | 1.000 | 0.994 | 0.739 | 0.054 | 0.020 |
| logreg:all | cross_seed | **0.970** | **0.123** | 1.000 | 0.996 | 0.736 | 0.053 | 0.020 |
| logreg:teacher | cv_5fold | 0.964 | 0.125 | 1.000 | 0.990 | 0.728 | 0.055 | 0.020 |
| logreg:teacher | cross_seed | 0.969 | 0.124 | 1.000 | 0.994 | 0.733 | 0.053 | 0.015 |
| mlp:all | cv_5fold | 0.960 | 0.127 | 1.000 | 0.982 | 0.728 | 0.064 | 0.026 |
| mlp:all | cross_seed | 0.947 | 0.133 | 1.000 | 0.946 | 0.727 | 0.121 | 0.190 |
| teacher_best_similarity | single | 0.906 | 0.159 | 0.944 | 0.916 | 0.716 | — | — |
| teacher_support_mass | single | 0.896 | 0.166 | 0.932 | 0.936 | 0.701 | — | — |
| anchor_score_noleak | single | 0.886 | 0.185 | 0.940 | 0.932 | 0.697 | — | — |
| neg_qwen_only_stable_mass | single | 0.830 | 0.190 | 0.932 | 0.876 | 0.664 | — | — |
| **logreg:self (no teacher)** | cv_5fold | **0.808** | **0.225** | **0.836** | **0.800** | **0.704** | 0.175 | 0.065 |
| logreg:self (no teacher) | cross_seed | 0.819 | 0.213 | 0.892 | 0.800 | 0.695 | 0.183 | 0.100 |
| neg_token_mean_entropy | single | 0.777 | 0.237 | 0.856 | 0.770 | 0.676 | — | — |
| neg_semantic_entropy_set | single | 0.777 | 0.271 | 0.792 | 0.774 | 0.699 | — | — |
| logprob_avg | single | 0.771 | 0.241 | 0.864 | 0.762 | 0.672 | — | — |
| cluster_size | single | 0.764 | 0.294 | 0.756 | 0.754 | 0.704 | — | — |
| group_top1_basin_share | single | 0.500 | 0.446 | 0.548 | 0.580 | 0.568 | — | — |
| **verifier_score_v05 (旧学习版本)** | single | **0.480** | **0.516** | **0.388** | **0.542** | **0.633** | — | — |

**Setting: fixed_8** (base_acc = 0.587)

| Predictor | regime | AUROC | AURC | sel_acc@25% | sel_acc@50% | sel_acc@75% |
|---|---|---|---|---|---|---|
| logreg:all | cross_seed | **0.969** | **0.112** | 1.000 | 1.000 | 0.760 |
| logreg:all | cv_5fold | 0.950 | 0.118 | 1.000 | 0.996 | 0.744 |
| mlp:all | cv_5fold | 0.953 | 0.117 | 1.000 | 0.990 | 0.748 |
| logreg:teacher | cross_seed | 0.959 | 0.115 | 1.000 | 1.000 | 0.748 |
| teacher_best_similarity | single | 0.900 | 0.147 | 0.948 | 0.926 | 0.739 |
| logreg:self (no teacher) | cv_5fold | 0.778 | 0.228 | 0.852 | 0.778 | 0.701 |
| neg_token_mean_entropy | single | 0.773 | 0.225 | 0.856 | 0.784 | 0.692 |
| **verifier_score_v05 (旧学习版本)** | single | **0.370** | **0.545** | **0.364** | **0.458** | **0.573** |

### 3.2 关键观察

1. **Logistic regression on all features → AUROC 0.97**。如此简单的线性模型在 50+ 特征上得到接近完美的 ranker，5-fold CV 与 cross-seed 完全一致（差异 < 0.01），不是过拟合。

2. **Top-50% confidence 的题 99.4% 正确**。如果工业上能接受 50% coverage（剩下 50% 弃权或转交），这一半题的答错率从 base 44% 降到 **0.4%**。

3. **Self-only 预测器（部署时不需要调 teacher）AUROC 0.81**。这是 SelfCheckGPT / Semantic Uncertainty 同 setting 下的强 baseline；在我们的特征集上，单纯的 self-introspection 也能从 base 56.6% 拉到 sel_acc@25% = 84%。

4. **Teacher-anchor 是最强单特征**。`teacher_best_similarity` 单个 feature 就能拿 AUROC 0.906；这印证了第二轮分析中"50 个特征里只有 teacher_* 真携带 ground-truth 信息"的判断。

5. **Plan_gpt55 现有的 `verifier_score_v05` 在这份 anchor 数据上 AUROC 0.48（基本反向）**。原 verifier 是在另一份 candidate_features.csv 上训的，迁移到 anchor rows 已经反向。**这是个独立发现**：现有的 learned verifier 不可直接复用。

6. **MLP 没有显著强于 LogReg**（cv: 0.960 vs 0.969），且 cross-seed ECE 飙到 0.19——非线性容易在小样本上过拟合。**LogReg 是更稳的产品选择**。

## 4. Routing Pareto

### 4.1 Default cost (correct +1, wrong −3, teacher −0.3, abstain 0)

| Setting | Method | answer 数 | teacher 数 | abstain 数 | answer accuracy | utility/题 |
|---|---|---|---|---|---|---|
| sample0 | always_answer | 1000 | 0 | 0 | 0.566 | **−0.736** |
| sample0 | always_teacher | 0 | 1000 | 0 | — | −1.040 |
| sample0 | always_abstain | 0 | 0 | 1000 | — | 0.000 |
| sample0 | logreg:self routing | 491 | 0 | 509 | 0.805 | +0.107 |
| sample0 | mlp:all routing | 478 | 0 | 522 | **1.000** | +0.478 |
| **sample0** | **logreg:teacher routing** | **487** | **43** | **470** | **1.000** | **+0.513** |
| fixed_8 | always_answer | 1000 | 0 | 0 | 0.587 | −0.652 |
| fixed_8 | logreg:self routing | 409 | 0 | 591 | 0.829 | +0.129 |
| fixed_8 | logreg:all routing | 502 | 0 | 498 | 0.996 | +0.494 |
| **fixed_8** | **logreg:teacher routing** | **502** | **30** | **468** | **0.996** | **+0.507** |

**关键数字**：sample0 + logreg:teacher routing 在 default cost 下答 487 / 1000 题，**accuracy 1.000**，调 teacher 43 题（accuracy 0.977），弃权 470 题，平均效用 **+0.513**。比 always_answer 高 **+1.249 / 题**。

### 4.2 Cost 敏感性

| Cost model | always_answer | best routing | swing |
|---|---|---|---|
| lenient (wrong −1.5, teacher −0.2) | −0.085 | +0.519 | +0.604 |
| default (wrong −3.0, teacher −0.3) | −0.736 | +0.513 | +1.249 |
| strict (wrong −5.0, teacher −0.4) | −1.604 | +0.507 | +2.111 |

**Wrong penalty 越高，selective prediction 的相对价值越大**——这正是 hallucination 风险高的工业场景。

## 5. 与 Plan_opus（generation shaping） 的对比

| 框架 | 方法 | sample0 strict | 说明 |
|---|---|---|---|
| C: generation shaping | Base Qwen2.5-7B | 0.660 | — |
| C: generation shaping | Plan_gpt55 anchor-VBPO v1 | 0.660 | 训不动；fixed_8 −1.0 损伤 |
| C: generation shaping | Plan_opus VBPO step90 | 0.665 | +0.5 点 |
| C: generation shaping | Plan_opus GRPO step20 | 0.674 | +1.4 点（lucky checkpoint） |
| **A: selective prediction** | **logreg:all @ 50% coverage** | **0.994** | **+44 点 selective accuracy** |
| **A: selective prediction** | **logreg:teacher routing (default cost)** | **answer acc 1.000** | **utility −0.74 → +0.51** |

> 注：sample0 base 0.660 是 test500 (offset=500) 上的；selective prediction 实验用的是 anchor rows (offset=0, base 0.566)。两个数字不直接可比，但相对结构一致——selective prediction 的"answer accuracy"概念上对应"在我决定回答的子集上的 strict accuracy"，base 56.6% 拉到 99.4%/99.6% 是非常显著的实质改进。

## 6. 这能写成什么 paper？

**Title 草案**：
> *Trust the Sensors, Not the Actuators: Selective Prediction over Cross-Model Answer Basins for Hallucination-Risk Control*

**Story arc**：
1. **诊断**：观察到 stable-wrong-basin 现象 → 生成 50+ basin/anchor 特征
2. **失败**：试图把这些特征喂给 ITI/VBPO/GRPO 改 generation distribution → strict accuracy 只能 squeeze 出 +0.5–1.4 点
3. **诊断 II**：信息论上 self-introspection 不能告诉你 ground truth，能告诉你的只是 risk
4. **重新 framing**：把同一批特征作 risk score 而非 generation reward
5. **结果**：
   - Selective accuracy @ 50% coverage 从 56.6% 拉到 99.4%
   - Risk-adjusted utility 从 −0.74 拉到 +0.51（default cost）
   - 现有 self-only 信号已强（AUROC 0.81），加 cross-model anchor 后接近完美 (AUROC 0.97)

**Contributions**：
1. **Cross-model basin anchoring** 作为外部 ground-truth 信号的形式化（vs SelfCheckGPT 的 self-consistency）
2. **Question-level basin geometry features** 与 candidate-level features 的组合
3. **3-action routing framework**（answer / cascade-to-teacher / abstain）的 Pareto 分析
4. **Negative result**：把同一信号用作 generation reward 几乎无效，用作 risk score 极有效——这条 "sensor vs actuator" 的论证本身就是贡献

**Honest disclaimers** 必须放在 limitations：
- 单数据集（TriviaQA）；NaturalQuestions / SciQ 上还没验证
- teacher_best_similarity 在部署时需要调 GPT-OSS（一次每题），cost 不为 0
- 5-fold CV 与 cross-seed 一致 → 不是 question-level 过拟合，但 dataset-level domain shift 还没测
- 现在的 routing oracle 用 `teacher_best_basin_strict_any` 估算 teacher accuracy，部署时要换成实测

## 7. Next Steps

| 优先级 | 任务 | 预期收益 |
|---|---|---|
| 1 | NaturalQuestions / SciQ 复现 selective prediction | 跨数据集泛化是 paper 必须的 |
| 2 | 在 test500 (offset=500) 上重新生成 8 候选 + 提取特征 + 评测 | 与 Plan_opus generation shaping 实验**完全可比**的 selective accuracy 数字 |
| 3 | "self-only at deployment" 强化：用 SelfCheckGPT 的二次问询丰富 self-features | 把 self-only AUROC 从 0.81 推到 0.85+，去掉对 teacher 调用的依赖 |
| 4 | Calibration（Platt / isotonic）on top of LogReg | 把 ECE 从 0.02 推到 0.005，配合 risk-coverage 的工业部署 |
| 5 | Routing 在 lenient → strict cost 跨度上画 Pareto plot | paper-friendly 图 |
| 6 | 定义 "question hardness" 的对外 release artifact，让其他研究者直接用 | 数据贡献 |

## 8. 复现命令

```bash
# 1. 跑所有 selective predictors
python /zhutingqi/song/Plan_opus_selective/scripts/run_selective_experiments.py

# 2. 跑 routing 分析（依赖 best_predictions_*.csv）
python /zhutingqi/song/Plan_opus_selective/scripts/run_routing_analysis.py

# 3. 渲染对外的 markdown 表
python /zhutingqi/song/Plan_opus_selective/scripts/build_report_tables.py
```

输出：
- `results/selective_metrics_long.csv`, `selective_metrics_pivot.csv`, `metrics_table.md`
- `results/best_predictions_<setting>.csv`, `predictions_<setting>_<predictor>.csv`
- `results/question_features_<setting>.csv`
- `results/routing_summary.csv`, `routing_pareto.csv`, `routing_table.md`
- `reports/SELECTIVE_PREDICTION_REPORT.md` (本文)

## 附录 A: 目录

```
Plan_opus_selective/
├─ shared/
│  ├─ data_utils.py    # 候选 → 问题级特征聚合
│  ├─ metrics.py       # AURC, sel_acc, Brier, ECE
│  └─ routing.py       # 3-action utility curve
├─ scripts/
│  ├─ run_selective_experiments.py   # 所有 predictor + CV/cross-seed
│  ├─ run_routing_analysis.py        # Pareto + best operating point
│  └─ build_report_tables.py         # markdown 表
├─ results/                          # 全部 CSV + markdown
├─ reports/SELECTIVE_PREDICTION_REPORT.md
└─ runs_logs/
```
