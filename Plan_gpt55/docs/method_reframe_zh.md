# 方法重构与下一阶段路线图

## 新叙事

当前项目不应叙事为：

> 我们提出 ITI 干预，并成功提升 QA。

这个叙事和现有证据不匹配。

更稳健的叙事是：

> 我们发现 Qwen2.5 内部存在与 semantic/token uncertainty 相关的层表征；这些表征可以被固定方向干预扰动，但直接使用 fixed-vector inference-time steering 作为部署控制器会表现出明显脆弱性。因此，下一步应从手写 ITI 转向 learned uncertainty-aware control。

这个叙事保留了项目价值，同时不夸大当前负结果。

## 证据支撑

### 方向资产有机制信号

`direction_asset_audit.md` 显示：

| Site | Pos/Neg | Projection separation | Cohen d |
| --- | ---: | ---: | ---: |
| `mlp.output|layer_18` | `46/93` | `4.7022` | `1.9191` |
| `mlp.output|layer_24` | `46/93` | `23.9535` | `1.4620` |

这说明 high/low semantic entropy 分组在这些方向上确实有明显 separation。

因此，不应该说“熵和层表征没有关系”。更准确的是：

> 方向作为 diagnostic probe 有价值。

### 但 fixed ITI 没有形成稳健部署解

`artifact_audit_summary.md` 显示 `Intervention_D` 的最终候选是 fallback representative：

- `semantic_band_success_rate=0.667 < 0.670`
- `token_mean_nonpositive_rate=0.667 < 0.670`
- `token_max_nonpositive_rate=0.333 < 0.500`
- `safety_success_rate=0.000 < 0.670`
- `stability_penalty=6.541 > 4.500`

这说明当前 Pareto 搜索没有找到 robust solution。

### 部署评估扰动过弱

`cand008_deploy_eval_v1` 的部署式评估显示：

- 400 question pairs；
- answer changed only `13`；
- answer change rate `3.25%`；
- correctness delta `0`；
- exact delta `0`；
- entropy/latency 变化全部处于 noise scale。

这说明固定 ITI 目前并没有实质改变最终回答行为。

## 需要停止的路线

### 1. 停止把 cand_008 当部署方案

`cand_008` 可以作为分析对象，但不应作为最终方案。

原因：

- 输出改变率低；
- correctness 没收益；
- token/semantic/latency 效应小于方差；
- 不满足 safety-first Pareto gate。

### 2. 停止只在固定 alpha 附近微调

继续在 `0.005 ~ 0.008` 附近微调很可能只是局部噪声优化。

除非目标是写 failure analysis，否则不建议继续把算力投入小范围固定向量搜索。

### 3. 停止把 elapsed_ms 当强部署证据

当前生成链路使用非 cache-aware 解码，`elapsed_ms` 只能作为辅助分析指标。

如果要主张加速，需要重新实现或单独评估 cache-aware decoding。

## 应保留的路线

### Track A: Diagnostic-Only ITI

目标：证明哪些层和哪些状态携带 uncertainty signal。

做法：

- 更系统地审计 direction separation；
- 引入 outlier dominance 检查；
- 检查不同 split 上 direction projection 是否一致；
- 检查 event count 与 outcome delta 的相关性。

产出：

- 机制图；
- 层级证据；
- 不把 ITI claim 成部署方法。

### Track B: Learned Controller

目标：把固定 alpha/gate 改成可学习策略。

输入特征可以包括：

- previous token entropy；
- entropy trend；
- layer projection score；
- token position；
- generated length；
- per-site hidden norm；
- question-level uncertainty estimate。

输出可以是：

- 是否介入；
- 介入哪个 site；
- alpha 大小；
- 目标 polarity。

训练目标：

- correctness non-drop；
- semantic breadth within band；
- token hesitation reduction；
- latency/token-count regularization。

### Track C: Decoding / Reranking

目标：不改 hidden state，而是在生成策略层利用 uncertainty。

可行方向：

- entropy-aware temperature schedule；
- high-uncertainty token reranking；
- multi-sample answer reranking；
- uncertainty-triggered shorter/longer decoding；
- semantic cluster-aware selection。

优势：

- 比 hidden-state steering 更可控；
- 更容易做 ablation；
- correctness 风险更低。

### Track D: Distillation / SFT

目标：把“少犹豫、不塌缩、正确率不掉”的行为训练进模型。

蒸馏不是学习 teacher 的熵分布，而是学习：

- teacher outputs；
- teacher token distributions；
- teacher preferences；
- teacher reasoning/answer style。

熵信号在这里作为辅助：

- sample weighting；
- curriculum；
- confidence-aware loss；
- reject/rerank teacher outputs；
- 标注哪些样本需要更果断。

## 下一阶段实验顺序

### Phase 1: Post-hoc audit

不重新跑大模型，先从已有 artifacts 中回答：

- direction separation 是否稳定；
- deploy selection 是否混入 train；
- event density 和指标变化是否相关；
- correctness evaluator 是否提供有效区分。

当前 `Plan_gpt55` 已经开始提供这类脚本：

- `scripts/audit_existing_artifacts.py`
- `scripts/audit_direction_assets.py`

### Phase 2: Small sanity experiments

只做小规模实验验证新的控制思路：

- 同一问题固定多 seed；
- cache-aware 和 non-cache-aware latency 分开；
- semantic samples 提高到至少 8；
- correctness 使用 exact/contains/NLI/judge 分层报告；
- 对比 no-intervention、fixed ITI、adaptive gate、reranking。

### Phase 3: Learned controller prototype

如果 Phase 2 支持 uncertainty signals 有预测力，再训练轻量 controller。

优先使用冻结 base model 的外部 controller，不直接微调 Qwen。

### Phase 4: Distillation or SFT

如果 controller 有效果，再考虑训练式路线：

- teacher-generated SFT；
- preference distillation；
- confidence-aware distillation；
- uncertainty-regularized decoding behavior。

## 建议论文定位

如果短期要写成 paper，建议定位为：

> mechanistic diagnosis + negative result + principled transition

而不是：

> direct ITI improves QA

更稳的贡献点：

1. 发现 entropy-related internal directions；
2. 系统评估 fixed-vector ITI 的脆弱性；
3. 提出 safety-first evaluation protocol；
4. 证明 naive uncertainty steering 很难稳定转化为 correctness-preserving deployment gains；
5. 给出 learned uncertainty-aware control 的路线。

## 最终建议

不要放弃“熵相关机制”这个核心 idea。

但应该放弃把 `cand_008` 或固定 ITI 当最终主线。

下一步最值得做的是：

1. 继续把 ITI 当探针；
2. 建立更可信的评估；
3. 训练或设计 adaptive controller；
4. 只在 controller/reranking/distillation 出现稳定收益后，再恢复大规模搜索。
