# Qwen2.5 候选空间普遍性验证：all-200 TriviaQA

## 目的

这轮实验不是换模型，也不是换数据集，而是在当前 Qwen2.5 和已有 TriviaQA artifact 上，把候选空间分析从 pilot 规模扩大到全 200 题。

核心问题是：

> 之前图里看到的 rescue candidate 结构，是否只是 31 个 test 问题上的偶然现象？

## 实验设置

- 模型：Qwen2.5，保持不变。
- 数据：`run_20260403_050400` 中的 200 个 TriviaQA 问题。
- 采样：每个 question-seed pair 生成 8 个候选答案。
- Seeds：`42, 43`。
- 总规模：400 个 question-seed pair，3200 个候选答案点。
- 正确性指标：strict correctness，即 exact match 或 contains match。

输出目录：

`/zhutingqi/song/Plan_gpt55/candidate_space_analysis/runs/run_20260427_102843_candidate_space_all200_qwen25`

## 主要结果

all-200 结果显示，候选空间里确实存在可救空间：

- 3200 个候选答案点中，1584 个 strict-correct，比例为 49.50%。
- 400 个 question-seed pair 中，200 个 sample0 本来正确。
- 34 个 question-seed pair 中，sample0 错，但后续 7 个候选里存在严格正确答案。
- 这对应 93 个 rescue candidate。
- 同时也存在 109 个 damage candidate，即 sample0 对，但某些候选会把答案换错。

这说明候选空间不是随机噪声。模型第一次回答错时，后续采样中确实有一部分答案能够把问题救回来。

## 对 pilot 图的修正

之前 31 题 pilot 图给人的强烈直觉是：

> rescue candidate 往往出现在高置信、低犹豫区域。

放大到 200 题后，这个说法需要收敛得更严谨：

> rescue candidate 不是随机分布的，但不能简单概括为“高置信、低犹豫就是 rescue”。高置信/低犹豫是有用信号之一，但不是稳定充分条件。

原因是 all-200 中，rescue 与 logprob/entropy 的相对分离度明显变弱：

- `rescue_vs_nonrescue` 的 `logprob_avg_z` Cohen's d 只有 0.087。
- `rescue_vs_nonrescue` 的 `token_mean_entropy_z` Cohen's d 约为 -0.008。
- 也就是说，仅靠“相对更高 logprob / 相对更低 entropy”已经不能稳健地区分 rescue。

但是，候选空间结构仍然存在。更稳定的强信号出现在两类地方：

- 正确答案整体更容易出现在更大的语义簇里，`correct_vs_wrong` 的 `cluster_size` Cohen's d 为 1.176，AUC 为 0.772。
- damage candidate 更危险且更可识别，通常更低 logprob、更高 entropy、更小 cluster、更长答案。例如 `damage_vs_nondamage` 的 `logprob_avg` Cohen's d 为 -1.333，`token_mean_entropy` Cohen's d 为 1.107。

## 对论文论点的影响

这轮结果让论文论点变得更稳，而不是更弱。

不建议把论文主张写成：

> 正确 rescue 答案通常位于高置信、低熵区域。

这个说法太强，容易被更大样本推翻。

更建议写成：

> Multiple sampled answers form a structured candidate space rather than random variation. Correctness, rescue potential, and damage risk are partially encoded in cheap output-level signals such as semantic cluster structure, token entropy, log probability, and answer length. However, no single signal is sufficient; reliable control requires a verifier/controller over this candidate space.

中文解释：

> 多次采样得到的答案不是一团随机噪声，而是形成了一个有结构的候选空间。正确性、可救性和损坏风险都部分体现在语义簇、token entropy、logprob、答案长度等低成本输出特征里。但任何单一特征都不够可靠，因此后续方法应该学习一个轻量 verifier/controller，而不是手写一个“最高置信就选它”的规则。

## 对后续方法的启发

all-200 结果给出三个直接启发：

1. 不应继续押注固定 hidden-state ITI。
   之前 Phase 1 的结果已经说明固定干预不稳；现在输出空间结果说明，模型内部确实有潜在正确答案，但它以候选分布的形式显现，而不是容易被一根固定方向推出。

2. 不应只做 naive best-of-N。
   3200 点中有 34 个 pair 可救，但也有 109 个 damage candidate。盲目切换答案会同时制造伤害。

3. 应该训练轻量 controller。
   输入可以是每个候选的 logprob、entropy、cluster size、cluster weight、length，以及候选间相对 rank；目标是判断什么时候保留 sample0，什么时候触发额外采样，什么时候切换到另一个候选。

## 当前简单规则的上限

手写几何分数能带来小幅提升，但不够强：

- `confidence_minus_entropy`: strict correct 50.25%，相对 sample0 提升 0.25%。
- `low_entropy_cluster`: strict correct 51.00%，相对 sample0 提升 1.00%。
- `cluster_confidence_entropy`: strict correct 52.25%，相对 sample0 提升 2.25%。

这说明多维组合方向是对的，但线性手写规则不足以成为最终方法。下一步需要训练和验证一个真正的轻量 verifier/controller，最好使用 leave-question-out 或 cross-validation，避免只记住题目。

## 推荐论文表述

这张图仍然值得作为论文关键图，但图注和正文要谨慎：

> Pilot visualization suggests that rescue answers can occupy distinctive regions in confidence-hesitation space. A larger all-200 validation shows that this structure persists at the candidate-space level, but the reliable signal is multi-dimensional rather than reducible to confidence or entropy alone. This motivates candidate-space control: learning when to trust the first answer, when to sample further, and when to switch to an alternative candidate.

一句话版本：

> 候选空间有结构，但结构不是单轴的；这正是 controller 方法的必要性。
