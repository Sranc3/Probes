# 候选答案空间分析初步结论

## 为什么要做这个

Phase 2A 说明一个关键事实：

> 模型第一次回答错的时候，正确答案有时已经出现在后续候选里。

这意味着问题不一定是“模型不会”，而可能是“模型不会稳定地选中自己已经生成出来的好答案”。

因此，我们把每个候选答案当成一个多维空间里的点，用以下坐标描述它：

- 平均 log probability；
- token mean/max entropy；
- 生成长度；
- 所属语义簇大小；
- 所属语义簇权重；
- 在同一问题候选集内的相对排名；
- strict correctness 标签。

这不是最终的“黎曼几何模型”，但可以理解成一个局部坐标图：先看这个空间里有没有可学习结构。

## 当前数据规模

来源：`/zhutingqi/song/Plan_gpt55/runs/run_20260427_074132_phase2a_reranking`

- 93 个 question-seed 组合；
- 每个组合 8 个候选；
- 共 744 个候选点；
- strict-correct 候选：321 个；
- rescue 候选：22 个；
- damage 候选：13 个；
- sample0 错但候选集中存在正确答案的问题：12 个。

## 最重要发现

### 1. rescue 候选确实有结构

rescue 候选不是随机散落的。它们常见特征是：

- 在同题候选内部，logprob 相对更高；
- token entropy 相对更低；
- 但不一定属于最大语义簇；
- 很多 rescue 答案反而在小簇里。

这说明简单的“多数派语义簇”不够。正确答案有时是少数派，但它在局部特征上更像一个高质量点。

### 2. damage 候选也有明显风险特征

damage 候选常见特征是：

- 生成更长；
- token entropy 相对更高；
- logprob 相对更差；
- 不一定能被“低熵”单独排除。

这解释了为什么只看 logprob 或 entropy 不稳：有些错误答案也非常自信。

### 3. 简单组合分数已经接近 best-of-8

当前手写几何分数：

| Formula | Strict Correct | Δ vs sample0 | Improved | Damaged |
| --- | ---: | ---: | ---: | ---: |
| `confidence_minus_entropy` | 47.31% | +5.38% | 6 | 1 |
| `cluster_confidence_entropy` | 48.39% | +6.45% | 7 | 1 |
| `low_entropy_cluster` | 48.39% | +6.45% | 7 | 1 |

这已经接近 Phase 2A 的 `best_of_n_logprob` 结果 49.46%，但仍然不是最终方案，因为它仍使用了 8 个候选。

## 对困境的意义

ITI 的问题是：它在 hidden state 上强行推，结果多数是 wrong-to-wrong。

候选空间分析显示另一条路：

> 正确答案有时已经在候选空间里，关键是学会低成本地识别它。

这比继续固定 ITI 搜索更有希望，因为目标变清楚了：

- 不再问“怎么推模型内部向量”；
- 而是问“什么特征说明这个候选值得被选中”；
- 进一步问“能不能在生成早期判断是否需要额外采样”。

## 下一步建议

1. 把当前特征表作为训练数据，做 leave-question-out 的轻量 verifier。
2. 预测两个任务：
   - candidate 是否 strict-correct；
   - sample0 错时，是否存在 rescue candidate。
3. 如果离线 verifier 能超过简单规则，再做低成本策略：
   - 默认生成一次；
   - 只有 verifier 判断 sample0 高风险时，触发第 2 或第 3 个候选；
   - 只在局部候选空间里做选择。

## 关于“黎曼/流形”的位置

现在还不应该直接声称黎曼几何。

更准确的说法是：

> 我们先构建 answer-candidate space 的局部经验坐标，并分析正确候选与错误候选的几何分布。如果这些结构稳定，再考虑用 learned metric、local curvature 或 manifold-aware controller 描述。

这样既保留了酷炫方向，也不会显得玄。
