# ITI-for-Entropy 诊断报告

## 结论摘要

当前项目不应被简单判定为“idea 错了”。

更准确的判断是：

1. **熵相关表征存在**：已有结果足以支持“某些层的激活和 semantic/token uncertainty 存在关系”。
2. **固定 ITI 路线过于苛刻**：把 question-level 的熵相关方向直接作为 token-level 固定控制向量，机制跳跃太大。
3. **实现和评估还比较粗**：搜索空间、方向构造、gate、correctness、semantic entropy 采样、latency 评估都有噪声源。

因此，当前证据更支持这个表述：

> Entropy-related representations are discoverable and partially steerable, but fixed-vector inference-time intervention is brittle as a deployment method.

## 证据链

### 1. 方向构造与实际干预之间存在层级错配

方向资产来自 high/low semantic entropy 的 question-level 分组。核心逻辑在：

`/zhutingqi/song/ITI_for_entropy/Intervention_A/intervention_utils.py`

```python
is_positive = float(row["semantic_entropy_weighted"]) >= semantic_threshold
pooled = entropy_weighted_pool(vectors, token_entropies)
raw_direction = positive_mean - negative_mean
unit_direction = raw_direction / raw_norm
```

这说明方向是“高 semantic entropy 问题的 pooled activation 均值”减去“低 semantic entropy 问题的 pooled activation 均值”。

但推理时应用方式是：

```python
modified[0, -1] = modified[0, -1] + delta
```

也就是在每个生成 step 的最后一个 token hidden state 上做局部平移。

这不是代码 bug，而是方法假设很强：

- question-level 统计差异必须是 causal direction；
- pooled sequence representation 的差异必须能迁移到 token-level；
- 同一个方向必须跨题目、跨 seed、跨生成阶段都适用；
- 同一 alpha 必须同时满足 correctness、breadth、hesitation、latency。

这个假设链条过长，是 ITI 路线不稳定的主要机制原因。

### 2. `Intervention_D` 的最终候选不是 robust success

`Intervention_D` 的 safety-first Pareto 搜索已经更接近真实目标：

- correctness 不掉；
- exact match 不掉；
- semantic breadth 不塌；
- token hesitation 降低；
- latency 降低。

但最终产物显示：

- final Pareto 只有 `ind_0067`；
- `robust_success_rate = 0.000`；
- `safety_success_rate = 0.000`；
- `stability_penalty = 6.541403`；
- held-out gate 未通过，只是 fallback Pareto ranking 后留下。

关键指标：

| Metric | Value |
| --- | ---: |
| `delta_final_semantic_correct_rate_mean` | `0.000000` |
| `delta_final_exact_match_rate_mean` | `0.000000` |
| `delta_semantic_entropy_weighted_mean` | `0.025961` |
| `delta_token_mean_entropy_mean` | `-0.006889` |
| `delta_token_max_entropy_mean` | `-0.037773` |
| `delta_elapsed_ms_mean` | `-5.527778` |
| `token_max_nonpositive_rate` | `0.333` |
| `safety_success_rate` | `0.000` |

解释：

- 平均上有一点 token hesitation 和 latency 改善；
- correctness 没掉；
- 但 seed 间不稳定，尤其 `token_max` 只有三分之一 seed 变好；
- 因此不能写成可部署成功。

### 3. 部署评估显示效果接近噪声边界

`cand008_deploy_eval_v1` 的主结果：

| Metric | Value |
| --- | ---: |
| question pairs | `400` |
| `delta_semantic_entropy_weighted_mean` | `0.002383` |
| `delta_semantic_entropy_weighted_std` | `0.070772` |
| `delta_token_mean_entropy_mean` | `-0.001044` |
| `delta_token_mean_entropy_std` | `0.025901` |
| `delta_token_max_entropy_mean` | `0.006795` |
| `delta_token_max_entropy_std` | `0.119411` |
| `delta_elapsed_ms_mean` | `1.947500` |
| `delta_elapsed_ms_std` | `124.517393` |
| `delta_final_semantic_correct_rate` | `0.000000` |
| `answer_changed_count` | `13 / 400` |

这说明 cand_008 的实际输出扰动很有限：

- 只有 13/400 个 paired outputs 改变；
- correctness 没变；
- token/semantic/latency 的均值远小于标准差；
- 不能证明部署收益。

这并不等价于“机制不存在”，而是说明当前控制策略太弱或太不稳定。

## 实现和评估中的粗糙点

### A. 搜索空间太窄

`Intervention_D` v1 主要围绕：

- `mlp.output|layer_18`
- `mlp.output|layer_24`
- `gate_entropy_quantile = 0.67`
- `alpha = 0.0048 ~ 0.0078`

这只能验证 `cand_008` 邻域，不足以覆盖全局控制策略。

### B. gate 太单一

当前 gate 基本是 previous token entropy 是否超过训练集 quantile threshold。

这个 gate 缺少：

- token position awareness；
- answer stage awareness；
- uncertainty trend；
- semantic cluster state；
- question difficulty calibration；
- per-site adaptive threshold。

### C. correctness evaluator 偏弱

当前 correctness 主要依赖：

- exact match；
- contains match；
- DeBERTa NLI semantic match。

问题：

- exact match 在 TriviaQA 上常常过于稀疏；
- contains match 容易过宽；
- NLI 对短答案和问答拼接形式可能不稳定；
- 没有独立 judge 或 dataset-native normalization。

### D. semantic entropy 方差高

semantic entropy 来自少量 sampled generations 和 NLI clustering。

如果每题只有 2-4 个 samples，那么 entropy estimate 方差很大。部署结果中 `delta_semantic_entropy_weighted_std` 远大于均值，说明许多结论处在噪声边界。

### E. latency 不是真部署指标

当前生成使用 `use_cache=False`，每步重算完整 prefix，方便 hook 和 trace，但不是部署式解码。

因此 `elapsed_ms` 混入了：

- 重算成本；
- hook 开销；
- GPU 状态波动；
- token 数变化；
- baseline/intervention 顺序效应。

不能直接作为“部署加速”证据。

### F. fallback 机制会造成解释风险

EA 在没有候选通过 held-out robustness gate 时，会 fallback 到 held-out Pareto ranking。

这适合保留调试信息，但报告时必须标注：

> final candidate is a fallback Pareto representative, not a robustly accepted solution.

## 哪些该保留

### validated

- 熵和部分层表征之间有可观察关系。
- ITI 可以改变 token/semantic entropy 的局部统计。
- `mlp.output|layer_18/24` 是值得继续作为 probe 的区域。

### weak_or_inconclusive

- 固定方向是否能稳定改善 QA。
- 当前 semantic entropy 指标是否足够精确地代表 exploration breadth。
- latency 是否真的可通过当前干预方式改善。

### likely_wrong_path

- 继续只在 `cand_008` 附近小范围调 alpha。
- 把 fallback Pareto candidate 写成成功。
- 继续以 semantic entropy 单调下降作为目标。
- 把固定 ITI 当最终部署方法。

## 诊断结论

当前项目的核心价值不是“找到了一个可部署 ITI 解”，而是：

1. 找到了 entropy-related representation；
2. 证明该 representation 可被干预扰动；
3. 发现 fixed-vector inference-time steering 不够稳；
4. 为 learned control / decoding policy / distillation 提供了机制线索。

所以建议把研究叙事改为：

> We identify entropy-related internal representations and show that direct fixed-vector intervention is brittle, motivating learned uncertainty-aware control rather than hand-coded inference-time steering.
