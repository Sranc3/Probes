# 新实验具体协议

## 通用报告规范

每个新实验必须报告：

- split 和 question IDs；
- 是否包含 train；
- seeds；
- semantic sample count；
- max_new_tokens；
- candidate / gate / alpha；
- answer change rate；
- correctness transition table；
- exact / contains / NLI semantic correct 分层；
- token entropy mean/max；
- semantic entropy weighted/uniform；
- event density；
- latency 是否 cache-aware。

## 通用通过标准

一个 candidate 只有同时满足以下条件，才可称为候选方案：

| Criterion | Required |
| --- | ---: |
| correctness non-drop rate | `1.0` |
| exact non-drop rate | `1.0` if exact has support |
| answer change rate | `>= 0.10` |
| semantic band success rate | `>= 0.80` |
| token mean nonpositive rate | `>= 0.67` |
| token max nonpositive rate | `>= 0.50` |
| safety success rate | `>= 0.67` |

如果只满足部分指标，只能称为 diagnostic signal，不称为 deployable solution。

## Experiment 1A: Pure Held-out Fixed ITI Recheck

### 假设

如果固定 ITI 仍有部署价值，那么在纯 test split 上应能产生非微弱行为改变，同时不降低 correctness。

### 数据

- split: `test`
- expected questions: 31
- seeds: `[42, 43, 52, 53, 54]`

### Candidates

1. `baseline`
2. `cand008_primary`
   - `target_mode=increase_semantic_high`
   - `alpha=0.005`
   - sites: `mlp.output|layer_18,24`
   - gate: `0.67`
3. `ind0067_safety_pareto`
   - `target_mode=reduce_semantic_high`
   - `alpha=0.007029674339123061`
   - layer weights from `Intervention_D`
4. `layer18_only`
5. `layer24_only`

### Metrics

- answer change rate；
- correctness transition；
- token entropy deltas；
- semantic entropy deltas；
- event density；
- per-site event count；
- seed stability。

### 判断

- 如果 answer change rate < 10%，说明固定 ITI 过弱；
- 如果 correctness drop > 0，说明固定 ITI 不安全；
- 如果 token 指标改善但 semantic/correctness 不稳，说明需要 learned/adaptive controller。

## Experiment 1B: Gate Ablation

### 假设

固定方向失败可能不是方向错，而是 gate 太粗。

### Gate Matrix

| Gate | Description |
| --- | --- |
| `always` | 每步干预 |
| `prev_entropy_q50` | previous entropy >= q50 |
| `prev_entropy_q67` | previous entropy >= q67 |
| `prev_entropy_q80` | previous entropy >= q80 |
| `trend_up` | entropy 比前两步上升 |
| `position_late` | 只在生成超过若干 token 后干预 |

### Candidates

优先只测两个方向：

- `cand008_primary`
- `ind0067_safety_pareto`

### 判断

如果某些 gate 显著提升 answer change rate 且不伤 correctness，说明 controller 路线值得推进。

## Experiment 1C: Alpha/Sign Stress Test

### 假设

如果 direction 是有效控制旋钮，alpha/sign 应存在可解释趋势。

### Grid

- alpha: `[0.001, 0.0025, 0.005, 0.0075, 0.010, 0.015]`
- sign:
  - `increase_semantic_high`
  - `reduce_semantic_high`
- sites:
  - layer 18
  - layer 24
  - layer 18 + 24

### 判断

- 有平滑趋势：继续 adaptive alpha；
- 无趋势但有局部 spike：可能是噪声或强上下文依赖；
- 大 alpha 伤 correctness：固定 ITI 安全边界窄。

## Experiment 2A: Decoding/Reranking Baseline

### 假设

不改 hidden state 的方法可能更稳地利用 uncertainty。

### 方法

1. Generate N candidate answers。
2. 用 semantic clusters 聚类。
3. 优先选择：
   - 与多数 cluster 一致；
   - token entropy 较低；
   - answer length 不异常；
   - NLI 与 question+ideal proxy 更一致的答案。

### 对照

- baseline single sample；
- baseline best-of-N by logprob；
- semantic cluster majority；
- entropy-aware reranking。

### 判断

如果 reranking 改善 correctness 或稳定性，优先走 Track C。

## Experiment 2B: Learned Controller Prototype

### 假设

固定 ITI 的失败来自 action 不随状态变化，而不是 direction 完全无用。

### 最小实现

第一版 controller 不直接接入生成，只做 offline classification：

- 输入：每个 question/seed/candidate 的 summary features；
- 标签：该 candidate 是否满足 safety success；
- 模型：logistic regression / small MLP；
- 评估：train/val/test split。

### 后续在线版本

如果 offline controller 有预测力，再做 online step-level controller。

## Experiment 3: Distillation/SFT Feasibility

### 进入条件

只有当 Experiment 2A 或 2B 至少一个成功，才进入训练式路线。

### 数据构造

- positive examples：correctness 不掉且 hesitation 更低的 outputs；
- negative examples：correctness drop 或 semantic collapse 的 outputs；
- teacher answers：强模型或 best-of-N/reranked answer；
- weights：由 entropy 和 correctness confidence 决定。

### 训练目标

- SFT loss；
- preference loss；
- confidence-aware weighting；
- 可选 token entropy regularization。

## 推荐执行顺序

1. Experiment 1A
2. Experiment 1B
3. Experiment 1C
4. Experiment 2A
5. Experiment 2B
6. Experiment 3

不要跳过 1A。它会先修正旧部署评估中 `all_rows` 混入 train 的解释风险。
