# Direction 与 Intervention Event 审计设计

## 审计目的

当前 ITI 方向来自 high/low semantic entropy 的均值差，但最终应用在 token-level hidden state 上。这个设计可能有效，也可能只是捕捉了 question difficulty、answer style、长度或 sampling noise。

因此需要审计三类问题：

1. 方向资产是否稳定；
2. gate 是否合理；
3. intervention event 是否真的对应有意义的 hesitation / uncertainty。

## 审计对象

- Direction assets：
  - `raw_direction`
  - `unit_direction`
  - `raw_norm`
  - `median_step_norm`
  - positive/negative counts
- Train split pooled activations：
  - 每个 train question 的 entropy-weighted pooled activation；
  - pooled activation 在 direction 上的 projection；
  - high/low semantic group 的 projection separation。
- Gate threshold：
  - `gate_entropy_quantile`
  - `gate_entropy_threshold`
  - train token entropy distribution。

## 关键检查

### 1. Positive / negative imbalance

如果 high semantic entropy 样本太少，方向可能被少数样本主导。

需要检查：

- positive count；
- negative count；
- positive ratio；
- split size。

### 2. Projection separation

对每个 site，计算每个 train row pooled activation 在 `unit_direction` 上的投影。

理想情况：

- positive group projection 均值显著高于 negative group；
- 两组方差不过大；
- separation 不只由一两个 outlier 贡献。

如果 separation 很弱，那么 ITI 方向更像噪声方向。

### 3. Outlier dominance

检查 projection 的极端值和 top absolute projections。

如果方向主要由少量问题支撑，应降低对固定 ITI 的信心。

### 4. Gate interpretability

当前 gate 是 previous token entropy 超过训练分位数。需要检查：

- gate threshold 是否过低，导致大多数 step 都触发；
- 是否不同 layer/site 使用同一个 gate；
- gate 是否与 actual intervention effect 相关。

### 5. Event-to-outcome relation

后续如果保留 step traces，应统计：

- event count vs token entropy delta；
- event count vs correctness transition；
- event count vs latency；
- per-site event density。

如果 event count 和 outcome 无关，说明 gate 并没有捕捉到有效干预时机。

## 推荐解释规则

- 如果 direction separation 强，但 event-to-outcome 弱：说明机制信号存在，但控制策略差。
- 如果 direction separation 弱：说明当前方向构造本身不可靠。
- 如果 separation 强且 event-to-outcome 强：才值得继续推进 ITI 或 learned controller。
- 如果 gate 触发过多：应转向 adaptive gate。
- 如果 gate 触发过少：应重新定义 uncertainty trigger。

## 当前建议

短期内不要继续把 fixed direction 当主线。更合理的定位是：

- 使用 direction audit 确认哪些层有稳定 uncertainty representation；
- 使用 event audit 找出哪些 token states 适合介入；
- 再训练 controller 或 decoding policy，而不是手写固定 alpha。
