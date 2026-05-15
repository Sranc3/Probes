# Controller Implications: Factual + Riemann

建议下一版 controller 不再只做 absolute basin scoring，而是使用 pairwise sample0-vs-alternative state:

\[
s_{0,k}=[S_k-S_0, U_k-U_0, R_{fact,k}-R_{fact,0}, K_k-K_0]
\]

其中 `R_fact` 来自 factual residual，`K` 来自 Riemann/curvature features。

本次结果显示，`R_fact` 的绝对值不足以单独替代 stability/waveform，但 `delta_R_fact` 对 rescue/damage 很有用。因此下一版 controller 应优先尝试：

- 保留现有 `theory_core` 作为主评分骨架。
- 增加 `delta_qwen_factual_residual_vs_sample0` 作为 damage veto / rescue bonus，而不是直接做全局 factual ranker。
- 使用 `riemann_anisotropy` 和 `riemann_curvature_proxy` 作为弱几何项；它们不应单独决策，但可以帮助区分 stable hallucination 的 basin shape。
- 暂时不要重依赖 hidden-state geometry；当前 mean-pooling 版本接近随机，下一步需要更细的 token-level manifold 或 layer sweep。

不要把本实验中的 label/grouping 字段放进 controller。
