# Controller Implications: Factual + Riemann

建议下一版 controller 不再只做 absolute basin scoring，而是使用 pairwise sample0-vs-alternative state:

\[
s_{0,k}=[S_k-S_0, U_k-U_0, R_{fact,k}-R_{fact,0}, K_k-K_0]
\]

其中 `R_fact` 来自 factual residual，`K` 来自 Riemann/curvature features。

增强版 factual verifier 的用法：

- `qwen_structured_factual_residual` 作为 absolute factual risk。
- `delta_qwen_structured_factual_residual_vs_sample0` 作为 alternative basin 是否比 sample0 更可信的核心信号。
- `qwen_world_knowledge_conflict_risk` 适合做 damage veto。
- `qwen_answer_responsiveness_score` 和 `qwen_constraint_satisfaction_score` 适合做 rescue bonus 的必要条件。
- `strong_verifier_controller` feature family 将这些 verifier 特征和 `delta_wf_prob_min_mean_vs_sample0`、`delta_cluster_weight_mass_vs_sample0`、`riemann_anisotropy` 合并，用来检验是否能比单独 verifier 更强。

不要把本实验中的 label/grouping 字段放进 controller。
