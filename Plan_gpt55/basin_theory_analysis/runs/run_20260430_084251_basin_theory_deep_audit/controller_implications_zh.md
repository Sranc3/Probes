# Controller Implications from Basin Theory Audit

## 核心结论

当前最值得投入的不是更重的 RL，而是把 controller state 从浅层 geometry 扩展为 stability + waveform dynamics + sample0-relative contrast + future factual residual。

建议的 state 形式：

\[
s_k = [S_k, G_k, U_k, \Delta_{0\rightarrow k}, R_{fact}]
\]

其中本实验覆盖了 `S_k`、`G_k`、`U_k` 和 `Delta`，尚未覆盖真正的 `R_fact`。

从本轮结果看，`rescue_vs_damage` 的 waveform family AUC 达到 `0.831`，高于 stability / geometry；而 `stable_correct_vs_hallucination` 仍主要由 stability/geometry 分开，但 AUC 只有约 `0.70`。这说明：

- waveform 更像 damage/rescue 的局部风险信号；
- stability 更像 basin attractor 强度信号；
- 两者都不是事实正确性的充分条件；
- 缺失的关键块仍然是 \(R_{fact}\)。

## 下一步优先级

1. 若 waveform functional 在 stable hallucination 上稳定分离，应把 `wf_entropy_max`、`roughness`、`prob_min` 接入 prefix risk model。
2. 若 sample0_delta 对 rescue/damage 更强，应把 controller 改成 pairwise sample0-vs-alternative scoring，而不是 absolute basin scoring。
3. 若所有 no-leak 数值特征仍弱，则下一步应做 semantic/factual basin verifier pilot，而不是继续调数值 controller。
