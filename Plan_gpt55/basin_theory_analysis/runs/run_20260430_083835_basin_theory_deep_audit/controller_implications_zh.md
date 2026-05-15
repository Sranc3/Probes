# Controller Implications from Basin Theory Audit

## 核心结论

当前最值得投入的不是更重的 RL，而是把 controller state 从浅层 geometry 扩展为 stability + waveform dynamics + sample0-relative contrast + future factual residual。

建议的 state 形式：

\[
s_k = [S_k, G_k, U_k, \Delta_{0\rightarrow k}, R_{fact}]
\]

其中本实验覆盖了 `S_k`、`G_k`、`U_k` 和 `Delta`，尚未覆盖真正的 `R_fact`。

## 下一步优先级

1. 若 waveform functional 在 stable hallucination 上稳定分离，应把 `wf_entropy_max`、`roughness`、`prob_min` 接入 prefix risk model。
2. 若 sample0_delta 对 rescue/damage 更强，应把 controller 改成 pairwise sample0-vs-alternative scoring，而不是 absolute basin scoring。
3. 若所有 no-leak 数值特征仍弱，则下一步应做 semantic/factual basin verifier pilot，而不是继续调数值 controller。
