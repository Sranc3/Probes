# Controller Implications: Factual + Riemann

建议下一版 controller 不再只做 absolute basin scoring，而是使用 pairwise sample0-vs-alternative state:

\[
s_{0,k}=[S_k-S_0, U_k-U_0, R_{fact,k}-R_{fact,0}, K_k-K_0]
\]

其中 `R_fact` 来自 factual residual，`K` 来自 Riemann/curvature features。

优先使用 question-heldout AUC 最高且无泄漏的 feature family；不要把本实验中的 label/grouping 字段放进 controller。
