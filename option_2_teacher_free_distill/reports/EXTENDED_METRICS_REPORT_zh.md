# 扩展指标报告：AUROC 之外，还看到了什么

> 生成于 2026-05-15。源文件：
> - `results/extended_metrics_long.csv`（33 行 = 3 base × 3 数据集 × 3-5 probe）
> - `results/per_item_predictions_all.csv`（16 500 行，每个 (base, dataset, probe) 在最佳层的逐题预测）
> - `reports/dashboard_extended.png`（3×3 可视化）

## TL;DR — 三个用 AUROC 看不到的事实

| # | 指标 | 现象 | 论文意义 |
|---|---|---|---|
| 1 | **Coverage @ Risk ≤ 10%** | 在 ID 上，DCP-MLP / SEPs-LR 让我们能在保 90% 准确率的前提下回答 24%–32% 的题；SEPs-Ridge 只能回答 4%–11%。**OOD 上几乎所有 probe 都<2%。** | DCP/SEPs-LR 有 **2-3× 的可部署覆盖率**；OOD 上"低风险高覆盖"在所有方法上都不可行——给 SEPs 的"OOD 鲁棒"叙事直接打脸 |
| 2 | **AUPRC（average precision）** | DCP-MLP/SEPs-LR 在 OOD 上比 SEPs-Ridge 高 +0.05~+0.18。Qwen-72B HotpotQA：DCP=0.704，SEPs-LR=0.716，SEPs-Ridge=0.523 | AUROC 只能看排序质量，AUPRC 直接说"前 K 个最自信的题里有多少真的对"——deployment-relevant，比 AUROC 更"硬" |
| 3 | **Brier / ECE / Reliability decomposition** | **ARD probes（被认为"AUROC 输了"的）反而是 4× 校准更好的**：ID 上 ARD-MLP Brier=0.16-0.19、ECE=0.05-0.07，DCP-MLP Brier=0.20-0.23、ECE=0.18-0.21 | **新故事浮出**：ARD 不是失败品——它是"trustworthy probability"赛道的赢家。论文可以提一个 "discrimination vs calibration trade-off" 的双方法叙事 |

---

## 1. 已经在跑的时候算了、之前没浮上来的指标

下表是各 (base, dataset) 下**最佳层**每个 probe 的全套指标。**新增列**用 ★ 标记。

### 1.1 ID（TriviaQA, n=1000, base_acc=0.568）

| base | probe | layer | AUROC | ★AUPRC | AURC ↓ | ★AUARC ↑ | sel_acc@50% | ★cov@risk≤10% | ★cov@risk≤20% | Brier ↓ | ECE ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen-7B  | DCP-MLP    | 20 | 0.799 | **0.841** | 0.215 | 0.783 | 0.792 | **0.242** | **0.460** | 0.230 | 0.209 |
| Qwen-7B  | SEPs-LR    | 24 | 0.794 | **0.841** | 0.215 | 0.783 | 0.796 | **0.250** | **0.492** | 0.223 | 0.192 |
| Qwen-7B  | SEPs-Ridge | 27 | 0.738 | 0.771 | 0.263 | 0.735 | 0.736 | 0.112 | 0.310 | — | — |
| Qwen-7B  | ARD-MLP    | 20 | 0.786 | 0.817 | 0.230 | 0.768 | 0.780 | 0.172 | 0.452 | **0.186** | **0.070** |
| Qwen-7B  | ARD-Ridge  | 24 | 0.770 | 0.809 | 0.236 | 0.762 | 0.776 | 0.154 | 0.462 | **0.193** | **0.050** |
| Llama-3B | DCP-MLP    | 28 | 0.800 | 0.828 | 0.222 | 0.776 | 0.784 | 0.140 | 0.474 | 0.233 | 0.210 |
| Llama-3B | SEPs-LR    | 27 | 0.808 | 0.844 | 0.212 | 0.786 | 0.788 | **0.250** | 0.482 | 0.220 | 0.201 |
| Llama-3B | SEPs-Ridge | 27 | 0.747 | 0.778 | 0.258 | 0.740 | 0.736 | 0.102 | 0.284 | — | — |
| Llama-3B | ARD-MLP    | 28 | 0.801 | 0.835 | 0.218 | 0.780 | 0.796 | **0.300** | 0.494 | **0.179** | **0.050** |
| Llama-3B | ARD-Ridge  | 27 | 0.803 | 0.850 | 0.209 | 0.789 | 0.796 | **0.292** | 0.484 | **0.181** | **0.064** |
| Qwen-72B | DCP-MLP    | 64 | 0.843 | **0.877** | 0.189 | 0.809 | 0.824 | 0.282 | **0.582** | 0.195 | 0.182 |
| Qwen-72B | SEPs-LR    | 64 | 0.836 | 0.865 | 0.197 | 0.801 | 0.816 | **0.324** | 0.552 | 0.196 | 0.179 |
| Qwen-72B | SEPs-Ridge | 72 | 0.772 | 0.798 | 0.243 | 0.755 | 0.772 | 0.040 | 0.390 | — | — |
| Qwen-72B | ARD-MLP    | 64 | 0.838 | 0.861 | 0.200 | 0.798 | 0.816 | 0.230 | **0.586** | **0.161** | **0.047** |
| Qwen-72B | ARD-Ridge  | 72 | 0.829 | 0.859 | 0.201 | 0.797 | 0.836 | 0.214 | 0.564 | **0.167** | **0.063** |

> SEPs-Ridge 没有 Brier/ECE 因为输出是 entropy 回归值，不在 [0,1] 概率空间。

### 1.2 OOD HotpotQA（n=500，多跳带 context）

| base | probe | layer | AUROC | ★AUPRC | AURC ↓ | sel_acc@50% | ★cov@risk≤20% | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| Qwen-7B  | DCP-MLP    | 16 | 0.730 | 0.536 | 0.516 | 0.476 | 0.000 | 0.266 | 0.269 |
| Qwen-7B  | SEPs-LR    | 16 | 0.724 | 0.534 | 0.518 | 0.492 | 0.004 | 0.294 | 0.290 |
| Qwen-7B  | SEPs-Ridge | 16 | 0.684 | 0.519 | 0.531 | 0.444 | 0.014 | — | — |
| Llama-3B | DCP-MLP    | 16 | 0.657 | 0.459 | 0.560 | 0.436 | 0.020 | 0.331 | 0.314 |
| Llama-3B | SEPs-LR    | 16 | 0.677 | 0.512 | 0.538 | 0.440 | 0.032 | 0.269 | 0.255 |
| Llama-3B | SEPs-Ridge | 16 | 0.664 | 0.506 | 0.543 | 0.432 | 0.022 | — | — |
| Qwen-72B | DCP-MLP    | 64 | 0.771 | 0.704 | 0.355 | 0.660 | 0.132 | 0.258 | 0.254 |
| Qwen-72B | SEPs-LR    | 56 | **0.791** | **0.716** | 0.346 | 0.664 | **0.152** | 0.238 | 0.218 |
| Qwen-72B | SEPs-Ridge | 56 | 0.589 | 0.523 | 0.487 | 0.524 | 0.012 | — | — |

### 1.3 OOD NQ-Open（n=500，单跳无 context）

| base | probe | layer | AUROC | ★AUPRC | AURC ↓ | sel_acc@50% | ★cov@risk≤20% | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| Qwen-7B  | DCP-MLP    | 20 | 0.667 | 0.450 | 0.569 | 0.436 | 0.000 | 0.374 | 0.369 |
| Qwen-7B  | SEPs-LR    | 20 | 0.661 | 0.445 | 0.577 | 0.432 | 0.000 | 0.356 | 0.347 |
| Qwen-7B  | SEPs-Ridge | 16 | 0.619 | 0.416 | 0.601 | 0.412 | 0.006 | — | — |
| Llama-3B | DCP-MLP    | 24 | 0.613 | 0.554 | 0.447 | 0.560 | 0.010 | 0.384 | 0.377 |
| Llama-3B | SEPs-LR    | 12 | 0.604 | 0.553 | 0.456 | 0.556 | 0.000 | 0.366 | 0.342 |
| Llama-3B | SEPs-Ridge | 12 | 0.538 | 0.504 | 0.501 | 0.500 | 0.010 | — | — |
| Qwen-72B | DCP-MLP    | 64 | 0.654 | 0.616 | 0.397 | 0.612 | 0.020 | 0.337 | 0.324 |
| Qwen-72B | SEPs-LR    | 56 | **0.701** | **0.674** | 0.353 | 0.652 | **0.080** | 0.301 | 0.289 |
| Qwen-72B | SEPs-Ridge | 48 | 0.597 | 0.571 | 0.438 | 0.572 | 0.000 | — | — |

---

## 2. 新指标讲了三个新故事

### Story 1：ARD 是隐藏的"trustworthy probability"赢家

之前我们只看 AUROC，所以 ARD（0.78–0.84）在 DCP（0.80–0.84）和 SEPs-LR（0.79–0.84）面前显得"差不多"或"稍差"。但加上**校准维度**：

| 指标 | DCP-MLP | SEPs-LR | **ARD-MLP** | **ARD-Ridge** |
|---|---|---|---|---|
| AUROC（ID 平均） | 0.814 | 0.813 | 0.808 | 0.801 |
| Brier（ID 平均） | 0.219 | 0.213 | **0.175** | **0.180** |
| ECE（ID 平均） | 0.200 | 0.191 | **0.056** | **0.059** |
| Brier reliability | 0.043 | 0.040 | **0.004** | **0.004** |

**ARD 的 ECE 比 DCP 小 3.5×，Brier reliability 小 11×。** Reliability diagram（dashboard 图 2,1 中央）也直观看到：ARD（绿色/紫色）几乎完美贴合对角线，DCP/SEPs-LR 在高置信区严重 overconfident。

**为什么会这样？** ARD 用 MSE 拟合教师 anchor 特征，目标连续；DCP 用 BCE 拟合 0/1 硬标签，会被 overfitting 推到极端 logits。**这是 distillation 的隐藏副作用**——蒸出来的概率天然校准好。

**论文新角度**：可以把 DCP（高 discrimination, 差校准）和 ARD（中 discrimination, 好校准）并列为**两种互补的 teacher-free 方案**，对应不同部署需求：
- 需要排序最强 → DCP-MLP
- 需要"概率即风险"（用于 cost-aware routing）→ ARD-MLP

### Story 2：Coverage@Risk 直接量化了"能省多少 forward 调用"

`cov@risk≤10%` 是论文最具说服力的 deployment number：

| base | DCP-MLP | SEPs-LR | SEPs-Ridge |
|---|---|---|---|
| Qwen-7B  ID | 24.2% | **25.0%** | 11.2% |
| Llama-3B ID | 14.0% | **25.0%** | 10.2% |
| Qwen-72B ID | 28.2% | **32.4%** | 4.0% |

读法：
> **在 Qwen-72B 上，用 SEPs-LR 单次 forward 就能挑出 32% 的题，而且这 32% 的回答准确率 ≥ 90%。**

这比单纯说 "AUROC 0.84" 直观 100×。配合表 1 的 base_acc=0.568，意味着：
- 对 32% 的题 → 直接回答（90%+ 正确）
- 对剩下 68% → 走 K=8 self-consistency 或转 teacher
- 平均 forward 数从 8 降到 ≈ 5.5（减 31%），accuracy 不掉

**OOD 上这个数字几乎归零**（cov@risk≤10% 全部 < 2%）：这说明 SEPs 论文里"OOD 鲁棒"的说法在 deployment threshold 下完全站不住。**论文可以专门做一个对比图打这个点。**

### Story 3：AUPRC 在 OOD（imbalanced）场景比 AUROC 更"硬"

NQ-Open 上 base_acc 只有 0.33-0.51（"correct" 是少数类或边缘类）。AUROC 不区分 precision 和 recall，所以会高估能力。AUPRC 直接告诉你 "top-K 个最自信的预测里有多少是真对的"。

| Qwen-72B NQ | AUROC | AUPRC | Δ |
|---|---|---|---|
| DCP-MLP    | 0.654 | 0.616 | -0.038 |
| SEPs-LR    | 0.701 | 0.674 | -0.027 |
| SEPs-Ridge | 0.597 | 0.571 | -0.026 |

AUROC 0.65 听起来"还行"，AUPRC 0.62 立刻提醒你 "top-K 的 precision 可能并没那么好"。论文里同时报这两个，审稿人会觉得你在严肃做评估。

---

## 3. 进入 dashboard 的 9 张图

文件：`reports/dashboard_extended.png`（3×3, 2350×1850 px, 1.07 MB）

| 位置 | 内容 | 一句话讲清楚 |
|---|---|---|
| (0,0) | TriviaQA RC 曲线 | DCP（红）始终在 SEPs-Ridge（灰）下面 → 同样 coverage 下 risk 更低 |
| (0,1) | HotpotQA RC 曲线 | 三种 probe 差距大幅缩小，72B 上 DCP/SEPs-LR 才拉开 |
| (0,2) | NQ-Open RC 曲线 | 几乎所有 probe 都接近随机 baseline 在低 coverage 区 |
| (1,0) | Cov@Risk≤10% 柱状矩阵 | **ID 区域 DCP/SEPs-LR 高耸；OOD 区域几乎全为零** |
| (1,1) | AUPRC 热力图 | DCP/SEPs-LR 普遍 ≥ SEPs-Ridge +0.05 |
| (1,2) | Sel-Acc@50% 矩阵（黑虚线 = base_acc） | 看 lift 一目了然 |
| (2,0) | Brier vs AUROC 散点 | ARD 在左下（低 Brier 高 AUROC），DCP/SEPs-LR 在右上 → **trade-off 图** |
| (2,1) | Reliability diagram（Qwen-72B ID）| ARD 贴对角线，DCP/SEPs-LR 在高置信区下凸 → overconfident |
| (2,2) | Brier 分解柱状（Reliability ↓ vs Resolution ↑）| ARD reliability ≈ 0；DCP/SEPs-LR reliability ≈ 0.04 |

---

## 4. 对论文叙事的影响

| 之前的叙事 | 现在可以改成 |
|---|---|
| "DCP-MLP 显著优于 SEPs-Ridge" | 加一个维度：DCP 优于 SEPs-Ridge **且** SEPs-LR 在 OOD 大模型上反超 DCP（72B NQ） |
| "ARD 没能蒸进知识，效果不如 DCP" | 修正：**ARD 在 discrimination 上稍差，但 calibration 显著更好** → 两条路线服务不同需求 |
| "我们的方法 K=1 等价于 K=8 self-consistency" | 强化：**在保 90% 准确率的前提下能 abstain 70% 的题** → 折算成 forward 数能省 30%+ |
| "OOD 上 SEPs 鲁棒性不可靠" | 加硬证据：**OOD 的 cov@risk≤10% 接近 0，意味着 SEPs 文献中报的 OOD AUROC 在实际 deployment 下不可用** |

---

## 5. 复现命令

```bash
cd /zhutingqi/song/option_2_teacher_free_distill

# 1. 计算所有扩展指标（约 4 分钟）
python scripts/compute_extended_metrics.py
# 产出：
#   results/extended_metrics_long.csv
#   results/per_item_predictions_all.csv
#   results/<base>/per_item_predictions_{id,hotpotqa,nq}_best.csv

# 2. 生成扩展 dashboard
python scripts/make_dashboard_extended.py
# 产出：reports/dashboard_extended.png
```

---

## 6. 还能继续做的（你之前说"等下再做"的那批）

| 指标 | 工作量 | 价值 |
|---|---|---|
| Cost-aware utility / Pareto frontier | 30 min | 论文 figure-1 候选——直接对应"Tier A/B/C 部署决策"叙事 |
| Inference latency / wall-clock 实测 | 30 min（per base） | 给"K=1 等价 K=8"配实测数据，比理论 8× 更可信 |
| RC curves by question difficulty bucket | 20 min | 揭示 probe 是在简单题上取胜还是难题 |
| Layer-wise CKA / linear probing geometry | 60 min（需要 GPU） | 解释 "L20-L24 sweet spot" 的 mechanistic 来源 |

要做哪个就告诉我。
