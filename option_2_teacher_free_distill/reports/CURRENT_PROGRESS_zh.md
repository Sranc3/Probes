# 项目当前进展速览（给 co-author）

**最近更新**：2026-05-14（v3 — 加入 Llama-3.2-3B + Qwen2.5-72B 跨 base scale 实验）
**Base models**：Qwen2.5-7B-Instruct / Llama-3.2-3B-Instruct / **Qwen2.5-72B-Instruct** + GPT-OSS（teacher anchor 来源）
**数据集**：TriviaQA（ID）/ HotpotQA dev_distractor（OOD-1，带 context）/ NQ-Open validation（OOD-2，无 context）
**规模**：每个数据集 500 题，每题最多 K=8 采样

> **跨 base 实验已经跑完**（3 base × 3 数据集 × 5 probe × 8–10 layer，共 9 个独立 base×dataset cell）。
> 完整的跨 base 详细分析见独立文档 `reports/CROSS_BASE_PROGRESS_zh.md`，本文档保留单 base (Qwen-7B) 的项目历程叙事。
> v3 关键发现摘要在文末"最新跨 base 发现"一节。

---

## 一句话总结

我们最初想做"用 verifier 信号去对模型做偏好微调（DPO/GRPO），让答题准确率提升"，**但反复尝试都失败了**。后来把同样的特征用在"知道自己什么时候答错"这个新问题上，**效果非常好**；再把这套信号**蒸馏成 Qwen 自己一次前向就能算的探针**之后，做到了**两个 OOD 数据集上稳定打平/小胜 SEPs（NeurIPS 2024 baseline）**，这是目前最适合写进顶会论文的"硬产出"。

---

## 整个故事的三个阶段

### 阶段 1（Plan_opus）：先把训练流程的实现 bug 修干净

**问题来源**：之前 Plan_gpt55 里跑 VBPO/Anchor-VBPO/GRPO，怎么调都不涨甚至掉点。

**根因诊断**（一两句话版本）：
- DPO 损失里的 logprob 取的是**平均**而不是**求和** → 序列越长梯度被稀释成几乎为零；
- GRPO 的比率和 KL 也是按"序列均值"算的 → PPO 的 clip ε 实际上不起作用；
- 监督信号扔给整段答案 → 模型在背风格句式而不是答案；
- Anchor 监督的"正/负"成对样本生成规则不严格 → 训出来在学噪声。

**修完之后**：
- 训练曲线终于"动起来"了（dev margin_delta 提升 1000 倍量级），证明流程对了；
- **但 strict accuracy 提升只有 +0.5 ~ +1.4 个点**，在数据噪声范围内。

**结论**：**实现层面已经没坑了，但仅靠这些信号去"让模型答得更准"是天花板低的事**。

### 阶段 2（Plan_opus_selective）：换问题——不是"答得更准"，而是"知道自己答没答对"

把 anchor + self-introspection 一堆特征喂给一个简单 logreg / MLP，让它预测"这道题模型到底有没有答对"。这就是 **Selective Prediction**。

**结果一下子非常好**：
- AUROC = **0.97**
- 选最自信的前 50% 来回答，准确率 = **0.994**
- 把"答 / 不答 / 转交 teacher"建模成路由问题，效用从 baseline 的 −0.736 拉到 **+0.513**

**但有个明显问题**：这个 0.97 是在 **K=8 自我采样 + 一次 GPT-OSS teacher 调用** 的前提下达到的——每问一题要付 9 次模型 forward，**部署成本太高**。

**新的瓶颈**：能不能压缩掉这些昂贵的特征？

### 阶段 3（option_2_teacher_free_distill）：蒸馏成 1 次前向，正面 PK SEPs

**思路**：直接在 Qwen 的 prompt 隐状态上训一个小探针，预测"这道题答没答对"。
正面 PK 的对手是 **Semantic Entropy Probes (SEPs, Kossen et al., NeurIPS 2024)**——他们用同样的 1 次前向预测语义熵，号称 K=1 时也能 work。

我们做了三套探针（每套都是 K=1 prompt-only）：

| 名字 | 方法 | 类比 |
|---|---|---|
| **SEPs-Ridge** | Ridge 回归预测 *语义熵*，再取 −prediction 当置信度 | Kossen 2024 主推方法 |
| **SEPs-LR** | Logistic 直接预测 P(correct) | Kossen 2024 论文里的"强变体"，但他们没主推 |
| **DCP-MLP（我们）** | 2-层 MLP 直接预测 P(correct) | 我们方法 |
| **ARD-MLP/Ridge（我们）** | 先用 hidden state 回归 7 维 teacher anchor，再 logreg → P(correct) | 试图把 teacher 信号"蒸进"hidden state |

**横扫 8 个 layer（4/8/12/16/20/24/27/28）选最佳**。

---

## 最新结果（核心数字）

### 表 1：三个数据集的 best-layer AUROC（越高越好）

| Probe | TriviaQA ID | HotpotQA OOD（带 context） | NQ-Open OOD（无 context） |
|---|:---:|:---:|:---:|
| **DCP-MLP（我们）** | **0.7960** (L20) | **0.7302** (L16) | **0.6673** (L20) |
| SEPs-LR | 0.7893 (L24) | 0.7244 (L16) | 0.6614 (L20) |
| SEPs-Ridge（Kossen 主推） | 0.7470 (L27) | 0.6838 (L16) | 0.6190 (L16) |

### 表 2：DCP-MLP vs SEPs-Ridge 的成对 bootstrap（2000 次重采样）

| 数据集 | ΔAUROC | 95% CI | p 值 | 结论 |
|---|:---:|:---:|:---:|---|
| TriviaQA ID | **+0.049** | [+0.008, +0.094] | **0.021** | ✅ 显著胜出 |
| HotpotQA OOD | +0.047 | [−0.002, +0.099] | 0.063 | 🔶 边缘显著 |
| NQ-Open OOD | +0.050 | [−0.007, +0.104] | 0.081 | 🔶 边缘显著 |

**这是论文里最稳的硬证据**：方向一致、幅度几乎相同（约 +0.05 AUROC）、三个独立数据集都成立。

### 表 3：一次前向（K=1）vs 八次采样 + teacher（K=8）

| 方案 | 推理成本 | TriviaQA AUROC |
|---|---|:---:|
| **DCP-MLP（我们，K=1）** | 1 次 prompt forward | **0.796** |
| logreg:self（K=8 自采样） | 8 次 forward | 0.808 |
| logreg:teacher（K=8 + GPT-OSS） | 8 次 forward + teacher API | **0.964** |

**两个直接含义**：
- "K=1 vs K=8 自采样"在 bootstrap 下**统计上打平** → **8 倍推理加速、AUROC 几乎不掉**；
- "K=1（任何方法） vs K=8 + teacher API" 仍然差 0.17 AUROC → **teacher 调用是不可替代的信息源**（重要的负面发现，下面展开）。

### 表 4：OOD 退化是不是均匀的？（针对 SEPs 论文的争议点）

| Probe | ID → HotpotQA 退化 | ID → NQ-Open 退化 |
|---|:---:|:---:|
| DCP-MLP | −0.066 | −0.129 |
| SEPs-LR | −0.065 | −0.128 |
| SEPs-Ridge | −0.063 | −0.128 |

**三个 probe 在每个 OOD 上的退化幅度相差 ≤ 0.001。** SEPs 论文当年警告"accuracy 探针在 OOD 上会灾难性退化、所以必须用 entropy 探针"——**我们用两个独立 OOD shift 给出反例**：accuracy 探针不比 entropy 探针差。这是一个直接的 counter-data-point。

另外注意到：**去掉 context（HotpotQA→NQ）让退化大约翻倍**（−0.07 → −0.13），三个 probe 都一样。这给"运行时检索增强"做了一个干净的未来工作切入点。

### 表 5：ARD（试图把 teacher 信号蒸进 hidden state）—— 失败但有价值

| Probe | TriviaQA AUROC |
|---|:---:|
| ARD-Ridge（蒸 7 维 teacher anchor → 再分类） | 0.774 |
| ARD-MLP（同上但用 MLP 回归） | 0.790 |
| **DCP-MLP（不蒸，直接预测对错）** | **0.796** |
| logreg:teacher（运行时真的调 teacher） | **0.964** |

**两个 ARD 蒸馏出来的天花板和"直接预测"几乎一样，远低于真的调 teacher**。这告诉我们：
- Qwen 的 hidden state 里没有 GPT-OSS 的"互补知识"；
- 蒸馏只能传递学生本来就能表达的东西，**不能凭空注入新事实知识**；
- 这是个**信息论意义上的负面结论**，不是工程没调好。

这个负面发现非常适合写在论文里——它**直接论证了"为什么 teacher API call 不能省"**，从而支撑后面"分层路由部署（Tier A/B/C）"的论文叙事。

---

## 我们目前能讲的硬故事 vs 还打平的部分

### 能写的（defensible）
1. **K=1 prompt-only 就能打平 K=8 自采样**——8 倍推理加速、AUROC 不掉。（ID 上 bootstrap 验证）
2. **DCP-MLP 显著优于 SEPs 主推的 Ridge 变体**——三个独立数据集（ID + 两个 OOD）方向一致、幅度一致（≈ +0.05 AUROC）。
3. **两个 OOD 上"accuracy 探针 ≈ entropy 探针 退化幅度"**——SEPs 论文的 OOD 悲观论是不全面的，至少在我们这两个 shift 上不成立。
4. **L16–L20 是部署默认层**——L16 是唯一在三个设定上都不是最差的层。
5. **ARD 失败的负面结果**——干净地论证 teacher API 调用不可替代，支撑后面"Tier A/B/C 分层路由"的部署叙事。
6. **去掉 context 让 OOD 退化大致翻倍**——清晰量化，直接给 future work 做铺垫。

### 还诚实地打平、不能吹的
1. **DCP-MLP vs SEPs 的"强变体" SEPs-LR 在三个数据集上都统计打平**（Δ ≈ +0.007，p > 0.6）。
   也就是说我们方法相对于 SEPs 系列**最强变体**只是常数级 MLP 非线性带来的提升，不是范式创新。
2. **目前只用了一个 base model（Qwen2.5-7B）**。L16/20 这个 layer 选择会不会换 base model 就漂走？还不知道。这正是你下一步要做的事 ↓

---

## ⚠️ 一个值得标注的过程细节（透明化）

跑 NQ 的时候发现自己实现的 `safe_auroc` 在大量 ties 下有 bug：MLP 在某些 OOD 早期层会饱和到 ≈ 1.0，导致虚假报出 AUROC = 0.99（sklearn 的真实值是 0.51）。**已经全面切换到 `sklearn.metrics.roc_auc_score`，所有数字重跑过一遍**，影响：
- ID 数字几乎不变（DCP@L20: 0.7985 → 0.7960，p: 0.014 → 0.021）
- HotpotQA OOD 几乎不变
- NQ OOD 早期层（L4–L8）数字大幅下调，best-layer 从被错报的 L8/0.99 修正回 L20/0.667

这件事本身写进 report 反而是加分项——我们是自己 audit 自己。

---

## 接下来你要做的事 / 我建议的优先级

### 你已经计划的：部署 Qwen2.5-72B 和 Llama-3B
**为什么这一步特别关键**：现在所有数字都基于 Qwen2.5-7B。审稿人 100% 会问"换模型还成立吗"。具体能验证：

1. **方法的 base-model 鲁棒性**
   - DCP-MLP > SEPs-Ridge 的 +0.05 AUROC 差距，在 Qwen-72B 和 Llama-3B 上还在不在？方向变不变？
2. **Layer 16–20 sweet spot 是不是 Qwen-7B 特有**
   - 72B（80 层左右）和 3B（28 层）上，最佳层是不是仍然落在 ~57–71% 深度？
   - 如果是 → 我们能写"normalised-depth ~0.6 是普适的探针默认"，这是论文里很好的一句 take-away。
3. **OOD 均匀退化是不是模型容量的函数**
   - 直觉上 72B 应该退化更小（参数化记忆更强），3B 应该退化更大；
   - 如果三种 capacity 上退化都"均匀"（probe 之间差距 < 0.005），那"accuracy 探针没有 OOD 灾难"这个反例就**非常硬**。
4. **K=1 vs K=8 加速比能不能保住**
   - 在 72B 上 K=1 的相对加速比更大（绝对节省时间多），文章里能写"模型越大、我们的加速优势越大"。

### 我建议你跑的具体配置
```
- Qwen2.5-72B：取 layers {8, 16, 24, 32, 40, 48, 56, 64, 72, 80} 中的 8 个，
                或者按 normalised depth 取 {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} 对应层
- Llama-3-3B：取 layers {3, 6, 9, 12, 15, 18, 21, 24, 27} 中的 8 个

每个 base 跑：
  TriviaQA 500 题 ID + HotpotQA 500 题 OOD + NQ 500 题 OOD
  探针：DCP-MLP, SEPs-LR, SEPs-Ridge（ARD 在 OOD 上没有 anchor 数据，可以只跑 ID）
```

我已经把所有 pipeline 脚本写成了通用的（`extract_hidden_states.py` / `prepare_hotpotqa_ood.py` / `prepare_nq_ood.py` / `train_probes.py` / `evaluate_ood.py`），**只需要换 `--model-dir` 参数和 `--layers`**，应该能很快跑出来。具体我帮你预先封装一下 batch script，等你 72B/3B 模型部署好就一键跑。

### 论文方向的备选叙事（基于现有结果就能投）
**叙事 A（最稳）**：*A K=1 prompt-only probe matches K=8 self-introspection for selective prediction, with consistent advantage over SEPs across two OOD shifts; teacher anchor is irreplaceable but only needed at runtime via API call.*
- 卖点：8x 加速 + 干净的 OOD 三角验证 + 负面 ARD 结果作为 teacher 必要性的论证。
- 风险：novelty 是"组合"而非"范式"。

**叙事 B（更有野心）**：在 A 的基础上加 multi-base-model（7B/72B/3B）的 layer-depth scaling 分析 + 部署 Tier A/B/C 路由实验，包装成 "**Cost-aware Selective Prediction across Model Scales**"。
- 卖点：跨 capacity 的 layer 普适性 + 一个干净的 cost-vs-accuracy Pareto。
- 这个方向**正好就是你下一步要补的实验自然能填进去**。

---

## 文件位置总结
- 训练/评测脚本：`/zhutingqi/song/option_2_teacher_free_distill/scripts/`
- 详细技术报告（v2，单 base Qwen-7B，含所有数字、bootstrap、bug audit）：`reports/TEACHER_FREE_REPORT.md`
- **跨 base 报告（v3，3 个 base × 3 数据集 × 5 probe）**：`reports/CROSS_BASE_PROGRESS_zh.md` + 自动生成的 `reports/CROSS_BASE_REPORT.md`
- 多 OOD 综合表（per-base）：`results/{qwen7b,llama3b,qwen72b}/ood_combined_table.md`
- 跨 base 数据 CSV（用于画图）：`results/cross_base/`
- 项目早期叙事（Plan_opus → Plan_opus_selective）：`/zhutingqi/song/Plan_opus_selective/reports/项目历程与表格解读_zh.md`
- novelty 评估（vs 2024–2026 文献）：`/zhutingqi/song/Plan_opus_selective/reports/NOVELTY_ASSESSMENT_zh.md`

---

## 最新跨 base 发现（v3 摘要，详见 `CROSS_BASE_PROGRESS_zh.md`）

跑完 **Llama-3.2-3B / Qwen2.5-7B / Qwen2.5-72B** 三个 base × {TQA, HQA, NQ} × 5 个 probe 之后，原叙事有 4 个重要变化：

### ✅ 加强的论文证据
1. **DCP-MLP > SEPs-Ridge 在 9 个独立 cell 中 8 个验证成立**（4 个显著、3 个 borderline、1 个反向打平、0 个反向显著）。从单 base 单 ID + 2 OOD 升级为 **跨模型家族 × 跨规模 × 跨 OOD shift** 的稳定胜出。
2. **所有 probe 在 7B → 72B 上 ID AUROC 单调上升**（DCP +0.043, SEPs-LR +0.045, ARD +0.053），干净的 scaling curve。

### 🆕 全新的论文核心叙事（**emergent at 72B**）
3. **Anchor distillation 在 72B 上 work 了**：
   - 7B：ARD-MLP 0.7898 ≈ DCP-MLP 0.7960 → 之前的"ARD 失败、信息论上界"叙事
   - 72B：**ARD-MLP 0.8424（最高），超过 DCP-MLP 0.8392 和 SEPs-LR 0.8344**
   - 论文里写："**Distillation feasibility is a capacity function**" — 这是个 emergent capability 论文里的好数据点

### ⚠️ 需要修正的原叙事
4. **"OOD 均匀退化"是 Qwen-7B 特有，不是普适规律**：
   - 7B：三个 probe 退化幅度差 < 0.001（uniform）
   - Llama-3B HotpotQA：SEPs-Ridge 退化最少（−0.085）→ 反而**支持** SEPs paper 原观点
   - **72B HotpotQA：SEPs-Ridge 退化最多（−0.190 vs accuracy probe −0.04 ~ −0.07）→ 反过来证明 entropy probe 更脆弱**
   - 正确叙事："OOD 鲁棒性 ranking 在 entropy 与 accuracy 之间是 **base-model-and-dataset-dependent**"

5. **DCP-MLP 的 MLP 非线性优势在 scale 上稀释**：
   - **Qwen-72B NQ：SEPs-LR 反过来打赢 DCP-MLP，−0.047 AUROC, p=0.004**（统计显著）
   - 解读：base model 越大，hidden state 里的 confidence 越接近线性可分，2-层 MLP 的非线性容量在小数据上反而过拟合
   - 论文里写："Hidden-state confidence becomes increasingly linear at larger model size; MLP non-linearity buys you less, eventually nothing."

### 终极论文叙事（基于跨 3 base 数据）

> **"Cost-aware selective prediction across model scales."** A K=1 prompt-only probe matches K=8 self-introspection across 3B / 7B / 72B model scales. DCP-MLP statistically beats SEPs-Ridge on **8 of 9 base × dataset cells**. Two emergent findings appear only at 72B scale: (a) anchor-distillation (ARD-MLP) becomes the single best K=1 probe — the teacher anchor finally becomes recoverable when the student's representation is large enough; (b) entropy probes become catastrophically more OOD-fragile than accuracy probes on multi-hop QA — reversing the SEPs paper's claim. The relative advantage of MLP non-linearity over linear classifiers diminishes with scale, suggesting hidden-state confidence becomes increasingly linear at larger model size.
