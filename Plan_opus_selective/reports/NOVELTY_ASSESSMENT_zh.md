# Novelty 评估 & 投稿定位建议

> 写作日期：2026-05-14
> 目的：在准备投顶会之前，把"我们到底新在哪"和"哪些已经被发表"画出来；
> 同时回答一个具体问题：**"以 multi-agent（共同分工商讨）角度叙述"是否站得住**。
>
> **核心结论先讲**：
> 1. **以"多智能体协作"为叙事主线，撞车风险很大**（CascadeDebate、DOWN、EdgeJury 已占位）；
> 2. **以"basin"为叙事主线也撞车了**（Cherukuri & Varshney, 2604.04743 已占位）；
> 3. **目前真正属于我们的、还没被发的**只有一个 narrow 的点：**用跨模型 basin 几何（不是答案一致性）做特征 → 选择性预测的 anchor**。
> 4. 这个点单独投顶会**不够厚**，需要一个明确的 methodological hook（最有戏的是 **teacher-free distillation**）才能撑起一篇主会论文。

---

## 一、过去 12 个月这块发生了什么（按和我们方法的相似度排序）

下面所有 arxiv ID 我都通过网络验证过；2604.* 是 2026 年 4 月，属于 4 周前到 1 个月前的工作。

### 1.1 直接相关：basin / cluster / geometric uncertainty

| 论文 | 时间 | 占了我们什么 |
|---|---|---|
| **Hallucination Basins** (Cherukuri & Varshney, [arXiv:2604.04743](https://arxiv.org/abs/2604.04743)) | 2026-04 | **"basin" 这个词已经被注册了**。他们用 latent-state 动力系统刻画"factoid → 单 basin 坍塌 / misconception → 多 basin"。他们也做了 geometry-aware steering。我们再讲 "answer basin" 必须解释和他们的关系。 |
| Geometric Uncertainty ([arXiv:2509.13813](https://arxiv.org/abs/2509.13813)) | 2025-09 | 用 archetypal analysis + convex hull volume 做 hallucination 检测。"几何形状作为不确定性"已经发了。 |
| Semantic Volume ([arXiv:2502.21239](https://arxiv.org/abs/2502.21239)) | 2025-02 | embedding gram 矩阵的 determinant 当不确定性。Cluster geometry 不是新角度。 |
| Semantic Entropy Probes (NeurIPS 2024) | 2024-06 | 把 Farquhar 2024 的 semantic entropy 蒸成一个 hidden-state 探针。我们的 self-feature 和它高度重叠。 |
| Beyond Semantic Entropy (ACL 2025) | 2025-07 | 加上 inter/intra cluster similarity，扩展了 semantic entropy。 |

**结论**：cluster / basin / geometric 这一系列特征本身**不是新东西**，最近 2 年已经被多次发表。

### 1.2 直接相关：cross-model 一致性 / anchoring

| 论文 | 时间 | 占了我们什么 |
|---|---|---|
| **FINCH-ZK** (EMNLP 2025 Industry) | 2025-10 | 黑盒 fine-grained cross-model consistency，hallucination F1 +6–39%。**就是用另一个 LLM 当锚点**。 |
| Teaming LLMs (arXiv:2510.19507) | 2025-10 | 多个不同来源 LLM 组队检测 hallucination，"consortium consistency"。 |
| Two-stage cross-consistency verifier (arXiv:2502.15845) | 2025-02 | self-consistency + cross-consistency 动态切换。 |

**结论**：拿 GPT-OSS 当 teacher 给 Qwen 提供 anchor 信号——**这件事本身已经不新**。

### 1.3 直接相关：selective prediction / abstention

| 论文 | 时间 | 占了我们什么 |
|---|---|---|
| **SelectLLM** (NeurIPS 2025 / ICLR 2026 重投) | 2025 | 把 selective prediction 集成到 fine-tuning，TriviaQA 等基准。**和我们用的数据集都重叠**。 |
| **UniCR** ([arXiv:2509.01455](https://arxiv.org/abs/2509.01455)) | 2025-09 | 统一框架，融合多种异构不确定性信号 + 风险控制 refusal + 共形保证。**这就是我们想做的事更通用版本**。 |
| I-CALM ([arXiv:2604.03904](https://www.arxiv.org/abs/2604.03904)) | 2026-04 | prompt-based abstention。 |
| Dynamic Abstention (ICLR 2026 提交中) | 2025 | mid-generation abstention with RL。 |
| **AbstentionBench** ([arXiv:2506.09038](https://arxiv.org/pdf/2506.09038)) | 2025-06 | **Reasoning fine-tuning average degrades abstention by 24%**——这条 finding 对我们其实是好消息（见 §3）。 |

**结论**：selective prediction + risk-controlled refusal **作为顶层框架已经不新**。

### 1.4 最危险：multi-agent + cascade + 信心触发

这是你想转向的方向，但**最近 6 个月这块刚被密集占完**：

| 论文 | 时间 | 你提的"分工商讨"框架被占的程度 |
|---|---|---|
| **CascadeDebate** ([arXiv:2604.12262](https://arxiv.org/abs/2604.12262)) | 2026-04 | **"信心驱动 + 在 cascade 升级边界插入多 agent 商讨"——这正是你的提法**。LG / Sogang / 首尔大学合作。5 个 benchmark 上 +26.75%，online threshold optimizer。 |
| **DOWN** "Debate Only When Necessary" ([arXiv:2504.05047](http://arxiv.org/abs/2504.05047v2)) | 2025-04 | 信心分数低才触发 debate，效率提升 6 倍。 |
| **EdgeJury** ([arXiv:2601.00850](https://arxiv.org/abs/2601.00850)) | 2026-01 | 3B–8B 小模型组成"陪审团"角色化生成 + 匿名互审 + 主席整合。TruthfulQA +21.4%。**就是"分工商讨"**。 |
| **PassiveQA** ([arXiv:2604.04565](https://www.arxiv.org/pdf/2604.04565)) | 2026-04 | **3 action：Answer / Ask / Abstain**——和我们的 3-action routing 完全同构。 |
| Dynamic Role Assignment ([arXiv:2601.17152](https://arxiv.org/pdf/2601.17152v1)) | 2026-01 | meta-debate，给不同模型分配不同角色，+74.8% over uniform。 |
| Decentralized debate (Agora-Opt, [arXiv:2604.25847](https://arxiv.org/html/2604.25847)) | 2026-04 | 分布式 debate + memory。 |

**坦白讲**：你说"以 multi-agent（共同分工商讨）角度叙述"——
这正是 **CascadeDebate + DOWN + EdgeJury** 三家 1 个月前到 4 个月前刚刚做完的事。
**reviewer 一定会拿这三篇来比，你必须 baseline against 至少这三篇**，而且要在它们的方法上跑赢。

---

## 二、我们手上还剩什么真正属于自己的东西

把上面的对手地图叠起来后，**唯一还没被人精确占的点**只有：

> **"用跨模型 basin 几何（不是答案一致性）作为特征 → 投喂 selective predictor"**

具体来说，下面这些是我们独有的细节：

1. **跨模型 basin 对齐特征**：FINCH-ZK / 一致性方法只看"两个模型答的是不是同一句话"；我们看的是"Qwen 的 8 个候选聚成几个 basin？这些 basin 和 GPT-OSS 的 basin 几何怎么对齐？top-1 basin 是否在 teacher support 下？" 这是**特征工程层面的差异**。
2. **特征数量与系统化对比**：相关工作通常用 1–5 个不确定性特征。我们用了 ~150 个，**做了系统的 subset 消融**（self-only / teacher-only / all × {logreg, mlp} × {5-fold CV, cross-seed}）。这是个小但具体的 empirical 贡献。
3. **明确的负面对照实验**：我们花了大功夫修 Plan_opus 的 6 个 bug，把 DPO/GRPO 推到代码级 clean，仍然只能 +0.5–1.4 strict accuracy。这条**精修过的负面证据**比一般论文 random baseline 更可信，恰好和 AbstentionBench (2025) "reasoning hurts abstention −24%" 的发现互为印证。
4. **TriviaQA 上 sel_acc@50% = 0.994 的 headline 数字**：在已知 selective prediction 工作里这是非常硬的数。SelectLLM、UniCR 这一档没有公布同口径数。

但实话实说：上面 4 点单独看都**只够投 workshop / findings**。要进主会，必须把这 4 点收成一个**单点突出的 methodological hook**。

---

## 三、对 "multi-agent 叙事"的具体判断

### 问题：能不能讲成"多智能体分工商讨"？

**答**：可以讲，但**单这条故事撑不起顶会**，原因如下：

1. **CascadeDebate** 已经做完"信心触发 + 多 agent 在 cascade 边界商讨"。你必须给出**和它不一样**的 mechanism，且**实验上跑赢它**。
2. **EdgeJury** 已经做完"小模型陪审团 + 角色化分工"。你的"GPT-OSS 当 teacher"反而退化为他们的 special case（teacher 做 chair, Qwen 做 juror）。
3. **PassiveQA** 已经做完 Answer / Ask / Abstain 3-action——和我们 routing table 完全同构。
4. 你目前**没有真正实现多 agent 之间的"对话/反驳"**——我们的 GPT-OSS 只是被用作离线 anchor 标注，不是 online debate。把它包装成"multi-agent collaboration"在审稿人眼里**像是 framing 修辞，不是真的算法新意**。

### 那如果真的要走 multi-agent，需要补什么？

至少做出**和 CascadeDebate 不一样的、可量化的差分**：

- 选项 A：**Geometry-conditioned debate triggering**——用我们的 basin 几何特征（top1 basin share, num basins, basin entropy）替代 CascadeDebate 的简单信心阈值，证明 trigger precision/recall 显著更高，从而 debate 调用次数更少 / 准确率更高。
- 选项 B：**Anchor-rooted debate**——让 teacher 不是"被升级时调用"，而是"在 debate 中作为外部专家被有选择地引入"，用 anchor 几何决定何时引入。这个 mechanism 没人做过。
- 选项 C：**Teacher-free distilled jury**——把 GPT-OSS 的 anchor 信号蒸馏成 Qwen 自带的一颗 confidence head + 一个 role-conditioned head，部署时**不需要任何外部 teacher**，单 8B 模型自己做 multi-agent 协作。这个有方法上的真创新，因为现在的 multi-agent 文章基本默认能调多个 API。

**风险预估**（如果只换叙事，不补实验）：

- 顶会主会（NeurIPS/ICLR/ACL）：**reject 概率 > 70%**，主要 attack 来自 CascadeDebate / EdgeJury 比对缺失。
- 顶会 Findings/Short：**reject 概率 ~50%**。
- 顶会 workshop（safety / calibration / agent）：**accept 概率 ~70%**。

---

## 四、我推荐的 4 条投稿定位（按性价比排序）

### 选项 1（最稳，推荐）：把"selective prediction with cross-model basin anchor"做扎实，不打 multi-agent 牌

**故事线**：
> 已有 selective prediction 工作（SEPs, semantic entropy, semantic volume）只看 self-introspection；
> 已有 cross-model 工作（FINCH-ZK, Teaming LLMs）只看 final-answer agreement；
> **我们提出 cross-model basin geometry features**，介于两者之间——既不依赖 self（避免 overconfident wrong），也不只看答案匹配（保留几何细节）。

**必须补的实验**：
- 至少 3 个数据集：TriviaQA + NaturalQuestions + HotpotQA（甚至加 MMLU）。
- 至少 2 个 teacher：GPT-OSS + 另一个（Llama-3-70B / Qwen2.5-72B）。
- baseline 至少 5 个：semantic entropy, SEPs, semantic volume, FINCH-ZK, 一个 logprob baseline。
- 必须跑统计显著性（多 seed，配对 t-test 或 bootstrap CI）。

**适合投**：ICLR / NeurIPS（小幅创新但实验扎实有戏）、ACL Findings（稳）、EMNLP main（中等概率）。

---

### 选项 2（最有 punch，但最累）：teacher-free distillation 主线

**故事线**：
> Cross-model anchoring 在测试时调 teacher 太贵；
> 我们把 GPT-OSS 提供的 basin anchor 信号**蒸成 Qwen 自带的一颗 lightweight confidence head**；
> 部署时单 8B 模型 + 一颗 head，**不调任何外部 model**，selective accuracy 接近 oracle teacher 版本。

**必须补的实验**：
- 训一颗 lightweight head（MLP 或几层 transformer adapter）从 Qwen 的 hidden state 直接预测 anchor 特征。
- 证明蒸馏后的 head 在 selective prediction 任务上 ≥ 用原始 anchor 特征的 80%。
- 与 SEPs（也是 head 蒸馏）正面对决。
- ablation：蒸 self-only head / 蒸 anchor-only head / 蒸 fused head。

**适合投**：NeurIPS / ICLR 主会（有 method novelty + 强实用价值，比较硬）。

**风险**：要花 1–2 个月额外的实验。

---

### 选项 3（你想要的方向，但有差分）：geometry-triggered selective debate

**故事线**：
> CascadeDebate 用信心阈值触发 debate，但信心分数本身就 miscalibrated；
> 我们用**跨模型 basin 几何**作为更鲁棒的 trigger，"何时商讨 / 商讨多少轮 / 何时升级 teacher"由几何决定；
> 在 5 个 benchmark 上节省 X% debate 调用，accuracy +Y%。

**必须补的实验**：
- 真的实现一个 multi-agent debate loop（哪怕用 Qwen 自我多轮 + GPT-OSS 抽查）。
- 直接对比 CascadeDebate / DOWN / EdgeJury（用他们 release 的 code 或重实现）。
- 三套 cost model 下都要赢。

**风险**：CascadeDebate 是 2 个月前的，你必须在它的实验框架内跑赢。reviewer 会用放大镜看 differential novelty。

**适合投**：ACL / EMNLP main，或 NeurIPS workshop。

---

### 选项 4（创新最弱、但最稳收）：把负面结果 + selective prediction 做成 position / analysis paper

**故事线**：
> 我们花一年试图用 RL / DPO / Anchor-DPO / GRPO 改 generation 来减幻觉；
> 系统化地修了 6 类工程 bug 后仍然只能 +0.5–1.4 strict accuracy；
> 同样的特征数据用 selective prediction 框架，**risk-adjusted utility 从 −0.74 → +0.51**；
> 联动 AbstentionBench 的 −24% finding，**论证 generation-shaping 这条路被信息论锁住，自省特征只能当 sensor，不能当 actuator**。

**必须补的**：把负面结果讲扎实（曲线 + 信息论分析），加 2 个数据集复现。

**适合投**：ACL/EMNLP Findings、TMLR、安全/校准 workshop。

---

## 五、对你这条问题的直接回答

> "我准备以 multi-agent 的角度去叙述（比如共同分工商讨解决问题的场景）"

我的建议是：**不要把它作为唯一叙事，至少不要这么直白地讲**。理由：

1. **撞车 3 篇近期工作**（CascadeDebate / DOWN / EdgeJury），且你目前没有 online debate 实现，audience 一看就觉得是"包装"。
2. **PassiveQA 已占了 3-action 框架**——你那张 routing table 必须明确和它差分。
3. 真要走这条，必须补 §4 选项 3 里说的实验，**而且要拿 CascadeDebate 当主 baseline 对决**，要赢。

**性价比最高的做法**：

- **主线**走选项 1（cross-model basin anchor for selective prediction），**辅助**讲选项 4（generation-shaping 失败 + selective 成功 = framing 转变的实证）。
- 如果时间允许，再加选项 2 的 teacher-free distillation 一节作为 method-level 加分项。
- multi-agent 这层只在 introduction / discussion 里**点一下**：我们的 routing table 可以**自然嵌入到 multi-agent cascade**（如 CascadeDebate）里，作为更精细的 trigger 信号；但**不要把整个 paper 框成 multi-agent paper**。

---

## 六、下一步具体要做的事（按优先级）

| 优先级 | 任务 | 估计 | 备注 |
|---|---|---|---|
| 🔴 必做 | 加 NaturalQuestions + HotpotQA 数据集 | 3–5 天 | 单数据集 = 顶会即拒 |
| 🔴 必做 | 加 1 个备用 teacher（如 Llama-3-70B 或 Qwen2.5-72B） | 2 天 | 证明结论不依赖 GPT-OSS |
| 🔴 必做 | baseline 比对：semantic entropy / SEPs / semantic volume / FINCH-ZK | 5 天 | 不和它们比就是空白论文 |
| 🟡 强烈推荐 | 训一颗 lightweight anchor head 做 teacher-free 部署（选项 2） | 7–10 天 | 主会的 method punch |
| 🟡 强烈推荐 | 多 seed + bootstrap CI 重做所有数字 | 1 天 | 现在 2 seed 不够 |
| 🟢 可选 | 真实跑一次 vs CascadeDebate 的 head-to-head | 5 天 | 若坚持走 multi-agent |
| 🟢 可选 | 写一节"为什么 generation shaping 失败"的 information-theoretic analysis | 3 天 | 加深选项 4 |

---

## 七、最后一句大实话

我们手上**确实有真东西**——50 万行实验、6 个被修好的 bug、−0.74 → +0.51 的 utility 跳变、AUROC 0.97 的 headline——但这些**不会自动等于一篇顶会**。当前空间里这条赛道太挤，不补 3 个数据集、不打 4–5 个强 baseline、不给一个明确的 methodological hook，**任何叙事都会被拒**。

**multi-agent 角度对你来说是个陷阱**：好讲故事但实验空间已经被前人填满。**最务实的 framing 是 selective prediction + cross-model basin anchor + teacher-free distillation**，把一年的负面经验作为引子，而不是包装成 multi-agent 叙事去和 CascadeDebate 对面碰瓷。
