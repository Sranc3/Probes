# 项目历程 & 表格怎么读（一份给自己看的复盘）

> 写给自己（和合作者）的一份"白话版"总结：
> 我们从哪里来 → 走过哪些弯路 → 现在在哪里 → 那两张大表怎么看。
>
> 没有公式，只有比喻和人话。

---

## 一句话先讲完

> **我们一直在尝试"让模型不胡说八道"。前面绕了很久试图"治病"——逼模型把错答案改成对的；最后发现真正起效的是"分诊"——让一个小助手判断"医生这次有没有把握"，没把握就不答 / 转给更强的医生。同样一份特征数据，前者只能多对 0.5–1.4 道题，后者能把"敢答的那一半题"准确率从 56.6% 拉到 99.4%。**

如果你只想看结论，看完上面这段就够了。下面是过程。

---

## 二、最早的目标：让 LLM 不要"自信地胡说"

**真实问题**：拿一道事实题让 Qwen 采样 8 次，常出现两种坏情况：

- **稳定错答（stable wrong basin）**：8 次答案高度一致，但都错。模型"很笃定地"错。
- **可救场答**：8 次里只有 1–2 次是对的，但在 sample0 这一票上选了错的那个。

我们一直关心的不是"模型不会"，而是：
> **"模型其实知道，但被自己的过度自信带偏了"** — 这一类。

为此积累了几十个**candidate 特征**，包括：

- 自省特征（self）：token 熵、logprob、cluster size、basin 形状……
- 跨模型特征（teacher）：用 GPT-OSS 当锚点，看 Qwen 的候选和 GPT-OSS 的候选有多近。

---

第一波尝试：直接动模型"内部"——ITI

**思路**：找到模型隐层里"低熵方向"，推理时直接把激活往那边推一推。

**结果**：太脆。
- 推强了模型变僵硬。
- 推弱了等于没干。
- 一个固定强度治不了所有题。

**教训**：固定 intervention 不靠谱，至少需要"按题决定推不推、推多少"的 controller。

---

第二波：用统计特征做"裁判"（fixed-k geometry verifier）

**思路**：8 个候选答案的几何（聚类、熵、basin 大小）能不能预测哪个对？

**结果**：
- 单特征（如 `teacher_best_similarity`）就有 AUROC ~0.90，**signal 是真的**。
- 但当 verifier 用得好的时候，**会偷偷利用"答案就在 8 个里"这个信息**——评测协议有泄露。
- **去掉泄露后，纯粹用 verifier 选答案的提升大幅缩水**。

**教训**：特征里有 signal，但"挑答案"这件事 ceiling 早被锁住了。

---

第三波：上"重型武器"——让模型自己学会避坑

既然挑答案不够爽，那让模型**学会**避开 stable wrong basin。

### 5.1 VBPO（基于 DPO）
- 把"对的候选 vs 错的候选"喂成偏好对，让模型 prefer 对的。
- **结果：训不动**。loss 在降，但 sample0 几乎不变。

### 5.2 Basin-GRPO（基于 PPO/GRPO）
- 用更"重型"的 RL 框架，配一个 reward 函数（strict + similarity + entropy 形状）。
- **结果：照样训不动**。reward 上去了，accuracy 没动。

### 5.3 Cross-Model Anchor（引入 GPT-OSS 当 teacher）
- **anchor-VBPO**：把 GPT-OSS 的支持作为 chosen 信号。
- **anchor-GRPO**：reward 加了 teacher_similarity 项。
- **结果**：报告里写着 +0.6 / +1.6 / +1.9 三个数，**但都是单 seed**。

> 这是当时最大的困惑：我们已经发现 basin 现象、记录了几十个特征、做了 anchor，为什么训练就是不响应？是 step 数不够？lr 不对？noise 太大？

---

## 六、Plan_opus：把"训不动"的根因一次性挖出来

我新建了 `Plan_opus/`，没有改思路，只是**把代码层逐行审了一遍**。结果发现了 6 个真问题：

| # | 哪里出事 | 直白解释 |
|---|---|---|
| 1 | DPO 用的是 token **平均** logprob | 等价于把 DPO 梯度按句子长度除一遍，信号被稀释 100 倍 |
| 2 | GRPO 也是用句子平均比率 | 导致 PPO 的 `clip_epsilon` 形同虚设，KL 也错算 |
| 3 | 训练时整段 verbose completion 都参与梯度 | "答案"信号被淹没在长解释里 |
| 4 | `teacher_anchor_vs_student_only` pair 类型噪声 38% | 标签都不可靠，模型当然学不到东西 |
| 5 | anchor-GRPO 的 reward 用了硬阈值（≥0.80 才给分） | "差一点"的信号全丢，gradient 稀疏 |
| 6 | 训练对覆盖率太低（500 题里只有 78 题构出对子） | 等于在很小的子集上训练 |

**修完之后**：
- dev margin_delta 从 ~0.002 涨到 ~2.17（≈1000 倍），训练**真的在学**。
- pair 覆盖率从 78 涨到 ~280（teacher_rescue 起作用）。

**但是**——
- sample0 strict accuracy 只多了 **0.5–1.4 个点**，且双 seed 才勉强稳。
- 重新用**多 seed 评测**跑一遍 Plan_gpt55 的旧模型，发现之前报的 +0.6/+1.6/+1.9 **大部分回到 +0**——原来的"提升"是单 seed 噪声。

> **这次复盘最重要的发现**：
> bug 是真的，修也是值得的；**但即便把训练流程修干净，这条"用 RL 治病"的路，天花板就这么高**。

---

## 七、关键反思：是不是路本身错了？

修完 bug 后我盯着特征列表看了很久，意识到一件事：

> **50 个特征里，只有 `teacher_*` 一族真携带 ground truth。**
> **其余 40+ 个 self-introspection 特征只能告诉你"我不确定"，不能告诉你"答案是什么"。**

这就是 stable wrong basin 的物理意义：模型**自己照镜子永远照不出来**它哪里错——它就是稳定地错。

把这种"诊断信号"当"治疗信号"用（去改 generation distribution），等于要求一支温度计帮你退烧。**温度计当温度计用是顶级的，逼它退烧只能失败**。

那特征该怎么用？答案是：

- 不要逼模型把错答案改对。
- **用特征当 risk score**：模型答完，特征告诉你这个答案有多大概率是错的。
  - 高风险 → **弃权**（abstain）。
  - 中风险 → **调更强的模型**（call teacher）。
  - 低风险 → **直接回答**。

这就是文献里的 **selective prediction / 三动作路由（Framework A）**。

---

## 八、Plan_opus_selective：换 framing 后的结果

新框架下，模型本身**完全不动**，只在外面接一个简单的逻辑回归（约等于"小助手"）：

- 输入：每道题的 ~150 维问题级特征（聚合后的 self + teacher + 几何）。
- 输出：一个 0–1 的 confidence。
- 用 5 折交叉验证 + cross-seed 双协议都测了一遍。

**结果**：

- **AUROC 0.97**（几乎完美区分对错）。
- **置信度最高的那 50% 题，准确率从 baseline 的 56.6% → 99.4%**。
- **三动作 routing 下，平均 utility 从 −0.74（永远回答）变成 +0.51**。

> 同样的数据、同样的特征，**framing 一换，效果差一个数量级**。这就是这个项目最值得记住的一课。

---

## 九、那两张大表怎么读（重点！）

刚才贴出来的表格主要是两张：`metrics_table.md` 和 `routing_table.md`。下面把每一列翻译成大白话。

---

### 9.1 `metrics_table.md` —— 哪个 confidence 信号最准？

每行代表"用一种方法给每道题打 confidence 分"，列代表"这分打得好不好"。

#### 行（predictor）大致分两类

- **single_feature**：直接拿某一个特征当 confidence。比如 `teacher_best_similarity_sel` = 直接看"被选答案和 GPT-OSS 锚点的最大相似度"。
- **logreg / mlp**：用一组特征训一个分类器。后缀 `:self` / `:teacher` / `:all` 表示用了哪一组特征：
  - `self`：只用模型自省的特征（**部署时完全不需要调 teacher**，便宜）。
  - `teacher`：只用 GPT-OSS anchor 特征（**每道题都要先调 teacher**，贵但准）。
  - `all`：两者都用。

`regime` 列：
- `cv_5fold`：5 折交叉验证（同一道题的两个 seed 一起切，**没有 leak**）。
- `cross_seed`：训 seed 42，测 seed 43（再反过来）。

#### 列（指标）翻译

| 列 | 大白话 |
|---|---|
| `n` | 测试题数（1000，500 题×2 seed） |
| `base_acc` | 不挑题、强答全部题的 baseline 准确率（sample0=0.566） |
| **`auroc`** | 这个 confidence 区分"对/错"的能力，0.5=瞎猜，1.0=完美 |
| **`aurc`** | 风险曲线下面积，**越小越好**（少错、少弃权之间的总体 trade-off） |
| **`sel_acc@0.50`** | **只回答 confidence 最高的 50% 题时，准确率多少**（0.994 = 几乎全对） |
| `sel_acc@0.25` | 只回答最有把握的 25% 题时的准确率 |
| `sel_acc@0.75` | 回答 75% 题时的准确率（覆盖越大，准确率自然下降） |
| `brier` | 概率打得有多准（**越小越好**） |
| `ece` | 校准误差，例如"我说 80% 把握"是不是真的 80% 命中（越小越好） |

#### 怎么扫一眼看出门道

1. **看 baseline**：always answer = 56.6%，always abstain 的 AUROC = 0.5。
2. **看最强的单特征**：`teacher_best_similarity_sel` AUROC=0.906。**说明 teacher 信号已经很强**。
3. **看 logreg:teacher**：AUROC=0.969，比单特征再上一截。
4. **看 logreg:self**：AUROC=0.808。**完全不调 teacher，光靠自省特征也能在 sel_acc@25% 把准确率打到 89%**。
5. **看 mlp:all**：略低于 logreg:all，**说明非线性没赚到——更说明信号本身已经够直白，不需要复杂模型**。

#### 一句话结论

> **`logreg:all`（cross_seed）：sel_acc@50% = 0.996。意思是：让小助手挑置信度最高的 500 道题去答，几乎全对**。剩下 500 道弃权或转 teacher，就避开了胡说。

---

### 9.2 `routing_table.md` —— 给每种策略算一个"经济账"

这张表把"答 / 转 teacher / 弃权"的代价折成数字，看每种策略**人均赚多少**。

`cost_model` 三套：
- `default`：答对 +1，答错 −3，调 teacher −0.3，弃权 0。
- `lenient`：答错只赔 −1（错答不太严重的场景）。
- `strict`：答错赔 −5（医疗法律等高风险场景）。

#### 列翻译

| 列 | 大白话 |
|---|---|
| `utility_per_item` | **人均效用**，越高越好。负数=亏 |
| `answer_count` | 这种策略最终答了多少题 |
| `teacher_count` | 转给 teacher 多少题 |
| `abstain_count` | 弃权多少题 |
| `answer_accuracy` | 它**答的那部分**对了多少 |
| `teacher_accuracy` | 它**转过去的那部分**teacher 对了多少 |
| `answer_threshold` / `defer_threshold` | 触发"答"和"转 teacher"的 confidence 阈值 |

#### 三个 baseline 先看

| 策略 | default 下 utility | 解读 |
|---|---|---|
| 永远弃权 | 0.000 | 不答不亏，但啥用没有 |
| **永远回答** | **−0.736** | 错 44%，每道平均亏 0.74 分（这是没小助手时的现状） |
| 永远转 teacher | −1.040 | teacher 也不能保证全对，而且每次 −0.3 调用费 |

**对比策略**

| 策略 | utility | 怎么读 |
|---|---|---|
| **logreg_teacher (default)** | **+0.513** | 答 487 题（accuracy = 100%），转 teacher 43 题（97.7% 对），弃权 470 题。**理论最优** |
| **mlp_all (default)** | **+0.478** | 答 478 题全对，剩下全弃权（不调 teacher，最便宜的省钱方案） |
| **logreg_self (default)** | **+0.107** | **完全不调 teacher**，纯靠自省，已把人均效用从 −0.736 拉回 +0.107 |
| `best`（事后最优） | +0.500 | "上帝模式"，给定数据能拿到的天花板 |

#### 三种典型部署场景对应到表里

- **想最便宜（不能调 teacher）**：选 `logreg_self`（全程自省）或 `mlp_all`（不调 teacher 的版本）。前者把 utility 从 −0.74 拉到 +0.11；后者拉到 +0.48。
- **可以付钱调 GPT-OSS，但要省着用**：选 `logreg_teacher` 路由——只对 ~43/1000 题调 teacher，但平均效用 +0.51，几乎是事后最优。
- **错一题特别贵（strict 场景）**：看 `strict` 区，`always_answer` 直接亏到 −1.6，但 `logreg_teacher` 还能稳在 +0.51。**风险越大，小助手价值越大**。

---

## 十、整体判断：这次到底做出了什么？

| 维度 | 之前（Plan_gpt55 / Plan_opus） | 现在（Plan_opus_selective） |
|---|---|---|
| 框架 | "用特征改模型生成" | "用特征判断要不要回答" |
| 直接修改模型 | 是（DPO/GRPO） | **否**（模型完全不动） |
| 训练成本 | 几小时～一天的 GPU | 一个 logreg，**几秒** |
| sample0 strict accuracy | +0.5 ~ +1.4（边界显著） | 不动（按定义） |
| **sel_acc@50%** | 不适用 | **0.566 → 0.994** |
| 风险加权 utility | 不适用 | **−0.736 → +0.513** |
| 可解释性 | 中（DPO loss / RL reward 难看） | **高**（一个线性模型，每个特征权重都看得见） |

**关键洞见**：

1. **basin 现象是真的**，但它的物理意义是"诊断"，不是"治疗"。
2. **熵相关特征确实有用**——但只在 selective prediction 里发挥得出来。
3. **teacher 信号是 ground truth 的唯一外源**，所以 `logreg:teacher` 拿到接近最优；但 `logreg:self`（无 teacher）也已经能赚正分，**说明自省特征也不是只能当噪声**。
4. **复杂模型反而没赚到**（mlp 不优于 logreg），说明信号本身够直白，不需要 RL 这种重型方法去开采。

---

## 十一、下一步可以做什么

按"最划算→最重要"排：

1. **省 teacher 钱版本**：当 `logreg:self` 已经判定"高把握"时直接答，**只对中等区间的题再去问 teacher**。可以把 teacher 调用从 1000 次降到 ~100 次而几乎不损 utility。
2. **小模型蒸馏 confidence 头**：把 logreg:self 的 confidence 蒸成模型自带的一颗小 head（不动 base），**部署时零额外特征计算成本**。
3. **覆盖更多数据集**（NQ / HotpotQA）验证 framing A 的迁移性——当前结论建立在 TriviaQA 500 题上。
4. **做一份对外汇报材料**：把"同一份特征数据，framing 一换，−0.74 变 +0.51"作为主轴讲故事。这是这个项目最有传播力的发现。
5. （**慎做**）回头看 VBPO/GRPO：现在 framing 想清楚了，可以考虑用 selective prediction 的高置信度子集**反过来当 RL 的高质量 chosen 信号**——但优先级低，先把 framing A 做扎实。

---

## 附：术语速查（防忘）

- **basin**：把 8 个采样答案做语义聚类，每个 cluster 就是一个 basin；stable wrong basin = 大家集中到同一个错答案。
- **strict_correct**：判答案是否对的标准（exact match + alias normalize）。
- **AUROC**：confidence 区分对错的能力，0.5=瞎猜，1.0=完美。
- **AURC**：风险-覆盖率曲线下面积，越小越好。
- **sel_acc@coverage**：只回答置信度前 coverage% 的题时，准确率是多少。
- **ECE / Brier**：概率打得校准吗？（越小越好）
- **utility**：把对/错/转/弃的代价折成钱算总账，越大越好。
- **framing A vs C**：A = selective prediction（不动模型，决定要不要答）；C = generation shaping（用 DPO/GRPO 改模型）。

---

**总结一句**：
> 不是熵没用、不是 basin 没用、不是 teacher 没用——是**之前用错了地方**。把诊断信号当治疗信号用了一年多，换成"分诊"框架后，同样的数据立刻给出了第一份真正能写进 paper 的数字。
