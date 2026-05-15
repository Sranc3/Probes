# 2-Tier Cascade — OOD 推广性测试

> Qwen2.5-7B (student) → GPT-OSS-120B (teacher) 两节点 cascade，从 TriviaQA (ID) 推广到 HotpotQA dev_distractor 和 NQ-Open validation 两个 OOD 数据集。

实验日期：2026-05-15
对应脚本：`scripts/run_teacher_on_ood.py` + `scripts/make_cascade_2tier_ood.py`
对应 dashboard：`reports/dashboard_cascade_2tier_ood.png` (3×3，ID + 2 OOD)
对应明细：`results/cascade_2tier_{hotpotqa,nq}_results.csv`、`cascade_2tier_ood_summary.csv`

---

## 1. 实验目的

之前在 TriviaQA ID 上做的两节点 cascade 拿到了 **+58% 成本节省**（在 0.70 准确率目标下）。这个数字在 ID 设定下很漂亮，但论文 reviewer 一定会问：

> "你这个方法在分布外数据集上还成立吗？probe 是不是过拟合 TriviaQA 了？"

所以这一轮我们做的事情是：把同一个 cascade 框架（Qwen-7B 用 ID 训练好的 DCP-MLP 做路由 → 不确定时 fallback 到 GPT-OSS-120B）原封不动地搬到 HotpotQA 和 NQ 两个 OOD 数据集，看：

- **probe 在 OOD 上还能不能给出有用的路由信号**
- **cascade 在 OOD 上还能不能省钱**
- **DCP-MLP 在 OOD 上是否仍然优于 SEPs**

---

## 2. 实验设置

| 维度 | 设定 |
|---|---|
| Student | Qwen2.5-7B-Instruct (ID 训练好的 DCP-MLP / SEPs-LR / SEPs-Ridge probes，**不重训**) |
| Teacher | GPT-OSS-120B (无任何 fine-tune，原版) |
| 路由方式 | 百分位阈值（escalation rate）：把 probe 分数最低的 X% 升级给 teacher |
| 成本单位 | Qwen-7B K=1 forward = 1.0；teacher cost = teacher 实测中位 latency / Qwen-7B latency |
| 题目数 | 每个 OOD 数据集 500 题，与 ID 严格 qid 对齐 |
| 真实 latency | 每条 teacher generation 的实测 wall-clock 都被记录 |

---

## 3. 头牌数字（OOD）

### 3.1 模型本身的准确率与成本

| 数据集 | Qwen-7B 单跑 | GPT-OSS-120B 单跑 | 提升幅度 | Teacher 中位 latency | 成本比 |
|---|---|---|---|---|---|
| **TriviaQA (ID)** | 0.566 | 0.764 | +19.8 pp | 4562 ms | 14.1× |
| **HotpotQA dev_distractor (OOD)** | 0.330 | 0.454 | +12.4 pp | 6993 ms | **21.6×** |
| **NQ-Open validation (OOD)** | 0.328 | 0.472 | +14.4 pp | 5639 ms | **17.4×** |

**两条第一观察：**

1. **OOD 上 teacher 自己也很弱**——GPT-OSS-120B 在 HotpotQA 上只能拿 45.4%，在 NQ 上 47.2%。这意味着 cascade 的"理论天花板"被压低了。
2. **OOD 上 teacher 的 token 数明显增加**（GPT-OSS 会做更长的 analysis），导致单次 teacher forward 的 cost ratio 从 ID 的 14.1× 涨到 OOD 的 17–22×。**OOD 让 teacher 既"更慢"又"提升幅度更小"**——这正是真实部署最难的场景。

### 3.2 Cascade 在不同准确率目标下的成本节省

#### HotpotQA（n=500，always-teacher cost = 21.6）

| 目标准确率 | DCP-MLP cascade | SEPs-LR cascade | SEPs-Ridge cascade | Qwen 单独够吗？ |
|---|---|---|---|---|
| ≥ 0.30 | **1.00 (esc=0%)** | 1.00 (esc=0%) | 1.00 (esc=0%) | ✓（Qwen 0.330 已达） |
| ≥ 0.40 | 8.77 (esc=36%) — **省 59%** | 10.28 (esc=43%) — 省 52% | 8.99 (esc=37%) — 省 58% | ✗ |
| ≥ 0.45 (≈ teacher) | 20.42 (esc=90%) — 省 5% | **16.54 (esc=72%) — 省 23%** | 21.07 (esc=93%) — 省 2% | ✗ |
| ≥ 0.50 | — | — | — | ✗（teacher 也到不了） |

#### NQ-Open（n=500，always-teacher cost = 17.4）

| 目标准确率 | DCP-MLP cascade | SEPs-LR cascade | SEPs-Ridge cascade | Qwen 单独够吗？ |
|---|---|---|---|---|
| ≥ 0.30 | **1.00 (esc=0%)** | 1.00 (esc=0%) | 1.00 (esc=0%) | ✓ |
| ≥ 0.40 | 7.61 (esc=38%) — 省 56% | 9.01 (esc=46%) — 省 48% | **5.70 (esc=27%) — 省 67%** | ✗ |
| ≥ 0.45 (≈ teacher) | 12.66 (esc=67%) — 省 27% | 13.01 (esc=69%) — 省 25% | **11.96 (esc=63%) — 省 31%** | ✗ |
| ≥ 0.50 | — | — | — | ✗（teacher 也到不了） |

---

## 4. 五个关键发现

### 发现 1 — Cascade 在 OOD 上**仍然有价值**，但价值变小

把 ID 的 "省 58%" 跟 OOD 的 "省 23–67%" 一对比：cascade 在 OOD 仍然比 always-teacher **明显便宜**。例如：

- 想拿到接近 teacher 的精度（HotpotQA 0.45 / NQ 0.45），cascade 省下 **23–31% 成本**
- 想在 Qwen 单跑（0.33）基础上加 +7pp（到 0.40），cascade 省下 **52–67% 成本**

**但是天花板被压低**——teacher 自己只能 45–47%，所以 cascade 不可能达到 0.50 以上。这是一个 *"问题本身就难"* 的限制，不是 cascade 框架的问题。

### 发现 2 — DCP-MLP **在 OOD 上不再独占头名**

ID 上 DCP-MLP 在所有目标下都是最便宜的 cascade。OOD 上情况复杂多了：

| 目标准确率 | HotpotQA 最佳 probe | NQ 最佳 probe |
|---|---|---|
| 中等（≥0.40） | DCP-MLP 微胜 | **SEPs-Ridge** 显著胜 |
| 偏高（≥0.45） | **SEPs-LR** 显著胜 | **SEPs-Ridge** 微胜 |

**诚实结论**：在 OOD 上，三个 probe 几乎打平，没有一个能 dominate。SEPs-Ridge 在某些 OOD 配置下甚至比 DCP-MLP 便宜——这跟我们 ID 上的"DCP-MLP 单调最佳"叙事不符，必须如实写进论文。

### 发现 3 — Probe 的"自信但错误"率在 OOD 上爆炸式上升 ⚠️

把 dashboard 第 3 行（Routing @ DCP escalation = 30%）按数据集横向对比：

| 数据集 | Qwen 提交时的 acc | "自信但错误" 占总题量 | "不确定但正确"（误升级） |
|---|---|---|---|
| TriviaQA (ID) | **70.3%** | 21% | 8% |
| HotpotQA (OOD) | **41.1%** | 41% | 4% |
| NQ-Open (OOD) | **40.3%** | 42% | 5% |

**这是这一轮最重要的发现**：在 OOD 上，DCP-MLP 给出"高分自信"的题里有 **40% 是错的**（vs. ID 时只有 21%）。这正是经典的 **"out-of-distribution overconfidence"** 问题——模型对自己其实不会的问题反而很有信心。

含义：

- **不能直接把 ID 训好的 probe 部署到 OOD**——必须有 calibration 修正、temperature scaling 或者 OOD detection 做前置过滤
- 如果用户把这个 cascade 拿去线上跑，**实际 teacher 救不了的"自信错误"** 会大大拉低端到端用户体验
- 这是论文必须坦诚汇报的 limitation，也是后续工作的最直接抓手

### 发现 4 — 等成本曲线出现"S 形"，意味着**升级低分题反而最有效率**

dashboard 第 2 行（accuracy vs escalation budget）显示：

- ID 上：DCP 在低 escalation rate (0–30%) 上斜率最陡，意味着"挑出最不确定的 30% 升级 → 已经能逼近 teacher"
- OOD 上：曲线变得**线性**——任何 escalation rate 带来的边际收益几乎相同

这意味着 OOD 上 probe 失去了"区分难易"的能力——它甚至无法可靠地识别出"哪些题是 Qwen 不会的"。从工程角度看，**OOD 部署需要的不再是 cascade 路由，而是 OOD detection 前置层**。

### 发现 5 — Cost ratio 在 OOD 上随 teacher token 数膨胀

| 数据集 | Teacher 中位 token | Teacher 中位 latency | Cost ratio |
|---|---|---|---|
| ID | 80 | 4562 ms | 14.1× |
| HotpotQA OOD | 121 | 6993 ms | **21.6×** |
| NQ OOD | 95 | 5639 ms | 17.4× |

GPT-OSS 在 HotpotQA 这种长上下文 + 多跳推理的任务上 token 数几乎翻倍。**这是工程部署不能忽视的事实**：teacher 的"贵"不是固定的，会随任务复杂度漂移。论文里写成本节省的时候必须把 cost ratio 当一个分布而不是常数来描述。

---

## 5. 结论 & 论文叙事建议

### 5.1 可以诚实主张的内容

1. **2-tier cascade 在 ID 和 OOD 上都比 always-teacher 便宜**——这条结论稳。在 ID 上省 58%，在 OOD 上保守省 23–31%（接近 teacher 准确率），激进省 52–67%（中等准确率目标）。
2. **OOD 上 cascade 框架本身是可推广的**——只要 teacher 比 student 强，就有路由价值。
3. **真实测得的 latency / cost ratio 比理论模型更可信**，且证明 cost ratio 会随 OOD 漂移（14× → 22×）。

### 5.2 必须如实承认的 limitations

1. **DCP-MLP 在 OOD 上失去单调优势**——三种 probe 几乎打平，需要 honest reporting，不能继续宣传 "DCP-MLP best"。
2. **Probe 的 "confident-but-wrong" 在 OOD 上从 21% 涨到 41%**——这是 OOD overconfidence 的直接证据，是当前方法的**最大局限**。
3. **OOD 上 probe 失去"挑难题"的能力**——accuracy-vs-escalation 曲线变线性，证明信号衰减。

### 5.3 论文章节里这一轮该出现在哪儿

| 论文章节 | 用什么内容 |
|---|---|
| **Main result table** | TriviaQA ID 数字（headline 58% saving） |
| **Generalisation section / OOD experiments** | 本文件全部内容；HotpotQA + NQ 双数据集 |
| **Limitations** | 发现 3（OOD overconfidence）+ 发现 2（DCP 优势消失） |
| **Future work** | "需要 OOD detection 前置层"、"需要 OOD-aware probe calibration"、"需要 Multi-tier cascade 加入 OOD 检测节点" |
| **Figure** | `dashboard_cascade_2tier_ood.png`（3×3 横向对比） |

---

## 6. 数据资产清单

| 文件 | 内容 |
|---|---|
| `runs/teacher_oss120b_hotpotqa.jsonl` | 500 条 GPT-OSS-120B HotpotQA greedy 生成（含 final/analysis、latency、token 数） |
| `runs/teacher_oss120b_nq.jsonl` | 同上，NQ-Open |
| `runs/qwen7b/{hotpotqa,nq}_ood.npz` | Qwen-7B sample0 strict_correct 与 hidden states（已存在） |
| `results/per_item_predictions_all.csv` | DCP-MLP / SEPs-LR / SEPs-Ridge 的 OOD probe 分数（已存在） |
| `results/cascade_2tier_{hotpotqa,nq}_results.csv` | 全 strategy × 全 escalation rate 的 cost/acc 表 |
| `results/cascade_2tier_{hotpotqa,nq}_savings.csv` | 各准确率目标下三 probe 的最便宜成本 |
| `results/cascade_2tier_ood_summary.csv` | 单文件汇总（推荐贴论文附录） |
| `reports/dashboard_cascade_2tier_{hotpotqa,nq}.png` | 单数据集 dashboard |
| `reports/dashboard_cascade_2tier_ood.png` | 3×3 ID + 2 OOD 横向对比（推荐贴论文 main figure） |

---

## 7. 一句话总结

> **2-tier cascade 在 OOD 上仍然能省 23–67% 的 teacher 成本，但 ID 上的 "DCP-MLP 单调最佳" 优势消失，并且 probe 出现严重的"自信但错误"问题——cascade 框架可推广，probe 本身需要 OOD-aware 的改进。**
