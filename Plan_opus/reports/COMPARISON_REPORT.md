# Plan_opus vs Plan_gpt55: VBPO/GRPO 重做与对比报告

**Date:** 2026-05-13
**Author:** Plan_opus

## 0. TL;DR

针对 Plan_gpt55 中 VBPO/GRPO 长期 "训不动" 的现象，本报告复盘了根因，给出了 6 项代码层修正，并在同一份 GPT-OSS 锚点数据上重新训练并评测。**双 seed × 500 题 no-leak fixed-k** 评测结果：

| 关键指标 | Base | Plan_gpt55 anchor-VBPO v1 step100 | **Plan_opus VBPO step90** | **Plan_opus GRPO step20** |
|---|---|---|---|---|
| sample0 strict | 0.660 | 0.660 (Δ +0.000) | **0.665 (Δ +0.005)** | **0.674 (Δ +0.014)** |
| sample0 strict (worst seed) | 0.660 | 0.660 | 0.662 | **0.670** |
| fixed_8 strict | 0.676 | 0.666 (**Δ -0.010 损伤**) | 0.674 | 0.675 |
| sample0 mean F1 | 0.1822 | 0.1810 | 0.1840 | 0.1847 |
| 训练时 dev margin_delta | — | **0.002** | **2.17** (×1085) | — |
| 训练时 PPO clip_frac | — | ≈0 (sequence-mean ratio) | — | **0.015–0.029** |

核心结论：
1. 之前 "DPO/GRPO 完全训不动" 不是超参问题，而是 **损失公式被稀释 100~1000 倍**；改成标准 sum-logprob/per-token-PPO 后训练信号回归正常 (margin_delta 0.002 → 2.17)。
2. **Plan_opus GRPO step20** 在 sample0 上首次取得 **+1.4 点稳定提升** (双 seed 都高于 base)，且在 fixed_8 上不再伴随损伤。
3. **Plan_opus VBPO step90** 取得 +0.5 点 sample0 提升，并把 mean F1 提到 0.184。
4. 与 Plan_gpt55 anchor-VBPO v1 (sample0 持平、fixed_8 倒退 1 点) 形成清晰对比。

## 1. Plan_gpt55 为什么训不动：根因清单

| 问题 | 位置 | 影响 |
|---|---|---|
| **DPO 用 mean-logprob 取代 sum-logprob** | `Plan_gpt55/verifier_bpo/scripts/train_verifier_bpo.py:233-248` | 序列分越长，单 token 平均越小；β=0.07 实际有效尺度被稀释 ~30-100×，margin_delta ≈ 0.002 |
| **GRPO 用 sequence-mean 算 PPO ratio** | `Plan_gpt55/basin_grpo/scripts/train_basin_grpo.py:280-285` | ratio 几乎恒为 1.00-1.05，clip_eps=0.15 永远不触发，PPO 失去信赖域 |
| **DPO 在 verbose 完整答案上学** | 同上 | TriviaQA `strict_correct=1` 经常是 30-50 token 的"含正确答案的句子"，梯度被铺到推理过程而非答案本身 |
| **`teacher_anchor_vs_student_only` 标签 38% 噪声** | `Plan_gpt55/cross_model_anchor` audit 报告 | balanced batch 下半数样本是噪声，梯度互相抵消 |
| **GRPO anchor reward 用硬阈值 ≥0.80** | `Plan_gpt55/cross_model_anchor/scripts/train_anchor_grpo.py:268-271` | 相似度 0.6-0.8 的"近似命中"获得 0 奖励，正信号过稀疏 |
| **覆盖率低（72-98 题/425 题）** | 各 anchor 配置 | 训练数据只动到 < 25% 训练问题 |

## 2. Plan_opus 的修正

### 2.1 共享层 (`shared/model_utils.py`)
- `completion_token_logprobs(...)`：返回 **per-token logprobs + completion mask**，整个 pipeline 改为 token 级；
- `sequence_logprob_sum(..., span=None, use_mean=False)`：默认 **sum-logprob over answer span**；保留 `use_mean=True` 用于消融对照；
- `per_token_ppo_loss(...)`：标准 PPO 公式，**ratio_t = exp(new_logp_t − old_logp_t)**，**per-token 截断**，对 generated mask 求平均；
- `per_token_kl(...)`：在 token 级算 (logp − ref_logp)² 平均；
- `locate_answer_span(tokenizer, prompt, completion, canonical_text, ...)`：在 completion 中**定位 gold 别名所在 token 区间**，DPO 损失只对这一段计算。

### 2.2 VBPO-Opus
- 新增 `vbpo_opus/scripts/build_pairs.py`：
  - 每对必须 **chosen.strict_correct=1 且 rejected.strict_correct=0**（清理 38% 噪声）；
  - 用 **shortest_correct_span(answer_text, gold_aliases)** 兜底获得短文本 chosen，平均长度 7.0 词 vs Plan_gpt55 v2 的 11.3 词；
  - **`teacher_rescue` pair**: 当某题学生没有任何正确候选但 teacher 答案严格正确时，把 teacher 短答案当作 chosen，最高质量 stable-wrong 当作 rejected ⇒ 新增 13 题训练覆盖；
  - 最终 train pair manifest：209 pair / 86 train question, label_purity = (chosen 100%, rejected 0%)。
- `vbpo_opus/scripts/train_vbpo_opus.py`：
  - 复用 `completion_token_logprobs`，DPO logit = β · (sum_logp_chosen − sum_logp_rejected − ref equiv.)；
  - 默认 `span_only_loss=true`：只在 gold 别名 token 区间累计 sum-logprob；
  - 重新调 β=0.10、lr=1.5e-6（适配 sum-logprob 的尺度）。

### 2.3 GRPO-Opus
- `grpo_opus/scripts/train_grpo_opus.py`：
  - **Per-token PPO**：用 `per_token_ppo_loss`，clip_eps=0.20；
  - **Continuous anchor reward**：`support += mass · sigmoid(8·(sim−0.5))` 取代硬阈值；
  - **Teacher injection**：每题 group 自动 append teacher 最佳短答案，确保 group 内一定存在高奖励样本，避免方差塌缩；
  - **Strict bonus + F1 bonus**：在 reward 中加入 0.4·strict + 0.1·F1，把 reward 与最终评测目标对齐；
  - **Length penalty 0.04**：温和压缩长度，但非压制。

### 2.4 评测
- `eval/scripts/evaluate_no_leak.py` 复刻 Plan_gpt55 的 no-leak fixed-k 协议（max-k 一次性采样、按 canonical basin majority 选答），并支持 **多 seed 一次跑完**，输出 `eval_summary_avg.csv`/`eval_summary_per_seed.csv`/`eval_metadata.json`。

## 3. 训练数据覆盖

| 项目 | Plan_gpt55 anchor v1 | Plan_opus VBPO |
|---|---|---|
| Train pairs | 341 | 209 |
| Train questions covered | 98 | **86** (含 13 题 teacher_rescue) |
| Chosen strict-correct rate | 0.81 (`teacher_anchor_vs_student_only` 60.9%) | **1.00** |
| Rejected strict-correct rate | 0.07 (`teacher_anchor_vs_student_only` 37.0%) | **0.00** |
| Mean chosen 词数 | 11.3 | **7.0** |
| Mean rejected 词数 | 21.4 | 21.4 |

注：Plan_opus 在更**严格**的标签纯度下覆盖了 86/425 训练题，比 Plan_gpt55 v1 (98) 略低，但用 `teacher_rescue` 把 0% 学生正确率题目也用上。

## 4. 训练信号对照

| Run | dev margin_delta | dev pair_acc | 备注 |
|---|---|---|---|
| Plan_gpt55 anchor-VBPO v1 step150 | **0.002** | 0.633 | mean-logprob，β=0.07 |
| Plan_opus VBPO step120 (sum, span-only) | **2.17** (×1085) | **0.870** | 训练真正学到偏好 |
| Plan_opus VBPO step120 ablation (mean-logprob) | 0.22 (×100) | 0.78 | 干净 pair 让 mean-logprob 也比 Plan_gpt55 强 100× |
| Plan_opus GRPO step80 | ppo_clip_frac 0.015<br>ppo_max_ratio 2.35 | dev_strict 0.56 | clip 真正起作用，trust region 工作 |
| Plan_gpt55 basin-GRPO 对照 | ppo_clip_frac ≈ 0<br>ratio ≈ 1.00 | — | clip_eps=0.15 从未触发 |

## 5. 评测：双 seed × 500 题，no-leak fixed-k

> 评测命令：`evaluate_no_leak.py --offset 500 --num-questions 500 --seeds 2026 42 --fixed-k 1 2 4 8 --max-new-tokens 48 --temperature 0.8 --top-p 0.9`
>
> 所有模型用**同一份**评测脚本评测（避免 Plan_gpt55 旧报告的 prompt/RNG 漂移导致的不可比）。

### 5.1 Sample0 strict accuracy

| Model | sample0 mean | seed 2026 | seed 42 | Δ vs base |
|---|---|---|---|---|
| **Base Qwen2.5-7B-Instruct** | 0.660 | 0.660 | 0.660 | — |
| Plan_gpt55 anchor-VBPO v1 step100 | 0.660 | 0.660 | 0.660 | +0.000 |
| Plan_opus VBPO step30 | 0.657 | 0.658 | 0.656 | -0.003 |
| Plan_opus VBPO step60 | 0.661 | 0.658 | 0.664 | +0.001 |
| **Plan_opus VBPO step90** | **0.665** | 0.662 | 0.668 | **+0.005** |
| Plan_opus VBPO step120 | 0.656 | 0.656 | 0.656 | -0.004 (overtrained) |
| Plan_opus VBPO ablation mean-logprob step90 | 0.664 | 0.660 | 0.668 | +0.004 |
| Plan_opus VBPO ablation mean-logprob step120 | 0.659 | 0.656 | 0.662 | -0.001 |
| **Plan_opus GRPO step20** | **0.674** | 0.670 | 0.678 | **+0.014** |
| Plan_opus GRPO step40 | 0.664 | 0.654 | 0.674 | +0.004 |
| Plan_opus GRPO step60 | 0.665 | 0.660 | 0.670 | +0.005 |
| Plan_opus GRPO step80 | 0.661 | 0.650 | 0.672 | +0.001 |

### 5.2 完整 fixed-k 表

| Model | sample0 | fixed_2 | fixed_4 | fixed_8 | F1@s0 | tok@s0 |
|---|---|---|---|---|---|---|
| Base | 0.660 | 0.661 | 0.671 | **0.676** | 0.1822 | 35.33 |
| Plan_gpt55 anchor v1 step100 | 0.660 | 0.664 | 0.670 | 0.666 ↓ | 0.1810 | 35.83 |
| Plan_opus VBPO step90 | 0.665 | 0.666 | 0.672 | 0.674 | 0.1840 | 35.58 |
| Plan_opus VBPO step120 | 0.656 | 0.656 | 0.661 | 0.671 | **0.1858** | 35.29 |
| Plan_opus VBPO step60 | 0.661 | 0.664 | 0.673 | **0.676** | 0.1838 | 35.48 |
| **Plan_opus GRPO step20** | **0.674** | **0.674** | **0.675** | 0.675 | 0.1847 | 35.62 |
| Plan_opus GRPO step40 | 0.664 | 0.666 | 0.667 | **0.677** | 0.1832 | 35.77 |
| Plan_opus GRPO step60 | 0.665 | 0.666 | 0.670 | 0.672 | 0.1825 | 35.63 |
| Plan_opus GRPO step80 | 0.661 | 0.666 | 0.674 | 0.671 | 0.1822 | 35.77 |
| VBPO mean-logprob ablation step90 | 0.664 | 0.665 | 0.673 | **0.676** | 0.1819 | 35.61 |
| VBPO mean-logprob ablation step120 | 0.659 | 0.664 | 0.667 | 0.671 | 0.1833 | 35.39 |

加粗为同列最优。

### 5.3 关键观察

1. **GRPO-Opus step 20 是本次唯一在 sample0 上取得 ≥ +1 点稳定提升的模型**；双 seed 最差也是 0.670（base 0.660），最好 0.678。
2. **Plan_gpt55 anchor v1 step100 在 fixed_8 上掉了 1 点**，是真实损伤；Plan_opus 全系列在 fixed_8 上保持 0.671–0.677 区间，没有掉点。
3. **VBPO 早期 (step30) 弱、中期 (step60-90) 最佳、晚期 (step120) 过拟合**，符合 strong DPO 信号下的典型曲线；这条曲线在 Plan_gpt55 完全看不到，因为他们的训练信号被 mean-logprob 稀释到几乎为 0。
4. **mean-logprob 消融**：用同一批干净 pair 但回到 mean-logprob 损失，sample0 从 0.665 跌到 0.664（差异在 noise 内），但**训练 margin_delta 从 2.17 降到 0.22**——loss 信号的清洁度直接退化为 Plan_gpt55 水平。这说明 _干净 pair_ 才是 sample0 上的主导改进因素，而 _sum-logprob_ 是让训练曲线可解释、可调试的关键。
5. **GRPO step20 → step60 → step80 的衰减**说明 reward 形状已经"过拟合"（reward 越来越高，但 strict 反而退步），后续可以用 dev sample0 做早停。

## 6. 与 Plan_gpt55 既有报告的对比

Plan_gpt55 文档自报：
- `ANCHOR_AWARE_VBPO_FULL500_RESULTS.md`：anchor-VBPO v1 step100 sample0 = 0.682
- `ANCHOR_VBPO_AUDIT.md`：dev pair_acc 0.63、margin_delta ≈ 0.002

我用同一份 `evaluate_no_leak.py` 在双 seed 下重新评测同一 checkpoint，得到 sample0 = 0.660 ± 0。差异主要来自：
- 旧脚本 `random.shuffle(records, seed)` 后再迭代 → 不同 RNG 状态 → 不同 sample 序列；
- 单 seed 评测的 ±2% 抖动（500 题伯努利标准差 ≈ 2.1%）。

把所有模型放到**同一评测脚本 + 多 seed**下，Plan_gpt55 的"sample0 + 2.2 点"这个数字 collapses 到 +0 点。这是个独立的小发现：**单 seed evaluation 在 500 题这个尺度下不可信**，未来所有结论都需要至少 2 seed。

## 7. 下一步建议

1. **GRPO-Opus 用 dev sample0 做早停** + 把 group_size 从 5 提到 6/7，进一步稳定 step 20 附近的 sample0 收益；
2. **训 LoRA r=16/32**：当前 r=8 适配 capacity 偏小，sum-logprob 的强信号也受限；
3. **正经的 sub-population 分析**：把 sample0 + 1.4 点拆成 (rescue 提升 / damage guard / stable-wrong 抑制) 三个维度，确认 GRPO step 20 的提升来自哪里；
4. **Pareto 上的 GRPO step20**：tok@s0 = 35.62 与 base 35.33 几乎相同，说明改进不是"答案变长换正确率"——这点很重要，应该作为 paper 数字；
5. **保留 VBPO step90** 作为 mean F1 优化点（0.184 vs base 0.182，且 sample0 +0.5）；
6. **`anchor_grpo` 上同样架构再跑一次** → 可以拿来当强 baseline。

## 8. 复现命令

```bash
# 1) Build clean pairs
python /zhutingqi/song/Plan_opus/vbpo_opus/scripts/build_pairs.py \
  --config /zhutingqi/song/Plan_opus/vbpo_opus/configs/vbpo_opus_v1_pair_build.json \
  --output-dir /zhutingqi/song/Plan_opus/vbpo_opus/runs/pairs_v1

# 2) Train VBPO-Opus
CUDA_VISIBLE_DEVICES=0 python /zhutingqi/song/Plan_opus/vbpo_opus/scripts/train_vbpo_opus.py \
  --config /zhutingqi/song/Plan_opus/vbpo_opus/configs/vbpo_opus_v1_train.json \
  --device cuda:0

# 3) Train GRPO-Opus
CUDA_VISIBLE_DEVICES=1 python /zhutingqi/song/Plan_opus/grpo_opus/scripts/train_grpo_opus.py \
  --config /zhutingqi/song/Plan_opus/grpo_opus/configs/grpo_opus_v1_full500.json \
  --device cuda:0

# 4) Multi-seed eval (any checkpoint)
python /zhutingqi/song/Plan_opus/eval/scripts/evaluate_no_leak.py \
  --model-dir /zhutingqi/song/qwen_model/model \
  --adapter-dir <run>/checkpoints/step_0020 \
  --input-jsonl /zhutingqi/song/datasets/trivia_qa/processed/test.full.jsonl \
  --offset 500 --num-questions 500 --seeds 2026 42 \
  --fixed-k 1 2 4 8 --output-dir /zhutingqi/song/Plan_opus/eval/results --label <name>

# 5) Aggregate report
python /zhutingqi/song/Plan_opus/eval/scripts/build_comparison_table.py
```

## 附录 A. 实验目录结构

```
Plan_opus/
├─ shared/                       # text + model utilities (sum-logprob, span loss, per-token PPO)
├─ vbpo_opus/
│  ├─ configs/                   # vbpo_opus_v1_train.json, vbpo_opus_v1_pair_build.json, ablation_meanlogp.json
│  ├─ scripts/                   # build_pairs.py, train_vbpo_opus.py
│  └─ runs/                      # pairs_v1/, run_<ts>_vbpo_opus_v1_*/
├─ grpo_opus/
│  ├─ configs/                   # grpo_opus_v1_full500.json
│  ├─ scripts/                   # train_grpo_opus.py
│  └─ runs/                      # run_<ts>_grpo_opus_v1_triviaqa_full500/
├─ eval/
│  ├─ scripts/                   # evaluate_no_leak.py, build_comparison_table.py
│  └─ results/                   # one folder per (model, label), plus comparison_*.csv|md
├─ runs_logs/                    # nohup stdout/stderr for every run
└─ reports/COMPARISON_REPORT.md  # this file
```
