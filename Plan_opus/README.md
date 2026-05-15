# Plan_opus — VBPO/GRPO 修复版与对照实验

## 这是什么

针对 `Plan_gpt55` 中 VBPO（DPO 派）和 Basin-GRPO（PPO 派）长期"训不动"的问题，本目录给出 6 项代码层修正、清洁的训练 pair、修过的 PPO loss，以及在同一无泄漏 fixed-k 评测下的多 seed 对照实验。

完整结论与表格见 [`reports/COMPARISON_REPORT.md`](reports/COMPARISON_REPORT.md)。

## 关键改动 vs Plan_gpt55

| 模块 | Plan_gpt55 实现 | Plan_opus 修正 |
|---|---|---|
| DPO 序列对数似然 | mean-logprob over completion | **sum-logprob over canonical answer span** |
| GRPO PPO ratio | exp(seq_mean − seq_mean) ≈ 1.0 | **per-token ratio + per-token clip** |
| GRPO anchor reward | 硬阈值 sim ≥ 0.80 | **sigmoid(8·(sim−0.5)) 连续 reward** |
| pair 构造 | `teacher_anchor_vs_student_only` 38% 标签噪声 | **必须 chosen strict-correct & rejected strict-wrong** |
| 训练覆盖 | 98/425 题 | **86/425 题（含 13 题 teacher_rescue）** |
| 评测协议 | 单 seed | **多 seed (2026, 42)** |

## 主要结果（双 seed × 500 题 no-leak fixed-k）

| Model | sample0 | fixed_8 | F1@s0 |
|---|---|---|---|
| Base Qwen2.5-7B | 0.660 | 0.676 | 0.1822 |
| Plan_gpt55 anchor-VBPO v1 step100 | 0.660 (+0.000) | 0.666 (-0.010) | 0.1810 |
| Plan_opus VBPO step90 | 0.665 (+0.005) | 0.674 | 0.1840 |
| **Plan_opus GRPO step20** | **0.674 (+0.014)** | 0.675 | 0.1847 |

GRPO-Opus step20 是项目历史上第一个在 sample0 strict 上取得**双 seed 全部高于 base** 的模型。

## 训练信号回归正常

| Run | dev margin_delta | dev pair_acc |
|---|---|---|
| Plan_gpt55 anchor-VBPO v1 step150 | 0.002 | 0.633 |
| **Plan_opus VBPO step120** | **2.17 (×1085)** | **0.870** |

| Run | PPO clip_frac | PPO max_ratio |
|---|---|---|
| Plan_gpt55 basin-GRPO | ≈0 | 1.05 |
| **Plan_opus GRPO step80** | **0.015** | **2.35** |

## 目录速查

```
shared/                              # sum-logprob, span loss, per-token PPO 工具
vbpo_opus/
  configs/                           # 训练 + pair build + 消融 配置
  scripts/build_pairs.py             # 构造严格干净的 chosen/rejected pair
  scripts/train_vbpo_opus.py         # span-only sum-logprob DPO 训练
  runs/pairs_v1/                     # 已构造好的 pair manifest
  runs/run_<ts>_vbpo_opus_v1_*/      # 训练 checkpoint
grpo_opus/
  configs/                           # 训练配置
  scripts/train_grpo_opus.py         # per-token PPO + continuous reward + teacher injection
  runs/run_<ts>_grpo_opus_v1_*/      # 训练 checkpoint
eval/
  scripts/evaluate_no_leak.py        # 多 seed no-leak fixed-k 评测
  scripts/build_comparison_table.py  # 聚合所有 eval 输出
  results/                           # 各模型 seed_<n>/eval_summary.csv + comparison_*.csv|md
runs_logs/                           # 所有训练/评测 stdout
reports/COMPARISON_REPORT.md         # 完整报告（含根因诊断 + 实验表 + 后续建议）
```

## 复现步骤

详见报告第 8 节。

```bash
# Build pairs → train VBPO → train GRPO → eval all → aggregate
python vbpo_opus/scripts/build_pairs.py --config vbpo_opus/configs/vbpo_opus_v1_pair_build.json --output-dir vbpo_opus/runs/pairs_v1
python vbpo_opus/scripts/train_vbpo_opus.py --config vbpo_opus/configs/vbpo_opus_v1_train.json --device cuda:0
python grpo_opus/scripts/train_grpo_opus.py --config grpo_opus/configs/grpo_opus_v1_full500.json --device cuda:0
python eval/scripts/evaluate_no_leak.py --model-dir <model> --adapter-dir <ckpt> ...
python eval/scripts/build_comparison_table.py
```
