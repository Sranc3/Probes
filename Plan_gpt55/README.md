# Plan_gpt55

这是一个独立的诊断与转向工作区，用来重新审视当前 ITI-for-entropy 项目。

核心问题不是简单判断“idea 行不行”，而是拆成两层：

1. **机制发现是否成立**：熵相关表征是否真实存在、是否和特定层/状态稳定相关。
2. **落地方式是否合适**：固定方向、固定 alpha、推理时局部加向量的 ITI 是否足够稳定，能否作为部署方法。

当前结论倾向于：

- 熵相关机制是有信号的。
- 固定 ITI 向量作为最终部署控制器过于苛刻。
- 现有评估和搜索也有若干粗糙点，导致不能直接把失败归因于“idea 本身完全错误”。

## 工作区结构

- `docs/diagnosis_report_zh.md`：代码和实验链路诊断报告。
- `docs/method_reframe_zh.md`：研究叙事重构和路线转向。
- `docs/direction_event_audit_design_zh.md`：方向资产和事件审计设计。
- `configs/artifact_audit_config.json`：现有 run artifacts 审计配置。
- `configs/direction_audit_config.json`：方向/事件审计配置草案。
- `configs/experiment_matrix.json`：下一阶段实验矩阵。
- `scripts/audit_existing_artifacts.py`：从现有实验产物生成诊断摘要。
- `scripts/audit_direction_assets.py`：审计方向资产、gate 和 split 的结构性风险。
- `scripts/summarize_experiment_matrix.py`：生成实验矩阵摘要。
- `experiments/`：分阶段实验说明。
- `templates/`：后续执行实验时可复用的配置模板。
- `reports/`：脚本生成的审计输出。

## 当前主要判断

当前失败更像是一个组合问题：

- `validated`：熵和若干层表征之间存在可观测关系。
- `weak_or_inconclusive`：固定 ITI 是否能稳定优化 QA 表现，目前证据不足。
- `likely_wrong_path`：继续只在 `cand_008` 附近微调 alpha/weight，很可能收益有限。

## 建议的新定位

把 ITI 从“最终部署方法”降级为“机制探针”，并把主线转向：

- learned controller
- uncertainty-aware decoding / reranking
- distillation / SFT
- 更严格的 correctness 与 cache-aware latency 评估

## 快速运行

生成现有实验产物审计：

```bash
"/root/miniconda3/envs/jyh_0526/bin/python" \
  "/zhutingqi/song/Plan_gpt55/scripts/audit_existing_artifacts.py" \
  --config "/zhutingqi/song/Plan_gpt55/configs/artifact_audit_config.json"
```

生成方向资产审计：

```bash
"/root/miniconda3/envs/jyh_0526/bin/python" \
  "/zhutingqi/song/Plan_gpt55/scripts/audit_direction_assets.py" \
  --config "/zhutingqi/song/Plan_gpt55/configs/direction_audit_config.json"
```

生成新实验矩阵摘要：

```bash
"/root/miniconda3/envs/jyh_0526/bin/python" \
  "/zhutingqi/song/Plan_gpt55/scripts/summarize_experiment_matrix.py" \
  --matrix "/zhutingqi/song/Plan_gpt55/configs/experiment_matrix.json" \
  --output "/zhutingqi/song/Plan_gpt55/reports/experiment_matrix_summary.md"
```
