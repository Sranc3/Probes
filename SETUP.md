# Setup — first-time install on a fresh GPU machine

> 如果你只想 vibe-read 论文素材，**不需要 setup**，直接读 `README.md` 推荐的文档即可。
> 这份 SETUP 是给想 reproduce 实验的协作者。

---

## 0. 硬件要求

| 实验 | 最低硬件 |
|---|---|
| Phase 3 (Qwen-7B probe) | 1× 24 GB GPU (A6000 / RTX 3090 / 4090) |
| Phase 4 cross-base (Qwen-72B) | 4× 80 GB GPU (A100/H100/H200) |
| Phase 5d 2-tier OOD (GPT-OSS-120B) | 8× 80 GB GPU |

CPU / 内存：Phase 5 dashboard 生成只要 16 GB 内存即可。

---

## 1. Conda envs（**两个**）

### Env A — `vllm`（主力）

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
cd /path/to/this/repo
pip install -r requirements.txt
```

### Env B — `deepseek_v4`（**仅**用于 GPT-OSS-120B teacher 生成）

```bash
conda create -n deepseek_v4 python=3.11 -y
conda activate deepseek_v4
pip install -r requirements-gpt-oss.txt
```

---

## 2. 数据集

放到任意目录，然后改对应脚本里的 path 常量。我们用过的目录是：

| Dataset | 来源 | 本地路径 |
|---|---|---|
| TriviaQA test | https://nlp.cs.washington.edu/triviaqa/ | `datasets/trivia_qa/processed/test.full.jsonl` |
| HotpotQA dev_distractor | https://hotpotqa.github.io/ | `datasets/HotpotQA/processed/hotpotqa.dev_distractor.context1200.jsonl` |
| NQ-Open validation | https://huggingface.co/datasets/nq_open | `datasets/nq_open/nq_open/validation-00000-of-00001.parquet` |

`option_2_teacher_free_distill/scripts/prepare_*_ood.py` 顶部有数据 path 默认值，按需修改。

---

## 3. 模型权重

从 HuggingFace Hub 拉取（需要本地至少几百 GB 磁盘）：

```bash
# Qwen 系列
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./qwen_model
huggingface-cli download Qwen/Qwen2.5-72B-Instruct --local-dir ./Qwen2.5-72B-Instruct

# Llama
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir ./Llama-3.2-3B-Instruct

# GPT-OSS-120B (~120 GB)
huggingface-cli download openai/gpt-oss-120b --local-dir ./gpt-oss-120b
```

每个 base model 在 `option_2_teacher_free_distill/scripts/run_all_models.py` 里的 `BASES` dict 都有对应路径常量，按需修改。

---

## 4. 一键 sanity check

```bash
conda activate vllm
cd option_2_teacher_free_distill
python -c "
import numpy as np, pandas as pd, torch, transformers, sklearn, matplotlib
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.device_count())
print('tf:', transformers.__version__, '  sklearn:', sklearn.__version__)
print('OK')
"

conda activate deepseek_v4
python -c "
import torch, transformers
from transformers import AutoConfig
c = AutoConfig.from_pretrained('/path/to/gpt-oss-120b')
print('GPT-OSS model_type:', c.model_type)  # should print 'gpt_oss'
print('OK')
"
```

如果两条都打印 OK，环境就配好了。然后按 `README.md` 的 "复现" 章节顺序跑实验。

---

## 5. 已知坑

### 5.1 GPT-OSS-120B kernel offline 加载

GPT-OSS 的 MXFP4 量化需要 `kernels-community/gpt-oss-triton-kernels` 这个包从 HF 下载。如果你的机器无法访问 huggingface.co：

1. 在能联网的机器上先跑一次 `transformers.AutoModelForCausalLM.from_pretrained("/path/to/gpt-oss-120b")`，让它把 kernel 缓存到 `~/.cache/huggingface/hub/kernels--kernels-community--gpt-oss-triton-kernels/snapshots/<sha>`
2. 把这个目录拷到目标机器
3. 修改 `option_2_teacher_free_distill/scripts/run_teacher_on_ood.py` 顶部的 `_KERNEL_SNAPSHOT` 常量指向你的本地路径

### 5.2 Qwen-72B device_map

`prepare_hotpotqa_ood.py` / `prepare_nq_ood.py` 默认 `device_map=args.device`（`cuda` = GPU 0）。Qwen-72B 即使能塞进单卡也很慢，**建议改成 `device_map="auto"` + 用 `accelerate` 自动分片**。

### 5.3 conda subprocess 找不到 conda

如果你用 shell wrapper（`subprocess.run(shell=True)`）调 `conda run -n vllm python ...`，可能会报 `conda: not found`，因为 sh 没有 conda 在 PATH 里。我们 `run_all_models.py` 里的解决方法是用 `sys.executable`（继承父进程的 python 解释器）。

---

## 6. 帮我快速验证 reproduction 是否成功

跑完 Phase 3 + Phase 4 + Phase 5 之后，对照下表的关键数字（来自 paper-ready snapshot）：

| 来源 | 期待值 | 容差 |
|---|---|---|
| `results/qwen7b/best_per_probe.csv` ROW probe=DCP_mlp | AUROC ≈ 0.838, layer=20, pooling=prompt_last | ±0.005 |
| `results/qwen72b/best_per_probe.csv` ROW probe=ARD_mlp | AUROC ≈ 0.842 (emergent winner) | ±0.005 |
| `results/cascade_2tier_savings.csv` ROW target_acc=0.70, probe=DCP-MLP | cost ≈ 5.93, savings = "−58%" | ±0.5 cost |
| `results/cascade_2tier_ood_summary.csv` ROW dataset=hotpotqa, probe=DCP_mlp | min_cost_to_target ≈ 14.81, saving ≈ +31% | ±2 cost |

如果数字差太多，**先怀疑随机种子**（GroupKFold 默认 seed=42）；其次怀疑 `safe_auroc` 是否切到了 sklearn（v2.0 之后是切的）。
