# 面向媒体领域的提示词优化模型

这是一个偏课程作业性质的轻量项目，结构尽量保持扁平：

1. `src/` 放可复用的业务逻辑
2. `scripts/` 放批处理和辅助脚本
3. 根目录保留少量直接运行的入口脚本
4. `prompt/` 放可单独修改的提示词文本

推荐流程：

1. 用 `scripts/data/generate_data.py` 生成多轮对话训练样本
2. 用 `scripts/data/merge_clean_data.py` 合并并清洗多 worker 分片
3. 用 `Qwen2.5-7B-train.py` 做 Unsloth QLoRA 微调
4. 用 `inference.py` 做多轮对话推理

---

## ⚡ 快速开始（推荐）

### 全流程由两个脚本组成

数据生成和模型训练建议分开执行：

#### 1. 数据生成脚本

优先使用多 worker 并行生成：

```powershell
conda activate prompt-opt
.\scripts\data\start_generate_workers.ps1 -WorkerCount 4 -TargetSizePerWorker 300 -OutputDir .\data
```

如果只想单进程生成，也可以直接运行：

```powershell
python .\scripts\data\generate_data.py --target-size 1000 --output .\train_data.jsonl --temperature 0.9
```

生成完成后，再合并清洗：

```powershell
python .\scripts\data\merge_clean_data.py --input-dir .\data --output .\train_data.cleaned.jsonl
```

#### 2. 训练脚本

```powershell
conda activate prompt-opt
.\train_pipeline.ps1
```

如果你只想跑训练，也可以直接调用训练入口：

```powershell
python .\Qwen2.5-7B-train.py `
  --train-file .\train_data.cleaned.jsonl `
  --output-dir .\outputs\qwen25_7b_prompt_optimizer
```

训练数据位置就是这里指定的 `--train-file`。默认值写在 [Qwen2.5-7B-train.py](Qwen2.5-7B-train.py#L25) 里，当前默认是 `train_data.cleaned.jsonl`；如果你想用别的数据，只要把这个参数改成你的路径即可。

`train_pipeline.ps1` 仍然可以作为一键流程使用，但是仍然建议将数据生成和训练分开。它会自动走完：
- ✅ 环境检查
- ✅ API 密钥配置（自动提示输入）
- ✅ 多 worker 并行数据生成
- ✅ 数据合并清理
- ✅ Qwen2.5-7B 模型训练
- ✅ 推理测试（可选）

**也支持直接指定参数**（跳过交互）：
```powershell
.\train_pipeline.ps1 -ApiKey "sk-xxxx" -WorkerCount 4 -TargetSizePerWorker 300
```

**或跳过数据生成，直接用已有数据训练**：
```powershell
.\train_pipeline.ps1 -SkipDataGen
```

---

## 1) 环境配置（推荐先做）

这个项目可以借助 `pyproject.toml` 做“半一键”安装：先准备 Python 和 PyTorch，再用 `pip install -e .[dev]` 安装项目本身和开发工具。

```powershell
# 创建环境（推荐 Python 3.11）
conda create -n prompt-opt python=3.11 -y
conda activate prompt-opt

# 先安装与 CUDA 匹配的 PyTorch
python -m pip install -U pip setuptools wheel
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 再安装项目依赖和开发工具
pip install -e .[dev]
```

> 如果你的机器已经有可用的 PyTorch/CUDA 环境，也可以直接执行 `pip install -e .[dev]`。
>
> `unsloth` 对 Python 版本比较敏感，推荐使用 Python 3.11。

## 2) 目录结构

```text
prompt_optimizer_model/
├── README.md
├── pyproject.toml
├── train_pipeline.ps1
├── Qwen2.5-7B-train.py
├── inference.py
├── evaluate.py
├── quick_test.py
├── data/
├── outputs/
├── prompt/
├── scripts/
│   └── data/
│       ├── generate_data.py
│       ├── merge_clean_data.py
│       └── start_generate_workers.ps1
└── src/
  └── data_pipeline/
    ├── __init__.py
    ├── core.py
    ├── generate.py
    └── merge.py
```

## 3) API 在哪里 / 入口脚本在哪

- DeepSeek API 调用在 `src/data_pipeline/generate.py`
- 数据处理核心逻辑在 `src/data_pipeline/core.py`
- 薄入口脚本在 `scripts/data/generate_data.py` 和 `scripts/data/merge_clean_data.py`
- 训练入口在 `Qwen2.5-7B-train.py`
- 推理入口在 `inference.py`
- 评估入口在 `evaluate.py`
- 一键流程在 `train_pipeline.ps1`

## 4) 环境准备

```powershell
Set-Location E:\01_workspace\prompt_optimizer_model
conda activate prompt-opt
```

如果你想先手动安装而不是走 `pip install -e .[dev]`，也可以用下面这组命令：

```powershell
python -m pip install -U pip setuptools wheel
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -U datasets trl accelerate openai unsloth -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **注意**：`unsloth` 依赖 Python **3.10+**，在 Python 3.9 及以下会抛出 `TypeError: unsupported operand type(s) for |`，请确保 conda 环境使用 Python 3.11。

---

## 5) 脚本选择指南

### 数据生成脚本

| 脚本 | 用途 | 何时使用 |
|-----|------|---------|
| **scripts/data/start_generate_workers.ps1** | 多 worker 并行生成数据 | 已有 API key，想批量生成训练样本 |
| **scripts/data/generate_data.py** | 单进程生成数据 | 调试生成逻辑或小规模试跑 |

### 训练脚本

| 脚本 | 用途 | 何时使用 |
|-----|------|---------|
| **Qwen2.5-7B-train.py** | 仅模型训练 | 数据已准备好，只想调整训练参数 |
| **train_pipeline.ps1** | 完整一键流程 | 想把生成、合并、训练和推理串起来 |
| **inference.py** | 推理测试 | 模型训练完成后，进行多轮对话测试 |

---

## 6) 数据生成配置 API（DeepSeek）

`scripts/data/generate_data.py` 默认就是 DeepSeek：

- `MODEL_NAME=deepseek-chat`
- `BASE_URL=https://api.deepseek.com/v1`

```powershell
$env:API_KEY = "你的DeepSeek密钥"
$env:BASE_URL = "https://api.deepseek.com/v1"
$env:MODEL_NAME = "deepseek-chat"
```

---

## 7) 生成数据

### 单进程

```powershell
python .\scripts\data\generate_data.py --target-size 1000 --output .\train_data.jsonl --temperature 0.9
```

### 多进程并行

```powershell
.\scripts\data\start_generate_workers.ps1 -WorkerCount 4 -TargetSizePerWorker 300 -OutputDir .\data
```

生成完成后合并清洗：

```powershell
python .\scripts\data\merge_clean_data.py --input-dir .\data --output .\train_data.cleaned.jsonl
```

---

## 8) 训练 Qwen2.5-7B（Unsloth QLoRA）

```powershell
# 使用清华镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"

python .\Qwen2.5-7B-train.py `
  --model-name Qwen/Qwen2.5-7B-Instruct `
  --train-file .\train_data.cleaned.jsonl `
  --output-dir .\outputs\qwen25_7b_prompt_optimizer `
  --max-seq-length 384 `
  --per-device-train-batch-size 1 `
  --gradient-accumulation-steps 8 `
  --save-steps 100 `
  --num-train-epochs 3
```

脚本默认启用 Unsloth 的梯度检查点优化；如需关闭：

```powershell
python .\Qwen2.5-7B-train.py --no-use-gradient-checkpointing
```

显存不足时优先降低：

1. `--max-seq-length`（512 -> 384 -> 256）
2. `--gradient-accumulation-steps`（16 -> 8）

---

## 9) 推理测试

```powershell
python .\inference.py `
  --base-model Qwen/Qwen2.5-7B-Instruct `
  --adapter-path .\outputs\qwen25_7b_prompt_optimizer `
  --max-new-tokens 384
```

可用命令：

- `clear` 清空历史
- `quit` / `exit` 退出

---

## 10) 关键默认值（当前代码）

- 数据生成：DeepSeek（`deepseek-chat`）
- 训练模型：`Qwen/Qwen2.5-7B-Instruct`（Unsloth 4-bit 加载）
- 训练输出目录：`outputs/qwen25_7b_prompt_optimizer`
- 推理默认 adapter：`outputs/qwen25_7b_prompt_optimizer`

## 11) TODO

### 已知问题

1. `clear` 现在还不能真正清空历史记录
2. 存在死循环输出问题