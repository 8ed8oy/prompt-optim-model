# 面向媒体领域的提示词优化模型（DeepSeek 造数 + Qwen2.5-7B Unsloth QLoRA）

本项目推荐流程：

1. 用 `generate_data.py`（DeepSeek API）生成多轮对话训练样本
2. 用 `merge_clean_data.py` 合并并清洗多 worker 分片
3. 用 `Qwen2.5-7B-train.py` 在 `Qwen/Qwen2.5-7B-Instruct` 上做 Unsloth QLoRA 微调
4. 用 `inference.py` 加载 LoRA 做多轮对话推理

---

## 1) 环境准备

```powershell
Set-Location E:\01_workspace\prompt_optimizer_model

# 创建环境（unsloth 要求 Python >= 3.10，推荐 3.11）
conda create -n prompt-opt python=3.11 -y
conda activate prompt-opt

python -m pip install -U pip setuptools wheel
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -U datasets trl accelerate openai -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 unsloth（国内镜像）
pip install -U unsloth -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **注意**：`unsloth` 依赖 Python **3.10+**，在 Python 3.9 及以下会抛出 `TypeError: unsupported operand type(s) for |`，请确保 conda 环境使用 Python 3.11。

---

## 2) 配置数据生成 API（DeepSeek）

`generate_data.py` 默认就是 DeepSeek：

- `MODEL_NAME=deepseek-chat`
- `BASE_URL=https://api.deepseek.com/v1`

```powershell
$env:API_KEY = "你的DeepSeek密钥"
$env:BASE_URL = "https://api.deepseek.com/v1"
$env:MODEL_NAME = "deepseek-chat"
```

---

## 3) 生成数据

### 单进程

```powershell
python .\generate_data.py --target-size 1000 --output .\train_data.jsonl --temperature 0.9
```

### 多进程并行

```powershell
.\start_generate_workers.ps1 -WorkerCount 4 -TargetSizePerWorker 300 -OutputDir .\data
```

生成完成后合并清洗：

```powershell
python .\merge_clean_data.py --input-dir .\data --output .\train_data.cleaned.jsonl
```

---

## 4) 训练 Qwen2.5-7B（Unsloth QLoRA）

```powershell
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

## 5) 推理测试

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

## 6) 关键默认值（当前代码）

- 数据生成：DeepSeek（`deepseek-chat`）
- 训练模型：`Qwen/Qwen2.5-7B-Instruct`（Unsloth 4-bit 加载）
- 训练输出目录：`outputs/qwen25_7b_prompt_optimizer`
- 推理默认 adapter：`outputs/qwen25_7b_prompt_optimizer`
