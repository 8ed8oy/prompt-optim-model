# 面向媒体领域的提示词优化模型（Qwen2.5-7B + QLoRA）一键运行说明

本项目目标：微调一个可进行**多轮追问**的提示词优化模型，将用户模糊意图转为专业级文生图/视频 Prompt。  
包含 3 个脚本：

- `generate_data.py`：调用强模型 API 合成多轮对话训练数据（支持断点续传）
- `train.py`：在 12GB 显存下进行 QLoRA 微调（支持 checkpoint 自动恢复）
- `inference.py`：加载 LoRA 后本地终端多轮对话测试（流式输出）

---

## 0. 项目结构

```text
prompt_optimizer_model/
├─ generate_data.py
├─ train.py
├─ inference.py
└─ README.md
```

---

## 1. 环境要求

- Python: `3.9`
- GPU: `12GB` 显存（建议 NVIDIA）
- OS: Windows（本说明按 PowerShell 给出）
- 建议 CUDA 驱动已正确安装

> 说明：`bf16` 仅在硬件支持时自动启用，否则脚本会自动回退到 `fp16`。

---

## 2. 一次性安装依赖

在项目根目录打开 PowerShell，执行：

```powershell
# 进入项目目录
Set-Location E:\01_workspace\prompt_optimizer_model

# （可选）创建并激活 conda 环境
# conda create -n prompt-opt python=3.9 -y
# conda activate prompt-opt

# 升级基础工具
python -m pip install -U pip setuptools wheel

# 安装核心依赖
pip install -U torch transformers datasets peft trl bitsandbytes accelerate openai
```

如果你使用国内镜像，可自行追加 `-i` 源。

---

## 3. 配置 API（用于合成数据）

`generate_data.py` 读取以下环境变量：

- `API_KEY`（必填）
- `BASE_URL`（可选，OpenAI 兼容网关）
- `MODEL_NAME`（可选，默认 `gpt-4o-mini`）

PowerShell 示例：

```powershell
$env:API_KEY = "你的API密钥"
$env:BASE_URL = "https://api.openai.com/v1"
$env:MODEL_NAME = "gpt-4o"
```

如使用其他兼容服务（例如 Qwen-Max 网关），只需替换 `BASE_URL` 与 `MODEL_NAME`。

---

## 4. 一键全流程（推荐直接复制执行）

> 下面命令按顺序跑完：生成数据 -> 训练 -> 推理。

```powershell
Set-Location E:\01_workspace\prompt_optimizer_model

# 1) 生成 1000 条训练数据（可断点续传）
python .\generate_data.py --target-size 1000 --output train_data.jsonl --temperature 0.9

# 2) QLoRA 微调（12GB 显存保守配置）
python .\train.py `
  --model-name Qwen/Qwen2.5-1.5B-Instruct `
  --train-file .\train_data.jsonl `
  --output-dir .\outputs\qwen25_prompt_optimizer `
  --max-seq-length 1024 `
  --per-device-train-batch-size 2 `
  --gradient-accumulation-steps 8 `
  --save-steps 100 `
  --num-train-epochs 3

# 3) 本地多轮对话推理（流式输出）
python .\inference.py `
  --base-model Qwen/Qwen2.5-1.5B-Instruct `
  --adapter-path .\outputs\qwen25_prompt_optimizer `
  --max-new-tokens 384
```

---

## 5. 各脚本详细用法

### 5.1 生成数据 `generate_data.py`

基础用法：

```powershell
python .\generate_data.py --target-size 1000 --output train_data.jsonl
```

常用参数：

- `--target-size`：目标样本数（默认 `1000`）
- `--output`：输出 JSONL 路径（默认 `train_data.jsonl`）
- `--model`：API 模型名（默认读 `MODEL_NAME`）
- `--temperature`：采样温度（默认 `0.9`）
- `--max-retries`：单样本最大重试次数（默认 `6`）

断点续传机制：

- 若 `train_data.jsonl` 已存在，脚本会先读取已有行数，然后继续补齐到目标数。
- 中途中断可直接重跑同一命令。

---

### 5.2 训练 `train.py`

基础用法：

```powershell
python .\train.py --model-name Qwen/Qwen2.5-1.5B-Instruct --train-file .\train_data.jsonl --output-dir .\outputs\qwen25_prompt_optimizer
```

12GB 显存推荐参数：

- `--max-seq-length 1024`
- `--per-device-train-batch-size 2`
- `--gradient-accumulation-steps 8`
- `--save-steps 100`

如果仍显存不足，按优先级调整：

1. 把 `--max-seq-length` 从 `1024` 降到 `768` 或 `512`
2. 把 `--gradient-accumulation-steps` 从 `8` 降到 `4`
3. 只在必要时把 batch 从 `2` 降到 `1`

Checkpoint 自动恢复：

- 脚本会自动检测 `output_dir/checkpoint-*` 的最新步数。
- 存在 checkpoint 时自动 resume；不存在则从头训练。

---

### 5.3 推理 `inference.py`

基础用法：

```powershell
python .\inference.py --base-model Qwen/Qwen2.5-1.5B-Instruct --adapter-path .\outputs\qwen25_prompt_optimizer
```

交互命令：

- 输入普通文本：继续对话
- 输入 `clear`：清空历史
- 输入 `quit` 或 `exit`：退出程序

LoRA 合并：

- 默认尝试合并 LoRA
- 如需关闭合并（更稳妥）：

```powershell
python .\inference.py --base-model Qwen/Qwen2.5-1.5B-Instruct --adapter-path .\outputs\qwen25_prompt_optimizer --no-merge-lora
```

---

## 6. 输出文件说明

- `train_data.jsonl`：合成训练数据，每行一个 JSON
- `outputs/qwen25_prompt_optimizer/`：LoRA adapter、tokenizer、checkpoint

---

## 7. 常见问题（FAQ）

### Q1: 报错 `无法解析导入 openai`

执行：

```powershell
pip install -U openai
```

并确保 VS Code 选择的是同一个 Python 环境。

### Q2: 训练时报 CUDA OOM

优先降低 `--max-seq-length`，其次降低 `--gradient-accumulation-steps`，最后再降 `batch size`。

### Q3: 训练意外中断了怎么办？

直接重复执行原训练命令，`train.py` 会自动从最新 checkpoint 恢复。

### Q4: 推理输出不够“追问式”

建议：

1. 提高训练数据中“助手追问”占比
2. 保证最终消息都带 `【最终优化提示词】`
3. 推理时把 `temperature` 设为 `0.6~0.8`

---

## 8. 最小验证清单

跑完后按以下检查：

1. `train_data.jsonl` 行数达到目标（如 1000）
2. `outputs/qwen25_prompt_optimizer` 下存在 adapter 与 checkpoint 目录
3. `inference.py` 中模型可进行至少 2 轮追问后输出优化 Prompt

---

## 9. 参考命令速查

```powershell
# 数据生成
python .\generate_data.py --target-size 1000 --output train_data.jsonl

# 训练
python .\train.py --model-name Qwen/Qwen2.5-1.5B-Instruct --train-file .\train_data.jsonl --output-dir .\outputs\qwen25_prompt_optimizer --max-seq-length 1024 --per-device-train-batch-size 2 --gradient-accumulation-steps 8 --save-steps 100

# 推理
python .\inference.py --base-model Qwen/Qwen2.5-1.5B-Instruct --adapter-path .\outputs\qwen25_prompt_optimizer
```

至此，你可以在当前目录直接完成从数据、训练到推理验证的完整闭环。
