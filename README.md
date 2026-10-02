# 提示词优化模型

媒体提示词优化助手 — 基于 Qwen2.5-7B-Instruct + QLoRA（4-bit）微调，将模糊的文生图/视频需求打磨为专业英文标签化提示词。

> 训练数据为 **DeepSeek API 合成的多轮对话数据**（V2：400 条，**非人工标注**）；模型效果**尚未经过充分评测**，不要把训练 loss 下降当作生成质量提升。详见 [docs/training_evaluation_report.md](docs/training_evaluation_report.md)。

## 快速开始

```powershell
conda activate prompt-opt
# 在本仓库根目录执行（不要再写死盘符路径）
```

### 推理（对话）

```powershell
# 8GB 笔记本显卡（推荐）
python scripts/inference.py --low-vram

# 12GB+ 显卡
python scripts/inference.py

# 或使用根目录快捷方式
python inference.py --low-vram
```

### 网页演示

```powershell
pip install gradio
python scripts/web_demo.py --low-vram
# 浏览器打开 http://127.0.0.1:7860
```

---

## 目录结构

```text
prompt_optimizer_model/
├── README.md
├── pyproject.toml
├── inference.py                  # 推理入口（wrapper → scripts/inference.py）
├── evaluate.py                   # 评测入口（wrapper → scripts/evaluate.py）
├── evaluation_results.json       # 4 条用例的规则打分结果（未记录所用 adapter）
├── data/                         # 训练数据（V2，400 条）
│   └── train_data.cleaned.jsonl
├── data.old/                     # 旧版训练数据（V1，800 条，不推荐使用）
│   └── train_data.cleaned.jsonl
├── docs/                         # 交接文档与评估报告
│   ├── PROJECT.md
│   └── training_evaluation_report.md
├── outputs/                      # 模型权重
│   └── qwen25_7b_prompt_optimizer_v2/
├── prompt/                       # 提示词模板
│   ├── inference_system_prompt.txt
│   ├── data_generation_system_prompt.txt
│   ├── evaluation_system_prompt.txt
│   └── evaluation_followup_assistant.txt
├── scripts/                      # 所有脚本
│   ├── inference.py              # 推理脚本（低显存优化）
│   ├── train.py                  # Unsloth QLoRA 训练
│   ├── web_demo.py               # Gradio 网页演示
│   ├── quick_test.py             # 模型文件完整性检查
│   ├── evaluate.py               # 规则打分评测（根目录 evaluate.py 是 wrapper）
│   ├── run_data_generation.py    # 数据生成便捷入口
│   └── data/
│       ├── generate_data.py      # 生成训练数据
│       ├── merge_clean_data.py   # 合并清洗数据
│       └── start_generate_workers.ps1  # 并行生成
└── src/                          # 业务逻辑
    ├── prompt_loader.py
    └── data_pipeline/
        ├── __init__.py
        ├── core.py
        ├── generate.py
        └── merge.py
```

---

## 1) 环境配置

```powershell
# 创建环境（推荐 Python 3.11）
conda create -n prompt-opt python=3.11 -y
conda activate prompt-opt

# 安装 PyTorch（CUDA 12.6）
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装项目依赖
pip install -U datasets trl accelerate openai unsloth gradio
# 或清华镜像
pip install -U datasets trl accelerate openai unsloth gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 2) 数据生成

> 数据来源：调用 **DeepSeek API（`deepseek-chat`）合成**多轮对话，**不是人工标注**。
> 现存数据规模：`data/train_data.cleaned.jsonl` = V2，**400 条**；`data.old/train_data.cleaned.jsonl` = V1，**800 条**。下面的 `-WorkerCount 4 -TargetSizePerWorker 300` 只是并行生成的示例命令（合计 1200 条），**并不是现存数据集的生成记录**——现存 V2 数据是 400 条（`data/train_data.worker0.jsonl` 与它字节完全相同）。
> ⚠️ 仓库当前把 `.env` 纳入了版本控制，其中含真实 API Key；请勿再提交，并尽快轮换该 Key（`.env` 应加入 `.gitignore`）。

配置 API 密钥（`.env` 文件或环境变量）：

```powershell
$env:API_KEY = "sk-xxxxxxxx"
$env:BASE_URL = "https://api.deepseek.com/v1"
$env:MODEL_NAME = "deepseek-chat"
```

### 单进程生成

```powershell
python scripts/run_data_generation.py --target-size 200 --output data --sleep 0.8
```

### 多 Worker 并行

```powershell
.\scripts\data\start_generate_workers.ps1 -WorkerCount 4 -TargetSizePerWorker 300 -OutputDir .\data
```

### 合并清洗

```powershell
python scripts/data/merge_clean_data.py --input-dir data --output data/train_data.cleaned.jsonl
```

---

## 3) 训练

```powershell
# 使用清华镜像（国内网络）
$env:HF_ENDPOINT = "https://hf-mirror.com"

python scripts/train.py `
  --train-file data/train_data.cleaned.jsonl `
  --output-dir outputs/qwen25_7b_prompt_optimizer_v2 `
  --num-train-epochs 5 `
  --learning-rate 1e-4

# 显存不足时降低序列长度
python scripts/train.py --max-seq-length 256
```

> 上面这组参数对应现存 V2 产物（`outputs/qwen25_7b_prompt_optimizer_v2/checkpoint-60/trainer_state.json`：5 epoch / 60 step / lr 峰值 1e-4）。
> ⚠️ `scripts/train.py` 自身的默认值是 `--num-train-epochs 3 --learning-rate 2e-4`，**与 V2 实际训练不一致**；直接用默认值跑复现不出 V2。

---

## 4) 推理

```powershell
# 8GB 笔记本显卡
python scripts/inference.py --low-vram

# 12GB+ 显卡  
python scripts/inference.py

# 自定义显存限制
python scripts/inference.py --max-gpu-memory "6000MiB"

# 批量推理
python scripts/inference.py --input-file test_queries.txt
```

交互命令：
- `clear` — 清空对话历史
- `quit` / `exit` — 退出

---

## 5) 网页演示

```powershell
python scripts/web_demo.py --low-vram --port 7860
```

可选参数：
- `--share` — 生成公网临时链接
- `--low-vram` — 8GB 显存优化

---

## 6) 模型切换

训练输出目录和推理 adapter 路径对应：

| 版本 | 训练输出 | 推理参数 |
|------|---------|---------|
| v2（当前） | `outputs/qwen25_7b_prompt_optimizer_v2` | `--adapter-path outputs/qwen25_7b_prompt_optimizer_v2` |
| v1（旧版，不推荐） | `outputs/qwen25_7b_prompt_optimizer` | `--adapter-path outputs/qwen25_7b_prompt_optimizer` |

> 两版口径不要混用：**V1** = 800 条数据（`data.old/`）/ 3 epoch / 300 step（另有 2026-03-19 那次 198 步、3 epoch 的训练）；**V2** = 400 条数据（`data/`）/ 5 epoch / 60 step。
> 根目录 `evaluation_results.json` 只是 4 条用例的规则打分，且 `scripts/evaluate.py` 默认指向 **V1**、结果文件未记录所用 adapter 与日期，**不能当作 V2 的效果证据**。详见 [docs/training_evaluation_report.md](docs/training_evaluation_report.md)。

---

## 7) 显存参考

| 显卡 | 推荐配置 |
|------|---------|
| 8GB（RTX 4060 Laptop 等） | `--low-vram`（Adapter 模式，默认显存上限 7500MiB） |
| 12GB（RTX 3060 等） | 默认配置（训练实测：RTX 3060 12GB / 35 分钟） |
| 16GB+ | 默认或 `--merge-lora` |
