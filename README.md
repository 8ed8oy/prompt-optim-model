# 提示词优化模型

媒体提示词优化助手 — 基于 Qwen2.5-7B + LoRA 微调，将模糊的文生图/视频需求打磨为专业英文标签化提示词。

## 快速开始

```powershell
conda activate prompt-opt
cd E:\01_workspace\prompt_optimizer_model
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
├── data/                         # 训练数据
│   └── train_data.cleaned.jsonl
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
  --num-train-epochs 3 `
  --learning-rate 1e-4

# 显存不足时降低序列长度
python scripts/train.py --max-seq-length 256
```

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

---

## 7) 显存参考

| 显卡 | 推荐配置 |
|------|---------|
| 8GB（RTX 4060 Laptop 等） | `--low-vram`（Adapter 模式） |
| 12GB（RTX 3060 等） | 默认配置（约 8-9GB） |
| 16GB+ | 默认或 `--merge-lora` |
