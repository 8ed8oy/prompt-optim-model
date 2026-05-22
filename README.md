# 提示词优化模型

这是一个围绕“提示词改造 -> 数据生成 -> 数据清洗 -> 模型微调 -> 推理验证”的轻量项目，主要服务于后续接手者根据新场景持续改 prompt、批量生成训练样本并迭代模型。

仓库的设计原则很简单：

1. `prompt/` 放可以直接改的提示词文本，修改这里就能改变数据生成与评估行为。
2. `src/` 放真正的业务逻辑，包括提示词加载、数据校验、规范化与合并。
3. `scripts/` 放可直接运行的薄入口脚本，便于批量生成和清洗数据。
4. 根目录保留训练、推理和评估入口，方便从命令行直接跑完整流程。

如果只记住一件事：这个仓库的核心不是训练脚本本身，而是用强模型生成高质量多轮样本，再把这些样本整理成稳定的 SFT 数据集。

## 1. 项目定位

这个项目面向的是提示词工程和模型微调的闭环工作流，目标是把“写 prompt、测效果、造数据、训模型”串成一条可重复执行的流程。

更具体地说，它做的事情是：

1. 由外部强模型根据指定场景生成多轮对话样本。
2. 将生成结果做结构化校验、规范化和去重，得到可训练的数据集。
3. 使用 Qwen2.5-7B-Instruct 做 Unsloth QLoRA 微调。
4. 用推理和评估脚本检查新 prompt 或新数据是否真的带来收益。

这意味着后续接手者最常做的工作会是两类：

1. 修改 `prompt/` 下的提示词，适配新的场景或新的输出风格。
2. 重新生成数据、重新训练，并对比新旧版本效果。

## 2. 仓库结构

```text
prompt-optim-model/
├── README.md
├── pyproject.toml
├── Qwen2.5-7B-train.py          # 训练入口
├── inference.py                 # 推理入口
├── evaluate.py                  # 评估入口
├── quick_test.py                # 快速验证脚本
├── train_pipeline.ps1           # 一键流程脚本
├── data/                        # 生成后的分片和整理后的数据
├── outputs/                     # 训练输出、adapter、checkpoint
├── prompt/                      # 可直接修改的提示词文件
├── scripts/
│   └── data/
│       ├── generate_data.py      # 单进程生成数据
│       ├── merge_clean_data.py   # 合并并清洗分片
│       └── start_generate_workers.ps1
└── src/
    ├── prompt_loader.py         # 提示词加载
    └── data_pipeline/
        ├── core.py              # 数据规范、校验、去重、归一化
        ├── generate.py          # 生成主逻辑
        └── merge.py             # 合并主逻辑
```

你接手这个仓库时，优先看这几个位置：

1. `prompt/` 里的 prompt 文本决定“生成什么样的数据”。
2. `src/data_pipeline/core.py` 决定“什么样的数据能被接受”。
3. `scripts/data/generate_data.py` 和 `scripts/data/start_generate_workers.ps1` 决定“怎么生成数据”。
4. `Qwen2.5-7B-train.py` 决定“训练怎么跑”。

## 3. 端到端流程

推荐把整个流程拆成四步执行，不要把生成、清洗、训练、推理混在一次命令里。

### 第一步：根据场景生成原始样本

优先使用多 worker 并行生成，因为它更适合批量造数据.下面的为4线程生成共1200条数据：

```powershell
.\scripts\data\start_generate_workers.ps1 -WorkerCount 4 -TargetSizePerWorker 300 -OutputDir .\data
```

如果只想先验证生成逻辑，可以用单进程入口：

```powershell
python .\scripts\data\generate_data.py --target-size 100 --output .\data\train_data.jsonl --temperature 0.9
```

### 第二步：合并并清洗分片

多 worker 生成后，用合并脚本把分片整理成最终训练集：

```powershell
python .\scripts\data\merge_clean_data.py --input-dir .\data --output .\data\train_data.cleaned.jsonl
```

### 第三步：微调模型

```powershell
python .\Qwen2.5-7B-train.py --train-file .\data\train_data.cleaned.jsonl --output-dir .\outputs\qwen25_7b_prompt_optimizer
```

### 第四步：推理和检查

```powershell
python .\inference.py --adapter-path .\outputs\qwen25_7b_prompt_optimizer
```

如果是新场景迁移，建议至少做一次小规模推理和人工检查，再决定是否继续放大数据规模。

## 4. 提示词改造约定

这一节是接手时最重要的部分。这个仓库的核心产物不是代码，而是 `prompt/` 里的四份提示词；后续大多数修改都会发生在这里。

### 四份 prompt 分别负责什么
只建议修改1和4

1. `prompt/data_generation_system_prompt.txt`
  这是最重要的训练数据生成提示词。`src/data_pipeline/generate.py` 会直接读取它，决定强模型如何生成多轮对话样本、如何追问、如何输出最终 JSON。
2. `prompt/evaluation_system_prompt.txt`
  这是评估脚本的系统提示词。`evaluate.py` 会读取它，用来测试模型在新场景下是否仍然具备“先追问、再优化”的能力。
3. `prompt/evaluation_followup_assistant.txt`
  这是评估脚本里的“上一轮 assistant 回复”。`evaluate.py` 会把它放进多轮测试中，用来模拟用户补充信息后的第二阶段对话。
4. `prompt/inference_system_prompt.txt`
  这是推理脚本使用的系统提示词。`inference.py` 会读取它，决定训练后模型在真实交互里的语气、角色设定和回答方式。

### 改造时要遵守的目标

这个项目的目标不是让模型“一次性吐出一个成品提示词”，而是让模型学会在信息不足时继续追问用户，并最终把模糊需求整理成可直接用于文生图或视频生成的结构化英文提示词。

所以，`data_generation_system_prompt.txt` 里必须始终保留这些能力要求：

1. 对话必须是多轮的，不能只做单轮问答。训练数据应该体现“先澄清、再补充、再输出”的过程。
2. 当用户信息不足时，模型必须追问关键维度，例如发布层级、平台、视觉风格、核心元素、情绪基调、镜头语言、时间地点、禁用元素等。
3. 最后一条 assistant 消息必须是纯 JSON 对象字符串，不能带解释、前缀、后缀、Markdown 或多余文本。
4. JSON 里必须至少有 `prompt` 字段，且该字段必须是英文、标签化、逗号分隔的结构化提示词。
5. 如果场景是长故事、连续动作或多阶段情节，JSON 里还要包含 `scenes`，长度为 3 到 4，每个子场景都要保持英文结构化写法。

### 训练数据和评估数据的区别

`data_generation_system_prompt.txt` 是“造训练数据”的主提示词，目标是让模型学会产出高质量多轮样本；`evaluation_system_prompt.txt` 和 `evaluation_followup_assistant.txt` 是“测模型是否学会了这些行为”的提示词，目标是验证模型在真实交互里会不会继续追问，而不是直接跳到结果。

换句话说：

1. 训练数据关注的是“样本质量”和“分布覆盖”。
2. 评估 prompt 关注的是“模型行为是否符合预期”。
3. 推理 prompt 关注的是“上线后助手的交互风格和稳定性”。

### 版本管理和 prompt_old 归档

修改 prompt 时，不要直接覆盖旧版本而不留痕。建议的做法是：

1. 每次改动前，先把旧版 prompt 手动复制到 `prompt_old/`。
2. 文件名建议加日期或版本号，例如 `prompt_old/data_generation_system_prompt_V1.0.txt`。
3. 如果一次改动涉及多个 prompt，要把它们作为同一批版本一起归档，避免训练集、评估集和推理集的逻辑不一致。
4. 每次归档时最好在文件名或旁边的说明里写清楚这次修改的目的，例如“增强追问能力”“收紧媒体安全约束”“支持长故事分镜”。

## 5. 数据规范

这个仓库的数据不是“随便让模型吐一段文本”就能用，生成数据必须满足一组固定约束。实际生成和清洗都由脚本负责，接手者不需要手工拼 JSON，但需要理解数据长什么样、为什么能过、为什么会被丢掉。

### 数据来源

训练数据由强模型通过 API 生成。当前脚本默认使用 DeepSeek 接口，也可以替换成兼容 OpenAI Chat Completions 的其他模型，当前默认建议使用 DeepSeek V4 Flash 或同等级别的强模型来造数据。

### 生成脚本

有两种常用方式：

1. 单进程生成：`scripts/data/generate_data.py`
2. 多 worker 并行生成：`scripts/data/start_generate_workers.ps1`

前者适合调试提示词和验证小样本，后者适合正式批量造数据。

### 数据格式约定

最终可训练数据是 JSONL，每一行是一个样本，且必须包含 `messages` 字段。每个样本本质上是一段多轮对话，角色只能是 `system`、`user`、`assistant`。

训练入口只读取 `messages`，所以真正重要的是对话内容是否稳定、最后一条 assistant 是否能产出结构化的最终提示词。

### 校验和清洗规则

数据在进入训练前会经过脚本校验，主要包括这些检查：

1. 对话轮数必须在允许范围内，不能过短或过长。
2. 每条消息必须是合法 JSON 字段，且内容不能为空。
3. 最后一条 assistant 消息必须能被解析成最终提示词 JSON。
4. 最终 prompt 必须是英文结构化标签风格，不能混入过多中文。
5. 重复样本、异常重复字符、明显不合格样本会被过滤。
6. 如果最终消息里包含 `scenes`，则需要是 3 到 4 条分镜结构，且每条也必须满足同样的结构要求。

换句话说，生成脚本负责“尽量让强模型吐出可用样本”，清洗脚本负责“把不合格的、重复的、格式不对的样本剔除掉”。

### 交接时最重要的约定

1. 改 prompt 以后，要重新生成数据，不要直接复用旧分布的数据。
2. 如果新场景需要不同的输出格式，先改 prompt，再改校验逻辑，最后再放大生成规模。
3. 若数据看起来能跑但训练效果变差，优先检查 `src/data_pipeline/core.py` 的校验规则是不是太宽或太严。

## 6. 环境配置与依赖

这个仓库假设接手者从零开始配置环境，所以这里按 Windows + Conda 的方式写，从安装 Conda 到安装项目依赖一步一步走。

### 6.1 安装 Conda

如果你的电脑还没有 Conda，建议直接安装 Miniconda 或 Anaconda。对于这个项目，更推荐 Miniconda，因为它更轻量。

1. 打开浏览器，下载 Miniconda for Windows x86_64。
2. 一路按默认选项安装即可。
3. 安装完成后，打开 PowerShell，执行 `conda --version` 确认可用。

如果系统提示找不到 `conda`，说明 Conda 没有加到 PATH 里。最省事的方式是重新打开一个终端，或者使用 Anaconda Prompt 再操作。

### 6.2 创建项目环境

建议使用 Python 3.11，新人直接用这个版本最稳：

```powershell
conda create -n prompt-opt python=3.11 -y
conda activate prompt-opt
```

### 6.3 安装 PyTorch

先安装和你机器 CUDA 匹配的 PyTorch。如果你不确定自己的 CUDA 版本，优先看显卡驱动和现有环境；如果只是先跑 CPU 或调试，也可以先装能用的版本再调整。

```powershell
python -m pip install -U pip setuptools wheel
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 6.4 安装项目依赖

项目依赖已经写在 `pyproject.toml` 里，推荐直接安装：

```powershell
pip install -e .
pip install -e .[dev]
```

如果你只想先把正式依赖装起来，也可以只执行 `pip install -e .`。开发依赖里的 `pytest` 和 `ruff` 主要用于测试和静态检查。

### 6.5 必要依赖说明

这个项目最关键的依赖是这些：

1. `torch`：模型训练和推理基础。
2. `unsloth`：用于 4-bit 加载和 QLoRA 微调。
3. `transformers`、`trl`、`accelerate`：训练管线和 SFT 支持。
4. `datasets`：加载和处理训练数据。
5. `openai`：数据生成阶段调用兼容接口。

### 6.6 API 配置

数据生成脚本需要你自己准备 API Key。当前默认使用 DeepSeek 兼容接口，推荐的配置方式如下：

```powershell
$env:API_KEY = "你的API密钥"
$env:BASE_URL = "https://api.deepseek.com/v1"
$env:MODEL_NAME = "deepseek-chat"
```

如果你换成其他强模型，只要接口是兼容的，就可以继续沿用同一套生成和清洗流程。

### 6.7 从零开始的最短检查

环境装好后，先做这三个检查：

1. `conda activate prompt-opt`
2. `python --version` 确认是 3.11 左右
3. `python .\scripts\data\generate_data.py --help` 能正常打开参数说明

只要这三步能过，说明环境基本可用。

## 7. 训练 Qwen2.5-7B（Unsloth QLoRA）

```powershell
# 使用清华镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"

python .\Qwen2.5-7B-train.py `
  --model-name Qwen/Qwen2.5-7B-Instruct `
  --train-file .\data\train_data.cleaned.jsonl `
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

## 8. 推理测试

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

## 9. 关键默认值（当前代码）

- 数据生成：DeepSeek（`deepseek-chat`）
- 训练模型：`Qwen/Qwen2.5-7B-Instruct`（Unsloth 4-bit 加载）
- 训练输出目录：`outputs/qwen25_7b_prompt_optimizer`
- 推理默认 adapter：`outputs/qwen25_7b_prompt_optimizer`

## 10. TODO

### 已知问题

1. `clear` 现在还不能真正清空历史记录
2. 存在死循环输出问题