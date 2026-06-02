# 项目交接文档 — 媒体提示词优化模型

> 人工智能课程项目 | Qwen2.5-7B + LoRA 微调

---

## 一、项目概述

### 1.1 要解决什么问题

文生图/视频工具（如 Midjourney、Stable Diffusion、Sora）需要用户输入**结构化的英文标签式提示词**才能生成高质量结果。但普通用户（文旅局宣传人员、景区新媒体运营等）只能用自然语言描述模糊需求，如"我要一张城市宣传海报"。

本项目的 AI 模型充当**媒体提示词优化专家**，通过多轮对话：
1. **追问澄清**：主动询问平台、风格、构图、光线、情绪等关键维度
2. **输出专业提示词**：生成可直接用于文生图/视频工具的英文标签化提示词

### 1.2 核心指标

| 指标 | 数值 |
|------|------|
| 基座模型 | Qwen2.5-7B-Instruct（7B 参数） |
| 微调方法 | QLoRA（4-bit 量化 + LoRA rank=16） |
| 训练样本 | 400 条高质量多轮对话 |
| 可训练参数 | 40M / 7600M（0.53%） |
| 训练损失 | 2.94 → 1.24（下降 57.7%） |
| 推理显存 | 6.1GB（--low-vram 模式） |
| 模型文件大小 | ~154MB（LoRA adapter） |

---

## 二、技术架构

### 2.1 整体流程

```
用户模糊需求
    ↓
系统提示词 → Qwen2.5-7B + LoRA → 追问澄清（1-2轮）
    ↓
信息足够后 → 输出紧凑 JSON
    ↓
{"prompt": "city skyline, golden hour, cinematic wide shot, ..."}
```

### 2.2 关键技术选型

| 技术 | 选择 | 原因 |
|------|------|------|
| 基座模型 | Qwen2.5-7B-Instruct | 中文理解能力强，7B 可在消费级显卡运行 |
| 量化 | 4-bit nf4 (QLoRA) | 将 7B 模型从 14GB 压缩到 ~4GB |
| 微调框架 | Unsloth | 训练速度 2x 提升，显存优化 |
| LoRA rank | 16, alpha=32 | 在效果与效率间平衡 |
| 训练数据生成 | DeepSeek API | 低成本生成 400 条高质量多轮对话 |
| 推理优化 | float16 计算 + Adapter 模式 | 峰值显存 6.1GB，适配 8GB 笔记本 |

### 2.3 对话数据格式

训练数据采用多轮对话结构（7-9 条消息），最后一条 assistant 消息为纯 JSON：

```json
{
  "messages": [
    {"role": "system", "content": "你是媒体提示词优化专家..."},
    {"role": "user", "content": "我需要一张城市宣传海报"},
    {"role": "assistant", "content": "好的，先确认几个关键点：1. 平台尺寸？..."},
    {"role": "user", "content": "微信头图，16:9，写实风格..."},
    {"role": "assistant", "content": "补充确认：光线时段？构图偏好？..."},
    {"role": "user", "content": "黄昏金色光线，中心构图..."},
    {"role": "assistant", "content": "{\"prompt\":\"city skyline at golden hour, ...>=400字符的标签化英文prompt\"}"}
  ],
  "meta": {"scene": "城市宣传海报", "difficulty": "medium"}
}
```

### 2.4 输出 JSON 规范

```json
// 静态图
{"prompt": "主体, 场景, 风格, 镜头, 光线, 色彩, 构图, 情绪, material details, ..."}

// 视频/长故事（含分镜）
{
  "prompt": "主 prompt >=400 字符",
  "scenes": [
    {"id": 1, "prompt": "分镜1 prompt >=200 字符"},
    {"id": 2, "prompt": "分镜2 prompt"},
    {"id": 3, "prompt": "分镜3 prompt"}
  ]
}
```

---

## 三、研发过程

### 3.1 迭代记录

| 阶段 | 工作 | 结果 |
|------|------|------|
| **V1 初始** | 用旧版 prompt 生成 ~800 条数据，训练 3 epoch | 损失降至 0.71，但模型直接输出 JSON 不追问，prompt 仅 77-158 字符 |
| **问题诊断** | 发现：1) 训练数据 prompt 偏短 2) 推理 system prompt 与训练不一致（91 vs 757 字符）| 确认根因 |
| **数据质量提升** | 重写 `data_generation_system_prompt.txt`，要求 prompt >=400 字符 + 紧凑 JSON | 新数据 prompt 平均 1073 字符 |
| **V2 训练** | 400 条高质量数据，5 epoch，lr=1e-4 | 损失 2.94→1.24（57.7%），模型正确追问 |
| **显存优化** | Adapter 模式 + float16 + 6800MiB 限制 | 9.7GB → 6.1GB（-37%） |
| **Web Demo** | Gradio 网页界面 | 支持多轮对话、JSON 格式化显示 |

### 3.2 关键经验

1. **System prompt 一致性至关重要**：训练数据中的 system 消息必须与推理时一致。V1 用 91 字符训练但 757 字符推理，导致模型行为异常。
2. **LoRA 不合并更省显存**：`merge_and_unload()` 会将 4-bit 权重 → fp16 → 合并 → 4-bit，中间产生 fp16 峰值，多了 2-3GB。
3. **float16 vs bfloat16**：bfloat16 精度更高但比 float16 多占 ~200MB 显存，8GB 卡建议 float16。
4. **数据质量 > 数量**：400 条高质量样本的效果远好于 800 条一般样本。

---

## 四、实验数据

### 4.1 训练损失曲线

```
Step  5: loss=2.940  ████████████████████████████
Step 10: loss=2.296  ██████████████████████▌
Step 20: loss=1.543  ███████████████▍
Step 30: loss=1.385  █████████████▋
Step 40: loss=1.319  █████████████▎
Step 50: loss=1.291  █████████████
Step 60: loss=1.237  ████████████▍  ← 最终
```

下降 57.7%，5 epoch，35 分钟（RTX 3060 12GB）。

### 4.2 模型效果对比

| 测试项 | V1（旧模型） | V2（当前模型） |
|--------|-------------|---------------|
| 追问行为 | 直接输出 JSON ❌ | 正确追问 ✅ |
| Prompt 长度 | 77-158 字符 | 平均 1073 字符 |
| 专业标签 | 缺少镜头/光线/构图 | 覆盖 6+ 维度 |
| JSON 格式 | 有换行缩进 | 紧凑一行 ✅ |
| 多轮对话 | 不支持 | 支持 ✅ |

---

## 五、仓库使用指南

### 5.1 快速启动

```powershell
conda activate prompt-opt
cd E:\01_workspace\prompt_optimizer_model

# 命令行推理
python scripts/inference.py --low-vram

# 网页演示（推荐展示用）
python scripts/web_demo.py --low-vram
```

### 5.2 生成新数据

```powershell
# 设置 API 密钥
$env:API_KEY = "sk-xxxxxxxx"
$env:BASE_URL = "https://api.deepseek.com/v1"
$env:MODEL_NAME = "deepseek-chat"

# 生成
python scripts/run_data_generation.py --target-size 200 --output data --sleep 0.8
python scripts/data/merge_clean_data.py --input-dir data --output data/train_data.cleaned.jsonl
```

### 5.3 训练新模型

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"  # 国内镜像
python scripts/train.py `
  --train-file data/train_data.cleaned.jsonl `
  --output-dir outputs/my_model `
  --num-train-epochs 3 --learning-rate 1e-4
```

---

## 六、展示方案建议

### 6.1 PPT 素材

- **架构图**：见 [二、技术架构](#二技术架构)
- **损失曲线**：见 [四、实验数据](#四实验数据)
- **效果对比**：见 [4.2 模型效果对比](#42-模型效果对比)
- **Demo 截图**：运行 `python scripts/web_demo.py --low-vram` 后截屏

### 6.2 现场演示流程

1. 启动 Web Demo → 浏览器打开 `http://127.0.0.1:7860`
2. 输入 "我需要一张城市宣传海报" → 展示追问
3. 逐步补充信息 → 展示多轮对话
4. 最终输出 JSON → 展示格式化结果

### 6.3 可讨论的技术亮点

- QLoRA 如何将 7B 模型压缩到 6GB 显存
- 多轮对话数据的自动生成方案
- System prompt 一致性对行为的影响
- LoRA Adapter 模式 vs Merge 模式的显存差异

---

## 七、文件清单

| 文件 | 用途 |
|------|------|
| `scripts/inference.py` | 命令行推理 |
| `scripts/web_demo.py` | Gradio 网页演示 |
| `scripts/train.py` | Unsloth QLoRA 训练 |
| `scripts/run_data_generation.py` | 调用 API 生成训练数据 |
| `scripts/data/generate_data.py` | 数据生成核心 |
| `scripts/data/merge_clean_data.py` | 数据清洗合并 |
| `prompt/inference_system_prompt.txt` | 推理系统提示词（91 字符） |
| `prompt/data_generation_system_prompt.txt` | 数据生成提示词（4363 字符） |
| `src/data_pipeline/core.py` | 数据验证、清洗、JSON 解析 |
| `src/data_pipeline/generate.py` | API 调用与数据生成逻辑 |
| `outputs/qwen25_7b_prompt_optimizer_v2/` | 当前最佳模型权重 |
| `data/train_data.cleaned.jsonl` | 训练数据（400 条） |

---

*最后更新：2026-06-02*
