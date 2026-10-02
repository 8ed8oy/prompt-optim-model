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
| 基座模型 | Qwen2.5-7B-Instruct（7B 参数）；**实际加载的是 4-bit 量化仓库** `unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit`（据 `outputs/*/adapter_config.json` 的 `base_model_name_or_path`） |
| 微调方法 | QLoRA（4-bit 量化 + LoRA rank=16） |
| 训练样本 | V2：400 条多轮对话（**DeepSeek API 合成**，经脚本清洗，非人工标注） |
| 可训练参数 | 40.37M / 约 7620M（0.53%，实测自 `adapter_model.safetensors`） |
| 训练损失 | 2.9398 → 1.2374（-57.9%）；**训练集拟合指标，不代表生成质量** |
| 验证集指标 | 无（训练未配置 evaluation，`log_history` 中无 `eval_loss`） |
| 模型效果评测 | 仅有 4 条用例的规则打分（`evaluation_results.json`，未记录所用 adapter）；无留出测试集、无人工评分、无 V1/V2 对照 |
| 推理显存 | 约 7.1GB（作者实测；`--low-vram` 默认显存上限 7500MiB） |
| 模型文件大小 | 161,533,192 字节（≈154 MiB，LoRA adapter） |

---

## 二、技术架构

### 2.1 整体流程

```
用户模糊需求
    ↓
系统提示词 → Qwen2.5-7B + LoRA → 追问澄清（训练数据按 7-9 条消息 = 3-4 轮往返构造；实际轮次未评测）
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
| 微调框架 | Unsloth | 据 Unsloth 官方宣称可提升训练速度、降低显存（**本仓库无对照实测**） |
| LoRA rank | 16, alpha=32 | 在效果与效率间平衡 |
| 训练数据生成 | DeepSeek API（`deepseek-chat`） | 低成本批量合成 400 条多轮对话；**为合成数据，非人工标注** |
| 推理优化 | float16 计算 + Adapter 模式 | 实测峰值显存约 7.1GB，适配 8GB 笔记本 |

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
| **V1 初始** | 用旧版 prompt 生成 800 条数据（现存 `data.old/train_data.cleaned.jsonl`，实测 800 行），训练 3 epoch | 2026-03-19 那次训练的 `max_steps` 就是 198，已按其自身调度跑满 3 epoch（`global_step=198`、`epoch=3.0`），训练损失最低 0.386（step 190）；2026-05-28/29 又从 `checkpoint-198` 以新调度（`max_steps=300`）继续训练，最终 1.271。旧文档写的「损失降至 0.71」在现存产物中找不到对应最终值（step 40 曾记录 0.7131），已按现存值改写 |
| **问题诊断** | 发现：1) 训练数据 prompt 偏短 2) 训练/推理 system prompt 不一致 | 现存数据可佐证：V1 数据里 system 消息有 **260 种不同写法**（长度 32–1569 字符），而当时推理用的 `prompt/inference_system_prompt.txt` 只有 52 字符（2026-05-21 提交）。旧文档写的「91 vs 757 字符」在产物中无出处，已按实测改写 |
| **数据质量提升** | 重写 `data_generation_system_prompt.txt`，要求 prompt >=400 字符 + 紧凑 JSON | 新数据（V2）`prompt` 字段平均 **1073 字符**（实测；最小 474、最大 2635） |
| **V2 训练** | 400 条数据，**5 epoch / 60 step**，lr=1e-4（由日志学习率反推） | 训练损失 2.9398→1.2374（-57.9%）；**注意：这是训练集拟合指标，不能推断生成质量；V2 未做效果评测** |
| **显存优化** | Adapter 模式 + float16 + 显存上限（`--low-vram` 默认 7500MiB） | 实测 9.7GB → 7.1GB（-27%）。（旧文档写的 6800MiB 与当前脚本默认值不符） |
| **Web Demo** | Gradio 网页界面 | 支持多轮对话、JSON 格式化显示 |

### 3.2 关键经验

1. **System prompt 一致性至关重要**：训练数据中的 system 消息必须与推理时一致。V1 数据里 system 消息有 260 种写法（32–1569 字符），推理 prompt 当时仅 52 字符，两者严重不一致。⚠️ 该问题在 V2 中**并未完全消除**：V2 训练数据的 system 消息为 91 字符，`prompt/inference_system_prompt.txt` 为 89 字符，文本仍不相同（待修正）。
2. **LoRA 不合并更省显存**：`merge_and_unload()` 会将 4-bit 权重 → fp16 → 合并 → 4-bit，中间产生 fp16 峰值，多了 2-3GB。（原理性说明；本仓库无对照日志）
3. **float16 vs bfloat16**：bfloat16 精度更高但比 float16 多占约 200MB 显存，8GB 卡建议 float16。（原理性说明；本仓库无对照日志，`trainer_state.json` 不含精度字段）
4. **数据质量 > 数量**：⚠️ 这是**假设，尚未验证**。仓库内没有 V1/V2 的同题对照评测，因此「400 条 V2 的效果远好于 800 条 V1」目前**不能断言**。

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

下降 57.9%（2.9398 → 1.2374），5 epoch、60 step。训练耗时 35 分钟，硬件为 RTX 3060 12GB（作者实测）。README 里的 `--low-vram` 面向 8GB 笔记本，是**推理**机器，与训练机器不是同一台，二者不矛盾。

> ⚠️ **口径提醒**：以上是**训练集训练损失**。loss 下降只说明模型在拟合这 400 条训练样本，**不能推断生成质量提升**；模型效果评测见 4.3。

### 4.2 模型效果对比（据实降级）

| 测试项 | V1（旧模型） | V2（当前模型） | 证据 |
|--------|-------------|---------------|------|
| 追问行为 | 作者当时亲测：V1 直接输出 JSON、不追问 | 作者当时亲测：V2 追问行为正常 | `evaluation_results.json` 只有 4 条用例且未记录所用 adapter，4/4 判为「在追问」，**不足以复现或推翻上述一手观察**；结论以一手观察为准，待补同题对照评测 |
| Prompt 长度 | 数据集实测：最小 131 / 平均 542 字符 | 数据集实测：平均 1073 字符（最小 474） | 脚本统计 `data.old/`、`data/`；旧文档的「77-158 字符」无产物支撑 |
| 专业标签 | 无对照评测 | 无对照评测 | — |
| JSON 格式 | 无对照评测 | 无对照评测（4 条用例均未走到输出 JSON） | — |
| 多轮对话 | 数据结构上两者都是 7-9 条消息多轮对话 | 同左 | `wc`/字段统计 |

> 上表只有「Prompt 长度」一行有产物支撑；其余在旧版本中是观察性描述，现无证据可查，故标注为「未评测」。**不要把这张表当作 A/B 效果对比使用。**

### 4.3 评测现状（重要）

| 项目 | 现状 |
|------|------|
| 评测方法 | `scripts/evaluate.py` 的**规则打分**（关键词命中、回复长度、是否分点、专业词计数）；不是人工评估 |
| 用例数 | 4 条写死的用例 |
| 结果文件 | `evaluation_results.json`（4 条；`questioning_score` 全为 1.0） |
| 所用模型 | 文件**未记录 adapter 路径与日期**；`scripts/evaluate.py` 默认指向 **V1**，且文件时间早于 V2 训练，故很可能只评了 V1 |
| 未做的部分 | 留出测试集、`eval_loss`、最终 JSON 的质量评测（4 条用例全部停在追问阶段）、人工评分、V1/V2/基座同题对照 |

> ⚠️ **评测基座不一致**：`scripts/evaluate.py` 的默认基座是 `Qwen/Qwen2.5-7B-Instruct`，而适配器实际训练时基于 `unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit`（见 `adapter_config.json`）；只有 `scripts/inference.py` 的默认基座才是后者。不加 `--base-model` 直接跑评测，基座与训练时不一致，结果不可比。

**结论：现有证据只能说明「训练过程跑通」+「被评测的那个 adapter 在 4 条用例上都被规则判为在追问」，不能说明生成质量提升。** 详见 `docs/training_evaluation_report.md`。

---

## 五、仓库使用指南

### 5.1 快速启动

```powershell
conda activate prompt-opt
# 在本仓库根目录执行（原文档写死的盘符绝对路径已失效，统一改用相对路径）

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

- QLoRA 如何把 7B 模型压到消费级显卡可跑的显存（实测约 7.1GB；`--low-vram` 默认上限 7500MiB）
- 多轮对话数据的自动生成方案
- System prompt 一致性对行为的影响（含 V2 仍未对齐的 91 vs 89 字符问题）
- LoRA Adapter 模式 vs Merge 模式的显存差异

---

## 七、文件清单

| 文件 | 用途 |
|------|------|
| `scripts/inference.py` | 命令行推理 |
| `scripts/web_demo.py` | Gradio 网页演示 |
| `scripts/train.py` | Unsloth QLoRA 训练 |
| `scripts/evaluate.py` | 规则打分评测（根目录 `evaluate.py` 为 wrapper） |
| `scripts/quick_test.py` | 模型文件完整性自检 |
| `scripts/run_data_generation.py` | 调用 API 生成训练数据 |
| `scripts/data/generate_data.py` | 数据生成核心 |
| `scripts/data/merge_clean_data.py` | 数据清洗合并 |
| `prompt/inference_system_prompt.txt` | 推理系统提示词（实测 89 字符；V2 训练数据里的 system 为 91 字符，**尚未对齐**） |
| `prompt/data_generation_system_prompt.txt` | 数据生成提示词（实测 4364 字符） |
| `src/data_pipeline/core.py` | 数据验证、清洗、JSON 解析 |
| `src/data_pipeline/generate.py` | API 调用与数据生成逻辑 |
| `outputs/qwen25_7b_prompt_optimizer_v2/` | 当前版本权重（V2） |
| `outputs/qwen25_7b_prompt_optimizer/` | 旧版本权重（V1，不推荐）；其中根目录 adapter 与 `checkpoint-300` 的 `adapter_model.safetensors` 字节一致（md5 相同）；另含 2026-03-19 那次 3 epoch / 198 步训练结束时的 `checkpoint-198` |
| `data/train_data.cleaned.jsonl` | V2 训练数据（400 条，DeepSeek API 合成） |
| `data.old/train_data.cleaned.jsonl` | V1 旧训练数据（800 条，不推荐使用） |
| `docs/training_evaluation_report.md` | 训练效果评估报告（含评测缺口说明） |

---

*最后更新：2026-06-02；2026-06-07 按仓库内现存产物核对并修订数字与结论；2026-10-02 复审：修正 `checkpoint-198` 成因描述（跑满自身 198 步调度后结束，非中途中断残留）、移除写死的盘符路径、补充基座模型与评测基座不一致的说明*
