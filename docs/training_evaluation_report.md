# 训练效果评估报告 - 媒体提示词优化模型

## 执行摘要

> **修订说明（核对仓库内真实产物后修订）**：本报告最初写于 2026-03-17，描述的是当时一次「1200 条数据 / 约 102 步」的训练。该次训练的 `trainer_state.json`（原文引用 `checkpoint-102`）**如今已不在仓库中**（`scripts/train.py` 设置 `save_total_limit=3`，旧 checkpoint 会被自动清理），因此其中「1200 条样本」「loss 2.449 → 0.709」「总 FLOPs 5.89e+16」等数字在现存产物中**无法核实**（作者本人亦无法确认该次训练的具体来源），本版已按现存 checkpoint 的真实值改写，或明确标注为「无产物支撑」。
>
> **最需要记住的一条**：训练损失（training loss）只反映模型对**训练集**的拟合程度，**不能用来推断生成质量**。本报告把「训练过程指标」与「模型效果」严格分开；目前仓库内**没有**足以证明「生成质量提升」的评测数据。

基于对训练过程、模型文件和训练日志的分析，结论如下：

- ⚠️ **训练过程正常完成**，但**模型效果尚未得到充分评测**（详见第 4、6 节与第 8 节）
- ✅ **训练损失下降（训练过程指标）**: V2 从 step 5 的 2.9398 降至 step 60 的 1.2374（-57.9%）；V1（`checkpoint-300`）末尾为 1.2705。**这不等于生成质量提升**
- ✅ **配置合理**: LoRA rank=16, alpha=32, dropout=0.05，目标模块覆盖 7 个投影层（与 `adapter_config.json` 逐项一致）
- ⚠️ **数据规模**: V2 训练集 **400 条**多轮对话（`data/train_data.cleaned.jsonl`），由 **DeepSeek API 合成**、脚本校验清洗，**不是人工标注**；V1 旧数据 800 条（`data.old/`，不推荐）
- ✅ **训练步数**: V2 = 60 步 / 5 epoch；V1 = 300 步 / 3 epoch（来源：各自 `trainer_state.json`），per-device batch size=2
- ✅ **文件完整**: 现存 adapter 与 checkpoint 文件都存在且可读取

## 1. 训练过程分析

### 1.1 损失变化曲线

**口径提醒**：以下都是**训练集上的训练损失**（`trainer_state.json` 的 `log_history`），不是验证集/测试集指标，也不能代表生成质量。

**V2（当前版本，`outputs/qwen25_7b_prompt_optimizer_v2/checkpoint-60/trainer_state.json`）**，`max_steps=60`、5 epoch、`logging_steps=5`：

```
Step  5: loss=2.9398
Step 10: loss=2.2965  (-21.9%)
Step 20: loss=1.5428  (-32.8%)
Step 30: loss=1.3846  (-10.3%)
Step 40: loss=1.3194  (-4.7%)
Step 50: loss=1.2912  (-2.1%)
Step 60: loss=1.2374  (-4.2%)   ← 最终记录值
```

**V1（旧版本，`outputs/qwen25_7b_prompt_optimizer/checkpoint-300/trainer_state.json`）**，`max_steps=300`、3 epoch、`logging_steps=10`：

```
Step  10: loss=2.5814
Step  30: loss=0.7402
Step 100: loss=0.4996
Step 190: loss=0.3861   ← 2026-03-19 那次训练的最低记录值
Step 200: loss=2.6662   ← 续训后学习率调度重启，loss 反弹（grad_norm=7.06）
Step 300: loss=1.2705   ← 最终记录值
```

**关键观察**（仅针对训练过程）:
1. **快速下降期**（V2 step 5-20）：训练损失从 2.94 降到 1.54，说明模型在拟合训练样本的格式与措辞
2. **平稳下降期**（V2 step 20-60）：无异常波动，最终落到 1.24
3. **V1 存在一次明显波动**：`checkpoint-300` 在 step 190 → 200 之间 loss 由 0.386 反弹到 2.666，原因是 2026-05-28/29 的续训从 `checkpoint-198` 恢复、学习率调度被重置（step 190 的 lr 已衰减到 1.08e-06，step 200 又回到 5.38e-05）。**因此不能说「训练过程无剧烈波动」**
4. 以上变化只说明**拟合训练集的过程**，与「模型生成的提示词是否更好用」是两件事

### 1.2 训练参数统计

以下数值分两类，请勿混用：

| 项目 | V1（旧，`qwen25_7b_prompt_optimizer`） | V2（当前，`qwen25_7b_prompt_optimizer_v2`） | 来源 |
|------|----------------------------------------|---------------------------------------------|------|
| 训练步数 | 300（2026-03-19 那次的 `max_steps` 就是 198，已按其自身调度跑满 3 epoch 后结束，不是中途崩溃；5/28-29 又从 `checkpoint-198` 以新调度 `max_steps=300` 继续训练） | 60 | `trainer_state.json` 的 `global_step` / `max_steps` |
| 训练周期 | 3 epoch | 5 epoch | 同上 `num_train_epochs` |
| 训练集规模 | 800 条（`data.old/`） | 400 条（`data/`） | `wc -l` 数据文件 |
| per-device batch size | 2 | 2 | `trainer_state.json` 的 `train_batch_size` |
| 日志间隔 | 10 步 | 5 步 | `trainer_state.json` 的 `logging_steps` |
| 总 FLOPs（记录值） | 6.517e+16（`checkpoint-300`；`checkpoint-198` 为 5.181e+16） | 3.277e+16 | `trainer_state.json` 的 `total_flos` |
| 梯度累积步数 | **无记录** | **无记录** | `trainer_state.json` 不含该字段；脚本默认 18，但 V1 的 epoch↔step 关系与之不符，待作者确认 |
| 学习率 | 余弦衰减，峰值约 2e-4（step 10 记录 lr=1.999e-4） | 余弦衰减，峰值约 1e-4（step 5 记录 lr=9.971e-5） | `log_history` 的 `learning_rate` |

> 说明：旧版报告写的「102 步 / 3 epoch / 总 FLOPs 5.89e+16 / 学习率 2e-4 / gradient accumulation=18」无法在现存产物中核实：`checkpoint-102` 不存在，现存 `trainer_state.json` 里没有任何一处是 5.89e+16。这里给出的是可核实的真实值。

## 2. 模型配置评估

### 2.1 LoRA配置
```json
{
  "lora_rank": 16,          // 适中的秩，平衡效果与参数效率
  "lora_alpha": 32,         // alpha=2*r，标准配置
  "lora_dropout": 0.05,     // 适中的dropout防止过拟合
  "target_modules": [       // 覆盖关键模块
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
  ]
}
```

**配置核对结果**（与 `outputs/*/adapter_config.json` 逐项比对，V1/V2 相同）:
- `r=16`、`lora_alpha=32`、`lora_dropout=0.05`、`bias="none"`、`task_type=CAUSAL_LM` ✅ 与 `scripts/train.py` 的默认值一致
- `target_modules` 覆盖 q/k/v/o/gate/up/down 共 7 个投影层 ✅ 与脚本一致
- 可训练参数按 `adapter_model.safetensors` 头部实测为 **40,370,176（≈40.37M）**，占 7B 基座的 **0.530%**
- ⚠️ 无法评价「配置好坏」：没有对照实验（不同 rank / lr / epoch 的对比）与效果指标，因此这里不再给星级评分

### 2.2 训练超参数
（以下来自 `scripts/train.py` 的默认值，除注明外未在 `trainer_state.json` 中留档）
- **最大序列长度**: 384 tokens（脚本默认 `--max-seq-length 384`）
- **学习率**: 余弦衰减；V1 峰值约 2e-4，**V2 峰值约 1e-4**（由 `log_history` 的 `learning_rate` 反推，脚本默认值为 2e-4）
- **优化器**: paged_adamw_8bit
- **梯度检查点**: 启用（`gradient_checkpointing=True`，脚本注释为 unsloth 模式）
- **权重衰减**: 0.01
- **warmup_ratio**: 0.03
- **seed**: 42

## 3. 数据质量评估

### 3.1 训练数据统计
- **样本数量**: V2 = 400 条（`data/train_data.cleaned.jsonl`，400 行 = 400 条样本）；V1 = 800 条（`data.old/train_data.cleaned.jsonl`）。旧版报告写的「1200 条」在仓库内**没有对应文件**
- **数据来源**: 调用 **DeepSeek API**（`deepseek-chat`）**合成**多轮对话，提示词见 `prompt/data_generation_system_prompt.txt`，生成入口 `scripts/run_data_generation.py` / `src/data_pipeline/generate.py`，随后经 `scripts/data/merge_clean_data.py` 去重清洗。**不是人工标注**，也未见人工逐条审核记录
- **数据格式**: JSONL，顶层字段固定为 `messages` + `meta`，`messages` 为 7-9 条消息的多轮对话
- **实际分布（脚本统计）**:
  - V2：消息条数 7 条 322 例 / 8 条 10 例 / 9 条 68 例；`meta.difficulty` = medium 311 / hard 68 / easy 21；`meta.scene` 去重后 **125 种**
  - V1：消息条数 7 条 497 / 8 条 60 / 9 条 243；`meta.difficulty` = medium 689 / hard 103 / easy 8；`meta.scene` 去重后 **225 种**
  - V2 最终 assistant 消息里 `prompt` 字段长度：最小 474 / 中位 1057 / 最大 2635 / **平均 1073 字符**（与 `docs/PROJECT.md` 的「平均 1073 字符」一致）；V1 为最小 131 / 平均 542，**旧文档写的「77-158 字符」在现存产物中找不到依据**

### 3.2 场景覆盖度
现存数据的 `meta.scene` 是自由文本而非固定分类，按场景名可粗分为（V2）：
1. **文旅/景区宣传**（海报、主视觉、宣传片镜头等，占比最大）
2. **城市形象与夜游/夜市**（城市宣传片、灯会、夜游品牌）
3. **乡村与生态旅游**（乡村振兴、民宿、田园综合体、湿地/海岛）
4. **文化遗产与非遗**（纪录片海报、非遗市集、手工艺体验）
5. **短视频/公众号封面**（竖版封面、头图）
6. **视频分镜类**（含 `scenes` 分镜数组，V2 中 126/400 条）

> 旧版报告列的「政务宣传/新闻播报/产品展示图/微电影片头」等分类，无法与现存 `meta.scene` 的取值对应，可能来自已丢失的早期数据集，已按现存数据改写。

### 3.3 对话模式分析
数据展现了完整的优化流程：
1. **用户提出模糊需求**
2. **助手追问澄清细节**
3. **用户补充信息**
4. **助手生成优化提示词**

## 4. 模型能力（设计目标，尚未评测）

**本节内容全部是「设计目标 / 待验证假设」，不是实测结果。** 除第 6.4 节列出的 4 条规则打分用例外，仓库内没有对模型输出质量的评测。

### 4.1 设计目标能力
1. **追问澄清**: 通过提问澄清模糊需求
2. **专业术语**: 使用媒体领域的专业术语
3. **结构化输出**: 生成结构化的紧凑 JSON 提示词
4. **多轮对话**: 支持多轮交互式优化

### 4.2 设计目标表现（未验证）
- **简单请求**: 追问 2-3 个关键细节
- **中等请求**: 生成包含 5-7 个要素的提示词
- **完整对话**: 完成 3-4 轮对话并生成最终提示词

> 上述数字是当初写数据生成提示词时的目标，没有任何测试集统计支撑，**不能当作评测结论引用**。

## 5. 潜在风险与改进建议

### 5.1 潜在风险
1. **评测缺口（当前最大风险）**: 没有留出验证/测试集，没有人工评分或自动化质量指标，因此**无法判断 V2 是否比基座模型或 V1 更好**
2. **过拟合风险**: 训练集仅 400 条（V2），且训练损失降到 1.24、后续几乎不再下降，存在过拟合可能；但也可能只是任务太窄，需验证集才能判断
3. **场景局限**: 数据集中在文旅/体制内宣传场景，通用性有限
4. **技术时效性**: 提示词风格可能随时间变化
5. **训练/推理 system prompt 一致性未完全对齐**: V2 训练数据里的 system 消息为 91 字符（去重后仅 1 种写法），而 `prompt/inference_system_prompt.txt` 当前为 89 字符，两者**文本并不相同**（旧版 `PROJECT.md` 把推理 prompt 记为 91 字符、暗示已对齐；实际仍有差异。当前 `docs/PROJECT.md` 已如实标注该问题未消除，见 6.4 节）

### 5.2 改进建议
1. **数据扩充**:
   - 增加至5000+训练样本
   - 扩展更多媒体场景 (教育、医疗、文旅等)
   - 加入不同层级案例 (中央、省、市、县)

2. **训练优化**:
   - 尝试rank=32或64提高表达能力
   - 调整学习率调度策略
   - 增加早停机制防止过拟合

3. **评估完善（优先级最高）**:
   - 从 400 条数据中划出验证/测试集，训练时记录 `eval_loss`（当前训练未做任何 evaluation，`log_history` 里没有 `eval_loss`）
   - 固化 V1 / V2 / 基座模型的同题对照（同一批测试输入、同一解码参数），并把 adapter 路径与日期写进结果文件（现有 `evaluation_results.json` 未记录这两项）
   - 自动指标：JSON 解析成功率、必需字段与分镜完整性、`prompt` 长度分布、关键维度（镜头/光线/构图/色彩/情绪）覆盖率
   - 人工盲评：请非作者按 6.3 的标准打分，并保留评分表

## 6. 推理测试指南

### 6.1 快速启动
```bash
conda activate prompt-opt
# 在本仓库根目录执行（不要写死盘符路径）
python inference.py --low-vram
```

### 6.2 测试用例建议

**测试1 - 简单请求验证追问能力**
```
输入: "党建宣传海报"
期望: 追问发布层级、视觉风格、核心元素
```

**测试2 - 中等复杂度测试专业度**
```
输入: "乡村振兴短视频封面，要体现希望和活力"
期望: 追问平台、具体元素、色彩风格、构图
```

**测试3 - 多轮对话测试连贯性**
```
第一轮: "人文纪录片海报"
第二轮: "省级卫视，电影感纪实，手工艺人工作"
期望: 生成包含具体细节的提示词
```

### 6.3 评估标准
1. **追问质量**: 问题是否切中要害
2. **专业程度**: 是否使用领域术语
3. **输出结构**: 是否清晰、易用
4. **实用性**: 生成的提示词是否可直接使用

### 6.4 仓库内现有的评测情况（据实说明）

**评测方法：规则打分，不是人工评估，也不是 LLM 评审。**
`scripts/evaluate.py`（根目录 `evaluate.py` 只是 wrapper）对 4 条写死的用例做关键词/长度/结构启发式打分，指标为：是否追问（关键词匹配）、`response_length`、`key_terms_score`（10 个词的命中率）、`has_structure`、`professional_terms`（12 个专业词的命中**个数**）以及 `professional_score = min(professional_terms / 5, 1.0)`。**不检查最终 JSON 是否可解析、是否可直接投喂文生图工具。**

> 另需注意：该脚本的默认基座是 `Qwen/Qwen2.5-7B-Instruct`，而适配器实际基于 `unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit`（见 `adapter_config.json`）。不加 `--base-model` 直接跑评测，基座与训练时不一致，结果不可比。

**现有结果**（`evaluation_results.json`，4 条用例）：

| 指标 | 数值 |
|------|------|
| `questioning_score` | 4/4 均为 1.0 |
| `key_terms_score` | 0.2 / 0.5 / 0.1 / 0.3 |
| `structure_score` | 0.0 / 1.0 / 0.0 / 1.0 |
| `professional_score` | 0.0 / 0.6 / 0.0 / 0.6 |

**必须同时说明的限制**：
1. 该文件**没有记录所用的 adapter 路径与运行日期**；`scripts/evaluate.py` 的 `--adapter-path` 默认值是 **V1**（`outputs/qwen25_7b_prompt_optimizer`），文件时间也早于 V2 训练，因此**现有结果很可能只是 V1 的，不能当作 V2 的评测**
2. 4 条用例**全部停留在追问阶段**，没有一条走到最终 JSON 输出，因此对「生成专业提示词」这一核心目标**没有任何评测数据**
3. 没有留出测试集、没有人工评分、没有与基座模型的对照，也没有 V1/V2 的同题对照
4. 评测用的 system prompt（`prompt/evaluation_system_prompt.txt`，370 字符）与推理用（`prompt/inference_system_prompt.txt`，89 字符）、训练数据里实际使用的（91 字符）**是三套不同的文本**，这会让评测结论难以归因

**因此**：现有证据只能支持「训练过程跑通」以及「被评测的那个 adapter 在 4 条用例上都被规则判为在追问」，**不能支持「生成质量提升」「效果好」这类结论**。

## 7. 技术指标总结

### 7.1 训练过程指标（**不是效果指标**）
- **训练损失下降**: V2 2.9398 → 1.2374（-57.9%）；V1 最终 1.2705（中途 2026-03-19 那次最低 0.3861）。**训练损失下降只说明在拟合训练集，不能推断生成质量提升**
- **下降速度**: V2 前 20 步内从 2.94 降到 1.54
- **训练稳定性**: V2 平稳；V1 在续训处（step 200）出现 loss 2.666 的反弹
- **验证集指标**: 无 —— 训练未配置 evaluation，`log_history` 中没有任何 `eval_loss`

### 7.2 模型效率指标
- **可训练参数量**: 40,370,176（≈40.37M），占 7B 基座 **0.530%**（实测自 `adapter_model.safetensors` 头部；旧版写的 0.1% 不准确）
- **adapter 文件大小**: 161,533,192 字节（≈154 MiB）
- **显存**: 4-bit 量化 + Adapter 模式；`scripts/inference.py --low-vram` 的默认显存上限是 7500MiB。**「9.7GB → 7.1GB」为作者实测数据**
- **推理速度**: 无实测数据

### 7.3 数据与训练过程质量
- **数据来源**: DeepSeek API 合成（**非人工标注**），经脚本去重清洗
- **数据规模**: V2 400 条 / V1 800 条
- **场景覆盖**: `meta.scene` 自由文本，V2 去重 125 种、V1 225 种
- **对话结构**: 7-9 条消息的多轮对话，末条 assistant 为紧凑 JSON

## 8. 结论与建议

### 8.1 主要结论
1. **训练过程正常完成**：V2 在 400 条合成数据上训练 60 步 / 5 epoch，checkpoint 与 adapter 文件完整可读，配置与脚本默认值一致
2. **训练损失显著下降，但这只是训练过程指标**：不能据此推断生成质量提升；两者之间隔着一次尚未完成的评测
3. **模型效果未经充分评测**：现有 `evaluation_results.json` 是 4 条用例的规则打分（很可能用的是 V1），没有测试集、没有人工评分、没有 V1/V2 对照
4. **数据为 API 合成，非人工标注**：V2 400 条，来源 DeepSeek API

### 8.2 最终建议
1. **先补评测，再谈效果**：划分测试集，跑 V1 / V2 / 基座三方同题对照，并把 adapter 路径与日期写进结果文件
2. **短期优化**: 根据评测结果再决定是否调参（rank / lr / epoch）或补充数据
3. **长期规划**: 建立可复现的评测流程，再谈持续迭代

### 8.3 风险评估
- **低风险**: 训练过程正常，文件完整，配置可复现
- **中风险**: 数据量有限（400 条），存在过拟合可能
- **高风险**: **效果未经验证**——若直接把「训练 loss 降到 1.24」当作「模型可用」的依据对外展示，属于结论超出证据范围

## 附件

1. 训练日志（真实存在）
   - V2 当前版本: [`checkpoint-60/trainer_state.json`](../outputs/qwen25_7b_prompt_optimizer_v2/checkpoint-60/trainer_state.json)
   - V1 旧版本: [`checkpoint-300/trainer_state.json`](../outputs/qwen25_7b_prompt_optimizer/checkpoint-300/trainer_state.json)
   - ~~`outputs/qwen25_7b_prompt_optimizer/checkpoint-102/trainer_state.json`~~ —— **该文件不存在**（旧版报告引用的失效链接，已移除）
2. [V2 模型配置文件](../outputs/qwen25_7b_prompt_optimizer_v2/adapter_config.json) / [V1 模型配置文件](../outputs/qwen25_7b_prompt_optimizer/adapter_config.json)
3. [评估脚本](../scripts/evaluate.py)（根目录 `evaluate.py` 是它的 wrapper）
4. [快速自检脚本](../scripts/quick_test.py)（检查模型文件完整性与配置，不加载模型）
5. [现有评测结果](../evaluation_results.json)
6. ~~`manual_evaluation_guide.md`~~ —— **该文件不存在**（旧版报告引用的失效链接，已移除）

---

**报告生成时间**: 2026-03-17
**本次修订**: 按仓库内现存产物（`data/*.jsonl`、`outputs/*/checkpoint-*/trainer_state.json`、`adapter_config.json`、`scripts/*.py`、`git log`）逐项核对后改写；数据来源由「人工标注」更正为「DeepSeek API 合成」，训练指标按现存日志改写，并删除失效引用；2026-10-02 复审：修正 `checkpoint-198` 的成因描述（该次训练是跑满自身 198 步调度后正常结束，非中途中断）、修正对 `docs/PROJECT.md` 的失效交叉引用、补全 `professional_score` 的实际计算方式与评测基座不一致的提示
**修订前评估方法**: 训练日志分析 + 文件完整性检查 + 配置合理性评估（所引用的 checkpoint 已丢失）
**修订后评估方法**: 产物核对 + 现存规则打分结果复核；**未做任何新的模型效果评测**
**评估结论**: ⚠️ **训练过程正常完成；模型效果尚未充分评测，不能据训练 loss 下降宣称效果良好**