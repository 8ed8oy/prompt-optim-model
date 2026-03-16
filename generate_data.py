#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
generate_data.py

用途：
1) 使用 DeepSeek V3（OpenAI 兼容接口）合成多轮对话训练数据。
2) 输出为 Hugging Face chat template 可直接消费的 JSONL（messages 列表）。
3) 支持断点续传：如果中断，重新运行会自动接着生成，直到达到目标条数。

运行前环境变量：
- API_KEY:    必填，DeepSeek API 密钥
- BASE_URL:   选填，默认 https://api.deepseek.com/v1
- MODEL_NAME: 选填，默认 deepseek-chat（即 DeepSeek V3）

示例：
  set API_KEY=sk-xxxx
  python generate_data.py --target-size 1000 --output train_data.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI


# -----------------------------
# 1) 媒体场景种子（可按需扩展）
# -----------------------------
SCENE_SEEDS: List[str] = [
    "新闻纪实报道封面图",
    "科幻城市夜景短视频",
    "体育赛事集锦开场画面",
    "人文纪录片海报",
    "财经栏目片头视觉",
    "美食探店短视频",
    "旅游宣传片镜头脚本",
    "医疗科普栏目主视觉",
    "教育课程宣传视频",
    "公益宣传短片",
    "影视预告片关键镜头",
    "天气新闻播报背景图",
    "电商直播间产品展示图",
    "企业品牌发布会开场视频",
    "城市形象宣传片",

    # 政务与形象宣传
    "全国两会期间地方政府成就展板背景图",
    "城市政务APP开屏科技感海报",
    "‘最多跑一次’政务服务大厅宣传片分镜",
    "廉政文化主题微电影片头视觉",
    "地方公安干警/交警风雪中执勤纪实封面",
    # 党建与思想建设
    "七一建党节微信公众号首图（大气红底金字风格）",
    "基层党支部活动室文化墙背景插画",
    "‘学习强国’专栏文章配图（历史厚重感）",
    "优秀共产党员先进事迹汇报短视频开场",
    # 宏观经济与国企风采
    "国有企业年度社会责任报告封面大图",
    "‘一带一路’十周年港口货运繁忙景象无人机视角",
    "重点工程（桥梁/高铁/基建）通车仪式宣传片镜头",
    "高新技术产业园航拍夜景视频素材",
    # 乡村振兴与文旅
    "乡村振兴主题纪录片海报（金黄麦浪与新农人）",
    "地方非遗文化传承短视频特写镜头",
    "绿水青山就是金山银山（两山理论）生态宣传画",
    "县域文旅局长‘变装’宣传片转场分镜",
    # 民生与突发报道
    "抗洪抢险/救灾前线消防员逆行背影特写",
    "节假日高铁站春运温馨护航纪实摄影",
    "公立医院抗击疫情/义诊活动医护人员特写",
    "社区网格员走访慰问孤寡老人暖心插画",

    "省委全会开幕会直播背景板设计",
    "县级融媒体中心抖音政务短视频封面",
    "学习贯彻党的二十届三中全会精神宣传海报",
    "政府工作报告一图读懂长图设计",
    "防汛抗洪应急报道前线记者连线画面",
    "共同富裕示范区建设成果展板",
    "文明城市建设公益广告（社区版本）",
    "基层党建活动室上墙制度设计图",
    "营商环境优化专题报道主视觉",
    "枫桥经验60周年纪念活动海报"
]


# -----------------------------
# 2) Meta Prompt：要求模型生成规范 JSON
# -----------------------------
META_PROMPT = """
你是“媒体提示词优化专家”，由传播大脑（浙江日报报业集团旗下媒体人工智能公司）打造，旨在服务体制内媒体工作者（记者、编辑、融媒体运营人员）。你的任务是帮助用户将模糊的创作意图，优化为高质量、符合主流媒体传播要求的文生图/视频提示词。

【核心原则：懂选题、知语境、守底线、能进化】
1.  **懂选题**：能准确理解不同类型新闻（时政新闻、社会新闻、民生新闻、突发报道、党建宣传、政务发布、公益广告等）的视觉表达特点。
2.  **知语境**：熟悉省、市、县三级融媒体中心的实际工作场景，如微信公众号首图、政务APP开屏、短视频封面、纪录片海报、应急报道配图等。
3.  **守底线**：严格遵守互联网新闻信息服务管理规定，生成内容必须符合主流媒体价值观导向，确保政治安全、内容安全。
4.  **能进化**：通过多轮追问，精准捕捉用户需求，最终输出专业、结构化的英文Prompt。

【语言与模板约束】
1. 除最后一条助手消息中的英文Prompt正文外，其余所有对话内容必须使用简体中文，不要中英混写。
2. 最后一条助手消息必须严格使用以下统一模板，不能增加任何额外段落：
    【最终优化提示词】
    <仅英文Prompt正文，使用英文逗号分隔，不要夹杂中文解释>
3. 英文Prompt正文中禁止出现整句中文、中文标点、平台口号、解释性括号说明。
4. system 消息可以简写，但必须保持“媒体提示词优化专家”角色设定。

【对话流程规范】
请严格按照以下逻辑生成2-3轮往返的多轮对话（总消息数5-7条，最后一条必须是助手）：

- **第1轮用户**：给出模糊的创作意图，必须是媒体内容创作相关场景（如“七一建党节公众号首图”“防汛抗洪报道配图”“共同富裕示范区展板”等）。
- **第1轮助手**：针对用户意图进行追问，必须覆盖以下维度中的至少3个：
    *   **发布层级**：中央/省级/县级？政务发布/商业平台/内部使用？
    *   **视觉风格**：庄重大气/温暖纪实/科技感/插画手绘/写实摄影？
    *   **核心元素**：是否需要党徽、红旗、特定标语、地标建筑、人物特写？
    *   **技术参数**：横版/竖版？图片/短视频？时长要求？画幅比例？
    *   **情绪基调**：严肃庄重/温暖感人/激昂奋进/冷静客观？
- **第2轮用户**：补充具体的约束条件和偏好，为助手提供更明确的信息。
- **第2轮（或第3轮）助手**：输出**最终优化提示词**，必须包含“【最终优化提示词】”标签，标签后紧跟高质量的英文Prompt。

【英文Prompt写作规范】
最终输出的英文Prompt必须结构清晰，包含以下要素：
*   **场景/主体**：核心画面内容是什么（如“基层党员在防汛一线搬运沙袋”）
*   **镜头语言**：特写/中景/远景/航拍？推/拉/摇/移？
*   **光线/色彩**：自然光/舞台光/逆光？红色调/金色调/冷色调/纪实黑白？
*   **风格/氛围**：纪实摄影风格/电影感/插画风格/X渲染风格？
*   **构图/技术**：中心构图/三分法构图？9:16竖屏/16:9宽屏？4K/高清？
*   **价值观导向**：确保符合主流价值，体现人文关怀或正能量。

【安全与合规红线】
- **严禁**生成包含血腥、暴力、敏感政治元素、争议性历史事件的内容。
- **涉及党政内容**（如党建、两会、政府成就），风格必须庄重、严肃，优先考虑红底金字、党徽、红旗、庄重建筑等元素。
- **涉及灾难/突发报道**（如防汛、抗疫），必须体现“逆行者”“众志成城”“人文关怀”，避免直接呈现伤亡惨状。
- **涉及民生/社会新闻**，可适当温暖、活泼，但不得低俗、猎奇。

【输出格式要求】
必须输出**单个JSON对象**，不要Markdown代码块，不要额外说明。JSON结构如下：
{
  "messages": [
        {"role": "system", "content": "你是媒体提示词优化专家..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ],
  "meta": {
    "scene": "从用户意图中提取的场景关键词",
    "difficulty": "easy|medium|hard"  // 根据追问复杂度和生成难度随机选择
  }
}
""".strip()


def build_user_instruction(scene: str) -> str:
    """构造每次请求给强模型的用户指令，加入场景与随机扰动以提高多样性。"""
    style_bias = random.choice(
        [
            "强调镜头语言与运镜",
            "强调视觉风格与色彩",
            "强调叙事节奏与情绪",
            "强调构图与光影细节",
        ]
    )
    round_hint = random.choice(["2轮往返", "3轮往返"])
    return (
        f"请围绕场景“{scene}”生成一条训练样本；{style_bias}；对话长度偏向{round_hint}。"
        "请确保追问部分全部为简体中文，最终仅按统一模板输出英文Prompt。请直接输出 JSON 对象。"
    )


FINAL_TAG = "【最终优化提示词】"


def normalize_text(text: str) -> str:
    """规范文本空白，减少格式抖动。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    compact_lines: List[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                compact_lines.append("")
            previous_blank = True
            continue
        compact_lines.append(re.sub(r"\s+", " ", line))
        previous_blank = False
    return "\n".join(compact_lines).strip()


def extract_final_prompt(text: str) -> str:
    """提取最终英文 Prompt 正文。"""
    normalized = normalize_text(text)
    if FINAL_TAG not in normalized:
        return ""
    _, prompt_body = normalized.split(FINAL_TAG, 1)
    return prompt_body.strip()


def ascii_letter_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-z]", text)
    visible_chars = re.findall(r"\S", text)
    if not visible_chars:
        return 0.0
    return len(letters) / len(visible_chars)


def cjk_ratio(text: str) -> float:
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    visible_chars = re.findall(r"\S", text)
    if not visible_chars:
        return 0.0
    return len(cjk_chars) / len(visible_chars)


def has_abnormal_repetition(text: str) -> bool:
    return bool(re.search(r"(.)\1{7,}", text))


def normalize_final_message(text: str) -> str:
    """统一最终消息模板。"""
    prompt_body = extract_final_prompt(text)
    if not prompt_body:
        return normalize_text(text)
    prompt_body = re.sub(r"[。；：“”‘’（）、，]", " ", prompt_body)
    prompt_body = re.sub(r"\s+", " ", prompt_body).strip(" ,\n")
    return f"{FINAL_TAG}\n{prompt_body}"


def normalize_record(obj: Dict, scene: str) -> Dict:
    """对模型输出做轻量标准化，方便后续去重与训练。"""
    messages = obj.get("messages", [])
    normalized_messages: List[Dict] = []
    for idx, message in enumerate(messages):
        role = message.get("role", "")
        content = normalize_text(str(message.get("content", "")))
        if idx == len(messages) - 1 and role == "assistant":
            content = normalize_final_message(content)
        normalized_messages.append({"role": role, "content": content})

    obj["messages"] = normalized_messages
    if "meta" not in obj or not isinstance(obj.get("meta"), dict):
        obj["meta"] = {}
    obj["meta"]["scene"] = normalize_text(str(obj["meta"].get("scene") or scene))
    obj["meta"].setdefault("difficulty", random.choice(["easy", "medium", "hard"]))
    return obj


def extract_json_object(text: str) -> Optional[Dict]:
    """
    尝试从模型输出中提取 JSON 对象。
    - 优先直接 json.loads
    - 失败后使用正则截取最外层 {...}
    """
    text = text.strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 回退：从文本中抓第一段可能的 JSON 对象
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    candidate = match.group(0)
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def validate_sample(obj: Dict) -> bool:
    """校验样本是否满足训练格式要求。"""
    if "messages" not in obj or not isinstance(obj["messages"], list):
        return False

    messages = obj["messages"]
    if len(messages) < 5 or len(messages) > 7:
        return False

    valid_roles = {"system", "user", "assistant"}
    for m in messages:
        if not isinstance(m, dict):
            return False
        if m.get("role") not in valid_roles:
            return False
        if not isinstance(m.get("content"), str) or not m.get("content").strip():
            return False
        if len(m["content"].strip()) < 4 or len(m["content"].strip()) > 2000:
            return False
        if has_abnormal_repetition(m["content"]):
            return False

    # 最后一条必须是 assistant
    if messages[-1].get("role") != "assistant":
        return False

    # 最终 assistant 文本必须带标签
    final_content = normalize_text(messages[-1].get("content", ""))
    if FINAL_TAG not in final_content:
        return False

    final_prompt = extract_final_prompt(final_content)
    if not final_prompt:
        return False

    # 非最终 Prompt 的对话尽量保持中文，减少中英混拼
    for i, message in enumerate(messages[:-1]):
        if message["role"] == "system":
            continue
        content = normalize_text(message["content"])
        if ascii_letter_ratio(content) > 0.35:
            return False
        if cjk_ratio(content) < 0.15:
            return False

    # 最终 Prompt 正文以英文为主，避免夹杂大量中文解释
    if ascii_letter_ratio(final_prompt) < 0.45:
        return False
    if cjk_ratio(final_prompt) > 0.08:
        return False

    # 样本内部不应出现完全重复消息
    normalized_contents = [normalize_text(m["content"]) for m in messages]
    if len(normalized_contents) != len(set(normalized_contents)):
        return False

    # 至少要有 2 条 user + 2 条 assistant（含最终）
    user_count = sum(1 for m in messages if m["role"] == "user")
    assistant_count = sum(1 for m in messages if m["role"] == "assistant")
    if user_count < 2 or assistant_count < 2:
        return False

    return True


def load_existing(output_path: Path) -> List[Dict]:
    """读取已生成数据，实现断点续传。"""
    if not output_path.exists():
        return []

    records: List[Dict] = []
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                # 遇到坏行直接跳过，保证任务可继续
                continue
    return records


def append_record(output_path: Path, record: Dict) -> None:
    """追加写入单条 JSONL。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_output_path(output: str, worker_id: str) -> Path:
    """根据 worker_id 生成分片文件路径，避免多进程同时写同一个文件。"""
    raw_path = Path(output)
    if raw_path.suffix.lower() == ".jsonl":
        if worker_id:
            return raw_path.with_name(f"{raw_path.stem}.{worker_id}{raw_path.suffix}")
        return raw_path

    target_dir = raw_path
    filename = f"train_data.{worker_id}.jsonl" if worker_id else "train_data.jsonl"
    return target_dir / filename


def is_non_retryable_error(error: Exception) -> bool:
    text = str(error)
    return any(code in text for code in ["Error code: 400", "Error code: 401", "Error code: 403", "Error code: 404", "invalid_request_error"])


def generate_one_sample(client: OpenAI, model_name: str, scene: str, temperature: float) -> Optional[Dict]:
    """调用 API 生成并解析单条样本。"""
    user_instruction = build_user_instruction(scene)

    completion = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": META_PROMPT},
            {"role": "user", "content": user_instruction},
        ],
    )

    content = completion.choices[0].message.content if completion.choices else ""
    obj = extract_json_object(content or "")
    if obj is None:
        return None

    obj = normalize_record(obj, scene)

    if not validate_sample(obj):
        return None

    # 兜底补齐 meta（方便后续分析）
    if "meta" not in obj or not isinstance(obj.get("meta"), dict):
        obj["meta"] = {}
    obj["meta"].setdefault("scene", scene)
    obj["meta"].setdefault("difficulty", random.choice(["easy", "medium", "hard"]))
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="生成多轮对话训练数据（JSONL）")
    parser.add_argument("--target-size", type=int, default=1000, help="目标样本数，默认 1000")
    parser.add_argument("--output", type=str, default="train_data.jsonl", help="输出 JSONL 文件路径；若传目录则自动写入分片")
    parser.add_argument("--worker-id", type=str, default="worker0", help="并行生成时的 worker 标识，用于分片输出")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_NAME", "deepseek-chat"), help="API 模型名，默认 deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.9, help="采样温度")
    parser.add_argument("--max-retries", type=int, default=6, help="单条样本最大重试次数")
    parser.add_argument("--sleep", type=float, default=0.8, help="每次成功调用后 sleep 秒数，避免限流")
    args = parser.parse_args()

    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL", "https://api.deepseek.com/v1")

    if not api_key:
        print("[错误] 未检测到环境变量 API_KEY", file=sys.stderr)
        sys.exit(1)

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    output_path = resolve_output_path(args.output, args.worker_id)

    existing = load_existing(output_path)
    current_size = len(existing)
    print(f"[信息] 已有样本数: {current_size}")
    print(f"[信息] 目标样本数: {args.target_size}")
    print(f"[信息] 当前输出文件: {output_path.resolve()}")

    if current_size >= args.target_size:
        print("[完成] 已达到目标，无需继续生成。")
        return

    # 为了防止重复，做一个轻量级内容签名集合（用最终消息文本）
    seen_final_messages = set()
    for r in existing:
        try:
            final_text = r["messages"][-1]["content"].strip()
            seen_final_messages.add(final_text)
        except Exception:
            continue

    generated = current_size
    while generated < args.target_size:
        scene = random.choice(SCENE_SEEDS)

        success = False
        for attempt in range(1, args.max_retries + 1):
            try:
                sample = generate_one_sample(
                    client=client,
                    model_name=args.model,
                    scene=scene,
                    temperature=args.temperature,
                )
                if sample is None:
                    raise ValueError("样本解析/校验失败")

                final_text = sample["messages"][-1]["content"].strip()
                if final_text in seen_final_messages:
                    # 重复样本，重试
                    raise ValueError("检测到重复样本")

                append_record(output_path, sample)
                seen_final_messages.add(final_text)
                generated += 1
                success = True

                print(f"[进度] {generated}/{args.target_size} (scene={scene})")
                time.sleep(args.sleep)
                break

            except Exception as e:
                if is_non_retryable_error(e):
                    print(f"[错误] 检测到不可重试错误: {e}", file=sys.stderr)
                    sys.exit(1)
                wait_s = min(2 ** attempt, 20)
                print(f"[警告] 第 {attempt} 次尝试失败: {e}; {wait_s}s 后重试")
                time.sleep(wait_s)

        if not success:
            print("[警告] 单条样本连续失败，跳过当前轮次继续下一条。")

    print(f"[完成] 数据生成结束，输出文件: {output_path.resolve()}")


if __name__ == "__main__":
    main()
