#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
generate_data.py

用途：
1) 使用 OpenAI 兼容接口（如 GPT-4o / Qwen-Max）合成多轮对话训练数据。
2) 输出为 Hugging Face chat template 可直接消费的 JSONL（messages 列表）。
3) 支持断点续传：如果中断，重新运行会自动接着生成，直到达到目标条数。

运行前环境变量：
- API_KEY: 必填，模型服务的密钥
- BASE_URL: 选填，OpenAI 兼容网关地址（如阿里云/其他代理）
- MODEL_NAME: 选填，默认 gpt-4o-mini（可改为 qwen-max 等）

示例：
  set API_KEY=xxxx
  set BASE_URL=https://api.openai.com/v1
  set MODEL_NAME=gpt-4o
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
    "商业广告短片（美妆）",
    "商业广告短片（汽车）",
    "科幻城市夜景短视频",
    "体育赛事集锦开场画面",
    "人文纪录片海报",
    "财经栏目片头视觉",
    "美食探店短视频",
    "旅游宣传片镜头脚本",
    "医疗科普栏目主视觉",
    "教育课程宣传视频",
    "公益宣传短片",
    "时尚大片封面图",
    "游戏宣传 CG 分镜",
    "影视预告片关键镜头",
    "天气新闻播报背景图",
    "电商直播间产品展示图",
    "企业品牌发布会开场视频",
    "城市形象宣传片",
    "音乐 MV 概念镜头",
]


# -----------------------------
# 2) Meta Prompt：要求模型生成规范 JSON
# -----------------------------
META_PROMPT = """
你是“数据合成引擎”。请为“面向媒体领域的文生图/视频提示词优化模型”生成一条训练样本。

严格要求：
1. 必须输出 **单个 JSON 对象**，不要输出 Markdown 代码块，不要任何额外说明。
2. 对话轮次：2-3 轮往返（即用户与助手至少 5 条消息，最多 7 条消息，最后一条必须是助手）。
3. 对话逻辑：
   - 用户先给出模糊意图（媒体内容创作相关）。
   - 助手追问关键创作参数（例如风格、镜头语言、光影、时长、画幅比例、情绪、主体细节等）。
   - 用户补充约束与偏好。
   - 助手最终输出专业英文 Prompt，并包含“【最终优化提示词】”标签。
4. 最终助手消息中：
   - 必须有“【最终优化提示词】”这段中文标签。
   - 标签后是高质量英文 prompt，结构清晰，适用于文生图/视频模型（包含场景、主体、镜头、光线、风格、色彩、构图、技术参数等）。
5. 角色仅允许：system / user / assistant。
6. JSON 必须符合以下结构：
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ],
  "meta": {
    "scene": "...",
    "difficulty": "easy|medium|hard"
  }
}
7. system 消息建议描述助手职责：你是提示词优化专家，擅长多轮追问。
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
        "请直接输出 JSON 对象。"
    )


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

    # 最后一条必须是 assistant
    if messages[-1].get("role") != "assistant":
        return False

    # 最终 assistant 文本必须带标签
    if "【最终优化提示词】" not in messages[-1].get("content", ""):
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
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    parser.add_argument("--output", type=str, default="train_data.jsonl", help="输出 JSONL 文件路径")
    parser.add_argument("--model", type=str, default=os.getenv("MODEL_NAME", "gpt-4o-mini"), help="API 模型名")
    parser.add_argument("--temperature", type=float, default=0.9, help="采样温度")
    parser.add_argument("--max-retries", type=int, default=6, help="单条样本最大重试次数")
    parser.add_argument("--sleep", type=float, default=0.8, help="每次成功调用后 sleep 秒数，避免限流")
    args = parser.parse_args()

    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        print("[错误] 未检测到环境变量 API_KEY", file=sys.stderr)
        sys.exit(1)

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    output_path = Path(args.output)

    existing = load_existing(output_path)
    current_size = len(existing)
    print(f"[信息] 已有样本数: {current_size}")
    print(f"[信息] 目标样本数: {args.target_size}")

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
                wait_s = min(2 ** attempt, 20)
                print(f"[警告] 第 {attempt} 次尝试失败: {e}; {wait_s}s 后重试")
                time.sleep(wait_s)

        if not success:
            print("[警告] 单条样本连续失败，跳过当前轮次继续下一条。")

    print(f"[完成] 数据生成结束，输出文件: {output_path.resolve()}")


if __name__ == "__main__":
    main()
