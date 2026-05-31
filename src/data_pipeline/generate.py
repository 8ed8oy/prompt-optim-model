import argparse
import random
import sys
import time
from typing import Dict, List, Optional

from openai import OpenAI

from src.prompt_loader import read_prompt

from .core import append_record, extract_json_object, is_non_retryable_error, load_existing, normalize_record, resolve_output_path, validate_sample


SCENE_SEEDS: List[str] = [
    "城市形象宣传片封面图",
    "城市文旅宣传主视觉",
    "文旅宣传海报",
    "文旅短视频竖版封面",
    "城市漫步线路宣传海报",
    "城市夜游品牌宣传封面",
    "城市夜景文旅短视频封面",
    "古城灯会宣传片镜头",
    "沉浸式夜游项目宣传封面",
    "滨水步道慢行系统推广海报",
    "山水风光旅游宣传片镜头脚本",
    "生态旅游宣传片画面",
    "湿地公园科普导览海报",
    "海岛度假目的地海报",
    "茶园采摘体验宣传片镜头",
    "田园综合体宣传视频开场画面",
    "乡村旅游推广海报",
    "乡村民宿品牌推广海报",
    "乡村振兴文旅融合宣传画",
    "四季旅游攻略封面图",
    "古镇保护与传承纪录片海报",
    "江南水乡夜游宣传片分镜",
    "古桥古巷人文纪录片海报",
    "老街区更新展示长图",
    "历史街区焕新宣传片镜头",
    "古村落航拍宣传图",
    "古城慢生活宣传海报",
    "景区开园活动主视觉",
    "主题公园开园宣传封面",
    "游客中心导览主视觉",
    "文旅招商推介会开场视频",
    "文旅局官方公众号头图",
    "研学旅行宣传海报",
    "文博融合活动主视觉",
    "博物馆特展宣传海报",
    "文化遗产数字化展示页面头图",
    "节庆民俗活动宣传海报",
    "非遗文化传承短视频特写镜头",
    "非遗市集活动宣传片封面",
    "传统手工艺体验活动主视觉",
    "地方戏曲惠民演出宣传图",
    "乡愁记忆主题摄影海报",
    "红色文旅线路宣传图",
    "文旅夜市促消费宣传海报",
    "景区直播带货开场画面",
    "文旅品牌发布会开场视频",
    "区域旅游目的地总览海报",
    "城市文旅地图导览长图",
]

META_PROMPT = read_prompt("data_generation_system_prompt.txt")


def build_user_instruction(scene: str) -> str:
    style_bias = random.choice(
        [
            "强调镜头语言与运镜",
            "强调视觉风格与色彩",
            "强调叙事节奏与情绪",
            "强调构图与光影细节",
        ]
    )
    round_hint = random.choice(["3轮往返", "4轮往返"])
    user_profile = random.choice(
        [
            "市级文旅局宣传人员",
            "县级融媒体中心编辑",
            "景区新媒体运营",
            "博物馆宣传策划",
            "非遗活动执行人员",
            "乡村旅游推广负责人",
        ]
    )
    delivery_hint = random.choice(
        [
            "优先适配微信头图和视频号",
            "优先适配抖音/快手竖屏",
            "优先适配公众号首图和长图",
            "优先适配景区大屏和手机端",
        ]
    )
    story_hint = random.choice(
        [
            "若用户是长故事/多阶段动作，最终 assistant JSON 必须包含 3-4 条 scenes 分镜",
            "遇到连续情节时，必须做分镜拆解并保持镜头连贯",
            "非长故事可只输出 prompt，但长故事必须输出 scenes",
        ]
    )
    return (
        f"请模拟一位{user_profile}围绕文旅宣传场景“{scene}”生成一条训练样本；"
        f"{style_bias}；{delivery_hint}；对话长度偏向{round_hint}。"
        f"{story_hint}。"
        "请确保每轮追问都围绕缺失信息推进，至少覆盖发布平台/尺寸、视觉风格、核心元素、情绪基调、镜头语言或构图中的3个维度。"
        "如果信息仍不够，继续追问，不要过早输出最终 JSON。"
        "最后一条 assistant 消息必须是纯 JSON 对象字符串，且 prompt 字段必须是英文、标签化、逗号分隔的结构化写法。"
        "请直接输出 JSON 对象。"
    )


def generate_one_sample(client: OpenAI, model_name: str, scene: str, temperature: float) -> Optional[Dict]:
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

    if "meta" not in obj or not isinstance(obj.get("meta"), dict):
        obj["meta"] = {}
    obj["meta"].setdefault("scene", scene)
    obj["meta"].setdefault("difficulty", random.choice(["easy", "medium", "hard"]))
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="生成多轮对话训练数据（JSONL）")
    parser.add_argument("--target-size", type=int, default=200, help="目标样本数，默认 200")
    parser.add_argument("--output", type=str, default="train_data.jsonl", help="输出 JSONL 文件路径；若传目录则自动写入分片")
    parser.add_argument("--worker-id", type=str, default="worker0", help="并行生成时的 worker 标识，用于分片输出")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="API 模型名，默认 deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.9, help="采样温度")
    parser.add_argument("--max-retries", type=int, default=6, help="单条样本最大重试次数")
    parser.add_argument("--sleep", type=float, default=0.8, help="每次成功调用后 sleep 秒数，避免限流")
    args = parser.parse_args()

    import os

    model_name = os.getenv("MODEL_NAME", args.model)
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

    seen_final_messages = set()
    for record in existing:
        try:
            final_text = record["messages"][-1]["content"].strip()
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
                    model_name=model_name,
                    scene=scene,
                    temperature=args.temperature,
                )
                if sample is None:
                    raise ValueError("样本解析/校验失败")

                final_text = sample["messages"][-1]["content"].strip()
                if final_text in seen_final_messages:
                    raise ValueError("检测到重复样本")

                append_record(output_path, sample)
                seen_final_messages.add(final_text)
                generated += 1
                success = True

                print(f"[进度] {generated}/{args.target_size} (scene={scene})")
                time.sleep(args.sleep)
                break

            except Exception as error:
                if is_non_retryable_error(error):
                    print(f"[错误] 检测到不可重试错误: {error}", file=sys.stderr)
                    sys.exit(1)
                wait_seconds = min(2 ** attempt, 20)
                print(f"[警告] 第 {attempt} 次尝试失败: {error}; {wait_seconds}s 后重试")
                time.sleep(wait_seconds)

        if not success:
            print("[警告] 单条样本连续失败，跳过当前轮次继续下一条。")

    print(f"[完成] 数据生成结束，输出文件: {output_path.resolve()}")


if __name__ == "__main__":
    main()