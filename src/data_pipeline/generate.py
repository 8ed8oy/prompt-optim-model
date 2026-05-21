import argparse
import random
import sys
import time
from typing import Dict, List, Optional

from openai import OpenAI

from src.prompt_loader import read_prompt

from .core import append_record, extract_json_object, is_non_retryable_error, load_existing, normalize_record, resolve_output_path, validate_sample


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
    "全国两会期间地方政府成就展板背景图",
    "城市政务APP开屏科技感海报",
    "‘最多跑一次’政务服务大厅宣传片分镜",
    "廉政文化主题微电影片头视觉",
    "地方公安干警/交警风雪中执勤纪实封面",
    "七一建党节微信公众号首图（大气红底金字风格）",
    "基层党支部活动室文化墙背景插画",
    "‘学习强国’专栏文章配图（历史厚重感）",
    "优秀共产党员先进事迹汇报短视频开场",
    "国有企业年度社会责任报告封面大图",
    "‘一带一路’十周年港口货运繁忙景象无人机视角",
    "重点工程（桥梁/高铁/基建）通车仪式宣传片镜头",
    "高新技术产业园航拍夜景视频素材",
    "乡村振兴主题纪录片海报（金黄麦浪与新农人）",
    "地方非遗文化传承短视频特写镜头",
    "绿水青山就是金山银山（两山理论）生态宣传画",
    "县域文旅局长‘变装’宣传片转场分镜",
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
    "枫桥经验60周年纪念活动海报",
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
    story_hint = random.choice(
        [
            "若用户是长故事/多阶段动作，最终 assistant JSON 必须包含 3-4 条 scenes 分镜",
            "遇到连续情节时，必须做分镜拆解并保持镜头连贯",
            "非长故事可只输出 prompt，但长故事必须输出 scenes",
        ]
    )
    return (
        f"请围绕场景“{scene}”生成一条训练样本；{style_bias}；对话长度偏向{round_hint}。"
        f"{story_hint}。"
        "请确保最后一条 assistant 消息是 JSON 对象字符串，且 prompt 字段必须是英文结构化标签写法。"
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