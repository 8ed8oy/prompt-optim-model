#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估脚本 - 测试训练好的模型在媒体提示词优化任务上的表现

使用方法：
python evaluate.py --adapter-path outputs/qwen25_7b_prompt_optimizer
"""

import argparse
import json
from typing import List, Dict
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

from src.prompt_loader import read_prompt


SYSTEM_PROMPT = read_prompt("evaluation_system_prompt.txt")
FOLLOWUP_ASSISTANT_PROMPT = read_prompt("evaluation_followup_assistant.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="评估训练好的提示词优化模型")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", type=str, default="outputs/qwen25_7b_prompt_optimizer")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def load_model(base_model: str, adapter_path: str):
    """加载4-bit模型和LoRA适配器"""
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, messages: List[Dict[str, str]],
                     max_new_tokens: int, temperature: float, top_p: float) -> str:
    """生成回复"""
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()


def get_test_cases() -> List[Dict]:
    """定义测试用例"""
    system_prompt = SYSTEM_PROMPT

    test_cases = [
        {
            "id": 1,
            "name": "简单请求 - 党建宣传海报",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "我需要一个党建宣传海报"}
            ],
            "expected_behavior": "应该追问具体细节，如发布层级、视觉风格、核心元素等"
        },
        {
            "id": 2,
            "name": "中等复杂度 - 乡村振兴短视频封面",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "乡村振兴主题的短视频封面，要体现希望和活力"}
            ],
            "expected_behavior": "应该追问平台、具体元素、色彩风格等细节"
        },
        {
            "id": 3,
            "name": "已有部分信息 - 城市宣传片",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "市级城市宣传片，要现代感强，体现创新活力"}
            ],
            "expected_behavior": "应该进一步追问具体场景、核心元素、情绪基调等"
        },
        {
            "id": 4,
            "name": "多轮对话测试 - 初始请求",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "人文纪录片海报"}
            ],
            "expected_behavior": "第一轮应追问发布层级、视觉风格等基本信息"
        }
    ]

    # 多轮对话的后续轮次
    multi_turn_followup = {
        "id": 4,
        "name": "多轮对话测试 - 用户补充信息",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "人文纪录片海报"},
            {"role": "assistant", "content": FOLLOWUP_ASSISTANT_PROMPT},
            {"role": "user", "content": "省级卫视出品，用于网络视频平台。希望是电影感纪实风格，核心是手工艺人工作场景。情绪要深沉厚重，体现传承。"}
        ],
        "expected_behavior": "应该进一步追问细节或直接生成优化后的提示词"
    }

    return test_cases


def evaluate_response(response: str, test_case: Dict) -> Dict:
    """评估单个回复的质量"""
    evaluation = {
        "test_case_id": test_case["id"],
        "test_case_name": test_case["name"],
        "response": response,
        "metrics": {}
    }

    # 基本质量指标
    response_lower = response.lower()

    # 1. 是否追问细节（对于初始请求）
    if test_case["id"] in [1, 2, 3, 4] and len(test_case["messages"]) == 2:  # 初始请求
        is_questioning = any(keyword in response_lower for keyword in ["发布", "层级", "平台", "风格", "元素", "情绪", "基调", "什么", "哪些", "如何", "?？"])
        evaluation["metrics"]["is_questioning"] = is_questioning
        evaluation["metrics"]["questioning_score"] = 1.0 if is_questioning else 0.0

    # 2. 回复长度
    evaluation["metrics"]["response_length"] = len(response)

    # 3. 是否包含关键术语
    key_terms = ["提示词", "优化", "海报", "封面", "宣传", "视觉", "构图", "色彩", "光线", "镜头"]
    found_terms = [term for term in key_terms if term in response]
    evaluation["metrics"]["key_terms_found"] = found_terms
    evaluation["metrics"]["key_terms_score"] = len(found_terms) / len(key_terms)

    # 4. 结构完整性（是否分点或分段）
    has_structure = any(marker in response for marker in ["\n1.", "\n2.", "\n3.", "\n- ", "\n* ", "首先", "其次", "最后", "一、", "二、"])
    evaluation["metrics"]["has_structure"] = has_structure
    evaluation["metrics"]["structure_score"] = 1.0 if has_structure else 0.0

    # 5. 专业度评估（简单启发式）
    professional_terms = ["电影感", "纪实", "构图", "色调", "景深", "特写", "中景", "远景", "光线", "阴影", "质感", "层次"]
    professional_count = sum(1 for term in professional_terms if term in response)
    evaluation["metrics"]["professional_terms"] = professional_count
    evaluation["metrics"]["professional_score"] = min(professional_count / 5, 1.0)  # 最多5个术语

    return evaluation


def main():
    args = parse_args()

    print("=" * 80)
    print("开始评估训练好的提示词优化模型")
    print(f"基础模型: {args.base_model}")
    print(f"适配器路径: {args.adapter_path}")
    print("=" * 80)

    # 加载模型
    print("正在加载模型...")
    model, tokenizer = load_model(args.base_model, args.adapter_path)
    print("模型加载完成!")

    # 获取测试用例
    test_cases = get_test_cases()

    all_results = []

    # 运行测试
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"测试用例 {i+1}/{len(test_cases)}: {test_case['name']}")
        print(f"预期行为: {test_case['expected_behavior']}")
        print(f"{'-'*60}")

        print("用户输入:", test_case['messages'][-1]['content'])

        try:
            # 生成回复
            response = generate_response(
                model, tokenizer, test_case["messages"],
                args.max_new_tokens, args.temperature, args.top_p
            )

            print("模型回复:")
            print(response)

            # 评估回复
            evaluation = evaluate_response(response, test_case)
            all_results.append(evaluation)

            # 打印评估结果
            print(f"\n评估结果:")
            for metric_name, metric_value in evaluation["metrics"].items():
                if isinstance(metric_value, (int, float)):
                    print(f"  {metric_name}: {metric_value:.3f}")
                elif isinstance(metric_value, list):
                    print(f"  {metric_name}: {', '.join(metric_value)}")
                else:
                    print(f"  {metric_name}: {metric_value}")

        except Exception as e:
            print(f"测试失败: {e}")
            all_results.append({
                "test_case_id": test_case["id"],
                "test_case_name": test_case["name"],
                "error": str(e)
            })

    # 汇总结果
    print("\n" + "="*80)
    print("评估汇总")
    print("="*80)

    successful_tests = [r for r in all_results if "error" not in r]
    if successful_tests:
        # 计算平均分
        avg_scores = {}
        score_fields = ["questioning_score", "key_terms_score", "structure_score", "professional_score"]

        for field in score_fields:
            scores = [r["metrics"].get(field, 0) for r in successful_tests if field in r["metrics"]]
            if scores:
                avg_scores[field] = sum(scores) / len(scores)

        print(f"完成测试: {len(successful_tests)}/{len(test_cases)}")
        print("\n平均得分:")
        for field, avg_score in avg_scores.items():
            print(f"  {field}: {avg_score:.3f}")

        # 总体评估
        if avg_scores.get("questioning_score", 0) > 0.5:
            print("\n✅ 模型表现出良好的追问能力，能够澄清用户意图")
        else:
            print("\n⚠️  模型的追问能力有待提高")

        if avg_scores.get("professional_score", 0) > 0.3:
            print("✅ 模型回复表现出一定的专业性")
        else:
            print("⚠️  模型的专业性需要加强")

        if avg_scores.get("structure_score", 0) > 0.5:
            print("✅ 模型回复结构清晰")
        else:
            print("⚠️  模型回复结构可以更清晰")

    # 保存结果
    output_file = "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()