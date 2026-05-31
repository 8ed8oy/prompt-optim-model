#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
inference.py

用途：
1) 加载 4-bit 基座模型 + LoRA adapter。
2) 支持（可选）合并 LoRA 权重到模型。
3) 在终端中进行多轮交互式对话，并流式输出生成结果。

示例：
    python inference.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --adapter-path outputs/qwen25_7b_prompt_optimizer \
    --max-new-tokens 384
"""

import argparse
import threading
from os import environ
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from src.data_pipeline.core import extract_final_payload
from src.prompt_loader import read_prompt


DEFAULT_SYSTEM_PROMPT = read_prompt("inference_system_prompt.txt")
CLARIFY_REMINDER = (
    "如果信息仍然不足，请只追问用户需要补充的关键信息；"
    "如果信息已经足够，请直接输出最终 JSON，不要输出解释性文字。"
)
QUESTION_HINTS = ("?", "？", "请补充", "需要", "是否", "哪些", "什么", "如何", "几", "还是")


def resolve_local_model_source(model_name: str) -> str:
    if Path(model_name).exists():
        return model_name

    cache_root = Path(environ.get("HF_HUB_CACHE") or (Path.home() / ".cache" / "huggingface" / "hub"))
    candidate_repos = [model_name]
    if model_name == "Qwen/Qwen2.5-7B-Instruct":
        candidate_repos = [
            "unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit",
            model_name,
        ]

    for repo in candidate_repos:
        repo_dir = cache_root / f"models--{repo.replace('/', '--')}" / "snapshots"
        if not repo_dir.exists():
            continue
        snapshots = [p for p in repo_dir.iterdir() if p.is_dir()]
        if snapshots:
            snapshots.sort(key=lambda p: p.name)
            return str(snapshots[-1])

    return model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA 推理 + 多轮终端对话")
    parser.add_argument("--base-model", type=str, default="unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit")
    parser.add_argument("--adapter-path", type=str, default="outputs/qwen25_7b_prompt_optimizer")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="可选：按行读取测试输入，便于非交互式批量推理",
    )
    parser.add_argument(
        "--clarify-retries",
        type=int,
        default=2,
        help="当模型未输出明确追问或最终 JSON 时，自动补充约束并重试的次数",
    )
    parser.add_argument("--merge-lora", action="store_true", default=True, help="是否尝试合并 LoRA 权重（默认开启）")
    parser.add_argument("--no-merge-lora", action="store_false", dest="merge_lora", help="关闭 LoRA 合并，直接以 Adapter 方式推理")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
    )
    return parser.parse_args()


def build_bnb_config() -> BitsAndBytesConfig:
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model_and_tokenizer(base_model: str, adapter_path: str, merge_lora: bool):
    base_model = resolve_local_model_source(base_model)
    print(f"[信息] 基座模型: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=build_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, adapter_path)

    # 注意：4-bit 下 merge_and_unload 可能受限，这里做“尽力合并”并兜底
    if merge_lora:
        try:
            model = model.merge_and_unload()
            print("[信息] 已成功合并 LoRA 权重。")
        except Exception as e:
            print(f"[警告] LoRA 合并失败，将以未合并方式推理: {e}")

    model.eval()
    return model, tokenizer


def stream_generate(
    model,
    tokenizer,
    history: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    """基于当前 history 生成 assistant 回复，并以流式方式打印。"""
    chat_inputs = tokenizer.apply_chat_template(
        history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # 兼容不同 transformers/tokenizer 版本的返回类型：Tensor / BatchEncoding / dict
    if isinstance(chat_inputs, torch.Tensor):
        model_inputs = {"input_ids": chat_inputs}
    elif hasattr(chat_inputs, "keys") and hasattr(chat_inputs, "__getitem__"):
        model_inputs = {
            key: value
            for key, value in chat_inputs.items()
            if isinstance(value, torch.Tensor)
        }
    else:
        raise TypeError(
            f"apply_chat_template 返回了不支持的类型: {type(chat_inputs)}"
        )

    if "input_ids" not in model_inputs:
        raise ValueError("apply_chat_template 返回结果中缺少 input_ids")

    # 如果模型分布在多卡，取第一个参数所在设备；单卡时即 cuda:0
    model_device = next(model.parameters()).device
    model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.1,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generation_kwargs.update(model_inputs)

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("助手: ", end="", flush=True)
    chunks: List[str] = []
    for new_text in streamer:
        print(new_text, end="", flush=True)
        chunks.append(new_text)
    print()

    thread.join()
    return "".join(chunks).strip()


def is_final_json(text: str) -> bool:
    return extract_final_payload(text) is not None


def looks_like_clarifying_question(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    return any(hint in normalized for hint in QUESTION_HINTS)


def generate_task_turn(
    model,
    tokenizer,
    history: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    clarify_retries: int,
) -> str:
    generation_history = history
    assistant_text = ""

    for attempt in range(clarify_retries + 1):
        assistant_text = stream_generate(
            model=model,
            tokenizer=tokenizer,
            history=generation_history,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        if is_final_json(assistant_text):
            print("[信息] 模型已输出最终 JSON。")
            break

        if looks_like_clarifying_question(assistant_text):
            print("[信息] 模型处于追问阶段，等待用户补充信息。")
            break

        if attempt < clarify_retries:
            print("[警告] 本轮回复未形成明确追问或最终 JSON，正在补充约束后重试。")
            generation_history = generation_history + [{"role": "system", "content": CLARIFY_REMINDER}]

    return assistant_text


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_path, args.merge_lora)

    history: List[Dict[str, str]] = [{"role": "system", "content": args.system_prompt}]

    print("=" * 72)
    print("多轮对话测试已启动。输入 quit/exit 退出，输入 clear 清空历史。")
    print("当前启用自动追问收敛：信息不足时会尝试补充约束并重试，直到模型给出明确追问或最终 JSON。")
    print("=" * 72)

    input_lines = None
    if args.input_file:
        input_path = Path(args.input_file)
        if input_path.exists():
            input_lines = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines()]
        else:
            print(f"[警告] 输入文件不存在：{input_path}，将进入交互模式。")

    line_iter = iter(input_lines) if input_lines is not None else None

    while True:
        if line_iter is None:
            user_text = input("用户: ").strip()
        else:
            try:
                user_text = next(line_iter)
            except StopIteration:
                break

        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            print("[信息] 会话结束。")
            break
        if user_text.lower() == "clear":
            history = [{"role": "system", "content": args.system_prompt}]
            print("[信息] 历史已清空。")
            continue

        history.append({"role": "user", "content": user_text})

        assistant_text = generate_task_turn(
            model=model,
            tokenizer=tokenizer,
            history=history,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            clarify_retries=args.clarify_retries,
        )

        history.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
