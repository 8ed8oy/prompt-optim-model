#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference.py — 低显存 LoRA 推理脚本（适配 8GB 笔记本显卡）

优化要点：
1) 默认不合并 LoRA（Adapter 模式），避免 fp16 合并时的显存峰值
2) --low-vram 模式可限制显存上限（8GB 卡用 7500MiB）
3) CPU offload 兜底

用法:
    # 8GB 显存笔记本（推荐）
    python scripts/inference.py --low-vram

    # 12GB+ 显存
    python scripts/inference.py

    # 批量推理
    python scripts/inference.py --input-file test_queries.txt
"""

import argparse
import gc
import sys
import threading
from os import environ
from pathlib import Path
from typing import Dict, List

# 项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
        candidate_repos = ["unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit", model_name]
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
    parser = argparse.ArgumentParser(description="LoRA 推理（低显存优化）")
    parser.add_argument("--base-model", type=str, default="unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit")
    parser.add_argument("--adapter-path", type=str, default="outputs/qwen25_7b_prompt_optimizer_v2")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--clarify-retries", type=int, default=2)
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    # 显存控制
    parser.add_argument("--low-vram", action="store_true", default=False,
                        help="8GB 显存优化模式：Adapter 模式 + 限制显存上限")
    parser.add_argument("--max-gpu-memory", type=str, default=None,
                        help="显存上限，如 '7GB'、'7000MiB'（--low-vram 时默认 7500MiB）")
    parser.add_argument("--merge-lora", action="store_true", default=False,
                        help="合并 LoRA（需要 10GB+ 显存，不推荐）")
    return parser.parse_args()


def build_bnb_config() -> BitsAndBytesConfig:
    # 使用 float16 以节省 ~200MB 显存
    compute_dtype = torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model_and_tokenizer(
    base_model: str, adapter_path: str, merge_lora: bool,
    max_gpu_memory: str | None = None,
):
    base_model = resolve_local_model_source(base_model)
    print(f"[信息] 基座模型: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        quantization_config=build_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if max_gpu_memory:
        load_kwargs["max_memory"] = {0: max_gpu_memory, "cpu": "16GB"}
        print(f"[信息] 显存上限: {max_gpu_memory}")

    base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    model = PeftModel.from_pretrained(base, adapter_path)

    if merge_lora:
        try:
            model = model.merge_and_unload()
            print("[信息] 已合并 LoRA 权重。")
        except Exception as e:
            print(f"[警告] LoRA 合并失败，以 Adapter 模式运行: {e}")

    model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[信息] 模型显存: {allocated:.1f}GB 已分配 / {reserved:.1f}GB 已保留")

    return model, tokenizer


def stream_generate(model, tokenizer, history, temperature, top_p, max_new_tokens) -> str:
    chat_inputs = tokenizer.apply_chat_template(
        history, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    if isinstance(chat_inputs, torch.Tensor):
        input_ids = chat_inputs
    elif hasattr(chat_inputs, "keys"):
        input_ids = chat_inputs["input_ids"]
    else:
        raise TypeError(f"Unexpected type: {type(chat_inputs)}")

    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=input_ids, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature, top_p=top_p,
        repetition_penalty=1.1, streamer=streamer,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("助手: ", end="", flush=True)
    chunks: List[str] = []
    for new_text in streamer:
        print(new_text, end="", flush=True)
        chunks.append(new_text)
    print()
    thread.join()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return "".join(chunks).strip()


def is_final_json(text: str) -> bool:
    return extract_final_payload(text) is not None


def looks_like_clarifying_question(text: str) -> bool:
    return any(hint in text for hint in QUESTION_HINTS) if text.strip() else False


def generate_task_turn(model, tokenizer, history, temperature, top_p,
                       max_new_tokens, clarify_retries) -> str:
    gen_history = history
    assistant_text = ""
    for attempt in range(clarify_retries + 1):
        assistant_text = stream_generate(
            model, tokenizer, gen_history, temperature, top_p, max_new_tokens,
        )
        if is_final_json(assistant_text):
            print("[信息] 模型已输出最终 JSON。")
            break
        if looks_like_clarifying_question(assistant_text):
            print("[信息] 模型处于追问阶段，等待用户补充信息。")
            break
        if attempt < clarify_retries:
            print("[警告] 本轮未形成明确追问或最终 JSON，补充约束后重试。")
            gen_history = gen_history + [{"role": "system", "content": CLARIFY_REMINDER}]
    return assistant_text


def main() -> None:
    args = parse_args()

    max_gpu_memory = args.max_gpu_memory
    merge_lora = args.merge_lora
    if args.low_vram:
        merge_lora = False
        if max_gpu_memory is None:
            max_gpu_memory = "7500MiB"
        print("[信息] 低显存模式：Adapter + float16 + 显存限制 7500MiB")

    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.adapter_path, merge_lora, max_gpu_memory,
    )

    history: List[Dict[str, str]] = [{"role": "system", "content": args.system_prompt}]
    print("=" * 72)
    print("多轮对话测试已启动。输入 quit/exit 退出，输入 clear 清空历史。")
    print(f"模式: {'LoRA Adapter' if not merge_lora else 'LoRA Merged'}"
          f" | 显存限制: {max_gpu_memory or '无'}")
    print("=" * 72)

    input_lines = None
    if args.input_file:
        input_path = Path(args.input_file)
        if input_path.exists():
            input_lines = [l.strip() for l in input_path.read_text(encoding="utf-8").splitlines()]
        else:
            print(f"[警告] 输入文件不存在：{input_path}")

    line_iter = iter(input_lines) if input_lines is not None else None

    while True:
        if line_iter is None:
            try:
                user_text = input("用户: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[信息] 会话结束。")
                break
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
            model, tokenizer, history, args.temperature, args.top_p,
            args.max_new_tokens, args.clarify_retries,
        )
        history.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
