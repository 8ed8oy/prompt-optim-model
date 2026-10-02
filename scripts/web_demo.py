#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
web_demo.py — Gradio 网页演示（适配 8GB 笔记本显卡）

启动后浏览器访问 http://127.0.0.1:7860

用法:
    # 低显存模式
    python scripts/web_demo.py --low-vram

    # 12GB+ 显存
    python scripts/web_demo.py

依赖: pip install gradio
"""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.data_pipeline.core import extract_final_payload
from src.prompt_loader import read_prompt


# ── 全局模型 ──────────────────────────────────────────────
_model = None
_tokenizer = None
_system_prompt = read_prompt("inference_system_prompt.txt")


def load_model(base_model: str, adapter_path: str, max_gpu_memory: str | None):
    """加载 4-bit 模型 + LoRA adapter（不合并）"""
    global _model, _tokenizer

    # 使用 float16 计算以节省 ~200MB 显存（bf16 虽精度更高但更占显存）
    compute_dtype = torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    _tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    load_kwargs = dict(
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if max_gpu_memory:
        load_kwargs["max_memory"] = {0: max_gpu_memory, "cpu": "16GB"}

    base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    _model = PeftModel.from_pretrained(base, adapter_path)
    _model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[信息] 显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB 已分配")


def chat_gradio(message: str, history: list) -> tuple:
    """Gradio 6 回调：history 是 list[dict] 格式，每个 dict 含 role + content"""
    if _model is None:
        return "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "模型未加载，请先启动服务。"},
        ]

    # 构建 messages 供 tokenizer 使用
    messages = [{"role": "system", "content": _system_prompt}]
    for h in (history or []):
        if isinstance(h, dict):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    # Tokenize
    chat_inputs = _tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    input_ids = chat_inputs if isinstance(chat_inputs, torch.Tensor) else chat_inputs["input_ids"]
    input_ids = input_ids.to(next(_model.parameters()).device)

    # Generate（限制 max_new_tokens 以节省显存）
    with torch.no_grad():
        output_ids = _model.generate(
            input_ids, max_new_tokens=384, do_sample=True,
            temperature=0.7, top_p=0.9, repetition_penalty=1.1,
            eos_token_id=_tokenizer.eos_token_id,
            pad_token_id=_tokenizer.pad_token_id,
        )

    response = _tokenizer.decode(
        output_ids[0][len(input_ids[0]):], skip_special_tokens=True,
    ).strip()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # JSON 格式化显示
    payload = extract_final_payload(response)
    if payload:
        prompt_text = payload.get("prompt", "")
        scenes = payload.get("scenes")
        parts = [
            "### ✅ 最终提示词",
            "",
            "**主 Prompt：**",
            "```",
            prompt_text,
            "```",
        ]
        if scenes:
            parts.append("")
            parts.append("**分镜 Scenes：**")
            for s in scenes:
                parts.append(f"- **Scene {s['id']}**: {s['prompt'][:150]}...")
        response = "\n".join(parts)

    # 使用 dict 格式（Gradio 6 要求）
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio 网页演示")
    parser.add_argument("--base-model", type=str, default="unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit")
    parser.add_argument("--adapter-path", type=str, default="outputs/qwen25_7b_prompt_optimizer_v2")
    parser.add_argument("--low-vram", action="store_true", default=False)
    parser.add_argument("--max-gpu-memory", type=str, default=None)
    parser.add_argument("--share", action="store_true", default=False, help="生成公网链接")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main():
    args = parse_args()

    max_mem = args.max_gpu_memory
    if args.low_vram and max_mem is None:
        max_mem = "7500MiB"
        print("[信息] 低显存模式：显存上限 7500MiB + float16 计算")

    print("[信息] 正在加载模型...")
    load_model(args.base_model, args.adapter_path, max_mem)
    print("[信息] 模型加载完成！")

    with gr.Blocks(title="媒体提示词优化助手") as demo:
        gr.Markdown("# 媒体提示词优化助手")
        gr.Markdown(
            "输入模糊的文生图/视频需求，AI 会先追问澄清细节，"
            "然后生成专业的英文标签化提示词。"
        )
        chatbot = gr.Chatbot(height=500, render_markdown=True)
        msg = gr.Textbox(placeholder="请输入您的媒体创作需求...", scale=4)
        clear = gr.ClearButton([msg, chatbot])

        gr.Examples(
            examples=[
                "我需要一张城市宣传海报",
                "帮我做一个非遗文化传承的短视频",
                "需要一张乡村振兴宣传图，要有希望和活力",
                "设计一个江南水乡夜游的宣传片分镜",
            ],
            inputs=msg,
        )

        msg.submit(chat_gradio, [msg, chatbot], [msg, chatbot])

    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
