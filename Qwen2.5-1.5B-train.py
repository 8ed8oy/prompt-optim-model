#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train.py

用途：
使用 QLoRA（4-bit + LoRA）在 Qwen2.5-1.5B-Instruct 上进行 SFT 微调。
设计目标：12GB 显存环境可稳定运行，并支持断点恢复训练。

依赖：
  pip install torch transformers datasets peft trl bitsandbytes accelerate

示例：
  python train.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --train-file train_data.jsonl \
    --output-dir outputs/qwen25_prompt_optimizer \
    --max-seq-length 1024 \
    --num-train-epochs 3
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT 训练脚本（12GB 显存优化）")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-file", type=str, default="train_data.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/qwen25_prompt_optimizer")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-gradient-checkpointing", action="store_true", default=True)
    return parser.parse_args()


def load_jsonl_messages(train_file: str) -> Dataset:
    """加载 train_data.jsonl，提取 messages 字段并转为 Hugging Face Dataset。"""
    path = Path(train_file)
    if not path.exists():
        raise FileNotFoundError(f"训练数据文件不存在: {train_file}")

    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" not in obj or not isinstance(obj["messages"], list):
                continue
            records.append({"messages": obj["messages"]})

    if not records:
        raise ValueError("未读取到有效训练样本，请检查 train_data.jsonl 格式")

    return Dataset.from_list(records)


def build_text_dataset(raw_dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """
    使用 tokenizer.apply_chat_template 将 messages 转为可训练文本字段 text。
    这样可以兼容多数 TRL 版本，不依赖额外的数据整理器。
    """

    def _to_text(example: Dict) -> Dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return raw_dataset.map(_to_text, remove_columns=raw_dataset.column_names)


def maybe_get_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    自动检测 output_dir 中最新 checkpoint。
    目录形如 checkpoint-100, checkpoint-200 ...
    """
    path = Path(output_dir)
    if not path.exists():
        return None

    checkpoints = []
    for p in path.glob("checkpoint-*"):
        if p.is_dir():
            m = re.search(r"checkpoint-(\d+)$", p.name)
            if m:
                checkpoints.append((int(m.group(1)), str(p)))

    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def build_bnb_config() -> BitsAndBytesConfig:
    """构建 4-bit 量化配置，优先使用 NF4，计算精度优先 bf16。"""
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    # 1) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) 4-bit 量化加载模型
    bnb_config = build_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3) 显存优化关键配置
    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 让输入嵌入支持梯度，配合 k-bit 训练
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # 4) LoRA 配置（重点注入注意力投影层）
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # 5) 数据集
    raw_dataset = load_jsonl_messages(args.train_file)
    train_dataset = build_text_dataset(raw_dataset, tokenizer)

    # 6) TrainingArguments（12GB 显存保守配置）
    bf16_flag = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=bf16_flag,
        fp16=not bf16_flag,
        max_grad_norm=1.0,
        weight_decay=0.01,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=args.use_gradient_checkpointing,
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    # 8) 自动断点恢复
    latest_ckpt = maybe_get_latest_checkpoint(args.output_dir)
    if latest_ckpt:
        print(f"[信息] 检测到 checkpoint，恢复训练: {latest_ckpt}")
        trainer.train(resume_from_checkpoint=latest_ckpt)
    else:
        print("[信息] 未检测到 checkpoint，从头开始训练")
        trainer.train()

    # 9) 保存最终 adapter + tokenizer
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[完成] 训练结束，模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
