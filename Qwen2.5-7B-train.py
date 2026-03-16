#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train.py

用途：
使用 Unsloth + QLoRA（4-bit + LoRA）在 Qwen2.5-7B-Instruct 上进行 SFT 微调。
设计目标：在有限显存环境稳定运行，并支持断点恢复训练。

依赖：
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    pip install unsloth datasets trl accelerate
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import unsloth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT 训练脚本（Qwen2.5-7B 显存优化）")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-file", type=str, default="train_data.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/qwen25_7b_prompt_optimizer")
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=18)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
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


def build_text_dataset(raw_dataset: Dataset, tokenizer) -> Dataset:
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


def main() -> None:
    args = parse_args()

    try:
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]
    except (ImportError, TypeError) as exc:
        import sys

        python_ver = sys.version_info
        hint = (
            f"\n当前 Python {python_ver.major}.{python_ver.minor}，"
            "unsloth 要求 Python >= 3.10。\n"
            "请用 Python 3.11 重建环境：\n"
            "  conda create -n prompt-opt python=3.11 -y\n"
            "  conda activate prompt-opt\n"
            "  pip install unsloth -i https://pypi.tuna.tsinghua.edu.cn/simple"
        )
        raise RuntimeError(f"无法导入 unsloth：{exc}{hint}") from exc

    torch.manual_seed(args.seed)

    # 1) 使用 Unsloth 载入 4-bit 模型 + tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) LoRA 注入（由 Unsloth 管理 k-bit 训练细节）
    gradient_checkpointing_mode = "unsloth" if args.use_gradient_checkpointing else False
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=gradient_checkpointing_mode,
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )

    # 3) 训练稳定性设置
    model.config.use_cache = False

    # 4) 数据集
    raw_dataset = load_jsonl_messages(args.train_file)
    train_dataset = build_text_dataset(raw_dataset, tokenizer)

    # 5) SFTConfig（12G 显存保守配置，兼容 TRL 0.24+）
    bf16_flag = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = SFTConfig(
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
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
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
