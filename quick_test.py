#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 检查模型文件完整性和基本功能

这个脚本只检查文件是否存在和配置是否正确，不加载完整模型。
"""

import json
import os
from pathlib import Path


def check_adapter_files(adapter_path: str) -> dict:
    """检查适配器文件完整性"""
    path = Path(adapter_path)
    results = {
        "adapter_path": str(path),
        "exists": path.exists(),
        "files": {},
        "issues": []
    }

    if not path.exists():
        results["issues"].append(f"适配器路径不存在: {adapter_path}")
        return results

    # 检查必要文件
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json"
    ]

    optional_files = [
        "chat_template.jinja",
        "README.md"
    ]

    for file in required_files:
        file_path = path / file
        results["files"][file] = {
            "path": str(file_path),
            "exists": file_path.exists(),
            "size": file_path.stat().st_size if file_path.exists() else 0
        }
        if not file_path.exists():
            results["issues"].append(f"缺失必要文件: {file}")

    for file in optional_files:
        file_path = path / file
        results["files"][file] = {
            "path": str(file_path),
            "exists": file_path.exists(),
            "size": file_path.stat().st_size if file_path.exists() else 0
        }

    # 检查检查点目录
    checkpoint_dirs = [d for d in path.glob("checkpoint-*") if d.is_dir()]
    results["checkpoints"] = {
        "count": len(checkpoint_dirs),
        "directories": [str(d) for d in checkpoint_dirs]
    }

    if checkpoint_dirs:
        # 检查最新检查点
        latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0)[-1]
        results["latest_checkpoint"] = str(latest_checkpoint)

        # 检查检查点内的文件
        checkpoint_files = list(latest_checkpoint.glob("*"))
        results["checkpoint_files"] = [str(f) for f in checkpoint_files]

    return results


def check_training_config(adapter_path: str) -> dict:
    """检查训练配置"""
    path = Path(adapter_path)
    results = {
        "training_config": {}
    }

    config_file = path / "adapter_config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 提取关键配置
            key_configs = [
                "r", "lora_alpha", "lora_dropout",
                "target_modules", "base_model_name_or_path",
                "bias", "fan_in_fan_out"
            ]

            for key in key_configs:
                if key in config:
                    results["training_config"][key] = config[key]

            # 检查LoRA配置
            if "r" in config:
                results["training_config"]["lora_rank"] = config["r"]
            if "lora_alpha" in config:
                results["training_config"]["lora_alpha"] = config["lora_alpha"]

        except Exception as e:
            results["issues"] = [f"读取配置文件失败: {e}"]

    return results


def check_tokenizer_config(adapter_path: str) -> dict:
    """检查分词器配置"""
    path = Path(adapter_path)
    results = {
        "tokenizer_config": {}
    }

    config_file = path / "tokenizer_config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 提取关键配置
            key_configs = [
                "model_max_length", "padding_side",
                "bos_token", "eos_token", "pad_token"
            ]

            for key in key_configs:
                if key in config:
                    results["tokenizer_config"][key] = config[key]

        except Exception as e:
            results["issues"] = [f"读取分词器配置失败: {e}"]

    return results


def analyze_training_logs(adapter_path: str) -> dict:
    """分析训练日志"""
    path = Path(adapter_path)
    results = {
        "training_logs": {}
    }

    # 查找训练日志文件
    trainer_state_files = list(path.rglob("trainer_state.json"))

    if trainer_state_files:
        latest_file = sorted(trainer_state_files, key=lambda x: x.stat().st_mtime)[-1]

        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                trainer_state = json.load(f)

            # 提取关键训练信息
            if "log_history" in trainer_state:
                logs = trainer_state["log_history"]
                if logs:
                    # 计算损失变化
                    first_loss = logs[0].get("loss", None)
                    last_loss = logs[-1].get("loss", None)

                    results["training_logs"]["loss_progress"] = {
                        "first_loss": first_loss,
                        "last_loss": last_loss,
                        "loss_reduction": None
                    }

                    if first_loss and last_loss:
                        reduction = (first_loss - last_loss) / first_loss * 100
                        results["training_logs"]["loss_progress"]["loss_reduction"] = f"{reduction:.1f}%"

            # 训练统计
            stats_keys = ["global_step", "num_train_epochs", "total_flos", "train_batch_size"]
            for key in stats_keys:
                if key in trainer_state:
                    results["training_logs"][key] = trainer_state[key]

        except Exception as e:
            results["issues"] = [f"读取训练日志失败: {e}"]

    return results


def main():
    adapter_path = "outputs/qwen25_7b_prompt_optimizer"

    print("=" * 80)
    print("快速测试 - 提示词优化模型训练效果评估")
    print("=" * 80)

    print(f"\n1. 检查适配器文件完整性: {adapter_path}")
    print("-" * 60)

    adapter_check = check_adapter_files(adapter_path)

    if not adapter_check["exists"]:
        print(f"[错误] 适配器路径不存在: {adapter_path}")
        return

    print(f"[成功] 适配器路径存在")

    # 检查文件
    for file_name, file_info in adapter_check["files"].items():
        if file_info["exists"]:
            size_mb = file_info["size"] / (1024 * 1024)
            print(f"   [成功] {file_name}: {size_mb:.2f} MB")
        else:
            if file_name in ["adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"]:
                print(f"   [错误] {file_name}: 缺失")
            else:
                print(f"   [警告] {file_name}: 缺失 (可选文件)")

    # 检查点
    print(f"\n   检查点数量: {adapter_check['checkpoints']['count']}")
    if adapter_check['checkpoints']['count'] > 0:
        print(f"   最新检查点: {adapter_check.get('latest_checkpoint', 'N/A')}")

    print(f"\n2. 检查训练配置")
    print("-" * 60)

    config_check = check_training_config(adapter_path)
    if "training_config" in config_check and config_check["training_config"]:
        config = config_check["training_config"]

        if "lora_rank" in config:
            print(f"   [成功] LoRA rank (r): {config['lora_rank']}")
        if "lora_alpha" in config:
            print(f"   [成功] LoRA alpha: {config['lora_alpha']}")
        if "target_modules" in config:
            modules = config['target_modules']
            print(f"   [成功] 目标模块: {len(modules)}个模块")
            print(f"      {', '.join(modules[:5])}{'...' if len(modules) > 5 else ''}")

    print(f"\n3. 检查分词器配置")
    print("-" * 60)

    tokenizer_check = check_tokenizer_config(adapter_path)
    if "tokenizer_config" in tokenizer_check and tokenizer_check["tokenizer_config"]:
        for key, value in tokenizer_check["tokenizer_config"].items():
            print(f"   [成功] {key}: {value}")

    print(f"\n4. 分析训练日志")
    print("-" * 60)

    log_analysis = analyze_training_logs(adapter_path)
    if "training_logs" in log_analysis and log_analysis["training_logs"]:
        logs = log_analysis["training_logs"]

        if "global_step" in logs:
            print(f"   [成功] 训练步数: {logs['global_step']}")
        if "num_train_epochs" in logs:
            print(f"   [成功] 训练周期: {logs['num_train_epochs']}")
        if "train_batch_size" in logs:
            print(f"   [成功] 批次大小: {logs['train_batch_size']}")

        if "loss_progress" in logs:
            loss_info = logs["loss_progress"]
            if loss_info["first_loss"] and loss_info["last_loss"]:
                print(f"   [成功] 损失变化: {loss_info['first_loss']:.3f} → {loss_info['last_loss']:.3f}")
                if loss_info["loss_reduction"]:
                    print(f"   [成功] 损失下降: {loss_info['loss_reduction']}")

    print(f"\n5. 问题总结")
    print("-" * 60)

    all_issues = []
    if adapter_check.get("issues"):
        all_issues.extend(adapter_check["issues"])
    if config_check.get("issues"):
        all_issues.extend(config_check["issues"])
    if tokenizer_check.get("issues"):
        all_issues.extend(tokenizer_check["issues"])
    if log_analysis.get("issues"):
        all_issues.extend(log_analysis["issues"])

    if all_issues:
        for issue in all_issues:
            print(f"   [错误] {issue}")
    else:
        print("   [成功] 未发现问题")

    print(f"\n" + "="*80)
    print("评估总结")
    print("="*80)

    # 总体评估
    has_critical_files = all(
        adapter_check["files"][file]["exists"]
        for file in ["adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"]
    )

    has_checkpoints = adapter_check["checkpoints"]["count"] > 0
    has_training_logs = "training_logs" in log_analysis and log_analysis["training_logs"]

    if has_critical_files and has_checkpoints:
        print("[成功] 模型文件完整，可以用于推理")

        if has_training_logs:
            logs = log_analysis["training_logs"]
            if "loss_progress" in logs and logs["loss_progress"].get("loss_reduction"):
                reduction = logs["loss_progress"]["loss_reduction"]
                print(f"[成功] 训练有效，损失下降: {reduction}")

                # 根据损失下降给出评估
                try:
                    reduction_pct = float(reduction.strip('%'))
                    if reduction_pct > 50:
                        print("[成功] 训练效果优秀")
                    elif reduction_pct > 30:
                        print("[成功] 训练效果良好")
                    else:
                        print("[警告]  训练效果一般，建议进一步优化")
                except:
                    pass
    else:
        print("[错误] 模型文件不完整，无法进行推理")

    print(f"\n下一步建议:")
    print("1. 运行推理测试: python inference.py")
    print("2. 查看详细评估指南: manual_evaluation_guide.md")
    print("3. 使用评估脚本: python evaluate.py")


if __name__ == "__main__":
    main()