#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
便捷数据生成脚本 — 读取 .env 中的 API 凭据，调用 DeepSeek API 生成训练数据。

用法:
    conda activate prompt-opt
    python run_data_generation.py --target-size 200 --output data --sleep 0.8
"""

import argparse
import os
import sys


def load_env_from_powershell_format(env_file: str = ".env") -> None:
    """Load environment variables from a PowerShell-formatted .env file."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), env_file)
    if not os.path.exists(env_path):
        print(f"[警告] 未找到 .env 文件: {env_path}")
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            # Handle PowerShell format: $env:KEY=value
            if line.startswith("$env:"):
                line = line[5:]
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())
                print(f"[信息] 已加载环境变量: {key.strip()}")


def main() -> None:
    load_env_from_powershell_format()

    if "API_KEY" not in os.environ:
        print("[错误] 未设置 API_KEY 环境变量，请检查 .env 文件。")
        sys.exit(1)

    # 导入并执行 generate.main()
    from src.data_pipeline.generate import main as generate_main

    generate_main()


if __name__ == "__main__":
    main()
