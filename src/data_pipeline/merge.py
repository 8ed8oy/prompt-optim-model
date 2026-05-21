import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from .core import extract_final_prompt, normalize_record, validate_sample


def iter_jsonl_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.glob("*.jsonl")):
        if path.is_file():
            yield path


def load_records(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line_no, line in enumerate(file_handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                print(f"[跳过] {path.name}:{line_no} 不是合法 JSON")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="合并并清洗多 worker 生成的训练数据")
    parser.add_argument("--input-dir", type=str, default="data", help="输入分片目录，默认 data")
    parser.add_argument("--output", type=str, default="train_data.cleaned.jsonl", help="清洗后的输出文件")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        raise SystemExit(f"[错误] 输入目录不存在: {input_dir.resolve()}")

    seen = set()
    kept = 0
    dropped_invalid = 0
    dropped_duplicate = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for path in iter_jsonl_files(input_dir):
            for record in load_records(path):
                scene = str(record.get("meta", {}).get("scene") or "未知场景")
                record = normalize_record(record, scene)
                if not validate_sample(record):
                    dropped_invalid += 1
                    continue

                final_prompt = extract_final_prompt(record["messages"][-1]["content"])
                signature = final_prompt.lower().strip()
                if signature in seen:
                    dropped_duplicate += 1
                    continue

                seen.add(signature)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[完成] 保留样本: {kept}")
    print(f"[完成] 过滤异常: {dropped_invalid}")
    print(f"[完成] 去重条数: {dropped_duplicate}")
    print(f"[完成] 输出文件: {output_path.resolve()}")


if __name__ == "__main__":
    main()