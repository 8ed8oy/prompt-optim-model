import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional


FINAL_TAG = "【最终优化提示词】"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    compact_lines: List[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                compact_lines.append("")
            previous_blank = True
            continue
        compact_lines.append(re.sub(r"\s+", " ", line))
        previous_blank = False
    return "\n".join(compact_lines).strip()


def extract_json_object(text: str) -> Optional[Dict]:
    text = text.strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    candidate = match.group(0)
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def extract_final_prompt(text: str) -> str:
    normalized = normalize_text(text)

    final_json = extract_json_object(normalized)
    if isinstance(final_json, dict):
        prompt = str(final_json.get("prompt", "")).strip()
        if prompt:
            return prompt

    if FINAL_TAG not in normalized:
        return ""
    _, prompt_body = normalized.split(FINAL_TAG, 1)
    return prompt_body.strip()


def ascii_letter_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-z]", text)
    visible_chars = re.findall(r"\S", text)
    if not visible_chars:
        return 0.0
    return len(letters) / len(visible_chars)


def cjk_ratio(text: str) -> float:
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    visible_chars = re.findall(r"\S", text)
    if not visible_chars:
        return 0.0
    return len(cjk_chars) / len(visible_chars)


def has_abnormal_repetition(text: str) -> bool:
    return bool(re.search(r"(.)\1{7,}", text))


def looks_structured_prompt(text: str) -> bool:
    prompt = normalize_text(text)
    if not prompt:
        return False

    parts = [segment.strip() for segment in prompt.split(",") if segment.strip()]
    if len(parts) < 6:
        return False

    keywords = [
        "style", "shot", "camera", "lighting", "color", "palette",
        "mood", "composition", "depth", "focus", "cinematic", "--ar",
    ]
    lower_prompt = prompt.lower()
    if not any(keyword in lower_prompt for keyword in keywords):
        return False

    return True


def extract_final_payload(text: str) -> Optional[Dict]:
    normalized = normalize_text(text)
    obj = extract_json_object(normalized)
    if isinstance(obj, dict) and isinstance(obj.get("prompt"), str):
        payload: Dict = {"prompt": normalize_text(str(obj.get("prompt", "")))}
        scenes = obj.get("scenes")
        if isinstance(scenes, list):
            converted_scenes = []
            for index, scene in enumerate(scenes, start=1):
                if isinstance(scene, dict) and isinstance(scene.get("prompt"), str):
                    converted_scenes.append(
                        {
                            "id": int(scene.get("id", index)),
                            "prompt": normalize_text(str(scene.get("prompt", ""))),
                        }
                    )
            if converted_scenes:
                payload["scenes"] = converted_scenes
        return payload

    legacy_prompt = extract_final_prompt(normalized)
    if legacy_prompt:
        return {"prompt": legacy_prompt}

    return None


def normalize_final_message(text: str) -> str:
    payload = extract_final_payload(text)
    if not payload:
        return normalize_text(text)

    prompt_body = re.sub(r"[。；：“”‘’（）、，]", " ", str(payload.get("prompt", "")))
    prompt_body = re.sub(r"\s+", " ", prompt_body).strip(" ,\n")

    final_payload: Dict = {"prompt": prompt_body}
    scenes = payload.get("scenes")
    if isinstance(scenes, list):
        cleaned_scenes = []
        for index, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict):
                continue
            scene_prompt = re.sub(r"[。；：“”‘’（）、，]", " ", str(scene.get("prompt", "")))
            scene_prompt = re.sub(r"\s+", " ", scene_prompt).strip(" ,\n")
            if scene_prompt:
                cleaned_scenes.append({"id": int(scene.get("id", index)), "prompt": scene_prompt})
        if cleaned_scenes:
            final_payload["scenes"] = cleaned_scenes

    return json.dumps(final_payload, ensure_ascii=False)


def normalize_record(obj: Dict, scene: str) -> Dict:
    messages = obj.get("messages", [])
    normalized_messages: List[Dict] = []
    for idx, message in enumerate(messages):
        role = message.get("role", "")
        content = normalize_text(str(message.get("content", "")))
        if idx == len(messages) - 1 and role == "assistant":
            content = normalize_final_message(content)
        normalized_messages.append({"role": role, "content": content})

    obj["messages"] = normalized_messages
    if "meta" not in obj or not isinstance(obj.get("meta"), dict):
        obj["meta"] = {}
    obj["meta"]["scene"] = normalize_text(str(obj["meta"].get("scene") or scene))
    obj["meta"].setdefault("difficulty", random.choice(["easy", "medium", "hard"]))
    return obj


def validate_sample(obj: Dict) -> bool:
    if "messages" not in obj or not isinstance(obj["messages"], list):
        return False

    messages = obj["messages"]
    if len(messages) < 7 or len(messages) > 9:
        return False

    valid_roles = {"system", "user", "assistant"}
    for message in messages:
        if not isinstance(message, dict):
            return False
        if message.get("role") not in valid_roles:
            return False
        if not isinstance(message.get("content"), str) or not message.get("content").strip():
            return False
        if len(message["content"].strip()) < 4 or len(message["content"].strip()) > 2000:
            return False
        if has_abnormal_repetition(message["content"]):
            return False

    if messages[-1].get("role") != "assistant":
        return False

    final_content = normalize_text(messages[-1].get("content", ""))
    final_payload = extract_final_payload(final_content)
    if not isinstance(final_payload, dict):
        return False

    final_prompt = normalize_text(str(final_payload.get("prompt", "")))
    if not final_prompt:
        return False

    for message in messages[:-1]:
        if message["role"] == "system":
            continue
        content = normalize_text(message["content"])
        if ascii_letter_ratio(content) > 0.35:
            return False
        if cjk_ratio(content) < 0.15:
            return False

    if ascii_letter_ratio(final_prompt) < 0.45:
        return False
    if cjk_ratio(final_prompt) > 0.08:
        return False
    if not looks_structured_prompt(final_prompt):
        return False

    scenes = final_payload.get("scenes")
    if scenes is not None:
        if not isinstance(scenes, list):
            return False
        if len(scenes) < 3 or len(scenes) > 4:
            return False
        for scene_item in scenes:
            if not isinstance(scene_item, dict):
                return False
            scene_prompt = normalize_text(str(scene_item.get("prompt", "")))
            if not scene_prompt:
                return False
            if ascii_letter_ratio(scene_prompt) < 0.45:
                return False
            if cjk_ratio(scene_prompt) > 0.08:
                return False
            if not looks_structured_prompt(scene_prompt):
                return False

    normalized_contents = [normalize_text(m["content"]) for m in messages]
    if len(normalized_contents) != len(set(normalized_contents)):
        return False

    user_count = sum(1 for m in messages if m["role"] == "user")
    assistant_count = sum(1 for m in messages if m["role"] == "assistant")
    if user_count < 2 or assistant_count < 2:
        return False

    return True


def load_existing(output_path: Path) -> List[Dict]:
    if not output_path.exists():
        return []

    records: List[Dict] = []
    with output_path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def append_record(output_path: Path, record: Dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_output_path(output: str, worker_id: str) -> Path:
    raw_path = Path(output)
    if raw_path.suffix.lower() == ".jsonl":
        if worker_id:
            return raw_path.with_name(f"{raw_path.stem}.{worker_id}{raw_path.suffix}")
        return raw_path

    target_dir = raw_path
    filename = f"train_data.{worker_id}.jsonl" if worker_id else "train_data.jsonl"
    return target_dir / filename


def is_non_retryable_error(error: Exception) -> bool:
    text = str(error)
    return any(
        code in text
        for code in ["Error code: 400", "Error code: 401", "Error code: 403", "Error code: 404", "invalid_request_error"]
    )