import os
import base64
import mimetypes
import pathlib
import time
import re
import json
from openai import OpenAI

# --- CONFIGURATION ---
FOLDER   = "/Users/yizhangwu/Downloads/Annotation With NividaPrompting/data_set_1"
MODEL    = "glm-v"
BASE_URL = "https://ellm.nrp-nautilus.io/v1"
API_KEY  = "bfQyWLaa8qFGyQ0iK91kwG1BIuxVIVk3"

N_FUTURE_FRAMES = 10

PROMPT_ROOT = pathlib.Path("/Users/yizhangwu/Downloads/Annotation With NividaPrompting")

P1 = PROMPT_ROOT / "prompt_stage_1"
P2 = PROMPT_ROOT / "prompt_stage_2"


def load_prompt(path: pathlib.Path) -> str:
    """
    Reads a prompt text file and returns its content as a stripped string.
    """
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()

# insert the overall navigation goal here
NAV_GOAL =" go through the intersection "  

# === Load Stage 1 Prompts ===
STAGE1_SYSTEM_INSTRUCTION = load_prompt(P1 / "STAGE1_SYSTEM_INSTRUCTION")
STAGE1_USER_TASK          = load_prompt(P1 / "STAGE1_USER_TASK")

# === Load Stage 2 Prompts ===
STAGE2_SYSTEM_INSTRUCTION = load_prompt(P2 / "STAGE2_SYSTEM_INSTRUCTION")
STAGE2_USER_TASK          = load_prompt(P2 / "STAGE2_USER_TASK").replace("__INSERT_NAVIGATION_GOAL_HERE__", NAV_GOAL)



def to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None: mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def list_images(folder: str):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    p = pathlib.Path(folder)
    return sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts])

def extract_text_from_chat_message(msg) -> str:
    c = msg.content
    if isinstance(c, str): return (c or "").strip()
    parts = []
    for part in (c or []):
        if isinstance(part, dict) and part.get("text"): parts.append(part["text"])
        elif hasattr(part, "text"): parts.append(part.text)
    return "".join(parts).strip()

def clean_model_output(raw_text: str) -> str:
    pattern = r"```json(.*?)```"
    match = re.search(pattern, raw_text, re.DOTALL)
    if match: return match.group(1).strip()
    start = raw_text.find('{')
    end = raw_text.rfind('}')
    if start != -1 and end != -1: return raw_text[start:end+1]
    return raw_text

def chat_once(client, messages):
    return client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=1500,
        messages=messages,
    )

IMAGE_VARIANTS = [
    lambda url: {"type": "image_url", "image_url": {"url": url}},
    lambda url: {"type": "image_url", "image_url": url},
]

def split_history_future(img_paths, n_future=N_FUTURE_FRAMES):
    img_paths = sorted(img_paths)
    if len(img_paths) <= n_future:
        raise ValueError(f"Not enough images")
    history = img_paths[: len(img_paths) - n_future]
    future = img_paths[len(img_paths) - n_future :]
    return history, future



def run_stage1(client: OpenAI, history_paths):
    data_urls = [to_data_url(str(p)) for p in history_paths]
    for build_image_part in IMAGE_VARIANTS:
        for attempt in range(3):
            user_content = [{"type": "text", "text": STAGE1_USER_TASK.strip()}]
            for idx, url in enumerate(data_urls):
                user_content.append({"type": "text", "text": f"[HISTORY FRAME {idx}]"})
                user_content.append(build_image_part(url))
            try:
                resp = chat_once(client, [
                    {"role": "system", "content": STAGE1_SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_content},
                ])
                return extract_text_from_chat_message(resp.choices[0].message)
            except Exception as e:
                time.sleep(1)
    return ""

def run_stage2(client: OpenAI, history_paths, future_paths, stage1_json_str: str):
    hist_urls = [to_data_url(str(p)) for p in history_paths]
    fut_urls  = [to_data_url(str(p)) for p in future_paths]
    prompt_text = STAGE2_USER_TASK.replace("__INSERT_STAGE1_JSON_HERE__", stage1_json_str)
    
    for build_image_part in IMAGE_VARIANTS:
        for attempt in range(3):
            user_content = [{"type": "text", "text": prompt_text.strip()}]
            for idx, url in enumerate(hist_urls):
                user_content.append({"type": "text", "text": f"[HISTORY FRAME {idx}]"})
                user_content.append(build_image_part(url))
            for idx, url in enumerate(fut_urls):
                user_content.append({"type": "text", "text": f"[FUTURE FRAME {idx}]"})
                user_content.append(build_image_part(url))
            try:
                resp = chat_once(client, [
                    {"role": "system", "content": STAGE2_SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_content},
                ])
                return extract_text_from_chat_message(resp.choices[0].message)
            except Exception as e:
                time.sleep(1)
    return ""

def get_dataset_name(folder: str) -> str:
    """
    Infer dataset name from the folder path.
    Example:
        FOLDER = "/data/nuscenes/clip_0001" -> dataset_name = "clip_0001"
    """
    return pathlib.Path(folder).name


def main():
    imgs = list_images(FOLDER)
    if not imgs:
        raise SystemExit(f"No images found in {FOLDER}")

    # Split history / future
    history, future = split_history_future(imgs, N_FUTURE_FRAMES)

    # Build client
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Prepare output directory and file names based on dataset name
    dataset_name = get_dataset_name(FOLDER)
    output_dir = pathlib.Path(FOLDER) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # File paths:
    #   data_set_<dataset_name>_stage_1.json
    #   data_set_<dataset_name>_stage_1_full.txt
    #   data_set_<dataset_name>_stage_2.json
    #   data_set_<dataset_name>_stage_2_full.txt
    stage1_json_path = output_dir / f"data_set_{dataset_name}_stage_1.json"
    stage1_full_path = output_dir / f"data_set_{dataset_name}_stage_1_full.txt"
    stage2_json_path = output_dir / f"data_set_{dataset_name}_stage_2.json"
    stage2_full_path = output_dir / f"data_set_{dataset_name}_stage_2_full.txt"

    print("\n[Stage I] Extraction...")
    s1_raw = run_stage1(client, history)
    s1_clean = clean_model_output(s1_raw)
    with open(stage1_json_path, "w") as f:
        f.write(s1_clean)
    with open(stage1_full_path, "w") as f:
        f.write(s1_raw)

    print("\n[Stage II] Decision Grounding...")
    s2_raw = run_stage2(client, history, future, s1_clean)
    s2_clean = clean_model_output(s2_raw)
    with open(stage2_json_path, "w") as f:
        f.write(s2_clean)
    with open(stage2_full_path, "w") as f:
        f.write(s2_raw)

    print(f"Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()