import json
import argparse
import os
import requests
import torch
import io
import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
BATCH_SIZE = 8
NUM_WORKERS = 8
MAX_PIXELS = 1024 * 1024
# EXPLICIT CACHE DIRECTORY
CACHE_DIR = "/home/jupyter/datasphere/filestore/storage/hf_cache"

SYSTEM_PROMPT = """You are an expert botanist. Classify the MAIN tree in the center.
JSON Output Format:
{
  "color": "Describe crown color (Dark Green, Light Green, Grey, Brown, White)",
  "features": "Describe needles vs leaves/bare branches",
  "reasoning": "Step-by-step logic",
  "class_id": 1 (Coniferous) or 2 (Deciduous) or 3 (Bad),
  "label": "coniferous" or "deciduous" or "bad",
  "confidence": 0.0 to 1.0
}
Rules:
- Green in winter = Coniferous.
- Bare branches (no needles) = Deciduous.
- Blurry/No tree = Bad.
"""


class ImageDataset(Dataset):
    def __init__(self, urls):
        self.urls = urls

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return {"image": image, "url": url, "error": None}
        except Exception as e:
            dummy = Image.new('RGB', (224, 224), color='black')
            return {"image": dummy, "url": url, "error": str(e)}


def collate_fn(batch):
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to txt file with URLs")
    parser.add_argument("--output", type=str, default="predictions_a100.jsonl")
    args = parser.parse_args()

    # Create cache dir if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Loading model: {MODEL_ID}...", flush=True)
    print(f"Using cache directory: {CACHE_DIR}", flush=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        cache_dir=CACHE_DIR
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=MAX_PIXELS,
        cache_dir=CACHE_DIR
    )

    with open(args.input, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and "http" in line]

    processed_urls = set()
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            for line in f:
                try:
                    processed_urls.add(json.loads(line)['url'])
                except:
                    pass

    new_urls = [u for u in urls if u not in processed_urls]
    print(f"Total: {len(urls)}, Processed: {len(processed_urls)}, To do: {len(new_urls)}", flush=True)

    dataset = ImageDataset(new_urls)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            collate_fn=collate_fn)

    print("Starting inference...", flush=True)
    with open(args.output, 'a') as f_out:
        for batch in dataloader:
            valid_images = []
            valid_urls = []
            messages = []

            for item in batch:
                if item['error']:
                    record = {"url": item['url'], "error": item['error'], "prediction": {"class_id": 3, "label": "bad"}}
                    f_out.write(json.dumps(record) + "\n")
                    continue

                valid_images.append(item['image'])
                valid_urls.append(item['url'])

                messages.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": item['image']},
                            {"type": "text", "text": SYSTEM_PROMPT},
                        ],
                    }
                ])

            if not valid_images:
                continue

            texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

            for url, text in zip(valid_urls, output_texts):
                try:
                    clean_text = text.replace("```json", "").replace("```", "").strip()
                    prediction = json.loads(clean_text)
                except:
                    prediction = {"raw_text": text, "class_id": 3, "label": "bad_parse"}

                record = {
                    "url": url,
                    "timestamp": time.time(),
                    "model": MODEL_ID,
                    "prediction": prediction
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            f_out.flush()
            print(f"Processed batch. Last URL: {valid_urls[-1]}", end="\r")


if __name__ == "__main__":
    main()
