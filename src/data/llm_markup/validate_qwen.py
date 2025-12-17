import json
import argparse
import os
import requests
import torch
import io
import time
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sklearn.metrics import confusion_matrix, classification_report

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
BATCH_SIZE = 8
NUM_WORKERS = 8
MAX_PIXELS = 1024 * 1024
# Using the same cache dir as in the reference script
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

class ValidationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        url = item['url']
        gt_label = item['gt_label']
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return {"image": image, "url": url, "gt_label": gt_label, "error": None}
        except Exception as e:
            dummy = Image.new('RGB', (224, 224), color='black')
            return {"image": dummy, "url": url, "gt_label": gt_label, "error": str(e)}

def collate_fn(batch):
    return batch

def parse_gt_file(file_path):
    data = []
    print(f"Parsing GT file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip header if present
            if "is_conifer" in line and "downloadUrl" in line:
                continue
            
            parts = line.split()
            # Try to find URL and label
            url = None
            gt_val = None
            
            # Simple heuristic parsing: looking for http for url, and TRUE/FALSE for label
            for p in parts:
                if "http" in p:
                    url = p
                if p == "TRUE":
                    gt_val = 1 # Coniferous
                elif p == "FALSE":
                    gt_val = 2 # Deciduous
            
            if url and gt_val:
                data.append({"url": url, "gt_label": gt_val})
            else:
                pass
                # print(f"Skipping line: {line}")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="src/data/gt/hw_3_markup_data.txt", help="Path to GT file")
    parser.add_argument("--output", type=str, default="src/data/llm_markup/validation_predictions.jsonl", help="Output file for predictions")
    args = parser.parse_args()

    # Create cache dir if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Loading model: {MODEL_ID}...", flush=True)
    try:
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
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you are running on a machine with GPU access and correct environment.")
        return

    # Prepare Data
    data_items = parse_gt_file(args.input)
    print(f"Loaded {len(data_items)} samples for validation.")

    dataset = ValidationDataset(data_items)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # Store results for metrics
    y_true = []
    y_pred = []

    print("Starting inference...", flush=True)
    with open(args.output, 'w', encoding='utf-8') as f_out:
        for i, batch in enumerate(dataloader):
            valid_images = []
            valid_urls = []
            valid_gts = []
            messages = []

            for item in batch:
                if item['error']:
                    record = {"url": item['url'], "error": item['error'], "prediction": {"class_id": 3, "label": "bad"}, "gt_label": item['gt_label']}
                    f_out.write(json.dumps(record) + "\n")
                    # Treat download error as 'Bad' (3) or ignore?
                    # Let's track it as 3 for now so we see it in confusion matrix if we want, or ignore.
                    # Here we will add to metrics as 3
                    y_true.append(item['gt_label'])
                    y_pred.append(3)
                    continue

                valid_images.append(item['image'])
                valid_urls.append(item['url'])
                valid_gts.append(item['gt_label'])

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

            try:
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
            except Exception as e:
                print(f"Error during batch inference: {e}")
                for url, gt in zip(valid_urls, valid_gts):
                     y_true.append(gt)
                     y_pred.append(3) # Error
                continue

            for url, text, gt in zip(valid_urls, output_texts, valid_gts):
                try:
                    clean_text = text.replace("```json", "").replace("```", "").strip()
                    prediction = json.loads(clean_text)
                    pred_class = prediction.get("class_id", 3)
                except:
                    prediction = {"raw_text": text, "class_id": 3, "label": "bad_parse"}
                    pred_class = 3

                # Ensure class is int
                try:
                    pred_class = int(pred_class)
                except:
                    pred_class = 3

                record = {
                    "url": url,
                    "timestamp": time.time(),
                    "gt_label": gt,
                    "prediction": prediction
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                y_true.append(gt)
                y_pred.append(pred_class)

            if i % 10 == 0:
                print(f"Processed batch {i}. Last URL: {valid_urls[-1]}", end="\r", flush=True)

    print("\nInference complete.")
    
    # Calculate Metrics
    print("-" * 30)
    print("VALIDATION RESULTS")
    print("-" * 30)
    
    # Map labels
    # 1: Coniferous
    # 2: Deciduous
    # 3: Bad/Error
    
    labels = [1, 2, 3]
    target_names = ["Coniferous", "Deciduous", "Bad"]
    
    # Filter only relevant classes if 3 is not in GT
    # GT only has 1 and 2. 3 appears in Pred.
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    print("Confusion Matrix:")
    print(f"{ '':>12} {'Pred Conif':>12} {'Pred Decid':>12} {'Pred Bad':>12}")
    print(f"{ 'Act Conif':>12} {cm[0][0]:>12} {cm[0][1]:>12} {cm[0][2]:>12}")
    print(f"{ 'Act Decid':>12} {cm[1][0]:>12} {cm[1][1]:>12} {cm[1][2]:>12}")
    # There are no 'Act Bad' in GT, so the 3rd row of CM (Act Bad) will be zeros effectively if we strictly follow GT.
    # But sklearn confusion_matrix with given labels will produce a 3x3 matrix. 
    # Row 3 is Actual Bad, which is 0.
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=[1, 2, 3], target_names=target_names, zero_division=0))

if __name__ == "__main__":
    main()
