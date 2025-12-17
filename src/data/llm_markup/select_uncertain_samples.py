import json
import os

INPUT_FILE = "src/data/llm_markup/predictions_a100.jsonl"
OUTPUT_FILE = "src/data/llm_markup/uncertain_3000.jsonl"
TARGET_COUNT = 3000

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    bad_records = []
    candidates = []

    print(f"Reading from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                prediction = data.get("prediction", {})
                
                # Определяем bad records
                # Критерии: class_id == 3, или label содержит 'bad' (например 'bad', 'bad_parse')
                class_id = prediction.get("class_id")
                label = str(prediction.get("label", "")).lower()
                
                is_bad = False
                if class_id == 3:
                    is_bad = True
                elif "bad" in label:
                    is_bad = True
                
                if is_bad:
                    bad_records.append(data)
                else:
                    # Остальные - кандидаты на добавление по низкому конфиденсу
                    # Если confidence нет, считаем его 0.0 (или можно игнорировать, но лучше включить как неуверенные)
                    if "confidence" not in prediction:
                        prediction["confidence"] = 0.0
                    
                    candidates.append(data)

            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {i+1}")
                continue

    print(f"Found {len(bad_records)} 'bad' records.")
    
    # Сортируем кандидатов по возрастанию confidence (самые неуверенные в начале)
    candidates.sort(key=lambda x: x["prediction"]["confidence"])
    
    needed = TARGET_COUNT - len(bad_records)
    
    final_list = []
    if needed <= 0:
        print(f"Warning: 'bad' records count ({len(bad_records)}) already meets or exceeds target ({TARGET_COUNT}). Keeping all bad records.")
        final_list = bad_records
    else:
        print(f"Need {needed} more records to reach {TARGET_COUNT}.")
        if len(candidates) < needed:
             print(f"Warning: Not enough candidates to reach {TARGET_COUNT}. Using all available ({len(candidates)}).")
             selected_candidates = candidates
        else:
             selected_candidates = candidates[:needed]
        
        final_list = bad_records + selected_candidates
    
    print(f"Writing {len(final_list)} records to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in final_list:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print("Done.")

if __name__ == "__main__":
    main()
