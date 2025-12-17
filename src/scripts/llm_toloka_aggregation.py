import pandas as pd
import os

# 1. Setup File Paths
llm_path = 'src/data/raw/vlm_predictions_a100.csv'
toloka_path = 'src/data/raw/aggregated_markup.csv'
output_path = 'src/data/raw/labels_llm_toloka.csv'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 2. Load Data
print("Loading data...")
try:
    df_llm = pd.read_csv(llm_path)
    df_toloka = pd.read_csv(toloka_path)
except FileNotFoundError as e:
    print(f"Error: Could not find file. {e}")
    exit()

# 3. Preprocessing
# Standardize column names for merging
df_llm = df_llm.rename(columns={'label': 'label_llm'})
df_toloka = df_toloka.rename(columns={'label': 'label_toloka'})

# Ensure filenames are strings and stripped of whitespace for clean matching
df_llm['filename'] = df_llm['filename'].astype(str).str.strip()
df_toloka['filename'] = df_toloka['filename'].astype(str).str.strip()

# Ensure labels are integers (handles cases like 1.0 vs 1)
df_llm['label_llm'] = df_llm['label_llm'].astype(int)
df_toloka['label_toloka'] = df_toloka['label_toloka'].astype(int)

# 4. Merge Dataframes
# We use 'inner' join because we need info from BOTH sources to make a decision
merged_df = pd.merge(df_toloka, df_llm, on='filename', how='inner')

print(f"Total overlapping images found: {len(merged_df)}")

# 5. Apply the Logic
# ---------------------------------------------------------
# Logic Refresher:
# Class 1: Trust Toloka (High Precision). Ignore LLM opinion here.
# Class 0: Trust Intersection (Toloka says 0 AND LLM says 0).
# Discard: Toloka says 0 but LLM says 1 (High risk of FN).
# ---------------------------------------------------------

# Filter for Positives
positives = merged_df[merged_df['label_toloka'] == 1].copy()
positives['final_label'] = 1

# Filter for Negatives (The "Safety Check")
# We only keep negative if BOTH agree it is negative
negatives = merged_df[
    (merged_df['label_toloka'] == 0) & 
    (merged_df['label_llm'] == 0)
].copy()
negatives['final_label'] = 0

# Combine them back together
final_dataset = pd.concat([positives, negatives])

# 6. Statistics & Sanity Check
discarded_count = len(merged_df) - len(final_dataset)
discarded_negatives = merged_df[(merged_df['label_toloka'] == 0) & (merged_df['label_llm'] == 1)]

print("\n--- Processing Statistics ---")
print(f"Original Toloka Count:  {len(df_toloka)}")
print(f"Original LLM Count:     {len(df_llm)}")
print(f"Merged Intersection:    {len(merged_df)}")
print("-" * 30)
print(f"Final Dataset Size:     {len(final_dataset)}")
print(f"  - Positive Samples:   {len(positives)} (Source: Toloka Positives)")
print(f"  - Negative Samples:   {len(negatives)} (Source: Toloka 0 + LLM 0)")
print("-" * 30)
print(f"Discarded Images:       {discarded_count}")
print(f"  - Risk Candidates:    {len(discarded_negatives)} (Toloka said 0, but LLM flagged 1)")

# 7. Save to CSV
# We only save filename and the new final_label
output_df = final_dataset[['filename', 'final_label']]
output_df.to_csv(output_path, index=False)
print(f"\nSaved cleaned dataset to: {output_path}")
# Why this works for your specific data:

# positives dataframe: This captures your Toloka 1s. Since humans have high precision (FP=2), we take these immediately. We don't care what the LLM thinks here (even if LLM missed it, the human saw it, so it's a valid object).

# negatives dataframe: This captures the intersection. This effectively removes the ~55 False Negatives where the human missed the object but the "paranoid" LLM spotted it.

# The Discard Pile: The script prints how many "Risk Candidates" were dropped. These are the images where Toloka=0 and LLM=1. These are the most dangerous images for your model, so removing them cleans your training data significantly.