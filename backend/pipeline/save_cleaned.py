# This script is responsible for saving the cleaned data to a JSON file.

import json
import os
from loader import get_all_files
from cleaner import read_and_clean

# Step 1: Load files
files = get_all_files("../data/raw/")  # Adjusted path

# Step 2: Clean and process
docs = read_and_clean(files)

# Step 3: Ensure output folder exists (relative to root)
output_path = "../output"
os.makedirs(output_path, exist_ok=True)

# Step 4: Save to ../output/cleaned_docs.json
output_file = os.path.join(output_path, "cleaned_docs.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print(f" Saved cleaned data to {output_file}")
