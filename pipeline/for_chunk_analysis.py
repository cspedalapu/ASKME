import json
import os

# Step 1: Load recursive chunk log
with open("../output/recursive_chunk_log.json", "r", encoding="utf-8") as f:
    raw_chunks = json.load(f)

# Step 2: Transform for chunk analysis structure
converted = []
for chunk in raw_chunks:
    converted.append({
        "chunk_id": chunk["chunk_id"],
        "file_name": chunk.get("file", "unknown"),
        "file_type": os.path.splitext(chunk.get("file", ""))[1].lower(),
        "num_tokens": chunk.get("tokens", 0),
        "source": chunk.get("source", ""),
        "truncated": chunk.get("tokens", 0) > 512
    })

# Step 3: Save as new analysis file
with open("../output/chunk_analysis_recursive.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2)

print("✅ Recursive chunk analysis saved as chunk_analysis_recursive.json")
