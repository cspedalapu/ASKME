

import os
import json
import time
import torch
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')  # Ensure sentence tokenizer is available
nltk.download('punkt_tab')  # Ensure punkt tokenizer is available

# Step 1: Load cleaned document content
with open("../output/cleaned_docs.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Step 2: Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="../output/chroma_store")
collection_name = "campus_documents"
collection = chroma_client.get_or_create_collection(name=collection_name)

# Step 3: Load model + tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using device: {device}")

model = SentenceTransformer("BAAI/bge-m3")
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
MAX_MODEL_TOKENS = tokenizer.model_max_length  # usually 8192
CHUNK_TOKEN_LIMIT = 512
BATCH_SIZE = 32  # Optimal for RTX 3080 (10GB VRAM)

# Optional: show GPU memory
if device == "cuda":
    print(" GPU Memory Check:")
    os.system("nvidia-smi")

# Step 4: Sentence-aware chunking with smart token fallback
def sentence_aware_chunking(text, tokenizer, chunk_token_limit=CHUNK_TOKEN_LIMIT):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sent in sentences:
        sent_tokens = tokenizer.encode(sent, truncation=False)

        # Fallback for long sentences
        if len(sent_tokens) > chunk_token_limit:
            print(f" Long sentence fallback: {len(sent_tokens)} tokens")
            for i in range(0, len(sent_tokens), chunk_token_limit):
                token_chunk = sent_tokens[i:i + chunk_token_limit]
                text_chunk = tokenizer.decode(token_chunk, skip_special_tokens=True)
                chunks.append(text_chunk.strip())
            continue

        proposed_chunk = current_chunk + " " + sent if current_chunk else sent
        proposed_tokens = tokenizer.encode(proposed_chunk, truncation=False)

        if len(proposed_tokens) <= chunk_token_limit:
            current_chunk = proposed_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Step 5: Start timer
start_time = time.time()

# Step 6: Initialize trackers
all_ids = []
all_texts = []
all_embeddings = []
all_metadatas = []
doc_id_counter = 0

total_skipped_chunks = 0

# Step 7: Embed safely
for doc in tqdm(documents, desc=" Embedding documents"):
    content = doc["content"]
    file_name = doc["file_name"]
    file_type = doc["file_type"]
    source = doc["source"]

    chunks = sentence_aware_chunking(content, tokenizer)

    verified_chunks = []
    for i, chunk in enumerate(chunks):
        token_len = len(tokenizer.encode(chunk, truncation=False))
        if token_len <= MAX_MODEL_TOKENS:
            verified_chunks.append(chunk)
        else:
            print(f" Chunk too long after fallback (should not happen): {token_len} tokens")
            total_skipped_chunks += 1

    if not verified_chunks:
        print(f" No valid chunks to embed for: {file_name}")
        continue

    try:
        embeddings = model.encode(
            verified_chunks,
            device=device,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f" OOM error at batch_size={BATCH_SIZE}, falling back to 16")
            torch.cuda.empty_cache()
            try:
                embeddings = model.encode(
                    verified_chunks,
                    device=device,
                    batch_size=16,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            except Exception as inner_e:
                print(f" Embedding still failed for {file_name}: {inner_e}")
                continue
        else:
            print(f" Embedding error in {file_name}: {e}")
            continue

    for chunk_index, (chunk, embedding) in enumerate(zip(verified_chunks, embeddings)):
        doc_id = f"doc_{doc_id_counter}"
        num_tokens = len(tokenizer.encode(chunk, truncation=False))

        all_ids.append(doc_id)
        all_texts.append(chunk)
        all_embeddings.append(embedding.tolist())
        all_metadatas.append({
            "file_name": file_name,
            "file_type": file_type,
            "source": source,
            "chunk_id": chunk_index,
            "num_tokens": num_tokens,
            "truncated": False
        })

        doc_id_counter += 1

# Step 8: Store in ChromaDB with batching
MAX_CHROMA_BATCH = 5000  # Safe limit per ChromaDB docs

for i in range(0, len(all_texts), MAX_CHROMA_BATCH):
    collection.add(
        documents=all_texts[i:i + MAX_CHROMA_BATCH],
        embeddings=all_embeddings[i:i + MAX_CHROMA_BATCH],
        ids=all_ids[i:i + MAX_CHROMA_BATCH],
        metadatas=all_metadatas[i:i + MAX_CHROMA_BATCH]
    )

# Step 9: Save metadata
with open("../output/chunk_analysis.json", "w", encoding="utf-8") as f:
    json.dump(all_metadatas, f, ensure_ascii=False, indent=2)

# Step 10: Log summary
end_time = time.time()
elapsed = round(end_time - start_time, 2)

print("\n Embedding complete!")
print(f" Chunks embedded: {len(all_texts)}")
print(f" Metadata file: ../output/chunk_analysis.json")
print(f" Vector DB saved at: ../output/chroma_store")
print(f" Time taken: {elapsed} seconds")
print(f" Total skipped chunks: {total_skipped_chunks}")