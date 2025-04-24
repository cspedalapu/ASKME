# This script reads raw files, applies minimal preprocessing, chunks them using RecursiveCharacterTextSplitter,
# embeds using BAAI/bge-m3, and stores them in ChromaDB with full logging support for analysis and visualization.

import os
import re
import json
import time
import torch
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import chromadb
from chromadb import PersistentClient
from docx import Document as DocxDocument
import fitz  # PyMuPDF for PDF reading

# -----------------------------
# CONFIGURATION
# -----------------------------
SUPPORTED_EXTENSIONS = ['.pdf', '.json', '.jsonl', '.csv', '.txt', '.docx']
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))
CHROMA_DB_PATH = "../output/chroma_store_recursive"
LOG_FILE_PATH = "../output/recursive_chunk_log.json"
EMBEDDING_VECTOR_PATH = "../output/recursive_vectors.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_TOKEN_LIMIT = 512
BATCH_SIZE = 32

# -----------------------------
# FILE LOADER
# -----------------------------
def get_all_files(data_dir=DATA_PATH):
    loaded_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, file)
                loaded_files.append(full_path)
    print(f"\n Total files found: {len(loaded_files)}")
    return loaded_files

# -----------------------------
# SMART FILE READER
# -----------------------------
def read_file_smart(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        elif ext == ".docx":
            doc = DocxDocument(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return json.dumps(data)
        else:  # .txt, .csv, .jsonl
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        return ""

# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
def preprocess_text(text):
    text = text.replace('\x00', '')
    text = re.sub(r'\r\n|\r', '\n', text)     # Normalize newlines
    text = re.sub(r'\n{2,}', '\n\n', text)     # Limit excessive line breaks
    text = re.sub(r'\s+', ' ', text)            # Normalize whitespace
    return text.strip()

# -----------------------------
# MAIN PIPELINE
# -----------------------------

def main():
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = SentenceTransformer("BAAI/bge-m3", device=DEVICE).to(DEVICE)

    if DEVICE == "cuda":
        print(" GPU Memory Check:")
        os.system("nvidia-smi")

    chroma_client = PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name="document_chunks_recursive")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_TOKEN_LIMIT,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    all_ids = []
    all_texts = []
    all_embeddings = []
    all_metadatas = []
    log_records = []
    vector_records = []
    doc_id_counter = 0

    files = get_all_files()

    for file_path in tqdm(files, desc="📄 Processing files"):
        raw_text = read_file_smart(file_path)
        if not raw_text.strip():
            print(f"⚠️ Empty or unreadable: {file_path}")
            continue

        clean_text = preprocess_text(raw_text)
        chunks = splitter.split_text(clean_text)

        if not chunks:
            print(f"⚠️ No chunks generated for {file_path}")
            continue

        try:
            embeddings = model.encode(
                chunks,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        except Exception as e:
            print(f"❌ Embedding failed for {file_path}: {e}")
            continue

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"doc_{doc_id_counter}"
            token_count = len(tokenizer.encode(chunk, truncation=False))

            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_embeddings.append(embedding.tolist())
            all_metadatas.append({
                "source": file_path,
                "filename": os.path.basename(file_path),
                "chunk_id": chunk_id,
                "num_tokens": token_count
            })

            log_records.append({
                "chunk_id": chunk_id,
                "file": os.path.basename(file_path),
                "tokens": token_count,
                "length": len(chunk),
                "source": file_path
            })

            vector_records.append({
                "chunk_id": chunk_id,
                "embedding": embedding.tolist(),
                "source": file_path
            })

            doc_id_counter += 1

    for i in range(0, len(all_texts), 5000):
        collection.add(
            documents=all_texts[i:i + 5000],
            embeddings=all_embeddings[i:i + 5000],
            ids=all_ids[i:i + 5000],
            metadatas=all_metadatas[i:i + 5000]
        )

    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(log_records, f, indent=2)

    with open(EMBEDDING_VECTOR_PATH, "w", encoding="utf-8") as f:
        json.dump(vector_records, f, indent=2)

    elapsed = round(time.time() - start_time, 2)
    print("\n✅ Recursive embedding complete!")
    print(f" Total chunks stored: {len(all_texts)}")
    print(f" Metadata file: {LOG_FILE_PATH}")
    print(f" Embedding vector file: {EMBEDDING_VECTOR_PATH}")
    print(f" ChromaDB location: {CHROMA_DB_PATH}")
    print(f" Time taken: {elapsed} seconds")

if __name__ == "__main__":
    main()