import chromadb
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
import openai
import os
import requests
import json
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_DB_PATH = os.path.abspath(os.path.join(os.getcwd(), "output/chroma_store_recursive"))
COLLECTION_NAME = "document_chunks_recursive"
TOP_K = 25
TOP_RERANKED = 5
USE_OPENAI = True  # Set to False to use Ollama
LOG_FILE = os.path.abspath(os.path.join(os.getcwd(), "output/rag_ui_query_log.jsonl"))

# -----------------------------
# MODEL & DB INIT
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBEDDING_MODEL).to(device)
reranker = CrossEncoder(RERANKER_MODEL, device=device)
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

# -----------------------------
# UTILITIES
# -----------------------------
def is_text_natural(text):
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    return alpha_ratio > 0.3 and '{' not in text and '}' not in text

# -----------------------------
# CORE FUNCTIONS
# -----------------------------
def get_reranked_chunks(user_query):
    query_embedding = embedder.encode([user_query], normalize_embeddings=True)

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=TOP_K,
        include=["documents", "metadatas"]
    )

    raw_chunks = list(zip(results["documents"][0], results["metadatas"][0]))
    candidates = [(doc, meta) for doc, meta in raw_chunks if is_text_natural(doc)]

    rerank_inputs = [[user_query, doc[:512]] for doc, _ in candidates]
    rerank_scores = reranker.predict(rerank_inputs)
    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    top_chunks = [doc for (doc, _), _ in ranked[:TOP_RERANKED]]

    return top_chunks

def generate_answer(context_chunks, question):
    print("📥 generate_answer called with:", question)
    context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)])
    prompt = f"""
You are a helpful university advisor assistant. Use the following context to answer the user's question truthfully.

Context:
{context}

Question: {question}

Answer:
"""

    if USE_OPENAI:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = response.choices[0].message.content.strip()
    else:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False}
        )
        answer = response.json()["response"].strip()

    log_interaction(question, answer, context_chunks)
    return answer

# -----------------------------
# LOGGING FUNCTION
# -----------------------------
def log_interaction(question, answer, chunks):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "chunks": [str(chunk) for chunk in chunks]
    }
    try:
        print("📝 Logging this interaction:", log_entry)
        print("🔍 Will write to:", LOG_FILE)
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        print("✅ Log saved to:", LOG_FILE)
    except Exception as e:
        print("❌ Failed to save log:", e)