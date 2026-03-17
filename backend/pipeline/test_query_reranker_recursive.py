# RAG pipeline step: Retrieve, Rerank, and Generate Answer via LLM (OpenAI or Ollama)

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
import torch
import openai
import os
import json
from datetime import datetime


# CONFIGURATION
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_DB_PATH = "../output/chroma_store_recursive"
COLLECTION_NAME = "document_chunks_recursive"
TOP_K = 25
TOP_RERANKED = 5
USE_OPENAI = True  # Set to False to use Ollama
LOG_FILE = "../output/rag_qa_logs.jsonl"


# MODEL & DB SETUP
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

embedder = SentenceTransformer(EMBEDDING_MODEL).to(device)
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANKER_MODEL, device=device)

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
except Exception as e:
    print(f" Failed to connect to ChromaDB collection: {e}")
    exit()

if collection.count() == 0:
    print(" No documents in collection. Please embed data first.")
    exit()


# FILTERING FUNCTION
def is_text_natural(text):
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    return alpha_ratio > 0.3 and '{' not in text and '}' not in text


# LOGGING FUNCTION
def log_interaction(question, answer, chunks):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "chunks": chunks
    }
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# LLM GENERATION
def generate_answer(context_chunks, question):
    context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)])
    prompt = f"""
            You are a helpful AI advisor. Use the following context to answer the user question truthfully. Also explain which paragraph supports your answer.

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
        return response.choices[0].message.content.strip()
    else:
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False}
        )
        return response.json()["response"].strip()


# QUERY LOOP
def query_loop():
    print("\n RAG QA System (Retrieve → Rerank → Generate Answer)")
    while True:
        user_query = input("\nEnter your question: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print(" Exiting.")
            break

        # Embed query
        query_embedding = embedder.encode([user_query], normalize_embeddings=True)

        # Step 1: Initial Vector Search
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=TOP_K,
            include=["documents", "metadatas"]
        )

        raw_candidates = list(zip(results["documents"][0], results["metadatas"][0]))
        candidates = [(doc, meta) for doc, meta in raw_candidates if is_text_natural(doc)]

        if not candidates:
            print(" No valid text chunks found after filtering.")
            continue

        # Step 2: Reranking
        rerank_inputs = [[user_query, doc[:512]] for doc, _ in candidates]
        rerank_scores = reranker.predict(rerank_inputs)
        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        top_chunks = [doc for (doc, _), _ in ranked[:TOP_RERANKED]]

        # Step 3: LLM Answer Generation
        print("\n Generating Answer with LLM...")
        answer = generate_answer(top_chunks, user_query)

        # Display
        print("\n Final Answer:")
        print(answer)

        # Save log
        log_interaction(user_query, answer, top_chunks)

if __name__ == "__main__":
    query_loop()