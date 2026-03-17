# test_query_reranker_recursive.py
# Command-line interface for querying ChromaDB with optional reranker and better filtering.

# old one


import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
import torch

# CONFIGURATION
EMBEDDING_MODEL = "BAAI/bge-m3"
# RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
CHROMA_DB_PATH = "../output/chroma_store_recursive"
COLLECTION_NAME = "document_chunks_recursive"
TOP_K = 10  # Increased for diversity
TOP_RERANKED = 5

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

# QUERY LOOP
def query_loop():
    print("\n Reranked Semantic Search Interface (type 'exit' to quit)")
    while True:
        user_query = input("\nEnter your question: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print(" Exiting search.")
            break

        # Embed query
        query_embedding = embedder.encode([user_query], normalize_embeddings=True)

        # Step 1: Initial Vector Search
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=TOP_K,
            include=["documents", "metadatas"]
        )

        # Step 2: Pre-filter chunks
        raw_candidates = list(zip(results["documents"][0], results["metadatas"][0]))
        candidates = [(doc, meta) for doc, meta in raw_candidates if is_text_natural(doc)]

        if not candidates:
            print(" No valid text chunks found after filtering. Try another query.")
            continue

        # Step 3: Apply CrossEncoder Reranking
        rerank_inputs = [[user_query, doc[:512]] for doc, _ in candidates]
        rerank_scores = reranker.predict(rerank_inputs)

        # Step 4: Sort by score
        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)

        # Step 5: Show results
        print("\n Reranked Top Matches:")
        for i, ((doc, meta), score) in enumerate(ranked[:TOP_RERANKED]):
            print(f"\nRank {i+1} | Relevance Score: {round(score, 4)}")
            print(f"Source: {meta.get('filename')} | Chunk ID: {meta.get('chunk_id')} | Tokens: {meta.get('num_tokens', 'N/A')}")
            print(f"Content Preview: {doc[:300]}{'...' if len(doc) > 300 else ''}")

if __name__ == "__main__":
    query_loop()
