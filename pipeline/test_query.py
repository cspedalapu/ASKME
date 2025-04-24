# This script provides a command-line interface for semantic search over a ChromaDB collection.
# It allows users to input queries and retrieves the top matching documents based on semantic similarity.

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch

# Load model and tokenizer for consistent embedding
model_name = "BAAI/bge-m3"
model = SentenceTransformer(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Connect to ChromaDB persistent store
chroma_client = chromadb.PersistentClient(path="../output/chroma_store")
collection = chroma_client.get_collection(name="campus_documents")

# Input loop
def query_loop():
    print("\n Semantic Search Interface (type 'exit' to quit)")
    while True:
        user_query = input("\n Enter your question: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print(" Exiting search.")
            break

        # Embed the query
        query_embedding = model.encode([user_query], normalize_embeddings=True)

        # Perform semantic search
        results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5,
        include=["documents", "metadatas", "distances"]
)

        # Display results
        print("\n Top Matches:")
        for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):

            score = round(1 - dist, 4) if dist is not None else "N/A"
            print(f"\nRank {i+1} | Score: {score}")
            print(f"Source: {meta['file_name']} | Chunk: {meta['chunk_id']} | Tokens: {meta['num_tokens']}")
            print(f"Content: {doc[:300]}{'...' if len(doc) > 300 else ''}")

if __name__ == "__main__":
    query_loop()
