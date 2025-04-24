# pipeline/query.py – Semantic Search + GPT-4o Answering with .env support

import os
import torch
import chromadb
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Load embedding model and tokenizer
model_name = "BAAI/bge-m3"
model = SentenceTransformer(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="../output/chroma_store")
collection = chroma_client.get_collection(name="campus_documents")

def generate_answer(query, context_docs):
    context = "\n\n".join(context_docs)
    prompt = f"""
You are a helpful academic assistant. Use the context below to answer the question as clearly and informatively as possible.

Context:
{context}

Question: {query}
Answer:
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def query_loop():
    print("\n Semantic Search + GPT-4o Answering (type 'exit' to quit)")
    while True:
        user_query = input("\n🔍 Enter your question: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print(" Exiting search.")
            break

        # Embed the query
        query_embedding = model.encode([user_query], normalize_embeddings=True)

        # Perform search
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )

        print("\n Top Matches:")
        context_chunks = []
        for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
            score = round(1 - dist, 4)
            context_chunks.append(doc)
            print(f"\nRank {i+1} | Score: {score}")
            print(f"Source: {meta['file_name']} | Chunk: {meta['chunk_id']} | Tokens: {meta['num_tokens']}")
            print(f"Content: {doc[:300]}{'...' if len(doc) > 300 else ''}")

        # Generate answer
        print("\n GPT Answer:")
        final_answer = generate_answer(user_query, context_chunks)
        print(f"\n{final_answer}")

if __name__ == "__main__":
    query_loop()