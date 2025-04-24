# This script retrieves the context from the vector database using the query and the embedding model.
# It then uses the context to generate an answer using the GPT model.

from sentence_transformers import SentenceTransformer
import torch
from chromadb import PersistentClient

# Load vector DB
client = PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="document_chunks")

# Load embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-m3", device=device)

from pipeline.query_index import query_vector, query_rerank

def get_top_vector(query):
    return query_vector(query, top_k=1)

def get_top_reranked(query, return_top_k=False):
    chunks = query_rerank(query, rerank_top_k=5)
    return chunks if return_top_k else chunks[:1]

def retrieve_context(query, mode="vector"):
    if mode == "vector":
        return get_top_vector(query)
    elif mode == "rerank":
        return get_top_reranked(query)
    elif mode == "gpt":
        return get_top_reranked(query, return_top_k=True)

