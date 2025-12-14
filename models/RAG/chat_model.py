import pickle
print("pickle ok")
import faiss
print("faiss ok")
from sentence_transformers import SentenceTransformer
print("sentence transformer ok")
model = SentenceTransformer("all-MiniLM-L6-v2")

def search_faiss(query, top_k=5, index_path="embeddings/faiss.index", meta_path="embeddings/metadata.pkl"):
    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    q_emb = model.encode([query]).astype("float32")
    distances, ids = index.search(q_emb, top_k)

    results = []
    for i in range(top_k):
        meta = metadata[ids[0][i]]
        results.append({
            "score": float(distances[0][i]),
            "file": meta["file"],
            "chunk_id": meta["chunk_id"],
            "text": meta["text"][:300] + "..."
        })

    return results

import faiss
import pickle
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

# -------------------------
# Load embedder
# -------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Load FAISS + metadata
# -------------------------
faiss_index = faiss.read_index(r"C:\Users\palya\Desktop\demancast\models\RAG\embeddings\faiss.index")

with open(r"C:\Users\palya\Desktop\demancast\models\RAG\embeddings\metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


# ----------------------------------------------------
# 1. Retrieve context from FAISS
# ----------------------------------------------------
def get_faiss_context(query, top_k=5):
    qv = embedder.encode([query]).astype("float32")
    distances, ids = faiss_index.search(qv, top_k)

    context = ""
    for idx in ids[0]:
        chunk = metadata[idx]
        context += f"\n\n---\n{chunk['text']}\n"
    return context


# ----------------------------------------------------
# 2. Chatbot function using FREE Llama3 (Ollama)
# ----------------------------------------------------
def chatbot(query):
    # Retrieve relevant chunks
    context = get_faiss_context(query)

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the following context to answer the user. 
If answer is not found, say: "Not available in knowledge base".

Context:
{context}

User question: {query}
"""

    # Call local llama3 model
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]

def chatbot_stream(query):
    context = get_faiss_context(query)

    prompt = f"""
Use this context to answer:

{context}

Question: {query}
"""

    stream = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)

if __name__ == "__main__":
    print(chatbot("who is the creater of this project?"))
