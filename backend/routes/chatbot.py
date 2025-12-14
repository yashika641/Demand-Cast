from fastapi import APIRouter, HTTPException
import faiss
import pickle
import ollama
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np

router = APIRouter()

# ------------------------------
# Load models only once
# ------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

faiss_index = faiss.read_index(r"C:\Users\palya\Desktop\demancast\models\RAG\embeddings\faiss.index")
with open(r"C:\Users\palya\Desktop\demancast\models\RAG\embeddings\metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


# ------------------------------
# Input schema
# ------------------------------
class ChatRequest(BaseModel):
    query: str


# ------------------------------
# FAISS Context Retrieval
# ------------------------------
def get_faiss_context(query: str, top_k: int = 5) -> str:
    q_emb = embedder.encode([query]).astype("float32")
    distances, ids = faiss_index.search(q_emb, top_k)

    context_chunks = []
    for idx in ids[0]:
        context_chunks.append(metadata[idx]["text"])

    return "\n\n---\n\n".join(context_chunks)


# ------------------------------
# Llama 3 Chatbot Logic
# ------------------------------
def llama3_answer(query: str) -> str:
    context = get_faiss_context(query)

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the following context to answer the question. 
If the answer is not available, reply: "Not available in the knowledge base."

Context:
{context}

User question: {query}
"""

    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------
# 🚀 Chatbot API Route
# ------------------------------
@router.post("/chatbot")
def chatbot_endpoint(request: ChatRequest):
    answer = llama3_answer(request.query)
    return {"answer": answer}
