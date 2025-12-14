print("🚀 Embedding script STARTED:", __file__)
print("TEST START")

import os
print("os OK")

import pickle
print("pickle OK")

import pandas as pd
print("pandas OK")

import numpy as np
print("numpy OK")

import faiss
print("faiss OK")
print("Loading torch...")
import torch
print("Torch loaded:", torch.__version__)
print("Loading transformers...")
import transformers
print("Transformers OK")
print("Loading tokenizers...")
import tokenizers
print("Tokenizers OK")
print("Trying *only* sentence-transformers")
import sentence_transformers
print("Imported base module")
from sentence_transformers import SentenceTransformer
print("SentenceTransformer imported successfully")
import datetime
from datetime import datetime
print("datetime OK")

print("ALL IMPORTS SUCCESS")

print("imports done")
# ---------------------------------------------
# Utility: timestamped logging
# ---------------------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]  {msg}")


# ---------------------------------------------
# 1. Load files
# ---------------------------------------------
def load_text_from_file(file_path):
    ext = file_path.split('.')[-1].lower()
    log(f"➡ Loading file: {file_path}")

    try:
        if ext == "pdf":
            # reader = PdfReader(file_path)
            # text = "\n".join([page.extract_text() or "" for page in reader.pages])
            # return text
            return

        elif ext == "csv":
            df = pd.read_csv(file_path)
            return df.to_string()

        elif ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        else:
            log(f"⚠ Unsupported file type: {file_path}")
            return ""

    except Exception as e:
        log(f"❌ Error reading file {file_path}: {e}")
        return ""


# ---------------------------------------------
# 2. Chunk long text
# ---------------------------------------------
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ---------------------------------------------
# 3. Embedding function
# ---------------------------------------------
log("🔄 Loading embedding model (MiniLM)...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(text_list):
    log(f"🔢 Creating embeddings for {len(text_list)} chunks...")
    return model.encode(text_list, show_progress_bar=True)


# ---------------------------------------------
# 4. Process all files + build embeddings
# ---------------------------------------------
def build_faiss_index(data_folder="data", save_folder="embeddings"):

    log("🚀 Starting FAISS index creation pipeline...")
    log(f"📁 Data folder: {data_folder}")
    log(f"💾 Save folder: {save_folder}")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        log(f"📂 Created folder: {save_folder}")

    all_chunks = []
    metadata = []

    # Scan directory
    log("🔍 Scanning data folder for files...")
    files_found = []

    for root, _, files in os.walk(data_folder):
        for file in files:
            files_found.append(os.path.join(root, file))

    log(f"📌 Total files found: {len(files_found)}")

    for idx, file_path in enumerate(files_found, start=1):
        log(f"\n📄 [{idx}/{len(files_found)}] Processing: {file_path}")

        text = load_text_from_file(file_path)
        if not text.strip():
            log(f"⚠ File empty or unreadable: {file_path}")
            continue

        chunks = chunk_text(text)
        log(f"🔹 Created {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({
                "file": os.path.basename(file_path),
                "chunk_id": i,
                "text": chunk,
            })

    log(f"\n🧩 TOTAL chunks collected: {len(all_chunks)}")

    # 5. Embed
    vectors = embed_texts(all_chunks)
    vectors = np.array(vectors).astype("float32")

    # 6. Build FAISS index
    dim = vectors.shape[1]
    log(f"📐 Embedding dimension: {dim}")

    log("📦 Building FAISS index...")
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # 7. Save files
    faiss_path = f"{save_folder}/faiss.index"
    meta_path = f"{save_folder}/metadata.pkl"

    faiss.write_index(index, faiss_path)
    log(f"💾 Saved FAISS index to: {faiss_path}")

    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    log(f"💾 Saved metadata ({len(metadata)} entries) to: {meta_path}")

    log("\n✅ FAISS index creation completed successfully!")
    return index, metadata


# ---------------------------------------------
# MAIN ENTRY
# ---------------------------------------------
def main():
    build_faiss_index(
        data_folder=r'C:\Users\palya\Desktop\demancast\models\RAG\knowledge_base',
        save_folder=r'C:\Users\palya\Desktop\demancast\models\RAG\embeddings'
    )


if __name__ == "__main__":
    main()
