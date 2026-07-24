# -*- coding: utf-8 -*-
"""
05_create_chroma_store.py
---------------------------
Step 5 of the RAG pipeline: VECTOR STORE.

Loads the chunk embeddings produced by 04_vector_representation.py and
writes them into a persistent ChromaDB collection on disk. This is the
store that 06_retrieve_context.py (and, at query time, streamlit_app.py)
will query for the most relevant chunks given a user question.

ChromaDB is used (rather than FAISS) because it persists to a simple local
folder with zero extra setup, which is convenient both for local
development and for committing/re-loading the store in a small student
project.

Output: chroma_store/  (persistent ChromaDB directory - commit this folder
        to the project/GitHub repo so the deployed Streamlit app can load
        it directly without re-embedding anything at deploy time)
"""

import os
import json
import numpy as np
import chromadb

EMBEDDINGS_FILE = "data/04_vector_representation/embeddings.npy"
CHUNKS_FILE = "data/04_vector_representation/chunks_with_ids.json"
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "credit_policy_chunks"


def load_data():
    if not os.path.isfile(EMBEDDINGS_FILE) or not os.path.isfile(CHUNKS_FILE):
        raise FileNotFoundError(
            "Embeddings not found. Run 04_vector_representation.py first "
            "(requires internet access to download the embedding model, "
            "e.g. run it on Colab / Lightning AI / your local machine)."
        )
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return embeddings, chunks


def build_store(embeddings: np.ndarray, chunks: list):
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Recreate the collection fresh each time this script runs, so re-running
    # it after re-chunking/re-embedding doesn't leave stale duplicate entries.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "source_file": chunk["source_file"],
            "text_quality": chunk["text_quality"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    return collection


def main():
    embeddings, chunks = load_data()
    print(f"Loaded {len(chunks)} chunks with embeddings of dim {embeddings.shape[1]}")

    os.makedirs(CHROMA_DIR, exist_ok=True)
    collection = build_store(embeddings, chunks)

    print(f"Chroma collection '{COLLECTION_NAME}' created with {collection.count()} items.")
    print(f"Persisted to: {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
