# -*- coding: utf-8 -*-
"""
04_vector_representation.py
-----------------------------
Step 4 of the RAG pipeline: VECTOR REPRESENTATION.

Encodes every chunk from 03_chunking.py into a dense embedding vector using
a multilingual sentence-transformers model. A multilingual model is used
(rather than an English-only one) because the source documents and the
questions users will ask are in Arabic.

Model: paraphrase-multilingual-MiniLM-L12-v2
- Supports Arabic (and 50+ other languages).
- Small enough (~470MB) to run on CPU, which matters for free-tier
  deployment (Streamlit Cloud, Railway, etc.) where no GPU is available.

Output: data/04_vector_representation/embeddings.npy  (float32 array, shape
        [num_chunks, embedding_dim])
        data/04_vector_representation/chunks_with_ids.json  (same chunk
        metadata as 03_chunking.py, in the exact same row order as the
        embeddings array, so row i of embeddings.npy corresponds to
        chunks_with_ids.json[i])
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/03_chunking/chunks.json"
OUTPUT_DIR = "data/04_vector_representation"
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
CHUNKS_FILE = os.path.join(OUTPUT_DIR, "chunks_with_ids.json")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks() -> list:
    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run 03_chunking.py first.")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_chunks(chunks: list, model_name: str = MODEL_NAME) -> np.ndarray:
    print(f"Loading embedding model: {model_name} (first run downloads the model)")
    model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]
    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # normalized vectors -> cosine similarity via dot product
    )
    return embeddings.astype(np.float32)


def main():
    chunks = load_chunks()
    embeddings = embed_chunks(chunks)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Saved embeddings to: {EMBEDDINGS_FILE}")
    print(f"Saved aligned chunk metadata to: {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
