# -*- coding: utf-8 -*-
"""
03_chunking.py
---------------
Step 3 of the RAG pipeline: CHUNKING.

Splits each cleaned document into overlapping, retrieval-sized chunks.

Chunking strategy: paragraph-aware sliding window.
- Documents are first split on blank lines into paragraphs (these regulatory
  texts are already organized into numbered clauses/paragraphs, so this keeps
  each numbered rule mostly intact).
- Paragraphs are then greedily packed into chunks of up to CHUNK_SIZE
  characters, so a chunk never cuts a short paragraph in half.
- Consecutive chunks overlap by CHUNK_OVERLAP characters so a rule that sits
  near a chunk boundary is not lost from the retriever's context.

Each chunk keeps a reference back to its source document (doc_id, title,
source_file) so retrieved chunks can be cited in the final answer.

Output: data/03_chunking/chunks.json
"""

import os
import json

INPUT_FILE = "data/02_preprocessing/cleaned_documents.json"
OUTPUT_DIR = "data/03_chunking"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "chunks.json")

CHUNK_SIZE = 800       # target max characters per chunk
CHUNK_OVERLAP = 150    # characters of overlap carried into the next chunk


def split_into_paragraphs(text: str) -> list:
    """Split cleaned text into paragraphs on blank lines."""
    raw_paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
    return paragraphs


def chunk_paragraphs(paragraphs: list, chunk_size: int, overlap: int) -> list:
    """Greedily pack paragraphs into overlapping chunks of up to chunk_size chars."""
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # If adding this paragraph would exceed the target size, close the
        # current chunk (if it has content) and start a new one that begins
        # with the trailing `overlap` characters of the chunk just closed.
        if current_chunk and len(current_chunk) + len(paragraph) + 1 > chunk_size:
            chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_text + "\n" + paragraph
        else:
            current_chunk = (current_chunk + "\n" + paragraph) if current_chunk else paragraph

        # Handle a single paragraph that is by itself longer than chunk_size.
        while len(current_chunk) > chunk_size * 1.5:
            chunks.append(current_chunk[:chunk_size].strip())
            current_chunk = current_chunk[chunk_size - overlap:]

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def build_chunks(documents: list) -> list:
    all_chunks = []
    chunk_counter = 0

    for doc in documents:
        paragraphs = split_into_paragraphs(doc["cleaned_text"])
        doc_chunks = chunk_paragraphs(paragraphs, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk_text in enumerate(doc_chunks):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}_chunk_{i:03d}",
                "doc_id": doc["doc_id"],
                "source_file": doc["source_file"],
                "title": doc["title"],
                "text_quality": doc["text_quality"],
                "chunk_index": i,
                "text": chunk_text,
            })
            chunk_counter += 1

        print(f"{doc['doc_id']}: {len(doc_chunks)} chunks")

    print(f"\nTotal chunks created: {chunk_counter}")
    return all_chunks


def main():
    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run 02_preprocessing.py first.")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    chunks = build_chunks(documents)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
