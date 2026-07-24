# -*- coding: utf-8 -*-
"""
02_preprocessing.py
--------------------
Step 2 of the RAG pipeline: PREPROCESSING / TEXT CLEANING.

Takes the raw text extracted in 01_documents.py and cleans it up before
chunking. Two real-world text quality problems were found in the source
PDFs (official CBE "Book 4" regulation sections) and are handled here:

1. CHARACTER-ORDER REVERSAL: pdfplumber extracts each line of Arabic text
   in reversed character order for these particular PDFs (a font-encoding
   quirk, not a pdfplumber bug per se). Reversing each line restores
   correct reading order. This affects ALL 6 source documents and is
   fixed automatically below.

2. OCR / FONT-SUBSTITUTION NOISE: one source document (section_08, the
   real-estate financing controls section) still contains garbled words
   after the reversal fix (e.g. "الوكا" instead of "البنك"), most likely
   because that particular PDF was produced from a scanned/OCR'd source
   with a different, lower-quality text layer. This cannot be reliably
   auto-corrected with rule-based text cleaning, so this script flags
   that document as "low_quality" in its metadata rather than silently
   pretending it's clean. It is still included in the pipeline (so the
   RAG assistant can retrieve it and a human can sanity-check answers
   coming from it), but this is a known, documented limitation.

Output: data/02_preprocessing/cleaned_documents.json
"""

import os
import json
import re

INPUT_FILE = "data/01_documents/documents.json"
OUTPUT_DIR = "data/02_preprocessing"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cleaned_documents.json")

# Known low-quality source documents (see module docstring above).
LOW_QUALITY_DOC_IDS = {"section_08_real_estate_financing"}


def fix_reversed_lines(raw_text: str) -> str:
    """Reverse each line's character order to restore correct Arabic reading order.

    Reversing the whole line also flips embedded LTR runs (numbers, percentages,
    dates like "2026") into the wrong order, since those were correct in the raw
    (pre-reversal) extraction. So after reversing the line, every run of digits
    (and common surrounding punctuation like '%', '.', '/') is reversed back a
    second time to restore correct left-to-right numeric order.
    """
    lines = raw_text.split("\n")
    fixed_lines = []
    for line in lines:
        if not line.strip():
            fixed_lines.append(line)
            continue
        reversed_line = line[::-1]
        fixed_line = re.sub(
            r"[0-9%\.,/]+",
            lambda m: m.group(0)[::-1],
            reversed_line,
        )
        fixed_lines.append(fixed_line)
    return "\n".join(fixed_lines)


def clean_text(text: str) -> str:
    """General cleanup: remove page-break markers, collapse whitespace, strip stray page numbers."""
    # Remove the page-break marker inserted in 01_documents.py; treat it as a paragraph break.
    text = text.replace("PAGE_BREAK", " ")
    text = text.replace("][", " ")

    # Collapse 3+ blank lines into a single paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces/tabs (but keep newlines).
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip a bare single digit or short number sitting alone on its own line
    # (typical page-number artifact left over from PDF extraction).
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)

    # Trim trailing/leading whitespace per line.
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Collapse remaining multiple blank lines again after stripping.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def preprocess_documents(documents: list) -> list:
    cleaned_documents = []

    for doc in documents:
        fixed = fix_reversed_lines(doc["raw_text"])
        cleaned = clean_text(fixed)

        quality = "low_quality" if doc["doc_id"] in LOW_QUALITY_DOC_IDS else "ok"

        cleaned_documents.append({
            "doc_id": doc["doc_id"],
            "source_file": doc["source_file"],
            "title": doc["title"],
            "num_pages": doc["num_pages"],
            "text_quality": quality,
            "cleaned_text": cleaned,
        })

        print(f"Cleaned: {doc['doc_id']}  (quality={quality}, {len(cleaned):,} chars)")

    return cleaned_documents


def main():
    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found. Run 01_documents.py first.")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    cleaned_documents = preprocess_documents(documents)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_documents, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(cleaned_documents)} cleaned documents to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
