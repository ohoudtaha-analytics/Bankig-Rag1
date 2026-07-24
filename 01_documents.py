# -*- coding: utf-8 -*-
"""
01_documents.py
----------------
Step 1 of the RAG pipeline: DOCUMENT LOADING.

Loads the raw source PDFs (official Central Bank of Egypt "Book 4 - Credit
Granting Controls" sections) from data/raw_pdfs/, extracts their raw text
using pdfplumber, and saves one JSON record per document to
data/01_documents/documents.json.

This step does NOT clean or normalize the text - that happens in
02_preprocessing.py. This step is only responsible for getting text out
of the PDF files and attaching basic metadata (source filename, page count,
section title) to each document.
"""

import os
import json
import pdfplumber

RAW_PDF_DIR = "data/raw_pdfs"
OUTPUT_DIR = "data/01_documents"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "documents.json")

# Human-readable titles for each source file, used as metadata.
# Update this dictionary whenever a new source PDF is added to data/raw_pdfs/.
SECTION_TITLES = {
    "section_01_general_controls.pdf": "الباب الرابع - الفصل الأول - القسم الأول: ضوابط عامة لمنح الائتمان",
    "section_02_real_estate_dev_companies.pdf": "الباب الرابع - الفصل الأول - القسم الثاني: تمويل نشاط شركات التنمية العقارية",
    "section_03_acquisition_goodwill.pdf": "الباب الرابع - الفصل الأول - القسم الثالث: ضوابط التمويل لالستحواذ على الشركات وتقييم الشهرة",
    "section_06_import_operations.pdf": "الباب الرابع - الفصل الأول - القسم السادس: العمليات الاستيرادية",
    "section_08_real_estate_financing.pdf": "الباب الرابع - الفصل الأول - القسم الثامن: ضوابط التمويل العقاري",
    "section_09_margin_trading.pdf": "الباب الرابع - الفصل الأول - القسم التاسع: ضوابط تمويل عمليات الشراء بالهامش",
}


def extract_pdf_text(pdf_path: str) -> str:
    """Extract raw text from every page of a PDF and join with page breaks."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    return "\n[PAGE_BREAK]\n".join(pages_text)


def load_documents() -> list:
    """Load every PDF in RAW_PDF_DIR and return a list of document records."""
    documents = []

    if not os.path.isdir(RAW_PDF_DIR):
        raise FileNotFoundError(
            f"Raw PDF directory not found: {RAW_PDF_DIR}. "
            "Place source PDFs there before running this script."
        )

    pdf_files = sorted(f for f in os.listdir(RAW_PDF_DIR) if f.lower().endswith(".pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {RAW_PDF_DIR}.")

    for filename in pdf_files:
        pdf_path = os.path.join(RAW_PDF_DIR, filename)
        print(f"Loading: {filename}")

        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)

        raw_text = extract_pdf_text(pdf_path)

        doc_id = os.path.splitext(filename)[0]
        documents.append({
            "doc_id": doc_id,
            "source_file": filename,
            "title": SECTION_TITLES.get(filename, doc_id),
            "num_pages": num_pages,
            "raw_text": raw_text,
        })

    return documents


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    documents = load_documents()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    total_chars = sum(len(d["raw_text"]) for d in documents)
    print(f"\nLoaded {len(documents)} documents ({total_chars:,} raw characters total).")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
