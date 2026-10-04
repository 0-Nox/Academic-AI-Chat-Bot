"""
PDF -> JSON converter (per mentor guidance: ingest PDFs directly and store
institutional data as structured JSON rather than plain .txt).

Converts a PDF into the project's canonical document schema:

{
  "source": "attendance_policy.pdf",
  "category": "attendance",
  "page_count": 2,
  "pages": [
    {"page_number": 1, "text": "..."},
    {"page_number": 2, "text": "..."}
  ]
}

This module only extracts and structures text — chunking for embeddings
still happens in rag.py's load_and_chunk(), which reads this same JSON
schema. Keeping extraction and chunking separate means either can change
independently (e.g. swapping in OCR for scanned PDFs later) without
touching the other.

Uses pdfplumber rather than pypdf: better text-layout handling for the
kind of multi-column/table-ish institutional PDFs (timetables, schedules)
this project targets. See /mnt/skills/public/pdf-reading/SKILL.md for the
broader set of extraction strategies (OCR, rasterization, tables) if a
given PDF's text layer turns out to be unreliable.
"""

import json
import os

import pdfplumber

from app import config


def pdf_to_doc(pdf_path: str) -> dict:
    """Extracts text per page from a PDF and returns the canonical doc dict.
    Raises ValueError if the PDF has no extractable text layer (i.e. it's
    scanned/raster-only) — those need OCR first, which this module does
    not attempt (see module docstring)."""
    source_name = os.path.basename(pdf_path)
    category = config.infer_category(source_name)

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page_number": i, "text": text})

    if not pages:
        raise ValueError(
            f"No extractable text found in {source_name}. It may be a "
            "scanned/image-only PDF — OCR would be needed first "
            "(see pdf-reading skill)."
        )

    return {
        "source": source_name,
        "category": category,
        "page_count": len(pages),
        "pages": pages,
    }


def convert_pdf_file(pdf_path: str, output_dir: str = None) -> str:
    """Converts a PDF on disk into a .json file in output_dir (default:
    config.DATA_DIR) using the same base filename. Returns the output path."""
    output_dir = output_dir or config.DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    doc = pdf_to_doc(pdf_path)
    json_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".json"
    out_path = os.path.join(output_dir, json_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return out_path


def text_to_doc(text: str, source_name: str) -> dict:
    """Wraps plain text (e.g. a legacy .txt upload) into the same schema,
    as a single 'page'. Kept for backward compatibility with .txt uploads."""
    category = config.infer_category(source_name)
    return {
        "source": source_name,
        "category": category,
        "page_count": 1,
        "pages": [{"page_number": 1, "text": text}],
    }


if __name__ == "__main__":
    # CLI usage: python -m app.pdf_ingest path/to/file.pdf [more.pdf ...]
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m app.pdf_ingest <pdf_path> [more_pdfs...]")
        raise SystemExit(1)
    for path in sys.argv[1:]:
        try:
            out_path = convert_pdf_file(path)
            print(f"{path} -> {out_path}")
        except ValueError as e:
            print(f"Skipped {path}: {e}")
