"""
Core RAG logic — no web framework code here on purpose.
This module owns: loading/chunking documents, building the embedding index,
retrieval (with optional metadata filtering, FR-4), and grounded generation
with short-term conversational memory (FR-7).

Embeddings are kept in an in-memory numpy array for similarity search (the
SRS lists FAISS / MongoDB Atlas Vector Search as options — either is a
drop-in replacement for `retrieve()` later without touching callers).
Chunk *text + metadata* is persisted to MongoDB (`document_chunks`) so it
survives restarts and satisfies FR-10/Section 14.1; only the embedding
vectors themselves are rebuilt in memory at startup.
"""

import glob
import os

import numpy as np
from google import genai
from google.genai import types

from app import config, db

client = genai.Client(api_key=config.GEMINI_API_KEY)

# Best-effort mapping from filename keywords to a domain category, so
# ingested documents get useful metadata (FR-4) even without a real ERP
# feeding structured fields yet.
_FILENAME_CATEGORY_HINTS = {
    "attendance": "attendance",
    "timetable": "timetable",
    "course": "course_information",
    "faculty": "faculty_details",
    "exam": "examination_schedule",
    "regulation": "academic_regulations",
    "notice": "notices",
    "assignment": "assignments",
}


def _infer_category(filename: str) -> str:
    lowered = filename.lower()
    for hint, category in _FILENAME_CATEGORY_HINTS.items():
        if hint in lowered:
            return category
    return "general_erp_usage"


def load_and_chunk(data_dir=None):
    """Reads every .txt file in data_dir, chunks it, and returns a list of
    dicts. Does NOT touch MongoDB or embeddings — pure text processing, so
    it stays easy to unit test."""
    data_dir = data_dir or config.DATA_DIR
    chunks = []
    for filepath in glob.glob(os.path.join(data_dir, "*.txt")):
        source_name = os.path.basename(filepath)
        category = _infer_category(source_name)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        start = 0
        chunk_index = 0
        while start < len(text):
            end = start + config.CHUNK_SIZE
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": source_name,
                        "category": category,
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1
            start += config.CHUNK_SIZE - config.CHUNK_OVERLAP
    return chunks


def persist_chunks(chunks):
    """Upserts chunk text + metadata into MongoDB (FR-10). Embeddings are
    NOT stored here — they live in memory for the lifetime of the process."""
    db.document_chunks.delete_many({})
    if not chunks:
        return
    docs = [
        {
            "source": c["source"],
            "category": c["category"],
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "created_at": db.utcnow(),
        }
        for c in chunks
    ]
    db.document_chunks.insert_many(docs)


def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
    result = client.models.embed_content(
        model=config.EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [np.array(e.values) for e in result.embeddings]


def build_index(chunks):
    """Embeds every chunk and persists chunk metadata to MongoDB. Returns
    the chunks with an in-memory `embedding` field attached."""
    if not chunks:
        persist_chunks(chunks)
        return chunks
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    persist_chunks(chunks)
    return chunks


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query, chunks, k=None, category=None):
    """Top-k similarity search, with optional metadata filter (FR-4)."""
    k = k or config.TOP_K
    candidates = chunks
    if category:
        candidates = [c for c in chunks if c.get("category") == category]
        if not candidates:  # filter too narrow — fall back to the full set
            candidates = chunks
    if not candidates:
        return []
    query_embedding = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
    scored = [(cosine_similarity(query_embedding, c["embedding"]), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def generate_answer(query, retrieved, history=None):
    """Generates a grounded answer. `history` is an optional list of
    {"role": "user"|"assistant", "text": ...} dicts for short-term
    conversational memory (FR-7) — kept small (last few turns) by the
    caller, not here."""
    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for _, c in retrieved
    )
    history_block = ""
    if history:
        history_block = "\n".join(
            f"{turn['role'].upper()}: {turn['text']}" for turn in history
        )
        history_block = f"\nRECENT CONVERSATION:\n{history_block}\n"

    prompt = f"""You are an academic assistant for a university ERP/LMS system.
Answer the user's question using ONLY the context provided below.
If the answer is not present in the context, say you don't have that information
— do not guess or use outside knowledge.
Cite the source file(s) you used at the end of your answer.
{history_block}
CONTEXT:
{context_block}

QUESTION:
{query}

ANSWER:"""
    response = client.models.generate_content(model=config.GEN_MODEL, contents=prompt)
    return response.text


def answer_query(query, chunks, category=None, history=None):
    """Single entry point: retrieve + generate. Returns (answer_text, source_list)."""
    retrieved = retrieve(query, chunks, category=category)
    if not retrieved:
        return (
            "I don't have any indexed documents to answer that from yet.",
            [],
        )
    sources = sorted(set(c["source"] for _, c in retrieved))
    answer = generate_answer(query, retrieved, history=history)
    return answer, sources
