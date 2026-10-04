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
import json
import os
import re
import time

import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app import config, db

client = genai.Client(api_key=config.GEMINI_API_KEY)

# Kept as a re-export so existing imports (e.g. app.main) don't break.
_infer_category = config.infer_category


def _chunk_text(text, start_index=0):
    """Splits a block of text into overlapping chunks. Returns a list of
    (chunk_index, chunk_text) tuples, continuing the index count from
    start_index (so chunk numbers stay unique across a multi-page document)."""
    pieces = []
    start = 0
    idx = start_index
    while start < len(text):
        end = start + config.CHUNK_SIZE
        chunk_text = text[start:end].strip()
        if chunk_text:
            pieces.append((idx, chunk_text))
            idx += 1
        start += config.CHUNK_SIZE - config.CHUNK_OVERLAP
    return pieces


def load_and_chunk(data_dir=None):
    """Reads every .json document in data_dir (the canonical schema produced
    by pdf_ingest.py — source/category/pages[{page_number, text}]) and
    chunks each page's text. Returns a list of chunk dicts. Does NOT touch
    MongoDB or embeddings — pure data processing, so it stays easy to test.

    Data is stored as JSON rather than plain .txt so each chunk can carry
    real structure (page numbers, explicit category) instead of it being
    inferred after the fact."""
    data_dir = data_dir or config.DATA_DIR
    chunks = []
    for filepath in glob.glob(os.path.join(data_dir, "*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                doc = json.load(f)
            except json.JSONDecodeError:
                continue  # skip malformed files rather than crash startup

        source_name = doc.get("source") or os.path.basename(filepath)
        category = doc.get("category") or config.infer_category(source_name)
        chunk_index = 0
        for page in doc.get("pages", []):
            page_text = page.get("text", "")
            page_number = page.get("page_number")
            for idx, chunk_text in _chunk_text(page_text, chunk_index):
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": source_name,
                        "category": category,
                        "chunk_index": idx,
                        "page": page_number,
                    }
                )
                chunk_index = idx + 1
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
            "page": c.get("page"),
            "text": c["text"],
            "created_at": db.utcnow(),
        }
        for c in chunks
    ]
    db.document_chunks.insert_many(docs)


EMBED_BATCH_SIZE = 100  # Gemini's embed_content caps a batch at 100 requests
MAX_RATE_LIMIT_RETRIES = 6


def _extract_retry_delay(error, default=30.0):
    """Gemini's 429 message includes 'Please retry in 30.3s' — pull that
    out so we wait exactly as long as asked rather than guessing."""
    match = re.search(r"retry in ([\d.]+)s", str(error))
    if match:
        return float(match.group(1)) + 1.0  # small safety buffer
    return default


def _embed_batch_with_retry(batch, task_type):
    """Calls embed_content with retry-and-wait on 429 RESOURCE_EXHAUSTED —
    the free tier caps embed_content at 100 requests/minute *total*, so a
    document with enough chunks to need multiple batches will reliably hit
    this on the second batch within the same minute. Waiting out the
    quota window (rather than failing) is the correct fix, not just a
    bigger batch size."""
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return client.models.embed_content(
                model=config.EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
        except genai_errors.ClientError as e:
            is_rate_limit = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
            if not is_rate_limit or attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            delay = _extract_retry_delay(e)
            print(
                f"Embedding rate limit hit — waiting {delay:.0f}s "
                f"(attempt {attempt}/{MAX_RATE_LIMIT_RETRIES})..."
            )
            time.sleep(delay)


def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
    """Embeds a list of texts, splitting into batches of EMBED_BATCH_SIZE
    (the API rejects a single call with more than 100 texts) and retrying
    with backoff on rate-limit errors between batches."""
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        result = _embed_batch_with_retry(batch, task_type)
        all_embeddings.extend(np.array(e.values) for e in result.embeddings)
    return all_embeddings


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
    def _label(c):
        return f"{c['source']}, p.{c['page']}" if c.get("page") else c["source"]

    context_block = "\n\n".join(
        f"[Source: {_label(c)}]\n{c['text']}" for _, c in retrieved
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
    def _label(c):
        return f"{c['source']} (p.{c['page']})" if c.get("page") else c["source"]

    sources = sorted(set(_label(c) for _, c in retrieved))
    answer = generate_answer(query, retrieved, history=history)
    return answer, sources