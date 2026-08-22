"""
Core RAG logic — no web framework code here on purpose.
This module owns: loading/chunking documents, building the in-memory
embedding index, retrieval, and grounded generation.

Kept framework-agnostic so it can be unit-tested directly (see stress_test.py)
without needing the API running.
"""

import os
import glob
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # reads .env in the project root and sets os.environ vars

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY as an environment variable first.")

client = genai.Client(api_key=API_KEY)

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.6-flash"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 3


def load_and_chunk(data_dir="data"):
    chunks = []
    for filepath in glob.glob(os.path.join(data_dir, "*.txt")):
        source_name = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "source": source_name})
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [np.array(e.values) for e in result.embeddings]


def build_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query, chunks, k=TOP_K):
    query_embedding = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
    scored = [(cosine_similarity(query_embedding, c["embedding"]), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def generate_answer(query, retrieved):
    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for _, c in retrieved
    )
    prompt = f"""You are an academic assistant for a university ERP/LMS system.
Answer the user's question using ONLY the context provided below.
If the answer is not present in the context, say you don't have that information
— do not guess or use outside knowledge.
Cite the source file(s) you used at the end of your answer.

CONTEXT:
{context_block}

QUESTION:
{query}

ANSWER:"""
    response = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    return response.text


def answer_query(query, chunks):
    """Single entry point: retrieve + generate. Returns (answer_text, source_list)."""
    retrieved = retrieve(query, chunks)
    sources = sorted(set(c["source"] for _, c in retrieved))
    answer = generate_answer(query, retrieved)
    return answer, sources
