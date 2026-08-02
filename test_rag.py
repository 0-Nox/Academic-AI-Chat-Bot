"""
Walking-skeleton RAG pipeline — validates retrieval + generation logic
before any database, API server, or Docker gets involved.

Usage:
    python test_rag.py

Folder layout expected:
    data/
        timetable.txt
        attendance.txt
        exam_schedule.txt
        notices.txt
        regulations.txt
"""

import os
import glob
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------------------
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY as an environment variable first.")

client = genai.Client(api_key=API_KEY)

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-pro"

CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 80  # overlap so context isn't cut mid-sentence
TOP_K = 3  # how many chunks to retrieve per query


# ---------------------------------------------------------------------------
# 1. Load & chunk documents
# ---------------------------------------------------------------------------
def load_and_chunk(data_dir="data"):
    chunks = []  # list of dicts: {text, source}
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

    print(f"Loaded {len(chunks)} chunks from {data_dir}/")
    return chunks


# ---------------------------------------------------------------------------
# 2. Embed chunks (in-memory store — no vector DB yet)
# ---------------------------------------------------------------------------
def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
    """Batch-embed a list of strings using Gemini's embedding model."""
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


# ---------------------------------------------------------------------------
# 3. Retrieve top-k chunks for a query
# ---------------------------------------------------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query, chunks, k=TOP_K):
    query_embedding = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
    scored = [(cosine_similarity(query_embedding, c["embedding"]), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


# ---------------------------------------------------------------------------
# 4. Augment prompt + generate grounded answer
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 5. Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    chunks = load_and_chunk("data")
    chunks = build_index(chunks)

    test_queries = [
        "What is the minimum attendance percentage required?",
        "When is the CS301 mid-semester exam?",
        "Are there any notices about fee deadlines?",
    ]

    for q in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {q}")
        retrieved = retrieve(q, chunks)
        print("Retrieved chunks from:", [c["source"] for _, c in retrieved])
        answer = generate_answer(q, retrieved)
        print(f"\nANSWER:\n{answer}")
