"""
FastAPI layer. Owns HTTP concerns only — request/response shapes, startup,
and error handling. All RAG logic lives in rag.py.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import rag

# In-memory index, built once at startup (still no MongoDB/vector DB yet —
# that's the next increment).
index_store = {"chunks": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading and embedding documents...")
    chunks = rag.load_and_chunk("data")
    chunks = rag.build_index(chunks)
    index_store["chunks"] = chunks
    print(f"Index ready: {len(chunks)} chunks loaded.")
    yield
    # (nothing to clean up yet — placeholder for when we add a DB connection)


app = FastAPI(title="Academic Chatbot API", lifespan=lifespan)


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(index_store["chunks"])}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    if not index_store["chunks"]:
        # Graceful degradation per NFR — don't crash, tell the caller clearly.
        raise HTTPException(status_code=503, detail="Knowledge base not ready yet.")

    try:
        answer, sources = rag.answer_query(request.query, index_store["chunks"])
    except Exception as e:
        # In production this would be logged with more detail (Section 9,
        # Evaluation & Monitoring). For now, fail safely rather than leak
        # internals to the caller.
        raise HTTPException(
            status_code=500, detail="Failed to generate a response."
        ) from e

    return ChatResponse(answer=answer, sources=sources)
