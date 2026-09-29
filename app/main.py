"""
FastAPI layer. Owns HTTP concerns only — request/response shapes, startup,
auth wiring, and error handling. RAG logic lives in rag.py, routing/intent
in agent.py, safety checks in guardrails.py, persistence in db.py/security.py.

Endpoint <-> requirement map:
  POST /auth/login   - issues a JWT (Security Requirements, Section 19)
  GET  /health        - FR-11, NFR Reliability (never raises on a dead dep)
  POST /chat          - FR-1, FR-2, FR-3, FR-5, FR-6, FR-7, FR-8
  POST /feedback      - FR-12
  POST /ingest        - FR-10 (Admin only)
  GET  /logs          - FR-9 / Auditability (Admin only)
"""

import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from app import agent, config, db, guardrails, rag, security

# In-memory index: chunk text/metadata + embeddings, rebuilt at startup.
# (Vector store, per Section 14: embeddings don't need to survive a
# restart, only the chunk text/metadata does — that's in MongoDB.)
index_store = {"chunks": []}

# Short-term conversational memory, keyed by session id (FR-7). Kept small
# and in-memory; MongoDB `chat_sessions` holds the durable copy for history
# and evaluation (FR-9).
MAX_HISTORY_TURNS = 6


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.ensure_indexes()
    security.seed_demo_users()
    print("Loading and embedding documents...")
    chunks = rag.load_and_chunk()
    chunks = rag.build_index(chunks)
    index_store["chunks"] = chunks
    print(f"Index ready: {len(chunks)} chunks loaded.")
    yield


app = FastAPI(title="Academic Chatbot API", lifespan=lifespan)


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


# --------------------------------------------------------------------------
# Auth (Section 19)
# --------------------------------------------------------------------------
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


@app.post("/auth/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = security.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    token = security.create_access_token(user["username"], user["role"])
    return TokenResponse(access_token=token, role=user["role"])


# --------------------------------------------------------------------------
# Health (FR-11) — deliberately unauthenticated so uptime checks work, and
# never raises even if MongoDB is down (NFR: Reliability).
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_loaded": len(index_store["chunks"]),
        "database": db.status_label(),
    }


# --------------------------------------------------------------------------
# Chat (FR-1, FR-2, FR-3, FR-5, FR-6, FR-7, FR-8)
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str
    intent: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, user: security.TokenData = Depends(security.get_current_user)):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    session_id = request.session_id or str(uuid.uuid4())
    session = db.chat_sessions.find_one({"_id": session_id})
    history = (session or {}).get("messages", [])[-MAX_HISTORY_TURNS:]

    # --- Guardrails: input validation (Figure 17.2, steps 2-3) ---
    input_check = guardrails.check_input(query)
    if not input_check.allowed:
        _log_turn(session_id, user.username, query, "blocked", input_check.reason,
                   input_check.safe_message, [])
        return ChatResponse(
            answer=input_check.safe_message, sources=[], session_id=session_id,
            intent=input_check.reason,
        )

    # --- Agent Orchestrator: intent classification / routing (FR-2, Section 18) ---
    intent = agent.classify_intent(query)
    if intent == config.OUT_OF_SCOPE_LABEL:
        _log_turn(session_id, user.username, query, intent, "out_of_scope",
                   guardrails.FALLBACK_OUT_OF_SCOPE, [])
        return ChatResponse(
            answer=guardrails.FALLBACK_OUT_OF_SCOPE, sources=[], session_id=session_id,
            intent=intent,
        )

    if not index_store["chunks"]:
        raise HTTPException(status_code=503, detail="Knowledge base not ready yet.")

    try:
        answer, sources = rag.answer_query(
            query, index_store["chunks"], category=intent, history=history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate a response.") from e

    # --- Guardrails: output validation (Figure 17.2, steps 12-13) ---
    answer = guardrails.check_output(answer)

    _log_turn(session_id, user.username, query, intent, None, answer, sources)
    return ChatResponse(answer=answer, sources=sources, session_id=session_id, intent=intent)


def _log_turn(session_id, username, query, intent, block_reason, answer, sources):
    """Persists the turn to chat_sessions (FR-7 memory) and eval_logs
    (FR-9 / Auditability — every query, retrieved context, response)."""
    now = db.utcnow()
    db.chat_sessions.update_one(
        {"_id": session_id},
        {
            "$setOnInsert": {"user_id": username, "started_at": now},
            "$push": {
                "messages": {
                    "$each": [
                        {"role": "user", "text": query, "ts": now},
                        {"role": "assistant", "text": answer, "citations": sources, "ts": now},
                    ]
                }
            },
        },
        upsert=True,
    )
    db.eval_logs.insert_one(
        {
            "session_id": session_id,
            "user": username,
            "query": query,
            "intent": intent,
            "blocked_reason": block_reason,
            "sources": sources,
            "timestamp": now,
        }
    )


# --------------------------------------------------------------------------
# Feedback (FR-12)
# --------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    session_id: str
    rating: int  # +1 thumbs up, -1 thumbs down
    comment: str | None = None


@app.post("/feedback")
def submit_feedback(
    request: FeedbackRequest, user: security.TokenData = Depends(security.get_current_user)
):
    if request.rating not in (-1, 1):
        raise HTTPException(status_code=400, detail="rating must be 1 or -1.")
    db.feedback.insert_one(
        {
            "session_id": request.session_id,
            "user": user.username,
            "rating": request.rating,
            "comment": request.comment,
            "created_at": db.utcnow(),
        }
    )
    return {"status": "recorded"}


# --------------------------------------------------------------------------
# Document ingestion (FR-10) — Admin only (Section 10, 19)
# --------------------------------------------------------------------------
@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    user: security.TokenData = Depends(security.require_role("admin")),
):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported for now.")
    contents = (await file.read()).decode("utf-8", errors="ignore")

    import os
    os.makedirs(config.DATA_DIR, exist_ok=True)
    dest_path = os.path.join(config.DATA_DIR, file.filename)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(contents)

    chunks = rag.load_and_chunk()
    chunks = rag.build_index(chunks)
    index_store["chunks"] = chunks
    return {"status": "ingested", "chunks_loaded": len(chunks)}


# --------------------------------------------------------------------------
# Logs (FR-9 / Auditability) — Admin only
# --------------------------------------------------------------------------
@app.get("/logs")
def get_logs(
    limit: int = 50, user: security.TokenData = Depends(security.require_role("admin"))
):
    logs = list(
        db.eval_logs.find({}, {"_id": 0}).sort("timestamp", -1).limit(min(limit, 200))
    )
    return {"count": len(logs), "logs": logs}
