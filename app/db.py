"""
MongoDB access layer (Section 14.1 — Data Architecture).

Collections match the SRS's logical schema:
  users, documents, document_chunks, chat_sessions, feedback, eval_logs

Kept as a thin wrapper around pymongo so the rest of the app never imports
pymongo directly — makes it easy to swap in Motor (async) or mongomock for
tests later without touching callers.
"""

from datetime import datetime, timezone
from pymongo import MongoClient, ASCENDING

from app import config

# MongoDB is convenient but not required to run the app locally: if nothing
# is listening at MONGODB_URI, fall back to an in-memory mock with the same
# collection API (mongomock). Same code path either way — only difference
# is data doesn't survive a restart in mock mode. Real MongoDB (or Atlas)
# is what you want for anything beyond local testing.
USING_REAL_MONGODB = True
try:
    _client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=1500)
    _client.admin.command("ping")
    print(f"Connected to MongoDB at {config.MONGODB_URI}.")
except Exception:
    import mongomock

    print(
        f"Could not reach MongoDB at {config.MONGODB_URI} — using an "
        "in-memory database for this run instead. Data (chat history, "
        "feedback, logs) will be lost when the app stops. Start a real "
        "MongoDB and re-run to persist data."
    )
    _client = mongomock.MongoClient()
    USING_REAL_MONGODB = False

_db = _client[config.MONGODB_DB_NAME]

users = _db["users"]
documents = _db["documents"]
document_chunks = _db["document_chunks"]
chat_sessions = _db["chat_sessions"]
feedback = _db["feedback"]
eval_logs = _db["eval_logs"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ping() -> bool:
    """Used by /health — never raises, just reports availability
    (NFR: Reliability — degrade gracefully if a dependency is down)."""
    try:
        _client.admin.command("ping")
        return True
    except Exception:
        return False


def status_label() -> str:
    if not ping():
        return "down"
    return "up (real MongoDB)" if USING_REAL_MONGODB else "up (in-memory mock, not persistent)"


def ensure_indexes() -> None:
    users.create_index("username", unique=True)
    document_chunks.create_index("document_id")
    document_chunks.create_index([("category", ASCENDING)])
    chat_sessions.create_index("user_id")
    feedback.create_index("chat_session_id")
    eval_logs.create_index("timestamp")
