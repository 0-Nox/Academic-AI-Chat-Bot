# Academic AI Chatbot

A RAG-based conversational assistant for academic ERP/LMS systems. See the
SRS for full requirements — this README covers running the MVP.

## Architecture

```
static/index.html  ──HTTP──►  app/main.py (FastAPI)
                                   │
                    ┌──────────────┼───────────────┐
                    ▼              ▼               ▼
             app/security.py  app/guardrails.py  app/agent.py
             (JWT auth)       (injection/PII)    (intent routing)
                    │                                │
                    ▼                                ▼
               app/db.py (MongoDB)            app/rag.py (retrieval +
                                                Gemini generation)
```

- **app/main.py** — HTTP endpoints only.
- **app/rag.py** — chunking, embeddings, retrieval, grounded generation.
- **app/agent.py** — single-router agent: intent classification (FR-2).
- **app/guardrails.py** — prompt-injection/harmful-input rejection, PII redaction.
- **app/security.py** — JWT auth, password hashing, role checks.
- **app/db.py** — MongoDB collections (Section 14.1 schema).

## Setup

1. **Python deps**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **MongoDB (optional)** — real MongoDB gives you persistence (chat
   history, feedback, logs survive a restart). If nothing is running at
   `MONGODB_URI`, the app automatically falls back to an in-memory database
   for that run — no Docker, no local install needed to just try it out.
   To get persistence later: `docker run -d -p 27017:27017 --name
   academic-mongo mongo:7`, or point `MONGODB_URI` at a free MongoDB Atlas
   cluster.

3. **Environment variables** — copy `.env.example` to `.env` and fill in
   your `GEMINI_API_KEY` (get one from Google AI Studio). Defaults work for
   a local MongoDB.

4. **Sample data** — a small synthetic dataset lives in `sample_data/` for
   local testing. Point the app at it (or drop your own `.txt` files into a
   `data/` folder — created automatically, and already gitignored):
   ```bash
   mkdir -p data && cp sample_data/*.txt data/
   ```

5. **Run**
   ```bash
   uvicorn app.main:app --reload
   ```
   Open http://localhost:8000 — sign in with one of the seeded demo
   accounts (see below).

## Demo accounts (seeded automatically on first run)

| Username  | Password    | Role    |
|-----------|-------------|---------|
| student1  | student123  | student |
| faculty1  | faculty123  | faculty |
| admin1    | admin123    | admin   |

Change or remove these before any real deployment.

## API summary

| Endpoint            | Auth      | Purpose                                    |
|----------------------|-----------|---------------------------------------------|
| `POST /auth/login`   | none      | Returns a JWT.                              |
| `GET /health`        | none      | Liveness + index/DB status.                 |
| `POST /chat`         | any user  | Ask a question; returns grounded answer.    |
| `POST /feedback`     | any user  | Thumbs up/down + optional comment.          |
| `POST /ingest`       | admin     | Upload a new `.txt` document, re-index.     |
| `GET /logs`          | admin     | Recent query/response logs for evaluation.  |

## What's implemented vs. the SRS

Implemented: FR-1, FR-2, FR-3, FR-5, FR-6, FR-7, FR-8, FR-9, FR-10, FR-11,
FR-12, and the Section 19 security requirements (JWT auth, no plaintext
secrets, RBAC on ingestion/logs).

Not yet implemented (see Section 22, Future Scope, and open items):
- Docker / Docker Compose packaging, Kubernetes manifests, cloud deployment
  (Section 3.1's containerization/orchestration bullets).
- A real vector index (FAISS / MongoDB Atlas Vector Search) — current
  retrieval is in-memory cosine similarity, fine at MVP data scale but not
  the production-shaped implementation the SRS names.
- Formal evaluation harness against the Section 21 metric targets — `/logs`
  gives you the raw data to compute these from, but nothing computes
  faithfulness/hallucination-rate automatically yet.
- NER-based query rewriting (Section 16) beyond basic whitespace normalization.
