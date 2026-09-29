"""
Central configuration. Every environment-dependent value lives here so the
rest of the codebase never calls os.environ directly (Security Requirements,
Section 19: secrets via env vars / secrets manager, never hard-coded).
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} as an environment variable first.")
    return value


# --- LLM / embeddings (Gemini) ---
GEMINI_API_KEY = _require("GEMINI_API_KEY")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
GEN_MODEL = os.environ.get("GEN_MODEL", "gemini-3.6-flash")

# --- MongoDB (Section 14.1 — Data Architecture) ---
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "academic_chatbot")

# --- Auth (Section 19 — Security Requirements: JWT-based auth) ---
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-only-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "120"))

# --- RAG pipeline tuning ---
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "80"))
TOP_K = int(os.environ.get("TOP_K", "3"))

# --- Data source for ingestion (FR-10) ---
DATA_DIR = os.environ.get("DATA_DIR", "data")

# Nine supported domains (Scope, Section 3.1) plus the guardrail catch-all.
SUPPORTED_DOMAINS = [
    "attendance",
    "timetable",
    "course_information",
    "faculty_details",
    "examination_schedule",
    "academic_regulations",
    "notices",
    "assignments",
    "general_erp_usage",
]
OUT_OF_SCOPE_LABEL = "out_of_scope"
