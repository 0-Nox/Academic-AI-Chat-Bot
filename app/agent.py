"""
Agent Orchestrator (Section 18 — Agentic AI Workflow).

Per the SRS, this is deliberately a *single* router agent, not a full
multi-agent system: it performs (1) intent classification / routing and
(2) triggers the guardrails gate. A full Planner-Retriever-Validator-Safety
design is out of scope for the MVP (Section 22, Future Scope).

Also folds in the lightweight NLP steps from Section 16 that matter for
routing: keyword-based intent shortcut (cheap, deterministic) with an LLM
fallback for anything ambiguous (FR-2), plus trivial query normalization.
"""

import json
import re

from google import genai

from app import config

_client = genai.Client(api_key=config.GEMINI_API_KEY)

# Cheap deterministic first pass — most queries hit this and never need an
# LLM call just to be routed. Falls through to the LLM classifier below only
# when nothing matches.
_KEYWORD_MAP = {
    "attendance": ["attendance", "present", "absent", "bunk"],
    "timetable": ["timetable", "time table", "schedule", "class timing", "period"],
    "course_information": ["course", "syllabus", "credits", "elective", "curriculum"],
    "faculty_details": ["faculty", "professor", "teacher", "instructor", "hod"],
    "examination_schedule": ["exam", "midsem", "mid-sem", "endsem", "end-sem", "test date"],
    "academic_regulations": ["regulation", "policy", "rule", "grading", "backlog", "detention"],
    "notices": ["notice", "announcement", "circular", "deadline"],
    "assignments": ["assignment", "homework", "submission", "project deadline"],
    "general_erp_usage": ["erp", "lms", "login", "portal", "password reset"],
}


def normalize_query(query: str) -> str:
    """Trivial text preprocessing (Section 16): collapse whitespace only.
    Full normalization/stop-word handling is delegated to the embedding
    model, which is more robust than hand-rolled rules for short queries."""
    return re.sub(r"\s+", " ", query).strip()


def _keyword_classify(query: str) -> str | None:
    lowered = query.lower()
    for domain, keywords in _KEYWORD_MAP.items():
        if any(kw in lowered for kw in keywords):
            return domain
    return None


def _llm_classify(query: str) -> str:
    domains = config.SUPPORTED_DOMAINS + [config.OUT_OF_SCOPE_LABEL]
    prompt = f"""Classify the user's query into exactly one of these labels:
{json.dumps(domains)}

Rules:
- If the query is not about academic/institutional topics a university ERP
  or LMS would cover, return "{config.OUT_OF_SCOPE_LABEL}".
- Respond with ONLY the label, nothing else. No punctuation, no explanation.

Query: "{query}"
Label:"""
    response = _client.models.generate_content(model=config.GEN_MODEL, contents=prompt)
    label = response.text.strip().strip('"').lower()
    return label if label in domains else config.OUT_OF_SCOPE_LABEL


def classify_intent(query: str) -> str:
    """Routes a query to one of the nine domains or 'out_of_scope' (FR-2)."""
    query = normalize_query(query)
    keyword_hit = _keyword_classify(query)
    if keyword_hit:
        return keyword_hit
    return _llm_classify(query)
