"""
Guardrails Layer (Section 20). Runs before generation (input validation) and
after generation (output validation), matching the sequence diagram in
Figure 17.2 (steps 2-3 and 12-13).

This is intentionally lightweight pattern/heuristic matching, not a trained
classifier — appropriate for the MVP scope (Section 12: limited compute
budget) and documented as a place to swap in a real moderation model later
(Section 22, Future Scope).
"""

import re
from dataclasses import dataclass

# --- Prompt injection / jailbreak patterns -----------------------------
_INJECTION_PATTERNS = [
    r"ignore (all|any|the|previous|prior)? ?(previous |prior )?instructions",
    r"disregard (all|any|the|previous|prior)? ?(previous |prior )?instructions",
    r"you are now",
    r"act as (a|an) (?!student|faculty|admin)",
    r"pretend (you are|to be)",
    r"system prompt",
    r"reveal (your|the) (prompt|instructions|system message)",
    r"jailbreak",
    r"dan mode",
    r"do anything now",
    r"bypass (your|the) (rules|guardrails|restrictions|filters)",
    r"forget (everything|all previous)",
    r"new instructions?:",
    r"</?(system|assistant|instructions)>",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# --- Harmful / toxic content (basic keyword heuristic) -------------------
_HARMFUL_PATTERNS = [
    r"\bhow to (make|build|synthesize) (a )?(bomb|explosive|weapon)",
    r"\bkill (myself|someone|yourself)",
    r"\bself[- ]harm\b",
    r"\bhack (into|the) .*(account|server|database|network)\b",
]
_HARMFUL_RE = re.compile("|".join(_HARMFUL_PATTERNS), re.IGNORECASE)

# --- PII patterns (redacted on the way out, Section 20: sensitive data leakage) --
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b")
_STUDENT_ID_RE = re.compile(r"\b\d{2}[A-Z]{3,6}\d{5,8}\b")  # e.g. 24STUCHH010784


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    safe_message: str | None = None


FALLBACK_INJECTION = (
    "I can't follow instructions embedded in a question like that. "
    "Please ask your academic question directly and I'll help."
)
FALLBACK_HARMFUL = (
    "I'm not able to help with that request. If this is something urgent "
    "or safety-related, please contact your institution's support services."
)
FALLBACK_OUT_OF_SCOPE = (
    "That's outside what I can help with — I only answer questions about "
    "attendance, timetables, courses, faculty, exams, regulations, notices, "
    "and assignments. Try rephrasing your question around one of those topics."
)


def check_input(query: str) -> GuardrailResult:
    """Runs before the query reaches the retriever/LLM."""
    if _INJECTION_RE.search(query):
        return GuardrailResult(False, "prompt_injection", FALLBACK_INJECTION)
    if _HARMFUL_RE.search(query):
        return GuardrailResult(False, "harmful_content", FALLBACK_HARMFUL)
    return GuardrailResult(True)


def redact_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _PHONE_RE.sub("[redacted-phone]", text)
    text = _STUDENT_ID_RE.sub("[redacted-id]", text)
    return text


def check_output(answer: str) -> str:
    """Runs on the generated answer before it's returned to the user.
    Returns the (possibly redacted) safe text."""
    return redact_pii(answer)
