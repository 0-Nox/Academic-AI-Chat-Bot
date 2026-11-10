"""
Summaries, PDF export and e-mail to staff (one APIRouter, mounted in main.py).

Endpoint                 Access            Purpose
GET    /documents        signed-in user    ingested documents that can be summarised
POST   /summarize        signed-in user    summarise a document or one's own conversation
POST   /export/pdf       signed-in user    PDF of a summary or a full transcript
GET    /staff            signed-in user    staff who can receive e-mail (no addresses)
POST   /staff            admin             add a staff contact
DELETE /staff/{id}       admin             remove a staff contact
POST   /email            signed-in user    e-mail that PDF to chosen staff

Design rules that keep e-mail safe:
  * recipients are chosen from the staff directory by id, never typed, so the
    chatbot cannot be used to send mail to arbitrary addresses;
  * the PDF is built on the server from stored data, so what staff receive is
    what the system produced and not text supplied by the sender;
  * every user is limited to EMAIL_MAX_PER_HOUR messages and each send is
    written to the email_log collection.
"""

import json
import re
import uuid
from datetime import timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from app import config, db, guardrails, mailer, pdf_export, security, summarize

router = APIRouter()
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# --------------------------------------------------------------------------- models
class SummarizeRequest(BaseModel):
    source: str | None = None          # a document (its "source" name) ...
    session_id: str | None = None      # ... or one of the user's conversations
    length: Literal["short", "medium", "detailed"] = "medium"
    page_from: int | None = Field(None, ge=1)
    page_to: int | None = Field(None, ge=1)


class ExportRequest(BaseModel):
    kind: Literal["summary", "transcript"]
    summary_id: str | None = None      # required for kind = summary
    session_id: str | None = None      # required for kind = transcript


class EmailRequest(ExportRequest):
    recipient_ids: list[str] = Field(min_length=1)
    message: str | None = Field(None, max_length=1000)


class StaffIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    designation: str = Field("", max_length=100)
    department: str = Field("", max_length=100)
    email: str = Field(max_length=254)


# --------------------------------------------------------------------------- helpers
def _session_for(session_id, user):
    s = db.chat_sessions.find_one({"_id": session_id})
    if not s:
        raise HTTPException(status_code=404, detail="Session not found.")
    if user.role != "admin" and s.get("user_id") != user.username:
        raise HTTPException(status_code=403, detail="Not your session.")
    return s


def _summary_for(summary_id, user):
    s = db.summaries.find_one({"_id": summary_id})
    if not s:
        raise HTTPException(status_code=404, detail="Summary not found.")
    if user.role != "admin" and s.get("owner") != user.username:
        raise HTTPException(status_code=403, detail="Not your summary.")
    return s


def _slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:60] or "document"


def _build_pdf(req, user):
    """Returns (filename, pdf_bytes, title). Always built from stored data."""
    who = ("Prepared for", f"{user.username} ({user.role})")
    if req.kind == "summary":
        if not req.summary_id:
            raise HTTPException(status_code=400, detail="summary_id is required for a summary PDF.")
        s = _summary_for(req.summary_id, user)
        meta = [who, ("Document" if s["target_type"] == "document" else "Conversation", s["target"] if s["target_type"] == "document" else s["target"][:8])]
        if s.get("pages") and s["pages"][0] is not None:
            meta.append(("Pages", f"{s['pages'][0]}-{s['pages'][1]}"))
        meta.append(("Length", s["length"]))
        return _slug(s["title"]) + ".pdf", pdf_export.summary_pdf(s["title"], s["text"], meta), s["title"]
    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id is required for a transcript PDF.")
    session = _session_for(req.session_id, user)
    messages = session.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="This conversation has no messages yet.")
    title = "Conversation transcript"
    meta = [who, ("Conversation", req.session_id[:8]), ("Messages", str(len(messages)))]
    return "conversation_transcript.pdf", pdf_export.transcript_pdf(title, messages, meta), title


def _ensure_staff_seeded():
    """If the staff directory is empty, seed it from the STAFF_CONTACTS
    environment variable (a JSON list). Invalid entries are skipped."""
    if db.staff_contacts.count_documents({}) > 0 or not config.STAFF_CONTACTS:
        return
    try:
        entries = json.loads(config.STAFF_CONTACTS)
    except json.JSONDecodeError:
        print("STAFF_CONTACTS is not valid JSON; staff directory not seeded.")
        return
    for e in entries if isinstance(entries, list) else []:
        email = str(e.get("email", "")).strip().lower()
        if str(e.get("name", "")).strip() and _EMAIL_RE.match(email) and not db.staff_contacts.find_one({"email": email}):
            db.staff_contacts.insert_one({"_id": str(uuid.uuid4()), "name": str(e["name"]).strip()[:100],
                                          "designation": str(e.get("designation", ""))[:100],
                                          "department": str(e.get("department", ""))[:100], "email": email})


def _public(contact, admin):
    out = {"id": contact["_id"], "name": contact["name"], "designation": contact.get("designation", ""),
           "department": contact.get("department", "")}
    if admin:
        out["email"] = contact["email"]
    return out


# --------------------------------------------------------------------------- documents and summaries
@router.get("/documents")
def documents(user: security.TokenData = Depends(security.get_current_user)):
    return {"documents": summarize.list_documents()}


@router.post("/summarize")
def summarize_endpoint(req: SummarizeRequest, user: security.TokenData = Depends(security.get_current_user)):
    if bool(req.source) == bool(req.session_id):
        raise HTTPException(status_code=400, detail="Provide either 'source' or 'session_id'.")
    if req.page_from and req.page_to and req.page_from > req.page_to:
        raise HTTPException(status_code=400, detail="page_from must not be greater than page_to.")
    try:
        if req.source:
            doc = summarize.find_document(req.source)
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found.")
            result = summarize.summarize_document(doc, req.length, req.page_from, req.page_to)
            target_type, target, title = "document", doc["source"], f"Summary of {doc['source']}"
        else:
            session = _session_for(req.session_id, user)
            result = summarize.summarize_session(session.get("messages", []), req.length)
            target_type, target, title = "session", req.session_id, "Conversation summary"
    except summarize.SummaryError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    text = guardrails.check_output(result["text"])   # same PII redaction as chat answers
    summary_id = str(uuid.uuid4())
    pages = list(result["pages"]) if result.get("pages") else None
    db.summaries.insert_one({"_id": summary_id, "owner": user.username, "target_type": target_type, "target": target,
                             "title": title, "length": req.length, "pages": pages, "parts": result.get("parts", 1),
                             "text": text, "created_at": db.utcnow()})
    return {"summary_id": summary_id, "title": title, "summary": text, "target_type": target_type, "target": target,
            "length": req.length, "pages": pages, "parts": result.get("parts", 1)}


# --------------------------------------------------------------------------- PDF
@router.post("/export/pdf")
def export_pdf(req: ExportRequest, user: security.TokenData = Depends(security.get_current_user)):
    filename, data, _ = _build_pdf(req, user)
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# --------------------------------------------------------------------------- staff directory
@router.get("/staff")
def list_staff(user: security.TokenData = Depends(security.get_current_user)):
    _ensure_staff_seeded()
    admin = user.role == "admin"
    return {"staff": [_public(c, admin) for c in db.staff_contacts.find({}).sort("name", 1)]}


@router.post("/staff", status_code=201)
def add_staff(body: StaffIn, user: security.TokenData = Depends(security.require_role("admin"))):
    email = body.email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="Enter a valid e-mail address.")
    if db.staff_contacts.find_one({"email": email}):
        raise HTTPException(status_code=409, detail="A contact with this e-mail address already exists.")
    doc = {"_id": str(uuid.uuid4()), "name": body.name.strip(), "designation": body.designation.strip(),
           "department": body.department.strip(), "email": email}
    db.staff_contacts.insert_one(doc)
    return _public(doc, True)


@router.delete("/staff/{staff_id}")
def delete_staff(staff_id: str, user: security.TokenData = Depends(security.require_role("admin"))):
    if db.staff_contacts.delete_one({"_id": staff_id}).deleted_count == 0:
        raise HTTPException(status_code=404, detail="Contact not found.")
    return {"status": "deleted"}


# --------------------------------------------------------------------------- e-mail
@router.post("/email")
def email_pdf(req: EmailRequest, user: security.TokenData = Depends(security.get_current_user)):
    if not mailer.is_configured():
        raise HTTPException(status_code=503, detail="E-mail is not configured on this server.")
    _ensure_staff_seeded()
    ids = list(dict.fromkeys(req.recipient_ids))
    if len(ids) > config.EMAIL_MAX_RECIPIENTS:
        raise HTTPException(status_code=400, detail=f"At most {config.EMAIL_MAX_RECIPIENTS} recipients are allowed.")
    contacts = list(db.staff_contacts.find({"_id": {"$in": ids}}))
    if len(contacts) != len(ids):
        raise HTTPException(status_code=400, detail="One or more recipients are not in the staff directory.")
    since = db.utcnow() - timedelta(hours=1)
    sent_recently = db.email_log.count_documents({"user": user.username, "status": "sent", "timestamp": {"$gte": since}})
    if sent_recently >= config.EMAIL_MAX_PER_HOUR:
        raise HTTPException(status_code=429, detail="E-mail limit reached. Please try again later.")
    filename, pdf, title = _build_pdf(req, user)       # validates ownership too
    note = f"Message from the sender:\n{req.message.strip()}\n\n" if req.message and req.message.strip() else ""
    body = (f"{user.username} ({user.role}) shared a document with you through the Campus Assistant.\n\n{note}"
            f"Attached: {filename}\n\n{pdf_export.DISCLAIMER}\n")
    log = {"user": user.username, "role": user.role, "kind": req.kind, "recipients": [c["_id"] for c in contacts],
           "recipient_names": [c["name"] for c in contacts], "file": filename, "timestamp": db.utcnow()}
    try:
        mailer.send_email([c["email"] for c in contacts], f"Campus Assistant: {title}", body, [(filename, pdf, "application/pdf")])
    except mailer.MailError as e:
        db.email_log.insert_one({**log, "status": "failed"})
        raise HTTPException(status_code=502, detail=str(e))
    db.email_log.insert_one({**log, "status": "sent"})
    return {"status": "sent", "sent_to": [c["name"] for c in contacts], "file": filename}