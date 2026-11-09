"""
PDF generation for summaries and conversation transcripts (reportlab).

Every PDF is built on the server from stored data, never from text sent by
the browser, so a PDF that is e-mailed to staff really is what the system
produced. Text is cleaned for the PDF's built-in Helvetica font, which only
covers Western European characters: a few common symbols are replaced by
ASCII equivalents and anything else unsupported becomes "?".
"""

import re
from datetime import datetime, timezone
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer

NAVY = colors.HexColor("#1b3a5c")
GREY = colors.HexColor("#666666")
DISCLAIMER = ("This document was generated automatically by the Campus Assistant. It may contain "
              "errors; please verify important details against the original institutional documents.")

_REPLACE = {"≥": ">=", "≤": "<=", "→": "->", "←": "<-", "…": "...", "\u00a0": " ", "−": "-",
            "✓": "v", "×": "x", "\u200b": ""}


def _safe(text):
    text = str(text or "")
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.encode("cp1252", errors="replace").decode("cp1252")


def _markup(line):
    """Escape text for reportlab and turn **bold** into <b>bold</b>."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(_safe(line)))


def _styles():
    base = dict(fontName="Helvetica", fontSize=10.5, leading=15, textColor=colors.black)
    return {
        "title": ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=NAVY, spaceAfter=6),
        "meta": ParagraphStyle("m", fontName="Helvetica", fontSize=9, leading=13, textColor=GREY),
        "h": ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=NAVY, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("b", alignment=TA_JUSTIFY, spaceAfter=4, **base),
        "bullet": ParagraphStyle("bl", leftIndent=16, bulletIndent=4, spaceAfter=2, **base),
        "who_user": ParagraphStyle("wu", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=colors.HexColor("#b3261e"), spaceBefore=8),
        "who_bot": ParagraphStyle("wb", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=NAVY, spaceBefore=8),
        "user": ParagraphStyle("u", backColor=colors.HexColor("#f2f2f2"), borderPadding=(4, 5, 4, 5), spaceBefore=3, spaceAfter=6, **base),
        "src": ParagraphStyle("s", fontName="Helvetica-Oblique", fontSize=8.5, leading=11, textColor=GREY, spaceAfter=4),
        "disc": ParagraphStyle("d", fontName="Helvetica-Oblique", fontSize=8.5, leading=11, textColor=GREY, spaceBefore=10),
    }


def _text_flowables(text, st, body="body"):
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            out.append(Paragraph(_markup(line.lstrip("# ")), st["h"]))
        elif line[:2] in ("- ", "* ") or line.startswith("• "):
            out.append(Paragraph(_markup(line[2:]), st["bullet"], bulletText="•"))
        else:
            out.append(Paragraph(_markup(line), st[body]))
    return out


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cccccc"))
    canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.0 * cm, "Campus Assistant")
    canvas.drawRightString(A4[0] - 2 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _header(title, meta, st):
    items = [Paragraph(_markup(title), st["title"])]
    for label, value in meta:
        items.append(Paragraph(f"<b>{escape(_safe(label))}:</b> {escape(_safe(value))}", st["meta"]))
    items += [Spacer(1, 6), HRFlowable(width="100%", thickness=1, color=NAVY), Spacer(1, 8)]
    return items


def _render(story, title):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2.2 * cm,
                            title=_safe(title), author="Campus Assistant")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def summary_pdf(title, summary_text, meta):
    """meta: list of (label, value) pairs shown under the title."""
    st = _styles()
    story = _header(title, [("Generated", _now())] + list(meta), st)
    story += _text_flowables(summary_text, st)
    story.append(Paragraph(escape(DISCLAIMER), st["disc"]))
    return _render(story, title)


def transcript_pdf(title, messages, meta):
    st = _styles()
    story = _header(title, [("Generated", _now())] + list(meta), st)
    for m in messages:
        is_user = m.get("role") == "user"
        ts = m.get("ts")
        stamp = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else ""
        block = [Paragraph(("YOU" if is_user else "ASSISTANT") + (f"  ·  {stamp}" if stamp else ""), st["who_user" if is_user else "who_bot"])]
        if is_user:
            block.append(Paragraph(_markup(m.get("text", "")), st["user"]))
        else:
            block += _text_flowables(m.get("text", ""), st)
            if m.get("citations"):
                block.append(Paragraph(escape(_safe("Sources: " + ", ".join(m["citations"]))), st["src"]))
        story.append(KeepTogether(block[:2]))
        story += block[2:]
    story.append(Paragraph(escape(DISCLAIMER), st["disc"]))
    return _render(story, title)