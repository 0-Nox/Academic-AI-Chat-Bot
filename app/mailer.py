"""
E-mail sending over SMTP (standard library only).

Credentials come from environment variables (see .env.example) and are never
logged. E-mail stays disabled until SMTP_HOST is configured. Callers decide
who may be written to; this module only delivers a message.
"""

import re
import smtplib
import ssl
from email.message import EmailMessage

from app import config


class MailError(Exception):
    """Delivery failed; the message is safe to show to the user."""


def is_configured() -> bool:
    return bool(config.SMTP_HOST and config.SMTP_FROM)


def clean_header(value: str, limit: int = 150) -> str:
    """Single-line header text: control characters (including CR/LF, which
    could be used to inject extra headers) are removed."""
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or "")).strip()[:limit]


def send_email(to_addrs, subject, body, attachments=()):
    """attachments: iterable of (filename, bytes, mime_type)."""
    if not is_configured():
        raise MailError("E-mail is not configured on this server.")
    msg = EmailMessage()
    msg["From"] = clean_header(config.SMTP_FROM, 254)
    msg["To"] = ", ".join(clean_header(a, 254) for a in to_addrs)
    msg["Subject"] = clean_header(subject, 150)
    msg.set_content(body)
    for filename, data, mime in attachments:
        maintype, _, subtype = mime.partition("/")
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=clean_header(filename, 100))
    try:
        ctx = ssl.create_default_context()
        if config.SMTP_SECURITY == "ssl":
            server = smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT, timeout=20, context=ctx)
        else:
            server = smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=20)
        with server:
            if config.SMTP_SECURITY == "starttls":
                server.starttls(context=ctx)
            if config.SMTP_USER:
                server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.send_message(msg)
    except (smtplib.SMTPException, OSError) as e:
        print(f"E-mail delivery failed: {type(e).__name__}")  # class only: never log credentials
        raise MailError("The e-mail could not be sent through the configured mail server.") from e