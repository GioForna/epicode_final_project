"""Guardrail: mascheramento dati personali e controlli di input."""
import re

PII_PATTERNS = {
    "<EMAIL>": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "<IBAN>": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "<CARTA>": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "<TELEFONO>": re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d"),
    "<IP>": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

MIN_CHARS = 20
MAX_CHARS = 5000


def mask_pii(text: str) -> str:
    """Sostituisce email, IBAN, carte, telefoni e IP con segnaposto prima di chiamare l'API."""
    for placeholder, pattern in PII_PATTERNS.items():
        text = pattern.sub(placeholder, text)
    return text


def validate_ticket(text: str):
    """Restituisce (ok, messaggio). Rifiuta input vuoti o troppo lunghi."""
    text = (text or "").strip()
    if len(text) < MIN_CHARS:
        return False, f"Ticket troppo breve (minimo {MIN_CHARS} caratteri)."
    if len(text) > MAX_CHARS:
        return False, f"Ticket troppo lungo (massimo {MAX_CHARS} caratteri)."
    return True, ""
