# domainrag/cleaning.py

import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters.

    Example:
    visually similar unicode forms become consistent.
    """
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    """
    Replace repeated spaces/newlines/tabs with a single space.
    """
    return re.sub(r"\s+", " ", text).strip()


def remove_control_chars(text: str) -> str:
    """
    Remove non-printable control characters.
    """
    return "".join(ch for ch in text if ch.isprintable() or ch.isspace())


def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline.
    """
    if text is None:
        return ""

    text = str(text)
    text = normalize_unicode(text)
    text = remove_control_chars(text)
    text = normalize_whitespace(text)

    return text


def is_good_text(
    text: str,
    min_chars: int = 100,
    max_chars: int = 20_000,
) -> bool:
    """
    Filter useless or pathological documents.
    """
    if not text:
        return False

    n = len(text)

    if n < min_chars:
        return False

    if n > max_chars:
        return False

    return True