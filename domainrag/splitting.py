# domainrag/splitting.py

import hashlib


def stable_hash_fraction(text: str) -> float:
    """
    Convert a string into a deterministic number in [0, 1).

    We use this for reproducible train/dev/test splitting.
    """
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    value = int(h, 16)
    return value / 2**128


def assign_split(
    doc_id: str,
    train_frac: float = 0.8,
    dev_frac: float = 0.1,
) -> str:
    """
    Deterministically assign a document to train/dev/test.

    Important:
    We split by doc_id, not chunk_id.
    """
    x = stable_hash_fraction(doc_id)

    if x < train_frac:
        return "train"

    if x < train_frac + dev_frac:
        return "dev"

    return "test"