"""Content key and content hash: pure leaf functions.

``content_key`` is the canonical comparison form (NFC, stripped, whitespace
runs collapsed, case kept). ``content_hash`` hashes the raw, unnormalised
text, so two raw texts sharing a key still differ by hash. The storage layer
derives both on every row write; callers never supply them.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WHITESPACE_RUN = re.compile(r"\s+")


def content_key(text: str) -> str:
    """Mechanical canonical form: NFC-normalise, strip, collapse whitespace
    runs to one space. Case is kept."""
    normalised = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RUN.sub(" ", normalised.strip())


def content_hash(text: str) -> str:
    """SHA-256 hex digest of the raw, unnormalised content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
