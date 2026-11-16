"""Discriminative guards: pure string comparison, no I/O."""

from __future__ import annotations

import re
from typing import Literal

GuardName = Literal["numeral", "version", "entity", "negation"]

_RE_NUMERALS = re.compile(r"\b\d+(?:\.\d+)?\b")
_RE_VERSIONS = re.compile(r"\bv?\d+(?:\.\d+){1,3}\b", re.IGNORECASE)
_RE_CAPS = re.compile(r"\b[A-Z][a-zA-Z0-9]+\b")
_NEGATIONS = frozenset(
    ["not", "no", "never", "without", "neither", "nor", "cannot", "can't", "don't",
     "won't", "doesn't", "isn't", "aren't", "wasn't", "weren't", "shouldn't",
     "wouldn't", "couldn't", "needn't", "mustn't", "hardly", "barely", "scarcely"]
)


def guard_check(content_a: str, content_b: str) -> GuardName | None:
    """Return the name of the first guard that trips, or None.

    Checked in order: ``numeral``, ``version``, ``entity`` (capitalized tokens), ``negation``.
    A guard trips when the two texts yield different sets for it.
    """
    a_lower = content_a.lower()
    b_lower = content_b.lower()

    nums_a = set(_RE_NUMERALS.findall(content_a))
    nums_b = set(_RE_NUMERALS.findall(content_b))
    if nums_a != nums_b:
        return "numeral"

    vers_a = set(_RE_VERSIONS.findall(content_a))
    vers_b = set(_RE_VERSIONS.findall(content_b))
    if vers_a != vers_b:
        return "version"

    ents_a = set(_RE_CAPS.findall(content_a))
    ents_b = set(_RE_CAPS.findall(content_b))
    if ents_a != ents_b:
        return "entity"

    words_a = frozenset(re.findall(r"\b\w+\b", a_lower))
    words_b = frozenset(re.findall(r"\b\w+\b", b_lower))
    negs_a = words_a & _NEGATIONS
    negs_b = words_b & _NEGATIONS
    if negs_a != negs_b:
        return "negation"

    return None
