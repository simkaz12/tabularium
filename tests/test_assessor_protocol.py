"""Tests for ``engine.assessor`` — the six-method Protocol and its envelope validators.

Falsifiability (brief Scope 1): the reason-class validator was removed, both
``test_replacement_requires_a_reason_class`` and ``test_non_replacement_rejects_a_reason_class``
went red for it, then it was restored. Command run: ``uv run pytest -q
tests/test_assessor_protocol.py``, both before and after — reported in the runner verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tabularium.engine.assessor import (
    Assessor,
    AssessorCandidate,
    ContradictionVerdict,
    PanelDisagreement,
    ReasonClass,
    Relation,
    RelationVerdict,
    Trace,
    Verdict,
)
from tabularium.engine.rubric import load_rubric

_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)


def _trace() -> Trace:
    return Trace(
        model="test", wording_hash="0" * 16, latency_ms=1,
        input_tokens=None, output_tokens=None, request_id=None,
    )


def _candidate(**overrides: object) -> AssessorCandidate:
    defaults: dict[str, object] = {
        "id": "c1",
        "content": "Existing content.",
        "kind": "fact",
        "scope": "global",
        "created_at": _NOW,
    }
    defaults.update(overrides)
    return AssessorCandidate(**defaults)  # type: ignore[arg-type]


def _relation_verdict(
    pick: Relation, reason_class: ReasonClass | None = None
) -> RelationVerdict:
    return RelationVerdict(
        method="relation", pick=pick, odds={pick: 1.0}, confidence=1.0,
        trace=_trace(), reason_class=reason_class,
    )


def test_replacement_requires_a_reason_class() -> None:
    with pytest.raises(ValidationError):
        _relation_verdict("replacement")


def test_non_replacement_rejects_a_reason_class() -> None:
    with pytest.raises(ValidationError):
        _relation_verdict("distinct", "changed")


@pytest.mark.parametrize(
    ("pick", "reason_class"),
    [
        ("restatement", None),
        ("distinct", None),
        ("follow_up", None),
        ("replacement", "correction"),
    ],
)
def test_valid_shapes_construct(pick: Relation, reason_class: ReasonClass | None) -> None:
    verdict = _relation_verdict(pick, reason_class)
    assert verdict.pick == pick
    assert verdict.reason_class == reason_class


def test_assessor_candidate_constructs() -> None:
    candidate = _candidate()
    assert candidate.id == "c1"
    assert candidate.kind == "fact"


class _EightMethodAssessor:
    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        return _relation_verdict("distinct")

    async def kind(self, text: str, declared_kind: str) -> Verdict:
        return Verdict(
            method="kind", pick="fact", odds={"fact": 1.0}, confidence=1.0, trace=_trace()
        )

    async def worth(
        self, text: str, *, neighbours: Sequence[AssessorCandidate] | None = None
    ) -> Verdict:
        return Verdict(
            method="worth", pick="keep", odds={"keep": 1.0}, confidence=1.0, trace=_trace()
        )

    async def perishable(self, text: str) -> Verdict:
        return Verdict(
            method="perishable", pick="standing", odds={"standing": 1.0}, confidence=1.0,
            trace=_trace(),
        )

    async def breadth(self, text: str, source_scope: str) -> Verdict:
        return Verdict(
            method="breadth", pick="owner_wide", odds={"owner_wide": 1.0}, confidence=1.0,
            trace=_trace(),
        )

    async def scope(self, text: str, labels: Sequence[tuple[str, str | None]]) -> Verdict:
        return Verdict(
            method="scope", pick="global", odds={"global": 1.0}, confidence=1.0, trace=_trace()
        )

    async def relevance(self, text: str) -> Verdict:
        return Verdict(
            method="relevance", pick="keep", odds={"keep": 1.0}, confidence=1.0, trace=_trace()
        )

    async def contradiction(
        self, existing: str, incoming: str, *, panel: PanelDisagreement | None = None
    ) -> ContradictionVerdict:
        return ContradictionVerdict(
            pick="coexist", reason_class=None,
            odds={"coexist": 1.0, "correction": 0.0, "changed": 0.0}, confidence=1.0,
            trace=_trace(),
        )


def _accepts_assessor(assessor: Assessor) -> Assessor:
    return assessor


async def test_a_minimal_class_satisfies_the_assessor_protocol() -> None:
    assessor = _accepts_assessor(_EightMethodAssessor())
    verdict = await assessor.relation("incoming text", _candidate(), similarity=None)
    assert verdict.pick == "distinct"


def test_caseset_prompt_template_carries_both_placeholders() -> None:
    rubric = load_rubric()
    assert rubric.caseset is not None, "packaged rubric artifact is missing [caseset]"
    template = rubric.caseset.prompt_template
    assert "{existing}" in template
    assert "{incoming}" in template
