"""``EchoAssessor`` — the default lane's fake Assessor.

Satisfies ``engine.assessor.Assessor`` in-process, no key, no network: ``relation`` echoes a
case's own material label, mapped down to a relation and reason class,
regardless of what it is actually asked to assess. ``relevance``/``contradiction`` (A3) default
to a fixed ``keep-clear``/``coexist`` pick, or the per-call scripted answer named by
``relevance_script``/``contradiction_script`` (keyed by the call's own record text) — this is
what lets a test or the admission runner's falsity axis vary answers per record. ``kind``/
``worth``/``scope`` return a fixed pick with full odds — their wording lands with their own
caller piece (A5/A6).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from tabularium.engine.assessor import (
    CONTRADICTION_QUESTION,
    RELATION_QUESTION,
    RELEVANCE_QUESTION,
    AssessorCandidate,
    ContradictionVerdict,
    Method,
    PanelDisagreement,
    RelationVerdict,
    Trace,
    Verdict,
    wording_hash,
)
from tests.support.cases import MATERIAL_TO_REASON, MATERIAL_TO_RELATION, MaterialLabel

#: Scripted per-call answers, keyed by the record text the case's own scoring already keys on —
#: ``relevance`` by the record's own text, ``contradiction`` by the pair's OLDER (``existing``)
#: text, matching the admission runner's own "score per case by the older record's id" convention
#: (D8). A mapped value that is a ``BaseException`` instance is raised rather than
#: returned — the fake's own stand-in for a scripted ``timeout``/``error`` degrade, since the
#: ``Assessor`` Protocol's methods raise on failure and never return ``Unavailable`` themselves
#: (``engine.admission.ask`` is what maps a raise to ``Unavailable``).
RelevanceScript = Mapping[str, Verdict | BaseException]
ContradictionScript = Mapping[str, ContradictionVerdict | BaseException]


def _fixed_verdict(method: Method, pick: str) -> Verdict:
    return Verdict(
        method=method,
        pick=pick,
        odds={pick: 1.0},
        confidence=1.0,
        trace=Trace(
            model="echo",
            wording_hash="echo",
            latency_ms=0,
            input_tokens=None,
            output_tokens=None,
            request_id=None,
        ),
    )


def _fixed_relevance_verdict(pick: str) -> Verdict:
    return Verdict(
        method="relevance",
        pick=pick,
        odds={pick: 1.0},
        confidence=1.0,
        trace=Trace(
            model="echo",
            wording_hash=wording_hash(RELEVANCE_QUESTION),
            latency_ms=0,
            input_tokens=None,
            output_tokens=None,
            request_id=None,
        ),
    )


def _fixed_contradiction_verdict(
    raw_pick: Literal["coexist", "correction", "changed"],
) -> ContradictionVerdict:
    reason_class: Literal["correction", "changed"] | None = (
        raw_pick if raw_pick in ("correction", "changed") else None
    )
    domain_pick: Literal["coexist", "replace"] = "coexist" if raw_pick == "coexist" else "replace"
    return ContradictionVerdict(
        pick=domain_pick,
        reason_class=reason_class,
        odds={raw_pick: 1.0},
        confidence=1.0,
        trace=Trace(
            model="echo",
            wording_hash=wording_hash(CONTRADICTION_QUESTION),
            latency_ms=0,
            input_tokens=None,
            output_tokens=None,
            request_id=None,
        ),
    )


class EchoAssessor:
    """Fixed-verdict fake: ``relation`` always returns the relation/reason ``label`` maps to,
    with a real relation wording hash. ``relevance``/``contradiction`` (A3) return a fixed
    ``keep-clear``/``coexist`` pick by default, or the scripted per-call answer named by
    ``relevance_script``/``contradiction_script`` when the call's own text is a key there — this
    is what lets a test or the admission runner vary answers per record rather than share one
    fixed verdict across an entire pass. ``kind``/``worth``/``scope`` return a fixed pick."""

    def __init__(
        self,
        label: MaterialLabel,
        *,
        relevance_script: RelevanceScript | None = None,
        contradiction_script: ContradictionScript | None = None,
        breadth_pick: str = "project_specific",
    ) -> None:
        self._label = label
        self._breadth_pick = breadth_pick
        self._relevance_script: RelevanceScript = dict(relevance_script or {})
        self._contradiction_script: ContradictionScript = dict(contradiction_script or {})
        self.calls: list[tuple[str, str, list[str]]] = []
        #: Every ``relevance`` call's own text, in order — finer-grained than ``calls`` for a
        #: test asserting a call count or a call's own input without the tuple padding.
        self.relevance_calls: list[str] = []
        #: Every ``contradiction`` call's ``(existing, incoming, panel)``, in order — carries the
        #: ``panel`` argument ``calls`` above has no slot for, which a falsity-split test needs to
        #: assert the call really carried both judges' votes and ``why``s (review stage).
        self.contradiction_calls: list[tuple[str, str, PanelDisagreement | None]] = []

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        self.calls.append(("relation", incoming, [candidate.id]))
        pick = MATERIAL_TO_RELATION[self._label]
        reason_class = MATERIAL_TO_REASON[self._label]
        reason_odds: dict[str, float] | None = (
            {reason_class: 1.0} if reason_class is not None else None
        )
        return RelationVerdict(
            method="relation",
            pick=pick,
            odds={pick: 1.0},
            confidence=1.0,
            trace=Trace(
                model="echo",
                wording_hash=wording_hash(RELATION_QUESTION),
                latency_ms=0,
                input_tokens=None,
                output_tokens=None,
                request_id=None,
            ),
            reason_class=reason_class,
            reason_odds=reason_odds,
        )

    async def kind(self, text: str, declared_kind: str) -> Verdict:
        self.calls.append(("kind", text, []))
        return _fixed_verdict("kind", declared_kind)

    async def worth(
        self, text: str, *, neighbours: Sequence[AssessorCandidate] | None = None
    ) -> Verdict:
        self.calls.append(("worth", text, []))
        return _fixed_verdict("worth", "keep")

    async def perishable(self, text: str) -> Verdict:
        self.calls.append(("perishable", text, []))
        return _fixed_verdict("perishable", "standing")

    async def breadth(self, text: str, source_scope: str) -> Verdict:
        self.calls.append(("breadth", text, []))
        return _fixed_verdict("breadth", self._breadth_pick)

    async def scope(self, text: str, labels: Sequence[tuple[str, str | None]]) -> Verdict:
        self.calls.append(("scope", text, []))
        return _fixed_verdict("scope", "global")

    async def relevance(self, text: str) -> Verdict:
        self.calls.append(("relevance", text, []))
        self.relevance_calls.append(text)
        scripted = self._relevance_script.get(text)
        if scripted is not None:
            if isinstance(scripted, BaseException):
                raise scripted
            return scripted
        return _fixed_relevance_verdict("keep-clear")

    async def contradiction(
        self, existing: str, incoming: str, *, panel: PanelDisagreement | None = None
    ) -> ContradictionVerdict:
        self.calls.append(("contradiction", incoming, []))
        self.contradiction_calls.append((existing, incoming, panel))
        scripted = self._contradiction_script.get(existing)
        if scripted is not None:
            if isinstance(scripted, BaseException):
                raise scripted
            return scripted
        return _fixed_contradiction_verdict("coexist")
