"""The scripted Assessor behind the tool golden (``tests/test_golden_tools.py``).

``ToolAssessor`` answers by marker substring, like the cleaner golden's ``ScriptedAssessor``, but
covers ``relation`` (by incoming text and by candidate text), ``worth``, ``perishable``, ``kind``,
``scope`` and ``breadth``. A scripted exception instance is raised, which ``admission.ask`` turns
into ``Unavailable``; ``ScriptedInterrupt`` is a ``BaseException`` that ``ask`` does not catch.
``log`` records every call to those methods, in order. ``relevance`` and ``contradiction`` are
``EchoAssessor``'s and are not logged here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from tabularium.engine.assessor import (
    AssessorCandidate,
    Method,
    ReasonClass,
    Relation,
    RelationVerdict,
    Trace,
    Verdict,
    compute_confidence,
)
from tests.support.fake_assessor import EchoAssessor

Label = Literal["repetition", "elaboration", "addition", "correction", "changed state", "unrelated"]

_LABEL_RELATION: dict[Label, Relation] = {
    "repetition": "restatement",
    "elaboration": "replacement",
    "addition": "follow_up",
    "correction": "replacement",
    "changed state": "replacement",
    "unrelated": "distinct",
}
_LABEL_REASON: dict[Label, ReasonClass | None] = {
    "repetition": None,
    "elaboration": "elaborated",
    "addition": None,
    "correction": "correction",
    "changed state": "changed",
    "unrelated": None,
}
_RELATIONS = ("restatement", "replacement", "follow_up", "distinct")


class ScriptedInterrupt(BaseException):
    """A scripted Assessor answer that escapes ``admission.ask``, which catches ``Exception`` only.

    It stands for a cancellation arriving while the Assessor runs.
    """


@dataclass(frozen=True)
class Pick:
    """One scripted pick and the odds mass on it. Below 1, the rest is spread evenly over the other
    labels of the method, and the pick keeps all the mass when it has no other label."""

    pick: str
    peak: float = 1.0


Answer = str | Pick | BaseException
RelationAnswer = Label | Pick | BaseException


def _as_label(name: str) -> Label:
    for label in _LABEL_RELATION:
        if label == name:
            return label
    raise ValueError(f"unknown relation label {name!r}")


def _trace(method: str) -> Trace:
    return Trace(
        model="tool-golden-assessor",
        wording_hash=f"tool-golden-{method}",
        latency_ms=0,
        input_tokens=None,
        output_tokens=None,
        request_id=None,
    )


def _odds(pick: str, peak: float, labels: Sequence[str]) -> dict[str, float]:
    if peak >= 1.0:
        return {pick: 1.0}
    others = [label for label in labels if label != pick]
    if not others:
        return {pick: 1.0}
    share = (1.0 - peak) / len(others)
    return {pick: peak, **dict.fromkeys(others, share)}


def _verdict(method: Method, answer: str | Pick, labels: Sequence[str]) -> Verdict:
    chosen = answer if isinstance(answer, Pick) else Pick(answer)
    odds = _odds(chosen.pick, chosen.peak, labels)
    return Verdict(
        method=method,
        pick=chosen.pick,
        odds=odds,
        confidence=compute_confidence(odds),
        trace=_trace(method),
    )


def _lookup[T](table: Mapping[str, T], text: str) -> T | None:
    for marker, answer in table.items():
        if marker in text:
            return answer
    return None


class ToolAssessor(EchoAssessor):
    """Scripted answers by marker substring; every unscripted call gets a plain default.

    - ``relation``: the label in ``by_incoming`` for a marker found in the incoming text, else the
      label in ``by_candidate`` for a marker found in the candidate text, else ``label``. A
      ``Pick`` as the label sets the odds peak. A scripted exception is raised.
    - ``on_call(method, text)`` is awaited in every call after it is logged and before the answer
      is chosen, so a scenario can change the store while the Assessor is busy.
    - ``worth`` (default ``keep``), ``perishable`` (``standing``), ``kind`` (the declared kind),
      ``breadth`` (``project_specific``) and ``scope`` (``default_scope``) answer by marker in
      ``text``.
    """

    def __init__(
        self,
        label: Label = "unrelated",
        *,
        by_incoming: Mapping[str, RelationAnswer] | None = None,
        by_candidate: Mapping[str, RelationAnswer] | None = None,
        worth: Mapping[str, Answer] | None = None,
        perishable: Mapping[str, Answer] | None = None,
        kind_script: Mapping[str, Answer] | None = None,
        scope_script: Mapping[str, Answer] | None = None,
        breadth_script: Mapping[str, Answer] | None = None,
        default_scope: str = "p",
        on_call: Callable[[Method, str], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(label)
        self._default_label: Label = label
        self._by_incoming = dict(by_incoming or {})
        self._by_candidate = dict(by_candidate or {})
        self._worth = dict(worth or {})
        self._perishable = dict(perishable or {})
        self._kind = dict(kind_script or {})
        self._scope = dict(scope_script or {})
        self._breadth = dict(breadth_script or {})
        self._default_scope = default_scope
        self._on_call = on_call
        self.log: list[dict[str, object]] = []

    async def _answer(
        self,
        method: Method,
        table: Mapping[str, Answer],
        text: str,
        default: str,
        labels: Sequence[str],
        extra: Mapping[str, object] | None = None,
    ) -> Verdict:
        self.log.append({"method": method, "text": text, **(extra or {})})
        if self._on_call is not None:
            await self._on_call(method, text)
        scripted = _lookup(table, text)
        if isinstance(scripted, BaseException):
            raise scripted
        return _verdict(method, scripted if scripted is not None else default, labels)

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        self.log.append(
            {
                "method": "relation",
                "incoming": incoming,
                "candidate": candidate.content,
                "candidate_kind": candidate.kind,
                "candidate_scope": candidate.scope,
                "similarity": similarity,
                "neighbours": [other.content for other in neighbours or ()],
            }
        )
        if self._on_call is not None:
            await self._on_call("relation", incoming)
        scripted = _lookup(self._by_incoming, incoming)
        if scripted is None:
            scripted = _lookup(self._by_candidate, candidate.content)
        if isinstance(scripted, BaseException):
            raise scripted

        label, peak = self._default_label, 1.0
        if isinstance(scripted, Pick):
            label, peak = _as_label(scripted.pick), scripted.peak
        elif scripted is not None:
            label = scripted
        relation = _LABEL_RELATION[label]
        reason = _LABEL_REASON[label]
        odds = _odds(relation, peak, _RELATIONS)
        return RelationVerdict(
            method="relation",
            pick=relation,
            odds=odds,
            confidence=compute_confidence(odds),
            trace=_trace("relation"),
            reason_class=reason,
            reason_odds={reason: 1.0} if reason is not None else None,
        )

    async def kind(self, text: str, declared_kind: str) -> Verdict:
        return await self._answer(
            "kind",
            self._kind,
            text,
            declared_kind,
            ("fact", "decision", "neither"),
            {"declared_kind": declared_kind},
        )

    async def worth(
        self, text: str, *, neighbours: Sequence[AssessorCandidate] | None = None
    ) -> Verdict:
        return await self._answer(
            "worth", self._worth, text, "keep", ("keep", "derivable", "covered")
        )

    async def perishable(self, text: str) -> Verdict:
        return await self._answer(
            "perishable", self._perishable, text, "standing", ("standing", "status", "sequencing")
        )

    async def breadth(self, text: str, source_scope: str) -> Verdict:
        return await self._answer(
            "breadth",
            self._breadth,
            text,
            "project_specific",
            ("owner_wide", "project_specific"),
            {"source_scope": source_scope},
        )

    async def scope(self, text: str, labels: Sequence[tuple[str, str | None]]) -> Verdict:
        names = [name for name, _ in labels]
        return await self._answer(
            "scope", self._scope, text, self._default_scope, names, {"labels": names}
        )
