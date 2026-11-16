"""The Assessor seam: engine-owned questions asked of an external judge over one envelope.

``relation`` judges an incoming save against one probe survivor. ``relevance`` and ``contradiction``
judge the Cleaner's pre-pass and split band. ``kind``, ``worth``, ``scope``, ``perishable``
and ``breadth`` are closed-label classifications, each one choice question whose
wording is a module-level ``QuestionSpec``. Each method answers with a pick, ``odds`` per label
summing to 1, a ``confidence`` the engine computes from those odds, and a ``trace``. The save and
Cleaner paths store the trace on a verdict log row. The methods return ``Verdict`` (``relation`` a
``RelationVerdict``), and ``contradiction`` returns its own ``ContradictionVerdict``. A failed or
absent call is a typed ``Unavailable``, never ``None`` and never a silent skip.

The engine wrapper ``engine.admission.ask`` owns the per-caller timeout and the mapping from a
raised exception to ``Unavailable``. This module owns the wire shapes, the question wording and the
transport.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

import httpx
from pydantic import (
    BaseModel,
    StrictBool,
    StrictFloat,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from tabularium.domain import AssessorPick, InvalidationReason, Kind

_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

Method = Literal[
    "relation",
    "kind",
    "worth",
    "scope",
    "perishable",
    "breadth",
    "relevance",
    "contradiction",
]
UnavailableReason = Literal["absent", "timeout", "error"]
Relation = Literal["restatement", "distinct", "follow_up", "replacement"]
ReasonClass = Literal["correction", "changed", "elaborated"]


class AssessorCandidate(BaseModel):
    """One existing record the Assessor weighs an incoming save against."""

    model_config = {"frozen": True}

    id: str
    content: str
    kind: Kind
    scope: str
    created_at: datetime


class Trace(BaseModel):
    """Provenance returned with a verdict. The save and Cleaner paths store it on a verdict log row.

    ``wording_hash`` is ``wording_hash(spec)`` for the question this verdict answers. ``model`` is
    the response's own ``model`` field, not the requested id, since the vendor may resolve an alias.
    ``request_id`` is the vendor's correlation id when the transport returns one.
    """

    model_config = {"frozen": True}

    model: str
    wording_hash: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    request_id: str | None


def compute_confidence(odds: Mapping[str, float]) -> float:
    """``(n*peak - 1)/(n - 1)`` clamped to ``[0, 1]``.

    One statistic for every method and adapter, computed by the engine and never taken from the
    vendor. A single label (``n == 1``) returns ``1.0``, since it carries no uncertainty to express.
    """
    n = len(odds)
    if n == 1:
        return 1.0
    peak = max(odds.values())
    confidence = (n * peak - 1) / (n - 1)
    return min(1.0, max(0.0, confidence))


def _check_odds(odds: Mapping[str, float], confidence: float, *, pick: str | None = None) -> None:
    """The envelope checks ``Verdict`` and ``ContradictionVerdict`` share, in firing order: odds
    non-empty, every value in ``[0, 1]``, odds summing to 1 (tolerance 1e-6), then ``pick`` in
    ``odds`` when a ``pick`` is given, then ``confidence`` equal to ``compute_confidence(odds)``.
    Raises ``ValueError`` on the first failure.
    """
    if not odds:
        raise ValueError("odds must not be empty")
    for label, value in odds.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"odds[{label!r}]={value!r} outside [0, 1]")
    total = sum(odds.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"odds sum to {total!r}, not 1")
    if pick is not None and pick not in odds:
        raise ValueError(f"pick={pick!r} not in odds")
    expected = compute_confidence(odds)
    if abs(confidence - expected) > 1e-9:
        raise ValueError(f"confidence={confidence!r} != compute_confidence(odds)={expected!r}")


class Verdict(BaseModel):
    """The shared envelope of an Assessor answer: a pick, the odds it came from, the confidence
    those odds imply and a trace.

    Validation rejects empty odds, a value outside ``[0, 1]``, odds that do not sum to 1 (tolerance
    1e-6), a pick not in ``odds``, and a ``confidence`` that differs from
    ``compute_confidence(odds)``.
    """

    model_config = {"frozen": True}

    method: Method
    pick: str
    odds: dict[str, float]
    confidence: float
    trace: Trace

    @model_validator(mode="after")
    def _validate_envelope(self) -> Verdict:
        _check_odds(self.odds, self.confidence, pick=self.pick)
        return self


class RelationVerdict(Verdict):
    """``relation``'s answer: ``pick`` narrowed to the four-way relation, plus the ``reason_class``
    that a ``replacement`` pick must carry.

    ``ReasonClass`` has no ``moved`` member, so the type itself excludes it. Any other pick must not
    carry a ``reason_class``.
    """

    model_config = {"frozen": True}

    pick: Relation
    reason_class: ReasonClass | None = None
    reason_odds: dict[str, float] | None = None

    @model_validator(mode="after")
    def _reason_class_matches_pick(self) -> RelationVerdict:
        if self.pick == "replacement" and self.reason_class is None:
            raise ValueError("pick='replacement' requires a reason_class")
        if self.pick != "replacement" and self.reason_class is not None:
            raise ValueError(f"pick={self.pick!r} must not carry a reason_class")
        return self


class Unavailable(BaseModel):
    """No verdict, typed and never ``None``. ``absent`` means no adapter is configured, ``timeout``
    that the engine's per-caller budget expired, and ``error`` that the call raised anything else.
    ``latency_ms`` is ``None`` on ``absent``, where the call never started.
    """

    model_config = {"frozen": True}

    method: Method
    reason: UnavailableReason
    detail: str | None
    latency_ms: int | None


class QuestionSpec(BaseModel):
    """One wire question's fixed wording: instructions plus criteria. ``wording_hash`` hashes both,
    so an edit to either changes the stamp the verdict log compares.
    """

    model_config = {"frozen": True}

    id: str
    type: Literal["choice", "score"]
    instructions: str
    criteria: dict[str, str | None] | list[str]


def wording_hash(spec: QuestionSpec) -> str:
    """sha256 of ``instructions``, a newline, then the sorted-key JSON of ``criteria``; the first 16
    hex characters. A rewording changes it.
    """
    payload = spec.instructions + "\n" + json.dumps(spec.criteria, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


RELATION_QUESTION = QuestionSpec(
    id="relation",
    type="choice",
    instructions=(
        "How does the 'incoming' text relate to the 'existing' record in a long-term memory "
        "store for a coding agent? 'existing' is the older record, already in the store; "
        "'incoming' is the newer text being evaluated against it."
    ),
    criteria={
        "restatement": (
            "same fact, no new information; the store should reinforce the existing record"
        ),
        "distinct": "a different fact that happens to be similar; both should be kept",
        "follow_up": (
            "same subject, a later step or a refinement that builds on the existing record "
            "without contradicting it; both should be kept"
        ),
        "replacement": (
            "same subject but the existing record is now wrong or outdated; the incoming text "
            "should supersede it"
        ),
    },
)

REASON_QUESTION = QuestionSpec(
    id="reason",
    type="choice",
    instructions=(
        "The 'relation' question above judged this a replacement. Why does the existing "
        "record no longer hold?"
    ),
    criteria={
        "correction": "the existing record was wrong when it was written",
        "changed": "the existing record was true when written and the world has since moved",
        "elaborated": (
            "the existing record is still true and the incoming text adds detail or precision"
        ),
    },
)


class PanelJudgeAnswer(BaseModel):
    """One judge's vote on a falsity pair the panel split on, passed into a ``contradiction`` call
    so the Assessor sees what each judge said and not only that they disagreed.
    """

    model_config = {"frozen": True}

    verdict: AssessorPick
    reason: str | None
    why: str


class PanelDisagreement(BaseModel):
    """Both judges' votes and reasoning on a falsity pair, passed as ``contradiction``'s ``panel``
    keyword when the panel split. A global-pair call or a plain falsity-split call has no
    disagreement to report and passes ``panel=None``.
    """

    model_config = {"frozen": True}

    judge_a: PanelJudgeAnswer
    judge_b: PanelJudgeAnswer


class ContradictionVerdict(BaseModel):
    """``contradiction``'s answer. The wire's three labels (``coexist``, ``correction``,
    ``changed``) are collapsed to a two-way ``pick`` (``coexist`` or ``replace``), and
    ``reason_class`` names which replace label won.

    ``odds`` keeps all three wire labels and ``confidence`` is computed from that three-way peak.
    This is a standalone envelope, not a ``Verdict`` subclass, because ``Verdict`` requires ``pick``
    to be a key of ``odds`` and a collapsed pick is not. A ``replace`` pick requires a
    ``reason_class`` and any other pick forbids one.
    """

    model_config = {"frozen": True}

    method: Literal["contradiction"] = "contradiction"
    pick: AssessorPick
    reason_class: InvalidationReason | None = None
    odds: dict[str, float]
    confidence: float
    trace: Trace

    @model_validator(mode="after")
    def _validate_envelope(self) -> ContradictionVerdict:
        _check_odds(self.odds, self.confidence)
        if self.pick == "replace" and self.reason_class is None:
            raise ValueError("pick='replace' requires a reason_class")
        if self.pick != "replace" and self.reason_class is not None:
            raise ValueError(f"pick={self.pick!r} must not carry a reason_class")
        return self


RELEVANCE_QUESTION = QuestionSpec(
    id="relevance",
    type="score",
    instructions=(
        "Should this record stay in a long-term memory store for a coding agent? "
        "Shelve it if a reader could land on it cheaply by reading the code, docs or git "
        "history; if it is false or irrelevant now; if it is a timeline event (task shipped, "
        "PR merged, suite green, deployed); or if it is session narrative rather than a fact "
        "the session produced. Keep it if it stands on its own: a convention, a gotcha, a "
        "defect or absence the code does not show, a correction, a preference, an "
        "infrastructure detail, or the reason something was done."
    ),
    criteria=["shelve-clear", "shelve-lean", "keep-lean", "keep-clear"],
)

CONTRADICTION_QUESTION = QuestionSpec(
    id="contradiction",
    type="choice",
    instructions=(
        "Two records from a long-term memory store for a coding agent. 'existing' is the "
        "older record, the newer. Does the newer record make the older one no "
        "longer true? When a 'panel' block is present, two judges disagreed on this pair; "
        "it gives each judge's answer and reasoning. Weigh their reasoning, but decide from "
        "the two records."
    ),
    criteria={
        "coexist": (
            "both records hold today; the newer one may add detail, precision or a separate "
            "point, but the older one is still true"
        ),
        "correction": "the older record was wrong when it was written; the newer one corrects it",
        "changed": (
            "the older record was true when written, but what it describes has since changed; "
            "the newer one describes the present"
        ),
    },
)


WORTH_QUESTION = QuestionSpec(
    id="worth",
    type="choice",
    instructions=(
        "Should this 'text' be written to a long-term memory store for a coding agent? "
        "When a 'neighbours' list is present it holds records the store already has that sit "
        "closest to the text."
    ),
    criteria={
        "keep": (
            "stands on its own and a reader could not get it cheaply from the code, docs or git "
            "history: a convention, a gotcha, a defect or absence the code does not show, a "
            "correction, a preference, an infrastructure detail, or the reason something was "
            "done"
        ),
        "derivable": (
            "a reader could land on it cheaply by reading the code, the docs or the git "
            "history, or it is a timeline event (task shipped, PR merged, suite green, "
            "deployed) or session narrative"
        ),
        "covered": "a record in 'neighbours' already says this; nothing new would be stored",
    },
)

KIND_QUESTION = QuestionSpec(
    id="kind",
    type="choice",
    instructions=(
        "What kind of record is this 'text' in a long-term memory store for a coding agent? "
        "'declared_kind' is what the writer called it; judge the text itself."
    ),
    criteria={
        "fact": "states how something is, was or behaves; a gotcha, a convention, a detail",
        "decision": (
            "records a choice that was made, with a reason or a rejected alternative: what was "
            "picked and why"
        ),
        "neither": (
            "a status update, a plan, a question, a task list or narrative; neither a fact "
            "worth keeping nor a decision"
        ),
    },
)

PERISHABLE_QUESTION = QuestionSpec(
    id="perishable",
    type="choice",
    instructions=(
        "Will this 'text' still be true and useful once the piece of work it came from is "
        "finished? It is a candidate record for a long-term memory store for a coding agent."
    ),
    criteria={
        "standing": (
            "holds regardless of the current task: a convention, a gotcha, a preference, an "
            "infrastructure detail, the reason a choice was made"
        ),
        "status": (
            "describes where the work stands right now: what is done, in progress, passing, "
            "failing or deployed; true today and stale after the work moves on"
        ),
        "sequencing": (
            "describes the order of steps, what to do next or what blocks what; only meaningful "
            "while the work is under way"
        ),
    },
)

BREADTH_QUESTION = QuestionSpec(
    id="breadth",
    type="choice",
    instructions=(
        "Does this 'text' apply to the owner's work across every project, or only to the "
        "project named in 'source_scope'? It is a candidate record for a long-term memory "
        "store for a coding agent, and a record that is owner-wide would be stored where every "
        "project reads it."
    ),
    criteria={
        "owner_wide": (
            "a working preference, a tooling or environment fact, or a rule that holds for the "
            "owner in every repository, whatever the project"
        ),
        "project_specific": (
            "true of one project, its code, its tickets, its infrastructure or its conventions; "
            "other projects would not need it"
        ),
    },
)


def scope_question(labels: Sequence[tuple[str, str | None]]) -> QuestionSpec:
    """The ``scope`` question over the caller's ``labels``: one criterion per label, its description
    when it has one, else the bare name.
    """
    return QuestionSpec(
        id="scope",
        type="choice",
        instructions=(
            "Which scope should this 'text' be stored under in a long-term memory store for a "
            "coding agent? Each label names a scope; pick the one the text belongs to."
        ),
        criteria={name: description or name for name, description in labels},
    )


@runtime_checkable
class Assessor(Protocol):
    """Async seam of fixed methods. Implementors raise on failure; the engine wrapper
    ``engine.admission.ask`` maps a raised ``Exception`` to ``Unavailable`` (``NotImplementedError``
    is re-raised), and this protocol does not.

    Callers may add keyword-only parameters to their own method; the return types are fixed here.
    """

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        """Judge ``incoming`` against one probe survivor, one call per survivor. ``neighbours`` are
        further records near ``incoming``, carried in ``state`` as an array; ``None`` or empty
        leaves the request as it would be without them.
        """
        ...

    async def kind(self, text: str, declared_kind: Kind) -> Verdict:
        """``pick`` one of ``fact``/``decision``/``neither`` (``KIND_QUESTION``)."""
        ...

    async def worth(
        self, text: str, *, neighbours: Sequence[AssessorCandidate] | None = None
    ) -> Verdict:
        """``pick`` one of ``keep``/``derivable``/``covered`` (``WORTH_QUESTION``);
        ``covered`` is only answerable against ``neighbours``."""
        ...

    async def scope(self, text: str, labels: Sequence[tuple[str, str | None]]) -> Verdict:
        """Which of ``labels`` best fits ``text`` — ``pick`` is one of the label names."""
        ...

    async def perishable(self, text: str) -> Verdict:
        """``pick`` one of ``standing``/``status``/``sequencing`` (``PERISHABLE_QUESTION``)."""
        ...

    async def breadth(self, text: str, source_scope: str) -> Verdict:
        """``pick`` one of ``owner_wide``/``project_specific`` (``BREADTH_QUESTION``) —
        whether ``text``, written under ``source_scope``, holds across every project."""
        ...

    async def relevance(self, text: str) -> Verdict:
        """The Cleaner's four ordered grades (``RELEVANCE_QUESTION``), one call per newcomer;
        ``pick`` is one of ``shelve-clear``, ``shelve-lean``, ``keep-lean`` or ``keep-clear``.
        """
        ...

    async def contradiction(
        self, existing: str, incoming: str, *, panel: PanelDisagreement | None = None
    ) -> ContradictionVerdict:
        """Coexist or replace, plus a reason class (``CONTRADICTION_QUESTION``). ``panel`` carries
        both judges' votes and reasoning when the Assessor is deciding a split or a reason
        disagreement; it is ``None`` for an ordinary global-pair call.
        """
        ...


class JevError(Exception):
    """A non-2xx or malformed response from typesafe.ai. ``status`` is the HTTP status.

    ``message`` is the body's ``detail.error_type`` when present, else the first of ``error``,
    ``message`` and ``detail`` (a string, or the ``msg`` of a list or dict), cut to 200 characters,
    else ``http_<status>``.
    """

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


class MalformedResponse(JevError):
    """A 2xx response whose body does not parse into the shape the question needs. ``message`` is
    always ``malformed_response``. ``status`` is the response's own status where the envelope itself
    is malformed and ``200`` where an answer inside a well-formed envelope is.
    """

    def __init__(self, status: int) -> None:
        super().__init__("malformed_response", status)


class NoReasonClass(Exception):
    """A Jev ``replacement`` pick carried no usable ``reason`` choice. ``engine.admission.ask``
    handles it like ``JevError`` and sets the wire ``detail`` string explicitly.
    """


Probability = StrictFloat | StrictBool


class JevAnswer(BaseModel):
    """One question's answer inside a Jev response.

    ``probabilities`` is required and numeric. ``choice`` is whatever the vendor sent: ``score``
    questions ignore it and the others check it against the odds. ``legend`` is the index-to-grade
    map of a ``score`` answer, and ``None`` when absent or not an object.
    """

    model_config = {"frozen": True}

    probabilities: dict[str, Probability]
    choice: object = None
    legend: dict[str, object] | None = None

    @field_validator("legend", mode="before")
    @classmethod
    def _legend_is_object_or_none(cls, value: object) -> object:
        return value if isinstance(value, dict) else None

    def picked(self, odds: Mapping[str, float]) -> str:
        """``choice`` when it is one of the labels in ``odds``, else ``MalformedResponse(200)``."""
        if not isinstance(self.choice, str) or self.choice not in odds:
            raise MalformedResponse(200)
        return self.choice


class JevUsage(BaseModel):
    """The token counts of a Jev response. ``Trace`` validates their types."""

    model_config = {"frozen": True}

    input_tokens: object = None
    output_tokens: object = None


class JevResponse(BaseModel):
    """The envelope of a 2xx Jev response: the resolved ``model``, ``usage`` (empty when absent or
    not an object) and one raw answer per question key.
    """

    model_config = {"frozen": True}

    model: StrictStr
    answers: dict[str, object]
    usage: JevUsage = JevUsage()

    @field_validator("usage", mode="before")
    @classmethod
    def _usage_is_object_or_empty(cls, value: object) -> object:
        return value if isinstance(value, dict) else {}

    def answer(self, key: str) -> JevAnswer:
        """The answer for question ``key``, validated. A missing key, a non-object answer or
        non-object, non-numeric ``probabilities`` raises ``MalformedResponse(200)``.
        """
        try:
            return JevAnswer.model_validate(self.answers.get(key))
        except ValidationError as exc:
            raise MalformedResponse(200) from exc


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:200] if text else f"http_{response.status_code}"
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict) and "error_type" in detail:
            return str(detail["error_type"])[:200]
        for key in ("error", "message", "detail"):
            value = body.get(key)
            if isinstance(value, str):
                return value[:200]
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, dict) and "msg" in first:
                    return str(first["msg"])[:200]
                return str(first)[:200]
    return f"http_{response.status_code}"


def _normalize_odds(probabilities: Mapping[str, float]) -> dict[str, float]:
    total = sum(probabilities.values())
    if total <= 0:
        raise MalformedResponse(200)
    if abs(total - 1.0) <= 1e-6:
        return dict(probabilities)
    return {label: value / total for label, value in probabilities.items()}


def _remap_score_odds(answer: JevAnswer, question: QuestionSpec) -> dict[str, float]:
    """Put a ``score`` answer's odds onto the grade names in ``question.criteria``, then normalize.

    A ``score`` answer has no ``choice``. It names its levels in an index-keyed ``legend`` and gives
    ``probabilities`` keyed by the same indices. Any mismatch raises ``MalformedResponse(200)``: a
    missing legend, a legend that is not a permutation of the grades, or probabilities keyed off
    something the legend does not define.
    """
    legend = answer.legend
    grades = question.criteria
    if legend is None or not isinstance(grades, list) or len(legend) != len(grades):
        raise MalformedResponse(200)
    named = {index: grade for index, grade in legend.items() if isinstance(grade, str)}
    if len(named) != len(legend) or set(named.values()) != set(grades):
        raise MalformedResponse(200)
    if set(answer.probabilities) != set(named):
        raise MalformedResponse(200)
    by_grade = {named[index]: answer.probabilities[index] for index in named}
    return _normalize_odds(by_grade)


_NEIGHBOUR_CHAR_CAP = 1500
_NEIGHBOUR_TOTAL_CHAR_CAP = 24000


def _neighbour_state(neighbours: Sequence[AssessorCandidate]) -> list[dict[str, str]]:
    """``neighbours`` as the array ``state`` carries: each record's content cut to
    ``_NEIGHBOUR_CHAR_CAP`` characters, and the whole array held to ``_NEIGHBOUR_TOTAL_CHAR_CAP``.
    In the order given, the first record that would pass the total cap and every record after it are
    dropped, not cut again.
    """
    entries: list[dict[str, str]] = []
    used = 0
    for neighbour in neighbours:
        content = neighbour.content[:_NEIGHBOUR_CHAR_CAP]
        if used + len(content) > _NEIGHBOUR_TOTAL_CHAR_CAP:
            break
        used += len(content)
        entries.append({"content": content, "kind": neighbour.kind, "scope": neighbour.scope})
    return entries


def _question_payload(spec: QuestionSpec) -> dict[str, object]:
    return {
        "type": spec.type,
        "instructions": spec.instructions,
        "criteria": spec.criteria,
    }


class JevAssessor:
    """Live adapter for typesafe.ai Jev over plain ``httpx``. It implements every ``Assessor``
    method: ``relation`` (optionally with neighbours), ``relevance`` and ``contradiction``, and the
    closed-label classifications over their own ``QuestionSpec``.

    The default client has no timeout (``httpx.AsyncClient(timeout=None)``). The per-call budget is
    the engine wrapper's ``asyncio.wait_for``, not a second ceiling here.
    """

    def __init__(
        self, api_key: str, model: str, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client if client is not None else httpx.AsyncClient(timeout=None)

    async def _ask(self, state: object, specs: Sequence[QuestionSpec]) -> tuple[JevResponse, Trace]:
        """Post ``specs`` as one request and parse the 2xx envelope. The trace carries the wording
        hash of ``specs[0]``, the question the verdict answers.
        """
        questions = {spec.id: _question_payload(spec) for spec in specs}
        start = time.monotonic()
        response = await self._client.post(
            _JEV_ENDPOINT,
            json={"state": state, "model": self._model, "questions": questions},
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        if response.status_code < 200 or response.status_code >= 300:
            raise JevError(_error_message(response), response.status_code)
        try:
            parsed = JevResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise MalformedResponse(response.status_code) from exc
        trace = Trace.model_validate(
            {
                "model": parsed.model,
                "wording_hash": wording_hash(specs[0]),
                "latency_ms": latency_ms,
                "input_tokens": parsed.usage.input_tokens,
                "output_tokens": parsed.usage.output_tokens,
                "request_id": response.headers.get("x-typesafe-request-id"),
            }
        )
        return parsed, trace

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        state: dict[str, object] = {
            "existing": candidate.content,
            "incoming": incoming,
            "embedding_similarity": similarity,
        }
        if neighbours:
            state["neighbours"] = _neighbour_state(neighbours)
        response, trace = await self._ask(state, (RELATION_QUESTION, REASON_QUESTION))
        answer = response.answer("relation")
        odds = _normalize_odds(answer.probabilities)
        pick = self._relation_pick(answer.picked(odds))

        reason_class: ReasonClass | None = None
        reason_odds: dict[str, float] | None = None
        if pick == "replacement":
            reason_class, reason_odds = self._reason(response)
            if reason_class is None:
                raise NoReasonClass(f"replacement with no usable reason pick: {pick!r}")

        return RelationVerdict(
            method="relation",
            pick=pick,
            odds=odds,
            confidence=compute_confidence(odds),
            trace=trace,
            reason_class=reason_class,
            reason_odds=reason_odds,
        )

    @staticmethod
    def _relation_pick(label: str) -> Relation:
        match label:
            case "restatement" | "distinct" | "follow_up" | "replacement" as pick:
                return pick
            case _:
                raise MalformedResponse(200)

    @staticmethod
    def _reason(response: JevResponse) -> tuple[ReasonClass | None, dict[str, float] | None]:
        try:
            reason = response.answer("reason")
        except MalformedResponse:
            return None, None
        reason_odds = _normalize_odds(reason.probabilities)
        match reason.choice:
            case "correction" | "changed" | "elaborated" as reason_class:
                return reason_class, reason_odds
            case _:
                return None, reason_odds

    async def _classify(
        self, state: dict[str, object], spec: QuestionSpec, method: Method
    ) -> Verdict:
        response, trace = await self._ask(state, (spec,))
        answer = response.answer(spec.id)
        if not set(answer.probabilities) <= set(spec.criteria):
            raise MalformedResponse(200)
        odds = _normalize_odds(answer.probabilities)
        pick = answer.picked(odds)
        return Verdict(
            method=method,
            pick=pick,
            odds=odds,
            confidence=compute_confidence(odds),
            trace=trace,
        )

    async def kind(self, text: str, declared_kind: Kind) -> Verdict:
        state: dict[str, object] = {"text": text, "declared_kind": declared_kind}
        return await self._classify(state, KIND_QUESTION, "kind")

    async def worth(
        self, text: str, *, neighbours: Sequence[AssessorCandidate] | None = None
    ) -> Verdict:
        state: dict[str, object] = {"text": text}
        if neighbours:
            state["neighbours"] = _neighbour_state(neighbours)
        return await self._classify(state, WORTH_QUESTION, "worth")

    async def scope(self, text: str, labels: Sequence[tuple[str, str | None]]) -> Verdict:
        state: dict[str, object] = {"text": text}
        return await self._classify(state, scope_question(labels), "scope")

    async def perishable(self, text: str) -> Verdict:
        state: dict[str, object] = {"text": text}
        return await self._classify(state, PERISHABLE_QUESTION, "perishable")

    async def breadth(self, text: str, source_scope: str) -> Verdict:
        state: dict[str, object] = {"text": text, "source_scope": source_scope}
        return await self._classify(state, BREADTH_QUESTION, "breadth")

    async def relevance(self, text: str) -> Verdict:
        response, trace = await self._ask({"text": text}, (RELEVANCE_QUESTION,))
        odds = _remap_score_odds(response.answer("relevance"), RELEVANCE_QUESTION)
        pick = max(odds, key=lambda grade: odds[grade])
        return Verdict(
            method="relevance",
            pick=pick,
            odds=odds,
            confidence=compute_confidence(odds),
            trace=trace,
        )

    async def contradiction(
        self, existing: str, incoming: str, *, panel: PanelDisagreement | None = None
    ) -> ContradictionVerdict:
        state: dict[str, object] = {"existing": existing, "incoming": incoming}
        if panel is not None:
            state["panel"] = panel.model_dump()
        response, trace = await self._ask(state, (CONTRADICTION_QUESTION,))
        answer = response.answer("contradiction")
        odds = _normalize_odds(answer.probabilities)
        pick: AssessorPick
        reason_class: InvalidationReason | None
        match answer.picked(odds):
            case "coexist":
                pick, reason_class = "coexist", None
            case "correction" | "changed" as replacing:
                pick, reason_class = "replace", replacing
            case _:
                raise MalformedResponse(200)
        return ContradictionVerdict(
            pick=pick,
            reason_class=reason_class,
            odds=odds,
            confidence=compute_confidence(odds),
            trace=trace,
        )
