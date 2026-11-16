"""``JevAssessor``'s closed-label classifications and ``relation`` with neighbours, over
``httpx.MockTransport`` — request bodies and response-to-verdict mappings, no network."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from tabularium.engine import assessor as assessor_module
from tabularium.engine.admission import relation_verdicts
from tabularium.engine.assessor import (
    BREADTH_QUESTION,
    KIND_QUESTION,
    PERISHABLE_QUESTION,
    WORTH_QUESTION,
    AssessorCandidate,
    JevAssessor,
    JevError,
    QuestionSpec,
    Verdict,
    scope_question,
    wording_hash,
)

_NOW = datetime(2025, 6, 1, tzinfo=UTC)
_MODEL = "jev-1.13.0"


def _candidate(cid: str = "c1", content: str = "existing content") -> AssessorCandidate:
    return AssessorCandidate(
        id=cid, content=content, kind="fact", scope="global", created_at=_NOW
    )


def _response(question_id: str, choice: str, probabilities: dict[str, float]) -> dict[str, Any]:
    return {
        "model": _MODEL,
        "answers": {
            question_id: {"type": "choice", "choice": choice, "probabilities": probabilities},
        },
        "usage": {"input_tokens": 50, "output_tokens": 5},
    }


class _Recorder:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.bodies: list[dict[str, Any]] = []

    def client(self) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            self.bodies.append(json.loads(request.content))
            return httpx.Response(200, json=self.response)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _assessor(recorder: _Recorder) -> JevAssessor:
    return JevAssessor("key", _MODEL, client=recorder.client())


ClassifierCall = Callable[[JevAssessor], Any]


@pytest.mark.parametrize(
    ("method", "spec", "call", "state"),
    [
        (
            "worth",
            WORTH_QUESTION,
            lambda a: a.worth("some text"),
            {"text": "some text"},
        ),
        (
            "kind",
            KIND_QUESTION,
            lambda a: a.kind("some text", "decision"),
            {"text": "some text", "declared_kind": "decision"},
        ),
        (
            "perishable",
            PERISHABLE_QUESTION,
            lambda a: a.perishable("some text"),
            {"text": "some text"},
        ),
        (
            "breadth",
            BREADTH_QUESTION,
            lambda a: a.breadth("some text", "project-a"),
            {"text": "some text", "source_scope": "project-a"},
        ),
    ],
)
async def test_classification_request_body_and_verdict(
    method: str, spec: QuestionSpec, call: ClassifierCall, state: dict[str, Any]
) -> None:
    labels = list(spec.criteria)
    odds = dict.fromkeys(labels, 0.1)
    odds[labels[1]] = 1.0 - 0.1 * (len(labels) - 1)
    recorder = _Recorder(_response(spec.id, labels[1], odds))

    verdict = await call(_assessor(recorder))

    body = recorder.bodies[0]
    assert body["model"] == _MODEL
    assert body["state"] == state
    assert set(body["questions"]) == {spec.id}
    question = body["questions"][spec.id]
    assert question["type"] == "choice"
    assert question["instructions"] == spec.instructions
    assert list(question["criteria"]) == labels

    assert isinstance(verdict, Verdict)
    assert verdict.method == method
    assert verdict.pick == labels[1]
    assert verdict.odds == pytest.approx(odds)
    assert verdict.trace.model == _MODEL
    assert verdict.trace.wording_hash == wording_hash(spec)
    assert verdict.trace.input_tokens == 50


def test_closed_label_sets() -> None:
    assert list(WORTH_QUESTION.criteria) == ["keep", "derivable", "covered"]
    assert list(KIND_QUESTION.criteria) == ["fact", "decision", "neither"]
    assert list(PERISHABLE_QUESTION.criteria) == ["standing", "status", "sequencing"]
    assert list(BREADTH_QUESTION.criteria) == ["owner_wide", "project_specific"]


async def test_scope_asks_over_the_passed_labels() -> None:
    labels = [("global", "every project"), ("project-a", None), ("tabularium", "the memory server")]
    odds = {"global": 0.1, "project-a": 0.8, "tabularium": 0.1}
    recorder = _Recorder(_response("scope", "project-a", odds))

    verdict = await _assessor(recorder).scope("export endpoint", labels)

    body = recorder.bodies[0]
    assert body["state"] == {"text": "export endpoint"}
    assert body["questions"]["scope"]["criteria"] == {
        "global": "every project",
        "project-a": "project-a",
        "tabularium": "the memory server",
    }
    assert verdict.method == "scope"
    assert verdict.pick == "project-a"
    assert verdict.trace.wording_hash == wording_hash(scope_question(labels))


async def test_scope_wording_hash_follows_the_labels() -> None:
    first = scope_question([("a", None), ("b", None)])
    second = scope_question([("a", None), ("c", None)])
    assert wording_hash(first) != wording_hash(second)


async def test_worth_carries_neighbours_as_an_array() -> None:
    odds = {"keep": 0.1, "derivable": 0.1, "covered": 0.8}
    recorder = _Recorder(_response("worth", "covered", odds))
    neighbours = [_candidate("n1", "first near record"), _candidate("n2", "second near record")]

    verdict = await _assessor(recorder).worth("some text", neighbours=neighbours)

    state = recorder.bodies[0]["state"]
    assert state["text"] == "some text"
    assert [n["content"] for n in state["neighbours"]] == [
        "first near record",
        "second near record",
    ]
    assert verdict.pick == "covered"


async def test_worth_without_neighbours_has_no_neighbours_key() -> None:
    odds = {"keep": 1.0, "derivable": 0.0, "covered": 0.0}
    recorder = _Recorder(_response("worth", "keep", odds))

    await _assessor(recorder).worth("some text", neighbours=[])

    assert "neighbours" not in recorder.bodies[0]["state"]


async def test_probabilities_are_normalised() -> None:
    odds = {"owner_wide": 3.0, "project_specific": 1.0}
    recorder = _Recorder(_response("breadth", "owner_wide", odds))

    verdict = await _assessor(recorder).breadth("text", "project-a")

    assert verdict.odds == pytest.approx({"owner_wide": 0.75, "project_specific": 0.25})


@pytest.mark.parametrize(
    "response",
    [
        _response("breadth", "not_a_label", {"owner_wide": 0.5, "project_specific": 0.5}),
        _response("breadth", "owner_wide", {"owner_wide": 0.5, "invented": 0.5}),
        {"model": _MODEL, "answers": {}, "usage": {}},
        {"model": _MODEL, "answers": {"breadth": {"choice": "owner_wide"}}, "usage": {}},
    ],
)
async def test_malformed_classification_answers_raise(response: dict[str, Any]) -> None:
    recorder = _Recorder(response)

    with pytest.raises(JevError) as caught:
        await _assessor(recorder).breadth("text", "project-a")

    assert caught.value.message == "malformed_response"


_RELATION_RESPONSE = {
    "model": _MODEL,
    "answers": {
        "relation": {
            "type": "choice",
            "choice": "distinct",
            "probabilities": {
                "restatement": 0.0,
                "distinct": 1.0,
                "follow_up": 0.0,
                "replacement": 0.0,
            },
        }
    },
    "usage": {},
}


async def test_relation_without_neighbours_keeps_the_original_state() -> None:
    recorder = _Recorder(_RELATION_RESPONSE)

    await _assessor(recorder).relation("incoming", _candidate(), similarity=0.5)

    assert recorder.bodies[0]["state"] == {
        "existing": "existing content",
        "incoming": "incoming",
        "embedding_similarity": 0.5,
    }


async def test_relation_with_neighbours_carries_them_in_state() -> None:
    recorder = _Recorder(_RELATION_RESPONSE)
    neighbours = [_candidate("n1", "near one"), _candidate("n2", "near two")]

    await _assessor(recorder).relation(
        "incoming", _candidate(), similarity=0.5, neighbours=neighbours
    )

    state = recorder.bodies[0]["state"]
    assert state["existing"] == "existing content"
    assert state["neighbours"] == [
        {"content": "near one", "kind": "fact", "scope": "global"},
        {"content": "near two", "kind": "fact", "scope": "global"},
    ]


async def test_neighbours_are_truncated_per_record_and_in_total() -> None:
    recorder = _Recorder(_RELATION_RESPONSE)
    huge = "x" * (assessor_module._NEIGHBOUR_CHAR_CAP * 3)
    neighbours = [_candidate(f"n{i}", huge) for i in range(40)]

    await _assessor(recorder).relation(
        "incoming", _candidate(), similarity=None, neighbours=neighbours
    )

    sent = recorder.bodies[0]["state"]["neighbours"]
    assert all(len(n["content"]) == assessor_module._NEIGHBOUR_CHAR_CAP for n in sent)
    assert sum(len(n["content"]) for n in sent) <= assessor_module._NEIGHBOUR_TOTAL_CHAR_CAP
    assert 0 < len(sent) < 40


class _NeighbourSpy:
    def __init__(self) -> None:
        self.seen: list[Any] = []

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Any = None,
    ) -> Any:
        self.seen.append((candidate.id, None if neighbours is None else [n.id for n in neighbours]))
        raise RuntimeError("stop")


class _LegacySpy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def relation(
        self, incoming: str, candidate: AssessorCandidate, *, similarity: float | None
    ) -> Any:
        self.calls.append(candidate.id)
        raise RuntimeError("stop")


async def test_relation_verdicts_passes_neighbours_minus_the_survivor() -> None:
    spy = _NeighbourSpy()
    survivors = [(_candidate("a"), 0.9), (_candidate("b"), 0.8)]
    neighbours = [_candidate("a"), _candidate("c"), _candidate("d")]

    await relation_verdicts(
        spy,  # type: ignore[arg-type]
        "incoming",
        survivors,
        timeout_s=1.0,
        neighbours=neighbours,
    )

    assert sorted(spy.seen) == [("a", ["c", "d"]), ("b", ["a", "c", "d"])]


async def test_relation_verdicts_without_neighbours_omits_the_keyword() -> None:
    spy = _LegacySpy()

    await relation_verdicts(
        spy,  # type: ignore[arg-type]
        "incoming",
        [(_candidate("a"), 0.9)],
        timeout_s=1.0,
        neighbours=[_candidate("a")],
    )

    assert spy.calls == ["a"]
