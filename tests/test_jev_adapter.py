"""``JevAssessor`` over ``httpx.MockTransport`` — no network, ever (Assessor).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from tabularium.engine.admission import ask
from tabularium.engine.assessor import (
    AssessorCandidate,
    ContradictionVerdict,
    JevAssessor,
    JevError,
    NoReasonClass,
    PanelDisagreement,
    PanelJudgeAnswer,
    RelationVerdict,
    Unavailable,
    Verdict,
)

_NOW = datetime(2025, 6, 1, tzinfo=UTC)
_MODEL = "jev-1.13.0"


def _candidate() -> AssessorCandidate:
    return AssessorCandidate(
        id="c1", content="existing content", kind="fact", scope="global", created_at=_NOW
    )


def _relation_response(
    *,
    model: str = _MODEL,
    choice: str = "restatement",
    probabilities: dict[str, float] | None = None,
    reason_choice: str | None = None,
    reason_probabilities: dict[str, float] | None = None,
    input_tokens: int = 100,
    output_tokens: int = 10,
) -> dict[str, Any]:
    probabilities = probabilities or {
        "restatement": 0.9, "distinct": 0.05, "follow_up": 0.03, "replacement": 0.02,
    }
    answers: dict[str, Any] = {
        "relation": {"type": "choice", "choice": choice, "probabilities": probabilities},
    }
    if reason_choice is not None:
        answers["reason"] = {
            "type": "choice",
            "choice": reason_choice,
            "probabilities": reason_probabilities or {reason_choice: 1.0},
        }
    return {
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _client_for(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


async def test_request_body_and_auth_header_shape() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_relation_response())

    assessor = JevAssessor("secret-key", _MODEL, client=_client_for(handler))
    await assessor.relation("incoming text", _candidate(), similarity=0.87)

    assert captured["auth"] == "Bearer secret-key"
    body = captured["body"]
    assert body["model"] == _MODEL
    assert body["state"] == {
        "existing": "existing content", "incoming": "incoming text", "embedding_similarity": 0.87,
    }
    assert set(body["questions"]) == {"relation", "reason"}
    assert body["questions"]["relation"]["type"] == "choice"
    assert "criteria" in body["questions"]["relation"]
    assert "criteria" in body["questions"]["reason"]


# ---------------------------------------------------------------------------
# 200 success shapes
# ---------------------------------------------------------------------------


async def test_200_yields_relation_verdict_with_trace_from_response_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_relation_response(model="jev-1.13.0-resolved"),
            headers={"x-typesafe-request-id": "req-123"},
        )

    assessor = JevAssessor(
        "key", "jev-latest-would-be-refused-elsewhere", client=_client_for(handler)
    )
    verdict = await assessor.relation("incoming", _candidate(), similarity=0.5)

    assert isinstance(verdict, RelationVerdict)
    assert verdict.pick == "restatement"
    assert verdict.trace.model == "jev-1.13.0-resolved"
    assert verdict.trace.request_id == "req-123"
    assert verdict.trace.input_tokens == 100
    assert verdict.trace.output_tokens == 10
    assert verdict.reason_class is None


async def test_drifted_probabilities_are_renormalised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_relation_response(
                choice="distinct",
                probabilities={
                    "restatement": 0.30, "distinct": 0.30, "follow_up": 0.30, "replacement": 0.30,
                },
            ),
        )

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    verdict = await assessor.relation("incoming", _candidate(), similarity=None)

    assert sum(verdict.odds.values()) == pytest.approx(1.0, abs=1e-9)
    assert verdict.odds["distinct"] == pytest.approx(0.25, abs=1e-9)


async def test_replacement_with_a_reason_answer_yields_reason_class() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_relation_response(
                choice="replacement",
                probabilities={
                    "restatement": 0.0, "distinct": 0.0, "follow_up": 0.0, "replacement": 1.0,
                },
                reason_choice="correction",
            ),
        )

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    verdict = await assessor.relation("incoming", _candidate(), similarity=0.9)

    assert verdict.pick == "replacement"
    assert verdict.reason_class == "correction"
    assert verdict.reason_odds == {"correction": 1.0}


async def test_replacement_with_no_usable_reason_pick_raises_no_reason_class() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_relation_response(
                choice="replacement",
                probabilities={
                    "restatement": 0.0, "distinct": 0.0, "follow_up": 0.0, "replacement": 1.0,
                },
            ),
        )

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    with pytest.raises(NoReasonClass):
        await assessor.relation("incoming", _candidate(), similarity=0.9)


async def test_no_reason_class_through_ask_reports_the_wire_string() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_relation_response(
                choice="replacement",
                probabilities={
                    "restatement": 0.0, "distinct": 0.0, "follow_up": 0.0, "replacement": 1.0,
                },
            ),
        )

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))

    async def _factory() -> RelationVerdict:
        return await assessor.relation("incoming", _candidate(), similarity=0.9)

    result = await ask(_factory, method="relation", timeout_s=1.0)
    assert isinstance(result, Unavailable)
    assert result.reason == "error"
    assert result.detail == "no_reason_class"


# ---------------------------------------------------------------------------
# Error shapes
# ---------------------------------------------------------------------------


async def test_400_max_tokens_exceeded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": {"error_type": "max_tokens_exceeded"}})

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    with pytest.raises(JevError) as excinfo:
        await assessor.relation("incoming", _candidate(), similarity=None)
    assert excinfo.value.message == "max_tokens_exceeded"
    assert excinfo.value.status == 400


async def test_422_list_body_raises_with_first_msg() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422, json={"detail": [{"loc": ["body", "model"], "msg": "field required"}]}
        )

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    with pytest.raises(JevError) as excinfo:
        await assessor.relation("incoming", _candidate(), similarity=None)
    assert excinfo.value.message == "field required"
    assert excinfo.value.status == 422


async def test_401_authentication_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "authentication_error"})

    assessor = JevAssessor("bad-key", _MODEL, client=_client_for(handler))
    with pytest.raises(JevError) as excinfo:
        await assessor.relation("incoming", _candidate(), similarity=None)
    assert excinfo.value.message == "authentication_error"
    assert excinfo.value.status == 401


async def test_malformed_200_body_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"not_the_expected_shape": True})

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    with pytest.raises(JevError) as excinfo:
        await assessor.relation("incoming", _candidate(), similarity=None)
    assert excinfo.value.message == "malformed_response"


# ---------------------------------------------------------------------------
# Through the wrapper: every error shape above becomes a typed Unavailable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status_and_body", "expected_detail"),
    [
        ((400, {"detail": {"error_type": "max_tokens_exceeded"}}), "max_tokens_exceeded"),
        ((401, {"detail": "authentication_error"}), "authentication_error"),
        ((200, {"not_the_expected_shape": True}), "malformed_response"),
    ],
)
async def test_every_error_shape_becomes_unavailable_through_ask(
    status_and_body: tuple[int, dict[str, Any]], expected_detail: str,
) -> None:
    status, body = status_and_body

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))

    async def _factory() -> RelationVerdict:
        return await assessor.relation("incoming", _candidate(), similarity=None)

    result = await ask(_factory, method="relation", timeout_s=1.0)
    assert isinstance(result, Unavailable)
    assert result.reason == "error"
    assert result.detail == expected_detail


# ---------------------------------------------------------------------------
# A3: relevance (score, four grades)
# ---------------------------------------------------------------------------


def _relevance_response(
    *,
    model: str = _MODEL,
    score: float = 2.8,
    legend: dict[str, str] | None = None,
    probabilities: dict[str, float] | None = None,
) -> dict[str, Any]:
    """The real wire shape for a ``score`` question: no ``choice`` key at all, an index-keyed
    ``legend`` naming each level, and ``probabilities`` keyed by that same index — confirmed
    live against api.typesafe.ai, 2026-09-24."""
    legend = legend or {
        "0": "shelve-clear", "1": "shelve-lean", "2": "keep-lean", "3": "keep-clear",
    }
    probabilities = probabilities or {"0": 0.02, "1": 0.03, "2": 0.05, "3": 0.90}
    return {
        "model": model,
        "answers": {
            "relevance": {
                "type": "score",
                "score": score,
                "confidence": 0.6,
                "legend": legend,
                "probabilities": probabilities,
            },
        },
        "usage": {"input_tokens": 50, "output_tokens": 5},
    }


async def test_relevance_request_body_shape() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_relevance_response())

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    await assessor.relevance("some record text")

    body = captured["body"]
    assert body["state"] == {"text": "some record text"}
    assert set(body["questions"]) == {"relevance"}
    assert body["questions"]["relevance"]["type"] == "score"
    # QuestionSpec.criteria is a list here (RELEVANCE_QUESTION's own literal shape) -- the four
    # grade names, in order (Constraints: "keep the four grade names in order").
    assert body["questions"]["relevance"]["criteria"] == [
        "shelve-clear", "shelve-lean", "keep-lean", "keep-clear",
    ]


async def test_relevance_200_yields_verdict_with_grade_pick() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_relevance_response(
                probabilities={"0": 0.1, "1": 0.6, "2": 0.2, "3": 0.1},
            ),
        )

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    verdict = await assessor.relevance("some record text")

    assert isinstance(verdict, Verdict)
    assert verdict.method == "relevance"
    assert verdict.pick == "shelve-lean"
    assert sum(verdict.odds.values()) == pytest.approx(1.0, abs=1e-9)


async def test_relevance_pick_is_the_argmax_of_the_remapped_odds() -> None:
    """The exact wire shape reported live (no ``choice`` key), grade names remapped through
    ``legend`` and ``pick`` taken as the highest-probability grade."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": _MODEL,
                "answers": {
                    "relevance": {
                        "type": "score",
                        "score": 1.6,
                        "confidence": 0.04,
                        "legend": {
                            "0": "shelve-clear", "1": "shelve-lean",
                            "2": "keep-lean", "3": "keep-clear",
                        },
                        "probabilities": {"0": 0.24, "1": 0.19, "2": 0.28, "3": 0.29},
                    },
                },
            },
        )

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    verdict = await assessor.relevance("some record text")

    assert verdict.pick == "keep-clear"
    assert verdict.odds["keep-clear"] == pytest.approx(0.29, abs=1e-9)
    assert verdict.odds["keep-lean"] == pytest.approx(0.28, abs=1e-9)


async def test_relevance_legend_index_order_can_differ_from_grade_order() -> None:
    """The wire's index keys carry no ordering guarantee relative to the four grades — the
    remap must key off the legend's own mapping, never off index position."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_relevance_response(
                legend={
                    "0": "keep-clear", "1": "keep-lean", "2": "shelve-lean", "3": "shelve-clear",
                },
                probabilities={"0": 0.1, "1": 0.1, "2": 0.1, "3": 0.7},
            ),
        )

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    verdict = await assessor.relevance("some record text")

    assert verdict.pick == "shelve-clear"
    assert verdict.odds["shelve-clear"] == pytest.approx(0.7, abs=1e-9)
    assert verdict.odds["keep-clear"] == pytest.approx(0.1, abs=1e-9)


async def test_relevance_malformed_response_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": _MODEL, "answers": {}})

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    with pytest.raises(JevError):
        await assessor.relevance("text")


@pytest.mark.parametrize(
    ("legend", "probabilities"),
    [
        pytest.param(
            {"0": "shelve-lean", "1": "keep-lean", "2": "keep-clear"},
            {"0": 0.2, "1": 0.3, "2": 0.5},
            id="legend_missing_a_grade",
        ),
        pytest.param(
            {"0": "shelve-clear", "1": "shelve-lean", "2": "keep-lean", "3": "unsure"},
            {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25},
            id="grade_outside_the_four",
        ),
        pytest.param(
            {"0": "shelve-clear", "1": "shelve-lean", "2": "keep-lean", "3": "keep-clear"},
            {"0": 0.25, "1": 0.25, "2": 0.25, "4": 0.25},
            id="probability_keys_not_in_legend",
        ),
        pytest.param(
            {"a": "north", "b": "south", "c": "east", "d": "west"},
            {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
            id="legend_not_matching_criteria",
        ),
        pytest.param(
            {
                "0": "shelve-clear", "1": "shelve-lean", "2": "keep-lean", "3": "keep-clear",
                "4": "keep-clear",
            },
            {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.199, "4": 0.201},
            id="legend_duplicates_a_grade",
        ),
    ],
)
async def test_relevance_malformed_score_shapes_raise(
    legend: dict[str, str], probabilities: dict[str, float]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_relevance_response(legend=legend, probabilities=probabilities)
        )

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    with pytest.raises(JevError) as excinfo:
        await assessor.relevance("text")
    assert excinfo.value.message == "malformed_response"


# ---------------------------------------------------------------------------
# A3: contradiction (three-label choice collapsed to coexist/replace, D12)
# ---------------------------------------------------------------------------


def _contradiction_response(
    *, model: str = _MODEL, choice: str = "coexist", probabilities: dict[str, float] | None = None
) -> dict[str, Any]:
    probabilities = probabilities or {"coexist": 0.8, "correction": 0.1, "changed": 0.1}
    return {
        "model": model,
        "answers": {
            "contradiction": {"type": "choice", "choice": choice, "probabilities": probabilities},
        },
        "usage": {"input_tokens": 80, "output_tokens": 8},
    }


async def test_contradiction_request_body_shape_without_panel() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_contradiction_response())

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    await assessor.contradiction("older text", "newer text")

    body = captured["body"]
    assert body["state"] == {"existing": "older text", "incoming": "newer text"}
    assert set(body["questions"]) == {"contradiction"}
    assert set(body["questions"]["contradiction"]["criteria"]) == {
        "coexist", "correction", "changed",
    }


async def test_contradiction_request_body_carries_panel_as_third_named_state_key() -> None:
    """Uncertainty (Constraints): the panel context goes in as a third named input on ``state`` —
    Jev's own ``state`` is a free-form JSON object (01-jev-api-limits.md), so this is directly
    supported, no labelled-incoming-block fallback needed."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_contradiction_response())

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    panel = PanelDisagreement(
        judge_a=PanelJudgeAnswer(verdict="replace", reason="correction", why="A's reason"),
        judge_b=PanelJudgeAnswer(verdict="coexist", reason=None, why="B's reason"),
    )
    await assessor.contradiction("older text", "newer text", panel=panel)

    body = captured["body"]
    assert body["state"]["existing"] == "older text"
    assert body["state"]["incoming"] == "newer text"
    assert body["state"]["panel"] == {
        "judge_a": {"verdict": "replace", "reason": "correction", "why": "A's reason"},
        "judge_b": {"verdict": "coexist", "reason": None, "why": "B's reason"},
    }


async def test_contradiction_coexist_yields_coexist_pick_no_reason_class() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_contradiction_response(choice="coexist"))

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    verdict = await assessor.contradiction("older", "newer")

    assert isinstance(verdict, ContradictionVerdict)
    assert verdict.pick == "coexist"
    assert verdict.reason_class is None
    # D12: all three raw wire labels are kept in `odds`, even though `pick` only takes two values.
    assert set(verdict.odds) == {"coexist", "correction", "changed"}


@pytest.mark.parametrize("raw_choice", ["correction", "changed"])
async def test_contradiction_replace_labels_collapse_to_replace_pick(raw_choice: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_contradiction_response(choice=raw_choice))

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    verdict = await assessor.contradiction("older", "newer")

    assert verdict.pick == "replace"
    assert verdict.reason_class == raw_choice


async def test_contradiction_malformed_choice_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_contradiction_response(choice="unknown_label"))

    assessor = JevAssessor("key", _MODEL, client=_client_for(handler))
    with pytest.raises(JevError):
        await assessor.contradiction("older", "newer")
