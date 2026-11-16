"""One live ``relation`` call against typesafe.ai — deployed-server only, the
key lives at ``/etc/tabularium/typesafe.env`` there. Skipped cleanly everywhere else.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from tabularium.engine.assessor import AssessorCandidate, JevAssessor

pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).parent / "fixtures" / "jev_pairs_sample.json"


@pytest.mark.skipif(
    not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set"
)
async def test_relation_call_on_fixture_pair_a000() -> None:
    pairs = json.loads(_FIXTURE.read_text())
    pair = next(p for p in pairs if p["pair_id"] == "a000")

    api_key = os.environ["TYPESAFE_API_KEY"]
    assessor = JevAssessor(api_key, "jev-1.13.0")
    candidate = AssessorCandidate(
        id=pair["pair_id"],
        content=pair["existing_content"],
        kind="fact" if pair["existing_type"] == "fact" else "decision",
        scope=pair["existing_scope"],
        created_at=datetime.fromisoformat(pair["existing_created_at"]),
    )

    verdict = await assessor.relation(
        pair["incoming_content"], candidate, similarity=pair["similarity"]
    )

    assert sum(verdict.odds.values()) == pytest.approx(1.0, abs=1e-6)
    assert verdict.pick in verdict.odds
    assert 0.0 <= verdict.confidence <= 1.0
    assert verdict.trace.model == "jev-1.13.0"
    assert verdict.trace.latency_ms < 3000
