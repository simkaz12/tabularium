"""Tests for ``tests/support/`` — the case-set runner's loader, fault hooks, fake Assessor,
seeding and ideal-tier prompt/parse/score round-trip (brief Scope 2).
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import HypomnemaId, Scope
from tabularium.embedding import EmbedderUnavailable
from tabularium.engine import Tabularium
from tabularium.engine.assessor import AssessorCandidate
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import StorageAdapter
from tests.conformance.test_conformance import (
    _fact,
    _make_lancedb_adapter,
    _make_memory_adapter,
    _vec,
)
from tests.support.cases import (
    RESERVED_EXISTING_ALIAS,
    Case,
    CaseLoadError,
    case_texts,
    load_cases,
)
from tests.support.doubles import FakeEmbedder, FakeJudgeAdapter, TestClock
from tests.support.fake_assessor import EchoAssessor
from tests.support.faults import (
    FaultInjectingAdapter,
    FaultInjectingEmbedder,
    read_only_table_dir,
)
from tests.support.ideal_judge import (
    IdealVerdict,
    build_ideal_prompt,
    parse_ideal_response,
    score_ideal,
)
from tests.support.seeding import ArchivedStartUnsupported, seed_case

_EPOCH = datetime(2025, 1, 1, tzinfo=UTC)

_FULL_CASE_YAML = """
id: full-01
class: material
origin: authored
existing:
  - alias: e1
    text: "Existing text one."
    kind: fact
    scope: global
    lifecycle: active
    validity: valid
    created_at: "2025-01-01T00:00:00+00:00"
incoming:
  content: "Incoming text."
  scope: global
  kind: fact
  supersedes: [e1]
  reason: elaborated
expect:
  selection:
    aliases: [e1]
    k: 5
  material: elaboration
  mutation:
    acceptable:
      - "rows == 2"
      - "e1.validity == invalid"
      - "e1.superseded_by == e2"
      - "e1.text unchanged"
      - "history says elaborated"
      - "pair(e1,e2) registered"
    ideal: []
why: "loader coverage fixture"
"""


def _write_case(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_load_cases_loads_every_field_and_all_six_check_kinds(tmp_path: Path) -> None:
    _write_case(tmp_path, "full-01.yaml", _FULL_CASE_YAML)

    cases = load_cases(dir=tmp_path)

    assert len(cases) == 1
    case = cases[0]
    assert case.id == "full-01"
    assert case.class_ == "material"
    assert case.origin == "authored"
    assert case.incoming.supersedes == ["e1"]
    assert case.incoming.reason == "elaborated"
    assert case.expect is not None
    assert case.expect.selection is not None
    assert case.expect.selection.aliases == ["e1"]
    kinds = {check.kind for check in case.expect.mutation.acceptable}
    assert kinds == {
        "rows_equal",
        "validity_equals",
        "superseded_by_equals",
        "text_unchanged",
        "history_says",
        "pair_registered",
    }


def test_load_cases_unknown_check_string_names_file_and_string(tmp_path: Path) -> None:
    bad = _FULL_CASE_YAML.replace('"rows == 2"', '"rows === 2"')
    _write_case(tmp_path, "bad-check.yaml", bad)

    with pytest.raises(CaseLoadError) as excinfo:
        load_cases(dir=tmp_path)

    message = str(excinfo.value)
    assert "bad-check.yaml" in message
    assert "rows === 2" in message


def test_load_cases_unknown_material_label_names_file_and_string(tmp_path: Path) -> None:
    bad = _FULL_CASE_YAML.replace("material: elaboration", "material: nonsense")
    _write_case(tmp_path, "bad-material.yaml", bad)

    with pytest.raises(CaseLoadError) as excinfo:
        load_cases(dir=tmp_path)

    message = str(excinfo.value)
    assert "bad-material.yaml" in message
    assert "nonsense" in message


def test_load_cases_misspelled_expect_key_names_file_and_key(tmp_path: Path) -> None:
    """Pydantic ignores unknown keys by default,
    so ``acceptablee`` (a typo for ``acceptable``) silently drops the assertion — the case loads
    with an empty ``acceptable`` list and passes without checking the store. ``extra="forbid"``
    on every case model turns the typo into a load-time error instead."""
    bad = _FULL_CASE_YAML.replace("acceptable:", "acceptablee:")
    _write_case(tmp_path, "bad-key.yaml", bad)

    with pytest.raises(CaseLoadError) as excinfo:
        load_cases(dir=tmp_path)

    message = str(excinfo.value)
    assert "bad-key.yaml" in message
    assert "acceptablee" in message


def test_load_cases_existing_row_reusing_the_incoming_alias_is_rejected(tmp_path: Path) -> None:
    """The runner binds the reserved alias
    ``"incoming"`` to the record ``engine.save(case.incoming...)`` just wrote, AFTER seeding —
    an ``existing[]`` row aliased ``incoming`` would have its id silently overwritten by that
    bind, so a check against ``incoming`` could pass against the wrong record. Rejected at load
    time instead."""
    bad = _FULL_CASE_YAML.replace("e1", RESERVED_EXISTING_ALIAS)
    _write_case(tmp_path, "reserved-alias.yaml", bad)

    with pytest.raises(CaseLoadError) as excinfo:
        load_cases(dir=tmp_path)

    message = str(excinfo.value)
    assert "reserved-alias.yaml" in message
    assert RESERVED_EXISTING_ALIAS in message


_DUPLICATE_ALIAS_CASE_YAML = """
id: duplicate-alias-01
class: material
origin: authored
existing:
  - alias: e1
    text: "Existing text one."
    kind: fact
    scope: global
    lifecycle: active
    validity: valid
    created_at: "2025-01-01T00:00:00+00:00"
  - alias: e1
    text: "Existing text two."
    kind: fact
    scope: global
    lifecycle: active
    validity: valid
    created_at: "2025-01-01T00:00:00+00:00"
incoming:
  content: "Incoming text."
  scope: global
  kind: fact
expect:
  material: unrelated
  mutation:
    acceptable: []
    ideal: []
why: "duplicate-alias loader fixture"
"""


def test_load_cases_duplicate_existing_aliases_are_rejected(tmp_path: Path) -> None:
    """Two ``existing[]`` rows sharing an alias make every check against that alias ambiguous —
    rejected at load time rather than silently resolving to whichever row a lookup happens to
    find first."""
    _write_case(tmp_path, "duplicate-alias.yaml", _DUPLICATE_ALIAS_CASE_YAML)

    with pytest.raises(CaseLoadError) as excinfo:
        load_cases(dir=tmp_path)

    message = str(excinfo.value)
    assert "duplicate-alias.yaml" in message
    assert "e1" in message


def test_case_texts_returns_existing_and_incoming_texts(tmp_path: Path) -> None:
    _write_case(tmp_path, "full-01.yaml", _FULL_CASE_YAML)
    cases = load_cases(dir=tmp_path)

    texts = case_texts(cases)

    assert "Existing text one." in texts
    assert "Incoming text." in texts


def test_load_cases_on_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    assert load_cases(dir=tmp_path) == []


# ---------------------------------------------------------------------------
# FaultInjectingAdapter
# ---------------------------------------------------------------------------


class _CountingMemoryAdapter:
    """Wraps a real ``MemoryAdapter`` and counts ``upsert`` calls that actually reached it —
    the falsifiable proof that ``FaultInjectingAdapter`` raises BEFORE delegation."""

    def __init__(self, dim: int) -> None:
        self.inner = _make_memory_adapter(dim=dim)
        self.upsert_calls = 0

    async def upsert(self, record: Any, vector: list[float] | None) -> None:
        self.upsert_calls += 1
        await self.inner.upsert(record, vector)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


async def test_fault_injecting_adapter_counts_only_after_arm_and_raises_before_delegation() -> (
    None
):
    dim = 4
    counting = _CountingMemoryAdapter(dim)
    wrapper = FaultInjectingAdapter(counting)
    wrapper.fault(on="upsert", call=2, exc=RuntimeError("boom"))

    await wrapper.upsert(_fact(content="pre-arm one"), _vec(dim))
    await wrapper.upsert(_fact(content="pre-arm two"), _vec(dim))
    assert counting.upsert_calls == 2, "calls before arm() must delegate, never raise"

    wrapper.arm()

    await wrapper.upsert(_fact(content="post-arm first"), _vec(dim))
    assert counting.upsert_calls == 3

    with pytest.raises(RuntimeError, match="boom"):
        await wrapper.upsert(_fact(content="post-arm second"), _vec(dim))
    assert counting.upsert_calls == 3, "the Nth call must raise BEFORE reaching the inner adapter"


# ---------------------------------------------------------------------------
# FaultInjectingEmbedder
# ---------------------------------------------------------------------------


async def test_fault_injecting_embedder_raises_after_arm_and_recovers_after_lift() -> None:
    inner = FakeEmbedder(dim=8)
    wrapper = FaultInjectingEmbedder(inner)

    before = await wrapper.embed("hello")
    assert before == await inner.embed("hello")

    wrapper.arm()
    with pytest.raises(EmbedderUnavailable):
        await wrapper.embed("hello")

    wrapper.lift()
    after = await wrapper.embed("hello")
    assert after == await inner.embed("hello")


async def test_fault_injecting_embedder_nth_call_form() -> None:
    inner = FakeEmbedder(dim=8)
    wrapper = FaultInjectingEmbedder(inner)
    wrapper.fault(on="embed", call=2, exc=ValueError("boom"))
    wrapper.arm()

    await wrapper.embed("first call passes")
    with pytest.raises(ValueError, match="boom"):
        await wrapper.embed("second call raises")


# ---------------------------------------------------------------------------
# read_only_table_dir
# ---------------------------------------------------------------------------


async def _active_total(adapter: StorageAdapter) -> int:
    return len(
        [
            record
            async for record in await adapter.scan_records(lifecycle_in=frozenset({"active"}))
            if record.scope == Scope("global") and record.validity == "valid"
        ]
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="running as root, read-only bit has no effect")
async def test_read_only_table_dir_blocks_writes_and_restores(tmp_path: Path) -> None:
    store_path = tmp_path / "store"
    embedder = FakeEmbedder(dim=256)
    clock = TestClock(_EPOCH)
    adapter = await LanceDBAdapter.open(
        str(store_path), embedding_model=embedder.model, dim=embedder.dim, clock=clock
    )
    engine = Tabularium(adapter, embedder, clock, EngineConfig())
    await engine.initialize()

    r1 = await engine.save("Row one for the disk-fault test.", scope="global")
    r2 = await engine.save("Row two for the disk-fault test.", scope="global")
    assert r1.outcome == "saved"
    assert r2.outcome == "saved"

    before = await _active_total(adapter)
    assert before == 2

    with read_only_table_dir(store_path, "records"):
        with pytest.raises(Exception):  # noqa: B017 — any OSError/PermissionError-derived surface
            await engine.save("A fresh row while the table dir is read-only.", scope="global")
        mid = await _active_total(adapter)
        assert mid == 2

    after = await engine.save("A fresh row after the read-only bit lifts.", scope="global")
    assert after.outcome == "saved"
    final = await _active_total(adapter)
    assert final == 3


# ---------------------------------------------------------------------------
# EchoAssessor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected_relation", "expected_reason"),
    [
        ("repetition", "restatement", None),
        ("elaboration", "replacement", "elaborated"),
        ("addition", "follow_up", None),
        ("correction", "replacement", "correction"),
        ("changed state", "replacement", "changed"),
        ("unrelated", "distinct", None),
    ],
)
async def test_echo_assessor_maps_every_label_per_d7(
    label: Any, expected_relation: Any, expected_reason: Any
) -> None:
    assessor = EchoAssessor(label)
    candidate = AssessorCandidate(
        id="c1", content="Existing content.", kind="fact", scope="global",
        created_at=datetime(2025, 6, 1, tzinfo=UTC),
    )

    verdict = await assessor.relation("incoming text", candidate, similarity=None)

    assert verdict.pick == expected_relation
    assert verdict.reason_class == expected_reason
    assert assessor.calls == [("relation", "incoming text", ["c1"])]


# ---------------------------------------------------------------------------
# seed_case — parametrized over both storage adapters
# ---------------------------------------------------------------------------


@pytest.fixture(params=["memory", "lancedb"])
async def engine_and_clock(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[Tabularium, TestClock, StorageAdapter]:
    clock = TestClock(_EPOCH)
    embedder = FakeEmbedder(dim=256)
    adapter: MemoryAdapter | LanceDBAdapter
    if request.param == "memory":
        adapter = _make_memory_adapter(dim=256, clock=clock)
    else:
        adapter = await _make_lancedb_adapter(tmp_path, dim=256, clock=clock)
    engine = Tabularium(adapter, embedder, clock, EngineConfig())
    await engine.initialize()
    return engine, clock, adapter


def _case(**overrides: Any) -> Case:
    defaults: dict[str, Any] = {
        "id": "seed-case",
        "class": "material",
        "origin": "authored",
        "existing": [],
        "incoming": {"content": "Incoming.", "scope": "global", "kind": "fact"},
        "expect": {"material": "unrelated", "mutation": {"acceptable": [], "ideal": []}},
        "why": "seed test fixture",
    }
    defaults.update(overrides)
    return Case.model_validate(defaults)


async def test_seed_case_sets_clock_before_each_save(
    engine_and_clock: tuple[Tabularium, TestClock, StorageAdapter],
) -> None:
    engine, clock, adapter = engine_and_clock
    t1 = datetime(2025, 1, 1, tzinfo=UTC)
    t2 = datetime(2025, 2, 1, tzinfo=UTC)
    case = _case(
        existing=[
            {
                "alias": "e1",
                "text": "Row one.",
                "kind": "fact",
                "scope": "global",
                "lifecycle": "active",
                "validity": "valid",
                "created_at": t1.isoformat(),
            },
            {
                "alias": "e2",
                "text": "Row two.",
                "kind": "fact",
                "scope": "global",
                "lifecycle": "active",
                "validity": "valid",
                "created_at": t2.isoformat(),
            },
        ]
    )

    alias_to_id = await seed_case(engine, clock, case)

    assert set(alias_to_id) == {"e1", "e2"}
    hit1 = await adapter.get_by_id(HypomnemaId(uuid.UUID(alias_to_id["e1"])))
    hit2 = await adapter.get_by_id(HypomnemaId(uuid.UUID(alias_to_id["e2"])))
    assert hit1.created_at == t1
    assert hit2.created_at == t2


async def test_seed_case_invalid_row_reads_back_invalid(
    engine_and_clock: tuple[Tabularium, TestClock, StorageAdapter],
) -> None:
    engine, clock, adapter = engine_and_clock
    case = _case(
        existing=[
            {
                "alias": "e1",
                "text": "Row invalid.",
                "kind": "fact",
                "scope": "global",
                "lifecycle": "active",
                "validity": "invalid",
                "created_at": _EPOCH.isoformat(),
            },
        ]
    )

    alias_to_id = await seed_case(engine, clock, case)

    hit = await adapter.get_by_id(HypomnemaId(uuid.UUID(alias_to_id["e1"])))
    assert hit.validity == "invalid"


async def test_seed_case_archived_row_raises_archived_start_unsupported(
    engine_and_clock: tuple[Tabularium, TestClock, StorageAdapter],
) -> None:
    engine, clock, _adapter = engine_and_clock
    case = _case(
        existing=[
            {
                "alias": "e1",
                "text": "Row archived.",
                "kind": "fact",
                "scope": "global",
                "lifecycle": "archived",
                "validity": "valid",
                "created_at": _EPOCH.isoformat(),
            },
        ]
    )

    with pytest.raises(ArchivedStartUnsupported):
        await seed_case(engine, clock, case)


# ---------------------------------------------------------------------------
# ideal_judge
# ---------------------------------------------------------------------------


def _ideal_case(**overrides: Any) -> Case:
    defaults: dict[str, Any] = {
        "existing": [
            {
                "alias": "e1",
                "text": "Existing content for the ideal-tier test.",
                "kind": "fact",
                "scope": "global",
                "lifecycle": "active",
                "validity": "valid",
                "created_at": _EPOCH.isoformat(),
            },
        ],
        "incoming": {
            "content": "Incoming content for the ideal-tier test.",
            "scope": "global",
            "kind": "fact",
        },
        "expect": {"material": "addition", "mutation": {"acceptable": [], "ideal": ["rows == 2"]}},
        "why": "the human reviewer's note, never shown to the judge",
    }
    defaults.update(overrides)
    return _case(**defaults)


def test_build_ideal_prompt_carries_existing_and_incoming_never_why() -> None:
    case = _ideal_case()

    prompt = build_ideal_prompt(case)

    assert "Existing content for the ideal-tier test." in prompt
    assert "Incoming content for the ideal-tier test." in prompt
    assert case.why not in prompt


def test_build_ideal_prompt_does_not_rescan_existing_text_for_the_incoming_token() -> None:
    """Two chained ``.replace()`` calls let the
    incoming substitution rescan text the existing substitution just inserted. An existing row
    whose content is the literal string ``"{incoming}"`` must survive verbatim; the incoming
    content must appear only in its own designated slot, not injected into that row too."""
    case = _ideal_case(
        existing=[
            {
                "alias": "e1",
                "text": "{incoming}",
                "kind": "fact",
                "scope": "global",
                "lifecycle": "active",
                "validity": "valid",
                "created_at": _EPOCH.isoformat(),
            },
        ],
        incoming={"content": "Real incoming text.", "scope": "global", "kind": "fact"},
    )

    prompt = build_ideal_prompt(case)

    assert "[e1] {incoming}" in prompt
    assert prompt.count("Real incoming text.") == 1


def test_parse_ideal_response_returns_label_and_checks() -> None:
    raw = json.dumps({"material": "addition", "checks": ["rows == 2"]})

    verdict = parse_ideal_response(raw)

    assert verdict.material == "addition"
    assert [check.raw for check in verdict.checks] == ["rows == 2"]


async def test_score_ideal_round_trips_through_a_fake_judge_with_zero_failures() -> None:
    case = _ideal_case()
    raw = json.dumps({"material": "addition", "checks": ["rows == 2"]})
    judge = FakeJudgeAdapter(response=raw)

    response = await judge.call(build_ideal_prompt(case))
    verdict = parse_ideal_response(response)
    failures = score_ideal(case, verdict)

    assert failures == []


def test_score_ideal_treats_a_whitespace_only_check_difference_as_a_match() -> None:
    """Raw-string comparison reported
    ``pair(e1,e2) registered`` (expected) and ``pair(e1, e2) registered`` (returned) as one
    missing plus one unexpected check, though both parse to the identical typed check."""
    case = _ideal_case(
        expect={
            "material": "addition",
            "mutation": {"acceptable": [], "ideal": ["pair(e1,e2) registered"]},
        },
    )
    verdict = IdealVerdict.model_validate(
        {"material": "addition", "checks": ["pair(e1, e2) registered"]}
    )

    failures = score_ideal(case, verdict)

    assert failures == []


def test_score_ideal_wrong_material_reports_wrong_replacement() -> None:
    """The sole prior ``score_ideal`` test fed
    an exact match, so replacing the scorer body with ``return []`` still passed it. A material
    mismatch alone (checks otherwise matching) must be reported under ``"wrong replacement"``."""
    case = _ideal_case()
    assert case.expect is not None
    verdict = IdealVerdict(material="correction", checks=case.expect.mutation.ideal)

    failures = score_ideal(case, verdict)

    assert [f.failure_class for f in failures] == ["wrong replacement"]


def test_score_ideal_missing_check_reports_its_own_failure_class() -> None:
    """A check the case expects but the verdict omits must be reported, under THAT check's own
    failure class (``validity_equals`` -> "live contradiction"), not silently dropped."""
    case = _ideal_case(
        expect={
            "material": "addition",
            "mutation": {"acceptable": [], "ideal": ["e1.validity == invalid"]},
        },
    )
    verdict = IdealVerdict(material="addition", checks=[])

    failures = score_ideal(case, verdict)

    assert len(failures) == 1
    assert failures[0].failure_class == "live contradiction"
    assert "e1.validity == invalid" in failures[0].detail


def test_score_ideal_unexpected_check_reports_its_own_failure_class() -> None:
    """A check the verdict returns but the case does not expect must be reported, under THAT
    check's own failure class (``history_says`` -> "lost history"), not silently ignored."""
    case = _ideal_case(expect={"material": "addition", "mutation": {"acceptable": [], "ideal": []}})
    verdict = IdealVerdict.model_validate(
        {"material": "addition", "checks": ["history says elaborated"]}
    )

    failures = score_ideal(case, verdict)

    assert len(failures) == 1
    assert failures[0].failure_class == "lost history"
    assert "history says elaborated" in failures[0].detail
