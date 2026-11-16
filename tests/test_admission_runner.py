"""The admission case-set runner — one pytest test per case per axis.

Ids look like ``test_admission[<case-id>][<axis>]``: the default lane collects ``material`` and
``mutation``; the ``-m
integration`` lane collects ``selection`` and ``ideal`` (live Ollama, and the case-set judge seat
for ``ideal``). ``CASES`` is empty until a later piece adds case rows — an empty parametrize set
is pytest's documented skip, not an error, so this module stays green with nothing to test.

Test-only switches, never set in a real run: ``TABULARIUM_ADMISSION_EMBEDDER=fake`` substitutes
``FakeEmbedder`` for ``FrozenEmbedder`` in the default lane (for a temporary case whose text was
never frozen — see ``tests/test_admission_runner_shape.py``); ``TABULARIUM_ADMISSION_CASES``
(honoured by ``tests.support.cases.load_cases``) points the loader at a different case directory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import DecisionLifecycle, FactLifecycle, HypomnemaId, Scope
from tabularium.embedding import Embedder, LitellmEmbedder
from tabularium.engine import Tabularium
from tabularium.engine.assessor import (
    AssessorCandidate,
    ContradictionVerdict,
    ReasonClass,
    Relation,
    Trace,
)
from tabularium.engine.clean import clean
from tabularium.engine.filters import read_span
from tabularium.engine.gate import probe_candidates
from tabularium.engine.judge import build_judge_adapter
from tabularium.engine.rubric import load_rubric
from tabularium.results import BounceResolution
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.types import CleanerState, StorageAdapter
from tests.support.cases import (
    MATERIAL_TO_REASON,
    RESERVED_EXISTING_ALIAS,
    Case,
    Check,
    Failure,
    FalsityAssessorPick,
    FalsityJudgeAnswer,
    HistorySays,
    InvalidationEquals,
    MaterialLabel,
    PairRegistered,
    RowsEqual,
    SupersededByEquals,
    TextUnchanged,
    ValidityEquals,
    load_cases,
)
from tests.support.doubles import FakeEmbedder, FrozenEmbedder, TestClock
from tests.support.fake_assessor import EchoAssessor
from tests.support.faults import FaultInjectingAdapter, as_adapter
from tests.support.ideal_judge import build_ideal_prompt, parse_ideal_response, score_ideal
from tests.support.seeding import ArchivedStartUnsupported, seed_case

Axis = Literal["material", "mutation", "selection", "ideal", "falsity"]

_DEFAULT_AXES: tuple[Axis, ...] = ("material", "mutation")
_INTEGRATION_AXES: tuple[Axis, ...] = ("selection", "ideal")

_EPOCH = datetime(2025, 1, 1, tzinfo=UTC)

_ALL_FACT_LIFECYCLES: frozenset[FactLifecycle] = frozenset(
    {"active", "hot", "ember", "archived", "shelved"}
)
_ALL_DECISION_LIFECYCLES: frozenset[DecisionLifecycle] = frozenset(
    {"in_force", "superseded", "revoked"}
)
_ALL_LIFECYCLES = _ALL_FACT_LIFECYCLES | _ALL_DECISION_LIFECYCLES
_ALL_VALIDITY: frozenset[Literal["valid", "invalid"]] = frozenset({"valid", "invalid"})

_EMBEDDER_ENV = "TABULARIUM_ADMISSION_EMBEDDER"
_DEFAULT_EMBEDDER_DIM = 768

#: Bound to the record ``engine.save(case.incoming...)`` just wrote — the incoming record has no
#: author-facing alias of its own in the D6 YAML shape, so a check naming it as a
#: ``superseded_by``/``pair(...)`` target (e.g. ``e1.superseded_by == incoming``) needs a name to
#: resolve against. ``Case``'s own loader validator rejects an ``existing[]`` row that reuses this
#: alias, so binding it here can never collide with a seeded row.
_INCOMING_ALIAS = RESERVED_EXISTING_ALIAS

#: The check grammar's ``history says`` word vocabulary spells the correction word differently
#: from ``domain.Fact.supersession_reason``'s own ``Literal`` ("corrected" vs "correction"); the
#: other three words already match verbatim.
_HISTORY_WORD_TO_DOMAIN_REASON: dict[str, str] = {
    "corrected": "correction",
    "changed": "changed",
    "elaborated": "elaborated",
    "moved": "moved",
}

#: The material axis's own expectation, per D7 — independently declared
#: here rather than imported from ``tests.support.cases.MATERIAL_TO_RELATION``/
#: ``MATERIAL_TO_REASON``: ``EchoAssessor``
#: itself echoes THAT mapping, so asserting against the identical dict object could never catch
#: a wrong entry in it — both sides would agree by construction. A second, hand-written table
#: means a typo or a swapped value in either one shows up as a real mismatch.
_EXPECTED_MATERIAL_VERDICT: dict[MaterialLabel, tuple[Relation, ReasonClass | None]] = {
    "repetition": ("restatement", None),
    "elaboration": ("replacement", "elaborated"),
    "addition": ("follow_up", None),
    "correction": ("replacement", "correction"),
    "changed state": ("replacement", "changed"),
    "unrelated": ("distinct", None),
}

CASES: list[Case] = load_cases()


def _admission_params() -> list[Any]:
    """One param per case per axis — a case carrying a ``falsity`` block scores ONLY the falsity
    axis (on the default lane, no ``-m integration`` marker); every other case scores
    material/mutation (default) plus selection/ideal (integration), unchanged."""
    params: list[Any] = []
    for case in CASES:
        if case.falsity is not None:
            continue
        for axis in _DEFAULT_AXES:
            params.append(pytest.param(case, axis, id=f"{case.id}][{axis}"))
    for case in CASES:
        if case.falsity is not None:
            params.append(pytest.param(case, "falsity", id=f"{case.id}][falsity"))
    for case in CASES:
        if case.falsity is not None:
            continue
        for axis in _INTEGRATION_AXES:
            params.append(
                pytest.param(case, axis, id=f"{case.id}][{axis}", marks=pytest.mark.integration)
            )
    return params


def _build_default_embedder() -> Embedder:
    if os.environ.get(_EMBEDDER_ENV) == "fake":
        return FakeEmbedder(dim=_DEFAULT_EMBEDDER_DIM)
    return FrozenEmbedder()


def _case_scopes(case: Case) -> set[str]:
    scopes = {row.scope for row in case.existing}
    scopes.add(case.incoming.scope)
    return scopes


def _supersedes_arg(
    case: Case, alias_to_id: dict[str, str]
) -> HypomnemaId | list[HypomnemaId] | None:
    """Translate ``case.incoming.supersedes`` (always a list of aliases in the D6 YAML shape)
    into the shape ``engine.save`` actually requires per arm: a LIST of ids on the Fact arm, a
    single bare id on the Decision arm (``core.py`` rejects a list there outright).
    """
    if not case.incoming.supersedes:
        return None
    ids = [HypomnemaId(uuid.UUID(alias_to_id[alias])) for alias in case.incoming.supersedes]
    if case.incoming.kind == "decision":
        if len(ids) != 1:
            raise AssertionError(
                f"case {case.id!r}: a Decision incoming save must name exactly one "
                f"supersedes alias, got {case.incoming.supersedes!r}"
            )
        return ids[0]
    return ids


async def _register_case_scopes(adapter: StorageAdapter, case: Case) -> None:
    for scope in _case_scopes(case):
        await adapter.create_scope(Scope(scope))


def _has_unsupported_archived_start(case: Case) -> bool:
    return any(row.lifecycle == "archived" for row in case.existing)


async def _row_count(adapter: StorageAdapter, case: Case) -> int:
    """Count every distinct record across every scope the case touches.

    A read span is ``[scope, global]`` only (SPEC), so counting from a single scope misses a
    predecessor or successor written to a DIFFERENT named scope — exactly the cross-tier
    duplicate the ``rows == N`` check exists to catch. Listing every scope the case names and
    deduplicating by id (scopes whose read span overlaps at ``global`` would otherwise
    double-count) gives the true row count for the whole case.
    """
    seen_ids: set[str] = set()
    for scope in _case_scopes(case):
        span = read_span(Scope(scope))
        seen_ids.update(
            [
                str(record.id)
                async for record in await adapter.scan_records(lifecycle_in=_ALL_LIFECYCLES)
                if record.scope in span and record.validity in _ALL_VALIDITY
            ]
        )
    return len(seen_ids)


async def _evaluate_check(
    check: Check,
    *,
    adapter: StorageAdapter,
    case: Case,
    alias_to_id: dict[str, str],
) -> Failure | None:
    """Run one parsed ``Check`` against the live store and return a ``Failure`` on mismatch.

    ``alias_to_id`` carries every existing row's alias PLUS the reserved ``"incoming"`` alias
    (``_INCOMING_ALIAS``, bound by the caller once the incoming save completes) — the one alias
    table every check resolves against, so an unresolved alias is a named ``Failure``, never a
    silent ``None == None`` false pass.

    Every check reads through the storage adapter (``scan_records``/``get_by_id`` and the
    distinct-pair registry): pure reads with no side effects.
    """
    if isinstance(check, RowsEqual):
        actual = await _row_count(adapter, case)
        if actual != check.n:
            return Failure(
                failure_class=check.failure_class, detail=f"rows == {check.n}: got {actual}"
            )
        return None

    if isinstance(check, ValidityEquals):
        record_id = alias_to_id.get(check.alias)
        if record_id is None:
            return Failure(
                failure_class=check.failure_class,
                detail=f"{check.alias}.validity == {check.validity}: unknown alias {check.alias!r}",
            )
        hit = await adapter.get_by_id(HypomnemaId(uuid.UUID(record_id)))
        if hit.validity != check.validity:
            return Failure(
                failure_class=check.failure_class,
                detail=f"{check.alias}.validity == {check.validity}: got {hit.validity}",
            )
        return None

    if isinstance(check, SupersededByEquals):
        record_id = alias_to_id.get(check.alias)
        target_id = alias_to_id.get(check.target_alias)
        if record_id is None or target_id is None:
            unknown = check.alias if record_id is None else check.target_alias
            return Failure(
                failure_class=check.failure_class,
                detail=(
                    f"{check.alias}.superseded_by == {check.target_alias}: "
                    f"unknown alias {unknown!r}"
                ),
            )
        hit = await adapter.get_by_id(HypomnemaId(uuid.UUID(record_id)))
        actual_target = str(hit.superseded_by) if hit.superseded_by is not None else None
        if actual_target != target_id:
            return Failure(
                failure_class=check.failure_class,
                detail=(
                    f"{check.alias}.superseded_by == {check.target_alias}: got {actual_target}"
                ),
            )
        return None

    if isinstance(check, TextUnchanged):
        row = next((r for r in case.existing if r.alias == check.alias), None)
        record_id = alias_to_id.get(check.alias)
        if row is None or record_id is None:
            return Failure(
                failure_class=check.failure_class,
                detail=f"{check.alias}.text unchanged: unknown alias {check.alias!r}",
            )
        hit = await adapter.get_by_id(HypomnemaId(uuid.UUID(record_id)))
        if hit.content != row.text:
            return Failure(
                failure_class=check.failure_class, detail=f"{check.alias}.text unchanged: differs"
            )
        return None

    if isinstance(check, HistorySays):
        predecessor_aliases = case.incoming.supersedes
        if not predecessor_aliases:
            return Failure(
                failure_class=check.failure_class,
                detail=f"history says {check.word}: incoming names no predecessor",
            )
        expected_reason = _HISTORY_WORD_TO_DOMAIN_REASON[check.word]
        mismatched: list[str] = []
        for alias in predecessor_aliases:
            predecessor_id = alias_to_id.get(alias)
            if predecessor_id is None:
                mismatched.append(f"{alias} (unknown alias)")
                continue
            record = await adapter.get_by_id(HypomnemaId(uuid.UUID(predecessor_id)))
            actual_reason = getattr(record, "supersession_reason", None)
            if actual_reason != expected_reason:
                mismatched.append(f"{alias} (got {actual_reason!r})")
        if mismatched:
            return Failure(
                failure_class=check.failure_class,
                detail=f"history says {check.word}: {', '.join(mismatched)}",
            )
        return None

    if isinstance(check, PairRegistered):
        id_a = alias_to_id.get(check.alias_a)
        id_b = alias_to_id.get(check.alias_b)
        if id_a is None or id_b is None:
            return Failure(
                failure_class=check.failure_class,
                detail=f"pair({check.alias_a},{check.alias_b}) registered: unknown alias",
            )
        registered = await adapter.is_distinct(
            HypomnemaId(uuid.UUID(id_a)), HypomnemaId(uuid.UUID(id_b))
        )
        if registered != check.expected:
            return Failure(
                failure_class=check.failure_class,
                detail=(
                    f"pair({check.alias_a},{check.alias_b}) "
                    f"{'registered' if check.expected else 'not registered'}: got {registered}"
                ),
            )
        return None

    if isinstance(check, InvalidationEquals):
        record_id = alias_to_id.get(check.alias)
        by_id = alias_to_id.get(check.by_alias)
        if record_id is None or by_id is None:
            unknown = check.alias if record_id is None else check.by_alias
            return Failure(
                failure_class=check.failure_class,
                detail=(
                    f"{check.alias}.invalidated_by == {check.by_alias} reason "
                    f"{check.reason}: unknown alias {unknown!r}"
                ),
            )
        record = await adapter.get_by_id(HypomnemaId(uuid.UUID(record_id)))
        actual_by = getattr(record, "invalidated_by", None)
        actual_reason = getattr(record, "invalidation_reason", None)
        if actual_by != by_id or actual_reason != check.reason:
            return Failure(
                failure_class=check.failure_class,
                detail=(
                    f"{check.alias}.invalidated_by == {check.by_alias} reason "
                    f"{check.reason}: got invalidated_by={actual_by!r} "
                    f"invalidation_reason={actual_reason!r}"
                ),
            )
        return None

    raise AssertionError(f"unreachable: unhandled check variant {check!r}")


async def _run_material(case: Case, request: pytest.FixtureRequest) -> None:
    request.node.user_properties.append(("admission_axis", "material"))
    if _has_unsupported_archived_start(case):
        pytest.skip("no production transition to archived yet")
    assert case.expect is not None, f"case {case.id!r}: material axis needs an expect block"

    assessor = EchoAssessor(case.expect.material)
    dummy_candidate = AssessorCandidate(
        id="dummy", content="dummy", kind="fact", scope="global", created_at=_EPOCH
    )
    verdict = await assessor.relation(case.incoming.content, dummy_candidate, similarity=None)

    expected_relation, expected_reason = _EXPECTED_MATERIAL_VERDICT[case.expect.material]
    assert verdict.pick == expected_relation
    assert verdict.reason_class == expected_reason


async def _run_mutation(
    case: Case, tmp_path: Path, request: pytest.FixtureRequest, caplog: pytest.LogCaptureFixture
) -> None:
    request.node.user_properties.append(("admission_axis", "mutation"))
    if _has_unsupported_archived_start(case):
        pytest.skip("no production transition to archived yet")
    assert case.expect is not None, f"case {case.id!r}: mutation axis needs an expect block"

    embedder = _build_default_embedder()
    clock = TestClock(_EPOCH)
    raw_adapter = await LanceDBAdapter.open(
        str(tmp_path / "admission_store"),
        embedding_model=embedder.model,
        dim=embedder.dim,
        clock=clock,
    )

    # A case naming `fault:` wraps the real adapter so the incoming save meets the configured
    # fault; seeding always runs over the SAME wrapper unarmed, so setup
    # writes never consume it (`arm()` only after seeding, below).
    fault_wrapper: FaultInjectingAdapter | None = None
    engine_adapter: StorageAdapter = raw_adapter
    if case.fault is not None:
        fault_wrapper = FaultInjectingAdapter(raw_adapter)
        engine_adapter = as_adapter(fault_wrapper)

    engine = Tabularium(
        engine_adapter,
        embedder,
        clock,
        EngineConfig(
            auto_exit_restatement=True,
            auto_exit_replacement=True,
            auto_exit_follow_up=True,
            auto_exit_distinct=True,
        ),
        assessor=EchoAssessor(case.expect.material),
    )
    await engine.initialize()
    await _register_case_scopes(raw_adapter, case)

    try:
        alias_to_id = await seed_case(engine, clock, case)
    except ArchivedStartUnsupported:
        pytest.skip("no production transition to archived yet")

    supersedes_arg = _supersedes_arg(case, alias_to_id)

    # Seeding above logs its own ``admission_resolved`` lines (one per seeded
    # row) — the harvest below is scoped to the incoming save alone, so it
    # is cleared and captured only from here (P7 admission-order piece,
    # decision D10).
    caplog.clear()
    incoming_id: HypomnemaId | None
    with caplog.at_level(logging.DEBUG, logger="tabularium"):
        if fault_wrapper is not None:
            assert case.fault is not None
            configured_exc = RuntimeError(f"admission case {case.id!r} fault: {case.fault.on}")
            fault_wrapper.fault(on=case.fault.on, call=case.fault.call, exc=configured_exc)
            fault_wrapper.arm()
            with pytest.raises(Exception) as excinfo:
                await engine.save(
                    case.incoming.content,
                    scope=case.incoming.scope,
                    kind=case.incoming.kind,
                    supersedes=supersedes_arg,
                    reason=case.incoming.reason,
                )
            if case.fault.on == "replace":
                # ``_write_fact``/``_write_decision`` catch the injected
                # exception, log ``supersession_failed``, and re-raise as
                # ``ToolError`` (P3 one-transaction-supersession piece) —
                # the chained ``__cause__`` is the identity check here,
                # never the raw raised object.
                assert excinfo.value.__cause__ is configured_exc, (
                    f"case {case.id!r}: engine.save raised {excinfo.value!r} whose "
                    f"__cause__ is not the configured fault on {case.fault.on!r} call "
                    f"{case.fault.call} -- an unrelated failure must never be mistaken "
                    "for the injected one"
                )
            else:
                assert excinfo.value is configured_exc, (
                    f"case {case.id!r}: engine.save raised {excinfo.value!r} before "
                    f"reaching the configured fault on {case.fault.on!r} call "
                    f"{case.fault.call} -- an unrelated failure must never be mistaken "
                    "for the injected one"
                )
            incoming_id = None
        else:
            incoming_result = await engine.save(
                case.incoming.content,
                scope=case.incoming.scope,
                kind=case.incoming.kind,
                supersedes=supersedes_arg,
                reason=case.incoming.reason,
            )
            assert incoming_result.embedded is True, (
                f"case {case.id!r}: incoming.content landed without a vector "
                "(embedder_unavailable) -- its text was never frozen into "
                "tests/fixtures/frozen_vectors.json; run scripts/freeze_vectors.py "
                "and try again rather than trusting this pass"
            )
            incoming_id = (
                incoming_result.hypomnema.id if incoming_result.hypomnema is not None else None
            )

    resolved_lines = [
        r for r in caplog.records if r.getMessage().startswith("admission_resolved")
    ]
    assert len(resolved_lines) == 1, (
        f"case {case.id!r}: expected exactly one admission_resolved line for the "
        f"incoming save, got {len(resolved_lines)}: {[r.getMessage() for r in resolved_lines]}"
    )
    if case.expect.step is not None:
        actual_step = resolved_lines[0].step  # type: ignore[attr-defined]
        assert actual_step == case.expect.step, (
            f"case {case.id!r}: expected admission_resolved step={case.expect.step!r}, "
            f"got {actual_step!r}"
        )

    failed_lines = [
        r for r in caplog.records if r.getMessage().startswith("supersession_failed")
    ]
    if case.fault is not None and case.fault.on == "replace":
        assert len(failed_lines) == 1, (
            f"case {case.id!r}: expected exactly one supersession_failed line, got "
            f"{len(failed_lines)}: {[r.getMessage() for r in failed_lines]}"
        )
        failed = failed_lines[0]
        expected_predecessor_ids = [alias_to_id[alias] for alias in case.incoming.supersedes]
        # A purely automatic invalidation (no caller-level supersedes at all — P8
        # assessor-automatic-exits piece) has no case.incoming.reason to compare
        # against; the logged reason falls back to the Assessor's own verdict, which
        # the case's material label already determines (tests.support.cases.
        # MATERIAL_TO_REASON — the SAME mapping EchoAssessor itself echoes).
        expected_reason = (
            case.incoming.reason
            if case.incoming.supersedes
            else MATERIAL_TO_REASON[case.expect.material]
        )
        assert failed.successor_id, case.id  # type: ignore[attr-defined]
        # Subset, not equality (P8 assessor-automatic-exits piece, composition):
        # a composed fault's logged predecessor_ids is the caller's own list PLUS
        # the Assessor-found candidate, an alias no case declares up front —
        # every caller-named id must still appear, but the log may legitimately
        # carry more than the case's own supersedes list names.
        actual_predecessor_ids = failed.predecessor_ids  # type: ignore[attr-defined]
        assert set(expected_predecessor_ids) <= set(actual_predecessor_ids), case.id
        assert failed.reason == expected_reason, case.id  # type: ignore[attr-defined]
        assert failed.scope == case.incoming.scope, case.id  # type: ignore[attr-defined]
    else:
        assert not failed_lines, (
            f"case {case.id!r}: expected no supersession_failed line, got "
            f"{[r.getMessage() for r in failed_lines]}"
        )

    eval_alias_to_id = dict(alias_to_id)
    if incoming_id is not None:
        eval_alias_to_id[_INCOMING_ALIAS] = str(incoming_id)

    failures: list[Failure] = []
    for check in case.expect.mutation.acceptable:
        failure = await _evaluate_check(
            check,
            adapter=raw_adapter,
            case=case,
            alias_to_id=eval_alias_to_id,
        )
        if failure is not None:
            failures.append(failure)

    for failure in failures:
        request.node.user_properties.append(("failure_class", failure.failure_class))
    assert not failures, "; ".join(failure.detail for failure in failures)


async def _run_selection(case: Case, tmp_path: Path, request: pytest.FixtureRequest) -> None:
    """Score the SAME candidate probe the dedup gate itself runs (``engine.gate.probe_candidates``)
    — never ``engine.recall``, which is a materially different feature: different filters
    (recall excludes shelved Facts and Decisions the gate probes; the gate probes both
    validities where recall defaults to valid-only) and a different ranking (recall's RRF
    times importance vs the gate's raw vector cosine). Scoring recall here would pass or fail
    candidates admission itself never sees.
    """
    request.node.user_properties.append(("admission_axis", "selection"))
    if _has_unsupported_archived_start(case):
        pytest.skip("no production transition to archived yet")
    assert case.expect is not None, f"case {case.id!r}: selection axis needs an expect block"
    if case.expect.selection is None:
        pytest.skip(f"case {case.id!r} does not score the selection axis")

    embedder = LitellmEmbedder()
    clock = TestClock(_EPOCH)
    adapter = await LanceDBAdapter.open(
        str(tmp_path / "admission_store"),
        embedding_model=embedder.model,
        dim=embedder.dim,
        clock=clock,
    )
    cfg = EngineConfig()
    engine = Tabularium(adapter, embedder, clock, cfg)
    await engine.initialize()
    await _register_case_scopes(adapter, case)

    try:
        alias_to_id = await seed_case(engine, clock, case)
    except ArchivedStartUnsupported:
        pytest.skip("no production transition to archived yet")

    selection = case.expect.selection
    vector = await embedder.embed(case.incoming.content)
    probe = await probe_candidates(
        content=case.incoming.content,
        vector=vector,
        span=read_span(Scope(case.incoming.scope)),
        k=selection.k,
        adapter=adapter,
        cfg=cfg,
    )
    expected_ids = {alias_to_id[alias] for alias in selection.aliases}
    surfaced_ids = {str(candidate_id) for candidate_id, _ in probe.fused_ids}
    missing = expected_ids - surfaced_ids
    assert not missing, f"selection missed aliases resolving to ids: {missing}"


async def _run_ideal(case: Case, request: pytest.FixtureRequest) -> None:
    request.node.user_properties.append(("admission_axis", "ideal"))
    if _has_unsupported_archived_start(case):
        pytest.skip("no production transition to archived yet")
    assert case.expect is not None, f"case {case.id!r}: ideal axis needs an expect block"

    prompt = build_ideal_prompt(case)
    rubric = load_rubric()
    assert rubric.caseset is not None, "packaged rubric artifact is missing [caseset]"
    judge = build_judge_adapter(rubric.caseset.judge)
    response = await judge.call(prompt)
    verdict = parse_ideal_response(response)
    failures = score_ideal(case, verdict)

    for failure in failures:
        request.node.user_properties.append(("failure_class", failure.failure_class))
    assert not failures, "; ".join(failure.detail for failure in failures)


# ---------------------------------------------------------------------------
# Falsity axis (A3 cleaner-integration piece, scope 5) — every falsity case shares ONE store and
# ONE scope, seeded once through production ``save``, then scored from a SINGLE ``clean(scope)``
# pass's ``CleanSummary.falsity``. The shared run is expensive (a real ``LanceDBAdapter`` plus one
# ``save`` per case's pair), so it executes once and every parametrized falsity test reads the
# cached, already-plain result — never the live adapter, which would tie a later test to whatever
# event loop pytest-asyncio happened to give the FIRST one.
# ---------------------------------------------------------------------------

_FALSITY_SCOPE = "falsity-admission"
_FALSITY_OLDER_AT = datetime(2024, 1, 1, tzinfo=UTC)
_FALSITY_NEWER_AT = datetime(2024, 6, 1, tzinfo=UTC)
_FALSITY_WATERMARK = datetime(2024, 3, 1, tzinfo=UTC)

_ITEM_HEADING_RE = re.compile(r"^## (\d+)\.$", re.MULTILINE)
_OLDER_BLOCK_RE = re.compile(r"O \(older, created [^)]*\):\n(.*)", re.DOTALL)
_FALSITY_MARKER = "# Cleaner — falsity rubric"

_FALSITY_RUN: dict[str, Any] = {}


async def _require_scope_always_ok(scope: Scope) -> None:
    del scope


class _KeyedFalsityJudge:
    """One judge seat over the WHOLE shared pass — every falsity case's pair lands in the same
    scope, so a batch may carry more than one case's pair at once. Each item is resolved by the
    OLDER record's own text (unique per case by construction, per the case-authoring rule: unrelated
    subjects, never two cases sharing a subject) rather than by position. A batch with no falsity
    marker is the RELEVANCE axis's own call on the same newcomers (``clean()`` runs it first, over
    the identical shared pool) — answered ``keep-clear`` unconditionally so no falsity fixture is
    ever shelved before the falsity axis gets to see it.
    """

    def __init__(self, identity: str, answers: dict[str, FalsityJudgeAnswer]) -> None:
        self._identity = identity
        self._answers = answers
        self.calls: list[str] = []

    @property
    def identity(self) -> str:
        return self._identity

    def instruction_fingerprint(self) -> str:
        return "not_applicable"

    async def call(self, prompt: str) -> str:
        self.calls.append(prompt)
        reply_format_at = prompt.find("# Reply format")
        batch_text = prompt if reply_format_at == -1 else prompt[:reply_format_at]
        headings = list(_ITEM_HEADING_RE.finditer(batch_text))
        if _FALSITY_MARKER not in prompt:
            return json.dumps(
                {
                    "verdicts": [
                        {"index": i, "grade": "keep-clear", "unclear": False, "why": "not falsity"}
                        for i in range(1, len(headings) + 1)
                    ]
                }
            )
        items: list[dict[str, Any]] = []
        for idx, match in enumerate(headings):
            start = match.end()
            end = headings[idx + 1].start() if idx + 1 < len(headings) else len(batch_text)
            body = batch_text[start:end].strip()
            older_match = _OLDER_BLOCK_RE.search(body)
            assert older_match is not None, (
                f"falsity batch item {idx + 1}: no older-record block found: {body!r}"
            )
            older_text = older_match.group(1).strip()
            answer = self._answers.get(older_text)
            assert answer is not None, (
                f"falsity batch item {idx + 1}: no scripted answer for older text {older_text!r}"
            )
            item: dict[str, Any] = {"index": idx + 1, "verdict": answer.verdict, "why": answer.why}
            if answer.reason is not None:
                item["reason"] = answer.reason
            items.append(item)
        return json.dumps({"verdicts": items})


def _contradiction_verdict(pick: FalsityAssessorPick) -> ContradictionVerdict:
    reason_class = pick if pick in ("correction", "changed") else None
    domain_pick: Literal["coexist", "replace"] = "coexist" if pick == "coexist" else "replace"
    return ContradictionVerdict(
        pick=domain_pick,
        reason_class=reason_class,
        odds={pick: 1.0},
        confidence=1.0,
        trace=Trace(
            model="echo", wording_hash="echo", latency_ms=0,
            input_tokens=None, output_tokens=None, request_id=None,
        ),
    )


async def _save_resolving_bounce(
    engine: Tabularium, *, content: str, scope: str, kind: Literal["fact", "decision"]
) -> HypomnemaId:
    """Production ``save``, replaying a bounce's own ``confirm_distinct`` ``how`` (Constraints:
    "A save that bounces is resolved by replaying its confirm_distinct how") — the case author
    chose every falsity pair as neighbours on purpose, so a bounce here is never a setup failure."""
    result = await engine.save(content, scope=scope, kind=kind)
    if result.outcome == "bounced":
        assert result.bounce is not None
        result = await engine.save(
            content, scope=scope, kind=kind,
            resolve=BounceResolution(
                candidate_id=result.bounce.candidate_id, action="confirm_distinct"
            ),
        )
    assert result.hypomnema is not None
    assert result.embedded is True, (
        "falsity case content landed without a vector (embedder_unavailable) -- its text was "
        "never frozen into tests/fixtures/frozen_vectors.json; run scripts/freeze_vectors.py"
    )
    return result.hypomnema.id


async def _get_falsity_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    if _FALSITY_RUN:
        return _FALSITY_RUN

    falsity_cases = [case for case in CASES if case.falsity is not None]
    embedder = _build_default_embedder()
    clock = TestClock(_FALSITY_OLDER_AT)
    store_dir = tmp_path_factory.mktemp("falsity_admission_store")
    raw_adapter = await LanceDBAdapter.open(
        str(store_dir), embedding_model=embedder.model, dim=embedder.dim, clock=clock
    )
    engine = Tabularium(
        raw_adapter,
        embedder,
        clock,
        EngineConfig(
            auto_exit_restatement=True,
            auto_exit_replacement=True,
            auto_exit_follow_up=True,
            auto_exit_distinct=True,
        ),
        assessor=EchoAssessor("unrelated"),
    )
    await engine.initialize()
    scopes = {_FALSITY_SCOPE}
    for case in falsity_cases:
        scopes |= _case_scopes(case)
    for scope in scopes:
        await raw_adapter.create_scope(Scope(scope))

    alias_to_id: dict[str, dict[str, str]] = {}
    for case in falsity_cases:
        older_row = case.existing[0]
        clock.set(older_row.created_at)
        older_id = await _save_resolving_bounce(
            engine, content=older_row.text, scope=older_row.scope, kind=older_row.kind
        )
        clock.set(_FALSITY_NEWER_AT)
        newer_id = await _save_resolving_bounce(
            engine,
            content=case.incoming.content,
            scope=case.incoming.scope,
            kind=case.incoming.kind,
        )
        alias_to_id[case.id] = {"older": str(older_id), "newer": str(newer_id)}

    rubric = load_rubric()
    cfg = EngineConfig()
    for case in falsity_cases:
        newer_id_str = alias_to_id[case.id]["newer"]
        older_id_str = alias_to_id[case.id]["older"]
        vector = await embedder.embed(case.incoming.content)
        probe = await probe_candidates(
            content=case.incoming.content,
            vector=vector,
            span=read_span(Scope(case.incoming.scope)),
            k=rubric.parameters.k,
            exclude={HypomnemaId(uuid.UUID(newer_id_str))},
            validity=frozenset({"valid"}),
            adapter=raw_adapter,
            cfg=cfg,
        )
        actual_best = str(probe.best_hit.record.id) if probe.best_hit is not None else None
        assert actual_best == older_id_str, (
            f"case {case.id!r}: its own pair was not generated -- the backward probe's best hit "
            f"was {actual_best}, not this case's own older record {older_id_str}; fix the "
            "fixture text, never the scoring"
        )

    judge_a_answers: dict[str, FalsityJudgeAnswer] = {}
    judge_b_answers: dict[str, FalsityJudgeAnswer] = {}
    contradiction_script: dict[str, ContradictionVerdict | BaseException] = {}
    for case in falsity_cases:
        assert case.falsity is not None
        older_text = case.existing[0].text
        judge_a_answers[older_text] = case.falsity.judge_a
        judge_b_answers[older_text] = case.falsity.judge_b
        if case.falsity.assessor is None:
            continue
        if case.falsity.assessor == "unavailable":
            contradiction_script[older_text] = TimeoutError(f"{case.id} assessor timeout")
        else:
            contradiction_script[older_text] = _contradiction_verdict(case.falsity.assessor)

    judge_a = _KeyedFalsityJudge("a", judge_a_answers)
    judge_b = _KeyedFalsityJudge("b", judge_b_answers)
    falsity_assessor = EchoAssessor("unrelated", contradiction_script=contradiction_script)

    await raw_adapter.set_cleaner_state(
        scope=Scope(_FALSITY_SCOPE),
        state=CleanerState(
            watermark_at=_FALSITY_WATERMARK, last_pass_at=_FALSITY_WATERMARK, report=None
        ),
    )
    clock.set(_FALSITY_NEWER_AT)
    summary = await clean(
        scope=_FALSITY_SCOPE,
        adapter=raw_adapter,
        embedder=embedder,
        clock=clock,
        lock=asyncio.Lock(),
        cfg=EngineConfig(),
        require_scope=_require_scope_always_ok,
        rubric=rubric,
        judge_a=judge_a,
        judge_b=judge_b,
        assessor=falsity_assessor,
    )
    assert summary.falsity is not None, f"falsity axis did not run: outcome={summary.outcome!r}"

    reason_by_older_id: dict[str, str | None] = {}
    for case in falsity_cases:
        older_id_str = alias_to_id[case.id]["older"]
        record = await raw_adapter.get_by_id(HypomnemaId(uuid.UUID(older_id_str)))
        reason_by_older_id[older_id_str] = getattr(record, "supersession_reason", None)

    _FALSITY_RUN["summary"] = summary
    _FALSITY_RUN["alias_to_id"] = alias_to_id
    _FALSITY_RUN["reason_by_older_id"] = reason_by_older_id
    return _FALSITY_RUN


async def _run_falsity(
    case: Case, tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> None:
    request.node.user_properties.append(("admission_axis", "falsity"))
    assert case.falsity is not None

    run = await _get_falsity_run(tmp_path_factory)
    summary = run["summary"]
    ids = run["alias_to_id"][case.id]
    older_id = HypomnemaId(uuid.UUID(ids["older"]))
    newer_id = HypomnemaId(uuid.UUID(ids["newer"]))

    edge = next(
        (e for e in summary.falsity.edges if e.id == older_id and e.superseded_by == newer_id),
        None,
    )
    contested = any(
        p.candidate_id == older_id and p.newcomer_id == newer_id
        for p in summary.falsity.contested_pairs
    )
    global_reported = any(
        g.candidate_id == older_id for g in summary.falsity.global_candidates
    )

    fb = case.falsity
    failures: list[str] = []
    if fb.expect == "coexist":
        if edge is not None:
            failures.append("expected coexist, but an edge was drawn")
        if contested:
            failures.append("expected coexist, but the pair is contested")
        if global_reported:
            failures.append("expected coexist, but the pair is global_reported")
    elif fb.expect == "contested":
        if edge is not None:
            failures.append("expected contested, but an edge was drawn")
        if not contested:
            failures.append("expected contested, but the pair is missing from contested_pairs")
    elif fb.expect == "global_reported":
        if edge is not None:
            failures.append("expected global_reported, but an edge was drawn")
        if not global_reported:
            failures.append("expected global_reported, but missing from global_candidates")
    else:
        expected_reason = fb.expect.split(":", 1)[1]
        if edge is None:
            failures.append(f"expected supersede:{expected_reason}, but no edge was drawn")
        else:
            actual_reason = run["reason_by_older_id"].get(ids["older"])
            if actual_reason != expected_reason:
                failures.append(
                    f"expected supersede reason={expected_reason!r}, got {actual_reason!r}"
                )

    assert not failures, f"case {case.id!r}: " + "; ".join(failures)


@pytest.mark.parametrize(("case", "axis"), _admission_params())
async def test_admission(
    case: Case,
    axis: Axis,
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
    caplog: pytest.LogCaptureFixture,
) -> None:
    if axis == "material":
        await _run_material(case, request)
    elif axis == "mutation":
        await _run_mutation(case, tmp_path, request, caplog)
    elif axis == "selection":
        await _run_selection(case, tmp_path, request)
    elif axis == "falsity":
        await _run_falsity(case, tmp_path_factory, request)
    else:
        await _run_ideal(case, request)
