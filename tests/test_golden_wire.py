"""Golden wire tests: byte-for-byte pins of every frozen surface.

Each golden file under ``tests/golden/`` is written once and never regenerated, except that
``tools.json`` and ``judge_argv.json`` were rewritten on 2026-11-03 when the eleven write tools left
the MCP surface. A mismatch fails with a unified diff. UUIDs and ISO timestamps are normalized to
``<uuid-N>`` and ``<ts-N>`` in order of first appearance before comparing.
"""

from __future__ import annotations

import difflib
import json
import math
import re
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pytest

import tabularium.engine.assessor as assessor
import tabularium.engine.clean as clean
import tabularium.engine.judge as judge
import tabularium.engine.report as report
from tabularium.config import EngineConfig
from tabularium.domain import HypomnemaId, Scope
from tabularium.engine import Tabularium
from tabularium.engine.axis_common import (
    BandCounts,
    BatchFailure,
    ContestedVerdict,
    RecordVerdict,
)
from tabularium.engine.cleaner_writes import TrimmedPair
from tabularium.engine.falsity import (
    FalsityAxisSummary,
    FalsityBandCounts,
    FalsityBatchFailure,
    FalsityContestedPair,
    FalsityEdge,
    FalsityReportEntry,
    FalsityVerdict,
)
from tabularium.engine.judge_policy import McpGrant
from tabularium.engine.relevance import RelevanceAxisSummary, RelevanceVerdict
from tabularium.engine.revisit import RevisitAxisSummary, RevisitVerdict
from tabularium.server import make_app
from tabularium.storage import lancedb_codec
from tabularium.storage.memory import MemoryAdapter
from tests.support.doubles import FakeEmbedder, FakeTokenCounter, TestClock
from tests.support.fake_assessor import EchoAssessor
from tests.support.faults import FaultInjectingEmbedder, as_embedder

GOLDEN_DIR = Path(__file__).parent / "golden"

Outcome = Literal["initialized", "empty_pool", "completed", "blocked"]
Cli = Literal["claude", "codex"]
Kind = Literal["fact", "decision"]

_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")

_DIM = 16
_EPOCH = datetime(2025, 1, 1, tzinfo=UTC)
_PASS_AT = datetime(2025, 1, 10, 12, 0, 0, tzinfo=UTC)


def normalize(text: str) -> str:
    uuids: dict[str, str] = {}
    stamps: dict[str, str] = {}

    def replace_uuid(match: re.Match[str]) -> str:
        key = match.group(0).lower()
        return uuids.setdefault(key, f"<uuid-{len(uuids) + 1}>")

    def replace_stamp(match: re.Match[str]) -> str:
        return stamps.setdefault(match.group(0), f"<ts-{len(stamps) + 1}>")

    return _TIMESTAMP.sub(replace_stamp, _UUID.sub(replace_uuid, text))


def dump_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def assert_golden(name: str, actual: str) -> None:
    path = GOLDEN_DIR / name
    expected = path.read_text(encoding="utf-8")
    normalized = normalize(actual)
    if normalized == expected:
        return
    diff = difflib.unified_diff(
        expected.splitlines(keepends=True),
        normalized.splitlines(keepends=True),
        fromfile=f"golden/{name}",
        tofile="actual",
    )
    pytest.fail("golden mismatch\n" + "".join(diff), pytrace=False)


def fixed_id(number: int) -> HypomnemaId:
    return HypomnemaId(uuid.UUID(int=number))


# ---------------------------------------------------------------------------
# tools.json
# ---------------------------------------------------------------------------


async def build_tools() -> str:
    adapter = MemoryAdapter(embedding_model="fake/nomic", dim=_DIM)
    embedder = FakeEmbedder(dim=_DIM)
    clock = TestClock(_EPOCH)
    cfg = EngineConfig(candidate_k=1000, gray_floor=0.80)
    engine = Tabularium(adapter, embedder, clock, cfg, token_counter=FakeTokenCounter())
    await engine.initialize()
    app = make_app(engine)
    tools = sorted(await app.list_tools(), key=lambda tool: tool.name)
    return dump_json(
        {
            "instructions": app.instructions,
            "tools": [tool.model_dump(mode="json", by_alias=True) for tool in tools],
        }
    )


async def test_tools_golden() -> None:
    assert_golden("tools.json", await build_tools())


# ---------------------------------------------------------------------------
# bounces.json
# ---------------------------------------------------------------------------


def _angle_vec(angle_deg: float) -> list[float]:
    radians = math.radians(angle_deg)
    return [math.cos(radians), math.sin(radians)] + [0.0] * (_DIM - 2)


class _TableEmbedder:
    dim: int = _DIM
    model: str = "test/golden"

    def __init__(self, table: Mapping[str, list[float]]) -> None:
        self._table = table

    async def embed(self, text: str) -> list[float]:
        return list(self._table[text])


_EXISTING = "the quick brown fox jumps over the lazy dog"
_INCOMING = "the quick brown fox jumps over the lazy hound"
_OUTAGE_EXISTING = "alpha bravo charlie delta echo"
_OUTAGE_INCOMING = "alpha bravo charlie delta echo foxtrot"

_TABLE: dict[str, list[float]] = {
    _EXISTING: _angle_vec(0),
    _INCOMING: _angle_vec(5),
}


async def _table_engine(
    *, assessor_double: EchoAssessor | None = None
) -> tuple[Tabularium, MemoryAdapter]:
    embedder = _TableEmbedder(_TABLE)
    clock = TestClock(_EPOCH)
    adapter = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim, clock=clock)
    cfg = EngineConfig(candidate_k=1000, gray_floor=0.80, decision_gray_floor=0.70)
    engine = Tabularium(adapter, embedder, clock, cfg, assessor=assessor_double)
    await engine.initialize()
    await adapter.create_scope(Scope("p"))
    await adapter.create_scope(Scope("q"))
    return engine, adapter


async def _outage_engine(
    *, assessor_double: EchoAssessor | None = None
) -> tuple[Tabularium, FaultInjectingEmbedder]:
    inner = FakeEmbedder(dim=_DIM)
    wrapper = FaultInjectingEmbedder(inner)
    clock = TestClock(_EPOCH)
    adapter = MemoryAdapter(embedding_model=inner.model, dim=inner.dim, clock=clock)
    cfg = EngineConfig(candidate_k=1000, gray_floor=0.80)
    engine = Tabularium(adapter, as_embedder(wrapper), clock, cfg, assessor=assessor_double)
    await engine.initialize()
    await adapter.create_scope(Scope("p"))
    return engine, wrapper


def _dump_bounce(result: Any) -> dict[str, Any]:
    assert result.outcome == "bounced"
    dumped: dict[str, Any] = result.model_dump(mode="json")
    return dumped


async def _fact_three_exits() -> dict[str, Any]:
    engine, _ = await _table_engine()
    await engine.save(_EXISTING, scope="p")
    return _dump_bounce(await engine.save(_INCOMING, scope="p"))


async def _fact_two_exits() -> dict[str, Any]:
    engine, _ = await _table_engine()
    await engine.save(_EXISTING, scope="p", kind="decision")
    return _dump_bounce(await engine.save(_INCOMING, scope="p"))


async def _fact_switch_off_judged() -> dict[str, Any]:
    engine, _ = await _table_engine(assessor_double=EchoAssessor("repetition"))
    await engine.save(_EXISTING, scope="p")
    return _dump_bounce(await engine.save(_INCOMING, scope="p"))


async def _decision_three_exits() -> dict[str, Any]:
    engine, _ = await _table_engine()
    await engine.save(_EXISTING, scope="p", kind="decision")
    return _dump_bounce(await engine.save(_INCOMING, scope="p", kind="decision"))


async def _decision_promoted_from() -> dict[str, Any]:
    engine, _ = await _table_engine()
    await engine.save(_EXISTING, scope="p")
    return _dump_bounce(await engine.save(_INCOMING, scope="p", kind="decision"))


async def _fact_upward_promotion() -> dict[str, Any]:
    engine, _ = await _table_engine(assessor_double=EchoAssessor("repetition"))
    await engine.save(_EXISTING, scope="p")
    return _dump_bounce(await engine.save(_INCOMING, scope="global"))


async def _decision_upward_promotion() -> dict[str, Any]:
    engine, _ = await _table_engine(assessor_double=EchoAssessor("repetition"))
    await engine.save(_EXISTING, scope="p", kind="decision")
    return _dump_bounce(await engine.save(_INCOMING, scope="global", kind="decision"))


async def _judge_unavailable(kind: Kind) -> dict[str, Any]:
    engine, wrapper = await _outage_engine()
    await engine.save(_OUTAGE_EXISTING, scope="p", kind=kind)
    wrapper.arm()
    saved = await engine.save(_OUTAGE_INCOMING, scope="p", kind=kind)
    return _dump_bounce(saved)


async def build_bounces() -> str:
    cases: dict[str, Callable[[], Any]] = {
        "decision_promoted_from": _decision_promoted_from,
        "decision_three_exits": _decision_three_exits,
        "decision_upward_promotion": _decision_upward_promotion,
        "fact_switch_off_judged": _fact_switch_off_judged,
        "fact_three_exits": _fact_three_exits,
        "fact_two_exits": _fact_two_exits,
        "fact_upward_promotion": _fact_upward_promotion,
        "judge_unavailable_decision": lambda: _judge_unavailable("decision"),
        "judge_unavailable_fact": lambda: _judge_unavailable("fact"),
    }
    return dump_json({name: await build() for name, build in cases.items()})


async def test_bounces_golden() -> None:
    assert_golden("bounces.json", await build_bounces())


# ---------------------------------------------------------------------------
# judge_prompts.txt and pack_batches.json
# ---------------------------------------------------------------------------

_ITEM_MODELS: dict[str, Any] = {
    "falsity": FalsityVerdict,
    "relevance": RelevanceVerdict,
    "revisit": RevisitVerdict,
}
_RUBRIC_PROSE = "Judge each record below.\n\nGrade it by the rubric."
_CONTENTS = [
    "The deploy script restarts the unit after rsync.",
    "Line one of a record.\nLine two of the same record.",
    "A third record with a trailing space. ",
]


def build_judge_prompts() -> str:
    sections: list[str] = []
    for name, model in _ITEM_MODELS.items():
        prompt = judge.render_batch_prompt(
            rubric_prose=_RUBRIC_PROSE, contents=_CONTENTS, item_model=model
        )
        sections.append(f"===== {name}: render_batch_prompt =====\n{prompt}")
        sections.append(
            f"===== {name}: render_reply_format =====\n{judge.render_reply_format(model)}"
        )
    return "\n".join(sections) + "\n"


def test_judge_prompts_golden() -> None:
    assert_golden("judge_prompts.txt", build_judge_prompts())


def build_pack_batches() -> str:
    result: dict[str, Any] = {}
    for name, model in _ITEM_MODELS.items():
        base = len(
            judge.render_batch_prompt(rubric_prose=_RUBRIC_PROSE, contents=[], item_model=model)
        )
        cases: dict[str, tuple[list[str], int, int | None]] = {
            "char_ceiling_split": (["x" * 100] * 10, 100, base + 8 + 3 * 110),
            "count_cap": (["short"] * 7, 3, None),
            "oversize_alone": (["a" * 50, "b" * 5000, "c" * 50, "d" * 50], 100, base + 8 + 400),
            "digit_boundary": (["y" * 20] * 12, 100, base + 8 + 9 * 30),
            "empty": ([], 5, None),
            "single_fits": (["only"], 5, None),
        }
        result[name] = {
            case: judge.pack_batches(
                contents,
                count_cap=cap,
                rubric_prose=_RUBRIC_PROSE,
                item_model=model,
                **({} if ceiling is None else {"ceiling": ceiling}),
            )
            for case, (contents, cap, ceiling) in cases.items()
        }
    return dump_json(result)


def test_pack_batches_golden() -> None:
    assert_golden("pack_batches.json", build_pack_batches())


# ---------------------------------------------------------------------------
# question_hashes.json
# ---------------------------------------------------------------------------


def build_question_hashes() -> str:
    hashes = {
        name: assessor.wording_hash(value)
        for name, value in vars(assessor).items()
        if name.endswith("_QUESTION") and isinstance(value, assessor.QuestionSpec)
    }
    return dump_json(hashes)


def test_question_hashes_golden() -> None:
    assert_golden("question_hashes.json", build_question_hashes())


# ---------------------------------------------------------------------------
# arrow_schemas.txt
# ---------------------------------------------------------------------------

_RECORDS_SCHEMA_DIM = 8


def build_arrow_schemas() -> str:
    schemas = {
        name.lstrip("_"): value
        for name, value in vars(lancedb_codec).items()
        if name.endswith("_SCHEMA") and isinstance(value, pa.Schema)
    }
    schemas[f"make_records_schema({_RECORDS_SCHEMA_DIM})"] = lancedb_codec.make_records_schema(
        _RECORDS_SCHEMA_DIM
    )
    sections = [
        f"===== {name} =====\n{schemas[name].to_string(show_schema_metadata=True)}"
        for name in sorted(schemas)
    ]
    return "\n".join(sections) + "\n"


def test_arrow_schemas_golden() -> None:
    assert_golden("arrow_schemas.txt", build_arrow_schemas())


# ---------------------------------------------------------------------------
# persisted_json.json
# ---------------------------------------------------------------------------


def _call_records() -> tuple[judge.JudgeCallRecord, ...]:
    return (
        judge.JudgeCallRecord(
            batch_index=0,
            judge="claude:sonnet",
            rubric_version="v1",
            instruction_fingerprint="abc123",
            duration_seconds=12.5,
            status="success",
            error=None,
            prompt_tokens=1200,
            completion_tokens=80,
            cost=0.0123,
            prompt_chars=4800,
        ),
        judge.JudgeCallRecord(
            batch_index=1,
            judge="codex:gpt",
            rubric_version="v1",
            duration_seconds=1200.0,
            status="judge_unavailable",
            error="codex CLI timed out after 1200.0s",
            prompt_tokens=900,
            completion_tokens=None,
            cost=None,
        ),
    )


def _long_error(label: str) -> str:
    return f"{label}: " + "boom " * 400


def _relevance_summary() -> RelevanceAxisSummary:
    return RelevanceAxisSummary(
        scope=Scope("proj-a"),
        rubric_version="v1",
        judge_pair=("claude:sonnet", "codex:gpt"),
        pool_size=5,
        excluded_decisions=(fixed_id(90),),
        excluded_global=(fixed_id(91),),
        excluded_already_shelved=(fixed_id(92),),
        counts=BandCounts(shelve=1, keep=1, unclear=1, contested=1),
        shelved_ids=(fixed_id(1),),
        unclear_ids=(fixed_id(2),),
        contested_ids=(fixed_id(3),),
        failed_batches=(
            BatchFailure(
                batch_index=2,
                record_ids=(fixed_id(4), fixed_id(5)),
                judge_a_outcome="unavailable",
                judge_b_outcome="invalid_output",
                error=_long_error("relevance"),
            ),
        ),
        calls=_call_records(),
        verdicts=(RecordVerdict(id=fixed_id(1), band="shelve", why="stale"),),
        contested_verdicts=(
            ContestedVerdict(
                id=fixed_id(3),
                judge_a_identity="claude:sonnet",
                judge_a_grade="shelve-lean",
                judge_a_why="A gate clause",
                judge_b_identity="codex:gpt",
                judge_b_grade="keep-clear",
                judge_b_why="B gate clause",
            ),
        ),
        trimmed=(TrimmedPair(old_id=fixed_id(1), new_id=fixed_id(6)),),
    )


def _falsity_summary() -> FalsityAxisSummary:
    return FalsityAxisSummary(
        scope=Scope("proj-a"),
        rubric_version="v1",
        judge_pair=("claude:sonnet", "codex:gpt"),
        newcomer_pool_size=3,
        excluded_decisions=(),
        excluded_global=(),
        excluded_shelved_this_pass=(fixed_id(1),),
        counts=FalsityBandCounts(replace=1, coexist=1, contested=1),
        edges=(
            FalsityEdge(
                id=fixed_id(10),
                superseded_by=fixed_id(11),
                similarity=0.93,
                deciding_judge="claude:sonnet",
                why="the newer record corrects it",
            ),
        ),
        decision_candidates=(
            FalsityReportEntry(
                newcomer_id=fixed_id(12),
                candidate_id=fixed_id(13),
                similarity=0.88,
                reason="candidate_is_decision",
                assessor_pick="coexist",
            ),
        ),
        global_candidates=(),
        hot_or_ember_candidates=(),
        contested_pairs=(
            FalsityContestedPair(
                newcomer_id=fixed_id(14),
                candidate_id=fixed_id(15),
                similarity=0.91,
                judge_a_identity="claude:sonnet",
                judge_a_verdict="replace",
                judge_a_why="what changed",
                judge_b_identity="codex:gpt",
                judge_b_verdict="coexist",
                judge_b_why="what stands",
            ),
        ),
        failed_batches=(
            FalsityBatchFailure(
                batch_index=0,
                pairs=((fixed_id(16), fixed_id(17)),),
                judge_a_outcome="invalid_output",
                judge_b_outcome="unavailable",
                error=_long_error("falsity"),
            ),
        ),
        calls=_call_records()[:1],
    )


def _revisit_summary() -> RevisitAxisSummary:
    return RevisitAxisSummary(
        scope=Scope("proj-a"),
        rubric_version="v1",
        judge_pair=("claude:sonnet", "codex:gpt"),
        pool_size=4,
        excluded_global=(),
        excluded_non_fact=(),
        protected_ids=(fixed_id(20),),
        skipped_no_evidence_ids=(fixed_id(21),),
        skipped_too_large_ids=(fixed_id(22), fixed_id(23)),
        counts=BandCounts(shelve=1, keep=2, unclear=0, contested=1),
        shelved_ids=(fixed_id(24),),
        unclear_ids=(),
        contested_ids=(fixed_id(25),),
        failed_batches=(
            BatchFailure(
                batch_index=1,
                record_ids=(fixed_id(26),),
                judge_a_outcome="unavailable",
                judge_b_outcome="unavailable",
                error=_long_error("revisit"),
            ),
        ),
        calls=_call_records(),
        verdicts=(RecordVerdict(id=fixed_id(24), band="shelve", why="superseded in fact"),),
        contested_verdicts=(),
        verification_failures=(fixed_id(27),),
        split_keep=1,
        trimmed=(
            TrimmedPair(old_id=fixed_id(24), new_id=fixed_id(28)),
            TrimmedPair(old_id=fixed_id(29), new_id=None),
        ),
    )


def _clean_summary(
    *, outcome: Outcome, with_axes: bool, dry_run: bool, revisit_error: str | None = None
) -> clean.CleanSummary:
    return clean.CleanSummary(
        scope=Scope("proj-a"),
        outcome=outcome,
        watermark_before=None if outcome == "initialized" else _EPOCH,
        watermark_after=_PASS_AT,
        pool_size=5,
        relevance=_relevance_summary() if with_axes else None,
        falsity=_falsity_summary() if with_axes else None,
        revisit=_revisit_summary() if with_axes and revisit_error is None else None,
        revisit_error=revisit_error,
        verification_failures=(fixed_id(30),) if with_axes else (),
        reason=f"golden {outcome}",
        dry_run=dry_run,
    )


def build_persisted_json() -> str:
    completed = _clean_summary(outcome="completed", with_axes=True, dry_run=False)
    errored = _clean_summary(
        outcome="completed", with_axes=True, dry_run=False, revisit_error="revisit axis crashed"
    )
    built = report.build_report(completed, pass_at=_PASS_AT)
    built_errored = report.build_report(errored, pass_at=_PASS_AT)
    assert built is not None
    assert built_errored is not None
    fallback_payload = clean.fallback_payload
    fallback_revisit_payload = clean.fallback_revisit_payload
    revisit_payload, remaining = fallback_revisit_payload(completed, budget=900)
    initialized = _clean_summary(outcome="initialized", with_axes=False, dry_run=False)
    empty_pool = _clean_summary(outcome="empty_pool", with_axes=False, dry_run=False)
    dry_run = _clean_summary(outcome="completed", with_axes=True, dry_run=True)
    return dump_json(
        {
            "judge_call_records": [call.model_dump_json() for call in _call_records()],
            "cleaner_report_completed": built.model_dump_json(),
            "cleaner_report_revisit_error": built_errored.model_dump_json(),
            "fallback_payload_initialized": fallback_payload(initialized),
            "fallback_payload_empty_pool": fallback_payload(empty_pool),
            "fallback_payload_dry_run_completed": fallback_payload(dry_run),
            "fallback_payload_revisit_error": fallback_payload(errored),
            "fallback_revisit_payload": json.dumps(revisit_payload),
            "fallback_revisit_payload_remaining_budget": remaining,
            "fallback_revisit_payload_without_axis": json.dumps(
                fallback_revisit_payload(initialized, budget=900)[0]
            ),
        }
    )


def test_persisted_json_golden() -> None:
    assert_golden("persisted_json.json", build_persisted_json())


# ---------------------------------------------------------------------------
# judge_argv.json
# ---------------------------------------------------------------------------

_GRANT = McpGrant(
    server_url="http://127.0.0.1:8765/mcp",
    server_name="tabularium",
    allowed_tools=("recall", "context"),
)


async def _capture_argv(cli: Cli, grant: McpGrant | None) -> list[str]:
    adapter = judge.SubprocessJudgeAdapter(
        cli=cli,
        binary=f"/usr/bin/{cli}",
        model="golden-model",
        grant=grant,
    )
    captured: list[list[str]] = []

    async def fake_run(argv: list[str], prompt: str) -> tuple[bytes, bytes, int]:
        captured.append(list(argv))
        if "--output-last-message" in argv:
            Path(argv[argv.index("--output-last-message") + 1]).write_text("{}")
        return b"{}", b"", 0

    adapter._run = fake_run  # type: ignore[method-assign]  # stub the subprocess seam
    await adapter.call("prompt")
    argv = captured[0]
    if "--output-last-message" in argv:
        argv[argv.index("--output-last-message") + 1] = "<outfile>"
    return argv


async def build_judge_argv() -> str:
    return dump_json(
        {
            "claude_without_grant": await _capture_argv("claude", None),
            "claude_with_grant": await _capture_argv("claude", _GRANT),
            "codex_without_grant": await _capture_argv("codex", None),
            "codex_with_grant": await _capture_argv("codex", _GRANT),
        }
    )


async def test_judge_argv_golden() -> None:
    assert_golden("judge_argv.json", await build_judge_argv())
