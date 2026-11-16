"""The world a tool-golden scenario runs in: engine, MCP app, fault doubles and the dump.

A scenario builds a ``ToolWorld``, drives the read tools with ``await world.call(name, args)``
exactly as an MCP client would (through ``make_app(engine).call_tool``), and returns
``await world.finish()``. The server registers no write tool, so a scenario seeds its store with
``await world.write(name, args)``, which runs the engine method of that name directly.
Every call is logged in order with its arguments and either the full ``structured_content`` or the
exact error text, and ``pending_backfill`` is awaited after each call so a background pass cannot
race the next step. ``finish`` adds the store: every record, vector presence, tombstones, verdict
rows, events, read events, unsupersede and revalidation rows, distinct pairs, sessions, reviews,
judgments and the cleaner's state and pass history. A collection with nothing in it is left out.
The scope, facet and role registries are not dumped; the scenario that re-runs ``initialize`` logs
them.

Nothing is normalized here. The clock ticks once per read, ``uuid4`` is pinned by
``tests.support.cleaner_world.pinned_environment``, the embedder is exact arithmetic, and the
assessor is scripted, so every value is deterministic.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as FastMCPToolError
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult
from pydantic import validate_call

from tabularium.config import EngineConfig
from tabularium.domain import Scope
from tabularium.engine import Tabularium
from tabularium.engine import ToolError as EngineToolError
from tabularium.engine.assessor import Assessor
from tabularium.server import make_app
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import StorageAdapter
from tests.support.cleaner_golden import ClusterEmbedder, TickingClock
from tests.support.doubles import FakeTokenCounter, TestClock
from tests.support.faults import (
    FaultInjectingAdapter,
    FaultInjectingEmbedder,
    as_adapter,
    as_embedder,
)
from tests.support.tool_golden_assessor import ScriptedInterrupt

Json = dict[str, Any]

START = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
DAY = timedelta(days=1)
SCOPES = ("p", "q")


def golden_config(**overrides: Any) -> EngineConfig:
    """The engine config every scenario shares, plus per-scenario overrides.

    ``candidate_k``, ``gray_floor`` and ``decision_gray_floor`` fit the exact cosines of
    ``ClusterEmbedder``; the digest cleaner stays off (the default).
    """
    values: dict[str, Any] = {
        "candidate_k": 1000,
        "gray_floor": 0.80,
        "decision_gray_floor": 0.70,
    }
    values.update(overrides)
    return EngineConfig(**values)


def cluster(text: str, number: int, variant: int = 0) -> str:
    """``text`` tagged so ``ClusterEmbedder`` puts it in cluster ``number`` at ``variant``.

    Variants 0 and 1 of one cluster have cosine 0.96, variants 0 and 2 have 0.936. The tag's
    digits are listed in the text too, so two variants never differ in their numerals and the
    numeral guard does not trip on the tag alone.
    """
    return f"{text} 0 1 2 3 <c{number}.{variant}>"


class _EngineSlot:
    """Stands in for the engine in the one shared MCP app and forwards every attribute to the engine
    of the world that is making a call.

    ``make_app`` builds the JSON schema of every tool, about 0.2 s a call, which dominated the run
    while every world built its own app. Building the app once and pointing it at each world's
    engine keeps the path a client takes, ``make_app(engine)`` then ``app.call_tool``; the app's
    only reference to an engine is this object. ``ToolWorld.call`` selects its own engine before
    every call, and no scenario runs two worlds at once.
    """

    def __init__(self) -> None:
        self.target: Tabularium | None = None

    def __getattr__(self, name: str) -> Any:
        if self.target is None:
            raise AttributeError(name)
        return getattr(self.target, name)


_SLOT = _EngineSlot()
_SHARED_APP: list[MCPServer[None]] = []


def _shared_app() -> MCPServer[None]:
    if not _SHARED_APP:
        _SHARED_APP.append(make_app(cast(Tabularium, _SLOT)))
    return _SHARED_APP[0]


@dataclass
class ToolWorld:
    engine: Tabularium
    app: MCPServer[None]
    adapter: MemoryAdapter
    storage: StorageAdapter
    clock: TickingClock
    embedder: FaultInjectingEmbedder
    adapter_faults: FaultInjectingAdapter
    calls: list[Json] = field(default_factory=list)

    async def call(self, name: str, args: Json) -> Json | None:
        """Call a tool as an MCP client does; log the full outcome; return the structured content.

        An error returns ``None``; the log keeps ``isError`` and the exact text.
        """
        entry: Json = {"tool": name, "args": args}
        result: Json | None = None
        _SLOT.target = self.engine
        try:
            outcome = await self.app.call_tool(name, args)
        except FastMCPToolError as exc:
            entry["isError"] = True
            entry["error"] = str(exc)
        except MCPError as exc:
            entry["protocolError"] = {"code": exc.error.code, "message": exc.error.message}
        except ScriptedInterrupt as exc:
            entry["interrupted"] = str(exc)
        else:
            assert isinstance(outcome, CallToolResult), (
                "the tools return results, not input requests"
            )
            entry["isError"] = bool(outcome.is_error)
            result = outcome.structured_content
            entry["result"] = result

        await self._settle_backfill(entry)
        self.calls.append(entry)
        return result

    async def write(self, name: str, args: Json) -> Json | None:
        """Run the engine write method ``name`` as setup; log the full outcome; return the result.

        The arguments are validated against the method's signature as the retired tool wrapper's
        were, and an ``EngineToolError`` or ``ValueError`` is logged as ``isError`` with its text.
        An error returns ``None``.
        """
        entry: Json = {"write": name, "args": args}
        result: Json | None = None
        try:
            outcome = await validate_call(getattr(self.engine, name))(**args)
        except (EngineToolError, ValueError) as exc:
            entry["isError"] = True
            entry["error"] = str(exc)
        else:
            entry["isError"] = False
            result = _plain(outcome)
            entry["result"] = result

        await self._settle_backfill(entry)
        self.calls.append(entry)
        return result

    async def _settle_backfill(self, entry: Json) -> None:
        try:
            await self.engine.pending_backfill
        except Exception as exc:
            entry["backfill_raised"] = {"type": type(exc).__name__, "message": str(exc)}

    async def direct(self, label: str, awaitable: Awaitable[Any]) -> Any:
        """Await an engine-level call that no tool reaches; log its return or exception."""
        entry: Json = {"direct": label}
        value: Any = None
        try:
            value = await awaitable
        except Exception as exc:
            entry["raised"] = {"type": type(exc).__name__, "message": str(exc)}
        else:
            entry["returned"] = _plain(value)

        await self._settle_backfill(entry)
        self.calls.append(entry)
        return value

    def note(self, label: str, value: object) -> None:
        """Log a value the scenario computed outside any tool call."""
        self.calls.append({"note": label, "value": _plain(value)})

    async def finish(self) -> Json:
        await self.engine.pending_backfill
        await assert_vectors_match_content(self.adapter)
        return {"calls": self.calls, "store": await dump_store(self.adapter)}


def _plain(value: object) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


async def make_world(
    *,
    assessor: Assessor | None = None,
    config: EngineConfig | None = None,
    scopes: tuple[str, ...] = SCOPES,
    inject_token_counter: bool = True,
) -> ToolWorld:
    """An initialized engine over a ``MemoryAdapter`` with scopes ``p`` and ``q`` registered."""
    embedder = ClusterEmbedder()
    adapter = MemoryAdapter(
        embedding_model=embedder.model,
        dim=embedder.dim,
        clock=TestClock(START),
    )
    adapter_faults = FaultInjectingAdapter(adapter)
    embedder_faults = FaultInjectingEmbedder(embedder)
    clock = TickingClock(START)
    engine = Tabularium(
        as_adapter(adapter_faults),
        as_embedder(embedder_faults),
        clock,
        config if config is not None else golden_config(),
        token_counter=FakeTokenCounter() if inject_token_counter else None,
        assessor=assessor,
    )
    await engine.initialize()
    for name in scopes:
        await adapter.create_scope(Scope(name))

    return ToolWorld(
        engine=engine,
        app=_shared_app(),
        adapter=adapter,
        storage=as_adapter(adapter_faults),
        clock=clock,
        embedder=embedder_faults,
        adapter_faults=adapter_faults,
    )


def saved_id(result: Json | None) -> str:
    """The record id a ``save`` or ``ember`` result names: the written record, else the bounce's
    candidate, else the reinforced record."""
    assert result is not None, "the setup call raised"
    for key in ("hypomnema", "bounce"):
        block = result.get(key)
        if block is not None:
            return str(block["id"] if key == "hypomnema" else block["candidate_id"])
    reinforced = result.get("reinforced_id")
    assert reinforced is not None, f"no record id in {result}"
    return str(reinforced)


def expect(result: Json | None, outcome: str) -> Json:
    """Assert a setup call returned ``outcome`` so a drifted scenario fails where it drifted."""
    assert result is not None, "the setup call raised"
    assert result.get("outcome") == outcome, f"expected {outcome!r}, got {result.get('outcome')!r}"
    return result


async def assert_vectors_match_content(adapter: MemoryAdapter) -> None:
    """Fail when a stored vector is not the embedder's vector for its record's content.

    The dump records only whether a row has a vector, so this closes the gap: a save, revival or
    backfill that stored a wrong or zeroed vector would otherwise leave the golden unchanged.
    """
    embedder = ClusterEmbedder()
    async for record in await adapter.scan_records():
        row = adapter._rows[record.id]
        if row.vector is None:
            continue
        expected = embedder.vector(record.content)
        assert list(row.vector) == list(expected), f"stored vector of {record.id} is not content's"


async def dump_store(adapter: MemoryAdapter) -> Json:
    """The adapter state a tool or an engine call can change, in store order; a collection with
    nothing in it is left out."""
    store = await _dump_all(adapter)
    return {name: value for name, value in store.items() if value}


async def _dump_all(adapter: MemoryAdapter) -> Json:
    records = [record async for record in await adapter.scan_records()]
    return {
        "records": [record.model_dump(mode="json") for record in records],
        "vectors": [
            {
                "id": str(record_id),
                "has_vector": row.vector is not None,
                "embedding_model": row.embedding_model,
            }
            for record_id, row in adapter._rows.items()
        ],
        "tombstones": [t.model_dump(mode="json") for t in adapter._tombstones.values()],
        "audit": {str(record_id): trail for record_id, trail in adapter._audit.items()},
        "events": [event.model_dump(mode="json") for event in adapter._events],
        "read_events": [event.model_dump(mode="json") for event in adapter._read_events],
        "distinct_pairs": [pair.model_dump(mode="json") for pair in adapter._distinct.values()],
        "verdict_rows": [row.model_dump(mode="json") for row in adapter._verdicts.values()],
        "unsupersedes": [event.model_dump(mode="json") for event in adapter._unsupersedes],
        "revalidations": [event.model_dump(mode="json") for event in adapter._revalidations],
        "record_reviews": [review.model_dump(mode="json") for review in adapter._record_reviews],
        "record_judgments": [j.model_dump(mode="json") for j in adapter._record_judgments],
        "sessions": [session.model_dump(mode="json") for session in adapter._sessions],
        "cleaner_pass_history": [
            record.model_dump(mode="json") for record in adapter._cleaner_pass_history
        ],
        "cleaner_state": {
            str(scope): state.model_dump(mode="json")
            for scope, state in adapter._cleaner_state.items()
        },
    }


_STANDARD_RECORD_FIELDS = frozenset(
    logging.LogRecord("x", logging.INFO, "x", 0, "x", None, None).__dict__
) | {"message", "asctime", "taskName"}


class _LogCapture(logging.Handler):
    """Collects what a ``tabularium`` logger emits as plain JSON: logger name, level, message
    and the ``extra`` fields, in emitted order.

    Neither the module, the function nor the line is recorded, so a log helper can move to another
    module without changing the golden while a changed logger name, level, message or ``extra``
    does. ``created`` and the other wall-clock fields of a record are left out as well; the message
    and ``extra`` of the records the scenarios reach carry only values the pinned clock, the pinned
    ``uuid4`` and the scripted doubles control, which the cleaner golden normalises the same way.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[Json] = []

    def emit(self, record: logging.LogRecord) -> None:
        extra = {
            key: _plain(value)
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_FIELDS
        }
        entry: Json = {
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "extra": extra,
        }
        if record.exc_info is not None and record.exc_info[1] is not None:
            error = record.exc_info[1]
            entry["exception"] = f"{type(error).__name__}: {error}"
        self.records.append(entry)


@contextmanager
def captured_logs() -> Iterator[list[Json]]:
    """Capture every record the ``tabularium`` loggers emit at any level for the block."""
    root = logging.getLogger("tabularium")
    handler = _LogCapture()
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


Scenario = Callable[[], Awaitable[Json]]
TOOL_SCENARIOS: dict[str, Scenario] = {}


def scenario(function: Scenario) -> Scenario:
    """Register ``function`` under its own name; the name is the golden's scenario key."""
    TOOL_SCENARIOS[function.__name__] = function
    return function


def inject(world: ToolWorld, on: str, exc: BaseException, *, call: int = 1) -> None:
    """Make the ``call``-th call of adapter method ``on`` from now raise ``exc`` before it runs."""
    world.adapter_faults.fault(on=on, call=call, exc=exc)
    world.adapter_faults.arm()


async def fact(world: ToolWorld, text: str, scope: str = "p", **extra: Any) -> str:
    """Save a Fact through ``Tabularium.save`` and return its id; the save must land."""
    result = await world.write("save", {"content": text, "scope": scope, **extra})
    return saved_id(expect(result, "saved"))


async def decision(world: ToolWorld, text: str, scope: str = "p", **extra: Any) -> str:
    """Save a Decision through ``Tabularium.save`` and return its id; the save must land."""
    result = await world.write(
        "save", {"content": text, "scope": scope, "kind": "decision", **extra}
    )
    return saved_id(expect(result, "saved"))


async def ember(
    world: ToolWorld,
    text: str,
    *,
    scope: str = "p",
    valence: str = "correction",
    **extra: Any,
) -> Json:
    """Call ``Tabularium.ember`` and return its result."""
    result = await world.write(
        "ember", {"content": text, "scope": scope, "valence": valence, **extra}
    )
    assert result is not None, "the ember call raised"
    return result
