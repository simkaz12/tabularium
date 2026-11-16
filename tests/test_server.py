"""MCP server tests — in-process round-trips (B4, SPEC §9).

Tests use FastMCP.call_tool / read_resource directly (no stdio transport
needed); the engine is wired with FakeEmbedder + MemoryAdapter.

Acceptance coverage
-------------------
  recall → hits                — basic hybrid search result
  decision → DecisionView      — rendered == resource read bytes for same id
  decisions_query → links      — DecisionLink shape, uri scheme
  ToolError → isError=true     — unknown decision id, unregistered scope
  4 tools registered, all async, read-only annotation flags
  context → ContextPayload     — end-to-end FastMCP-surface tool round-trip
  (SPEC §7 / §9.1 / §9.2): populated hot tier, registered-but-empty
  scope is NOT an error, unregistered scope/role IS a ToolError, two
  registered roles return identical payloads, negative/non-integer budget is
  a pydantic validation failure surfaced by FastMCP.

The server registers no write tool, so a test that needs a stored record seeds it with the
engine (``engine.save``) and then reads it through the tool under test.

Hermeticity: every engine built here injects ``tokens.FakeTokenCounter`` (see
``_make_engine_components``) so the ``context`` tool round-trips in this file
never construct a real ``LitellmTokenCounter`` (tests/test_tokens.py's rule).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Fact, Scope, new_id
from tabularium.engine import Tabularium
from tabularium.results import (
    ContextPayload,
    DecisionLink,
    DecisionView,
    RecallResult,
)
from tabularium.server import make_app
from tabularium.storage.memory import MemoryAdapter
from tabularium.tool_text import SERVER_INSTRUCTIONS
from tests.support.doubles import FakeEmbedder, FakeTokenCounter, TestClock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 16


def _s(result: Any) -> dict[str, Any]:
    """Extract the structured output dict from a ``call_tool`` result.

    v2's ``MCPServer.call_tool`` returns a single ``CallToolResult`` (no
    longer a ``(content, structured)`` tuple) whose ``structured_content``
    carries the dict every call site here actually wants.
    """
    return cast(dict[str, Any], result.structured_content)


def _make_engine_components(
    *,
    dim: int = _DIM,
    should_fail: bool = False,
) -> tuple[Tabularium, MemoryAdapter, TestClock]:
    adapter = MemoryAdapter(embedding_model="fake/nomic", dim=dim)
    embedder = FakeEmbedder(dim=dim, should_fail=should_fail)
    clock = TestClock(datetime(2025, 1, 1, tzinfo=UTC))
    cfg = EngineConfig(candidate_k=1000, gray_floor=0.80)
    # token_counter=FakeTokenCounter(): keeps every ``context`` tool
    # round-trip in this file hermetic — a real LitellmTokenCounter is
    # never constructed here (tests/test_tokens.py's rule).
    engine = Tabularium(adapter, embedder, clock, cfg, token_counter=FakeTokenCounter())
    return engine, adapter, clock


def _hot_fact(
    *,
    scope: str = "global",
    content: str = "a hot fact",
    feedback_score: float = 0.9,
    audience_hint: tuple[str, ...] | None = ("decider",),
) -> Fact:
    """A minimal ``hot``-lifecycle Fact for context-tool tests — the ``context``
    tool never promotes anything itself (that is the sweep's job), so
    tests that need a populated hot tier write one directly via the adapter.

    Defaults to ``audience_hint=("decider",)`` (curation block for
    the routing semantics; retargeted off the retired ``"default"`` role, which also makes
    ``role`` REQUIRED on the ``context`` tool —
    there is no longer an "omit role" case to exercise): explicit
    enumeration only means an unhinted Fact reaches NO role on the push
    tiers, and most context-tool tests here query ``role="decider"`` to
    exercise mechanics unrelated to routing itself. Pass
    ``audience_hint=None`` explicitly for a genuinely untagged Fact.
    """
    now = datetime(2025, 1, 1, tzinfo=UTC)
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope(scope),
        lifecycle="hot",
        validity="valid",
        feedback_score=feedback_score,
        created_at=now,
        updated_at=now,
        audience_hint=audience_hint,
    )


def _vec(dim: int = _DIM, seed: float = 1.0) -> list[float]:
    return [seed] * dim


async def _make_world(
    *,
    dim: int = _DIM,
    should_fail: bool = False,
    extra_scope: str | None = None,
) -> tuple[Tabularium, Any]:
    """Build an initialized engine and the FastMCP app over it."""
    engine, adapter, _ = _make_engine_components(dim=dim, should_fail=should_fail)
    await engine.initialize()
    if extra_scope:
        await adapter.create_scope(Scope(extra_scope))
    return engine, make_app(engine)


async def _make_app(
    *,
    dim: int = _DIM,
    should_fail: bool = False,
    extra_scope: str | None = None,
) -> Any:
    """Build a FastMCP app backed by an initialized engine."""
    _engine, app = await _make_world(dim=dim, should_fail=should_fail, extra_scope=extra_scope)
    return app


# ---------------------------------------------------------------------------
# recall → hits
# ---------------------------------------------------------------------------


async def test_recall_returns_hits() -> None:
    engine, app = await _make_world(extra_scope="global")
    await engine.save("Python packaging uses uv.", scope="global")
    r = await app.call_tool("recall", {"query": "Python packaging", "scope": "global"})
    result = RecallResult(**_s(r))
    assert not result.degraded
    assert len(result.hits) > 0
    assert result.hits[0].kind in {"fact", "decision"}


# ---------------------------------------------------------------------------
# decision → DecisionView with rendered == resource bytes
# ---------------------------------------------------------------------------


async def test_decision_rendered_equals_resource() -> None:
    """DecisionView.rendered must be byte-identical to resource read."""
    from tabularium.domain import Scope

    adapter = MemoryAdapter(embedding_model="fake/nomic", dim=_DIM)
    embedder = FakeEmbedder(dim=_DIM)
    clock = TestClock(datetime(2025, 1, 1, tzinfo=UTC))
    cfg = EngineConfig(candidate_k=1000, gray_floor=0.80)
    engine = Tabularium(adapter, embedder, clock, cfg)
    await engine.initialize()
    await adapter.create_scope(Scope("global"))

    app = make_app(engine)

    save_result = await engine.save(
        "Use FastMCP for the MCP server layer.", scope="global", kind="decision"
    )
    assert save_result.hypomnema is not None
    dec_id = str(save_result.hypomnema.id)

    # Fetch DecisionView
    dv_raw = await app.call_tool("decision", {"id": dec_id})
    dv = DecisionView(**_s(dv_raw))

    # Read resource. ``read_resource``'s return type is
    # ``Iterable[ReadResourceContents] | InputRequiredResult`` (v2's
    # multi-round-trip elicitation flow) -- this resource never elicits, so
    # the cast narrows to the branch that's actually reachable here.
    from mcp.server.lowlevel.helper_types import ReadResourceContents

    resource_contents = cast(
        list[ReadResourceContents], list(await app.read_resource(f"decision://{dec_id}"))
    )
    assert len(resource_contents) == 1
    resource_text = resource_contents[0].content

    assert dv.rendered == resource_text, (
        "DecisionView.rendered must be byte-identical to decision://{id} resource"
    )


# ---------------------------------------------------------------------------
# decisions_query → DecisionLink list
# ---------------------------------------------------------------------------


async def test_decisions_query_returns_links() -> None:
    engine, app = await _make_world(extra_scope="global")
    await engine.save("Use pydantic v2 for schema validation.", scope="global", kind="decision")
    r = await app.call_tool(
        "decisions_query", {"query": "pydantic schema validation", "scope": "global"}
    )
    structured = _s(r)
    # Structured output wraps a list in {"result": [...]}
    raw_list = cast(
        list[dict[str, Any]],
        structured["result"]
        if isinstance(structured, dict) and "result" in structured
        else structured,
    )
    links = [DecisionLink(**item) for item in raw_list]
    assert len(links) > 0
    link = links[0]
    assert link.uri.startswith("decision://")
    assert link.lifecycle in {"in_force", "superseded", "revoked"}
    assert link.validity == "valid"
    assert link.scope == "global"


async def test_decisions_query_and_decision_descriptions_document_invalidation() -> None:
    """P9 invalid-Decision read contract: ``decisions_query``'s description
    documents ``include_invalid`` beside ``include_superseded``; ``decision``'s
    names the two new columns and ``revalidations``."""
    app = await _make_app()
    tools = await app.list_tools()
    decisions_query_tool = next(t for t in tools if t.name == "decisions_query")
    decision_tool = next(t for t in tools if t.name == "decision")

    dq_description = decisions_query_tool.description or ""
    assert "include_superseded" in dq_description
    assert "include_invalid" in dq_description

    d_description = decision_tool.description or ""
    assert "invalidated_by" in d_description
    assert "invalidation_reason" in d_description
    assert "revalidations" in d_description


# ---------------------------------------------------------------------------
# ToolError → isError=true (FastMCPToolError raised)
# ---------------------------------------------------------------------------


async def test_unknown_decision_id_raises_tool_error() -> None:
    """Unknown decision id → FastMCPToolError."""
    from mcp.server.mcpserver.exceptions import ToolError as FastMCPToolError

    app = await _make_app()
    fake_id = str(uuid.uuid4())
    with pytest.raises(FastMCPToolError):
        await app.call_tool("decision", {"id": fake_id})


async def test_fact_id_for_decision_tool_raises_tool_error() -> None:
    """Fact id passed to ``decision`` tool → FastMCPToolError."""
    from mcp.server.mcpserver.exceptions import ToolError as FastMCPToolError

    engine, app = await _make_world(extra_scope="global")
    saved = await engine.save("A regular fact.", scope="global")
    assert saved.hypomnema is not None
    with pytest.raises(FastMCPToolError):
        await app.call_tool("decision", {"id": str(saved.hypomnema.id)})


async def test_decision_tool_value_error_becomes_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``ValueError`` from ``engine.decision`` reaches the client as a tool error."""
    from mcp.server.mcpserver.exceptions import ToolError as FastMCPToolError

    engine, _adapter, _clock = _make_engine_components()
    await engine.initialize()

    async def _bad(*args: object, **kwargs: object) -> None:
        raise ValueError("malformed decision request")

    monkeypatch.setattr(engine, "decision", _bad)
    app = make_app(engine)

    with pytest.raises(FastMCPToolError) as exc_info:
        await app.call_tool("decision", {"id": str(uuid.uuid4())})
    assert "malformed decision request" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tool list: verify four tools are registered
# ---------------------------------------------------------------------------


async def test_four_read_tools_registered() -> None:
    """The server lists exactly the four read tools and registers no write tool; new records
    reach the live store through t3's ``memory_propose``, not through this server."""
    app = await _make_app()
    tools = await app.list_tools()
    names = {t.name for t in tools}
    assert {"recall", "decision", "decisions_query", "context"} == names, (
        f"Expected exactly 4 tools; got: {names}"
    )


async def test_write_tool_is_not_registered() -> None:
    """A v1 write tool is absent from the server, not registered and refusing: a client that
    still calls ``save`` gets an unknown-tool error and nothing is written."""
    from mcp.server.mcpserver.exceptions import ToolError as FastMCPToolError

    engine, app = await _make_world(extra_scope="global")

    with pytest.raises(FastMCPToolError, match="Unknown tool: save"):
        await app.call_tool("save", {"content": "x", "scope": "global"})
    recalled = await engine.recall("x", scope="global")
    assert recalled.hits == []


async def test_server_instructions_send_writes_to_memory_propose() -> None:
    app = await _make_app()

    assert app.instructions == SERVER_INSTRUCTIONS
    assert "memory_propose" in SERVER_INSTRUCTIONS
    assert "``save``" not in SERVER_INSTRUCTIONS


async def test_all_tools_are_async() -> None:
    """Every registered tool must stay ``async def`` (SDK v2 migration,
    2026-08-12 -- see the module docstring's "Concurrency invariants"
    section). v2 runs a *sync* handler on a worker thread, which
    ``_write_lock`` (an ``asyncio.Lock``) does not span -- a sync handler
    would run concurrently with it, breaking the dedup gate's
    check-then-write atomicity. v2's internal ``Tool`` carries
    ``is_async`` (``mcpserver/tools/base.py``); the tool manager (not the
    client-facing ``list_tools()`` schema view, which drops that field)
    is where it lives.

    Also asserts the count, so a new tool cannot land ``def`` (sync) by
    mistake without this test noticing it exists at all.
    """
    app = await _make_app()
    registered = app._tool_manager.list_tools()
    assert len(registered) == 4, f"Expected exactly 4 tools; got {len(registered)}"
    not_async = [t.name for t in registered if not t.is_async]
    assert not_async == [], f"Sync (non-async) tool handlers found: {not_async}"


async def test_tool_annotation_flags() -> None:
    """Annotation flags per SPEC §9.1's table (§9.1 line 405-410)."""
    app = await _make_app()
    tools = {t.name: t for t in (await app.list_tools())}

    def _flags(name: str) -> tuple[bool | None, bool | None, bool | None]:
        ann = tools[name].annotations
        assert ann is not None
        return ann.read_only_hint, ann.idempotent_hint, ann.destructive_hint

    assert _flags("recall") == (True, None, None)
    assert _flags("decision") == (True, None, None)
    assert _flags("decisions_query") == (True, None, None)
    assert _flags("context") == (True, None, None)


# ---------------------------------------------------------------------------
# Resource template: decision://{id}
# ---------------------------------------------------------------------------


async def test_resource_template_listed() -> None:
    app = await _make_app()
    templates = await app.list_resource_templates()
    uris = {t.uri_template for t in templates}
    assert "decision://{id}" in uris


async def test_unknown_resource_id_raises_error() -> None:
    """decision://<unknown-uuid> must raise an error. Scope 3 step 5 (SEP-2164,
    SDK v2 migration 2026-08-12): an id that parses as a UUID but has no
    Decision is a ``ResourceNotFoundError`` (-32602), which propagates
    untouched through v2's resource-template wrapper (``except (ResourceError,
    MCPError): raise``) with the original engine message intact -- no longer
    a ``ValueError`` and no longer needing ``__cause__`` to find the text."""
    from mcp.server.mcpserver.exceptions import ResourceNotFoundError

    app = await _make_app()
    fake_id = str(uuid.uuid4())
    with pytest.raises(ResourceNotFoundError, match="unknown or buried id"):
        await app.read_resource(f"decision://{fake_id}")


# ---------------------------------------------------------------------------
# context — session-start grounding payload (SPEC §7 / §9.1 / §9.2)
#
# End-to-end tests: every one of these drives the tool through the FastMCP
# surface (``app.call_tool("context", ...)``), never the engine method
# directly — the acceptance-criteria bar for this brief.
# ---------------------------------------------------------------------------


async def test_context_description_names_two_tiers_not_decisions() -> None:
    """M2 slice C (2026-08-01) deleted the in-force Decision tier
    from ``select_default`` — the tool description must not still advertise
    it, or claim a three-tier empty-scope shape. Pinned so this cannot
    silently rot again."""
    app = await _make_app()
    tools = await app.list_tools()
    context_tool = next(t for t in tools if t.name == "context")
    description = context_tool.description or ""
    assert "in-force Decisions" not in description
    assert "three" not in description.lower()
    assert "decisions_query" in description


async def test_context_populated_hot_tier_end_to_end() -> None:
    """A hot Fact written directly via the adapter (the sweep's job,
    is out of scope here) shows up in the ``context`` tool's ``hot`` tier."""
    engine, adapter, _ = _make_engine_components()
    await engine.initialize()
    await adapter.create_scope(Scope("global"))
    hot = _hot_fact(content="a hot fact for the context tool end-to-end test")
    await adapter.upsert(hot, _vec())

    app = make_app(engine)
    structured = await app.call_tool(
        "context", {"scope": "global", "budget": 1000, "role": "decider"}
    )
    payload = ContextPayload(**_s(structured))

    assert [t.name for t in payload.tiers] == ["hot", "ember"]
    hot_tier = next(t for t in payload.tiers if t.name == "hot")
    assert [str(e.id) for e in hot_tier.entries] == [str(hot.id)]
    assert payload.budget_used > 0
    assert payload.skipped == 0


async def test_context_registered_but_empty_scope_is_not_an_error() -> None:
    """§9.2 draws this line: registered-but-empty scope -> all tiers
    present and empty, isError=false -- never confused with an unregistered
    scope (which IS a ToolError, see below)."""
    app = await _make_app(extra_scope="proj-empty")
    structured = await app.call_tool(
        "context", {"scope": "proj-empty", "budget": 100, "role": "decider"}
    )
    payload = ContextPayload(**_s(structured))

    assert [t.name for t in payload.tiers] == ["hot", "ember"]
    for tier in payload.tiers:
        assert tier.entries == []
    assert payload.budget_used == 0
    assert payload.skipped == 0


async def test_context_unregistered_scope_raises_tool_error() -> None:
    from mcp.server.mcpserver.exceptions import ToolError as FastMCPToolError

    app = await _make_app()  # "never-registered" is not registered
    with pytest.raises(FastMCPToolError):
        await app.call_tool("context", {"scope": "never-registered", "budget": 100})


async def test_context_unregistered_role_raises_tool_error() -> None:
    from mcp.server.mcpserver.exceptions import ToolError as FastMCPToolError

    app = await _make_app(extra_scope="global")
    with pytest.raises(FastMCPToolError):
        await app.call_tool(
            "context",
            {"scope": "global", "budget": 100, "role": "never-registered"},
        )


async def test_context_omitted_role_raises_tool_error_at_the_mcp_boundary() -> None:
    """Curation block: ``role`` is now REQUIRED -- an MCP-protocol
    ``context`` call that omits it is bounced with a ``FastMCPToolError``
    naming the registered roles, exactly like a direct engine call (see
    ``tests/test_context.py::TestEngineContextRoleValidation::
    test_omitted_role_raises_tool_error_naming_registered_roles``). The tool
    itself does no extra validation (server.py stays "near-1:1 projection,
    ZERO memory semantics") -- this is the engine's ``ToolError`` propagating
    through the SAME try/except every other engine error already uses.

    Falsifiability (step 6): restoring the retired ``role if role is not
    None else _DEFAULT_ROLE`` fallback in ``Tabularium.context`` makes this
    test's ``pytest.raises`` block fail (no exception raised) -- see the
    task verdict for the captured failure output.
    """
    from mcp.server.mcpserver.exceptions import ToolError as FastMCPToolError

    app = await _make_app(extra_scope="global")
    with pytest.raises(FastMCPToolError, match="requires a role"):
        await app.call_tool("context", {"scope": "global", "budget": 100})


async def test_context_two_registered_roles_return_identical_payloads() -> None:
    """The Curator is deferred out of this slice (this plan's locked
    decision) -- with an UNHINTED Fact, every registered role resolves to
    the same (empty) selection: explicit enumeration only (curation block) means an unhinted
    record reaches NO role, so ``decider`` and
    ``implementer`` agree by both being empty, not by both being full.
    Pinned here so the day the Curator makes them differ, this test is the
    thing that has to be updated deliberately."""
    engine, adapter, _ = _make_engine_components()
    await engine.initialize()
    await adapter.create_scope(Scope("global"))
    await adapter.create_role("implementer")
    hot = _hot_fact(content="an unhinted fact routed to no role", audience_hint=None)
    await adapter.upsert(hot, _vec())

    app = make_app(engine)
    r1 = await app.call_tool("context", {"scope": "global", "budget": 1000, "role": "decider"})
    r2 = await app.call_tool("context", {"scope": "global", "budget": 1000, "role": "implementer"})

    assert ContextPayload(**_s(r1)) == ContextPayload(**_s(r2))


async def test_context_negative_budget_raises_tool_error() -> None:
    """Negative budget is a pydantic validation failure surfaced by
    FastMCP, not a silent clamp (this brief's acceptance criteria)."""
    from mcp.server.mcpserver.exceptions import ToolError as FastMCPToolError

    app = await _make_app(extra_scope="global")
    with pytest.raises(FastMCPToolError):
        await app.call_tool("context", {"scope": "global", "budget": -1})


async def test_context_non_integer_budget_raises_tool_error() -> None:
    """A non-coercible string budget is a pydantic validation failure
    surfaced by FastMCP, not a silent clamp."""
    from mcp.server.mcpserver.exceptions import ToolError as FastMCPToolError

    app = await _make_app(extra_scope="global")
    with pytest.raises(FastMCPToolError):
        await app.call_tool("context", {"scope": "global", "budget": "not-a-number"})


async def test_context_fractional_float_budget_raises_tool_error() -> None:
    """A fractional float (3.5) cannot coerce to ``int`` -- also a pydantic
    validation failure surfaced by FastMCP, not silently truncated."""
    from mcp.server.mcpserver.exceptions import ToolError as FastMCPToolError

    app = await _make_app(extra_scope="global")
    with pytest.raises(FastMCPToolError):
        await app.call_tool("context", {"scope": "global", "budget": 3.5})


# ---------------------------------------------------------------------------
# Two-tier error split (SDK v2 migration, 2026-08-12, scope 3 step 6)
# ---------------------------------------------------------------------------


async def test_two_tier_category_error_still_isError_true_message_intact() -> None:
    """(a) A handler raising ``EngineToolError`` still yields ``isError=true``
    with the message intact -- the split adds a tier only at the *uncaught*
    end (locked decision); everything already caught (``EngineToolError``,
    ``ValueError``) is unchanged."""
    from mcp.server.mcpserver.exceptions import ToolError as FastMCPToolError

    app = await _make_app()  # no extra scope -- "missing-scope" not registered
    with pytest.raises(FastMCPToolError) as exc_info:
        await app.call_tool("recall", {"query": "x", "scope": "missing-scope"})
    assert "missing-scope" in str(exc_info.value)


async def test_two_tier_unexpected_exception_becomes_protocol_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) A handler raising an unexpected ``KeyError`` (the
    concurrent-bury class named in the brief) now yields a JSON-RPC
    protocol error (``MCPError``, ``code=INTERNAL_ERROR``), NOT an
    ``isError=true`` result -- the new behaviour this migration adds.
    Repo falsifiability bar: this test was shown RED (the KeyError
    surfacing as ``FastMCPToolError``/``isError=true`` instead) with
    ``@_protocol_error_boundary`` temporarily removed from ``recall``,
    before the decorator landed permanently."""
    from mcp.shared.exceptions import MCPError
    from mcp.types import INTERNAL_ERROR

    engine, adapter, _clock = _make_engine_components()
    await engine.initialize()
    await adapter.create_scope(Scope("global"))

    async def _boom(*args: object, **kwargs: object) -> None:
        raise KeyError("simulated concurrent-bury race")

    monkeypatch.setattr(engine, "recall", _boom)

    app = make_app(engine)
    with pytest.raises(MCPError) as exc_info:
        await app.call_tool("recall", {"query": "x", "scope": "global"})
    assert exc_info.value.code == INTERNAL_ERROR
