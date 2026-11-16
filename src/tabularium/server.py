"""Tabularium MCP server: a FastMCP projection of the engine, plus the ``tabularium-mcp`` entry
point.

``make_app(engine)`` registers the tools and the ``decision://{id}`` resource on an ``MCPServer``.
It registers the four read tools and the resource; the v1 store takes no writes through MCP.
A tool body forwards its arguments to the matching ``Tabularium`` method and returns the result;
memory semantics and invariants live in ``tabularium.engine``. The agent-facing description
strings live in ``tabularium.tool_text`` and are part of the wire contract.

Invariants:

- Every tool handler is ``async def``. A sync handler would run on a worker thread, where the
  engine's ``asyncio.Lock`` does not exclude it. ``test_all_tools_are_async``
  (``tests/test_server.py``) pins it.
- Run one server process per store. The engine's write lock is a per-process ``asyncio.Lock``,
  and the dedup gate's check-then-write relies on one process holding it.

Errors:

- ``_protocol_error_boundary`` wraps every tool handler. It re-raises an ``EngineToolError`` or
  ``ValueError`` as FastMCP's ``ToolError``, which the client sees as a tool result with
  ``isError=true``. Any other exception becomes ``MCPError(INTERNAL_ERROR)``, a JSON-RPC protocol
  error. ``MCPError`` and ``ToolError`` pass through untouched.
- Bounces and degraded recall are ordinary results, not errors.
- The ``decision://{id}`` resource is not wrapped by the boundary. An unknown or buried id
  raises ``ResourceNotFoundError``.

Environment, read at startup by ``main()`` through the functions in ``tabularium.config``, which
this module re-exports: ``config_from_env`` (the store and embedder), ``transport_config_from_env``,
``engine_config_from_env`` and ``build_assessor_from_env``. The variables and defaults are listed in
the ``tabularium.config`` docstring. An unparseable value raises ``ValueError`` before the store is
opened. ``TYPESAFE_API_KEY``, when non-empty, makes ``main()`` build a ``JevAssessor``; otherwise
the engine runs with no Assessor.

Transport. ``stdio`` calls ``run_stdio_async()``. ``http`` calls ``run_streamable_http_async``
with ``stateless_http=True``, so a request without an MCP protocol-version header needs no
session, and with the explicit allowlist from ``_transport_security_for_host``.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import uuid as _uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError
from mcp.server.mcpserver.exceptions import ToolError as FastMCPToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.exceptions import MCPError
from mcp.types import INTERNAL_ERROR, ToolAnnotations
from pydantic import Field

from tabularium.config import (
    build_assessor_from_env as build_assessor_from_env,
)
from tabularium.config import (
    config_from_env as config_from_env,
)
from tabularium.config import (
    engine_config_from_env as engine_config_from_env,
)
from tabularium.config import (
    transport_config_from_env as transport_config_from_env,
)
from tabularium.domain import HypomnemaId
from tabularium.embedding import LitellmEmbedder
from tabularium.engine import Tabularium
from tabularium.engine import ToolError as EngineToolError
from tabularium.engine.assessor import Assessor
from tabularium.results import (
    ContextPayload,
    DecisionLink,
    DecisionView,
    RecallResult,
)
from tabularium.storage import open_adapter
from tabularium.tool_text import (
    CONTEXT_DESCRIPTION,
    DECISION_DESCRIPTION,
    DECISION_RESOURCE_DESCRIPTION,
    DECISIONS_QUERY_DESCRIPTION,
    RECALL_DESCRIPTION,
    SERVER_INSTRUCTIONS,
)


def _transport_security_for_host(host: str, port: int) -> TransportSecuritySettings | None:
    """Build the DNS-rebinding allowlist for the http transport.

    Loopback hosts return ``None``: the SDK enables its own allowlist for them. Any other host
    gets an allowlist of exactly ``{host}:{port}`` and ``http://{host}:{port}``, because
    ``None`` there would leave the check disabled.
    """
    if host in ("127.0.0.1", "localhost", "::1"):
        return None
    hostport = f"{host}:{port}"
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[hostport],
        allowed_origins=[f"http://{hostport}"],
    )


# ---------------------------------------------------------------------------
# Protocol-error boundary
# ---------------------------------------------------------------------------


def _protocol_error_boundary[**P, T](fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """Translate a tool handler's exceptions into the two MCP error tiers.

    ``MCPError`` and ``FastMCPToolError`` propagate untouched. An ``EngineToolError`` or
    ``ValueError`` becomes ``FastMCPToolError(str(exc))``, which the client sees as a tool result
    with ``isError=true``. Any other ``Exception`` becomes
    ``MCPError(code=INTERNAL_ERROR, message=str(exc))``, a JSON-RPC protocol error, instead of the
    SDK's own handler turning it into ``isError=true`` text. ``functools.wraps`` keeps the wrapped
    signature, which the SDK reads to build the tool's input schema.
    """

    @functools.wraps(fn)
    async def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await fn(*args, **kwargs)
        except (MCPError, FastMCPToolError):
            raise
        except (EngineToolError, ValueError) as exc:
            raise FastMCPToolError(str(exc)) from exc
        except Exception as exc:
            raise MCPError(code=INTERNAL_ERROR, message=str(exc)) from exc

    return _wrapper


# ---------------------------------------------------------------------------
# FastMCP application factory
# ---------------------------------------------------------------------------


def _register_read_tools(app: MCPServer[None], engine: Tabularium) -> None:
    """Register the four read tools and the ``decision://{id}`` resource on ``app``."""

    # ------------------------------------------------------------------
    # recall
    # ------------------------------------------------------------------

    @app.tool(
        annotations=ToolAnnotations(read_only_hint=True),
        description=RECALL_DESCRIPTION,
    )
    @_protocol_error_boundary
    async def recall(
        query: str,
        scope: str,
        limit: int = 10,
        include_invalid: bool = False,
        include_archived: bool = False,
    ) -> RecallResult:
        return await engine.recall(
            query,
            scope=scope,
            limit=limit,
            include_invalid=include_invalid,
            include_archived=include_archived,
        )

    # ------------------------------------------------------------------
    # decision
    # ------------------------------------------------------------------

    @app.tool(
        annotations=ToolAnnotations(read_only_hint=True),
        description=DECISION_DESCRIPTION,
    )
    @_protocol_error_boundary
    async def decision(id: HypomnemaId) -> DecisionView:
        return await engine.decision(id)

    # ------------------------------------------------------------------
    # decisions_query
    # ------------------------------------------------------------------

    @app.tool(
        annotations=ToolAnnotations(read_only_hint=True),
        description=DECISIONS_QUERY_DESCRIPTION,
    )
    @_protocol_error_boundary
    async def decisions_query(
        query: str,
        scope: str,
        limit: int = 10,
        include_superseded: bool = False,
        include_invalid: bool = False,
    ) -> list[DecisionLink]:
        return await engine.decisions_query(
            query,
            scope=scope,
            limit=limit,
            include_superseded=include_superseded,
            include_invalid=include_invalid,
        )

    # ------------------------------------------------------------------
    # context
    # ------------------------------------------------------------------

    @app.tool(
        annotations=ToolAnnotations(read_only_hint=True),
        description=CONTEXT_DESCRIPTION,
    )
    @_protocol_error_boundary
    async def context(
        scope: str,
        budget: Annotated[int, Field(ge=0)],
        role: str | None = None,
    ) -> ContextPayload:
        return await engine.context(scope, budget, role=role)

    # ------------------------------------------------------------------
    # decision://{id} resource
    # ------------------------------------------------------------------

    @app.resource(
        "decision://{id}",
        mime_type="text/markdown",
        description=DECISION_RESOURCE_DESCRIPTION,
    )
    async def decision_resource(id: str) -> str:
        """Serve ``DecisionView.rendered`` for ``decision://{id}``.

        A malformed id raises ``FastMCPToolError``; an id with no Decision raises
        ``ResourceNotFoundError``.
        """
        try:
            parsed_id = HypomnemaId(_uuid.UUID(id))
        except Exception as exc:
            raise FastMCPToolError(f"Invalid decision id: {id!r}") from exc

        try:
            view = await engine.decision(parsed_id)
        except EngineToolError as exc:
            # A well-formed id with no Decision (unknown or buried) is resource-not-found,
            # which the SDK reports as -32602.
            raise ResourceNotFoundError(str(exc)) from exc

        return view.rendered


def make_app(
    engine: Tabularium,
) -> MCPServer[None]:
    """Build an ``MCPServer`` whose tools call ``engine``.

    The app holds no state beyond the engine reference. ``host``, ``port`` and transport
    security are passed to ``run_streamable_http_async`` by ``main()``, not here.

    The app registers ``recall``, ``decisions_query``, ``decision`` and ``context`` plus the
    ``decision://{id}`` resource, and carries ``SERVER_INSTRUCTIONS``. No tool writes to the store.
    """

    @asynccontextmanager
    async def _noop_lifespan(
        _app: MCPServer[None],
    ) -> AsyncGenerator[None, None]:
        """No-op lifespan; the caller has already initialized the engine."""
        yield

    app: MCPServer[None] = MCPServer(
        "Tabularium",
        instructions=SERVER_INSTRUCTIONS,
        lifespan=_noop_lifespan,
    )

    _register_read_tools(app, engine)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Send the ``"tabularium"`` logger's INFO lines to stderr.

    Only that logger is touched, and only when it has no handler yet.
    """
    log = logging.getLogger("tabularium")
    if log.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def main() -> None:
    """``tabularium-mcp`` entry point: build the engine from the environment and serve it.

    The transport and engine settings are parsed before the store is opened, so an invalid
    value raises ``ValueError`` without touching the store.
    """
    _configure_logging()
    transport_cfg = transport_config_from_env()
    engine_cfg = engine_config_from_env()
    store_cfg = config_from_env()
    assessor: Assessor | None = build_assessor_from_env(engine_cfg.assessor_model)

    async def _run() -> None:
        adapter = await open_adapter(
            store_cfg.adapter,
            embedding_model=store_cfg.model,
            dim=store_cfg.dim,
            store_path=store_cfg.path,
        )
        embedder = LitellmEmbedder(
            model=store_cfg.model,
            dim=store_cfg.dim,
            api_base=store_cfg.api_base,
        )
        eng = Tabularium(adapter, embedder, config=engine_cfg, assessor=assessor)
        await eng.initialize()

        if transport_cfg.transport == "http":
            host = transport_cfg.host
            port = transport_cfg.port
            app = make_app(eng)
            await app.run_streamable_http_async(
                host=host,
                port=port,
                transport_security=_transport_security_for_host(host, port),
                stateless_http=True,
            )
        else:
            app = make_app(eng)
            await app.run_stdio_async()

    asyncio.run(_run())
