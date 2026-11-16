"""MCP transport selection tests (server deployment)

Covers env-driven stdio/http selection (``TABULARIUM_MCP_TRANSPORT`` /
``_HOST`` / ``_PORT``), the loud-failure requirement for an invalid
transport value, ``main()``'s http-branch host/port/transport_security
threading into ``run_streamable_http_async`` (SDK v2 migration,
2026-08-12 -- v2's ``MCPServer`` constructor dropped these three params;
``make_app`` no longer takes them either, they only exist as call-time
kwargs on ``run_streamable_http_async`` now), and the DNS-rebinding
allowlist builder (``_transport_security_for_host``).

A pytest suite cannot prove a real socket binds and answers without a 421
Misdirected Request -- that proof is a live curl against an actually-bound
server (see the build brief's Verify section). This file proves the
plumbing that feeds that server: the env parsing, the fail-loudly
contract, and that ``main()``'s http branch threads the right values into
the call that actually binds the socket.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from tabularium import server as server_mod
from tabularium.config import EngineConfig, TransportConfig
from tabularium.engine import Tabularium
from tabularium.server import (
    _transport_security_for_host,
    make_app,
    transport_config_from_env,
)
from tabularium.storage.memory import MemoryAdapter
from tests.support.doubles import FakeEmbedder, TestClock

_DIM = 16


def _engine() -> Tabularium:
    adapter = MemoryAdapter(embedding_model="fake/nomic", dim=_DIM)
    embedder = FakeEmbedder(dim=_DIM)
    clock = TestClock(datetime(2025, 1, 1, tzinfo=UTC))
    cfg = EngineConfig(candidate_k=1000, gray_floor=0.80)
    return Tabularium(adapter, embedder, clock, cfg)


# ---------------------------------------------------------------------------
# transport_config_from_env
# ---------------------------------------------------------------------------


def test_transport_config_defaults_to_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TABULARIUM_MCP_TRANSPORT", raising=False)
    monkeypatch.delenv("TABULARIUM_MCP_HOST", raising=False)
    monkeypatch.delenv("TABULARIUM_MCP_PORT", raising=False)
    assert transport_config_from_env() == TransportConfig(
        transport="stdio",
        host="127.0.0.1",
        port=8787,
    )


def test_transport_config_reads_http_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TABULARIUM_MCP_TRANSPORT", "http")
    monkeypatch.setenv("TABULARIUM_MCP_HOST", "192.0.2.1")
    monkeypatch.setenv("TABULARIUM_MCP_PORT", "8787")
    assert transport_config_from_env() == TransportConfig(
        transport="http",
        host="192.0.2.1",
        port=8787,
    )


def test_transport_config_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TABULARIUM_MCP_TRANSPORT", "carrier-pigeon")
    with pytest.raises(ValueError, match="TABULARIUM_MCP_TRANSPORT"):
        transport_config_from_env()


def test_main_fails_loudly_for_invalid_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid transport raises synchronously, before any adapter/embedder
    setup -- never a silent fall back to stdio."""
    monkeypatch.setenv("TABULARIUM_MCP_TRANSPORT", "smoke-signal")
    with pytest.raises(ValueError, match="TABULARIUM_MCP_TRANSPORT"):
        server_mod.main()


# ---------------------------------------------------------------------------
# make_app -- no longer takes host/port/transport_security (v2)
# ---------------------------------------------------------------------------


def test_make_app_takes_no_transport_kwargs() -> None:
    """v2's ``MCPServer`` constructor dropped host/port/transport_security
    (SDK v2 migration, 2026-08-12); ``make_app`` no longer accepts them
    either -- confirm the stdio-only call shape every existing caller uses
    still builds a working app."""
    app = make_app(_engine())
    assert app is not None


# ---------------------------------------------------------------------------
# main() http branch -- threads host/port/transport_security into
# run_streamable_http_async (v2 moved them there from the constructor)
# ---------------------------------------------------------------------------


def test_main_threads_http_host_port_security_into_run_streamable_http_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove main()'s http branch threads TABULARIUM_MCP_HOST/_PORT and the
    computed transport_security allowlist into
    ``run_streamable_http_async(host=, port=, transport_security=,
    stateless_http=True)`` -- the call-time seam v2 moved host/port/
    transport_security to (scope 1 steps 7-8), plus ``stateless_http=True``
    itself: without it, a bare request carrying no
    ``MCP-Protocol-Version`` header falls into v2's *legacy* stateful
    session path (``StreamableHTTPSessionManager._handle_stateful_request``)
    and a session-less ``tools/call`` 400s with "Missing session ID" --
    reproduced locally on loopback before this fix landed. A client sending
    the modern ``MCP-Protocol-Version: 2026-07-28`` header always reaches
    the session-free path regardless of this flag; ``stateless_http=True``
    is what makes a bare, header-less request (exactly what the brief's own
    scope 4 Verify curl recipe sends) session-free too, matching the
    locked "Stateless transport" decision."""
    monkeypatch.setenv("TABULARIUM_MCP_TRANSPORT", "http")
    monkeypatch.setenv("TABULARIUM_MCP_HOST", "192.0.2.1")
    monkeypatch.setenv("TABULARIUM_MCP_PORT", "8787")
    monkeypatch.setenv("TABULARIUM_STORE_PATH", "/tmp/unused-by-this-test")

    calls: dict[str, Any] = {}

    class _StubApp:
        async def run_streamable_http_async(self, **kwargs: Any) -> None:
            calls.update(kwargs)

    class _StubEmbedder:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    async def _fake_open_adapter(*_args: Any, **_kwargs: Any) -> MemoryAdapter:
        return MemoryAdapter(embedding_model="fake/nomic", dim=_DIM)

    async def _fake_initialize(_self: Tabularium) -> None:
        return None

    def _fake_make_app(_engine: Tabularium) -> _StubApp:
        return _StubApp()

    monkeypatch.setattr(server_mod, "open_adapter", _fake_open_adapter)
    monkeypatch.setattr(server_mod, "LitellmEmbedder", _StubEmbedder)
    monkeypatch.setattr(Tabularium, "initialize", _fake_initialize)
    monkeypatch.setattr(server_mod, "make_app", _fake_make_app)

    server_mod.main()

    assert calls["host"] == "192.0.2.1"
    assert calls["port"] == 8787
    assert calls["stateless_http"] is True
    security = calls["transport_security"]
    assert security is not None
    assert security.allowed_hosts == ["192.0.2.1:8787"]
    assert security.allowed_origins == ["http://192.0.2.1:8787"]


# ---------------------------------------------------------------------------
# _transport_security_for_host
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_transport_security_none_for_loopback(host: str) -> None:
    """Loopback hosts defer to FastMCP's own auto-enabled allowlist."""
    assert _transport_security_for_host(host, 8787) is None


def test_transport_security_targeted_allowlist_for_wg_host() -> None:
    security = _transport_security_for_host("192.0.2.1", 8787)
    assert security is not None
    assert security.enable_dns_rebinding_protection is True
    assert security.allowed_hosts == ["192.0.2.1:8787"]
    assert security.allowed_origins == ["http://192.0.2.1:8787"]
