# 0002 — MCP interface built on the official SDK's FastMCP

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted

Tabularium's MCP surface is built on the **official MCP Python SDK's FastMCP** (`mcp`, pinned `>=1.27,<2`) — not the standalone `fastmcp` 3.x (Prefect-owned), nor the low-level `Server`. For a small surface (4 tools + a `decision://{id}` resource template, stdio-first), the official FastMCP is the exact fit: type-hinted functions returning pydantic models, `@mcp.resource` templates, a lean (~10) dependency tree, Anthropic-backed and production-recommended.

## Considered options

- **Standalone `fastmcp` 3.4.4 (Prefect)** — more actively developed, cleaner test ergonomics (`Client(transport=mcp)`), insulated from the eventual `mcp` v2 import break; but ~2× the dependencies + Prefect coupling + auth/OpenAPI/Apps machinery this surface won't use. Rejected as overkill. It shares the same `@mcp.tool`/`@mcp.resource` API, so it's a low-cost swap later.
- **Low-level `mcp` Server** — raw JSON-RPC boilerplate, wrong direction for a 4-tool surface. Rejected.
- **Official SDK FastMCP** — chosen.

## Consequences

- Pin `mcp>=1.27,<2`: the official SDK v1.x is in maintenance mode; the `mcp.server.fastmcp` import path may change when v2 stabilizes. Do NOT adopt the `mcp` 2.0.0bN pre-release.
- Standalone `fastmcp` 3.x is the escape hatch if the surface grows to need HTTP transport, auth providers, or its cleaner testing API — same decorators, low-cost migration.
- Advances T03 (interface). The tool-surface *design* (exact args/returns/error modes) is separate and still pending — it depends on the hypomnema schema + the curation model.
