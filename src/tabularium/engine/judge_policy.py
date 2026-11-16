"""Which MCP tools a judge seat may be granted, and the grant itself.

A leaf module: ``rubric.py`` validates seats against this policy and ``judge.py`` renders the grant
into each CLI's permission syntax, so neither imports the other for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: A tool that must never appear in a judge seat's MCP allowlist. ``recall`` has side effects: it
#: reinforces a hit and can flip archived to active on its archived-fallback branch, and the MCP
#: layer has no ``dry_run``, so that write lands regardless.
FORBIDDEN_MCP_TOOL = "recall"

#: Every MCP server name and tool name must match this shape: a bare lowercase identifier with no
#: wildcard, glob character, separator, whitespace or control character. Claude's
#: ``--allowed-tools`` accepts pattern syntax, and ``mcp__<server>__*`` was observed to grant every
#: tool on the server, a destructive one included, even though the allowlist never named it. Real
#: Tabularium tool names are lowercase snake_case.
BARE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")

#: The only MCP tools a judge seat may be granted: those verified side-effect-free. A positive
#: allowlist rather than a blocklist of dangerous verbs, so any tool added to the MCP surface later
#: is refused until someone verifies it and widens this set. A well-formed name that is not in this
#: set passes the shape check, so this set is what blocks it. ``recall`` is the exception: it is
#: rejected earlier, by ``FORBIDDEN_MCP_TOOL``.
JUDGE_SAFE_MCP_TOOLS: frozenset[str] = frozenset({"decisions_query", "decision", "context"})

#: Companion to ``JUDGE_SAFE_MCP_TOOLS``, rendered by ``judge.py`` into the claude seat's
#: ``--disallowed-tools`` and the codex seat's ``disabled_tools``.
#:
#: Security-relevant. ``--allowed-tools`` is not exhaustive on the claude side: claude merges
#: ambient permission settings from the subprocess's inherited working directory on top of it, so
#: an ambient allow entry could run ``recall`` for a seat granted only ``context``. An explicit deny
#: closes that, because deny takes precedence over allow.
#:
#: Hand-maintained, not computed as a complement of the safe set: that would need this module to
#: know the server's full tool list, which only ``server.py`` has, and ``server.py`` already imports
#: the engine layer. A test in ``tests/test_rubric.py``
#: (``test_judge_deny_list_covers_every_registered_tool_outside_the_safe_set``) fails when a
#: registered tool is in neither set.
JUDGE_DENY_MCP_TOOLS: frozenset[str] = frozenset({"recall"})


@dataclass(frozen=True, slots=True)
class McpGrant:
    """The MCP access one CLI seat has: a server URL, the name it is registered under for the call,
    and the bare tool names it may call.

    Intent only, never a CLI-specific config shape: each CLI call in ``judge.py`` renders it into
    that CLI's own flags. Holding one does not mean it passed ``validate_mcp_grant``.
    """

    server_url: str
    server_name: str
    allowed_tools: tuple[str, ...]


def reject_non_bare_identifier(value: str, *, seat_label: str, field: str) -> None:
    """Raise ``ValueError`` unless ``value`` matches ``BARE_IDENTIFIER``.

    Used for the server name and every allowlist entry. Both are interpolated into a CLI's
    permission or config syntax in ``judge.py``, so a wildcard, separator or control character must
    not reach it.
    """
    if not BARE_IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"judges.{seat_label}: {field} {value!r} is not a bare lowercase identifier "
            "([a-z][a-z0-9_]*) -- wildcards, glob metacharacters, dots, whitespace and control "
            "characters are all rejected. '*' was verified live to grant every tool on the "
            "server (including a destructive one) through claude's own Bash(...)-style "
            "--allowed-tools pattern matching, silently widening the grant past what the tuple "
            "appears to say."
        )


def validate_mcp_grant(grant: McpGrant, *, seat_label: str) -> None:
    """Fail-closed validation of one grant. ``seat_label`` (``"seat_a"``, ``"seat_b"``,
    ``"backstop"`` or a caller's own tag) is named in every message.

    Raises ``ValueError`` unless the server name is a bare identifier and the allowlist is
    non-empty, with every entry a bare identifier, not ``recall``, and in ``JUDGE_SAFE_MCP_TOOLS``.

    Two callers share this one function: ``JudgesConfig``'s validator, which has the real seat
    names, and ``judge.py``'s ``build_judge_adapter``, which turns a grant into a live CLI
    permission and cannot assume its caller went through ``JudgesConfig``.
    """
    reject_non_bare_identifier(grant.server_name, seat_label=seat_label, field="mcp_server_name")
    if not grant.allowed_tools:
        raise ValueError(
            f"judges.{seat_label}: mcp_server_url is set ({grant.server_url!r}) but "
            "mcp_allowed_tools is empty -- a judge seat must never open an MCP connection "
            "without an explicit, non-empty tool allowlist"
        )
    for tool in grant.allowed_tools:
        reject_non_bare_identifier(tool, seat_label=seat_label, field="mcp_allowed_tools entry")
        if tool == FORBIDDEN_MCP_TOOL:
            raise ValueError(
                f"judges.{seat_label}: mcp_allowed_tools must not include "
                f"{FORBIDDEN_MCP_TOOL!r} -- it is not side-effect-free: it reinforces a hit and "
                "can flip archived->active on its archived-fallback branch, and dry_run does not "
                "exist at the MCP layer, so that write would land regardless of any dry-run "
                "intent. Use decisions_query / decision / context instead."
            )
        if tool not in JUDGE_SAFE_MCP_TOOLS:
            raise ValueError(
                f"judges.{seat_label}: mcp_allowed_tools entry {tool!r} is not one of the "
                f"tools established side-effect-free {sorted(JUDGE_SAFE_MCP_TOOLS)!r} -- a "
                "judge seat may only be granted a tool this codebase has actually verified "
                "read-only. Widening this set requires verifying the new tool is read-only, not "
                "just naming it in a rubric."
            )
