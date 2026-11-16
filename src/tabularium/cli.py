"""``tabularium`` command-line entry point.

Commands:

  tabularium sync <path>                  export every Decision to <path>/<id>.md
  tabularium scope add <name>             register a scope
  tabularium facet add <name>             register a purpose-facet value
  tabularium role add <name>              register a role
  tabularium reclaim [--yes]              irreversible version pruning (disk only)
  tabularium doctor                       facet coverage, dead vocabulary, rows awaiting
                                          embedding, projection drift
  tabularium backfill                     run one backfill pass
  tabularium supersede <pred> <succ> --reason <class> [--note TEXT]
                                          draw a Fact-arm supersede edge by hand
  tabularium clean --scope <name> [--dry-run] [--rubric PATH]
                                          run one Cleaner pass by hand
  tabularium clean --scope <name> --history [--limit N]
                                          print pass-history rows, run no pass

Store selection. The environment variables are the ones ``config.config_from_env`` reads, with
the same defaults:

  TABULARIUM_STORE_PATH   store directory; required for the lancedb adapter
  TABULARIUM_ADAPTER      "lancedb" (default) or "memory"
  TABULARIUM_MODEL        litellm embedding model (default ollama/nomic-embed-text)
  TABULARIUM_DIM          embedding dimensions (default 768)
  TABULARIUM_API_BASE     embedder API base (default http://localhost:11434)

``--store`` overrides ``TABULARIUM_STORE_PATH`` and ``--adapter`` overrides
``TABULARIUM_ADAPTER``. A bad ``--adapter`` or ``TABULARIUM_ADAPTER`` raises ``ValueError`` from
``config_from_env`` before any command runs; it is not a ``CliError``.

Dispatch. Each subparser sets ``handler`` through ``set_defaults``; ``main`` runs
``asyncio.run(args.handler(args, store))`` and exits with the integer it returns. A command that
fails raises ``CliError``; ``main`` prints ``error: <message>`` to stderr and exits 1.

Every command opens the adapter directly through ``open_store`` and builds no ``Tabularium``
engine. Only ``backfill`` and ``clean`` build an embedder (``_build_embedder``). Function-local
imports defer the engine, storage, embedding, rendering and pydantic-model imports to command
run time, because they cost from 0.2 s to several seconds and ``tabularium --help`` must not
pay it. ``doctor`` and ``clean`` read the engine settings through
``config.engine_config_from_env``.

- ``sync`` never runs git. It rewrites a file only when its bytes differ, and removes a
  UUID-named ``.md`` file only when the store holds a Decision tombstone for that id. It exits 1
  without removing anything when the store looks misconfigured (memory adapter, or a
  ``lancedb`` path that did not exist) and the scan found no Decisions while the directory
  already holds UUID-named files.
- ``reclaim`` prints its no-other-writer precondition and asks for ``y`` unless ``--yes`` is
  given; declining exits 1.
- ``doctor`` exits 0 after printing its report. It builds the report with
  ``engine_config_from_env()``; a malformed ``TABULARIUM_*`` engine value prints
  ``error: <message>`` and exits 1 before the store is opened. See ``engine/doctor.py``.
- ``supersede`` calls ``engine.dispositions.supersede`` with a fresh ``asyncio.Lock`` and
  ``SystemClock``; it needs no embedder. ``--reason`` is parsed to a ``SupersessionReason`` by
  argparse, which exits 2 on any other value. It exits 1 on a malformed id or a ``ToolError``.
- ``backfill`` prints the concurrent-write warning, runs ``populate_content_keys`` and then one
  pass, and exits 1 when any row failed.
- ``clean`` runs ``engine.report.run_clean_pass``, the function the nightly scheduler uses. It
  writes from a second OS process; the engine's write lock and the adapter's locks are
  per-process, so nothing serializes it against a running server, and it prints a warning first
  unless ``--dry-run`` is given. A dry run judges for real and writes only the pass-history row; see
  ``engine/clean.py``. ``--rubric`` loads a rubric from a TOML path and requires ``--dry-run``,
  because a live pass would stamp shelve reasons with a rubric version that exists nowhere else.
  ``--history`` and ``--rubric`` together exit 1. The memory adapter is refused before the store
  is opened. With ``TABULARIUM_READ_ONLY`` true, a pass or dry run is refused with one line and
  exit 1 before the store is opened; ``--history`` still prints. ``clean`` exits 1 when the
  outcome is ``blocked`` or a ``ToolError`` is raised.
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
import re
import sys
import uuid as _uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, get_args

from tabularium.clock import SystemClock

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from tabularium.clock import Clock
    from tabularium.config import StoreConfig
    from tabularium.domain import BatchOutcome, HypomnemaId, Scope, SupersessionReason
    from tabularium.embedding import Embedder
    from tabularium.engine.assessor import Assessor
    from tabularium.engine.clean import CleanSummary
    from tabularium.engine.falsity import FalsityAxisSummary
    from tabularium.engine.judge import JudgeAdapter
    from tabularium.engine.relevance import RelevanceAxisSummary
    from tabularium.engine.revisit import RevisitAxisSummary
    from tabularium.engine.rubric import Rubric
    from tabularium.results import DoctorReport
    from tabularium.storage.port import StorageAdapter

    StoreAdapter = StorageAdapter
    Handler = Callable[[argparse.Namespace, StoreConfig], Awaitable[int]]

_REASON_METAVAR = "{correction,changed,elaborated,moved}"


class CliError(Exception):
    """A command failure that ``main`` prints as ``error: <message>`` before exiting 1."""


def _is_uuid_stem(stem: str) -> bool:
    """Return True if ``stem`` parses as a UUID."""
    try:
        _uuid.UUID(stem)
        return True
    except ValueError:
        return False


def _hypomnema_id(raw: str) -> HypomnemaId:
    """Parse the UUID string ``raw`` into a ``HypomnemaId``; raises ``ValueError`` if malformed."""
    # Deferred: the domain module imports pydantic (about 0.2 s).
    from tabularium.domain import HypomnemaId

    return HypomnemaId(_uuid.UUID(raw))


# ---------------------------------------------------------------------------
# Store access
# ---------------------------------------------------------------------------


async def open_store(store: StoreConfig) -> StoreAdapter:
    """Open the adapter ``store`` selects; an empty ``store.path`` counts as unset.

    Raises ``ValueError`` from ``open_adapter`` when ``lancedb`` has no path.
    """
    # Deferred: the storage package imports lancedb (about 2 s).
    from tabularium.storage import open_adapter

    return await open_adapter(
        store.adapter,
        embedding_model=store.model,
        dim=store.dim,
        store_path=store.path or None,
    )


def _build_embedder(store: StoreConfig) -> Embedder:
    """Build a ``LitellmEmbedder`` from the resolved store config, including ``api_base``."""
    # Deferred: the embedding module imports litellm (about 3.5 s).
    from tabularium.embedding import LitellmEmbedder

    return LitellmEmbedder(model=store.model, dim=store.dim, api_base=store.api_base)


def _concurrent_write_warning(command: str) -> str:
    """The warning ``command`` prints before it writes from a second OS process."""
    return (
        f"{command}: this command writes from a second OS process.\n"
        "Nothing serializes it against a running MCP server: the engine write lock "
        "(engine.core.Tabularium._write_lock) and the adapters' locks are per-process, and this "
        "invocation builds its own. A write here can interleave with one the server makes.\n"
        "Run it against a local store or a stopped service, never against a store a live "
        "tabularium-mcp instance is writing to."
    )


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ExportResult:
    """What ``_export_decisions`` wrote: the counts and the ids it fetched and rendered."""

    written: int
    unchanged: int
    exported_ids: frozenset[str]


def _store_is_suspect(store: StoreConfig) -> bool:
    """True when an empty scan of ``store`` says nothing about what the export directory holds.

    Must run before the store is opened: opening creates a missing ``lancedb`` path.
    """
    return (
        store.adapter == "memory"
        or store.path is None
        or not pathlib.Path(store.path).exists()
    )


async def _export_decisions(adapter: StoreAdapter, path: pathlib.Path) -> _ExportResult:
    """Write ``path/<id>.md`` for every Decision, skipping files whose bytes already match."""
    # Deferred: the domain, engine and render modules import pydantic, litellm and lancedb.
    from tabularium.domain import Decision
    from tabularium.engine.decision_read import load_revalidation_records
    from tabularium.render import render_decision

    scan_ids: set[str] = set()
    async for row in await adapter.scan():
        if row.type == "decision":
            scan_ids.add(str(row.id))

    written = 0
    unchanged = 0
    exported_ids: set[str] = set()

    for str_id in scan_ids:
        try:
            record = await adapter.get_by_id(_hypomnema_id(str_id))
        except KeyError:
            continue

        if not isinstance(record, Decision):
            continue

        revalidations = await load_revalidation_records(record.id, adapter)
        rendered = render_decision(record, revalidations=revalidations)
        file_path = path / f"{str_id}.md"
        new_bytes = rendered.encode("utf-8")
        exported_ids.add(str_id)

        if file_path.exists() and file_path.read_bytes() == new_bytes:
            unchanged += 1
            continue

        file_path.write_bytes(new_bytes)
        written += 1

    return _ExportResult(written, unchanged, frozenset(exported_ids))


def _guard_suspect_store(
    store: StoreConfig,
    exported_ids: frozenset[str],
    existing_files: Sequence[pathlib.Path],
) -> None:
    """Raise ``CliError`` when a suspect store exported nothing and the directory has UUID files.

    That combination usually means ``--store``, ``--adapter`` or ``TABULARIUM_STORE_PATH`` is
    misconfigured, and the prune that follows would delete a real export.
    """
    if exported_ids or not existing_files:
        return

    adapter_msg = (
        "the memory adapter never persists data"
        if store.adapter == "memory"
        else "the store path did not exist before this run"
    )
    raise CliError(
        f"sync found 0 decisions ({adapter_msg}) but the output "
        f"directory already contains {len(existing_files)} UUID-named "
        "Markdown file(s).  This usually means --store / --adapter / "
        "TABULARIUM_STORE_PATH is misconfigured.  No files were removed."
    )


async def _prune_buried(
    adapter: StoreAdapter,
    existing_files: Sequence[pathlib.Path],
    exported_ids: frozenset[str],
) -> int:
    """Remove each unexported file whose id has a Decision tombstone; returns the count."""
    removed = 0

    for existing_file in existing_files:
        if existing_file.stem in exported_ids:
            continue

        tombstone = await adapter.get_tombstone(_hypomnema_id(existing_file.stem))
        if tombstone is None or tombstone.type != "decision":
            continue

        existing_file.unlink()
        removed += 1
        print(f"  removed {existing_file.name} (buried)")

    return removed


async def _cmd_sync(path: pathlib.Path, store: StoreConfig) -> int:
    """Export every Decision to ``path/<id>.md``; raises ``CliError`` if the guard trips.

    A file is rewritten only when its bytes differ. A UUID-named ``.md`` file whose id was not
    exported this run is removed only when the store has a Decision tombstone for it. Other
    files in ``path`` are left alone.
    """
    path.mkdir(parents=True, exist_ok=True)
    store_is_suspect = _store_is_suspect(store)
    adapter = await open_store(store)

    exported = await _export_decisions(adapter, path)
    existing_files = [f for f in path.glob("*.md") if _is_uuid_stem(f.stem)]

    if store_is_suspect:
        _guard_suspect_store(store, exported.exported_ids, existing_files)

    removed = await _prune_buried(adapter, existing_files, exported.exported_ids)

    print(
        f"sync: {exported.written} written, {exported.unchanged} unchanged, {removed} removed"
        f" → {path}"
    )
    return 0


# ---------------------------------------------------------------------------
# scope / facet / role add
# ---------------------------------------------------------------------------


async def _cmd_registry_add(
    kind: Literal["scope", "facet", "role"],
    name: str,
    store: StoreConfig,
) -> int:
    """Register a scope, facet (purpose) or role name; raises ``CliError`` on an invalid name."""
    adapter = await open_store(store)

    if kind == "scope":
        # Deferred: the domain module imports pydantic (about 0.2 s).
        from tabularium.domain import parse_scope

        try:
            typed_scope = parse_scope(name)
        except ValueError:
            raise CliError(
                f"invalid scope name {name!r}: "
                "must match ^[a-z][a-z0-9-]{{0,31}}$"
            ) from None

        await adapter.create_scope(typed_scope)
        scopes = await adapter.list_scopes()
        print(f"scope {name!r} registered (total: {len(scopes)})")

    elif kind == "facet":
        if not name.strip():
            raise CliError("facet name must be non-empty")
        await adapter.create_facet(name)
        facets = await adapter.list_facets()
        print(f"facet {name!r} registered (total: {len(facets)})")

    elif kind == "role":
        if not name.strip():
            raise CliError("role name must be non-empty")
        await adapter.create_role(name)
        roles = await adapter.list_roles()
        print(f"role {name!r} registered (total: {len(roles)})")

    return 0


# ---------------------------------------------------------------------------
# reclaim
# ---------------------------------------------------------------------------

_RECLAIM_PRECONDITION = (
    "reclaim: irreversible version pruning (buys disk only).\n"
    "PRECONDITION: no other process (e.g. the MCP server) may be concurrently "
    "working on this dataset — stop it first (same precondition as "
    "docs/backup.md's cold-copy procedure)."
)


async def _cmd_reclaim(store: StoreConfig, *, yes: bool) -> int:
    """Prune every non-latest version of the store through ``StorageAdapter.reclaim``. Irreversible.

    Prints the no-other-writer precondition and asks for ``y`` unless ``yes`` is set; anything
    else returns 1 without opening the store.
    """
    print(_RECLAIM_PRECONDITION)

    if not yes:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer != "y":
            print("reclaim: aborted (not confirmed)")
            return 1

    adapter = await open_store(store)
    await adapter.reclaim()
    print("reclaim: complete")
    return 0


# ---------------------------------------------------------------------------
# supersede
# ---------------------------------------------------------------------------


def _parse_reason(raw: str) -> SupersessionReason:
    """Argparse ``type`` for ``--reason``: ``raw`` as a ``SupersessionReason``.

    Raises ``argparse.ArgumentTypeError`` with argparse's own invalid-choice wording, so a bad
    value exits 2 with the same message a ``choices`` check gives.
    """
    # Deferred: the domain module imports pydantic (about 0.2 s).
    from tabularium.domain import SupersessionReason

    reasons: tuple[SupersessionReason, ...] = get_args(SupersessionReason)
    for reason in reasons:
        if raw == reason:
            return reason

    raise argparse.ArgumentTypeError(
        f"invalid choice: {raw!r} (choose from {', '.join(reasons)})"
    )


async def _cmd_supersede(
    predecessor: str,
    successor: str,
    reason: SupersessionReason,
    note: str | None,
    store: StoreConfig,
) -> int:
    """Draw a Fact-arm supersede edge by hand through ``engine.dispositions.supersede``.

    Builds no engine and no embedder: it passes a fresh ``asyncio.Lock`` and a ``SystemClock``
    to ``supersede``. Raises ``CliError`` for a malformed id or a ``ToolError``.
    """
    # Deferred: the engine package imports litellm and lancedb (about 6 s).
    from tabularium.engine._errors import ToolError
    from tabularium.engine.dispositions import supersede

    if not _is_uuid_stem(predecessor):
        raise CliError(f"predecessor {predecessor!r} is not a valid UUID")
    if not _is_uuid_stem(successor):
        raise CliError(f"successor {successor!r} is not a valid UUID")

    adapter = await open_store(store)

    try:
        result = await supersede(
            successor_id=_hypomnema_id(successor),
            predecessor_ids=[_hypomnema_id(predecessor)],
            reason=reason,
            note=note,
            adapter=adapter,
            clock=SystemClock(),
            lock=asyncio.Lock(),
        )
    except ToolError as exc:
        raise CliError(str(exc)) from exc

    edge_id = result.edges[0][1]
    print(
        f"supersede: {predecessor} -> {successor} "
        f"(reason={reason!r}, edge_id={edge_id})"
    )
    return 0


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _pct(numerator: int, denominator: int) -> str:
    """Format ``numerator/denominator`` as a percentage, ``0.0%`` for a zero denominator."""
    if denominator == 0:
        return "0.0%"
    return f"{100.0 * numerator / denominator:.1f}%"


def _render_doctor_report(report: DoctorReport) -> str:
    """Render a ``DoctorReport`` as fixed-order terminal text.

    Dicts are iterated in sorted-key order and no timestamp is printed, so two runs against an
    unchanged store produce identical output. Each ratio prints its denominator.
    """
    fc = report.facet_coverage
    dv = report.dead_vocabulary
    pk = report.awaiting_embedding
    pd = report.projection_drift

    lines: list[str] = []
    lines.append("tabularium doctor")
    lines.append("=" * 60)

    # Section 1: facet coverage
    lines.append("")
    lines.append("1. Facet coverage (SPEC §10.2)")
    lines.append(
        f"   scanned (denominator): {fc.scanned} live records "
        f"(facts={fc.fact_count}, decisions={fc.decision_count})"
    )
    if fc.scanned == 0:
        lines.append("   store is empty — nothing to report for this section.")
    else:
        lines.append(
            f"   with purpose set: {fc.with_purpose}/{fc.scanned} "
            f"({_pct(fc.with_purpose, fc.scanned)})"
        )
        lines.append(f"   purpose value distribution (of {fc.scanned} scanned):")
        if fc.purpose_counts:
            for name in sorted(fc.purpose_counts):
                count = fc.purpose_counts[name]
                lines.append(f"     {name}: {count} ({_pct(count, fc.scanned)})")
        else:
            lines.append("     (no facet values registered)")
        lines.append(
            f"   with audience_hint set: {fc.with_audience_hint}/{fc.scanned} "
            f"({_pct(fc.with_audience_hint, fc.scanned)})"
        )

    # Section 2: dead vocabulary
    lines.append("")
    lines.append("2. Dead-vocabulary flags")
    lines.append(f"   registered facet values ({len(dv.registered)}): "
                  f"{', '.join(dv.registered) if dv.registered else '(none)'}")
    lines.append(
        f"   consumed by a policy row (code-level — some dispatch path "
        f"branches on this value): {len(dv.registered) - len(dv.dead)}/{len(dv.registered)}"
    )
    if dv.dead:
        lines.append(
            f"   dead — registered but NO policy row consumes them: "
            f"{', '.join(dv.dead)}"
        )
        lines.append(
            "     (structural finding, not a per-value anomaly: the Curator — "
            "the only design consumer that would branch on purpose — is "
            "deferred per the M2 slice C decision, so today's build has zero "
            "purpose-consuming dispatch code at all.)"
        )
    else:
        lines.append("   dead: none — every registered value has a consumer.")
    if dv.unused_in_data:
        lines.append(
            f"   unused in this scan's data (zero live occurrences right now, "
            f"independent of the code-level finding above): "
            f"{', '.join(dv.unused_in_data)}"
        )
    else:
        lines.append(
            "   unused in data: none — every registered value has at least "
            "one live occurrence this scan."
        )

    # Section 3: rows awaiting embedding
    lines.append("")
    lines.append("3. Rows awaiting embedding")
    lines.append(f"   awaiting embedding: {pk.count}")
    if pk.count == 0:
        lines.append("   nothing awaiting a vector — every live row carries one.")
    else:
        lines.append(
            "   these rows landed without a vector (an embedder outage at "
            "write time) or under a stale model stamp; `tabularium backfill` "
            "fills them in, and the next successful embed schedules a pass "
            "automatically."
        )

    # Section 4: projection drift
    lines.append("")
    lines.append("4. Projection drift [A9]")
    lines.append(f"   mode: {pd.mode} (report-only — this command never repairs)")
    lines.append(f"   scanned: {pd.scanned}")
    lines.append(f"   drifted: {len(pd.drifted)}/{pd.scanned}"
                  + (f" ({_pct(len(pd.drifted), pd.scanned)})" if pd.scanned else ""))
    if not pd.drifted:
        lines.append("   no drift found — every scanned record's stored "
                      "projections match the event-log recomputation.")
    else:
        lines.append(
            "   run `rebuild_projections(adapter, cfg, repair=True)` "
            "(no CLI flag yet — engine call only) to persist the fix."
        )

    lines.append("")
    return "\n".join(lines)


async def _cmd_doctor(store: StoreConfig) -> int:
    """Print the doctor report built with ``engine_config_from_env()``.

    Raises ``CliError`` before the store is opened when a ``TABULARIUM_*`` engine value is
    malformed.
    """
    # Deferred: the config module imports pydantic (about 0.2 s).
    from tabularium.config import engine_config_from_env

    # Deferred: the engine package imports litellm and lancedb (about 6 s).
    from tabularium.engine.doctor import build_doctor_report

    try:
        engine_cfg = engine_config_from_env()
    except ValueError as exc:
        raise CliError(str(exc)) from exc

    adapter = await open_store(store)

    report = await build_doctor_report(adapter, engine_cfg)
    print(_render_doctor_report(report))
    return 0


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------


async def _cmd_backfill(store: StoreConfig, *, embedder: Embedder | None = None) -> int:
    """Run one backfill pass against the configured store and print the counts.

    Returns 0 when no row failed, else 1. ``embedder`` is a test-injection seam; ``None`` builds
    a ``LitellmEmbedder``. ``populate_content_keys`` runs first, under a fresh lock, because this
    command opens the adapter directly instead of through ``Tabularium.initialize()``, and a row in
    a legacy store may still lack its ``content_hash``.
    """
    # Deferred: the engine package imports litellm and lancedb (about 6 s).
    from tabularium.engine.backfill import run_pass

    print(_concurrent_write_warning("backfill"))

    adapter = await open_store(store)
    if embedder is None:
        embedder = _build_embedder(store)

    lock = asyncio.Lock()
    async with lock:
        await adapter.populate_content_keys()

    report = await run_pass(
        adapter, embedder, lock, embedding_model=store.model, trigger="cli"
    )
    print(f"filled={report.filled} skipped={report.skipped_hash} failed={report.failed}")
    return 0 if report.failed == 0 else 1


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------

_CLEAN_RUBRIC_REQUIRES_DRY_RUN = (
    "clean: --rubric is dry-run only. A live pass under a non-canonical rubric would "
    "write shelve reasons stamped with a rubric_version that does not exist anywhere in the "
    "repo, making a destructive, owner-supervised write unreproducible. Pass --dry-run, or drop "
    "--rubric to run live against the packaged rubric."
)

_CLEAN_RUBRIC_HISTORY_MUTUALLY_EXCLUSIVE = (
    "clean: --rubric is meaningless with --history — --history runs no pass and never "
    "consults a rubric of any kind. Drop one or the other."
)

_CLEAN_READ_ONLY_REFUSAL = (
    "clean: refused, TABULARIUM_READ_ONLY is set and this store is a read-only archive"
)

_CLEAN_MEMORY_ADAPTER_REFUSAL = (
    "clean: the memory adapter never persists data — a write command has no business "
    "targeting an ephemeral store, so this is refused outright as a preflight check, before "
    "anything is opened (unlike `sync`, which is read-only on the store and only refuses "
    "conditionally, inside its own fail-closed guard, when a stale export directory looks "
    "populated against an empty scan). Point --store / TABULARIUM_STORE_PATH at a lancedb "
    "store."
)

_WHITESPACE_RUN_RE = re.compile(r"\s+")
# C0 controls except tab, LF and CR (which _WHITESPACE_RUN_RE collapses), plus DEL and the C1 range.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _single_line(text: str) -> str:
    """Drop control characters from free text, then collapse every whitespace run to one space.

    Judge ``why`` text is only length-checked upstream, so this keeps a newline or a terminal
    escape sequence out of the one-record-per-line output.
    """
    return _WHITESPACE_RUN_RE.sub(" ", _CONTROL_CHAR_RE.sub("", text)).strip()


# A failed batch's error goes through _single_line, then keeps its last 300 characters.
_BATCH_ERROR_RENDER_CAP = 300


def _render_batch_error(error: str, *, cap: int = _BATCH_ERROR_RENDER_CAP) -> str:
    """Render a failed batch's ``.error`` as one line of at most ``cap`` characters (tail kept)."""
    single = _single_line(error)
    if len(single) <= cap:
        return single
    return "..." + single[-(cap - 3) :]


def _nonzero_counts(**counts: int) -> str:
    """Render the non-zero keyword counts as ``", 2 trimmed, 1 merge-reverted"``.

    Order follows the keywords and underscores become hyphens. Returns ``""`` when all are zero.
    """
    return "".join(f", {n} {label.replace('_', '-')}" for label, n in counts.items() if n)


class _BatchFailureView(Protocol):
    """The fields of a failed batch the summary prints; all three axes' failure models have them."""

    batch_index: int
    judge_a_outcome: BatchOutcome
    judge_b_outcome: BatchOutcome
    error: str


def _failed_batch_lines(axis: str, failures: Sequence[_BatchFailureView]) -> list[str]:
    """The ``N <axis> batch(es) failed:`` header and one line per failed batch; empty for none."""
    if not failures:
        return []

    lines = [f"  {len(failures)} {axis} batch(es) failed:"]
    for failure in failures:
        lines.append(
            f"    batch {failure.batch_index} "
            f"(judge_a={failure.judge_a_outcome}, "
            f"judge_b={failure.judge_b_outcome}): "
            f"{_render_batch_error(failure.error)}"
        )
    return lines


def _shelved_lines(
    verb: str,
    shelved_ids: Sequence[HypomnemaId],
    why_by_id: Mapping[HypomnemaId, str],
) -> list[str]:
    """The ``  <verb>:`` header and one line per shelved id with its deciding judge's why."""
    if not shelved_ids:
        return []

    lines = [f"  {verb}:"]
    for record_id in shelved_ids:
        raw_why = why_by_id.get(record_id)
        why = _single_line(raw_why) if raw_why else None
        lines.append(f"    {record_id}" + (f" — {why}" if why else ""))
    return lines


def _unclear_lines(unclear_ids: Sequence[HypomnemaId]) -> list[str]:
    """The ``unclear`` line listing the ids the panel declined to judge; empty for none."""
    if not unclear_ids:
        return []

    ids = ", ".join(str(i) for i in unclear_ids)
    return [f"  unclear (panel declined to judge): {ids}"]


def _judge_text(identity: str, grade: str, why: str) -> str:
    """One judge's side of a disagreement: ``identity=grade ("why")``."""
    return f'{identity}={grade} ("{_single_line(why)}")'


def _disagreement_line(subject: str, judge_a: str, judge_b: str) -> str:
    """One contested entry: the subject, then the two judges' rendered sides."""
    return f"    {subject} — {judge_a} vs {judge_b}"


def _relevance_lines(relevance: RelevanceAxisSummary, *, dry_run: bool) -> list[str]:
    """The relevance axis section of the clean summary."""
    shelve_verb = "would shelve" if dry_run else "shelved"
    counts = relevance.counts
    lines = [
        f"relevance: {counts.shelve} {shelve_verb}, {counts.keep} keep, "
        f"{counts.unclear} unclear, {counts.contested} contested"
        f"{_nonzero_counts(trimmed=len(relevance.trimmed))} "
        f"(judges: {relevance.judge_pair[0]} + {relevance.judge_pair[1]}, "
        f"rubric {relevance.rubric_version})"
    ]

    why_by_id = {v.id: v.why for v in relevance.verdicts}
    lines += _shelved_lines(shelve_verb, relevance.shelved_ids, why_by_id)
    lines += _unclear_lines(relevance.unclear_ids)

    if relevance.contested_ids:
        contested_by_id = {v.id: v for v in relevance.contested_verdicts}
        lines.append("  contested (judges disagreed -- no write, routed to owner):")
        for record_id in relevance.contested_ids:
            detail = contested_by_id.get(record_id)
            if detail is None:
                lines.append(f"    {record_id}")
                continue
            lines.append(
                _disagreement_line(
                    str(record_id),
                    _judge_text(detail.judge_a_identity, detail.judge_a_grade, detail.judge_a_why),
                    _judge_text(detail.judge_b_identity, detail.judge_b_grade, detail.judge_b_why),
                )
            )

    lines += _failed_batch_lines("relevance", relevance.failed_batches)
    return lines


def _falsity_lines(falsity: FalsityAxisSummary, *, dry_run: bool) -> list[str]:
    """The falsity axis section of the clean summary."""
    replace_verb = "would supersede" if dry_run else "superseded"
    extras = _nonzero_counts(
        merged=len(falsity.merged),
        merge_reverted=len(falsity.merge_reverted),
        merge_revert_failed=len(falsity.merge_revert_failed),
    )
    lines = [
        f"falsity: {falsity.counts.replace} {replace_verb}, {falsity.counts.coexist} coexist, "
        f"{falsity.counts.contested} contested"
        f"{extras} "
        f"(judges: {falsity.judge_pair[0]} + {falsity.judge_pair[1]}, "
        f"rubric {falsity.rubric_version})"
    ]

    if falsity.contested_pairs:
        lines.append("  contested (judges disagreed -- no edge drawn, routed to owner):")
        for pair in falsity.contested_pairs:
            lines.append(
                _disagreement_line(
                    f"{pair.newcomer_id} vs {pair.candidate_id}",
                    _judge_text(pair.judge_a_identity, pair.judge_a_verdict, pair.judge_a_why),
                    _judge_text(pair.judge_b_identity, pair.judge_b_verdict, pair.judge_b_why),
                )
            )

    lines += _failed_batch_lines("falsity", falsity.failed_batches)
    return lines


def _revisit_lines(revisit: RevisitAxisSummary, *, dry_run: bool) -> list[str]:
    """The revisit axis section of the clean summary."""
    shelve_verb = "would shelve" if dry_run else "shelved"
    counts = revisit.counts
    lines = [
        f"revisit: {counts.shelve} {shelve_verb}, {counts.keep} keep, "
        f"{revisit.split_keep} split-keep, "
        f"{counts.unclear} unclear, {counts.contested} contested"
        f"{_nonzero_counts(trimmed=len(revisit.trimmed))} "
        f"(judges: {revisit.judge_pair[0]} + {revisit.judge_pair[1]}, "
        f"rubric {revisit.rubric_version}; "
        f"pool {revisit.pool_size}, protected {len(revisit.protected_ids)}, "
        f"skipped-no-evidence {len(revisit.skipped_no_evidence_ids)}, "
        f"skipped-too-large {len(revisit.skipped_too_large_ids)})"
    ]

    why_by_id = {v.id: v.why for v in revisit.verdicts}
    lines += _shelved_lines(shelve_verb, revisit.shelved_ids, why_by_id)
    lines += _unclear_lines(revisit.unclear_ids)

    if revisit.contested_verdicts:
        lines.append("  contested (judges disagreed -- no write, routed to owner):")
        for detail in revisit.contested_verdicts:
            lines.append(
                _disagreement_line(
                    str(detail.id),
                    _judge_text(detail.judge_a_identity, detail.judge_a_grade, detail.judge_a_why),
                    _judge_text(detail.judge_b_identity, detail.judge_b_grade, detail.judge_b_why),
                )
            )

    lines += _failed_batch_lines("revisit", revisit.failed_batches)

    if revisit.verification_failures:
        ids = ", ".join(str(i) for i in revisit.verification_failures)
        lines.append(f"  revisit verification failures: {ids}")
    return lines


def _render_clean_summary(
    summary: CleanSummary,
    *,
    rubric_path: pathlib.Path | None = None,
    rubric_version: str | None = None,
) -> str:
    """Render a ``CleanSummary`` as fixed-order terminal text.

    The ``mode:`` line states ``summary.dry_run``. In a dry run, ``watermark`` is labelled
    ``watermark (would move)``, verbs read ``would shelve`` and ``would supersede``, and a
    ``completed`` outcome is marked unverified, because ``clean()`` skips write verification
    under ``dry_run``. The ``rubric:`` line names the packaged rubric when ``rubric_path`` is
    ``None``, else the path and ``rubric_version``. The ``revisit`` section is printed after the
    two birth axes; its results do not change ``outcome``, ``watermark`` or ``reason``.
    """
    lines: list[str] = []
    lines.append(f"tabularium clean — scope {summary.scope!s}")
    lines.append("=" * 60)
    lines.append(
        f"mode: {'DRY RUN — nothing written, watermark not moved' if summary.dry_run else 'LIVE'}"
    )
    lines.append(
        f"rubric: {rubric_path} (version {rubric_version})"
        if rubric_path is not None
        else "rubric: packaged"
    )
    outcome_line = f"outcome: {summary.outcome}"
    if summary.dry_run and summary.outcome == "completed":
        outcome_line += (
            " (dry run — UNVERIFIED: no write was attempted, so nothing was re-read to "
            "confirm it would have landed)"
        )
    lines.append(outcome_line)
    before = (
        summary.watermark_before.isoformat()
        if summary.watermark_before is not None
        else "(never judged)"
    )
    watermark_label = "watermark (would move)" if summary.dry_run else "watermark"
    lines.append(f"{watermark_label}: {before} -> {summary.watermark_after.isoformat()}")
    lines.append(f"pool size: {summary.pool_size}")

    if summary.relevance is not None:
        lines += _relevance_lines(summary.relevance, dry_run=summary.dry_run)

    if summary.falsity is not None:
        lines += _falsity_lines(summary.falsity, dry_run=summary.dry_run)

    if summary.revisit is not None:
        lines += _revisit_lines(summary.revisit, dry_run=summary.dry_run)
    if summary.revisit_error is not None:
        lines.append(
            f"revisit: axis errored (isolated from this pass's outcome): "
            f"{summary.revisit_error}"
        )

    if summary.verification_failures:
        ids = ", ".join(str(i) for i in summary.verification_failures)
        lines.append(f"verification failures ({len(summary.verification_failures)}): {ids}")

    lines.append(f"reason: {summary.reason}")
    lines.append("")
    return "\n".join(lines)


async def _cmd_clean_history(
    scope: str,
    store: StoreConfig,
    *,
    limit: int,
) -> int:
    """Print ``scope``'s Cleaner pass-history rows, newest first, as JSON lines.

    Runs no pass. The scope string is validated but need not be registered; a scope with no
    history prints nothing. Raises ``CliError`` for a malformed scope. The store comes from
    ``store`` like every other command.
    """
    # Deferred: the engine package imports litellm and lancedb (about 6 s).
    from tabularium.engine._errors import ToolError
    from tabularium.engine.filters import validate_and_resolve_scope

    try:
        typed_scope = validate_and_resolve_scope(scope)
    except ToolError as exc:
        raise CliError(str(exc)) from exc

    adapter = await open_store(store)

    records = await adapter.list_cleaner_passes(scope=typed_scope, limit=limit)
    for record in records:
        print(record.model_dump_json())
    return 0


def _check_clean_flags(*, history: bool, dry_run: bool, rubric_path: pathlib.Path | None) -> None:
    """Raise ``CliError`` for a flag combination ``clean`` refuses.

    ``--history`` with ``--rubric`` is refused, and ``--rubric`` without ``--dry-run`` is refused.
    """
    if history and rubric_path is not None:
        raise CliError(_CLEAN_RUBRIC_HISTORY_MUTUALLY_EXCLUSIVE)

    if rubric_path is not None and not dry_run:
        raise CliError(_CLEAN_RUBRIC_REQUIRES_DRY_RUN)


def _load_clean_rubric(rubric_path: pathlib.Path | None) -> Rubric | None:
    """Load the rubric at ``rubric_path``; ``None`` for the packaged rubric. Raises ``CliError``."""
    if rubric_path is None:
        return None

    # Deferred: the engine package imports litellm and lancedb (about 6 s).
    from tabularium.engine.rubric import RubricLoadError, load_rubric_from_path

    try:
        return load_rubric_from_path(rubric_path)
    except RubricLoadError as exc:
        raise CliError(str(exc)) from exc


async def _cmd_clean(
    scope: str,
    store: StoreConfig,
    *,
    dry_run: bool,
    history: bool = False,
    limit: int = 20,
    rubric_path: pathlib.Path | None = None,
    embedder: Embedder | None = None,
    clock: Clock | None = None,
    judge_a: JudgeAdapter | None = None,
    judge_b: JudgeAdapter | None = None,
    assessor: Assessor | None = None,
) -> int:
    """Run one Cleaner pass by hand through ``engine.report.run_clean_pass``; the module docstring
    lists the refusals and exit codes.

    Order of checks: ``_check_clean_flags``; ``history`` delegates to ``_cmd_clean_history`` and
    returns; ``TABULARIUM_READ_ONLY`` raises ``CliError``; a ``RubricLoadError`` raises
    ``CliError``, as does the memory adapter. The concurrent-write warning is then printed unless
    ``dry_run``, and the store is opened. A ``ToolError`` from the pass, such as an unregistered
    scope, raises ``CliError``. The return is 1 when the outcome is ``blocked`` and 0 otherwise.

    ``embedder``, ``clock``, ``judge_a``, ``judge_b`` and ``assessor`` are test-injection seams.
    When ``None``, the embedder is a ``LitellmEmbedder``, the clock a ``SystemClock``, the judges
    are resolved inside ``clean()``, and the Assessor comes from
    ``config.build_assessor_from_env``. The engine config is ``engine_config_from_env()``, so
    the ``TABULARIUM_*`` engine overrides apply.
    """
    # Deferred: the engine package imports litellm and lancedb (about 6 s).
    from tabularium.config import env_bool
    from tabularium.engine._errors import ToolError
    from tabularium.engine.report import run_clean_pass

    _check_clean_flags(history=history, dry_run=dry_run, rubric_path=rubric_path)

    if history:
        return await _cmd_clean_history(scope, store, limit=limit)

    if env_bool("TABULARIUM_READ_ONLY"):
        raise CliError(_CLEAN_READ_ONLY_REFUSAL)

    resolved_rubric = _load_clean_rubric(rubric_path)

    if store.adapter == "memory":
        raise CliError(_CLEAN_MEMORY_ADAPTER_REFUSAL)

    if not dry_run:
        print(_concurrent_write_warning("clean"))

    adapter = await open_store(store)

    async def _require_scope(typed_scope: Scope) -> None:
        """Raise ``ToolError`` unless ``typed_scope`` is registered; mirrors the engine's check."""
        registered = await adapter.list_scopes()
        if typed_scope not in registered:
            raise ToolError(
                f"Unregistered scope {typed_scope!r}: "
                "register with `tabularium scope add <name>` before use. "
                f"Registered scopes: {registered!r}"
            )

    if embedder is None:
        embedder = _build_embedder(store)
    if clock is None:
        clock = SystemClock()

    # Deferred: the config module imports pydantic (about 0.2 s).
    from tabularium.config import build_assessor_from_env, engine_config_from_env

    engine_cfg = engine_config_from_env()
    if assessor is None:
        assessor = build_assessor_from_env(engine_cfg.assessor_model)

    try:
        summary = await run_clean_pass(
            scope=scope,
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            lock=asyncio.Lock(),
            cfg=engine_cfg,
            require_scope=_require_scope,
            rubric=resolved_rubric,
            judge_a=judge_a,
            judge_b=judge_b,
            assessor=assessor,
            dry_run=dry_run,
        )
    except ToolError as exc:
        raise CliError(str(exc)) from exc

    print(
        _render_clean_summary(
            summary,
            rubric_path=rubric_path,
            rubric_version=resolved_rubric.version if resolved_rubric is not None else None,
        )
    )
    return 1 if summary.outcome == "blocked" else 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _help_handler(parser: argparse.ArgumentParser) -> Handler:
    """A handler that prints ``parser``'s help and returns 0, for a command group named alone."""

    async def handler(args: argparse.Namespace, store: StoreConfig) -> int:
        parser.print_help()
        return 0

    return handler


async def _run_supersede(args: argparse.Namespace, store: StoreConfig) -> int:
    return await _cmd_supersede(args.predecessor, args.successor, args.reason, args.note, store)


async def _run_clean(args: argparse.Namespace, store: StoreConfig) -> int:
    return await _cmd_clean(
        args.scope,
        store,
        dry_run=args.dry_run,
        history=args.history,
        limit=args.limit,
        rubric_path=pathlib.Path(args.rubric) if args.rubric is not None else None,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tabularium",
        description="Tabularium — long-term memory system for LLM agents",
    )
    parser.add_argument(
        "--store",
        metavar="PATH",
        default=None,
        help="Override TABULARIUM_STORE_PATH",
    )
    parser.add_argument(
        "--adapter",
        metavar="NAME",
        default=None,
        help="Override TABULARIUM_ADAPTER (lancedb|memory)",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sync_p = sub.add_parser(
        "sync",
        help="Export every Decision to <path>/<id>.md (idempotent)",
    )
    sync_p.add_argument("path", type=pathlib.Path, help="Output directory")
    sync_p.set_defaults(handler=lambda args, store: _cmd_sync(args.path, store))

    scope_p = sub.add_parser("scope", help="Manage scopes")
    scope_p.set_defaults(handler=_help_handler(scope_p))
    scope_sub = scope_p.add_subparsers(dest="scope_command", metavar="ACTION")
    scope_add = scope_sub.add_parser("add", help="Register a scope name")
    scope_add.add_argument("name", help="Scope name (^[a-z][a-z0-9-]{0,31}$)")
    scope_add.set_defaults(handler=lambda args, store: _cmd_registry_add("scope", args.name, store))

    facet_p = sub.add_parser("facet", help="Manage purpose-facet values")
    facet_p.set_defaults(handler=_help_handler(facet_p))
    facet_sub = facet_p.add_subparsers(dest="facet_command", metavar="ACTION")
    facet_add = facet_sub.add_parser("add", help="Register a facet value")
    facet_add.add_argument("name", help="Facet value name")
    facet_add.set_defaults(handler=lambda args, store: _cmd_registry_add("facet", args.name, store))

    role_p = sub.add_parser("role", help="Manage roles")
    role_p.set_defaults(handler=_help_handler(role_p))
    role_sub = role_p.add_subparsers(dest="role_command", metavar="ACTION")
    role_add = role_sub.add_parser("add", help="Register a role name")
    role_add.add_argument("name", help="Role name")
    role_add.set_defaults(handler=lambda args, store: _cmd_registry_add("role", args.name, store))

    reclaim_p = sub.add_parser(
        "reclaim",
        help="Irreversible version pruning (disk only) — requires confirmation",
    )
    reclaim_p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt",
    )
    reclaim_p.set_defaults(handler=lambda args, store: _cmd_reclaim(store, yes=args.yes))

    doctor_p = sub.add_parser(
        "doctor",
        help="Report facet coverage, dead vocabulary, rows awaiting embedding, "
        "and projection drift (SPEC §13) — observes only, never gates",
    )
    doctor_p.set_defaults(handler=lambda args, store: _cmd_doctor(store))

    backfill_p = sub.add_parser(
        "backfill",
        help="Run one pass filling vectors for null-vector/stale-model rows "
        "(P6 vector-less-rows piece) — prints filled/skipped/failed",
    )
    backfill_p.set_defaults(handler=lambda args, store: _cmd_backfill(store))

    supersede_p = sub.add_parser(
        "supersede",
        help="Draw a Fact-arm supersede edge by hand: <predecessor-id> <successor-id>",
    )
    supersede_p.add_argument("predecessor", help="Id of the Fact being superseded")
    supersede_p.add_argument("successor", help="Id of the Fact that replaces it")
    supersede_p.add_argument(
        "--reason",
        required=True,
        type=_parse_reason,
        metavar=_REASON_METAVAR,
        help="Required supersession reason class",
    )
    supersede_p.add_argument(
        "--note", default=None, help="Optional free-text note landing beside --reason"
    )
    supersede_p.set_defaults(handler=_run_supersede)

    clean_p = sub.add_parser(
        "clean",
        help="Run one Cleaner pass over a scope by hand (judged relevance + falsity) — "
        "writes from a second OS process; see --dry-run",
    )
    clean_p.add_argument("--scope", required=True, help="Target scope")
    clean_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Judge and print without writing anything or moving the watermark",
    )
    clean_p.add_argument(
        "--history",
        action="store_true",
        help="Print --scope's pass-history rows newest-first as JSON lines and exit; "
        "runs no pass (mutually exclusive with a live/dry-run pass)",
    )
    clean_p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max rows to print with --history (default 20)",
    )
    clean_p.add_argument(
        "--rubric",
        metavar="PATH",
        default=None,
        help="Load the rubric from PATH instead of the packaged artifact — dry-run only, "
        "exits non-zero without --dry-run",
    )
    clean_p.set_defaults(handler=_run_clean)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse the arguments and run the chosen command; the process exit code is its return.

    A ``CliError`` from the command prints as ``error: <message>`` on stderr and exits 1.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Deferred: the config module imports pydantic (about 0.2 s).
    from tabularium.config import config_from_env

    store = config_from_env(
        path_override=args.store,
        adapter_override=args.adapter,
    )

    try:
        rc = asyncio.run(args.handler(args, store))
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.exit(rc)
