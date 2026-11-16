"""The Cleaner's rubric — versioned data, loaded, never a Python string literal
(2026-08-10 cleaner plan).

Falsifiability (step 6 of the task): the version-pinning test below was proven sensitive to a real
defect by editing ``src/tabularium/engine/rubric.toml`` without touching any version field,
watching this file's ``test_version_is_pinned_to_content`` go red for exactly that reason, then
reverting. The captured command and output are reported in the task verdict, not duplicated here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from tabularium.engine import Tabularium
from tabularium.engine.judge_policy import JUDGE_DENY_MCP_TOOLS, JUDGE_SAFE_MCP_TOOLS
from tabularium.engine.rubric import (
    _ARTIFACT_PATH,
    CliSeat,
    FalsityRubric,
    JudgesConfig,
    JudgeSeatConfig,
    LitellmSeat,
    RelevanceRubric,
    Rubric,
    RubricLoadError,
    RubricParameters,
    load_rubric,
    load_rubric_from_path,
)
from tabularium.server import make_app
from tabularium.storage.memory import MemoryAdapter
from tests.support.doubles import FakeEmbedder

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Pinned against the artifact as shipped. Any edit to `rubric.toml` — prose, parameters, anchors —
# changes this hash-derived version. If this test goes red after an edit, the fix is to recompute
# and re-pin the new value here, never to hand-roll a version bump inside the artifact itself.
# The hash covers the whole artifact, including judge seats, parameters and the worked examples
# in the relevance prose.
_EXPECTED_VERSION = "cleaner-rubric-v1-7f7e8cdc7d89"

_FORBIDDEN_PROJECT_A_DISTRIBUTION_PHRASES = (
    "one in three",
    "one in two",
    "1 in 3",
    "1 in 2",
    "1/3",
    "1/2",
)


def test_version_is_pinned_to_content() -> None:
    """The version is a fingerprint of the artifact's bytes, not a hand-maintained field."""
    rubric = load_rubric()
    assert rubric.version == _EXPECTED_VERSION


def test_version_changes_when_artifact_changes(tmp_path: Path) -> None:
    """A byte-for-byte independent proof that content changes the version, without touching the
    shipped artifact: hash two different payloads through the same derivation the loader uses and
    confirm they differ. Complements (does not replace) the deliberate-break Verify step, which
    edits the real file.
    """
    from tabularium.engine.rubric import _compute_version

    v1 = _compute_version(b"alpha", schema=1)
    v2 = _compute_version(b"bravo", schema=1)
    assert v1 != v2
    # Same bytes, same schema -> same version (content-addressed, not random).
    assert _compute_version(b"alpha", schema=1) == v1


def test_load_rubric_is_cached_and_returns_the_same_object() -> None:
    assert load_rubric() is load_rubric()


def test_both_axes_present() -> None:
    rubric = load_rubric()
    assert isinstance(rubric.relevance, RelevanceRubric)
    assert isinstance(rubric.falsity, FalsityRubric)
    assert rubric.relevance.prose.strip()
    assert rubric.falsity.prose.strip()


def test_relevance_scale_is_four_point_ordered_with_no_middle() -> None:
    rubric = load_rubric()
    assert rubric.relevance.scale == (
        "shelve-clear",
        "shelve-lean",
        "keep-lean",
        "keep-clear",
    )
    # No neutral grade name anywhere in the scale.
    assert not any("neutral" in grade or "unsure" in grade for grade in rubric.relevance.scale)


def test_falsity_verdicts_are_a_forced_two_way_choice() -> None:
    rubric = load_rubric()
    assert rubric.falsity.verdicts == ("coexist", "replace")


def test_relevance_prose_carries_the_admission_questions_and_keep_test() -> None:
    prose = load_rubric().relevance.prose
    for marker in ("Derivable?", "Perishable?", "sequencing atom", "session narrative"):
        assert marker in prose
    # Normalize whitespace (the artifact wraps prose at ~100 cols) before matching a phrase that
    # may span a line break in the source.
    flat = " ".join(prose.lower().split())
    assert "delete the" in flat and "still a fact" in flat
    assert "why" in flat and "mandatory" in flat


def test_revisit_prose_is_present_and_carries_the_admission_markers() -> None:
    """Revisit rubric: `[revisit].prose` must be present, non-empty,
    and carry the marker phrases the brief names verbatim — "Silence proves nothing" and
    "correctness" — same style as ``test_relevance_prose_carries_the_admission_questions_and_keep_
    test`` above."""
    prose = load_rubric().revisit.prose
    assert prose.strip()
    assert "Silence proves nothing" in prose
    assert "correctness" in prose.lower()


def test_falsity_prose_frames_replacement_not_incompatibility() -> None:
    prose = load_rubric().falsity.prose
    assert "REPLACE" in prose and "COEXIST" in prose
    assert "not as logical incompatibility" in prose
    assert "why" in prose.lower() and "mandatory" in prose.lower()


def test_relevance_prose_tells_the_judge_how_to_use_keep() -> None:
    prose = load_rubric().relevance.prose
    assert "## Keeping the surviving part" in prose
    assert prose.index("## Keeping the surviving part") < prose.index("## `why` is mandatory")
    assert "never put it on a `keep-*` grade" in prose


def test_falsity_prose_routes_a_surviving_point_to_carry() -> None:
    prose = load_rubric().falsity.prose
    assert "that is REPLACE with `carry`, not COEXIST" in prose
    assert "Omit `carry` when N says" in prose


def test_relevance_anchor_is_marked_provisional() -> None:
    anchor = load_rubric().relevance.anchor
    assert "provisional" in anchor.lower()
    assert "most records" in anchor.lower() and "kept" in anchor.lower()


def test_relevance_anchor_does_not_carry_the_project_a_distribution() -> None:
    """Acceptance criterion: the project-a one-in-three-to-one-in-two framing must not survive
    into this rubric. That ratio was measured on a retired seed corpus, not this tail pass.
    """
    rubric = load_rubric()
    parts = [
        rubric.relevance.anchor,
        rubric.relevance.prose,
        rubric.falsity.anchor,
        rubric.falsity.prose,
    ]
    haystack = " ".join(" ".join(parts).lower().split())
    for phrase in _FORBIDDEN_PROJECT_A_DISTRIBUTION_PHRASES:
        assert phrase not in haystack, f"distribution phrase leaked into rubric: {phrase!r}"


def test_sweep_parameters_are_data_not_code_constants() -> None:
    """Every parameter must come from the artifact — no field has a Python-side default, so a
    parameters block missing a key fails construction rather than silently falling back.
    """
    rubric = load_rubric()
    assert isinstance(rubric.parameters, RubricParameters)
    assert rubric.parameters.k > 0
    assert 0.0 <= rubric.parameters.similarity_floor <= 1.0
    assert rubric.parameters.batch_cap > 0
    assert 0.0 <= rubric.parameters.unclear_ceiling <= 1.0
    assert rubric.parameters.pool_limit > 0
    assert rubric.parameters.report_render_cap > 0
    assert rubric.parameters.revisit_batch_cap > 0
    assert rubric.parameters.revisit_pass_cap > 0
    assert rubric.parameters.revisit_min_age_days > 0
    assert rubric.parameters.revisit_interval_days > 0
    assert rubric.parameters.revisit_stable_multiplier > 0
    assert rubric.parameters.revisit_contested_multiplier > 0
    assert rubric.parameters.revisit_max_interval_days > 0

    incomplete: dict[str, Any] = {
        "similarity_floor": 0.8,
        "batch_cap": 56,
        "unclear_ceiling": 0.1,
        "pool_limit": 100_000,
        "report_render_cap": 5,
    }  # `k` deliberately missing
    with pytest.raises(ValidationError):
        RubricParameters(**incomplete)


def test_judges_config_present_and_pinned_explicitly() -> None:
    """Judge ordering is data in this artifact, not a branch in judge.py."""
    judges = load_rubric().judges
    assert isinstance(judges, JudgesConfig)

    assert isinstance(judges.seat_a, CliSeat)
    assert isinstance(judges.seat_b, CliSeat)
    assert judges.seat_a.adapter == "cli"
    assert judges.seat_a.cli == "claude"
    assert judges.seat_a.binary
    assert judges.seat_a.model  # explicit, non-empty

    assert judges.seat_b.adapter == "cli"
    assert judges.seat_b.cli == "codex"
    assert judges.seat_b.binary
    # Explicit pin, not the CLI's own silently-resolved default. Re-pinned
    # 2026-08-11 (deployed-server defect): the original pin, "gpt-5.1-codex", 400'd on the account's
    # ChatGPT subscription; the replacement, "gpt-5.6-sol", happens to equal the CLI's current
    # resolved default, but that is coincidence of which models this account can serve today, not
    # a regression to the unpinned failure mode this test guards — the value is still named
    # explicitly in the artifact rather than left to CLI resolution, so no `!=` check against any
    # particular string belongs here any more.
    assert judges.seat_b.model

    assert isinstance(judges.backstop, LitellmSeat)
    assert judges.backstop.adapter == "litellm"
    assert judges.backstop.model.startswith("deepseek/")


def test_judge_seat_config_timeout_defaults_to_1200_and_validates_positive() -> None:
    """Owner-requested (2026-08-11): the timeout is configurable data on the seat, defaulting to
    1200s, never a value that can be zero or negative."""
    seat = CliSeat(cli="claude", model="sonnet", binary="/usr/bin/claude")
    assert seat.timeout == 1200.0

    overridden = CliSeat(cli="claude", model="sonnet", binary="/usr/bin/claude", timeout=30.0)
    assert overridden.timeout == 30.0

    for bad in (0.0, -1.0):
        with pytest.raises(ValidationError):
            CliSeat(cli="claude", model="sonnet", binary="/usr/bin/claude", timeout=bad)


def test_judge_seat_config_rejects_cli_fields_on_wrong_adapter() -> None:
    """A litellm seat must not carry cli/binary; a cli seat must carry both."""
    seats: TypeAdapter[CliSeat | LitellmSeat] = TypeAdapter(JudgeSeatConfig)
    with pytest.raises(ValidationError):
        seats.validate_python({"adapter": "litellm", "cli": "claude", "model": "x"})
    with pytest.raises(ValidationError):
        seats.validate_python({"adapter": "litellm", "binary": "/bin/x", "model": "x"})
    with pytest.raises(ValidationError):
        seats.validate_python({"adapter": "cli", "model": "x"})  # missing cli + binary


def test_judge_seat_config_default_mcp_grant_is_empty() -> None:
    """A plain seat carries no MCP grant at all -- default off, matching every seat in the
    packaged artifact (2026-08-14 judge-seat MCP-access brief)."""
    seat = CliSeat(cli="claude", model="sonnet", binary="/usr/bin/claude")
    assert seat.mcp_server_url is None
    assert seat.mcp_server_name is None
    assert seat.mcp_allowed_tools == ()
    assert seat.mcp_grant is None


def test_judge_seat_config_mcp_name_or_tools_without_url_is_rejected() -> None:
    """Structural consistency, enforced on ``CliSeat`` itself (not seat-labelled -- that
    is ``JudgesConfig``'s job below): a name or a tool tuple with no URL to point at is malformed
    regardless of which seat it is filed under."""
    with pytest.raises(ValidationError):
        CliSeat(
            cli="claude",
            model="sonnet",
            binary="/usr/bin/claude",
            mcp_server_name="tabularium",
        )
    with pytest.raises(ValidationError):
        CliSeat(
            cli="claude",
            model="sonnet",
            binary="/usr/bin/claude",
            mcp_allowed_tools=("context",),
        )


def test_judge_seat_config_litellm_seat_rejects_mcp_fields() -> None:
    """MCP judge access is rendered by the subprocess CLI adapters only -- a litellm seat (the
    backstop, in the packaged artifact) must never carry a grant nothing would ever render."""
    with pytest.raises(ValidationError):
        TypeAdapter(JudgeSeatConfig).validate_python(
            {
                "adapter": "litellm",
                "model": "deepseek/deepseek-chat",
                "mcp_server_url": "http://127.0.0.1:8787/mcp",
                "mcp_server_name": "tabularium",
                "mcp_allowed_tools": ("context",),
            }
        )


def _cli_seat_fields(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "adapter": "cli",
        "cli": "claude",
        "model": "sonnet",
        "binary": "/usr/bin/claude",
    }
    base.update(overrides)
    return base


def _cli_seat(**overrides: Any) -> CliSeat:
    return CliSeat(**_cli_seat_fields(**overrides))


def _judges_config(**seat_overrides: JudgeSeatConfig) -> JudgesConfig:
    """A well-formed three-seat panel, with individual seats swapped out by name -- lets a test
    plant a malformed grant on exactly one seat and check the resulting message names THAT seat."""
    seats: dict[str, JudgeSeatConfig] = {
        "seat_a": _cli_seat(),
        "seat_b": _cli_seat(cli="codex", binary="/usr/bin/codex"),
        "backstop": LitellmSeat(model="deepseek/deepseek-chat"),
    }
    seats.update(seat_overrides)
    return JudgesConfig(**seats)


def test_judges_config_rejects_a_seat_with_url_but_no_name() -> None:
    """Fail-closed (task brief point 4): a URL with no name is a configuration error at rubric
    load, naming the offending seat."""
    bad_seat = _cli_seat_fields(mcp_server_url="http://127.0.0.1:8787/mcp")
    with pytest.raises(ValidationError, match="seat_a"):
        JudgesConfig.model_validate(
            {
                "seat_a": bad_seat,
                "seat_b": _cli_seat_fields(cli="codex", binary="/usr/bin/codex"),
                "backstop": {"adapter": "litellm", "model": "deepseek/deepseek-chat"},
            }
        )


def test_judges_config_rejects_a_seat_with_url_but_empty_allowlist() -> None:
    """Fail-closed (task brief point 4): a URL with an empty tool allowlist must never open a
    connection, naming the offending seat."""
    bad_seat = _cli_seat(mcp_server_url="http://127.0.0.1:8787/mcp", mcp_server_name="tabularium")
    with pytest.raises(ValidationError, match="seat_a"):
        _judges_config(seat_a=bad_seat)


def test_judges_config_rejects_recall_in_the_allowlist() -> None:
    """Hard guard (task brief point 5), not a lint: 'recall' mutates on its archived-fallback
    branch and has no dry_run at the MCP layer, so it must never reach a judge's allowlist."""
    bad_seat = _cli_seat(
        mcp_server_url="http://127.0.0.1:8787/mcp",
        mcp_server_name="tabularium",
        mcp_allowed_tools=("recall",),
    )
    with pytest.raises(ValidationError, match="recall") as excinfo:
        _judges_config(seat_a=bad_seat)
    assert "seat_a" in str(excinfo.value)


def test_judges_config_rejects_recall_alongside_other_tools() -> None:
    """'recall' is rejected even when it rides alongside otherwise-legitimate tools -- the guard
    checks membership, not whether it is the ONLY tool granted."""
    bad_seat = _cli_seat(
        mcp_server_url="http://127.0.0.1:8787/mcp",
        mcp_server_name="tabularium",
        mcp_allowed_tools=("decision", "recall", "decisions_query"),
    )
    with pytest.raises(ValidationError, match="recall"):
        _judges_config(seat_a=bad_seat)


@pytest.mark.parametrize(
    "unsafe_tool",
    [
        "bury",
        "save",
        "ember",
        "revoke",
        "set_hot",
        "set_audience",
        "feedback",
        "set_shelved",
        "unsupersede",
        "set_validity",
    ],
)
def test_judges_config_rejects_a_bare_but_unsafe_tool_name(unsafe_tool: str) -> None:
    """The bare-identifier shape check alone let
    a fully well-formed ``mcp_allowed_tools=("bury",)`` -- the store's lone destructive tool --
    pass ``JudgesConfig`` validation, since ``"bury"`` is a valid bare identifier and is not the
    literal string ``"recall"``. Every non-read-only tool on the MCP surface, not just ``bury``
    and ``recall``, must be refused by the same positive allowlist."""
    bad_seat = _cli_seat(
        mcp_server_url="http://127.0.0.1:8787/mcp",
        mcp_server_name="tabularium",
        mcp_allowed_tools=("context", unsafe_tool),
    )
    with pytest.raises(ValidationError, match="established side-effect-free"):
        _judges_config(seat_a=bad_seat)


@pytest.mark.parametrize(
    "hostile_tool",
    [
        "*",
        "mcp__tabularium__*",
        "context*",
        "context bury",
        "context,bury",
        "context.bury",
        "",
        "  ",
        "context\n",
        "Context",
        "-context",
    ],
)
def test_judges_config_rejects_a_non_bare_identifier_tool_name(hostile_tool: str) -> None:
    """The recall guard only rejected the exact
    string 'recall', so ``mcp_allowed_tools = ["*"]`` passed validation and rendered as claude's
    own ``--allowed-tools mcp__<server>__*`` -- verified LIVE 2026-08-14 against a real local
    Tabularium MCP server that this wildcard grants EVERY tool on the configured server, including
    a destructive one (``bury`` actually executed, reaching "Unknown id" only because the probed
    id happened not to exist). Every tool name must now be a bare lowercase identifier -- no
    wildcards, glob metacharacters, separators, whitespace, or mixed case -- closing the bypass
    regardless of which exact tool name it was trying to reach."""
    bad_seat = _cli_seat(
        mcp_server_url="http://127.0.0.1:8787/mcp",
        mcp_server_name="tabularium",
        mcp_allowed_tools=("context", hostile_tool),
    )
    with pytest.raises(ValidationError, match="bare lowercase identifier"):
        _judges_config(seat_a=bad_seat)


@pytest.mark.parametrize(
    "hostile_name",
    ["*", "tab.ularium", "tab ularium", "tabularium\n", "Tabularium", "-tabularium"],
)
def test_judges_config_rejects_a_non_bare_identifier_server_name(hostile_name: str) -> None:
    """Mirrors ``test_judges_config_rejects_a_non_bare_identifier_tool_name`` for
    ``mcp_server_name`` -- it is interpolated into both claude's ``mcp__<name>__<tool>`` tool-name
    prefix and codex's ``mcp_servers.<name>.*`` dotted config path, so a dot, wildcard, or
    whitespace in the name could itself misparse or widen scope on either CLI. An EMPTY name is
    deliberately not parametrized here -- it already hits the dedicated "mcp_server_name is
    missing" message (falsy string), covered by
    ``test_judges_config_rejects_a_seat_with_url_but_no_name`` instead of this bare-identifier
    path."""
    bad_seat = _cli_seat(
        mcp_server_url="http://127.0.0.1:8787/mcp",
        mcp_server_name=hostile_name,
        mcp_allowed_tools=("context",),
    )
    with pytest.raises(ValidationError, match="bare lowercase identifier"):
        _judges_config(seat_a=bad_seat)


def test_judges_config_error_names_the_specific_offending_seat() -> None:
    """The message identifies WHICH seat is malformed, not just 'some seat' -- proven by planting
    the same defect on seat_b instead of seat_a and checking the label follows."""
    bad_seat = _cli_seat(
        cli="codex",
        binary="/usr/bin/codex",
        mcp_server_url="http://127.0.0.1:8787/mcp",
        mcp_server_name="tabularium",
        mcp_allowed_tools=("recall",),
    )
    with pytest.raises(ValidationError) as excinfo:
        _judges_config(seat_b=bad_seat)
    message = str(excinfo.value)
    assert "seat_b" in message
    assert "seat_a:" not in message


def test_judges_config_accepts_a_well_formed_mcp_grant() -> None:
    """The happy path: url + name + a non-empty, recall-free allowlist loads cleanly and the
    fields round-trip onto the seat exactly as given."""
    seat = _cli_seat(
        mcp_server_url="http://127.0.0.1:8787/mcp",
        mcp_server_name="tabularium",
        mcp_allowed_tools=("decisions_query", "decision", "context"),
    )
    judges = _judges_config(seat_a=seat)
    assert isinstance(judges.seat_a, CliSeat)
    assert isinstance(judges.seat_b, CliSeat)
    assert judges.seat_a.mcp_server_url == "http://127.0.0.1:8787/mcp"
    assert judges.seat_a.mcp_server_name == "tabularium"
    assert judges.seat_a.mcp_allowed_tools == (
        "decisions_query",
        "decision",
        "context",
    )
    # Untouched seats keep their own default-off grant.
    assert judges.seat_b.mcp_server_url is None
    assert isinstance(judges.backstop, LitellmSeat)


def test_packaged_rubric_ships_with_no_mcp_grant_on_any_seat() -> None:
    """Default off, byte-for-byte (task brief point 3): the shipped artifact configures MCP
    access on no seat at all -- this task only makes it POSSIBLE, a separate change turns it on."""
    judges = load_rubric().judges
    assert isinstance(judges.backstop, LitellmSeat)
    for seat in (judges.seat_a, judges.seat_b):
        assert isinstance(seat, CliSeat)
        assert seat.mcp_server_url is None
        assert seat.mcp_server_name is None
        assert seat.mcp_allowed_tools == ()
        assert seat.mcp_grant is None


async def test_judge_deny_list_covers_every_registered_tool_outside_the_safe_set() -> None:
    """Complete-by-construction guard (critical finding, 2026-08-14 ambient-permission-bypass
    fix): ``JUDGE_DENY_MCP_TOOLS`` is hand-maintained data, and hand-maintained data drifts. This
    test is the actual deliverable, not the list itself -- it introspects the MCP server's REAL
    registered tool names (``app.list_tools()``, not a hand-copied inventory) and fails the moment
    a tool added to ``server.py`` later lands outside BOTH ``JUDGE_SAFE_MCP_TOOLS`` (the positive
    grant allowlist) and ``JUDGE_DENY_MCP_TOOLS`` (the claude ``--disallowed-tools`` / codex
    ``disabled_tools`` belt-and-braces deny list) -- exactly the gap that let the ambient-permission
    bypass do real damage: a tool nobody had classified either way.

    Falsify: add a new tool to a throwaway app registration below without adding it to either
    constant -- this test goes red naming the orphaned tool; delete the throwaway registration to
    restore.
    """
    adapter = MemoryAdapter(embedding_model="fake/nomic", dim=16)
    embedder = FakeEmbedder(dim=16)
    engine = Tabularium(adapter, embedder)
    await engine.initialize()
    app = make_app(engine)

    registered = {t.name for t in await app.list_tools()}

    assert JUDGE_SAFE_MCP_TOOLS.isdisjoint(JUDGE_DENY_MCP_TOOLS), (
        "a tool cannot be both explicitly granted and explicitly denied"
    )
    classified = JUDGE_SAFE_MCP_TOOLS | JUDGE_DENY_MCP_TOOLS
    unclassified = registered - classified
    assert not unclassified, (
        f"tool(s) {sorted(unclassified)!r} registered on the MCP server but absent from BOTH "
        "JUDGE_SAFE_MCP_TOOLS and JUDGE_DENY_MCP_TOOLS -- classify explicitly, one way or the "
        "other, before this can ship"
    )
    stale = classified - registered
    assert not stale, (
        f"{sorted(stale)!r} named in rubric.py's safety sets but no longer a registered tool -- "
        "stale entries are harmless here but signal the two files have drifted"
    )


def test_load_rubric_from_path_rejects_recall_in_mcp_allowlist_naming_the_seat(
    tmp_path: Path,
) -> None:
    """Bridges the unit-level ``JudgesConfig`` validator tests above with the real ``--rubric``
    CLI seam: a custom artifact carrying 'recall' in a seat's allowlist must fail to load with a
    ``RubricLoadError`` naming both the path and the offending seat."""
    text = _MINIMAL_VALID_TOML.replace(
        '[judges.seat_a]\nadapter = "cli"\ncli = "claude"\nmodel = "test-model-a"\n'
        'binary = "/usr/bin/true"\n',
        '[judges.seat_a]\nadapter = "cli"\ncli = "claude"\nmodel = "test-model-a"\n'
        'binary = "/usr/bin/true"\n'
        'mcp_server_url = "http://127.0.0.1:8787/mcp"\n'
        'mcp_server_name = "tabularium"\n'
        'mcp_allowed_tools = ["recall"]\n',
    )
    assert "mcp_allowed_tools" in text, "fixture substitution did not match -- test is inert"
    custom_path = tmp_path / "custom.toml"
    custom_path.write_text(text, encoding="utf-8")
    with pytest.raises(RubricLoadError) as excinfo:
        load_rubric_from_path(custom_path)
    message = str(excinfo.value)
    assert str(custom_path) in message
    assert "seat_a" in message
    assert "recall" in message


def test_rubric_parts_are_frozen() -> None:
    rubric = load_rubric()
    with pytest.raises(ValidationError):
        rubric.version = "tampered"


def test_nothing_in_this_module_reaches_a_judge_or_the_store() -> None:
    """Guards the acceptance criterion mechanically: the rubric module's own source imports neither
    a storage adapter nor litellm (the engine's generative-call path).

    Checks for the actual import statement, not the bare substring "litellm": the seats' ``adapter``
    tag is a legitimate DATA tag naming which adapter *kind* a seat uses, which is exactly what this
    test wants to allow. The rubric knows the shape of judge config without ever calling litellm.
    """
    source = (_REPO_ROOT / "src" / "tabularium" / "engine" / "rubric.py").read_text()
    assert "tabularium.storage" not in source
    assert "import litellm" not in source


def test_loader_is_the_only_reader_of_the_artifact() -> None:
    """Nothing besides ``engine/rubric.py`` should open ``rubric.toml`` — every other module that
    wants rubric content must go through ``load_rubric()``.
    """
    rubric_module = _REPO_ROOT / "src" / "tabularium" / "engine" / "rubric.py"
    this_file = Path(__file__).resolve()
    offenders: list[Path] = []
    for base in (_REPO_ROOT / "src", _REPO_ROOT / "tests", _REPO_ROOT / "scripts"):
        if not base.exists():
            continue
        for py_file in base.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            if py_file in (rubric_module, this_file):
                continue
            if "rubric.toml" in py_file.read_text(encoding="utf-8", errors="ignore"):
                offenders.append(py_file)
    assert offenders == []


# ---------------------------------------------------------------------------
# load_rubric_from_path — the ``--rubric PATH`` CLI seam (dry-run only, cli.py's own gate).
#
# This is the one file besides ``engine/rubric.py`` allowed to read the packaged artifact's own
# bytes (``_ARTIFACT_PATH``) — the "only rubric.py opens rubric.toml" guard above carves out
# exactly this file. ``packaged_toml_with_revisit_min_age_days`` below is exported so
# ``test_cli.py``'s own end-to-end ``--rubric`` acceptance test can build a temp file whose only
# delta from the packaged artifact is that one parameter without gaining a second reader of the
# raw artifact itself.
# ---------------------------------------------------------------------------

# A minimal, self-contained, schema-valid rubric TOML — every required field present, values
# chosen only to satisfy the schema's own bounds. Used as the base for the mutation tests below,
# which each break exactly one thing about it.
_MINIMAL_VALID_TOML = """
[meta]
schema = 1

[parameters]
k = 3
similarity_floor = 0.5
batch_cap = 10
unclear_ceiling = 0.2
pool_limit = 100
report_render_cap = 5
revisit_batch_cap = 5
revisit_pass_cap = 10
revisit_min_age_days = 30
revisit_interval_days = 90
revisit_stable_multiplier = 2.0
revisit_contested_multiplier = 0.5
revisit_max_interval_days = 360

[relevance]
scale = ["shelve-clear", "shelve-lean", "keep-lean", "keep-clear"]
anchor = "test anchor"
prose = "test relevance prose"

[revisit]
prose = "test revisit prose"

[falsity]
verdicts = ["coexist", "replace"]
anchor = "test anchor"
prose = "test falsity prose"

[judges.seat_a]
adapter = "cli"
cli = "claude"
model = "test-model-a"
binary = "/usr/bin/true"

[judges.seat_b]
adapter = "cli"
cli = "codex"
model = "test-model-b"
binary = "/usr/bin/true"

[judges.backstop]
adapter = "litellm"
model = "deepseek/deepseek-chat"

[caseset]
prompt_template = "test caseset prompt {existing} {incoming}"

[caseset.judge]
adapter = "cli"
cli = "claude"
model = "test-model-c"
binary = "/usr/bin/true"
"""


def packaged_toml_with_revisit_min_age_days(days: int) -> str:
    """The packaged artifact's own text with ``revisit_min_age_days`` swapped to ``days`` — the
    ONLY delta. Exported for ``test_cli.py``'s end-to-end ``--rubric`` acceptance test, which
    needs a temp file that differs from the packaged artifact in exactly this one parameter.
    """
    text = _ARTIFACT_PATH.read_text(encoding="utf-8")
    original = "revisit_min_age_days = 30"
    assert original in text, "packaged artifact's revisit_min_age_days line changed shape"
    return text.replace(original, f"revisit_min_age_days = {days}", 1)


def test_load_rubric_from_path_loads_a_valid_custom_file(tmp_path: Path) -> None:
    """Happy path: a well-formed custom TOML loads to a real ``Rubric``, distinct from the
    packaged one only in content."""
    custom_path = tmp_path / "custom.toml"
    custom_path.write_text(_MINIMAL_VALID_TOML, encoding="utf-8")

    rubric = load_rubric_from_path(custom_path)

    assert isinstance(rubric, Rubric)
    assert rubric.parameters.k == 3
    assert rubric.relevance.prose == "test relevance prose"
    assert rubric.judges.seat_a.model == "test-model-a"


def test_load_rubric_from_path_accepts_a_schema_1_rubric_with_no_caseset_section(
    tmp_path: Path,
) -> None:
    """A pre-PR schema-1 rubric — the shape
    shipped on ``main`` before the admission case-set landed, e.g. a deployed Cleaner
    rubric — carries no ``[caseset]`` section at all. It must still load and run a live pass;
    only the admission test harness needs the section, never the Cleaner itself."""
    pre_caseset_toml = _MINIMAL_VALID_TOML.rsplit("[caseset]", 1)[0]
    assert "[caseset]" not in pre_caseset_toml

    custom_path = tmp_path / "pre-caseset.toml"
    custom_path.write_text(pre_caseset_toml, encoding="utf-8")

    rubric = load_rubric_from_path(custom_path)

    assert isinstance(rubric, Rubric)
    assert rubric.caseset is None


def test_load_rubric_from_path_missing_file_names_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.toml"
    with pytest.raises(RubricLoadError) as excinfo:
        load_rubric_from_path(missing)
    assert str(missing) in str(excinfo.value)


def test_load_rubric_from_path_unreadable_names_the_path(tmp_path: Path) -> None:
    """A directory is not readable as a file — portable across environments (unlike chmod, which
    is a no-op for a process running as root)."""
    directory_path = tmp_path / "a-directory"
    directory_path.mkdir()
    with pytest.raises(RubricLoadError) as excinfo:
        load_rubric_from_path(directory_path)
    assert str(directory_path) in str(excinfo.value)


def test_load_rubric_from_path_malformed_toml_names_the_path_and_problem(tmp_path: Path) -> None:
    custom_path = tmp_path / "custom.toml"
    custom_path.write_text("this is [ not valid toml =", encoding="utf-8")
    with pytest.raises(RubricLoadError) as excinfo:
        load_rubric_from_path(custom_path)
    message = str(excinfo.value)
    assert str(custom_path) in message
    assert "TOML" in message


def test_load_rubric_from_path_missing_section_names_the_path_and_problem(tmp_path: Path) -> None:
    """Well-formed TOML, but an entire required section (``[judges]``) is absent — a ``KeyError``
    inside ``_rubric_from_toml``, not a pydantic ``ValidationError``."""
    text = _MINIMAL_VALID_TOML.split("[judges.seat_a]")[0]
    custom_path = tmp_path / "custom.toml"
    custom_path.write_text(text, encoding="utf-8")
    with pytest.raises(RubricLoadError) as excinfo:
        load_rubric_from_path(custom_path)
    message = str(excinfo.value)
    assert str(custom_path) in message
    assert "missing a required section or field" in message


def test_load_rubric_from_path_schema_failure_names_the_path_and_problem(tmp_path: Path) -> None:
    """Every section is present, but one value fails the rubric's own pydantic schema
    (``similarity_floor`` must be within ``[0.0, 1.0]``) — a ``pydantic.ValidationError``, not a
    ``KeyError``, and the message must say so distinctly (round-trip proof for the ordering fix:
    ``ValidationError`` subclasses ``ValueError``, so the except clauses must be ordered
    specific-before-broad or every schema failure mislabels itself)."""
    text = _MINIMAL_VALID_TOML.replace("similarity_floor = 0.5", "similarity_floor = 5.0")
    custom_path = tmp_path / "custom.toml"
    custom_path.write_text(text, encoding="utf-8")
    with pytest.raises(RubricLoadError) as excinfo:
        load_rubric_from_path(custom_path)
    message = str(excinfo.value)
    assert str(custom_path) in message
    assert "failed schema validation" in message


@pytest.mark.parametrize("bad_schema", ["inf", "nan", "true", "1.9"])
def test_load_rubric_from_path_malformed_meta_schema_never_raises_a_bare_traceback(
    tmp_path: Path, bad_schema: str
) -> None:
    """``_rubric_from_toml`` used to compute
    ``int(data["meta"]["schema"])`` directly -- a syntactically valid TOML float like ``schema =
    inf`` parses fine and then ``int(float("inf"))`` raises an UNCAUGHT ``OverflowError``, past
    ``load_rubric_from_path``'s own exception translation, a bare traceback reaching the CLI.
    ``schema = true``/``schema = 1.9`` were worse: silently ACCEPTED (``int()`` coerces both
    without complaint) rather than rejected. All four must now raise ``RubricLoadError`` naming
    the path, via ``_RubricMeta``'s ``strict=True`` int field."""
    text = _MINIMAL_VALID_TOML.replace("schema = 1", f"schema = {bad_schema}")
    custom_path = tmp_path / "custom.toml"
    custom_path.write_text(text, encoding="utf-8")
    with pytest.raises(RubricLoadError) as excinfo:
        load_rubric_from_path(custom_path)
    message = str(excinfo.value)
    assert str(custom_path) in message
    assert "failed schema validation" in message


@pytest.mark.parametrize(
    ("field_assignment", "replacement"),
    [
        ("k = 3", "k = true"),
        ("batch_cap = 10", "batch_cap = true"),
        ("revisit_min_age_days = 30", "revisit_min_age_days = true"),
        ("revisit_stable_multiplier = 2.0", "revisit_stable_multiplier = inf"),
    ],
)
def test_load_rubric_from_path_rejects_malformed_parameter_values(
    tmp_path: Path, field_assignment: str, replacement: str
) -> None:
    """Before ``RubricParameters`` gained
    ``strict=True``/``allow_inf_nan=False``, a bare pydantic ``int``/``float`` field silently
    COERCED a bool to ``1``/``0`` and accepted an unbounded ``inf`` past its own ``gt=0`` check —
    a custom rubric with ``k = true`` or ``revisit_stable_multiplier = inf`` loaded cleanly and
    would judge the wrong pool or batch shape with no error at all. All four must now raise
    ``RubricLoadError`` naming the path."""
    text = _MINIMAL_VALID_TOML.replace(field_assignment, replacement)
    assert text != _MINIMAL_VALID_TOML, "fixture assignment string not found -- test is inert"
    custom_path = tmp_path / "custom.toml"
    custom_path.write_text(text, encoding="utf-8")
    with pytest.raises(RubricLoadError) as excinfo:
        load_rubric_from_path(custom_path)
    message = str(excinfo.value)
    assert str(custom_path) in message
    assert "failed schema validation" in message


def test_load_rubric_from_path_rejects_an_unrecognized_parameter_key(tmp_path: Path) -> None:
    """Pydantic v2 ignores unrecognized keys by
    default, so a custom rubric that keeps every required key intact but ALSO carries an extra one
    -- a misspelled second attempt at overriding a value, e.g. `kk = 999` alongside the untouched,
    still-required `k = 3` -- used to load cleanly with the extra key silently dropped, leaving
    the caller believing they had overridden a value they had not. `extra="forbid"` on
    `RubricParameters` turns this into a named `RubricLoadError`."""
    text = _MINIMAL_VALID_TOML.replace("k = 3", "k = 3\nkk = 999")
    assert "kk = 999" in text
    custom_path = tmp_path / "custom.toml"
    custom_path.write_text(text, encoding="utf-8")
    with pytest.raises(RubricLoadError) as excinfo:
        load_rubric_from_path(custom_path)
    message = str(excinfo.value)
    assert str(custom_path) in message
    assert "failed schema validation" in message


def test_load_rubric_from_path_rejects_an_unrecognized_seat_key(tmp_path: Path) -> None:
    """``extra="forbid"`` on the seat models makes a misspelled seat key a ``RubricLoadError``
    instead of a value silently dropped."""
    text = _MINIMAL_VALID_TOML.replace("[judges.seat_a]", '[judges.seat_a]\nmodle = "typo"', 1)
    assert 'modle = "typo"' in text
    custom_path = tmp_path / "custom.toml"
    custom_path.write_text(text, encoding="utf-8")
    with pytest.raises(RubricLoadError) as excinfo:
        load_rubric_from_path(custom_path)
    message = str(excinfo.value)
    assert str(custom_path) in message
    assert "failed schema validation" in message


def test_load_rubric_from_path_version_differs_from_packaged_for_differing_content(
    tmp_path: Path,
) -> None:
    """Prove ``rubric_version`` is content-addressed off the CUSTOM file's own bytes: two
    differing custom files get two differing versions, and both differ from the packaged one."""
    packaged = load_rubric()

    path_a = tmp_path / "a.toml"
    path_a.write_text(_MINIMAL_VALID_TOML, encoding="utf-8")
    path_b = tmp_path / "b.toml"
    path_b.write_text(_MINIMAL_VALID_TOML.replace("k = 3", "k = 4"), encoding="utf-8")

    rubric_a = load_rubric_from_path(path_a)
    rubric_b = load_rubric_from_path(path_b)

    assert rubric_a.version != packaged.version
    assert rubric_b.version != packaged.version
    assert rubric_a.version != rubric_b.version


def test_load_rubric_from_path_does_not_poison_the_packaged_cache(tmp_path: Path) -> None:
    """The cache-poisoning hazard, proven directly: loading a custom rubric — including one that
    happens to run BEFORE the packaged rubric has ever been loaded in this process — must never
    change what ``load_rubric()`` itself subsequently returns. ``load_rubric.cache_clear()`` is
    called first so this test does not depend on test execution order having already warmed the
    packaged loader's cache."""
    load_rubric.cache_clear()

    custom_path = tmp_path / "custom.toml"
    custom_path.write_text(_MINIMAL_VALID_TOML.replace("k = 3", "k = 99"), encoding="utf-8")
    custom = load_rubric_from_path(custom_path)

    packaged_after = load_rubric()

    assert custom.version != packaged_after.version
    assert packaged_after.version == _EXPECTED_VERSION
    assert packaged_after.parameters.k != 99
    # Cached identity: a second call returns the SAME object, proving the packaged loader's own
    # `lru_cache` is intact and was never primed with the custom rubric.
    assert load_rubric() is packaged_after
