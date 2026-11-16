"""Judge seam: the engine's generative call, made out of band and never on the synchronous save
path.

``JudgeAdapter`` is the interface. Two adapters implement it:

- ``SubprocessJudgeAdapter`` runs ``claude -p`` or ``codex exec``, selected by ``cli=``. The prompt
  goes on stdin and never in argv, because a single argv element is capped at 128 KiB and a rendered
  prompt can exceed that.
- ``LitellmJudgeAdapter`` is the key-based backstop: DeepSeek through litellm, which reads
  ``DEEPSEEK_API_KEY``.

Which judge fills which seat is data in the rubric (``JudgesConfig`` in ``engine/rubric.py``).
``build_judge_adapter`` maps one seat config to one adapter; it makes no routing decision.

Index protocol: a judge receives and returns batch-local integers 1..N. ``render_batch_prompt``
takes bare content strings, never records, so no record id reaches a model. The caller resolves
indices back to ids from the batch it holds.

Outcomes: ``adapter.call`` raises ``JudgeUnavailable`` and ``parse_batch_verdicts`` raises
``JudgeInvalidOutput`` (or ``UnclearFieldNotStrict`` for a bad schema). ``judge_batch`` catches
``Exception`` from both and returns a ``JudgeOutcome`` whose status is ``success``,
``judge_unavailable``, ``invalid_output``, ``unexpected`` or ``prompt_too_large``. Each outcome
records the judge identity, the rubric version and an instruction-file fingerprint, failures
included; ``JudgeOutcome.as_call_record`` projects it to a ``JudgeCallRecord`` for the pass report.

No retry loop wraps a call here. ``SubprocessJudgeAdapter`` makes one attempt per call and
``LitellmJudgeAdapter`` passes ``num_retries`` to litellm.

``unclear`` is parsed strictly. ``parse_batch_verdicts`` inspects any ``item_model`` and raises
``UnclearFieldNotStrict`` if it declares a non-strict ``unclear`` field, so lax coercion of
``0``/``1``/``"false"`` cannot turn an unjudgeable record into an answered one.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, get_args, get_origin, runtime_checkable

import litellm
from pydantic import BaseModel, Field, Strict, ValidationError

from tabularium.engine.judge_policy import JUDGE_DENY_MCP_TOOLS, McpGrant, validate_mcp_grant
from tabularium.engine.rubric import CliSeat, JudgeSeatConfig, LitellmSeat
from tabularium.tokens import TokenCounter

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The three outcomes
# ---------------------------------------------------------------------------


class JudgeUnavailable(Exception):
    """The judge could not be reached: auth failure, timeout, non-zero subprocess exit, or a
    provider error from litellm. A response that arrived and failed to parse is
    ``JudgeInvalidOutput``.
    """


class JudgeInvalidOutput(Exception):
    """A response arrived but does not satisfy the schema or the index protocol."""


class UnclearFieldNotStrict(TypeError):
    """An ``item_model`` passed to ``parse_batch_verdicts`` declares an ``unclear`` field that is
    not strict, neither on the field nor through ``model_config = {"strict": True}``.

    A schema-authoring defect, so a ``TypeError`` and not ``JudgeInvalidOutput``. It is raised
    whatever the response contains.
    """


# ---------------------------------------------------------------------------
# The index protocol
# ---------------------------------------------------------------------------


class IndexedVerdictBase(BaseModel):
    """Base of every per-record verdict item.

    ``index`` is the only identity a judge sees: a batch-local integer 1..N. It is strict because a
    JSON boolean would otherwise coerce to an in-range int (``bool`` subclasses ``int``).
    """

    model_config = {"extra": "forbid"}

    reply_rules: ClassVar[tuple[str, ...]] = ()

    index: int = Field(strict=True)


class UnclearVerdictBase(IndexedVerdictBase):
    """Base for verdict items that carry a strict ``unclear`` flag.

    Inheriting from it does not enforce strictness; ``parse_batch_verdicts`` checks any
    ``item_model``. Items without an ``unclear`` field extend ``IndexedVerdictBase`` directly.
    """

    unclear: bool = Field(strict=True)


def _check_unclear_is_strict(item_model: type[BaseModel]) -> None:
    """Raise ``UnclearFieldNotStrict`` if ``item_model`` declares a non-strict ``unclear`` field.

    A model with no ``unclear`` field passes. This is a schema check and does not look at any
    response.
    """
    if item_model.model_config.get("strict"):
        return  # model-level strict=True covers every field, `unclear` included
    info = item_model.model_fields.get("unclear")
    if info is None:
        return  # no such field on this schema -- nothing to enforce (e.g. the falsity axis)
    if any(isinstance(m, Strict) and m.strict for m in info.metadata):
        return
    raise UnclearFieldNotStrict(
        f"{item_model.__name__} declares an 'unclear' field without strict=True (and without "
        "model_config = {'strict': True}). This reproduces the "
        "bool(v.get('unclear', False)) defect: a missing or malformed field would silently "
        "coerce instead of raising. Declare `unclear: bool = Field(strict=True)`, or extend "
        "UnclearVerdictBase."
    )


def parse_batch_verdicts[V: IndexedVerdictBase](
    raw: str, *, item_model: type[V], batch_size: int
) -> list[V]:
    """Parse ``raw`` against the index protocol and return verdicts ordered by index
    1..``batch_size``.

    Raises ``UnclearFieldNotStrict`` before reading ``raw`` when ``item_model`` has a non-strict
    ``unclear`` field. Raises ``JudgeInvalidOutput`` for any of:

    - malformed JSON, a non-object response, or no ``verdicts`` list;
    - an item that fails ``item_model`` validation, including a non-int index or a missing or
      non-boolean strict field;
    - a duplicate index or an index outside ``1..batch_size``;
    - a missing verdict for any index in ``1..batch_size``.

    The result covers every index exactly once or the call raises; there is no partial list. The
    function does no I/O.
    """
    _check_unclear_is_strict(item_model)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JudgeInvalidOutput(f"response is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise JudgeInvalidOutput(f"response is not a JSON object (got {type(payload).__name__})")

    raw_verdicts = payload.get("verdicts")
    if not isinstance(raw_verdicts, list):
        raise JudgeInvalidOutput("response missing a 'verdicts' array")

    by_index: dict[int, V] = {}
    for raw_item in raw_verdicts:
        try:
            item = item_model.model_validate(raw_item)
        except ValidationError as exc:
            raise JudgeInvalidOutput(f"verdict item failed schema validation: {exc}") from exc
        if item.index in by_index:
            raise JudgeInvalidOutput(f"duplicate index {item.index}")
        if not (1 <= item.index <= batch_size):
            raise JudgeInvalidOutput(f"index {item.index} out of range 1..{batch_size}")
        by_index[item.index] = item

    missing = sorted(set(range(1, batch_size + 1)) - by_index.keys())
    if missing:
        raise JudgeInvalidOutput(f"{len(missing)} record(s) missing a verdict: {missing}")

    return [by_index[i] for i in range(1, batch_size + 1)]


_REPLY_ENVELOPE_RULES: tuple[str, ...] = (
    "The first character of your reply is `{`.",
    "No code fences and no text before or after the JSON object.",
    "Give one entry for each index from 1 to N, each exactly once.",
    "Add no fields beyond the ones listed.",
)


def _literal_values(annotation: Any) -> tuple[Any, ...] | None:
    if get_origin(annotation) is Literal:
        return get_args(annotation)
    for arg in get_args(annotation):
        if arg is not type(None):
            values = _literal_values(arg)
            if values is not None:
                return values
    return None


def _unwrap_optional(annotation: Any) -> Any:
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1 and len(get_args(annotation)) == 2 and get_origin(annotation) is not Literal:
        return args[0]
    return annotation


def _field_type_text(info: Any) -> str:
    annotation = _unwrap_optional(info.annotation)
    literals = _literal_values(annotation)
    if literals is not None:
        return "one of " + ", ".join(json.dumps(v) for v in literals)

    strict = any(isinstance(m, Strict) and m.strict for m in info.metadata)
    if annotation is int:
        return "a JSON integer, not a string" if strict else "a JSON integer"
    if annotation is bool:
        return "a JSON boolean, not a string" if strict else "a JSON boolean"
    if annotation is str:
        return "a JSON string"
    return "a JSON value"


def render_reply_format(item_model: type[IndexedVerdictBase]) -> str:
    """Render the ``# Reply format`` section for ``item_model``.

    The section is the fixed envelope rules, one line per model field (name, JSON type, allowed
    Literal values, ``Field(description=...)``), then the model's ``reply_rules``. It is generated
    from the model so it follows the schema ``parse_batch_verdicts`` enforces. Class docstrings are
    not emitted.
    """
    lines = [
        "# Reply format",
        "",
        'Reply with one JSON object: `{"verdicts": [ ... ]}`, holding one entry per record.',
        "",
    ]
    lines.extend(f"- {rule}" for rule in _REPLY_ENVELOPE_RULES)
    lines.extend(["", "Each entry has these fields:", ""])
    for name, info in item_model.model_fields.items():
        requirement = "required" if info.is_required() else "optional"
        text = f"- `{name}` ({requirement}): {_field_type_text(info)}"
        if info.description:
            text += f". {info.description}"
        lines.append(text)
    if item_model.reply_rules:
        lines.append("")
        lines.extend(f"- {rule}" for rule in item_model.reply_rules)
    lines.append("")
    return "\n".join(lines)


def render_batch_prompt(
    *, rubric_prose: str, contents: Sequence[str], item_model: type[IndexedVerdictBase]
) -> str:
    """Render a judge prompt: the rubric prose, each record under a batch-local ``## N.`` heading (N
    = 1..len(contents)), then the reply format generated from ``item_model``.

    Takes bare content strings so no record id can reach a judge. The caller keeps its own
    index-to-id mapping.
    """
    lines = [rubric_prose, "", "---", "", f"# Your batch — {len(contents)} records", ""]
    lines.extend(_render_entry(i, content) for i, content in enumerate(contents, start=1))
    lines.append(render_reply_format(item_model))
    return "\n".join(lines)


def _render_entry(index: int, content: str) -> str:
    """One record's block in a batch prompt: the ``## index.`` heading line, the content line and a
    blank line. ``render_batch_prompt`` joins blocks with a newline, so a block adds
    ``_entry_size`` characters to the prompt.
    """
    return f"## {index}.\n{content}\n"


def _entry_size(index: int, content: str) -> int:
    """The characters one entry adds to a rendered prompt: its block plus the joining newline."""
    return len(_render_entry(index, content)) + 1


#: No rendered judge prompt may exceed this many chars. Codex rejects input over 1,048,576
#: characters; 600K leaves a wide margin below that.
PROMPT_CHAR_CEILING = 600_000


_BATCH_COUNT_DIGIT_SLACK = 8


def pack_batches(
    contents: Sequence[str],
    *,
    count_cap: int,
    rubric_prose: str,
    item_model: type[IndexedVerdictBase],
    ceiling: int = PROMPT_CHAR_CEILING,
) -> list[list[int]]:
    """Group positions in ``contents`` into batches, input order kept.

    A batch is closed when the next unit would push its rendered prompt past ``ceiling`` or reach
    ``count_cap`` units. A unit too large on its own becomes a batch of one, which ``judge_batch``
    refuses with ``prompt_too_large``.

    The unit size of an entry is ``_entry_size``, the same renderer ``render_batch_prompt`` joins,
    so the two cannot drift. The base size comes from a real render with no contents.
    """
    base = _BATCH_COUNT_DIGIT_SLACK + len(
        render_batch_prompt(rubric_prose=rubric_prose, contents=[], item_model=item_model)
    )
    batches: list[list[int]] = []
    current: list[int] = []
    current_size = base
    for position, content in enumerate(contents):
        unit = _entry_size(len(current) + 1, content)
        if current and (len(current) >= count_cap or current_size + unit > ceiling):
            batches.append(current)
            current = []
            current_size = base
            unit = _entry_size(1, content)
        current.append(position)
        current_size += unit
    if current:
        batches.append(current)
    return batches


# ---------------------------------------------------------------------------
# One interface
# ---------------------------------------------------------------------------


@runtime_checkable
class JudgeAdapter(Protocol):
    """Async judge seam: send a prompt, get raw text back.

    Implementors raise ``JudgeUnavailable`` when the judge could not be reached. A response that
    arrives is returned unparsed; ``parse_batch_verdicts`` validates it for every adapter.
    """

    @property
    def identity(self) -> str:
        """Human-readable judge identity, stamped on every ``JudgeOutcome``."""
        ...

    def instruction_fingerprint(self) -> str:
        """Fingerprint of the local instruction files this seat's next call would read.

        Computed on each call, never cached at construction. One of ``"sha256:<hex>"`` (at least one
        file readable), ``"absent"`` (none found, never the hash of an empty input) or
        ``"not_applicable"`` (a seat kind that reads none). Implementations should not raise;
        ``judge_batch`` records ``"absent"`` if one does.
        """
        ...

    async def call(self, prompt: str) -> str:
        """Send ``prompt`` to the judge and return its raw text response.

        Raises:
            JudgeUnavailable: on any failure to reach the judge or get a response back.
        """
        ...


# ---------------------------------------------------------------------------
# Instruction-file fingerprinting. The CLAUDE.md/AGENTS.md chain a seat's subprocess reads is
# outside rubric_version's hash; the fingerprint of that chain is recorded next to it on every
# JudgeCallRecord, so a behaviour shift with no rubric change can be attributed.
# ---------------------------------------------------------------------------

#: No instruction file was found in the chain. Distinct from the sha256 of empty input.
_FINGERPRINT_ABSENT = "absent"

#: A seat kind that reads no instruction file (the litellm backstop). Distinct from
#: `_FINGERPRINT_ABSENT`, which means a file was looked for and not found.
_FINGERPRINT_NOT_APPLICABLE = "not_applicable"

#: Default only for `JudgeCallRecord.instruction_fingerprint`, so stored rows from before the field
#: existed parse. Live calls never produce it.
_FINGERPRINT_UNRECORDED = "unrecorded"


def _find_git_root(start: Path) -> Path | None:
    """Return the nearest directory at or above ``start`` that contains ``.git`` (file or
    directory), or ``None`` when ``start`` is not inside a git repository.
    """
    current = start
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _claude_instruction_chain(cwd: Path) -> list[Path]:
    """``CLAUDE.md`` candidate paths for ``cwd``, ordered from the filesystem root down to ``cwd``.

    Models observed claude CLI behaviour: it reads the whole directory chain and does not stop at a
    git repository boundary.
    """
    dirs: list[Path] = []
    current = cwd.resolve()
    while True:
        dirs.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    dirs.reverse()
    return [d / "CLAUDE.md" for d in dirs]


def _codex_instruction_chain(cwd: Path) -> list[Path]:
    """``AGENTS.md`` candidate paths for ``cwd``, ordered from the git root down to ``cwd``.

    Models observed codex CLI behaviour, which differs from claude's: the walk stops at the nearest
    enclosing git root, and outside a git repository only ``cwd`` itself is checked.

    NOTE: the codex CLI also appears to name a global ``~/AGENTS.md``, ``/AGENTS.md`` and
    per-directory ``AGENTS.override.md``. None of that was exercised, so none of it is hashed.
    """
    resolved = cwd.resolve()
    root = _find_git_root(resolved)
    if root is None:
        return [resolved / "AGENTS.md"]
    dirs: list[Path] = []
    current = resolved
    while True:
        dirs.append(current)
        if current == root:
            break
        current = current.parent
    dirs.reverse()
    return [d / "AGENTS.md" for d in dirs]


def _fingerprint_chain(paths: Sequence[Path]) -> str:
    """Hash ``paths`` in order, folding in each readable file's path string and bytes.

    A content change and a move or reorder both change the result. A file that cannot be read
    (``OSError``) is skipped with a warning. A chain that contributes nothing returns
    ``_FINGERPRINT_ABSENT``, never the hash of empty input.
    """
    hasher = hashlib.sha256()
    contributed = False
    for path in paths:
        try:
            content = path.read_bytes()
        except OSError as exc:
            _log.warning("instruction fingerprint: could not read %s: %s", path, exc)
            continue
        hasher.update(str(path).encode("utf-8", errors="replace"))
        hasher.update(b"\0")
        hasher.update(content)
        hasher.update(b"\0")
        contributed = True
    if not contributed:
        return _FINGERPRINT_ABSENT
    return f"sha256:{hasher.hexdigest()}"


# ---------------------------------------------------------------------------
# Adapter 1: the two subscription CLIs
# ---------------------------------------------------------------------------


def _stderr_diagnostic(stderr: bytes, *, limit: int) -> str:
    """Return the last ``limit`` characters of ``stderr``, decoded with replacement.

    NOTE: only the tail is used, with no keyword or line preference. A failed CLI can echo the whole
    prompt before its real error, and a prompt can contain any pattern a scan would look for, so a
    scan can select text from the echo instead of the failure.
    """
    return stderr.decode(errors="replace")[-limit:]


def _failure_diagnostic(stdout: bytes, stderr: bytes, *, limit: int) -> str:
    """Diagnostic for a non-zero CLI exit: the stderr tail when stderr has non-whitespace content,
    else the stdout tail prefixed ``(stdout) ``.

    ``claude -p`` reports some errors on stdout and leaves stderr empty. Stdout is not appended to a
    non-blank stderr: codex stdout is the echoed transcript and would push the real error out of the
    excerpt.
    """
    if stderr.strip():
        return _stderr_diagnostic(stderr, limit=limit)
    return "(stdout) " + _stderr_diagnostic(stdout, limit=limit)


class SubprocessJudgeAdapter:
    """One adapter for two CLIs, ``claude -p`` and ``codex exec``, selected by ``cli``.

    Both invocations:

    - Carry the prompt on stdin through a fresh pipe that ``communicate`` closes, never in argv.
    - Run ``binary`` under the inherited environment and working directory; no ``env=`` or ``cwd=``
      override is passed.
    - Pass the configured ``model`` explicitly.

    Any non-zero exit, including an auth failure, raises ``JudgeUnavailable``. A call makes one
    attempt. ``timeout`` (seconds) applies per call; ``build_judge_adapter`` passes the seat's
    ``CliSeat.timeout``.
    """

    _DEFAULT_TIMEOUT = 1200.0
    _STDERR_DIAGNOSTIC_LIMIT = 2000

    def __init__(
        self,
        *,
        cli: Literal["claude", "codex"],
        binary: str,
        model: str,
        timeout: float = _DEFAULT_TIMEOUT,
        grant: McpGrant | None = None,
    ) -> None:
        self._cli = cli
        self._binary = binary
        self._model = model
        self._timeout = timeout
        self._grant = grant

    @property
    def identity(self) -> str:
        return f"{self._cli}:{self._model}"

    @property
    def timeout(self) -> float:
        """The per-call timeout in seconds."""
        return self._timeout

    @property
    def grant(self) -> McpGrant | None:
        """The MCP grant rendered into the CLI's flags, or ``None`` for no MCP access."""
        return self._grant

    def instruction_fingerprint(self) -> str:
        """Fingerprint of the instruction-file chain this seat's subprocess would read from the
        current working directory, which the subprocess inherits.

        Computed on each call. An unreadable file is handled by ``_fingerprint_chain``; an
        ``OSError`` while resolving the chain returns ``"absent"``.
        """
        cwd = Path.cwd()
        try:
            chain = (
                _claude_instruction_chain(cwd)
                if self._cli == "claude"
                else _codex_instruction_chain(cwd)
            )
        except OSError as exc:
            _log.warning(
                "instruction fingerprint: chain resolution failed for %s: %s", self._cli, exc
            )
            return _FINGERPRINT_ABSENT
        return _fingerprint_chain(chain)

    async def call(self, prompt: str) -> str:
        if self._cli == "claude":
            return await self._call_claude(prompt)
        return await self._call_codex(prompt)

    def _claude_mcp_argv(self) -> list[str]:
        """Render this seat's MCP grant into claude's flags.

        No grant (``grant is None``): ``["--allowed-tools", ""]``.

        A grant: ``--mcp-config`` with an inline JSON server entry, ``--strict-mcp-config``,
        ``--allowed-tools mcp__<name>__<tool>`` for each tool in the grant's ``allowed_tools``, then
        ``--disallowed-tools mcp__<name>__<tool>`` for each tool in ``JUDGE_DENY_MCP_TOOLS``,
        sorted.

        NOTE: security-relevant. ``--allowed-tools`` is not exhaustive: observed live, claude merges
        ambient permission settings from the inherited working directory on top of it, so an ambient
        allow entry can enable a tool outside the grant. The explicit deny list closes that, since
        deny takes precedence. It is rendered from the fixed ``JUDGE_DENY_MCP_TOOLS``, not from the
        complement of this seat's grant. Observed: an ungranted tool is shown to the model and its
        call is refused.
        """
        grant = self._grant
        if grant is None:
            return ["--allowed-tools", ""]

        mcp_config = json.dumps(
            {"mcpServers": {grant.server_name: {"type": "http", "url": grant.server_url}}}
        )
        allowed_tools = [f"mcp__{grant.server_name}__{tool}" for tool in grant.allowed_tools]
        disallowed_tools = [
            f"mcp__{grant.server_name}__{tool}" for tool in sorted(JUDGE_DENY_MCP_TOOLS)
        ]
        return [
            "--mcp-config",
            mcp_config,
            "--strict-mcp-config",
            "--allowed-tools",
            *allowed_tools,
            "--disallowed-tools",
            *disallowed_tools,
        ]

    async def _call_claude(self, prompt: str) -> str:
        """Run ``claude -p --model <model>`` plus ``_claude_mcp_argv()`` with ``prompt`` on stdin
        and return stdout, unparsed.

        A non-zero exit raises ``JudgeUnavailable`` with the failure diagnostic. argv carries no
        prompt; ``--allowed-tools`` is variadic and would swallow a trailing positional prompt.
        """
        argv = [self._binary, "-p", "--model", self._model, *self._claude_mcp_argv()]
        stdout, stderr, returncode = await self._run(argv, prompt)
        if returncode != 0:
            raise JudgeUnavailable(
                f"claude CLI exited {returncode}: "
                f"{_failure_diagnostic(stdout, stderr, limit=self._STDERR_DIAGNOSTIC_LIMIT)}"
            )
        return stdout.decode(errors="replace")

    def _codex_mcp_argv(self) -> list[str]:
        """Render this seat's MCP grant into codex's flags.

        No grant: ``[]``.

        A grant: ``--ignore-user-config`` and three ``-c`` overrides, ``mcp_servers.<name>.url``,
        ``.enabled_tools`` (the grant) and ``.disabled_tools`` (``JUDGE_DENY_MCP_TOOLS``, sorted).
        The URL and tool names go through ``json.dumps``, which is valid TOML string syntax, so no
        value can break out of its string.

        NOTE: security-relevant. Without ``--ignore-user-config``, the ``-c`` overrides add this
        server to the servers already declared in ``$CODEX_HOME/config.toml`` instead of replacing
        them, which would expose ambient servers to the judge. Observed: when a tool is in both
        lists, ``disabled_tools`` wins, and codex omits an ungranted tool from the model's function
        surface.
        """
        grant = self._grant
        if grant is None:
            return []

        tools_toml = ", ".join(json.dumps(tool) for tool in grant.allowed_tools)
        disabled_toml = ", ".join(json.dumps(tool) for tool in sorted(JUDGE_DENY_MCP_TOOLS))
        return [
            "--ignore-user-config",
            "-c",
            f"mcp_servers.{grant.server_name}.url={json.dumps(grant.server_url)}",
            "-c",
            f"mcp_servers.{grant.server_name}.enabled_tools=[{tools_toml}]",
            "-c",
            f"mcp_servers.{grant.server_name}.disabled_tools=[{disabled_toml}]",
        ]

    async def _call_codex(self, prompt: str) -> str:
        """Run ``codex exec --skip-git-repo-check`` plus ``_codex_mcp_argv()``, then
        ``--output-last-message <file> -m <model>``, with ``prompt`` on stdin. Return the file's
        text.

        Only the ``--output-last-message`` file is read; stdout is a session transcript and is never
        parsed. A non-zero exit, or exit 0 without the file, raises ``JudgeUnavailable``.
        ``--skip-git-repo-check`` is needed because codex refuses to run outside a trusted git
        directory without it.
        """
        with tempfile.TemporaryDirectory() as td:
            outfile = Path(td) / "last-message.txt"
            argv = [
                self._binary,
                "exec",
                "--skip-git-repo-check",
                *self._codex_mcp_argv(),
                "--output-last-message",
                str(outfile),
                "-m",
                self._model,
            ]
            stdout, stderr, returncode = await self._run(argv, prompt)
            if returncode != 0:
                raise JudgeUnavailable(
                    f"codex CLI exited {returncode}: "
                    f"{_failure_diagnostic(stdout, stderr, limit=self._STDERR_DIAGNOSTIC_LIMIT)}"
                )
            if not outfile.exists():
                raise JudgeUnavailable(
                    "codex CLI exited 0 but wrote no --output-last-message file "
                    f"(stdout was {len(stdout)} bytes of transcript, deliberately not scraped)"
                )
            return outfile.read_text(errors="replace")

    async def _run(self, argv: list[str], prompt: str) -> tuple[bytes, bytes, int]:
        """Spawn ``argv``, feed ``prompt`` on stdin and return ``(stdout, stderr, returncode)``.

        A spawn ``OSError`` or a timeout raises ``JudgeUnavailable``; on timeout the process is
        killed and awaited first. The return code is ``-1`` if the process reports none.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise JudgeUnavailable(f"{self._cli} CLI unavailable: {exc}") from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode()), timeout=self._timeout
            )
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise JudgeUnavailable(f"{self._cli} CLI timed out after {self._timeout}s") from exc
        returncode = proc.returncode if proc.returncode is not None else -1
        return stdout, stderr, returncode


# ---------------------------------------------------------------------------
# Adapter 2: DeepSeek via litellm — the key-based backstop
# ---------------------------------------------------------------------------


class LitellmJudgeAdapter:
    """DeepSeek through litellm, the key-based backstop behind whichever primary seat is
    unavailable.

    Reads ``DEEPSEEK_API_KEY``. Timeout and retries are delegated to litellm (``timeout``,
    ``num_retries``); no retry loop is built here.
    """

    _DEFAULT_TIMEOUT = 60.0
    _DEFAULT_NUM_RETRIES = 2

    def __init__(
        self,
        *,
        model: str,
        timeout: float = _DEFAULT_TIMEOUT,
        num_retries: int = _DEFAULT_NUM_RETRIES,
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._num_retries = num_retries

    @property
    def identity(self) -> str:
        return f"litellm:{self._model}"

    def instruction_fingerprint(self) -> str:
        """``_FINGERPRINT_NOT_APPLICABLE``: this seat runs no subprocess and reads no instruction
        file. This differs from ``_FINGERPRINT_ABSENT``, which means a file was looked for and not
        found.
        """
        return _FINGERPRINT_NOT_APPLICABLE

    async def call(self, prompt: str) -> str:
        """Raise ``JudgeUnavailable`` on any litellm error, a streaming response, or an empty
        completion.
        """
        try:
            response = await litellm.acompletion(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                timeout=self._timeout,
                num_retries=self._num_retries,
            )
        except Exception as exc:
            _log.warning("Judge unavailable (%s): %s", self._model, exc)
            raise JudgeUnavailable(str(exc)) from exc

        if not isinstance(response, litellm.ModelResponse):
            raise JudgeUnavailable(f"litellm returned a streaming response for model={self._model}")
        content = response.choices[0].message.content
        if not content:
            raise JudgeUnavailable(f"litellm returned no content for model={self._model}")
        return str(content)


# ---------------------------------------------------------------------------
# Configuration -> adapter (mechanical; the ordering itself lives in the rubric artifact)
# ---------------------------------------------------------------------------


def build_judge_adapter(seat: JudgeSeatConfig) -> JudgeAdapter:
    """Instantiate the adapter a rubric seat config names.

    A CLI seat's MCP grant is validated with ``validate_mcp_grant`` first, because this function
    turns a grant into a live CLI permission and a seat built outside ``JudgesConfig`` has not been
    through that check. It is the same function ``JudgesConfig`` uses.
    """
    match seat:
        case CliSeat():
            grant = seat.mcp_grant
            if grant is not None:
                validate_mcp_grant(grant, seat_label="<build_judge_adapter>")

            return SubprocessJudgeAdapter(
                cli=seat.cli,
                binary=seat.binary,
                model=seat.model,
                timeout=seat.timeout,
                grant=grant,
            )
        case LitellmSeat():
            return LitellmJudgeAdapter(model=seat.model)


# ---------------------------------------------------------------------------
# The total outcome: judge_batch returns one of these on every call.
# ---------------------------------------------------------------------------

#: The closed set of statuses ``judge_batch`` reports. The axis modules fold it with
#: ``assert_never``, so adding a member without handling it there fails mypy at each call site.
JudgeOutcomeStatus = Literal[
    "success", "judge_unavailable", "invalid_output", "unexpected", "prompt_too_large"
]


class JudgeCallRecord(BaseModel):
    """One ``judge_batch`` call without ``verdicts``: the per-call telemetry unit each axis collects
    and ``engine/report.py`` nests in the pass report.

    Every field mirrors ``JudgeOutcome``. ``batch_index`` is the caller's loop counter. Calls are
    recorded for every status, since a seat degrading toward its timeout shows only as a trend
    across successful calls. ``instruction_fingerprint`` defaults to ``_FINGERPRINT_UNRECORDED`` so
    stored reports from before the field existed still parse; live calls always set a real value.
    """

    model_config = {"frozen": True}

    batch_index: int
    judge: str
    rubric_version: str
    instruction_fingerprint: str = _FINGERPRINT_UNRECORDED
    duration_seconds: float
    status: JudgeOutcomeStatus
    error: str | None
    prompt_tokens: int
    completion_tokens: int | None
    cost: float | None
    prompt_chars: int = 0


class JudgeOutcome[V: IndexedVerdictBase](BaseModel):
    """The total outcome of one ``judge_batch`` call.

    ``judge``, ``rubric_version``, ``instruction_fingerprint``, ``duration_seconds``, ``status``,
    ``error`` and ``prompt_tokens`` are present on every status. ``duration_seconds`` is 0.0 for
    ``prompt_too_large``; otherwise it covers the adapter call and excludes parsing. ``verdicts`` is
    non-empty only on ``success``. ``completion_tokens`` and ``cost`` are never set by
    ``judge_batch``.
    """

    model_config = {"frozen": True}

    judge: str
    rubric_version: str
    instruction_fingerprint: str
    duration_seconds: float
    status: JudgeOutcomeStatus
    error: str | None
    prompt_tokens: int
    completion_tokens: int | None = None
    cost: float | None = None
    verdicts: list[V] = Field(default_factory=list)
    prompt_chars: int = 0

    def as_call_record(self, *, batch_index: int) -> JudgeCallRecord:
        """Project this outcome to a ``JudgeCallRecord``: every field except ``verdicts``.
        ``batch_index`` is the caller's own counter.
        """
        return JudgeCallRecord(
            batch_index=batch_index,
            judge=self.judge,
            rubric_version=self.rubric_version,
            instruction_fingerprint=self.instruction_fingerprint,
            duration_seconds=self.duration_seconds,
            status=self.status,
            error=self.error,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cost=self.cost,
            prompt_chars=self.prompt_chars,
        )


async def judge_batch[V: IndexedVerdictBase](
    adapter: JudgeAdapter,
    *,
    prompt: str,
    item_model: type[V],
    batch_size: int,
    rubric_version: str,
    token_counter: TokenCounter,
) -> JudgeOutcome[V]:
    """Call ``adapter`` with ``prompt``, parse the response against the index protocol and return
    the total outcome.

    The adapter and the parser raise; this seam turns anything derived from ``Exception`` into a
    status: ``JudgeUnavailable`` becomes ``judge_unavailable``, ``JudgeInvalidOutput`` becomes
    ``invalid_output``, and any other exception becomes ``unexpected`` with its text in ``error``.
    ``asyncio.CancelledError`` and a failure in ``token_counter.count`` propagate.

    Order: count tokens, collect the instruction fingerprint (``"absent"`` and a warning if that
    fails), refuse a prompt over ``PROMPT_CHAR_CEILING`` with ``prompt_too_large`` without calling
    the adapter, call, parse. ``token_counter`` is injected so a pass shares one counter across
    batches.
    """
    prompt_tokens = token_counter.count(prompt)

    # Collected before the call so every status carries it; a failure here must not fail the call.
    try:
        instruction_fingerprint = adapter.instruction_fingerprint()
    except Exception as exc:  # broad on purpose: provenance collection is non-fatal
        _log.warning(
            "instruction fingerprint collection failed for %s: %s", adapter.identity, exc
        )
        instruction_fingerprint = _FINGERPRINT_ABSENT

    def outcome(
        *,
        status: JudgeOutcomeStatus,
        duration_seconds: float,
        error: str | None,
        verdicts: list[V] | None = None,
    ) -> JudgeOutcome[V]:
        return JudgeOutcome(
            judge=adapter.identity,
            rubric_version=rubric_version,
            instruction_fingerprint=instruction_fingerprint,
            duration_seconds=duration_seconds,
            status=status,
            error=error,
            prompt_tokens=prompt_tokens,
            prompt_chars=len(prompt),
            verdicts=verdicts or [],
        )

    if len(prompt) > PROMPT_CHAR_CEILING:
        return outcome(
            status="prompt_too_large",
            duration_seconds=0.0,
            error=(
                f"prompt is {len(prompt)} chars, over the {PROMPT_CHAR_CEILING}-char ceiling; "
                "the judge was not called"
            ),
        )

    start = time.monotonic()
    try:
        raw = await adapter.call(prompt)
    except JudgeUnavailable as exc:
        return outcome(
            status="judge_unavailable",
            duration_seconds=time.monotonic() - start,
            error=str(exc),
        )
    except Exception as exc:  # broad on purpose: any adapter or parser failure becomes a status
        return outcome(
            status="unexpected",
            duration_seconds=time.monotonic() - start,
            error=str(exc),
        )

    # Duration covers the adapter call only; parsing is excluded.
    duration_seconds = time.monotonic() - start

    try:
        verdicts = parse_batch_verdicts(raw, item_model=item_model, batch_size=batch_size)
    except JudgeInvalidOutput as exc:
        return outcome(status="invalid_output", duration_seconds=duration_seconds, error=str(exc))
    except Exception as exc:  # broad on purpose: any adapter or parser failure becomes a status
        return outcome(status="unexpected", duration_seconds=duration_seconds, error=str(exc))

    return outcome(
        status="success",
        duration_seconds=duration_seconds,
        error=None,
        verdicts=verdicts,
    )
