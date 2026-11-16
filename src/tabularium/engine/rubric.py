"""The Cleaner's rubric: versioned data that the engine loads.

The rubric is the grading standard that decides what the memory keeps. It ships as a static TOML
artifact (``rubric.toml``, next to this module) rather than as a Python literal or prompt text in a
judge call site, so an eval harness can sweep it.

Other modules reach the packaged artifact through ``load_rubric()``, and a caller-supplied file
through ``load_rubric_from_path()``.

``Rubric.version`` is content-addressed: it hashes the raw artifact bytes, so the artifact cannot
change without the version changing and there is no version field to forget to bump.
"""

from __future__ import annotations

import hashlib
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from tabularium.engine.judge_policy import McpGrant, validate_mcp_grant

_ARTIFACT_PATH = Path(__file__).with_name("rubric.toml")


class RubricParameters(BaseModel):
    """Sweep parameters carried as data beside the prose, not constants in code.

    No field has a Python default: a missing key is a ``pydantic.ValidationError`` at load time
    instead of a silent code-side fallback.

    ``strict=True`` and ``allow_inf_nan=False`` on every float field stop a caller-supplied rubric
    from passing ``k = true`` (``bool`` coerces to ``int``) or ``revisit_stable_multiplier = inf``,
    either of which would otherwise change the pool or batch shape with no error. ``extra="forbid"``
    makes an unrecognized key, such as a misspelled override, a ``RubricLoadError`` instead of being
    dropped while the run uses the unchanged value. The packaged artifact satisfies both.
    """

    model_config = {"frozen": True, "strict": True, "extra": "forbid"}

    k: int = Field(gt=0, description="Candidates retrieved per newcomer in the falsity pair sweep.")
    similarity_floor: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Minimum probe similarity for a falsity-sweep candidate pair.",
    )
    batch_cap: int = Field(gt=0, description="Maximum records per judge batch, either axis.")
    unclear_ceiling: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description=(
            "Fraction of a relevance batch that may be graded `unclear` before the batch is "
            "treated as a reading failure rather than a genuine mix of unjudgeable records."
        ),
    )
    pool_limit: int = Field(
        gt=0,
        description=(
            "Ceiling on one Cleaner pass's `list_born_since` call. The pass orchestration in "
            "`engine/clean.py` and each axis's own re-query must read the same value, or the "
            "watermark-advance snapshot and what the axes actually judged can diverge. Real batch "
            "size is set by the write rate since the last pass; this is a safety ceiling on the "
            "Protocol's mandatory `limit` argument, not a number to tune for judging quality."
        ),
    )
    report_render_cap: int = Field(
        gt=0,
        description=(
            "Ceiling on the number of owner-attention lines "
            "`engine.report.render_context_summary` puts in context()'s per-pass rendering, so a "
            "pass with hundreds of unclear, contradicted or failed items does not crowd grounding "
            "out of the session-start budget. The full report still lands in the `report` column; "
            "this cap only bounds the short rendering."
        ),
    )
    revisit_batch_cap: int = Field(
        gt=0, description="Maximum records per revisit-axis judge batch."
    )
    revisit_pass_cap: int = Field(
        gt=0,
        description=(
            "Ceiling on one revisit-axis pass's `list_due_for_review` call. It also sets the "
            "cold-start ramp: the pool it bounds is wider than `revisit_batch_cap`, so the drain "
            "keeps moving on a pass with many protected or no-evidence exclusions."
        ),
    )
    revisit_min_age_days: int = Field(
        gt=0,
        description=(
            "A record must be at least this many days old before its first revisit-axis look. The "
            "revisit interval is itself the throttle, and a younger record has not had time to "
            "accumulate evidence beyond its own birth digest."
        ),
    )
    revisit_interval_days: int = Field(
        gt=0, description="Base interval between revisit-axis looks at a record, in days."
    )
    revisit_stable_multiplier: float = Field(
        gt=0,
        allow_inf_nan=False,
        description=(
            "Multiplier applied to the interval per consecutive unanimous-keep review — a record "
            "repeatedly confirmed stable earns a longer gap."
        ),
    )
    revisit_contested_multiplier: float = Field(
        gt=0,
        allow_inf_nan=False,
        description=(
            "Multiplier applied to the interval when the last review was contested — a shorter "
            "gap, not a longer one."
        ),
    )
    revisit_max_interval_days: int = Field(
        gt=0, description="Ceiling on the compounded revisit interval, in days."
    )


class RelevanceRubric(BaseModel):
    """The relevance axis: does this record still deserve to be in the store."""

    model_config = {"frozen": True}

    scale: tuple[str, str, str, str] = Field(
        description="The four-point ordered scale, worst to best, with no neutral middle."
    )
    anchor: str = Field(
        description="The tail-pass anchor — provisional until measured against real passes."
    )
    prose: str = Field(description="The full grading standard shown to each judge.")


class RevisitRubric(BaseModel):
    """The revisit axis: a second, evidence-in-hand pass over old records.

    Its prose is rendered after ``relevance.prose``
    (``render_batch_prompt(rubric_prose=relevance.prose + revisit.prose, ...)``), so the shared
    grading standard stays in one place. This section only adds interpretation rules for the history
    block each record carries.
    """

    model_config = {"frozen": True}

    prose: str = Field(
        description="Interpretation rules for the history block, shown after relevance.prose."
    )


class FalsityRubric(BaseModel):
    """The falsity axis: given a newer record and an older one, replace or coexist."""

    model_config = {"frozen": True}

    verdicts: tuple[str, str] = Field(description="The two-point forced choice; no unclear option.")
    anchor: str = Field(description="The provisional anchor for the pair sweep's expected shape.")
    prose: str = Field(description="The full grading standard shown to each judge.")


class _SeatBase(BaseModel):
    """Fields every judge seat carries."""

    model_config = {"frozen": True, "extra": "forbid"}

    model: str = Field(
        min_length=1,
        description="Model passed explicitly to the adapter, never left to the provider's default.",
    )


class CliSeat(_SeatBase):
    """A seat filled by an installed CLI (``claude -p`` or ``codex exec``).

    The ``mcp_*`` fields are flat so the TOML shape stays flat. ``mcp_grant`` is the typed view
    of them.
    """

    adapter: Literal["cli"] = "cli"
    cli: Literal["claude", "codex"] = Field(description="Which installed CLI fills this seat.")
    binary: str = Field(description="CLI binary: an absolute path or a name resolved on PATH.")
    timeout: float = Field(
        default=1200.0,
        gt=0,
        description="Per-call subprocess timeout in seconds.",
    )
    mcp_server_url: str | None = Field(
        default=None,
        description=(
            "URL of a Tabularium MCP server this seat may call while judging, so the judge pulls "
            "evidence itself instead of receiving it in the prompt. ``None`` (the default, and "
            "every seat in the packaged artifact) means no MCP access at all. The three ``mcp_*`` "
            "fields carry intent only -- a server URL, a server name and a tuple of bare tool "
            "names -- and each CLI call in ``judge.py`` renders them into that CLI's own flags "
            "(claude: --mcp-config JSON, --strict-mcp-config, --allowed-tools "
            "mcp__<name>__<tool>; codex: -c mcp_servers.<name>.url=... and "
            "-c mcp_servers.<name>.enabled_tools=[...], on argv only)."
        ),
    )
    mcp_server_name: str | None = Field(
        default=None,
        description=(
            "Name this MCP server is registered under for the call: the tool-name prefix for "
            "claude (mcp__<name>__<tool>) and the config key for codex (mcp_servers.<name>). "
            "Required whenever mcp_server_url is set and must be None otherwise."
        ),
    )
    mcp_allowed_tools: tuple[str, ...] = Field(
        default=(),
        description=(
            "Bare tool names this seat may call on the configured MCP server, for example "
            "('decisions_query', 'decision', 'context'). Never 'recall': it is not "
            "side-effect-free, because it reinforces a hit and can flip archived to active on its "
            "archived-fallback branch, and the MCP layer has no dry_run. "
            "JudgesConfig rejects any seat whose allowlist contains it, or any tool outside "
            "``JUDGE_SAFE_MCP_TOOLS``. Must be non-empty whenever mcp_server_url is set, so a "
            "seat never opens an MCP connection with an empty grant, and empty otherwise."
        ),
    )

    @model_validator(mode="after")
    def _mcp_fields_are_consistent(self) -> CliSeat:
        """Structural consistency only: a name or allowlist needs a URL, and a URL needs a name.

        ``JudgesConfig`` runs the seat-labelled fail-closed and ``recall`` checks at rubric load.
        """
        if self.mcp_server_url is None:
            if self.mcp_server_name is not None or self.mcp_allowed_tools:
                raise ValueError(
                    "mcp_server_name/mcp_allowed_tools set without mcp_server_url -- an MCP "
                    "grant needs a server to point at"
                )
        elif self.mcp_server_name is None:
            raise ValueError(
                f"mcp_server_url is set ({self.mcp_server_url!r}) but mcp_server_name is "
                "missing -- a judge seat must never open an MCP connection without a name"
            )

        return self

    @property
    def mcp_grant(self) -> McpGrant | None:
        """The seat's MCP grant, or ``None`` when ``mcp_server_url`` is unset."""
        if self.mcp_server_url is None or self.mcp_server_name is None:
            return None

        return McpGrant(
            server_url=self.mcp_server_url,
            server_name=self.mcp_server_name,
            allowed_tools=self.mcp_allowed_tools,
        )


class LitellmSeat(_SeatBase):
    """A seat filled by a key-based litellm route. It carries no CLI and no MCP access."""

    adapter: Literal["litellm"] = "litellm"


JudgeSeatConfig = Annotated[CliSeat | LitellmSeat, Field(discriminator="adapter")]


class JudgesConfig(BaseModel):
    """The judge panel: two primary seats plus one shared key-based backstop.

    The backstop (DeepSeek via litellm) stands behind whichever primary seat is unavailable, not
    behind one seat, because a key-based route cannot lapse the way a CLI's cached login can.
    """

    model_config = {"frozen": True}

    seat_a: JudgeSeatConfig
    seat_b: JudgeSeatConfig
    backstop: JudgeSeatConfig

    @model_validator(mode="after")
    def _mcp_grants_are_safe(self) -> JudgesConfig:
        """Run ``validate_mcp_grant`` for every CLI seat whenever a full ``Rubric`` is loaded, the
        packaged one or a ``--rubric`` file. A seat built directly gets only its own structural
        checks.
        """
        for seat_label, seat in (
            ("seat_a", self.seat_a),
            ("seat_b", self.seat_b),
            ("backstop", self.backstop),
        ):
            if isinstance(seat, CliSeat) and (grant := seat.mcp_grant) is not None:
                validate_mcp_grant(grant, seat_label=seat_label)

        return self


class CaseSetRubric(BaseModel):
    """The admission case-set's ``ideal``-tier judge: a seat using the same ``JudgeSeatConfig``
    shape as the Cleaner panel, plus the prompt template that renders a case's
    ``existing``/``incoming`` pair.

    The judge sees ``existing`` and ``incoming`` only, never a case's ``why``, so
    ``prompt_template`` has exactly the placeholders ``{existing}`` and ``{incoming}``, filled by
    ``tests.support.ideal_judge.build_ideal_prompt``.
    """

    model_config = {"frozen": True}

    judge: JudgeSeatConfig
    prompt_template: str


class Rubric(BaseModel):
    """The Cleaner's rubric: relevance, falsity, sweep parameters, judges and a content-addressed
    version.

    ``version`` is stamped on every verdict next to the judge pair, so panel behaviour can be
    measured against a specific grading standard.

    ``caseset`` is optional. No production Cleaner path reads it, only the admission test harness,
    so a rubric without a ``[caseset]`` section still loads and runs a live pass.
    """

    model_config = {"frozen": True}

    version: str
    parameters: RubricParameters
    relevance: RelevanceRubric
    revisit: RevisitRubric
    falsity: FalsityRubric
    judges: JudgesConfig
    caseset: CaseSetRubric | None = None


def _compute_version(raw: bytes, *, schema: int) -> str:
    """Content-addressed version: any change to the artifact's bytes changes it.

    ``schema`` namespaces intentional format-breaking rewrites, a loader-shape change, from ordinary
    edits to the prose or parameters.
    """
    digest = hashlib.sha256(raw).hexdigest()[:12]
    return f"cleaner-rubric-v{schema}-{digest}"


class _RubricMeta(BaseModel):
    """The ``[meta]`` section: ``schema``, the tag ``_compute_version`` folds into the version
    string.

    It is a strict pydantic model so a malformed file raises the same ``ValidationError`` as any
    other field. A bare ``int()`` coercion would let ``schema = inf`` escape as an uncaught
    ``OverflowError`` and accept ``true`` or ``1.9``.
    """

    model_config = {"frozen": True}

    schema_: int = Field(
        alias="schema",
        strict=True,
        description="Namespaces intentional format-breaking rewrites (a loader-shape change) "
        "from ordinary content edits, inside `_compute_version`.",
    )


def _rubric_from_toml(data: dict[str, Any], raw: bytes) -> Rubric:
    """Build a ``Rubric`` from parsed TOML data and the raw bytes it came from.

    Shared by ``load_rubric()`` and ``load_rubric_from_path()``. ``version`` is always derived from
    ``raw``, so a custom file's version differs from the packaged one whenever the bytes differ.

    Raises what the constructors raise: ``KeyError`` or ``TypeError`` for a missing section or
    field, ``pydantic.ValidationError`` for a bad value. No translation happens here;
    ``load_rubric_from_path`` wraps these in ``RubricLoadError``, while ``load_rubric()`` leaves the
    packaged artifact's errors unwrapped.
    """
    meta = _RubricMeta(**data["meta"])
    version = _compute_version(raw, schema=meta.schema_)
    judges_data = data["judges"]
    caseset_data = data.get("caseset")
    caseset = (
        CaseSetRubric(
            judge=caseset_data["judge"],
            prompt_template=caseset_data["prompt_template"],
        )
        if caseset_data is not None
        else None
    )
    return Rubric(
        version=version,
        parameters=RubricParameters(**data["parameters"]),
        relevance=RelevanceRubric(**data["relevance"]),
        revisit=RevisitRubric(**data["revisit"]),
        falsity=FalsityRubric(**data["falsity"]),
        judges=JudgesConfig(
            seat_a=judges_data["seat_a"],
            seat_b=judges_data["seat_b"],
            backstop=judges_data["backstop"],
        ),
        caseset=caseset,
    )


@lru_cache(maxsize=1)
def load_rubric() -> Rubric:
    """Load, validate and return the packaged rubric. Cached, since the artifact is static for the
    process lifetime.

    ``load_rubric_from_path`` shares only the stateless ``_rubric_from_toml`` and
    ``_compute_version`` with this function. It never calls it and never touches this cache or
    ``_ARTIFACT_PATH``.
    """
    raw = _ARTIFACT_PATH.read_bytes()
    data: dict[str, Any] = tomllib.loads(raw.decode("utf-8"))
    return _rubric_from_toml(data, raw)


class RubricLoadError(Exception):
    """A rubric file given with ``--rubric PATH`` (dry run only, gated in ``cli.py``) could not be
    loaded: the path is missing or unreadable, the bytes are not valid TOML, or the TOML fails the
    schema. The message names the path and the problem.
    """


def load_rubric_from_path(path: Path) -> Rubric:
    """Load, validate and return a ``Rubric`` from an arbitrary TOML path, for the ``--rubric`` CLI
    seam. Dry-run-only use is enforced by the caller, not here.

    Not cached, and independent of ``load_rubric()``'s state. ``version`` is derived from this
    file's bytes by the same rule as the packaged one.

    Raises:
        RubricLoadError: ``path`` is missing, unreadable, not UTF-8 or not valid TOML, or the
            content fails the schema (a missing section or field, an out-of-range value).
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise RubricLoadError(f"rubric file not found: {path}") from exc
    except OSError as exc:
        raise RubricLoadError(f"rubric file at {path} could not be read: {exc}") from exc

    try:
        data: dict[str, Any] = tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise RubricLoadError(f"rubric file at {path} is not valid UTF-8: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RubricLoadError(f"rubric file at {path} is not valid TOML: {exc}") from exc

    try:
        return _rubric_from_toml(data, raw)
    except ValidationError as exc:
        # Caught BEFORE the broader `ValueError` clause below: `pydantic.ValidationError`
        # subclasses `ValueError`, so the order here is load-bearing -- reversed, every schema
        # failure would mislabel itself as a missing section/field instead.
        raise RubricLoadError(f"rubric file at {path} failed schema validation: {exc}") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise RubricLoadError(
            f"rubric file at {path} is missing a required section or field: {exc}"
        ) from exc
