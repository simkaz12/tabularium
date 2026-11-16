"""Configuration: the frozen ``EngineConfig`` model, the store and transport settings, and the
``TABULARIUM_*`` environment parsing that builds them.

``EngineConfig`` has a default for every field; callers override by constructor keyword. It
rejects unknown keys and is immutable, and each field's ``description`` states its contract.
Its validators reject an ``assessor_model`` that is not ``jev-<major>.<minor>.<patch>``, a
``cleaner_nightly_at`` that is not 24-hour ``HH:MM``, ``archive_threshold >= revival_threshold``,
and ``cleaner_prepass_skip_keep=True`` with no ``cleaner_prepass_skip_cutoff``.

Environment parsing. The ``*_from_env`` functions read ``os.environ`` when called and raise
``ValueError`` for a value they cannot parse; the ``env_*`` helpers name the variable in the
message. ``config_from_env`` returns a ``StoreConfig``:

    TABULARIUM_ADAPTER      "lancedb" (default) or "memory"
    TABULARIUM_STORE_PATH   store directory; required for "lancedb"
    TABULARIUM_MODEL        litellm embedding model (default ollama/nomic-embed-text)
    TABULARIUM_DIM          embedding dimensions (default 768)
    TABULARIUM_API_BASE     embedder API base (default http://localhost:11434)

``transport_config_from_env`` returns a ``TransportConfig``:

    TABULARIUM_MCP_TRANSPORT  "stdio" (default) or "http"; any other value raises
    TABULARIUM_MCP_HOST       bind host for "http" (default 127.0.0.1)
    TABULARIUM_MCP_PORT       bind port for "http" (default 8787)

``engine_config_from_env`` maps these onto ``EngineConfig``. Booleans accept ``true``, ``1`` or
``yes`` and ``false``, ``0`` or ``no``, case-insensitively, and default to false:

    TABULARIUM_ENABLE_CLEANER, TABULARIUM_AUTO_EXIT_RESTATEMENT,
    TABULARIUM_AUTO_EXIT_REPLACEMENT, TABULARIUM_AUTO_EXIT_FOLLOW_UP,
    TABULARIUM_AUTO_EXIT_DISTINCT, TABULARIUM_CLEANER_PREPASS_SKIP_KEEP,
    TABULARIUM_CLEANER_SPLIT_SHELVE, TABULARIUM_READ_ONLY
    TABULARIUM_ASSESSOR_MODEL                 default jev-1.13.0
    TABULARIUM_ASSESSOR_SAVE_TIMEOUT_S        default 2.0
    TABULARIUM_ASSESSOR_CLEANER_TIMEOUT_S     default 10.0
    TABULARIUM_CLEANER_PREPASS_SKIP_CUTOFF    float, default unset
    TABULARIUM_CLEANER_NIGHTLY_AT             24-hour HH:MM, default 03:00

``TABULARIUM_READ_ONLY`` freezes maintenance writes; see ``EngineConfig.read_only``.

``build_assessor_from_env`` reads ``TYPESAFE_API_KEY``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

from tabularium.domain import StoreBackend

if TYPE_CHECKING:
    from tabularium.engine.assessor import Assessor

_ASSESSOR_MODEL_PATTERN = re.compile(r"^jev-\d+\.\d+\.\d+$")
_HHMM_PATTERN = re.compile(r"([01]\d|2[0-3]):[0-5]\d")

DEFAULT_STORE_ADAPTER: StoreBackend = "lancedb"
DEFAULT_EMBEDDER_MODEL = "ollama/nomic-embed-text"
DEFAULT_EMBEDDER_DIM = 768
DEFAULT_EMBEDDER_API_BASE = "http://localhost:11434"

TransportName = Literal["stdio", "http"]


class EngineConfig(BaseModel):
    """Frozen engine configuration; see the module docstring for validation."""

    model_config = {"frozen": True, "extra": "forbid"}

    # -----------------------------------------------------------------------
    # Retrieval / RRF
    # -----------------------------------------------------------------------

    rrf_k: int = Field(default=60, ge=1, description="RRF constant k (SPEC §5)")
    candidate_k: int = Field(
        default=50,
        ge=1,
        description="Per-leg candidate fan-out floor (ADR 0018); `recall` sizes the "
        "actual per-leg fetch as max(limit, candidate_k) so a larger caller-stated "
        "limit is never starved by this seed.",
    )
    probe_k: int = Field(
        default=25,
        ge=1,
        description="The admission gate's serial probe width (P7 admission-order "
        "piece) — vector top-k, BM25 narrows inside it. The "
        "ONLY width the gate reads; `candidate_k` keeps sizing recall.",
    )
    beta: float = Field(
        default=0.25,
        ge=0.0,
        description=(
            "Importance boost weight: final = RRF · (1 + β · imp_norm). "
            "Eval-swept; 0.25 is the M0 seed."
        ),
    )
    min_active_results: int = Field(
        default=3,
        ge=0,
        description=(
            "Fallback floor: when the primary lifecycle pass returns fewer than "
            "this many results, widen to archived."
        ),
    )

    # -----------------------------------------------------------------------
    # Dedup gate thresholds
    # -----------------------------------------------------------------------

    gray_floor: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine floor for the serial probe: cosine ≥ gray_floor → survivor, "
            "judged by the Assessor when one is configured, else bounces as today. "
            "Below the floor the save writes through. Applies to Fact arm."
        ),
    )
    decision_gray_floor: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Per-arm gray_floor for the Decision arm (SPEC §4.3 / §12). "
            "Lower than gray_floor: Decisions are recall-favoured on the bounce "
            "because a Decision false-pass corrupts 'what's in force'. "
            "Eval-tunable seed; not a claim about the optimal operating value."
        ),
    )
    # -----------------------------------------------------------------------
    # Importance composite weights
    # -----------------------------------------------------------------------

    w_recall: float = Field(default=0.5, ge=0.0, description="Weight on ln(1 + recall_count)")
    w_recency: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight on exp(-Δdays / half_life_days)",
    )
    w_feedback: float = Field(
        default=1.5,
        ge=0.0,
        description="Weight on feedback_score ∈ [0, 1]",
    )
    half_life_days: float = Field(
        default=30.0,
        gt=0.0,
        description="Recency decay half-life in days",
    )

    # -----------------------------------------------------------------------
    # Embedder config
    # -----------------------------------------------------------------------

    embedder_dim: int = Field(
        default=768,
        ge=1,
        description="Vector dimensionality expected from the embedder",
    )
    embedder_api_base: str = Field(
        default="http://localhost:11434",
        description="API base URL for the embedder (Ollama default)",
    )

    # -----------------------------------------------------------------------
    # Lifecycle thresholds and hot-tier caps
    # -----------------------------------------------------------------------

    archive_threshold: float = Field(
        default=0.10,
        ge=0.0,
        description=(
            "[M3] Importance threshold below which a Fact decays to archived. "
            "Calibrated; seed is conservative."
        ),
    )
    revival_threshold: float = Field(
        default=0.15,
        ge=0.0,
        description=(
            "[M3] Importance threshold for recall-fallback revival (a notch above "
            "archive_threshold). Drive no M0 code paths."
        ),
    )
    hot_cap_global: int = Field(
        default=8,
        ge=1,
        description=(
            "Per-(scope, role) hot-tier cap for the `global` scope (SPEC §8.5; "
            "curation block, owner decision 2026-08-03). Counted via "
            "engine.context.routes_to -- the SAME predicate the push-tier "
            "delivery filter uses, so the cap and the delivered payload cannot "
            "drift apart. `global` gets the LARGER cap: every session spans "
            "[scope, global], making it the scarcest real estate in the system. "
            "GOVERNING INVARIANT (tied to hot_cap_scope below): "
            "hot_cap_global + hot_cap_scope must be <= what the SMALLEST "
            "consumer budget delivers. A cap above budget capacity is not a "
            "cap, it is a queue -- the cut still happens, it just moves from an "
            "explicit ToolError naming occupants to a silent rank-based "
            "truncation (the 'which 18 of 484 appear is close to arbitrary' "
            "failure that killed the Decision tier). The consumer-budget knobs "
            "are NOT in this repo: a session-start hook can request a larger "
            "budget than a per-subagent hook, and the smaller one governs. The "
            "sub-agent budget was raised specifically to keep this invariant "
            "true at 8+5=13 atoms ~= 1300 tokens, measured at ~100 "
            "tokens/atom. Raising either cap here without checking the "
            "consumer hooks is how this invariant breaks silently."
        ),
    )
    hot_cap_scope: int = Field(
        default=5,
        ge=1,
        description=(
            "Per-(scope, role) hot-tier cap for every scope OTHER than "
            "`global` (SPEC §8.5; curation block). See "
            "hot_cap_global's description for the full invariant this pairs "
            "with -- the two numbers are never tuned independently."
        ),
    )
    escalation_threshold: int = Field(
        default=3,
        ge=1,
        description=(
            "Owner-asserted recurrence count (r_owner) at which the §8.4 rung-2 "
            "escalation suggestion fires (codify or foreground — both named)."
        ),
    )

    # -----------------------------------------------------------------------
    # Maintenance
    # -----------------------------------------------------------------------

    maintenance_threshold: int = Field(
        default=20,
        ge=1,
        description=(
            "Under-target-fragment count (adapter.maintenance_debt().num_small_fragments) "
            "at which the save path synchronously invokes adapter.upkeep(). 20 is "
            "vendor-aligned — LanceDB's own compaction cadence guidance — not a tuned "
            "value. The debt signal floors at ~1 for a healthy store, never 0: this is "
            "an absolute count, never a 'greater than zero' test."
        ),
    )

    # -----------------------------------------------------------------------
    # Read-only archive mode
    # -----------------------------------------------------------------------

    read_only: bool = Field(
        default=False,
        description=(
            "When True the engine freezes its maintenance writes: `initialize()` writes nothing "
            "(no registry seeding, no startup passes, no backfill pass, no nightly Cleaner "
            "scheduler, whatever `enable_digest_cleaner` says), `recall` never revives an "
            "archived hit, and the `tabularium clean` CLI refuses a pass or dry run. "
            "`read_events` rows still append, as telemetry. The MCP server registers no write "
            "tool in either mode. `server.py`'s `engine_config_from_env()` reads "
            "`TABULARIUM_READ_ONLY` and wires it to this field."
        ),
    )

    # -----------------------------------------------------------------------
    # Cleaner trigger
    # -----------------------------------------------------------------------

    enable_digest_cleaner: bool = Field(
        default=False,
        description=(
            "When True, the server starts the nightly Cleaner scheduler at `initialize()`: at "
            "`cleaner_nightly_at` it runs `Tabularium.clean(scope)` for every registered scope, "
            "one at a time, through the per-scope single-flight. `digest()` starts no pass. "
            "Defaults False so a fresh process (tests) never starts an unattended pass that "
            "WRITES (shelving Facts, drawing supersession edges). Name kept from when `digest()` "
            "was the trigger. `server.py`'s `engine_config_from_env()` reads "
            "`TABULARIUM_ENABLE_CLEANER` and wires it to this field; the deployed service "
            "has it set to `true`."
        ),
    )

    auto_exit_restatement: bool = Field(
        default=False,
        description=(
            "When True, an Assessor verdict of `restatement` against a legal same-scope "
            "or downward candidate reinforces automatically instead of bouncing for a "
            "human to adjudicate. Defaults False: this switch is gated on the verdict "
            "log's own override rate — read it at "
            "`GET /api/verdicts/tuning` before flipping it; flipping it without that "
            "evidence is a plain owner act, not a graduation."
        ),
    )
    auto_exit_replacement: bool = Field(
        default=False,
        description=(
            "When True, an Assessor verdict of `replacement` against a legal same-scope "
            "candidate supersedes it automatically (or invalidates it, when the "
            "candidate is an in-force Decision and the reason class is `correction`/"
            "`changed`) instead of bouncing. Defaults False: this switch is gated on the "
            "verdict log's own override rate — read it at "
            "`GET /api/verdicts/tuning` before flipping it; flipping it without that "
            "evidence is a plain owner act, not a graduation."
        ),
    )
    auto_exit_follow_up: bool = Field(
        default=False,
        description=(
            "When True, an Assessor verdict of `follow_up` against a legal candidate "
            "writes the new row automatically, registering no pair, instead of "
            "bouncing. Defaults False: this switch is gated on the verdict log's own "
            "override rate — read it at `GET /api/verdicts/tuning` "
            "before flipping it; flipping it without that evidence is a plain owner "
            "act, not a graduation."
        ),
    )
    auto_exit_distinct: bool = Field(
        default=False,
        description=(
            "When True, an Assessor verdict of `distinct` against a legal candidate "
            "writes the new row automatically, registering no pair, instead of "
            "bouncing. Defaults False: this switch is gated on the verdict log's own "
            "override rate — read it at `GET /api/verdicts/tuning` "
            "before flipping it; flipping it without that evidence is a plain owner "
            "act, not a graduation."
        ),
    )

    # -----------------------------------------------------------------------
    # Assessor
    # -----------------------------------------------------------------------

    assessor_model: str = Field(
        default="jev-1.13.0",
        description=(
            "The versioned Jev model id the adapter requests, e.g. `jev-1.13.0`. Never an "
            "alias (`jev-latest`, `jev-preview`) — the validator below refuses one, since an "
            "alias moving under a pinned threshold would be a silent regression. Env override "
            "`TABULARIUM_ASSESSOR_MODEL`, wired in `server.py`'s `engine_config_from_env`."
        ),
    )
    assessor_save_timeout_s: float = Field(
        default=2.0,
        gt=0.0,
        description=(
            "Per-caller timeout for a `relation` call issued from the save path (~2 s, ticket "
            "02's own bound). The Cleaner's own 10 s timeout is A3's config field, not this "
            "one. Env override `TABULARIUM_ASSESSOR_SAVE_TIMEOUT_S`."
        ),
    )

    # -----------------------------------------------------------------------
    # Cleaner Assessor integration
    # -----------------------------------------------------------------------

    cleaner_prepass_skip_keep: bool = Field(
        default=False,
        description=(
            "When True, a relevance pre-pass pick of `keep-clear` above "
            "`cleaner_prepass_skip_cutoff` skips the panel entirely for that record. Defaults "
            "False: the pre-pass ships in shadow first (D1) — read the verdict log's own "
            "override rate at `GET /api/verdicts/tuning` before flipping it; flipping it without "
            "that evidence is a plain owner act, not a graduation. Env override "
            "`TABULARIUM_CLEANER_PREPASS_SKIP_KEEP`."
        ),
    )
    cleaner_prepass_skip_cutoff: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "Confidence cutoff a pre-pass `keep-clear` pick must clear to skip the panel, when "
            "`cleaner_prepass_skip_keep` is True. No default value (D1) — a model validator "
            "below refuses `cleaner_prepass_skip_keep=True` with no cutoff set. Env override "
            "`TABULARIUM_CLEANER_PREPASS_SKIP_CUTOFF`."
        ),
    )
    cleaner_split_shelve: bool = Field(
        default=False,
        description=(
            "When True, an Assessor shelve vote on a relevance or revisit split shelves the "
            "record instead of only annotating and sorting the contested queue entry (D2). "
            "Defaults False: read the verdict log's own override rate at "
            "`GET /api/verdicts/tuning` before flipping it; flipping it without that evidence is "
            "a plain owner act, not a graduation. Env override `TABULARIUM_CLEANER_SPLIT_SHELVE`."
        ),
    )
    assessor_cleaner_timeout_s: float = Field(
        default=10.0,
        gt=0.0,
        description=(
            "Per-caller timeout for a `relevance`/`contradiction` call issued from the Cleaner "
            "(the save path's own 2 s timeout is `assessor_save_timeout_s`, not this one). Env "
            "override `TABULARIUM_ASSESSOR_CLEANER_TIMEOUT_S`."
        ),
    )

    cleaner_nightly_at: str = Field(
        default="03:00",
        description=(
            "Local wall-clock `HH:MM` at which the in-server scheduler runs one Cleaner pass per "
            "registered scope, `global` included, one at a time. Only runs when "
            "`enable_digest_cleaner` is True. Env override `TABULARIUM_CLEANER_NIGHTLY_AT`."
        ),
    )

    # -----------------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------------

    @field_validator("assessor_model")
    @classmethod
    def _check_assessor_model_is_versioned(cls, value: str) -> str:
        if not _ASSESSOR_MODEL_PATTERN.match(value):
            raise ValueError(
                f"assessor_model={value!r} must be a versioned id matching "
                f"{_ASSESSOR_MODEL_PATTERN.pattern!r}, never an alias"
            )
        return value

    @field_validator("cleaner_nightly_at")
    @classmethod
    def _check_cleaner_nightly_at(cls, value: str) -> str:
        if not _HHMM_PATTERN.fullmatch(value):
            raise ValueError(f"cleaner_nightly_at={value!r} must be a 24-hour HH:MM time")

        return value

    @model_validator(mode="after")
    def _check_threshold_ordering(self) -> EngineConfig:
        # NOTE: no check relates decision_gray_floor to gray_floor.
        if self.archive_threshold >= self.revival_threshold:
            raise ValueError(
                f"archive_threshold ({self.archive_threshold}) must be < "
                f"revival_threshold ({self.revival_threshold})"
            )
        return self

    @model_validator(mode="after")
    def _check_cleaner_prepass_skip_cutoff(self) -> EngineConfig:
        if self.cleaner_prepass_skip_keep and self.cleaner_prepass_skip_cutoff is None:
            raise ValueError(
                "cleaner_prepass_skip_keep=True requires cleaner_prepass_skip_cutoff to be set"
            )
        return self


def parse_store_backend(raw: str) -> StoreBackend:
    """Return ``raw`` as a ``StoreBackend``; raises ``ValueError("Unknown store_adapter: ...")``."""
    match raw:
        case "memory" | "lancedb":
            return raw
        case _:
            raise ValueError(f"Unknown store_adapter: {raw!r}")


def parse_transport(raw: str) -> TransportName:
    """Return ``raw`` as a ``TransportName``; raises ``ValueError`` for any other value."""
    match raw:
        case "stdio" | "http":
            return raw
        case _:
            raise ValueError(
                f"Invalid TABULARIUM_MCP_TRANSPORT={raw!r}; must be one of "
                f"{get_args(TransportName)!r}"
            )


@dataclass(frozen=True, slots=True)
class StoreConfig:
    """Which store to open and which embedder settings stamp it.

    Raises ``ValueError("Unknown store_adapter: ...")`` on construction when ``adapter`` is not
    a ``StoreBackend``. ``path`` is required by the ``lancedb`` adapter at open time, not here.
    """

    adapter: StoreBackend
    path: str | None
    model: str
    dim: int
    api_base: str

    def __post_init__(self) -> None:
        parse_store_backend(self.adapter)


@dataclass(frozen=True, slots=True)
class TransportConfig:
    """How ``tabularium-mcp`` serves: ``stdio``, or ``http`` bound to ``host`` and ``port``."""

    transport: TransportName
    host: str
    port: int


_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes"})
_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no"})


def env_str(name: str, default: str) -> str:
    """Return the variable's raw value, or ``default`` when it is unset."""
    return os.environ.get(name, default)


def env_bool(name: str) -> bool:
    """Parse a boolean variable; unset means false.

    Raises ``ValueError`` for a value outside the accepted spellings.
    """
    raw = os.environ.get(name, "false")
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid {name}={raw!r}; must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)!r}"
    )


def env_float(name: str, default: float) -> float:
    """Parse a float variable, or return ``default`` when it is unset.

    Raises ``ValueError`` for a value that is not a float.
    """
    value = env_optional_float(name)
    return default if value is None else value


def env_optional_float(name: str) -> float | None:
    """Parse a float variable, or return ``None`` when it is unset.

    Raises ``ValueError`` for a value that is not a float.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw!r}; must be a float") from exc


def env_int(name: str, default: int) -> int:
    """Parse an integer variable, or return ``default`` when it is unset.

    Raises ``ValueError`` for a value that is not an integer.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw!r}; must be an integer") from exc


def config_from_env(
    *,
    path_override: str | None = None,
    adapter_override: str | None = None,
) -> StoreConfig:
    """Read the store and embedder settings; a non-empty override wins over its variable.

    Raises ``ValueError`` for a non-integer ``TABULARIUM_DIM`` or an unknown adapter.
    """
    adapter = adapter_override or env_str("TABULARIUM_ADAPTER", DEFAULT_STORE_ADAPTER)
    path = path_override or os.environ.get("TABULARIUM_STORE_PATH")
    return StoreConfig(
        adapter=parse_store_backend(adapter),
        path=path,
        model=env_str("TABULARIUM_MODEL", DEFAULT_EMBEDDER_MODEL),
        dim=env_int("TABULARIUM_DIM", DEFAULT_EMBEDDER_DIM),
        api_base=env_str("TABULARIUM_API_BASE", DEFAULT_EMBEDDER_API_BASE),
    )


def transport_config_from_env() -> TransportConfig:
    """Read the transport settings.

    Raises ``ValueError`` for a ``TABULARIUM_MCP_TRANSPORT`` other than ``stdio`` or ``http``
    and for a non-integer ``TABULARIUM_MCP_PORT``.
    """
    transport = parse_transport(env_str("TABULARIUM_MCP_TRANSPORT", "stdio"))

    return TransportConfig(
        transport=transport,
        host=env_str("TABULARIUM_MCP_HOST", "127.0.0.1"),
        port=env_int("TABULARIUM_MCP_PORT", 8787),
    )


def engine_config_from_env() -> EngineConfig:
    """Build an ``EngineConfig`` from the ``TABULARIUM_*`` variables in the module docstring.

    Other fields keep their defaults. Raises ``ValueError`` for a boolean outside the accepted
    spellings, a float or integer that does not parse, a malformed
    ``TABULARIUM_CLEANER_NIGHTLY_AT``, or a combination ``EngineConfig`` rejects.
    """
    enable_cleaner = env_bool("TABULARIUM_ENABLE_CLEANER")
    read_only = env_bool("TABULARIUM_READ_ONLY")
    assessor_model = env_str("TABULARIUM_ASSESSOR_MODEL", "jev-1.13.0")
    assessor_save_timeout_s = env_float("TABULARIUM_ASSESSOR_SAVE_TIMEOUT_S", 2.0)
    assessor_cleaner_timeout_s = env_float("TABULARIUM_ASSESSOR_CLEANER_TIMEOUT_S", 10.0)
    cleaner_prepass_skip_cutoff = env_optional_float("TABULARIUM_CLEANER_PREPASS_SKIP_CUTOFF")
    cleaner_nightly_at = env_str("TABULARIUM_CLEANER_NIGHTLY_AT", "03:00").strip()
    if not _HHMM_PATTERN.fullmatch(cleaner_nightly_at):
        raise ValueError(
            f"Invalid TABULARIUM_CLEANER_NIGHTLY_AT={cleaner_nightly_at!r}; must be HH:MM (24-hour)"
        )

    return EngineConfig(
        enable_digest_cleaner=enable_cleaner,
        read_only=read_only,
        auto_exit_restatement=env_bool("TABULARIUM_AUTO_EXIT_RESTATEMENT"),
        auto_exit_replacement=env_bool("TABULARIUM_AUTO_EXIT_REPLACEMENT"),
        auto_exit_follow_up=env_bool("TABULARIUM_AUTO_EXIT_FOLLOW_UP"),
        auto_exit_distinct=env_bool("TABULARIUM_AUTO_EXIT_DISTINCT"),
        assessor_model=assessor_model,
        assessor_save_timeout_s=assessor_save_timeout_s,
        cleaner_prepass_skip_keep=env_bool("TABULARIUM_CLEANER_PREPASS_SKIP_KEEP"),
        cleaner_prepass_skip_cutoff=cleaner_prepass_skip_cutoff,
        cleaner_split_shelve=env_bool("TABULARIUM_CLEANER_SPLIT_SHELVE"),
        assessor_cleaner_timeout_s=assessor_cleaner_timeout_s,
        cleaner_nightly_at=cleaner_nightly_at,
    )


def build_assessor_from_env(assessor_model: str) -> Assessor | None:
    """Return a ``JevAssessor`` when ``TYPESAFE_API_KEY`` is non-empty, else ``None``.

    ``server.main()`` and the ``tabularium clean`` CLI both call it, so a scheduled pass and a
    manual pass build the Assessor the same way. ``JevAssessor`` is imported on call because
    importing ``tabularium.engine.assessor`` runs ``tabularium.engine``, which imports this
    module.
    """
    from tabularium.engine.assessor import JevAssessor

    typesafe_api_key = os.environ.get("TYPESAFE_API_KEY", "")
    return JevAssessor(typesafe_api_key, assessor_model) if typesafe_api_key else None
