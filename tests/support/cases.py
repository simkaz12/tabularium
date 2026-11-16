"""The admission case-set's YAML shape, one file per case, loaded into typed models — never a case
body written as inline
assertion code.

Each check in ``expect.mutation.acceptable``/``expect.mutation.ideal`` is parsed, by a small
regex grammar, from a fixed-vocabulary string into one of the typed ``Check`` variants below,
each carrying the mutation-axis failure class it maps to. ``pair
registered``/``invalidated_by``/etc. join the original six with the P8 assessor-automatic-exits
piece's own additions (``PairRegistered.expected``, ``InvalidationEquals``). A raw string that
matches no pattern, or a ``material`` label outside the six D7 labels, fails loading with a
message naming both the case file and the offending string.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from tabularium.engine.assessor import ReasonClass, Relation

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CASES_DIR = _REPO_ROOT / "tests" / "cases" / "admission"
_CASES_DIR_ENV = "TABULARIUM_ADMISSION_CASES"

# ---------------------------------------------------------------------------
# Material labels — six, finer than the Assessor's four relations.
# ---------------------------------------------------------------------------

MaterialLabel = Literal[
    "repetition", "elaboration", "addition", "correction", "changed state", "unrelated"
]

MATERIAL_TO_RELATION: dict[MaterialLabel, Relation] = {
    "repetition": "restatement",
    "elaboration": "replacement",
    "addition": "follow_up",
    "correction": "replacement",
    "changed state": "replacement",
    "unrelated": "distinct",
}

MATERIAL_TO_REASON: dict[MaterialLabel, ReasonClass | None] = {
    "repetition": None,
    "elaboration": "elaborated",
    "addition": None,
    "correction": "correction",
    "changed state": "changed",
    "unrelated": None,
}

# ---------------------------------------------------------------------------
# Check grammar — the mutation axis's fixed vocabulary.
# ---------------------------------------------------------------------------

FailureClass = Literal["lost history", "wrong replacement", "live contradiction", "kept duplicate"]

_ALIAS = r"[A-Za-z0-9_-]+"


class RowsEqual(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    kind: Literal["rows_equal"] = "rows_equal"
    failure_class: Literal["kept duplicate"] = "kept duplicate"
    raw: str
    n: int


class ValidityEquals(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    kind: Literal["validity_equals"] = "validity_equals"
    failure_class: Literal["live contradiction"] = "live contradiction"
    raw: str
    alias: str
    validity: Literal["valid", "invalid"]


class SupersededByEquals(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    kind: Literal["superseded_by_equals"] = "superseded_by_equals"
    failure_class: Literal["wrong replacement"] = "wrong replacement"
    raw: str
    alias: str
    target_alias: str


class TextUnchanged(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    kind: Literal["text_unchanged"] = "text_unchanged"
    failure_class: Literal["lost history"] = "lost history"
    raw: str
    alias: str


class HistorySays(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    kind: Literal["history_says"] = "history_says"
    failure_class: Literal["lost history"] = "lost history"
    raw: str
    word: Literal["corrected", "changed", "elaborated", "moved"]


class PairRegistered(BaseModel):
    """``pair(a, b) registered`` / ``pair(a, b) not registered`` — ``expected`` (P8
    assessor-automatic-exits piece) distinguishes the two: ``True`` (default, the
    original grammar) asserts the pair IS registered; ``False`` asserts the automatic
    ``follow_up``/``distinct`` exits' own "registers no pair" rule (D5/D8a) — the
    pair must NOT appear in the distinct-pair registry."""

    model_config = {"frozen": True, "extra": "forbid"}

    kind: Literal["pair_registered"] = "pair_registered"
    failure_class: Literal["wrong replacement"] = "wrong replacement"
    raw: str
    alias_a: str
    alias_b: str
    expected: bool = True


class InvalidationEquals(BaseModel):
    """``alias.invalidated_by == by_alias reason <class>`` (P8 assessor-automatic-exits
    piece) — the Fact-invalidates-Decision transaction's own two columns, read
    directly off the adapter (not projected onto ``BrowseHit`` — see P9)."""

    model_config = {"frozen": True, "extra": "forbid"}

    kind: Literal["invalidation_equals"] = "invalidation_equals"
    failure_class: Literal["live contradiction"] = "live contradiction"
    raw: str
    alias: str
    by_alias: str
    reason: Literal["correction", "changed"]


Check = Annotated[
    RowsEqual
    | ValidityEquals
    | SupersededByEquals
    | TextUnchanged
    | HistorySays
    | PairRegistered
    | InvalidationEquals,
    Field(discriminator="kind"),
]

_ROWS_EQUAL = re.compile(r"^rows == (?P<n>\d+)$")
_VALIDITY_EQUALS = re.compile(rf"^(?P<alias>{_ALIAS})\.validity == (?P<validity>valid|invalid)$")
_SUPERSEDED_BY_EQUALS = re.compile(
    rf"^(?P<alias>{_ALIAS})\.superseded_by == (?P<target>{_ALIAS})$"
)
_TEXT_UNCHANGED = re.compile(rf"^(?P<alias>{_ALIAS})\.text unchanged$")
_HISTORY_SAYS = re.compile(r"^history says (?P<word>corrected|changed|elaborated|moved)$")
_PAIR_REGISTERED = re.compile(rf"^pair\((?P<a>{_ALIAS}),\s*(?P<b>{_ALIAS})\) registered$")
_PAIR_NOT_REGISTERED = re.compile(
    rf"^pair\((?P<a>{_ALIAS}),\s*(?P<b>{_ALIAS})\) not registered$"
)
_INVALIDATION_EQUALS = re.compile(
    rf"^(?P<alias>{_ALIAS})\.invalidated_by == (?P<by_alias>{_ALIAS}) "
    r"reason (?P<reason>correction|changed)$"
)


def parse_check(raw: str) -> Check:
    """Parse one check string from the fixed vocabulary into a typed ``Check``.

    Raises:
        ValueError: ``raw`` matches none of the fixed patterns — the message names the offending
            string; ``load_cases`` wraps it with the file name.
    """
    if match := _ROWS_EQUAL.fullmatch(raw):
        return RowsEqual(raw=raw, n=int(match["n"]))
    if match := _VALIDITY_EQUALS.fullmatch(raw):
        validity = cast("Literal['valid', 'invalid']", match["validity"])
        return ValidityEquals(raw=raw, alias=match["alias"], validity=validity)
    if match := _SUPERSEDED_BY_EQUALS.fullmatch(raw):
        return SupersededByEquals(raw=raw, alias=match["alias"], target_alias=match["target"])
    if match := _TEXT_UNCHANGED.fullmatch(raw):
        return TextUnchanged(raw=raw, alias=match["alias"])
    if match := _HISTORY_SAYS.fullmatch(raw):
        word = cast("Literal['corrected', 'changed', 'elaborated', 'moved']", match["word"])
        return HistorySays(raw=raw, word=word)
    if match := _PAIR_NOT_REGISTERED.fullmatch(raw):
        return PairRegistered(raw=raw, alias_a=match["a"], alias_b=match["b"], expected=False)
    if match := _PAIR_REGISTERED.fullmatch(raw):
        return PairRegistered(raw=raw, alias_a=match["a"], alias_b=match["b"])
    if match := _INVALIDATION_EQUALS.fullmatch(raw):
        reason = cast("Literal['correction', 'changed']", match["reason"])
        return InvalidationEquals(
            raw=raw, alias=match["alias"], by_alias=match["by_alias"], reason=reason
        )
    raise ValueError(f"unknown check kind: {raw!r}")


def coerce_checks(value: Any) -> Any:
    """Parse every string in ``value`` into a typed ``Check``; anything else already-typed rides
    through unchanged. Shared by ``MutationExpect``'s own field validator and
    ``ideal_judge.IdealVerdict``, so both the case file and a judge's parsed response run the
    identical grammar."""
    if not isinstance(value, list):
        return value
    return [parse_check(item) if isinstance(item, str) else item for item in value]


class Failure(BaseModel):
    """One check that did not hold — the mutation and ideal axes' shared unit of report."""

    model_config = {"frozen": True, "extra": "forbid"}

    failure_class: FailureClass
    detail: str


# ---------------------------------------------------------------------------
# Case shape
# ---------------------------------------------------------------------------

StartLifecycle = Literal["active", "shelved", "archived", "buried"]


class ExistingRow(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    alias: str
    text: str
    kind: Literal["fact", "decision"]
    scope: str
    lifecycle: StartLifecycle
    validity: Literal["valid", "invalid"]
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def _require_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("existing[].created_at must be tz-aware")
        return v


class Incoming(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    content: str
    scope: str
    kind: Literal["fact", "decision"]
    supersedes: list[str] = Field(default_factory=list)
    reason: ReasonClass | None = None


class Selection(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    aliases: list[str]
    k: int


class MutationExpect(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    acceptable: list[Check] = Field(default_factory=list)
    ideal: list[Check] = Field(default_factory=list)

    @field_validator("acceptable", "ideal", mode="before")
    @classmethod
    def _parse_check_strings(cls, value: Any) -> Any:
        return coerce_checks(value)


class Expect(BaseModel):
    """``step`` (P8 assessor-automatic-exits piece): the ``admission_resolved`` step the
    incoming save must log. ``None`` (default) means the mutation-axis runner does not
    check it, matching every P7 case row's own silence on the step vocabulary."""

    model_config = {"frozen": True, "extra": "forbid"}

    selection: Selection | None = None
    material: MaterialLabel
    mutation: MutationExpect
    step: str | None = None


class FalsityJudgeAnswer(BaseModel):
    """One judge seat's scripted answer on a falsity pair (A3 cleaner-integration piece, scope
    5) — the same shape ``FalsityVerdict`` enforces in production: ``reason`` required exactly
    when ``verdict == "replace"``, forbidden otherwise."""

    model_config = {"frozen": True, "extra": "forbid"}

    verdict: Literal["coexist", "replace"]
    reason: Literal["correction", "changed"] | None = None
    why: str = "because"

    @model_validator(mode="after")
    def _reason_matches_verdict(self) -> FalsityJudgeAnswer:
        if self.verdict == "replace" and self.reason is None:
            raise ValueError("falsity judge_a/judge_b: verdict='replace' requires reason")
        if self.verdict == "coexist" and self.reason is not None:
            raise ValueError("falsity judge_a/judge_b: verdict='coexist' must not carry reason")
        return self


#: The wire's own three-label choice, plus the case grammar's fourth word for a scripted
#: Cleaner-timeout degrade — never sent to ``ContradictionVerdict`` itself, resolved by the
#: runner into a raised exception instead (``ask()`` is what turns a raise into ``Unavailable``).
FalsityAssessorPick = Literal["coexist", "correction", "changed", "unavailable"]

#: ``supersede:<reason>``'s reason half is parsed out by ``FalsityBlock``'s own validator, never
#: carried as a free string on this Literal — the four fixed shapes below are validated against
#: the wire vocabulary AS the incoming string is parsed.
FalsityExpectValue = Literal[
    "coexist", "contested", "supersede:correction", "supersede:changed", "global_reported"
]


class FalsityBlock(BaseModel):
    """A case's falsity-axis block (Constraints, A3 cleaner-integration piece) — mutually
    exclusive with ``expect`` (a falsity case scores the Cleaner's falsity axis on the case's
    own ``existing[0]``/``incoming`` pair, never the material/mutation/selection/ideal axes)."""

    model_config = {"frozen": True, "extra": "forbid"}

    judge_a: FalsityJudgeAnswer
    judge_b: FalsityJudgeAnswer
    assessor: FalsityAssessorPick | None = None
    expect: FalsityExpectValue


class Fault(BaseModel):
    """A case's own fault target: raise before delegation on the ``call``-th invocation of
    method ``on``.

    YAML gotcha: a BARE ``on:`` key parses as the boolean ``True`` under PyYAML's default (YAML
    1.1) resolver, not the string ``"on"`` — a case file must write ``"on": upsert`` (quoted) or
    loading fails with a missing-field error naming this model, not the case file.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    on: str
    call: int


#: Reserved: bound by the runner to the record ``engine.save(case.incoming...)`` just wrote, so
#: a check can name the incoming save as a ``superseded_by``/``pair(...)`` target (e.g.
#: ``e1.superseded_by == incoming``). An ``existing[]`` row must never reuse it — a case that did
#: would evaluate checks against the incoming save's id instead of that row's, silently.
RESERVED_EXISTING_ALIAS = "incoming"


class Case(BaseModel):
    model_config = {"frozen": True, "populate_by_name": True, "extra": "forbid"}

    id: str
    class_: str = Field(alias="class")
    origin: Literal["store", "authored"]
    existing: list[ExistingRow] = Field(default_factory=list)
    incoming: Incoming
    expect: Expect | None = None
    falsity: FalsityBlock | None = None
    why: str
    fault: Fault | None = None

    @model_validator(mode="after")
    def _validate_existing_aliases(self) -> Case:
        aliases = [row.alias for row in self.existing]

        if RESERVED_EXISTING_ALIAS in aliases:
            raise ValueError(
                f"existing[].alias must not reuse the reserved alias {RESERVED_EXISTING_ALIAS!r}"
            )

        duplicates = sorted({alias for alias in aliases if aliases.count(alias) > 1})
        if duplicates:
            raise ValueError(f"existing[].alias must be unique, duplicates: {duplicates}")

        return self

    @model_validator(mode="after")
    def _validate_expect_xor_falsity(self) -> Case:
        if (self.expect is None) == (self.falsity is None):
            raise ValueError(
                "a case must carry exactly one of expect (material/mutation/selection/ideal) "
                "or falsity, never both, never neither"
            )
        if self.falsity is not None and len(self.existing) != 1:
            raise ValueError(
                "a falsity case's existing[] must name exactly one row -- the pair's older "
                "record; incoming is the newer one"
            )
        return self


class CaseLoadError(Exception):
    """A case YAML file could not be loaded — names the file and the offending string."""


def load_cases(dir: Path | str | None = None) -> list[Case]:
    """Read every ``*.yaml`` case file, sorted by name, into a ``Case``.

    Resolution order for the directory: an explicit ``dir`` argument; else the
    ``TABULARIUM_ADMISSION_CASES`` environment variable (the pytester-driven shape test's own
    switch); else ``tests/cases/admission``. An empty or absent directory returns ``[]`` — the
    harness stays green with zero cases.

    Raises:
        CaseLoadError: a file's YAML is well-formed but fails the case schema (an unknown check
            string, an unknown material label, a missing field) — the message names the file.
    """
    if dir is not None:
        cases_dir = Path(dir)
    else:
        env_dir = os.environ.get(_CASES_DIR_ENV)
        cases_dir = Path(env_dir) if env_dir else _DEFAULT_CASES_DIR

    if not cases_dir.is_dir():
        return []

    cases: list[Case] = []
    for path in sorted(cases_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            cases.append(Case.model_validate(raw))
        except ValidationError as exc:
            raise CaseLoadError(f"{path.name}: {exc}") from exc
    return cases


def case_texts(cases: Sequence[Case]) -> set[str]:
    """Every ``existing[].text`` and ``incoming.content`` across ``cases`` — for the freeze
    script, which must freeze a real vector for each before the default lane can embed it."""
    texts: set[str] = set()
    for case in cases:
        for row in case.existing:
            texts.add(row.text)
        texts.add(case.incoming.content)
    return texts
