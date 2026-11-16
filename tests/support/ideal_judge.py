"""The ``ideal`` tier's prompt/parse/score round-trip: a pluggable judge
seat, reusing ``JudgeAdapter``/``JudgeSeatConfig``, that sees a case's ``existing`` and
``incoming`` only — never its ``why`` — and answers with a material label plus expected checks.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from tabularium.engine.rubric import load_rubric
from tests.support.cases import Case, Check, Failure, MaterialLabel, coerce_checks


class IdealVerdict(BaseModel):
    """The ``ideal``-tier judge's parsed answer."""

    model_config = {"frozen": True}

    material: MaterialLabel
    checks: list[Check] = Field(default_factory=list)

    @field_validator("checks", mode="before")
    @classmethod
    def _parse_check_strings(cls, value: Any) -> Any:
        return coerce_checks(value)


_TEMPLATE_TOKEN = re.compile(r"\{existing\}|\{incoming\}")


def build_ideal_prompt(case: Case) -> str:
    """Render the case-set rubric's prompt template for ``case``.

    Fills ``{existing}`` with every existing row's alias and text, and ``{incoming}`` with the
    incoming content — never ``case.why``, which stays for the human reviewer only.

    Both tokens are substituted in a single pass over the template: two chained ``.replace()``
    calls would let the SECOND call rescan the
    text the FIRST call just inserted, so an existing row whose content is literally the string
    ``"{incoming}"`` would have the incoming content substituted into it too.
    """
    rubric = load_rubric()
    assert rubric.caseset is not None, "packaged rubric artifact is missing [caseset]"
    template = rubric.caseset.prompt_template
    existing_block = (
        "\n\n".join(f"[{row.alias}] {row.text}" for row in case.existing) or "(none)"
    )
    filled = {"{existing}": existing_block, "{incoming}": case.incoming.content}
    return _TEMPLATE_TOKEN.sub(lambda match: filled[match.group(0)], template)


def parse_ideal_response(text: str) -> IdealVerdict:
    """Parse a judge's raw text response — a JSON object ``{"material": ..., "checks": [...]}``.

    Raises:
        ValueError: ``text`` is not valid JSON, or fails ``IdealVerdict``'s own schema (an
            unrecognized material label or check string).
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ideal judge response is not valid JSON: {exc}") from exc
    return IdealVerdict.model_validate(data)


def _check_identity(check: Check) -> tuple[tuple[str, Any], ...]:
    """A hashable key for ``check``, its parsed fields only — ``raw`` excluded so two strings
    that parse to the same meaning (``pair(e1,e2) registered`` vs ``pair(e1, e2) registered``,
    whitespace aside) compare equal."""
    return tuple(sorted(check.model_dump(exclude={"raw"}).items()))


def score_ideal(case: Case, verdict: IdealVerdict) -> list[Failure]:
    """Score a parsed ``IdealVerdict`` against ``case.expect`` — zero failures for an exact
    match on both the material label and the ``mutation.ideal`` check set.

    A material mismatch is reported under ``"wrong replacement"`` — the material label is what
    determines which mutation the store should have performed, so a wrong label is, at root, the
    judge naming the wrong replacement. A missing or unexpected check is reported under that
    check's own ``failure_class``.
    """
    failures: list[Failure] = []

    assert case.expect is not None, f"case {case.id!r}: ideal scoring needs an expect block"
    if verdict.material != case.expect.material:
        failures.append(
            Failure(
                failure_class="wrong replacement",
                detail=f"material: expected {case.expect.material!r}, got {verdict.material!r}",
            )
        )

    expected_by_identity = {_check_identity(check): check for check in case.expect.mutation.ideal}
    actual_by_identity = {_check_identity(check): check for check in verdict.checks}

    for identity, check in expected_by_identity.items():
        if identity not in actual_by_identity:
            failures.append(
                Failure(failure_class=check.failure_class, detail=f"missing: {check.raw}")
            )
    for identity, check in actual_by_identity.items():
        if identity not in expected_by_identity:
            failures.append(
                Failure(failure_class=check.failure_class, detail=f"unexpected: {check.raw}")
            )

    return failures
