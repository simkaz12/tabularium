"""The judge seam (2026-08-10 cleaner plan) — the engine's first generative call.

Hermetic lane only, per the task brief ("This task is testable without credentials. A fake adapter
is the primary test surface; that is the point of a seam."). No test spawns a real ``claude``/
``codex`` process or calls a real litellm provider: subprocess mechanics are exercised against
``asyncio.create_subprocess_exec`` monkeypatched to a recording double, and litellm mechanics
against ``litellm.acompletion`` monkeypatched the same way. No credential and no network call is
required anywhere in this file. One deliberate exception (2026-08-13 ``MAX_ARG_STRLEN`` fix): the
two ``test_prompt_far_larger_than_max_arg_strlen_reaches_*_seat_via_stdin`` tests spawn a REAL
subprocess, because the property under test (does a huge prompt actually survive the OS-level argv
limit) cannot be proven against a mock that never goes through ``execve(2)`` at all -- they still
use a trivial local script standing in for the binary, never the credentialed claude/codex CLIs, so
this stays fully hermetic (no network, no credential).

Falsifiability: the
strict-parse property is enforced by PRODUCTION code two ways, and both were proven sensitive to a
real defect, then reverted:

1. ``_GradedVerdict`` below extends ``UnclearVerdictBase`` (production, ``engine/judge.py``) rather
   than redeclaring ``unclear: bool = Field(strict=True)`` itself. Removing ``strict=True`` from
   ``UnclearVerdictBase.unclear`` in ``engine/judge.py`` and re-running
   ``test_unclear_non_boolean_is_invalid_output`` / ``test_unclear_missing_is_invalid_output`` turns
   both red for exactly that reason (silent lax coercion instead of a raise).
2. Commenting out the ``_check_unclear_is_strict(item_model)`` call at the top of
   ``parse_batch_verdicts`` in ``engine/judge.py`` turns
   ``test_unclear_field_not_strict_is_caught_even_without_extending_the_base`` red: a schema that
   does NOT extend ``UnclearVerdictBase`` and declares a lax ``unclear: bool`` field stops being
   caught at all, and a payload with ``"unclear": 0`` parses as ``ok`` instead of raising.

Strict-``unclear`` enforcement used to exist ONLY in this file's own test-local
schema, which no production code would ever import — ``grep -rn "unclear" src/tabularium/`` found
zero production fields named ``unclear``. Both breaks above were checked by hand.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, ClassVar, Literal

import pytest
from litellm.types.utils import Choices, Message, ModelResponse
from pydantic import ConfigDict, Field, ValidationError

from tabularium.engine.falsity import FalsityVerdict
from tabularium.engine.judge import (
    PROMPT_CHAR_CEILING,
    IndexedVerdictBase,
    JudgeCallRecord,
    JudgeInvalidOutput,
    JudgeOutcome,
    JudgeUnavailable,
    LitellmJudgeAdapter,
    SubprocessJudgeAdapter,
    UnclearFieldNotStrict,
    UnclearVerdictBase,
    build_judge_adapter,
    judge_batch,
    pack_batches,
    parse_batch_verdicts,
    render_batch_prompt,
)
from tabularium.engine.judge_policy import McpGrant
from tabularium.engine.relevance import RelevanceVerdict
from tabularium.engine.revisit import RevisitVerdict
from tabularium.engine.rubric import CliSeat, LitellmSeat, load_rubric
from tests.support.doubles import FakeJudgeAdapter, FakeTokenCounter

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# A concrete verdict item schema for these tests: index/grade/unclear/why is the literal
# shape of a real failure seen in an earlier implementation, so it is the most direct
# fixture for this seam's index-protocol tests. Extends PRODUCTION `UnclearVerdictBase`
# (engine/judge.py) for its `index`/`unclear` fields -- an earlier
# version redeclared `unclear: bool = Field(strict=True)` locally, so the strict-parse
# tests below were never exercising anything a real axis schema would import.
# (The falsity axis has no `unclear` field at all -- see
# test_schema_with_no_unclear_field_passes_the_check below.)
# ---------------------------------------------------------------------------


class _GradedVerdict(UnclearVerdictBase):
    grade: str
    why: str


def _ok_item(index: int) -> dict[str, Any]:
    return {"index": index, "grade": "keep-clear", "unclear": False, "why": "because"}


def _payload(verdicts: list[dict[str, Any]]) -> str:
    return json.dumps({"verdicts": verdicts})


# ---------------------------------------------------------------------------
# The index protocol: parse_batch_verdicts
# ---------------------------------------------------------------------------


def test_parse_batch_verdicts_ok_orders_by_index_regardless_of_input_order() -> None:
    raw = _payload([_ok_item(3), _ok_item(1), _ok_item(2)])
    verdicts = parse_batch_verdicts(raw, item_model=_GradedVerdict, batch_size=3)
    assert [v.index for v in verdicts] == [1, 2, 3]


def test_missing_record_is_invalid_output() -> None:
    """A response omitting one record in the batch is invalid_output -- exactly one verdict
    per record, no omissions, no extras, no duplicates (acceptance criterion)."""
    raw = _payload([_ok_item(1), _ok_item(3)])  # index 2 never shows up
    with pytest.raises(JudgeInvalidOutput, match="missing"):
        parse_batch_verdicts(raw, item_model=_GradedVerdict, batch_size=3)


def test_duplicate_index_is_invalid_output() -> None:
    raw = _payload([_ok_item(1), _ok_item(1), _ok_item(2)])
    with pytest.raises(JudgeInvalidOutput, match="duplicate"):
        parse_batch_verdicts(raw, item_model=_GradedVerdict, batch_size=2)


def test_out_of_range_index_is_invalid_output() -> None:
    raw = _payload([_ok_item(1), _ok_item(2), _ok_item(99)])
    with pytest.raises(JudgeInvalidOutput, match="out of range"):
        parse_batch_verdicts(raw, item_model=_GradedVerdict, batch_size=2)


def test_malformed_json_is_invalid_output() -> None:
    with pytest.raises(JudgeInvalidOutput):
        parse_batch_verdicts("not json{{{", item_model=_GradedVerdict, batch_size=1)


def test_missing_verdicts_key_is_invalid_output() -> None:
    with pytest.raises(JudgeInvalidOutput, match="verdicts"):
        parse_batch_verdicts(json.dumps({"foo": "bar"}), item_model=_GradedVerdict, batch_size=1)


def test_response_that_is_not_a_json_object_is_invalid_output() -> None:
    with pytest.raises(JudgeInvalidOutput):
        parse_batch_verdicts(json.dumps([1, 2, 3]), item_model=_GradedVerdict, batch_size=1)


def test_bool_masquerading_as_index_is_invalid_output() -> None:
    """bool is a subclass of int in Python -- "index": true must not silently pass as index 1
    (the exact concern an indexed-verdict check must cover)."""
    bad = _ok_item(1)
    bad["index"] = True
    raw = _payload([bad])
    with pytest.raises(JudgeInvalidOutput):
        parse_batch_verdicts(raw, item_model=_GradedVerdict, batch_size=1)


def test_id_shaped_string_instead_of_index_is_invalid_output() -> None:
    """Acceptance criterion: a judge returning an id-shaped string instead of an index is
    invalid_output."""
    bad = _ok_item(1)
    bad["index"] = "93894994-2938-4abc-9a12-c7f312345678"  # ordinary UUID-shaped string literal
    raw = _payload([bad])
    with pytest.raises(JudgeInvalidOutput):
        parse_batch_verdicts(raw, item_model=_GradedVerdict, batch_size=1)


def test_id_shaped_string_fixture_raises_no_partial_set_no_disk_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real failure from an earlier implementation, as a fixture: a response containing a
    record-id-shaped string where a batch-local integer index was required parses as
    invalid_output and raises. It must not produce a partial verdict set and must not be
    written to a file as a side effect.
    """
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.iterdir())

    bad = _ok_item(1)
    bad["index"] = "93894994-2938-4abc-9a12-c7f312345678"  # ordinary UUID-shaped string literal
    raw = _payload([bad, _ok_item(2)])

    # No partial verdict set: the only way to observe a return value here is via
    # pytest.raises catching the exception -- there is no code path in
    # parse_batch_verdicts that returns before every index is accounted for.
    with pytest.raises(JudgeInvalidOutput):
        parse_batch_verdicts(raw, item_model=_GradedVerdict, batch_size=2)

    after = set(tmp_path.iterdir())
    assert after == before, "parsing a bad response must never write anything to disk"


def test_unclear_missing_is_invalid_output() -> None:
    item = _ok_item(1)
    del item["unclear"]
    raw = _payload([item])
    with pytest.raises(JudgeInvalidOutput):
        parse_batch_verdicts(raw, item_model=_GradedVerdict, batch_size=1)


@pytest.mark.parametrize("bad_value", [0, 1, "false", "true", None, "yes"])
def test_unclear_non_boolean_is_invalid_output(bad_value: object) -> None:
    """`unclear` is parsed strictly: a missing or malformed field is invalid_output, never
    coerced to False (the `bool(v.get("unclear", False))` defect)."""
    item = _ok_item(1)
    item["unclear"] = bad_value
    raw = _payload([item])
    with pytest.raises(JudgeInvalidOutput):
        parse_batch_verdicts(raw, item_model=_GradedVerdict, batch_size=1)


def test_unclear_true_boolean_is_accepted() -> None:
    """Sanity: the strict field still accepts a genuine JSON boolean."""
    item = _ok_item(1)
    item["unclear"] = True
    raw = _payload([item])
    verdicts = parse_batch_verdicts(raw, item_model=_GradedVerdict, batch_size=1)
    assert verdicts[0].unclear is True


# ---------------------------------------------------------------------------
# The mechanical guarantee (required): strict-`unclear` enforcement must not
# depend on a caller extending `UnclearVerdictBase` -- `parse_batch_verdicts` must catch a
# non-strict `unclear` field on ANY item_model, by inspecting the schema itself.
# ---------------------------------------------------------------------------


class _LaxUnclearVerdict(IndexedVerdictBase):
    """A deliberately BAD axis-schema stand-in: declares `unclear` WITHOUT strict=True and does
    NOT extend `UnclearVerdictBase` -- exactly the shape a future schema could ship if
    it merely forgot the annotation rather than skipping the base class outright. This is what
    proves the enforcement is mechanical: nothing here opts into the check, and it still fires.
    """

    grade: str
    unclear: bool  # BAD: no strict=True -- this is the property under test
    why: str


class _NoUnclearVerdict(IndexedVerdictBase):
    """A schema with no `unclear` field at all -- the falsity axis's real shape. The
    check must pass trivially: there is nothing to enforce.
    """

    verdict: str


class _ModelLevelStrictUnclearVerdict(IndexedVerdictBase):
    """`unclear` has no per-field `Field(strict=True)`, but `model_config = {"strict": True}`
    covers every field including it -- the check's other satisfying shape, per its own docstring.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(strict=True)

    grade: str
    unclear: bool
    why: str


def test_unclear_field_not_strict_is_caught_even_without_extending_the_base() -> None:
    """A schema that neither extends `UnclearVerdictBase` nor otherwise
    declares `unclear` as strict is rejected by `parse_batch_verdicts` BEFORE the response is
    even looked at -- proven here with a fully well-formed, otherwise-valid payload, so the only
    thing that can be firing is the schema check.
    """
    raw = _payload([{"index": 1, "grade": "keep-clear", "unclear": False, "why": "because"}])
    with pytest.raises(UnclearFieldNotStrict, match="_LaxUnclearVerdict"):
        parse_batch_verdicts(raw, item_model=_LaxUnclearVerdict, batch_size=1)


def test_unclear_field_not_strict_is_a_type_error() -> None:
    """Deliberately a caller-schema authoring defect (TypeError), not a judge-response outcome
    (JudgeInvalidOutput) -- the fix is to change the schema, not to blame the judge."""
    assert issubclass(UnclearFieldNotStrict, TypeError)
    assert not issubclass(UnclearFieldNotStrict, JudgeInvalidOutput)


def test_schema_with_no_unclear_field_passes_the_check() -> None:
    """The falsity axis's shape (no `unclear` at all) is not penalized by a check that exists
    only for schemas that carry the field."""
    raw = _payload([{"index": 1, "verdict": "replace"}])
    verdicts = parse_batch_verdicts(raw, item_model=_NoUnclearVerdict, batch_size=1)
    assert verdicts[0].verdict == "replace"


def test_model_level_strict_config_also_satisfies_the_check() -> None:
    """`model_config = {"strict": True}` is an equally acceptable way to satisfy the guarantee,
    per `_check_unclear_is_strict`'s own docstring -- not just a per-field `Field(strict=True)`.
    """
    raw = _payload([{"index": 1, "grade": "keep-clear", "unclear": False, "why": "because"}])
    verdicts = parse_batch_verdicts(raw, item_model=_ModelLevelStrictUnclearVerdict, batch_size=1)
    assert verdicts[0].unclear is False

    # And the model-level strict config still rejects a lax value for `unclear`, same as the
    # per-field case.
    bad_raw = _payload([{"index": 1, "grade": "keep-clear", "unclear": 0, "why": "because"}])
    with pytest.raises(JudgeInvalidOutput):
        parse_batch_verdicts(bad_raw, item_model=_ModelLevelStrictUnclearVerdict, batch_size=1)


def test_unclear_verdict_base_is_the_production_source_of_strictness() -> None:
    """`_GradedVerdict` (this file's own test fixture) gets its strict `unclear` field from
    PRODUCTION `UnclearVerdictBase`, not from a private redeclaration -- so the check
    cannot drift from the schema. A model built the same way `UnclearVerdictBase` intends -- extend
    it,
    add axis fields -- passes the mechanical check without redeclaring anything.
    """

    class _Extends(UnclearVerdictBase):
        note: str

    assert issubclass(_Extends, UnclearVerdictBase)
    raw = _payload([{"index": 1, "unclear": True, "note": "x"}])
    verdicts = parse_batch_verdicts(raw, item_model=_Extends, batch_size=1)
    assert verdicts[0].unclear is True


# ---------------------------------------------------------------------------
# render_batch_prompt: judges see only batch-local integers
# ---------------------------------------------------------------------------


def test_no_record_id_in_rendered_prompt() -> None:
    contents = ["Record one content, about project-a policies.", "Record two content, about scans."]
    prompt = render_batch_prompt(
        rubric_prose="RUBRIC PROSE GOES HERE", contents=contents, item_model=_GradedVerdict
    )

    record_ids = [
        "fact-11111111-1111-1111-1111-111111111111",
        "93894994-2938-4abc-9a12-c7f312345678",
    ]
    for rid in record_ids:
        assert rid not in prompt

    assert "## 1." in prompt
    assert "## 2." in prompt
    assert "RUBRIC PROSE GOES HERE" in prompt
    for c in contents:
        assert c in prompt


def test_render_batch_prompt_signature_cannot_receive_a_record_id() -> None:
    """Structural guarantee, not merely a convention: the function has no parameter through
    which an id could even be passed."""
    import inspect

    sig = inspect.signature(render_batch_prompt)
    assert set(sig.parameters) == {"rubric_prose", "contents", "item_model"}


def _section(prompt: str) -> str:
    assert "# Reply format" in prompt
    return prompt[prompt.index("# Reply format") :]


@pytest.mark.parametrize("model", [RelevanceVerdict, RevisitVerdict, FalsityVerdict])
def test_prompt_ends_with_reply_format_generated_from_the_model(
    model: type[IndexedVerdictBase],
) -> None:
    prompt = render_batch_prompt(rubric_prose="PROSE", contents=["a", "b"], item_model=model)
    section = _section(prompt)

    assert prompt.endswith(section)
    assert prompt.index("# Your batch — 2 records") < prompt.index("# Reply format")
    for name in model.model_fields:
        assert f"`{name}`" in section
    for value in ("shelve-clear", "shelve-lean", "keep-lean", "keep-clear"):
        if "grade" in model.model_fields:
            assert f'"{value}"' in section
    if "unclear" in model.model_fields:
        assert "a JSON boolean, not a string" in section
    assert "a JSON integer, not a string" in section
    assert "A3" not in section
    assert model.__doc__ is not None
    assert model.__doc__.strip().splitlines()[0] not in section
    assert not [ln for ln in section.splitlines() if re.fullmatch(r"## \d+\.", ln)]
    assert "record id" not in section.lower()


def test_falsity_reply_format_carries_the_conditional_reason_rule() -> None:
    section = _section(
        render_batch_prompt(rubric_prose="P", contents=["a"], item_model=FalsityVerdict)
    )

    assert '"coexist", "replace"' in section
    assert '"correction", "changed"' in section
    assert "required when `verdict` is `replace` and omitted when it is `coexist`" in section
    assert "wrong when it was written" in section


def test_reply_format_follows_a_new_field_without_touching_judge() -> None:
    class _Extended(UnclearVerdictBase):
        severity: Literal["low", "high"]
        note: str = Field(description="One short note.")

    section = _section(render_batch_prompt(rubric_prose="P", contents=["a"], item_model=_Extended))

    assert "`severity`" in section
    assert '"low", "high"' in section
    assert "`note`" in section
    assert "One short note." in section


@pytest.mark.parametrize(
    ("model", "entries"),
    [
        (
            RelevanceVerdict,
            [{"index": 1, "grade": "keep-clear", "why": "w", "unclear": False}],
        ),
        (
            RevisitVerdict,
            [{"index": 1, "grade": "shelve-lean", "why": "w", "unclear": True}],
        ),
        (
            FalsityVerdict,
            [
                {"index": 1, "verdict": "replace", "reason": "correction", "why": "w"},
                {"index": 2, "verdict": "coexist", "why": "w"},
            ],
        ),
    ],
)
def test_reply_written_to_the_section_spec_parses(
    model: type[IndexedVerdictBase], entries: list[dict[str, Any]]
) -> None:
    section = _section(
        render_batch_prompt(rubric_prose="P", contents=[f"c{i}" for i in entries], item_model=model)
    )
    assert all(f"`{name}`" in section for name in entries[0])

    parsed = parse_batch_verdicts(
        json.dumps({"verdicts": entries}), item_model=model, batch_size=len(entries)
    )

    assert [v.index for v in parsed] == list(range(1, len(entries) + 1))


# ---------------------------------------------------------------------------
# judge_batch / JudgeOutcome (cleaner-telemetry change): judge_batch is now
# TOTAL -- it never raises, on any of the four paths. Every test below asserts through the
# returned JudgeOutcome, never through pytest.raises.
# ---------------------------------------------------------------------------


async def test_judge_batch_success_stamps_identity_rubric_version_and_prompt_tokens() -> None:
    adapter = FakeJudgeAdapter(identity="fake:v1", response=_payload([_ok_item(1), _ok_item(2)]))
    outcome = await judge_batch(
        adapter,
        prompt="PROMPT",
        item_model=_GradedVerdict,
        batch_size=2,
        rubric_version="rv-1",
        token_counter=FakeTokenCounter(),
    )
    assert isinstance(outcome, JudgeOutcome)
    assert outcome.status == "success"
    assert outcome.judge == "fake:v1"
    assert outcome.rubric_version == "rv-1"
    assert [v.index for v in outcome.verdicts] == [1, 2]
    assert adapter.calls == ["PROMPT"]
    assert outcome.error is None
    # FakeTokenCounter's cost function is len(text) // 4 (tokens.py) -- computed from the fixture,
    # never copied from observed output.
    assert outcome.prompt_tokens == len("PROMPT") // 4
    assert outcome.completion_tokens is None
    assert outcome.cost is None
    assert outcome.duration_seconds >= 0.0


async def test_judge_batch_unavailable_returns_outcome_not_raise() -> None:
    """Acceptance criterion: judge_batch raises nothing -- an unavailable judge is reported, not
    propagated."""
    adapter = FakeJudgeAdapter(should_fail=True, fail_reason="401 OAuth access token has expired")
    outcome = await judge_batch(
        adapter,
        prompt="p",
        item_model=_GradedVerdict,
        batch_size=1,
        rubric_version="rv",
        token_counter=FakeTokenCounter(),
    )
    assert outcome.status == "judge_unavailable"
    assert outcome.error is not None
    assert "401 OAuth access token has expired" in outcome.error
    assert outcome.verdicts == []
    assert outcome.judge == adapter.identity
    assert outcome.rubric_version == "rv"
    assert outcome.prompt_tokens == len("p") // 4


async def test_judge_batch_invalid_output_returns_outcome_not_raise() -> None:
    adapter = FakeJudgeAdapter(response="not json")
    outcome = await judge_batch(
        adapter,
        prompt="p",
        item_model=_GradedVerdict,
        batch_size=1,
        rubric_version="rv",
        token_counter=FakeTokenCounter(),
    )
    assert outcome.status == "invalid_output"
    assert outcome.error is not None
    assert outcome.verdicts == []
    assert outcome.prompt_tokens == len("p") // 4


async def test_judge_batch_unexpected_exception_becomes_a_status_not_a_swallowed_pass() -> None:
    """The load-bearing case (acceptance criterion): an adapter raising an exception OUTSIDE the
    two named types (here, a plain RuntimeError -- neither JudgeUnavailable nor
    JudgeInvalidOutput) yields status="unexpected" with the exception text in `error` -- not a
    success classification, not an AssertionError, not a swallowed pass.

    Before this task, this exact shape (a raised RuntimeError from `adapter.call`) was
    misclassified as "ok" by the now-deleted `_judge_outcome` in both `engine/relevance.py`
    (line 362-363 at c56a06b) and `engine/falsity.py` (line 540-541 at c56a06b), which then
    tripped `assert isinstance(result_a, JudgeBatchResult)` -- an AssertionError on the CLI, a
    silently swallowed pass behind `_run_cleaner_pass`'s broad `except Exception`
    (`engine/core.py:1221`) in the background.
    """
    adapter = FakeJudgeAdapter(raise_exception=RuntimeError("boom: unanticipated adapter failure"))
    outcome = await judge_batch(
        adapter,
        prompt="p",
        item_model=_GradedVerdict,
        batch_size=1,
        rubric_version="rv",
        token_counter=FakeTokenCounter(),
    )
    assert outcome.status == "unexpected"
    assert outcome.error is not None
    assert "boom: unanticipated adapter failure" in outcome.error
    assert outcome.verdicts == []
    assert outcome.judge == adapter.identity
    assert outcome.rubric_version == "rv"
    assert outcome.prompt_tokens == len("p") // 4


async def test_judge_batch_duration_measured_across_the_adapter_call() -> None:
    """Acceptance criterion: duration is measured across the adapter call, and a fake adapter
    that sleeps produces a duration greater than that sleep."""
    delay = 0.05
    adapter = FakeJudgeAdapter(response=_payload([_ok_item(1)]), delay=delay)
    outcome = await judge_batch(
        adapter,
        prompt="p",
        item_model=_GradedVerdict,
        batch_size=1,
        rubric_version="rv",
        token_counter=FakeTokenCounter(),
    )
    assert outcome.status == "success"
    assert outcome.duration_seconds > delay


async def test_judge_batch_never_raises_for_any_of_the_four_paths() -> None:
    """Structural guarantee: none of the four paths propagates an exception out of judge_batch --
    every one of these calls would have raised (or, for the unexpected case, propagated a bare
    RuntimeError) before this task."""
    counter = FakeTokenCounter()
    adapters = [
        FakeJudgeAdapter(response=_payload([_ok_item(1)])),
        FakeJudgeAdapter(should_fail=True),
        FakeJudgeAdapter(response="not json"),
        FakeJudgeAdapter(raise_exception=ValueError("surprise")),
    ]
    for adapter in adapters:
        outcome = await judge_batch(
            adapter,
            prompt="p",
            item_model=_GradedVerdict,
            batch_size=1,
            rubric_version="rv",
            token_counter=counter,
        )
        assert outcome.status in ("success", "judge_unavailable", "invalid_output", "unexpected")


# ---------------------------------------------------------------------------
# JudgeOutcome.as_call_record (cleaner-telemetry change): the per-call
# telemetry projection an axis (relevance.py, falsity.py) collects into its own summary.
# ---------------------------------------------------------------------------


async def test_as_call_record_mirrors_every_field_except_verdicts() -> None:
    adapter = FakeJudgeAdapter(identity="fake:v1", response=_payload([_ok_item(1), _ok_item(2)]))
    outcome = await judge_batch(
        adapter,
        prompt="PROMPT",
        item_model=_GradedVerdict,
        batch_size=2,
        rubric_version="rv-1",
        token_counter=FakeTokenCounter(),
    )
    assert outcome.status == "success"
    assert len(outcome.verdicts) == 2  # the outcome itself still carries verdicts

    record = outcome.as_call_record(batch_index=3)

    assert isinstance(record, JudgeCallRecord)
    assert record.batch_index == 3
    assert record.judge == outcome.judge
    assert record.rubric_version == outcome.rubric_version
    assert record.duration_seconds == outcome.duration_seconds
    assert record.status == outcome.status
    assert record.error == outcome.error
    assert record.prompt_tokens == outcome.prompt_tokens
    assert record.completion_tokens == outcome.completion_tokens
    assert record.cost == outcome.cost
    # `verdicts` has no counterpart on `JudgeCallRecord` at all -- a structural guarantee, not
    # merely "not asserted": the model has no such field to carry it.
    assert "verdicts" not in JudgeCallRecord.model_fields


async def test_as_call_record_on_a_failed_outcome_carries_the_error_and_null_tokens() -> None:
    adapter = FakeJudgeAdapter(should_fail=True, fail_reason="401 OAuth access token has expired")
    outcome = await judge_batch(
        adapter,
        prompt="p",
        item_model=_GradedVerdict,
        batch_size=1,
        rubric_version="rv",
        token_counter=FakeTokenCounter(),
    )
    assert outcome.status == "judge_unavailable"

    record = outcome.as_call_record(batch_index=0)

    assert record.status == "judge_unavailable"
    assert record.error is not None
    assert "401 OAuth access token has expired" in record.error
    assert record.completion_tokens is None
    assert record.cost is None
    assert record.prompt_tokens == outcome.prompt_tokens


def test_judge_call_record_is_frozen() -> None:
    record = JudgeCallRecord(
        batch_index=0,
        judge="a",
        rubric_version="v1",
        duration_seconds=0.1,
        status="success",
        error=None,
        prompt_tokens=1,
        completion_tokens=None,
        cost=None,
    )
    with pytest.raises(ValidationError):
        record.status = "unexpected"


# ---------------------------------------------------------------------------
# build_judge_adapter: configuration from the rubric artifact selects the adapter
# ---------------------------------------------------------------------------


def test_build_judge_adapter_cli_seat_returns_subprocess_adapter() -> None:
    seat = CliSeat(cli="claude", model="sonnet", binary="/usr/bin/claude")
    adapter = build_judge_adapter(seat)
    assert isinstance(adapter, SubprocessJudgeAdapter)
    assert adapter.identity == "claude:sonnet"


def test_build_judge_adapter_litellm_seat_returns_litellm_adapter() -> None:
    seat = LitellmSeat(model="deepseek/deepseek-chat")
    adapter = build_judge_adapter(seat)
    assert isinstance(adapter, LitellmJudgeAdapter)
    assert adapter.identity == "litellm:deepseek/deepseek-chat"


def test_build_judge_adapter_defaults_the_timeout_to_1200_seconds() -> None:
    """Owner-requested (2026-08-11): the timeout is configurable via rubric data, defaulting to
    1200s (raised from the adapter's original hard-coded 120s). Asserted through
    ``build_judge_adapter``, never by reading ``SubprocessJudgeAdapter._DEFAULT_TIMEOUT`` — the
    class constant could agree with itself while the wiring silently dropped the value."""
    seat = CliSeat(cli="claude", model="sonnet", binary="/usr/bin/claude")
    adapter = build_judge_adapter(seat)
    assert isinstance(adapter, SubprocessJudgeAdapter)
    assert adapter.timeout == 1200.0


def test_build_judge_adapter_wires_a_seat_configured_timeout_override() -> None:
    """Falsifiability (Verify step 1): commenting out the ``timeout=seat.timeout`` kwarg in
    ``build_judge_adapter`` turns this assertion red on its own value (45.0 != 1200.0), not on a
    missing symbol — reported verbatim in the task verdict rather than duplicated here."""
    seat = CliSeat(cli="codex", model="gpt-5.1-codex", binary="/usr/bin/codex", timeout=45.0)
    adapter = build_judge_adapter(seat)
    assert isinstance(adapter, SubprocessJudgeAdapter)
    assert adapter.timeout == 45.0


def test_build_judge_adapter_from_the_shipped_rubric_config() -> None:
    """Ordering -- which judge fills which seat, and what backs it up -- is data in the
    rubric artifact, not a branch here."""
    judges = load_rubric().judges
    seat_a = build_judge_adapter(judges.seat_a)
    seat_b = build_judge_adapter(judges.seat_b)
    backstop = build_judge_adapter(judges.backstop)

    assert isinstance(seat_a, SubprocessJudgeAdapter)
    assert seat_a.identity.startswith("claude:")
    assert isinstance(seat_b, SubprocessJudgeAdapter)
    assert seat_b.identity.startswith("codex:")
    assert isinstance(backstop, LitellmJudgeAdapter)
    assert backstop.identity.startswith("litellm:deepseek/")


# ---------------------------------------------------------------------------
# SubprocessJudgeAdapter mechanics -- a recording double stands in for the real CLI, so
# these tests need neither the claude/codex binaries nor a network call.
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode: int | None = returncode
        #: Whatever ``communicate(input=...)`` was called with -- ``None`` if the caller never
        #: passed one. Lets a test assert the prompt actually reached stdin (2026-08-13 fix: the
        #: prompt moved off argv onto stdin).
        self.received_input: bytes | None = None

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        self.received_input = input
        return self._stdout, self._stderr

    def kill(self) -> None:
        pass

    async def wait(self) -> int:
        return self.returncode or 0


class _SubprocessDouble:
    """Records every call and returns a canned ``_FakeProc``.

    If ``output_file_content`` is set and the call carries ``--output-last-message <path>``,
    writes that content to ``<path>`` the way the real codex CLI does -- proving the codex
    adapter reads from that file rather than from stdout.
    """

    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        output_file_content: bytes | None = None,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.output_file_content = output_file_content
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, *argv: str, **kwargs: Any) -> _FakeProc:
        proc = _FakeProc(self.stdout, self.stderr, self.returncode)
        self.calls.append({"argv": argv, "kwargs": kwargs, "proc": proc})
        if self.output_file_content is not None and "--output-last-message" in argv:
            idx = argv.index("--output-last-message")
            Path(argv[idx + 1]).write_bytes(self.output_file_content)
        return proc


def _patch_subprocess_exec(monkeypatch: pytest.MonkeyPatch, double: _SubprocessDouble) -> None:
    monkeypatch.setattr("tabularium.engine.judge.asyncio.create_subprocess_exec", double)


async def test_claude_argv_carries_no_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Acceptance criterion (2026-08-13 fix): argv is a fixed shape regardless of prompt size --
    the prompt is never one of its elements, closing the ``MAX_ARG_STRLEN`` defect a prompt-as-
    argv-element invocation structurally could not avoid."""
    double = _SubprocessDouble(stdout=_payload([_ok_item(1)]).encode())
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli="claude", binary="/bin/claude", model="sonnet")

    raw = await adapter.call("MY PROMPT TEXT")

    assert raw == _payload([_ok_item(1)])
    argv = double.calls[0]["argv"]
    assert argv == ("/bin/claude", "-p", "--model", "sonnet", "--allowed-tools", "")
    assert "MY PROMPT TEXT" not in argv


async def test_claude_prompt_sent_via_stdin_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prompt reaches the CLI over stdin (``communicate(input=...)``), not argv -- stdin is
    ``PIPE``, not ``DEVNULL`` (2026-08-13 fix; ``DEVNULL`` was correct only while the prompt rode
    argv and stdin carried nothing)."""
    double = _SubprocessDouble(stdout=b"{}")
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli="claude", binary="/bin/claude", model="sonnet")

    await adapter.call("MY PROMPT TEXT")

    call = double.calls[0]
    assert call["kwargs"]["stdin"] is asyncio.subprocess.PIPE
    assert call["proc"].received_input == b"MY PROMPT TEXT"


async def test_claude_explicit_model_not_the_cli_default(monkeypatch: pytest.MonkeyPatch) -> None:
    double = _SubprocessDouble(stdout=b"{}")
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli="claude", binary="/bin/claude", model="opus-4.6")

    await adapter.call("p")

    argv = double.calls[0]["argv"]
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "opus-4.6"


async def test_claude_nonzero_exit_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    double = _SubprocessDouble(
        stdout=b"",
        stderr=(
            b"Input must be provided either through stdin or as a prompt argument "
            b"when using --print"
        ),
        returncode=1,
    )
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli="claude", binary="/bin/claude", model="sonnet")

    with pytest.raises(JudgeUnavailable):
        await adapter.call("p")


async def test_claude_auth_failure_401_is_unavailable_not_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance criterion: an auth failure (401 OAuth access token has expired) is
    unavailable, not invalid_output."""
    double = _SubprocessDouble(
        stdout=b"", stderr=b"401 OAuth access token has expired", returncode=1
    )
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli="claude", binary="/bin/claude", model="sonnet")

    with pytest.raises(JudgeUnavailable, match="401 OAuth access token has expired"):
        await adapter.call("p")


async def test_codex_uses_output_last_message_never_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance criterion: the codex adapter reads its result from
    --output-last-message, never from stdout. Feeds a full transcript -- banner, echoed
    prompt, bubblewrap warning, JSON, token count -- to the stdout path and asserts it is
    refused rather than scraped."""
    clean = _payload([_ok_item(1)]).encode()
    transcript = (
        b"Codex CLI v1.2.3 (build 9999)\n"
        b"> PROMPT TEXT ECHOED HERE\n"
        b"Codex could not find bubblewrap on PATH\n" + clean + b"\n"
        b"tokens used: 1234\n"
    )
    double = _SubprocessDouble(stdout=transcript, returncode=0, output_file_content=clean)
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli="codex", binary="/bin/codex", model="gpt-5.1-codex")

    raw = await adapter.call("PROMPT TEXT ECHOED HERE")

    assert raw == clean.decode()
    # Proves the transcript's stdout content never reached the return value.
    assert "Codex CLI v1.2.3" not in raw
    assert "bubblewrap" not in raw
    assert "tokens used" not in raw


async def test_codex_argv_carries_no_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Acceptance criterion (2026-08-13 fix): mirrors ``test_claude_argv_carries_no_prompt`` --
    argv is a fixed shape regardless of prompt size."""
    double = _SubprocessDouble(
        stdout=b"transcript", returncode=0, output_file_content=_payload([_ok_item(1)]).encode()
    )
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli="codex", binary="/bin/codex", model="gpt-5.1-codex")

    await adapter.call("MY PROMPT")

    argv = double.calls[0]["argv"]
    assert argv[0] == "/bin/codex"
    assert argv[1] == "exec"
    assert "--skip-git-repo-check" in argv
    assert "--output-last-message" in argv
    assert "-m" in argv
    assert argv[argv.index("-m") + 1] == "gpt-5.1-codex"  # explicit, never the CLI default
    assert "MY PROMPT" not in argv
    # --ignore-user-config is only ever added inside the CONFIGURED branch of _codex_mcp_argv (it
    # is structurally impossible for it to leak into the empty-list unconfigured return) -- this
    # asserts that structural guarantee directly, since the equality checks above are per-flag
    # `in` checks rather than a full-tuple pin the way the claude equivalent is.
    assert "--ignore-user-config" not in argv


# ---------------------------------------------------------------------------
# MCP judge-seat access (2026-08-14 judge-seat MCP-access brief) -- an opt-in grant rendered into
# each CLI's own argv shape. Every test above this section already re-pins the UNCONFIGURED argv
# byte-for-byte (no seat in the tests above ever passes an mcp_server_url); the tests below cover
# only the CONFIGURED path, added as new coverage rather than mutating the pinned tests.
# ---------------------------------------------------------------------------


async def test_claude_argv_with_mcp_grant_renders_config_and_scoped_allowed_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    double = _SubprocessDouble(stdout=_payload([_ok_item(1)]).encode())
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(
        cli="claude",
        binary="/bin/claude",
        model="sonnet",
        grant=McpGrant(
            server_url="http://127.0.0.1:8787/mcp",
            server_name="tabularium",
            allowed_tools=("decisions_query", "decision"),
        ),
    )

    await adapter.call("MY PROMPT TEXT")

    argv = double.calls[0]["argv"]
    assert argv[:4] == ("/bin/claude", "-p", "--model", "sonnet")
    assert "--mcp-config" in argv
    mcp_config = json.loads(argv[argv.index("--mcp-config") + 1])
    assert mcp_config == {
        "mcpServers": {"tabularium": {"type": "http", "url": "http://127.0.0.1:8787/mcp"}}
    }
    assert "--strict-mcp-config" in argv
    tools_idx = argv.index("--allowed-tools")
    disallowed_idx = argv.index("--disallowed-tools")
    assert argv[tools_idx + 1 : disallowed_idx] == (
        "mcp__tabularium__decisions_query",
        "mcp__tabularium__decision",
    )
    # Critical finding (2026-08-14): --allowed-tools alone is not exhaustive against an ambient
    # `.claude/settings.local.json` permission grant -- --disallowed-tools closes it. Every
    # non-safe tool must be named, regardless of what this seat's own grant happens to include.
    assert argv[disallowed_idx + 1 :] == ("mcp__tabularium__recall",)
    # Never the old blanket-deny shape once a grant is configured.
    assert "" not in argv
    assert "MY PROMPT TEXT" not in argv


async def test_codex_argv_with_mcp_grant_renders_c_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    double = _SubprocessDouble(
        stdout=b"transcript", returncode=0, output_file_content=_payload([_ok_item(1)]).encode()
    )
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(
        cli="codex",
        binary="/bin/codex",
        model="gpt-5.1-codex",
        grant=McpGrant(
            server_url="http://127.0.0.1:8787/mcp",
            server_name="tabularium",
            allowed_tools=("context",),
        ),
    )

    await adapter.call("MY PROMPT")

    argv = double.calls[0]["argv"]
    assert argv[0] == "/bin/codex"
    assert argv[1] == "exec"
    assert "--skip-git-repo-check" in argv
    # Without this flag, the -c overrides below
    # only ADD the configured server on top of whatever ~/.codex/config.toml already declares
    # (verified live: ambient servers -- github, a browser server, a memory server --
    # stayed reachable alongside it) rather than isolating the judge to ONLY this grant, the same
    # ambient-leak --strict-mcp-config exists to close on the claude side.
    assert "--ignore-user-config" in argv
    c_indices = [i for i, element in enumerate(argv) if element == "-c"]
    assert len(c_indices) == 3
    assert argv[c_indices[0] + 1] == 'mcp_servers.tabularium.url="http://127.0.0.1:8787/mcp"'
    assert argv[c_indices[1] + 1] == 'mcp_servers.tabularium.enabled_tools=["context"]'
    # Belt-and-braces companion to claude's --disallowed-tools (critical finding, 2026-08-14):
    # verified live that disabled_tools wins over enabled_tools on conflict, so naming the full
    # deny list here can only narrow codex's own already-omission-based enforcement, never widen it.
    assert argv[c_indices[2] + 1] == 'mcp_servers.tabularium.disabled_tools=["recall"]'
    assert "--output-last-message" in argv
    assert argv[argv.index("-m") + 1] == "gpt-5.1-codex"
    assert "MY PROMPT" not in argv


async def test_codex_argv_with_mcp_grant_renders_multiple_tools_as_a_toml_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    double = _SubprocessDouble(
        stdout=b"t", returncode=0, output_file_content=_payload([_ok_item(1)]).encode()
    )
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(
        cli="codex",
        binary="/bin/codex",
        model="gpt-5.1-codex",
        grant=McpGrant(
            server_url="http://127.0.0.1:8787/mcp",
            server_name="tabularium",
            allowed_tools=("decisions_query", "decision", "context"),
        ),
    )

    await adapter.call("p")

    argv = double.calls[0]["argv"]
    enabled_tools_flag = next(a for a in argv if a.startswith("mcp_servers.tabularium.enabled"))
    assert enabled_tools_flag == (
        'mcp_servers.tabularium.enabled_tools=["decisions_query", '
        '"decision", "context"]'
    )


async def test_codex_argv_escapes_a_url_containing_toml_hostile_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_codex_mcp_argv`` used to interpolate
    ``mcp_server_url`` inside a bare ``f'"{url}"'`` wrapper, which does not escape an embedded
    quote/backslash -- a URL containing one could break out of the intended TOML string and make
    codex parse a different value than the one configured. Rendering through ``json.dumps`` (valid
    TOML string syntax) closes this: the hostile characters below survive as DATA inside the one
    scalar value, never as syntax."""
    hostile_url = 'http://127.0.0.1:8787/mcp?x="}][evil={\\'
    double = _SubprocessDouble(
        stdout=b"t", returncode=0, output_file_content=_payload([_ok_item(1)]).encode()
    )
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(
        cli="codex",
        binary="/bin/codex",
        model="gpt-5.1-codex",
        grant=McpGrant(
            server_url=hostile_url,
            server_name="tabularium",
            allowed_tools=("context",),
        ),
    )

    await adapter.call("p")

    argv = double.calls[0]["argv"]
    url_flag = next(a for a in argv if a.startswith("mcp_servers.tabularium.url"))
    rendered_value = url_flag.removeprefix("mcp_servers.tabularium.url=")
    # The rendered value must be a single, well-formed JSON/TOML string literal that decodes back
    # to exactly the hostile URL -- not a value an attacker-controlled fragment could have escaped
    # out of.
    assert json.loads(rendered_value) == hostile_url


def test_build_judge_adapter_wires_mcp_fields_from_a_seat_config() -> None:
    """``build_judge_adapter`` is purely mechanical -- the fields travel from the rubric's data
    straight through to the adapter's own constructor, exposed via the same public-property
    convention ``timeout`` already established ."""
    seat = CliSeat(
        cli="claude",
        model="sonnet",
        binary="/usr/bin/claude",
        mcp_server_url="http://127.0.0.1:8787/mcp",
        mcp_server_name="tabularium",
        mcp_allowed_tools=("decisions_query", "context"),
    )
    adapter = build_judge_adapter(seat)
    assert isinstance(adapter, SubprocessJudgeAdapter)
    assert adapter.grant == McpGrant(
        server_url="http://127.0.0.1:8787/mcp",
        server_name="tabularium",
        allowed_tools=("decisions_query", "context"),
    )


def test_build_judge_adapter_from_the_shipped_rubric_config_has_no_mcp_grant() -> None:
    """Default off (task brief point 3): the packaged rubric configures MCP access on no seat, so
    every adapter it builds carries an empty grant -- the pre-existing, unconfigured argv shape."""
    judges = load_rubric().judges
    for seat in (judges.seat_a, judges.seat_b):
        adapter = build_judge_adapter(seat)
        assert isinstance(adapter, SubprocessJudgeAdapter)
        assert adapter.grant is None


# ---------------------------------------------------------------------------
# build_judge_adapter re-validates a STANDALONE seat config: JudgesConfig's own seat-labelled
# validator only fires for a seat loaded through
# a full Rubric -- a CliSeat constructed directly, bypassing JudgesConfig entirely, was NOT
# rejected by CliSeat on its own and would have rendered a destructive grant cleanly. These
# prove build_judge_adapter itself is the hard guard, not just the artifact-loading path.
# ---------------------------------------------------------------------------


def test_build_judge_adapter_rejects_a_standalone_seat_with_an_unsafe_tool() -> None:
    seat = CliSeat(
        cli="claude",
        model="sonnet",
        binary="/usr/bin/claude",
        mcp_server_url="http://127.0.0.1:8787/mcp",
        mcp_server_name="tabularium",
        mcp_allowed_tools=("bury",),
    )
    with pytest.raises(ValueError, match="established side-effect-free"):
        build_judge_adapter(seat)


def test_build_judge_adapter_rejects_a_standalone_seat_with_recall() -> None:
    seat = CliSeat(
        cli="claude",
        model="sonnet",
        binary="/usr/bin/claude",
        mcp_server_url="http://127.0.0.1:8787/mcp",
        mcp_server_name="tabularium",
        mcp_allowed_tools=("recall",),
    )
    with pytest.raises(ValueError, match="recall"):
        build_judge_adapter(seat)


def test_build_judge_adapter_rejects_a_standalone_seat_with_an_empty_allowlist() -> None:
    seat = CliSeat(
        cli="claude",
        model="sonnet",
        binary="/usr/bin/claude",
        mcp_server_url="http://127.0.0.1:8787/mcp",
        mcp_server_name="tabularium",
    )
    with pytest.raises(ValueError, match="mcp_allowed_tools is empty"):
        build_judge_adapter(seat)


def test_build_judge_adapter_rejects_a_standalone_seat_with_a_wildcard_tool() -> None:
    seat = CliSeat(
        cli="claude",
        model="sonnet",
        binary="/usr/bin/claude",
        mcp_server_url="http://127.0.0.1:8787/mcp",
        mcp_server_name="tabularium",
        mcp_allowed_tools=("*",),
    )
    with pytest.raises(ValueError, match="bare lowercase identifier"):
        build_judge_adapter(seat)


def test_build_judge_adapter_accepts_a_standalone_seat_with_only_safe_tools() -> None:
    """The happy path still works -- this is a re-validation, not a second, stricter gate."""
    seat = CliSeat(
        cli="claude",
        model="sonnet",
        binary="/usr/bin/claude",
        mcp_server_url="http://127.0.0.1:8787/mcp",
        mcp_server_name="tabularium",
        mcp_allowed_tools=("decisions_query", "decision", "context"),
    )
    adapter = build_judge_adapter(seat)
    assert isinstance(adapter, SubprocessJudgeAdapter)
    assert adapter.grant is not None
    assert adapter.grant.allowed_tools == ("decisions_query", "decision", "context")


async def test_codex_prompt_sent_via_stdin_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors ``test_claude_prompt_sent_via_stdin_pipe`` for the codex seat (2026-08-13 fix)."""
    double = _SubprocessDouble(
        stdout=b"t", returncode=0, output_file_content=_payload([_ok_item(1)]).encode()
    )
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli="codex", binary="/bin/codex", model="gpt-5.1-codex")

    await adapter.call("MY PROMPT")

    call = double.calls[0]
    assert call["kwargs"]["stdin"] is asyncio.subprocess.PIPE
    assert call["proc"].received_input == b"MY PROMPT"


async def test_codex_missing_output_file_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    double = _SubprocessDouble(stdout=b"transcript", returncode=0, output_file_content=None)
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli="codex", binary="/bin/codex", model="gpt-5.1-codex")

    with pytest.raises(JudgeUnavailable):
        await adapter.call("p")


async def test_codex_stderr_diagnostic_survives_a_long_prompt_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the deployed-server defect (fixed 2026-08-11, back when the prompt still rode
    argv):
    codex would echo the whole rendered prompt into stderr before its real error, so a plain head
    truncation of a long stderr blob was pure prompt echo with the actual error line always cut
    off. That specific echo source closed with the 2026-08-13 stdin fix, but the defensive TAIL-
    not-HEAD posture stays for any other large stderr blob — build one well past the 2000-char cap
    whose real error text sits at the very end and confirm it still reaches the raised message —
    this must go RED against a head-based `[:2000]` truncation (verified by hand, reported in the
    task verdict, not re-asserted here since it would require reverting the fix).
    """
    echoed_prompt = "PROMPT ECHO LINE\n" * 200  # >> 2000 chars, mimics the CLI echoing the prompt
    stderr = (echoed_prompt + "ERROR: model not supported for this account\n").encode()
    assert len(stderr) > 2000
    double = _SubprocessDouble(stdout=b"", stderr=stderr, returncode=1)
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli="codex", binary="/bin/codex", model="gpt-5.6-sol")

    with pytest.raises(JudgeUnavailable, match="ERROR: model not supported for this account"):
        await adapter.call("p")


async def test_claude_stderr_diagnostic_survives_a_long_prompt_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same exposure, applied to the claude call site for symmetry: the TAIL-not-HEAD posture is
    generic to any large stderr blob (the prompt no longer rides argv on either seat since the
    2026-08-13 fix, so a long echo-heavy stderr is not codex-only in principle even though it was
    only observed live on codex).
    """
    echoed_prompt = "PROMPT ECHO LINE\n" * 200
    stderr = (echoed_prompt + "ERROR: something actually broke\n").encode()
    assert len(stderr) > 2000
    double = _SubprocessDouble(stdout=b"", stderr=stderr, returncode=1)
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli="claude", binary="/bin/claude", model="sonnet")

    with pytest.raises(JudgeUnavailable, match="ERROR: something actually broke"):
        await adapter.call("p")


async def test_codex_stderr_diagnostic_prefers_tail_over_uppercase_error_substring_in_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for a deployed-server defect: the OLD implementation preferred ANY
    line containing the bare, case-sensitive substring "ERROR" wherever it appeared in stderr —
    codex's own marker, per the 2026-08-11 fix. Once the harness started feeding session digests
    describing this helper's own engineering history into the judge prompt, an echoed digest line
    containing the literal uppercase string "ERROR-lines-first" (naming that very fix) won the
    scan, and the REAL error — a title-case ``Error: ... (code -32602)`` line at the TRUE tail —
    was discarded entirely; the helper returned prose about its own prior bug fix instead of the
    live failure.

    This fixture reproduces that exact shape: a large stderr blob (~15 KB, well past the 2000-char
    cap) that is mostly prompt echo containing an uppercase "ERROR" substring, with the real error
    — the exact text observed on the deployed-server failure — at the tail.

    Falsifiability: this test is RED against the CURRENT (pre-fix) implementation — confirmed by
    hand before the fix landed (`_stderr_diagnostic`'s old `"ERROR" in line` scan matches every
    repetition of `echoed_digest_line` below and NONE of the real error line, so the old
    implementation's returned diagnostic is 2000 chars of pure digest-echo repetition with no
    ``code -32602`` substring anywhere in it — `pytest.raises(match=...)` fails to find the pattern
    and the test errors out rather than raising `JudgeUnavailable` with the expected text).
    """
    echoed_digest_line = (
        "session digest: fixed _stderr_diagnostic to prefer ERROR-lines-first over a head "
        "truncation\n"
    )
    echoed_prompt = echoed_digest_line * 200  # >> 2000 chars, well past the diagnostic cap
    real_error = (
        "Error: turn/start: turn/start failed: Input exceeds the maximum length of 1048576 "
        'characters. (code -32602), data: {"input_error_code":"input_too_large"}'
    )
    stderr = (echoed_prompt + real_error + "\n").encode()
    assert len(stderr) > 2000
    assert "ERROR" in echoed_prompt  # the substring the OLD implementation preferred
    assert "ERROR" not in real_error  # title-case "Error:" only -- excluded by the old scan
    double = _SubprocessDouble(stdout=b"", stderr=stderr, returncode=1)
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli="codex", binary="/bin/codex", model="gpt-5.6-sol")

    with pytest.raises(JudgeUnavailable, match="code -32602"):
        await adapter.call("p")


async def test_claude_empty_stderr_reports_stdout_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    double = _SubprocessDouble(stdout=b"Prompt is too long", stderr=b"", returncode=1)
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli="claude", binary="/bin/claude", model="sonnet")

    with pytest.raises(JudgeUnavailable, match="Prompt is too long"):
        await adapter.call("p")


async def test_codex_empty_stderr_reports_stdout_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    double = _SubprocessDouble(stdout=b"Prompt is too long", stderr=b"  \n", returncode=1)
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli="codex", binary="/bin/codex", model="gpt-5.1-codex")

    with pytest.raises(JudgeUnavailable, match=r"\(stdout\) Prompt is too long"):
        await adapter.call("p")


@pytest.mark.parametrize("cli", ["claude", "codex"])
async def test_nonblank_stderr_wins_and_stdout_stays_out_of_the_message(
    monkeypatch: pytest.MonkeyPatch, cli: str
) -> None:
    double = _SubprocessDouble(
        stdout=b"STDOUT TRANSCRIPT TEXT", stderr=b"real stderr failure", returncode=1
    )
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli=cli, binary=f"/bin/{cli}", model="m")  # type: ignore[arg-type]

    with pytest.raises(JudgeUnavailable) as excinfo:
        await adapter.call("p")

    assert "real stderr failure" in str(excinfo.value)
    assert "STDOUT TRANSCRIPT TEXT" not in str(excinfo.value)


async def test_codex_nonzero_exit_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    double = _SubprocessDouble(
        stdout=b"",
        stderr=b"Not inside a trusted directory and --skip-git-repo-check was not specified.",
        returncode=1,
    )
    _patch_subprocess_exec(monkeypatch, double)
    adapter = SubprocessJudgeAdapter(cli="codex", binary="/bin/codex", model="gpt-5.1-codex")

    with pytest.raises(JudgeUnavailable):
        await adapter.call("p")


# ---------------------------------------------------------------------------
# The MAX_ARG_STRLEN regression itself -- the one pair of tests in this file that spawns a REAL
# subprocess rather than a monkeypatched double. Not the credentialed claude/codex CLIs (the
# module docstring's "no test spawns a real claude/codex process" still holds): a trivial local
# script stands in for the binary, so this stays fully hermetic -- no network, no credential.
# ---------------------------------------------------------------------------


async def test_prompt_far_larger_than_max_arg_strlen_reaches_claude_seat_via_stdin(
    tmp_path: Path,
) -> None:
    """Regression for the 2026-08-13 defect: a live revisit-batch prompt (246k tokens, ~1 MB
    rendered) was passed as a single argv element and blew Linux's ``MAX_ARG_STRLEN`` (128 KiB
    per argv element, enforced inside ``execve(2)`` before any Python-level code runs) -- both
    judge seats failed identically with ``[Errno 7] Argument list too long``. Exercises the REAL
    size regime (well past 128 KiB), not a token count standing in for one. The stand-in script
    (``cat``) echoes stdin straight back to stdout; ``SubprocessJudgeAdapter.call`` returns that
    unparsed, so an exact round-trip proves the huge prompt reached the process intact, and only
    reachable via stdin -- an argv-based invocation of this same size raises ``OSError: [Errno 7]
    Argument list too long`` before the process ever starts (verified by hand against the pre-fix
    argv-based invocation; the exact command and its ``OSError`` are recorded in the task verdict,
    not re-asserted here since it would require reverting the fix).
    """
    script = tmp_path / "echo_stdin.sh"
    script.write_text("#!/bin/sh\ncat\n")
    script.chmod(0o755)

    huge_prompt = "A" * (200 * 1024)  # 200 KiB, comfortably past the 128 KiB MAX_ARG_STRLEN cap
    assert len(huge_prompt.encode()) > 131072

    adapter = SubprocessJudgeAdapter(cli="claude", binary=str(script), model="sonnet")
    raw = await adapter.call(huge_prompt)

    assert raw == huge_prompt


async def test_prompt_far_larger_than_max_arg_strlen_reaches_codex_seat_via_stdin(
    tmp_path: Path,
) -> None:
    """Codex-seat mirror of the claude-seat regression above. The stand-in script reads its own
    ``--output-last-message <path>`` argument, then copies stdin into that file the way the real
    codex CLI writes its final message there -- proving the codex call path also survives a
    prompt far past ``MAX_ARG_STRLEN``.
    """
    script = tmp_path / "echo_stdin_to_output_file.sh"
    script.write_text(
        "#!/bin/sh\n"
        "outfile=\n"
        "while [ $# -gt 0 ]; do\n"
        '  if [ "$1" = "--output-last-message" ]; then\n'
        '    outfile="$2"\n'
        "  fi\n"
        "  shift\n"
        "done\n"
        'cat > "$outfile"\n'
    )
    script.chmod(0o755)

    huge_prompt = "B" * (200 * 1024)
    assert len(huge_prompt.encode()) > 131072

    adapter = SubprocessJudgeAdapter(cli="codex", binary=str(script), model="gpt-5.1-codex")
    raw = await adapter.call(huge_prompt)

    assert raw == huge_prompt


# ---------------------------------------------------------------------------
# LitellmJudgeAdapter mechanics -- litellm.acompletion is monkeypatched, so no network call
# and no DEEPSEEK_API_KEY is ever needed here.
# ---------------------------------------------------------------------------


def _model_response(content: str | None, *, model: str) -> ModelResponse:
    msg = Message(content=content, role="assistant")
    choice = Choices(message=msg, index=0, finish_reason="stop")
    return ModelResponse(id="x", created=0, model=model, object="chat.completion", choices=[choice])


async def test_litellm_passes_explicit_model_and_returns_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> ModelResponse:
        captured.update(kwargs)
        return _model_response(_payload([_ok_item(1)]), model=str(kwargs["model"]))

    monkeypatch.setattr("tabularium.engine.judge.litellm.acompletion", fake_acompletion)
    adapter = LitellmJudgeAdapter(model="deepseek/deepseek-chat")

    raw = await adapter.call("PROMPT")

    assert captured["model"] == "deepseek/deepseek-chat"
    assert raw == _payload([_ok_item(1)])


async def test_litellm_exception_maps_to_unavailable_not_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance criterion: an auth failure (401 OAuth access token has expired) is
    unavailable, not invalid_output -- also true for the litellm adapter."""

    async def fake_acompletion(**kwargs: Any) -> ModelResponse:
        raise RuntimeError("401 OAuth access token has expired")

    monkeypatch.setattr("tabularium.engine.judge.litellm.acompletion", fake_acompletion)
    adapter = LitellmJudgeAdapter(model="deepseek/deepseek-chat")

    with pytest.raises(JudgeUnavailable, match="401 OAuth access token has expired"):
        await adapter.call("p")


async def test_litellm_empty_content_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**kwargs: Any) -> ModelResponse:
        return _model_response(None, model=str(kwargs["model"]))

    monkeypatch.setattr("tabularium.engine.judge.litellm.acompletion", fake_acompletion)
    adapter = LitellmJudgeAdapter(model="deepseek/deepseek-chat")

    with pytest.raises(JudgeUnavailable):
        await adapter.call("p")


# ---------------------------------------------------------------------------
# Static guard: nothing in this module writes an unparsed response to disk and returns a
# non-failing status (Verify step 2). Mirrors test_rubric.py's own source-scan convention.
# ---------------------------------------------------------------------------


async def test_judge_batch_stamps_instruction_fingerprint_from_the_adapter() -> None:
    """Owner: `judge_batch` reads `adapter.instruction_fingerprint()` and stamps
    it on the outcome on every one of the four paths, alongside `rubric_version`."""
    adapter = FakeJudgeAdapter(
        identity="fake:v1",
        response=_payload([_ok_item(1)]),
        instruction_fingerprint="sha256:" + "cd" * 32,
    )
    outcome = await judge_batch(
        adapter,
        prompt="p",
        item_model=_GradedVerdict,
        batch_size=1,
        rubric_version="rv",
        token_counter=FakeTokenCounter(),
    )
    assert outcome.status == "success"
    assert outcome.instruction_fingerprint == "sha256:" + "cd" * 32

    record = outcome.as_call_record(batch_index=0)
    assert record.instruction_fingerprint == "sha256:" + "cd" * 32


async def test_judge_batch_stamps_instruction_fingerprint_even_on_a_failed_call() -> None:
    """Present on every status, failures included -- mirrors `rubric_version`'s own contract."""
    adapter = FakeJudgeAdapter(should_fail=True, instruction_fingerprint="absent")
    outcome = await judge_batch(
        adapter,
        prompt="p",
        item_model=_GradedVerdict,
        batch_size=1,
        rubric_version="rv",
        token_counter=FakeTokenCounter(),
    )
    assert outcome.status == "judge_unavailable"
    assert outcome.instruction_fingerprint == "absent"


async def test_judge_batch_never_fails_when_instruction_fingerprint_raises() -> None:
    """Cheap and non-fatal (task brief): if `adapter.instruction_fingerprint()` itself raises
    something `judge_batch` never anticipated, the judge call still completes and reports a
    status -- provenance collection must never break a judge call."""

    class _BoomingFingerprintAdapter:
        identity = "fake:boom"

        def instruction_fingerprint(self) -> str:
            raise RuntimeError("simulated: instruction fingerprint collection blew up")

        async def call(self, prompt: str) -> str:
            return _payload([_ok_item(1)])

    outcome = await judge_batch(
        _BoomingFingerprintAdapter(),
        prompt="p",
        item_model=_GradedVerdict,
        batch_size=1,
        rubric_version="rv",
        token_counter=FakeTokenCounter(),
    )
    assert outcome.status == "success"  # the actual judge call still went through
    assert outcome.instruction_fingerprint == "absent"  # degraded, not propagated


def test_judge_call_record_defaults_instruction_fingerprint_for_pre_existing_json() -> None:
    """Backward compatibility: a `JudgeCallRecord` built the way `test_judge_call_record_is_frozen`
    already does (no `instruction_fingerprint` kwarg at all) still constructs -- the same shape a
    pre-this-task persisted `calls` entry has when read back off a real store."""
    record = JudgeCallRecord(
        batch_index=0,
        judge="a",
        rubric_version="v1",
        duration_seconds=0.1,
        status="success",
        error=None,
        prompt_tokens=1,
        completion_tokens=None,
        cost=None,
    )
    assert record.instruction_fingerprint == "unrecorded"


# ---------------------------------------------------------------------------
# Instruction-file fingerprinting -- SubprocessJudgeAdapter.
# instruction_fingerprint(), and the chain-resolution helpers it calls for each seat. No real
# claude/codex process is ever spawned here -- every test drives instruction_fingerprint()
# directly against tmp_path fixtures, same hermetic posture as the rest of this file. The chain
# geometry each helper implements (root-to-cwd order for claude with no git-boundary stop; a
# git-repo-root-bounded walk for codex) was verified against the REAL CLIs locally (no live
# store) -- see
# `_claude_instruction_chain`/`_codex_instruction_chain`'s own docstrings in
# `engine/judge.py` for the exact scenarios and observed output.
# ---------------------------------------------------------------------------


def _claude_adapter() -> SubprocessJudgeAdapter:
    return SubprocessJudgeAdapter(cli="claude", binary="/bin/claude", model="sonnet")


def _codex_adapter() -> SubprocessJudgeAdapter:
    return SubprocessJudgeAdapter(cli="codex", binary="/bin/codex", model="gpt-5.1-codex")


class TestInstructionFingerprintSingleFilePresent:
    def test_claude_single_file_yields_a_sha256_fingerprint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "CLAUDE.md").write_text("be terse")
        monkeypatch.chdir(tmp_path)

        fingerprint = _claude_adapter().instruction_fingerprint()

        assert fingerprint.startswith("sha256:")
        assert len(fingerprint) == len("sha256:") + 64  # a real hex digest, not a placeholder

    def test_codex_single_file_yields_a_sha256_fingerprint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cwd IS the git repo root -- the chain collapses to exactly one candidate file."""
        (tmp_path / ".git").mkdir()
        (tmp_path / "AGENTS.md").write_text("strict JSON only")
        monkeypatch.chdir(tmp_path)

        fingerprint = _codex_adapter().instruction_fingerprint()

        assert fingerprint.startswith("sha256:")

    def test_fingerprint_is_deterministic_for_the_same_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "CLAUDE.md").write_text("stable content")
        monkeypatch.chdir(tmp_path)

        first = _claude_adapter().instruction_fingerprint()
        second = _claude_adapter().instruction_fingerprint()
        assert first == second

    def test_fingerprint_changes_when_content_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "CLAUDE.md"
        target.write_text("version one")
        monkeypatch.chdir(tmp_path)
        before = _claude_adapter().instruction_fingerprint()

        target.write_text("version two")
        after = _claude_adapter().instruction_fingerprint()

        assert before != after


class TestInstructionFingerprintAbsence:
    """Absence is a distinct value, not an empty hash (task brief) -- 'no instruction file
    present' is the exact condition that made every early Cleaner pass come back `blocked`."""

    def test_claude_no_file_anywhere_in_the_chain_is_the_absent_sentinel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        fingerprint = _claude_adapter().instruction_fingerprint()

        assert fingerprint == "absent"
        assert "sha256" not in fingerprint

    def test_codex_no_file_and_no_git_repo_is_the_absent_sentinel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        assert _codex_adapter().instruction_fingerprint() == "absent"

    def test_codex_git_repo_present_but_no_agents_md_is_the_absent_sentinel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / ".git").mkdir()
        monkeypatch.chdir(tmp_path)

        assert _codex_adapter().instruction_fingerprint() == "absent"

    def test_absent_is_not_the_hash_of_an_empty_string(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import hashlib

        monkeypatch.chdir(tmp_path)
        fingerprint = _claude_adapter().instruction_fingerprint()

        assert fingerprint != f"sha256:{hashlib.sha256(b'').hexdigest()}"
        assert fingerprint == "absent"


class TestInstructionFingerprintMultiFileChainOrderSignificance:
    """Order matters -- hashed in the order the CLI would apply them (task brief). Isolated at the
    `_fingerprint_chain` level: the SAME two files (identical paths, identical content) fed in
    reversed order must hash differently, proving order alone (not content, not path identity) is
    what changed."""

    def test_fingerprint_chain_reversed_order_differs(self, tmp_path: Path) -> None:
        from tabularium.engine.judge import _fingerprint_chain

        root_file = tmp_path / "CLAUDE.md"
        root_file.write_text("root content")
        leaf_dir = tmp_path / "leaf"
        leaf_dir.mkdir()
        leaf_file = leaf_dir / "CLAUDE.md"
        leaf_file.write_text("leaf content")

        root_to_leaf = _fingerprint_chain([root_file, leaf_file])
        leaf_to_root = _fingerprint_chain([leaf_file, root_file])

        assert root_to_leaf.startswith("sha256:")
        assert leaf_to_root.startswith("sha256:")
        assert root_to_leaf != leaf_to_root

    def test_claude_instruction_chain_orders_root_to_cwd(self, tmp_path: Path) -> None:
        """`_claude_instruction_chain` itself -- verified live against the real CLI (see its
        docstring): a marker planted at the root surfaces to the model BEFORE one planted at cwd.
        This test pins that the candidate PATH list this engine builds is ordered the same way."""
        from tabularium.engine.judge import _claude_instruction_chain

        mid = tmp_path / "mid"
        leaf = mid / "leaf"
        leaf.mkdir(parents=True)

        chain = _claude_instruction_chain(leaf)

        assert chain[-1] == leaf / "CLAUDE.md"  # cwd's own file is LAST
        assert chain[-2] == mid / "CLAUDE.md"
        assert chain[-3] == tmp_path / "CLAUDE.md"
        # root-to-cwd: shallower directories appear strictly before deeper ones.
        assert list(chain) == sorted(chain, key=lambda p: len(p.parts))

    def test_claude_adapter_multi_file_chain_both_levels_contribute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Integration check at the adapter level: both a root-level and a cwd-level CLAUDE.md
        contribute to the fingerprint -- removing either one changes the result."""
        leaf = tmp_path / "leaf"
        leaf.mkdir()
        (tmp_path / "CLAUDE.md").write_text("root instructions")
        (leaf / "CLAUDE.md").write_text("leaf instructions")
        monkeypatch.chdir(leaf)

        both_present = _claude_adapter().instruction_fingerprint()

        (leaf / "CLAUDE.md").unlink()
        root_only = _claude_adapter().instruction_fingerprint()

        assert both_present != root_only
        assert both_present.startswith("sha256:")
        assert root_only.startswith("sha256:")

    def test_codex_stays_within_the_git_repo_and_composes_root_plus_subdir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verified live (see `_codex_instruction_chain`'s docstring, scenarios 3-4): codex walks
        up to the nearest enclosing git repo root and no further. An outer, unrelated repo's
        AGENTS.md must NOT contribute -- proven here by comparing against the fingerprint computed
        with the outer file removed, which must be identical."""
        outer = tmp_path / "outer"
        inner = outer / "inner"
        sub = inner / "sub"
        sub.mkdir(parents=True)
        (outer / ".git").mkdir()
        (outer / "AGENTS.md").write_text("outer, unrelated repo -- must not be read")
        (inner / ".git").mkdir()
        (inner / "AGENTS.md").write_text("inner repo root instructions")
        monkeypatch.chdir(sub)

        with_outer_present = _codex_adapter().instruction_fingerprint()

        (outer / "AGENTS.md").unlink()
        without_outer = _codex_adapter().instruction_fingerprint()

        assert with_outer_present == without_outer  # the outer file never contributed
        assert with_outer_present.startswith("sha256:")


class TestInstructionFingerprintUnreadableFile:
    """Falsifiability bar: this degrades via `_fingerprint_chain`'s own `except OSError` clause
    (`engine/judge.py`) -- commenting that clause out and re-running this test raises
    `PermissionError` straight out of `instruction_fingerprint()` instead of returning a sentinel
    (verified by hand for this task's verdict, not re-asserted here since it would require
    reverting the guard)."""

    def test_unreadable_file_degrades_to_the_absent_sentinel_without_raising(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "CLAUDE.md"
        target.write_text("normally readable")
        monkeypatch.chdir(tmp_path)

        original_read_bytes = Path.read_bytes

        def _boom(self: Path, *a: object, **kw: object) -> bytes:
            if self == target:
                raise PermissionError(f"simulated: cannot read {self}")
            return original_read_bytes(self, *a, **kw)

        monkeypatch.setattr(Path, "read_bytes", _boom)

        fingerprint = _claude_adapter().instruction_fingerprint()  # must not raise

        assert fingerprint == "absent"

    def test_one_unreadable_file_does_not_block_a_readable_sibling_in_the_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unreadable file degrades to contributing nothing, not to failing the whole chain --
        a readable file elsewhere in the same chain still produces a real fingerprint, never
        'absent'."""
        leaf = tmp_path / "leaf"
        leaf.mkdir()
        unreadable_root = tmp_path / "CLAUDE.md"
        unreadable_root.write_text("root, will be made unreadable")
        (leaf / "CLAUDE.md").write_text("leaf, stays readable")
        monkeypatch.chdir(leaf)

        original_read_bytes = Path.read_bytes

        def _boom(self: Path, *a: object, **kw: object) -> bytes:
            if self == unreadable_root:
                raise PermissionError(f"simulated: cannot read {self}")
            return original_read_bytes(self, *a, **kw)

        monkeypatch.setattr(Path, "read_bytes", _boom)

        fingerprint = _claude_adapter().instruction_fingerprint()

        assert fingerprint.startswith("sha256:")  # the leaf file still contributed
        assert fingerprint != "absent"


class TestInstructionFingerprintOtherAdapters:
    """Every `JudgeAdapter` implementor carries `instruction_fingerprint()` -- the litellm backstop
    and the fake both return the `not_applicable` sentinel by default, distinct from `absent`."""

    def test_litellm_adapter_is_not_applicable(self) -> None:
        adapter = LitellmJudgeAdapter(model="deepseek/deepseek-chat")
        assert adapter.instruction_fingerprint() == "not_applicable"

    def test_fake_adapter_defaults_to_not_applicable(self) -> None:
        adapter = FakeJudgeAdapter()
        assert adapter.instruction_fingerprint() == "not_applicable"

    def test_fake_adapter_is_configurable(self) -> None:
        adapter = FakeJudgeAdapter(instruction_fingerprint="sha256:" + "11" * 32)
        assert adapter.instruction_fingerprint() == "sha256:" + "11" * 32


def test_no_write_call_anywhere_in_the_judge_module() -> None:
    """``engine/judge.py`` contains no file-write call at all: the codex adapter only ever
    READS the file the CLI itself populates via ``--output-last-message`` (inside a
    ``TemporaryDirectory`` that is always cleaned up), and that read happens unconditionally,
    before any parsing -- never gated behind a status check that would let a bad response
    through labelled "ok".
    """
    source = (_REPO_ROOT / "src" / "tabularium" / "engine" / "judge.py").read_text()
    assert ".write_text(" not in source
    assert ".write_bytes(" not in source
    assert "open(" not in source
    assert ".write(" not in source


# ---------------------------------------------------------------------------
# Bounded prompts: the packer, the ceiling guard, prompt_chars
# ---------------------------------------------------------------------------


def _pack(contents: list[str], *, count_cap: int, ceiling: int) -> list[list[int]]:
    return pack_batches(
        contents,
        count_cap=count_cap,
        rubric_prose="RUBRIC",
        item_model=_GradedVerdict,
        ceiling=ceiling,
    )


def _rendered_size(contents: list[str], group: list[int]) -> int:
    return len(
        render_batch_prompt(
            rubric_prose="RUBRIC",
            contents=[contents[i] for i in group],
            item_model=_GradedVerdict,
        )
    )


def test_packer_splits_units_whose_sum_passes_the_ceiling() -> None:
    contents = ["x" * 1_500 for _ in range(10)]
    empty = render_batch_prompt(rubric_prose="RUBRIC", contents=[], item_model=_GradedVerdict)
    ceiling = len(empty) + 4_000

    groups = _pack(contents, count_cap=100, ceiling=ceiling)

    assert len(groups) > 1
    assert [i for g in groups for i in g] == list(range(10))
    for group in groups:
        assert _rendered_size(contents, group) <= ceiling


def test_packer_honours_the_count_cap() -> None:
    contents = ["tiny"] * 7

    groups = _pack(contents, count_cap=3, ceiling=600_000)

    assert groups == [[0, 1, 2], [3, 4, 5], [6]]


def test_packer_puts_an_oversize_unit_alone() -> None:
    contents = ["small", "y" * 50_000, "small too"]

    groups = _pack(contents, count_cap=10, ceiling=10_000)

    assert groups == [[0], [1], [2]]


def test_packer_output_always_renders_within_the_ceiling() -> None:
    sizes = [100, 2_900, 4_000, 150, 3_333, 70, 5_000, 1_234, 999, 3_001, 2_222]
    contents = [("z" * n) for n in sizes]
    ceiling = 9_000

    groups = _pack(contents, count_cap=100, ceiling=ceiling)

    for group in groups:
        assert _rendered_size(contents, group) <= ceiling


async def test_judge_batch_over_the_ceiling_never_calls_the_adapter() -> None:
    adapter = FakeJudgeAdapter(identity="fake:v1", response=_payload([_ok_item(1)]))
    prompt = "p" * (PROMPT_CHAR_CEILING + 1)

    outcome = await judge_batch(
        adapter,
        prompt=prompt,
        item_model=_GradedVerdict,
        batch_size=1,
        rubric_version="rv-1",
        token_counter=FakeTokenCounter(),
    )

    assert outcome.status == "prompt_too_large"
    assert adapter.calls == []
    assert outcome.error is not None
    assert str(len(prompt)) in outcome.error
    assert str(PROMPT_CHAR_CEILING) in outcome.error
    assert outcome.prompt_chars == len(prompt)
    assert outcome.as_call_record(batch_index=0).prompt_chars == len(prompt)


async def test_judge_batch_records_prompt_chars_on_success() -> None:
    adapter = FakeJudgeAdapter(identity="fake:v1", response=_payload([_ok_item(1)]))

    outcome = await judge_batch(
        adapter,
        prompt="PROMPT",
        item_model=_GradedVerdict,
        batch_size=1,
        rubric_version="rv-1",
        token_counter=FakeTokenCounter(),
    )

    assert outcome.status == "success"
    assert outcome.prompt_chars == len("PROMPT")
    assert outcome.as_call_record(batch_index=0).prompt_chars == len("PROMPT")


def test_an_old_call_record_without_prompt_chars_still_loads() -> None:
    record = JudgeCallRecord.model_validate(
        {
            "batch_index": 0,
            "judge": "j",
            "rubric_version": "v",
            "duration_seconds": 0.1,
            "status": "success",
            "error": None,
            "prompt_tokens": 1,
            "completion_tokens": None,
            "cost": None,
        }
    )

    assert record.prompt_chars == 0


@pytest.mark.parametrize("model", [RelevanceVerdict, RevisitVerdict])
def test_shelve_axes_render_the_optional_keep_field(model: type[IndexedVerdictBase]) -> None:
    section = _section(render_batch_prompt(rubric_prose="P", contents=["a"], item_model=model))

    assert "`keep` (optional): a JSON string" in section
    assert "never put it on a `keep-*` grade" in section


def test_falsity_renders_the_optional_carry_field() -> None:
    section = _section(
        render_batch_prompt(rubric_prose="P", contents=["a"], item_model=FalsityVerdict)
    )

    assert "`carry` (optional): a JSON string" in section
    assert "`carry` goes only when `verdict` is `replace`" in section


@pytest.mark.parametrize("model", [RelevanceVerdict, RevisitVerdict])
@pytest.mark.parametrize(
    ("grade", "keep", "expected"),
    [
        ("shelve-clear", "A durable rule.", "A durable rule."),
        ("shelve-lean", "  padded  ", "padded"),
        ("shelve-lean", "   ", None),
        ("shelve-lean", "", None),
        ("shelve-clear", None, None),
        ("keep-lean", "A durable rule.", None),
        ("keep-clear", "A durable rule.", None),
    ],
)
def test_keep_is_normalized_never_rejected(
    model: type[RelevanceVerdict | RevisitVerdict],
    grade: str,
    keep: str | None,
    expected: str | None,
) -> None:
    entry = {"index": 1, "grade": grade, "why": "w", "unclear": False, "keep": keep}

    parsed = parse_batch_verdicts(json.dumps({"verdicts": [entry]}), item_model=model, batch_size=1)

    assert parsed[0].keep == expected


def test_keep_defaults_to_none_when_absent() -> None:
    entry = {"index": 1, "grade": "shelve-clear", "why": "w", "unclear": False}

    parsed = parse_batch_verdicts(
        json.dumps({"verdicts": [entry]}), item_model=RelevanceVerdict, batch_size=1
    )

    assert parsed[0].keep is None


@pytest.mark.parametrize(
    ("verdict", "reason", "carry", "expected"),
    [
        ("replace", "changed", "Still true.", "Still true."),
        ("replace", "changed", "   ", None),
        ("replace", "correction", None, None),
        ("coexist", None, "Still true.", None),
    ],
)
def test_carry_is_normalized_never_rejected(
    verdict: str, reason: str | None, carry: str | None, expected: str | None
) -> None:
    entry: dict[str, Any] = {"index": 1, "verdict": verdict, "why": "w", "carry": carry}
    if reason is not None:
        entry["reason"] = reason

    parsed = parse_batch_verdicts(
        json.dumps({"verdicts": [entry]}), item_model=FalsityVerdict, batch_size=1
    )

    assert parsed[0].carry == expected
