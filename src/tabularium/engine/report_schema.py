"""The persisted schema of a Cleaner pass report: the models whose JSON is written into
``CleanerState.report`` and the pass history's payload column.

Field names, order and defaults are the stored key order and what lets older rows parse, so a change
here changes stored bytes. ``engine/report.py`` re-exports every model, builds them in
``build_report`` and renders them for ``context()``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from tabularium.domain import BatchOutcome, HypomnemaId, Scope
from tabularium.engine.axis_common import (
    BandCounts,
    ContestedVerdict,
    JudgmentAppendFailure,
    RefusedVerdict,
)
from tabularium.engine.cleaner_writes import MergedTriple, MergeRevertedEntry, TrimmedPair
from tabularium.engine.falsity import (
    FalsityBandCounts,
    FalsityContestedPair,
    FalsityRefusedPair,
    FalsityReportEntry,
)
from tabularium.engine.judge import JudgeCallRecord
from tabularium.engine.revisit import RevisitReviewAppendFailure


class CleanerBatchFailureReport(BaseModel):
    """One batch whose judge call failed, flattened from an axis's ``*BatchFailure`` and tagged with
    its axis. ``judge_a_outcome`` and ``judge_b_outcome`` say whether each judge was unavailable or
    returned an unparseable response.
    """

    model_config = {"frozen": True}

    axis: Literal["relevance", "falsity", "revisit"]
    batch_index: int
    record_ids: tuple[HypomnemaId, ...]
    judge_a_outcome: BatchOutcome
    judge_b_outcome: BatchOutcome
    error: str


class RelevanceReportSection(BaseModel):
    """The relevance axis's slice of one report. ``judge_pair`` and ``rubric_version`` are stamped
    on each section.

    ``counts.shelve`` counts every shelve that stands, split shelves included (``split_shelve`` is a
    sub-count). Rows persisted before that rule count unanimous shelves only and are not rewritten.

    Fields mirror ``RelevanceAxisSummary``. ``calls`` is one entry per judge call, both seats, every
    batch, success or failure. ``contested_verdicts`` carries both judges' identity, grade and
    ``why`` for the ids in ``contested_ids``. ``refused`` holds shelve votes the store's write guard
    refused. ``judgment_append_failures`` holds contested records whose ``RecordJudgment`` append
    failed; the contested verdict itself stands. Every field with a default has one so rows
    persisted without it still parse.
    """

    model_config = {"frozen": True}

    judge_pair: tuple[str, str]
    rubric_version: str
    counts: BandCounts
    shelved_ids: tuple[HypomnemaId, ...]
    unclear_ids: tuple[HypomnemaId, ...]
    contested_ids: tuple[HypomnemaId, ...] = ()
    contested_verdicts: tuple[ContestedVerdict, ...] = ()
    failures: tuple[CleanerBatchFailureReport, ...]
    calls: tuple[JudgeCallRecord, ...] = ()
    refused: tuple[RefusedVerdict, ...] = ()
    judgment_append_failures: tuple[JudgmentAppendFailure, ...] = ()
    # Mirrored from `RelevanceAxisSummary`; see its docstring for how these counts reconcile.
    skipped: int = 0
    split_keep: int = 0
    split_annotated: int = 0
    split_shelve: int = 0
    # Mirrored from `RelevanceAxisSummary.trimmed`: (shelved id, id of the Fact saved from its
    # `keep` text).
    trimmed: tuple[TrimmedPair, ...] = ()


class FalsityReportSection(BaseModel):
    """The falsity axis's slice of one report, mirroring ``RelevanceReportSection``.

    ``counts.replace``, ``split_supersede`` and ``reason_by_assessor`` count the supersede edges
    that stand, so a merge revert decrements them. Rows persisted before that rule still count a
    reverted merge in ``replace`` and are not rewritten.

    ``contested`` holds bare ``(newcomer_id, candidate_id)`` pairs and ``contested_verdicts`` the
    per-pair detail for the same pairs. ``refused`` holds replace votes the store's write guard
    refused. ``judgment_append_failures`` holds contested-pair participants whose ``RecordJudgment``
    append failed. ``split_supersede``, ``split_coexist`` and ``reason_by_assessor`` are already
    folded into ``counts.replace`` and ``counts.coexist``, not a second tally. Every field with a
    default has one so rows persisted without it still parse.
    """

    model_config = {"frozen": True}

    judge_pair: tuple[str, str]
    rubric_version: str
    counts: FalsityBandCounts
    superseded_ids: tuple[HypomnemaId, ...]
    global_contradicted: tuple[FalsityReportEntry, ...]
    decision_contradicted: tuple[FalsityReportEntry, ...]
    contested: tuple[tuple[HypomnemaId, HypomnemaId], ...] = ()
    contested_verdicts: tuple[FalsityContestedPair, ...] = ()
    failures: tuple[CleanerBatchFailureReport, ...]
    calls: tuple[JudgeCallRecord, ...] = ()
    refused: tuple[FalsityRefusedPair, ...] = ()
    judgment_append_failures: tuple[JudgmentAppendFailure, ...] = ()
    split_supersede: int = 0
    split_coexist: int = 0
    reason_by_assessor: int = 0
    # Mirrored from `FalsityAxisSummary`. `merged` and `merge_revert_failed` (the undo failed, the
    # edge stands) are subsets of `counts.replace`; `merge_reverted` (the edge was undone) is not.
    merged: tuple[MergedTriple, ...] = ()
    merge_reverted: tuple[MergeRevertedEntry, ...] = ()
    merge_revert_failed: tuple[MergeRevertedEntry, ...] = ()


class RevisitReportSection(BaseModel):
    """The revisit axis's slice of one report, mirroring ``RelevanceReportSection``.

    Isolated in meaning: nothing here feeds ``CleanerReport.outcome``, ``verification_failures`` or
    the birth axes' watermark. ``protected_count`` and the ``skipped_*`` fields are counts, not id
    lists. The section is optional on ``CleanerReport``: a pass whose revisit axis raised before
    returning a summary carries ``revisit_error`` instead of a fabricated section.
    """

    model_config = {"frozen": True}

    judge_pair: tuple[str, str]
    rubric_version: str
    counts: BandCounts
    shelved_ids: tuple[HypomnemaId, ...]
    unclear_ids: tuple[HypomnemaId, ...]
    contested_ids: tuple[HypomnemaId, ...] = ()
    contested_verdicts: tuple[ContestedVerdict, ...] = ()
    protected_count: int = 0
    skipped_no_evidence_count: int = 0
    skipped_too_large_count: int = 0
    failures: tuple[CleanerBatchFailureReport, ...] = ()
    calls: tuple[JudgeCallRecord, ...] = ()
    verification_failures: tuple[HypomnemaId, ...] = ()
    # Mirrored from `RevisitAxisSummary.review_append_failures`: the write landed, only its audit
    # `RecordReview` row failed to append.
    review_append_failures: tuple[RevisitReviewAppendFailure, ...] = ()
    # Mirrored from `RevisitAxisSummary.refused`: shelve votes `set_shelved`'s legality guard
    # refused.
    refused: tuple[RefusedVerdict, ...] = ()
    # Mirrored from `RevisitAxisSummary.judgment_append_failures`: contested records whose
    # `RecordJudgment` append failed.
    judgment_append_failures: tuple[JudgmentAppendFailure, ...] = ()
    pool_size: int = 0
    split_keep: int = 0
    # Mirrored from `RevisitAxisSummary.trimmed`, a subset of `counts.shelve`.
    trimmed: tuple[TrimmedPair, ...] = ()


class CleanerReport(BaseModel):
    """One Cleaner pass, persisted: the JSON written to ``CleanerState.report`` and, as the payload,
    to the pass history.

    Carries ``pass_at``, ``outcome``, the relevance and falsity sections (judge pair, rubric
    version, band counts, ids written), and what needs the owner rather than the panel:
    ``unclear_ids``, contested records and pairs, ``global_contradicted`` and
    ``decision_contradicted`` (detected, never acted on), batch failures, and
    ``verification_failures`` (a write that did not land, non-empty only on ``"blocked"``).

    Each section's ``calls`` is the full per-call telemetry. ``render_context_summary`` never reads
    it.

    ``outcome`` describes the birth axes alone. ``revisit`` and ``revisit_error`` default to
    ``None`` so older rows parse, and neither is read to decide ``outcome``.
    """

    model_config = {"frozen": True}

    # Discriminator against `CleanerFailureReport`, read by `_render_judged`. Defaulted so rows
    # persisted without a `kind` key still parse.
    kind: Literal["report"] = "report"
    scope: Scope
    pass_at: datetime
    outcome: Literal["completed", "blocked"]
    relevance: RelevanceReportSection
    falsity: FalsityReportSection
    revisit: RevisitReportSection | None = None
    revisit_error: str | None = None
    verification_failures: tuple[HypomnemaId, ...]


class CleanerFailureReport(BaseModel):
    """Persisted by ``record_pass_failure`` when a pass raises outside ``clean()``'s own handling.

    A separate model, not a third ``outcome`` on ``CleanerReport``, because a crash before the axes
    run has no sections to report. ``kind`` is the discriminator. Carries ``str(error)`` only; the
    traceback goes to the ERROR log line.
    """

    model_config = {"frozen": True}

    kind: Literal["failure"] = "failure"
    scope: Scope
    pass_at: datetime
    error: str
