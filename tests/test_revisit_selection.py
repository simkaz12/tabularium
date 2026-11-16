"""The Cleaner's revisit axis — selection rules and the history block (2026-08-12
cleaner-revisit-pass plan, scope 3).

Hermetic, pure-function lane only — no adapter, no judge, no I/O anywhere in this file. Every
function under test (``select_revisit_pool``, ``assemble_history_block``, ``next_review_due_at``)
is synchronous and takes its inputs as plain Python values.

Falsifiability (step 6 of the task — both breaks target PRODUCTION code in
``src/tabularium/engine/revisit_pool.py``):

1. Protected-record rule: change ``is_protected`` to always ``return False`` and re-run
   ``TestProtectedRule::test_active_fact_with_two_shelved_events_is_protected``. It goes red
   because the previously-shelved-then-unshelved Fact is no longer removed from the pool via the
   ``protected`` bucket — it shows up in ``candidates`` instead.
2. Skipped-no-evidence rule: change ``select_revisit_pool``'s
   ``if not later_sessions and not events_since_birth:`` guard to ``if False:`` (never skip) and
   re-run ``TestNoEvidenceRule::test_no_later_session_and_no_event_is_skipped``. It goes red
   because the record with zero evidence beyond its own birth digest now appears in
   ``candidates`` instead of ``skipped_no_evidence``.

Both commands and their captured red output are recorded verbatim in the task verdict, not
duplicated here (matching the sibling axis test files' own convention).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tabularium.domain import Decision, Fact, Scope, Session, new_id
from tabularium.engine.review_schedule import next_review_due_at
from tabularium.engine.revisit_pool import (
    EXCERPT_CAP_CHARS,
    HISTORY_BUDGET_CHARS,
    assemble_history_block,
    find_birth_session,
    find_later_sessions,
    is_protected,
    select_revisit_pool,
    session_mentions_subject,
)
from tabularium.engine.rubric import load_rubric
from tabularium.storage.types import RecordReview, ReinforcementEvent

_EPOCH = datetime(2025, 1, 1, tzinfo=UTC)
_PARAMS = load_rubric().parameters


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _fact(
    *,
    content: str = "A durable gotcha worth remembering about the deployment pipeline.",
    scope: str = "proj-a",
    lifecycle: str = "active",
    created_at: datetime = _EPOCH,
) -> Fact:
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity="valid",
        created_at=created_at,
        updated_at=created_at,
    )


def _decision(*, scope: str = "proj-a", created_at: datetime = _EPOCH) -> Decision:
    return Decision(
        id=new_id(),
        type="decision",
        content="a decision",
        scope=Scope(scope),
        lifecycle="in_force",
        validity="valid",
        created_at=created_at,
        updated_at=created_at,
    )


def _session(
    *, scope: str = "proj-a", started_at: datetime, ended_at: datetime, content: str
) -> Session:
    return Session(
        id=new_id(),
        scope=scope,
        started_at=started_at,
        ended_at=ended_at,
        content=content,
    )


def _shelved_event(*, target_id: object, at: datetime) -> ReinforcementEvent:
    return ReinforcementEvent(
        target_id=target_id,  # type: ignore[arg-type]
        at=at,
        source="shelved",
        reason="test shelve/unshelve",
    )


def _dedup_event(*, target_id: object, at: datetime) -> ReinforcementEvent:
    return ReinforcementEvent(
        target_id=target_id,  # type: ignore[arg-type]
        at=at,
        source="dedup",
        matched_from_id=new_id(),
    )


def _review(*, band: str | None, outcome: str = "judged") -> RecordReview:
    return RecordReview(
        target_id=new_id(),
        scope=Scope("proj-a"),
        reviewed_at=_EPOCH,
        next_review_due_at=_EPOCH + timedelta(days=90),
        outcome=outcome,  # type: ignore[arg-type]
        band=band,  # type: ignore[arg-type]
        grades=("keep-clear", "keep-lean") if band == "keep" else (),
        rubric_version="cleaner-rubric-v1-test",
    )


# ---------------------------------------------------------------------------
# Belt-and-braces exclusions (decisions 7, 8)
# ---------------------------------------------------------------------------


class TestBeltAndBracesExclusions:
    def test_global_scope_is_excluded(self) -> None:
        record = _fact(scope="global")
        pool = select_revisit_pool([record], events_by_id={}, sessions_by_scope={}, params=_PARAMS)
        assert pool.excluded_global == (record.id,)
        assert pool.candidates == ()

    def test_decision_is_excluded(self) -> None:
        record = _decision()
        pool = select_revisit_pool([record], events_by_id={}, sessions_by_scope={}, params=_PARAMS)
        assert pool.excluded_non_fact == (record.id,)
        assert pool.candidates == ()


# ---------------------------------------------------------------------------
# Protected records (decision 9)
# ---------------------------------------------------------------------------


class TestProtectedRule:
    def test_active_fact_with_two_shelved_events_is_protected(self) -> None:
        """A currently-active Fact that carries a completed shelve-then-unshelve round trip (one
        `shelved`-source event per leg) must have been un-shelved to get back to `active` --
        never re-shelved by this panel."""
        record = _fact(lifecycle="active")
        events = {
            record.id: (
                _shelved_event(target_id=record.id, at=_EPOCH),
                _shelved_event(target_id=record.id, at=_EPOCH + timedelta(days=1)),
            )
        }
        pool = select_revisit_pool(
            [record], events_by_id=events, sessions_by_scope={}, params=_PARAMS
        )
        assert pool.protected == (record.id,)
        assert pool.candidates == ()
        assert pool.skipped_no_evidence == ()

    def test_active_fact_with_a_single_shelved_event_is_not_protected(self) -> None:
        """`set_shelved` appends its audit
        event BEFORE the lifecycle transition, so a single FAILED shelve attempt on an active
        Fact leaves exactly one stray `shelved`-source event behind despite no un-shelve ever
        having happened. A lone event must not be enough to permanently protect the record --
        only a completed round trip (two events) does."""
        record = _fact(lifecycle="active")
        events = {record.id: (_shelved_event(target_id=record.id, at=_EPOCH),)}
        pool = select_revisit_pool(
            [record], events_by_id=events, sessions_by_scope={}, params=_PARAMS
        )
        assert pool.protected == ()
        # No later session, and the one event since birth is at `_EPOCH == created_at`, not
        # strictly after it -- falls to skipped_no_evidence, never candidates.
        assert pool.skipped_no_evidence == (record.id,)

    def test_active_fact_with_no_shelved_event_is_not_protected(self) -> None:
        record = _fact(lifecycle="active")
        events = {record.id: (_dedup_event(target_id=record.id, at=_EPOCH),)}
        pool = select_revisit_pool(
            [record], events_by_id=events, sessions_by_scope={}, params=_PARAMS
        )
        assert pool.protected == ()
        # No later session, and the one event since birth is at `_EPOCH == created_at`, not
        # strictly after it -- falls to skipped_no_evidence, never candidates.
        assert pool.skipped_no_evidence == (record.id,)

    def test_archived_fact_with_shelved_events_is_not_protected(self) -> None:
        """`archived`, not `active` -- the structural inference only holds for `active` (only an
        un-shelve lands a Fact back on `active` from `shelved`)."""
        record = _fact(lifecycle="archived")
        events = {
            record.id: (
                _shelved_event(target_id=record.id, at=_EPOCH),
                _shelved_event(target_id=record.id, at=_EPOCH + timedelta(days=1)),
            )
        }
        pool = select_revisit_pool(
            [record], events_by_id=events, sessions_by_scope={}, params=_PARAMS
        )
        assert pool.protected == ()

    def test_is_protected_unit(self) -> None:
        active = _fact(lifecycle="active")
        assert (
            is_protected(
                active,
                (
                    _shelved_event(target_id=active.id, at=_EPOCH),
                    _shelved_event(target_id=active.id, at=_EPOCH + timedelta(days=1)),
                ),
            )
            is True
        )
        # A single stray event (e.g. a failed shelve attempt -- event appended, transition
        # raised) is not enough.
        assert is_protected(active, (_shelved_event(target_id=active.id, at=_EPOCH),)) is False
        assert is_protected(active, ()) is False
        archived = _fact(lifecycle="archived")
        assert (
            is_protected(
                archived,
                (
                    _shelved_event(target_id=archived.id, at=_EPOCH),
                    _shelved_event(target_id=archived.id, at=_EPOCH + timedelta(days=1)),
                ),
            )
            is False
        )


# ---------------------------------------------------------------------------
# No covering session -- annotation, not exclusion (decision 18)
# ---------------------------------------------------------------------------


class TestProvenanceGuard:
    def test_no_covering_session_is_annotated_not_excluded(self) -> None:
        record = _fact(created_at=_EPOCH)
        # A later session exists (evidence), but nothing covers the record's own birth.
        later = _session(
            started_at=_EPOCH + timedelta(days=5),
            ended_at=_EPOCH + timedelta(days=5, hours=1),
            content="Revisited the deployment pipeline gotcha again today.",
        )
        pool = select_revisit_pool(
            [record],
            events_by_id={},
            sessions_by_scope={Scope("proj-a"): (later,)},
            params=_PARAMS,
        )
        assert len(pool.candidates) == 1
        assert pool.candidates[0].birth_session is None
        assert pool.candidates[0].later_sessions == (later,)

    def test_covering_session_is_found_by_range_match(self) -> None:
        record = _fact(created_at=_EPOCH)
        birth = _session(
            started_at=_EPOCH - timedelta(hours=2),
            ended_at=_EPOCH + timedelta(hours=2),
            content="Session where the deployment pipeline gotcha was written down.",
        )
        # Evidence so the record survives to candidates.
        later = _session(
            started_at=_EPOCH + timedelta(days=5),
            ended_at=_EPOCH + timedelta(days=5, hours=1),
            content="Revisited the deployment pipeline gotcha again today.",
        )
        pool = select_revisit_pool(
            [record],
            events_by_id={},
            sessions_by_scope={Scope("proj-a"): (birth, later)},
            params=_PARAMS,
        )
        assert len(pool.candidates) == 1
        assert pool.candidates[0].birth_session is birth

    def test_find_birth_session_unit_no_match(self) -> None:
        record = _fact(created_at=_EPOCH)
        elsewhere = _session(
            started_at=_EPOCH + timedelta(days=1),
            ended_at=_EPOCH + timedelta(days=1, hours=1),
            content="unrelated",
        )
        assert find_birth_session(record, (elsewhere,)) is None


# ---------------------------------------------------------------------------
# No evidence since birth (decision 1 / decision 10's calibration concern)
# ---------------------------------------------------------------------------


class TestNoEvidenceRule:
    def test_no_later_session_and_no_event_is_skipped(self) -> None:
        record = _fact(created_at=_EPOCH)
        pool = select_revisit_pool([record], events_by_id={}, sessions_by_scope={}, params=_PARAMS)
        assert pool.skipped_no_evidence == (record.id,)
        assert pool.candidates == ()

    def test_later_session_not_mentioning_subject_does_not_count_as_evidence(self) -> None:
        record = _fact(
            content="A durable gotcha worth remembering about the deployment pipeline.",
            created_at=_EPOCH,
        )
        unrelated = _session(
            started_at=_EPOCH + timedelta(days=5),
            ended_at=_EPOCH + timedelta(days=5, hours=1),
            content="Today we discussed lunch plans and the weather forecast.",
        )
        pool = select_revisit_pool(
            [record],
            events_by_id={},
            sessions_by_scope={Scope("proj-a"): (unrelated,)},
            params=_PARAMS,
        )
        assert pool.skipped_no_evidence == (record.id,)

    def test_a_lifecycle_event_since_birth_counts_as_evidence(self) -> None:
        record = _fact(created_at=_EPOCH)
        events = {record.id: (_dedup_event(target_id=record.id, at=_EPOCH + timedelta(days=1)),)}
        pool = select_revisit_pool(
            [record], events_by_id=events, sessions_by_scope={}, params=_PARAMS
        )
        assert pool.skipped_no_evidence == ()
        assert len(pool.candidates) == 1

    def test_a_matching_later_session_counts_as_evidence(self) -> None:
        record = _fact(
            content="A durable gotcha worth remembering about the deployment pipeline.",
            created_at=_EPOCH,
        )
        later = _session(
            started_at=_EPOCH + timedelta(days=5),
            ended_at=_EPOCH + timedelta(days=5, hours=1),
            content="Revisited the deployment pipeline gotcha again today.",
        )
        pool = select_revisit_pool(
            [record],
            events_by_id={},
            sessions_by_scope={Scope("proj-a"): (later,)},
            params=_PARAMS,
        )
        assert pool.skipped_no_evidence == ()
        assert len(pool.candidates) == 1
        assert pool.candidates[0].later_sessions == (later,)

    def test_session_mentions_subject_unit(self) -> None:
        content = "A durable gotcha worth remembering about the deployment pipeline."
        assert session_mentions_subject(
            content, "Revisited the deployment pipeline gotcha again today."
        )
        assert not session_mentions_subject(content, "Totally unrelated small talk.")

    def test_find_later_sessions_orders_ascending_and_excludes_the_birth_row(self) -> None:
        record = _fact(
            content="A durable gotcha worth remembering about the deployment pipeline.",
            created_at=_EPOCH,
        )
        birth = _session(
            started_at=_EPOCH - timedelta(hours=1),
            ended_at=_EPOCH + timedelta(hours=1),
            content="Deployment pipeline gotcha written down here at birth.",
        )
        newer = _session(
            started_at=_EPOCH + timedelta(days=10),
            ended_at=_EPOCH + timedelta(days=10, hours=1),
            content="Deployment pipeline gotcha mentioned again, newer.",
        )
        older = _session(
            started_at=_EPOCH + timedelta(days=3),
            ended_at=_EPOCH + timedelta(days=3, hours=1),
            content="Deployment pipeline gotcha mentioned again, older.",
        )
        later = find_later_sessions(record, birth, (birth, newer, older))
        assert later == (older, newer)


# ---------------------------------------------------------------------------
# The history block (decision 10, 13) -- no record id, birth digest labelled
# ---------------------------------------------------------------------------


class TestHistoryBlock:
    def test_no_covering_session_renders_the_provenance_marker(self) -> None:
        block = assemble_history_block(birth_session=None, later_sessions=())
        assert "provenance unavailable (no covering session)" in block

    def test_birth_digest_is_labelled_as_the_authors_own_account(self) -> None:
        birth = _session(
            started_at=_EPOCH, ended_at=_EPOCH, content="THE-BIRTH-DIGEST-CONTENT-MARKER"
        )
        block = assemble_history_block(birth_session=birth, later_sessions=())
        assert "author's own account" in block
        assert "THE-BIRTH-DIGEST-CONTENT-MARKER" in block

    def test_later_sessions_are_listed(self) -> None:
        later_a = _session(
            started_at=_EPOCH + timedelta(days=1),
            ended_at=_EPOCH + timedelta(days=1),
            content="LATER-SESSION-A-MARKER",
        )
        later_b = _session(
            started_at=_EPOCH + timedelta(days=2),
            ended_at=_EPOCH + timedelta(days=2),
            content="LATER-SESSION-B-MARKER",
        )
        block = assemble_history_block(
            birth_session=None, later_sessions=(later_a, later_b)
        )
        assert "LATER-SESSION-A-MARKER" in block
        assert "LATER-SESSION-B-MARKER" in block

    def test_a_large_history_stays_inside_the_budget_newest_sessions_first(self) -> None:
        birth = _session(started_at=_EPOCH, ended_at=_EPOCH, content="B" * 20_000)
        later = [
            _session(
                started_at=_EPOCH + timedelta(days=day),
                ended_at=_EPOCH + timedelta(days=day),
                content=f"S{day:02d}-" + "x" * 8_995,
            )
            for day in range(1, 11)
        ]

        block = assemble_history_block(birth_session=birth, later_sessions=later)

        assert len(block) <= HISTORY_BUDGET_CHARS + 1_000
        assert block.count("B") >= EXCERPT_CAP_CHARS
        assert "B" * (EXCERPT_CAP_CHARS + 1) not in block
        assert "[... clipped, 12000 more chars]" in block
        assert block.index("author's own account") < block.index("Later sessions")
        included = [day for day in range(1, 11) if f"S{day:02d}-" in block]
        assert included == [8, 9, 10]
        assert block.index("S10-") < block.index("S09-") < block.index("S08-")
        assert "(7 older later session(s) omitted" in block

    def test_small_inputs_render_whole_with_no_omitted_line(self) -> None:
        birth = _session(started_at=_EPOCH, ended_at=_EPOCH, content="short birth")
        later = _session(
            started_at=_EPOCH + timedelta(days=1),
            ended_at=_EPOCH + timedelta(days=1),
            content="short later",
        )

        block = assemble_history_block(
            birth_session=birth, later_sessions=(later,)
        )

        assert "short birth" in block
        assert "- short later" in block
        assert "clipped" not in block
        assert "omitted" not in block

    def test_no_record_id_ever_appears_in_the_block(self) -> None:
        record = _fact()
        birth = _session(started_at=_EPOCH, ended_at=_EPOCH, content=f"mentions {record.id} not")
        block = assemble_history_block(birth_session=birth, later_sessions=())
        # The block itself never formats the id in -- it only carries whatever the session
        # content already had (this fixture deliberately embeds it in the SESSION content to
        # prove the function does not additionally strip session content; the structural
        # guarantee this function itself provides is that it never APPENDS the id).
        assert str(record.id) not in block.replace(f"mentions {record.id} not", "")


# ---------------------------------------------------------------------------
# Adaptive interval maths (decision 13)
# ---------------------------------------------------------------------------


class TestNextReviewDueAt:
    def test_empty_history_returns_base_interval(self) -> None:
        due = next_review_due_at(_EPOCH, history=(), params=_PARAMS)
        assert due == _EPOCH + timedelta(days=_PARAMS.revisit_interval_days)

    def test_one_consecutive_keep_compounds_once(self) -> None:
        history = (_review(band="keep"),)
        due = next_review_due_at(_EPOCH, history=history, params=_PARAMS)
        expected_days = _PARAMS.revisit_interval_days * _PARAMS.revisit_stable_multiplier
        assert due == _EPOCH + timedelta(days=expected_days)

    def test_two_consecutive_keeps_compound_twice(self) -> None:
        history = (_review(band="keep"), _review(band="keep"))
        due = next_review_due_at(_EPOCH, history=history, params=_PARAMS)
        expected_days = _PARAMS.revisit_interval_days * (_PARAMS.revisit_stable_multiplier**2)
        assert due == _EPOCH + timedelta(days=expected_days)

    def test_a_non_keep_breaks_the_streak(self) -> None:
        history = (_review(band="keep"), _review(band="keep"), _review(band="shelve"))
        due = next_review_due_at(_EPOCH, history=history, params=_PARAMS)
        # Most recent is "shelve" -- streak is 0, no stable multiplier, no contested multiplier.
        assert due == _EPOCH + timedelta(days=_PARAMS.revisit_interval_days)

    def test_last_review_contested_shrinks_the_interval(self) -> None:
        history = (_review(band="keep"), _review(band="contested"))
        due = next_review_due_at(_EPOCH, history=history, params=_PARAMS)
        expected_days = _PARAMS.revisit_interval_days * _PARAMS.revisit_contested_multiplier
        assert due == _EPOCH + timedelta(days=expected_days)

    def test_clamped_to_max_interval(self) -> None:
        history = tuple(_review(band="keep") for _ in range(20))
        due = next_review_due_at(_EPOCH, history=history, params=_PARAMS)
        assert due == _EPOCH + timedelta(days=_PARAMS.revisit_max_interval_days)
