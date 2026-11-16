"""Seed an admission case's ``existing`` rows through production tools only.

Every row lands through ``engine.save`` with the ``TestClock`` set to the row's own
``created_at`` first; any starting lifecycle or validity is then reached through
``set_validity``/``set_shelved``/``bury`` — never a direct ``StorageAdapter`` call.
"""

from __future__ import annotations

from tabularium.domain import HypomnemaId
from tabularium.engine import Tabularium
from tabularium.results import BounceResolution
from tests.support.cases import Case
from tests.support.doubles import TestClock


class ArchivedStartUnsupported(Exception):
    """A case's existing row must start ``lifecycle: archived`` — no production transition
    reaches ``archived`` yet (a prior decision: the engine has no decay sweep). The
    default-lane runner catches this and skips the case with the named reason."""


async def seed_case(engine: Tabularium, clock: TestClock, case: Case) -> dict[str, str]:
    """Seed every ``case.existing`` row, returning ``alias -> id`` (string ids).

    A bounce during the seeding save is resolved with ``confirm_distinct`` against the bounce's
    own candidate — the case author chose these rows as neighbours on purpose, so seeding must
    never treat a bounce as a setup failure.

    Raises:
        ArchivedStartUnsupported: a row's ``lifecycle`` is ``"archived"``.
    """
    alias_to_id: dict[str, str] = {}

    for row in case.existing:
        if row.lifecycle == "archived":
            raise ArchivedStartUnsupported(
                f"case {case.id!r}: existing row {row.alias!r} starts archived, but no "
                "production transition reaches archived yet"
            )

        clock.set(row.created_at)
        result = await engine.save(row.text, scope=row.scope, kind=row.kind)
        if result.outcome == "bounced":
            assert result.bounce is not None
            result = await engine.save(
                row.text,
                scope=row.scope,
                kind=row.kind,
                resolve=BounceResolution(
                    candidate_id=result.bounce.candidate_id, action="confirm_distinct"
                ),
            )

        assert result.hypomnema is not None
        record_id: HypomnemaId = result.hypomnema.id
        alias_to_id[row.alias] = str(record_id)

        if row.validity == "invalid":
            await engine.set_validity(record_id, valid=False, reason="admission case seed")
        if row.lifecycle == "shelved":
            await engine.set_shelved(record_id, shelved=True, reason="admission case seed")
        if row.lifecycle == "buried":
            await engine.bury(record_id, reason="admission case seed")

    return alias_to_id
