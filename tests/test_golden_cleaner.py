"""Golden cleaner test: the observable output of the three axes and ``clean()``, pinned.

``tests/golden/cleaner_runs.json`` holds one section per scenario. A scenario runs one axis, or
``clean()``, over a ``MemoryAdapter`` with scripted judges and a scripted Assessor, a clock that
ticks once per read, ``uuid4`` pinned to a counter, the judge-call and Assessor-call timers
pinned to zero and the token counter replaced by ``len // 4``
(``tests/support/cleaner_world.pinned_environment``). Its section dumps:

- the axis summary, or the ``CleanSummary`` of a ``clean()`` run, with the CLI text for it;
- for an axis scenario, the ``CleanerReport`` JSON, the fallback pass-history payload and the
  context summary built from a ``CleanSummary`` composed around that axis summary;
- for a ``clean()`` scenario, the pass-history rows and ``CleanerState`` as stored, and the context
  summary rendered from each row;
- the records the run changed or added, every verdict row, the record reviews, record judgments and
  reinforcement events added after seeding, all in store order;
- one entry per judge call (call number, fault, prompt length and sha256) and every Assessor call.

The ``report_*`` scenarios render hand-built summaries and pass-history rows, and the
``verdicts_*`` scenarios settle the pending rows a crashed pass leaves for the startup reconcile.

Lists stay in the order the code emitted them. Two things are normalized: UUIDs become ``<uuid-N>``
in order of first appearance within a scenario, and the pydantic minor version in the error URL
that ends a persisted schema-validation message becomes ``<version>``. Timestamps stay exact
because the clock is deterministic and the offsets between them (``now + 1µs``) are part of what is
pinned.

The golden was written once at 735f087 and is read-only. To regenerate it deliberately, write
``await build_golden()`` to ``tests/golden/cleaner_runs.json``. A mismatch names the scenario and
prints a unified diff of its section.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

import pytest

import tests.support.cleaner_scenarios_clean as _clean_scenarios
import tests.support.cleaner_scenarios_falsity as _falsity_scenarios
import tests.support.cleaner_scenarios_relevance as _relevance_scenarios
import tests.support.cleaner_scenarios_report as _report_scenarios
import tests.support.cleaner_scenarios_revisit as _revisit_scenarios
import tests.support.cleaner_scenarios_verdicts as _verdicts_scenarios
from tests.support.cleaner_world import SCENARIOS, UuidPin, pinned_environment

GOLDEN_PATH = Path(__file__).parent / "golden" / "cleaner_runs.json"

_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_PYDANTIC_DOCS = re.compile(r"https://errors\.pydantic\.dev/\d+\.\d+/")

REGISTERED_MODULES = (
    _relevance_scenarios,
    _falsity_scenarios,
    _revisit_scenarios,
    _clean_scenarios,
    _report_scenarios,
    _verdicts_scenarios,
)


def dump_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def normalize(text: str) -> str:
    """Number UUIDs in order of first appearance and drop the pydantic minor version from the
    documentation URL its validation errors end with.
    """
    seen: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        return seen.setdefault(match.group(0).lower(), f"<uuid-{len(seen) + 1}>")

    return _PYDANTIC_DOCS.sub("https://errors.pydantic.dev/<version>/", _UUID.sub(replace, text))


async def build_sections() -> dict[str, Any]:
    """Run every scenario in name order and return its normalized section."""
    pin = UuidPin()
    sections: dict[str, Any] = {}
    with pinned_environment(pin):
        for name in sorted(SCENARIOS):
            pin.reset()
            dump = await SCENARIOS[name]()
            sections[name] = json.loads(normalize(dump_json(dump)))
    return sections


async def build_golden() -> str:
    return dump_json(await build_sections())


def _diff(name: str, expected: object, actual: object) -> str:
    return "".join(
        difflib.unified_diff(
            dump_json(expected).splitlines(keepends=True),
            dump_json(actual).splitlines(keepends=True),
            fromfile=f"golden/{name}",
            tofile=f"actual/{name}",
            n=2,
        )
    )


async def test_cleaner_golden() -> None:
    expected: dict[str, Any] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    actual = await build_sections()

    problems: list[str] = []
    for name in sorted(set(expected) - set(actual)):
        problems.append(f"scenario {name} is in the golden and no longer runs")
    for name in sorted(set(actual) - set(expected)):
        problems.append(f"scenario {name} runs and is not in the golden")
    for name in sorted(set(expected) & set(actual)):
        if expected[name] != actual[name]:
            diff = _diff(name, expected[name], actual[name])
            problems.append(f"scenario {name} differs\n{diff[:6000]}")

    if problems:
        pytest.fail("cleaner golden mismatch\n" + "\n".join(problems), pytrace=False)
