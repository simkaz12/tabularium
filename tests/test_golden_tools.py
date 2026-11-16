"""Golden tool test: every read tool's full response, and the store it leaves behind, pinned.

``tests/golden/tool_runs.json`` holds one section per scenario. A scenario builds an engine over a
``MemoryAdapter`` with the cluster embedder, a clock that ticks once per read, ``uuid4`` pinned to a
counter and the digest cleaner off, with a scripted Assessor where the path needs one, then drives
the read tools through ``app.call_tool`` of ``make_app(engine)`` as an MCP client does. The server
registers no write tool, so the writes that seed a store run on the engine through
``ToolWorld.write`` and are logged as ``write`` entries. The scenarios share
one app whose engine is a forwarding object (``tool_golden_world._EngineSlot``), because building
the app costs about 0.2 s. Its section dumps:

- every call in order, with its arguments and either the full ``structured_content`` or the exact
  error text (``isError``), plus the engine-level calls no tool reaches (labelled ``direct``);
- every record the ``tabularium`` loggers emitted, at any level, in emitted order, as logger name,
  level, message and ``extra`` (``logs``; the calls and store above are under ``run``);
- the store afterwards: every record, vector presence, tombstones, verdict rows, events, read
  events, unsupersede and revalidation rows, distinct pairs, sessions, reviews, judgments and the
  cleaner's state and pass history. A collection with nothing in it is left out; the scope, facet
  and role registries are not dumped, and the scenario that re-runs ``initialize`` logs them.

The scenario modules are ``tests/support/tool_scenarios_*.py``:

- ``reads``: ``recall``, ``decision``, ``decisions_query`` and ``context``, over stores seeded with
  ``ToolWorld.write``;
- ``direct``: engine functions that no tool reaches (``initialize``, ``save`` with a kind or
  content it refuses, ``context`` and ``pack`` with a negative budget, ``clean``, the nightly loop
  and tick).

Lists stay in the order the code emitted them. Normalization is the cleaner golden's: UUIDs become
``<uuid-N>`` in order of first appearance within a scenario, and the pydantic minor version in the
error URL that ends a validation message becomes ``<version>``. Timestamps stay exact because the
clock is deterministic. Floats stay exact: importance comes from ``math.exp`` and
``math.log1p`` and the RRF scores from division. One machine reproduces them bit for bit across
runs, also under other ``PYTHONHASHSEED`` and ``TZ`` values; another ``libm`` may not.

The golden was written once at 2cd02e6 and is read-only. On 2026-11-03 the removal of the eleven
write tools from the MCP surface pruned it from 145 sections to 26: every scenario that drove a
write tool as its subject was deleted, and the 26 that remain reproduce their old sections apart
from the ``tool`` key of a seeding call becoming ``write``, and the empty-decision refusals becoming
``direct`` calls. To regenerate it deliberately, write
``await build_golden()`` to ``tests/golden/tool_runs.json``. A mismatch names the scenario and
prints a unified diff of its section.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

import pytest

import tests.support.tool_scenarios_direct as _direct_scenarios
import tests.support.tool_scenarios_reads as _read_scenarios
from tabularium.server import make_app
from tests.support.cleaner_world import UuidPin, pinned_environment
from tests.support.tool_golden_world import TOOL_SCENARIOS, captured_logs, make_world
from tests.test_golden_cleaner import dump_json, normalize

GOLDEN_PATH = Path(__file__).parent / "golden" / "tool_runs.json"

REGISTERED_MODULES = (
    _read_scenarios,
    _direct_scenarios,
)


async def build_sections() -> dict[str, Any]:
    """Run every scenario in name order and return its normalized section."""
    pin = UuidPin()
    sections: dict[str, Any] = {}
    with pinned_environment(pin):
        for name in sorted(TOOL_SCENARIOS):
            pin.reset()
            with captured_logs() as records:
                dump = await TOOL_SCENARIOS[name]()
            sections[name] = json.loads(normalize(dump_json({"run": dump, "logs": records})))
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


async def test_tool_golden() -> None:
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
        pytest.fail("tool golden mismatch\n" + "\n".join(problems), pytrace=False)


def _tools_called(section: object) -> set[str]:
    called: set[str] = set()
    if isinstance(section, dict):
        tool = section.get("tool")
        if isinstance(tool, str):
            called.add(tool)
        for value in section.values():
            called |= _tools_called(value)
    elif isinstance(section, list):
        for item in section:
            called |= _tools_called(item)
    return called


async def test_golden_drives_every_registered_tool() -> None:
    world = await make_world()
    registered = {tool.name for tool in await make_app(world.engine).list_tools()}
    expected: dict[str, Any] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    assert registered == {"recall", "decision", "decisions_query", "context"}
    assert _tools_called(expected) == registered
