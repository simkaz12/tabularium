"""Session-finish reporting for the admission case-set runner.

Collects, via ``item.user_properties``, which axis each admission test belonged to
(``("admission_axis", axis)``) and which mutation-axis failure classes it hit
(``("failure_class", cls)``, zero or more per item). At session finish, prints one table per
axis that had items, worst failure class first — lost history, wrong replacement, live
contradiction, kept duplicate — or the single line ``admission: no cases ran`` when zero
admission tests were collected at all.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from _pytest.terminal import TerminalReporter

_FAILURE_CLASSES_WORST_FIRST: tuple[str, ...] = (
    "lost history",
    "wrong replacement",
    "live contradiction",
    "kept duplicate",
)

_axis_totals: Counter[str] = Counter()
_axis_failure_classes: dict[str, Counter[str]] = {}


def _axis_of(item: pytest.Item) -> str | None:
    for name, value in item.user_properties:
        if name == "admission_axis":
            return str(value)
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, Any, None]:
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return

    axis = _axis_of(item)
    if axis is None:
        return

    _axis_totals[axis] += 1
    for name, value in item.user_properties:
        if name == "failure_class":
            _axis_failure_classes.setdefault(axis, Counter())[str(value)] += 1


def pytest_terminal_summary(terminalreporter: TerminalReporter) -> None:
    if not _axis_totals:
        terminalreporter.write_line("admission: no cases ran")
        return

    for axis in sorted(_axis_totals):
        counts = _axis_failure_classes.get(axis, Counter())
        terminalreporter.write_line(f"admission [{axis}] failure classes (worst first):")
        for failure_class in _FAILURE_CLASSES_WORST_FIRST:
            terminalreporter.write_line(f"  {failure_class}: {counts.get(failure_class, 0)}")
