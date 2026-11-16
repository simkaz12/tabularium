"""Maintenance policy tests — ``maybe_upkeep`` in isolation (ADR 0017).

The policy is unit-testable against a fake adapter alone: it never talks to
LanceDB.  ``FakeAdapter`` subclasses ``MemoryAdapter`` (for full ``StorageAdapter``
structural conformance) and overrides ``maintenance_debt``/``upkeep`` with
scripted, instrumented behaviour.
"""

from __future__ import annotations

import ast
import inspect
import logging

from tabularium.config import EngineConfig
from tabularium.engine import maintenance as maintenance_module
from tabularium.engine.maintenance import maybe_upkeep
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import MaintenanceDebt

_THRESHOLD = 20


class FakeAdapter(MemoryAdapter):
    """Instrumented adapter: scripted ``maintenance_debt()`` value, optional raises."""

    def __init__(
        self,
        *,
        num_small_fragments: int,
        upkeep_raises: Exception | None = None,
        debt_raises: Exception | None = None,
    ) -> None:
        super().__init__(embedding_model="fake/nomic", dim=4)
        self._debt_value = num_small_fragments
        self._upkeep_raises = upkeep_raises
        self._debt_raises = debt_raises
        self.maintenance_debt_calls = 0
        self.upkeep_calls = 0

    async def maintenance_debt(self) -> MaintenanceDebt:
        self.maintenance_debt_calls += 1
        if self._debt_raises is not None:
            raise self._debt_raises
        return MaintenanceDebt(
            num_small_fragments=self._debt_value,
            num_rows=0,
            num_unindexed_rows=0,
        )

    async def upkeep(self) -> None:
        self.upkeep_calls += 1
        if self._upkeep_raises is not None:
            raise self._upkeep_raises


async def test_triggers_at_threshold() -> None:
    """A debt report of exactly ``threshold`` triggers upkeep()."""
    cfg = EngineConfig(maintenance_threshold=_THRESHOLD)
    adapter = FakeAdapter(num_small_fragments=_THRESHOLD)
    await maybe_upkeep(adapter, cfg)
    assert adapter.upkeep_calls == 1


async def test_does_not_trigger_below_threshold() -> None:
    """A debt report of ``threshold - 1`` never triggers upkeep()."""
    cfg = EngineConfig(maintenance_threshold=_THRESHOLD)
    adapter = FakeAdapter(num_small_fragments=_THRESHOLD - 1)
    await maybe_upkeep(adapter, cfg)
    assert adapter.upkeep_calls == 0


async def test_does_not_trigger_at_healthy_floor() -> None:
    """A debt report of 1 (the healthy floor) never triggers upkeep()."""
    cfg = EngineConfig(maintenance_threshold=_THRESHOLD)
    adapter = FakeAdapter(num_small_fragments=1)
    await maybe_upkeep(adapter, cfg)
    assert adapter.upkeep_calls == 0


async def test_raising_upkeep_is_absorbed_and_logged(
    caplog: object,
) -> None:
    """An exception from upkeep() is caught, logged at WARNING, and does not propagate."""
    cfg = EngineConfig(maintenance_threshold=_THRESHOLD)
    boom = RuntimeError("boom: upkeep failed")
    adapter = FakeAdapter(num_small_fragments=_THRESHOLD, upkeep_raises=boom)
    with caplog.at_level(logging.WARNING, logger="tabularium"):  # type: ignore[attr-defined]
        await maybe_upkeep(adapter, cfg)  # must return normally, not raise
    assert adapter.upkeep_calls == 1
    warnings = [
        r for r in caplog.records if r.levelno == logging.WARNING  # type: ignore[attr-defined]
    ]
    assert any(r.exc_info is not None for r in warnings), (
        "expected a WARNING log record with the exception attached"
    )
    assert any(
        r.exc_info is not None and r.exc_info[1] is boom for r in warnings
    ), "expected the attached exception to be the one raised by upkeep()"


async def test_maintenance_debt_read_exactly_once() -> None:
    """``maintenance_debt()`` is read exactly once per ``maybe_upkeep`` call."""
    cfg = EngineConfig(maintenance_threshold=_THRESHOLD)
    adapter = FakeAdapter(num_small_fragments=_THRESHOLD)
    await maybe_upkeep(adapter, cfg)
    assert adapter.maintenance_debt_calls == 1


def test_maintenance_module_has_no_lancedb_import() -> None:
    """The policy module never talks to LanceDB — the property that keeps it
    unit-testable against a fake adapter alone (SPEC / ADR 0017)."""
    source = inspect.getsource(maintenance_module)
    tree = ast.parse(source)
    imported_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_names.append(node.module)
    assert not any("lancedb" in name for name in imported_names), (
        f"engine/maintenance.py must not import lancedb; found imports: {imported_names!r}"
    )
