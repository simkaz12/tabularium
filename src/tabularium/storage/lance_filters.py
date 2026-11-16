"""Typed builders for LanceDB SQL filter strings.

Every filter the LanceDB adapter and codec send to the engine is built here, so a
value never reaches a query by raw interpolation. Strings and UUIDs are quoted by
``sql_quote``; timestamps are formatted as ``timestamp '<iso>'`` literals and the
column names and operators are fixed by the caller, never taken from data.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Literal

TimestampOp = Literal[">", ">=", "<="]


def sql_quote(value: str) -> str:
    """Escape a string for use inside a single-quoted LanceDB SQL literal."""
    return value.replace("'", "''")


def _literal(value: str | uuid.UUID) -> str:
    return f"'{sql_quote(str(value))}'"


def eq(column: str, value: str | uuid.UUID) -> str:
    return f"{column} = {_literal(value)}"


def ne(column: str, value: str | uuid.UUID) -> str:
    return f"{column} != {_literal(value)}"


def in_(column: str, values: Iterable[str | uuid.UUID]) -> str:
    return f"{column} IN ({', '.join(_literal(value) for value in values)})"


def is_null(column: str) -> str:
    return f"{column} IS NULL"


def is_not_null(column: str) -> str:
    return f"{column} IS NOT NULL"


def ts_cmp(column: str, op: TimestampOp, moment: datetime) -> str:
    return f"{column} {op} timestamp '{moment.isoformat()}'"


def and_(*parts: str) -> str:
    return " AND ".join(parts)


def or_(*parts: str) -> str:
    return f"({' OR '.join(parts)})"
