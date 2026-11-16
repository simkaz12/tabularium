"""Clock seam: the engine reads the current time through an injected ``Clock``.

Code under ``src/`` calls ``Clock.now()`` instead of ``datetime.datetime.now()`` so a test can
supply a fixed or advancing clock. ruff's TID251 bans ``datetime.datetime.now`` and ``utcnow``;
``pyproject.toml`` exempts this module from that rule.
"""

from __future__ import annotations

import datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Injectable time source."""

    def now(self) -> datetime.datetime:
        """Return the current instant as a tz-aware UTC datetime."""
        ...


class SystemClock:
    """Wall clock."""

    def now(self) -> datetime.datetime:
        """Return the current instant as a tz-aware UTC datetime."""
        return datetime.datetime.now(tz=datetime.UTC)
