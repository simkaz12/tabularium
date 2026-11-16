"""Raw access to LanceDB table handles for tests."""

from __future__ import annotations

from typing import Any, cast

from tabularium.storage.lance_protocols import LanceTable


def raw_table(handle: LanceTable) -> Any:  # tests drive methods the typed handle omits
    return handle


def proxy_table(proxy: object) -> LanceTable:
    return cast(LanceTable, proxy)  # fault proxy forwards every other attribute to the real handle
