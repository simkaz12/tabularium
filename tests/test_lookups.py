from __future__ import annotations

from uuid import uuid4

import pytest

from tabularium.domain import HypomnemaId
from tabularium.engine._errors import ToolError
from tabularium.engine.lookups import get_fact, get_record, unknown_id
from tabularium.storage.memory import MemoryAdapter


class TestNamedLookups:
    async def test_get_record_raises_the_unknown_id_message_without_a_cause(self) -> None:
        adapter = MemoryAdapter(embedding_model="fake", dim=4)
        missing = HypomnemaId(uuid4())

        with pytest.raises(ToolError) as caught:
            await get_record(adapter, missing)

        assert str(caught.value) == f"Unknown id: {missing}"
        assert type(caught.value) is ToolError
        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__ is True

    async def test_get_fact_raises_the_same_refusal_for_a_missing_id(self) -> None:
        missing = HypomnemaId(uuid4())

        with pytest.raises(ToolError) as caught:
            await get_fact(MemoryAdapter(embedding_model="fake", dim=4), missing)

        assert str(caught.value) == f"Unknown id: {missing}"

    def test_unknown_id_builds_the_message(self) -> None:
        target = HypomnemaId(uuid4())

        assert str(unknown_id(target)) == f"Unknown id: {target}"
