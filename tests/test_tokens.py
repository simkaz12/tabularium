"""``tabularium.tokens`` — the token-counter seam (M2 slice C, SPEC §7).

Hermetic lane
~~~~~~~~~~~~~
``FakeTokenCounter`` only — no network, no disk, no litellm model call. This
is the counter the default test suite (including ``tests/test_context.py``'s
packer tests) injects everywhere.

Integration lane (``-m integration``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Exercises ``LitellmTokenCounter`` — every construction of that class in this
repo lives inside ``TestLitellmTokenCounter`` below, which is marked
``@pytest.mark.integration`` at the class level, so the default lane never
instantiates it (the acceptance criteria). The protocol-conformance
check for this class in the default lane is therefore structural
(``hasattr``/``callable`` against the class object), not an ``isinstance``
against a real instance.
"""

from __future__ import annotations

import pytest

from tabularium.tokens import LitellmTokenCounter, TokenCounter
from tests.support.doubles import FakeTokenCounter

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_fake_counter_satisfies_protocol(self) -> None:
        assert isinstance(FakeTokenCounter(), TokenCounter)

    def test_litellm_counter_class_declares_the_protocol_method(self) -> None:
        # Structural check against the CLASS, not an instance — the default
        # lane must never construct a real LitellmTokenCounter, so this does not instantiate it at
        # all.
        assert hasattr(LitellmTokenCounter, "count")
        assert callable(LitellmTokenCounter.count)


# ---------------------------------------------------------------------------
# FakeTokenCounter — trivially predictable cost function
# ---------------------------------------------------------------------------


class TestFakeTokenCounter:
    def test_cost_is_characters_floor_div_four(self) -> None:
        counter = FakeTokenCounter()
        text = "x" * 37
        assert counter.count(text) == 37 // 4 == 9

    def test_empty_string_costs_zero(self) -> None:
        counter = FakeTokenCounter()
        assert counter.count("") == 0

    def test_same_entry_counted_twice_yields_same_number(self) -> None:
        """No hidden state in the counter (the acceptance criteria)."""
        counter = FakeTokenCounter()
        text = "the exact same content, counted more than once"
        first = counter.count(text)
        second = counter.count(text)
        assert first == second

    def test_counting_one_text_does_not_affect_counting_another(self) -> None:
        """A second, unrelated call must not perturb a prior instance's result
        — proves there is no running/hidden counter state at all, not merely
        that repeat calls with the SAME text agree."""
        counter = FakeTokenCounter()
        a = "a" * 40
        b = "b" * 100
        before = counter.count(a)
        counter.count(b)
        after = counter.count(a)
        assert before == after


# ---------------------------------------------------------------------------
# LitellmTokenCounter — real adapter, integration-gated
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLitellmTokenCounter:
    def test_satisfies_protocol_as_a_real_instance(self) -> None:
        assert isinstance(LitellmTokenCounter(), TokenCounter)

    def test_counts_a_real_sentence_with_the_default_model(self) -> None:
        counter = LitellmTokenCounter()
        n = counter.count("hello world, this is a test sentence")
        assert isinstance(n, int)
        assert n > 0

    def test_same_entry_counted_twice_yields_same_number(self) -> None:
        counter = LitellmTokenCounter()
        text = "a stable sentence for repeat counting"
        assert counter.count(text) == counter.count(text)
