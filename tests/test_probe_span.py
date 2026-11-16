"""``probe_span_for`` property test.

``probe_span_for(W)`` must equal the union of ``read_span(S)`` for every
registered scope ``S`` whose read span contains ``W`` — exactly what D6
words, verified against a Hypothesis-generated registered-scope list of at
least four scopes.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from tabularium.domain import Scope
from tabularium.engine.filters import probe_span_for, read_span

_GLOBAL = Scope("global")

_SCOPE_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-"


def _scope_name_strategy() -> st.SearchStrategy[str]:
    """Generate valid scope names matching ``^[a-z][a-z0-9-]{0,31}$``."""
    return st.builds(
        lambda first, rest: first + rest,
        st.sampled_from("abcdefghijklmnopqrstuvwxyz"),
        st.text(alphabet=_SCOPE_ALPHABET, min_size=0, max_size=10),
    )


def _registered_scopes_strategy(min_size: int = 4) -> st.SearchStrategy[list[Scope]]:
    """At least ``min_size`` distinct registered scopes, ``global`` always
    among them — the engine registers it unconditionally at ``initialize()``."""
    return st.builds(
        lambda names: [_GLOBAL, *(Scope(n) for n in names)],
        st.lists(
            _scope_name_strategy().filter(lambda n: n != "global"),
            min_size=min_size - 1,
            max_size=min_size + 6,
            unique=True,
        ),
    )


def _union_of_read_spans_containing(target: Scope, registered: list[Scope]) -> set[Scope]:
    """The brute-force definition D6 words: the union of every registered
    scope's own read span, for every registered scope whose read span
    contains ``target``."""
    union: set[Scope] = set()
    for s in registered:
        span = read_span(s)
        if target in span:
            union.update(span)
    return union


@given(registered=_registered_scopes_strategy())
def test_probe_span_equals_union_of_read_spans_containing_it(
    registered: list[Scope],
) -> None:
    for target in registered:
        expected = _union_of_read_spans_containing(target, registered)
        actual = set(probe_span_for(target, registered))
        assert actual == expected, (target, registered)


@given(registered=_registered_scopes_strategy())
def test_probe_span_matches_the_pieces_worked_shape(registered: list[Scope]) -> None:
    """Restates D6 in the piece's own words: a named scope probes ``[itself,
    global]``; ``global`` probes every registered scope."""
    for target in registered:
        if target == _GLOBAL:
            assert set(probe_span_for(target, registered)) == set(registered)
        else:
            assert probe_span_for(target, registered) == [target, _GLOBAL]


def _global_alone(_scope: Scope, _registered: list[Scope]) -> list[Scope]:
    """The rejected alternative (D6): probe ``global`` alone regardless of
    the target. Kept local to this falsifiability check only — never
    imported by production code."""
    return [_GLOBAL]


@given(registered=_registered_scopes_strategy())
def test_reverting_to_global_alone_fails_the_property(registered: list[Scope]) -> None:
    """Falsifiability: swap in the rejected global-alone implementation and
    watch the same property the tests above pin go red for any named scope
    — proves the property above is not vacuously true."""
    named = [s for s in registered if s != _GLOBAL]
    assert named, "the strategy always includes at least three named scopes"
    target = named[0]
    expected = _union_of_read_spans_containing(target, registered)
    broken = set(_global_alone(target, registered))
    assert broken != expected
