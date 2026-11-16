"""``EngineConfig`` construction — the auto-reinforce band's field-level deletion.

P7 admission-order deleted the cosine auto-reinforce band (``auto_ceiling``,
``paranoid``, ``effective_auto_ceiling``) outright rather than leaving it as
inert dead config: a caller still passing ``paranoid=`` is a stale caller,
not a no-op one, so construction must fail loud rather than silently accept
and ignore the kwarg.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tabularium.config import EngineConfig


def test_paranoid_kwarg_raises() -> None:
    with pytest.raises(ValidationError):
        EngineConfig(paranoid=False)  # type: ignore[call-arg]


def test_auto_ceiling_kwarg_raises() -> None:
    with pytest.raises(ValidationError):
        EngineConfig(auto_ceiling=0.9)  # type: ignore[call-arg]


def test_band_fields_are_gone() -> None:
    cfg = EngineConfig()
    assert not hasattr(cfg, "paranoid")
    assert not hasattr(cfg, "auto_ceiling")
    assert not hasattr(cfg, "effective_auto_ceiling")


# ---------------------------------------------------------------------------
# A3 cleaner-integration piece: cleaner_prepass_skip_keep requires a cutoff (D1)
# ---------------------------------------------------------------------------


def test_prepass_skip_keep_without_cutoff_raises() -> None:
    with pytest.raises(ValidationError, match="cleaner_prepass_skip_cutoff"):
        EngineConfig(cleaner_prepass_skip_keep=True)


def test_prepass_skip_keep_with_cutoff_constructs() -> None:
    cfg = EngineConfig(cleaner_prepass_skip_keep=True, cleaner_prepass_skip_cutoff=0.9)
    assert cfg.cleaner_prepass_skip_keep is True
    assert cfg.cleaner_prepass_skip_cutoff == 0.9


def test_prepass_skip_keep_defaults_off_with_no_cutoff() -> None:
    cfg = EngineConfig()
    assert cfg.cleaner_prepass_skip_keep is False
    assert cfg.cleaner_prepass_skip_cutoff is None


@pytest.mark.parametrize("cutoff", [0.0, -0.1, 1.1])
def test_prepass_skip_cutoff_out_of_range_raises(cutoff: float) -> None:
    with pytest.raises(ValidationError):
        EngineConfig(cleaner_prepass_skip_cutoff=cutoff)
