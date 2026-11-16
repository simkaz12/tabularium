"""``engine_config_from_env`` — the environment-variable override surface for
``EngineConfig`` (currently wires exactly ``enable_digest_cleaner`` via
``TABULARIUM_ENABLE_CLEANER``).

The deployed service could not turn the Cleaner on before this landed: the
flag lived on ``EngineConfig``, but ``config_from_env`` covered only
store/embedder settings, and ``main()`` never passed a ``config=`` at all —
so every deployed engine ran ``EngineConfig()`` defaults regardless of the
environment. Coverage here therefore has two parts, deliberately kept
separate: parsing (``engine_config_from_env`` itself) and wiring (that its
output actually reaches the engine's ``_cfg`` via ``main()``). A parsing-only
suite would leave the real defect — the missing ``config=`` — uncovered.
"""

from __future__ import annotations

from typing import Any

import pytest

from tabularium import server as server_mod
from tabularium.config import EngineConfig
from tabularium.engine.assessor import Assessor, JevAssessor
from tabularium.server import engine_config_from_env

# ---------------------------------------------------------------------------
# engine_config_from_env — parsing
# ---------------------------------------------------------------------------


def test_unset_defaults_to_cleaner_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TABULARIUM_ENABLE_CLEANER", raising=False)
    cfg = engine_config_from_env()
    assert isinstance(cfg, EngineConfig)
    assert cfg.enable_digest_cleaner is False


@pytest.mark.parametrize("spelling", ["true", "1", "yes", "TRUE", "Yes"])
def test_accepted_true_spellings_enable_the_cleaner(
    monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    monkeypatch.setenv("TABULARIUM_ENABLE_CLEANER", spelling)
    assert engine_config_from_env().enable_digest_cleaner is True


@pytest.mark.parametrize("spelling", ["false", "0", "no", "FALSE", "No"])
def test_accepted_false_spellings_keep_the_cleaner_disabled(
    monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    monkeypatch.setenv("TABULARIUM_ENABLE_CLEANER", spelling)
    assert engine_config_from_env().enable_digest_cleaner is False


def test_unrecognized_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TABULARIUM_ENABLE_CLEANER", "maybe")
    with pytest.raises(ValueError, match="TABULARIUM_ENABLE_CLEANER"):
        engine_config_from_env()


def test_bool_of_the_string_would_have_been_wrong(monkeypatch: pytest.MonkeyPatch) -> None:
    """``bool("0")`` and ``bool("false")`` are both True in Python -- the exact
    silent-typo failure mode the brief forbids. "0" and "false" must resolve
    to False, not raise, and not resolve True."""
    monkeypatch.setenv("TABULARIUM_ENABLE_CLEANER", "0")
    assert engine_config_from_env().enable_digest_cleaner is False

    monkeypatch.setenv("TABULARIUM_ENABLE_CLEANER", "false")
    assert engine_config_from_env().enable_digest_cleaner is False


# ---------------------------------------------------------------------------
# engine_config_from_env — the four P8 auto-exit switches (decision D9),
# each wired through TABULARIUM_AUTO_EXIT_<RELATION> with the SAME accepted
# spellings and fail-loudly discipline as TABULARIUM_ENABLE_CLEANER above.
# ---------------------------------------------------------------------------

_AUTO_EXIT_ENV_TO_FIELD: dict[str, str] = {
    "TABULARIUM_AUTO_EXIT_RESTATEMENT": "auto_exit_restatement",
    "TABULARIUM_AUTO_EXIT_REPLACEMENT": "auto_exit_replacement",
    "TABULARIUM_AUTO_EXIT_FOLLOW_UP": "auto_exit_follow_up",
    "TABULARIUM_AUTO_EXIT_DISTINCT": "auto_exit_distinct",
}


@pytest.mark.parametrize("env_var", sorted(_AUTO_EXIT_ENV_TO_FIELD))
def test_auto_exit_unset_defaults_to_off(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    for other in _AUTO_EXIT_ENV_TO_FIELD:
        monkeypatch.delenv(other, raising=False)
    cfg = engine_config_from_env()
    assert getattr(cfg, _AUTO_EXIT_ENV_TO_FIELD[env_var]) is False


@pytest.mark.parametrize("env_var", sorted(_AUTO_EXIT_ENV_TO_FIELD))
@pytest.mark.parametrize("spelling", ["true", "1", "yes", "TRUE", "Yes"])
def test_auto_exit_accepted_true_spellings(
    monkeypatch: pytest.MonkeyPatch, env_var: str, spelling: str
) -> None:
    for other in _AUTO_EXIT_ENV_TO_FIELD:
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(env_var, spelling)
    cfg = engine_config_from_env()
    assert getattr(cfg, _AUTO_EXIT_ENV_TO_FIELD[env_var]) is True


@pytest.mark.parametrize("env_var", sorted(_AUTO_EXIT_ENV_TO_FIELD))
@pytest.mark.parametrize("spelling", ["false", "0", "no", "FALSE", "No"])
def test_auto_exit_accepted_false_spellings(
    monkeypatch: pytest.MonkeyPatch, env_var: str, spelling: str
) -> None:
    for other in _AUTO_EXIT_ENV_TO_FIELD:
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(env_var, spelling)
    cfg = engine_config_from_env()
    assert getattr(cfg, _AUTO_EXIT_ENV_TO_FIELD[env_var]) is False


@pytest.mark.parametrize("env_var", sorted(_AUTO_EXIT_ENV_TO_FIELD))
def test_auto_exit_unrecognized_value_raises(monkeypatch: pytest.MonkeyPatch, env_var: str) -> None:
    for other in _AUTO_EXIT_ENV_TO_FIELD:
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(env_var, "maybe")
    with pytest.raises(ValueError, match=env_var):
        engine_config_from_env()



# ---------------------------------------------------------------------------
# engine_config_from_env — the Assessor's two env-driven fields (Assessor map
# ticket 02): TABULARIUM_ASSESSOR_MODEL and
# TABULARIUM_ASSESSOR_SAVE_TIMEOUT_S.
# ---------------------------------------------------------------------------


def test_assessor_model_unset_defaults_to_pinned_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TABULARIUM_ASSESSOR_MODEL", raising=False)
    assert engine_config_from_env().assessor_model == "jev-1.13.0"


def test_assessor_model_accepts_a_versioned_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TABULARIUM_ASSESSOR_MODEL", "jev-2.0.1")
    assert engine_config_from_env().assessor_model == "jev-2.0.1"


def test_assessor_model_refuses_an_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TABULARIUM_ASSESSOR_MODEL", "jev-latest")
    with pytest.raises(ValueError, match="jev-latest"):
        engine_config_from_env()


def test_assessor_save_timeout_unset_defaults_to_two_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TABULARIUM_ASSESSOR_SAVE_TIMEOUT_S", raising=False)
    assert engine_config_from_env().assessor_save_timeout_s == 2.0


def test_assessor_save_timeout_accepts_a_float_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TABULARIUM_ASSESSOR_SAVE_TIMEOUT_S", "3.5")
    assert engine_config_from_env().assessor_save_timeout_s == 3.5


def test_assessor_save_timeout_unrecognized_value_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TABULARIUM_ASSESSOR_SAVE_TIMEOUT_S", "not-a-float")
    with pytest.raises(ValueError, match="TABULARIUM_ASSESSOR_SAVE_TIMEOUT_S"):
        engine_config_from_env()


# ---------------------------------------------------------------------------
# main() — TYPESAFE_API_KEY gates whether an Assessor is constructed, no
# network call either way.
# ---------------------------------------------------------------------------


class _AssessorSpyTabularium:
    captured_assessor: Assessor | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _AssessorSpyTabularium.captured_assessor = kwargs.get("assessor")
        raise _StopEarly()


def test_main_constructs_no_assessor_when_key_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TABULARIUM_MCP_TRANSPORT", raising=False)
    monkeypatch.setenv("TABULARIUM_ADAPTER", "memory")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    _AssessorSpyTabularium.captured_assessor = None
    monkeypatch.setattr(server_mod, "Tabularium", _AssessorSpyTabularium)

    with pytest.raises(_StopEarly):
        server_mod.main()

    assert _AssessorSpyTabularium.captured_assessor is None


def test_main_constructs_a_jev_assessor_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TABULARIUM_MCP_TRANSPORT", raising=False)
    monkeypatch.setenv("TABULARIUM_ADAPTER", "memory")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-never-logged")
    _AssessorSpyTabularium.captured_assessor = None
    monkeypatch.setattr(server_mod, "Tabularium", _AssessorSpyTabularium)

    with pytest.raises(_StopEarly):
        server_mod.main()

    assessor = _AssessorSpyTabularium.captured_assessor
    assert isinstance(assessor, JevAssessor)


def test_auto_exit_switches_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting one switch never sets the others — each of the four fields is its own
    env var, never a single combined flag."""
    for other in _AUTO_EXIT_ENV_TO_FIELD:
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv("TABULARIUM_AUTO_EXIT_REPLACEMENT", "true")
    cfg = engine_config_from_env()
    assert cfg.auto_exit_replacement is True
    assert cfg.auto_exit_restatement is False
    assert cfg.auto_exit_follow_up is False
    assert cfg.auto_exit_distinct is False


# ---------------------------------------------------------------------------
# main() — the wiring: engine_config_from_env()'s output must reach the
# Tabularium(...) construction, not just exist as a standalone function.
# ---------------------------------------------------------------------------


class _StopEarly(Exception):
    """Raised by the Tabularium spy to short-circuit main() right after
    construction -- before it reaches eng.initialize() / the transport run
    loop, neither of which this test wants to actually drive."""


class _SpyTabularium:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _SpyTabularium.captured_config = kwargs.get("config")
        raise _StopEarly()

    captured_config: EngineConfig | None = None


def test_main_wires_engine_config_from_env_into_tabularium_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TABULARIUM_MCP_TRANSPORT", raising=False)
    monkeypatch.setenv("TABULARIUM_ADAPTER", "memory")
    monkeypatch.setenv("TABULARIUM_ENABLE_CLEANER", "true")
    _SpyTabularium.captured_config = None
    monkeypatch.setattr(server_mod, "Tabularium", _SpyTabularium)

    with pytest.raises(_StopEarly):
        server_mod.main()

    config = _SpyTabularium.captured_config
    assert config is not None, (
        "Tabularium(...) was never constructed with a `config=` kwarg at all -- "
        "this is the exact defect the brief describes: the flag existed but "
        "nothing threaded it through main()."
    )
    assert config.enable_digest_cleaner is True


def test_main_fails_loudly_for_invalid_cleaner_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid TABULARIUM_ENABLE_CLEANER raises synchronously, before the
    store is ever opened -- matching transport_config_from_env's own
    fail-loudly-at-startup contract."""
    monkeypatch.setenv("TABULARIUM_ENABLE_CLEANER", "carrier-pigeon")
    with pytest.raises(ValueError, match="TABULARIUM_ENABLE_CLEANER"):
        server_mod.main()


# ---------------------------------------------------------------------------
# engine_config_from_env — the A3 cleaner-integration piece's four fields:
# TABULARIUM_CLEANER_PREPASS_SKIP_KEEP, TABULARIUM_CLEANER_PREPASS_SKIP_CUTOFF,
# TABULARIUM_CLEANER_SPLIT_SHELVE, TABULARIUM_ASSESSOR_CLEANER_TIMEOUT_S.
# ---------------------------------------------------------------------------


def _clear_cleaner_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "TABULARIUM_CLEANER_PREPASS_SKIP_KEEP",
        "TABULARIUM_CLEANER_PREPASS_SKIP_CUTOFF",
        "TABULARIUM_CLEANER_SPLIT_SHELVE",
        "TABULARIUM_ASSESSOR_CLEANER_TIMEOUT_S",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    ("env_var", "field"),
    [
        ("TABULARIUM_CLEANER_PREPASS_SKIP_KEEP", "cleaner_prepass_skip_keep"),
        ("TABULARIUM_CLEANER_SPLIT_SHELVE", "cleaner_split_shelve"),
    ],
)
def test_cleaner_bool_switches_unset_default_off(
    monkeypatch: pytest.MonkeyPatch, env_var: str, field: str
) -> None:
    _clear_cleaner_env(monkeypatch)
    assert getattr(engine_config_from_env(), field) is False


@pytest.mark.parametrize(
    ("env_var", "field"),
    [
        ("TABULARIUM_CLEANER_PREPASS_SKIP_KEEP", "cleaner_prepass_skip_keep"),
        ("TABULARIUM_CLEANER_SPLIT_SHELVE", "cleaner_split_shelve"),
    ],
)
@pytest.mark.parametrize("spelling", ["true", "1", "yes", "TRUE", "Yes"])
def test_cleaner_bool_switches_accepted_true_spellings(
    monkeypatch: pytest.MonkeyPatch, env_var: str, field: str, spelling: str
) -> None:
    _clear_cleaner_env(monkeypatch)
    monkeypatch.setenv(env_var, spelling)
    if env_var == "TABULARIUM_CLEANER_PREPASS_SKIP_KEEP":
        monkeypatch.setenv("TABULARIUM_CLEANER_PREPASS_SKIP_CUTOFF", "0.9")
    assert getattr(engine_config_from_env(), field) is True


@pytest.mark.parametrize(
    ("env_var", "field"),
    [
        ("TABULARIUM_CLEANER_PREPASS_SKIP_KEEP", "cleaner_prepass_skip_keep"),
        ("TABULARIUM_CLEANER_SPLIT_SHELVE", "cleaner_split_shelve"),
    ],
)
def test_cleaner_bool_switches_unrecognized_value_raises(
    monkeypatch: pytest.MonkeyPatch, env_var: str, field: str
) -> None:
    _clear_cleaner_env(monkeypatch)
    monkeypatch.setenv(env_var, "carrier-pigeon")
    with pytest.raises(ValueError, match=env_var):
        engine_config_from_env()


def test_cleaner_prepass_skip_cutoff_unset_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cleaner_env(monkeypatch)
    assert engine_config_from_env().cleaner_prepass_skip_cutoff is None


def test_cleaner_prepass_skip_cutoff_accepts_a_float_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cleaner_env(monkeypatch)
    monkeypatch.setenv("TABULARIUM_CLEANER_PREPASS_SKIP_CUTOFF", "0.85")
    assert engine_config_from_env().cleaner_prepass_skip_cutoff == 0.85


def test_cleaner_prepass_skip_cutoff_malformed_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cleaner_env(monkeypatch)
    monkeypatch.setenv("TABULARIUM_CLEANER_PREPASS_SKIP_CUTOFF", "not-a-float")
    with pytest.raises(ValueError, match="TABULARIUM_CLEANER_PREPASS_SKIP_CUTOFF"):
        engine_config_from_env()


def test_assessor_cleaner_timeout_unset_defaults_to_ten_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cleaner_env(monkeypatch)
    assert engine_config_from_env().assessor_cleaner_timeout_s == 10.0


def test_assessor_cleaner_timeout_accepts_a_float_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_cleaner_env(monkeypatch)
    monkeypatch.setenv("TABULARIUM_ASSESSOR_CLEANER_TIMEOUT_S", "5.0")
    assert engine_config_from_env().assessor_cleaner_timeout_s == 5.0


def test_assessor_cleaner_timeout_malformed_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cleaner_env(monkeypatch)
    monkeypatch.setenv("TABULARIUM_ASSESSOR_CLEANER_TIMEOUT_S", "not-a-float")
    with pytest.raises(ValueError, match="TABULARIUM_ASSESSOR_CLEANER_TIMEOUT_S"):
        engine_config_from_env()


def test_spy_tabularium_sees_the_four_new_cleaner_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring half (module docstring): ``engine_config_from_env()``'s output must actually
    reach the ``Tabularium(...)`` construction, not just exist as a standalone function."""
    monkeypatch.delenv("TABULARIUM_MCP_TRANSPORT", raising=False)
    monkeypatch.setenv("TABULARIUM_ADAPTER", "memory")
    _clear_cleaner_env(monkeypatch)
    monkeypatch.setenv("TABULARIUM_CLEANER_PREPASS_SKIP_KEEP", "true")
    monkeypatch.setenv("TABULARIUM_CLEANER_PREPASS_SKIP_CUTOFF", "0.92")
    monkeypatch.setenv("TABULARIUM_CLEANER_SPLIT_SHELVE", "true")
    monkeypatch.setenv("TABULARIUM_ASSESSOR_CLEANER_TIMEOUT_S", "7.5")
    _SpyTabularium.captured_config = None
    monkeypatch.setattr(server_mod, "Tabularium", _SpyTabularium)

    with pytest.raises(_StopEarly):
        server_mod.main()

    config = _SpyTabularium.captured_config
    assert config is not None
    assert config.cleaner_prepass_skip_keep is True
    assert config.cleaner_prepass_skip_cutoff == 0.92
    assert config.cleaner_split_shelve is True
    assert config.assessor_cleaner_timeout_s == 7.5


def test_cleaner_nightly_at_defaults_to_three_am(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TABULARIUM_CLEANER_NIGHTLY_AT", raising=False)
    assert engine_config_from_env().cleaner_nightly_at == "03:00"


def test_cleaner_nightly_at_reads_its_env_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TABULARIUM_CLEANER_NIGHTLY_AT", "04:30")
    assert engine_config_from_env().cleaner_nightly_at == "04:30"


@pytest.mark.parametrize("value", ["4:30", "25:00", "03:75", "nightly"])
def test_cleaner_nightly_at_rejects_a_malformed_value(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("TABULARIUM_CLEANER_NIGHTLY_AT", value)
    with pytest.raises(ValueError, match="TABULARIUM_CLEANER_NIGHTLY_AT"):
        engine_config_from_env()
