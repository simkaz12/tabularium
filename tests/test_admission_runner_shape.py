"""Proof of the runner's own shape, without committing a real case (brief Scope 3).

Writes a temporary case YAML, points ``tests.support.cases.load_cases`` at it through
``TABULARIUM_ADMISSION_CASES``, and drives ``tests/test_admission_runner.py`` as a real
subprocess against this repo's own tree — the only way to observe collected ids, the
session-finish summary and pass/fail together, exactly as an owner running ``uv run pytest``
would see them.

``pytest_plugins = ["pytester"]`` is enabled for its scratch-directory management. The actual
pytest invocation runs through a plain subprocess rather than ``pytester.run()``: the ``pytester``
fixture chdirs the process into an isolated temp root at setup, and this test deliberately runs
the REAL ``tests/test_admission_runner.py`` against the REAL repo tree — its real
``tests/conftest.py`` hook, its real ``tests.support`` imports — not an isolated copy.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

_ELAB_CASE_TEMPLATE = """\
id: elab-01
class: material
origin: authored
existing:
  - alias: e1
    text: "The runner shape probe's original fact text."
    kind: fact
    scope: global
    lifecycle: active
    validity: valid
    created_at: "2025-01-01T00:00:00+00:00"
incoming:
  content: "The runner shape probe's elaborated fact text, superseding e1."
  scope: global
  kind: fact
  supersedes: [e1]
  reason: elaborated
expect:
  material: elaboration
  mutation:
    acceptable:
      - "{check}"
    ideal: []
why: "P2 runner-harness shape proof -- not a real admission case"
"""


def _run_admission_suite(
    *args: str, cases_dir: Path, fake_embedder: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run ``uv run pytest <args>`` as a real subprocess against this repo's own tree."""
    env = os.environ.copy()
    env["TABULARIUM_ADMISSION_CASES"] = str(cases_dir)
    if fake_embedder:
        env["TABULARIUM_ADMISSION_EMBEDDER"] = "fake"
    return subprocess.run(
        ["uv", "run", "pytest", *args],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_admission_runner_shape(pytester: pytest.Pytester) -> None:
    cases_dir = pytester.path / "cases"
    cases_dir.mkdir()
    case_path = cases_dir / "elab-01.yaml"
    case_path.write_text(_ELAB_CASE_TEMPLATE.format(check="rows == 2"), encoding="utf-8")

    default_collect = _run_admission_suite(
        "--collect-only", "-q", "tests/test_admission_runner.py", cases_dir=cases_dir
    )
    assert "test_admission[elab-01][material]" in default_collect.stdout, default_collect.stdout
    assert "test_admission[elab-01][mutation]" in default_collect.stdout, default_collect.stdout
    assert "test_admission[elab-01][selection]" not in default_collect.stdout
    assert "test_admission[elab-01][ideal]" not in default_collect.stdout

    integration_collect = _run_admission_suite(
        "-m",
        "integration",
        "--collect-only",
        "-q",
        "tests/test_admission_runner.py",
        cases_dir=cases_dir,
    )
    assert "test_admission[elab-01][selection]" in integration_collect.stdout
    assert "test_admission[elab-01][ideal]" in integration_collect.stdout

    passing_run = _run_admission_suite(
        "-q", "-rs", "tests/test_admission_runner.py", cases_dir=cases_dir, fake_embedder=True
    )
    assert passing_run.returncode == 0, passing_run.stdout + passing_run.stderr
    assert "2 passed" in passing_run.stdout
    assert "admission [mutation] failure classes (worst first):" in passing_run.stdout
    assert "  lost history: 0" in passing_run.stdout
    assert "  wrong replacement: 0" in passing_run.stdout
    assert "  live contradiction: 0" in passing_run.stdout
    assert "  kept duplicate: 0" in passing_run.stdout

    case_path.write_text(_ELAB_CASE_TEMPLATE.format(check="rows == 3"), encoding="utf-8")

    failing_run = _run_admission_suite(
        "-q", "-rs", "tests/test_admission_runner.py", cases_dir=cases_dir, fake_embedder=True
    )
    assert failing_run.returncode != 0
    assert "1 failed, 1 passed" in failing_run.stdout
    assert "  kept duplicate: 1" in failing_run.stdout


def test_all_texts_includes_case_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Insert the scripts dir temporarily so freeze_vectors.py is importable without making it a
    # package (the same import convention as the other script tests).
    sys.path.insert(0, str(_SCRIPTS_DIR))
    try:
        import freeze_vectors
    finally:
        sys.path.pop(0)

    from tests.support.cases import Case

    case = Case.model_validate(
        {
            "id": "freeze-probe",
            "class": "material",
            "origin": "authored",
            "existing": [
                {
                    "alias": "e1",
                    "text": "Freeze probe existing text.",
                    "kind": "fact",
                    "scope": "global",
                    "lifecycle": "active",
                    "validity": "valid",
                    "created_at": "2025-01-01T00:00:00+00:00",
                }
            ],
            "incoming": {
                "content": "Freeze probe incoming text.",
                "scope": "global",
                "kind": "fact",
            },
            "expect": {"material": "unrelated", "mutation": {"acceptable": [], "ideal": []}},
            "why": "freeze script coverage",
        }
    )
    monkeypatch.setattr(freeze_vectors, "load_cases", lambda: [case])

    texts = freeze_vectors._all_texts()

    assert "Freeze probe existing text." in texts
    assert "Freeze probe incoming text." in texts
