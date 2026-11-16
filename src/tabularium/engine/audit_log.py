"""Structured audit-log helpers for the save, supersede and unsupersede paths.

Every helper logs through the ``tabularium`` logger, the name ``dispositions.py`` used before the
helpers moved here. Each emits one record whose fields are both ``extra`` and rendered into the
message text. These records are log lines only and never ``append_event`` calls.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

from tabularium.domain import HypomnemaId, Kind, Scope, SupersessionReason, UnsupersedeShape
from tabularium.engine.assessor import Method, UnavailableReason

_log = logging.getLogger("tabularium")


def log_failed_supersession(
    *,
    op: Literal["save", "supersede"],
    arm: Kind,
    successor_id: HypomnemaId,
    predecessor_ids: Sequence[HypomnemaId],
    reason: SupersessionReason,
    scope: Scope,
    error: Exception,
) -> None:
    """Log one WARNING, message ``supersession_failed``, when ``adapter.replace`` raises.

    Called once by each arm before it converts to ``ToolError``. It is a log line and never an
    ``append_event`` call. Every field is both ``extra`` and rendered into the message text, so a
    plain handler shows all eight.
    """
    error_class = type(error).__name__
    detail = str(error)
    predecessor_id_strs = [str(pid) for pid in predecessor_ids]
    _log.warning(
        "supersession_failed op=%s arm=%s successor_id=%s predecessor_ids=%s "
        "reason=%s scope=%s error=%s detail=%s",
        op,
        arm,
        str(successor_id),
        predecessor_id_strs,
        reason,
        str(scope),
        error_class,
        detail,
        extra={
            "op": op,
            "arm": arm,
            "successor_id": str(successor_id),
            "predecessor_ids": predecessor_id_strs,
            "reason": reason,
            "scope": str(scope),
            "error": error_class,
            "detail": detail,
        },
    )


AdmissionStep = Literal[
    "replay",
    "exact_key",
    "excluded_only",
    "below_floor",
    "survivors",
    "bm25_only",
    "vectorless",
    "resolve",
    "auto_reinforce",
    "auto_supersede",
    "auto_invalidate",
    "auto_write",
    "withheld",
]


def log_admission_resolved(
    *,
    arm: Kind,
    step: AdmissionStep,
    outcome: str,
    id: str,
    scope: Scope,
    span: Sequence[Scope],
) -> None:
    """Log one DEBUG, message ``admission_resolved``: the admission step that resolved this save.

    Called once on every return path of ``save_internal`` and ``save_decision``. ``id`` is a row
    or candidate id where one applies and ``"-"`` where none does. Every field is both ``extra``
    and rendered into the message text, so a plain handler shows all six.
    """
    span_str = ",".join(str(s) for s in span)
    _log.debug(
        "admission_resolved arm=%s step=%s outcome=%s id=%s scope=%s span=%s",
        arm,
        step,
        outcome,
        id,
        str(scope),
        span_str,
        extra={
            "arm": arm,
            "step": step,
            "outcome": outcome,
            "id": id,
            "scope": str(scope),
            "span": span_str,
        },
    )


def log_assessor_call(
    *,
    method: Method,
    outcome: Literal["verdict", "unavailable"],
    reason: UnavailableReason | None,
    latency_ms: int | None,
) -> None:
    """Log one DEBUG, message ``assessor_call``, per ``engine.admission.ask`` call.

    ``reason`` is ``None`` when ``outcome="verdict"``.
    """
    _log.debug(
        "assessor_call method=%s outcome=%s reason=%s latency_ms=%s",
        method,
        outcome,
        reason,
        latency_ms,
        extra={
            "method": method,
            "outcome": outcome,
            "reason": reason,
            "latency_ms": latency_ms,
        },
    )


def log_unsupersede_completed(
    *,
    arm: Kind,
    predecessor_id: HypomnemaId,
    successor_id: HypomnemaId,
    edge_id: str,
    shape: UnsupersedeShape,
) -> None:
    """Log one INFO, message ``unsupersede_completed``, per finished undo."""
    _log.info(
        "unsupersede_completed arm=%s predecessor_id=%s successor_id=%s "
        "edge_id=%s shape=%s",
        arm,
        str(predecessor_id),
        str(successor_id),
        edge_id,
        shape,
        extra={
            "arm": arm,
            "predecessor_id": str(predecessor_id),
            "successor_id": str(successor_id),
            "edge_id": edge_id,
            "shape": shape,
        },
    )


def log_unsupersede_failed(
    *,
    op: str,
    arm: Kind,
    predecessor_id: HypomnemaId,
    successor_id: HypomnemaId,
    edge_id: str,
    reason: str,
    error: Exception,
) -> None:
    """Log one WARNING, message ``unsupersede_failed``, per failed undo (the eight-field shape of
    ``log_failed_supersession``)."""
    error_class = type(error).__name__
    detail = str(error)
    _log.warning(
        "unsupersede_failed op=%s arm=%s predecessor_id=%s successor_id=%s "
        "edge_id=%s reason=%s error=%s detail=%s",
        op,
        arm,
        str(predecessor_id),
        str(successor_id),
        edge_id,
        reason,
        error_class,
        detail,
        extra={
            "op": op,
            "arm": arm,
            "predecessor_id": str(predecessor_id),
            "successor_id": str(successor_id),
            "edge_id": edge_id,
            "reason": reason,
            "error": error_class,
            "detail": detail,
        },
    )
