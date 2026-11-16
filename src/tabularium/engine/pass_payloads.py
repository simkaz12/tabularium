"""The JSON shapes stored in a pass-history row's ``payload`` besides ``CleanerReport``.

``clean.py`` builds them and serializes with ``json.dumps(model.model_dump(mode="json"))``;
``report.py`` reads them back with ``model_validate_json``. Field order is the stored key order.

The fields that readers of older rows may find missing have defaults: every field of
``RevisitCounts``, ``RevisitFallbackPayload``, ``FallbackPayload``, ``ErroredPayload`` and
``PayloadKind``, and every field of ``UnreadableBatchEntry``. ``BatchFailureEntry``,
``OmittedMarker`` and ``TrimmedEntry`` have required fields. Validating a stored string is lenient:
a field with a default takes it when the stored value has the wrong type, and unknown keys are
ignored. A required field that is missing or wrong-typed still fails, and ``FailedBatchEntry`` then
falls through to ``UnreadableBatchEntry``. Building a model in Python is strict.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
)

from tabularium.domain import BatchOutcome


class _PayloadModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    @field_validator("*", mode="wrap")
    @classmethod
    def _default_when_stored_value_is_invalid(
        cls, value: object, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
    ) -> object:
        try:
            return handler(value)
        except ValidationError:
            if info.mode != "json" or info.field_name is None:
                raise

            field = cls.model_fields[info.field_name]
            if field.is_required():
                raise

            return field.get_default(call_default_factory=True)


def _as_text(value: object) -> str:
    return value if isinstance(value, str) else str(value)


def _stringified_items(value: object) -> object:
    if not isinstance(value, list | tuple):
        return value

    return [str(item) for item in value]


def _entry_objects(value: object) -> object:
    if not isinstance(value, list | tuple):
        return []

    return [entry for entry in value if isinstance(entry, dict | _PayloadModel)]


class BatchFailureEntry(_PayloadModel):
    batch_index: int
    judge_a_outcome: BatchOutcome
    judge_b_outcome: BatchOutcome
    error: str


class OmittedMarker(_PayloadModel):
    omitted_count: int


class UnreadableBatchEntry(_PayloadModel):
    batch_index: object = None
    judge_a_outcome: object = None
    judge_b_outcome: object = None
    error: object = None


FailedBatchEntry = OmittedMarker | BatchFailureEntry | UnreadableBatchEntry
FailedBatches = Annotated[tuple[FailedBatchEntry, ...], BeforeValidator(_entry_objects)]


class RevisitCounts(_PayloadModel):
    shelve: int = 0
    keep: int = 0
    unclear: int = 0
    contested: int = 0


class TrimmedEntry(_PayloadModel):
    old_id: str
    new_id: str | None


class RevisitFallbackPayload(_PayloadModel):
    pool_size: int = 0
    counts: RevisitCounts = RevisitCounts()
    split_keep: int = 0
    shelved_ids: tuple[str, ...] = ()
    contested_ids: Annotated[tuple[str, ...], BeforeValidator(_stringified_items)] = ()
    protected_count: int = 0
    skipped_no_evidence_count: int = 0
    skipped_too_large_count: int = 0
    failed_batch_count: int = 0
    failed_batches: FailedBatches = ()
    verification_failure_count: int = 0
    review_append_failure_count: int = 0
    refused_count: int = 0
    trimmed_count: int = 0
    trimmed: tuple[TrimmedEntry, ...] = ()


class FallbackPayload(_PayloadModel):
    watermark_before: str | None = None
    watermark_after: str | None = None
    pool_size: int = 0
    reason: Annotated[str, BeforeValidator(_as_text)] = ""
    relevance_failed_batches: FailedBatches = ()
    falsity_failed_batches: FailedBatches = ()
    revisit: RevisitFallbackPayload | None = None
    revisit_error: str | None = None


class ErroredPayload(_PayloadModel):
    error: Annotated[str, BeforeValidator(_as_text)] = "unknown error"


class PayloadKind(_PayloadModel):
    kind: object = None
