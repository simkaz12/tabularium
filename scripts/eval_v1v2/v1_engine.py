from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from eval_v1v2.rows import CorpusRow, NoiseRow
from tabularium.clock import Clock, SystemClock
from tabularium.config import EngineConfig, build_assessor_from_env
from tabularium.domain import Decision, Fact, HypomnemaId, Kind, parse_scope
from tabularium.embedding import LitellmEmbedder
from tabularium.engine import Tabularium
from tabularium.engine.assessor import Assessor
from tabularium.storage.lancedb_adapter import LanceDBAdapter

EVAL_SCOPE = "eval"
EMBEDDING_MODEL = "ollama/nomic-embed-text"
EMBEDDING_DIM = 768
FORBIDDEN_ROOT = "/var/lib"
FACT_LIFECYCLES = frozenset({"active", "hot", "ember"})
LOG_EVERY = 250
TIMESTAMP_FIELDS = ("created_at", "updated_at", "last_referenced_at")
CARRIED_SKIP = frozenset(
    {"id", "type", "kind", "content", "scope", "lifecycle", "vector", "recall_count", "created_at"}
)
SNAPSHOT_MARGIN = timedelta(seconds=1)


class FixedClock:
    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment


@dataclass
class V1Harness:
    engine: Tabularium
    adapter: LanceDBAdapter
    embedder: LitellmEmbedder
    assessor: Assessor | None
    dir: Path


def assert_temp_path(path: str | Path) -> Path:
    target = Path(path).expanduser()
    real = target.resolve()
    root = Path(tempfile.gettempdir()).resolve()
    inside = real == root or root in real.parents

    if not inside or str(real).startswith(FORBIDDEN_ROOT):
        raise ValueError(f"store path {real} must be under {root} and never under {FORBIDDEN_ROOT}")

    return real


def require_assessor_key() -> None:
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("TYPESAFE_API_KEY is not set; the v1 save replay needs the Jev Assessor")


async def open_v1(
    store: Path | None = None,
    *,
    use_assessor: bool = True,
    clock: Clock | None = None,
) -> V1Harness:
    directory = assert_temp_path(store) if store is not None else _fresh_dir()
    directory.mkdir(parents=True, exist_ok=True)

    config = EngineConfig()
    clock = clock if clock is not None else SystemClock()
    embedder = LitellmEmbedder()
    adapter = await LanceDBAdapter.open(
        str(directory),
        embedding_model=embedder.model,
        dim=embedder.dim,
        clock=clock,
    )
    assessor = build_assessor_from_env(config.assessor_model) if use_assessor else None
    engine = Tabularium(adapter, embedder, clock, config, assessor=assessor)

    await engine.initialize()

    return V1Harness(engine, adapter, embedder, assessor, directory)


def _fresh_dir() -> Path:
    return assert_temp_path(tempfile.mkdtemp(prefix="v1v2-eval-v1-"))


async def register_scopes(harness: V1Harness, names: Iterable[str]) -> None:
    for name in sorted(set(names)):
        if name != "global":
            await harness.adapter.create_scope(parse_scope(name))


def parse_moment(raw: str) -> datetime:
    moment = datetime.fromisoformat(raw)

    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def build_record(
    *,
    record_id: uuid.UUID,
    kind: Kind,
    content: str,
    scope: str,
    created_at: datetime,
    recall_count: int,
    lifecycle: str,
    carried: Mapping[str, Any] | None = None,
) -> Fact | Decision:
    arm = Decision if kind == "decision" else Fact
    accepted = {
        name: value
        for name, value in (carried or {}).items()
        if name in arm.model_fields and name not in CARRIED_SKIP and value is not None
    }
    common = {
        "id": HypomnemaId(record_id),
        "content": content,
        "scope": parse_scope(scope),
        "recall_count": recall_count,
        "created_at": created_at,
        "updated_at": created_at,
        "embedding_model": EMBEDDING_MODEL,
        **accepted,
    }

    if kind == "decision":
        return Decision(lifecycle="in_force", **common)

    state = lifecycle if lifecycle in FACT_LIFECYCLES else "active"

    return Fact(lifecycle=state, **common)  # type: ignore[arg-type]


def record_from_noise(row: NoiseRow, scope: str = EVAL_SCOPE) -> Fact | Decision:
    return build_record(
        record_id=uuid.UUID(row.id),
        kind=row.kind,
        content=row.content,
        scope=scope,
        created_at=parse_moment(row.created_at),
        recall_count=row.recall_count,
        lifecycle=row.lifecycle,
    )


def record_from_corpus(row: CorpusRow) -> Fact | Decision:
    return build_record(
        record_id=uuid.UUID(row.id),
        kind=row.kind,
        content=row.content,
        scope=row.scope,
        created_at=parse_moment(row.created_at),
        recall_count=row.recall_count,
        lifecycle=row.lifecycle,
        carried=row.model_extra,
    )


def snapshot_clock(rows: Iterable[CorpusRow]) -> FixedClock:
    moments: list[datetime] = []

    for row in rows:
        for name in TIMESTAMP_FIELDS:
            raw = getattr(row, name, None)

            if raw:
                moments.append(parse_moment(raw))

    if not moments:
        raise ValueError("corpus carries no timestamps to pin the clock to")

    return FixedClock(max(moments) + SNAPSHOT_MARGIN)


async def load_noise(harness: V1Harness, rows: list[NoiseRow], log: Callable[[int], None]) -> None:
    await register_scopes(harness, [EVAL_SCOPE])

    for done, row in enumerate(rows, start=1):
        await harness.adapter.upsert(record_from_noise(row), row.vector)

        if done % LOG_EVERY == 0:
            log(done)


async def load_corpus(
    harness: V1Harness, rows: list[CorpusRow], log: Callable[[int], None]
) -> None:
    await register_scopes(harness, [row.scope for row in rows])

    for done, row in enumerate(rows, start=1):
        await harness.adapter.upsert(record_from_corpus(row), row.vector)

        if done % LOG_EVERY == 0:
            log(done)


async def count_records(harness: V1Harness) -> int:
    total = 0

    async for _ in await harness.adapter.scan_records():
        total += 1

    return total
