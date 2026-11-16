# 0005 — Hypomnema schema (discriminated union) + storage seam (legs-only, fusion owned above)

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted — **amended by** [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md) (`delete`→`bury` + `set_validity`; `+validity`, `+promoted_from`; `FactLifecycle` slot), [0007](0007-recall-pipeline-and-context-composition.md) (legs grow a `filters:{scope,validity,lifecycle_in}` object; retrieval feeds a `RecallResult` `degraded` envelope; `reinforce` emitted by write-path + fallback-revival only), and [0008](0008-dedup-gate-and-reinforcement.md) (`reinforce(id, at)`→append a reinforcement event, counters materialized; adapter grows aux-persistence ops `append_event`/`query_events`/`mark_distinct`/`is_distinct`; the dedup probe's `filters` scan broader than recall's — all lifecycle + both validity), [0017](0017-store-maintenance-and-ranked-visibility.md) (the seam grows a maintenance operation; the write-visibility clause becomes *ranked* visibility with an explicit staleness window), and [0018](0018-bounded-recall-no-unbounded-mode.md) (the retrieval legs take `k: int` — the `k=None` exhaustive clause is deleted).

Tabularium's core record and the interface every adapter sits behind. Two joined decisions from T13; recorded together because the seam's types are the record.

## The record: a discriminated union, not a flat nullable row

The domain model is a **pydantic discriminated union** — `Hypomnema = Annotated[Fact | Decision, Field(discriminator="type")]` over a shared `BaseHypomnema` — **not** one unified model with nullable per-type columns (project-b's `kind`-enum + nullable-columns shape, R05 §1). The union puts each type's invariants in the type system: a `Decision` carries `superseded_by` and frozen `content`; a `Fact` carries the decay machinery; an illegal combination will not construct. This enforces ADR 0001's thesis (decay = Facts-only, supersession = Decisions-only) *structurally* instead of via runtime validators — the exact conflation we are greenfielding away from.

**Field inventory** (reconciled against R05 §1, carry-vs-drop each):

- `BaseHypomnema`: `id: HypomnemaId` (UUID — adapter-portable, not autoincrement), `type` (discriminator), `content: str`, `scope: Scope` (slot; T04 fills), `facets: Facets` (slot; T12 fills), `source: str | None`, `recall_count: int`, `last_referenced_at: datetime | None`, `created_at`/`updated_at`.
- `Fact(Base)`: `lifecycle: FactLifecycle` (slot; enum is T05), `importance_score`, `feedback_score`, `hot_rejection_count`.
- `Decision(Base)`: `content` frozen (immutable — ADR 0001), `lifecycle: DecisionLifecycle` = `in_force | superseded | revoked`, `superseded_by: HypomnemaId | None`.
- **Ranking split:** shared *reference-tracking* (`recall_count`, `last_referenced_at`) on the base since both types track references; *decay machinery* (`importance_score`, `feedback_score`, `hot_rejection_count`, `FactLifecycle`) is Fact-only — so a Decision structurally cannot carry a decay field.
- **Dropped from project-b:** `subject_type/subject_id` (Eloquent morph; aboutness = content+facets+embedding), `immutable` column (structural now), `is_hot` bool (→ a `FactLifecycle` state), `deleted_at`/`*_reason` (→ lifecycle states; no hard delete).
- **Embedding is NOT on the domain model.** An embedding is a derived index over content, not domain data a caller reads. The vector lives below the seam only.

Storage is **one physical table**; the **seam owns the union↔row mapping** (union→row on write, discriminator-driven row→correct-subclass on read), so `get_by_id(id) -> Fact | Decision` hides the impedance behind a small surface — the mapping lives once at the seam (locality), not smeared across consumers branching on `.type`.

## The seam: raw retrieval legs; the fusion loop lives above it

The store seam exposes **raw retrieval legs** (`vector_search`, `text_search`), not a fused `recall`. The **RRF fusion loop** — plus importance-layering and type-branching — lives in the engine's RecallPipeline (T06), which is **Tabularium's own code**. This honors the engine-not-consumer principle (ADR 0003): "recall ranking/fusion" is a memory *semantic* Tabularium must own, not import. LanceDB still provides native ANN and native BM25 *indexes* for the legs (mechanisms below the seam); Tabularium declines only LanceDB's native *fusion* step (semantic above the seam) — that is the mechanism/semantic line.

**Embedding crosses the seam as a parameter, computed above it.** The engine embeds (litellm, ADR 0003) and passes the vector into `upsert(record, vector)`. Every store adapter is a **pure vector store** that never imports litellm. Dimensionality is a store-init config value (LanceDB/Qdrant fix vector width at table/collection creation; one instance commits to one embedder → one dim). An `embedding_model` provenance string on the row makes staleness detectable (rows whose model ≠ the configured embedder are re-embed candidates — the fix for project-b's silent dim-mismatch on model swap). Null-on-outage: a write survives with a null vector when the embedder is down, invisible to `vector_search` until a backfill op fills it.

**Interface** (async; `scope` required-kw-only on every retrieval; `Hit = {record, score, rank, leg}` so the engine's RRF consumes `rank`):

```
vector_search / text_search      — raw legs (non-null vectors only for vector_search; both scope-filtered)
upsert(record, vector)           — Decision content-immutability enforced here
reinforce(id, at)                — bump recall_count + last_referenced_at; never rewrites content
transition(id, lifecycle, ...)   — the ONLY path to change a Decision's state (supersede/revoke/promote)
delete(id, reason)               — Facts-only; refuses Decisions (they transition, never hard-delete)
get_by_id / scan                 — scan is metadata-only (eviction / decision-view / backfill discovery)
backfill_targets / backfill      — null-or-stale rows; engine re-embeds above, feeds vectors back
```

## Considered options

- **Unified `Hypomnema` + nullable per-type fields** (project-b-shaped) — rejected: the model lies about its invariants (a `Fact` with a `superseded_by`), pushing type rules into validators.
- **Fusion inside the store adapter** (use LanceDB's native hybrid) — rejected *for placement*: it imports a memory semantic into the one slot Tabularium must own (7217), and would force every adapter to replicate RRF. The below-seam simplicity was not worth ceding the differentiator.
- **Full capability-negotiation + above-seam strategy layer** (the "maximise flexibility" design-it-twice candidate) — rejected: its machinery existed only to delegate fusion natively, which owning-the-loop makes moot; its `native_hybrid` path carried a documented drift risk.
- **Discriminated union + single table + seam-owned mapping + legs-only seam + fusion above** — chosen. (Design-it-twice: three parallel architect proposals — minimise-surface / maximise-flexibility / optimise-common-caller — synthesized to common-caller ergonomics + minimise-surface rigor + fusion owned above.)

## Consequences

- Fusion is tested **once**, above the seam — the **in-memory adapter is the reference implementation** (cosine-leg + BM25-leg only, no RRF to replicate); it is the seam's honesty check. Two production adapters (LanceDB, Qdrant) + the reference = a real seam by the two-adapter rule.
- The eval harness (T01) swaps fusion strategies (RRF → weighted-linear → DBSF) trivially — it is Tabularium's own loop, zero adapter involvement.
- Cost: Tabularium forgoes LanceDB's in-process native fusion — two index queries + a Python RRF over a few hundred candidates. Negligible at Tabularium's scale.
- `similar_pairs` is **not** a seam op — T07's dedup gate builds pairwise near-duplicate detection on a scoped `vector_search` + `min_similarity` probe (the store stays a pure vector store; dedup semantics stay above the seam).
- Feeds the typed interfaces T06 (recall) / T07 (dedup) / T08 (eviction) build against; unblocks T12 (facets live on the record).
- Open slots firm as their tickets land: `Scope` (T04), `Facets` (T12), `FactLifecycle` enum (T05).
