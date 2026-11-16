# Tabularium — Canonical SPEC

> **Frozen 2026-08-07.** Historical prior — it records the amendment-resolved design intent as
> of the time, is not maintained, and diverges from the built system in places. There is no
> present-truth document: the code is what is true now, and new architectural decisions are
> recorded in the Tabularium store, not as new ADR files. `README.md` orients.

This is the **amendment-resolved normative end-state**: a builder reads only this document to know WHAT to build. Every ADR amendment chain (0005 amended by 0006/0007/0008/0013/0015/0016; 0006 by 0020 (hot's earned-only half only); 0007 by 0008/0012/0015/0016; 0008 by 0009; 0009 by 0012; 0012 by 0013/0015/0016; 0004 by 0005) is resolved here — the ADRs remain the WHY-archive, linked per section, never authoritative over this text. Items pinned during assembly (not traceable to a single ADR clause) are marked **[A#]** and indexed in §17.

Domain vocabulary is fixed by [CONTEXT.md](../CONTEXT.md) (the glossary); this SPEC uses it without redefining it.

---

## 1. Domain model

*(WHY: [ADR 0001](adr/0001-decision-immutable-content-mutable-lifecycle.md), [ADR 0005](adr/0005-hypomnema-schema-and-storage-seam.md), [ADR 0006](adr/0006-fact-taxonomy-validity-and-monotonic-dispositions.md), [ADR 0013](adr/0013-decision-record-view.md), [ADR 0016](adr/0016-curation-audience-routing.md), [ADR 0020](adr/0020-hot-tier-owner-curated-not-earned.md))*

The record is a **pydantic discriminated union** — `Hypomnema = Annotated[Fact | Decision, Field(discriminator="type")]` — never a flat nullable row. Illegal combinations must not construct: a Decision structurally cannot carry decay machinery; a Fact cannot carry supersession fields.

### 1.1 Final field inventory

```python
HypomnemaId = <UUID newtype>            # adapter-portable, never autoincrement
Scope       = NewType("Scope", str)     # §2

class BaseHypomnema(BaseModel):
    id: HypomnemaId
    type: Literal["fact", "decision"]           # discriminator
    content: str
    scope: Scope
    purpose: str | None = None                  # registered facet value or None (§10)
    audience_hint: list[str] | None = None      # role names; additive-only on pull, routes the push tiers (§7)
    source: str | None = None                   # free provenance (agent identity rides here)
    validity: Literal["valid", "invalid"] = "valid"   # bidirectional truth axis
    reconciles: HypomnemaId | None = None       # set on a valid record written past an invalid match (§4.5)
    recall_count: int = 0                       # materialized projection of the event log (§8.2)
    last_referenced_at: datetime | None = None  # materialized projection; None ⇒ treat as created_at
    created_at: datetime
    updated_at: datetime

class Fact(BaseHypomnema):
    lifecycle: Literal["active", "hot", "ember", "archived", "shelved"]  # shelved: ADR 0021
    valence: Literal["correction", "guideline"] | None = None   # ember-born Facts only [A1]
    feedback_score: float = 0.0                 # EWMA projection ∈ [0,1] (§8.1)
    hot_rejection_count: int = 0                # observability only; drives no transition
    decay_due_at: datetime | None = None        # precomputed archive-crossing time (§8.3)
    owner_recurrence_count: int = 0             # r_owner — §8.4 rung-2 crossing input [ADR 0020 amendment]
    agent_recurrence_count: int = 0             # keeps an ember warm; never arms rung 2 (§8.4)
    escalation_baseline: int = 0                # re-arm point for r_owner's crossing test (§8.4/§8.5)
    provisional: bool = False                   # owner-verdict-outstanding marker (§8.4 amendment)

class SupersedesSnapshot(BaseModel):
    id: HypomnemaId
    title: str                                  # derived from predecessor content at snapshot time
    created_at: datetime

class PromotedFromSnapshot(BaseModel):
    id: HypomnemaId
    content: str
    valence: Literal["correction", "guideline"]
    recurrence_count: int

class Decision(BaseHypomnema):
    content: str                                # FROZEN — immutable, never edited in place
    lifecycle: Literal["in_force", "superseded", "revoked"]
    superseded_by: HypomnemaId | None = None    # forward pointer, set exactly once
    supersession_chain: list[SupersedesSnapshot] = []   # full backward history, birth-frozen
    promoted_from: list[PromotedFromSnapshot] = []      # justification snapshots, frozen at promotion
```

Normative resolutions folded in:

- **No stored `importance_score`.** ADR 0005 listed it; ADR 0009 removes it — importance is always computed lazily (§8.1). The stored inputs are `recall_count`, `feedback_score`, `last_referenced_at`, plus derived `decay_due_at`.
- **No stored `title` on any record** — titles are always derived (first sentence of content, truncated ~80 chars). The `title` field inside `SupersedesSnapshot` is that derivation captured at snapshot time (a frozen copy), not a record column.
- **Embedding is not domain data.** The vector lives below the seam only; an `embedding_model` provenance string sits on the stored row (not the pydantic model) so staleness is detectable (§3.4).
- **`facets` slot resolution:** ADR 0005's open `Facets` slot resolves to exactly two columns — `purpose` + `audience_hint`. No generic facet container exists.
- **`valence` physical form [A1]:** semantically "rides `source`" (ADR 0006); physically a nullable column on the Fact arm so `PromotedFromSnapshot` and the eval harness can read it without parsing `source`.
- **Dropped from project-b, permanently:** `subject_type/subject_id`, `immutable` column, `is_hot` flag, `deleted_at`/`*_reason` soft-delete, owner/user columns of any kind.

### 1.2 The three orthogonal axes

1. **Lifecycle** (attention): Fact `active | hot | ember | archived | shelved` **[Amended, ADR 0021 — `shelved` added]**; Decision `in_force | superseded | revoked`.
2. **Validity** (truth): `valid ⇄ invalid`, both arms, bidirectional, metadata about frozen content — never an edit of it.
3. **Valence** (ember direction): `correction | guideline`, embers only.

Legal Fact edges: born `active` or `ember` (never `hot`); `{active,ember}→hot` **[Amended 2026-07-31 — ADR 0020; further amended same day — ember/hot rationale]** only via the owner-curated `set_hot` tool, never earned by an importance sweep — `hot` is always-on (injected into every session-start payload), so an automatic promotion would be permanent context bloat; the `ember→hot` edge is §8.4 rung 2's **Foreground** exit (an owner-curated act, not a computed one — the recurrence count that motivates it is informational, never a gate `set_hot` itself enforces); `hot→active` via `set_hot` (the deliberate demotion — **never back to `ember`**, ember is a write-time-seeded provisional state and a record that has been through the ladder is not) or as **housekeeping**, a structural side effect of burial/invalidation/(future) decay removing the record from the tier's presented set without any policy re-judging it; `{active,ember}→archived` decay; `archived→active` revival. **[Amended, ADR 0021]** `{active,archived}→shelved` via the owner-curated `set_shelved` tool (bidirectional with `shelved→active`, which always lands on `active`, never restoring the prior lifecycle) — `hot`/`ember` cannot be shelved directly, and `shelved→hot` directly via `set_hot` is illegal (un-shelve first); see §4.1's probe-exclusion amendment and §9.1's `set_shelved` signature. Decision edges: `in_force→superseded` (via a superseding save) and `in_force→revoked`; both permanent. The `in_force + invalid` cell is legal and meaningful.

### 1.3 Dispositions (monotonic archive)

Most-retained → least: **archive** (Facts, reversible) · **invalidate** (both arms, keeps whole record) · **bury** (both arms, drops content + every derived index, always writes a **tombstone**: id, type, reason, timestamp, model-generated flag). **No true hard-delete exists.** Burial is explicit and human-decided, never automated. Automation may only re-rank or *suggest* — it never auto-retires or deletes a Decision.

### 1.4 Choosing the kind

`kind` is chosen once by the caller at `save()` and is immutable thereafter (§3.1: `transition()` mutates `lifecycle` only; there is no kind migration). It fixes which arm's machinery the atom receives for life, and which retrieval surface can reach it — `recall()` returns Facts only; `decisions_query()` and `decision://{id}` return Decisions only. A misfiled atom is not merely mislabelled: it is unreachable from the surface its content belongs to.

**The discriminator.** One question: *what would make this stop being true?*

- **"We choose otherwise, and record that we did"** → `decision`.
- **"The world moves on without anyone recording it"** → `fact`.

**Confirming tests.** Both must agree with the discriminator.

1. **Road-not-taken.** Can you name what was rejected? A Decision has alternatives that were live at the moment of choosing. A Fact has none — the world is not a menu.
2. **Supersession authority.** If this turns out wrong, would the correction be a record *in this store* that cites and overturns it? Decisions are overturned from inside. Facts are falsified from outside — a refactor, a vendor change, a version bump — with nothing to cite.

| Content | kind | Why |
|---|---|---|
| Burial always leaves a tombstone. | `decision` | Rejected: hard delete. Overturned only by a superseding record. |
| `bury()` deletes the row; `_decision_view` raises on the id. | `fact` | Describes current code. A refactor falsifies it silently. *(Also likely fails the derivability bar — see below.)* |
| Vector store is LanceDB. Rejected: Qdrant, hand-rolled cosine. | `decision` | Named alternatives; ours to overturn. |
| LanceDB is pre-1.0 and ships breaking changes on minor bumps. | `fact` | Nobody chose this. Upstream falsifies it. |
| LanceDB is version-pinned (ADR 0004). | `decision` | The response to the fact above. Note these are **two atoms**. |
| The server service does not restart on `git pull`. | `fact` | Environment gotcha; a unit-file change falsifies it with nothing to cite. |
| Changelog entries are one sentence per merged pull request. | `decision` | Rejected: free-form entries. Stops holding only when the maintainers say otherwise — never by age. |
| The Curator is deferred out of the grounding slice. | `decision` | Sequencing is chosen. Rejected: ship it in slice C. |

**Mixed atoms.** If the two confirming tests disagree, the content holds a state claim *and* a choice. Split it and save both. One atom, one kind.

**Recurrence is not a discriminator test.** A Fact that keeps mattering does not become a Decision by repetition. If it recurs because it is a standing rule rather than a choice, its exit is the hot tier (§8.4), not codification.

**Derivability cross-check.** Most agonised fact-or-decision calls are really save-or-don't calls. If the atom describes what the code currently does, apply the derivability bar first — a reader with the repo can find that out by looking, so it should not be saved at all. The bar cuts across the kind split, not along it ([CONTEXT.md](../CONTEXT.md), *Derivability bar*).

**Default.** `kind="fact"`. The two errors are not symmetric: a Fact that should have been a Decision can be re-saved as one, because Facts carry no immutability contract. A Decision that should have been a Fact is unrepairable — content is frozen, kind cannot migrate, `recall()` will never surface it, and burial is the only exit.

### 1.5 Session — a record that is not a hypomnema



```python
class Session(BaseModel):
    id: UUID
    scope: str
    started_at: datetime
    ended_at: datetime
    content: str
    source: str | None = None
```

A `Session` is an append-only work-session digest — the prose narrative of a work session, one row per
`digest()` call. It is explicitly **not** a hypomnema: it carries **no lifecycle** and **no validity**,
does not subclass `BaseHypomnema`, and never appears in the `Hypomnema = Fact | Decision` discriminated
union. It is **never embedded** — no vector column exists on its storage row at all, not even a nullable
one (§3.3) — and it is therefore never deduped, never decayed, and never reachable from `recall`,
`decisions_query`, or `context`.

**Why not a fourth `kind`.** `kind` exists to select *embedding-driven* machinery — dedup, recall
ranking and importance-decay are all cosine-based (§1.4's discriminator: *what would make this stop
being true?*, answered once per record at `save()` and fixed for life). A Session opts out of every
piece of that machinery at once: nothing about it ever stops being true (it is a closed, immutable
record of what happened), so forcing it through the `kind` discriminator would consist entirely of
declining what `kind` exists to select. It is therefore its own module — parallel seam methods (§3),
own storage table (§3.3), own MCP tools (`digest`/`sessions`, §9.1) — rather than a third arm on the
`Hypomnema` union.

The session↔hypomnema relationship is **computed, never stored**: a `Session` carries `scope` + a time
range; every `Fact`/`Decision` already carries `scope` + `created_at`; which hypomnema a session
"covers" is a range predicate, not a foreign key, join table, or `produced: [ids]` list.

---

## 2. Scopes

*(WHY: [ADR 0015](adr/0015-owner-scope-seam.md))*

- **No owner concept.** One deployment = one store directory = one human's archive. No user table, owner column, principal argument, or resolver anywhere. Agents are agents of the same owner; agent identity rides `source`. Multi-owner = a second store.
- **`Scope = NewType('Scope', str)`**, pattern `^[a-z][a-z0-9-]{0,31}$`, no normalization. Wire shape stays `scope: str`.
- **Two tiers only:** the reserved owner-wide scope **`global`** + flat peer named scopes. No nesting, no paths, ever.
- **Granularity = working context** (product/effort), never a repo. Cross-stack pairs share one scope; repo→scope binding is client-side config; the engine never knows what a repo is.
- **Read span:** a read in scope X (`recall`, `decisions_query`, `context`) sees **X + `global`**, always, no flag; a read on `global` sees `global` alone. Hits are labeled with their origin scope. Strict single-scope reads exist only as an engine-internal parameter (eval fixtures), never model-facing.
- **Write span:** a write targets exactly one scope; `global` is an ordinary explicit target.
- **Registered vocabulary:** `tabularium scope add <name>` (CLI, human act; `global` pre-registered). Any model-facing call naming an unregistered scope → `ToolError`.
- **Hot tiers are per-scope arenas** (including `global`); eviction contests never cross scopes.

---

## 3. Storage seam (canonical adapter interface)

*(WHY: [ADR 0004](adr/0004-vector-store-lancedb-swappable-seam.md), [ADR 0005](adr/0005-hypomnema-schema-and-storage-seam.md) + its amendments)*

Two adapters: **LanceDB** (default; pinned, pre-1.0), **in-memory** (the reference implementation and hermetic-test backend). **[Amended 2026-08-02 — ADR 0004 amendment.]** The Qdrant adapter is cut, not deferred to M4: its only justification was proving the seam is real, and the seam already proves itself — the two adapters above share one dual-parameterized conformance suite that has already caught real defects. Revival trigger: a second deployment target appears, embedded LanceDB hits a concurrency or scale wall (it has no concurrency mediation today and runs single-process), or a pre-1.0 LanceDB upgrade breaks compatibility. The adapter is **chosen at store-open via config** — a seam + config selection, not hot-swap machinery. Every adapter is a **pure vector store**: it never imports litellm, never fuses, never ranks by importance.

### 3.1 The interface (async; `filters` required on every retrieval)

```python
class Filters(BaseModel):
    scopes: list[Scope]                 # engine-computed span (read: [X, global]; dedup probe: same; global: [global])
    validity: set[Literal["valid", "invalid"]]
    lifecycle_in: set[str]              # per-arm lifecycle values to include

class Hit(BaseModel):
    record: Fact | Decision
    score: float
    rank: int
    leg: Literal["vector", "text"]

class StorageAdapter(Protocol):
    # retrieval legs — raw, never fused
    async def vector_search(self, vector: list[float], *, filters: Filters, k: int) -> list[Hit]: ...
    async def text_search(self, query: str, *, filters: Filters, k: int) -> list[Hit]: ...
    # k is required — there is no unbounded mode (ADR 0018). vector_search sees non-null-vector rows only.

    # writes
    async def upsert(self, record: Hypomnema, vector: list[float]) -> None: ...
        # Decision content-immutability enforced here; vector required on the normal path [A2]
    async def transition(self, id: HypomnemaId, lifecycle: str, *, at: datetime,
                         superseded_by: HypomnemaId | None = None,
                         audience_hint: tuple[str, ...] | None = None) -> None: ...
        # the ONLY path to change lifecycle; persists updated_at = at. updated_at ≡ time of last
        # lifecycle transition (init created_at); set_validity and update_projections never touch it —
        # this is what makes the render's superseded-date truthful [M1 amendment 2026-07-23].
        # Supersession is one atomic call:
        # new.supersession_chain = old.supersession_chain + [snapshot(old)]; old.superseded_by = new.id
        # Explicit typed keyword-only params, no **kw bag [storage-seam-explicit-typing, 2026-08-02] —
        # an unsupported keyword is now a static (mypy) and runtime (TypeError) error, never a silent no-op.
        # audience_hint tri-state — the whole contract:
        # None = leave the stored value untouched (silence, not an opinion); a tuple = replace it;
        # () = clear it (restoring universal reach). None and () are NOT the same request — conflating
        # them would make un-hinting impossible. Persisted in the SAME row write as lifecycle/updated_at.
        # This is the routing key for the two push tiers (§7); it never touches recall (§5, ADR 0016).
    async def set_validity(self, id: HypomnemaId, valid: bool, reason: str) -> None: ...
    async def bury(self, id: HypomnemaId, reason: str) -> TombstoneId: ...
        # drops content AND the vector row AND the FTS row; writes the tombstone

    # reads
    async def get_by_id(self, id: HypomnemaId) -> Fact | Decision: ...
    async def scan(self, ...) -> AsyncIterator[RowMeta]: ...   # metadata-only; engine-internal maintenance family
    async def scan_records(self, ...) -> AsyncIterator[Fact | Decision]: ...   # content-carrying sibling of scan;
        # same lifecycle_in/buried-exclusion contract, but yields full records — collapses a scan-then-per-id
        # get_by_id round trip into one round trip (push-side selection, §7)
    async def get_tombstone(self, id: HypomnemaId) -> Tombstone | None: ...   # tombstone read-back (sync prune, doctor) [M1 amendment 2026-07-23]

    # event log + pair registry (aux persistence; ids and timestamps only, never content)
    async def append_event(self, event: ReinforcementEvent) -> None: ...
    async def query_events(self, *, target_id: ... , source: ...) -> list[ReinforcementEvent]: ...
    async def update_projections(self, id: HypomnemaId, *, recall_count, last_referenced_at,
                                 feedback_score, decay_due_at) -> None: ...          # [A3]
    async def update_recurrence(self, id: HypomnemaId, *, owner_recurrence_count,
                                agent_recurrence_count, escalation_baseline) -> None: ...
        # Facts-only; §8.4 counters — verbatim-persist, same [A3] contract as update_projections,
        # kept a separate call because these three fields are not §8.2 projections
    async def mark_distinct(self, pair: DistinctPair) -> None: ...   # unordered UNIQUE, first-judgment-wins, idempotent
    async def is_distinct(self, a: HypomnemaId, b: HypomnemaId) -> bool: ...

    # embedding maintenance
    async def backfill_targets(self) -> list[HypomnemaId]: ...   # rows whose embedding_model ≠ configured embedder
    async def backfill(self, id: HypomnemaId, vector: list[float]) -> None: ...

    # registries (same aux family)
    async def create_scope / list_scopes / create_facet / list_facets / create_role / list_roles: ...
    async def delete_role(self, name: str) -> None: ...   # roles-only deregistration verb — idempotent no-op if unregistered; never touches audience_hint

    # maintenance (ranked visibility — ADR 0017)
    async def maintenance_debt(self) -> MaintenanceDebt: ...   # pure read: num_small_fragments, num_rows, num_unindexed_rows
    async def upkeep(self) -> None: ...   # folds pending writes into every index + compacts fragments; never prunes
    async def reclaim(self) -> None: ...   # IRREVERSIBLE version pruning; buys only disk; never automatic;
        # reachable only by name — never a parameter on upkeep(); no-other-writer precondition (docs/backup.md)

    # session methods (own table, parallel seam — 2026-08-02 sessions plan; §1.5)
    async def append_session(self, session: Session) -> None: ...   # append-only, no update/delete leg
    async def latest_session_end(self, *, scope: str) -> datetime | None: ...
        # per-scope max(ended_at); None if scope has no rows yet
    async def search_sessions(self, *, scopes: list[str], query: str | None,
                              since: datetime | None, until: datetime | None,
                              limit: int) -> list[Session]: ...
        # these three deliberately do NOT take Filters: Filters carries validity/lifecycle_in,
        # both meaningless for a record with no lifecycle and no validity (§1.5); a Session's
        # only retrieval predicates are scope-span + time-range + optional BM25 term, so a
        # narrower `scopes` list plus explicit kwargs says exactly that, instead of widening
        # Filters with fields half the callers must pass as empty/no-op
```

**Engine-computed, seam-persisted [A3]:** all projection formulas (EWMA, `decay_due_at`) live above the seam; the engine appends the event then persists recomputed projection fields via `update_projections`. `reinforce(id, at)` in earlier ADRs = `append_event` + `update_projections` composed by the engine.

### 3.2 What lives above the seam (never in an adapter)

RRF fusion · importance computation and boost · the dedup verdict/guards/dispatch · lifecycle trigger logic · token counting/packing · the Curator · rendering. The seam carries **mechanisms** (indexes, rows, log persistence); Tabularium owns **semantics**.

### 3.3 Storage layout and conformance rules

- **One physical table**, seam-owned union↔row mapping (union→row on write; discriminator-driven row→subclass on read).
- `scope` is an **indexed filter column**; filters are pushed down as native metadata **pre-filters** (never over-fetch-then-drop above the seam).
- **Isolation is interface-shaped:** `filters`/`scope` required-kw-only on every retrieval and model-facing engine method; the only scopeless family is engine-internal maintenance (`scan`/backfill/sweep).
- **Ranked-visibility conformance rule [A4] (ADR 0017):** a committed write must be visible to **both** legs on the next same-connection search — and correctly *ranked*, not merely present. The contract names its staleness window rather than denying one:

  > After maintenance, every record ranks correctly. Between maintenance runs, records written since the last run may rank stale.

  `upkeep()` is the operation that closes the window (compaction + index fold-in, across every table, always passing an explicit long `cleanup_older_than` so it never prunes); `maintenance_debt()` reports how much of the window is currently open. How an engine-specific quirk is absorbed (e.g. LanceDB FTS index is a build-time snapshot) is the adapter's problem; the shared conformance suite (anchored by the in-memory reference, which is never stale) asserts the property across both adapters. *When* `upkeep()` fires is store-health policy above the seam, not this section's concern.
- **Content lives in exactly one place:** the event log, pair registry, counters, and tombstone `reason` carry ids/timestamps, never content — this is what makes `bury` complete (§10.4).
- **`sessions` is the first non-`records` table carrying a full-text index** (2026-08-02 sessions plan). It has no vector column at all, not even a nullable one — BM25 on `content` composes with a scalar range predicate on `ended_at` to serve `search_sessions` (§3.1, §1.5) without ever touching `vector_search`.

### 3.4 Embedding

The engine embeds (litellm, §11) and passes the vector to `upsert`. Vector dimensionality is store-init config (one store = one embedder = one dim). Each row carries `embedding_model`; rows whose value ≠ the configured embedder are backfill candidates. **[A2]** `upsert` requires a vector on the normal path — an embedder outage **parks** the write (§4.7); the ADR 0005 null-on-outage write is superseded by ADR 0012 B2b. Backfill ops serve model-version re-embeds only.

---

## 4. Write path — the dedup gate

*(WHY: [ADR 0008](adr/0008-dedup-gate-and-reinforcement.md), [ADR 0015](adr/0015-owner-scope-seam.md) Q4, [ADR 0016](adr/0016-curation-audience-routing.md) B8.5)*

Every `save`/`ember` runs the gate **before writing**: probe → classify best match → dispatch to a per-case process. A shallow classifier in front of deep per-case processes.

### 4.1 Probe

- **Full hybrid**: `vector_search` ∪ `text_search`, **RRF for candidate selection only** — never a blended score driving the verdict. BM25 is a recall aid, never a precision signal.
- **Probe span = read span** (save into X probes X + `global`; save into `global` probes `global` alone; siblings never probe each other). Cross-tier verdicts are normal.
- **Scan filters broader than Recall's:** all lifecycle states + both validity values — "have we seen this before, in any state?" **[Amended, ADR 0021]** — except `shelved`: the probe's one deliberate blind spot. A curator judgement that a Fact is not worth carrying (`set_shelved(id, shelved=True, reason=...)`) withdraws the gate's defence of that content — a later save of the same atom is free to succeed — without hiding the content itself (`shelved` stays in Recall's primary filter, §5). `archived` is unaffected by this amendment and does not absorb `shelved`: `archived` is cold-but-still-authoritative (revivable, still deduped against); `shelved` is judged-not-worth-carrying (never revived, never deduped against).
- The **importance boost is absent** from the probe (identity, not value).
- The gate **ignores `purpose` and `audience_hint`** entirely — the probe is content-based; same content, different facet = still a duplicate.

### 4.2 Verdict

Anchored on **raw cosine** (interpretable, per-embedder-calibratable), guarded by near-free pure-Python **discriminative checks** — numeral / entity / version / negation-cue diff — which veto an auto-reinforce even at ~0.99 cosine:

- `cosine < gray_floor` → **pass** (write novel).
- `gray_floor ≤ cosine < auto_ceiling` → **bounce**.
- `cosine ≥ auto_ceiling`, no guard trip, Fact → **auto-reinforce**.
- A **guard trip downgrades** reinforce → bounce (a downgrade rule, not a fourth outcome).

The distinct-pair registry is consulted first: a pair already judged distinct never re-bounces.

### 4.3 Dispatch matrix

| Best match | Process | Action |
|---|---|---|
| valid Fact (active/hot) | common | reinforce / bounce / pass |
| valid ember | recurrence | reinforce + recurrence counter (`source=dedup` events) → escalation ladder §8.4 |
| valid archived Fact | revival | reinforce as **in-place** `archived→active`, same id, history preserved |
| Decision | supersession | **bounce eagerly** (lower `gray_floor` than Facts); never auto-reinforce |
| guard trip | false-merge defense | bounce + surface the differing token |
| invalid (any arm) | reconciliation | **non-blocking**: the valid save always writes, append-only, carrying `reconciles: <invalid_id>`; the invalid record is never touched |

Asymmetry (normative): revival is an in-place transition of the same record; reconciliation is a new record — the gate never auto-flips a truth bit. Facts are **precision-favored on auto-reinforce** (high `auto_ceiling`); Decisions are **recall-favored on the bounce** (low `gray_floor`) because a Decision false-pass corrupts "what's in force".

### 4.4 Bounce (agent adjudication — no internal LLM, ever)

The gray band returns a structured bounce to the calling agent; **nothing is written until the agent acts** (an ignored bounce is a safe failure).

```python
class BounceNeighbor(BaseModel):
    candidate_id: HypomnemaId
    similarity: float
    guard_flag: str | None

class BounceExit(BaseModel):
    action: Literal["reinforce", "confirm_distinct", "supersede", "promoted_from"]
    how: str
    when: str

class FactBounce(BaseModel):
    bounce_kind: Literal["fact"]
    candidate_id: HypomnemaId
    candidate_content: str
    candidate_scope: str                # [A5] — the agent sees a cross-tier match for what it is
    similarity: float
    guard_flag: str | None
    verdict: Literal["near_dup", "decision_adjacent", "guard_trip"]
    exits: tuple[BounceExit, BounceExit]                 # always reinforce, confirm_distinct
    other_candidates: list[BounceNeighbor] = []   # other above-floor neighbours, no content

class DecisionBounce(BaseModel):
    bounce_kind: Literal["decision"]
    candidate_id: HypomnemaId
    candidate_content: str
    candidate_scope: str
    similarity: float
    guard_flag: str | None
    verdict: Literal["near_dup", "decision_adjacent", "guard_trip"]
    exits: tuple[BounceExit, BounceExit] | tuple[BounceExit, BounceExit, BounceExit]
    # three (+ supersede or promoted_from) when the candidate is promotable —
    # a Decision, or a Fact with lifecycle in {active, hot, ember}; two
    # (reinforce, confirm_distinct only) when the candidate is a Fact whose
    # lifecycle is archived or shelved — promoted_from is not legal against
    # either, so the offer site suppresses it rather than naming an exit the
    # engine would then refuse [Amended, ADR 0022, 2026-08-07]
    other_candidates: list[BounceNeighbor] = []

class BounceResolution(BaseModel):
    candidate_id: HypomnemaId
    action: Literal["confirm_distinct", "reinforce"]
```

**[Amended, ADR 0022, 2026-08-06]** `BounceInfo`/`suggested_action` are replaced by `FactBounce`/`DecisionBounce`, discriminated on `bounce_kind` — the kind being **saved**, never the candidate's kind — each carrying named `exits` (an exact `how` call plus the `when` judgement that selects it); the engine names the exits and chooses none. `suggested_action`'s `drop` value is gone (declining every exit is expressed by making no call); `verdict`'s `decision_supersede` is renamed `decision_adjacent` — all three verdict values name observations, not conclusions.

**[Further amended, ADR 0022, 2026-08-07]** What the caller is able to do is set by what they are writing **and by the lifecycle of what they matched** — the original framing's "never on the candidate's kind" named only one axis, and a Decision-save-against-Fact-candidate bounce could offer `promoted_from` against an archived Fact the engine then refused. `DecisionBounce.exits` is now three exits for a promotable candidate (a Decision, or a Fact with `lifecycle` in `{active, hot, ember}`) and two for a non-promotable one (a Fact `archived` or `shelved`) — see §8.4's amendment for the full promotable-lifecycle rule.

**[Second amendment, ADR 0022, 2026-08-07]** The lifecycle fix above made `promoted_from` legal to *call*; it did not make the exit's own `how` string *true*. Calling it verbatim — `save(kind="decision", content=…, promoted_from=[candidate_id])`, no `resolve` — still re-bounced, because the engine had nowhere checked `promoted_from` against the candidate that caused the bounce; only `resolve` short-circuited it. The fix: when a Decision save's `promoted_from` names exactly the `candidate_id` that caused this bounce, the save writes past it **one-shot** — no `resolve`, and **nothing is registered in the distinct-pair registry**. The promotion target was already run through `is_promotable` by the same predicate the offer site uses, so the pair is engine-linked by construction; spending a permanent, first-judgment-wins `confirm_distinct` on top of that buys nothing and costs an irreversible judgement for free. The bypass is scoped to exactly the primary `candidate_id` — `promoted_from` naming an unrelated record, or naming one of `other_candidates` instead of the primary, still bounces normally.

Resolution paths: `save(resolve=BounceResolution(...))` for confirm-distinct / it-was-a-false-alarm-reinforce; `save(supersedes=candidate_id)` for supersession (write-by-creation); `save(promoted_from=[candidate_id])` naming the bounce's own candidate for one-shot promotion (write-by-creation, no distinct-pair registration — second 2026-08-07 amendment above); declining every exit is expressed by not re-calling. `resolve` still only ever targets one `candidate_id` — `other_candidates` is read-only visibility into the same probe, not an alternate resolution target list.

`other_candidates`: the primary candidate is the top-1 by cosine, unchanged; `other_candidates` surfaces every *other* record from the same probe above its own arm floor (`gray_floor` for a Fact neighbour, `decision_gray_floor` for a Decision neighbour), descending by similarity, content omitted (the primary's content is all adjudication needs — this corpus's atoms are long enough that N neighbours' content would blow the caller's token budget). Empty when no other candidate clears its floor.

### 4.5 Bounce/park idempotency contract [A6]

- A `save(resolve=…)` **re-runs the gate**. If the named `candidate_id` no longer matches (buried, superseded, or no longer the best match), the engine returns a **fresh bounce** — it never blindly applies a stale resolution. `resolve.action="reinforce"` targets the candidate directly; if the id is gone → `ToolError` (unknown id).
- **Content contract:** every `save` call re-embeds its `content` argument, resolve calls included — there is no server-side pinning of the originally-bounced content. A `resolve` call must therefore resubmit the **same** `content` as the bounced save; editing it between the bounce and the resolve restarts adjudication against whatever is now closest, which is a fresh probe, not a stale one.
- `confirm_distinct` is idempotent by construction. [Amended] The pair registry alone cannot express "distinct from itself" (unordered `frozenset({a, b})` key collapses to one element when `a == b`), so it suppresses the re-*bounce* but not the re-*write* on its own — a byte-identical replay of the same `resolve` call is additionally detected by content-equality against the already-excluded, already-distinct-judged candidate and returns `outcome="already_saved"` naming the pre-existing matching record (earliest `created_at`, id string tie-break when more than one already-distinct duplicate exists — the storage seam carries no separate write-order field), instead of writing a duplicate. Shared by **both arms** (Fact and Decision — same engine-layer helper, one liveness predicate per arm: a Decision candidate must additionally be `lifecycle="in_force"`, since superseded/revoked Decisions are retired-but-not-buried and can still surface via the pair registry; a Fact candidate has no such retired-but-live state short of burial, which drops it from search results entirely, so `validity="valid"` alone gates it). An edited resubmission (different `content`) is a fresh judgment and still writes. [Further amended] If the matching prior record is a Fact currently `archived`, the ack additionally revives it in place (`archived→active`, the same reinforce+transition the gate's own archived-hit dispatch performs — §4.3) before returning `already_saved`: leaving it dormant while telling the caller the content is already saved is not idempotence, it is silent data loss by omission — the caller believes the fact is live and cannot find it again. An `active`/`hot`/`ember` match stays non-writing, as before. Decisions have no archived-equivalent dormant state (retired lifecycles are `superseded`/`revoked`, both already excluded from `is_live_match`), so this addendum is Fact-arm-only.
- A **parked** save is retried by the caller verbatim. Idempotency falls out of the gate: if a retry races an unnoticed success, the probe finds the just-written record and auto-reinforces instead of duplicating.

### 4.6 Reinforcement — append-only event log

```python
class ReinforcementEvent(BaseModel):
    target_id: HypomnemaId
    at: datetime
    source: Literal["dedup", "feedback", "revival", "compact", "shelved"]  # shelved: ADR 0021
    matched_from_id: HypomnemaId | None = None
    value: float | None = None          # feedback events only, ∈ [0,1]
    reason: str | None = None           # feedback OR shelved events only; carried as-is
```

The log is the **source of truth**; `recall_count` / `last_referenced_at` / `feedback_score` / `decay_due_at` are materialized projections (§8.2). A dedup-reinforce touches counters only — **never content** (no merge). `source=compact` is emitted only by offline reconciliation (§8.6). **** `reason` is the caller-supplied audit-trail string on `feedback(id, value, reason=None)`, carried as-is; `None` for every other source and for a feedback event whose caller omitted it (never an empty string). **[Further amended, ADR 0021]** `source="shelved"` (`set_shelved(id, shelved, reason)`, both directions) is the second source that populates `reason` — deliberately an event, not `set_validity`'s in-memory audit map, which does not survive a restart. `source="shelved"` is excluded from every projection formula below (§8.2) — shelving is a curator judgement, not a reference or a feedback signal.

**Distinct-pair registry:** `DistinctPair = {lower_id, higher_id, source ∈ {save, compact}, judged_at}` — the unordered pair stored **normalized, smaller id first** (adapters normalize on write; this is what the UNIQUE constraint binds). Append-only, first-judgment-wins, idempotent, **never pruned** (a buried entry's pairs harmlessly remain — monotonic, like the tombstone archive).

### 4.7 Park (embedder outage at write)

`save`/`ember` write **nothing** and return `outcome="parked", reason="embedder_unavailable"` — retryable; the caller holds the content. Reads degrade (§5.4); writes park — the principled asymmetry. **[A7]** The server logs parked outcomes and `doctor` reports park counts, so a never-retried pattern is visible in dogfooding.

### 4.8 Thresholds — per-embedder, seed → calibrate → curate

Thresholds are per-embedder **versioned config**, never magic numbers: seed conservatively per embedder family → the gate **self-labels** in operation (every bounce resolution is a label) → an **offline** calibration pass sweeps labeled pairs + the golden set and picks the operating point maximizing recall s.t. **precision ≥ `P_floor`** → curated config replaces the seed. The write path only ever *reads* config. Recalibrate from scratch on embedder swap.

**Cold-start (fail-loud-never-silent), five layers:** guards are threshold-independent · pre-deployment golden-set calibration · **paranoid mode** (until `min_label_count`, `auto_ceiling` sits maxed — bounce in doubt) · instrumented auto-reinforce-rate alert · direction-gated application (`auto_ceiling` loosens only on evidence, tightens freely).

---

## 5. Read path — Recall

*(WHY: [ADR 0007](adr/0007-recall-pipeline-and-context-composition.md) + amendments in 0012/0015/0016)*

`recall(query, *, scope, …) -> RecallResult` — query-driven hybrid retrieval; a **pure read** (reinforces nothing, except the fallback-revival case below). Consumers: the topic-shift path + the dedup probe (§4.1) reusing the same legs.

- **Fusion:** the two legs run **in parallel** (honestly parallel — never "two-stage"); the engine fuses with **RRF** (`k=60` seed); importance applies as a separate **post-fusion multiplicative boost**: `final = RRF(v, t) · (1 + β · importance_norm)` — never a third RRF leg. `importance_norm` = per-candidate-set normalization at read (`imp / max(imp)` in the fused pool). Lifecycle attention rides the same boost through importance — no separate knob.
- **Candidate fan-out:** each leg fetches `max(limit, candidate_k)` (config default `candidate_k=50`, a floor never a cap — ADR 0018), so a caller-stated `limit` is never silently starved; there is no unbounded mode — `k` is always a required, bounded `int` at the seam.
- **Filters pushed to the seam:** scope span (X + `global`) + validity + lifecycle-set, as native pre-filters. No hydration stage exists — every column is cheap.
- **Validity:** `invalid` excluded by default; `include_invalid=True` opts in, hits carry their validity flag.
- **Two-tier lifecycle:** primary pass = `active | hot | ember | shelved` **[Amended, ADR 0021 — `shelved` added]**; a **fallback pass** widens to `archived` when the primary returns fewer than `min_active_results` (engine config, never model-facing) — this auto-widen supplement is deliberately `archived`-only, never `shelved` (see below). Routine recall never revives; a returned **fallback-tier hit is reinforced** (genuine reference), and transitions `archived→active` only when the composite crosses `revival_threshold` (§8.3). Explicit `include_archived=True` (history queries) is separate, never revives, and its lifecycle FILTER is a strict superset of the primary pass's (`active | hot | ember | shelved | archived`, not just `archived`) — `shelved` is never structurally excluded from it the way it would be if the filter had stopped at four values. Like every retrieval call in this system, the actual candidates returned are still bounded by `candidate_k` per leg (no unbounded mode — ADR 0018): a `shelved` (or `active`/`hot`/`ember`) hit can in principle be pushed out of a `include_archived=True` call's bounded top-k by enough higher-scoring `archived` competitors, exactly as any lifecycle value already sharing that combined filter could before this amendment — this is a pre-existing, general property of bounded top-k retrieval over a widened filter, not something `shelved` changes or is exempt from. A `shelved` hit never reaches the fallback-tier **revival** path either way: the auto-widen supplement that feeds that path is `archived`-only, and a `shelved` hit is always already counted among the primary pass's own results, so there is nothing left for that supplement to add for it — the deliberate difference from `archived`, whose auto-widen hits do revive. Demotion to `shelved` is a curator judgement; a usage signal must never silently overturn it.
- **Degradation (embedder down):** detect via a bounded try (litellm timeout/retry/fallback — no pre-flight probe); degraded path = BM25-leg-only + boost; result carries typed `degraded: bool`, `reason`, `legs_used`. One-directional: store outage = hard error, not degraded success.
- **Hits are labeled:** `RecalledHit` carries `scope` (origin) and `purpose`.

---

## 6. Decisions — lifecycle, lineage, view

*(WHY: [ADR 0001](adr/0001-decision-immutable-content-mutable-lifecycle.md), [ADR 0013](adr/0013-decision-record-view.md))*

- **Immutable content, mutable lifecycle.** Change = supersession (`save(kind=decision, supersedes=…)`) or revocation. Retired Decisions leave active recall, stay queryable forever. Relevance for Decisions is **reference-based** (recall_count / last_referenced), never time-decay. Evergreen-vs-dated is emergent, never a flag.
- **Lineage is birth-frozen** (the organizing principle): `supersession_chain` (full backward history as snapshots, captured at the superseding transition) and `promoted_from` (ember snapshots frozen at promotion). The forward pointer `superseded_by` is set exactly once. Chains are linear (supersession targets an in-force decision); chain identity, if needed, is the root id; no chain entity is stored.
- **The render is a frame, never a parser:** status line (lifecycle · validity · recorded date · superseded date) + verbatim body + lineage section (full backward from own row, **one-hop forward only**) + `promoted_from` inline with valence and recurrence count. Title = derived. Plain honestly-labeled timestamps only — **provenance is the commit-as-you-go practice**, not engine machinery.
- **Byte-stable resource:** `decision://{id}` bytes change **iff** the decision has its own semantic transition — a pure function of its own row, zero ancestor reads. Live fields (`recall_count`, `last_referenced_at`, `tip`) appear **only** in the model-facing `DecisionView` (the `decision` tool), never in the resource; `decisions_query`'s `DecisionLink` stays a lean triage link (title / lifecycle / forward pointer / scope) by design [A11]. `purpose` is likewise excluded from the resource render.
- **Subscribability: de-scoped, seam preserved.** The capability is not declared until real; all state changes flow through `transition`/`set_validity`, so a later notification hook is one emission point.
- **`DecisionView.rendered` is byte-identical to the resource** — one render function, two exposures.

---

## 7. Push side — Curator, delivery profiles, context composition

*(WHY: [ADR 0007](adr/0007-recall-pipeline-and-context-composition.md) Module B, [ADR 0016](adr/0016-curation-audience-routing.md))*

**Pull is open; push is curated.** Audience machinery never touches Recall.

- **Curator** (selection): `curate(scope, role) -> CuratedSelection` — query-less, **token-blind**, push-only. Consumes delivery profiles + `purpose` facets + `audience_hint`s; emits ordered tiers `[Tier{name, budget_share, ordered_entries}]`. Never rewrites content (rendering = selection only).
- **Context composition** (packing): `compose_context(scope, budget, role="default") = pack(curate(scope, role), budget)`. Mechanics locked: **model-aware token counting** (`litellm.token_counter` behind a thin `TokenCounter` seam) · greedy whole-entry fill per tier · **waterfall spill** of unused budget · **never truncate** (skip whole + observable `skipped`/`budget_used`).
- **Delivery profiles are data, not code** — store-level config keyed by Role: facet rules per tier, tier order, per-tier sort keys, budget shares, hint handling. (Serialization format + versioning: build-time with the config file format, §16.) The **`default` profile** reproduces the pre-Curator behavior exactly: tiers hot Facts (`importance DESC, last_referenced DESC`) → embers (nudge-rung set), presentation order **hot → ember**. Two real consumers from day one: composition + the eval harness (profile-swapping). **Decisions are pull-only, not a push tier** — see the amendment below.
- **`audience_hint` is additive-only on pull** — unchanged: `recall` never filters or excludes on it; stale hints just stop matching. **On the two push tiers (`hot`/`ember`) it is the routing key**: absence of a hint means universal reach (falsy `audience_hint` routes to every role); a hint routes the candidate only to a role named in it. Not a boost — routing is binary inclusion/exclusion at the push tier, never a rank nudge. Nothing is hidden from anyone: `recall` stays open to every caller regardless of hint, so this narrows an already-owner-curated set rather than gatekeeping content — ADR 0016's own *attention economy, never secrecy* principle, not an exception to it.
- **Composition merges the X + `global` pool by rank within each tier — no scope quotas** (per-scope hot caps bound contributions naturally).
- General non-hot active Facts are excluded from push (Recall's on-demand job).
- **Roles** are cooperative self-identification (no principal exists); `role` is REQUIRED on `context` — omission raises rather than resolving to a fallback.
- **amendment (2026-07-31):** composition ships with **fixed default selection**, not the Curator above — `context(scope, budget, role) = pack(select_default(scope, role), budget, counter)`, never `pack(curate(scope, role), budget)`. The Curator's *configurability* (delivery profiles as data, facet rules per tier, audience-hint boosting as opposed to routing) **remains cut, not merely deferred** — `select_default` still hard-codes the one `default` tier shape (hot → ember) and there is no `curate()` step; role-specific *selection logic* (different tiers, different sort keys per role) is still exactly one adapter away from existing. The §9.1 signature stays final regardless — `role` is accepted and **validated** against the role registry (unregistered → `ToolError`, same as `purpose`/`audience_hint`). **`role` is no longer inert:** both push tiers now apply the routing filter — a candidate reaches `role`'s payload when its `audience_hint` is empty (universal reach) or names `role`. On the corpus as it stood the day this landed, every record was unhinted, so every registered role still resolved to the identical selection — narrowing became opt-in, one atom at a time, via `set_hot(id, hot=True, roles=[...])`, never a flag day. 
- **amendment (2026-08-01):** `select_default` drops the in-force Decision tier. **Default selection ships `hot` + `ember` only** — `select_default(scope)` returns exactly two tiers, in that presentation order; the third tier described above (`in-force Decisions`, recency/reference) is deleted, not disabled (no `enable_*` flag, no dormant branch). **Decisions are pull-only at session start** — reachable through `decisions_query` (hybrid retrieval by natural-language query) and `decision(id)` (full view) when an agent is actually deciding something, never a ranked slice of all of them injected up front. Owner decision, 2026-08-01, on three measurements: the Decision tier was 2315 of 2997 tokens — 77% of the payload — for a representative scope at `budget=3000`; it was the only **uncurated** tier (484 in-force candidates sliced by rank to 18, versus `hot`, which the owner fills one record at a time via `set_hot`); and that rank is not trustworthy until slice B, since `w_feedback` is the largest term in the §8.1 composite (1.5) and `feedback_score` is pinned at 0.0 on every record — so which 18 of 484 appeared was close to arbitrary. This is also what the owner's standing rule warns against: session-start injection must be a curated hot set, never a ranked slice of the whole ledger. **Ember stays in the push tiers**: embers are few and they are corrections; an ember's escalation exit is `set_hot` (§8.4 rung 2), so surfacing one at session start is how the owner learns a correction is recurring and worth promoting — the only mechanism in default selection that raises something the owner has not already chosen. `pack()`'s mechanics (equal split, waterfall spill, never-truncate) are unchanged; with two tiers they simply apply to two. 
- **Amendment (2026-08-03) — explicit enumeration only.** The "absence of a hint means universal reach" half of the two amendments above is **superseded**: an unhinted (falsy `audience_hint`) record now reaches **NO** role on either push tier, not every role. "For everyone" must be spelled by naming all four roles (`decider`/`builder`/`reviewer`/`navigator`) in the hint; omission is no longer a way to say it. **What survived unedited:** the routing mechanism itself — binary inclusion/exclusion at the push tier, never a rank nudge; one shared predicate (`engine/context.py::_routes_to`) applied identically by both `_select_hot` and `_select_embers`, so the rule cannot drift between tiers; and pull stays completely untouched — `recall` and `decisions_query` apply no `audience_hint` filter, unchanged by this amendment, same as every amendment before it. **Why:** implicit universality could not distinguish "deliberately for everyone" from "nobody has curated this yet", and made the untagged state a lazy default that would silently flood every one of the four newly-registered roles — the exact ritual-stamp failure ADR 0016's own Context section documents from project-b. **Sequencing:** the ten then-existing hot records with explicit hints *before* this semantics flip landed, so no record went dark in the transition. ADR 0016's own further amendment (below).
- **Amendment (2026-08-03) — `role` is now REQUIRED on `context`, `default` is retired.** The earlier amendment (above) closed WHAT a falsy `audience_hint` means (no role, not every role); this closes a live consequence it opened: `role` omitted still resolved to `"default"`, and no real record was ever hinted `"default"`, so a role-less `context` call returned an EMPTY payload with no error at all — the exact "nobody has curated this yet" ambiguity already banned, just reappearing one layer up, at the caller. `context(scope, budget, role)`'s `role` parameter is now REQUIRED: a caller supplying `None` is bounced with a `ToolError` naming the registered roles, at both the engine boundary (`Tabularium.context`) and the MCP tool boundary (`tabularium.server`'s `context` tool — a thin pass-through, per its "near-1:1 projection, ZERO memory semantics" contract, not a second independent check). `"default"` — the always-present fallback role that predated per-role grounding — is retired: `initialize()` no longer seeds it, so a fresh store's role registry is exactly the four function-named roles (`decider`/`builder`/`reviewer`/`navigator`). Owner decision, 2026-08-02: "no caller has no role — bounce back to get it" — an unspecified audience is a caller bug, not a state the system papers over with a default. 
- **Amendment (2026-08-03) — `delete_role` closes the honesty gap above.** The previous amendment could only retire `"default"` in the weak sense of "no longer seeded on a fresh store" — the seam exposed `create_role`/`list_roles` only, so a store that already held the row kept it forever. Owner decision: the existing store is a proof-of-concept data-acquisition artifact, not a production store to preserve, and the seed will not be re-run, so "registered but unused" is not an acceptable permanent state and design must not be weakened to keep that store's contents intact. `delete_role` now exists on `StorageAdapter` and both adapters — idempotent no-op when the name is not registered (symmetric with `create_role`'s duplicate-no-op, and what lets `initialize()` call it unconditionally with no existence check first); it never touches any record's `audience_hint`, and never refuses because some record still names the role — a stale hint entry goes **inert** (nobody can request a deleted role, since `context()` validates `role` against the registry), never dangling, and refusing would make deregistration hostage to stale data the operator cannot easily enumerate. `initialize()` now calls `delete_role("default")` unconditionally, after seeding the four canonical roles, on every engine startup — so "retired" is now a real guarantee: any store, fresh or pre-existing, ends up `default`-free the next time the engine runs. 

---

## 8. Importance, decay, eviction, offline jobs

*(WHY: [ADR 0009](adr/0009-importance-decay-eviction.md) + [ADR 0012](adr/0012-mcp-tool-surface-design.md) B4; §8.5/§8.6 amended by [ADR 0020](adr/0020-hot-tier-owner-curated-not-earned.md))*

### 8.1 The composite (computed-lazy, never stored)

```
importance = w_recall   · ln(1 + recall_count)
           + w_recency  · exp(−Δdays / half_life)        # Δdays since last_referenced_at (default created_at)
           + w_feedback · feedback_score                  # feedback_score ∈ [0,1]
```

- **As ranking** (recall boost, hot ordering, composition): computed lazily at read over the candidate set — recency is always `now()`-fresh; **the freeze bug is structurally impossible**.
- **As transition trigger**: a scheduled sweep evaluates the same formula (idle rows can't self-report).
- **Canonical feedback semantics [A8]:** events carry `value ∈ [0,1]` (anchors bad≈0.0 · neutral≈0.5 · helpful≈1.0); `feedback_score = EWMA(α=0.3, init 0.0) ∈ [0,1]`; the term is **non-negative additive with no centering transform** — feedback can erode a prior boost toward 0 but can never push below the floor or force archival (no backdoor eviction). Consequence, stated openly: an explicit neutral is mild positive signal by design — attention is signal; silence leaves the score untouched.
- Additive (not multiplicative-recency): the permanent recall/feedback terms **are** the archive floor — usage buys permanence; keepers stay active when quiet.
- Seeds (eval-tunable, not claims): `w_recall=0.5, w_recency=1.0, w_feedback=1.5, half_life=30d`.

### 8.2 Projections (deterministic functions of the event log)

| Stored input | Projection |
|---|---|
| `recall_count` | `count(events where source ∈ {dedup, revival, compact})` [A10] |
| `last_referenced_at` | `max(at where source ∈ {dedup, revival})`, default `created_at` — deliberately excludes `compact`: an offline sweep is not a reference [A10] |
| `feedback_score` | `EWMA(α=0.3, init 0.0)` over `source=feedback` values, each ∈ [0,1] → score ∈ [0,1] |
| `decay_due_at` | precomputed composite-crossing time, recomputed on each reinforcement |

**Feedback ⊥ references** (normative): feedback events never touch `last_referenced_at`/`recall_count`; reference events never touch `feedback_score`. **[A9]** A maintenance verb rebuilds all projections from the log and reports drift; a T01 property test asserts projection determinism.

### 8.3 Lifecycle transitions

- **Decay (`active→archived`):** sweep archives any active Fact whose composite `< archive_threshold`. **No hysteresis** (importance only rises on a reference — flapping is structurally impossible). Facts only; `hot` exempt; Decisions never decay; embers fade through this same crossing.
- **Revival (`archived→active`), split by source strength:** dedup-revival = **unconditional, in-place** (a deliberate re-assertion); recall-fallback revival = **threshold-gated** — the hit reinforces, the transition fires only when the composite crosses `revival_threshold` (a notch above `archive_threshold`; defensive dead-band).

### 8.4 Ember ladder

Rung 1 — **born as the nudge**: embers surface in the session-start ember tier from birth (those above the fade floor, importance-ranked; the cap self-limits).

**Rung 2 — recurrence.** `r_owner = count(owner-asserted recurrence events) ≥ escalation_threshold` (seed 3) → Tabularium surfaces the ember with **two exits** and names both. Suggested, never automatic; the owner takes one, or neither, and the ember otherwise stays a Fact and fades.

- **Codify** — `save(kind="decision", promoted_from=[ember_id])`. Take this when the ember records a choice: something was chosen and something else was live and rejected (§1.4). Content freezes and the ember is left untouched at `lifecycle="ember"`. [Amended 2026-08-07: this bullet previously also claimed "decay stops". Nothing in `engine/dispositions.py` or `engine/maintenance.py` reads `promoted_from`, so decay scheduling has never been aware that a Fact was codified — the claim was aspirational text that never shipped. Removed rather than implemented; if promotion should halt decay, that is a feature to decide on its own merits.]
- **Foreground** — `set_hot(ember_id, hot=True)`. Take this when the ember is a standing rule that keeps being violated, rather than a choice that keeps being re-derived. Nothing new is recorded; the atom stops being optional to see.

The engine does not choose between them. Recurrence is a deterministic count; which exit fits is an owner judgement about content, and §1.4's road-not-taken test is the question to ask. An ember with no road not taken has no Decision available to it — that is the case this rung exists for.

**`promoted_from` is not ember-only.** [Amended 2026-08-07 — ADR 0022 relapse fix.] The Codify bullet above names the ember-ladder's rung-2 use of `promoted_from` specifically — it is not the full extent of what the parameter accepts. Promotion is pure provenance: `_write_decision` upserts only the new Decision and never mutates the target Fact, for any Fact — the "ember is left untouched" guarantee above generalizes. So the real precondition was never "was this born an ember," it is "is this still a live, engine-defended assertion a Decision can credibly rest on": `active`, `hot` and `ember` Facts are all legal `promoted_from` targets; `archived` (decayed out of relevance) and `shelved` (curator-judged not worth carrying) are not. Gating on `lifecycle == "ember"` was considered and rejected — there is no `active→ember` transition anywhere in the engine (`ember` is written only at birth or ember→ember reignite), so an ember-only rule would make the exit permanently unreachable for a plain Fact worth codifying directly, the common case. The offer site (`engine/exits.py::build_exits`/`is_promotable`) and the engine guard (`engine/decisions.py::_save_decision`) share one predicate so the two cannot drift apart again.

Agent-sourced recurrence (`source=dedup` events, an agent re-saving matching content on its own initiative) keeps the ember warm and feeds the anti-decay signal, but **does not** count toward `r_owner` and cannot trigger this rung.

**Rung 0 — creation-time audience, and the authorship split.** Per-role grounding (§7's push-tier routing) made an unhinted record invisible on every push tier — including a brand-new ember, since `ember()` kept `audience_hint: list[str] | None = None` with no requirement at write time. Nothing prevented minting one that reached nobody, silently. The fix splits by `asserted_by`:

- **`asserted_by="owner"`** — `audience_hint` is **required and non-empty**; a falsy value (omitted, or an explicit `[]`) raises a `ToolError` naming the registered roles. An ember is authored, not inferred: the author is right there and can say who it is for, so requiring a real audience costs them nothing.
- **`asserted_by="agent"`** (the default) — a falsy `audience_hint` defaults to the **four canonical roles** (`decider`/`builder`/`reviewer`/`navigator` — `engine._types._ALL_ROLES`, never a live registry read, which could carry a store's historical cruft), and the written Fact is marked `provisional` (`Fact.provisional`, above): an agent surfaces a correction, it does not get to decide who it is for, so the owner's verdict is left outstanding rather than the engine inventing a narrower default on the agent's behalf.

The check runs unconditionally (before the dedup gate resolves reinforce-vs-fresh-write), mirroring `set_hot(hot=True)`'s unconditional `roles`-required check (§8.5) — the discipline is "always state who this is for," not "only when it happens to be new." **There is no "deliberately reach nobody" option at creation time** — the headline acceptance property is unqualified ("no ember can be born reaching nobody, by either authorship path"), so an explicit `audience_hint=[]` is treated exactly like an omitted one, on both paths. This differs from the SEPARATE `roles` retag parameter (below), whose own `roles=[]` legitimately un-hints an EXISTING record — a different parameter governing a different operation, untouched by this rung.

`provisional` is a **persisted field, not an inference from the hint's shape** — an all-four-roles hint alone cannot distinguish "the owner said everyone" from "an agent guessed and nobody has ruled," which is the exact ambiguity per-role grounding exists to kill; recreating it one layer up (inferring provisional from an all-four hint) would relocate the bug rather than close it. The defined path by which the owner rules and the mark clears: an **owner-asserted** `roles=[...]` retag on the same ember's reignite (`ember(asserted_by="owner", audience_hint=[...], roles=[...])` — see §9.1) sets `provisional=False`. **Both `audience_hint` and `roles` are required on that call, together** — `audience_hint` is required unconditionally on every `asserted_by="owner"` call regardless of outcome (the check above runs before the dedup gate knows this is a reignite, not a fresh write), so a call illustrating only `roles=[...]` would raise before ever reaching the retag branch. An agent-asserted retag narrows the hint but leaves the mark untouched, since narrowing is not ruling. This is entirely separate from the RETAG path's own optionality (`roles=None` leaves the stored hint untouched on reignite, unchanged by this rung, `TestNoAutoPromotion` unaffected).

An open-time schema migration backfills `provisional=False` on every pre-existing row (the same `_migrate_table_schema` mechanism the recurrence counters and `events.reason` used) — existing records are not re-opened for adjudication; a pre-existing unhinted record is simply off the push tiers already, same as before this rung existed.

**A second, definitely-reachable clearing path ** The ember-reignite `roles` retag above only runs on the dedup gate's auto-reinforce branch, which is unreachable under the real default `paranoid=True` (§4.8) — every reignite bounces instead, and `ember` has no `resolve` argument to adjudicate a bounce (unlike `save`). `set_hot(id, hot=True, roles=[...])` — itself always an explicit owner-curated audience decision, since `roles` is required there (rung 2's Foreground exit, this section) — never touches the dedup gate at all, so it also sets `provisional=False` on promotion, regardless of `paranoid`. Demotion (`hot=False`) leaves the mark untouched. This is the property this rung's own acceptance criterion ("a defined path by which the owner rules and the mark clears") actually needs to hold under default configuration, not merely under a test harness's `paranoid=False` override.

**A third path — ruling without foregrounding.** Both clearing paths above tie the ruling to something else: the reignite retag is unreachable under the shipped default (per the previous paragraph), and `set_hot(hot=True, ...)` ties the ruling to promotion into the always-on hot tier. Neither expresses the ordinary case: *"yes, this correction is for builders and reviewers; no, it does not belong in every session's payload yet."* `set_audience(id, roles)` (§9.1) is a dedicated verb for exactly that — it sets `audience_hint` and clears `provisional`, and **never touches `lifecycle`**, separating curation of audience from promotion into the tier. Facts-only (`provisional` does not exist on `Decision`); a `hot` Fact is a `ToolError` naming `set_hot` instead, since that tool's promote-side retag already re-checks `hot_cap` (§8.5) and a second, uncapped route to the same field on a hot Fact would let a caller grow past the cap silently. `roles` is required (no default) — there is no "leave the hint untouched" mode, since the tool exists to make an explicit ruling; `roles=[]` is a legal, deliberate "reaches no role" ruling. This does not resolve, and must not be read as resolving, whether the dedup gate belongs in front of a curation operation at all — that question stays open, tracked separately.

### 8.5 `hot` — owner-curated, never earned (no sweep, no LLM-judge)

**[Amended 2026-07-31 — ADR 0020, supersedes the earned-only half of ADR 0006.]** `hot` is always-on: every hot record is injected into every session-start payload (§7 / §9.1 `context`). A pure-importance promotion contest was designed, built, gated PASS, and withdrawn the same day on the discovery that an importance heuristic filling an always-on, fixed-budget tier converges on permanent context bloat — occupants only ever get sticky (hysteresis keeps them once in), never actually bounded by anything but the cap. The resolved design:

- **Promotion is a single owner action:** `set_hot(id, hot=True)` (§9.1) is the *only* `{active,ember}→hot` edge — the `ember` case is §8.4 rung 2's Foreground exit. No sweep, no scheduler, no background contest exists anywhere in the engine.
- **Deliberate demotion** is the same tool, reversed: `set_hot(id, hot=False)` is the only tool-driven `hot→active` edge.
- **Automatic demotion survives, but only as housekeeping, never curation:** a hot Fact that is buried, invalidated, or (once decay ships, M3) archived leaves the tier as a structural consequence of no longer being a live active-attention Fact — never because a policy re-judged the owner's pin. Invalidation in particular never touches `lifecycle` at all; the record simply stops qualifying for the tier's existing valid-only selection filter (§7).
- **`hot_cap` is a payload-size guardrail, enforced by rejection, not a contest parameter:** promoting past the cap raises `ToolError` naming every Fact currently hot there, so the owner — never the engine — decides what leaves. It never silently evicts the lowest-ranked occupant.
- **Amendment (2026-08-03) — `hot_cap` is per (scope, role), not per scope.** Five per scope was correct when there was one audience; with four roles (§10.3) sharing the same slots, one role's promotions could starve another's, defeating the point of routing (§9.1 `set_hot`'s `roles` parameter). The cap now counts occupants via `engine.context._routes_to` — the SAME predicate the push-tier delivery filter (`_select_hot`/`_select_embers`) applies, so the cap and the delivered payload cannot drift apart. Split into two knobs, never tuned independently: `hot_cap_global` (seed **8**) governs the `global` scope — every session spans `[scope, global]`, making it the scarcest real estate in the system, hence the larger cap; `hot_cap_scope` (seed **5**) governs every other scope. **Governing invariant:** `hot_cap_global + hot_cap_scope` must be ≤ what the smallest consumer budget delivers — a cap above budget capacity is not a cap, it is a queue: the cut still happens, it just moves from an explicit `ToolError` naming occupants to a silent rank-based truncation (the "which 18 of 484 appear is close to arbitrary" failure that killed the Decision tier, §7). **The consumer-budget knobs are not in this repo:** a session-start hook can request a larger budget than a per-subagent hook, and the smaller one governs. The sub-agent budget was raised specifically to keep the invariant true at 8+5=13 atoms ≈ 1300 tokens, measured at ~100 tokens/atom — raising either cap here without checking the consumer hooks is how this invariant breaks silently. Owner decision 2026-08-03; a Tabularium Decision. 
- `hot_promote_threshold` / `hot_demote_threshold` (the withdrawn contest's promotion floor and hysteresis dead-band) are now **dead config** — no code path reads them; left in place, unused, the same inert-observability treatment `hot_rejection_count` already had on the Fact model.
- `hot` is still **persisted** (a `set_hot` call must stick, not be re-derived per read) — that half of ADR 0006 stands unamended, along with `hot`/`pinned`/`is_hot` collapsing into one state (now an explicit owner write instead of an emergent one). `ember` remains a distinct, write-time-seeded state, but is **no longer excluded from promotion** — see the amendment immediately below.
- **[Further amended 2026-07-31 — §8.4 rung 2's Foreground exit.]** `set_hot(id, hot=True)` now accepts a Fact in `active` **or** `ember` (previously `active`-only; a Decision id is still a `ToolError`). The original prohibition guarded against unverified atoms reaching an always-injected tier. After ADR 0020 there is no automatic path to `hot` at all — the only remaining actor is the owner, who can see what they are promoting. An ember at `r_owner ≥ escalation_threshold` is the opposite of unverified: the recurrence is the evidence. `set_hot(id, hot=False)` demotes to `active`, never back to `ember` — ember is a write-time-seeded provisional state; a record that has been through the ladder is not provisional.
- The recurrence counters (`owner_recurrence_count` / `agent_recurrence_count`) live on the record, not the lifecycle, and survive both promotion and demotion.
- §8.4 rung 2 fires on the **crossing** — an owner-asserted recurrence event that takes `r_owner` from below the threshold to at-or-above — never on the standing value, so a Fact recurring past the threshold many times over only ever fires the suggestion once. `set_hot(id, hot=False)` records a fresh `escalation_baseline` (= the record's `owner_recurrence_count` at demotion) so the ladder can re-arm; without it, a later owner-asserted event on the same record would find the standing count already past threshold and re-fire immediately, which is a nag, not a suggestion.

### 8.6 Offline-job family (all `withoutOverlapping`)

1. **Decay sweep** (seed cadence 15–30 min): an **indexed range scan** on `decay_due_at ≤ now()` — never a full-scope recompute. **[Amended 2026-07-31 — ADR 0020.]** No longer a combined "decay + hot" job: `hot` has no sweep of any kind (§8.5) — its only transitions are the owner's `set_hot` calls plus housekeeping demotion on the existing bury/invalidate/(future) decay paths, so there is nothing for an offline job to do to it.
2. **Calibration pass** (slow cadence; gated on ≥ `min_label_count` new labels): T01 owns the sweep harness, this job owns **live application** — direction-gated, versioned per-embedder config.
3. **Offline dedup reconciliation** (seed weekly): re-runs the §4 classifier over existing records; **monotonic** — never merges or deletes; auto bands stay auto; the **gray band goes to an LLM provider-judge** (litellm-routed; the *only* LLM-judge seat in the system): `same`→reinforce (`source=compact`) · `distinct`→mark registry · Decision-supersession→**suggest**. Unreachable provider ⇒ pairs defer to the next run. Every call carries telemetry; its prompt lives in the prompt registry (§16).

**Three-way adjudicator split (locked):** write-time gray band → the **present calling agent** · eviction → the **calibrated number** · offline gray band → an **LLM judge**. No write-time LLM of any kind (including facet classification).

### 8.7 Clock

All time flows through an injectable **`Clock` seam** — no bare `datetime.now()` in engine code. With the append-only log, every trajectory is a pure function of `(event_log, clock)` — correctness is replayable by construction.

---

## 9. MCP surface

*(WHY: [ADR 0002](adr/0002-mcp-interface-official-fastmcp.md), [ADR 0012](adr/0012-mcp-tool-surface-design.md) + amendments in 0013/0015/0016/0020)*

**Layering:** `tabularium.server` = near-1:1 FastMCP projections, ZERO memory semantics; `tabularium` engine = the deep module owning every invariant (pydantic-in/pydantic-out; the same result models the tools expose); storage seam below. The engine stands alone — the eval harness and the `sync` CLI import it directly.

**Framework:** official MCP SDK FastMCP, pinned **`mcp>=1.27,<2`**; never the v2 pre-release; standalone `fastmcp` 3.x is the named escape hatch.

### 9.1 The surface — 15 tools + 1 resource (final signatures)

This is a **live-registration count**: `feedback` shipped (2026-08-02), reaching the
then-specced surface for the first time — `grep -c "@app.tool" src/tabularium/server.py` read **13** at
that point, matching every row below except `set_audience`. Curation block (2026-08-03) adds
`set_audience` (§8.4 rung 0's third clearing path), bringing the live count to **14**. ADR 0021
(2026-08-04) adds `set_shelved` — a fifth Fact lifecycle value and its verb — bringing the live count to
**15**, the current row set.

| Tool | readOnly | idempotent | destructive |
|---|---|---|---|
| `save` · `ember` · `digest` · `feedback` | — | — | — |
| `recall` · `context` · `decisions_query` · `decision` · `sessions` | ✓ | — | — |
| `set_validity` · `revoke` · `set_hot` · `set_audience` · `set_shelved` | — | ✓ | — |
| `bury` | — | ✓ | **✓** (the lone destructive tool) |
| `decision://{id}` resource (text/markdown) | ✓ | — | — |

```python
save(content: str, *, kind: Literal["fact","decision"]="fact", scope: str,
     source: str | None = None, supersedes: HypomnemaId | None = None,
     promoted_from: list[HypomnemaId] | None = None,
     resolve: BounceResolution | None = None,
     purpose: str | None = None, audience_hint: list[str] | None = None) -> SaveResult
    # tool description carries the derivability bar (save what cost something to learn;
    # skip what the code already shows) and the kind-routing rule (§1.4, Choosing the
    # kind) — the description is the single source of both texts

ember(content: str, *, scope: str, valence: Literal["correction","guideline"],
      asserted_by: Literal["owner","agent"] = "agent",
      source: str | None = None,
      purpose: str | None = None, audience_hint: list[str] | None = None,
      roles: list[str] | None = None) -> SaveResult
      # thin projection presetting lifecycle=ember; same engine gate; same envelope.
      # asserted_by is RECORDED, never enforced -- MCP gives the server no way to
      # verify who is actually calling. Only "owner"-asserted recurrence advances
      # r_owner / can cross escalation_threshold and trigger the §8.4 rung-2
      # suggestion; "agent" (the default) keeps the ember warm (agent_recurrence_
      # count) without ever arming the ladder.
      # Creation-time audience rule (§8.4 rung 0): a falsy audience_hint
      # (None or []) with asserted_by="owner" raises ToolError naming the
      # registered roles; with asserted_by="agent" (the default) it defaults to
      # the four canonical roles and marks the written Fact provisional=True.
      # roles (above) also RETAGS an existing ember's audience_hint on reignite
      # (unchanged by this rung, including roles=[]'s "un-hint an
      # EXISTING record" meaning) -- an owner-asserted retag additionally
      # clears provisional.

digest(content: str, *, scope: str) -> SessionRef
    # appends one append-only session-digest row (§1, §3). NOT deduplicated --
    # two identical calls write two rows, deliberately (a Session has no dedup
    # gate, no bounce). started_at/ended_at are server-derived, never supplied:
    # ended_at = now; started_at = the same scope's previous row's ended_at (or
    # this row's own ended_at if it is the first in the scope) -- the caller
    # states no times, so it can state none of them wrongly.

class SessionRef(BaseModel):
    id: UUID; scope: str; started_at: datetime; ended_at: datetime
    # minimal pointer to the written row -- deliberately excludes content and
    # source, same convention as HypomnemaRef; Session is not a Hypomnema (§1)
    # so this model does not reuse HypomnemaRef

class SaveResult(BaseModel):
    outcome: Literal["saved","reinforced","revived","reconciled","bounced","parked","already_saved"]
    hypomnema: HypomnemaRef | None = None
    bounce: FactBounce | DecisionBounce | None = None   # §4.4, discriminated on bounce_kind [ADR 0022]
    reason: str | None = None
    escalation_suggestion: EscalationSuggestion | None = None
    # already_saved [amended]: idempotent ack for a byte-identical replay of a
    # confirm_distinct resolve, EITHER arm — hypomnema carries the id of the
    # pre-existing match, not a new one (§4.5 [A6])
    # escalation_suggestion [§8.4 rung 2, amended 2026-07-31]: set on an
    # outcome="reinforced" ember when the reinforcing event was owner-
    # asserted AND it crossed escalation_threshold; names both exits
    # (codify / foreground). None on every other outcome, and None on an
    # agent-asserted or below-threshold ember reinforcement.

class EscalationExit(BaseModel):
    action: Literal["codify", "foreground"]
    how: str                                  # the exact call to make
    when: str                                 # the content judgement that selects it

class EscalationSuggestion(BaseModel):
    ember_id: HypomnemaId
    owner_recurrence_count: int
    threshold: int
    exits: tuple[EscalationExit, EscalationExit]   # always both, never one

recall(query: str, *, scope: str, limit: int = 10,
       include_invalid: bool = False, include_archived: bool = False) -> RecallResult

class RecalledHit(BaseModel):
    id: HypomnemaId; kind: Literal["fact","decision"]; content: str
    lifecycle: str; validity: Literal["valid","invalid"]; score: float
    scope: str                     # origin label (0015)
    purpose: str | None = None     # (0016)

class RecallResult(BaseModel):
    hits: list[RecalledHit]; degraded: bool; reason: str | None; legs_used: list[str]

sessions(scope: str, *, query: str | None = None, since: datetime | None = None,
         until: datetime | None = None, limit: int = 20) -> list[Session]
    # search span is [scope, global], same convention as recall/decisions_query
    # (§2). No query and no since/until -> most recent rows, ended_at descending,
    # capped at limit. Session is the §1 domain model returned directly -- there
    # is no RecalledHit-style projection because a Session carries no score,
    # lifecycle or validity to report (§1, §3).

context(scope: str, budget: int, role: str | None = None) -> ContextPayload
    # §7 amendment (2026-08-01): two packed tiers, hot -> ember, in that
    # order. No Decision tier -- Decisions are pull-only at session start, via
    # decisions_query / decision.
    # §7 amendment (2026-08-02): role now filters both tiers (routing,
    # not boosting) -- see §7's audience_hint bullet for the rule.
    # §7 amendment (2026-08-03): role is now REQUIRED --
    # role=None raises ToolError naming the registered roles, at both the engine
    # boundary and the MCP tool boundary (thin pass-through, no second check).
    # The type stays Optional so the caller gets a clean ToolError rather than a
    # raw TypeError for a missing argument. "default" is retired -- omission no
    # longer resolves to it.

class PackedTier(BaseModel):
    name: str; entries: list[RecalledHit]

class ContextPayload(BaseModel):              # final shape — tiers, not named fields (0016 B8)
    tiers: list[PackedTier]; skipped: int; budget_used: int

feedback(id: HypomnemaId, value: float, reason: str | None = None) -> FeedbackResult
    # Registered (2026-08-02) -- live in server.py.
    # value ∈ [0,1] (Field(ge=0, le=1)); anchors bad≈0.0 · neutral≈0.5 · helpful≈1.0
    # Fact-only: a Decision id → ToolError. About relevance quality, not truth.

set_validity(id: HypomnemaId, valid: bool, reason: str) -> ReviseResult   # both arms, bidirectional
bury(id: HypomnemaId, reason: str) -> BuryResult                          # both arms; tombstone_id returned
revoke(id: HypomnemaId, reason: str) -> ReviseResult                      # Decision-only; Fact id → ToolError
    # reason required on all three; supersession is save(supersedes=…), never a disposition tool

set_hot(id: HypomnemaId, hot: bool, roles: list[str] | None = None) -> Hypomnema
    # Fact-only; a Decision id -> ToolError. hot=True promotes an active OR ember Fact
    # (archived -> ToolError; §8.4 rung 2's Foreground exit for the ember case);
    # hot=False demotes a hot Fact to active, NEVER back to ember (ember is a
    # write-time-seeded provisional state; a record that has been through the
    # ladder is not) -- also records a fresh escalation_baseline so the ladder can
    # re-arm (§8.4/§8.5). Owner-curated, never earned (ADR 0020, §1.2/§8.4/§8.5) --
    # the ONLY {active,ember}->hot edge. Idempotent: hot=False on an already-active
    # Fact with roles=None is a no-op (no adapter write); hot=True on an already-hot
    # Fact is a no-op ONLY when roles equals the Fact's current audience_hint exactly
    # (roles is required there -- see below). Past hot_cap for ANY
    # requested role in the Fact's scope -> ToolError naming the overflowing
    # role and every Fact currently hot for that role in that scope; never a
    # silent eviction. Every requested role is checked before any write, so a
    # violation on any single role rejects the WHOLE promotion -- never a
    # partial write that tags some roles and leaves others untagged
    # (hot_cap is now per (scope, role), not per
    # scope; see §8.5).
    # roles: sets the Fact's
    # audience_hint -- the routing key for the two push tiers (§7). REPLACES any
    # existing hint, never unions (owner-ratified: curation-time judgment beats a
    # writer's capture-time guess). roles=None leaves the stored hint untouched;
    # [] clears it. Validated
    # through the same role registry purpose/audience_hint already gate against;
    # unregistered role -> ToolError. `_set_hot` remains the sole writer of
    # lifecycle="hot" (ADR 0020) -- the role tag rides with the pin, in this same
    # call.
    # roles ON an already-hot/already-active call is a retag, not a no-op: it
    # persists via a same-lifecycle transition. On the demote side (already
    # active, hot=False) this is unconditional -- no hot_cap re-check (no
    # slot is consumed by leaving the tier), no escalation-baseline re-arm
    # (that only fires on a genuine hot->active edge). On the promote side
    # (already hot, hot=True) the per-(scope, role) hot_cap check above DOES
    # run again for every requested role -- a retag
    # is not exempt merely because the lifecycle edge is a no-op, since
    # adding a role the Fact did not previously hold consumes a real slot in
    # that role's cap; the Fact being retagged is excluded from its own
    # occupant count in every role's tally, so retagging while KEEPING a role
    # it already holds never trips that role's own cap. Either way, this is
    # the live-store curation path, since the atoms worth reviewer-tagging
    # are typically already hot.
    # Amendment (2026-08-03): roles is now REQUIRED when
    # hot=True -- roles=None raises ToolError unconditionally, regardless of
    # lifecycle state and regardless of whatever audience_hint the Fact already
    # carries from write time (this REVERSES the original "existing callers
    # unaffected" clause quoted above). roles=[] (explicit empty) is still legal.
    # hot=False is unchanged: roles stays optional there -- removing something
    # from the tier needs no audience decision.

set_audience(id: HypomnemaId, roles: list[str]) -> Hypomnema
    # Fact-only; a Decision id -> ToolError. Rules on a Fact's audience
    # WITHOUT foregrounding it into hot -- sets audience_hint and clears
    # provisional, and NEVER touches lifecycle. Closes the gap set_hot
    # promotion alone left: under the default paranoid=True, the only
    # reachable way to clear a provisional ember's mark was set_hot(hot=
    # True, roles=[...]), which also promotes it -- there was no path for
    # "yes, this is for builders and reviewers, no it does not belong in
    # every session's payload yet". roles is REQUIRED (no default) --
    # there is no "leave the hint untouched" mode, since the call exists
    # to make an explicit ruling; roles=[] is a legal, deliberate "reaches
    # no role" ruling. A hot Fact is a ToolError naming set_hot instead --
    # that tool's promote-side retag already re-checks hot_cap (§8.5)
    # before persisting a new hint, and this tool must not become a
    # second, uncapped route to the same field. Idempotent: re-calling
    # with the SAME roles on a Fact whose provisional is already False is
    # a no-op; any other call always writes, even when roles matches the
    # stored hint, since clearing provisional for the first time is a
    # genuine ruling. Does NOT move, bypass, or special-case the dedup
    # gate guarding ember's reignite retag -- that remains a separate,
    # open question.

set_shelved(id: HypomnemaId, shelved: bool, reason: str) -> ReviseResult
    # Fact-only; a Decision id -> ToolError (ADR 0021). Bidirectional, one
    # verb, shaped after set_validity. shelved is the fifth Fact lifecycle
    # value and the ONLY one the dedup gate's candidate probe never sees
    # (§4.1) -- a near-duplicate of shelved content is free to save again.
    # Stays in recall's primary filter (§5) -- ordinary recall still
    # returns it -- and a recall hit on it never revives it (the
    # deliberate difference from archived). Never reaches context() --
    # automatic, the packer only reads hot/ember (§7). Legal edges:
    # active/archived -> shelved (shelved=True); shelved -> active
    # (shelved=False, always lands on active, never restores the prior
    # lifecycle). hot/ember Facts cannot be shelved directly; shelved ->
    # hot directly via set_hot is a ToolError -- un-shelve first.
    # Idempotent: shelved=True on an already-shelved Fact, or shelved=
    # False on an already-active Fact, is outcome="unchanged", no write.
    # reason is REQUIRED and persisted AS AN EVENT
    # (ReinforcementEvent(source="shelved", reason=reason)) -- deliberately
    # NOT set_validity's in-memory audit map, which does not survive a
    # restart.

decisions_query(query: str, *, scope: str, limit: int = 10,
                include_superseded: bool = False) -> list[DecisionLink]   # resource-link blocks

class DecisionLink(BaseModel):
    uri: str; id: HypomnemaId; title: str
    lifecycle: Literal["in_force","superseded","revoked"]
    superseded_by: HypomnemaId | None
    scope: str                     # (0015)

decision(id: HypomnemaId) -> DecisionView

class DecisionView(BaseModel):
    id: HypomnemaId; content: str
    lifecycle: Literal["in_force","superseded","revoked"]
    validity: Literal["valid","invalid"]
    supersedes: HypomnemaId | None            # = chain[-1].id
    superseded_by: HypomnemaId | None
    tip: HypomnemaId | None                   # derived live end of the chain (0013)
    promoted_from: list[HypomnemaId] | None
    recall_count: int; last_referenced_at: datetime | None
    purpose: str | None = None
    created_at: datetime
    rendered: str                             # byte-identical to the resource
```

**Scope arg rule:** required on create/search tools (`save`, `ember`, `digest`, `recall`, `context`, `decisions_query`, `sessions`); id-addressed tools derive scope from the record.

**Projection rule:** model-facing args are semantic (`value`, `include_invalid`, `include_archived`); tuning knobs (`candidate_k`, `min_active_results`, thresholds, β) stay in engine config, never model-facing. `purpose` is **CLI-correctable only** — no model-facing mutation of routing metadata.

**Extraction-prompt rule (ADR 0016 B7):** the `save` and `ember` tool descriptions carry the five purpose values with their one-line anchors — the same anchors idiom as `feedback` — because the tool interface *is* the extraction prompt. Capture guidance beyond the vocabulary lives client-side (CLAUDE.md / agent definitions), never in the engine; the engine observes via `doctor` (§13), it does not police.

### 9.2 Error taxonomy

**Envelope (`isError=false` — the agent branches on normal flow):** `bounced` · `parked` · `degraded` · registered-but-empty scope → empty hits · monotonic re-ops (re-bury, same-value `set_validity`, re-revoke, replayed `confirm_distinct` on either arm → `outcome="already_saved"`) → idempotent ack.

**`ToolError` (`isError=true` — wrong call / missing target):** unknown id · category errors (`feedback` on a Decision, `revoke` on a Fact) · **unregistered scope, purpose value, or role** · `role` omitted on `context`, or `roles` omitted on `set_hot(hot=True)` · pydantic validation failures (surfaced by FastMCP).

---

## 10. Registries and erasure

*(WHY: [ADR 0015](adr/0015-owner-scope-seam.md) Q6/Q7, [ADR 0016](adr/0016-curation-audience-routing.md) B3/B4/B6)*

### 10.1 Three registries, one contract

CLI-declared human acts; agents consume, never mint; unregistered value anywhere model-facing → `ToolError`; persistence behind the seam.

| Registry | CLI | Reserved | Pre-registered |
|---|---|---|---|
| Scopes | `tabularium scope add` | `global` | `global` |
| Facets (purpose) | `tabularium facet add` | — | the five values below |
| Roles | `tabularium role add` | — | `decider`, `builder`, `reviewer`, `navigator` — `default` RETIRED: no longer seeded, and actively DELETED from any store that holds it — `initialize()` calls `delete_role("default")` unconditionally on every engine startup |

### 10.2 Purpose vocabulary (five values, each backed by a default policy row)

`constraint` (hard guardrail — every acting role) · `convention` (house style — producing roles) · `preference` (owner taste — user-facing roles) · `pitfall` (known trap — doing roles) · `reference` (external pointer — scout/doing roles). **Optional, single-valued, kind-agnostic**; `None` is the honest default. `finding` is the documented first candidate registrant, not registered day one. Killed: `grounding`, `rationale`; `warning` renamed `pitfall`. Capture side: the tool descriptions are the extraction prompt (§9.1); client-side discipline owns when/what to save; `doctor` reports coverage — observability, not enforcement; no write-time inference, ever.

### 10.3 Roles

Working functions, never agent products. One vocabulary shared by profiles and hints. Agent→role binding is client config.

**Sub-agent grounding landed:** four profiles, keyed on function rather than agent name: `decider` (the main agent, already shipped, unchanged), `navigator` (scout and research agents; payload mostly static tooling orientation, kept outside the store), `builder` (implementer agents) and `reviewer` (reviewer agents). Only `reviewer` is registered and delivered so far, seeded by `initialize()` itself (§10.1) rather than a manual `tabularium role add` step, since the sub-agent start hook hardcodes `role="reviewer"` and a not-yet-registered role would fail every dispatch. Its payload is purely *learned* (recurring defect classes), the only profile whose effect is measurable. `builder` and `navigator` need no new mechanism, only a hook mapping and, for `builder`, a curation pass; they remain unregistered until that lands. Delivery is a sub-agent start hook mirroring the session-start hook, not a prompt rewrite, with a `role`-keyed `context()` call and a smaller budget than the main agent's.

**All four registered, `default` retired (curation block, 2026-08-03):** `initialize()` now seeds all four of `decider`/`builder`/`reviewer`/`navigator` on a fresh store, and no longer seeds `default` — `role` on `context` and `roles` on `set_hot(hot=True)` are both now REQUIRED, so an unregistered/omitted role is always a loud `ToolError`, never a silent resolution to a fallback. **Task 15 closes the remaining gap:** `delete_role` now exists on `StorageAdapter` and both adapters, and `initialize()` calls `delete_role("default")` unconditionally after seeding the four canonical roles — a store from before this task that already holds a `default` row has it removed the next time the engine starts (§10.1's table).

### 10.4 Erasure (three layers + the invariant)

1. **Whole archive:** delete the store directory (the owner is the data subject; the embedded-store idiom is the boundary).
2. **One record:** `bury` drops content + vector + FTS row — complete because **content lives in exactly one place** (§3.3).
3. **Lineage exception, stated not engineered:** content quoted in a descendant's birth-frozen lineage survives ancestor burial (it is the descendant's identity fact); if it must truly go, bury the quoting record too — a deliberate human escalation.
4. **Scope erasure:** iterate `bury` (a maintenance CLI verb only when a real obligation appears). The monotonic archive holds throughout.

---

## 11. Provider layer

*(WHY: [ADR 0003](adr/0003-provider-layer-litellm-pydantic-ai.md))*

**litellm** (pinned) underneath: provider routing, ordered local→cloud fallback, retry/backoff, embeddings (local **Ollama** first-party — no PyTorch/sentence-transformers), per-call cost/token/latency callbacks. **Pydantic AI** on top for structured-output validation (`output_validator` + `ModelRetry`). **No LangChain / LlamaIndex.**

Tabularium owns exactly three pieces: the **escalation/degradation policy**, the **telemetry sink** — instrumented inside `judge_batch`, not a litellm `CustomLogger`: both configured judge seats (`claude -p`, `codex exec`) are subprocess CLIs that never touch litellm, so a provider-level callback would observe no judge traffic at all (§16) — and the **retry-on-malformed-JSON loop** (litellm won't retry an HTTP-200 bad body).

Every AI surface in the system: embeddings (every write + query) · the offline reconciliation judge (§8.6) · model-aware token counting. Nothing else calls a model. **Telemetry's own scope is narrower than that full surface — judge calls only.** Embeddings are excluded deliberately: an embedding failure already produces a durable park record (§4.7), embedding cost is zero against the local Ollama embedder, embedding quality already has a measurement surface in the eval harness, and embeddings are the highest-volume AI call in the system — instrumenting them would add a row on every save and every recall to populate columns nobody queries.

---

## 12. Configuration (consolidated inventory)

All knobs live in **engine config** — never model-facing (§9.1 projection rule). Per-embedder calibration output is **versioned config, direction-gated on apply** (§4.8). Exact file format/migration = build-time detail; the inventory and placement below are normative.

| Knob | Seed | Owner section |
|---|---|---|
| `rrf_k` | 60 | §5 |
| `β` (importance boost) | eval-swept | §5 |
| `candidate_k` | 50 (floor; `_recall` sizes `max(limit, candidate_k)`) | §5 |
| `min_active_results` (fallback floor) | conservative-small | §5 |
| `auto_ceiling` / `gray_floor` — per-embedder; per-arm (Decision `gray_floor` lower) | golden-set calibrated | §4.8 |
| `P_floor` (calibration precision floor) | — | §4.8 / §14 |
| `min_label_count` (paranoid-mode exit + calibration gate) | — | §4.8 / §8.6 |
| `w_recall / w_recency / w_feedback` | 0.5 / 1.0 / 1.5 | §8.1 |
| `half_life` | 30d | §8.1 |
| feedback EWMA `α` (init 0.0) | 0.3 | §8.1 |
| `archive_threshold` / `revival_threshold` | calibrated; revival a notch above | §8.3 |
| `escalation_threshold` (ember recurrence) | 3 | §8.4 |
| `hot_cap_global` + `hot_cap_scope` (payload-size guardrail, enforced by rejection; per (scope, role), never tuned independently — invariant `hot_cap_global + hot_cap_scope` ≤ smallest consumer budget, whose knobs live outside this repo) | 8 + 5 | §8.5 |
| `hot_promote_threshold` / `hot_demote_threshold` — **dead config since ADR 0020**; no code path reads them | 0.60 / 0.40 | §8.5 |
| lifecycle-sweep cadence / reconciliation cadence | 15–30 min / weekly | §8.6 |
| context total budget + tier budget shares | profile data (`default` profile) | §7 |
| token-counter model binding | consumer model via litellm | §7 |
| store adapter + path; embedder model + dim | store-open config | §3 |
| provider routing/fallback policy | — | §11 |
| delivery profiles | `default` ships | §7 |

Every seed is an **eval-tunable seed, not a claim** (T01 sweeps them).

---

## 13. CLI

The CLI imports the engine library directly (it is the standing proof the engine stands alone).

- `tabularium sync <path>` — idempotent export: every Decision's byte-stable render → one file under `docs/decisions/`; after-run `git status` = exactly the semantic transitions since last sync. Never commits — the commit is a human act; git never enters the engine.
- `tabularium scope|facet|role add <name>` — the three registries (§10.1).
- `tabularium reclaim [--yes]` — CLI surface for the seam's third maintenance method (`StorageAdapter.reclaim`, §3.1, ADR 0017): irreversible version pruning, buys disk only. A separate named command — never a flag on `sync` or any other verb, since a boolean that flips a routine operation into irreversible behaviour is the exact failure this command exists to avoid. Prints the no-other-writer precondition (same as `docs/backup.md`'s cold-copy procedure) and requires interactive `y` confirmation unless `--yes` is passed.
- `tabularium doctor` — facet-coverage report (share of saves carrying purpose, value distribution, hint frequency) · dead-vocabulary flags (registered value no policy row consumes) · park counts [A7] · projection-drift report [A9].
- `tabularium backup …` — deferred verb (§16); the M1 base is a **documented cold-copy procedure** . Not a T-zero gate.
- Seed script (one-time, T-zero).
- Maintenance verbs (build-named): purpose correction (§9.1), projection rebuild [A9], scope-erase iteration (§10.4, only when a real obligation appears).

---

## 14. Eval harness

*(WHY: [ADR 0011](adr/0011-eval-harness-architecture.md))*

**Two modules over one substrate** (golden corpus + injectable `Clock` + event-log replay; `trajectory = f(log, clock)`):

1. **Regression GATE** — pass/fail on every pytest/CI run; **bit-deterministic**; blocks merge.
2. **Calibration harness** — sweeps Pareto frontiers (β, RRF weights, cosine thresholds, archive threshold); outputs versioned per-embedder config; never a gate.

**Direction axis — promote and demote co-equal.** Demote-side: anti-freeze (idle importance strictly decreasing between references — the headline golden regression) · archive at `decay_due_at ± tol` · no false-merge · invalid excluded by default · correct eviction victim. Promote-side (the intelligence tests): importance rises on genuine reference · ember earns codification at `r ≥ escalation_threshold` · Fact earns `hot` at threshold · archived revives on dedup re-assertion · dedup-reinforce increments and logs correctly.

**The inversion:** recall *relevance* is capped to a binary floor-gated smoke test — frozen ~20–40 hand-curated queries, fixed k, recall@k + precision@k ≥ floor; no graded labels, no LLM judge in the loop. Harness weight sits on deterministic trajectory evals (lifecycle correctness is the differentiator).

**Corpus (three sources):** previous memory store export (real input, **never oracle** — the circularity guard: ground truth derives from spec threshold arithmetic; project-b's freeze-bug log = the anti-freeze fixture) · Hypothesis-generated logs · the dogfood log post-build. Form: **~30% concrete named fixtures / ~70% Hypothesis property tests** (properties incl.: anti-freeze generalized; every burial leaves a tombstone; additive floor never breached downward; no false-merge across generated near-pairs; **projection determinism [A9]**).

**Hermetic gate = FROZEN vectors only** (hand-written synthetic for fusion math; snapshotted-real for floors). Never re-embed inside the gate. **Live-embedder lane:** `pytest -m integration` re-embeds the frozen set via local Ollama and re-asserts the floors — env-sensitive by design, human-look on red. Hermetic = no remote provider, not no embedder.

**Gate-blocking:** all property tests · project-b-sourced fixtures · recall/precision floors · dedup precision ≥ `P_floor`. **Report-only:** latency p50/p95 + token cost · model-slip rate (correction-valence embers + invalidated records + burial tombstones) · **LongMemEval_S run once** as a reference number (never CI-wired) · the calibration harness. **LoCoMo: not reported** (corrupted gold set).

**Adapter attribution (2026-07-29):** the recall/precision floors are measured and gated **per storage adapter** (memory reference + LanceDB production), not as one number characterizing "the system" — the two adapters run different keyword-search implementations, so identical rankings were never achievable or required. `P_floor` (dedup) is adapter-independent by construction (pure cosine geometry, no adapter search in the path). See ADR [0011](adr/0011-eval-harness-architecture.md) amendment and an internal planning note.

---

## 15. Dogfooding integration (engine-side)

*(WHY: [ADR 0014](adr/0014-dogfooding-mechanism.md); the practice/build sequencing is not recorded here)*

- **Tabularium is the store of record for its own construction decisions from T-zero.** Committed files under `docs/decisions/` are published renders (one file per Decision, byte-identical to the resource, Decisions only) — never the source.
- **Write loop:** deliberate `save(kind=decision)` speech act → `tabularium sync docs/decisions/` → human commit alongside implementation. No hooks, no conversation extractors (the gate needs an agent to bounce to), no auto-commit.
  - **[A12] Amendment 2026-07-28 — seed mechanics, recording settled practice.** Three clarifications, none reopening the clause above.
    1. **The gate is off for the import, not detuned.** The seed runs with `gray_floor=1.0`/`decision_gray_floor=1.0`, importing the archive whole: zero bounces, zero distinct-pair-registry writes. The dedup gate polices *incoming* saves; retro-applying write-time adjudication to a historical archive already curated under the prior practice is a category error, and blanket-resolving would write unreviewed judgments into the registry as if human, suppressing the M3 offline judge (§8.6 job 3) that exists to do exactly that review. Live thresholds are unchanged — this is a seed-run config, not a calibration change.
    2. **"House shape" is prose, not a template.** Per ADR 0013 §C1, decision content is a free-form blob and the render supplies the frame; structured content fields were considered and rejected there. A house-shape body is a dense paragraph *covering* statement, why, and rejected alternatives — never literal `## Statement / ## Why / ## Rejected` headers. The ADR file's own sectioning is a rendering, not the target form.
- **Cutover:** hard for decisions at T-zero (the previous store's Tabularium records freeze; one pointer record saved there); **phased for grounding** (the split-brain interval is named: session-start hot-load stays on project-b until context composition ships); **no dual-write, ever**.

---

## 16. Deferred register

Everything in-scope but not designed here. Each entry names its graduation trigger. Unmapped-anywhere = coverage-audit defect; mapped-here = a deliberate deferral, not a hole.

| Item | Trigger / placement |
|---|---|
| **Telemetry — landed 2026-08-11, judge calls only** (§11): pass-grain, append-only rows instrumented inside `judge_batch`, each carrying latency, status, judge-seat identity, rubric version, and locally-counted prompt tokens. **Still deferred and unbuilt**: the **escalation/degradation policy** and the **retry-on-malformed-JSON loop** — §11's other two provider-layer pieces did not graduate with it; this row is not "the whole provider-layer design shipped." | Escalation/degradation policy + retry-on-malformed-JSON loop: the first live LLM call (the offline judge) already fired and did not carry them — no new trigger named |
| **Prompt registry** (versioned prompts; no heredocs) — **graduated sideways, not as named**: the Cleaner's rubrics are content-addressed and their version is stamped on every `judge_batch` result, which is the versioning property this row asked for, delivered for the Cleaner's rubrics rather than for "the judge" as scoped, and never called a registry. | Resolved sideways 2026-08-11 — see Item; no registry exists |
| **Store backup/durability design** (verb · cadence · durability) | Deferred, **no T-zero gate**; base = documented cold-copy procedure; trigger: user-scheduled post-T-zero |
| Multimodal content (image/STT for Agent-X) | When Agent-X integration is scoped; content model + embedding seam must not preclude it |
| Resource subscribability | When a consuming host exists AND the SDK can express it; seam preserved (§6) |
| `recall`/`decisions_query` purpose filter param | Cheap future arg; when a pull consumer wants it |
| `finding` purpose value | One `facet add` command, when a policy row consumes it |
| Offline-judge purpose-suggestion for old unfaceted entries | Documented extension, zero commitment |
| Model-slip-rate exact formula · calibration objectives detail · zero-setup public test lane | Build-time choices (ADR 0011) |
| Config file format + migration/validation — **including delivery-profile serialization/versioning** (needed by M2) | Build-time; inventory + placement are §12-normative |
| Scope-erase CLI verb | Only when a real erasure obligation appears |
| Qdrant adapter | **Cut, not deferred (2026-08-02 — ADR 0004 amendment).** Revival trigger: a second deployment target appears, embedded LanceDB hits a concurrency or scale wall (no concurrency mediation today, single-process), or a pre-1.0 LanceDB upgrade breaks compatibility |
| Client-supplied correlation id for exact session attribution (§1.5) | Only if the computed time-range link proves insufficient in practice |

---

## 17. Assembly clarifications index

Pinned during final assembly; each is a resolution of an ambiguity between or within ADRs, or a small additive surface — none contradicts an ADR Decision clause.

- **[A1]** `valence` physical form: nullable column on the Fact arm (semantics per ADR 0006 unchanged).
- **[A2]** `upsert` requires a vector on the normal path — ADR 0012 B2b (park) supersedes ADR 0005's null-on-outage write; backfill ops are for model-version re-embeds only.
- **[A3]** Projection persistence op: `append_event` + `update_projections` compose the ADR 0008 `reinforce` contract; formulas stay above the seam.
- **[A4]** Adapter conformance rule: a committed write is visible to both legs on the next same-connection search (absorbs LanceDB FTS optimize behavior).
- **[A5]** `candidate_scope` (`FactBounce`/`DecisionBounce`, formerly `BounceInfo`, ADR 0022) — cross-tier bounces are visibly cross-tier (extends ADR 0015's hit-origin labeling; answers R07 R6).
- **[A6]** Bounce/park idempotency contract (§4.5) — stale `candidate_id` ⇒ fresh bounce; parked retries are safe via the gate itself (answers R07 G6). A `resolve` call must resubmit the same `content` as the bounced save — every call re-embeds `content` with no server-side pinning, so an edit is a fresh probe, not a stale one. `other_candidates` (§4.4, `FactBounce`/`DecisionBounce`) is additive visibility into the same probe's other above-floor neighbours; it does not change [A6]'s one-candidate-at-a-time resolve semantics.
- **[A7]** Parked outcomes are logged; `doctor` reports park counts (answers R07 R4).
- **[A8]** Canonical feedback formula: `[0,1]` events, EWMA α=0.3 init 0.0, non-negative additive term, **no centering transform** (answers R07 C2).
- **[A9]** Projection rebuild/drift maintenance verb + projection-determinism property test (answers R07 R5).
- **[A10]** `source=compact` projection: counts toward `recall_count` (it consolidates a genuinely-missed re-assertion) but **never bumps `last_referenced_at`** (an offline sweep is not a reference; recency reflects live use only). Resolves an ADR 0009-internal ambiguity.
- **[A11]** Live stats (`recall_count` · `last_referenced_at` · `tip`) appear in `DecisionView` only; `DecisionLink` stays a lean triage link by design — ADR 0013 D3's operative amendment governs over its prose; ADR 0012's lean-triage rationale.
