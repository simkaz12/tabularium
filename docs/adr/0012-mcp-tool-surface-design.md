# 0012 — MCP tool-surface design (T03)

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted — **amended by** [0018](0018-bounded-recall-no-unbounded-mode.md) (`exhaustive` leaves the `recall` tool surface — with no unbounded mode remaining, the parameter has no referent).

Closes T03 Q2 (exact tool surface — args/returns/error modes) and Q4 (library-underneath boundary). Q1 (MCP-first) and Q3 (primitive mapping) were closed in ADR [0002](0002-mcp-interface-official-fastmcp.md).

## Six cross-cutting principles

These principles are the "why this is a deep surface" story — every tool decision below traces back to one or more of them:

1. **Engine-deep / MCP-thin** — the engine owns every invariant; MCP tools are near-1:1 projections.
2. **Semantic-in / numeric-config-below** — model-facing args are semantic (`exhaustive: bool`, `value: float`); tuning knobs stay in engine config.
3. **Read-degrades / write-parks** — reads have no durable consequence so partial answers beat none (BM25-only degraded); writes are durable so a partial write is worse than a retryable non-write (park on embedder fault).
4. **Speech-act splits** — `save` vs `ember` and the three disposition tools reflect distinct speech acts, not surface ergonomics.
5. **Envelope vs ToolError** — normal-flow outcomes (bounce, parked, degraded) go in a structured envelope; wrong-call / missing-target raise `ToolError`.
6. **Interface honesty via truthful annotations** — splitting the disposition tools is what makes each tool's MCP annotation honest.

## B0 — Layering (resolves Q4)

Three layers, top to bottom:

1. **`tabularium.server`** — the FastMCP adapter (ADR [0002](0002-mcp-interface-official-fastmcp.md)). Near-1:1 `@mcp.tool` projections: translate args in, map engine result-models → structured output and engine exceptions → `ToolError`. ZERO memory semantics at this layer.
2. **`tabularium` engine** — the deep module. Owns every invariant: dedup verdict (ADR [0008](0008-dedup-gate-and-reinforcement.md)), degraded envelope (ADR [0007](0007-recall-pipeline-and-context-composition.md)), monotonic dispositions (ADR [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md)), never-truncate, RRF fusion. Typed pydantic-in / pydantic-out methods returning the same result models the MCP tools expose.
3. **Storage seam** (ADR [0005](0005-hypomnema-schema-and-storage-seam.md)) — below the engine.

**Deletion test (the core rationale):** the eval harness (T01, ADR [0011](0011-eval-harness-architecture.md)) drives the engine directly, and dogfood-from-code (T10) imports the library — two non-MCP consumers exist from day one, so the engine must stand alone. This makes the original project brief's "MCP-native with a thin Python library underneath" literally true, but *inverts* the naive reading: the library is the substance, MCP the thin part.

**Design principle carried throughout:** Tabularium's memories are an impactful, decision-leaning source. The surface biases toward capturing decision-grade content, not a generic append-everything store. This shapes `save`'s defaults, the precision-favored dedup bias (ADR [0008](0008-dedup-gate-and-reinforcement.md)), and the first-class decision resource.

**Rejected:** MCP-first with the library as a thin wrapper re-calling tools — fails the deletion test (eval and dogfood would either boot a server or duplicate logic) and scatters invariants into transport code.

## B1 — Surface skeleton: 10 tools + 1 resource

**Small-and-deep, not verb-sprawl.** The full surface:

| Tool / Resource | Engine method | readOnly | Notes |
|---|---|---|---|
| `save` | `save(…)` | no | write → dedup gate → verdict envelope |
| `ember` | `save(…)` preset | no | speech-act split; shared engine gate |
| `recall` | `recall(…)` | yes | hybrid RRF |
| `context` | `compose_context(…)` | yes | query-less session-start |
| `feedback` | `feedback(…)` | no | importance signal |
| `set_validity` | `set_validity(…)` | no | idempotentHint |
| `bury` | `bury(…)` | no | destructiveHint + idempotentHint |
| `revoke` | `revoke(…)` | no | idempotentHint; Decision-only |
| `decisions_query` | `decisions_query(…)` | yes | returns resource-link blocks |
| `decision` | `decision(…)` | yes | structured view + `rendered` markdown |
| `decision://{id}` *(resource)* | — | yes | host-facing rendered markdown |

**Rejected extremes:** a single `memory(action=…)` god-tool (opaque, no per-tool schema or annotations); project-b-style one-verb-per-tool (~11 shallow tools — the surface this beats).

## B2 — `save`

```python
save(
    content: str,
    *,
    kind: Literal["fact", "decision"] = "fact",
    scope: str,
    source: str | None = None,
    supersedes: HypomnemaId | None = None,
    promoted_from: list[HypomnemaId] | None = None,
    resolve: BounceResolution | None = None,
) -> SaveResult
```

`kind` is caller-declared (default `"fact"`). The engine cannot reliably infer fact-vs-decision — it is a semantic act with different lifecycle and dedup bands. Most writes are facts; a decision is the explicitly marked act.

**Verdict envelope** — one typed union so the agent always branches on the same shape:

```python
class SaveResult(BaseModel):
    outcome: Literal["saved", "reinforced", "revived", "reconciled", "bounced", "parked"]
    hypomnema: HypomnemaRef | None = None
    bounce: FactBounce | DecisionBounce | None = None
    reason: str | None = None

class BounceExit(BaseModel):
    action: Literal["reinforce", "confirm_distinct", "supersede", "promoted_from"]
    how: str
    when: str

class FactBounce(BaseModel):
    bounce_kind: Literal["fact"]
    candidate_id: HypomnemaId
    candidate_content: str
    similarity: float
    guard_flag: str | None
    verdict: Literal["near_dup", "decision_adjacent", "guard_trip"]
    exits: tuple[BounceExit, BounceExit]

class DecisionBounce(BaseModel):
    bounce_kind: Literal["decision"]
    candidate_id: HypomnemaId
    candidate_content: str
    similarity: float
    guard_flag: str | None
    verdict: Literal["near_dup", "decision_adjacent", "guard_trip"]
    exits: tuple[BounceExit, BounceExit, BounceExit]

class BounceResolution(BaseModel):
    candidate_id: HypomnemaId
    action: Literal["confirm_distinct", "reinforce"]
```

**[Amended 2026-08-06 — ADR [0022](0022-bounce-envelope-names-exits.md).]** `BounceInfo`/`suggested_action` are replaced by `FactBounce`/`DecisionBounce`, discriminated on the kind being **saved** (never the candidate's kind), each carrying named `exits` instead of a concluded action — the engine names the exits and chooses none. `suggested_action`'s `drop` value is gone (declining every exit is expressed by making no call); `verdict`'s `decision_supersede` is renamed `decision_adjacent`.

`bounced` is the one outcome demanding follow-up (ADR [0008](0008-dedup-gate-and-reinforcement.md) — nothing is written until the agent acts). The other non-`saved` outcomes (`reinforced`, `revived`, `reconciled`) are successful writes needing no agent action.

**B2a — Bounce resolution** is a small `resolve` object, not a bare boolean. It earns its keep by naming the guard-trip false-alarm path (`action="reinforce"` — a bare `confirm_distinct` cannot express this). Supersession is expressed via `save(supersedes=candidate_id)` (write-by-creation per ADR [0001](0001-decision-immutable-content-mutable-lifecycle.md)); declining every exit is expressed by not re-calling.

**B2b — Embedder outage at write** → **PARK, not degraded-write.** The embedder is a 24/7 co-located dependency (local Ollama, part of the Tabularium service footprint); its absence is a service fault, not a routine degraded mode. `save` writes nothing and returns `outcome="parked", reason="embedder_unavailable"` (retryable; the caller still holds the content). Nothing is lost because nothing is half-written.

**Principled read/write asymmetry (principle 3):** recall *degrades* (BM25-only, ADR [0007](0007-recall-pipeline-and-context-composition.md) — a read has no durable consequence; a partial answer beats none). `save` *parks* (a write is durable; a partial write is worse than a retryable non-write). Backfill seam ops exist for model-version re-embeds, not the normal write path.

## B2½ — `ember`

```python
ember(
    content: str,
    *,
    scope: str,
    valence: Literal["correction", "guideline"],
    source: str | None = None,
) -> SaveResult
```

A separate tool by speech-act distinction: `save` asserts durable knowledge; `ember` leaves a decaying directive carrying `valence` first-class. The separation buys legibility (the model sees "leave a correction" rather than `save(lifecycle=ember, valence=…)`) and keeps `valence` off plain `save`.

The engine has one deep save/gate; `ember` is a thin projection presetting `lifecycle=ember` and `valence`, sharing the same dedup gate. Recurrence (matched a valid ember → reinforce + bump counter → escalation ladder) is a dedup outcome, so `ember` also returns the `SaveResult` envelope.

## B3 — `recall` and `context`

```python
recall(
    query: str,
    *,
    scope: str,
    limit: int = 10,
    include_invalid: bool = False,
    include_archived: bool = False,
    exhaustive: bool = False,
) -> RecallResult

class RecalledHit(BaseModel):
    id: HypomnemaId
    kind: Literal["fact", "decision"]
    content: str
    lifecycle: str
    validity: Literal["valid", "invalid"]
    score: float

class RecallResult(BaseModel):
    hits: list[RecalledHit]
    degraded: bool
    reason: str | None
    legs_used: list[str]
```

**Projection rule (refining B0):** MCP tools expose semantic args; engine-internal tuning knobs stay in engine config and are never model-facing. `min_active_results` is an example of a knob that stays below.

**No hard candidate cap** (amends ADR [0007](0007-recall-pipeline-and-context-composition.md)'s `candidate_k=50`): a fixed cap is a silent invisible ceiling — a memory ranked 51st by both legs becomes unrecallable in principle. Engine `candidate_k: int | None = <config default>` where `None` = exhaustive (full scope-filtered scan through both legs). This earns a model-facing semantic knob: `exhaustive: bool = False` on the recall tool, meaning "search the entire archive, accuracy over latency." The cap is never an unpassable wall.

**Lean hit view:** internal fields (`decay_due_at`, counters, raw leg ranks) stay engine-side. `validity` rides along so an `include_invalid=True` caller can distinguish which hits are false.

**`context` — query-less session-start:**

```python
context(scope: str, budget: int) -> ContextPayload

class ContextPayload(BaseModel):
    hot_facts: list[RecalledHit]
    embers: list[RecalledHit]
    in_force_decisions: list[RecalledHit]
    skipped: int
    budget_used: int
```

**Presentation order: hot → ember → decision** (amends ADR [0007](0007-recall-pipeline-and-context-composition.md)'s decisions-first default; ADR [0007](0007-recall-pipeline-and-context-composition.md) marked order configurable, so this sets the new default). Rationale: orient → adjust → ground (what is happening now → nudges modifying how to handle it → the choices surrounding it). As the session-start payload, the last section sits closest to the incoming task; decisions-last keeps the governing frame freshest at query time.

`skipped` and `budget_used` surface the never-truncate guarantee as an observable budget signal. Budget reservation and spill logic are unchanged (ADR [0007](0007-recall-pipeline-and-context-composition.md)).

## B4 — `feedback`

```python
feedback(
    id: HypomnemaId,
    value: float,            # Field(ge=0.0, le=1.0)
    reason: str | None = None,
) -> FeedbackResult

class FeedbackResult(BaseModel):
    id: HypomnemaId
    feedback_score: float
    recorded: bool
```

`value` is a float in `[0.0, 1.0]`, anchors: `bad ≈ 0.0` / `neutral ≈ 0.5` / `helpful ≈ 1.0` (in the tool description; `Field(ge=0, le=1)` in the pydantic model). This amends ADR [0009](0009-importance-decay-eviction.md)'s `[-1, +1]` feedback range.

**Float-not-enum:** the anchored 0→1 float is the standard LLM-judge scoring idiom; models produce calibrated scalars with named anchors routinely. An explicit `neutral` distinct from no-feedback is real EWMA signal (neutral pulls the average toward the midpoint; silence leaves it untouched).

**Range shift `[-1,+1]` → `[0,1]` is principled:** it makes feedback a purely additive, non-negative term. Bad feedback can erode a prior boost (EWMA drifts toward 0) but can never punish an entry below the floor or force archival. Feedback cannot become a backdoor eviction lever; demotion stays with the explicit disposition tools. This reinforces ADR [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md)'s conservative-removal model.

**Fact-only:** `feedback_score` is a Fact attribute (ADR [0005](0005-hypomnema-schema-and-storage-seam.md)/[0009](0009-importance-decay-eviction.md)); Decisions derive relevance from reference-frequency, not feedback (ADR [0001](0001-decision-immutable-content-mutable-lifecycle.md)). Calling `feedback` on a Decision id is a category error → `ToolError` (the agent should learn the rule, not receive a silent no-op).

**About relevance quality, not truth:** `unhelpful ≠ wrong`. Wrongness is the `set_validity` axis. This preserves the three-axis orthogonality (lifecycle × validity × valence). `feedback_score` echoed in the return is the observability hook for the agent and the eval harness (ADR [0011](0011-eval-harness-architecture.md)).

## B5 — Dispositions: three precise tools (amends B1)

The single `revise` enum from the initial B1 sketch is **replaced by three precise tools**:

```python
set_validity(id: HypomnemaId, valid: bool, reason: str) -> ReviseResult
bury(id: HypomnemaId, reason: str) -> BuryResult
revoke(id: HypomnemaId, reason: str) -> ReviseResult

class ReviseResult(BaseModel):
    id: HypomnemaId
    lifecycle: str
    validity: Literal["valid", "invalid"]

class BuryResult(BaseModel):
    id: HypomnemaId
    tombstone_id: HypomnemaId
```

- **`set_validity`** — flip the truth axis; bidirectional as a bool (invalidate + revalidate in one tool, matching the glossary: validity is a bidirectional toggle). Applies to both `Fact` and `Decision`. Not destructive (reversible). `idempotentHint`.
- **`bury`** — strongest removal; always leaves a tombstone. Applies to both arms. `destructiveHint: true` + `idempotentHint`. The lone destructive tool.
- **`revoke`** — retire an in-force Decision with no replacement. Decision-only (not applicable to Facts; calling on a Fact id → `ToolError`). Monotonic retirement; the record persists queryable. `idempotentHint`, not destructive.

`reason` is required on all three. Supersession remains `save(supersedes=…)` (write-by-creation, ADR [0001](0001-decision-immutable-content-mutable-lifecycle.md)).

**Primary reason for the split — interface honesty (principle 6):** MCP annotations cannot be honest on a mixed enum. `bury` is destructive, `set_validity` is reversible, `revoke` is a retirement — annotating one `revise` tool as destructive would over-warn every validity flip. The split lets each tool carry a truthful annotation. The arm-applicability constraint (`revoke` is simply not a Fact tool) also becomes unrepresentable at the type level rather than a runtime branch, and future dispositions get precise verbs rather than enum bloat.

Cost: surface grows from 7 to 9 tools. The count that matters for per-turn model load is the hot-path tools (`save`, `ember`, `recall`, `context`). Rare, precisely-named governance tools are easier for a model to reason about than an overloaded enum.

## B6 — Decisions view: resource + tool

**Two consumers, two shapes:**

1. **`decision://{id}` resource** (mime `text/markdown`, read-only) — application-driven. For the host or user to display, pin, or cite a decision in rendered markdown.
2. **`decision(id: HypomnemaId) -> DecisionView` tool** (readOnly) — model-driven. For the agent to reason over structured fields, plus a `rendered: str` markdown field that also covers hosts that cannot render resources.

```python
class DecisionView(BaseModel):
    id: HypomnemaId
    content: str
    lifecycle: Literal["in_force", "superseded", "revoked"]
    validity: Literal["valid", "invalid"]
    supersedes: HypomnemaId | None
    superseded_by: HypomnemaId | None
    promoted_from: list[HypomnemaId] | None
    created_at: datetime
    rendered: str
```

Not a redundant duplicate — the resource serves the host rendered markdown; the tool serves the model structured data and doubles as the render guarantee regardless of host resource-UI support. (R04 flagged uneven host resource support as a real T10 risk.)

**`decisions_query`** returns resource-link blocks, not inline content, because decisions are long (an ADR is heavy; inlining N decisions is token-hostile). The agent triages on title/lifecycle then reads by URI. This is also why `decisions_query` stays separate from `recall` rather than being `recall(kind=decision)` — different content weight, different return shape.

```python
decisions_query(
    query: str,
    *,
    scope: str,
    limit: int = 10,
    include_superseded: bool = False,
) -> list[DecisionLink]

class DecisionLink(BaseModel):
    uri: str                  # → MCP resource_link block
    id: HypomnemaId
    title: str
    lifecycle: Literal["in_force", "superseded", "revoked"]
    superseded_by: HypomnemaId | None
```

`include_superseded=False` default — in-force decisions by default; history is opt-in (the decision analog of `recall`'s `include_archived`).

**Deferred to T09:** lineage rendering richness, timestamps ↔ git-history provenance correlation (kills the "written after the fact" skeptic, the original project brief), and resource subscribability (notify on supersession — a preserved capability).

## B7 — `scope`

`scope: str`, required on all create/search tools (`save`, `ember`, `recall`, `context`, `decisions_query`). Id-addressed tools (`feedback`, `set_validity`, `bury`, `revoke`, `decision`) do not take `scope` — a record's scope is derived from its id.

A plain string now is a forward-compatible seam: T04 later widens `Scope` from a bare string into whatever it needs (owner identity, scope_kind + scope_id like the previous memory store, facets, GDPR/erasure hook) behind the same arg name. Tool signatures barely move. T03 commits only that `scope` is a required explicit namespacing param and does not decide the owner model, kind/id split, or erasure semantics. Deliberately not adopting project-b's `scope_kind` + `scope_id` split now — that richness is exactly what T04 owns.

## B8 — Error taxonomy + annotations

**Governing split:** does the agent reason about it as normal flow → structured envelope (`isError=false`); or did it call wrong / target doesn't exist → `ToolError` (`isError=true`)?

**Envelope (isError=false):**
- `save` bounce (`outcome="bounced"`)
- `save` parked (`outcome="parked"` — a known typed "retry later" condition, not a usage error; most legible as a branchable outcome)
- `recall` degraded (ADR [0007](0007-recall-pipeline-and-context-composition.md))
- `recall` / `decisions_query` on unknown or empty scope → empty hits (absence ≠ error)
- Monotonic re-ops (re-bury / `set_validity` to same value / re-revoke) → idempotent ack

**ToolError (isError=true):**
- Unknown id (target does not exist)
- Category error: `feedback` on a Decision id, `revoke` on a Fact id (the agent should learn the rule, not swallow it)
- Bad enum / out-of-range value / missing required arg → pydantic validation error, surfaced automatically by FastMCP

**MCP annotations:**

| Tool | readOnlyHint | idempotentHint | destructiveHint |
|---|---|---|---|
| `recall`, `context`, `decisions_query`, `decision` | true | — | — |
| `set_validity`, `revoke` | — | true | — |
| `bury` | — | true | **true** |
| `save`, `ember`, `feedback` | — | — | — |

`bury` is the lone destructive tool. Clients treat annotations as advisory (untrusted unless the server is trusted).

## Considered options

- **One `memory(action=…)` god-tool** — opaque, no per-tool schema or annotations, model must re-read arg docs every call. Rejected.
- **Project-b-style one-verb-per-tool (~11 shallow tools)** — the surface this design beats; each tool has minimal depth and the annotation honesty problem still exists on the disposition verbs. Rejected.
- **Single `revise(disposition=…)` enum** — correct that governance acts are rare; wrong that bundling beats precision for rare acts (the per-turn-reasoning-load argument is weakest exactly where the acts are rare). Annotation honesty and unrepresentable-invalid-states win. Rejected.
- **`recall(kind=decision)` instead of `decisions_query`** — decisions are long; inlining N is token-hostile; the return shape is resource-links not hit objects. Rejected.
- **Degraded-write + backfill on embedder outage** — over-engineers a graceful path for a dependency that should be always-on; park gives the never-lose-a-decision guarantee without the machinery. Rejected.

## Consequences

**Amendments to prior ADRs:**

- **Amends ADR [0009](0009-importance-decay-eviction.md):** feedback value range changed from `[-1, +1]` to `[0, 1]` (additive, non-negative; conservative-removal consistency).
- **Amends ADR [0007](0007-recall-pipeline-and-context-composition.md):** (a) `candidate_k` hard cap removed — engine `candidate_k: int | None = <config default>`, `None` = exhaustive; (b) context presentation order default changed from decisions-first to hot → ember → decision.
- **Amends ADR [0005](0005-hypomnema-schema-and-storage-seam.md):** storage seam leg ops (`vector_search`, `text_search`) accept `k: int | None` where `None` = exhaustive (full scope-filtered scan).

**Framework unchanged:** ADR [0002](0002-mcp-interface-official-fastmcp.md) (FastMCP, pin `mcp>=1.27,<2`) is unaffected.

**Closes:** T03 (all four questions). Unblocks T09 (decision-record view lineage richness) and T10 (dogfooding — the library's public API is now fully specified as the engine methods, returning the same pydantic models, callable without MCP).
