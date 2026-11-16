# 0007 — Recall pipeline + Context composition (the two recall-family modules)

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted — **amended by** [0008](0008-dedup-gate-and-reinforcement.md) (reinforcement is recorded as an append-only `ReinforcementEvent` log — the source of truth — with `recall_count`/`last_referenced_at` as materialized projections, not an in-place counter), and [0018](0018-bounded-recall-no-unbounded-mode.md) (the candidate pool is sized from the caller's `limit` rather than a fixed config constant; no unbounded mode remains).

Resolves T06 — the headline §4 recall differentiator, designed concretely against the locked model (ADR [0005](0005-hypomnema-schema-and-storage-seam.md) legs-only seam, ADR [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md) Fact taxonomy) and the R05 project-b trace. Amends ADR 0005's seam interface (noted inline + in Consequences). Every trigger/threshold left "T07/T08's to set" stays deferred; this ADR fixes the *recall* behavior only.

## The split: two distinct modules, not one pipeline

The recall family is **two modules behind the engine**, not one pipeline with two modes:

- **Recall** — `recall(query, scope, …) -> RecallResult`. Query-driven hybrid retrieval; fans out the two legs, fuses, ranks. Consumers: the session interface's topic-shift path + the pairwise probe T07's dedup reuses.
- **Context composition** — `compose_context(scope, budget) -> payload`. **Query-less**; bin-packs the highest-attention entries into a token budget for the automatic session opener (project-b's `GroundingService`/`HotContextPayload`).

They have genuinely different interfaces (query-in/ranked-hits-out vs. budget-in/packed-payload-out), different core algorithms (fusion+rank vs. greedy bin-pack), different consumers and failure modes. Unifying them yields one shallow interface branching on "query or budget?" and leaks token-budget machinery into the hot recall/dedup path. They **share primitives** (both read via the seam; composition may *call* Recall for a topic-shift sub-case) — a seam between them, not a merge.

## Module A — Recall (query-driven)

### Fusion: RRF over two relevance legs, importance as a post-fusion boost

The store returns two **relevance** legs (`vector_search`, `text_search`). The engine fuses them **above the seam** with **RRF** (`k` default 60), then applies importance as a **separate, pluggable post-fusion multiplicative boost** — *not* a third RRF leg:

```
relevance = RRF(vector_leg, text_leg)                 # query-relevance only
final     = relevance · (1 + β · importance_norm)     # post-fusion boost stage
```

RRF legs must each be a *query-relevance ranking*; importance is a **query-independent global prior** (same ordering for every query), so as a leg it floats globally-important-but-off-topic entries into the top-k. Multiplicative post-fusion boost keeps relevance (RRF) and value (importance) orthogonal and **cannot manufacture relevance** — an entry neither leg surfaced can't be boosted in. importance-as-a-leg is not even an eval arm; the T01 ablation set is RRF-only vs. RRF+boost with a β-sweep.

**Lifecycle rides this same boost, through importance — no separate knob.** Embers are born-hot/high-importance and `hot` Facts earned high importance, so the importance boost already front-of-minds them; a parallel lifecycle-boost term would double-count attention. (A `hot`-floor guarantee, for a persisted-`hot` Fact whose importance momentarily dips, is a possible eval-tunable later.)

### Validity: exclude by default

Recall **excludes `invalid` records by default** — the primary consumer is an agent grounding *current* work, and surfacing retained-false content is an active hazard and a budget waste. An `include_invalid` param (default `false`) opts specialized consumers in (T01's invalidated-records stream, history/audit queries); included hits **carry their validity flag**.

### Lifecycle: two-tier retrieval, write-side revival

Recall is **two-tier**:

- **Primary pass** surfaces `active | hot | ember`; excludes `archived` (that *is* the meaning of archived).
- **Fallback pass** widens to include `archived` when the primary returns fewer than a small configurable `min_active_results` floor (start conservative; eval-tunable).

**Inclusion ≠ transition.** Routine recall is a pure read and never revives. But a returned **fallback-tier** hit is a genuine reference (the fallback fires only when there was *no* active knowledge and this archived note was the answer): it is **reinforced** (`last_referenced_at` + `recall_count`), and crossing T08's revival threshold performs the `archived → active` transition. So revival is **write-side**, and the fallback tier is the single principled read-side exception. An explicit `include_archived` param (history/deep queries) is separate from the automatic fallback and does not revive.

### Retrieval: no hydration stage, filters pushed to the seam

project-b's "two-stage cheap-column hydration" is **moot** — ADR 0005 already took the embedding off the domain record, so every `Hypomnema` column is cheap. There is **no hydration stage**: each leg returns `Hit{record, score, rank, leg}` with the full record; the engine fuses → boosts → returns top-`limit`. Honestly **parallel** legs (ADR 0006), powered by the store's native ANN + native BM25 *indexes* (ADR 0005 declines only native *fusion*).

All three filters — **scope** (already on the legs) + **validity** + **lifecycle-set** — are **pushed down to the seam** as the store's native metadata pre-filter (`filters: {scope, validity, lifecycle_in}`). Pushed down, each leg returns a full `candidate_k` of *usable* candidates even in a scope dominated by filtered-out rows; over-fetch-then-drop-above-seam would over-fetch by an unknown factor and can silently under-return. **This amends the ADR 0005 seam** (the legs grow the filter object beyond scope). Fan-out: each leg returns `candidate_k` (default ~50, configurable) → fuse → top `limit` (default ~10).

### Graceful degradation (embedder down)

1. **Detect via a bounded try, not a pre-call health probe** — embed the query through litellm's timeout + retry + provider fallback (T11); a *terminal* embed failure is the signal. No pre-flight health check (race + extra happy-path round-trip). Kills project-b's hang bug (project-b had no timeout).
2. **Degraded path = text-leg-only** — vector leg skipped → RRF over the single BM25 leg + the importance boost still applies; recall still returns ranked results.
3. **Signal with a typed field, never silently** — `RecallResult` carries `degraded: bool` + `reason` (`"embedder_unavailable"`) + `legs_used`, surfaced through the MCP response so the agent *and* T01 know the recall was vector-less.
4. **One-directional** — embedder-down → BM25-only = graceful; but BM25 lives *in* the store, so a store outage fails both legs = **hard error**, not a degraded success. (No `require_vector` fail-closed knob — the typed `degraded` field already lets a caller fail-closed on its own side.)

### Recall is a pure read

Ordinary recall emits only hits + the degraded flag. `recall_count` is bumped **only** by genuine reinforcement — re-save (T07), explicit feedback, and the fallback-tier revival hit — never by passive retrieval (retrieval ≠ usefulness; auto-bumping inflates importance for merely-frequently-*queried* topics and puts writes on the hot read path). T06 decides *what recall emits* (nothing); T08 owns *what reinforcement does*.

## Module B — Context composition (session-start)

### Payload: typed sub-budgets, not one ranked stream

Facts and Decisions are **not rank-comparable** — ADR 0005 put `importance_score` on `Fact` only; a `Decision` has none — so the payload cannot be one importance-ranked stream. It is **typed sub-budgets**, three tiers each with its own selection:

- **Embers** at the *session-start-nudge* escalation rung — first, small, high-priority (their designed purpose, ADR 0006). Which embers qualify = T08's threshold; composition consumes the qualifying set.
- **Hot Facts** — the capped attention tier, `importance DESC` then `last_referenced DESC`.
- **In-force Decisions** — by recency / reference-frequency (no importance to sort on), reserved sub-budget.
- **Excluded:** general non-hot active Facts (Recall's on-demand job; they'd bloat the always-on opener).

Reserved per-type sub-budgets with **waterfall spill** of unused budget to the next tier; total + splits configurable/eval-tunable (not project-b's hard-coded 1500 · 35/45/5). **Scope-splitting (user vs project) is a T04 slot** — the structure holds it without resolving owner/scope.

### Packing mechanics

- **Real, model-aware token counting** — count against the *consumer* model's tokenizer via `litellm.token_counter(model=…)`, behind a thin swappable `TokenCounter` seam (prod litellm / test stub / eval-varied). Not project-b's `bytes/4` (under-counts CJK/emoji → overflow).
- **Greedy whole-entry fill** within the sub-budgets — sort by the tier's key, add while it fits, skip an overflowing entry and keep filling. Not an optimal knapsack (noisy value/size signals; Decisions have no value number; the goal is priority-first grounding).
- **Never truncate content** — pack whole entries only; an entry that can't fit whole is **skipped and the skip is observable** (a truncated Decision misrepresents an immutable record; a repeatedly-skipped oversized entry is itself a signal).
- **Presentation order** — rules/nudges first (in-force Decisions, embers) then current context (hot Facts); configurable.

## Interface sketch

```python
# Module A
recall(query, *, scope, limit=10, candidate_k=50,
       include_invalid=False, include_archived=False,
       min_active_results=…) -> RecallResult
# RecallResult = { hits: list[Hit], degraded: bool, reason: str | None, legs_used: list[str] }

# Module B
compose_context(*, scope, budget) -> ContextPayload   # query-less; typed sub-budgets + waterfall
```

## Amendments to ADR 0005

- The retrieval legs grow a structured **`filters: {scope, validity, lifecycle_in}`** (native metadata pre-filter), beyond scope-only.
- Retrieval returns feed a `RecallResult` envelope carrying the **`degraded`** signal (fusion/degradation live in the engine, above the seam — no adapter change).
- `reinforce(id, at)` is emitted by the **write path + the fallback-tier revival hit only**, never by ordinary recall.

## Considered options

- **One unified recall pipeline (two modes)** — rejected: one shallow interface branching on query-vs-budget; leaks token machinery into the hot path.
- **Importance as a third RRF leg** — rejected: a query-independent prior masquerading as a relevance ranking; floats important-but-irrelevant entries in. Not even worth an eval arm.
- **Surface-and-flag invalid by default** — rejected: dumps false statements into the working context every recall; the opt-in param delivers it on demand instead.
- **Exclude archived entirely / read-side revival on any recall** — both rejected: the two-tier fallback resurfaces lapsed truth when it's the only answer, while write-side revival keeps a history peek from silently promoting.
- **Over-fetch then filter above the seam** — rejected: unknown over-fetch factor, silent under-return; the seam already filters scope.
- **Pre-call embedder health probe / `require_vector` fail-closed** — rejected: probe races and taxes the happy path; the typed `degraded` field already lets callers fail-closed.
- **Single importance-ranked payload / optimal knapsack packing / content truncation** — rejected: Facts and Decisions aren't rank-comparable; knapsack over-engineers a soft constraint; truncation misrepresents immutable records.
- **Reinforce recall winners (project-b-ish)** — rejected: inflates on query repetition, conflates retrieved/used, writes on every read.

## Consequences

- The eval harness (T01) swaps fusion strategies and β trivially (own loop, zero adapter involvement) and can measure degraded-mode recall via the typed flag.
- The in-memory adapter (ADR 0005 reference impl) exercises the same `filters` pre-filter — trivial in Python, the seam's honesty check.
- T07 builds its pairwise dedup probe on the same legs + `min_similarity`; the fallback-tier revival hit feeds T07/T08 reinforcement.
- T08 owns every threshold this ADR defers to: `min_active_results`, the `archived → active` revival threshold, ember escalation rungs, `hot`-floor, and what reinforcement does to importance/decay.
- T04 fills the `Scope` slot the two-tier filter and the composition sub-budgets both carry.
- No new glossary terms: T06 is a pipeline design (mechanism); the two domain nouns (**Recall**, **Context composition**) landed at the branch open.
