# 0006 — Fact taxonomy: lifecycle, the validity axis, and monotonic dispositions

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted

Resolves the Fact-side half of T05 (the Decision-side landed in ADR [0001](0001-decision-immutable-content-mutable-lifecycle.md); the record shape + storage seam in ADR [0005](0005-hypomnema-schema-and-storage-seam.md)). Defines the states a Fact moves through, adds a truth axis shared by both arms of the union, and fixes how anything leaves active memory. Amends ADR 0005's seam + field inventory and clarifies (does **not** weaken) ADR 0001's automation invariant — noted inline.

## Three orthogonal axes

A hypomnema carries three independent axes. Conflating them is precisely the project-b mistake we greenfield away from — project-b's single `kind` enum tangled *type* + *attention* + *removal*, plus a separate `is_hot` flag and a soft-delete, which is the root of its decay-freeze and stale-flag bugs (R05 §6):

1. **Lifecycle** — attention / relevance state. Fact: `active | hot | ember | archived`. Decision: `in_force | superseded | revoked` (ADR 0001).
2. **Validity** — truth state, **shared by both arms**: `valid | invalid`. A mutable, *bidirectional* toggle (the environment changes truth both ways). It is metadata *about* content, never an edit of it — so it does not violate a Decision's content-immutability.
3. **Valence** — embers only: `correction | guideline`, carried on `source`. Orthogonal to lifecycle; the eval harness filters on it.

## FactLifecycle state machine

- **`active`** — normal, importance-ranked, decaying.
- **`hot`** — the capped, contended session-start attention tier. **Earned only** (importance + winning a slot; LLM-judge eviction when the tier is full) — never born-into. **Persisted** (the eviction verdict must stick, not be silently re-derived on every read as importance jitters); `hot_rejection_count` lives here. Collapses project-b's `pinned` *kind* + `is_hot` *flag* into one state: importance-promotion needs no discrete kind, and keeper-status is emergent from reference frequency (ADR 0001), never a write-time "keep forever" flag.
- **`ember`** — a Fact **born hot at write-time**: a transient, self-decaying, directional nudge ("a warm coal in the pocket that keeps pointing the right way"). The one legitimate write-time-seeded attention path, distinct from earned `hot`. Fades to `archived` if unreferenced; matures to `active`/`hot` if reinforced. **Valence-neutral** (axis 3): a *correction* (an agent broke a guideline) or a neutral *guideline/pointer*.
- **`archived`** — decayed out of active recall but **retained and revivable**: re-reference restores it to `active` with its history intact. Symmetry with a superseded Decision "leaving active recall but staying queryable." Reversible — unlike burial.

**Birth states:** a Fact is born `active` (normal save) or `ember` (write-time nudge) — never `hot`.

T05 fixes the *states and legal edges*. Every **trigger/threshold** — when decay fires, when `hot` evicts, when an ember escalates, recurrence counts — is **T08 / T07 mechanics**, not decided here.

## Dispositions — a monotonic archive

How anything leaves active memory, most-retained → least:

- **archive** (Facts only) — the reversible decay exit; the Fact is kept whole.
- **invalidate** (both arms) — `valid → invalid`; the whole record is kept, marked false. For wrong-but-meaningful content (env changed / turned out false / derived-from-something) it carries signal (history + eval), so it is never removed. The `in_force + invalid` cell is a genuine Decision state ("still nominally in force, but its premise turned out false") that supersession and revocation cannot express.
- **bury** (both arms) — the strongest removal, for *truly-useless* garbage (a slip / mis-save / hallucination carrying no signal): the content is dropped but a **tombstone** is always left (id, type, reason, timestamp, model-generated flag). Explicit / manual, never automated.

**There is no true hard-delete.** Even garbage leaves a tombstone, so the archive is **monotonic** — Tabularium never fully forgets. The discriminator between *invalidate* and *bury* is **signal**: does the entry still carry meaning worth keeping? "Bury only garbage" is a **policy** (a human judgment at burial time), not a structural guarantee like `decay = Facts-only` (which the discriminated union enforces); burial is a sharp, explicit, rare tool that always carries a `reason`. (Dropping buried content also eases a future erasure/GDPR story — only metadata remains — parked at T04.)

## Ember escalation — Fact→Decision by creation

A recurring ember escalates up a ladder: fresh ember → session-start nudge → codified rule. The top rung is a **Decision**, created by **promotion-by-creation** — a *new* Decision that references the source embers via `promoted_from: list[HypomnemaId]`; the embers stay Facts and fade. The discriminator stays **immutable** (a record never changes type), preserving ADR 0005's structural guarantee. Suggested-not-automatic (ADR 0001 spirit); recurrence thresholds are T07 / T08. The `promoted_from` link is a self-documenting win: a rule that points at the corrections that justified it ("this exists because you corrected me five times — here are the instances").

## Recall shape (carry-vs-re-derive, Q2)

Recall runs its two legs (`vector_search`, `text_search`) **in parallel**, then fuses above the seam (RRF, ADR 0005). It is honestly **parallel**, not a narrowing "two-stage" pipeline — project-b's "two-stage" was only a cheap-column select (R05 §4). No claim the code can't back.

## Considered options

- **project-b's tangled `kind` enum** (type + attention + removal in one field, plus `is_hot` and soft-delete) — rejected: conflates three orthogonal axes; the source of project-b's decay-freeze and stale-flag bugs.
- **`hot` as a computed view** (enum `active | archived` only) — rejected: an expensive, race-guarded eviction verdict can't be re-derived cheaply per read, and it strands `hot_rejection_count`.
- **A distinct `pinned` state** (4-state) — rejected: reintroduces a write-time "keep forever" flag; importance-promotion is already a float, keeper-status is emergent.
- **Folding `ember` into `hot` + `source`** — rejected: overloads `source` (which now carries valence), makes a neutral guideline-ember inexpressible, and lets corrections evict knowledge from the capped tier.
- **Terminal `archived`** (no revival) — rejected: loses reinforcement continuity and fights the T07 dedup path (a re-write of lapsed-but-true knowledge should revive, not duplicate).
- **True hard-delete / prune** (nothing retained) — rejected: a removal is usually a model slip-up, so the burial *event* is eval signal; every removal leaves a tombstone.
- **Facts-only validity** — rejected: the `in_force + invalid` Decision state is too useful to the self-documenting / eval story.
- **Fact→Decision by type-mutation** — rejected: breaks discriminator + content immutability.
- **Chosen:** three orthogonal axes; `active | hot | ember | archived`; shared `valid | invalid`; archive / invalidate / bury with no hard-delete; ember escalation by creation.

## Consequences

- **Amends ADR 0005**: strike `delete(id, reason)` → **`bury(id, reason)`** (both arms, writes a tombstone) + **`set_validity(id, valid, reason)`** (both arms, bidirectional); add **`validity`** to `BaseHypomnema` and **`promoted_from: list[HypomnemaId]`** to `Decision`; **`FactLifecycle = active | hot | ember | archived`** fills the enum slot ADR 0005 left open; **`valence`** (correction/guideline) rides `source` for embers.
- **ADR 0001 stands unamended**: burial and Fact→Decision codification are *explicit owner actions*, never automation — so the "automation may never auto-retire or delete a Decision" invariant holds and nothing *decided* is ever lost.
- **Feeds T06 / T07 / T08**: the typed states + legal edges are what recall / dedup / eviction build against; every trigger and threshold is theirs to design.
- **Feeds T01**: the model-slip corpus = correction-valence embers + invalidated records + burial tombstones (three streams).
- **Feeds T12**: valence-on-`source` and the facets slot interact with curation / audience routing.
- **Noted future (MAP "Not yet specified")**: multimodal (image / STT) memory for Agent-X — the content model + provider/embedding seam must not be designed to preclude it.
