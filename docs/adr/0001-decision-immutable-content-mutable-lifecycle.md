# 0001 — A Decision has immutable content but a mutable lifecycle

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted

Tabularium separates a Decision's **content** (immutable — never edited, so the "what we chose and why" audit chain survives intact) from its **lifecycle** (mutable — a Decision can be superseded or revoked). This is a deliberate divergence from the previous memory store, which treats decisions as immutable *and* never-decaying: project-b has a `superseded_by_id` column but never wires it into recall or decay, so superseded decisions pile up in active recall forever (R05 §5–6). Change is modeled as **supersession** (a new Decision references the one it replaces) or **revocation** (retire with no replacement); the retired Decision leaves active recall but is never deleted.

## Considered options

- **Mutable decisions** (edit in place) — rejected: destroys the provenance chain that *is* the self-documenting differentiator (T09).
- **Immutable + time-decay** (project-b's model) — rejected: age ≠ staleness. Evergreen decisions ("we store vectors in LanceDB") stay in force indefinitely; dated ones ("try X this sprint") should retire. Time-decay conflates them.
- **Immutable content + explicit supersession/revocation, relevance by reference not age** — chosen.

## Consequences

- Recall ranking for Decisions uses a **reference-based** relevance signal (recall count / last-referenced), *not* time-decay — a divergence T08 (importance/decay) must honor.
- Evergreen-vs-dated is **emergent** (supersession + reference frequency), never a write-time flag.
- **Invariant:** automation may only re-rank or *suggest* retirement — an on-demand LLM-judge compaction pass proposes candidates — but may **never** auto-retire or delete a Decision. One wrongly-evicted decision breaks the "ask it why any decision was made" demo.
- Feeds T05 (extract-vs-greenfield boundary), T08 (importance/decay), T09 (decision-record view).
