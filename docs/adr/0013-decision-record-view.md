# 0013 — Decision-record view: frame rendering, byte-stable resource, provenance as practice (T09)

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted

Closes T09 (decision-record view — lineage rendering richness, provenance, resource subscribability), which was deferred from ADR [0012](0012-mcp-tool-surface-design.md) B6.

## Organizing principle: lineage is captured at birth and frozen

A Decision's lineage — `supersedes` (what it replaces) and `promoted_from` (what justified it) — is part of its identity, like `created_at`. Both are captured at birth and frozen. This single principle drives every rendering and schema decision in this ADR:

- The resource is byte-stable because lineage is frozen and backward history is immutable.
- `promoted_from` stores snapshots (not live pointers) because the justification must reflect the world as it stood when the decision was codified.
- The supersession chain is a birth-frozen list of snapshots (not live pointers or derived queries) because a new decision's statement of what it replaces is itself part of the decision.
- The forward pointer (`superseded_by`) is set exactly once, coinciding with the decision's own lifecycle transition — so rendered bytes change if and only if the decision has its own semantic transition.

Net effect: **the rendered resource is a pure function of the decision's own row** — zero ancestor reads at render time; byte-stability follows by construction rather than by argument.

## B — Provenance de-scoped to practice

Provenance is a property of the **dogfooding practice** (T10), not an engine feature.

the original project brief's "timestamps correlated to git history" — the mechanism meant to kill the "written after the fact" skeptic — is resolved as: **correlation done by the reader against the repo, enabled by T10's commit-as-you-go practice.** T10's practice commits decision artifacts continuously, interleaved with implementation commits. Git's DAG ordering does all the proving: backdating a decision record would require rewriting all subsequent pushed history.

The T09 view renders plain, honestly-labeled timestamps (`recorded_at`, `superseded_at`) and nothing more. No provenance subsystem, no git dependency, no new schema fields. `created_at` already exists on `BaseHypomnema` (ADR [0005](0005-hypomnema-schema-and-storage-seam.md)).

**Why.** Real-tool-first — the in-engine correlation machinery (content-hash manifests, earliest-containing-commit lookups, anchor columns) serves only a skeptical reader and adds zero probative value. The proof strength comes entirely from the practice; rendering the correlation is presentation sugar. A decision driven purely by portfolio appeal violates the project's decision-driver rule (ADR [0010](0010-project-posture-private-indefinite-full-build.md)).

This makes T10's commit-as-you-go practice **load-bearing** for the original project brief's provenance resolution. T09 hands the obligation to T10.

## C — Frame rendering

The rendered view is a **metadata frame around verbatim immutable content**, never a parser.

**(C1) Content is a free-form blob.** `save(content: str)` is unchanged (per ADR [0012](0012-mcp-tool-surface-design.md)). The render wraps content with column-derived truth — a status line containing lifecycle, validity, recorded date, and (if superseded) supersession date — then prints the body untouched. The frame is the view's contribution; the body is the decision as spoken.

**(C2) Title is derived.** The title slot in `DecisionLink` (ADR [0012](0012-mcp-tool-surface-design.md)) is filled by **the first sentence of `content`, truncated to ~80 characters**. No `title` column, no `title` arg on `save`. Decision content naturally front-loads its subject; an explicit title arg would need the derivation fallback anyway.

**(C3) Schema amendment — the supersession chain is birth-frozen.** `Decision` gains `supersession_chain: list[SupersedesSnapshot]` — its full backward history, captured frozen at birth (empty for a chain root):

```python
class SupersedesSnapshot(BaseModel):
    id: HypomnemaId
    title: str            # derived from the predecessor's content at snapshot time
    created_at: datetime
```

When C supersedes B, one transition sets `C.supersession_chain = B.supersession_chain + [snapshot(B)]` and `B.superseded_by = C.id`. This is well-defined because a chain only grows at the tip: B's own chain is already birth-frozen, and B's snapshot fields are immutable at snapshot time (`title` derives from immutable content, `created_at` never changes). The immediate predecessor is `chain[-1]`; `DecisionView.supersedes` (ADR [0012](0012-mcp-tool-surface-design.md)) remains its bare id. Assumes supersession targets an in-force decision — chains are linear.

The initial resolution here was a bare `supersedes: HypomnemaId | None` column; a burial edge case forced the widening to the full snapshot chain — see Considered options (D).

## D — Byte-stable resource + subscribability de-scoped

**(D1) Subscribability de-scoped; seam preserved.** `resources/subscribe` is de-scoped from the build:

- No dogfood client can consume it: Claude Code does not subscribe to resources; R04 flagged host support as uneven.
- FastMCP v1 cannot express it (requires low-level server mixing).
- Declaring the capability without emitting notifications is a claim the code can't back (the original project brief).

The seam cost is negligible: all state changes flow through single engine paths (`transition`, `set_validity`) — a later notification hook is one event emission at one place. Choosing the resource primitive (T03/R04) already preserved native subscribability. The capability is NOT declared until it is real.

**(D2) The resource doc is byte-stable.** The `decision://{id}` resource renders:

- Own content (verbatim)
- Status line (lifecycle + validity + recorded date + superseded date if applicable)
- Full **backward** supersession history — rendered entirely from the birth-frozen `supersession_chain`, zero ancestor reads
- **One-hop forward pointer only** (the immediate `superseded_by`, if set)

Structural fact: a supersession chain only grows at the tip. Therefore backward history is immutable at birth; the forward pointer is set exactly once, coinciding with the decision's own lifecycle transition. **Rendered bytes change if and only if the decision has its own semantic transition.** When C supersedes B: B's bytes update (B gains its one-hop forward pointer and fires a future notification), C is born, A is untouched.

**(D3) Consequences of byte-stability.**

- **No "jump to current tip" link** in superseded resource docs. A tip link in every chain member's doc would drift on every subsequent supersession — the drift eliminated by D2, re-entering through the link. Instead, the model-driven `decision()` tool exposes a derived `DecisionView.tip` field; tool output is transient, not subscribable, so volatile fields are free there.
- **Reference stats leave the resource.** `recall_count` and `last_referenced_at` are live-changing fields; placing them in the resource doc would break byte-stability. They move into `DecisionView` and `decisions_query` — the model-facing views that are transient by design. The resource is the stable, citable artifact (like a published ADR page); live evergreen-vs-dated signals live in the model-facing views.
- **`DecisionView` amendment.** ADR [0012](0012-mcp-tool-surface-design.md)'s `DecisionView` gains three fields: `tip: HypomnemaId | None` (the current live end of the chain, derived), `recall_count: int`, and `last_referenced_at: datetime | None`. `DecisionView.rendered` is byte-identical to the resource content — one render function, two exposures.

**(D4) The supersession "group" is a derived view, never a stored row.** The chain is fully reconstructible from `supersedes` / `superseded_by` pointers. Chain identity, if ever needed, is the root's id.

## E — `promoted_from` snapshot-at-birth

`promoted_from` widens from `list[HypomnemaId]` (ADR [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md)) to `list[PromotedFromSnapshot]`:

```python
class PromotedFromSnapshot(BaseModel):
    id: HypomnemaId
    content: str
    valence: Literal["correction", "guideline"]
    recurrence_count: int
```

The snapshot is captured **frozen at promotion time**. The Decision is born carrying its justification as it stood when codified.

**Why.** Live ember state changes over time — the fade-to-archived lifecycle transition, the moving recurrence counter, a burial into a tombstone. Any of these would silently change the promoted decision's rendered bytes — the drift eliminated by D2, sneaking back through live `promoted_from` pointers. The snapshot is also historically honest: "this rule exists because of these five corrections, as they were when they justified it." A later ember burial cannot retroactively hollow out the justification.

The render shows embers inline from the snapshot with valence and recurrence count, yielding the payoff: "this exists because you corrected me five times — here are the instances."

## Strawman render (illustrative)

```markdown
# a3f2 — Feedback range is [0,1], not [-1,+1]

**Superseded** by [d410] on 2026-08-02 · recorded 2026-07-21 · valid
scope: tabularium · source: grill T03

---

Feedback value is a float in [0,1], anchors bad≈0.0 / neutral≈0.5 / helpful≈1.0 …
*(content verbatim)*

---

## Lineage

[7c19] 2026-07-14 → **[a3f2] this** 2026-07-21 → superseded by [d410] 2026-08-02

- supersedes [7c19] "Feedback range [-1,+1] EWMA" (2026-07-14)

## Promoted from

- [e881] correction ×5 — "you keep proposing negative feedback values"
- [f209] guideline ×3 — "prefer float anchors over enums for judge-style scoring"
```

## Considered options

**Provenance (B):**
- **Git-anchored append-only export with content-hash manifest + earliest-commit correlation** — most verifiable on paper; adds zero probative value over the practice alone; every record requires a git lookup at render time; git becomes a seam the engine must depend on. Rejected.
- **Committing the binary LanceDB store** — undiffable, merge-hostile; the files git tracks would be the store artifact, not the decisions. Rejected.
- **Render-time git-log juxtaposition computed by the engine** — suggestive but not probative (a bad actor rewrites commits before the lookup too); adds a git dependency; the result is presentation sugar not proof. Rejected.
- **External trusted timestamping (OpenTimestamps / RFC 3161)** — the known stronger alternative to DAG-ordering; external dependency for marginal gain; neither OpenTimestamps nor RFC 3161 is in scope for a self-contained embedded memory system. Rejected; named here as the principled next step if stronger provenance is required.

**Frame rendering (C):**
- **Structured decision content** (`rationale` / `rejected_alternatives` fields on `save`) — taxes every write with form-filling; real dogfood decisions are dense single paragraphs that don't decompose cleanly; fields would rot; the ADR format is one rendering of a Decision, not its required shape. Rejected.
- **Explicit `title` arg** — needs the derivation fallback anyway (content always exists; title might not be supplied); derivation alone is sufficient. Rejected.
- **Derive `supersedes` by inverse-query** (scan `superseded_by` to infer the backward pointer) — one scan per backward hop per render; the birth-frozen chain makes the render zero-read. Rejected.

**Byte-stability (D):**
- **Full forward chain in every member's resource doc** — every supersession in the chain touches every ancestor's doc (drift); the initial strawman; caught before any decision was recorded. Rejected.
- **Reference stats in the resource doc** — live-updating fields break byte-stability and notification semantics. Rejected.
- **"Notifications ignore ember drift" carve-out on `promoted_from`** — treating live pointers specially is complexity that the snapshot eliminates cleanly. Rejected.
- **Bare-id `supersedes` column** (the initial C3 resolution) — backward titles would derive from ancestors' *live* content; burial of a mid-chain ancestor (legal per ADR [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md): both arms) tombstones its content and silently changes every successor's rendered bytes — the same drift, re-entering through burial. Rejected.
- **Single-hop `SupersedesSnapshot`** (snapshot only the immediate predecessor) — fixes the direct predecessor but deeper titles still require walking ancestor rows, and a buried intermediate's tombstone loses *its* snapshot too — the same failure one hop deeper. The full birth-frozen chain closes the hole completely and eliminates the walk. Rejected.
- **Tombstone keeps lineage fields** — would amend the tombstone's minimal definition (ADR [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md)) and still costs k ancestor reads per render. Rejected.

## Consequences

**Schema amendments:**

- **Amends ADR [0005](0005-hypomnema-schema-and-storage-seam.md):** `Decision` gains `supersession_chain: list[SupersedesSnapshot]` — the full backward history, birth-frozen; immediate predecessor = `chain[-1]`; empty for a chain root. The seam's `transition` op sets `new.supersession_chain = old.supersession_chain + [snapshot(old)]` and `old.superseded_by = new.id` in one call.
- **Amends ADR [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md):** `promoted_from` on `Decision` widens from `list[HypomnemaId]` to `list[PromotedFromSnapshot]` (snapshot-at-birth, frozen; carries `id`, `content`, `valence`, `recurrence_count`).
- **Amends ADR [0012](0012-mcp-tool-surface-design.md):** (a) `DecisionView` gains `tip: HypomnemaId | None`, `recall_count: int`, `last_referenced_at: datetime | None`; (b) `DecisionLink.title` derivation is now defined (first sentence of content, truncated ~80 chars); (c) `DecisionView.rendered` is byte-identical to the `decision://{id}` resource content — one render function, two exposures.

**Reinforces ADR [0001](0001-decision-immutable-content-mutable-lifecycle.md):** the provenance chain (the self-documenting audit trail) survives because content is immutable. Freezing lineage at birth is the same principle applied to the relational pointers, not just the content.

**Provenance responsibility:** T09-B makes T10's commit-as-you-go practice **load-bearing** for the original project brief's provenance resolution. T10 must commit decision artifacts continuously and interleaved with implementation commits; that is now the practice this design depends on, not a nice-to-have.

**Closes T09.** Unblocks T10 — both of T10's blockers (T03 + T09) are now closed.
