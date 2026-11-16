# 0017 — Store maintenance is a seam operation; the contract asserts *ranked* visibility

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted — **amends** [0005](0005-hypomnema-schema-and-storage-seam.md) (the storage seam grows a maintenance operation; the write-visibility clause is restated as ranked visibility with an explicit staleness window).

Discovered in production on 2026-07-29, the day after T-zero. Nothing in Tabularium owned index or store maintenance, and in LanceDB **every index is a snapshot taken at build time** — nothing folds later rows in implicitly. The seeded Pi store had run 7 602 saves against a keyword index built once, at `open()`, on an empty table: `num_indexed_rows: 0 / num_unindexed_rows: 7602`. BM25 was matching correctly and *scoring against empty corpus statistics* the entire time.

## The clause that was missing

The seam asserted **visibility** — a committed write is findable on the next search — and `TestWriteVisibility.test_text_leg_sees_upsert` (`tests/conformance/test_conformance.py:229-237`) faithfully checked exactly that: one record written, assert its id is *in* the results. Membership, at a sample size where BM25 statistics cannot be wrong. The adapter's own docstring went further and claimed the property the test did not check — *"rows added to the table are indexed incrementally and visible to `nearest_to_text` on the next query without an explicit `optimize()` call. The conformance suite asserts this"* (`lancedb_adapter.py:34-40`). Half true, and the half it got right is the half the test covered.

What the system needs is **ranked visibility**: findable *and correctly ranked*. That clause is the whole defect. It cannot be strengthened to "always ranked" — holding that true costs a full index rebuild per write (3.7 s at 2 997 rows, measured) — so the contract instead **names a staleness window** rather than denying one:

> After maintenance, every record ranks correctly. Between maintenance runs, records written since the last run may rank stale.

The false docstring is deleted. The conformance suite gains the property with teeth, asserted across both adapters: *write, maintain, assert a record written since the last maintenance ranks where it should.* The reference adapter passes trivially; LanceDB fails today.

## Placement: an explicit verb on the seam

*What* maintenance means is irreducibly adapter-specific — merging fragments and folding indexes is a LanceDB fact, and the in-memory reference adapter has no maintenance debt at all, so it no-ops honestly. *When* it should happen is store-health policy, and therefore a memory semantic, which ADR 0005 places above the seam. That is the same mechanisms-below / semantics-above line the seam was drawn on.

Scoped deliberately to LanceDB: Qdrant is **not** a design input, and the interface is not contorted to pre-satisfy an adapter that does not exist.

## Shape

- **Two operations, never conflated.** *Routine upkeep* = compaction + index fold-in, passing an explicit long `cleanup_older_than` so it never prunes; safe alongside concurrent readers; this is what keeps ranking correct, and the engine triggers it. *Reclaim* = version pruning with `delete_unverified=True`; irreversible, buys only disk, and carries LanceDB's own corruption warning ("only… if you can guarantee that no other process is currently working on this dataset"). Never automatic. Conflating the two is what let an unattended ops script run the irreversible half past its own printed warning banner.
- **Triggered on measured debt**, specifically `stats().fragment_stats.num_small_fragments` — LanceDB's own count of fragments that ought to be merged. Beats a write counter, which must be persisted or it resets, and which goes wrong whenever the store is maintained out of band. Measured read cost: 0.45 / 2.18 / 6.76 ms at 100 / 500 / 1 500 fragments — linear at ~4.5 µs per fragment, so ~35 ms at the 7 602 the Pi reached. The read is expensive exactly when it is about to fix itself. Note the floor is ~1, not 0 — a healthy store still reports one under-target fragment — so the threshold is an absolute count (20, vendor-aligned, configurable), never a `> 0` test.
- **Run synchronously**, on the save that crosses the threshold. One save in N pays a small bounded cost; in exchange there is no background task, no overlapping runs and no scheduler inside an MCP server, and the path stays deterministic and testable.
- **All seven tables** are maintained when a run fires, though only `records` is checked to decide. `records` is not a strict superset — a bounce resolved via `confirm_distinct` writes `pairs` without touching it — so stragglers are covered by maintaining everything rather than by the trigger.
- **Indexes created at `open()`**, idempotently, reusing the existing suppress-if-exists pattern: FTS on `content` (already there), BTree on `id`, Bitmap on `scope`. Verified to work on an empty table. This is the part that makes a store built *from source* born correctly indexed.
- **Upkeep failure never fails the save.** The record is already committed; a maintenance error is operational, not a write outcome. Catch, log, continue, and keep it out of the returned verdict.

## Considered options

- **Below the seam** — the adapter maintains itself invisibly. No interface change, and deep in the leverage sense. Rejected: store health becomes untestable from outside, and the policy hides inside an adapter where no sweep reaches it.
- **Outside the engine** — an ops command in the shape of `sync`. Smallest change, no contract debate. Rejected: an untracked maintenance script run by hand *is* this option informally, and its being outside the tested surface is precisely why the store's health could not be reproduced from source. Formalising the shape that just failed would be a strange conclusion.
- **Background/async upkeep** — nobody ever eats a pause. Rejected for now: untestable in the suite that just proved its own blind spots, can overlap itself, hides failures. A fine optimisation once the synchronous cost is measured and worth removing.
- **No automatic trigger, explicit command only** — zero policy. Rejected: it is how this situation arose. LanceDB's own guidance is to optimize every *20 data modification operations*, and Tabularium performs at least one per save; expecting a human to notice that cadence is the "relies on remembering" failure.
- **An explicit seam verb with the contract clause as the real deliverable** — chosen.

## Consequences

- The strongest objection is recorded rather than dismissed: this is an interface shaped by a single implementation. The verb is a no-op in the reference adapter and would likely be one in Qdrant, whose optimizers are internal. The answer is that the *verb* is unevenly needed but the *property* is universal — Qdrant has the same staleness window, it simply closes it for you — so the clause earns its seat even where the method body is empty.
- Measured effect of running this once against the seeded Pi store: `recall()` 7 163 ms → 152 ms, `get_by_id` 3 052 ms → 4.2 ms, disk 3.68 GB → 29 MB, fragments 7 602 → 1.
- A bare `optimize()` is **not** purely additive — `cleanup_older_than` defaults to 7 days, so it silently prunes week-old versions. The routine path must always pass the window explicitly.
- Bitmap indexes on `lifecycle`, `validity` and `type` are plausible further wins (every recall pre-filter touches them) and are deliberately **not** shipped here — unmeasured. They go to the eval harness as candidates.
- The fourth index candidate — an approximate vector index — was measured and rejected outright rather than left as a candidate; see [0019](0019-approximate-vector-index-dropped.md).
