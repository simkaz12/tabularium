# 0019 — The approximate vector index is dropped, not deferred

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted — **amends** [0004](0004-vector-store-lancedb-swappable-seam.md) (LanceDB's native approximate-nearest-neighbour capability is declined by measurement, not merely unused pending scale) and retires the "50 K note for future maintainer" in `lancedb_adapter.py`.

Discovered in production on 2026-07-29, the day after T-zero, alongside the maintenance defect fixed by [0017](0017-store-maintenance-and-ranked-visibility.md). The adapter's own module docstring told a future maintainer to build an IVF index once the store passed ~50 000 rows. Two things were wrong with that note before it was ever acted on: the ~50 K premise, and the parameter it told the maintainer to pass.

## What was found

The 50 K premise was falsified on Pi hardware: a measurable slowdown appears at 500–1 000 rows, two orders of magnitude below the stated threshold — the figure was evidently calibrated for workstation CPU/memory bandwidth, not a Raspberry Pi 5.

The note's own code sample does not run. It says to call:

```python
config=lancedb.index.IvfFlat(num_partitions=256, num_sub_vectors=96)
```

`IvfFlat.__init__` on the pinned `lancedb==0.34.0` has no `num_sub_vectors` parameter — that parameter belongs to `IvfPq`. Following the note literally raises `TypeError`. A future maintainer would have hit a broken instruction, not a working shortcut.

## What was measured

An `IvfFlat` index was actually built and measured, rather than argued about, against about 3k real rows through the real `LanceDBAdapter`, real `nomic-embed-text` embeddings:

- **Speed:** 1.01× on a fragmented store, 1.05× once compacted — no meaningful speedup at this scale, with or without the fragmentation defect ADR 0017 fixes separately.
- **Correctness:** mean top-10 overlap against exact brute-force search was **0.918** over 40 real queries, with per-query spread 0.60–1.00. Several queries silently lost 3–4 true neighbours — this is not noise, it is the index doing its job (approximating) at a cost the speed number does not justify.

Every other optimisation lever tested (a BTree index on `id`, a bitmap index on `scope`, fragment compaction) is semantically neutral — it changes nothing about *which* rows match or how they rank, only how fast the answer arrives. The approximate vector index is the only lever tested that is not neutral: it changed the result set for zero measured gain.

## Considered options

- **Build it at ~50 K rows, as the original note said.** Rejected: the premise is falsified (the real threshold on the deployed hardware is 500–1 000, not 50 000) and the note's own code sample does not run against the pinned version.
- **Build it now, since Tabularium already runs a live Pi store.** Rejected: the measured trade is negative at every scale tested — no speed win, and a real correctness cost. There is nothing to buy at the price it charges.
- **Drop it — exact brute-force search is the permanent design.** Chosen. `bypass_vector_index()` (`lancedb_adapter.py`) stays, now because exact search is the decided answer, not because a scale threshold hasn't been crossed yet.

## Consequences

- The module docstring's "Vector search" section, the "50 K note for future maintainer," and the `vector_search` docstring no longer mention building an approximate index — the stale instruction is deleted rather than updated, so a future reader inherits a conclusion instead of a suggestion.
- `bypass_vector_index()` is now explained by this ADR, not by a row-count threshold.
- This is not permanently closed: if the corpus one day grows far beyond present scale, the question can be reopened — but it reopens against fresh measurement and the per-adapter eval floors (ADR 0011), which by then will actually reflect the production search path. Under today's evidence there is nothing to defer; the lever loses.
- No conformance or eval-harness change follows from this decision — brute-force exact search was already the only path exercised; this ADR records why no other path is coming.
