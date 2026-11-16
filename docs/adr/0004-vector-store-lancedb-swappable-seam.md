# 0004 — Vector store: LanceDB primary, in-memory + Qdrant adapters, behind a swappable seam

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted — **amended by** [0005](0005-hypomnema-schema-and-storage-seam.md): the seam exposes raw legs only (`vector_search` / `text_search`); hybrid fusion (RRF) moved **above** the seam into the engine — strike "hybrid fusion (RRF)" from the required-operations list below.

Tabularium's vector store is **LanceDB** (embedded, native in-process hybrid: vector + BM25 + RRF + rerank), pinned (pre-1.0), behind a **runtime-swappable store seam**. The seam ships three adapters: **LanceDB** (default/production), an **in-memory** reference (hermetic tests + the reference implementation of the interface), and **Qdrant** (the server-scale production adapter that proves the seam is real). Replaces project-b's hand-rolled cosine loop (§4-OUT).

## Considered options

- **Qdrant as primary** — the more mature engine (well past 1.0, richest fusion: RRF+DBSF+ColBERT). Rejected *as primary*: its real engine needs Docker/Cloud (local `:memory:` mode is testing-only and feature-gapped), and off-Cloud you hand-build client-side sparse-vector generation. That breaks the self-contained, zero-infra shape the tool needs to be **run and dogfooded day-to-day** (dogfooding from commit one is the flagship feature), and makes the headline hybrid *more* DIY, not less. The store is not the differentiator; the recall/dedup/eviction on top of it is.
- **Milvus-lite as 2nd adapter** — also in-process hybrid, but Beta, a 2-month-old rewrite, segment-local BM25 IDF (weak on small corpora), heavier deps. Passed over for Qdrant, which proves the seam across a real embedded↔server structural difference (two near-identical embedded stores prove less).
- **sqlite-vec / pgvector** — vectors-only (fusion hand-built; pgvector `ts_rank` ≠ BM25). **Chroma** — OSS hybrid is Cloud-gated. All rejected for the hybrid-recall differentiator.
- **LanceDB primary + in-memory + Qdrant** — chosen.

## Consequences

- LanceDB is pre-1.0 (v0.34.0) → pin the version; the swappable seam is the mitigation and its reason to exist.
- BM25 + RRF + rerank are native and in-process → the recall pipeline (T06) builds hybrid *on* the store, not by hand.
- The seam's required operations: upsert/merge, vector top-k, FTS/BM25 search, hybrid fusion (RRF), scope filter, delete, get-by-id. The *typed* seam interface + the hypomnema schema are the immediate next design step.
- Qdrant is a need-driven swap for server-scale, not a default. Resolves T02 (backend + adapters + BM25 + seam operation-set).

---

## Amendment (2026-08-02) — the Qdrant adapter is cut

**Owner decision, 2026-08-02.** The Qdrant adapter described above is cut from the plan, not merely deferred. Its sole justification in this ADR's own words was that it "proves the seam is real" — a proof, not a capability. The seam already has that proof: two adapters (LanceDB + the in-memory reference) share one dual-parameterized conformance suite that has caught real defects. A third adapter built only to re-demonstrate what is already demonstrated would add cost without adding evidence.

The cost this ADR itself already priced in still stands: Qdrant's real engine needs Docker/Cloud, which this ADR rejected *as primary* for breaking "the self-contained, zero-infra shape" the tool needs to run and dogfood daily. The Pi deployment — the live daily-dogfood target — deliberately runs no Docker at all, so building the adapter now would mean building infrastructure the deployment cannot use.

This ADR's **Decision** and **Considered options** sections above stand unedited as the record of what was decided in July 2026 (Qdrant considered and accepted as a planned second adapter, rejected only *as primary*). This amendment supersedes that plan going forward; it does not retroactively rewrite it.

**Revival trigger** (build the Qdrant adapter when, and only when, one of these actually happens):

- a second deployment target appears,
- embedded LanceDB hits a concurrency or scale wall (it has no concurrency mediation today and runs single-process),
- or a pre-1.0 LanceDB upgrade breaks compatibility.

Living docs updated in the same pass: `docs/SPEC.md` §3 and the §16 deferred register. ADR [0005](0005-hypomnema-schema-and-storage-seam.md)'s Consequences line — "Two production adapters (LanceDB, Qdrant) + the reference = a real seam by the two-adapter rule" — is now historically-true-but-superseded by this amendment; ADR 0005 itself is left unedited, per the same frozen-ADR rule this amendment follows.
