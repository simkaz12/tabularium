# 0014 — Dogfooding mechanism: store of record, committed artifact, hybrid seed, write loop, build order, previous memory store cutover (T10)

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted

Closes T10 (dogfooding mechanism — capture, bootstrapping, captured shape, provenance proof), which was blocked by T03 and T09.

## A — Store of record

Once the dogfood loop is live, **Tabularium is the primary record of its own construction decisions.** A decision made mid-build is saved into Tabularium first — a deliberate `save(kind=decision)` call in the working session — and anything in the repository is an exported rendering of the T09 decision-record view, never the source. "Ask it why any decision was made" queries the actual store; the committed markdown files answer the same question by way of a published rendering.

**Why.** The original thesis — "dogfood it on its own construction and it becomes its own architecture-decision-record" — is literally false in any other arrangement. The glossary has already committed: "an ADR is one *rendering* of a Decision, not the Decision itself." A files-primary arrangement turns the killer feature into a photocopy of a file cabinet, deflating and dishonest as a demo.

**Accepted cost.** The practice cannot start until the tool minimally works. The pre-tool corpus — ADR files 0001–0013 plus the previous memory store atoms — lives under the current file-and-ledger practice and migrates in at T-zero via the seed. The seed is a cutover, not a convenience copy.

## B — Committed artifact

The commit-as-you-go practice commits **one markdown file per Decision** under `docs/decisions/`, byte-identical to the T09 `decision://{id}` rendered resource. **Decisions only** — Facts, embers, reinforcement events, and counters never touch git. The LanceDB store remains authoritative (A); the files are its published pages, exported renders.

**Why.** T09's byte-stability means a file's diff *is* the decision's semantic transition — when C supersedes B, `c410.md` is born and `a3f2.md` gains exactly one forward-pointer line; nothing else moves. Git's DAG timestamps the pages, discharging ADR [0013](0013-decision-record-view.md) B's provenance obligation. Decision and code land in the same commit, so `git show` answers "what changed and why" in one screen. Pushed history on `main` is **append-only** — no squash, rebase, or force-push; the DAG's probative value depends on it.

**Practice start.** The `docs/adr/` files are the pre-T-zero renderings this ADR codifies as the seed source.

**Deferred.** Backup and durability of the LanceDB store itself — external drive is a candidate — is a separate, later question not resolved here.

## C — Seed source: hybrid

At T-zero, one curated import builds the initial Decision records. The **previous-store records are the skeleton** (one atom → one record; atom granularity so recall answers with the decision, not a document; real supersession links replayed via `save(supersedes=…)` so chains come over intact) and the **ADR files are the content source** (each record's body is rewritten at seed time into the house decision shape — statement + why + rejected, dense free-form per ADR 0013 C1 — folding in the richness the atom compressed away). The `source` field dual-points: `"seed: ledger NNNN · ADR NNNN §X"`.

**Honesty note** (recorded as part of the decision): seed content is new prose written at seed time — a curated re-expression, not a verbatim copy. This is principled: atoms and ADRs are both renderings of the decision (glossary), and the seed is labeled via `source`. Originals stay frozen: ADRs in `docs/adr/` and git history, atoms in the previous memory store.

**Seed scope.** Everything up to cutover — not a fixed 40; T04, T12 decisions continue accruing under the current regime during the gap between now and T-zero, and all go into the seed.

**Cost accepted.** Approximately 40 curated records, roughly one batch of delegated drafting plus spot-checks. The seed corpus doubles as T01's dogfood-source eval fixtures (ADR [0011](0011-eval-harness-architecture.md) three-source corpus).

## C2 — Seed timestamps

Seeded records carry `recorded_at` = **seed day, truthfully**. The real decision date rides as plain text in the `source` line: `"seed: previous memory store · ADR 0012 §B4 · decided 2026-07-21"`, provable via the ADRs' git history.

**Principle.** *The store's timestamps tell the truth about the store; git tells the truth about history.*

**Accepted cost.** Seeded supersession chains render with uniform seed-day dates in their birth-frozen snapshots — the pre-tool history looks flat. Flat-but-true over pretty-but-fabricated; the real chronology survives in source lines and git.

## D — Write loop

A decision travels conversation → commit in three steps:

1. **Capture.** A deliberate `save(kind=decision)` speech act by the working session on Tabularium's own MCP server, at the moment the decision lands. No commit hooks; no conversation-parsing extractors. This is principled, not lazy: the dedup gate (ADR [0008](0008-dedup-gate-and-reinforcement.md)) bounces gray-band duplicates to the calling agent, and a hook has nobody to bounce to.

2. **Export.** One idempotent CLI command — `tabularium sync docs/decisions/` — loops every Decision and writes each byte-stable render to its file. Safely re-runnable; after-run `git status` = exactly the semantic transitions since the last sync; catches the sibling update a session would forget (C supersedes B → B's page gains its forward-pointer line). This CLI *is* the "dogfood-from-code imports the library" consumer ADR [0012](0012-mcp-tool-surface-design.md) B0 promised — the layering argument made flesh.

3. **Commit.** Human, alongside the implementation work it belongs to. Git never enters the engine (ADR 0013 B holds); the CLI writes files and stops.

## E — T-zero: build-order constraint

T10 dictates build order. **Milestone one of the build is the dogfood slice:**

- `save` with the full dedup gate (not a gateless stub — Decisions bounce eagerly by design; a gateless save dogfoods a different tool)
- `recall` (the bounce flow and "ask it why" both need it)
- the T09 render
- the `sync` CLI
- a minimal MCP server exposing `save` / `recall` / `decision` / `decisions_query`

**Not in the slice** (all post-T-zero): context composition, feedback, disposition tools, decay sweeps, ember ladder, calibration.

**T-zero** = the day the slice passes its own T01 hermetic eval gate (ADR [0011](0011-eval-harness-architecture.md)). Ships tested, not on vibes. At T-zero: the seed runs, and from that day every construction decision goes in-tool.

**Why.** The slice is approximately the first third of the build. Everything after — eviction calibration, offline-reconciliation judge, eval corpus fights, the genuinely interesting decisions — gets recorded by the tool itself, live, interleaved in git. The demo upgrades from "we seeded it with our decisions" to "the tool recorded most of its own construction — here are the commits."

**Accepted cost.** Milestone one is fat: schema + seam + LanceDB adapter + embedder + gate + recall + render + CLI + thin server. Every piece is critical-path anyway; the constraint only orders work, it adds none.

## F — Previous memory store cutover

**Hard cutover for decisions; phased cutover for grounding; no dual-write, ever.**

From T-zero, tabularium-scope decisions go into Tabularium, full stop. The previous memory store's tabularium atoms freeze as pre-cutover history — still recallable; seed source lines point into them. One final pointer record is saved there at T-zero: "tabularium decisions now live in Tabularium." Cross-project and user-scope memory stays on the previous memory store; Tabularium inherits only its own project scope.

**Grounding wrinkle (named, not hidden).** The dogfood slice excludes context composition (T10-E), so between T-zero and context shipping there is a split-brain interval: session-start hot-load keeps coming from the previous memory store while decisions already flow into Tabularium. Each store's cutover happens when its module is real — no capability claimed before it exists.

**Mechanics at T-zero.** The repo `.mcp.json` gains the Tabularium server entry; the repo `CLAUDE.md` retargets decision capture from the previous memory store to Tabularium `save`. File-based auto-memory is unaffected (harness plumbing, not project decision storage).

## Considered options

**Store of record (A):**
- **Files-primary with Tabularium as a synced mirror** — the killer feature becomes a photocopy of a file cabinet; deflating and dishonest as a demo. Rejected.
- **Coequal dual-write** — drift by construction; two authored sources with no machine link; "the store is the record" (A) is quietly false. Rejected.

**Committed artifact (B):**
- **Committing the LanceDB binary store** — undiffable, merge-hostile; already rejected in ADR 0013. Rejected.
- **JSONL as the committed artifact** — a backup shape, not a provenance shape; not human-readable or citable. Rejected.
- **External-HDD-only, nothing in git** — an HDD proves no ordering to anyone; would void ADR 0013 B's resolution of the provenance question. Rejected.

**Seed source (C):**
- **Atoms-only verbatim seed** — loses the ADR richness; each atom compresses away edge-case stories and full rejected lists. Rejected.
- **ADR files as records** — wrong granularity; one ADR bundles approximately nine decisions; the tool becomes a file-fetcher. Rejected.
- **No seed** — an empty store fails the self-documenting thesis for the very decisions that built the tool. Rejected.

**Seed timestamps (C2):**
- **Backdating `created_at` via the library import path** — writing records after the fact with earlier timestamps is the literal act the provenance design exists to disprove; it would make `created_at` silently mean "when decided" for seed records but "when saved" for everything after — an invisible inconsistency baked into the archive's first day; violates ADR 0013's "plain, honestly-labeled timestamps." Rejected.
- **A `decided_at` schema field** — schema machinery for a one-time seed event, the exact bloat ADR 0013 B refused. Rejected.

**Write loop (D):**
- **Engine auto-writing files on save/transition** — welds a filesystem side-effect and repo layout into the engine, the exact coupling ADR 0013 kept out; export is a consumer of the engine, not engine behavior. Rejected.
- **Session hand-writing the rendered field to a file** — misses sibling transitions (C supersedes B → B's page update would be forgotten). Rejected.
- **Auto-commit machinery** — git never enters the engine (ADR 0013 B); the commit is a human act alongside the work. Rejected.

**Build order (E):**
- **Build-order-agnostic switch-on** (primitives first, dogfood last) — makes the seed giant, the live era short, and the story retrospective; measurably weaker for zero engineering benefit. Every slice piece is critical-path anyway; the constraint only orders work, adds none. Rejected.

**Previous memory store cutover (F):**
- **Dual-write validation window** — two saves and two dedup-bounce resolutions per decision; stores drift; "the store is the record" (A) becomes quietly false during the window; the committed renders are already the safety net. Rejected.

## Consequences

**All open questions resolved.** With T10 closed, all eight §9 open questions are resolved:
- §9.4 (provenance) — resolved here: commit-as-you-go practice (B + D) + append-only history is the mechanism; git's DAG ordering does the proving.
- §9.5 (dogfooding mechanism) — resolved here: store of record (A), committed artifact (B), hybrid seed (C/C2), write loop (D), T-zero build-order constraint (E), previous memory store cutover (F).
- §9.1–9.3, §9.6–9.8 — closed by earlier ADRs.

**Build-order constraint imposed.** Milestone one of the build = the dogfood slice (save + gate + recall + T09 render + sync CLI + minimal MCP server). T-zero = the day it passes the T01 hermetic eval gate.

**Build deliverables of the dogfood slice.** The `sync` CLI (`tabularium sync <path>`) and the seed script are explicit deliverables of milestone one, not post-build tooling.

**Append-only history.** Pushed history on `main` is append-only — no squash, rebase, or force-push. This is a practice rule, not a git configuration; its load-bearing role (the DAG ordering is what makes the provenance credible) means violating it defeats ADR 0013 B's resolution of the provenance question

**Sync CLI as library consumer.** The `tabularium sync` CLI is the promised "dogfood-from-code imports the library" consumer from ADR 0012 B0 — it imports `tabularium` engine directly (not via MCP transport), confirming that the engine stands alone, which the eval harness also depends on.

**Closes T10.** Frontier after: **T04** (owner/scope seam, recommended next) + **T12** (curation/audience routing).
