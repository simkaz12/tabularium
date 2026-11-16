# 0015 — Owner/scope seam: no owner, two-tier registered scopes (T04)

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted

Closes T04 — all three ticket questions: (Q1) owner posture, (Q2/Q2b) scope structure + granularity, (Q3/Q4/Q5/Q6/Q7) seam design, storage enforcement, vocabulary registration, and the erasure story. Resolves project-b's §7 single-user-hardcode weak spot.

## Q1 — No owner

Tabularium models **no owner concept**. One deployment = one store directory = one human's archive — the embedded-DB idiom, where the LanceDB directory is the tenant boundary the same way a SQLite file is. No user table, no owner column, no resolver, no principal argument on any tool.

Multi-owner means running a second store. If owner identity ever mattered (it does not now), it would attach as a facet of the scope key — ADR [0012](0012-mcp-tool-surface-design.md) B7's promised widening: one type widens, zero tool signatures move.

**Why this resolves project-b's weak spot.** Project-b's failure mode was implicit single-tenancy: the owner was resolved as the first user row, in several call sites — a resolver that *guesses* the principal. With no owner concept, there is nothing to guess; the bug is structurally unwritable. Agents (Claude Code, Agent-X) are agents of the same owner; agent identity rides `source`, never scope. Auth and multi-tenancy are explicitly out-of-scope (the original project brief).

## Q2 + Q2b — What a scope is: structure and granularity

**Scope = a flat validated name.** Engine-side: `Scope = NewType('Scope', str)`. Wire stays `scope: str` (ADR [0012](0012-mcp-tool-surface-design.md) B7 — one type widens, zero tool signatures move).

**One reserved name: `global`** — the owner-wide tier. All other names are peer named scopes. The engine owns exactly this two-tier semantic: reads see the named scope + `global`; `global` is its own explicit write target. No nesting, no hierarchy, ever — "project:x:sub" paths and three-tier trees are rejected as speculative generality; finer partitioning belongs to T12 facets.

**Grounding.** Project-b's own schema proves the kind+id two-field structure collapses: `user`-kind rows have `scope_id` normalized to the lone user's id — zero information carried. The reserved name is `global`, not `user`, because no users exist (Q1).

**Scope granularity: a working context (product/effort), never a repo.** A cross-stack pair shares one scope: `project-a-api + project-a-web → project-a`; `a project's API repo + its mobile client → one scope`. The repo→scope binding is client-side config (each repo's `.mcp.json` / `CLAUDE.md`); the engine never knows what a repo is.

**Why.** Project-b's per-repo project ids split cross-stack knowledge arbitrarily — an API-contract decision filed under whichever repo the session happened to run in, invisible from the partner repo. Product-unit scopes heal the pair.

**Consequence for the T-zero seed (ADR [0014](0014-dogfooding-mechanism.md) C).** Project-b's per-repo pair ids merge into single scopes at import; scope names are chosen at seed time.

## Q3 — Read span

A **READ** in named scope X (`recall`, `decisions_query`, `context`) sees **X + `global`**, always — no flag. A read on `scope=global` sees global alone (no parent). A **WRITE** targets exactly one scope.

Hits are labeled with their origin scope: `RecalledHit` and `DecisionLink` each grow a `scope: str` field (T04 amendment to ADR [0012](0012-mcp-tool-surface-design.md) result models) so the agent distinguishes an owner-wide guideline from a project fact.

**Why.** The governing entry for a project query often lives in global ("never add inline comments"); strict single-scope reads force every consumer to remember a separate global call — smeared-knowledge disease, the same failure mode as project-b's owner resolver. Folding the two-call pattern into the engine once removes the smear. Noise risk is handled by ranking plus the global tier's small size.

**Strict single-scope** remains available as an engine-method parameter for internal consumers (eval hermetic fixtures) — never model-facing tool surface.

## Q4 — Machinery boundedness

**(1) Dedup gate probe span = read span.** "The gate guards the recall space, so it probes what recall sees": save into X probes X + `global`; save into `global` probes `global` alone; sibling named scopes never probe each other. Cross-tier verdicts are normal: `reinforce` may land on a `global` record from a project-targeted save; Decisions bounce eagerly across the tier boundary (ADR [0008](0008-dedup-gate-and-reinforcement.md) unchanged).

**Why probe-target-only fails.** Project-b's behavior: a project-local near-copy of a global guideline writes through, then the spanning read (Q3) surfaces both records forever, drifting. The gate would fail exactly at the tier boundary; the probe span must match the recall span.

**(2) Hot tier: per-scope arenas.** Every scope, including `global`, has its own capped contended hot tier. Project-b's single 5-slot user tier generalizes to per-scope arenas. Eviction contests never cross scopes.

**(3) Composition: no scope quotas.** ADR [0012](0012-mcp-tool-surface-design.md) type sub-budgets (`hot → ember → decision`) each fill by rank from the merged X+`global` pool. Project-b's hardcoded 35/45 user/project split is eliminated; per-scope hot caps already bound contributions naturally. A scope ratio, if evals ever want one, is numeric-config-below.

**(4) Sweeps are store-wide; eviction is per-scope.** Decay is record-local; one `decay_due_at` indexed scan covers all scopes. Seam scan/backfill maintenance ops are the one legitimately scopeless read family — engine-internal, never model-facing.

## Q5 — Storage enforcement + scope type

**One physical table.** `scope` = an indexed filter column (LanceDB scalar index; rides ADR [0007](0007-recall-pipeline-and-context-composition.md)'s existing `filters` object on both legs).

**Isolation enforced at the seam as an interface-shaped guarantee.** No scopeless model-facing read is expressible: `scope` is required-kw-only on both seam legs (ADR [0005](0005-hypomnema-schema-and-storage-seam.md)) and on every read/write engine method. The only scopeless operations are the engine-internal maintenance family (scan/backfill/sweep).

**Scope type.** `Scope = NewType('Scope', str)`. Pattern: `^[a-z][a-z0-9-]{0,31}$`. `global` reserved. No normalization magic — no case-folding; what you wrote is what it is. Pydantic validates at the tool boundary; the seam trusts validated input.

**Why.** Project-b's isolation was actually correct (indexed scope columns); its bug was the guessed owner, not the filter. Table-per-scope is a multi-owner pattern — Q1 deleted the owner — and every Q3 spanning read would become a two-table merge.

## Q6 — Registered scope vocabulary

Scopes are a **registered vocabulary**, not open. A scope is declared once via CLI (`tabularium scope add <name>`) — a human act, like `sync`; agents consume scopes, never create them; no MCP tool. `global` is pre-registered at store init.

Registry = tiny metadata behind the same storage seam (aux-persistence family like ADR [0008](0008-dedup-gate-and-reinforcement.md)'s event log: `create_scope` / `list_scopes` ops). No rename, no delete for now.

**Any model-facing call naming an unregistered scope → `ToolError`.** This amends ADR [0012](0012-mcp-tool-surface-design.md) B8: the line moves — registered-but-empty scope → empty hits (absence ≠ error, unchanged); unregistered scope → `ToolError` (wrong call — the same split B8 already draws for unknown ids).

**Why.** Scope names are model-typed strings on every call. Open vocabulary lets `save(scope="tabulraium")` silently fork a ghost namespace — entries invisible to correctly-scoped sessions; ghost reads indistinguishable from "no memories yet." Same silent-failure species as project-b's silent BM25 degradation. Registration makes the ghost-namespace bug structurally unwritable. Cost: ~9 CLI commands ever, one per product scope.

## Q7 — Erasure story (resolves ADR [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md)'s parked GDPR hook)

Three layers + one invariant:

**(1) Whole archive.** The owner IS the data subject (Q1). Erasure = delete the store directory. Complete by construction — the embedded-store idiom provides the boundary.

**(2) One record.** `bury` drops the content AND every derived index of it (vector row, FTS row). Enabled by a **new invariant** promoted from ADR [0008](0008-dedup-gate-and-reinforcement.md)'s accidental correctness: **content lives in exactly one place** — the event log, distinct-pair registry, and counters carry ids and timestamps, never content; tombstone `reason` describes why removed, never quotes content.

**(3) Lineage exception, stated not engineered.** Content quoted in a descendant Decision's birth-frozen lineage (ADR [0013](0013-decision-record-view.md) `PromotedFromSnapshot` carries content; `supersession_chain` carries derived titles) survives ancestor burial — it is the descendant's own identity fact (a book quoting a retracted paper). Remedy if quoted content must truly go: bury the quoting record too — a deliberate human escalation. The spec states this openly; an interviewer finds a decision, not a hole.

**(4) Scope erasure.** No new machinery — iterate `bury` over the scope's records (maintenance CLI verb only when a real obligation appears; pure iteration over an existing operation). The scope stays registered holding tombstones; the monotonic archive (ADR [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md)) holds.

## Considered options

**Owner identity:**
- Owner_id column + principal threading from day one — row-level multi-tenancy for a tool with exactly one user forever; an owner column without auth is worse than none (unanswerable "who validates the principal?"). Rejected.

**Scope structure:**
- Flat opaque strings with no engine semantics — the "this one is owner-wide" knowledge smears into every caller. Rejected.
- Project-b-style `scope_kind + scope_id` struct — breaks the B7 wire promise; a 2-value kind with one vestigial id carries zero information. Rejected.
- Hierarchical scope paths ("project:x:sub") — speculative generality; the inherited workload has no three-tier shape. Rejected.

**Scope granularity:**
- Scope-per-repo — a repo is an accident of stack layout, not a knowledge boundary; project-b's split filed cross-stack decisions by whichever repo the session ran in. Rejected.

**Read span:**
- Strict single-scope reads — silently miss governing guidelines; smeared-knowledge disease. Rejected.
- `include_global` flag on read tools — exposes plumbing; violates ADR [0012](0012-mcp-tool-surface-design.md)'s semantic-in principle; the default would be True and nobody ever flips it. Rejected.

**Storage:**
- Table-per-scope physical partitions — spanning reads become two-table merges; exhaustive scans multiply; scope creation becomes DDL; a multi-owner pattern without a multi-owner need. Rejected.

**Vocabulary:**
- Open first-write-creates vocabulary (project-b's way) — silent ghost-namespace bug; same failure species as project-b's silent BM25 degradation. Rejected.

**Erasure:**
- True purge (rows + tombstones gone) — breaks the monotonic archive for a scenario layer 1 already handles. Rejected.
- Scrubbing lineage snapshots on ancestor burial — silently rewrites Decision identity; the exact thing ADR [0013](0013-decision-record-view.md) prevents. Rejected.

## Consequences

**Fills ADR [0005](0005-hypomnema-schema-and-storage-seam.md)'s `Scope` slot.** `Scope = NewType('Scope', str)`, pattern `^[a-z][a-z0-9-]{0,31}$`, `global` reserved. `scope: Scope` required-kw-only on all model-facing read/write engine methods and on both seam legs.

**Amends ADR [0012](0012-mcp-tool-surface-design.md):**
- `RecalledHit` and `DecisionLink` grow `scope: str` (hit origin, so the agent distinguishes project facts from global guidelines).
- B8 error line moves: registered-but-empty scope → empty hits (unchanged); unregistered scope → `ToolError`.
- B7's promised `scope: str` widening lands with tool signatures untouched.

**Amends ADR [0007](0007-recall-pipeline-and-context-composition.md).** Composition merges X+`global` by rank from the type sub-budget pools; no scope quotas; project-b's hardcoded 35/45 split eliminated; per-scope hot caps bound contributions naturally.

**Sharpens ADR [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md).** `bury` drops content AND every derived index; the parked GDPR hook is resolved. The monotonic archive holds (no true purge).

**Reinforces ADR [0013](0013-decision-record-view.md).** The lineage exception (Q7 layer 3) is named and stated, not engineered — consistent with the lineage-birth-frozen principle.

**Consequence for ADR [0014](0014-dogfooding-mechanism.md)'s T-zero seed.** Project-b's per-repo pair ids merge into single product scopes at import; scope names are chosen at seed time.

**Adds the content-lives-in-one-place invariant** binding ADR [0008](0008-dedup-gate-and-reinforcement.md)'s aux stores (event log, distinct-pair registry, counters): those structures carry ids and timestamps, never content. `bury` can therefore drop content completely and durably from the one place it lives.

**Frontier.** Closes T04. The only remaining open design ticket is **T12** (curation / audience routing). Final-PRD-assembly graduates into a ticket once T12 closes.
