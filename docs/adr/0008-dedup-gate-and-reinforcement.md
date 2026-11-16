# 0008 — Dedup gate + reinforcement (the write-time near-duplicate gate)

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted

Resolves T07 — the §4 "genuinely novel bit" most memory libraries lack. Designed concretely against the locked model (ADR [0005](0005-hypomnema-schema-and-storage-seam.md) legs-only seam, ADR [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md) Fact taxonomy, ADR [0007](0007-recall-pipeline-and-context-composition.md) recall + reinforcement events), the R05 project-b trace, and the R06 dedup landscape. Amends ADR 0005's seam and ADR 0007's reinforcement contract (noted inline + in Consequences).

## The gate is a broad-scan classifier + dispatcher

On every `save`, before writing, the engine runs a **write-time dedup gate**: it probes for near-duplicates, classifies the best match by its full state, and **dispatches to a per-case process**. The reinforce / bounce / pass trichotomy is the *common-case* process (a valid active/hot Fact); the special cases each get a purpose-built handler. This is a **shallow classifier in front of several deep per-case processes** — routing is the small interface; each process owns its case's semantics (locality).

## The probe: full hybrid retrieval, cosine-anchored verdict, cheap guards

The probe uses the **full hybrid retrieval** — both legs, `vector_search` (raw cosine) **∪** `text_search` (BM25) — **not** vector-only. The two legs are the highest-leverage seam in the system (Recall hits them per query, dedup per write, both degraded modes fall back on them; ADR 0005 has dedup *reuse* recall's legs), so shallowing them to vector-only would trade away leverage at the most-used seam and violate the quality-first stance.

A **funnel** above the legs-only seam:

- **Stage 0 — candidate generation:** both legs, union, **RRF for candidate *selection* only** (rank-based) — never a blended score that drives the verdict. BM25 is a **recall** aid (verbatim / numeric / rare-term / degraded-vector dups the embedder under-ranks), never a precision signal (R06).
- **Stage 1 — verdict:** anchored on **raw cosine** (interpretable, per-embedder-calibratable), **guarded** by near-free pure-Python **discriminative checks** (numeral / entity / version / negation-cue diff) that veto an auto-reinforce even at ~0.99 cosine. This is the defense against the false-merge cosine is *provably blind to* (two facts identical but for a critical token — `port 5432` vs `5433`, "prefer X" vs "prefer **NOT** X"; NevIR arXiv 2305.07614). The **importance boost is absent** — poison for identity (it biases by the neighbour's value, not identity).

Native LanceDB backs this cheaply: scoped metadata **prefilter** (`prefilter=True`), a native **`distance_range`** cutoff, **exact** distances on a flat scan below ~50K rows/scope, and a write **immediately visible** to the next same-connection search (R06).

## Three outcomes; guards downgrade; the gray band bounces to the agent

- `cosine < gray_floor` → **pass** (write novel).
- `gray_floor ≤ cosine < auto_ceiling` → **bounce**.
- `cosine ≥ auto_ceiling`, no guard trip, Fact → **auto-reinforce**.
- A **guard trip** *downgrades* an auto-reinforce to a **bounce** (the false-merge trap is the highest-value place to ask) — a downgrade rule, not a fourth outcome.

The gray band **bounces to the calling agent** — a structured tool response (candidate id/content/similarity/which-guard-flagged); the agent resolves `confirm_distinct` / update-or-supersede / drop. **There is no internal LLM-judge.** Tabularium's consumer *is* a reasoning agent already in the MCP loop, so it is the cheapest (zero LLM on the hot write path), most context-rich (it just authored the save), MCP-native adjudicator. An ignored bounce is a **safe failure** — nothing is written until the agent acts.

## The verdict matrix: scan broader than Recall; per-case processes

The gate scans **all lifecycle states and both validity values** (never go blind). It scans *broader than Recall* — Recall (ADR 0007) excludes `invalid` + `archived` to surface *currently-useful* knowledge; dedup asks "*have we seen this before, in any state?*" **Same legs, different `filters`.** The best match dispatches:

| Matched candidate | Process | Action + side-effect |
|---|---|---|
| valid Fact (active/hot) | **common** | reinforce / bounce / pass |
| valid **ember** | **recurrence** | reinforce + bump recurrence counter (→ T08 escalation ladder) |
| valid **archived** Fact | **revival** | reinforce as an **in-place** transition `archived→active`, same id/history preserved |
| **Decision** | **supersession** | **bounce** (eager, see below); never auto-reinforce |
| guard trip | **false-merge** | bounce + surface the differing token |
| **invalid** (any type) | **reconciliation** | **non-blocking** (see below) |

**Invalid → non-blocking reconciliation.** An old `invalid` memory must never *block* a now-`valid` same memory — otherwise a stale invalid ghost gatekeeps fresh true knowledge and the gate overreaches as a truth-*oracle* rather than a dup-*detector*. So the now-valid save **always writes**, **append-only**: a **new** valid record carrying an inline `reconciles: <invalid_id>` link; the old invalid record is **never touched** (stays `invalid` = accurate "false during [period]" history). A **principled asymmetry** with revival: `archived→revival` is an *in-place transition of the same record* (a relevance exit; truth unchanged; "revivable with history" per ADR 0006 requires the same id), whereas `invalid→reconciliation` is a *new record* (the truth changed over time; the gate never auto-flips a truth bit).

**Decisions bounce more eagerly than Facts — a lower gray bar — because the cost structure inverts.** A Fact false-pass = cheap, recoverable bloat → Facts are **precision-favored on auto-reinforce** (high `auto_ceiling`; don't merge distinct things). A Decision false-pass = *two active decisions on one topic* → corrupts "what's in force?" (the T09 query) → Decisions are **recall-favored on the bounce** (low `gray_floor`; don't let a refinement slip past as a parallel record). A decision is an **evolving entity** — it starts broad and is superseded through design, so near-dup decision saves are common; the right representation is a **supersession lineage** (ADR 0001), not a mutated record (erases evolution) nor N parallel near-dups (bloat). T07 *catches* the refinement (eager bounce → supersession); T09 surfaces the lineage.

## Reinforcement: an append-only event log

Reinforcement is recorded as an **append-only `ReinforcementEvent` log** (`{target_id, at, source ∈ dedup|feedback|revival, matched_from_id?}`) = the **source of truth**; `recall_count` + `last_referenced_at` are a **materialized projection** on the record for O(1) reads. A dedup-reinforce touches **counters only — no content merge** (a ≥`auto_ceiling` match is redundant by definition; merging would risk the very false-merge the guards defend). `importance_score` / `feedback_score` are **not** gate-touched: **T07 emits the signal, T08's rescore values it.** **Ember recurrence falls out for free** — `count(events where source=dedup)` on that ember; T08 pins the escalation threshold. The append-only log is the observability project-b utterly lacks (R05: zero gate/scoring telemetry) and lets T01 replay every reinforcement.

## Auxiliary persistence lives behind the same swappable seam

The gate's two auxiliary structures — the reinforcement event log and the **distinct-pair registry** — are engine-owned and **append-only**, and their *persistence* lives **behind the same swappable storage seam** as adapter ops (`append_event`/`query_events` + `mark_distinct`/`is_distinct`). One backend owns *all* persistence (the modular choice); the engine stays backend-agnostic. This honours ADR 0005's mechanism/semantic line: dedup **semantics** (classifier, verdict, when-to-mark) stay above the seam; only the **persistence primitives** go below, like the vector legs.

**Distinct-pair registry:** unordered pair `{lower_id, higher_id}` **UNIQUE** + `source` (`save`; `compact` reserved) + `judged_at`; **append-only, first-judgment-wins, idempotent**. Under no-hard-delete it is **never pruned — monotonic**, same philosophy as the tombstone archive (a buried entry's pairs harmlessly remain; no FK-cascade). The classifier consults it to not re-bounce a pair already judged distinct.

## Thresholds: precision-favored, per-embedder, seed→calibrate→curate

The error asymmetry is the fixed point: **false-merge ≫ false-pass** (silent irreversible data loss vs. recoverable bloat), so the gate is **precision-favored** and `auto_ceiling` is precision-critical. Thresholds are **not** hardcoded magic numbers (R06: no published per-model threshold; cosine ranges are model-specific anisotropy artifacts) — they are **per-embedder config** set by a **seed → operate → calibrate → curate** loop:

1. **Seed** — a conservative per-embedder-family base (high `auto_ceiling`), flagged seed-not-claim.
2. **Operate** — the gate **self-labels**: every bounce resolution *is* a label (`confirm_distinct`=distinct, drop/reinforce=same), landing in the event log + registry. Dogfooding *is* the calibration corpus (a differentiator — no competitor recalibrates).
3. **Calibrate** — an **offline** periodic eval pass (never the hot path; the write path only *reads* the config) sweeps the labeled pairs and picks the operating point maximizing recall subject to precision ≥ `P_floor`.
4. **Curate** — replace the seed; re-run as the corpus grows.

Safeguards: a **hand-built golden set** (R01) is the unbiased backbone (self-labels are gray-band-biased); the two thresholds carry different risk (`auto_ceiling` precision-critical vs. `gray_floor` bounce-rate tradeoff); min-label-count before curating (Platt early, isotonic at ~1000+ pairs); per-embedder isolation + recalibrate on embedder swap (ADR 0005 `embedding_model` provenance). **T07 owns** the seed, threshold params, precision-favored objective + `P_floor` knob, corpus shape (incl. critical-token cases), and the direction-gated-application rule; **T01 owns** the calibration harness.

### Cold-start protocol (fail-loud-never-silent)

The blind window (seed threshold, no labels, irreversible auto-reinforce) gets five layers, on the stance *assume it's bad at the start, then servo toward the operating point*: (1) the **guards are threshold-independent** — worst false-merges caught day 1; (2) **pre-deployment golden-set calibration** — deploy with a real per-embedder threshold; (3) **paranoid mode** — until a min label count `auto_ceiling` sits maxed, so the gate **bounces in doubt** (defers to the agent) and nothing is silently dropped; self-limiting (each bounce generates a label); (4) **instrumented** — alert if the auto-reinforce rate exceeds expectation, so a bad seed surfaces loudly; (5) **direction-gated** — `auto_ceiling` loosens only as evidence accrues, tightens freely. The only cost of caution is extra bounces (the safe, recoverable direction).

## Considered options

- **Vector-only probe** — rejected: shallows the most-used seam; misses verbatim/numeric/degraded dups; not a paper plane.
- **RRF-blended score driving the verdict** — rejected: an RRF artifact isn't interpretable or per-embedder-calibratable; thresholds lose meaning.
- **Lexical-overlap as the precision guard** — rejected: wrong instrument — BM25 is a *recall* aid (R06); precision comes from the token/entity/negation guards + cosine.
- **Internal litellm LLM-judge for the gray band** — rejected entirely (not deferred): re-imports the magic-number-LLM the calibrated gray band exists to replace; costs an LLM per gray write; has less context than the caller. The agent in the MCP loop is the adjudicator.
- **Invalid → bounce/block** — rejected: the failure path (a stale invalid ghost blocking now-valid knowledge); the gate overreaching as a truth-oracle.
- **Invalid invisible to the gate** — rejected: loses the reconciliation signal; the fix is *see it, dispatch a non-blocking process*, not hide it.
- **Auto-revalidate the old invalid record** — rejected: the gate guessing truth, the same overreach in the opposite direction.
- **In-place counter bump (project-b)** — rejected: an in-place update with zero history — exactly project-b's telemetry gap.
- **Hardcoded thresholds (project-b / Mem0 / Graphiti)** — rejected: no per-model calibration; the unmet gap Tabularium owns.
- **Designing batch/compaction in T07** — rejected: deferred to T08's offline-job family (same classifier re-run offline; T07 leaves `source=compact` reserved and the classifier reusable).

## Consequences

- **Amends ADR 0005** (seam): `reinforce(id, at)` becomes **"append a reinforcement event"** (counters are materialized views); the adapter grows **aux-persistence ops** (`append_event`/`query_events`, `mark_distinct`/`is_distinct`); the dedup probe's `filters` scan **broader** than Recall's (all lifecycle + both validity).
- **Amends ADR 0007**: reinforcement is an append-only event log (source of truth), not an in-place counter.
- **Feeds T08**: every threshold this ADR defers — `auto_ceiling`/`gray_floor` calibration targets, the revival cutoff, the ember-recurrence escalation rung, what reinforcement *does* to importance/decay — plus the offline-job family (calibration pass, compaction, decay, hot-eviction) and the compaction-time registry read-integration.
- **Feeds T01**: the calibration harness + labeled-corpus shape (dup/near-dup/distinct incl. critical-token cases); the reinforcement event log for replay; the model-slip corpus already defined (ADR 0006).
- **Feeds T09**: the supersession lineage the eager decision-bounce creates is what the decision-record view surfaces as an evolving entity.
- **Not precluded:** batch compaction (registry `source=compact` reserved, classifier reusable).
- No new glossary terms: the domain nouns (dedup gate, reinforcement, distinct pair) are mechanism-level; the record states landed in ADR 0006.
