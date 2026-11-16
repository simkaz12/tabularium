# 0009 — Importance, decay, and eviction (the third and final core primitive)

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted — **amended by** [0012](0012-mcp-tool-surface-design.md) B4: feedback value range `[-1,+1]` → `[0,1]` (additive, non-negative; no centering transform — canonical formula in SPEC §Importance: `feedback_score = EWMA(α=0.3, init 0.0) ∈ [0,1]`, term stays `w_feedback·feedback_score`).

Resolves T08 — the last of the three core primitives (Recall = ADR [0007](0007-recall-pipeline-and-context-composition.md), Dedup = ADR [0008](0008-dedup-gate-and-reinforcement.md), and now importance/decay/eviction). This is where *every* trigger and threshold the prior branches deferred comes due. Built against the locked model (ADR [0005](0005-hypomnema-schema-and-storage-seam.md) seam, ADR [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md) Fact taxonomy, ADR [0007](0007-recall-pipeline-and-context-composition.md) recall + importance boost, ADR [0008](0008-dedup-gate-and-reinforcement.md) reinforcement event log) and the R05 project-b trace (§6–7). Amends ADR 0008's event schema (feedback events carry a `value`) and reaffirms ADR 0006's persisted-`hot` under a new rationale — noted inline + in Consequences.

## The root: importance is computed-lazy-for-ranking, swept-only-for-transitions

project-b's decay-freeze bug (R05 §6 — idle rows kept a stale score right up to the 90-day delete because the nightly rescore only touched rows used in the last 24h) is **architectural**, not a typo. The recency term is time-varying, so importance drifts every day even on untouched rows; materialising it and updating lazily *guarantees* staleness somewhere.

The fix splits the two jobs importance actually does:

- **As a ranking signal** (the ADR 0007 recall boost, `hot` ordering, context composition) importance is **computed lazily at read** from stored *time-invariant inputs* (`recall_count`, `feedback_score`, `last_referenced_at`) over the already-small `candidate_k` set. The recency term is always `now()`-fresh → **the freeze bug is structurally impossible**, not merely fixed by widening a filter.
- **As a lifecycle trigger** (`active→archived` decay, `hot` eviction) the transition is an *event that must fire on idle rows nobody reads* — a ranking number can't carry it — so a **scheduled sweep** evaluates the same lazy formula and performs the transition.

The realisation the project-b bug teaches: **decay-as-ranking and decay-as-transition are different problems.** Ranking must be lazy (to kill the freeze permanently); transitions must be swept. The T08 offline-job family therefore exists for **transitions + calibration + reconciliation**, *not* for keeping a ranking number warm. No frozen `importance_score` column is stored.

## The formula: additive, because the permanent terms *are* the archive floor

```
importance = w_recall·ln(1 + recall_count)
           + w_recency·exp(−Δdays / half_life)      # Δdays since last_referenced_at
           + w_feedback·feedback_score
```

Kept **additive** (over multiplicative-recency) deliberately: under additive the `recall`/`feedback` terms **never decay**, so a genuine keeper (high `recall_count`) holds a permanent floor and *stays active when quiet* (a much-referenced Fact is near-Decision-grade — don't archive it on a slow week and lean on fallback-revival to drag it back), while an incidental row (recall = 1, neutral feedback) sinks below the floor once recency decays. **One self-selecting formula serves both ranking and the decay transition — usage buys permanence.** Multiplicative-recency was rejected (it decays keepers on idle → a lossier session opener).

- `Δdays` is measured from **`last_referenced_at`**, bumped only by *genuine reinforcement* (ADR 0007/0008), never passive query; it defaults to `created_at`, so a fresh unused row reads recency ≈ 1 (removing project-b's separate `created_at` decay-guard).
- Seeds — `w_recall=0.5`, `w_recency=1.0`, `w_feedback=1.5`, `half_life=30d`, `feedback_score = clamped EWMA(α=0.3) ∈ [−1,1]` — all **eval-tunable seeds, not claims** (same no-magic-numbers ethos as ADR 0008's thresholds; T01 sweeps them).
- **Normalisation:** the raw unbounded composite is what the decay threshold reads; the ADR 0007 boost needs `importance_norm`, so importance is normalised **per-scope over the candidate set at read** (`imp / max(imp)` in the fused pool).

## Reinforcement → importance: a projection of the event log, feedback and references orthogonal

ADR 0008's append-only `ReinforcementEvent` log is the source of truth; under this ADR there is **no rescore-write** — the three stored inputs are **deterministic projections**:

| Input | Projection |
|---|---|
| `recall_count` | `count(events where source ∈ {dedup, revival})` |
| `last_referenced_at` | `max(at where source ∈ {dedup, revival})`, default `created_at` |
| `feedback_score` | `EWMA(α=0.3)` over `source=feedback` events, each carrying `value ∈ [−1,+1]` |

**Feedback (either sign) touches only `feedback_score`; reference events (dedup, revival) touch only `recall_count` + `last_referenced_at`.** They are orthogonal — which dodges a real trap: if a *negative* feedback also bumped `last_referenced_at`, it would jump the recency term by +`w_recency` while the feedback term fell by only ~`w_feedback·α`, so importance would rise on a thumbs-down. Orthogonality means negative feedback lowers importance purely via its term (a genuine push-toward-archive), and positive feedback raises it via its term without masquerading as a reference. This is a **small amendment to ADR 0008** (feedback events grow a `value` field; the emitting tool is T03's concern).

## Lifecycle transitions

**`active → archived` (decay).** The sweep archives any `active` Fact whose lazy composite `< archive_threshold`. **No hysteresis is needed** — importance can only *rise* on a reference (recency, the sole time-varying term, monotonically decreases; the other terms change only on a write), so a sunk row cannot bob back on its own; flapping is structurally impossible. Facts-only; `hot` is exempt (it leaves via eviction, not decayed out from under itself); Decisions never decay (ADR 0006). **Embers fade for free** through this same crossing (born high but `recall_count=0`, so once recency decays they sink and archive).

**`archived → active` (revival) — split by source strength:**
- **Dedup-revival → unconditional, in-place** (ADR 0008): re-saving identical knowledge is a deliberate re-assertion, one act per real event, not amplifiable — the archived Fact transitions immediately, same id/history.
- **Recall-fallback revival → threshold-gated** (ADR 0007): the fallback fires *because* active recall was thin, so unconditional revival would re-activate a thin scope's whole archive on incidental queries. A fallback hit reinforces, but only **transitions when the reinforced composite crosses `revival_threshold`** (surfacing ≠ usage). `revival_threshold` sits a notch above `archive_threshold` (a defensive dead-band).

## Ember escalation: born-as-nudge, one dedup-only recurrence threshold

- **Recurrence counter** `r = count(events where source=dedup)` — deliberately narrower than `recall_count` (a dedup hit re-*asserts* the corrective pattern; a revival is a mere retrieval, so only re-assertion builds codification pressure).
- **Rung 1 — born → warm nudge:** an ember surfaces in the session-start ember sub-budget *from birth* (ADR 0007 Module B) — that *is* "born hot"; recurrence is not required to nudge. ADR 0007's "which embers qualify" resolves to *embers still above the fade floor*, importance-ranked into the sub-budget (the cap self-limits). Fades via the decay sweep.
- **Rung 2 — `r ≥ escalation_threshold` → codification suggestion:** Tabularium *surfaces a suggestion* to codify the ember as a **Decision** via promotion-by-creation (`promoted_from`, ADR 0006). **Suggested, never automatic** (ADR 0001 invariant) — the agent creates the Decision; embers stay Facts and serve as provenance. Seed `r ≥ 3`, eval-tunable.

## `hot`: pure-importance eviction with hysteresis — no LLM-judge

project-b ran an LLM-judge `voteEviction`. This ADR **drops it**:

- **Promotion (`active→hot`):** the sweep promotes the highest-importance `active` Facts above `hot_promote_threshold` into free slots. Cap = seed **5/scope**. Hot transitions happen in the sweep, not synchronously on write (hot need only be correct at session-start read).
- **Eviction (`hot→active`, reversible):** when the tier is full and a higher candidate qualifies, evict the **lowest-importance hot Fact** back to `active` (not deleted — still recallable + re-promotable). **No LLM-judge**, because (1) eviction is **reversible + low-stakes** (unlike the dedup false-merge that justified heavy machinery); (2) importance is *already* the purpose-built ranking composite — an LLM on top re-imports the magic-number-LLM ADR 0008 rejected, with *less* grounding (it sees content, not usage history); (3) **there is no calling agent in an offline sweep** to adjudicate, so the choice is number-vs-LLM and the number wins.
- **Hysteresis delivers ADR 0007's "hot-floor guarantee" for free:** promote at `hot_promote_threshold`, evict only when displaced by a strictly-higher candidate *or* importance falls below a lower `hot_demote_threshold` — the dead-band stops a momentary dip churning a persisted-hot Fact.
- **project-b's reachability bug dissolves:** no judge → no judge-model health gate to get wrong (project-b checked the *embed* model). Eviction is pure-Python deterministic, always available.
- `hot_rejection_count` is kept as **pure observability** (count of sweeps where an `active` Fact qualified but lost the slot contest; a seed-5 threshold surfaces an "undersized tier" advisory) — it drives no transition.

## The offline-job family

Three jobs, **all `withoutOverlapping`** (fixing project-b's missing guard) + per-scope locks where they mutate `hot`:

1. **Lifecycle sweep** (decay → hot, one combined job, seed ~15–30 min cadence). Decay is made scalable by a precomputed **`decay_due_at`** crossing-time — another projection, recomputed on each reinforcement — so the sweep is an **indexed range scan** (`state=active AND decay_due_at ≤ now()`), never a full-scope recompute. This keeps the lazy architecture honestly scalable and keeps decay near-zero-cost when nothing is due. **`hot` stays *persisted*** (reaffirming ADR 0006): cheap eviction weakens the original "verdict too expensive to re-derive" reason, but two new reasons hold — hysteresis is *path-dependent* (a read-time "top-5" can't express "don't evict on a dip") and `hot_rejection_count` needs a home.
2. **Calibration pass** (slow cadence, gated on ≥ `min_label_count` new labels). Reads the ADR 0008 self-labels + the R01/T01 golden set; T01 owns the sweep harness, T08 owns the **live application** — pick the operating point maximising recall s.t. precision ≥ `P_floor`, applied **direction-gated** (`auto_ceiling` loosens only as evidence accrues, tightens freely) over **versioned per-embedder config** (the write path only reads it; recalibrate from scratch on embedder swap).
3. **Offline dedup reconciliation** (seed weekly). Re-runs the ADR 0008 classifier over existing records to catch near-dups the write-time gate missed (embedder-outage exact-match fallback, pre-recalibration accumulation). **Monotonic — never merges or deletes content** (no-hard-delete holds): it consolidates the reinforcement signal + surfaces missed redundancy. Renamed *reconciliation*, not "compaction" (it doesn't shrink storage).

### The three-way adjudicator split (the through-line)

The gray band gets a **different adjudicator per context — use an LLM only where no cheaper/better one exists and there is no hot-path cost:**

| Seam | Adjudicator | Why |
|---|---|---|
| **Write-time dedup gray band** | the **present calling agent** | in the MCP loop, context-rich, free (ADR 0008) |
| **`hot` eviction** | the **calibrated number** | reversible/low-stakes; importance is purpose-built (this ADR) |
| **Offline reconciliation gray band** | an **LLM provider judge** (litellm-routed) | *no agent is present offline*, and it's not a mere ranking — so the LLM is the right call here |

The offline judge fires **only on the gray band** (auto bands stay auto), and its verdict maps to the **same monotonic, non-destructive actions** — `same`→reinforce (`source=compact`), `distinct`→mark the registry, `Decision supersession`→**suggest** (never auto-retire, ADR 0001). It never merges/deletes. Reachability is handled honestly by litellm's timeout + retry + fallback (ADR 0003): an unreachable provider defers those pairs to the next run — graceful, observable, off the hot path (the honest version of project-b's botched gate). Every call carries latency/token/verdict **telemetry** and its prompt lives in the **prompt registry** — TB's third genuine litellm AI surface (embeddings + offline judge), earning the "real router" claim. This is *not* a reversal of ADR 0008's no-internal-LLM-judge: that rejection was scoped to the write path *because the agent is present there*; offline is the distinct context ADR 0008 deferred here.

## Correctness eval (feeds T01)

Architecture C hands testability for free through **two seams the code must expose**:

1. **A `Clock` seam** — injectable `now()` (not `datetime.now()` scattered about), so the eval **time-travels** and asserts trajectories. project-b's freeze survived partly *because* decay was untestable; the clock-as-seam makes decay a first-class eval target.
2. **Event-log replay** — importance is a pure projection + lazy formula over the append-only log, so the full trajectory is a pure function of `(event_log, clock)`; the eval replays a log against a simulated clock and asserts states/scores. **Correctness is replayable by construction.**

Metrics handed to T01 (which owns the harness + corpus): the **anti-freeze property as the headline golden regression** (idle importance strictly decreasing between references — the R05 §6 bug inverted; plus archive-at-`decay_due_at` and revival per the rules); the ADR 0007 RRF-only-vs-boost ablation + β-sweep; eviction hit-rate + churn/flap + re-promotion latency; calibration precision/recall with `precision ≥ P_floor` the hard gate; reconciliation injected-dup recovery rate.

## Considered options

- **Materialised `importance_score` + batch-rescore-all** (project-b de-bugged) — rejected: the freeze is fixed by discipline, not construction; O(N) nightly write; stale between runs.
- **Pure lazy-at-read, no job at all** — rejected: leaves lifecycle transitions with no trigger (idle rows never archive).
- **Multiplicative-recency composite** — rejected: decays genuine keepers on idle, forcing reliance on fallback-revival for the session opener.
- **Feedback bumps `last_referenced_at`** — rejected: a thumbs-down would perversely raise importance via the recency jump.
- **Hysteresis / dead-band on decay** — rejected as unnecessary: importance can't rise without a reference, so decay can't flap.
- **LLM-judge for `hot` eviction** (project-b) — rejected: reversible low-stakes, the number is purpose-built, no agent present; re-imports the magic-number-LLM ADR 0008 rejected.
- **Queue the offline gray band for a future session's agent** — rejected (the pushback that reshaped this ADR): that agent didn't author the saves and lacks context; offline is exactly where an LLM provider *is* the right adjudicator.
- **Destructive compaction (merge/delete dups)** — rejected: violates no-hard-delete + no-content-merge; reconciliation is monotonic.
- **`decay_due_at` as a full-scan alternative** — the scan was rejected: the precomputed crossing-time makes decay an indexed lookup.

## Consequences

- **Amends ADR 0008:** feedback `ReinforcementEvent`s carry `value ∈ [−1,+1]`; `source=compact` events are now emitted by the offline reconciliation job; the offline gray-band adjudicator is an LLM provider-judge (the write-path no-LLM rule is unchanged).
- **Reaffirms ADR 0006:** `hot` stays a *persisted* state, now justified by hysteresis-path-dependence + `hot_rejection_count`, not by eviction cost.
- **Record/field inventory:** stores the time-invariant inputs (`recall_count`, `feedback_score`, `last_referenced_at`) + the derived `decay_due_at`; **no** stored `importance_score` composite. Adds a `Clock` seam and (from ADR 0008) the prompt-registry + telemetry sink used by the offline judge.
- **Feeds T01:** the Clock seam + replay + the full metric set (anti-freeze golden regression as centerpiece), the calibration harness ownership boundary, the injected-dup reconciliation test.
- **Feeds T09:** the ember→Decision codification suggestion and the Decision-supersession suggestion from offline reconciliation both create supersession lineage the decision-record view surfaces.
- **Closes the three core primitives** (Recall / Dedup / Importance-decay-eviction). No new glossary terms — the domain nouns (`hot`, `ember`, importance, decay, eviction) landed in ADR 0006 / CONTEXT.md; T08 is mechanism design.
```
