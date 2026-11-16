# 0016 — Curation & audience routing: the Curator, purpose facets, and roles (T12)

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted — **narrowly amended 2026-08-02** (see Amendment below): on the two push tiers, `audience_hint` is now the routing key, not merely additive/boost. Pull (Recall) is untouched — B1/B2's *"audience machinery never touches Recall"* stands unedited. **Further amended 2026-08-03**: the 2026-08-02 amendment's "absence means universal reach" default is itself superseded — an unhinted record now reaches no role on push. Pull remains untouched by both amendments.

Closes T12 — all four ticket decide-items, resolved as eight grill decisions: (Q1) the three-concern split, (Q2) routing knowledge placement, (Q3) purpose vocabulary, (Q4) capture mechanics, (Q5) Curator module shape, (Q6) role vocabulary, (Q7) extraction quality, (Q8) surface wiring and cross-ADR amendments. Resolves the third orthogonal axis (type × lifecycle × **audience**) that surfaced grilling T05, and reduces project-b's conflated `audiences` label to three clean, separable concerns.

## Context: the project-b story

Project-b's `audiences` column conflates three unrelated concerns in one label: who may see (visibility / ACL), who gets the entry pushed proactively (delivery), and how the content is projected per consumer (rendering). The result is a triple mess:

**The leaky ACL.** Project-b applies the audience filter in exactly one place — the grounding service's context composer (via the entry model's audience scope). The recall tool ignores audiences entirely, returning `['doer']`-labeled entries to any caller. The "ACL" is unenforceable by construction: project-b accidentally built open-pull / curated-push and then labelled it an access control. With no principal (ADR [0015](0015-owner-scope-seam.md) Q1 removed the owner concept), any ACL is theater — nothing to authenticate against.

**The never-opened apply gate.** During project-b's own audience rollout, a dry-run of its bulk-labeling pass proposed audience labels for about 1,200 decisions in the live store. 74% of the proposals included the `doer` label — a near-constant column carrying near-zero information. The apply gate was never opened. Production project-b held 2 labeled rows out of approximately 1,200 total entries. The ritual-stamp effect confirmed pre-launch: mandatory audience metadata on saves decays into noise that nobody fixes.

**The `work` category error.** Project-b's audience enum mixes consumer roles (`agent-x`, `main`, `doer`, `explorer`) with a topic (`work`) that its config fans out to three consumers. The original intent, per the user who built project-b, was context-budget routing — "doer needs the thin-controllers convention; main doesn't" — never secrecy. Naming the intent correctly is the fix.

These three pathologies — unenforceable visibility, ritual-stamp delivery metadata, and a vocabulary mixing functions with topics — each demand a different resolution. T12 separates them.

## B1 — Pull / push split: recall is open, composition is curated

**Pull (Recall) is fully open.** Any consumer that calls `recall` or `decisions_query` gets an answer; audience filtering never touches the retrieval path. This is the principled design of what project-b accidentally built.

**Push (context composition) is curated.** The session-start payload is the scarce, attention-weighted resource. Per-consumer delivery policy — the Curator — is T12's whole substance.

**Rendering reduces to selection.** ADR [0012](0012-mcp-tool-surface-design.md) B3 and ADR [0013](0013-decision-record-view.md) spent the rendering decisions (structured tool views, byte-stable resource, frame-not-parser). The Curator selects and orders — it never rewrites content. Never-truncate and frame-not-parser forbid rewriting.

**Why visibility is out.** T04 removed the principal — an ACL with nobody to authenticate is unenforceable theater. Structural isolation already exists as T04 scopes. Lying about one's role gains nothing when nothing is hidden.

## B2 — Routing knowledge placement: facets + policy as the primary router, audience hint as additive supplement

**Facets + policy** is the primary router. No load-bearing consumer name appears on any hypomnema. The entry carries what it IS (intrinsic, stable at write time); the Curator's delivery profile carries who WANTS it (changes with the agent roster, lives in one place). Adding a new agent role = one policy row, zero entries relabeled. Project-b's bulk-relabel sweeps become unwritable by design.

**`audience_hint: list[Role] | None = None`** survives as a supplementary signal. The saving model forms its who-needs-this opinion with full working context — context no later policy evaluation sees. The hint records the write-time exception that policy can't express. Semantics are **additive-only**: the hint can add an entry to a hinted role's push candidate pool and boost its rank; it can never exclude. It never touches pull — Recall reads no hints. A stale hint degrades gracefully: it simply stops matching; nothing ever goes dark.

The additive-only shape mirrors feedback's `[0, 1]` range (ADR [0012](0012-mcp-tool-surface-design.md) B4): no backdoor ACL, as feedback has no backdoor eviction.

## B3 — Purpose vocabulary: five registered values, each backed by a default policy row

`purpose` = a single-valued optional facet on any hypomnema (both `Fact` and `Decision` arms) naming its **intrinsic utterance type** — what the writer knowably said, never a delivery destination and never an axis that already exists.

**Five starting values**, each shipped only because a default delivery-policy row consumes it ("no claim the code can't back," facet edition):

| Value | Meaning | Default routing |
|---|---|---|
| `constraint` | Hard guardrail — violating = incident | Every acting role, always |
| `convention` | House style — violating = bad work | Implementer / reviewer roles |
| `preference` | Owner taste | User-facing roles only |
| `pitfall` | Known trap — where you'll trip next time | Doing roles |
| `reference` | Pointer to an external resource | Scout / doing roles |

**`constraint` vs `convention` is a real split**, not a style preference. They route differently (everyone vs producing roles) and mis-filing costs differently (incident vs style drift). If they routed identically they would merge; they don't.

**Killed from the candidate list:**
- `grounding` — names where content goes (a delivery outcome the Curator decides); storing it as a purpose value re-smuggles routing metadata into the row, the same mistake as project-b's `work` label.
- `rationale` — already `kind=decision` per ADR [0001](0001-decision-immutable-content-mutable-lifecycle.md); a `purpose=rationale` Fact is a mis-filed Decision.

**Renamed:** `warning` → `pitfall`. The glossary reserves warning-adjacent language away from valence-neutral embers (a correction-ember is a transient behavior nudge; a pitfall is durable world-knowledge about where one trips). The ember valence axis (ADR [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md), locked) is untouched and non-overlapping once the rename lands.

**`finding` — first documented candidate registrant, not registered day one.** Empirical discovery ("what was true when you looked," as distinct from a pitfall's "future trip-wire") appeared in 9 of 60 sampled live project-b saves. Findings are query-shaped — pull already serves them well; no default policy row pushes them at session start. Later adoption = one `tabularium facet add finding` command: the registered-vocabulary model earning its keep.

## B4 — Capture mechanics: optional, single-valued, kind-agnostic, registered vocabulary

**Optional (`purpose: str | None = None`).** Data-backed: a sample of the 60 most recent live project-b entries found that 75% fit none of the five values — plain working state, architecture decisions, session digests. Forcing a value on these saves manufactures ritual metadata. The mandatory-on-decisions variant was project-b's failure; it produced the never-opened apply gate.

**Single-valued.** The sample surfaced 6 mild edge collisions (mostly `pitfall` ↔ `constraint` shading). Adjacent values route similarly, so an edge mis-file is cheap; a multi-value folksonomy costs more in API surface and policy logic than it recovers.

**Kind-agnostic (both arms).** Purpose is an utterance-type facet, not a Fact/Decision split. "Thin controllers" is a Decision with `purpose=convention`; embers are covered free as a Fact lifecycle state.

**Registered vocabulary, same contract as scopes.** `tabularium facet add <value>` — a CLI human act; agents consume, never mint; an unregistered value on `save` or `ember` → `ToolError`. A typo dies loudly at the gate; no ghost facet accumulates in the store. A `doctor` check flags any registered value that no policy row consumes, so dead vocabulary is surfaced, not silently preserved.

**Empirical confirmation of the ritual-stamp pattern.** Project-b's bulk-label dry-run (1,199 decision candidates; 74% include `doer`; apply gate never opened; 2 labeled rows out of ~1,200 in production) is recorded here as the canonical prior-art cautionary data point. The design is deliberately architected against repeating it.

## B5 — Curator module shape: two modules, seam between selection and packing, policy as data

**Two modules.** The user-directed split: curation is the precise-tinkering surface — the tuning knob concentration point — and must not be merged into the packing machinery that ADR [0007](0007-recall-pipeline-and-context-composition.md) already locked.

**Curator** (new module): `curate(scope, role) → CuratedSelection` — query-less, token-blind, push-side only. Consumes delivery profiles + purpose facets + audience hints. Emits ordered tiers `[Tier{name, budget_share, ordered_entries}]`. Token-blind by design: the Curator decides *what deserves attention and in what order*; composition decides *what fits*. Every tunable for push delivery concentrates here.

**Context composition** (ADR [0007](0007-recall-pipeline-and-context-composition.md) Module B, mechanics unchanged): `compose_context(scope, budget, role="default") = pack(curate(scope, role), budget)`. Model-aware token counting, greedy whole-entry fill, waterfall spill, never-truncate — all unchanged. The function signature grows one optional `role` arg; the packing machinery is not otherwise touched.

**Delivery profiles** are data, not code — declarative objects keyed by Role: facet rules per tier, tier order, per-tier sort keys, budget shares, hint handling. Lives in store-level config, human-edited, versioned with the store. Adding a consumer role = writing a profile, zero code. The `default` profile reproduces ADR [0007](0007-recall-pipeline-and-context-composition.md)/ADR [0012](0012-mcp-tool-surface-design.md) behavior verbatim — the three tiers (hot → ember → decision) in the old order.

**Two real Curator consumers from day one** — context composition in production and the eval harness driving the Curator directly, swapping profiles like it swaps fusion strategies (ADR [0011](0011-eval-harness-architecture.md)). The seam earns its keep immediately; it is not a hypothetical future interface.

**`role` is cooperative self-identification, never authentication.** No principal exists (ADR [0015](0015-owner-scope-seam.md) Q1); nothing is hidden (Q1 above); a false claim gains nothing.

## B6 — Role vocabulary: working functions, third registry, `default` reserved

**Roles name working functions, never agent products.** `implementer`, `investigator`, `scout` — what the consumer does. Not product names such as `agent-x` — brand names for agent products that can be renamed or replaced. The agent→role binding lives in client config (the T04 repo→scope move; each repo's `.mcp.json` / `CLAUDE.md`). Agent identity already rides `source`; role names the function, not the model.

**One vocabulary** shared by delivery profiles and `audience_hint` values. The two always agree: a hint naming a role the profile doesn't recognize is already a `ToolError`.

**Third registry, same contract.** `tabularium role add <name>` — CLI human act; agents consume, never mint; unregistered role on `context(role=…)` or in any `audience_hint` value → `ToolError`. Hinting a not-yet-real role requires registering it first (one command). Three registries, one pattern:

| Registry | Reserved member | Pre-registered members |
|---|---|---|
| Scopes | `global` | `global` |
| Facets | — | five purpose values |
| Roles | `default` | `default` |

`default` is reserved + pre-registered: the profile used when `role` is omitted. The main session is the default consumer; it never claims anything.

**Ship discipline.** Day-one registry = `default` alone. `implementer`, `investigator`, and `scout` are documented in the spec as expected first registrants, registered the day sub-agent grounding becomes real. Context composition is not in the dogfood slice; nothing blocks.

## B7 — Extraction quality: capture-side only, zero engine inference

T12's "improve extraction for the right consumer" decomposes, in a deliberate-save world (every `save` is a speech act — ADR [0012](0012-mcp-tool-surface-design.md), ADR [0014](0014-dogfooding-mechanism.md)), into three mechanisms:

**(1) The tool interface is the extraction prompt.** The `save` and `ember` tool descriptions carry the five purpose values with one-line anchors — the feedback-anchors idiom of ADR [0012](0012-mcp-tool-surface-design.md) B4. Teaching the frontier writer at the tool description is the highest-leverage intervention, and it is pure semantic-in.

**(2) Capture guidance lives client-side.** What to save, which scope, when to hint — each client's memory discipline (`CLAUDE.md` / agent definitions), versioned with the client, outside the engine. The engine supplies a vocabulary sharp enough to teach; it does not police usage.

**(3) Observability, not enforcement.** The `doctor` command grows a facet-coverage report: share of saves that carry a purpose value, value distribution, hint frequency. Capture drift becomes visible in dogfooding. The T01 eval harness evaluates the Curator deterministically (faceted golden corpus + profile → expected selection) and never grades the writer in the hermetic gate: writer quality is a dogfood audit, not a CI assertion.

**Future door, not built.** T08's offline reconciliation family (the one legitimate LLM-judge seat, per ADR [0009](0009-importance-decay-eviction.md)) could someday suggest purpose values for old unfaceted entries — documented as a possible extension, zero commitment.

**Rejected: write-time LLM facet classifier.** Violates the locked three-way adjudicator split (ADR [0009](0009-importance-decay-eviction.md) parked nuance — no write-time LLM-judge). A model call on the hot save path; machine-confident ritual metadata industrialized on the 75%-NONE sample.

## B8 — Surface wiring and cross-ADR amendments

**(1) `save` and `ember` gain two new optional args.**

```python
save(
    content: str,
    *,
    kind: Literal["fact", "decision"] = "fact",
    scope: str,
    source: str | None = None,
    supersedes: HypomnemaId | None = None,
    promoted_from: list[HypomnemaId] | None = None,
    resolve: BounceResolution | None = None,
    purpose: str | None = None,           # T12 addition
    audience_hint: list[Role] | None = None,  # T12 addition
) -> SaveResult

ember(
    content: str,
    *,
    scope: str,
    valence: Literal["correction", "guideline"],
    source: str | None = None,
    purpose: str | None = None,           # T12 addition
    audience_hint: list[Role] | None = None,  # T12 addition
) -> SaveResult
```

`ember` gains both fields because a correction-ember hinting the role that erred is the natural use case. Both new args are optional; existing callers require no changes. Amends ADR [0012](0012-mcp-tool-surface-design.md) B2/B2½.

**(2) `ContextPayload` generalizes to profile-defined tiers.**

```python
class PackedTier(BaseModel):
    name: str
    entries: list[RecalledHit]

class ContextPayload(BaseModel):
    tiers: list[PackedTier]     # replaces hot_facts / embers / in_force_decisions
    skipped: int
    budget_used: int

context(scope: str, budget: int, role: str | None = None) -> ContextPayload
```

Three hard-coded `ContextPayload` fields cannot hold profile-defined tiers. The `default` profile emits the old three tiers in the old order (`hot_facts` → `embers` → `in_force_decisions`) byte-identically, so existing consumers see no behavioral change. `context` grows one optional `role` arg; omitting it = the `default` profile. Amends ADR [0012](0012-mcp-tool-surface-design.md) B3.

**(3) Result models gain `purpose`.**

```python
class RecalledHit(BaseModel):
    ...
    purpose: str | None = None        # T12 addition

class DecisionView(BaseModel):
    ...
    purpose: str | None = None        # T12 addition
```

The `decision://{id}` **resource render does not show `purpose`** — routing metadata is not decision identity; surfacing it would change the rendered bytes for no consumer benefit and violates the byte-stability guarantee (ADR [0013](0013-decision-record-view.md)).

**(4) `purpose` is CLI-correctable only, never model-facing-mutable.** Mis-filed purpose values are fixed by a CLI maintenance command (exact name settled at build time; it edits one entry's routing metadata, not the registry) — a human act, like registry edits. No new MCP tool. Disposition semantics are untouched; content immutability is untouched (`purpose` is routing metadata, not content).

**(5) Non-interactions stated.**
- The dedup gate ignores `purpose` and `audience_hint` entirely — the probe is content-based. Same content, different facet = still a duplicate.
- `recall` and `decisions_query` expose no `purpose` filter. This is a cheap future parameter; not built because pull is already open and purpose is push metadata.
- The two new columns land on the single physical table through the storage seam like any other metadata. Amends ADR [0005](0005-hypomnema-schema-and-storage-seam.md) column list.

**Headline: T12 adds zero MCP tools.** 10 tools + 1 resource stands. Surface growth = CLI (`tabularium facet add`, `tabularium role add`, `tabularium doctor`) + two optional write args + one optional `context` arg.

## Considered options

**Visibility:**
- Per-entry audience ACL enforced on every read — with no principal (ADR [0015](0015-owner-scope-seam.md) Q1), the ACL is unenforceable; `recall` would refuse to surface an entry the store owner themselves queried. Rejected.

**Routing knowledge:**
- Audience-on-entry as sole router — the writer at save time answers an unanswerable question: who will want this, forever, regardless of how the agent fleet changes. Project-b's relabeling sweeps that never run are the evidence. Rejected.
- Boost-only-within-policy-eligible hint — the hint acts only when policy already agrees, making it decorative exactly where the writer's opinion carries information (the exception case). Rejected.
- Pure facets + policy, no hint — discards genuine write-time signal the saving model holds with working context no later evaluation sees. Rejected.

**Purpose vocabulary:**
- `grounding` as a purpose value — names a delivery outcome, not an utterance type; re-smuggles routing into the row. Rejected.
- `rationale` as a purpose value — already `kind=decision`; storing it as a Fact purpose is a mis-filed Decision. Rejected.
- `warning` as a purpose value — conflicts with the glossary's reservation of warning-adjacent language away from valence-neutral embers. Renamed to `pitfall`.
- Required on decisions — project-b's ritual; produces the never-opened apply gate. Rejected.
- Closed enum (hard-coded in code) — a new utterance type requires a code release. Rejected.
- Free tags — open folksonomy accumulates ghost facets from typos, the same silent failure species as open scope vocabulary. Rejected.

**Capture mechanics:**
- Multi-valued purpose — 6 mild edge collisions in the sample; adjacent values route similarly; the folksonomy cost exceeds the recovery. Rejected.

**Curator shape:**
- One code adapter per consumer — one real push consumer at T-zero means a hypothetical seam; multiplying adapters re-encodes the consumer roster as N classes (project-b's config-map problem in OO dress). Rejected.
- Policy threaded through the packer as parameters — tunable policy smeared through frozen packing machinery. Rejected.

**Role vocabulary:**
- Roles as agent product names (such as `agent-x`) — the registry re-encodes a volatile roster that Q2 deliberately evicted from the archive; fleet renames churn the registry and stale hints. Rejected.
- Project-b's mixed enum (functions + brand names + topics in one vocabulary) — mixes three orthogonal namespaces. Rejected.

**Extraction quality:**
- Write-time LLM facet classifier — violates the three-way adjudicator split (ADR [0009](0009-importance-decay-eviction.md)); LLM call on the hot save path; machine-confident ritual metadata industrialized from the 75%-NONE sample. Rejected.

## Consequences

**Amends ADR [0012](0012-mcp-tool-surface-design.md):**
- B2/B2½: `save` and `ember` gain `purpose: str | None = None` and `audience_hint: list[Role] | None = None`.
- B3: `ContextPayload` generalizes from three named fields to `tiers: list[PackedTier]` + `skipped` + `budget_used`; `context` gains `role: str | None = None`; the `default` profile reproduces the old three-tier behavior byte-identically.
- Result models: `RecalledHit` and `DecisionView` gain `purpose: str | None`.

**Amends ADR [0007](0007-recall-pipeline-and-context-composition.md):**
- Module B is restructured — the Curator is extracted as the selection module, sitting between the data and the packer. `compose_context` becomes `pack(curate(scope, role), budget)`.
- Tier structure is profile-driven; the `default` profile reproduces the old waterfall (hot → ember → decision) and per-type packing mechanics unchanged.
- The packing machinery itself (model-aware token counting, greedy whole-entry fill, waterfall spill, never-truncate) is untouched.

**Amends ADR [0005](0005-hypomnema-schema-and-storage-seam.md):**
- The hypomnema record gains two columns: `purpose: str | None` and `audience_hint: list[str] | None` (stored as a nullable JSON array). Both land on the single physical table through the seam.

**Reinforces ADR [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md) and ADR [0009](0009-importance-decay-eviction.md).**
- The three-way adjudicator split (write-time → agent; eviction → the number; offline → LLM) is honored: no write-time LLM facet classifier is ever built.

**Reinforces ADR [0013](0013-decision-record-view.md).**
- Resource render byte-stability preserved: `purpose` is intentionally excluded from the `decision://{id}` rendered frame. Routing metadata is not decision identity.

**Closes T12** — the last open design ticket. All thirteen design tickets are now resolved. 

---

## Amendment (2026-08-02) — `audience_hint` becomes the push-tier routing key

**Owner decision, 2026-08-02**. B2 above specced `audience_hint` as **additive-only**: it "can add an entry to a hinted role's push candidate pool and boost its rank; it can never exclude." That was written when the Curator (`curate(scope, role)`) was the imagined consumer — a ranked pool where a hint nudges order. The Curator never shipped (M2 slice C deferred it; see SPEC §7's slice-C amendment); what shipped instead is `select_default`, a fixed two-tier selection with no ranked-pool concept for a hint to nudge into. Sub-agent grounding — four profiles keyed on function, `reviewer` first — needed a real mechanism to keep a `builder`-only correction out of a `reviewer`'s payload, and "boost the rank of something already in a role-agnostic pool" cannot do that; only exclusion can.

**What changed.** On the two push tiers (`hot`/`ember`, `select_default`/`context()`), `audience_hint` is now the **routing key**: a record whose hint is empty routes to every role (universal reach, unchanged default); a record with a hint routes *only* to a role named in it. This is binary inclusion/exclusion, not a rank nudge — the "never exclude" half of B2's additive-only rule is superseded for push. `set_hot` gained a `roles` parameter that persists this hint on promotion (**overwrites, never unions** — owner-ratified: curation-time judgment beats a writer's capture-time guess, and a union would mean an incidental capture-time hint could only ever widen, never be corrected downward).

**What survived unedited.** B1's pull/push split and B2's "it never touches pull — Recall reads no hints" stand exactly as written. `recall` and `decisions_query` apply no `audience_hint` filter today, and this amendment adds none — a hinted record is exactly as recallable by every caller as an unhinted one. The additive-only *language* is retired for push; the pull guarantee it was partly protecting is untouched by a separate, still-standing rule.

**Why this is the attention economy principle, not an exception to it.** B1's own framing: pull is open, push is the scarce, attention-weighted resource, and "nothing is hidden — a false claim gains nothing" because there is no principal to lie to. Routing on push does not create a principal or hide content: any caller can still `recall` a `reviewer`-hinted Fact by asking for it. What routing does is stop spending a `builder`'s always-on session-start budget on a correction addressed to `reviewer` — the exact "budget routing, never secrecy" intent ADR 0016's own Context section credits to project-b's original author (§"The `work` category error"). Narrowing an already-owner-curated set's *delivery* is attention economy; it only becomes secrecy if it also closed off retrieval, which it structurally cannot — pull has no hint-awareness to close.

**Why not the Curator instead.** Building `curate(scope, role)` now, just to keep the ranked-pool-plus-boost shape B2 specced, would be solving a problem sub-agent grounding doesn't have: `reviewer`'s payload needs binary routing over a fixed two-tier shape, not per-role tier composition, sort keys, or budget shares. `select_default`'s locked decision (SPEC §7) — the Curator's *configurability* remains cut, not merely deferred — is unaffected by this amendment; the routing predicate (`engine/context.py::_routes_to`) is one function shared by both tiers inside the existing fixed selection, not a step toward `curate()`.

**No schema change.** `audience_hint` was already a nullable column (ADR 0016 B8's Consequences, `docs/storage/lancedb_codec.py`); this amendment is a read-filter (engine-level, in `select_default`) plus one new write path (`transition`'s `audience_hint` tri-state, previously silently discarded by every caller — see the storage-seam-explicit-typing prerequisite). No new field on `Fact`/`Decision`, no Arrow column, no backfill.

Living docs updated in the same pass: `docs/SPEC.md` §3.1 (`transition` tri-state), §7 (routing rule + slice-C paragraph), §9.1 (`set_hot(roles=...)`, `context` note), §10.3 (registered profiles); ADR [0020](0020-hot-tier-owner-curated-not-earned.md) (role tag rides with the pin); the Curator bullet restated.

---

## Amendment (2026-08-03) — explicit enumeration only; an unhinted record reaches no role

**Owner decision, 2026-08-02** (curation block). The amendment immediately above made `audience_hint` the push-tier routing key but kept one half of B2's old shape: "a record whose hint is empty routes to every role (universal reach, unchanged default)." That default held up only while the corpus was small and unhinted-by-construction (ten hot records, zero embers). The reviewer profile's first live dispatch — four atoms delivered, six skipped, from the same undifferentiated pool every other role would also have received — surfaced why: an untagged record is ambiguous between *deliberately for everyone* and *nobody has curated this yet*, and treating that ambiguity as "everyone" silently floods every newly-registered role with every atom nobody has looked at, the identical ritual-stamp failure this ADR's own Context section documents from project-b's `doer`-heavy audience column.

**What changed.** On the two push tiers (`hot`/`ember`), `_routes_to` drops the empty-hint short-circuit entirely: a record reaches a role's payload **only** when that role is named in its `audience_hint`. A falsy hint (`None` or `()`) now reaches **no** role, not every role. "For everyone" is spelled by enumerating all four roles (`decider`, `builder`, `reviewer`, `navigator`) in the hint, never by omitting it. Rejected here, same as at B2/the prior amendment: an `all`/universal token — it recreates the retired lazy default under a new name, collides with `global` (already a *scope*, not a role), and would auto-include any role registered in the future, which is the bug this amendment closes, not a feature to reopen. `set_hot`'s and `ember`'s `roles` parameters are unchanged in shape (still overwrite, never union) — only what an *empty* list now means at read time changes, from "reaches everyone" to "reaches no one."

**What survived unedited.** Everything the 2026-08-02 amendment itself carried forward survives this amendment too, unedited a second time: B1's pull/push split; "`audience_hint` never touches pull — `recall` and `decisions_query` read no hints, so a hinted record is exactly as recallable as an unhinted one"; the routing mechanism's shape — binary inclusion/exclusion at the push tier, never a rank nudge; one predicate (`engine/context.py::_routes_to`) shared verbatim by `_select_hot` and `_select_embers`, so the rule cannot drift between tiers; `select_default`'s locked non-Curator shape (SPEC §7) — this is still a read-filter inside the existing fixed selection, not a step toward `curate(scope, role)`; and the "no schema change" fact — `audience_hint` was already nullable, this is a read-filter edit plus no new write path beyond what the prior amendment already added.

**Why this is still the attention economy principle, not a reintroduced exception.** The 2026-08-02 amendment's own framing holds without amendment: narrowing an already-owner-curated set's *delivery* is budget routing, not secrecy, because pull is structurally incapable of being narrowed by a push-side hint. Making the default *exclusive* rather than *inclusive* does not change that argument — it changes who has to spend a curation action to reach every role (the writer/curator, by naming all four, rather than nobody, by naming none), which is exactly the fix for the ritual-stamp failure: a `doer`-style catch-all label decayed into meaningless noise in project-b precisely because it cost nothing to apply by default. Costing a slot in every cap for genuine "everyone" content (curation block's locked decision) makes that catch-all a deliberate, tracked choice instead of a free default.

**No schema change; no data migration.** Same nullable `audience_hint` column as before. The curation-block sequencing note (`plan.md`) is the operational safeguard, not a schema one: the ten then-existing hot records with explicit hints *before* this semantics flip landed, so no record went invisible in the transition — the corpus was made compliant with the new rule before the rule took effect, rather than the rule being flipped and the corpus racing to catch up.

Living docs updated in the same pass: `docs/SPEC.md` §7.

---

## Amendment (2026-08-03) — `role` required on `context`, `roles` required on `set_hot(hot=True)`, `default` retired

**Owner decision, 2026-08-02** (curation block). The amendment immediately above closed WHAT a falsy `audience_hint` means (no role, not every role); it left one consequence open at the caller's end: `context`'s `role` parameter still defaulted an omission to `"default"`, and no real record was ever hinted `"default"` — so a role-less `context` call now returned an EMPTY payload with no error at all, the identical "nobody has curated this yet" ambiguity the prior amendment banned at the record level, reappearing one layer up.

**What changed.** `context(scope, budget, role)`'s `role` is now REQUIRED: a caller supplying `None` is bounced with a `ToolError` naming the registered roles, at both the engine boundary (`Tabularium.context`) and the MCP tool boundary (`tabularium.server`'s `context` tool — a thin pass-through per its "near-1:1 projection, ZERO memory semantics" contract; the same raise propagates through both, not two independent checks). `set_hot(id, hot, roles)`'s `roles` is now REQUIRED when `hot=True`: `roles=None` raises unconditionally, regardless of lifecycle state and regardless of whatever `audience_hint` the Fact already carries from write time — this DELIBERATELY REVERSES the 2026-08-02 amendment's "`roles=None` (default) leaves the stored hint untouched; every pre-existing caller is unaffected" clause, because letting a Fact enter an always-on push tier without ever forcing the caller to consider an audience is exactly the silent-empty-payload failure this amendment closes. `roles=[]` (explicit empty) is still legal — a deliberate "hot, reaching no role" choice, distinct from silently omitting the argument. `set_hot(hot=False)` is unchanged: `roles` stays optional there, since removing something from the tier needs no audience decision. `"default"` — the always-present fallback role that predated per-role grounding — is retired from `initialize()`'s seed list: a fresh store's role registry is now exactly the four function-named roles.

**What survived unedited.** Everything both prior amendments carried forward survives this one too: B1's pull/push split; `audience_hint` never touches pull; the routing predicate and its binary inclusion/exclusion shape; `select_default`'s locked non-Curator shape; no schema change (this amendment touches validation and seeding only, not storage).

**Honesty finding (this task's acceptance criteria):** neither `StorageAdapter` implementation (`MemoryAdapter`, `LanceDBAdapter`) exposes a role *deletion* path — `create_role`/`list_roles` only, no `delete_role`/`remove_role` on the `StorageAdapter` Protocol or either concrete adapter. "Retired" therefore means "no longer seeded by `initialize()` on a fresh store," NOT "guaranteed absent" — a store that already holds a `"default"` role row keeps that row forever; nothing in the engine breaks on its continued presence, since no code path resolves an omission to it any more.

Living docs updated in the same pass: `docs/SPEC.md` §7 (new paragraph), §9.1 (`context`/`set_hot` signature comments), §10.1 (registries table), §10.3 (new paragraph).

---

## Amendment (2026-08-03) — a brand-new ember cannot be born reaching nobody; the owner/agent authorship split; `Fact.provisional`

**Owner decision, 2026-08-03** (curation block). Every amendment above closed the RETAG surface (`set_hot`'s and `ember`'s `roles` params) and the READ surface (`context`'s `role`). One write path was still open: `ember()` kept `audience_hint: list[str] | None = None` with no requirement at CREATION time, and `_routes_to` (the earlier amendment) returns `False` for every registered role when the hint is falsy — so a brand-new ember could be minted today reaching no push tier at all, silently, with no error and nothing in the return value that says so. This is the third instance of the silent-dark class in this block (after the 04→05 ordering hazard and the role-less `context` call that was closed): explicit-enumeration routing turns every "optional audience" parameter into a way to write an invisible record, one at a time, until each is closed.

**What changed — the rule splits by `asserted_by`, not a single across-the-board requirement.** Embers are not exclusively owner-made: `ember()`'s `asserted_by: Literal["owner", "agent"] = "agent"` already defaults to `"agent"`, and SPEC §8.4's escalation ladder counts agent-asserted recurrences as the route toward owner promotion (a local slash command is the owner entry point, not the only one).

- **`asserted_by="owner"`**: `audience_hint` becomes **required and non-empty** — a FALSY value (omitted, or an explicit `[]`) raises a `ToolError` naming the registered roles, mirroring `set_hot(hot=True)`'s already-established "no caller has no audience" precedent. An ember is authored, not inferred: the author is right there and can say who it is for, so requiring a real audience costs them nothing. Rejected: defaulting a new owner-authored ember to all four roles — that is the universal token wearing a different hat, the exact lazy default this block's earlier amendments retired, relocated from `_routes_to` into the ember arm where it would be harder to see.
- **`asserted_by="agent"`** (the default): a FALSY `audience_hint` defaults to the **four canonical roles** — `decider`/`builder`/`reviewer`/`navigator`, a FIXED tuple (`engine._types._ALL_ROLES`), deliberately NOT a live `adapter.list_roles()` read (a store that still carries a stale `"default"` role row from before it was retired from seeding would otherwise tag a brand-new provisional ember with that historical cruft too, which is not "the four roles" by any reading) — AND marks the written Fact `provisional=True`. An agent surfaces a correction; it does not get to decide who that correction is for, so the widest reach is the honest default *while an owner verdict is outstanding* — not a claim that the owner actually meant "everyone."

**There is no "deliberately reach nobody" option at ember creation, on either path**: an earlier version of this rule treated only an omitted `audience_hint` as missing, letting an explicit `audience_hint=[]` slip through honored verbatim — which still produces an ember reaching no push tier at all, the identical observable failure this whole amendment exists to close, just gated behind one extra explicit argument. The task's own headline acceptance criterion is unqualified — "no ember can be born reaching nobody, by either authorship path" — so a falsy `audience_hint` (`None` or `[]`) is treated identically by both branches above. This is deliberately narrower than the SEPARATE `roles` retag parameter (unaffected by this amendment): `roles=[]` legitimately un-hints an EXISTING record on reignite — a different parameter governing a different operation, not a creation-time escape hatch.

**The check is unconditional, not outcome-gated.** It runs before scope/purpose/role validation, the embed call, and the dedup gate — i.e. before it is known whether this call will end up writing a brand-new record or reinforcing an existing ember via the dedup gate's "reignite" branch. This mirrors `set_hot(hot=True)`'s own precedent exactly (checked "before the record is even fetched," per that tool's docstring): the discipline the caller pays is "always state who this is for when you author a correction," not "state it only when it happens to be new." `audience_hint` itself still only ever governs a fresh write; on a reignite it is simply unused unless `roles` (a separate, already-optional parameter — see the 2026-08-03 amendment above) retags the stored hint.

**`Fact.provisional` — a persisted field, not an inference from the hint's shape.** The owner explicitly rejected treating "hinted all four roles" as *meaning* provisional and letting a future catalog tool infer its review queue from that shape: `plan.md`'s argument for retiring implicit universality is that an untagged atom "cannot distinguish *deliberately for everyone* from *nobody has curated this yet*" — and an all-four hint recreates precisely that ambiguity one layer up (it could mean "the owner said everyone" or "an agent guessed and nobody has ruled"). A known, testable, one-time schema-migration cost beats a permanent distinction the data cannot express. `provisional` is Fact-only, defaults `False`, and is set `True` at exactly one write site (`engine.embers.save_ember`'s agent-default branch).

**The defined path by which the owner rules and the mark clears.** An **owner-asserted** `roles=[...]` retag on the SAME ember's reignite (`ember(asserted_by="owner", audience_hint=[...], roles=[...])`) sets `provisional=False` on the persisted record, riding the same `transition()` call the retag itself already makes — one write, not two. **`audience_hint` must accompany `roles` on this call, not stand alone** — it is required unconditionally on every `asserted_by="owner"` call (checked before the dedup gate resolves reinforce-vs-fresh-write, above), so a call shape showing only `roles=[...]` would raise before this branch is ever reached (an earlier version of this prose illustrated the retag-and-clear call with `roles` alone). `audience_hint` itself has no further effect on a reinforce outcome — only `roles` retags the stored hint — but its presence is what lets the call through the creation-time guard at all. Gated on `asserted_by`, mirroring the owner/agent split `record_ember_recurrence` already applies to the recurrence counters just above in the same dispatch branch: an **agent**-asserted retag narrows the audience hint but leaves `provisional` untouched, since an agent narrowing is not the owner ruling. A cleared record stays cleared across a reopen (persisted field, not derived).

**A second, definitely-reachable path ** retag above only fires inside the ember auto-reinforce branch, which is unreachable under the real default `EngineConfig(paranoid=True)` — every reignite bounces instead, and `ember` carries no `resolve` argument to adjudicate a bounce (unlike `save`). `set_hot(id, hot=True, roles=[...])` is itself always an owner-curated audience decision (`roles` is required there — the 2026-08-03 amendment, above) and never touches the dedup gate, so `_set_hot` now also clears `provisional=False` on promotion, independent of `paranoid`. `hot=False` demotion leaves the mark untouched. Without this second path, the task's own headline acceptance criterion — "a defined path by which the owner rules and the mark clears" — would hold only under a test harness's `paranoid=False` override, not under the store's actual default configuration.

**Schema consequence — the single most-repeated hazard in this project.** Adding a domain field to `Fact` breaks every pre-existing LanceDB store unconditionally: write paths build Arrow rows against the new schema, and `merge_insert` fails with `Field 'provisional' not found in target schema` against a table that predates the column — the exact defect class that has already reached this store's production twice (the recurrence-counter migration; the conditional-`merge_insert` panic on a migrated table). Closed by the SAME open-time `_migrate_table_schema` mechanism those two used: `LanceDBAdapter.open()` backfills `provisional` on any pre-existing `records` table, defaulting **`False`** — the ten hot records were explicitly curated by the owner on 2026-08-03 and must not be re-opened for adjudication, and everything else pre-existing is unhinted and already off the push tiers, so a non-provisional default is both safe and semantically right. The conformance suite (which only ever builds a fresh, already-current store) structurally cannot exercise this — coverage lives in an old-schema-then-migrate fixture (`tests/test_lancedb_provisional_migration.py`), the same shape `tests/test_escalation.py`'s recurrence-counter migration test and `tests/test_lancedb_schema_migration_supersession.py` already established.

**What survived unedited.** The RETAG path (`ember(roles=...)`) stays optional exactly as the 2026-08-03 amendment (above) settled — `roles=None` still leaves the stored hint untouched on reignite; this is the CREATION path (`audience_hint`), a different parameter with a different job, and does not reopen that decision. `TestNoAutoPromotion` (an unhinted ember reinforcing with no `roles` by design) passes unmodified. B1's pull/push split, the routing predicate's binary shape, and `select_default`'s locked non-Curator shape are all untouched by this amendment, same as every amendment before it.

Living docs updated in the same pass: `docs/SPEC.md` §1.1 (`Fact.provisional` field), §8.4 (new "Rung 0" paragraph, including its second-path addendum — never §8.5), §9.1 (`ember` signature gains `roles`, both signature and comment updated). The four canonical role-name constants moved from `engine/core.py` into the shared leaf module `engine/_types.py` so `engine.embers.save_ember`'s agent-default and `Tabularium.initialize()`'s seeding read the SAME fixed tuple and can never drift; `core.py` imports it rather than defining it. `engine/dispositions.py::_set_hot` now also clears `Fact.provisional` on `hot=True` promotion — a `set_hot` behavior change, not an `ember`-surface one, so it is not itself a §8.5 edit.

---

## Amendment (2026-08-03) — `delete_role` closes the honesty gap

**Owner decision, 2026-08-03** (curation block). The earlier amendment above named an honesty gap rather than papering over it: the storage seam exposed `create_role`/`list_roles` only, so retiring `"default"` could only mean "no longer seeded on a fresh store" — a store that already held the row kept it forever. The owner ruled that weaker guarantee unacceptable: the existing store is a proof-of-concept data-acquisition artifact, not a production store to be preserved, and the seed will not be run again, so design must not be weakened — and a compatibility shim must not be added — to keep that store's current contents intact.

**What changed.** `delete_role(name: str) -> None` now exists on the `StorageAdapter` Protocol and on both `MemoryAdapter` and `LanceDBAdapter`. Two interface questions the task brief required settling explicitly, not left to adapter accident:

- **Deleting a name that is not currently registered is an idempotent no-op — it does not raise.** Symmetric with `create_role`'s duplicate-registration no-op. This is load-bearing: `Tabularium.initialize()` now calls `delete_role("default")` **unconditionally**, on every engine startup, including against a fresh store that never had `"default"` at all — a raising `delete_role` would force that call (and every future caller) to pre-check `list_roles()` first, for no benefit `create_role` doesn't already spare its own callers.
- **Deleting a role does not touch any record's `audience_hint`, and does not refuse because some record still names it.** The roles table and the `records` table are unrelated stores — a hint is a list of strings, not a foreign key. A hint entry naming a just-deleted role goes **inert**, not dangling: `context(role=...)` already validates the requested role against the registry (`_require_roles`) before using it as a routing key, so nobody can ever again *request* a deleted role — the stale hint entry can therefore never again route a record to any payload, while any OTHER role named in the same hint is completely unaffected. Refusing the delete on stale-hint grounds was considered and rejected: it would make deregistration hostage to data an operator has no easy way to enumerate or fix, in exchange for a guarantee the registry gate already provides for free.

**No schema change.** The `roles` table's shape (`_REGISTRY_SCHEMA` — a single `name` column) is unchanged; `delete_role` only removes a row via each adapter's native delete path (`LanceDBAdapter` via `AsyncTable.delete(predicate)`, mirroring `bury`'s existing unconditional-predicate delete of a records row; `MemoryAdapter` via a plain list removal). Because no column is added or altered, this closes without needing an old-schema-then-migrate fixture — the schema-evolution blind spot the conformance suite structurally cannot catch (`Field '<name>' not found in target schema`; the conditional-`merge_insert` panic) applies to a MISSING column, not to a row-level delete against an already-current schema. The "already holds `default`" scenario this task closes is a **data** fixture, not a schema one, and is covered directly: the conformance suite's dual-parameterized `TestRegistries` exercises `delete_role` against both adapters, and `tests/test_engine_m1.py`'s `test_initialize_idempotent_against_live_store_shape` pre-seeds `default` + `reviewer` (the live store's actual shape as of 2026-08-03) before calling `initialize()`, asserting `default` is now gone rather than surviving.

**What changed in prior wording.** The earlier amendment's "Honesty finding" paragraph (above) and its counterparts in `docs/SPEC.md` §7/§10.1/§10.3 and `engine/core.py`'s module + `initialize()` docstrings described a real, dated limitation — it is left as the historical record of what was actually shipped, not edited in place. It is superseded by this amendment: neither `StorageAdapter` implementation lacks a deletion path any longer, and "retired" now means what it always sounded like it meant.

**What survived unedited.** Everything every prior amendment in this ADR carries forward survives this one too: B1's pull/push split; `audience_hint` never touches pull; the routing predicate and its binary inclusion/exclusion shape; `select_default`'s locked non-Curator shape; `Fact.provisional`'s semantics and the two clearing paths (ember reignite, `set_hot` promotion) — this amendment adds a registry-deletion verb, it does not touch provisional or the routing predicate.

Living docs updated in the same pass: `docs/SPEC.md` §3.1 (storage seam signature list), §7 (new paragraph, appended after the paragraph rather than editing it), §10.1 (Roles registry table row), §10.3 (paragraph); `engine/core.py` (module docstring, roles-seeding comment block, `initialize()` docstring — all corrected in place, since these describe current behavior rather than archiving a decision); `engine/_types.py` and `engine/embers.py` (citation comments naming the now-closed "no deletion path" case updated so they no longer point at a stale claim, while the underlying live-registry-snapshot risk they document — general, not `"default"`-specific — is unchanged).
