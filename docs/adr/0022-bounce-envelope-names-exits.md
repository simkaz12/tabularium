# The bounce envelope reports evidence and names exits; it never suggests an action

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted (2026-08-06)

The dedup gate derived `suggested_action` from cosine similarity alone, and hardcoded
`suggested_action="supersede"` with the guard check suppressed whenever the best *candidate* was a
Decision (`engine/decisions.py:606-612`, `engine/facts.py:565-573`). Dogfooding recorded it wrong on
roughly 26 consecutive bounces, and in the Fact-save cell the suggestion was not merely wrong but
unexecutable — a Fact cannot supersede anything. The gate is therefore changed to report **evidence
only**: `BounceInfo` splits into `FactBounce` and `DecisionBounce`, discriminated on the kind being
**saved**, each carrying a list of named **exits** (an exact `how` call plus the `when` judgement that
selects it) in place of a single asserted action. The engine names the exits and chooses none —
the shape `EscalationSuggestion` already established for the ember ladder (`results.py:224-245`).

The root cause is epistemic, not a tuning error: **similarity measures topic adjacency, and
supersession is a claim about intent.** A restatement, an elaboration, a sequencing note and a genuine
replacement all produce the same high cosine, because they all share a subject. No threshold over that
signal can separate them.

## Considered options

- **Drop `supersede` entirely.** Rejected by the owner. Supersession is real and is precisely what the
  T09 lineage render exists to display — "x is achieved by y; later, x is achieved by c". Removing the
  value would delete the prompt that makes an agent notice a replacement at all. The value survives;
  what stops is *asserting* it from similarity.
- **Un-suppress the existing guards and let them discriminate.** Rejected on measurement. The guards are
  set-inequality checks over numerals, version strings, capitalised tokens and negation words
  (`engine/guards.py:20-55`); they fire on virtually all real prose written months apart — every one of
  the 985 fact bounces in the seed run carried a guard flag. A signal that trips ~100% of the time
  cannot separate restatement from replacement; wiring it in would swap one unconditional answer for
  another.
- **Add a second, higher threshold (`supersede_floor`) — the literal "two knobs, not one"
  proposal.** Rejected. Calibrating it needs its own labelled set, which does not exist, and the
  calibration consumption path is dead code (`config.py` loads no files; `calibration/*.json` is read by
  nothing). The knob would be a number with no evidence behind it. The answer is
  therefore **no second knob and an honest envelope**, not a second threshold.
- **One uniform envelope for both kinds.** Rejected by the owner in favour of two. The exit sets
  genuinely differ — `FactBounce` is always a fixed pair, `DecisionBounce` is a triple whose third
  member varies with the candidate — so one type would carry a conditional the caller must learn.
- **Split on the candidate's kind rather than the saved kind.** Rejected: that is the current structure,
  and it is what offers a Fact save an exit it cannot take. What the caller is *able to do* is set by
  what they are writing, not by what they matched.

## Consequences

- `verdict`'s `decision_supersede` becomes `decision_adjacent`, so all three values name observations
  rather than one naming a conclusion. Both `verdict` and `suggested_action` are transient — constructed
  in `_make_bounce`, never persisted to any row, column or event — so the rename needs no migration and
  no dual-read. Historical artifacts (the seed report's 258 `decision_supersede` rows, T-ZERO-LOG,
  the session notes) keep the old string as a record of what happened.
- `suggested_action`'s `drop` value is removed. Declining every exit is already expressed by making no
  call — the ember ladder's "the owner takes one, or neither".
- The `Decision`-save-against-`Fact`-candidate cell gains an exit the gate never surfaced:
  `promoted_from`, the same codify exit the ember ladder names. A Decision bouncing off a Fact usually
  *is* that Fact being codified.
- The six `when` strings are **owner-negotiated and pinned literally**, not paraphrasable guidance. They
  are the behavioural lever that stands between an agent correctly reaching for `supersedes=` and
  burning a permanent, first-judgment-wins `confirm_distinct`. Task 12 criterion 4 recorded the cost of
  letting agent-facing wording ship unratified.
- This is an interface break across the MCP surface. `SPEC.md:346` and `ADR 0012:83` carry the old
  `verdict` literal and are amended in the same commit per the standing same-day rule. The owner's
  global agent-instructions dedup-gate rule encodes the guard→`confirm_distinct` heuristic in prose;
  that logic now lives in the `when` strings and the rule should shrink to point at exits.
- `is_decision=True` was never exercised anywhere in the suite (`tests/test_eval_b5.py:589,625` both pass
  `False`), which is why this survived ~26 wrong suggestions unnoticed. Coverage of every cell is part of
  the change, held to the falsifiability bar: break each exit deliberately and watch a test go red.

## Amendment (2026-08-07) — the offered exit must also match the candidate's lifecycle

Filed live: a Decision save bounced against an `active` Fact, the envelope named `promoted_from` as
its third exit, and calling exactly that call raised `ToolError` — "only embers may be promoted (SPEC
§8.4)". This is this ADR's own guarantee failing one field over: `build_exits()` branched only on
`saved_kind` and the candidate's **kind** (Fact vs Decision), never on the candidate's **lifecycle** — a
third axis the original framing did not contain. Line 41-42's "What the caller is *able to do* is set by
what they are writing, not by what they matched" (kind-only) is corrected to: **and by the lifecycle of
what they matched**, for the one exit whose legality genuinely depends on it.

Two remedies were weighed. Gating `promoted_from` on `lifecycle == "ember"` — matching the engine guard
this ADR's `promoted_from` exit was calling into — was rejected: there is no `active→ember` transition
anywhere in the engine (`ember` is written only at birth, `engine/embers.py`, or ember→ember reignite,
`engine/facts.py`), so an ember-only gate would make the exit permanently unreachable for every Fact not
born an ember, which is the common case this ADR added the exit for. The chosen fix instead widens what
the engine accepts: `promoted_from` is pure provenance — `_write_decision` upserts only the new Decision
and never touches the target Fact, for any Fact, not only an ember (`tests/test_engine_b2.py`'s
`test_ember_stays_fact_after_promotion` and its 2026-08-07 sibling for the active case) — so the real
precondition was never "was this born an ember," it was "is this Fact still a live, engine-defended
assertion." `active`, `hot` and `ember` Facts satisfy that; `archived` (decayed out) and `shelved`
(curator-judged not worth carrying) do not.

Consequence for line 53-55 (the exit this ADR added): the Decision-save-against-Fact-candidate cell now
offers `promoted_from` only when the Fact candidate's lifecycle is `active`/`hot`/`ember`; an `archived`
or `shelved` Fact candidate drops `DecisionBounce` to the two-exit set (`reinforce`, `confirm_distinct`).
`DecisionBounce.exits` (`results.py`, `docs/SPEC.md` §4.4) widens from an unconditional 3-tuple to
`tuple[BounceExit, BounceExit] | tuple[BounceExit, BounceExit, BounceExit]` to carry this.
`PromotedFromSnapshot.valence` (`domain.py`) widens from `Literal["correction", "guideline"]` to
`| None` — an honest null for a promoted Fact that was never an ember, not a fabricated value. The offer
site (`engine/exits.py::is_promotable`) and the engine guard (`engine/decisions.py::_save_decision`)
share one predicate so the two cannot drift apart again — the same defect class this whole ADR exists to
close, recurring in a field the original fix did not cover.

Living docs updated in the same pass: `docs/SPEC.md` §8.4 (the promotable-lifecycle rule) and §4.4 (the
`DecisionBounce.exits` type and the lifecycle-axis correction to this ADR's own guarantee).

## Second amendment (2026-08-07) — the offered exit must also *resolve*, not merely be legal to call

Filed live, same day, one layer further in: a Decision save bounced against an `active` Fact, the
envelope named `promoted_from` as its third exit — now legal to call after the amendment above — and
calling it exactly as written, `save(kind="decision", content=…, promoted_from=[candidate_id])`, no
`resolve`, produced **the same bounce again**. Nothing was written. The save only landed once
`resolve={"candidate_id": …, "action": "confirm_distinct"}` was added on top, which the `how` string
never mentions and never has.

This is this ADR's own defect class — the gate naming an action it cannot support — recurring a third
time in the same envelope, past `suggested_action` and past the lifecycle-gating fix above. Both of
those closed *offer-site* mistakes: the string named something the engine outright refused. This one is
subtler — the engine no longer refuses the call, it just doesn't act on it, because `_save_decision`
checked `promoted_from` only during validation (does the target exist, is it promotable, is it in
scope) and never again during candidate dispatch — dispatch only ever looked at `resolve.candidate_id ==
candidate.id`. `promoted_from` shaped the birth-frozen lineage snapshot on the eventual Decision and
nothing else; it carried no instruction to the gate at all. The `how` string promised a resolution; the
engine read it as a footnote.

**The fix: make the promote exit a one-shot bypass.** When a Decision save's `promoted_from` names
exactly the `candidate_id` that caused this bounce, the save writes past the bounce — content saved,
lineage snapshot attached — and registers **nothing** in the distinct-pair registry. No `resolve`
required; the `how` string is now literally sufficient, which was always the point of an exit's `how`
being a call rather than a description of one. The codebase already had this shape: `reconciles`, the
single-shot write-past-an-invalid-bounce mechanism (`engine/facts.py`'s reinforce-on-invalid-candidate
branch, ~1029-1050) writes past a candidate the engine has already judged unfit to block on, without
spending an adjudication to do it. `promoted_from` is the same move for a different judgement — not
"this candidate is invalid," but "this candidate is what my decision rests on, and I already said so
by naming it."

Two remedies were weighed.

- **Route `promoted_from` through `resolve=confirm_distinct` automatically** — i.e., keep the
  `how` string as documentation and have the server synthesize the resolve call underneath it. Rejected:
  `confirm_distinct` is specified as "first-judgment-wins and permanent" (§4.5 [A6]) precisely because it
  registers a `DistinctPair` — a claim that survives every future bounce between that pair, forever.
  Manufacturing that claim automatically, for every promotion, spends an irreversible judgement the
  caller never actually made and never asked for. Worse, it is redundant: the engine already knows the
  two records are related — that is what `promoted_from` *means* — so registering them as merely
  "distinct" (as opposed to "one names the observation the other rests on") throws away the more precise
  relationship the save itself just recorded, in exchange for a cruder one that costs a one-shot resource.
  The rejected alternative would have made `promoted_from` **more** expensive to call correctly than
  discovering the bug and hand-adding `resolve` yourself — exactly backwards for an exit whose `how` is
  supposed to be sufficient on its own.
- **Widen `_handle_resolve_decision`'s existing `resolve` dispatch to also accept `promoted_from` as an
  implicit `action`, still writing a `DistinctPair`.** Same objection as above, one layer down — it still
  spends the registry entry, just via a different code path.

Both alternatives share the same flaw: they treat `promoted_from` as a variant *spelling* of
`confirm_distinct` rather than what it actually is — a stronger, already-adjudicated claim that makes
`confirm_distinct`'s job (park two similar-looking records so neither blocks the other later) moot. The
candidate a Decision is promoted from is never going to look like a coincidental near-duplicate of that
Decision again; there is nothing left for the registry to protect against.

Scope, deliberately narrow: the bypass fires only when `promoted_from` names the bounce's own primary
`candidate_id` — the record actually causing this bounce, from this probe. A `promoted_from` list naming
some other record, or naming one of `other_candidates` rather than the primary, does not bypass anything;
the save bounces exactly as before, and the (validated, but irrelevant-to-this-bounce) promotion
snapshots go unused because the call never reaches a write on that branch. Widening the bypass to "any
promotable id in `promoted_from`" was not seriously considered — it would let an unrelated `promoted_from`
entry silently absorb a bounce it had nothing to do with, the same "gate asserting something it cannot
back" shape this whole ADR exists to close, just moved from the offer side to the accept side.

Consequence for constraint 1 (the lifecycle-gating fix, previous section): unchanged, and re-asserted as
a regression guard here rather than re-litigated — `is_promotable`/`_PROMOTABLE_FACT_LIFECYCLES`
(`engine/exits.py`) stays the one shared predicate; the pre-lock `promoted_from` validation loop in
`_save_decision` still raises `ToolError` for an `archived`/`shelved` target before the bypass is ever
reached, and the bypass condition re-checks `is_promotable` on the locked-section candidate read anyway,
closing the narrow race where a Fact's lifecycle moves between that early unlocked validation and the
later candidate fetch.

No new config knob — the bypass is unconditional on the one narrow match it checks, not a tunable.

Living docs updated in the same pass: `docs/SPEC.md` §4.4 (the promote exit resolves the bounce one-shot
and registers nothing in the distinct-pair registry).
