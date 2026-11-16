# 0021 — `shelved`: demotion the dedup gate does not see

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted — **amends** [0005](0005-hypomnema-schema-and-storage-seam.md) (the Fact lifecycle gains a fifth value), [0007](0007-recall-pipeline-and-context-composition.md) (a lifecycle that is in the primary recall pass but never a dedup candidate) and [0012](0012-mcp-tool-surface-design.md) (`set_shelved` joins the tool surface).

The curation pass on `tabularium` + `global` buried 233 records on 2026-08-04, and in doing so established exactly what burial costs. `bury` drops the row from `records` and writes a six-field tombstone carrying no content (`lancedb_adapter.py:1113`), and `_load_tombstones` re-deletes any `records` row whose id appears in the tombstone set on *every* open — so a "restore" is a new record with a new id, and the re-save can itself be refused by the dedup gate. The only real recovery is a cold-copy snapshot of the whole store, which by the time that pass ran had aged 35 records behind the live store and would have discarded three unrelated scopes' writes to use.

That is the right price for destroying something genuinely wrong. It is the wrong price for **demotion**, and demotion is what the curation passes are actually doing. The next pool — `project-a`, 2967 eligible records, 30% of them mentioning a ticket — is overwhelmingly material where the judgement is *"this is not standing knowledge"*, not *"this is false"*.

## The harm junk actually does

Two harms, and they are not equally live:

- **It costs budget in a grounding payload.** Already solved: `context` reads `hot` + `ember` only, and nothing is born hot.
- **It generates nonsense bounce candidates against genuinely new atoms, forever.** This is the real one. `_probe_filters` (`filters.py:59`) requests all seven lifecycle values *and* both validity states, deliberately (SPEC §4.1) — the gate is supposed to see everything, so that a duplicate cannot hide behind a demotion.

The consequence of that design is that **no existing demotion reaches the gate**:

| state | out of grounding | out of recall | out of the dedup probe | reversible |
|---|---|---|---|---|
| `archived` | yes | no — the fallback widen pass reaches it, **and a hit revives it to `active`** (`facts.py:312`) | **no** | n/a |
| `set_validity(valid=False)` | yes | yes (recall is valid-only by default) | **no** — the probe requests both validity states | yes |
| `bury` | yes | yes | yes | **no** |

So there is no state that means *"a curator judged this not worth carrying, but it is still true and still findable."* `archived` un-demotes itself the moment anyone recalls it, `set_validity` asserts a falsehood claim that is usually wrong about the record, and `bury` solves it by destruction.

## The decision

A fifth Fact lifecycle value, **`shelved`**. Its one novel property is that it is **the only lifecycle the dedup probe does not see**. Every other property is chosen to keep that one honest:

- **It stays in `_primary_recall_filters`.** Explicit `recall` still returns it — demotion must not hide knowledge, only stop the store defending it.
- **It never revives on a hit**, unlike `archived`. Shelving is a curator judgement; a usage signal must not silently overturn it.
- **It never reaches grounding.** Automatic — the packer reads `hot` + `ember`.
- **It is reversible in one call:** `set_shelved(id, shelved: bool, reason: str)`, shaped after `set_validity(id, valid, reason)` so both directions live in one verb.

`archived` and `shelved` stay distinct and neither absorbs the other. `archived` means *cold but still authoritative* — revivable on demand, still deduped against. `shelved` means *judged not worth carrying* — no revival, not deduped against.

## Considered options

- **Keep burying and accept the cost.** Rejected. It makes every curation pass irreversible, which forces a conservative panel — and the `tabularium`/`global` panel was *already* systematically more conservative than the owner, who never once rescued a record both judges condemned but condemned 33 they both wanted to keep. Irreversibility buys caution the owner does not want and does not pay for the under-burial it causes.
- **Reuse `archived`, and remove `archived` from the probe.** Rejected: it conflates "cold, still authoritative, revive on demand" with "a curator says this is not worth carrying". Revival is correct behaviour for the first and exactly wrong for the second, and one value cannot have both.
- **Use `set_validity(valid=False)` as the demotion verb.** Rejected: it asserts *this is false*, which is a different and usually incorrect claim — most junk is perfectly true and merely not worth carrying — and it does not leave the probe, so it fixes nothing that matters.
- **A `junk` boolean orthogonal to lifecycle.** Rejected: a second axis that every filter, census and drift report must learn to consult. Lifecycle already *is* that axis; adding a value to it costs one `Literal` entry.

## Consequences

- **`Fact.lifecycle` gains a value** (`domain.py:199`). Decisions are untouched — they have their own arm and their own three states.
- **`_probe_filters` stops being "all lifecycle values".** SPEC §4.1's guarantee that the gate sees everything gains an explicit exception, and that is the load-bearing amendment in this ADR. Everything else here is mechanism.
- **Duplicates of shelved content can now be saved.** Intended, and stated plainly because it is the cost side of the trade: shelving withdraws the store's defence of that content. A later save of the same atom succeeds, and the two coexist — the shelved one reachable only through `recall`.
- **The reason is persisted, as an event.** `events.reason` has existed since M2 slice B but is documented as feedback-only (`lancedb_codec.py:104-107`); `set_shelved` becomes its second source. This is deliberately *not* how `set_validity` behaves — its reason goes to an in-memory map that does not survive a restart, and the comment at `lancedb_adapter.py:1103-1104` concedes the conformance suite does not test it. Repairing `set_validity` is out of scope here; copying it is not an option.
- **The `project-a` pass becomes reversible**, so its judge panel can run wide and unattended rather than narrow and supervised. A mistake costs one `set_shelved(id, shelved=False)`.
- **Shelved records still occupy `recall` results.** Accepted on the owner's call: the harm being removed is the dedup gate, not search noise.
- **`bury` survives unchanged**, for records that are genuinely wrong rather than merely not worth carrying. The 233 already buried stay buried.
