# 0018 — Every search is bounded: `candidate_k` derives from `limit`, and the unbounded mode is removed

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted — **amends** [0005](0005-hypomnema-schema-and-storage-seam.md) (the retrieval legs take `k: int`; the `k=None` exhaustive clause is deleted), [0007](0007-recall-pipeline-and-context-composition.md) (the candidate pool is sized from the caller's `limit`, not from a fixed config constant) and [0012](0012-mcp-tool-surface-design.md) (`exhaustive` leaves the `recall` tool surface).

Two defects of the same family, found on 2026-07-29. Both are a caller-facing parameter silently overridden by an internal constant.

**Asking for more did not fetch more.** `engine/recall.py:74` sized the candidate pool as `cfg.candidate_k` (default 50) and never consulted `limit`, which was applied only after RRF fusion. So `recall(limit=200)` returned at most the fused pool drawn from 50 candidates per leg — roughly 100 records — silently, with no degraded flag. The owner had hit exactly this shape before on the predecessor system ("I once did a full sweep and there was a 50 max") and named not repeating it as a requirement.

**"Give me everything" returned exactly ten.** The seam documented an exhaustive mode twice as a MUST (`storage/types.py:300`, `:319`): `k=None` ⇒ return *all* matching rows. The reference adapter obeyed. `LanceDBAdapter` guarded as `if k is not None: q = q.limit(k)`, so with `k=None` no limit was ever set and LanceDB applied its own hard-coded default of **10** — verified against queries whose true match count was 100 and 300. The mode advertised as *more thorough* returned fewer results than the ordinary path's 50.

## The decision

`candidate_k` **derives from `limit`**. Asking for 500 fetches 500 per leg: explicit, predictable, the cost stated by the caller and paid exactly. That is the entire fix for the sweep case, and it needs no gate because it is bounded by construction.

And **there is no unbounded mode anywhere**. `k=None` is not repaired — it is deleted from the storage contract. Both adapters take `k: int`. This resolves the silent-ten defect by removing the broken concept rather than fixing it, leaving a smaller system with no path whose cost is unbounded in archive size.

## Considered options

- **Fix `k=None` to be genuinely exhaustive and keep it on the tool.** Rejected: an unbounded scan over a corpus that grows forever is a footgun for every caller who touches it, and ADR 0012 fought for a small deep surface over verb-sprawl.
- **Fix it, remove it from the tool, and expose it on the CLI as a human act** — gated by the shell, since Tabularium has no authentication by design (no owner, no principal; Role is "cooperative self-identification, never authentication", ADR 0015/0016) and an honour-system flag would be unenforceable. This was accepted and then superseded within the same session: it preserves an unbounded-cost path whose only justified consumer, a full sweep, is served just as well by an explicit large `limit` at a bounded and stated cost.
- **An honour-system flag** — "only set this when the user explicitly asks." Rejected: unenforceable, and the Role precedent does not transfer. Role is safe because nothing is hidden, so a false claim gains nothing; here a false claim would gain unbounded compute.
- **Bounded everywhere, `candidate_k` from `limit`** — chosen.

## Consequences

- `exhaustive: bool` leaves the MCP `recall` tool (`server.py:278,287`). With nothing unbounded to request, the parameter has no referent.
- The eval harness stops hardcoding `candidate_k: None` (`tests/test_eval_b5.py:84-91`) and passes a number large enough to cover its 162-text frozen corpus — it states its bound like every other caller.
- `test_hypothesis_k_none_exhaustive` (`tests/conformance/test_conformance.py:1697-1707`) is replaced by "assert `k` is honoured exactly." The test it replaces was doubly broken — hardcoded to `_make_memory_adapter()` so it never ran against LanceDB, *and* capped at `n=8`, below the threshold at which the defect becomes visible. The replacement is testable at any sample size, which is why it could not have hidden this bug.
- No gated CLI sweep is built. Its only purpose was to gate the unbounded mode.
- Raising `limit` now genuinely costs more work — the honest trade. A caller asking for 500 pays for 500 per leg, and that is visible in the request rather than hidden in a config constant.
