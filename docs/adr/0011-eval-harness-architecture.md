# 0011 — Eval harness architecture (T01)

> **Frozen 2026-08-07.** Historical prior — it records what was decided and why at the time, is
> not maintained, and diverges from the built system in places. There is no present-truth
> document: the code is what is true now, and new architectural decisions are recorded in the
> Tabularium store, not as new ADR files. `README.md` orients.

**Status:** accepted

Resolves T01 — the #1 rigor gap, and per §10 the highest-leverage first artifact (it's language-independent and can run against project-b's existing ledger before any Tabularium Python exists). Built against the locked model (ADR [0005](0005-hypomnema-schema-and-storage-seam.md)–[0009](0009-importance-decay-eviction.md)), the R05 previous memory store trace (§4–7), and R01 (benchmark landscape).

## Two modules over one substrate

The eval is **two modules sharing one substrate** — a hand-built golden corpus + an injectable `Clock` seam + event-log replay (where `trajectory = f(log, clock)`):

1. **Regression GATE** — pass/fail, runs on every `pytest` / CI push, **blocks merge on a behavioral regression**. Bit-deterministic; no nondeterministic components allowed inside this module.
2. **Calibration harness** — sweeps a Pareto frontier of tunable config values (β, RRF weights, cosine thresholds, archive threshold) and outputs a versioned per-embedder threshold config. Occasional runs; no pass/fail; purely an optimizer.

**Why split?** Conflating them would either (a) put a nondeterministic optimizer in the CI gate (flaky passes/failures from the sweep) or (b) freeze calibration targets into assertions (can't sweep to find the optimal operating point). The shared substrate avoids double-building the costly parts — the golden corpus, the Clock seam, and the replay harness serve both modules.

The substrate is specified in ADR [0009](0009-importance-decay-eviction.md) §"Correctness eval": the `Clock` seam makes time-travel possible; the append-only event log makes the full lifecycle trajectory a pure function of `(event_log, clock)` — correctness is replayable by construction.

## The direction axis: promote-side and demote-side are co-equal

Every ranking and lifecycle mechanism in Tabularium is **bidirectional**. The eval covers **both directions**:

**Demote-side** (defensive, table-stakes — the things a system must not do wrong):
- idle importance strictly decreasing between references (the R05 §6 anti-freeze property — the project-b bug inverted into a golden regression)
- archive fires at `decay_due_at ± tol`
- dedup gate never false-merges distinct records
- recall excludes `invalid` memories by default
- eviction evicts the correct (lowest-importance) hot Fact

**Promote-side** (the actual intelligence — the things a useful system must do right):
- importance **rises** on genuine reference
- an ember earns a codification suggestion when its dedup-only recurrence counter `r ≥ escalation_threshold`
- a Fact earns `hot` when its importance crosses `hot_promote_threshold`
- an `archived` Fact is **revived** (in-place, same id) on a dedup-hit re-assertion
- dedup-reinforce increments `recall_count` + logs the event correctly (the counter is not silently dropped)

**Rationale for co-equal elevation.** An eval that checks only the demote side would pass a write-only archive that promotes nothing — a useless system. Promotion tests are the intelligence tests; demotion tests are the correctness floor. Both belong in the gate. T08's anti-freeze assertion is the headline *example* of a demote-side trajectory family, not the centerpiece of the whole harness.

## Metric families × {gate, calibrate} matrix

| Family | Gate | Calibrate |
|---|---|---|
| **Recall quality** | recall@k + precision@k ≥ floor (binary relevance, frozen set) | sweep β/RRF weights to maximize recall@k |
| **Dedup precision/recall** | dedup precision ≥ `P_floor` over labeled pairs | sweep cosine threshold → maximize recall s.t. precision ≥ `P_floor` |
| **Decay/eviction correctness** | property tests + project-b-sourced trajectory fixtures (promote/demote) | calibrate archive threshold → minimize false-archival (revived-within-N vs store bloat) |
| **Latency & cost** | report-only (p50/p95 + token cost; never blocks) | — |
| **Model-slip rate** | report-only (dogfood trend; never blocks) | — |

## The inversion: recall relevance is the black hole

The most expensive place to sink eval effort in the memory-system literature is **recall relevance quality** — graded relevance labels, LLM-judge calibration, endless IR metrics (MRR / nDCG / MAP), living-benchmark upkeep. Memory papers spend there because retrieval is *their* differentiator.

**Tabularium's differentiator is lifecycle correctness**, which is **more** deterministically evaluable (crisp assertions over replayed logs, bounded by construction). So the harness weight sits on the deterministic trajectory evals; recall quality is deliberately **capped** to a smoke tripwire:

- **Binary relevance only** — no graded labels, no LLM judge in the recurring loop.
- **Frozen set of ~20–40 queries** — hand-curated dogfood queries, committed, never re-generated per run.
- **Fixed k** — recall@k + precision@k at a single fixed k.
- **A floor gate** ("recall is not obviously broken"), not an optimization target.

This is a feature, not a limitation. Honesty about small-n at a binary floor **is** the §8 discipline-not-superiority framing (§7: no claim the code can't back).

## Three-source corpus

The golden corpus is assembled from three layers:

**(1) previous memory store export (R05-traced)** — a real pre-existing memory dataset, traced in detail in R05. This is the **authenticity win**: real-world memory patterns from a production-like system, not synthetic proxies. It also provides the **§10 unlock**: because the corpus is project-b-sourced (raw PHP export, no Tabularium Python), T01 can run before a single line of Tabularium's core engine exists — the highest-leverage possible bootstrapping order. The previous memory store seeds the recall set, the dedup labeled pairs, the model-slip corpus, and the concrete trajectory fixtures.

**Circularity guard (hard rule).** Ground truth is derived from the **spec's threshold arithmetic**, never harvested from an engine's own output. This applies doubly to project-b-sourced fixtures: project-b's logs are realistic **inputs** to Tabularium; project-b's *behavior* is **not** the oracle, because project-b carries the R05 §6 decay-freeze bug Tabularium is testing against.

**Gift:** a real project-b log where the freeze bug fired becomes the best anti-freeze regression fixture — authentic pathological input, assert Tabularium does **not** freeze on it.

**(2) Hypothesis-generated synthetic logs** — Hypothesis property tests generate logs on the fly; no hand-authoring of synthetic data for the property layer. This is the primary adversarial coverage mechanism.

**(3) Tabularium's own dogfood log** — grows post-build; the §2 self-documenting recursion. The recall quality frozen set will ultimately stabilise as a dogfood snapshot.

**Hand-authored synthetic fixtures** demote to a fallback, used only for scenarios the project-b seed doesn't cover (e.g. edge-case numeric-diff guard trips not present in the project-b export).

## Eval form: 30% concrete fixtures / 70% property tests

**(a) ~30% concrete golden fixtures** — sourced from the previous memory store seed. Named scenarios (readable, demo-ready), covering the promote-side regression suite (ember→Decision escalation, hot-earn, archived→revived, dedup-reinforce) plus the anti-freeze fixture. These are the **readable spec-by-example** layer — a reviewer can see exactly what event sequence produces what trajectory.

**(b) ~70% property-based (Hypothesis)** — the primary discipline. Invariants over generated logs:
- importance is monotonic-non-decreasing between references (the anti-freeze generalized to arbitrary sequences)
- every burial leaves a tombstone (no silent hard-delete)
- the additive floor is never breached downward (usage-earned permanence invariant)
- dedup never false-merges distinct records (a guard invariant holding across all generated near-pairs)

Property tests carry the heavier weight because lifecycle correctness is Tabularium's differentiator and invariants give the broadest adversarial coverage. Hypothesis is also a loud "modern Python rigor" signal (§5/§8 of the original project brief).

**Fixtures are the readable anchors; properties are the adversarial net.**

## Hermetic gate: bit-deterministic, frozen vectors only

**The determinism rule.** The regression GATE operates **only on FROZEN vectors** — never a freshly-computed embedding:

- **Hand-written synthetic vectors** for exact fusion-math / ranking assertions (the pure-float fusion arithmetic is under test; the vector values are fixed inputs).
- **Snapshotted real-embedder vectors** committed into the fixture file for the recall/dedup floor assertions (a frozen snapshot of what the local Ollama model produced at corpus-construction time).

**Why freeze, not pin?** Two distinct rationales:

1. A test fixture has **fixed inputs**, and an embedding is an input to the recall/dedup logic — re-embedding per run is equivalent to re-fetching test data from a live model each run. A gate that re-fetches its own test data on every run is not a gate.
2. A regression gate must **isolate its variable-under-test = Tabularium's own code**. Re-embedding would inject a non-Tabularium failure source (embedder / model / runtime / quant drift) into a Tabularium gate — producing a worse gate and a latent flake.

**A live neural forward pass is not bit-identical** across hardware / runtime / time: float non-associativity under parallelism, x86-vs-ARM / FMA differences, BLAS and runtime version, quantization, batching and padding. Pinning a model version buys reproducible-on-one-box, not true cross-env determinism — the only route to bit-identical gate results with real embeddings in play is to freeze the vector. "Deterministic enough" is not accepted: jitter can flip a gate result at a near-tie straddling the k-th rank boundary — bounded but real, hence killed.

**What gates the merge (blocks CI on failure):**
- All property tests (Hypothesis, pure `f(log, clock)`)
- The project-b-sourced golden fixtures (including the anti-freeze fixture)
- recall@k / precision@k FLOOR over the frozen ~20–40-query set
- dedup precision ≥ `P_floor` over labeled pairs

**Amendment (2026-07-29, storage-stabilization plan):** the recall@k/precision@k floor is **per-adapter**, not a single shared number — `tests/test_eval_b5.py`'s `RECALL_FLOOR`/`PRECISION_FLOOR` are dicts keyed by adapter id (`memory`/`lancedb`), each measured and gated independently, because the two adapters run different keyword-search implementations and were never expected to produce identical rankings. Before this amendment the floor was measured and gated only against the in-memory reference adapter; the number never characterized the deployed (LanceDB) adapter. `P_floor` (the dedup gate) is unaffected — that gate computes cosine directly from embeddings and never invokes adapter search, so it has no adapter-specific value to measure. See an internal planning note for the measured comparison.

**Report-only (never blocks merge):**
- Latency p50 / p95 + token cost (telemetry regression alert, not a gate)
- Model-slip rate (dogfood trend, not a gate)
- LongMemEval_S score (one-time reference number, never wired to CI)
- The entire calibration harness (it is an optimizer, not a gate; its outputs are versioned config, not pass/fail assertions)

## Live embedder lane: integration, not hermetic

The hermetic gate tests Tabularium's own code against frozen vectors. The **live embedder** is still essential and still exercised — in a separate lane:

```
pytest -m integration
```

This lane re-embeds the frozen set using the real local Ollama model (via litellm, ADR [0003](0003-provider-layer-litellm-pydantic-ai.md)) and asserts the live vectors still clear the recall and dedup floors. It answers: "did an embedder or model-version change break real recall?" and validates that the committed snapshot remains representative.

**"Hermetic" means: no remote provider** (no network dependency, no cost, no rate limits), **not: no embedder.** A pinned local embedder has no supply problem. This lane is allowed to be env-sensitive because it tests the integration, not Tabularium's logic; a red result is a human-look signal, not an unrelated-PR merge-block.

An optional **zero-setup lane** (clone → `uv run pytest` → green without pulling Ollama) is possible if the repo goes public (ADR [0010](0010-project-posture-private-indefinite-full-build.md)) — the cached frozen vectors already provide it; whether to make it a first-class advertised lane is a build-time call.

## Public benchmark

**LongMemEval_S** — run **once** as a documented reference-frame number (R01 finding: the LongMemEval leaderboard is unverifiable, but the benchmark itself is usable as a reference point). This number is never wired to CI and never used as a gate or optimization target. Cited with the §8 "discipline not superiority" framing.

**LoCoMo** — not reported (R01 finding: the gold set is corrupted; citing a number from a corrupted benchmark violates §7).

## Considered alternatives

- **One conflated harness (gate + calibration merged)** — rejected: puts a nondeterministic optimizer inside the CI gate (flaky) or freezes calibration targets into assertions (can't sweep). Rejected, not deferred.
- **Demote-side eval only** — rejected: would pass a write-only archive that promotes nothing; the promote-side IS the intelligence test. Demote-only is table-stakes without the differentiator.
- **Graded relevance + LLM judge + living benchmark for recall quality** — rejected: the black hole. Memory papers spend their entire eval budget there because recall is their differentiator; Tabularium's differentiator is lifecycle correctness, which is more deterministically evaluable. A graded recall eval would optimize the wrong seam.
- **Re-embedding inside the gate ("deterministic enough")** — rejected: float non-associativity + hardware/runtime/quant variance means near-ties at the k-th rank boundary flip — bounded but real; killed at source by freezing. "Deterministic enough" is not accepted.
- **Remote embedder for the integration lane** — rejected: network dependency, cost, rate-limit exposure. Local Ollama (ADR 0003) is the only embedder in scope.

## Consequences and deferred-to-build items

**Relationships:** consumes ADR [0009](0009-importance-decay-eviction.md) (Clock seam, event-log replay, anti-freeze property, calibration-harness ownership boundary, injected-dup reconciliation test), ADR [0008](0008-dedup-gate-and-reinforcement.md) (event log for replay, P_floor, labeled dedup pairs, reinforcement-event fixtures), ADR [0007](0007-recall-pipeline-and-context-composition.md) (RRF/β ablation, degraded-mode measurement, typed `degraded` flag), ADR [0006](0006-fact-taxonomy-validity-and-monotonic-dispositions.md) (model-slip corpus: correction-valence embers + invalidated records + burial tombstones), ADR [0003](0003-provider-layer-litellm-pydantic-ai.md) (local Ollama embedder in the integration lane). References R01 (benchmark landscape, LongMemEval_S, LoCoMo ruling), R05 (previous memory store trace: the corpus seed and the freeze-bug log). **Does not amend any prior ADR.**

**Implementation-tunable at build time (not open design questions):**
- Exact model-slip-rate formula (the metric family is locked; the formula is a build-time choice).
- Calibration-sweep objectives in full detail (defaults: β/RRF sweep → maximize recall@k; cosine threshold sweep → precision-favored F-β subject to precision ≥ `P_floor`; archive threshold sweep → minimize false-archival measured as revived-within-N vs store bloat).
- Whether to advertise the zero-setup cached-vector lane publicly (build-time call, ADR 0010).

**Closes:** T01. The #1 rigor gap is now specified. All three core primitives plus the eval harness are designed; the spec phase continues with T03 / T04 / T12.
