"""Engine tests — hermetic + integration lanes (SPEC C4).

Hermetic lane
~~~~~~~~~~~~~
Uses ``FakeEmbedder``, ``MemoryAdapter``, and ``TestClock`` — no network, no disk.
All tests are async (pytest-asyncio auto mode).

Coverage (hermetic):
- Gate pass / bounce at the gray_floor boundary
- Paranoid = nothing auto-reinforces (effective ceiling == floor)
- Guard-flag annotation on near-dup with differing numeral / version / negation
- Bounce carries candidate_scope across the X<->global tier (A5)
- Embedder outage lands a vector-less row with embedded=False
- Recall RRF math on a constructed fixture (known per-leg ranks -> expected fused order)
- Boost changes order (beta > 0 with unequal recall_counts)
- Degraded recall = text-leg-only with flags
- Scope-span isolation (unrelated scope never surfaces; global always co-read)
- Fallback widen-only when primary is thin
- Candidate pool derives from the caller's limit (ADR 0018)

Integration lane (``-m integration``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Requires local Ollama with nomic-embed-text.  Skips cleanly if Ollama unreachable.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Fact, HypomnemaId, Scope, new_id
from tabularium.embedding import EmbedderUnavailable, LitellmEmbedder
from tabularium.engine import Tabularium, ToolError
from tabularium.engine.guards import guard_check
from tabularium.engine.scoring import _rrf_score
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import Filters
from tests.support.doubles import FakeEmbedder, TestClock

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DIM = 16  # small dim — faster hash-position spacing


def _make_adapter(dim: int = _DIM) -> MemoryAdapter:
    return MemoryAdapter(embedding_model="fake/nomic", dim=dim)


def _make_clock(year: int = 2025) -> TestClock:
    return TestClock(datetime(year, 1, 1, tzinfo=UTC))


def _make_config(**kwargs: Any) -> EngineConfig:
    """Return config with the production candidate_k seed (50) -- every fixture
    corpus in this file stays well under it unless a test overrides the value
    itself to exercise the cap explicitly."""
    defaults: dict[str, Any] = {"gray_floor": 0.80}
    defaults.update(kwargs)
    return EngineConfig(**defaults)


async def _engine(
    *,
    dim: int = _DIM,
    config: EngineConfig | None = None,
    should_fail: bool = False,
) -> Tabularium:
    """Build and initialize a Tabularium with FakeEmbedder."""
    adapter = _make_adapter(dim=dim)
    embedder = FakeEmbedder(dim=dim, should_fail=should_fail)
    clock = _make_clock()
    cfg = config if config is not None else _make_config()
    engine = Tabularium(adapter, embedder, clock, cfg)
    await engine.initialize()
    return engine


def _sha256_collision_variant(embedder: FakeEmbedder, base: str, *, tries: int = 500) -> str:
    """A text distinct from ``base`` (different content, different content
    key) whose ``FakeEmbedder`` vector collides with ``base``'s (SHA-256
    slot match) — for a fixture that needs a genuine near-duplicate the
    admission order's exact-key step (P7 admission-order piece) must NOT
    catch, so the probe is what bounces or reinforces it, not the exact
    key check."""
    target = embedder.peek(base)
    for i in range(tries):
        candidate = f"{base} (variant {i})"
        if embedder.peek(candidate) == target:
            return candidate
    raise AssertionError(f"no SHA-256 slot collision found for {base!r} in {tries} tries")


# ---------------------------------------------------------------------------
# C4: Gate pass / bounce at the gray_floor boundary
# ---------------------------------------------------------------------------


class TestGatePassBounce:
    async def test_save_novel_returns_saved(self) -> None:
        engine = await _engine()
        result = await engine.save("The capital of France is Paris.", scope="global")
        assert result.outcome == "saved"
        assert result.hypomnema is not None
        assert result.bounce is None

    async def test_save_identical_already_saved(self) -> None:
        """Saving the exact same string twice acks the second as already_saved
        (P7 admission-order piece, step 1 — no bounce, no reinforcement event)."""
        engine = await _engine()
        text = "The capital of France is Paris."
        r1 = await engine.save(text, scope="global")
        assert r1.outcome == "saved"

        r2 = await engine.save(text, scope="global")
        assert r2.outcome == "already_saved"
        assert r2.bounce is None
        assert r2.hypomnema is not None
        assert r2.hypomnema.id == r1.hypomnema.id  # type: ignore[union-attr]

    async def test_exact_floor_bounces(self) -> None:
        """A cosine exactly at gray_floor must bounce (boundary inclusive).

        Two DIFFERENT-key texts (P7 admission-order piece): the exact-key
        step only fires on a matching canonical key, so a genuine near-dup
        via a forced SHA-256 slot collision reaches the probe instead —
        exactly what this boundary test needs.
        """
        cfg = _make_config(gray_floor=0.0)
        engine = await _engine(config=cfg)
        embedder = FakeEmbedder(dim=_DIM)
        text = "Test sentence."
        variant = _sha256_collision_variant(embedder, text)
        r1 = await engine.save(text, scope="global")
        assert r1.outcome == "saved"
        r2 = await engine.save(variant, scope="global")
        assert r2.outcome == "bounced"

    async def test_dissimilar_texts_both_saved(self) -> None:
        """Two orthogonal vectors (different hash positions) should both pass.

        We use a custom embedder that guarantees orthogonality by returning
        unit vectors at distinct fixed positions (no hash randomization).
        """
        dim = 8
        adapter = _make_adapter(dim=dim)
        clock = _make_clock()
        cfg = _make_config()

        # Produce orthogonal unit vectors at known positions
        class _OrthogonalEmbedder:
            dim: int = 8
            model: str = "orth"
            _call_count: int = 0
            _positions: tuple[int, ...] = (0, 4)  # guaranteed different positions

            async def embed(self, text: str) -> list[float]:
                pos = self._positions[self._call_count % len(self._positions)]
                self._call_count += 1
                v = [0.0] * self.dim
                v[pos] = 1.0
                return v

        oe = _OrthogonalEmbedder()
        engine = Tabularium(adapter, oe, clock, cfg)
        await engine.initialize()

        r1 = await engine.save("first text", scope="global")
        r2 = await engine.save("second text", scope="global")
        assert r1.outcome == "saved"
        assert r2.outcome == "saved"


# ---------------------------------------------------------------------------
# C4: the auto-reinforce band is gone — a probe survivor never auto-merges
# ---------------------------------------------------------------------------


class TestNoAutoReinforceBand:
    async def test_near_dup_always_bounces(self) -> None:
        """Even a near-perfect duplicate bounces, never auto-reinforces —
        the band is deleted.

        A genuine near-dup (different canonical key, forced SHA-256 slot
        collision) — P7's exact-key step only fires on a matching key, so
        this reaches the probe, which is what this test means to pin.
        """
        cfg = _make_config(gray_floor=0.80)
        engine = await _engine(config=cfg)
        embedder = FakeEmbedder(dim=_DIM)
        text = "The sky is blue."
        variant = _sha256_collision_variant(embedder, text)
        await engine.save(text, scope="global")
        r = await engine.save(variant, scope="global")
        assert r.outcome == "bounced"
        # No path to reinforced here — outcome is always bounced, not a write variant
        assert r.hypomnema is None


# ---------------------------------------------------------------------------
# C4: Guard-flag annotation on near-dup with differing tokens
# ---------------------------------------------------------------------------


class TestGuardFlags:
    """Guard checks operate on raw string content, not vectors."""

    def test_numeral_guard_trips_on_different_numbers(self) -> None:
        assert guard_check("We have 5 cats.", "We have 3 cats.") == "numeral"

    def test_numeral_guard_passes_on_same_numbers(self) -> None:
        assert guard_check("We have 5 cats.", "We have 5 cats.") is None

    def test_version_guard_trips(self) -> None:
        # To isolate the version guard we need numerals to be identical but version strings
        # to differ. Example: "v10.2.3" vs "v20.2.3".
        # _RE_NUMERALS (\b\d+(?:\.\d+)?\b) extracts "2.3" from both (the leading "v10."
        # or "v20." has no word-boundary before the digit after the first dot, but ".2"
        # has one before "2" -> extracts "2.3").  So numeral sets match; version sets differ.
        assert guard_check(
            "Upgrade the dependency to v10.2.3",
            "Upgrade the dependency to v20.2.3",
        ) == "version"

    def test_version_guard_passes_same_version(self) -> None:
        assert guard_check("Use version 3.1.4", "Use version 3.1.4") is None

    def test_negation_guard_trips(self) -> None:
        # Use all-lowercase strings to avoid triggering the entity guard first.
        a = "please do not run the process."
        b = "please run the process."
        assert guard_check(a, b) == "negation"

    def test_negation_guard_passes_both_negate(self) -> None:
        assert guard_check("do not run.", "do not run.") is None

    def test_entity_guard_trips(self) -> None:
        assert guard_check("Alice owns the car.", "Bob owns the car.") == "entity"

    def test_entity_guard_passes_same_entities(self) -> None:
        assert guard_check("Alice owns the car.", "Alice owns the car.") is None

    async def test_guard_annotates_bounce(self) -> None:
        """A near-dup with a numeral difference annotates guard_flag on the bounce."""
        # Use dim=4 to maximise hash collision probability (force same vector for similar strings)
        dim = 4
        cfg = _make_config(gray_floor=0.0)  # floor=0.0 -> any cosine bounces
        adapter = _make_adapter(dim=dim)
        embedder = FakeEmbedder(dim=dim)

        # Find two strings that produce identical vectors (same hash position)
        base = "We have 5 cats"
        variant = "We have 3 cats"
        v_base = embedder.peek(base)
        v_variant = embedder.peek(variant)

        # Skip if the vectors would not trigger a bounce at gray_floor=0.0.
        # The engine's best_cosine starts at 0.0 and only updates on strict '>',
        # so cosine must be > 0.0 to register a candidate and reach the guard path.
        # With single-hot sha256 vectors this is equivalent to checking for a slot
        # collision (cosine=1.0 on collision, 0.0 otherwise).
        cosine = sum(a * b for a, b in zip(v_base, v_variant, strict=False))
        if cosine <= 0.0:
            # Vectors are non-overlapping: gate will pass the save; cannot exercise guard
            pytest.skip("vector cosine ≤ 0.0 (no slot collision); cannot exercise guard via cosine")

        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        await engine.save(base, scope="global")
        r = await engine.save(variant, scope="global")
        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.guard_flag == "numeral"

    async def test_guard_annotates_negation_bounce(self) -> None:
        """Guard annotates a negation difference on the bounce.

        Uses a fixed stub embedder so the vectors always collide (gray_floor=0.0
        means any similarity bounces), and all-lowercase text so the entity guard
        does not fire before the negation guard.
        """
        dim = 8

        class _IdenticalEmbedder:
            """Always returns the same vector so both texts collide."""
            dim: int = 8
            model: str = "identical"

            async def embed(self, text: str) -> list[float]:
                del text
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        cfg = _make_config(gray_floor=0.0)
        adapter = _make_adapter(dim=dim)
        clock = _make_clock()
        engine = Tabularium(adapter, _IdenticalEmbedder(), clock, cfg)
        await engine.initialize()

        # All-lowercase so entity guard does not trip before negation guard
        base = "please do not run the process."
        variant = "please run the process."

        await engine.save(base, scope="global")
        r = await engine.save(variant, scope="global")
        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.guard_flag == "negation"


# ---------------------------------------------------------------------------
# C4: Bounce carries candidate_scope across X<->global tier (A5)
# ---------------------------------------------------------------------------


class TestBounceScope:
    async def test_bounce_candidate_scope_reflects_global(self) -> None:
        """Save a record into global; try to save a near-dup into a named scope.
        The bounce must carry candidate_scope='global'.

        A genuine near-dup, not an exact-key restatement: an exact-key
        match saved downward (project hitting a global row, P7 admission-
        order piece) reinforces silently rather than bouncing, so this
        property needs a probe-level bounce instead.
        """
        cfg = _make_config(gray_floor=0.0)
        adapter = _make_adapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()
        await adapter.create_scope(Scope("myproject"))

        text = "Shared knowledge."
        variant = _sha256_collision_variant(embedder, text)
        r1 = await engine.save(text, scope="global")
        assert r1.outcome == "saved"

        r2 = await engine.save(variant, scope="myproject")
        assert r2.outcome == "bounced"
        assert r2.bounce is not None
        assert r2.bounce.candidate_scope == "global"  # origin scope of the candidate

    async def test_bounce_candidate_scope_reflects_named_scope(self) -> None:
        """Save a record into a named scope; try to save a near-dup into the
        same scope. A byte-identical restatement is an already_saved ack (P7
        admission-order piece, step 1), not a bounce, so this needs a
        genuine near-dup to still exercise the bounce path.
        """
        cfg = _make_config(gray_floor=0.0)
        adapter = _make_adapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()
        await adapter.create_scope(Scope("myproject"))

        text = "Project-specific knowledge."
        variant = _sha256_collision_variant(embedder, text)
        await engine.save(text, scope="myproject")
        r = await engine.save(variant, scope="myproject")
        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.candidate_scope == "myproject"


# ---------------------------------------------------------------------------
# C4: Embedder outage lands a vector-less row with embedded=False
# ---------------------------------------------------------------------------


class TestVectorlessWrite:
    async def test_embedder_unavailable_lands_saved_with_embedded_false(self) -> None:
        engine = await _engine(should_fail=True)
        result = await engine.save("Remember this.", scope="global")
        assert result.outcome == "saved"
        assert result.embedded is False
        assert result.bounce is None
        assert result.hypomnema is not None

    async def test_vectorless_row_carries_null_vector_and_stamp(self) -> None:
        adapter = _make_adapter()
        embedder = FakeEmbedder(dim=_DIM, should_fail=True)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, _make_config())
        await engine.initialize()

        result = await engine.save("test", scope="global")

        assert result.hypomnema is not None
        record = await adapter.get_by_id(result.hypomnema.id)
        assert record.content == "test"
        assert record.embedding_model is None

    async def test_byte_exact_restatement_under_outage_is_already_saved(self) -> None:
        """Admission steps 1-3 (P7 admission-order piece) run inside the
        FIRST lock hold, before the embedder — a byte-identical restatement
        is caught there regardless of whether the embedder is up, so the
        outage never even reaches the vector-less write path."""
        engine = await _engine(should_fail=True)
        r1 = await engine.save("the exact same content twice", scope="global")
        r2 = await engine.save("the exact same content twice", scope="global")
        assert r1.outcome == "saved"
        assert r2.outcome == "already_saved"
        assert r1.hypomnema is not None
        assert r2.hypomnema is not None
        assert r2.hypomnema.id == r1.hypomnema.id

    async def test_vectorless_row_invisible_to_vector_search(self) -> None:
        adapter = _make_adapter()
        embedder = FakeEmbedder(dim=_DIM, should_fail=True)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, _make_config())
        await engine.initialize()

        await engine.save("test", scope="global")

        hits = await adapter.vector_search(
            [0.0] * _DIM,
            filters=Filters(
                scopes=[Scope("global")],
                validity=frozenset({"valid"}),
                lifecycle_in=frozenset({"active"}),
            ),
            k=10,
        )
        assert hits == []


# ---------------------------------------------------------------------------
# C4: Recall RRF math on constructed fixture
# ---------------------------------------------------------------------------


class TestRecallRRF:
    """
    Build a fixture with 3 records where we control per-leg ranks explicitly,
    then assert the fused order matches hand-computed RRF.

    RRF(k=60): score = 1/(60+rank_v) + 1/(60+rank_t)

    We control ranks by directly constructing vectors and text content so that
    both legs agree on their ranking (A > B > C on both), which gives:
      RRF(A) = 1/61 + 1/61 = 0.03279
      RRF(B) = 1/62 + 1/62 = 0.03226
      RRF(C) = 1/63 + 1/63 = 0.03175

    Expected order: A > B > C.
    With beta=0.0 importance boost is disabled so this is a pure RRF test.
    """

    async def test_rrf_fusion_order(self) -> None:
        import math as _math

        cfg = _make_config(beta=0.0)
        dim = 8
        adapter = MemoryAdapter(embedding_model="fake", dim=dim)
        await adapter.create_scope(Scope("global"))
        # Use FakeEmbedder for query only; records use hand-crafted vectors
        embedder = FakeEmbedder(dim=dim)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        now = clock.now()

        # Query vector: [1, 0, 0, ...]
        # Record vectors at decreasing cosine similarity to query:
        #   A: angle=0    -> cosine 1.0  (vector rank 1)
        #   B: angle=30deg -> cosine 0.866 (vector rank 2)
        #   C: angle=60deg -> cosine 0.5   (vector rank 3)
        vec_a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        vec_b = [_math.cos(_math.radians(30)), _math.sin(_math.radians(30)),
                 0, 0, 0, 0, 0, 0]
        vec_c = [_math.cos(_math.radians(60)), _math.sin(_math.radians(60)),
                 0, 0, 0, 0, 0, 0]

        # Text content designed so BM25 ranks A > B > C for query "unique common rare":
        # A has all 3 query terms -> BM25 rank 1
        # B has 2 query terms -> BM25 rank 2
        # C has 1 query term -> BM25 rank 3
        query_text = "unique common rare"

        id_a, id_b, id_c = new_id(), new_id(), new_id()

        def _fact(fid: HypomnemaId, content: str) -> Fact:
            return Fact(
                id=fid,
                type="fact",
                content=content,
                scope=Scope("global"),
                lifecycle="active",
                created_at=now,
                updated_at=now,
            )

        fact_a = _fact(id_a, "unique common rare document")
        fact_b = _fact(id_b, "unique common document")
        fact_c = _fact(id_c, "unique document")

        await adapter.upsert(fact_a, vec_a)
        await adapter.upsert(fact_b, vec_b)
        await adapter.upsert(fact_c, vec_c)

        # Override the embedder with a stub that always returns the query vector [1,0,...].
        # FakeEmbedder uses hash(text) % dim for position, which we can't control.
        _query_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        class _FixedEmbedder:
            dim: int = 8
            model: str = "fixed"

            async def embed(self, text: str) -> list[float]:
                del text
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        engine._embedder = _FixedEmbedder()

        result = await engine.recall(query_text, scope="global")
        assert result.degraded is False
        assert len(result.hits) == 3

        # Both legs rank A>B>C so RRF order is A>B>C
        ids = [h.id for h in result.hits]
        assert ids[0] == id_a, f"Expected A first, got order {ids}"
        assert ids[1] == id_b, f"Expected B second, got order {ids}"
        assert ids[2] == id_c, f"Expected C third, got order {ids}"

        # Verify the actual scores match hand-computed RRF
        scores = {h.id: h.score for h in result.hits}
        expected_a = 1 / 61 + 1 / 61
        expected_b = 1 / 62 + 1 / 62
        expected_c = 1 / 63 + 1 / 63
        assert math.isclose(scores[id_a], expected_a, rel_tol=1e-6), (
            f"A score {scores[id_a]:.6f} != expected {expected_a:.6f}"
        )
        assert math.isclose(scores[id_b], expected_b, rel_tol=1e-6)
        assert math.isclose(scores[id_c], expected_c, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# C4: Boost changes order (beta>0 with unequal recall_counts)
# ---------------------------------------------------------------------------


class TestImportanceBoost:
    async def test_high_recall_count_bumps_rank(self) -> None:
        """A record with high recall_count should outscore a semantically-equal
        record with recall_count=0 when beta > 0."""
        cfg = _make_config(beta=0.5)
        dim = 8
        adapter = MemoryAdapter(embedding_model="fake", dim=dim)
        await adapter.create_scope(Scope("global"))
        embedder = FakeEmbedder(dim=dim)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        now = clock.now()
        import math as _math

        # Two records with identical cosine similarity to the query
        # query vector = [1, 0, ...]
        # Both records at the same angle
        angle = _math.acos(0.8)
        vec = [_math.cos(angle), _math.sin(angle), 0, 0, 0, 0, 0, 0]

        id_hot = new_id()
        id_cold = new_id()

        fact_hot = Fact(
            id=id_hot,
            type="fact",
            content="frequently recalled content",
            scope=Scope("global"),
            lifecycle="active",
            created_at=now,
            updated_at=now,
        )
        fact_cold = Fact(
            id=id_cold,
            type="fact",
            content="rarely recalled content",
            scope=Scope("global"),
            lifecycle="active",
            created_at=now,
            updated_at=now,
        )

        await adapter.upsert(fact_hot, vec)
        await adapter.upsert(fact_cold, vec)

        # Boost recall_count on hot via update_projections
        await adapter.update_projections(
            id_hot,
            recall_count=100,
            last_referenced_at=now,
            feedback_score=0.0,
            decay_due_at=None,
        )

        # query with matching text so both legs fire
        result = await engine.recall("frequently recalled content", scope="global")
        ids = [h.id for h in result.hits]
        assert len(ids) >= 2
        assert ids[0] == id_hot, f"High recall_count record should rank first; got {ids}"


# ---------------------------------------------------------------------------
# C4: Degraded recall = text-leg-only with flags
# ---------------------------------------------------------------------------


class TestDegradedRecall:
    async def test_degraded_when_embedder_fails(self) -> None:
        engine = await _engine()

        # Save a record while embedder works
        await engine.save("The quick brown fox.", scope="global")

        # Break the embedder
        engine._embedder = FakeEmbedder(dim=_DIM, should_fail=True)
        result = await engine.recall("quick brown fox", scope="global")

        assert result.degraded is True
        assert result.reason is not None
        assert result.legs_used == ["text"]

    async def test_degraded_still_returns_bm25_hits(self) -> None:
        engine = await _engine()
        await engine.save("The quick brown fox jumped.", scope="global")

        engine._embedder = FakeEmbedder(dim=_DIM, should_fail=True)
        result = await engine.recall("quick brown fox", scope="global")

        assert result.degraded is True
        assert len(result.hits) >= 1
        assert result.hits[0].content == "The quick brown fox jumped."

    async def test_degraded_recall_finds_a_row_saved_under_the_same_outage(self) -> None:
        """A row that landed vector-less (P6 vector-less-rows piece, D4) is
        still readable through Recall's text leg while the outage lasts."""
        engine = await _engine(should_fail=True)
        saved = await engine.save("The quick brown fox jumped over.", scope="global")
        assert saved.embedded is False

        result = await engine.recall("quick brown fox", scope="global")

        assert result.degraded is True
        assert result.legs_used == ["text"]
        assert any(h.content == "The quick brown fox jumped over." for h in result.hits)


# ---------------------------------------------------------------------------
# C4: Scope-span isolation
# ---------------------------------------------------------------------------


class TestScopeSpan:
    async def test_global_not_visible_to_unrelated_scope(self) -> None:
        """Records in 'other' scope should not appear in recall for 'myproject'."""
        cfg = _make_config()
        adapter = _make_adapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()
        await adapter.create_scope(Scope("myproject"))
        await adapter.create_scope(Scope("other"))

        # Save only into 'other'
        await engine.save("secret knowledge", scope="other")

        # Recall in 'myproject' should NOT find it (other is not in [myproject, global])
        result = await engine.recall("secret knowledge", scope="myproject")
        assert all(h.scope != "other" for h in result.hits)
        other_hits = [h for h in result.hits if h.scope == "other"]
        assert other_hits == []

    async def test_global_co_read_in_named_scope_recall(self) -> None:
        """Records saved in 'global' ARE visible when recalling in 'myproject'."""
        cfg = _make_config()
        adapter = _make_adapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()
        await adapter.create_scope(Scope("myproject"))

        await engine.save("global insight", scope="global")
        result = await engine.recall("global insight", scope="myproject")
        global_hits = [h for h in result.hits if h.scope == "global"]
        assert len(global_hits) >= 1

    async def test_global_reads_only_global(self) -> None:
        """Recall on 'global' should ONLY see global-scoped records."""
        cfg = _make_config()
        adapter = _make_adapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()
        await adapter.create_scope(Scope("myproject"))

        # Save into both scopes with different text
        await engine.save("global content only", scope="global")
        await engine.save("project content only", scope="myproject")

        result = await engine.recall("content", scope="global")
        scopes_found = {h.scope for h in result.hits}
        assert "myproject" not in scopes_found

    async def test_unregistered_scope_raises(self) -> None:
        engine = await _engine()
        with pytest.raises(ToolError, match="Unregistered scope"):
            await engine.save("test", scope="nonexistent")

    async def test_unregistered_scope_recall_raises(self) -> None:
        engine = await _engine()
        with pytest.raises(ToolError, match="Unregistered scope"):
            await engine.recall("test", scope="nonexistent")


# ---------------------------------------------------------------------------
# C4: Fallback widen-only when primary is thin
# ---------------------------------------------------------------------------


class TestFallbackWiden:
    async def test_widen_to_archived_when_primary_thin(self) -> None:
        """When active results < min_active_results, widen to archived."""
        cfg = _make_config(min_active_results=2)
        adapter = _make_adapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        now = clock.now()

        # Upsert an archived Fact directly (engine save always creates active)
        archived_fact = Fact(
            id=new_id(),
            type="fact",
            content="archived useful knowledge",
            scope=Scope("global"),
            lifecycle="archived",
            created_at=now,
            updated_at=now,
        )
        vec = embedder.peek("archived useful knowledge")
        await adapter.upsert(archived_fact, vec)

        # Only this archived record exists; primary pass (active|hot|ember) finds 0
        # which is < min_active_results=2, so we widen.  The fallback revival path
        # transitions the record to active (revival_threshold satisfied with fresh recency),
        # so the hit is returned with lifecycle="active" after refresh.
        result = await engine.recall("archived useful knowledge", scope="global")
        assert result.hits, "fallback widen should find the archived record"
        assert result.hits[0].content == "archived useful knowledge"

    async def test_widen_does_not_evict_active_hits(self) -> None:
        """The fallback widen supplements active hits — it must not replace them.

        Scenario: 1 active record + 1 archived record; min_active_results=2 triggers widen.
        The active record must appear in the final result alongside the archived one.
        """
        cfg = _make_config(min_active_results=2)
        adapter = _make_adapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        now = clock.now()

        active_id = new_id()
        active_fact = Fact(
            id=active_id,
            type="fact",
            content="active knowledge here",
            scope=Scope("global"),
            lifecycle="active",
            created_at=now,
            updated_at=now,
        )
        archived_fact = Fact(
            id=new_id(),
            type="fact",
            content="archived useful knowledge",
            scope=Scope("global"),
            lifecycle="archived",
            created_at=now,
            updated_at=now,
        )

        active_vec = embedder.peek("active knowledge here")
        archived_vec = embedder.peek("archived useful knowledge")
        await adapter.upsert(active_fact, active_vec)
        await adapter.upsert(archived_fact, archived_vec)

        # Primary (active|hot|ember) finds 1 record < min_active_results=2 -> widen
        result = await engine.recall("knowledge", scope="global")
        result_ids = {h.id for h in result.hits}

        # Both active and archived should appear (merge, not replace)
        assert active_id in result_ids, "Active record must not be evicted by fallback widen"

    async def test_widen_with_bounded_candidate_k(self) -> None:
        """Regression: with candidate_k=1 and min_active_results=2, the fallback must
        still surface an archived record alongside the active one.

        Uses hand-crafted vectors and a fixed query-embedder stub so the result is
        deterministic regardless of PYTHONHASHSEED.

        Vector layout (dim=8):
          query   = [1, 0, 0, 0, 0, 0, 0, 0]  (position 0)
          active  = [1, 0, 0, 0, 0, 0, 0, 0]  (position 0 -> cosine=1.0 with query)
          archived= [0, 1, 0, 0, 0, 0, 0, 0]  (position 1 -> cosine=0.0 with query)

        Under a widened active+archived fallback with k=1:
          vector_search returns only the active record (cos=1.0 wins the k=1 slot)
          -> archived record is never returned -> bug.

        Under the archived-only fallback:
          vector_search scans only archived records -> archived record returned.
          Primary hits (active) are preserved, merged -> both appear.
        """
        cfg = _make_config(min_active_results=2, candidate_k=1, beta=0.0)
        dim = 8
        adapter = MemoryAdapter(embedding_model="fixed", dim=dim)
        clock = _make_clock()

        # Stub embedder: always returns [1, 0, ...] (same as active record vector)
        class _FixedEmbedder2:
            dim: int = 8
            model: str = "fixed"

            async def embed(self, text: str) -> list[float]:
                del text
                return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        engine = Tabularium(adapter, _FixedEmbedder2(), clock, cfg)
        await engine.initialize()

        now = clock.now()
        active_id = new_id()
        archived_id = new_id()

        # Active: at position 0 (cosine=1.0 with query -> wins k=1 in vector leg)
        vec_active = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Archived: at position 1 (cosine=0.0 with query -> loses to active in k=1 slot)
        vec_archived = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        active_fact = Fact(
            id=active_id,
            type="fact",
            content="active record",
            scope=Scope("global"),
            lifecycle="active",
            created_at=now,
            updated_at=now,
        )
        archived_fact = Fact(
            id=archived_id,
            type="fact",
            content="archived record",
            scope=Scope("global"),
            lifecycle="archived",
            created_at=now,
            updated_at=now,
        )

        await adapter.upsert(active_fact, vec_active)
        await adapter.upsert(archived_fact, vec_archived)

        # Confirm: widened k=1 vector search would miss archived (both active+archived
        # are candidates; active cosine=1.0 > archived cosine=0.0 -> k=1 returns only active).
        # This proves the old bug would fail here.
        from tabularium.storage.types import Filters as _F
        widened_hits = await adapter.vector_search(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            filters=_F(
                scopes=[Scope("global")],
                validity=frozenset({"valid"}),
                lifecycle_in=frozenset({"active", "archived"}),
            ),
            k=1,
        )
        assert len(widened_hits) == 1
        assert widened_hits[0].record.id == active_id, (
            "Sanity: widened k=1 returns only active (archived would be missed by old bug)"
        )

        # Now confirm the engine returns BOTH with the archived-only fix:
        result = await engine.recall("active archived", scope="global")
        result_ids = {h.id for h in result.hits}
        assert active_id in result_ids, "Active record must be present"
        assert archived_id in result_ids, (
            "Archived record must be surfaced via archived-only fallback; "
            "would be missed if fallback used widened active+archived filter with k=1"
        )

    async def test_no_widen_when_primary_sufficient(self) -> None:
        """When active results >= min_active_results, archived records are excluded."""
        cfg = _make_config(min_active_results=1)
        adapter = _make_adapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        now = clock.now()

        # Save an active record via engine
        await engine.save("active knowledge here", scope="global")

        # Also directly upsert an archived record
        archived = Fact(
            id=new_id(),
            type="fact",
            content="archived secret",
            scope=Scope("global"),
            lifecycle="archived",
            created_at=now,
            updated_at=now,
        )
        archived_vec = embedder.peek("archived secret")
        await adapter.upsert(archived, archived_vec)

        # Primary finds 1 active record which is >= min_active_results=1 -> no widen
        result = await engine.recall("knowledge", scope="global")
        assert not any(h.lifecycle == "archived" for h in result.hits)


# ---------------------------------------------------------------------------
# candidate pool derives from the caller's limit (ADR 0018)
# ---------------------------------------------------------------------------


class _KSpyAdapter(MemoryAdapter):
    """MemoryAdapter subclass that records every ``k`` handed to the two
    retrieval legs -- proves the sized candidate pool reaches every storage
    call (primary pass AND fallback/archived-widen pass)."""

    def __init__(self, *, dim: int = _DIM) -> None:
        super().__init__(embedding_model="fake/nomic", dim=dim)
        self.vector_search_ks: list[int] = []
        self.text_search_ks: list[int] = []

    async def vector_search(
        self, vector: list[float], *, filters: Filters, k: int
    ) -> Any:
        self.vector_search_ks.append(k)
        return await super().vector_search(vector, filters=filters, k=k)

    async def text_search(
        self, query: str, *, filters: Filters, k: int
    ) -> Any:
        self.text_search_ks.append(k)
        return await super().text_search(query, filters=filters, k=k)


class TestCandidatePoolFromLimit:
    async def test_limit_sizes_candidate_pool(self) -> None:
        """recall(limit=200) must issue vector_search/text_search with k=200 --
        on BOTH the primary pass and the fallback/archived-widen pass.

        With an empty store, the primary pass surfaces 0 active records, which
        is < the default min_active_results(3), so the fallback pass fires
        unconditionally too -- this single call exercises both passes.
        """
        cfg = _make_config(candidate_k=50)
        adapter = _KSpyAdapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        await engine.recall("anything", scope="global", limit=200)

        assert adapter.vector_search_ks, "vector_search was never called"
        assert adapter.text_search_ks, "text_search was never called"
        assert all(k == 200 for k in adapter.vector_search_ks), (
            f"expected every vector_search k==200, got {adapter.vector_search_ks}"
        )
        assert all(k == 200 for k in adapter.text_search_ks), (
            f"expected every text_search k==200, got {adapter.text_search_ks}"
        )
        # Both passes must have fired (primary + fallback widen).
        assert len(adapter.vector_search_ks) == 2, (
            f"expected 2 vector_search calls (primary + fallback), "
            f"got {len(adapter.vector_search_ks)}"
        )
        assert len(adapter.text_search_ks) == 2, (
            f"expected 2 text_search calls (primary + fallback), "
            f"got {len(adapter.text_search_ks)}"
        )

    async def test_small_limit_keeps_config_floor(self) -> None:
        """recall(limit=5) must still fetch cfg.candidate_k (50) per leg --
        the config value is a floor, so the fused pool is never shrunk below
        today's behaviour."""
        cfg = _make_config(candidate_k=50)
        adapter = _KSpyAdapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        await engine.recall("anything", scope="global", limit=5)

        assert adapter.vector_search_ks, "vector_search was never called"
        assert adapter.text_search_ks, "text_search was never called"
        assert all(k == 50 for k in adapter.vector_search_ks), (
            f"expected every vector_search k==50, got {adapter.vector_search_ks}"
        )
        assert all(k == 50 for k in adapter.text_search_ks), (
            f"expected every text_search k==50, got {adapter.text_search_ks}"
        )

    async def test_wide_sweep_returns_more_than_the_old_cap(self) -> None:
        """Behavioural proof (not merely a call-signature check): with >=120
        matching active records, recall(limit=120) must return all 120 hits.

        Under the old fixed candidate_k=50-per-leg sizing, the fused pool
        (union of two 50-record legs) could not supply 120 results no matter
        what limit the caller asked for -- this is the test that would have
        caught the original defect.
        """
        cfg = _make_config(candidate_k=50, beta=0.0)
        dim = _DIM
        adapter = _make_adapter(dim=dim)
        embedder = FakeEmbedder(dim=dim)
        clock = _make_clock()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        now = clock.now()
        n = 120
        for i in range(n):
            content = f"widget entry number {i}"
            vec = embedder.peek(content)
            fact = Fact(
                id=new_id(),
                type="fact",
                content=content,
                scope=Scope("global"),
                lifecycle="active",
                created_at=now,
                updated_at=now,
            )
            await adapter.upsert(fact, vec)

        result = await engine.recall("widget", scope="global", limit=n)
        assert len(result.hits) == n, (
            f"expected {n} hits at limit={n}, got {len(result.hits)} "
            f"(fused pool starved by a fixed per-leg cap)"
        )


# ---------------------------------------------------------------------------
# C4: Purpose raises in M0
# ---------------------------------------------------------------------------


class TestEmptyQueryGate:
    async def test_empty_query_raises(self) -> None:
        engine = await _engine()
        with pytest.raises(ValueError, match="non-empty"):
            await engine.recall("", scope="global")

    async def test_whitespace_query_raises(self) -> None:
        engine = await _engine()
        with pytest.raises(ValueError, match="non-empty"):
            await engine.recall("   ", scope="global")


class TestPurposeValidation:
    async def test_save_with_registered_purpose_succeeds(self) -> None:
        """Registered purpose values (pre-seeded by initialize) are accepted."""
        engine = await _engine()
        # "constraint" is one of the five pre-registered purpose values
        r = await engine.save("test", scope="global", purpose="constraint")
        assert r.outcome == "saved"

    async def test_save_with_unregistered_purpose_raises(self) -> None:
        """Unregistered purpose value raises ToolError (SPEC §10.2 / §9.2)."""
        engine = await _engine()
        with pytest.raises(ToolError, match="Unregistered purpose"):
            await engine.save("test", scope="global", purpose="not-a-real-purpose")


# ---------------------------------------------------------------------------
# C4: SaveResult / RecalledHit field correctness
# ---------------------------------------------------------------------------


class TestResultFields:
    async def test_saved_result_has_hypomnema_ref(self) -> None:
        engine = await _engine()
        r = await engine.save("fact", scope="global")
        assert r.outcome == "saved"
        assert r.hypomnema is not None
        assert r.hypomnema.kind == "fact"

    async def test_recalled_hit_fields_populated(self) -> None:
        engine = await _engine()
        await engine.save("recall me please", scope="global")
        result = await engine.recall("recall me please", scope="global")
        assert len(result.hits) >= 1
        hit = result.hits[0]
        assert hit.kind == "fact"
        assert hit.lifecycle == "active"
        assert hit.validity == "valid"
        assert hit.scope == "global"
        assert hit.purpose is None
        assert hit.score >= 0.0

    async def test_legs_used_normal_path(self) -> None:
        engine = await _engine()
        await engine.save("test sentence", scope="global")
        result = await engine.recall("test sentence", scope="global")
        assert "vector" in result.legs_used
        assert "text" in result.legs_used


# ---------------------------------------------------------------------------
# C4: RRF helper unit test
# ---------------------------------------------------------------------------


class TestRRFHelper:
    def test_rrf_score_formula(self) -> None:
        """1 / (k + rank), rank is 1-indexed."""
        assert math.isclose(_rrf_score(1, 60), 1 / 61)
        assert math.isclose(_rrf_score(2, 60), 1 / 62)
        assert math.isclose(_rrf_score(10, 60), 1 / 70)


# ---------------------------------------------------------------------------
# C4: Integration lane (requires Ollama + nomic-embed-text)
# ---------------------------------------------------------------------------


async def _ollama_available() -> bool:
    """Check if Ollama is up and has nomic-embed-text."""
    try:
        embedder = LitellmEmbedder(timeout=3.0, num_retries=0)
        await embedder.embed("ping")
        return True
    except EmbedderUnavailable:
        return False
    except Exception:
        return False


@pytest.mark.integration
async def test_integration_save_recall_semantic_ranking() -> None:
    """Live Ollama: save 4 sentences, recall with a paraphrase, assert correct ranking.

    Also verifies that a near-duplicate re-save fires a bounce.
    Skip cleanly if Ollama is unreachable.
    """
    if not await _ollama_available():
        pytest.skip("Ollama not reachable — skipping integration test")

    embedder = LitellmEmbedder()
    dim = 768
    adapter = MemoryAdapter(embedding_model="ollama/nomic-embed-text", dim=dim)
    clock = TestClock(datetime(2025, 1, 1, tzinfo=UTC))
    cfg = EngineConfig(gray_floor=0.80, candidate_k=50, beta=0.0)
    engine = Tabularium(adapter, embedder, clock, cfg)
    await engine.initialize()

    sentences = [
        "The mitochondria is the powerhouse of the cell.",
        "Python is a high-level programming language known for its readability.",
        "Machine learning models require large amounts of training data.",
        "The Eiffel Tower is located in Paris, France.",
    ]

    for s in sentences:
        r = await engine.save(s, scope="global")
        assert r.outcome == "saved", f"Unexpected outcome {r.outcome!r} for {s!r}"

    # Paraphrase query: semantically closest to sentence[1] (Python programming)
    query = "a popular language used in software development"
    result = await engine.recall(query, scope="global")

    assert not result.degraded
    assert len(result.hits) >= 1
    top_hit = result.hits[0]

    # The semantically closest result should be the Python sentence
    print(f"\n[integration] query={query!r}")
    for i, h in enumerate(result.hits):
        print(f"  rank {i+1}: score={h.score:.4f} content={h.content!r}")

    assert "Python" in top_hit.content or "programming" in top_hit.content, (
        f"Expected Python/programming sentence to rank first; got {top_hit.content!r}\n"
        f"All hits: {[(h.score, h.content) for h in result.hits]}"
    )

    # Byte-identical re-save is a step-1 replay, not a bounce.
    near_dup = sentences[1]
    r_dup = await engine.save(near_dup, scope="global")
    print(f"\n[integration] byte-identical re-save: outcome={r_dup.outcome!r}")
    assert r_dup.outcome == "already_saved", (
        f"Expected already_saved on re-save of identical sentence; got {r_dup.outcome!r}"
    )


# ---------------------------------------------------------------------------
# Maintenance policy wiring (SPEC S3.3, ADR 0017)
# ---------------------------------------------------------------------------


class _CountingAdapter(MemoryAdapter):
    """MemoryAdapter subclass that counts maintenance_debt() calls and can be
    scripted to report a chosen debt / raise from upkeep() -- used to prove
    save-path wiring and outcome integrity without mocking StorageAdapter."""

    def __init__(
        self,
        *,
        dim: int = _DIM,
        num_small_fragments: int = 0,
        upkeep_raises: Exception | None = None,
    ) -> None:
        super().__init__(embedding_model="fake/nomic", dim=dim)
        self._num_small_fragments = num_small_fragments
        self._upkeep_raises = upkeep_raises
        self.maintenance_debt_calls = 0
        self.upkeep_calls = 0

    async def maintenance_debt(self) -> Any:
        self.maintenance_debt_calls += 1
        from tabularium.storage.types import MaintenanceDebt

        return MaintenanceDebt(
            num_small_fragments=self._num_small_fragments,
            num_rows=len(self._rows),
            num_unindexed_rows=0,
        )

    async def upkeep(self) -> None:
        self.upkeep_calls += 1
        if self._upkeep_raises is not None:
            raise self._upkeep_raises


class TestMaintenancePolicyWiring:
    async def test_save_fact_invokes_maintenance_policy_once(self) -> None:
        adapter = _CountingAdapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        cfg = _make_config()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        result = await engine.save(
            "The Nile is the longest river in Africa.", scope="global"
        )

        assert result.outcome == "saved"
        assert adapter.maintenance_debt_calls == 1

    async def test_save_decision_invokes_maintenance_policy_once(self) -> None:
        adapter = _CountingAdapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        cfg = _make_config()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        result = await engine.save(
            "We standardize on uv for Python dependency management.",
            scope="global",
            kind="decision",
        )

        assert result.outcome == "saved"
        assert adapter.maintenance_debt_calls == 1

    async def test_ember_invokes_maintenance_policy_once(self) -> None:
        adapter = _CountingAdapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        cfg = _make_config()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        result = await engine.ember(
            "Always run tests before pushing.", scope="global", valence="guideline"
        )

        assert result.outcome == "saved"
        assert adapter.maintenance_debt_calls == 1

    async def test_upkeep_failure_does_not_change_save_outcome(self) -> None:
        """A failing upkeep() must not change save()'s returned outcome, and the
        record it committed must still be retrievable by id -- the record is
        already committed; a maintenance error is operational, not a write
        outcome (SPEC / ADR 0017)."""
        adapter = _CountingAdapter(
            num_small_fragments=999, upkeep_raises=RuntimeError("upkeep exploded")
        )
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        cfg = _make_config()  # default maintenance_threshold=20; 999 crosses it
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        result = await engine.save(
            "Distinct content for outcome integrity.", scope="global"
        )

        assert result.outcome == "saved"
        assert result.hypomnema is not None
        assert adapter.upkeep_calls == 1
        fetched = await adapter.get_by_id(result.hypomnema.id)
        assert fetched.content == "Distinct content for outcome integrity."

    async def test_exact_key_reinforce_invokes_maintenance_policy_once(self) -> None:
        """The exact content-key match (P7 admission-order piece, step 3)
        commits a reinforcement event -- also a durable write that must
        trigger the policy. A whitespace variant shares the first save's
        canonical key without being byte-identical, so it reaches step 3's
        reinforce rather than step 1's ``already_saved`` (which writes
        nothing and must not invoke maintenance at all)."""
        adapter = _CountingAdapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        cfg = _make_config()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        r1 = await engine.save("sky is blue", scope="global")
        assert r1.outcome == "saved"
        adapter.maintenance_debt_calls = 0  # isolate the reinforce call

        r2 = await engine.save("sky is blue ", scope="global")
        assert r2.outcome == "reinforced"
        assert adapter.maintenance_debt_calls == 1

    async def test_archived_revival_invokes_maintenance_policy_once(self) -> None:
        """Dedup-revival (archived Fact -> active transition + reinforce) is a
        durable write that must trigger the policy."""
        adapter = _CountingAdapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        cfg = _make_config(gray_floor=0.0)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        now = clock.now()
        vec = [1.0] + [0.0] * (_DIM - 1)
        archived = Fact(
            id=new_id(),
            type="fact",
            content="archived fact content",
            scope=Scope("global"),
            lifecycle="archived",
            created_at=now,
            updated_at=now,
        )
        await adapter.upsert(archived, vec)
        adapter.maintenance_debt_calls = 0  # isolate the revival call

        class _IdenticalEmbedder:
            async def embed(self, text: str) -> list[float]:
                return vec

        engine._embedder = _IdenticalEmbedder()  # type: ignore[assignment]
        # A different restatement — not a byte-identical or same-key match
        # (P7 admission-order piece, step 1/3 would otherwise intercept it
        # before the probe ever runs) — ``_IdenticalEmbedder`` still forces
        # the cosine collision this dedup-revival dispatch needs.
        r = await engine.save(
            "a differently worded restatement of the archived content",
            scope="global",
        )

        assert r.outcome == "revived"
        assert adapter.maintenance_debt_calls == 1

    async def test_ember_recurrence_invokes_maintenance_policy_once(self) -> None:
        """Ember recurrence auto-reinforce is a durable write that must trigger
        the policy."""
        adapter = _CountingAdapter()
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        cfg = _make_config(gray_floor=0.0)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        r1 = await engine.ember(
            "ember content here", scope="global", valence="guideline"
        )
        assert r1.outcome == "saved"
        adapter.maintenance_debt_calls = 0  # isolate the recurrence call

        r2 = await engine.save("ember content here", scope="global")
        assert r2.outcome == "reinforced"
        assert adapter.maintenance_debt_calls == 1
