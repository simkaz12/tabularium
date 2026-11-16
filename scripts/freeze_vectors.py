"""Freeze vector fixtures via live Ollama.

Run ONCE to generate ``tests/fixtures/frozen_vectors.json``.  Re-run only when
the embedder model changes.  The hermetic gate imports this fixture via
``FrozenEmbedder`` — it never re-embeds inside the gate.

Usage::

    uv run python scripts/freeze_vectors.py

Embeds every text in CORPUS_TEXTS + QUERY_TEXTS + GOLDEN_PAIR_TEXTS using
``ollama/nomic-embed-text`` at ``http://localhost:11434``.

Output format (deterministic JSON — sorted keys, fixed float repr)::

    {
      "embedder_model": "ollama/nomic-embed-text",
      "dim": 768,
      "vectors": {
        "<text>": [<float>, ...],
        ...
      }
    }

Gate contract: the fixture is a committed artifact.  Regeneration produces a
reviewable diff (sorted-key JSON, deterministic float repr).
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

import litellm

# Running this script directly (`uv run python scripts/freeze_vectors.py`, the documented
# usage above) puts only this file's own directory on sys.path[0], not the repo root -- the
# editable install adds src/ but never the repo root itself, so `tests.support.cases` is not
# importable without this. pytest never hits this gap (it puts the repo root on sys.path via
# tests/__init__.py), which is exactly why it would otherwise go unnoticed.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.support.cases import case_texts, load_cases  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "ollama/nomic-embed-text"
API_BASE = "http://localhost:11434"
DIM = 768
FIXTURE_PATH = pathlib.Path(__file__).parent.parent / "tests" / "fixtures" / "frozen_vectors.json"

# ---------------------------------------------------------------------------
# Recall mini-corpus (56 invented notes in six topical clusters)
#
# All text here is invented. Ground truth is hand-curated relevance over these texts.
# Six topic clusters:
#   A  reminder and digest service internals (a fictional todo app)
#   B  design decisions of a fictional search library ("Atlas")
#   C  backend codebase conventions (a fictional bookstore app)
#   D  testing patterns
#   E  repo, tooling and infrastructure notes
#   F  frontend pages and conventions
# ---------------------------------------------------------------------------

CORPUS_TEXTS: list[str] = [
    # Cluster A
    "The reminder table is keyed by (user_id, task_id, channel). channel is email, push or sms, and each key holds at most one pending reminder.",
    "Open issue: ReminderService.dueSoon() loads every pending reminder into memory and filters them in a loop. This is a known scaling bottleneck; the plan is to push the due-date filter into the database query.",
    "ReminderService.compose returns a typed DigestPayload with overdue, due-today and upcoming sections. The morning digest job calls this method to assemble the email body.",
    "mirrorFor tolerates a mail provider outage through a private safeSend() helper that catches the exception, logs a warning and returns null. The mirror row is stored with no delivery id and skipped by the retry job until it is backfilled.",
    "Mirror rows (source=calendar) are written by CalendarEventObserver and TaskNoteObserver, which hook model events (created, updated, deleted) and call ReminderService.mirrorFor synchronously.",
    "DigestRunner replaces the previous digest_summary row on every run (delete, then insert) so each user has at most one active digest.",
    "ReminderService.dueSoon() compares due dates in application code over every pending reminder. It is a known bottleneck; the eventual plan is to delegate the comparison to a database index.",
    "Digest payload tiers: pinned reminders are always included regardless of the query, the due tier is ranked by due date, and the summary tier is the digest row.",
    "The reminder table uses user_id plus workspace_id to namespace every row. Cross-workspace leakage is prevented by an explicit workspace filter on every query.",
    "ReminderEntry has a delivery_token column holding a 36-character string. It is null when the mail provider was unreachable at write time.",
    "DigestPayload packs pinned and due reminders into a character budget. The counter uses a simple word-count approximation.",
    "The reminders command line has reminder_list and reminder_add commands. reminder_digest prints the morning digest for a user.",
    # Cluster B
    "Atlas decision: reinforcement recording appends a ReinforcementEvent to an append-only event log. Counters such as hit_count and last_hit_at are recomputed from the log.",
    "Atlas decision: Atlas models no tenants. One deployment is one index directory for one team, with no user table, tenant column or principal argument anywhere.",
    "Atlas decision: the primary store is an embedded columnar database with native hybrid search (vector, full text and rank fusion), so no separate service is required.",
    "Atlas decision: the importance score is never stored. It is computed lazily from the event log (hit_count, feedback_score, last_hit_at). No importance column exists on any row.",
    "Atlas spec: purge removes the content, vector and text-index entry and writes a tombstone (id, kind, reason, timestamp, generated_by_model). There is no hard delete; purging is explicit and human-decided.",
    "Atlas decision: DocId is a UUID newtype that works with any storage adapter and is never an auto-increment integer. Document = Note | Ruling is a union discriminated by the kind field.",
    "Atlas decision: search runs two legs, vector_search and text_search, in parallel and fuses them above the storage seam with reciprocal rank fusion. It is not a two-stage narrowing pipeline.",
    "Atlas milestone one: add, search, ruling and rulings_query tools over the server, plus an export command that writes rulings as markdown files. No feedback, no digest and no sweeps yet.",
    "Atlas duplicate gate: strict mode bounces when in doubt. The gray floor and auto ceiling are per-embedder calibration seeds, not production claims. Guards: numeral, version, entity, negation.",
    "Atlas projection determinism: computing projections twice from the same event log and clock yields identical values. A property test gates this as a release blocker.",
    "Atlas namespace pattern: a global namespace plus flat named peer namespaces. A read in namespace X sees X plus global. Two tiers only, no nesting, no paths.",
    "Atlas rulings arm: Rulings bounce eagerly at a lower gray floor than Notes. Auto-reinforce of a Ruling is always forbidden; every Ruling match needs a human decision.",
    # Cluster C
    "House convention: console commands live in app/Modules/<Module>/Commands/ rather than Console/ and are registered in the module's CommandServiceProvider.",
    "Subscription has no is_active column. SubscriptionTransformer computes is_active dynamically by comparing the plan's date range against today.",
    "The OrderPlaced event fires two synchronous listeners in order: (1) OrderTotalsListener recalculates line totals and tax; (2) OrderIndexListener queues a search-index update.",
    "DiscountList and DiscountView endpoints now include ['rules'] by default. Callers that do not want the rules must pass includes([]) explicitly.",
    "Diagnostic recipe for 'is a listener actually firing?': dispatch the event from tinker, check the queue, or add a temporary Log::info in the listener's handle method.",
    "Queries that branch on the database driver must keep PostgreSQL production semantics under the SQLite test database: guard them with DB::connection()->getDriverName() === 'pgsql'.",
    "Module layout: each module has Controllers/, Models/, Transformers/, Requests/, Observers/ and Commands/ subdirectories under app/Modules/<Module>/.",
    "ShipmentData gained a public ?string $carrierCode (null by default). It affects the carrier_sync_tool workflow: check the carrier before correcting a shipment's state.",
    # Cluster D
    "Test pattern for transformer fields derived from a pivot table: write a list test that POSTs a create, then GETs the list and asserts the pivot field appears in the transformer output.",
    "Re-run gotcha: the smoke-test command and the verifier script both run against the same order in sequence, so the second run sees the side effects of the first.",
    "For quick manual checks of Subscriptions and Orders in the seeded local database, use the fixture data from DatabaseSeeder; no manual record creation is needed.",
    "Review workflow: when an implementer reports a fix that is on disk but unstaged, the reviewer must stage the file before running the test suite.",
    "A candidate config for the search benchmark is a JSON file with name, adapter (cloud, ollama or fake), model and baseline_query fields. The harness reads it and emits a scored report.",
    "Test doubles for the embedder seam: FakeEmbedder (single-hot sha256, hermetic), FrozenEmbedder (real geometry from a frozen fixture, hermetic) and LiveEmbedder (local Ollama, integration lane only).",
    "The pytest marker -m integration gates the live Ollama lane. Hermetic tests never call the embedder directly. Fixture files are committed artifacts, so regeneration diffs are reviewable.",
    "Hypothesis property tests carry about 70 percent of the eval harness weight. Properties include no importance freeze, purge leaves a tombstone, additive floor, projection determinism and no false merge.",
    # Cluster E
    "The local stack is two repos in one project group: bookstore-api (Laravel backend) and bookstore-app (mobile client). The setup script registers both under matching names so cross-repo impact analysis works.",
    "A project group needs every repo registered under an explicit name. The group link is derived from each repo's analyzed symbol graph, not from source files alone.",
    "FeedMatcher TopicMap::zoneFor scores matches through matchScore(): segment count dominates, and ties break on the specificity score from the registered topic list.",
    "The importer binds the reporting window (WindowResolver::covering) only when the first data row is inserted, not when the observer is constructed.",
    "The sample app's docker-compose file publishes the database on host port 5432, which collides with a locally installed PostgreSQL. Workaround: remap the host port in docker-compose.override.yml.",
    "Changelog convention: one entry per merged pull request, a single sentence in the past tense, with no ticket numbers in the sentence.",
    "A cross-repo pair that shares one project group lets impact analysis follow a change across the PHP and JavaScript boundary.",
    "The code-graph tool's context() call returns incoming and outgoing references for a symbol, grouped by relationship type (calls, imports, extends, implements, has-method).",
    # Cluster F
    "Two frontend pages render order detail: src/resources/orders/pages/ViewOrder.vue (view mode) and EditOrder.vue (edit mode). Both share the OrderHeader component.",
    "Nuxt 3 composable convention: useXxx returns reactive refs and async fetch functions. Side-effect-free composables live in composables/; page-specific logic stays in the page component.",
    "Order line items are rendered by the OrderLineTable component. It accepts a lines prop (an array of transformer output objects) and emits line-level edit events.",
    "Frontend route for the order list: /orders with an optional ?customer_id= filter. The list page uses the useOrders() composable, which wraps the paginated API call.",
    "ShipmentData was extended with a public ?string $carrierCode. The frontend carrier panel reads it to pre-fill the carrier field in the shipment form.",
    "Frontend Nuxt config: auto-imports are enabled for the composables/ and components/ directories, so no explicit import statements are needed for those symbols.",
    "Order invoice generation endpoint: POST /api/orders/{id}/invoice returns a binary stream. The download button calls it with a fetch using responseType 'blob'.",
    "Nuxt page components in src/resources/<module>/pages/ follow the <Module>List, <Module>View and <Module>Edit naming convention, with route files in src/routes/.",
]

# ---------------------------------------------------------------------------
# Query set: 30 queries with labeled relevant-id sets.
# Relevance is hand-curated ground truth.
# Index references are 0-based into CORPUS_TEXTS above.
# ---------------------------------------------------------------------------

QUERIES: list[dict[str, object]] = [
    {
        "query": "how does the reminder service compute due dates for pending reminders",
        "relevant_indices": [1, 6, 2, 7],
    },
    {
        "query": "what happens when the mail provider is unavailable while writing a mirror row",
        "relevant_indices": [3, 9],
    },
    {
        "query": "digest payload structure and reminder packing",
        "relevant_indices": [2, 7, 10, 11],
    },
    {
        "query": "calendar and note observers that write mirror rows on model events",
        "relevant_indices": [4],
    },
    {
        "query": "digest summary row replacement",
        "relevant_indices": [5],
    },
    {
        "query": "workspace namespacing and cross-workspace leakage prevention",
        "relevant_indices": [0, 8],
    },
    {
        "query": "purge tombstone and hard delete policy",
        "relevant_indices": [16],
    },
    {
        "query": "duplicate gate gray floor auto ceiling bounce guards",
        "relevant_indices": [20, 23],
    },
    {
        "query": "importance score lazy computation from the event log",
        "relevant_indices": [15, 12, 21],
    },
    {
        "query": "embedded columnar database hybrid vector text search",
        "relevant_indices": [14],
    },
    {
        "query": "namespace global named peer read span",
        "relevant_indices": [22, 13],
    },
    {
        "query": "search parallel legs rank fusion above the storage seam",
        "relevant_indices": [18, 14],
    },
    {
        "query": "Ruling arm bounce human decision auto-reinforce forbidden",
        "relevant_indices": [23, 20],
    },
    {
        "query": "projection determinism property test event log replay",
        "relevant_indices": [21, 12],
    },
    {
        "query": "console command registration module service provider",
        "relevant_indices": [24, 30],
    },
    {
        "query": "OrderPlaced listeners recalculate totals firing order",
        "relevant_indices": [26, 28],
    },
    {
        "query": "subscription is_active computed dynamically from plan dates",
        "relevant_indices": [25],
    },
    {
        "query": "driver branching PostgreSQL SQLite test compatibility",
        "relevant_indices": [29],
    },
    {
        "query": "transformer pivot field test list endpoint",
        "relevant_indices": [32],
    },
    {
        "query": "FakeEmbedder FrozenEmbedder hermetic test double embedder seam",
        "relevant_indices": [37, 38],
    },
    {
        "query": "Hypothesis property tests eval harness importance freeze purge",
        "relevant_indices": [39, 38],
    },
    {
        "query": "verifier script smoke test same order side effects",
        "relevant_indices": [33],
    },
    {
        "query": "project group cross-repo setup bookstore-api bookstore-app impact analysis",
        "relevant_indices": [40, 41, 46, 47],
    },
    {
        "query": "changelog convention single sentence per pull request",
        "relevant_indices": [45],
    },
    {
        "query": "docker compose host port collision workaround",
        "relevant_indices": [44],
    },
    {
        "query": "order view edit page Nuxt component",
        "relevant_indices": [48, 50, 55],
    },
    {
        "query": "Nuxt composable convention useXxx reactive refs",
        "relevant_indices": [49],
    },
    {
        "query": "order invoice generation download endpoint binary stream",
        "relevant_indices": [54],
    },
    {
        "query": "Nuxt auto-import composables components directory",
        "relevant_indices": [53],
    },
    {
        "query": "ShipmentData carrierCode frontend panel",
        "relevant_indices": [52, 31],
    },
]

# ---------------------------------------------------------------------------
# Golden dedup pair set (40 pairs, three labeled classes)
#
# Class labels:
#   "same"         - must reinforce/bounce-as-dup (cosine should be high)
#   "distinct_near" - topically close but genuinely distinct - must NOT auto-reinforce;
#                    guard-trip pairs live here (numbers/versions/entities/negation)
#   "unrelated"    - must pass clean (cosine should be low)
# ---------------------------------------------------------------------------

GOLDEN_PAIRS: list[dict[str, object]] = [
    {
        "text_a": "The reminder table is namespaced by (user_id, task_id, channel).",
        "text_b": "Reminders are keyed by user_id, task_id and channel; channel is email, push or sms, and one pending reminder is allowed per key.",
        "label": "same",
    },
    {
        "text_a": "Atlas purge drops content, vector and text-index entry and writes a tombstone.",
        "text_b": "When a document is purged, its content and vector are removed and a tombstone is written with id, kind, reason, timestamp and generated_by_model.",
        "label": "same",
    },
    {
        "text_a": "Search runs two legs in parallel: vector_search and text_search, fused via RRF.",
        "text_b": "The search pipeline runs vector and text legs in parallel, then fuses the results above the storage seam using reciprocal rank fusion.",
        "label": "same",
    },
    {
        "text_a": "An embedded columnar database is the primary store, with no separate service.",
        "text_b": "Atlas uses an embedded columnar database as its store, so no external service is needed to run it.",
        "label": "same",
    },
    {
        "text_a": "Console commands live in app/Modules/<Module>/Commands/ and must be registered in CommandServiceProvider.",
        "text_b": "Module console commands go under app/Modules/<Name>/Commands/ and are registered through the module's CommandServiceProvider.",
        "label": "same",
    },
    {
        "text_a": "Nuxt composables follow the useXxx naming convention and return reactive refs.",
        "text_b": "Nuxt 3 composable pattern: the name starts with use, and the function returns reactive refs and async fetch functions.",
        "label": "same",
    },
    {
        "text_a": "The importance score is not stored; it is computed lazily from the event log.",
        "text_b": "No importance column exists on any row; importance is always computed lazily from hit_count, feedback_score and last_hit_at.",
        "label": "same",
    },
    {
        "text_a": "A read in namespace X sees X plus global, always.",
        "text_b": "A search in namespace X automatically includes the global namespace in addition to X; the read span is always X union global.",
        "label": "same",
    },
    {
        "text_a": "The OrderPlaced event fires OrderTotalsListener and then OrderIndexListener in order.",
        "text_b": "OrderPlaced has two synchronous listeners: the totals recalculation fires first, then the index update listener.",
        "label": "same",
    },
    {
        "text_a": "The duplicate gate in strict mode bounces when in doubt.",
        "text_b": "Strict gate mode: when uncertain, always bounce and surface the candidate for human resolution.",
        "label": "same",
    },
    {
        "text_a": "DocId is a UUID newtype, never an auto-increment integer.",
        "text_b": "Document ids are UUID newtypes (DocId), portable across adapters and never auto-incremented.",
        "label": "same",
    },
    {
        "text_a": "Changelog entries are one sentence per merged pull request.",
        "text_b": "Changelog convention: one past-tense sentence for each merged pull request, nothing more.",
        "label": "same",
    },
    {
        "text_a": "Order invoice endpoint: POST /api/orders/{id}/invoice returns a binary stream.",
        "text_b": "Generating an order invoice calls POST /api/orders/{id}/invoice; the response is a binary blob.",
        "label": "same",
    },
    {
        "text_a": "Atlas namespaces: global plus flat named peers, two tiers only, no nesting.",
        "text_b": "The namespace hierarchy is flat: one reserved global namespace plus any number of named peers. No nesting, no paths.",
        "label": "same",
    },
    {
        "text_a": "Hypothesis property tests make up about 70 percent of the eval harness.",
        "text_b": "The eval harness is about 70 percent Hypothesis property tests and 30 percent concrete named fixtures.",
        "label": "same",
    },
    {
        "text_a": "Rulings bounce eagerly at the lower gray floor; auto-reinforce is always forbidden.",
        "text_b": "The Ruling arm always bounces at or above its lower gray floor; no auto-reinforce path exists for Rulings.",
        "label": "same",
    },
    {
        "text_a": "A cross-repo project group needs every repo registered under an explicit name.",
        "text_b": "To set up a cross-repo project group, register each repo under an explicit name before analysis.",
        "label": "same",
    },
    {
        "text_a": "Subscription is_active is computed dynamically by comparing the plan's date range against today.",
        "text_b": "There is no is_active column on Subscription; the transformer derives it by checking whether today falls inside the plan's date range.",
        "label": "same",
    },
    {
        "text_a": "mirrorFor uses safeSend() to tolerate a provider outage and stores a mirror row without a delivery id.",
        "text_b": "When the mail provider fails in mirrorFor, safeSend() catches the exception, logs a warning and stores the mirror row with a null delivery id.",
        "label": "same",
    },
    {
        "text_a": "The digest row is replaced on each DigestRunner run by deleting and then inserting.",
        "text_b": "DigestRunner keeps exactly one digest per user: each run deletes the previous row and inserts a new one.",
        "label": "same",
    },
    {
        "text_a": "The order line table shows a maximum of 50 lines per page.",
        "text_b": "The order line table shows a maximum of 100 lines per page.",
        "label": "distinct_near",
        "guard_note": "numeral: 50 vs 100",
    },
    {
        "text_a": "The search library is pinned at version 0.9.x for the first milestone.",
        "text_b": "The search library is pinned at version 0.14.x for the second milestone.",
        "label": "distinct_near",
        "guard_note": "version: 0.9.x vs 0.14.x",
    },
    {
        "text_a": "Alice is the designated code reviewer for the bookstore-api backend.",
        "text_b": "Bob is the designated code reviewer for the bookstore-api backend.",
        "label": "distinct_near",
        "guard_note": "entity: Alice vs Bob",
    },
    {
        "text_a": "Auto-reinforce is allowed for active Note candidates above the auto ceiling.",
        "text_b": "Auto-reinforce is NOT allowed for Ruling candidates; they always require a human decision.",
        "label": "distinct_near",
        "guard_note": "negation: allowed vs not allowed",
    },
    {
        "text_a": "ReminderService.dueSoon() uses a plain loop to rank reminders by due date.",
        "text_b": "DigestRunner.compose() assembles a budgeted payload from pinned and due reminders.",
        "label": "distinct_near",
    },
    {
        "text_a": "The embedded database provides native full-text search through its FTS index.",
        "text_b": "The in-memory adapter implements full-text search in pure Python for hermetic tests.",
        "label": "distinct_near",
    },
    {
        "text_a": "Purge writes a tombstone with the purged document's id and reason.",
        "text_b": "The distinct-pair registry stores pairs of ids that the gate judged distinct.",
        "label": "distinct_near",
    },
    {
        "text_a": "Order view page: ViewOrder.vue renders read-only fields in the orders module.",
        "text_b": "Order edit page: EditOrder.vue renders form inputs in the orders module.",
        "label": "distinct_near",
    },
    {
        "text_a": "Atlas's ruling_gray_floor is lower than gray_floor to catch Ruling matches earlier.",
        "text_b": "Atlas's auto_ceiling threshold triggers automatic reinforcement without bouncing.",
        "label": "distinct_near",
    },
    {
        "text_a": "pytest -m integration runs the live Ollama embedding lane.",
        "text_b": "uv run pytest runs the hermetic gate (no Ollama, no network).",
        "label": "distinct_near",
    },
    {
        "text_a": "The reminder table is namespaced by (user_id, task_id, channel).",
        "text_b": "The order invoice endpoint returns a binary stream.",
        "label": "unrelated",
    },
    {
        "text_a": "Console commands live in app/Modules/<Module>/Commands/.",
        "text_b": "Atlas purge drops content and vector and writes a tombstone.",
        "label": "unrelated",
    },
    {
        "text_a": "Nuxt composables follow the useXxx naming convention.",
        "text_b": "An embedded columnar database is the primary store, with no separate service.",
        "label": "unrelated",
    },
    {
        "text_a": "Changelog entries use one sentence per pull request.",
        "text_b": "DigestRunner.compose returns a typed DigestPayload.",
        "label": "unrelated",
    },
    {
        "text_a": "The code-graph tool needs each repo registered under an explicit name.",
        "text_b": "OrderPlaced fires the totals listener, then the index listener.",
        "label": "unrelated",
    },
    {
        "text_a": "Subscription is_active is computed dynamically.",
        "text_b": "Hypothesis property tests make up about 70 percent of the harness.",
        "label": "unrelated",
    },
    {
        "text_a": "The compose file's database port collides with a local PostgreSQL.",
        "text_b": "DocId is a UUID newtype, never an auto-increment integer.",
        "label": "unrelated",
    },
    {
        "text_a": "Frontend route for the order list: /orders with an optional customer filter.",
        "text_b": "Atlas namespaces: global plus flat named peers, two tiers only.",
        "label": "unrelated",
    },
    {
        "text_a": "ShipmentData gained a carrierCode field.",
        "text_b": "Search runs vector and text legs in parallel, fused via RRF.",
        "label": "unrelated",
    },
    {
        "text_a": "The digest row is replaced on each run.",
        "text_b": "Queries that branch on the database driver must keep PostgreSQL semantics under SQLite.",
        "label": "unrelated",
    },
]

# ---------------------------------------------------------------------------
# Collect all unique texts to embed
# ---------------------------------------------------------------------------

def _all_texts() -> list[str]:
    texts: set[str] = set()
    for t in CORPUS_TEXTS:
        texts.add(t)
    for q in QUERIES:
        texts.add(str(q["query"]))
    for pair in GOLDEN_PAIRS:
        texts.add(str(pair["text_a"]))
        texts.add(str(pair["text_b"]))
    texts |= case_texts(load_cases())
    return sorted(texts)  # sorted for deterministic order


# ---------------------------------------------------------------------------
# Embed via litellm
# ---------------------------------------------------------------------------

async def _embed(text: str) -> list[float]:
    """Embed a single text via litellm."""
    resp = await litellm.aembedding(
        model=MODEL,
        input=[text],
        api_base=API_BASE,
        timeout=30.0,
    )
    return list(resp.data[0]["embedding"])


async def main() -> None:
    all_texts = _all_texts()
    print(f"Embedding {len(all_texts)} unique texts via {MODEL}...")

    vectors: dict[str, list[float]] = {}
    for i, text in enumerate(all_texts, 1):
        try:
            vec = await _embed(text)
            if len(vec) != DIM:
                print(
                    f"ERROR: expected dim={DIM}, got {len(vec)} for text: {text[:60]!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
            vectors[text] = vec
            if i % 10 == 0:
                print(f"  {i}/{len(all_texts)}...")
        except Exception as exc:
            print(f"ERROR embedding text {i}: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"Done. Embedded {len(vectors)} texts.")

    # Write fixture — deterministic JSON (sorted keys)
    fixture: dict[str, object] = {
        "embedder_model": MODEL,
        "dim": DIM,
        "vectors": vectors,
    }

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixture_json = json.dumps(fixture, sort_keys=True, ensure_ascii=False)
    FIXTURE_PATH.write_text(fixture_json, encoding="utf-8")
    print(f"Fixture written to {FIXTURE_PATH} ({FIXTURE_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    asyncio.run(main())
