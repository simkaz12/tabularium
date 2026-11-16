# Tabularium

Long-term memory for LLM agents, built as an MCP server on a Python engine, with a second TypeScript server (v2) that took over the day-to-day writes.

The thesis: memory is a harness concern, not a database. Hybrid vector and keyword search is the easy part. The parts that decide whether a memory is any good are what gets written, what gets refused, what gets surfaced to which agent, and what gets cleaned out later. Tabularium owns that whole path instead of leaving it to the caller.

## Status

This is a solo project that I use every day for my own coding agents. It is a public snapshot of a private repository, with the history squashed and the personal data removed. Nothing here is a product, and whether it beats having no memory layer at all is a hypothesis I am still testing on myself. No benchmark in this repo says otherwise.

The design documents under `docs/` ([`SPEC.md`](docs/SPEC.md) and [`docs/adr/`](docs/adr/)) are historical records of what I decided and why, written before and during the build. They diverge from the code in places. When they disagree, the code is right.

## What is in the repo

| Path | What it is |
| --- | --- |
| `src/tabularium/` | The Python engine (v1), the MCP server and the CLI. About 34k lines. |
| `v2/` | The TypeScript server (v2): an HTTP API with proposals, fold asks, a weekly cleaner and a daily defect sweep that queue tiles for a human, and a LanceDB store. See [`v2/README.md`](v2/README.md). |
| `tests/` | The Python test suite, including a conformance suite that runs both storage adapters through one contract. |
| `scripts/eval_v1v2/` | A bench that compares v1 and v2 on the same inputs. See [`scripts/eval_v1v2/README.md`](scripts/eval_v1v2/README.md). |
| `deploy.sh`, `v2/deploy/` | Deploy scripts for a systemd host over ssh. |
| `CONTEXT.md` | Glossary of the domain terms the code and ADRs use. |

## How it works

### Records

A record is a hypomnema, a durable note kept for later recall. A scope is a flat namespace per project. Reading in scope X sees X plus a reserved `global` scope.

There are three kinds of record:

- A Fact is knowledge that may be revised and decays out of use.
- A Decision records what was chosen and why. In v1 the reason is part of the Decision's text. A v2 record has no `why` field: the thread excerpt in `provenance.excerpt` is the reason, and a v2 Decision names the option that was ruled out in `rejected`. Its content never changes. A change is a new Decision that supersedes the old one, and the old one stays queryable as history.
- A Defect is a bug an agent saw and left alone. It records where the bug is, what triggers it, the symptom, and why it was left. It moves from `open` to `taken` (an agent is working on it) to `closed`. Closing it takes a proposal whose excerpt is the proof of the fix. A daily sweep sends stale takes back to `open`, and asks a human whether a defect open for 90 days is still real.

The v1 engine has only Facts and Decisions. Defects came with v2.

### Read side

- `recall` (v1) runs a vector leg and a BM25 leg in parallel through the storage interface and fuses them with reciprocal rank fusion in the engine.
- `context(scope, budget, role)` builds the payload an agent gets at session start. It fills two tiers, `hot` and then `ember`, filters by role and packs the result into a token budget. Hot is a small capped tier that is curated by hand. An ember is a transient nudge that fades if nothing references it.
- `decisions_query` and the `decision://{id}` resource return Decisions with their supersession chain in a byte-stable render.

The MCP server exposes four tools (`recall`, `decision`, `decisions_query`, `context`) and the `decision://{id}` resource. It has no write tools, and its tools read the v1 store only. That store is now a read-only archive.

### Write side and the dedup gate

`Tabularium.save` is the engine's write path. Before a record is written, the gate embeds it and probes the same scope for near neighbours. A neighbour at or above the cosine floor (0.80 for Facts, 0.70 for Decisions, both configurable) is a survivor. With an assessor configured, the assessor judges the pair. Without one, the save bounces. A bounce writes nothing. It returns the candidate and the exits the caller may take (reinforce the old record, confirm the two are distinct, supersede it).

Four plain string checks (numbers, versions, named entities, negation) catch pairs that cosine similarity treats as one but that differ in a way that matters, such as "50 per page" against "100 per page".

When `TYPESAFE_API_KEY` is set, the assessor (a hosted model from typesafe.ai) classifies the relation between the incoming record and its neighbour as restatement, replacement, follow-up or distinct. Four flags, one per label, can turn a label into an automatic exit. They are off by default.

The outcomes of a save are `saved`, `reinforced`, `revived`, `reconciled`, `bounced` and `already_saved`.

### Cleaner

The Cleaner is a periodic pass over a scope that judges what the memory should stop holding. It has three axes:

- relevance: records born since the last pass are graded against a rubric and the ones not worth keeping are shelved. Shelved records leave recall but stay in the store.
- falsity: a newer record is compared with older ones it may have overtaken, and a contradicted Fact gets a supersession edge.
- revisit: a rotation over old records, with no watermark, so a quiet scope still gets its old records looked at.

Two judges from different model families grade independently, by default the `claude` and `codex` CLIs, with a litellm backstop (needs `DEEPSEEK_API_KEY`) when a seat is down. Disagreements are handled by policy. The rubric is a TOML file hashed into a version that is stamped on every verdict. The pass advances its watermark only after it re-reads every write it made and finds it present. It runs nightly when `TABULARIUM_ENABLE_CLEANER=true` (at 03:00 by default, set with `TABULARIUM_CLEANER_NIGHTLY_AT`), or by hand with `tabularium clean`.

### Storage

LanceDB is the production store, behind a storage port. An in-memory adapter implements the same port for tests. One conformance suite runs against both.

### v2

v2 is a separate TypeScript server with its own LanceDB store. Agents reach it over plain HTTP, since it has no MCP surface; T3 Code, the coding-agent app I use, calls it from its memory tools. Agents submit proposals (`POST /v2/proposals`) for any of the three kinds, and an agent proposal applies on create unless a fold ask holds it (below). Agents take and release defects directly, with no proposal. A checker model (the same hosted assessor) labels each proposal and flags the ones that look like junk or restatements.

With fold asks on (`TABULARIUM_FOLD_ASK`, off in code and on in the shipped systemd unit), a fact or decision proposal with a close live neighbour in the same scope is not written at once. It is held as `asked` until a human answers `fold` (a merged record is written and the old one is superseded) or `keep_both`, or until the ask expires, when the original proposal is written as submitted.

A cleaner runs weekly (Sunday 03:00 server time) and queues merge, supersede, retire and promote suggestions as tiles for a human to approve or reject. A defect sweep runs daily at 05:00 and queues a close tile for each defect that has been open 90 days. Records imported from a v1 store sit in a separate legacy table that recall searches alongside the v2 records; legacy rows never change through the API and drop out of recall once a v2 record replaces them. v2 recall is hybrid search over the v2 and legacy rows, reranked by the Jev checker, with `kind` and `audience` filters. It is a different path from the v1 fusion described above. The route list and every status code are in [`v2/README.md`](v2/README.md).

## Quick start

You need Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), and for real embeddings an [Ollama](https://ollama.com) server with `nomic-embed-text` on `localhost:11434`. The test suite does not need Ollama.

```bash
uv sync
export TABULARIUM_STORE_PATH=/tmp/tabularium-demo
uv run tabularium scope add demo
uv run tabularium doctor
uv run tabularium-mcp
```

`tabularium-mcp` serves over stdio unless `TABULARIUM_MCP_TRANSPORT=http` is set. With the HTTP transport it binds `TABULARIUM_MCP_HOST` and `TABULARIUM_MCP_PORT`.

To use the engine as a library:

```python
import asyncio
import tempfile

from tabularium.clock import SystemClock
from tabularium.config import EngineConfig
from tabularium.embedding import LitellmEmbedder
from tabularium.engine import Tabularium
from tabularium.storage.lancedb_adapter import LanceDBAdapter


async def main() -> None:
    embedder = LitellmEmbedder()
    clock = SystemClock()
    adapter = await LanceDBAdapter.open(
        tempfile.mkdtemp(), embedding_model=embedder.model, dim=embedder.dim, clock=clock
    )
    engine = Tabularium(adapter, embedder, clock, EngineConfig())
    await engine.initialize()

    await engine.save("The todo app syncs tasks to Postgres.", scope="global")
    second = await engine.save("The todo app syncs its tasks to a Postgres database.", scope="global")
    print(second.outcome)

    result = await engine.recall("where do tasks sync to", scope="global", limit=3)
    print([hit.content for hit in result.hits])


asyncio.run(main())
```

With Ollama running, the second save is bounced as a near-duplicate of the first.

The CLI has nine subcommands: `sync`, `scope`, `facet`, `role`, `reclaim`, `doctor`, `backfill`, `supersede` and `clean`.

## Tests

```bash
uv run pytest          # default lane, no network, no Ollama
uv run pytest -m integration   # needs a live Ollama (and some tests an API key)
uv run ruff check
uv run mypy .

cd v2 && npm ci && npm run typecheck && npm test
```

The default Python lane is hermetic. Embeddings come from a frozen fixture of vectors (`tests/fixtures/frozen_vectors.json`) built from a small synthetic corpus. `scripts/freeze_vectors.py` regenerates it against a local Ollama. The same lane enforces recall and precision floors on that corpus and a precision floor on the dedup gate's golden pairs. Those floors are regression gates for this synthetic corpus, not quality claims about real data.

## Eval bench

`scripts/eval_v1v2/` replays labelled record pairs through the v1 gate and through v2's proposal path, and asks both engines the same questions, on throwaway stores. Real data is bring-your-own. The README there lists the inputs, the requirements and the run commands.

## Deploy

Both deploy scripts take their target from environment variables and fail with a usage message when one is missing. They assume a Linux host with systemd that you reach over ssh.

```bash
DEPLOY_HOST=user@host DEPLOY_DIR=tabularium DEPLOY_URL=http://host:8787/mcp ./deploy.sh
DEPLOY_HOST=user@host DEPLOY_DIR=tabularium-v2 DEPLOY_HEALTH_URL=http://host:8789/v2/health v2/deploy/deploy.sh
```

`deploy.sh` pulls and syncs the Python checkout on the host, waits for the Cleaner's judge subprocesses to finish so a restart does not kill a pass half way, restarts the unit, and lists the tools the new process serves. `./deploy.sh --drain-only` runs only the wait. `v2/deploy/deploy.sh` syncs `v2/`, installs a systemd unit rendered from `v2/deploy/tabularium-v2.service`, restarts it and polls the health URL. Neither script ships a unit for the Python server, so create `tabularium-mcp.service` yourself or set `DEPLOY_UNIT`.

The Python server has no authentication. v2 only checks the `Host` header against `TABULARIUM_ALLOWED_HOSTS`. Bind both to loopback or a private interface. The MCP HTTP transport defaults to `127.0.0.1:8787`.

## Layout notes

- Python 3.12, pydantic, LanceDB, litellm for embeddings and judge backstops, the official MCP Python SDK for the server. ruff, and mypy in strict mode.
- v2 needs Node 22.18 or newer, the first 22.x that strips types without a flag. It runs TypeScript directly with no build step.

## License

MIT. See [LICENSE](LICENSE).
