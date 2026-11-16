# Tabularium v2

TypeScript memory server, run on a small always-on host such as a Raspberry Pi. Agents save records through proposals that apply on create unless a fold ask holds them, the cleaner queues its doubts as tiles for the owner in a tray in my fork of [T3 Code](https://github.com/pingdotgg/t3code) ([simkaz12/t3code](https://github.com/simkaz12/t3code)), which this server is wired into, and recall runs over the v2 store plus a legacy archive built from v1. Legacy rows never change content through the API (`importLegacy --replace` rewrites them) and drop out of recall once a v2 record takes their place.

## Run locally

```
npm ci
npx tsc --noEmit
npx vitest run
node src/main.ts
```

Node 22.18 or newer. The server strips types itself, so there is no build step.

## Configuration

| Variable | Default |
| --- | --- |
| `TABULARIUM_HOST` | `127.0.0.1` |
| `TABULARIUM_PORT` | `8789` |
| `TABULARIUM_STORE` | `./tabularium-v2-store` |
| `TABULARIUM_ALLOWED_HOSTS` | `<host>:<port>` |
| `TABULARIUM_EMBED_URL` | `http://localhost:11434/api/embed` |
| `TABULARIUM_EMBED_MODEL` | `nomic-embed-text` |
| `TABULARIUM_EMBED_DIM` | `768` |
| `TABULARIUM_EMBED_TIMEOUT_MS` | `10000`; one total deadline per embedding request, a miss answers 503 |
| `TABULARIUM_ASSESSOR_MODEL` | `jev-1.13.0` |
| `TABULARIUM_TOPICS_PATH` | `topics.json` next to `package.json` |
| `TABULARIUM_JEV_TIMEOUT_MS` | `15000`; one total deadline per Jev call, the SDK does not retry |
| `TABULARIUM_CONTAINMENT` | `on`; accepts `on`, `true`, `1`, `yes`, `off`, `false`, `0`, `no`; flags a proposal that covers a live neighbour |
| `TABULARIUM_CONTAINMENT_FLOOR` | `0.95`; how fully the new text must cover the neighbour, 0 to 1 |
| `TABULARIUM_FOLD_ASK` | `off`; same accepted values; the shipped unit sets `on` |
| `TABULARIUM_FOLD_SIMILARITY_FLOOR` | `0.8`; minimum similarity for a fold ask, 0 to 1 |
| `TABULARIUM_FOLD_SUBJECT_FLOOR` | `0.7`; minimum same-subject score for a fold ask, 0 to 1 |
| `TABULARIUM_FOLD_ASK_TIMEOUT_MS` | `600000`; how long an ask waits for an answer |
| `TYPESAFE_API_KEY` | none; without it checks and rerank are skipped and proposals carry `jev_unavailable` |

Containment and fold asks apply only to fact and decision proposals with no `target`. Requests whose `Host` header is not in `TABULARIUM_ALLOWED_HOSTS` get a 403. There is no other auth; bind it to loopback or a private interface (for example a VPN address) and keep it off the public internet.

## API

Base `/v2`. Bodies and responses are JSON.

| Route | Purpose |
| --- | --- |
| `POST /proposals` | create a proposal and run the Jev checks; an `agent` proposal applies on create: the response is `201` with `status: "approved"` and `resolution.recordId` (the new record, or the closed defect for a close target), and a Jev `reinforce` suggestion counts the reinforcement on the neighbour and answers `status: "reinforced"` with that neighbour's id and no new record; a `supersede` suggestion does not apply and stays on the proposal row for the cleaner; with fold asks on, a close match returns `201` with `status: "asked"` and an `ask` object instead (see Fold asks); a reinforce on a legacy neighbour answers with the id of the promoted v2 copy, not the neighbour's id |
| `GET /proposals?status=pending&since=<iso>` | list proposals; `status` is `pending` (the default), `approved`, `rejected`, `reinforced`, `asked`, `folded` or `all`; any other status or a `since` that is not a timestamp returns 400; an unknown query parameter such as `section` is ignored |
| `GET /asks?since=<iso>` | list held and answered fold asks, `{"counts": {"fold", "keep_both", "unanswered", "open"}, "asks": [...]}` |
| `POST /proposals/:id/answer` | answer an `asked` proposal with `{"action": "keep_both", "reason": "..."}` or `{"action": "fold", "kind": "fact" or "decision", "headline": "...", "fields": {...}}`; repeating the same answer is safe, a different one returns 409 |
| `POST /proposals/:id/approve` | write a record or apply the proposal's target action; `edits` may carry `headline`, `scope`, `purpose`, `topic`, `audience` and `fields.{claim,rejected,where,trigger,leftBecause,fixHint,impact}`; the body also takes top-level `supersedes` (record ids) and `reason` (a reason class: `correction`, `changed` or `elaborated`, required when a supersede applies) |
| `POST /proposals/:id/reject` | close without a record, body `{"reason": "..."}` with a non-empty reason |
| `POST /proposals/:id/reinforce` | body `{"recordId": "..."}`, required; count a reinforcement on that record, close without a record, return the record; a legacy `recordId` promotes a copy |
| `GET /recall?q=&scope=&kind=&audience=&limit=` | `q` is required (400 when empty); `limit` is a positive integer, default 10, at most 50; a bad `kind` or `audience` returns 400; hybrid search over v2 and legacy, reranked by Jev; `kind` is `fact`, `decision` or `defect`; `audience` keeps records for that role and unlabelled ones |
| `GET /records/:id` | one record with lineage and `takenBy` |
| `POST /records/:id/take` | mark an open defect as taken, body `{"threadId": "...", "link": "..."}` with both non-empty, returns the record |
| `POST /records/:id/release` | return a taken defect to open, empty body, returns the record; `takenAt` keeps the last-take time and `takenBy` is `null`; a defect that is not taken comes back unchanged |
| `POST /records/:id/unsupersede` | undo a supersede, `reason` required |
| `DELETE /records/:id` | delete a record that holds a secret, body `{"reason": "...", "category": "secret"}` |
| `GET /settings` | server settings, currently `{"cleanerConfidenceFloor": 0.7}` |
| `PUT /settings` | change settings, body `{"cleanerConfidenceFloor": 0.8}`, returns the full settings |
| `GET /health` | liveness |

### Proposal body

`POST /proposals` takes `kind` (`fact`, `decision` or `defect`), `headline` (100 characters at most, the claim in one sentence), `fields`, `scope`, `provenance`, `origin` (`agent` when omitted), an optional `audience` (`decider`, `builder`, `reviewer` or `navigator`) and an optional `target`.

- `fields.claim` is required for every kind. The record has no `why`: the thread excerpt in `provenance.excerpt` is the reason. A `fields.why` key is accepted and ignored.
- A decision names the ruled-out option in `fields.rejected`.
- A defect needs `fields.where`, `fields.trigger` and `fields.left_because`, with `claim` as the symptom. `fix_hint` and `impact` are optional. The parser also reads `leftBecause`, `fixHint` and `symptom`. A missing field returns 400 naming it, such as `where is required for a defect`.
- `provenance.stage` is the skill or phase that proposed it, `""` when unknown.
- A body `audience` wins over the one Jev infers.
- `target.action` is `retire`, `supersede`, `merge`, `promote` or `close`. A close proposal needs only `claim` (the closing note) and the target, whatever its `kind`, and skips the Jev checks.

Every proposal carries `audience` (or null) and `reader` (`lookup`, `trap`, `ruling`, `none`, `unclear` or null). The record copies `audience`, `reader` and `stage` from the proposal; on a cleaner tile, `edits.audience` at approve overrides the audience. Only `origin: "agent"` is accepted on `POST /proposals`; `cleaner` and `promotion` proposals are created inside the server and stay pending until the owner approves or rejects them.

### Jev questions and flags

Jev gets one request about the record and two per neighbour (relation, then the pair questions), for the nearest three, so up to seven calls per proposal. The gate questions are `kind`, `worth`, `perishable` and `reader`; `perishable` is not asked on a defect. `audience`, `breadth`, `scope`, `purpose` and the optional `topic` are labels. `headline_faithful`, `claim_standing` and `excerpt_grounds` are shape checks, plus `rejected_names_alternative` on a decision that names one.

Every choice question has an `unclear` answer for when the text does not give enough to tell. An `unclear` answer flags `unclear:<question>`, never demotes, never produces a suggestion, and stores a null audience or purpose. The confidence floor is 0.5.

A gate answer at confidence 0.5 or higher raises a `demote:<question>=<answer>` flag when `kind` is `neither`, `worth` is not `keep`, `perishable` is `status` or `sequencing`, or `reader` is `none`. A flag never blocks or places the record: it saves live, and the cleaner reads the flags on the proposal row to queue a retire tile. `shape_low:<question>` flags and `low_confidence:<question>` flags on a gate question queue a retire tile the same way; relation and reason `low_confidence` flags do not. `low_confidence:<question>` flags appear only for the four gate questions. When Jev is unavailable the proposal carries `jev_unavailable` and no demote flags.

### Defects

A defect record moves `open`, `taken`, `closed`. `POST /records/:id/take` sets `defectStatus: "taken"` with `takenThreadId`, `takenLink` and `takenAt`, also returned as `takenBy: {threadId, link, at}`; no proposal and no approval. Taking a taken or closed defect returns 409, and a record that is not a defect returns 400. Release returns the defect to `open` and clears `takenThreadId` and `takenLink`; `takenAt` stays as the last-take time, and `takenBy` is `null` while no thread holds the defect. Closing goes through a proposal with `target: {recordId, source: "v2", action: "close"}`: an agent close proposal sets `defectStatus: "closed"`, `closedBy` to the proposal id and `closedAt` on create, writes no new record and answers `approved` with the defect's id; a cleaner close tile does the same on approve. The proposal's excerpt is the proof. A close proposal for a missing record returns 404 and for a record that is not an open or taken v2 defect returns 409.

Recall results carry `audience`, `reader`, `stage`, `where`, `trigger`, `leftBecause`, `defectStatus` and `takenBy` next to the existing fields; `fields.why` is `null` unless the row is a legacy one that has it.

Recall with a `scope` also returns `global` records. Quotes and other full-text syntax in `q` are escaped, so any query text returns hits instead of an error. If a Jev rerank call fails, that hit keeps its hybrid position while the scored hits reorder among theirs. Recall falls back to hybrid order only when every call fails.

### Status codes

| Status | When |
| --- | --- |
| 400 | invalid JSON or fields; a defect missing `where`, `trigger`, `left_because` or its symptom; a reject without a `reason`; an unknown `audience`; take or release on a record that is not a defect; a malformed percent escape in the path; `origin` other than `agent` on `POST /proposals`, including `harvest`; `DELETE` without a non-empty `reason` or without `category: "secret"`; a `PUT /settings` value outside 0 to 1, a non-number or an unknown key |
| 403 | `Host` header not allowed |
| 404 | unknown route, record or proposal |
| 413 | request body over 1,000,000 bytes |
| 409 | approve, reject or reinforce on a proposal that is not pending; take on a defect that is taken or closed; a close for a record that is not an open or taken v2 defect; a reinforce, supersede or retire on a record that is not live; a second promote of the same legacy row; `unsupersede` on a record that is not superseded, or on a legacy row a v2 record replaced; `DELETE` on a record that is the replacement for another record |
| 500 | internal error |
| 503 | the embedder is down (Ollama unreachable or returning bad vectors) |

A Jev outage is not an error. `POST /proposals` returns within `TABULARIUM_JEV_TIMEOUT_MS` plus a small margin and the proposal carries `jev_unavailable`.

### Fold asks

With `TABULARIUM_FOLD_ASK` on, an agent fact or decision proposal with no `target` that has a close live neighbour in the same scope is held with `status: "asked"` and an `ask` object. The ask names the existing record. A human answers through `POST /proposals/:id/answer`. `fold` writes the merged record, supersedes the old one with reason `elaborated`, and ends the proposal as `folded`. `keep_both` writes the original proposal and ends it as `approved`. When `TABULARIUM_FOLD_ASK_TIMEOUT_MS` passes with no answer, the original is written and the ask is marked `unanswered`. A sweep every 60 seconds does the expiry and logs `{"event":"asks-expired",...}`.

Containment is separate. When the new text covers a live neighbour at or above `TABULARIUM_CONTAINMENT_FLOOR`, the proposal gets a `covers:<recordId>` flag. With fold asks on it is held against that neighbour; with them off it is written and keeps the flag.

The other flags a proposal can carry are `kind_differs`, `worth_not_keep`, `perishable_not_standing`, `rejected_missing`, `relation_unavailable:<id>`, `pair_unavailable:<id>`, `restatement:<id>` and `replacement:<id>`.

### Retries

Approve is safe to retry. The new record takes the proposal's id, so a retry after a partial failure finds the earlier write and finishes the rest: one record, each superseded record superseded once, the proposal `approved`. Reinforce resolves the proposal before it bumps the count, so a crash can lose a bump but never double it.

### Deleting

`DELETE` is for secrets that landed in a record. Garbage gets retired through a proposal, not deleted. Legacy rows can be deleted. A record that another record's `supersededBy` points at returns 409 with the dependent ids; restore or delete those first.

### Legacy retirement

Legacy rows are created only by the import. Their content never changes, and their lifecycle moves only from `live` to `retired`, with `supersededBy` set to the v2 record that took over when promote, supersede or reinforce retired it. A plain retire leaves `supersededBy` null and sets `retiredReason`. Retired legacy rows do not appear in recall. Approving any of these retires the row:

- promote: a v2 copy carrying `legacyId`
- supersede with a legacy id in `supersedes`: a new v2 record
- reinforce on a legacy `recordId`: promotes it to a v2 copy with `reinforcedCount` 1, and the proposal ends as `reinforced`
- retire on a legacy id

`unsupersede` on a retired legacy row restores it to `live` unless a v2 record carries its `legacyId`, in which case it returns 409.

## Settings

Settings live in the store's `settings` table on the server, so every T3 Code machine sees the same values. The only one is `cleanerConfidenceFloor`, default 0.7. To change it, open T3 Code Settings, go to the memory section and edit "Cleaner confidence floor" (0 to 1, step 0.05). It saves on blur and calls `PUT /v2/settings`. The cleaner reads it at the start of each run.

## Cleaner

Sundays at 03:00 server time the cleaner proposes at most five tiles: legacy promotions, merges of near-duplicates, supersedes for contradicted records and retirements of stale ones. A doubts pass runs first, outside that cap. It never writes a record. A record with a pending proposal or a rejection in the last 90 days is skipped. Each tile blocks both of its record ids for the rest of the run, so three identical records produce one merge tile, not a chain. The same applies to the target and the replacement of every pending or recently rejected proposal from earlier runs.

### Doubts pass

Agent proposals save live on create, so the cleaner is where a record Jev doubted gets a second look. The pass reads proposals with `status = 'approved'` and `origin = 'agent'` whose `resolvedAt` is later than the doubts cursor, newest first, and keeps those whose record is still `live` and not blocked by a pending or recently rejected tile. The cursor (`meta:lastDoubtsAt`) is the start time of the last run whose doubts pass cleared its backlog, 7 days back when none is saved; it is separate from the last run time the scheduler records. A run that hits the cap of 10 leaves the cursor in place, so the dropped records come up on the next run, and a record that already carries a pending or recently rejected tile is not queued again. A proposal whose `suggestion` is `supersede` with a live older record and a reason class gets the usual supersede tile: the older record is the target, the new record the `replacementId`, the reason the suggestion's class. Otherwise a proposal with a `demote:` flag gets a `retire` tile on the new record with reason `jev doubted at propose: <flags>`. A record that qualifies for both gets the supersede tile only, and once that tile is approved and the older record is superseded by the new one, a later run does not follow it with a retire tile on the new record. If the older record was retired or superseded by something else, the supersede tile is dropped and the retire tile applies. The proposals read is limited to 20,000 rows (`CLEANER_DOUBT_READ_LIMIT`); a read that fills its limit leaves the cursor in place, as a capped pass does. The pass queues at most 10 tiles per run (`CLEANER_DOUBT_CAP`), and they do not count against the cap of five.

### Pair routing

The cleaner pairs live records of at least 30 days old by cosine similarity, highest first. The Jev contradiction check runs first, with a budget of 200 Jev calls per run.

- Cosine 0.9 or higher with Jev configured and budget left: if Jev answers "replace" (correction or changed) at confidence at or above the floor, the tile is a supersede with the newer record superseding the older. Any other answer gives a merge tile. Without Jev or budget the tile is a merge.
- Cosine 0.9 or higher when the contradiction call fails: no tile for that pair this run. Merging on failure would turn a changed value into a merge during an outage.
- Cosine 0.75 up to 0.9: a supersede tile only when Jev answers "replace" at confidence at or above the floor.
- A `shelve-lean` retirement also needs confidence at or above the floor.

Tiles that Jev assessed show its confidence: a supersede tile appends it to the tile's `claim` text and keeps `target.reason` as the bare class word that approve validates, and merge and retire tiles carry it in `target.reason`. Promotion tiles and merges without a Jev verdict show none. The floor is the `cleanerConfidenceFloor` setting above. A pair that Jev rates "coexist" and a record it rates "keep" are remembered for 90 days in the `cleaner_state` table, keyed by id and creation time. The next runs skip them without spending budget, and a pair at 0.9 or higher that is remembered as coexist gets a merge tile with no Jev call.

### Failures

One bad candidate never stops the run. Each Jev call and each tile write is guarded on its own, and the log line says which failed. After 3 consecutive outage failures (timeout or abort, network error, HTTP 5xx, 401, 403 or 429) the Jev stages end for that run with one log line. Other 4xx responses and malformed answers count as a bad candidate and reset the count. Once the stages have ended, a pair at 0.9 or higher gets no tile, except a remembered coexist pair, which still gets its merge tile. Promotions still run. A run never holds the store lock while it waits on Jev.

### Catch-up

The last run time is saved in `cleaner_state`. On startup, if the most recent Sunday 03:00 is later than the saved time, the cleaner runs about 30 seconds after start. With no saved time it records the current time and does not run.

### Defect sweep

Daily at 05:00 server time, two rules run over live defects and the server logs `{"event":"defect-sweep","released":n,"asked":m}`. A `taken` defect whose `takenAt` is more than 14 days old goes back to `open`, its thread and link are cleared and `takenAt` is kept as the last-take time. An `open` defect created more than 90 days ago, with no take in that window (the kept `takenAt` counts, so a released defect waits out 90 days from that take), gets one close tile with `origin: "cleaner"`, headline `Still open? <headline>` cut with an ellipsis so the whole headline is at most 100 characters, the defect's symptom as the claim, and `target.reason` `open 90d`. A record that already has a pending tile of any action, or a tile of any action rejected within 90 days, gets none, and one run asks for at most 5. A defect released in the same run is not asked about. The last run time is kept under `meta:lastDefectSweepAt` in `cleaner_state`; a missed 05:00 slot runs about 30 seconds after startup.

## Deploy

```
DEPLOY_HOST=user@host DEPLOY_DIR=tabularium-v2 \
DEPLOY_HEALTH_URL=http://host:8789/v2/health \
bash v2/deploy/deploy.sh
```

Syncs the `v2/` working tree to `DEPLOY_DIR` on the host, runs `npm ci --omit=dev`, renders `deploy/tabularium-v2.service` with the remote directory and ssh user, installs it and restarts only that unit. It excludes `node_modules`, `*.log` and `eval/`. It ships whatever else is on disk, committed or not, and `rsync --delete` removes remote files that are not in `v2/`. Set `DEPLOY_UFW_CIDR` to open port 8789 to a source range with ufw. The script fails with a usage message when a required variable is unset.

Edit the unit before the first deploy: `TABULARIUM_HOST` binds to loopback, `TABULARIUM_STORE` points at `/var/lib/tabularium-v2/store` and must be writable by the service user, and the unit reads `TYPESAFE_API_KEY` from `/etc/tabularium/typesafe.env`. The unit also sets the fold, containment and `TABULARIUM_TOPICS_PATH` variables. The unit expects `topics.json` next to `package.json`; without it the topic label is skipped.

Starting the server against an older store adds the missing columns to each table and logs `{"event":"columns-added",...}`; a store that already has them is left alone.

## Legacy import

The server can serve a read-only archive of records that came from the v1 store. `src/scripts/importLegacy.ts` loads that archive from a JSON-lines file, one record per line. Work on a copy of the store, never the live one.

```
node src/scripts/importLegacy.ts [--replace] legacy.jsonl
```

Set `TABULARIUM_STORE` to the target store first. The import embeds each row and prints `{"event":"legacy-import","lines":..,"written":..,"replaced":..,"skipped":..}`. Without `--replace` it skips every `legacyId` already in the store. With `--replace` it overwrites rows whose kind, scope, headline, claim, why or rejected changed, re-embeds them, and counts them as `replaced`. The line format is the `LegacyLine` interface in `importLegacy.ts`.
