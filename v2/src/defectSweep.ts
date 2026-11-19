import { hostname } from "node:os";

import type { Mode } from "./config.ts";
import { DEFECT_ASK_CAP, DEFECT_STALE_DAYS, DEFECT_TAKEN_DAYS } from "./config.ts";
import { HEADLINE_MAX } from "./domain.ts";
import type { MemoryRecord } from "./domain.ts";
import type { ProposalService } from "./proposals.ts";
import type { RecordService } from "./records.ts";
import { andAll, selectWhere, sqlString } from "./store.ts";
import type { Store } from "./store.ts";

const DAY_MS = 86_400_000;
const ASK_PREFIX = "Still open? ";

export interface DefectSweepOptions {
  store: Store;
  records: Pick<RecordService, "release">;
  proposals: Pick<ProposalService, "createInternal">;
  mode?: Mode;
  now?: () => Date;
  log?: (message: string) => void;
}

export interface DefectSweepReport {
  released: number;
  asked: number;
}

export async function sweepDefects(options: DefectSweepOptions): Promise<DefectSweepReport> {
  const now = options.now ?? (() => new Date());
  const log = options.log ?? ((message) => console.error(message));
  const released = await releaseStaleTakes(options, now(), log);
  const asked = options.mode === "standalone" ? 0 : await askAboutStaleOpen(options, now(), released, log);

  return { released: released.size, asked };
}

function daysAgo(now: Date, days: number): string {
  return new Date(now.getTime() - days * DAY_MS).toISOString();
}

async function releaseStaleTakes(
  options: DefectSweepOptions,
  now: Date,
  log: (message: string) => void,
): Promise<Set<string>> {
  const where = andAll([
    "kind = 'defect'",
    "lifecycle = 'live'",
    "defectStatus = 'taken'",
    `takenAt < ${sqlString(daysAgo(now, DEFECT_TAKEN_DAYS))}`,
  ]);
  const stale = await selectWhere<{ id: string }>(options.store.records, where);
  const released = new Set<string>();

  for (const { id } of stale) {
    try {
      await options.records.release(id, { sweep: true });
      released.add(id);
    } catch (error) {
      log(`defect-sweep: releasing ${id} failed: ${String(error)}`);
    }
  }

  return released;
}

async function askAboutStaleOpen(
  options: DefectSweepOptions,
  now: Date,
  skip: Set<string>,
  log: (message: string) => void,
): Promise<number> {
  const cutoff = sqlString(daysAgo(now, DEFECT_STALE_DAYS));
  const where = andAll([
    "kind = 'defect'",
    "lifecycle = 'live'",
    "defectStatus = 'open'",
    `createdAt < ${cutoff}`,
    `(takenAt IS NULL OR takenAt < ${cutoff})`,
  ]);
  const stale = await selectWhere<MemoryRecord>(options.store.records, where);
  const blocked = await blockedRecordIds(options.store, now);
  const oldestFirst = stale.sort((a, b) => (a.createdAt < b.createdAt ? -1 : 1));
  let asked = 0;

  for (const record of oldestFirst) {
    if (asked >= DEFECT_ASK_CAP) {
      break;
    }

    if (skip.has(record.id) || blocked.has(record.id)) {
      continue;
    }

    try {
      await options.proposals.createInternal(closeBody(record, now));
      blocked.add(record.id);
      asked += 1;
    } catch (error) {
      log(`defect-sweep: creating the close tile for ${record.id} failed: ${String(error)}`);
    }
  }

  return asked;
}

async function blockedRecordIds(store: Store, now: Date): Promise<Set<string>> {
  const since = sqlString(daysAgo(now, DEFECT_STALE_DAYS));
  const rows = await selectWhere<{ targetRecordId: string | null }>(
    store.proposals,
    `status = 'pending' OR (status = 'rejected' AND resolvedAt >= ${since})`,
  );
  const blocked = new Set<string>();

  for (const row of rows) {
    if (row.targetRecordId !== null) {
      blocked.add(row.targetRecordId);
    }
  }

  return blocked;
}

function askHeadline(headline: string): string {
  const room = HEADLINE_MAX - ASK_PREFIX.length;
  const cut = headline.length > room ? `${headline.slice(0, room - 1)}…` : headline;

  return `${ASK_PREFIX}${cut}`;
}

function closeBody(record: MemoryRecord, now: Date) {
  return {
    kind: "defect",
    headline: askHeadline(record.headline),
    fields: { claim: record.claim },
    scope: record.scope,
    provenance: {
      threadId: "cleaner",
      threadTitle: `Defect sweep ${now.toISOString().slice(0, 10)}`,
      projectId: "",
      branch: "",
      machine: hostname(),
      link: "",
      excerpt: "",
      stage: "",
    },
    origin: "cleaner",
    target: {
      recordId: record.id,
      source: "v2",
      action: "close",
      reason: "open 90d",
    },
  };
}
