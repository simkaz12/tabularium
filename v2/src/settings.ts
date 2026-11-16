import { CLEANER_VERDICT_TTL_DAYS, DEFAULT_CLEANER_CONFIDENCE_FLOOR } from "./config.ts";
import { HttpError } from "./domain.ts";
import { findById, sqlString, upsert } from "./store.ts";
import type { Store } from "./store.ts";

export interface Settings {
  cleanerConfidenceFloor: number;
}

export interface RecordStamp {
  id: string;
  updatedAt: string;
}

export interface CleanerState {
  rememberPair(a: RecordStamp, b: RecordStamp): Promise<void>;
  isPairRemembered(a: RecordStamp, b: RecordStamp): Promise<boolean>;
  rememberRecord(record: RecordStamp): Promise<void>;
  isRecordRemembered(record: RecordStamp): Promise<boolean>;
  getLastRunAt(key?: string): Promise<string | null>;
  setLastRunAt(at: string, key?: string): Promise<void>;
  getLastDoubtsAt(): Promise<string | null>;
  setLastDoubtsAt(at: string): Promise<void>;
}

export interface RunClock {
  getLastRunAt(): Promise<string | null>;
  setLastRunAt(at: string): Promise<void>;
}

export function keyedRunClock(state: CleanerState, key: string): RunClock {
  return {
    getLastRunAt: () => state.getLastRunAt(key),
    setLastRunAt: (at) => state.setLastRunAt(at, key),
  };
}

interface SettingRow {
  key: string;
  value: string | null;
  updatedAt: string | null;
}

interface StateRow {
  id: string;
  kind: string | null;
  fingerprint: string | null;
  value: string | null;
  expiresAt: string | null;
  updatedAt: string | null;
}

const FLOOR_KEY = "cleanerConfidenceFloor";
const LAST_RUN_ID = "meta:lastRunAt";
const LAST_DOUBTS_ID = "meta:lastDoubtsAt";
const DAY_MS = 24 * 60 * 60 * 1000;

export class SettingsService {
  private readonly store: Store;
  private readonly now: () => string;

  constructor(store: Store, now: () => string = () => new Date().toISOString()) {
    this.store = store;
    this.now = now;
  }

  async getSettings(): Promise<Settings> {
    const row = await findByKey(this.store, FLOOR_KEY);
    const stored = row === undefined ? undefined : parseFloor(row.value);

    return { cleanerConfidenceFloor: stored ?? DEFAULT_CLEANER_CONFIDENCE_FLOOR };
  }

  async updateSettings(patch: unknown): Promise<Settings> {
    const floor = validatePatch(patch);

    await this.store.withLock(async () => {
      await this.store.settings
        .mergeInsert("key")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute([{ key: FLOOR_KEY, value: JSON.stringify(floor), updatedAt: this.now() }]);
    });

    return this.getSettings();
  }
}

export class StoreCleanerState implements CleanerState {
  private readonly store: Store;
  private readonly now: () => string;

  constructor(store: Store, now: () => string = () => new Date().toISOString()) {
    this.store = store;
    this.now = now;
  }

  async rememberPair(a: RecordStamp, b: RecordStamp): Promise<void> {
    const [first, second] = orderPair(a, b);

    await this.remember(pairId(first, second), "pair", pairFingerprint(first, second));
  }

  async isPairRemembered(a: RecordStamp, b: RecordStamp): Promise<boolean> {
    const [first, second] = orderPair(a, b);

    return this.isRemembered(pairId(first, second), pairFingerprint(first, second));
  }

  async rememberRecord(record: RecordStamp): Promise<void> {
    await this.remember(recordId(record), "record", record.updatedAt);
  }

  async isRecordRemembered(record: RecordStamp): Promise<boolean> {
    return this.isRemembered(recordId(record), record.updatedAt);
  }

  async getLastRunAt(key: string = LAST_RUN_ID): Promise<string | null> {
    const row = await findById<StateRow>(this.store.cleanerState, key);

    return row?.value ?? null;
  }

  async setLastRunAt(at: string, key: string = LAST_RUN_ID): Promise<void> {
    await this.store.withLock(async () => {
      await upsert(this.store.cleanerState, [
        {
          id: key,
          kind: "meta",
          fingerprint: null,
          value: at,
          expiresAt: null,
          updatedAt: this.now(),
        },
      ]);
    });
  }

  async getLastDoubtsAt(): Promise<string | null> {
    return this.getLastRunAt(LAST_DOUBTS_ID);
  }

  async setLastDoubtsAt(at: string): Promise<void> {
    await this.setLastRunAt(at, LAST_DOUBTS_ID);
  }

  private async remember(id: string, kind: string, fingerprint: string): Promise<void> {
    const stamp = this.now();
    const expiresAt = new Date(Date.parse(stamp) + CLEANER_VERDICT_TTL_DAYS * DAY_MS).toISOString();

    await this.store.withLock(async () => {
      await upsert(this.store.cleanerState, [
        { id, kind, fingerprint, value: null, expiresAt, updatedAt: stamp },
      ]);
    });
  }

  private async isRemembered(id: string, fingerprint: string): Promise<boolean> {
    const row = await findById<StateRow>(this.store.cleanerState, id);

    if (row === undefined) {
      return false;
    }

    if (row.fingerprint !== fingerprint) {
      return false;
    }

    return row.expiresAt !== null && Date.parse(row.expiresAt) > Date.parse(this.now());
  }
}

function orderPair(a: RecordStamp, b: RecordStamp): [RecordStamp, RecordStamp] {
  return a.id <= b.id ? [a, b] : [b, a];
}

function pairId(first: RecordStamp, second: RecordStamp): string {
  return `pair:${first.id}|${second.id}`;
}

function pairFingerprint(first: RecordStamp, second: RecordStamp): string {
  return `${first.updatedAt}|${second.updatedAt}`;
}

function recordId(record: RecordStamp): string {
  return `record:${record.id}`;
}

async function findByKey(store: Store, key: string): Promise<SettingRow | undefined> {
  const rows = await store.settings.query().where(`key = ${sqlString(key)}`).limit(1).toArray();
  const first = rows[0];

  return first === undefined ? undefined : (first as unknown as SettingRow);
}

function parseFloor(raw: string | null): number | undefined {
  if (raw === null) {
    return undefined;
  }

  const value: unknown = JSON.parse(raw);

  return typeof value === "number" ? value : undefined;
}

function validatePatch(patch: unknown): number {
  if (typeof patch !== "object" || patch === null || Array.isArray(patch)) {
    throw new HttpError(400, "body must be an object");
  }

  const keys = Object.keys(patch);
  const unknown = keys.filter((key) => key !== FLOOR_KEY);

  if (unknown.length > 0) {
    throw new HttpError(400, `unknown setting: ${unknown.join(", ")}`);
  }

  if (keys.length === 0) {
    throw new HttpError(400, `${FLOOR_KEY} is required`);
  }

  const value = (patch as Record<string, unknown>)[FLOOR_KEY];

  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new HttpError(400, `${FLOOR_KEY} must be a number between 0 and 1`);
  }

  return value;
}
