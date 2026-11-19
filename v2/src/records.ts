import { randomUUID } from "node:crypto";

import type { Table } from "@lancedb/lancedb";

import { HttpError, REASON_CLASSES, recordText } from "./domain.ts";
import type {
  Audience,
  DefectStatus,
  Kind,
  LegacyRecord,
  MemoryRecord,
  Origin,
  Provenance,
  Reader,
  ReasonClass,
  Source,
  StoredRecord,
} from "./domain.ts";
import type { Embedder } from "./embed.ts";
import { findById, insert, selectWhere, sqlString, upsert } from "./store.ts";
import type { Store } from "./store.ts";

export interface Located {
  table: Table;
  source: Source;
  row: StoredRecord;
}

export interface NewRecord {
  id?: string;
  kind: Kind;
  headline: string;
  claim: string;
  why?: string | null;
  rejected: string | null;
  scope: string;
  purpose: string | null;
  topic: string | null;
  audience?: Audience | null;
  reader?: Reader | null;
  where?: string | null;
  trigger?: string | null;
  leftBecause?: string | null;
  fixHint?: string | null;
  impact?: string | null;
  defectStatus?: DefectStatus | null;
  provenance: Provenance;
  origin: Origin;
  legacyId: string | null;
  vector?: number[];
  reinforcedCount?: number;
  lastReinforcedAt?: string | null;
}

export type PublicRecord = Omit<StoredRecord, "vector">;

export interface TakenBy {
  threadId: string;
  link: string;
  at: string;
}

export interface RecordView extends PublicRecord {
  source: Source;
  takenBy: TakenBy | null;
  lineage: {
    supersededBy: { id: string; headline: string } | null;
    supersedes: Array<{ id: string; headline: string }>;
  };
}

export function stripVector(row: StoredRecord): PublicRecord {
  const { vector: _vector, ...rest } = row;

  return rest;
}

export function takenByOf(row: StoredRecord): TakenBy | null {
  if (row.takenThreadId == null || row.takenAt == null) {
    return null;
  }

  return { threadId: row.takenThreadId, link: row.takenLink ?? "", at: row.takenAt };
}

function parseTaker(raw: unknown): { threadId: string; link: string } {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new HttpError(400, "body must be an object");
  }

  const body = raw as Record<string, unknown>;

  return { threadId: requireText(body.threadId, "threadId"), link: requireText(body.link, "link") };
}

export function threadIdOf(raw: unknown): string | null {
  if (typeof raw === "object" && raw !== null && !Array.isArray(raw)) {
    const value = (raw as Record<string, unknown>).threadId;

    return typeof value === "string" && value !== "" ? value : null;
  }

  return null;
}

export interface ReleaseActor {
  threadId?: string | null;
  sweep?: boolean;
}

export type DefectEventName = "take" | "release" | "auto-release" | "close";

export function requireReasonClass(value: unknown): ReasonClass {
  if (typeof value === "string" && (REASON_CLASSES as readonly string[]).includes(value)) {
    return value as ReasonClass;
  }

  throw new HttpError(400, `reason must be one of ${REASON_CLASSES.join(", ")}`);
}

export function requireText(value: unknown, name: string): string {
  if (typeof value === "string" && value.trim() !== "") {
    return value.trim();
  }

  throw new HttpError(400, `${name} is required`);
}

export class RecordService {
  private readonly store: Store;
  private readonly embedder: Embedder;
  private readonly now: () => string;

  constructor(store: Store, embedder: Embedder, now: () => string = () => new Date().toISOString()) {
    this.store = store;
    this.embedder = embedder;
    this.now = now;
  }

  async locate(id: string): Promise<Located | undefined> {
    const current = await findById<MemoryRecord>(this.store.records, id);

    if (current !== undefined) {
      return { table: this.store.records, source: "v2", row: current };
    }

    const archived = await findById<LegacyRecord>(this.store.legacy, id);

    if (archived !== undefined) {
      return { table: this.store.legacy, source: "legacy", row: archived };
    }

    return undefined;
  }

  async require(id: string): Promise<Located> {
    const found = await this.locate(id);

    if (found === undefined) {
      throw new HttpError(404, `record ${id} not found`);
    }

    return found;
  }

  async get(id: string): Promise<RecordView> {
    const found = await this.require(id);
    const row = found.row;
    const replacement =
      row.supersededBy === null ? undefined : await this.locate(row.supersededBy);
    const predecessors = await this.predecessors(id);

    return {
      ...stripVector(row),
      source: found.source,
      takenBy: takenByOf(row),
      lineage: {
        supersededBy:
          replacement === undefined
            ? null
            : { id: replacement.row.id, headline: replacement.row.headline },
        supersedes: predecessors,
      },
    };
  }

  async read(id: string, threadId?: string | null): Promise<RecordView> {
    const view = await this.get(id);

    await this.logRead(view, threadId ?? null);

    return view;
  }

  private async logRead(view: RecordView, threadId: string | null): Promise<void> {
    try {
      await insert(this.store.readLog, [
        {
          id: randomUUID(),
          ts: this.now(),
          query: null,
          scope: view.scope,
          resultId: view.id,
          source: view.source,
          rank: null,
          score: null,
          threadId,
          task: null,
          via: "get",
        },
      ]);
    } catch (error) {
      console.error(JSON.stringify({ event: "read-log-failed", id: view.id, error: String(error) }));
    }
  }

  private async logDefectEvent(event: {
    recordId: string;
    event: DefectEventName;
    threadId?: string | null;
    link?: string | null;
    proposalId?: string | null;
  }): Promise<void> {
    try {
      await insert(this.store.defectEvents, [
        {
          id: randomUUID(),
          ts: this.now(),
          recordId: event.recordId,
          event: event.event,
          threadId: event.threadId ?? null,
          link: event.link ?? null,
          proposalId: event.proposalId ?? null,
        },
      ]);
    } catch (error) {
      console.error(
        JSON.stringify({ event: "defect-event-failed", recordId: event.recordId, error: String(error) }),
      );
    }
  }

  async requireClosableDefect(id: string): Promise<Located> {
    const found = await this.require(id);

    if (found.source !== "v2" || found.row.kind !== "defect") {
      throw new HttpError(409, `record ${id} is not a v2 defect`);
    }

    if (found.row.defectStatus === "closed") {
      throw new HttpError(409, `defect ${id} is already closed`);
    }

    return found;
  }

  private async requireDefect(id: string): Promise<Located> {
    const found = await this.require(id);

    if (found.source !== "v2" || found.row.kind !== "defect") {
      throw new HttpError(400, `record ${id} is not a defect`);
    }

    return found;
  }

  async take(id: string, raw: unknown): Promise<RecordView> {
    const taker = parseTaker(raw);

    return this.store.withLock(async () => {
      const found = await this.requireDefect(id);

      if (found.row.defectStatus === "closed") {
        throw new HttpError(409, `defect ${id} is closed`);
      }

      if (found.row.defectStatus === "taken") {
        throw new HttpError(409, `defect ${id} is already taken`);
      }

      const updated = {
        ...found.row,
        defectStatus: "taken" as const,
        takenThreadId: taker.threadId,
        takenLink: taker.link,
        takenAt: this.now(),
      };
      await upsert(found.table, [updated as unknown as Record<string, unknown>]);
      await this.logDefectEvent({ recordId: id, event: "take", threadId: taker.threadId, link: taker.link });

      return this.get(id);
    });
  }

  async release(id: string, actor: ReleaseActor = {}): Promise<RecordView> {
    return this.store.withLock(async () => {
      const found = await this.requireDefect(id);

      if (found.row.defectStatus === "taken") {
        const updated = {
          ...found.row,
          defectStatus: "open" as const,
          takenThreadId: null,
          takenLink: null,
        };
        await upsert(found.table, [updated as unknown as Record<string, unknown>]);
        await this.logDefectEvent({
          recordId: id,
          event: actor.sweep === true ? "auto-release" : "release",
          threadId: actor.sweep === true ? null : (actor.threadId ?? null),
        });
      }

      return this.get(id);
    });
  }

  async markClosed(id: string, byId: string, threadId: string | null = null): Promise<StoredRecord> {
    const found = await this.requireClosableDefect(id);
    const updated = {
      ...found.row,
      defectStatus: "closed" as const,
      closedBy: byId,
      closedAt: this.now(),
    };
    await upsert(found.table, [updated as unknown as Record<string, unknown>]);
    await this.logDefectEvent({ recordId: id, event: "close", threadId, proposalId: byId });

    return updated;
  }

  private async predecessors(id: string): Promise<Array<{ id: string; headline: string }>> {
    const where = `supersededBy = ${sqlString(id)}`;
    const current = await selectWhere<MemoryRecord>(this.store.records, where);
    const archived = await selectWhere<LegacyRecord>(this.store.legacy, where);

    return [...current, ...archived].map((row) => ({ id: row.id, headline: row.headline }));
  }

  async insertRecord(input: NewRecord): Promise<MemoryRecord> {
    if (input.id !== undefined) {
      const existing = await findById<MemoryRecord>(this.store.records, input.id);

      if (existing !== undefined) {
        return existing;
      }
    }

    const vector = input.vector ?? (await this.embedder.embed(recordText(input)));
    const stamp = this.now();
    const record: MemoryRecord = {
      id: input.id ?? randomUUID(),
      kind: input.kind,
      headline: input.headline,
      claim: input.claim,
      why: input.why ?? null,
      rejected: input.rejected,
      scope: input.scope,
      purpose: input.purpose,
      topic: input.topic,
      audience: input.audience ?? null,
      reader: input.reader ?? null,
      where: input.where ?? null,
      trigger: input.trigger ?? null,
      leftBecause: input.leftBecause ?? null,
      fixHint: input.fixHint ?? null,
      impact: input.impact ?? null,
      defectStatus: input.defectStatus ?? null,
      takenThreadId: null,
      takenLink: null,
      takenAt: null,
      closedBy: null,
      closedAt: null,
      lifecycle: "live",
      supersededBy: null,
      supersessionReason: null,
      retiredReason: null,
      lifecycleNote: null,
      ...input.provenance,
      stage: input.provenance.stage ?? "",
      origin: input.origin,
      legacyId: input.legacyId,
      reinforcedCount: input.reinforcedCount ?? 0,
      lastReinforcedAt: input.lastReinforcedAt ?? null,
      createdAt: stamp,
      approvedAt: stamp,
      vector,
    };

    await upsert(this.store.records, [record as unknown as Record<string, unknown>]);

    return record;
  }

  async markSuperseded(oldId: string, byId: string, reason: ReasonClass): Promise<StoredRecord> {
    if (oldId === byId) {
      throw new HttpError(409, "a record cannot supersede itself");
    }

    const old = await this.require(oldId);
    const replacement = await this.require(byId);

    if (old.row.lifecycle !== "live") {
      throw new HttpError(409, `record ${oldId} is ${old.row.lifecycle}, not live`);
    }

    if (replacement.row.lifecycle !== "live") {
      throw new HttpError(409, `replacement ${byId} is ${replacement.row.lifecycle}, not live`);
    }

    if (old.source === "legacy") {
      return this.writeLegacyRetirement(old, byId, `superseded by ${byId}`, reason);
    }

    const updated = {
      ...old.row,
      lifecycle: "superseded" as const,
      supersededBy: byId,
      supersessionReason: reason,
    };
    await upsert(old.table, [updated as unknown as Record<string, unknown>]);

    return updated;
  }

  async retireLegacy(id: string, byId: string | null, reason: string): Promise<StoredRecord> {
    const found = await this.require(id);

    if (found.source !== "legacy") {
      throw new HttpError(409, `record ${id} is not a legacy record`);
    }

    if (found.row.lifecycle === "retired" && found.row.supersededBy === byId) {
      return found.row;
    }

    if (found.row.lifecycle !== "live") {
      throw new HttpError(409, `record ${id} is ${found.row.lifecycle}, not live`);
    }

    if (byId !== null) {
      const replacement = await this.require(byId);

      if (replacement.row.lifecycle !== "live") {
        throw new HttpError(409, `replacement ${byId} is ${replacement.row.lifecycle}, not live`);
      }
    }

    return this.writeLegacyRetirement(found, byId, reason, null);
  }

  private async writeLegacyRetirement(
    found: Located,
    byId: string | null,
    retiredReason: string,
    supersessionReason: ReasonClass | null,
  ): Promise<StoredRecord> {
    const updated = {
      ...found.row,
      lifecycle: "retired" as const,
      supersededBy: byId,
      supersessionReason,
      retiredReason,
    };
    await upsert(found.table, [updated as unknown as Record<string, unknown>]);

    return updated;
  }

  async markRetired(id: string, reason: string): Promise<StoredRecord> {
    const found = await this.require(id);

    if (found.row.lifecycle !== "live") {
      throw new HttpError(409, `record ${id} is ${found.row.lifecycle}, not live`);
    }

    const updated = {
      ...found.row,
      lifecycle: "retired" as const,
      retiredReason: reason,
    };
    await upsert(found.table, [updated as unknown as Record<string, unknown>]);

    return updated;
  }

  async bumpReinforcement(id: string): Promise<StoredRecord> {
    const found = await this.require(id);

    if (found.row.lifecycle !== "live") {
      throw new HttpError(409, `record ${id} is ${found.row.lifecycle}, not live`);
    }

    const updated = {
      ...found.row,
      reinforcedCount: (found.row.reinforcedCount ?? 0) + 1,
      lastReinforcedAt: this.now(),
    };
    await upsert(found.table, [updated as unknown as Record<string, unknown>]);

    return updated;
  }

  async unsupersede(id: string, reason: unknown): Promise<RecordView> {
    const note = requireText(reason, "reason");

    return this.store.withLock(async () => {
      const found = await this.require(id);

      await this.assertRestorable(found);

      const updated = {
        ...found.row,
        lifecycle: "live" as const,
        supersededBy: null,
        supersessionReason: null,
        retiredReason: found.source === "legacy" ? null : found.row.retiredReason,
        lifecycleNote: note,
      };
      await upsert(found.table, [updated as unknown as Record<string, unknown>]);

      return this.get(id);
    });
  }

  async archive(id: string, reason: unknown): Promise<RecordView> {
    const note = requireText(reason, "reason");

    return this.store.withLock(async () => {
      const found = await this.require(id);

      if (found.row.kind === "defect") {
        throw new HttpError(409, `record ${id} is a defect and cannot be archived`);
      }

      if (found.row.lifecycle !== "live") {
        throw new HttpError(409, `record ${id} is ${found.row.lifecycle}, not live`);
      }

      const updated = {
        ...found.row,
        lifecycle: "archived" as const,
        lifecycleNote: note,
      };
      await upsert(found.table, [updated as unknown as Record<string, unknown>]);

      return this.get(id);
    });
  }

  async unarchive(id: string): Promise<RecordView> {
    return this.store.withLock(async () => {
      const found = await this.require(id);

      if (found.row.lifecycle !== "archived") {
        throw new HttpError(409, `record ${id} is ${found.row.lifecycle}, not archived`);
      }

      const updated = {
        ...found.row,
        lifecycle: "live" as const,
        lifecycleNote: null,
      };
      await upsert(found.table, [updated as unknown as Record<string, unknown>]);

      return this.get(id);
    });
  }

  private async assertRestorable(found: Located): Promise<void> {
    const lifecycle = found.row.lifecycle;

    if (found.source !== "legacy") {
      if (lifecycle !== "superseded") {
        throw new HttpError(409, `record ${found.row.id} is ${lifecycle}, not superseded`);
      }

      return;
    }

    if (lifecycle === "live" || lifecycle === "archived") {
      throw new HttpError(409, `record ${found.row.id} is ${lifecycle}, not superseded or retired`);
    }

    const promoted = await selectWhere<MemoryRecord>(
      this.store.records,
      `legacyId = ${sqlString(found.row.id)}`,
      1,
    );

    if (promoted.length > 0) {
      throw new HttpError(409, `record ${found.row.id} was promoted to ${promoted[0]?.id}`);
    }
  }

  async remove(id: string, reason: unknown): Promise<{ id: string; deleted: true }> {
    const note = requireText(reason, "reason");

    return this.store.withLock(async () => {
      const found = await this.require(id);
      const dependents = await this.dependents(id);

      if (dependents.length > 0) {
        throw new HttpError(
          409,
          `record ${id} is the replacement for ${dependents.join(", ")}; restore or delete those first`,
        );
      }

      await found.table.delete(`id = ${sqlString(id)}`);
      console.log(JSON.stringify({ event: "record-deleted", id, source: found.source, reason: note }));

      return { id, deleted: true as const };
    });
  }

  private async dependents(id: string): Promise<string[]> {
    const where = `supersededBy = ${sqlString(id)}`;
    const [current, archived] = await Promise.all([
      selectWhere<MemoryRecord>(this.store.records, where),
      selectWhere<LegacyRecord>(this.store.legacy, where),
    ]);

    return [...current, ...archived].map((row) => row.id);
  }
}
