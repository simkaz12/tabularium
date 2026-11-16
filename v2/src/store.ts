import * as lancedb from "@lancedb/lancedb";
import type { Table } from "@lancedb/lancedb";
import { Bool, Field, FixedSizeList, Float32, Float64, Int32, Schema, Utf8 } from "apache-arrow";

import type { Proposal } from "./domain.ts";

export interface Store {
  readonly dim: number;
  readonly records: Table;
  readonly legacy: Table;
  readonly proposals: Table;
  readonly readLog: Table;
  readonly settings: Table;
  readonly cleanerState: Table;
  withLock<T>(work: () => Promise<T>): Promise<T>;
}

export interface Hit<Row> {
  row: Row;
  score: number;
}

export interface HybridParams {
  query: string;
  vector: number[];
  where?: string;
  limit: number;
}

const text = (name: string, nullable = true) => new Field(name, new Utf8(), nullable);
const int = (name: string) => new Field(name, new Int32(), true);

function recordFields(dim: number): Field[] {
  return [
    text("id", false),
    text("kind"),
    text("headline"),
    text("claim"),
    text("why"),
    text("rejected"),
    text("scope"),
    text("purpose"),
    text("topic"),
    text("audience"),
    text("reader"),
    text("stage"),
    text("where"),
    text("trigger"),
    text("leftBecause"),
    text("fixHint"),
    text("impact"),
    text("defectStatus"),
    text("takenThreadId"),
    text("takenLink"),
    text("takenAt"),
    text("closedBy"),
    text("closedAt"),
    text("lifecycle"),
    text("supersededBy"),
    text("supersessionReason"),
    text("retiredReason"),
    text("lifecycleNote"),
    text("threadId"),
    text("threadTitle"),
    text("projectId"),
    text("branch"),
    text("machine"),
    text("link"),
    text("excerpt"),
    text("origin"),
    text("legacyId"),
    int("reinforcedCount"),
    text("lastReinforcedAt"),
    text("createdAt"),
    text("approvedAt"),
    new Field("vector", new FixedSizeList(dim, new Field("item", new Float32(), true)), false),
  ];
}

function legacyFields(dim: number): Field[] {
  return [
    ...recordFields(dim),
    text("oldScope"),
    text("oldSource"),
    int("readCount"),
    text("originalCreatedAt"),
    new Field("reviewed", new Bool(), true),
    text("collapsedIds"),
  ];
}

const PROPOSAL_FIELDS: Field[] = [
  text("id", false),
  text("status"),
  text("kind"),
  text("headline"),
  text("scope"),
  text("origin"),
  text("purpose"),
  text("topic"),
  text("audience"),
  text("reader"),
  text("section"),
  text("targetRecordId"),
  text("targetAction"),
  text("createdAt"),
  text("resolvedAt"),
  text("fields"),
  text("provenance"),
  text("target"),
  text("checks"),
  text("flags"),
  text("neighbours"),
  text("suggestion"),
  text("resolution"),
  text("ask"),
];

const READ_LOG_FIELDS: Field[] = [
  text("id", false),
  text("ts"),
  text("query"),
  text("scope"),
  text("resultId"),
  text("source"),
  int("rank"),
  new Field("score", new Float64(), true),
];

const SETTINGS_FIELDS: Field[] = [
  text("key", false),
  text("value"),
  text("updatedAt"),
];

const CLEANER_STATE_FIELDS: Field[] = [
  text("id", false),
  text("kind"),
  text("fingerprint"),
  text("value"),
  text("expiresAt"),
  text("updatedAt"),
];

async function openOrCreate(db: lancedb.Connection, name: string, fields: Field[]): Promise<Table> {
  const names = await db.tableNames();

  if (names.includes(name)) {
    const table = await db.openTable(name);
    await ensureColumns(table, fields);

    return table;
  }

  return db.createEmptyTable(name, new Schema(fields));
}

export async function ensureColumns(table: Table, fields: Field[]): Promise<string[]> {
  const live = new Set((await table.schema()).fields.map((field) => field.name));
  const missing = fields.filter((field) => !live.has(field.name));

  if (missing.length === 0) {
    return [];
  }

  const unsupported = missing.filter((field) => !(field.type instanceof Utf8));

  if (unsupported.length > 0) {
    const names = unsupported.map((field) => field.name).join(", ");

    throw new Error(`table ${table.name} is missing non-string columns: ${names}`);
  }

  await table.addColumns(missing.map((field) => ({ name: field.name, valueSql: "cast(NULL as string)" })));

  const columns = missing.map((field) => field.name);
  console.log(JSON.stringify({ event: "columns-added", table: table.name, columns }));

  return columns;
}

async function ensureIndex(
  table: Table,
  column: string,
  config: ReturnType<typeof lancedb.Index.fts> | ReturnType<typeof lancedb.Index.btree>,
): Promise<void> {
  const existing = await table.listIndices();

  if (existing.some((index) => index.columns.includes(column))) {
    return;
  }

  await table.createIndex(column, { config });
}

class Mutex {
  private tail: Promise<unknown> = Promise.resolve();

  run<T>(work: () => Promise<T>): Promise<T> {
    const result = this.tail.then(work, work);
    this.tail = result.catch(() => undefined);
    return result;
  }
}

export async function openStore(path: string, dim: number): Promise<Store> {
  const db = await lancedb.connect(path, { readConsistencyInterval: 0 });
  const records = await openOrCreate(db, "records", recordFields(dim));
  const legacy = await openOrCreate(db, "legacy", legacyFields(dim));
  const proposals = await openOrCreate(db, "proposals", PROPOSAL_FIELDS);
  const readLog = await openOrCreate(db, "read_log", READ_LOG_FIELDS);
  const settings = await openOrCreate(db, "settings", SETTINGS_FIELDS);
  const cleanerState = await openOrCreate(db, "cleaner_state", CLEANER_STATE_FIELDS);

  for (const table of [records, legacy]) {
    await ensureIndex(table, "headline", lancedb.Index.fts());
    await ensureIndex(table, "claim", lancedb.Index.fts());
    await ensureIndex(table, "scope", lancedb.Index.btree());
    await ensureIndex(table, "lifecycle", lancedb.Index.btree());
  }

  await ensureIndex(proposals, "status", lancedb.Index.btree());

  const mutex = new Mutex();

  return {
    dim,
    records,
    legacy,
    proposals,
    readLog,
    settings,
    cleanerState,
    withLock: (work) => mutex.run(work),
  };
}

export function sqlString(value: string): string {
  return `'${value.replaceAll("'", "''")}'`;
}

export function scopeClause(scope: string | undefined | null): string | null {
  if (!scope) {
    return null;
  }

  return `(scope = ${sqlString(scope)} OR scope = 'global')`;
}

export function andAll(clauses: Array<string | null | undefined>): string | undefined {
  const present = clauses.filter((clause): clause is string => Boolean(clause));

  return present.length === 0 ? undefined : present.join(" AND ");
}

function toPlain<Row>(raw: Record<string, unknown>): Row {
  const row: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(raw)) {
    if (key === "vector" && value !== null && value !== undefined) {
      row[key] = Array.from(value as ArrayLike<number>);
    } else if (key === "_score" || key === "_relevance_score" || key === "_distance") {
      continue;
    } else {
      row[key] = value === undefined ? null : value;
    }
  }

  return row as Row;
}

export async function findById<Row>(table: Table, id: string): Promise<Row | undefined> {
  const rows = await table.query().where(`id = ${sqlString(id)}`).limit(1).toArray();
  const first = rows[0];

  return first === undefined ? undefined : toPlain<Row>(first);
}

export async function selectWhere<Row>(table: Table, where: string | undefined, limit = 10000): Promise<Row[]> {
  const query = table.query();

  if (where !== undefined) {
    query.where(where);
  }

  const rows = await query.limit(limit).toArray();

  return rows.map((raw) => toPlain<Row>(raw));
}

export async function upsert(table: Table, rows: Array<Record<string, unknown>>): Promise<void> {
  if (rows.length === 0) {
    return;
  }

  await table
    .mergeInsert("id")
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .execute(rows);
}

export async function insert(table: Table, rows: Array<Record<string, unknown>>): Promise<void> {
  if (rows.length === 0) {
    return;
  }

  await table.add(rows);
}

const RRF_K = 60;

export function ftsSafeText(query: string): string {
  return query
    .replace(/[^\p{L}\p{N}\s]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

async function vectorSearch<Row>(table: Table, params: HybridParams): Promise<Array<Hit<Row>>> {
  const search = table.query().nearestTo(params.vector).distanceType("cosine");

  if (params.where !== undefined) {
    search.where(params.where);
  }

  const raw = await search.limit(params.limit).toArray();

  return raw.map((entry, index) => ({
    row: toPlain<Row>(entry),
    score: 1 / (RRF_K + index),
  }));
}

export async function hybridSearch<Row>(table: Table, params: HybridParams): Promise<Array<Hit<Row>>> {
  const { query, vector, where, limit } = params;

  if (query.trim() === "" || limit <= 0) {
    return [];
  }

  const safe = ftsSafeText(query);

  if (safe === "") {
    return vectorSearch<Row>(table, params);
  }

  const rrf = await lancedb.rerankers.RRFReranker.create();
  const hybrid = table
    .query()
    .nearestTo(vector)
    .distanceType("cosine")
    .fullTextSearch(safe, { columns: ["headline", "claim"] })
    .rerank(rrf);

  if (where !== undefined) {
    hybrid.where(where);
  }

  const raw = await hybrid.limit(limit).toArray();

  return raw.map((entry) => ({
    row: toPlain<Row>(entry),
    score: Number(entry._relevance_score ?? 0),
  }));
}

export function proposalToRow(proposal: Proposal): Record<string, unknown> {
  return {
    id: proposal.id,
    status: proposal.status,
    kind: proposal.kind,
    headline: proposal.headline,
    scope: proposal.scope,
    origin: proposal.origin,
    purpose: proposal.purpose,
    topic: proposal.topic,
    audience: proposal.audience,
    reader: proposal.reader,
    targetRecordId: proposal.target?.recordId ?? null,
    targetAction: proposal.target?.action ?? null,
    createdAt: proposal.createdAt,
    resolvedAt: proposal.resolvedAt,
    fields: JSON.stringify(proposal.fields),
    provenance: JSON.stringify(proposal.provenance),
    target: JSON.stringify(proposal.target ?? null),
    checks: JSON.stringify(proposal.checks),
    flags: JSON.stringify(proposal.flags),
    neighbours: JSON.stringify(proposal.neighbours),
    suggestion: JSON.stringify(proposal.suggestion),
    resolution: JSON.stringify(proposal.resolution),
    ask: proposal.ask === null || proposal.ask === undefined ? null : JSON.stringify(proposal.ask),
  };
}

function parseJson<T>(raw: unknown, fallback: T): T {
  if (typeof raw !== "string") {
    return fallback;
  }

  return JSON.parse(raw) as T;
}

function withHeadlines(target: Proposal["target"]): Proposal["target"] {
  if (!target) {
    return null;
  }

  return {
    ...target,
    recordHeadline: target.recordHeadline ?? null,
    replacementHeadline: target.replacementHeadline ?? null,
  };
}

function withStage(provenance: Proposal["provenance"]): Proposal["provenance"] {
  return { ...provenance, stage: provenance.stage ?? "" };
}

export function rowToProposal(row: Record<string, unknown>): Proposal {
  return {
    id: row.id as string,
    status: row.status as Proposal["status"],
    kind: row.kind as Proposal["kind"],
    headline: row.headline as string,
    scope: row.scope as string,
    origin: row.origin as Proposal["origin"],
    purpose: (row.purpose as string | null) ?? null,
    topic: (row.topic as string | null) ?? null,
    audience: (row.audience as Proposal["audience"]) ?? null,
    reader: (row.reader as Proposal["reader"]) ?? null,
    createdAt: row.createdAt as string,
    resolvedAt: (row.resolvedAt as string | null) ?? null,
    fields: parseJson(row.fields, { claim: "" }),
    provenance: withStage(parseJson(row.provenance, {} as Proposal["provenance"])),
    target: withHeadlines(parseJson<Proposal["target"]>(row.target, null)),
    checks: parseJson(row.checks, { status: "skipped" } as Proposal["checks"]),
    flags: parseJson(row.flags, []),
    neighbours: parseJson(row.neighbours, []),
    suggestion: parseJson(row.suggestion, null),
    resolution: parseJson(row.resolution, null),
    ask: parseJson(row.ask, null),
  };
}
