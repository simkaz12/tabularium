import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { loadConfig } from "../config.ts";
import { KINDS, recordText } from "../domain.ts";
import type { Kind, LegacyRecord, Lifecycle } from "../domain.ts";
import { OllamaEmbedder } from "../embed.ts";
import type { Embedder } from "../embed.ts";
import { insert, openStore, selectWhere, sqlString, upsert } from "../store.ts";
import type { Store } from "../store.ts";

export interface LegacyLine {
  legacyId: string;
  kind: Kind;
  headline: string;
  claim: string;
  why: string;
  rejected: string | null;
  scope: string;
  oldScope: string;
  oldSource: string;
  purpose: string | null;
  topic: string | null;
  lifecycle: Lifecycle;
  readCount: number;
  originalCreatedAt: string;
  createdAt: string;
  reviewed: boolean;
  collapsedIds: string[];
}

export interface ImportReport {
  lines: number;
  written: number;
  replaced: number;
  skipped: number;
}

export interface ImportOptions {
  batchSize?: number;
  replace?: boolean;
}

export interface CliArgs {
  path: string;
  replace: boolean;
}

const LIFECYCLES: readonly Lifecycle[] = ["live", "superseded", "retired"];
const DEFAULT_BATCH = 50;
const LOOKUP_CHUNK = 200;

function fail(lineNumber: number, message: string): never {
  throw new Error(`legacy.jsonl line ${lineNumber}: ${message}`);
}

function requireString(raw: Record<string, unknown>, key: string, lineNumber: number): string {
  const value = raw[key];

  if (typeof value !== "string" || value.trim() === "") {
    return fail(lineNumber, `${key} must be a non-empty string`);
  }

  return value;
}

function textField(raw: Record<string, unknown>, key: string, lineNumber: number): string {
  const value = raw[key];

  if (typeof value !== "string") {
    return fail(lineNumber, `${key} must be a string`);
  }

  return value;
}

function optionalString(raw: Record<string, unknown>, key: string, lineNumber: number): string | null {
  const value = raw[key];

  if (value === null || value === undefined) {
    return null;
  }

  if (typeof value !== "string") {
    return fail(lineNumber, `${key} must be a string or null`);
  }

  return value;
}

function requireCount(raw: Record<string, unknown>, key: string, lineNumber: number): number {
  const value = raw[key];

  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    return fail(lineNumber, `${key} must be a non-negative integer`);
  }

  return value;
}

function requireStringList(raw: Record<string, unknown>, key: string, lineNumber: number): string[] {
  const value = raw[key];

  if (!Array.isArray(value) || value.some((entry) => typeof entry !== "string")) {
    return fail(lineNumber, `${key} must be a list of strings`);
  }

  return value as string[];
}

export function parseLine(text: string, lineNumber: number): LegacyLine {
  let parsed: unknown;

  try {
    parsed = JSON.parse(text);
  } catch {
    return fail(lineNumber, "not valid JSON");
  }

  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    return fail(lineNumber, "not a JSON object");
  }

  const raw = parsed as Record<string, unknown>;
  const kind = requireString(raw, "kind", lineNumber);
  const lifecycle = requireString(raw, "lifecycle", lineNumber);

  if (!KINDS.includes(kind as Kind)) {
    return fail(lineNumber, `kind must be one of ${KINDS.join(", ")}`);
  }

  if (!LIFECYCLES.includes(lifecycle as Lifecycle)) {
    return fail(lineNumber, `lifecycle must be one of ${LIFECYCLES.join(", ")}`);
  }

  if (typeof raw.reviewed !== "boolean") {
    return fail(lineNumber, "reviewed must be a boolean");
  }

  return {
    legacyId: requireString(raw, "legacyId", lineNumber),
    kind: kind as Kind,
    headline: requireString(raw, "headline", lineNumber),
    claim: requireString(raw, "claim", lineNumber),
    why: textField(raw, "why", lineNumber),
    rejected: optionalString(raw, "rejected", lineNumber),
    scope: requireString(raw, "scope", lineNumber),
    oldScope: requireString(raw, "oldScope", lineNumber),
    oldSource: textField(raw, "oldSource", lineNumber),
    purpose: optionalString(raw, "purpose", lineNumber),
    topic: optionalString(raw, "topic", lineNumber),
    lifecycle: lifecycle as Lifecycle,
    readCount: requireCount(raw, "readCount", lineNumber),
    originalCreatedAt: requireString(raw, "originalCreatedAt", lineNumber),
    createdAt: requireString(raw, "createdAt", lineNumber),
    reviewed: raw.reviewed,
    collapsedIds: requireStringList(raw, "collapsedIds", lineNumber),
  };
}

export function readLegacyFile(path: string): LegacyLine[] {
  const lines: LegacyLine[] = [];
  const seen = new Set<string>();
  const entries = readFileSync(path, "utf8").split("\n");

  entries.forEach((entry, index) => {
    if (entry.trim() === "") {
      return;
    }

    const line = parseLine(entry, index + 1);

    if (seen.has(line.legacyId)) {
      fail(index + 1, `duplicate legacyId ${line.legacyId}`);
    }

    seen.add(line.legacyId);
    lines.push(line);
  });

  return lines;
}

async function existingRows(store: Store, ids: string[]): Promise<Map<string, LegacyRecord>> {
  const found = new Map<string, LegacyRecord>();

  for (let start = 0; start < ids.length; start += LOOKUP_CHUNK) {
    const chunk = ids.slice(start, start + LOOKUP_CHUNK);
    const where = `legacyId IN (${chunk.map(sqlString).join(", ")})`;
    const rows = await selectWhere<LegacyRecord>(store.legacy, where, chunk.length);

    for (const row of rows) {
      found.set(row.legacyId ?? row.id, row);
    }
  }

  return found;
}

export function toLegacyRow(line: LegacyLine, vector: number[]): LegacyRecord {
  return {
    id: line.legacyId,
    kind: line.kind,
    headline: line.headline,
    claim: line.claim,
    why: line.why,
    rejected: line.rejected,
    scope: line.scope,
    purpose: line.purpose,
    topic: line.topic,
    audience: null,
    reader: null,
    where: null,
    trigger: null,
    leftBecause: null,
    fixHint: null,
    impact: null,
    defectStatus: null,
    takenThreadId: null,
    takenLink: null,
    takenAt: null,
    closedBy: null,
    closedAt: null,
    lifecycle: line.lifecycle,
    supersededBy: null,
    supersessionReason: null,
    retiredReason: null,
    lifecycleNote: null,
    threadId: "",
    threadTitle: "",
    projectId: "",
    branch: "",
    machine: "",
    link: "",
    excerpt: "",
    stage: "",
    origin: "agent",
    legacyId: line.legacyId,
    reinforcedCount: 0,
    lastReinforcedAt: null,
    createdAt: line.createdAt,
    approvedAt: line.createdAt,
    vector,
    oldScope: line.oldScope,
    oldSource: line.oldSource,
    readCount: line.readCount,
    originalCreatedAt: line.originalCreatedAt,
    reviewed: line.reviewed,
    collapsedIds: line.collapsedIds.length === 0 ? null : JSON.stringify(line.collapsedIds),
  };
}

function hasChanged(line: LegacyLine, row: LegacyRecord): boolean {
  return (
    row.kind !== line.kind ||
    row.scope !== line.scope ||
    row.headline !== line.headline ||
    row.claim !== line.claim ||
    row.why !== line.why ||
    (row.rejected ?? null) !== line.rejected
  );
}

function replacementRow(line: LegacyLine, existing: LegacyRecord, vector: number[]): LegacyRecord {
  return {
    ...toLegacyRow(line, vector),
    lifecycle: existing.lifecycle,
    supersededBy: existing.supersededBy,
    supersessionReason: existing.supersessionReason,
    retiredReason: existing.retiredReason,
    lifecycleNote: existing.lifecycleNote,
    reinforcedCount: existing.reinforcedCount,
    lastReinforcedAt: existing.lastReinforcedAt,
  };
}

async function embedAll(embedder: Embedder, lines: LegacyLine[]): Promise<number[][]> {
  const vectors: number[][] = [];

  for (const line of lines) {
    vectors.push(await embedder.embed(recordText(line)));
  }

  return vectors;
}

export async function importLegacy(
  store: Store,
  embedder: Embedder,
  lines: LegacyLine[],
  options: ImportOptions = {},
): Promise<ImportReport> {
  const batchSize = options.batchSize ?? DEFAULT_BATCH;
  const present = await existingRows(store, lines.map((line) => line.legacyId));
  const pending = lines.filter((line) => !present.has(line.legacyId));
  const changed = options.replace === true
    ? lines.filter((line) => {
        const row = present.get(line.legacyId);

        return row !== undefined && hasChanged(line, row);
      })
    : [];
  let written = 0;
  let replaced = 0;

  for (let start = 0; start < pending.length; start += batchSize) {
    const batch = pending.slice(start, start + batchSize);
    const vectors = await embedAll(embedder, batch);
    const rows = batch.map((line, index) => toLegacyRow(line, vectors[index] as number[]));

    await insert(store.legacy, rows as unknown as Array<Record<string, unknown>>);
    written += rows.length;
  }

  for (let start = 0; start < changed.length; start += batchSize) {
    const batch = changed.slice(start, start + batchSize);
    const vectors = await embedAll(embedder, batch);
    const rows = batch.map((line, index) =>
      replacementRow(line, present.get(line.legacyId) as LegacyRecord, vectors[index] as number[]),
    );

    await upsert(store.legacy, rows as unknown as Array<Record<string, unknown>>);
    replaced += rows.length;
  }

  return {
    lines: lines.length,
    written,
    replaced,
    skipped: lines.length - written - replaced,
  };
}

export function parseCliArgs(argv: string[]): CliArgs {
  const usage = "usage: node src/scripts/importLegacy.ts [--replace] <legacy.jsonl>";
  const args = argv.slice(2);
  const unknown = args.filter((arg) => arg.startsWith("-") && arg !== "--replace");
  const paths = args.filter((arg) => !arg.startsWith("-"));
  const [path] = paths;

  if (unknown.length > 0 || paths.length !== 1 || path === undefined) {
    throw new Error(usage);
  }

  return { path, replace: args.includes("--replace") };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const config = loadConfig();
  const cli = parseCliArgs(process.argv);
  const lines = readLegacyFile(cli.path);
  const store = await openStore(config.storePath, config.embedDim);
  const embedder = new OllamaEmbedder(
    config.embedUrl,
    config.embedModel,
    config.embedDim,
    fetch,
    config.embedTimeoutMs,
  );
  const report = await importLegacy(store, embedder, lines, { replace: cli.replace });

  console.log(JSON.stringify({ event: "legacy-import", ...report }));
}
