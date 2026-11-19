import { randomUUID } from "node:crypto";

import { AUDIENCES, HttpError, KINDS } from "./domain.ts";
import type {
  Audience,
  DefectStatus,
  JevSettings,
  Kind,
  LegacyRecord,
  MemoryRecord,
  Reader,
  ReadLogEntry,
  Source,
  StoredRecord,
} from "./domain.ts";
import type { Embedder } from "./embed.ts";
import { createJevReranker } from "./jev/rerank.ts";
import { takenByOf } from "./records.ts";
import type { TakenBy } from "./records.ts";
import { andAll, hybridSearch, insert, scopeClause, sqlString } from "./store.ts";
import type { Hit, Store } from "./store.ts";

export const RERANK_POOL = 20;
export const DEFAULT_LIMIT = 17;
export const MAX_LIMIT = 50;
export const SCORE_FLOOR = 0.6;
export const CLAIM_LINE = 1.0;
export const CLAIM_SLOTS = 5;
export const TASK_MAX_CHARS = 300;

export interface RecallParams {
  q: string;
  scope?: string | null;
  kind?: string | null;
  audience?: string | null;
  limit?: number | null;
  task?: string | null;
  threadId?: string | null;
}

export interface RecallCandidate {
  id: string;
  source: Source;
  kind: Kind;
  headline: string;
  fields: { claim: string; why: string | null; rejected: string | null };
  scope: string;
  purpose: string | null;
  topic: string | null;
  audience: Audience | null;
  reader: Reader | null;
  stage: string | null;
  where: string | null;
  trigger: string | null;
  leftBecause: string | null;
  defectStatus: DefectStatus | null;
  takenBy: TakenBy | null;
  score: number;
  jevScored?: boolean;
}

export interface RecallResult {
  id: string;
  source: Source;
  kind: Kind;
  headline: string;
  scope: string;
  score: number;
  fields?: { claim: string };
  purpose?: string;
  topic?: string;
  audience?: Audience;
  reader?: Reader;
  stage?: string;
  where?: string;
  trigger?: string;
  leftBecause?: string;
  defectStatus?: DefectStatus;
  takenBy?: TakenBy;
}

export type Reranker = (query: string, candidates: RecallCandidate[]) => Promise<RecallCandidate[] | null>;

export function toResult(row: StoredRecord, source: Source, score: number): RecallCandidate {
  return {
    id: row.id,
    source,
    kind: row.kind,
    headline: row.headline,
    fields: { claim: row.claim, why: row.why ?? null, rejected: row.rejected },
    scope: row.scope,
    purpose: row.purpose,
    topic: row.topic,
    audience: row.audience ?? null,
    reader: row.reader ?? null,
    stage: row.stage ?? null,
    where: row.where ?? null,
    trigger: row.trigger ?? null,
    leftBecause: row.leftBecause ?? null,
    defectStatus: row.defectStatus ?? null,
    takenBy: takenByOf(row),
    score,
  };
}

export function mergeBySource(v2: Array<Hit<StoredRecord>>, legacy: Array<Hit<StoredRecord>>): RecallCandidate[] {
  const tagged = [
    ...v2.map((hit, order) => ({ result: toResult(hit.row, "v2", hit.score), order, rank: 0 })),
    ...legacy.map((hit, order) => ({ result: toResult(hit.row, "legacy", hit.score), order, rank: 1 })),
  ];

  tagged.sort((a, b) => {
    if (b.result.score !== a.result.score) {
      return b.result.score - a.result.score;
    }

    if (a.rank !== b.rank) {
      return a.rank - b.rank;
    }

    return a.order - b.order;
  });

  return tagged.map((entry) => entry.result);
}

function present<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined && value !== "";
}

export function shapeResult(candidate: RecallCandidate, withClaim: boolean): RecallResult {
  const optional: Partial<RecallResult> = {
    purpose: candidate.purpose ?? undefined,
    topic: candidate.topic ?? undefined,
    audience: candidate.audience ?? undefined,
    reader: candidate.reader ?? undefined,
    stage: candidate.stage ?? undefined,
    where: candidate.where ?? undefined,
    trigger: candidate.trigger ?? undefined,
    leftBecause: candidate.leftBecause ?? undefined,
    defectStatus: candidate.defectStatus ?? undefined,
    takenBy: candidate.takenBy ?? undefined,
  };
  const kept = Object.fromEntries(Object.entries(optional).filter(([, value]) => present(value)));

  return {
    id: candidate.id,
    source: candidate.source,
    kind: candidate.kind,
    headline: candidate.headline,
    scope: candidate.scope,
    score: candidate.score,
    ...(withClaim ? { fields: { claim: candidate.fields.claim } } : {}),
    ...kept,
  };
}

function fuseByScore<Row extends { id: string }>(
  hybrid: Array<Hit<Row>>,
  taskHits: Array<Hit<Row>>,
  limit: number,
): Array<Hit<Row>> {
  const fused = new Map<string, Hit<Row>>();

  for (const hit of hybrid) {
    fused.set(hit.row.id, { row: hit.row, score: hit.score });
  }

  for (const hit of taskHits) {
    const seen = fused.get(hit.row.id);

    if (seen) {
      seen.score += hit.score;
    } else {
      fused.set(hit.row.id, { row: hit.row, score: hit.score });
    }
  }

  return [...fused.values()].sort((a, b) => b.score - a.score).slice(0, limit);
}

export function liveFilter(
  scope: string | null | undefined,
  kind: string | null | undefined,
  audience: string | null | undefined,
  shared = "global",
): string {
  return andAll([
    "lifecycle = 'live'",
    scopeClause(scope, shared),
    kind ? `kind = ${sqlString(kind)}` : null,
    audience ? `(audience = ${sqlString(audience)} OR audience IS NULL)` : null,
  ]) as string;
}

function createReranker(jev: JevSettings): Reranker | null {
  if (!jev.jev) {
    return null;
  }

  return createJevReranker({
    client: jev.jev,
    model: jev.model ?? "jev-1.13.0",
    timeoutMs: jev.timeoutMs,
    onError: (error) => console.error("jev rerank failed, using RRF order", error),
  });
}

function taskOf(raw: string | null | undefined): string | null {
  const task = (raw ?? "").trim().slice(0, TASK_MAX_CHARS).trim();

  return task === "" ? null : task;
}

export class RecallService {
  private readonly store: Store;
  private readonly embedder: Embedder;
  private readonly rerank: Reranker | null;
  private readonly now: () => string;
  private readonly sharedScopeOf: (scope: string) => string;

  constructor(
    store: Store,
    embedder: Embedder,
    jev: JevSettings = {},
    now: () => string = () => new Date().toISOString(),
    sharedScopeOf: (scope: string) => string = () => "global",
  ) {
    this.sharedScopeOf = sharedScopeOf;
    this.store = store;
    this.embedder = embedder;
    this.rerank = createReranker(jev);
    this.now = now;
  }

  async recall(params: RecallParams): Promise<{ results: RecallResult[] }> {
    const query = (params.q ?? "").trim();

    if (query === "") {
      throw new HttpError(400, "q is required");
    }

    if (params.kind && !(KINDS as readonly string[]).includes(params.kind)) {
      throw new HttpError(400, `kind must be one of ${KINDS.join(", ")}`);
    }

    if (params.audience && !(AUDIENCES as readonly string[]).includes(params.audience)) {
      throw new HttpError(400, `audience must be one of ${AUDIENCES.join(", ")}`);
    }

    const limit = this.limitOf(params.limit);
    const task = taskOf(params.task);
    const candidates = await this.candidates(query, task, params, Math.max(limit, RERANK_POOL));
    const ordered = await this.order(query, candidates);
    const results = this.shape(ordered, limit);

    await this.log(query, params.scope ?? null, results, task, params.threadId ?? null);

    return { results };
  }

  private limitOf(raw: number | null | undefined): number {
    if (raw === undefined || raw === null) {
      return DEFAULT_LIMIT;
    }

    if (!Number.isInteger(raw) || raw <= 0) {
      throw new HttpError(400, "limit must be a positive integer");
    }

    return Math.min(raw, MAX_LIMIT);
  }

  private async candidates(
    query: string,
    task: string | null,
    params: RecallParams,
    perTable: number,
  ): Promise<RecallCandidate[]> {
    const vector = await this.embedder.embed(query);
    const taskVector = task === null ? null : await this.embedder.embed(task);
    const shared = params.scope ? this.sharedScopeOf(params.scope) : "global";
    const where = liveFilter(params.scope, params.kind, params.audience, shared);
    const [v2, legacy] = await Promise.all([
      this.searchTable<MemoryRecord>(this.store.records, { query, vector, where, limit: perTable }, taskVector),
      this.searchTable<LegacyRecord>(this.store.legacy, { query, vector, where, limit: perTable }, taskVector),
    ]);

    return mergeBySource(v2, legacy);
  }

  private async searchTable<Row extends StoredRecord>(
    table: Store["records"] | Store["legacy"],
    search: { query: string; vector: number[]; where: string; limit: number },
    taskVector: number[] | null,
  ): Promise<Array<Hit<Row>>> {
    const hybrid = await hybridSearch<Row>(table, search);

    if (taskVector === null) {
      return hybrid;
    }

    const taskHits = await hybridSearch<Row>(table, { ...search, query: "-", vector: taskVector });

    return fuseByScore(hybrid, taskHits, search.limit);
  }

  private shape(ordered: RecallCandidate[], limit: number): RecallResult[] {
    const kept = ordered
      .filter((candidate) => candidate.jevScored !== true || candidate.score >= SCORE_FLOOR)
      .slice(0, limit);
    let slots = CLAIM_SLOTS;

    return kept.map((candidate) => {
      const eligible = candidate.jevScored !== true || candidate.score >= CLAIM_LINE;
      const withClaim = eligible && slots > 0;

      if (withClaim) {
        slots -= 1;
      }

      return shapeResult(candidate, withClaim);
    });
  }

  private async order(query: string, candidates: RecallCandidate[]): Promise<RecallCandidate[]> {
    if (this.rerank === null) {
      return candidates;
    }

    const head = candidates.slice(0, RERANK_POOL);
    const tail = candidates.slice(RERANK_POOL);
    const reranked = await this.rerank(query, head);

    return reranked ?? [...head, ...tail];
  }

  private async log(
    query: string,
    scope: string | null,
    results: RecallResult[],
    task: string | null,
    threadId: string | null,
  ): Promise<void> {
    const ts = this.now();
    const rows: ReadLogEntry[] = results.map((result, index) => ({
      id: randomUUID(),
      ts,
      query,
      scope,
      resultId: result.id,
      source: result.source,
      rank: index,
      score: result.score,
      threadId,
      task,
      via: "recall",
    }));

    await insert(this.store.readLog, rows as unknown as Array<Record<string, unknown>>);
  }
}
