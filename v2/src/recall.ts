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
export const DEFAULT_LIMIT = 10;
export const MAX_LIMIT = 50;

export interface RecallParams {
  q: string;
  scope?: string | null;
  kind?: string | null;
  audience?: string | null;
  limit?: number | null;
}

export interface RecallResult {
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
}

export type Reranker = (query: string, candidates: RecallResult[]) => Promise<RecallResult[] | null>;

export function toResult(row: StoredRecord, source: Source, score: number): RecallResult {
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

export function mergeBySource(v2: Array<Hit<StoredRecord>>, legacy: Array<Hit<StoredRecord>>): RecallResult[] {
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

export function liveFilter(
  scope: string | null | undefined,
  kind: string | null | undefined,
  audience: string | null | undefined,
): string {
  return andAll([
    "lifecycle = 'live'",
    scopeClause(scope),
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

export class RecallService {
  private readonly store: Store;
  private readonly embedder: Embedder;
  private readonly rerank: Reranker | null;
  private readonly now: () => string;

  constructor(
    store: Store,
    embedder: Embedder,
    jev: JevSettings = {},
    now: () => string = () => new Date().toISOString(),
  ) {
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
    const candidates = await this.candidates(query, params, Math.max(limit, RERANK_POOL));
    const ordered = await this.order(query, candidates);
    const results = ordered.slice(0, limit);

    await this.log(query, params.scope ?? null, results);

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

  private async candidates(query: string, params: RecallParams, perTable: number): Promise<RecallResult[]> {
    const vector = await this.embedder.embed(query);
    const where = liveFilter(params.scope, params.kind, params.audience);
    const [v2, legacy] = await Promise.all([
      hybridSearch<MemoryRecord>(this.store.records, { query, vector, where, limit: perTable }),
      hybridSearch<LegacyRecord>(this.store.legacy, { query, vector, where, limit: perTable }),
    ]);

    return mergeBySource(v2, legacy);
  }

  private async order(query: string, candidates: RecallResult[]): Promise<RecallResult[]> {
    if (this.rerank === null) {
      return candidates;
    }

    const head = candidates.slice(0, RERANK_POOL);
    const tail = candidates.slice(RERANK_POOL);
    const reranked = await this.rerank(query, head);

    return [...(reranked ?? head), ...tail];
  }

  private async log(query: string, scope: string | null, results: RecallResult[]): Promise<void> {
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
    }));

    await insert(this.store.readLog, rows as unknown as Array<Record<string, unknown>>);
  }
}
