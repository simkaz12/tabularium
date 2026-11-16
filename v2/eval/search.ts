import { fileURLToPath } from "node:url";

import { KINDS } from "../src/domain.ts";
import type { Kind } from "../src/domain.ts";
import type { RecallService } from "../src/recall.ts";
import { importLegacy } from "../src/scripts/importLegacy.ts";
import type { LegacyLine } from "../src/scripts/importLegacy.ts";
import {
  appendJsonl,
  assertTempPath,
  fallbackShape,
  liveEnvironment,
  limitOf,
  openEval,
  parseArgs,
  readJsonl,
  resetJsonl,
} from "./engine.ts";
import type { CorpusRow, EvalHarness, QuestionRow } from "./engine.ts";

export type Arm = "v2_rerank" | "v2_plain" | "v2_rerank_facts" | "v2_plain_facts";

export const SELF_CHECK_SIZE = 20;
export const SELF_CHECK_MIN_RANK_ONE = 18;
export const SELF_CHECK_SEED = 20261112;

export interface SelfCheck {
  sampled: number;
  rank_one: number;
  required: number;
  ranks: Array<number | null>;
  passed: boolean;
}

export interface SearchRow {
  question_id: string;
  arm: Arm;
  rank: number;
  record_id: string;
  score: number;
  elapsed_ms: number;
}

export function toLegacyLine(row: CorpusRow): LegacyLine {
  const kind: Kind = (KINDS as readonly string[]).includes(row.kind) ? (row.kind as Kind) : "fact";
  const shape = fallbackShape(kind, row.content, row.id);

  return {
    legacyId: row.id,
    kind,
    headline: shape.headline,
    claim: shape.claim,
    why: shape.why,
    rejected: shape.rejected,
    scope: row.scope,
    oldScope: row.scope,
    oldSource: "v1",
    purpose: null,
    topic: null,
    lifecycle: "live",
    readCount: row.recall_count,
    originalCreatedAt: row.created_at,
    createdAt: row.created_at,
    reviewed: false,
    collapsedIds: [],
  };
}

export async function loadCorpus(harness: EvalHarness, corpus: CorpusRow[]) {
  return importLegacy(harness.store, harness.embedder, corpus.map(toLegacyLine));
}

function seededOrder(count: number, seed: number): number[] {
  let state = seed >>> 0;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;

    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
  const order = Array.from({ length: count }, (_, index) => index);

  for (let index = count - 1; index > 0; index -= 1) {
    const swap = Math.floor(next() * (index + 1));
    const held = order[index] as number;

    order[index] = order[swap] as number;
    order[swap] = held;
  }

  return order;
}

export function sampleKnownItems(
  corpus: CorpusRow[],
  size: number = SELF_CHECK_SIZE,
  seed: number = SELF_CHECK_SEED,
): CorpusRow[] {
  const copies = new Map<string, number>();

  for (const row of corpus) {
    const key = `${row.scope}\u0000${row.content}`;

    copies.set(key, (copies.get(key) ?? 0) + 1);
  }

  const unique = corpus.filter((row) => copies.get(`${row.scope}\u0000${row.content}`) === 1);

  return seededOrder(unique.length, seed)
    .slice(0, size)
    .map((index) => unique[index] as CorpusRow);
}

export async function selfCheck(recall: RecallService, sample: CorpusRow[]): Promise<SelfCheck> {
  const ranks: Array<number | null> = [];

  for (const row of sample) {
    const { results } = await recall.recall({ q: row.content, scope: row.scope, limit: 10 });
    const at = results.findIndex((result) => result.id === row.id);

    ranks.push(at === -1 ? null : at + 1);
  }

  const rankOne = ranks.filter((rank) => rank === 1).length;
  const required = Math.min(SELF_CHECK_MIN_RANK_ONE, sample.length);

  return {
    sampled: sample.length,
    rank_one: rankOne,
    required,
    ranks,
    passed: sample.length > 0 && rankOne >= required,
  };
}

export function zeroHitCalls(questions: QuestionRow[], rows: SearchRow[]): number {
  return questions.length - new Set(rows.map((row) => row.question_id)).size;
}

export async function runArm(
  recall: RecallService,
  arm: Arm,
  questions: QuestionRow[],
  out: string,
  kind?: "fact",
): Promise<SearchRow[]> {
  const kindFilter = kind === undefined ? {} : { kind };
  const first = questions[0];
  const rows: SearchRow[] = [];

  if (first === undefined) {
    return rows;
  }

  await recall.recall({ q: first.query, scope: first.scope, limit: 10, ...kindFilter });

  for (const question of questions) {
    const started = performance.now();
    const { results } = await recall.recall({
      q: question.query,
      scope: question.scope,
      limit: 10,
      ...kindFilter,
    });
    const elapsed = Math.round((performance.now() - started) * 100) / 100;

    results.forEach((result, index) => {
      const row: SearchRow = {
        question_id: question.question_id,
        arm,
        rank: index + 1,
        record_id: result.id,
        score: result.score,
        elapsed_ms: elapsed,
      };

      appendJsonl(out, row);
      rows.push(row);
    });
  }

  return rows;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv, ["corpus", "questions", "out-dir", "limit", "store"]);
  const { corpus: corpusPath, questions: questionsPath, "out-dir": outDir } = args;

  if (corpusPath === undefined || questionsPath === undefined || outDir === undefined) {
    throw new Error("--corpus, --questions and --out-dir are required");
  }

  const environment = liveEnvironment();

  if (environment.jev === null) {
    throw new Error("TYPESAFE_API_KEY is not set; the rerank arm needs Jev");
  }

  const dir = args.store === undefined ? undefined : assertTempPath(args.store);
  const harness = await openEval({
    embedder: environment.embedder,
    jev: environment.jev,
    model: environment.config.assessorModel,
    topicsPath: environment.config.topicsPath,
    timeoutMs: environment.config.jevTimeoutMs,
    ...(dir === undefined ? {} : { dir }),
  });
  const corpus = readJsonl<CorpusRow>(corpusPath);
  const questions = readJsonl<QuestionRow>(questionsPath).slice(0, limitOf(args.limit));
  const started = performance.now();
  const report = await loadCorpus(harness, corpus);

  console.error(`loaded ${report.written} legacy rows in ${Math.round(performance.now() - started)} ms`);

  const check = await selfCheck(harness.seed.recall, sampleKnownItems(corpus));

  if (!check.passed) {
    console.error(`known-item self-check failed: ${check.rank_one} of ${check.sampled} at rank 1`);
    console.error(`ranks: ${JSON.stringify(check.ranks)}`);
    console.log(JSON.stringify({ event: "v2-search-selfcheck", ...check, store: harness.dir }));
    process.exitCode = 1;

    return;
  }

  const arms: Array<[Arm, RecallService, "fact" | undefined]> = [
    ["v2_rerank", harness.judged.recall, undefined],
    ["v2_plain", harness.seed.recall, undefined],
    ["v2_rerank_facts", harness.judged.recall, "fact"],
    ["v2_plain_facts", harness.seed.recall, "fact"],
  ];

  for (const [arm, recall, kind] of arms) {
    const out = `${outDir}/search-${arm}.jsonl`;

    resetJsonl(out);

    const rows = await runArm(recall, arm, questions, out, kind);

    console.log(
      JSON.stringify({ event: "v2-search", arm, questions: questions.length, rows: rows.length, zero_hit_calls: zeroHitCalls(questions, rows) }),
    );
  }

  console.log(
    JSON.stringify({
      event: "v2-search-jev",
      calls: harness.jev?.calls ?? 0,
      failures: harness.jev?.failures ?? 0,
      self_check: check,
      store: harness.dir,
    }),
  );

  if ((harness.jev?.failures ?? 0) > 0) {
    process.exitCode = 2;
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await main();
}
