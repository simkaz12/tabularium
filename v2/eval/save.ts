import { fileURLToPath } from "node:url";

import type { Config } from "../src/config.ts";
import type { Proposal } from "../src/domain.ts";
import { createServices } from "../src/http.ts";
import type { ProposalOptions } from "../src/proposals.ts";
import { findById, rowToProposal, sqlString, upsert } from "../src/store.ts";
import {
  EVAL_SCOPE,
  appendJsonl,
  assertTempPath,
  fallbackShape,
  foldWhy,
  liveEnvironment,
  limitOf,
  openEval,
  parseArgs,
  readJsonl,
} from "./engine.ts";
import type { EvalHarness, NoiseRow, PairRow } from "./engine.ts";
import { answerAsked, readReplies } from "./reply.ts";
import type { AnswerOutcome, AskReply } from "./reply.ts";

export type Bucket = "caught" | "flagged" | "new" | "wrong_target" | "asked";

export interface NeighbourRow {
  id: string;
  source: string;
  similarity: number;
  relation_pick: string | null;
  relation_confidence: number | null;
  new_covers_old: number | null;
  same_subject: number | null;
  is_seed: boolean;
}

export interface SaveRow {
  pair_id: string;
  system: "v2";
  bucket: Bucket;
  found_existing: boolean;
  relation_pick: string | null;
  raw_outcome: string;
  elapsed_ms: number;
  degraded: string | null;
  neighbours: NeighbourRow[];
  answer: AnswerOutcome | null;
}

const NOISE_BATCH_LOG = 250;
const COVERS_PREFIX = "covers:";

export interface Touched {
  createdId: string | null;
  reinforcedId: string | null;
}

export function coversId(proposal: Proposal): string | null {
  const flag = proposal.flags.find((candidate) => candidate.startsWith(COVERS_PREFIX));

  return flag === undefined ? null : flag.slice(COVERS_PREFIX.length);
}

export function touchedBy(proposal: Proposal): Touched {
  return {
    createdId: proposal.status === "approved" ? (proposal.resolution?.recordId ?? null) : null,
    reinforcedId: proposal.status === "reinforced" ? (proposal.resolution?.recordId ?? null) : null,
  };
}

export function replayOptions(config: Config): ProposalOptions {
  return {
    model: config.assessorModel,
    topicsPath: config.topicsPath,
    timeoutMs: config.jevTimeoutMs,
    containment: config.containment,
    containmentFloor: config.containmentFloor,
    foldAsk: true,
    foldSimilarityFloor: config.foldSimilarityFloor,
    foldSubjectFloor: config.foldSubjectFloor,
    foldAskTimeoutMs: config.foldAskTimeoutMs,
  };
}

export function withProposalOptions(harness: EvalHarness, options: ProposalOptions): EvalHarness {
  const judged = createServices(harness.store, harness.embedder, {
    proposals: { jev: harness.jev, ...options },
  });

  return { ...harness, judged };
}

export function bucketOf(proposal: Proposal, existingId: string): Bucket {
  if (proposal.status === "asked") {
    return proposal.ask?.existing.id === existingId ? "asked" : "wrong_target";
  }

  if (proposal.status === "reinforced") {
    return proposal.resolution?.recordId === existingId ? "caught" : "wrong_target";
  }

  if (proposal.status !== "approved") {
    throw new Error(`unexpected proposal status ${proposal.status}`);
  }

  const suggestion = proposal.suggestion;

  if (suggestion === null) {
    return "new";
  }

  return suggestion.recordId === existingId ? "flagged" : "wrong_target";
}

export function neighbourRowsOf(proposal: Proposal, seedId: string): NeighbourRow[] {
  return proposal.neighbours.map((neighbour) => ({
    id: neighbour.id,
    source: neighbour.source,
    similarity: neighbour.similarity,
    relation_pick: neighbour.relation?.pick ?? null,
    relation_confidence: neighbour.relation?.confidence ?? null,
    new_covers_old: neighbour.pair?.newCoversOld ?? null,
    same_subject: neighbour.pair?.sameSubject ?? null,
    is_seed: neighbour.id === seedId,
  }));
}

export function degradedOf(proposal: Proposal): string | null {
  if (proposal.flags.includes("jev_unavailable")) {
    return "jev_unavailable";
  }

  if (proposal.flags.some((flag) => flag.startsWith("relation_unavailable:"))) {
    return "relation_unavailable";
  }

  return proposal.flags.some((flag) => flag.startsWith("pair_unavailable:")) ? "pair_unavailable" : null;
}

export function rawOutcomeOf(proposal: Proposal): string {
  const suggestion = proposal.suggestion;

  if (proposal.status === "asked") {
    const via = coversId(proposal) === null ? "" : "covers:";

    return `asked:${via}${proposal.ask?.existing.id ?? ""}`;
  }

  return suggestion === null
    ? proposal.status
    : `${proposal.status}:${suggestion.action}:${suggestion.recordId}`;
}

export async function seedNoise(harness: EvalHarness, noise: NoiseRow[], log?: (count: number) => void): Promise<void> {
  let done = 0;

  for (const row of noise) {
    const shape = fallbackShape(row.kind, row.content, row.id);

    await harness.seed.records.insertRecord({
      id: row.id,
      kind: row.kind,
      headline: shape.headline,
      claim: foldWhy(shape),
      rejected: shape.rejected,
      scope: EVAL_SCOPE,
      purpose: null,
      topic: null,
      provenance: emptyProvenance(),
      origin: "agent",
      legacyId: null,
      vector: row.vector,
    });

    done += 1;

    if (log !== undefined && done % NOISE_BATCH_LOG === 0) {
      log(done);
    }
  }
}

export function emptyProvenance() {
  return {
    threadId: "",
    threadTitle: "",
    projectId: "",
    branch: "",
    machine: "",
    link: "",
    excerpt: "",
    stage: "",
  };
}

const DIRTY_BACKGROUND = "reinforcedCount > 0 OR lastReinforcedAt IS NOT NULL OR lifecycle != 'live' OR supersededBy IS NOT NULL";

export async function countRecords(harness: EvalHarness): Promise<number> {
  return harness.store.records.countRows();
}

export async function assertAtBase(harness: EvalHarness, pairId: string, baseCount: number): Promise<void> {
  const dirty = await harness.store.records.countRows(DIRTY_BACKGROUND);

  const proposals = await harness.store.proposals.countRows();

  if ((await countRecords(harness)) !== baseCount || dirty !== 0 || proposals !== 0) {
    throw new Error(
      `store is not at base before ${pairId}: ${dirty} background rows carry changes, ${proposals} proposal rows remain`,
    );
  }
}

export async function replayPair(
  harness: EvalHarness,
  pair: PairRow,
  baseCount: number,
  reply: AskReply | undefined = undefined,
): Promise<SaveRow> {
  await assertAtBase(harness, pair.pair_id, baseCount);

  const existingShape = fallbackShape(pair.kind, pair.existing, `${pair.pair_id}-existing`);
  const incomingShape = fallbackShape(pair.kind, pair.incoming, `${pair.pair_id}-incoming`);
  const existing = await harness.seed.records.insertRecord({
    kind: pair.kind,
    headline: existingShape.headline,
    claim: foldWhy(existingShape),
    rejected: existingShape.rejected,
    scope: EVAL_SCOPE,
    purpose: null,
    topic: null,
    provenance: emptyProvenance(),
    origin: "agent",
    legacyId: null,
  });
  let proposalId: string | null = null;
  let touched: Touched = { createdId: null, reinforcedId: null };
  const extra: Array<string | null> = [];

  try {
    const started = performance.now();
    const proposal = await harness.judged.proposals.create({
      kind: pair.kind,
      headline: incomingShape.headline,
      fields: { claim: foldWhy(incomingShape), rejected: incomingShape.rejected },
      scope: EVAL_SCOPE,
      origin: "agent",
    });
    const elapsed = performance.now() - started;
    const neighbour = proposal.neighbours.find((candidate) => candidate.id === existing.id);
    const bucket = bucketOf(proposal, existing.id);

    proposalId = proposal.id;
    touched = touchedBy(proposal);

    let answer: AnswerOutcome | null = null;

    if (bucket === "asked") {
      const answered = await answerAsked(harness, proposal, pair, existing.id, reply);

      answer = answered.outcome;

      if (answered.proposal !== null) {
        touched = touchedBy(answered.proposal);
        extra.push(answered.proposal.resolution?.recordId ?? null);
      }
    }

    return {
      pair_id: pair.pair_id,
      system: "v2",
      bucket,
      found_existing: neighbour !== undefined,
      relation_pick: neighbour?.relation?.pick ?? null,
      raw_outcome: rawOutcomeOf(proposal),
      elapsed_ms: Math.round(elapsed * 100) / 100,
      degraded: degradedOf(proposal),
      neighbours: neighbourRowsOf(proposal, existing.id),
      answer,
    };
  } finally {
    await restoreBase(
      harness,
      [existing.id, touched.createdId, ...extra, ...(await persistedRecordIds(harness, proposalId))],
      proposalId,
      touched.reinforcedId,
    );
  }
}

async function persistedRecordIds(harness: EvalHarness, proposalId: string | null): Promise<Array<string | null>> {
  if (proposalId === null) {
    return [];
  }

  const row = await findById<Record<string, unknown>>(harness.store.proposals, proposalId);

  if (row === undefined) {
    return [];
  }

  const { ask } = rowToProposal(row);

  return [ask?.foldRecordId ?? null, ask?.heldRecordId ?? null];
}

async function resetReinforcement(harness: EvalHarness, recordId: string): Promise<void> {
  const row = await findById<Record<string, unknown>>(harness.store.records, recordId);

  if (row !== undefined) {
    await upsert(harness.store.records, [{ ...row, reinforcedCount: 0, lastReinforcedAt: null }]);
  }
}

export async function restoreBase(
  harness: EvalHarness,
  recordIds: Array<string | null>,
  proposalId: string | null,
  reinforcedId: string | null,
) {
  if (reinforcedId !== null && !recordIds.includes(reinforcedId)) {
    await resetReinforcement(harness, reinforcedId);
  }

  const ids = recordIds.filter((id): id is string => id !== null);

  await harness.store.records.delete(`id IN (${ids.map(sqlString).join(", ")})`);

  if (proposalId !== null) {
    await harness.store.proposals.delete(`id = ${sqlString(proposalId)}`);
  }
}

export async function runSave(
  harness: EvalHarness,
  pairs: PairRow[],
  out: string,
  done: ReadonlySet<string> = new Set(),
  replies: ReadonlyMap<string, AskReply> = new Map(),
): Promise<SaveRow[]> {
  const baseCount = await countRecords(harness);
  const rows: SaveRow[] = [];

  for (const pair of pairs) {
    if (done.has(pair.pair_id)) {
      continue;
    }

    const row = await replayPair(harness, pair, baseCount, replies.get(pair.pair_id));

    appendJsonl(out, row);
    rows.push(row);
  }

  return rows;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const args = parseArgs(process.argv, ["pairs", "noise", "asks", "asks-extra", "out", "limit", "store", "arm"]);
  const { pairs: pairsPath, noise: noisePath, asks: asksPath, out } = args;

  if (pairsPath === undefined || noisePath === undefined || asksPath === undefined || out === undefined) {
    throw new Error("--pairs, --noise, --asks and --out are required");
  }

  const environment = liveEnvironment();

  if (environment.jev === null) {
    throw new Error("TYPESAFE_API_KEY is not set; v2 save needs Jev on");
  }

  const dir = args.store === undefined ? undefined : assertTempPath(args.store);
  const opened = await openEval({
    embedder: environment.embedder,
    jev: environment.jev,
    model: environment.config.assessorModel,
    topicsPath: environment.config.topicsPath,
    timeoutMs: environment.config.jevTimeoutMs,
    ...(dir === undefined ? {} : { dir }),
  });
  const harness = withProposalOptions(opened, replayOptions(environment.config));
  const pairs = readJsonl<PairRow>(pairsPath).slice(0, limitOf(args.limit));
  const noise = readJsonl<NoiseRow>(noisePath);
  const replies = readReplies(asksPath, args["asks-extra"], args.arm);
  const finished = new Set(readJsonl<SaveRow>(out).map((row) => row.pair_id));

  const present = await countRecords(harness);

  if (present !== 0 && present !== noise.length) {
    throw new Error(`store holds ${present} rows, expected 0 or ${noise.length}`);
  }

  if (present === 0) {
    const started = performance.now();

    await seedNoise(harness, noise, (count) => console.error(`seeded ${count}/${noise.length}`));
    console.error(`seeded ${noise.length} noise rows in ${Math.round(performance.now() - started)} ms`);
  }

  const rows = await runSave(harness, pairs, out, finished, replies);
  const degraded = rows.filter((row) => row.degraded !== null).length;

  console.log(
    JSON.stringify({
      event: "v2-save",
      pairs: pairs.length,
      written: rows.length,
      degraded,
      asked: rows.filter((row) => row.answer !== null).length,
      folded: rows.filter((row) => row.answer?.end_state === "folded").length,
      bothLive: rows.filter((row) => row.answer?.end_state === "both_live").length,
      noReply: rows.filter((row) => row.answer?.end_state === "no_reply").length,
      failed: rows.filter((row) => row.answer?.end_state === "failed").length,
      jevCalls: harness.jev?.calls ?? 0,
      jevFailures: harness.jev?.failures ?? 0,
      store: harness.dir,
    }),
  );
}
