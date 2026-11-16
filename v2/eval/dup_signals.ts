import { fileURLToPath } from "node:url";

import type { JevClient, Neighbour, Proposal, ProposalBody } from "../src/domain.ts";
import { ask, choiceOf, recordSummary } from "../src/jev/checks.ts";
import type { QuestionSpec } from "../src/jev/questions.ts";
import {
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
import {
  bucketOf,
  countRecords,
  degradedOf,
  emptyProvenance,
  assertAtBase,
  rawOutcomeOf,
  restoreBase,
  seedNoise,
} from "./save.ts";
import type { Bucket } from "./save.ts";

export const ADDS_INFORMATION_QUESTION: QuestionSpec = {
  id: "adds_information",
  type: "choice",
  instructions:
    "Does the 'incoming' text add any information that the 'existing' record does not already " +
    "contain? 'existing' is the older record in a long-term memory store for a coding agent; " +
    "'incoming' is the newer text being evaluated against it. Rewording, summarising or " +
    "restating what 'existing' already says adds nothing.",
  criteria: {
    yes: "the incoming text states at least one fact, detail, reason or change that the existing record lacks",
    no: "everything the incoming text says is already contained in the existing record",
  },
};

export interface NeighbourSignal {
  id: string;
  pick: string | null;
  confidence: number | null;
  probabilities: Record<string, number> | null;
  reason_class: string | null;
  similarity: number;
}

export interface AddsInformation {
  answer: "yes" | "no" | null;
  probability: number | null;
  probabilities: Record<string, number> | null;
  elapsed_ms: number;
  error: string | null;
}

export interface SignalRow {
  pair_id: string;
  bucket: Bucket;
  raw_outcome: string;
  found_existing: boolean;
  relation_pick: string | null;
  degraded: string | null;
  existing: NeighbourSignal | null;
  top_other: NeighbourSignal | null;
  adds_information: AddsInformation;
  propose_ms: number;
}

export function neighbourSignal(neighbour: Neighbour): NeighbourSignal {
  const relation = neighbour.relation;

  return {
    id: neighbour.id,
    pick: relation?.pick ?? null,
    confidence: relation?.confidence ?? null,
    probabilities: relation?.probabilities ?? null,
    reason_class: relation?.reasonClass ?? null,
    similarity: neighbour.similarity,
  };
}

export function signalsOf(proposal: Proposal, existingId: string) {
  const existing = proposal.neighbours.find((candidate) => candidate.id === existingId);
  const top = proposal.neighbours[0];
  const other = top !== undefined && top.id !== existingId ? top : undefined;

  return {
    existing: existing === undefined ? null : neighbourSignal(existing),
    top_other: other === undefined ? null : neighbourSignal(other),
  };
}

export async function askAddsInformation(
  client: JevClient,
  model: string,
  existing: ProposalBody,
  incoming: ProposalBody,
  timeoutMs?: number,
): Promise<AddsInformation> {
  const state = { existing: recordSummary(existing).slice(0, 1500), incoming: recordSummary(incoming) };
  const started = performance.now();

  try {
    const response = await ask(client, model, state, [ADDS_INFORMATION_QUESTION], timeoutMs);
    const choice = choiceOf(response, ADDS_INFORMATION_QUESTION.id);
    const answer = choice.choice === "yes" || choice.choice === "no" ? choice.choice : null;

    return {
      answer,
      probability: answer === null ? null : (choice.probabilities[choice.choice] ?? choice.confidence),
      probabilities: choice.probabilities,
      elapsed_ms: elapsedSince(started),
      error: answer === null ? `unexpected choice ${choice.choice}` : null,
    };
  } catch (error) {
    return {
      answer: null,
      probability: null,
      probabilities: null,
      elapsed_ms: elapsedSince(started),
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function elapsedSince(started: number): number {
  return Math.round((performance.now() - started) * 100) / 100;
}

function bodyOf(pair: PairRow, text: string, label: string): ProposalBody {
  const shape = fallbackShape(pair.kind, text, `${pair.pair_id}-${label}`);

  return {
    kind: pair.kind,
    headline: shape.headline,
    fields: { claim: foldWhy(shape), rejected: shape.rejected },
    scope: "eval",
    provenance: emptyProvenance(),
    origin: "agent",
  };
}

export async function replayPairSignals(
  harness: EvalHarness,
  pair: PairRow,
  baseCount: number,
  model: string,
  timeoutMs?: number,
): Promise<SignalRow> {
  await assertAtBase(harness, pair.pair_id, baseCount);

  const existingBody = bodyOf(pair, pair.existing, "existing");
  const incomingBody = bodyOf(pair, pair.incoming, "incoming");
  const existing = await harness.seed.records.insertRecord({
    kind: pair.kind,
    headline: existingBody.headline,
    claim: existingBody.fields.claim,
    rejected: existingBody.fields.rejected ?? null,
    scope: "eval",
    purpose: null,
    topic: null,
    provenance: emptyProvenance(),
    origin: "agent",
    legacyId: null,
  });
  let proposalId: string | null = null;
  let createdId: string | null = null;
  let reinforcedId: string | null = null;

  try {
    const started = performance.now();
    const proposal = await harness.judged.proposals.create({
      kind: pair.kind,
      headline: incomingBody.headline,
      fields: incomingBody.fields,
      scope: "eval",
      origin: "agent",
    });
    const proposeMs = elapsedSince(started);

    proposalId = proposal.id;
    createdId = proposal.status === "approved" ? (proposal.resolution?.recordId ?? null) : null;
    reinforcedId = proposal.status === "reinforced" ? (proposal.resolution?.recordId ?? null) : null;

    const signals = signalsOf(proposal, existing.id);
    const adds = harness.jev === null
      ? noQuestion("no jev client")
      : await askAddsInformation(harness.jev, model, existingBody, incomingBody, timeoutMs);

    return {
      pair_id: pair.pair_id,
      bucket: bucketOf(proposal, existing.id),
      raw_outcome: rawOutcomeOf(proposal),
      found_existing: signals.existing !== null,
      relation_pick: signals.existing?.pick ?? null,
      degraded: degradedOf(proposal),
      existing: signals.existing,
      top_other: signals.top_other,
      adds_information: adds,
      propose_ms: proposeMs,
    };
  } finally {
    await restoreBase(harness, [existing.id, createdId], proposalId, reinforcedId);
  }
}

function noQuestion(error: string): AddsInformation {
  return { answer: null, probability: null, probabilities: null, elapsed_ms: 0, error };
}

export async function runSignals(
  harness: EvalHarness,
  pairs: PairRow[],
  out: string,
  model: string,
  timeoutMs?: number,
  done: ReadonlySet<string> = new Set(),
): Promise<SignalRow[]> {
  const baseCount = await countRecords(harness);
  const rows: SignalRow[] = [];

  for (const pair of pairs) {
    if (done.has(pair.pair_id)) {
      continue;
    }

    const row = await replayPairSignals(harness, pair, baseCount, model, timeoutMs);

    appendJsonl(out, row);
    rows.push(row);
  }

  return rows;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const args = parseArgs(process.argv, ["pairs", "noise", "out", "limit", "store"]);
  const { pairs: pairsPath, noise: noisePath, out } = args;

  if (pairsPath === undefined || noisePath === undefined || out === undefined) {
    throw new Error("--pairs, --noise and --out are required");
  }

  const environment = liveEnvironment();

  if (environment.jev === null) {
    throw new Error("TYPESAFE_API_KEY is not set; the signal capture needs Jev on");
  }

  const dir = args.store === undefined ? undefined : assertTempPath(args.store);
  const model = environment.config.assessorModel;
  const harness = await openEval({
    embedder: environment.embedder,
    jev: environment.jev,
    model,
    topicsPath: environment.config.topicsPath,
    timeoutMs: environment.config.jevTimeoutMs,
    ...(dir === undefined ? {} : { dir }),
  });
  const pairs = readJsonl<PairRow>(pairsPath).slice(0, limitOf(args.limit));
  const noise = readJsonl<NoiseRow>(noisePath);
  const finished = new Set(readJsonl<SignalRow>(out).map((row) => row.pair_id));
  const present = await countRecords(harness);

  if (present !== 0 && present !== noise.length) {
    throw new Error(`store holds ${present} rows, expected 0 or ${noise.length}`);
  }

  if (present === 0) {
    await seedNoise(harness, noise, (count) => console.error(`seeded ${count}/${noise.length}`));
  }

  const rows = await runSignals(harness, pairs, out, model, environment.config.jevTimeoutMs, finished);
  const degraded = rows.filter((row) => row.degraded !== null).length;
  const unanswered = rows.filter((row) => row.adds_information.answer === null).length;

  console.log(
    JSON.stringify({
      event: "v2-dup-signals",
      pairs: pairs.length,
      written: rows.length,
      degraded,
      unanswered,
      jevCalls: harness.jev?.calls ?? 0,
      jevFailures: harness.jev?.failures ?? 0,
      store: harness.dir,
    }),
  );
}
