import { fileURLToPath } from "node:url";

import type { JevClient } from "../src/domain.ts";
import { ask, recordSummary } from "../src/jev/checks.ts";
import { NEW_COVERS_OLD_QUESTION, SAME_SUBJECT_QUESTION } from "../src/jev/questions.ts";
import type { QuestionSpec } from "../src/jev/questions.ts";
import {
  OLD_COVERS_NEW_QUESTION,
  bodyOf,
  elapsedSince,
  evenSample,
  failed,
  yesNoOf,
} from "./covers_signals.ts";
import type { YesNo } from "./covers_signals.ts";
import { appendJsonl, limitOf, liveEnvironment, parseArgs, readJsonl } from "./engine.ts";
import type { PairRow } from "./engine.ts";

export const FOLD_COVERS_OLD_QUESTION: QuestionSpec = { ...NEW_COVERS_OLD_QUESTION, id: "fold_covers_old" };
export const FOLD_COVERS_NEW_QUESTION: QuestionSpec = { ...NEW_COVERS_OLD_QUESTION, id: "fold_covers_new" };
export const SOURCES_COVER_FOLD_QUESTION: QuestionSpec = { ...OLD_COVERS_NEW_QUESTION, id: "sources_cover_fold" };

export const FOLD_SINGLE_QUESTION: QuestionSpec = {
  id: "fold_single",
  type: "choice",
  instructions:
    "Does the 'text' state one fact or one decision, rather than two separate facts or decisions " +
    "joined into one note? The text is a note in a long-term memory store for a coding agent. " +
    "A single fact with its details, reasons and rejected options is one. Two points that could " +
    "each be saved as their own note are two.",
  criteria: {
    yes: "the text makes one point; its sentences all serve a single fact or decision",
    no: "the text makes at least two separate points that could each stand as their own note",
  },
};

export const GUARD_NAMES = ["fold_covers_old", "fold_covers_new", "sources_cover_fold", "fold_single"] as const;

export type GuardName = (typeof GUARD_NAMES)[number];

export interface TimedYesNo extends YesNo {
  elapsed_ms: number;
}

export interface FoldRow {
  pair_id: string;
  writer: string;
  text: string | null;
  error: string | null;
}

export interface BandRow {
  pair_id: string;
}

export type GuardRow = {
  pair_id: string;
  writer: string;
  run: number;
  error: string | null;
} & Record<GuardName, TimedYesNo>;

export interface BandSignalRow {
  pair_id: string;
  run: number;
  same_subject: TimedYesNo;
  error: string | null;
}

async function askOne(
  client: JevClient,
  model: string,
  state: unknown,
  spec: QuestionSpec,
  timeoutMs?: number,
): Promise<TimedYesNo> {
  const started = performance.now();

  try {
    const response = await ask(client, model, state, [spec], timeoutMs);

    return { ...yesNoOf(response, spec), elapsed_ms: elapsedSince(started) };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);

    return { ...failed(message), elapsed_ms: elapsedSince(started) };
  }
}

function summaryOf(pair: PairRow, text: string, label: string): string {
  return recordSummary(bodyOf(pair, text, label));
}

function firstError(...answers: TimedYesNo[]): string | null {
  return answers.find((answer) => answer.error !== null)?.error ?? null;
}

export async function guardsForFold(
  client: JevClient,
  pair: PairRow,
  writer: string,
  fold: string,
  model: string,
  run: number,
  timeoutMs?: number,
): Promise<GuardRow> {
  const older = summaryOf(pair, pair.existing, "existing");
  const incoming = summaryOf(pair, pair.incoming, "incoming");
  const merged = summaryOf(pair, fold, `fold-${writer}`);
  const foldCoversOld = await askOne(
    client,
    model,
    { existing: older, incoming: merged },
    FOLD_COVERS_OLD_QUESTION,
    timeoutMs,
  );
  const foldCoversNew = await askOne(
    client,
    model,
    { existing: incoming, incoming: merged },
    FOLD_COVERS_NEW_QUESTION,
    timeoutMs,
  );
  const sourcesCoverFold = await askOne(
    client,
    model,
    { existing: `${older}\n\n${incoming}`, incoming: merged },
    SOURCES_COVER_FOLD_QUESTION,
    timeoutMs,
  );
  const foldSingle = await askOne(client, model, { text: merged }, FOLD_SINGLE_QUESTION, timeoutMs);

  return {
    pair_id: pair.pair_id,
    writer,
    run,
    fold_covers_old: foldCoversOld,
    fold_covers_new: foldCoversNew,
    sources_cover_fold: sourcesCoverFold,
    fold_single: foldSingle,
    error: firstError(foldCoversOld, foldCoversNew, sourcesCoverFold, foldSingle),
  };
}

export async function sameSubjectForPair(
  client: JevClient,
  pair: PairRow,
  model: string,
  run: number,
  timeoutMs?: number,
): Promise<BandSignalRow> {
  const state = {
    existing: summaryOf(pair, pair.existing, "existing"),
    incoming: summaryOf(pair, pair.incoming, "incoming"),
  };
  const sameSubject = await askOne(client, model, state, SAME_SUBJECT_QUESTION, timeoutMs);

  return { pair_id: pair.pair_id, run, same_subject: sameSubject, error: sameSubject.error };
}

export interface FoldTarget {
  pair: PairRow;
  writer: string;
  fold: string;
}

export function foldKey(pairId: string, writer: string): string {
  return `${pairId}~${writer}`;
}

export function foldTargets(folds: readonly FoldRow[], pairs: ReadonlyMap<string, PairRow>): FoldTarget[] {
  const latest = new Map<string, FoldRow>();

  for (const fold of folds) {
    latest.set(foldKey(fold.pair_id, fold.writer), fold);
  }

  const targets: FoldTarget[] = [];

  for (const fold of latest.values()) {
    const pair = pairs.get(fold.pair_id);

    if (pair !== undefined && fold.error === null && fold.text) {
      targets.push({ pair, writer: fold.writer, fold: fold.text });
    }
  }

  return targets;
}

export async function runGuards(
  client: JevClient,
  targets: readonly FoldTarget[],
  out: string,
  model: string,
  run: number,
  timeoutMs?: number,
  done: ReadonlySet<string> = new Set(),
): Promise<GuardRow[]> {
  const rows: GuardRow[] = [];

  for (const target of targets) {
    if (done.has(foldKey(target.pair.pair_id, target.writer))) {
      continue;
    }

    const row = await guardsForFold(client, target.pair, target.writer, target.fold, model, run, timeoutMs);

    appendJsonl(out, row);
    rows.push(row);
  }

  return rows;
}

export async function runBand(
  client: JevClient,
  pairs: readonly PairRow[],
  out: string,
  model: string,
  run: number,
  timeoutMs?: number,
  done: ReadonlySet<string> = new Set(),
): Promise<BandSignalRow[]> {
  const rows: BandSignalRow[] = [];

  for (const pair of pairs) {
    if (done.has(pair.pair_id)) {
      continue;
    }

    const row = await sameSubjectForPair(client, pair, model, run, timeoutMs);

    appendJsonl(out, row);
    rows.push(row);
  }

  return rows;
}

export function finishedKeys<Row extends { error: string | null }>(
  rows: readonly Row[],
  keyOf: (row: Row) => string,
): Set<string> {
  return new Set(rows.filter((row) => row.error === null).map(keyOf));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const flags = ["mode", "pairs", "band", "folds", "out", "sample", "run", "limit"];
  const args = parseArgs(process.argv, flags);
  const { mode, pairs: pairsPath, out } = args;

  if (mode === undefined || pairsPath === undefined || out === undefined) {
    throw new Error("--mode, --pairs and --out are required");
  }

  const environment = liveEnvironment();

  if (environment.jev === null) {
    throw new Error("TYPESAFE_API_KEY is not set; the guard questions need Jev on");
  }

  const pairs = new Map(readJsonl<PairRow>(pairsPath).map((pair) => [pair.pair_id, pair]));
  const run = args.run === undefined ? 1 : Number(args.run);
  const model = environment.config.assessorModel;
  const timeoutMs = environment.config.jevTimeoutMs;
  const limit = limitOf(args.limit);

  if (mode === "guard") {
    if (args.folds === undefined) {
      throw new Error("--folds is required for --mode guard");
    }

    const all = foldTargets(readJsonl<FoldRow>(args.folds), pairs).slice(0, limit);
    const targets = args.sample === undefined ? all : evenSample(all, limitOf(args.sample));
    const finished = finishedKeys(readJsonl<GuardRow>(out), (row) => foldKey(row.pair_id, row.writer));
    const rows = await runGuards(environment.jev, targets, out, model, run, timeoutMs, finished);
    const unanswered = rows.filter((row) => row.error !== null).length;

    const report = { event: "v2-fold-guard-signals", folds: targets.length, written: rows.length, unanswered };

    console.log(JSON.stringify(report));
  } else if (mode === "band") {
    if (args.band === undefined) {
      throw new Error("--band is required for --mode band");
    }

    const all = readJsonl<BandRow>(args.band)
      .flatMap((row) => pairs.get(row.pair_id) ?? [])
      .slice(0, limit);
    const selected = args.sample === undefined ? all : evenSample(all, limitOf(args.sample));
    const finished = finishedKeys(readJsonl<BandSignalRow>(out), (row) => row.pair_id);
    const rows = await runBand(environment.jev, selected, out, model, run, timeoutMs, finished);
    const unanswered = rows.filter((row) => row.error !== null).length;

    const report = { event: "v2-fold-band-signals", pairs: selected.length, written: rows.length, unanswered };

    console.log(JSON.stringify(report));
  } else {
    throw new Error(`--mode must be guard or band, got ${mode}`);
  }
}
