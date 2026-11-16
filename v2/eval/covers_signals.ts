import { fileURLToPath } from "node:url";

import type { JevClient, ProposalBody } from "../src/domain.ts";
import { ask, choiceOf, recordSummary } from "../src/jev/checks.ts";
import { FRAMING, NEW_COVERS_OLD_QUESTION } from "../src/jev/questions.ts";
import type { QuestionSpec } from "../src/jev/questions.ts";
import { appendJsonl, fallbackShape, liveEnvironment, limitOf, parseArgs, readJsonl } from "./engine.ts";
import type { PairRow } from "./engine.ts";
import { emptyProvenance } from "./save.ts";

export const OLD_COVERS_NEW_QUESTION: QuestionSpec = {
  id: "old_covers_new",
  type: "choice",
  instructions:
    "Does the 'existing' record state everything that the 'incoming' text states? " +
    `${FRAMING} Rewording counts as stating it. Extra detail in the existing record does not ` +
    "matter; only whether something in the incoming text is missing from the existing record.",
  criteria: {
    yes: "every fact, detail, reason and decision in the incoming text also appears in the existing record",
    no: "the incoming text states at least one fact, detail, reason or decision that the existing record lacks or contradicts",
  },
};

export const ENHANCES_QUESTION: QuestionSpec = {
  id: "enhances",
  type: "choice",
  instructions:
    "Is the 'incoming' text the same fact as the 'existing' record, restated with more detail? " +
    `${FRAMING} A different subject, a change to the fact, or a later step is not the same fact.`,
  criteria: {
    yes: "the incoming text makes the same point as the existing record and adds at least one concrete detail to it",
    no: "the incoming text is about something else, changes the existing fact, is a later step, or adds nothing",
  },
};

export const COVERS_QUESTIONS: readonly QuestionSpec[] = [
  NEW_COVERS_OLD_QUESTION,
  OLD_COVERS_NEW_QUESTION,
  ENHANCES_QUESTION,
];

export interface YesNo {
  answer: "yes" | "no" | null;
  p_yes: number | null;
  probabilities: Record<string, number> | null;
  error: string | null;
}

export interface CoversRow {
  pair_id: string;
  run: number;
  new_covers_old: YesNo;
  old_covers_new: YesNo;
  enhances: YesNo;
  elapsed_ms: number;
  error: string | null;
}

export function elapsedSince(started: number): number {
  return Math.round((performance.now() - started) * 100) / 100;
}

export function bodyOf(pair: PairRow, text: string, label: string): ProposalBody {
  const shape = fallbackShape(pair.kind, text, `${pair.pair_id}-${label}`);
  const claim = shape.why === "" ? shape.claim : `${shape.claim}\n${shape.why}`;

  return {
    kind: pair.kind,
    headline: shape.headline,
    fields: { claim, rejected: shape.rejected },
    scope: "eval",
    provenance: emptyProvenance(),
    origin: "agent",
  };
}

export function failed(message: string): YesNo {
  return { answer: null, p_yes: null, probabilities: null, error: message };
}

export function yesNoOf(response: Awaited<ReturnType<typeof ask>>, spec: QuestionSpec): YesNo {
  try {
    const choice = choiceOf(response, spec.id);

    if (choice.choice !== "yes" && choice.choice !== "no") {
      return failed(`unexpected choice ${choice.choice}`);
    }

    const yes = choice.probabilities.yes ?? (choice.choice === "yes" ? choice.confidence : 1 - choice.confidence);

    return { answer: choice.choice, p_yes: yes, probabilities: choice.probabilities, error: null };
  } catch (error) {
    return failed(error instanceof Error ? error.message : String(error));
  }
}

export async function askCovers(
  client: JevClient,
  model: string,
  existing: ProposalBody,
  incoming: ProposalBody,
  timeoutMs?: number,
): Promise<Omit<CoversRow, "pair_id" | "run">> {
  const state = { existing: recordSummary(existing).slice(0, 1500), incoming: recordSummary(incoming) };
  const started = performance.now();

  try {
    const response = await ask(client, model, state, COVERS_QUESTIONS, timeoutMs);

    return {
      new_covers_old: yesNoOf(response, NEW_COVERS_OLD_QUESTION),
      old_covers_new: yesNoOf(response, OLD_COVERS_NEW_QUESTION),
      enhances: yesNoOf(response, ENHANCES_QUESTION),
      elapsed_ms: elapsedSince(started),
      error: null,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);

    return {
      new_covers_old: failed(message),
      old_covers_new: failed(message),
      enhances: failed(message),
      elapsed_ms: elapsedSince(started),
      error: message,
    };
  }
}

export async function coversForPair(
  client: JevClient,
  pair: PairRow,
  model: string,
  run: number,
  timeoutMs?: number,
): Promise<CoversRow> {
  const existing = bodyOf(pair, pair.existing, "existing");
  const incoming = bodyOf(pair, pair.incoming, "incoming");
  const answers = await askCovers(client, model, existing, incoming, timeoutMs);

  return { pair_id: pair.pair_id, run, ...answers };
}

export function evenSample<Item>(items: readonly Item[], size: number): Item[] {
  if (size >= items.length) {
    return [...items];
  }

  return Array.from({ length: size }, (_, index) => items[Math.floor((index * items.length) / size)] as Item);
}

export async function runCovers(
  client: JevClient,
  pairs: readonly PairRow[],
  out: string,
  model: string,
  run: number,
  timeoutMs?: number,
  done: ReadonlySet<string> = new Set(),
): Promise<CoversRow[]> {
  const rows: CoversRow[] = [];

  for (const pair of pairs) {
    if (done.has(pair.pair_id)) {
      continue;
    }

    const row = await coversForPair(client, pair, model, run, timeoutMs);

    appendJsonl(out, row);
    rows.push(row);
  }

  return rows;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const args = parseArgs(process.argv, ["pairs", "out", "limit", "sample", "run"]);
  const { pairs: pairsPath, out } = args;

  if (pairsPath === undefined || out === undefined) {
    throw new Error("--pairs and --out are required");
  }

  const environment = liveEnvironment();

  if (environment.jev === null) {
    throw new Error("TYPESAFE_API_KEY is not set; the containment questions need Jev on");
  }

  const all = readJsonl<PairRow>(pairsPath).slice(0, limitOf(args.limit));
  const pairs = args.sample === undefined ? all : evenSample(all, limitOf(args.sample));
  const run = args.run === undefined ? 1 : Number(args.run);
  const finished = new Set(readJsonl<CoversRow>(out).map((row) => row.pair_id));
  const model = environment.config.assessorModel;
  const rows = await runCovers(environment.jev, pairs, out, model, run, environment.config.jevTimeoutMs, finished);
  const unanswered = rows.filter((row) => row.error !== null).length;

  console.log(JSON.stringify({ event: "v2-covers-signals", pairs: pairs.length, written: rows.length, unanswered }));
}
