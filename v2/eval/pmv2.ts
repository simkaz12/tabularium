import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

import type { RecallService } from "../src/recall.ts";
import {
  appendJsonl,
  clipHeadline,
  liveEnvironment,
  openEval,
  parseArgs,
  readJsonl,
  resetJsonl,
} from "./engine.ts";
import type { EvalHarness } from "./engine.ts";
import { replayOptions, withProposalOptions } from "./save.ts";

export type PmArm = "v2_rerank" | "v2_plain";

export interface PmNote {
  note_id: string;
  persona_id: string;
  chunk_id: string;
  headline: string;
  claim: string;
}

export interface PmQuestion {
  question_id: string;
  persona_id: string;
  query: string;
}

export interface WriteRow {
  system: "v2";
  note_id: string;
  persona_id: string;
  status: string;
  record_id: string | null;
  elapsed_ms: number;
}

export interface HitRow {
  arm: PmArm;
  question_id: string;
  rank: number;
  record_id: string;
  text: string;
}

export interface AskEndRow {
  note_id: string;
  persona_id: string;
  status: string;
  ask_answer: string | null;
  record_id: string | null;
}

export interface PersonaOutputs {
  writes: string;
  asks: string;
  hits: Record<PmArm, string>;
}

export function scopeOf(personaId: string): string {
  return `pmv2-${personaId}`;
}

export function outputsIn(dir: string): PersonaOutputs {
  return {
    writes: `${dir}/writes-v2.jsonl`,
    asks: `${dir}/asks-v2.jsonl`,
    hits: {
      v2_rerank: `${dir}/hits-v2_rerank.jsonl`,
      v2_plain: `${dir}/hits-v2_plain.jsonl`,
    },
  };
}

export async function writeNotes(
  harness: EvalHarness,
  personaId: string,
  notes: PmNote[],
  out: PersonaOutputs,
): Promise<Map<string, string>> {
  const held = new Map<string, string>();

  for (const note of notes) {
    const started = performance.now();
    const proposal = await harness.judged.proposals.create({
      kind: "fact",
      headline: clipHeadline(note.headline),
      fields: { claim: note.claim },
      scope: scopeOf(personaId),
      origin: "agent",
    });
    const elapsed = Math.round((performance.now() - started) * 100) / 100;
    const row: WriteRow = {
      system: "v2",
      note_id: note.note_id,
      persona_id: personaId,
      status: proposal.status,
      record_id: proposal.resolution?.recordId ?? null,
      elapsed_ms: Math.round(elapsed),
    };

    if (proposal.status === "asked") {
      held.set(proposal.id, note.note_id);
    }

    appendJsonl(out.writes, row);
  }

  return held;
}

export async function expireHeld(
  harness: EvalHarness,
  personaId: string,
  held: Map<string, string>,
  out: PersonaOutputs,
): Promise<AskEndRow[]> {
  const rows: AskEndRow[] = [];

  if (held.size === 0) {
    return rows;
  }

  await harness.judged.proposals.expireAsks(new Date(Date.now() + 365 * 24 * 3600 * 1000).toISOString());

  for (const [proposalId, noteId] of held) {
    const proposal = await harness.judged.proposals.get(proposalId);
    const row: AskEndRow = {
      note_id: noteId,
      persona_id: personaId,
      status: proposal.status,
      ask_answer: proposal.ask?.answer ?? null,
      record_id: proposal.resolution?.recordId ?? null,
    };

    appendJsonl(out.asks, row);
    rows.push(row);
  }

  return rows;
}

export async function recallArm(
  recall: RecallService,
  arm: PmArm,
  personaId: string,
  questions: PmQuestion[],
  out: string,
): Promise<HitRow[]> {
  const rows: HitRow[] = [];

  for (const question of questions) {
    const { results } = await recall.recall({ q: question.query, scope: scopeOf(personaId), limit: 10 });

    results.forEach((result, index) => {
      const row: HitRow = {
        arm,
        question_id: question.question_id,
        rank: index + 1,
        record_id: result.id,
        text: `${result.headline}\n${result.fields?.claim ?? ""}`,
      };

      appendJsonl(out, row);
      rows.push(row);
    });
  }

  return rows;
}

export async function runPersona(
  harness: EvalHarness,
  personaId: string,
  notes: PmNote[],
  questions: PmQuestion[],
  out: PersonaOutputs,
): Promise<{ asks: AskEndRow[]; hits: Record<PmArm, HitRow[]> }> {
  const held = await writeNotes(harness, personaId, notes, out);
  const asks = await expireHeld(harness, personaId, held, out);
  const hits = {
    v2_rerank: await recallArm(harness.judged.recall, "v2_rerank", personaId, questions, out.hits.v2_rerank),
    v2_plain: await recallArm(harness.seed.recall, "v2_plain", personaId, questions, out.hits.v2_plain),
  };

  return { asks, hits };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv, ["notes", "questions", "out", "persona"]);
  const { notes: notesPath, questions: questionsPath, out: outDir } = args;

  if (notesPath === undefined || questionsPath === undefined || outDir === undefined) {
    throw new Error("--notes, --questions and --out are required");
  }

  const environment = liveEnvironment();

  if (environment.jev === null) {
    throw new Error("TYPESAFE_API_KEY is not set; v2 save needs Jev on");
  }

  mkdirSync(outDir, { recursive: true });

  const notes = readJsonl<PmNote>(notesPath).map((note) => ({ ...note, persona_id: String(note.persona_id) }));
  const questions = readJsonl<PmQuestion>(questionsPath).map((question) => ({
    ...question,
    persona_id: String(question.persona_id),
  }));
  const wanted = args.persona;
  const personas = [...new Set(notes.map((note) => note.persona_id))].filter(
    (id) => wanted === undefined || id === wanted,
  );
  const out = outputsIn(outDir);

  for (const file of [out.writes, out.asks, out.hits.v2_rerank, out.hits.v2_plain]) {
    resetJsonl(file);
  }

  for (const personaId of personas) {
    const opened = await openEval({
      embedder: environment.embedder,
      jev: environment.jev,
      model: environment.config.assessorModel,
      topicsPath: environment.config.topicsPath,
      timeoutMs: environment.config.jevTimeoutMs,
    });
    const harness = withProposalOptions(opened, replayOptions(environment.config));
    const result = await runPersona(
      harness,
      personaId,
      notes.filter((note) => note.persona_id === personaId),
      questions.filter((question) => question.persona_id === personaId),
      out,
    );

    console.log(
      JSON.stringify({
        event: "v2-pmv2",
        persona: personaId,
        asks: result.asks.length,
        records: await harness.store.records.countRows(),
        hits: { v2_rerank: result.hits.v2_rerank.length, v2_plain: result.hits.v2_plain.length },
        jev_calls: harness.jev?.calls ?? 0,
        jev_failures: harness.jev?.failures ?? 0,
        store: harness.dir,
      }),
    );
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await main();
}
