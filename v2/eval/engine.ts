import { appendFileSync, existsSync, mkdtempSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve, sep } from "node:path";

import { loadConfig } from "../src/config.ts";
import type { Config } from "../src/config.ts";
import { HEADLINE_MAX } from "../src/domain.ts";
import type { JevClient, Kind } from "../src/domain.ts";
import { OllamaEmbedder } from "../src/embed.ts";
import type { Embedder } from "../src/embed.ts";
import { createServices } from "../src/http.ts";
import type { Services } from "../src/http.ts";
import { createJevClient } from "../src/main.ts";
import { openStore } from "../src/store.ts";
import type { Store } from "../src/store.ts";

export const EVAL_SCOPE = "eval";
export const EVAL_DIM = 768;

export interface PairRow {
  pair_id: string;
  source: string;
  kind: "fact" | "decision";
  existing: string;
  incoming: string;
  similarity: number | null;
  old_label: string | null;
}

export interface NoiseRow {
  id: string;
  kind: "fact" | "decision";
  content: string;
  vector: number[];
  recall_count: number;
  created_at: string;
  lifecycle: string;
}

export interface CorpusRow {
  id: string;
  kind: string;
  content: string;
  scope: string;
  lifecycle: string;
  vector: number[];
  recall_count: number;
  created_at: string;
}

export interface QuestionRow {
  question_id: string;
  query: string;
  scope: string;
  origin: "mined" | "generated";
}

export interface Shape {
  headline: string;
  claim: string;
  why: string;
  rejected: string | null;
}

export interface CountingJev extends JevClient {
  calls: number;
  failures: number;
}

export interface EvalHarness {
  dir: string;
  store: Store;
  embedder: Embedder;
  seed: Services;
  judged: Services;
  jev: CountingJev | null;
}

const DECISION_HEAD = /^\s*Decision(?:\s*\([^)]*\))?\s*:\s*/;
const DECISION_LABEL = /\s(Why|Rejected)\s*:\s*/g;
const SENTENCE_END = /(?<=[.!?])\s/;

function resolveThroughSymlinks(target: string): string {
  const tail: string[] = [];
  let probe = target;

  while (!existsSync(probe) && dirname(probe) !== probe) {
    tail.unshift(basename(probe));
    probe = dirname(probe);
  }

  return join(realpathSync(probe), ...tail);
}

export function assertTempPath(path: string): string {
  const root = realpathSync(tmpdir());
  const target = resolve(path);
  const real = resolveThroughSymlinks(target);
  const inside = (candidate: string) => candidate.startsWith(root + sep);
  const forbidden = (candidate: string) => candidate === "/var/lib" || candidate.startsWith("/var/lib/");

  if (!inside(target) || !inside(real) || forbidden(target) || forbidden(real)) {
    throw new Error(`store path must be under ${root}, got ${path}`);
  }

  return target;
}

export function readJsonl<Row>(path: string): Row[] {
  if (!existsSync(path)) {
    return [];
  }

  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line) as Row);
}

export function resetJsonl(path: string): void {
  writeFileSync(path, "");
}

export function appendJsonl(path: string, row: unknown): void {
  appendFileSync(path, `${JSON.stringify(row)}\n`);
}

function collapse(text: string): string {
  return text.split(/\s+/).filter((part) => part !== "").join(" ");
}

export function clipHeadline(text: string, limit: number = HEADLINE_MAX): string {
  const flat = collapse(text).replace(/\.+$/, "");

  if (flat.length <= limit) {
    return flat;
  }

  const cut = flat.slice(0, limit);
  const boundary = cut.lastIndexOf(" ");

  return (boundary > Math.floor(limit / 2) ? cut.slice(0, boundary) : cut).replace(/[ ,;:-]+$/, "");
}

function firstSentence(text: string): string {
  const trimmed = text.trim();
  const match = SENTENCE_END.exec(trimmed);

  return match === null ? trimmed : trimmed.slice(0, match.index);
}

function headlineFor(preferred: string, content: string, recordId: string): string {
  const headline = clipHeadline(preferred);

  if (headline !== "") {
    return headline;
  }

  const raw = collapse(content).slice(0, HEADLINE_MAX).trim();

  return raw !== "" ? raw : `untitled ${recordId.slice(0, 8)}`;
}

function splitDecision(content: string): { claim: string; why: string; rejected: string | null } | null {
  const head = DECISION_HEAD.exec(content);

  if (head === null) {
    return null;
  }

  const rest = content.slice(head[0].length);
  const seen = new Set<string>();
  const cuts: RegExpExecArray[] = [];

  for (const match of rest.matchAll(DECISION_LABEL)) {
    if (!seen.has(match[1] as string)) {
      seen.add(match[1] as string);
      cuts.push(match as RegExpExecArray);
    }
  }

  const claimEnd = cuts.length > 0 ? (cuts[0] as RegExpExecArray).index : rest.length;
  const claim = rest.slice(0, claimEnd).trim();

  if (claim === "") {
    return null;
  }

  const sections: Record<string, string> = {};

  cuts.forEach((match, position) => {
    const next = cuts[position + 1];
    const stop = next === undefined ? rest.length : next.index;

    sections[match[1] as string] = rest.slice(match.index + match[0].length, stop).trim();
  });

  const rejected = sections.Rejected;

  return { claim, why: sections.Why ?? "", rejected: rejected === undefined || rejected === "" ? null : rejected };
}

export function fallbackShape(kind: string, content: string, recordId = ""): Shape {
  if (kind === "decision") {
    const parts = splitDecision(content);

    if (parts !== null) {
      return {
        headline: headlineFor(firstSentence(parts.claim), content, recordId),
        claim: parts.claim,
        why: parts.why,
        rejected: parts.rejected,
      };
    }
  }

  return {
    headline: headlineFor(firstSentence(content), content, recordId),
    claim: content.trim(),
    why: "",
    rejected: null,
  };
}

export function foldWhy(shape: Shape): string {
  return shape.why === "" ? shape.claim : `${shape.claim}\n${shape.why}`;
}

export function countingJev(inner: JevClient): CountingJev {
  const counting: CountingJev = {
    calls: 0,
    failures: 0,
    async systemOne(request, options) {
      counting.calls += 1;

      try {
        return await inner.systemOne(request, options);
      } catch (error) {
        counting.failures += 1;

        throw error;
      }
    },
  };

  return counting;
}

export interface OpenOptions {
  embedder: Embedder;
  jev?: JevClient | null;
  dir?: string;
  model?: string;
  topicsPath?: string | null;
  timeoutMs?: number;
}

export async function openEval(options: OpenOptions): Promise<EvalHarness> {
  const dir = assertTempPath(options.dir ?? mkdtempSync(join(tmpdir(), "tabularium-eval-")));
  const store = await openStore(dir, options.embedder.dim);
  const jev = options.jev ? countingJev(options.jev) : null;
  const seed = createServices(store, options.embedder);
  const proposals = {
    jev,
    ...(options.model === undefined ? {} : { model: options.model }),
    ...(options.topicsPath === undefined ? {} : { topicsPath: options.topicsPath }),
    ...(options.timeoutMs === undefined ? {} : { timeoutMs: options.timeoutMs }),
  };
  const judged = createServices(store, options.embedder, { proposals });

  return { dir, store, embedder: options.embedder, seed, judged, jev };
}

export interface LiveEnvironment {
  config: Config;
  embedder: Embedder;
  jev: JevClient | null;
}

export function liveEnvironment(): LiveEnvironment {
  const config = loadConfig();
  const embedder = new OllamaEmbedder(
    config.embedUrl,
    config.embedModel,
    EVAL_DIM,
    fetch,
    config.embedTimeoutMs,
  );

  return { config, embedder, jev: createJevClient(config) };
}

export function parseArgs(argv: string[], flags: readonly string[]): Record<string, string> {
  const parsed: Record<string, string> = {};
  const args = argv.slice(2);

  for (let index = 0; index < args.length; index += 2) {
    const name = (args[index] ?? "").replace(/^--/, "");
    const value = args[index + 1];

    if (!flags.includes(name) || value === undefined) {
      throw new Error(`usage: ${flags.map((flag) => `--${flag} <value>`).join(" ")}`);
    }

    parsed[name] = value;
  }

  return parsed;
}

export function limitOf(raw: string | undefined): number {
  if (raw === undefined) {
    return Number.POSITIVE_INFINITY;
  }

  const value = Number(raw);

  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`--limit must be a positive integer, got ${raw}`);
  }

  return value;
}

export type { Kind };
