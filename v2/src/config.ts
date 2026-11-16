import { fileURLToPath } from "node:url";

export interface Config {
  host: string;
  port: number;
  storePath: string;
  embedUrl: string;
  embedModel: string;
  embedDim: number;
  embedTimeoutMs: number;
  assessorModel: string;
  typesafeApiKey: string | null;
  allowedHosts: string[];
  topicsPath: string;
  jevTimeoutMs: number;
  containment: boolean;
  containmentFloor: number;
  foldAsk: boolean;
  foldSimilarityFloor: number;
  foldSubjectFloor: number;
  foldAskTimeoutMs: number;
}

export const DEFAULT_CLEANER_CONFIDENCE_FLOOR = 0.7;
export const CLEANER_VERDICT_TTL_DAYS = 90;
export const CLEANER_DOUBT_CAP = 10;
export const CLEANER_DOUBT_READ_LIMIT = 20_000;
export const DEFECT_TAKEN_DAYS = 14;
export const DEFECT_STALE_DAYS = 90;
export const DEFECT_ASK_CAP = 5;
export const CONFIDENCE_FLOOR = 0.5;
export const GATE_QUESTIONS: readonly string[] = ["kind", "worth", "perishable", "reader"];

const DEFAULT_TOPICS_PATH = fileURLToPath(new URL("../topics.json", import.meta.url));

function positiveInt(raw: string | undefined, fallback: number, name: string): number {
  if (raw === undefined || raw === "") {
    return fallback;
  }

  const value = Number(raw);

  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer, got ${raw}`);
  }

  return value;
}

function toggle(raw: string | undefined, fallback: boolean, name: string): boolean {
  if (raw === undefined || raw === "") {
    return fallback;
  }

  const value = raw.trim().toLowerCase();

  if (["on", "true", "1", "yes"].includes(value)) {
    return true;
  }

  if (["off", "false", "0", "no"].includes(value)) {
    return false;
  }

  throw new Error(`${name} must be on or off, got ${raw}`);
}

function probability(raw: string | undefined, fallback: number, name: string): number {
  if (raw === undefined || raw === "") {
    return fallback;
  }

  const value = Number(raw);

  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${name} must be a number between 0 and 1, got ${raw}`);
  }

  return value;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const host = env.TABULARIUM_HOST || "127.0.0.1";
  const port = positiveInt(env.TABULARIUM_PORT, 8789, "TABULARIUM_PORT");
  const allowed = (env.TABULARIUM_ALLOWED_HOSTS || `${host}:${port}`)
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);

  return {
    host,
    port,
    storePath: env.TABULARIUM_STORE || "./tabularium-v2-store",
    embedUrl: env.TABULARIUM_EMBED_URL || "http://localhost:11434/api/embed",
    embedModel: env.TABULARIUM_EMBED_MODEL || "nomic-embed-text",
    embedDim: positiveInt(env.TABULARIUM_EMBED_DIM, 768, "TABULARIUM_EMBED_DIM"),
    embedTimeoutMs: positiveInt(env.TABULARIUM_EMBED_TIMEOUT_MS, 10000, "TABULARIUM_EMBED_TIMEOUT_MS"),
    assessorModel: env.TABULARIUM_ASSESSOR_MODEL || "jev-1.13.0",
    typesafeApiKey: env.TYPESAFE_API_KEY || null,
    allowedHosts: allowed,
    topicsPath: env.TABULARIUM_TOPICS_PATH || DEFAULT_TOPICS_PATH,
    jevTimeoutMs: positiveInt(env.TABULARIUM_JEV_TIMEOUT_MS, 15000, "TABULARIUM_JEV_TIMEOUT_MS"),
    containment: toggle(env.TABULARIUM_CONTAINMENT, true, "TABULARIUM_CONTAINMENT"),
    containmentFloor: probability(env.TABULARIUM_CONTAINMENT_FLOOR, 0.95, "TABULARIUM_CONTAINMENT_FLOOR"),
    foldAsk: toggle(env.TABULARIUM_FOLD_ASK, false, "TABULARIUM_FOLD_ASK"),
    foldSimilarityFloor: probability(
      env.TABULARIUM_FOLD_SIMILARITY_FLOOR,
      0.8,
      "TABULARIUM_FOLD_SIMILARITY_FLOOR",
    ),
    foldSubjectFloor: probability(env.TABULARIUM_FOLD_SUBJECT_FLOOR, 0.7, "TABULARIUM_FOLD_SUBJECT_FLOOR"),
    foldAskTimeoutMs: positiveInt(env.TABULARIUM_FOLD_ASK_TIMEOUT_MS, 600000, "TABULARIUM_FOLD_ASK_TIMEOUT_MS"),
  };
}
