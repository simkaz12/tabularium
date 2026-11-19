import { sqlString } from "./store.ts";
import type { Store } from "./store.ts";

export interface EmbedderBuild {
  model: string;
  digest: string;
  ollamaVersion: string;
}

export interface EmbedderStamp extends EmbedderBuild {
  stampedAt: string;
}

export type EmbedderCheck =
  | { status: "unchecked" }
  | { status: "match"; stamp: EmbedderStamp }
  | { status: "mismatch"; differs: Array<keyof EmbedderBuild>; stamp: EmbedderStamp; current: EmbedderBuild }
  | { status: "missing"; current: EmbedderBuild }
  | { status: "unreachable"; error: string };

export const EMBEDDER_STAMP_KEY = "embedderBuild";

const BUILD_FIELDS: Array<keyof EmbedderBuild> = ["model", "digest", "ollamaVersion"];
const BUILD_TIMEOUT_MS = 5000;

async function getJson(fetchImpl: typeof fetch, url: string): Promise<unknown> {
  const response = await fetchImpl(url, { signal: AbortSignal.timeout(BUILD_TIMEOUT_MS) });

  if (!response.ok) {
    throw new Error(`${url} answered ${response.status}`);
  }

  return response.json();
}

export async function fetchEmbedderBuild(
  embedUrl: string,
  model: string,
  fetchImpl: typeof fetch = fetch,
): Promise<EmbedderBuild> {
  const origin = new URL(embedUrl).origin;
  const version = (await getJson(fetchImpl, `${origin}/api/version`)) as { version?: unknown };
  const tags = (await getJson(fetchImpl, `${origin}/api/tags`)) as {
    models?: Array<{ name?: unknown; digest?: unknown }>;
  };
  const entry = (tags.models ?? []).find((candidate) => candidate.name === model || candidate.name === `${model}:latest`);

  if (typeof version.version !== "string") {
    throw new Error("ollama /api/version carries no version");
  }

  if (typeof entry?.digest !== "string") {
    throw new Error(`ollama /api/tags does not list model ${model}`);
  }

  return { model, digest: entry.digest, ollamaVersion: version.version };
}

export async function readEmbedderStamp(store: Store): Promise<EmbedderStamp | null> {
  const rows = await store.settings.query().where(`key = ${sqlString(EMBEDDER_STAMP_KEY)}`).limit(1).toArray();
  const value = (rows[0] as { value?: string | null } | undefined)?.value;

  return typeof value === "string" ? (JSON.parse(value) as EmbedderStamp) : null;
}

export async function writeEmbedderStamp(store: Store, build: EmbedderBuild, stampedAt: string): Promise<void> {
  const stamp: EmbedderStamp = { ...build, stampedAt };

  await store.withLock(async () => {
    await store.settings
      .mergeInsert("key")
      .whenMatchedUpdateAll()
      .whenNotMatchedInsertAll()
      .execute([{ key: EMBEDDER_STAMP_KEY, value: JSON.stringify(stamp), updatedAt: stampedAt }]);
  });
}

export async function checkEmbedderBuild(
  store: Store,
  embedUrl: string,
  model: string,
  fetchImpl: typeof fetch = fetch,
): Promise<EmbedderCheck> {
  let current: EmbedderBuild;

  try {
    current = await fetchEmbedderBuild(embedUrl, model, fetchImpl);
  } catch (error) {
    return { status: "unreachable", error: error instanceof Error ? error.message : String(error) };
  }

  const stamp = await readEmbedderStamp(store);

  if (stamp === null) {
    return { status: "missing", current };
  }

  const differs = BUILD_FIELDS.filter((field) => stamp[field] !== current[field]);

  return differs.length === 0 ? { status: "match", stamp } : { status: "mismatch", differs, stamp, current };
}

export const EMBEDDER_RETRY_MS = 30_000;

export class EmbedderHealth {
  private check: EmbedderCheck = { status: "unchecked" };
  private timer: NodeJS.Timeout | null = null;
  private generation = 0;

  get(): EmbedderCheck {
    return this.check;
  }

  async refresh(run: () => Promise<EmbedderCheck>, retryMs: number = EMBEDDER_RETRY_MS): Promise<EmbedderCheck> {
    const wasUnreachable = this.check.status === "unreachable";

    this.clearTimer();
    this.generation += 1;

    const generation = this.generation;
    const result = await run();

    if (generation !== this.generation) {
      return this.check;
    }

    this.check = result;

    const stillUnreachable = wasUnreachable && this.check.status === "unreachable";

    if (this.check.status !== "match" && this.check.status !== "unchecked" && !stillUnreachable) {
      console.warn(JSON.stringify({ event: "embedder-build-warning", ...this.check }));
    }

    if (this.check.status === "unreachable") {
      this.timer = setTimeout(() => this.retry(run, retryMs), retryMs);
      this.timer.unref();
    }

    return this.check;
  }

  stop(): void {
    this.generation += 1;
    this.clearTimer();
  }

  private clearTimer(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private retry(run: () => Promise<EmbedderCheck>, retryMs: number): void {
    this.timer = null;
    this.refresh(run, retryMs).catch((error) => console.error(error));
  }
}
