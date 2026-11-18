import { fileURLToPath } from "node:url";

import { loadConfig } from "../config.ts";
import { recordText } from "../domain.ts";
import { OllamaEmbedder } from "../embed.ts";
import type { Embedder } from "../embed.ts";
import { fetchEmbedderBuild, readEmbedderStamp, writeEmbedderStamp } from "../embedderBuild.ts";
import type { EmbedderBuild } from "../embedderBuild.ts";
import { openStore, selectWhere, upsert } from "../store.ts";
import type { Store } from "../store.ts";

const DEFAULT_BATCH = 50;
const DRY_RUN_SAMPLE = 10;
const READ_LIMIT = 1_000_000;

type TableName = "records" | "legacy";

interface TextRow {
  id: string;
  kind?: "fact" | "decision" | "defect";
  headline: string;
  claim: string;
  where?: string | null;
  trigger?: string | null;
  vector: number[];
}

export interface ReembedOptions {
  batchSize?: number;
  onProgress?: (event: Record<string, unknown>) => void;
}

export interface ReembedReport {
  records: number;
  legacy: number;
  vectorIndexRebuilt: string[];
}

export interface DryRunReport {
  records: number;
  legacy: number;
  sampled: number;
  msPerEmbed: number;
  estimatedSeconds: number;
}

async function reembedTable(
  store: Store,
  name: TableName,
  embedder: Embedder,
  batchSize: number,
  onProgress: (event: Record<string, unknown>) => void,
): Promise<number> {
  const table = store[name];
  const rows = await selectWhere<TextRow>(table, undefined, READ_LIMIT);

  for (let start = 0; start < rows.length; start += batchSize) {
    const batch = rows.slice(start, start + batchSize);
    const refreshed: TextRow[] = [];

    for (const row of batch) {
      refreshed.push({ ...row, vector: await embedder.embed(recordText(row)) });
    }

    await store.withLock(() => upsert(table, refreshed as unknown as Array<Record<string, unknown>>));
    onProgress({ event: "reembed-progress", table: name, done: start + batch.length, total: rows.length });
  }

  return rows.length;
}

async function refreshIndices(store: Store): Promise<string[]> {
  const rebuilt: string[] = [];

  for (const name of ["records", "legacy"] as const) {
    const table = store[name];
    const vectorIndexed = (await table.listIndices()).some((index) => index.columns.includes("vector"));

    if (vectorIndexed) {
      await table.createIndex("vector", { replace: true });
      rebuilt.push(name);
    }

    await table.optimize();
  }

  return rebuilt;
}

export async function reembed(
  store: Store,
  embedder: Embedder,
  options: ReembedOptions = {},
): Promise<ReembedReport> {
  const batchSize = options.batchSize ?? DEFAULT_BATCH;
  const onProgress = options.onProgress ?? (() => undefined);
  const records = await reembedTable(store, "records", embedder, batchSize, onProgress);
  const legacy = await reembedTable(store, "legacy", embedder, batchSize, onProgress);
  const vectorIndexRebuilt = await refreshIndices(store);

  return { records, legacy, vectorIndexRebuilt };
}

export async function reembedAndStamp(
  store: Store,
  embedder: Embedder,
  build: () => Promise<EmbedderBuild>,
  options: ReembedOptions & { now?: () => string } = {},
): Promise<ReembedReport & { stamp: EmbedderBuild }> {
  const before = await build();
  const report = await reembed(store, embedder, options);
  const after = await build();

  if (JSON.stringify(before) !== JSON.stringify(after)) {
    throw new Error(
      `ollama build changed during the re-embed (${JSON.stringify(before)} then ${JSON.stringify(after)}); stamp not written, run again`,
    );
  }

  await writeEmbedderStamp(store, after, (options.now ?? (() => new Date().toISOString()))());

  return { ...report, stamp: after };
}

export async function dryRun(store: Store, embedder: Embedder): Promise<DryRunReport> {
  const records = await selectWhere<TextRow>(store.records, undefined, READ_LIMIT);
  const legacy = await selectWhere<TextRow>(store.legacy, undefined, READ_LIMIT);
  const sample = [...records, ...legacy].slice(0, DRY_RUN_SAMPLE);
  const started = performance.now();

  for (const row of sample) {
    await embedder.embed(recordText(row));
  }

  const msPerEmbed = sample.length === 0 ? 0 : (performance.now() - started) / sample.length;
  const estimatedSeconds = Math.round(((records.length + legacy.length) * msPerEmbed) / 1000);

  return {
    records: records.length,
    legacy: legacy.length,
    sampled: sample.length,
    msPerEmbed: Math.round(msPerEmbed * 10) / 10,
    estimatedSeconds,
  };
}

export function parseCliArgs(argv: string[]): { dryRun: boolean; batchSize: number } {
  const usage = "usage: node src/scripts/reembed.ts [--dry-run] [--batch <n>]";
  const args = argv.slice(2);
  const batchAt = args.indexOf("--batch");
  const batchSize = batchAt === -1 ? DEFAULT_BATCH : Number(args[batchAt + 1]);
  const rest = args.filter(
    (arg, index) => arg !== "--batch" && arg !== "--dry-run" && (batchAt === -1 || index !== batchAt + 1),
  );

  if (rest.length > 0 || !Number.isInteger(batchSize) || batchSize < 1) {
    throw new Error(usage);
  }

  return { dryRun: args.includes("--dry-run"), batchSize };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const config = loadConfig();
  const cli = parseCliArgs(process.argv);
  const store = await openStore(config.storePath, config.embedDim);
  const embedder = new OllamaEmbedder(
    config.embedUrl,
    config.embedModel,
    config.embedDim,
    fetch,
    config.embedTimeoutMs,
  );
  const build = () => fetchEmbedderBuild(config.embedUrl, config.embedModel);
  const log = (event: Record<string, unknown>) => console.log(JSON.stringify(event));

  if (cli.dryRun) {
    log({
      event: "reembed-dry-run",
      ollama: await build(),
      stamp: await readEmbedderStamp(store),
      ...(await dryRun(store, embedder)),
    });
  } else {
    const report = await reembedAndStamp(store, embedder, build, { batchSize: cli.batchSize, onProgress: log });

    log({ event: "reembed-done", ...report });
  }
}
