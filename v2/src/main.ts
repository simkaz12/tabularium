import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { TypeSafeClient } from "@typesafe-ai/sdk";

import { CLEANER_SLOT, Cleaner, scheduleAt } from "./cleaner.ts";
import { loadConfig } from "./config.ts";
import type { Config } from "./config.ts";
import { sweepDefects } from "./defectSweep.ts";
import type { JevClient } from "./domain.ts";
import { OllamaEmbedder } from "./embed.ts";
import { checkEmbedderBuild } from "./embedderBuild.ts";
import { createHttpServer, createServices } from "./http.ts";
import { keyedRunClock } from "./settings.ts";
import { openStore } from "./store.ts";

const ASK_SWEEP_INTERVAL_MS = 60_000;
const DEFECT_SWEEP_STATE_KEY = "meta:lastDefectSweepAt";

export function createJevClient(config: Config): JevClient | null {
  if (config.typesafeApiKey === null) {
    return null;
  }

  return new TypeSafeClient({
    apiKey: config.typesafeApiKey,
    retry: { maxRetries: 0 },
  }) as unknown as JevClient;
}

export async function start(config: Config = loadConfig()) {
  mkdirSync(config.storePath, { recursive: true });

  const store = await openStore(config.storePath, config.embedDim);
  const embedder = new OllamaEmbedder(
    config.embedUrl,
    config.embedModel,
    config.embedDim,
    fetch,
    config.embedTimeoutMs,
  );
  const jev = createJevClient(config);
  const services = createServices(store, embedder, {
    proposals: {
      jev,
      model: config.assessorModel,
      topicsPath: config.topicsPath,
      timeoutMs: config.jevTimeoutMs,
      containment: config.containment,
      containmentFloor: config.containmentFloor,
      foldAsk: config.foldAsk,
      foldSimilarityFloor: config.foldSimilarityFloor,
      foldSubjectFloor: config.foldSubjectFloor,
      foldAskTimeoutMs: config.foldAskTimeoutMs,
    },
  });
  const server = createHttpServer(services, config.allowedHosts);
  const cleaner = new Cleaner({
    store,
    proposals: services.proposals,
    jev,
    model: config.assessorModel,
    timeoutMs: config.jevTimeoutMs,
    settings: services.settings,
    state: services.cleanerState,
  });
  const schedule = scheduleAt(
    async () => {
      const report = await cleaner.run();

      console.log(
        JSON.stringify({
          event: "cleaner-run",
          proposed: report.proposed.length,
          candidates: report.candidates,
          jevCalls: report.jevCalls,
        }),
      );
    },
    { ...CLEANER_SLOT, state: services.cleanerState },
  );

  const defectSweepSchedule = scheduleAt(
    async () => {
      const report = await sweepDefects({
        store,
        records: services.records,
        proposals: services.proposals,
      });

      console.log(
        JSON.stringify({ event: "defect-sweep", released: report.released, asked: report.asked }),
      );
    },
    {
      weekday: null,
      hour: 5,
      label: "defect-sweep",
      state: keyedRunClock(services.cleanerState, DEFECT_SWEEP_STATE_KEY),
    },
  );

  const expireAsks = async () => {
    try {
      const expired = await services.proposals.expireAsks();

      if (expired.length > 0) {
        console.log(JSON.stringify({ event: "asks-expired", count: expired.length }));
      }
    } catch (error) {
      console.error(error);
    }
  };
  const asksTimer = setInterval(expireAsks, ASK_SWEEP_INTERVAL_MS);

  asksTimer.unref();

  void expireAsks();

  services.embedderHealth
    .refresh(() => checkEmbedderBuild(store, config.embedUrl, config.embedModel))
    .catch((error) => console.error(error));

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(config.port, config.host, resolve);
  });

  console.log(
    JSON.stringify({ event: "listening", host: config.host, port: config.port, jev: jev !== null }),
  );

  return { server, store, services, schedule, defectSweepSchedule, asksTimer };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { server, schedule, defectSweepSchedule, asksTimer } = await start();

  for (const signal of ["SIGINT", "SIGTERM"] as const) {
    process.on(signal, () => {
      schedule.stop();
      defectSweepSchedule.stop();
      clearInterval(asksTimer);
      server.close(() => process.exit(0));
    });
  }
}
