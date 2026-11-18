import { createServer } from "node:http";
import type { IncomingMessage, Server, ServerResponse } from "node:http";

import { HttpError } from "./domain.ts";
import { EmbedError } from "./embed.ts";
import { EmbedderHealth } from "./embedderBuild.ts";
import type { Embedder } from "./embed.ts";
import { ProposalService } from "./proposals.ts";
import type { ProposalOptions } from "./proposals.ts";
import { RecallService } from "./recall.ts";
import { RecordService } from "./records.ts";
import { SettingsService, StoreCleanerState } from "./settings.ts";
import type { CleanerState } from "./settings.ts";
import type { Store } from "./store.ts";

const BODY_LIMIT = 1_000_000;
const PREFIX = "/v2";

export interface Services {
  records: RecordService;
  proposals: ProposalService;
  recall: RecallService;
  settings: SettingsService;
  cleanerState: CleanerState;
  embedderHealth: EmbedderHealth;
}

export interface ServiceOptions {
  proposals?: ProposalOptions;
  now?: () => string;
}

export function createServices(store: Store, embedder: Embedder, options: ServiceOptions = {}): Services {
  const now = options.now ?? (() => new Date().toISOString());
  const records = new RecordService(store, embedder, now);

  return {
    records,
    proposals: new ProposalService(store, embedder, records, now, options.proposals),
    recall: new RecallService(store, embedder, options.proposals ?? {}, now),
    settings: new SettingsService(store, now),
    cleanerState: new StoreCleanerState(store, now),
    embedderHealth: new EmbedderHealth(),
  };
}

async function readBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;

  for await (const chunk of req) {
    const buffer = chunk as Buffer;
    size += buffer.length;

    if (size > BODY_LIMIT) {
      throw new HttpError(413, "request body too large");
    }

    chunks.push(buffer);
  }

  if (size === 0) {
    return undefined;
  }

  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new HttpError(400, "request body is not valid JSON");
  }
}

function send(res: ServerResponse, status: number, payload: unknown): void {
  const body = JSON.stringify(payload);

  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

function intParam(value: string | null): number | null {
  if (value === null || value === "") {
    return null;
  }

  return Number(value);
}

function decodeSegment(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    throw new HttpError(400, "malformed percent escape in path");
  }
}

function secretDeletion(body: unknown): string {
  const fields = (typeof body === "object" && body !== null ? body : {}) as {
    reason?: unknown;
    category?: unknown;
  };

  if (typeof fields.reason !== "string" || fields.reason.trim() === "") {
    throw new HttpError(400, "reason is required");
  }

  if (fields.category !== "secret") {
    throw new HttpError(400, 'delete is for secrets only: category must be "secret"');
  }

  return fields.reason;
}

async function route(services: Services, req: IncomingMessage, url: URL): Promise<[number, unknown]> {
  const method = req.method ?? "GET";
  const path = url.pathname.replace(/\/+$/, "");
  const query = url.searchParams;

  if (!path.startsWith(`${PREFIX}/`) && path !== PREFIX) {
    throw new HttpError(404, "not found");
  }

  const parts = path.slice(PREFIX.length + 1).split("/").map(decodeSegment);
  const [head, id, action] = parts;

  if (method === "GET" && head === "health" && parts.length === 1) {
    return [200, { ok: true, embedder: services.embedderHealth.get() }];
  }

  if (head === "settings" && parts.length === 1) {
    if (method === "GET") {
      return [200, await services.settings.getSettings()];
    }

    if (method === "PUT") {
      return [200, await services.settings.updateSettings(await readBody(req))];
    }
  }

  if (head === "proposals") {
    if (method === "POST" && parts.length === 1) {
      return [201, await services.proposals.create(await readBody(req))];
    }

    if (method === "GET" && parts.length === 1) {
      return [200, await services.proposals.list(query.get("status") ?? "pending", query.get("since"))];
    }

    if (method === "POST" && parts.length === 3 && id !== undefined) {
      const body = await readBody(req);

      if (action === "approve") {
        return [200, await services.proposals.approve(id, body)];
      }

      if (action === "reject") {
        return [200, await services.proposals.reject(id, body)];
      }

      if (action === "reinforce") {
        return [200, await services.proposals.reinforce(id, body)];
      }

      if (action === "answer") {
        return [200, await services.proposals.answer(id, body)];
      }
    }
  }

  if (head === "asks" && method === "GET" && parts.length === 1) {
    return [200, await services.proposals.listAsks(query.get("since"))];
  }

  if (head === "recall" && method === "GET" && parts.length === 1) {
    const limit = intParam(query.get("limit"));

    return [
      200,
      await services.recall.recall({
        q: query.get("q") ?? "",
        scope: query.get("scope"),
        kind: query.get("kind"),
        audience: query.get("audience"),
        limit,
        task: query.get("task"),
      }),
    ];
  }

  if (head === "records" && id !== undefined) {
    if (method === "GET" && parts.length === 2) {
      return [200, await services.records.get(id)];
    }

    if (method === "POST" && parts.length === 3 && action === "unsupersede") {
      const body = (await readBody(req)) as { reason?: unknown } | undefined;

      return [200, await services.records.unsupersede(id, body?.reason)];
    }

    if (method === "POST" && parts.length === 3 && action === "archive") {
      const body = (await readBody(req)) as { reason?: unknown } | undefined;

      return [200, await services.records.archive(id, body?.reason)];
    }

    if (method === "POST" && parts.length === 3 && action === "unarchive") {
      await readBody(req);

      return [200, await services.records.unarchive(id)];
    }

    if (method === "POST" && parts.length === 3 && action === "take") {
      return [200, await services.records.take(id, await readBody(req))];
    }

    if (method === "POST" && parts.length === 3 && action === "release") {
      await readBody(req);

      return [200, await services.records.release(id)];
    }

    if (method === "DELETE" && parts.length === 2) {
      const reason = secretDeletion(await readBody(req));

      return [200, await services.records.remove(id, reason)];
    }
  }

  throw new HttpError(404, "not found");
}

export function createHttpServer(services: Services, allowedHosts: string[]): Server {
  const allowed = new Set(allowedHosts);

  return createServer(async (req, res) => {
    try {
      const host = req.headers.host ?? "";

      if (!allowed.has(host)) {
        throw new HttpError(403, "host not allowed");
      }

      const url = new URL(req.url ?? "/", `http://${host}`);
      const [status, payload] = await route(services, req, url);

      send(res, status, payload);
    } catch (error) {
      if (error instanceof HttpError) {
        send(res, error.status, { error: error.message });
        return;
      }

      if (error instanceof EmbedError) {
        console.error(error);
        send(res, 503, { error: "embedding service unavailable" });
        return;
      }

      console.error(error);
      send(res, 500, { error: "internal error" });
    }
  });
}
