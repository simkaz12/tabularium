import { createHash, timingSafeEqual } from "node:crypto";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import type { IncomingMessage, Server, ServerResponse } from "node:http";

import { DEFAULT_PROFILE, emptyScopeMap, loadScopeMap } from "./scopeMap.ts";
import type { ScopeMap } from "./scopeMap.ts";
import { HttpError } from "./domain.ts";
import { EmbedError } from "./embed.ts";
import { handleMcp } from "./mcp.ts";
import { EmbedderHealth } from "./embedderBuild.ts";
import type { Embedder } from "./embed.ts";
import { ProposalService } from "./proposals.ts";
import type { ProposalOptions } from "./proposals.ts";
import { RecallService } from "./recall.ts";
import { RecordService, threadIdOf } from "./records.ts";
import { SettingsService, StoreCleanerState } from "./settings.ts";
import type { CleanerState } from "./settings.ts";
import { sqlString } from "./store.ts";
import type { Store } from "./store.ts";

const BODY_LIMIT = 1_000_000;
const PREFIX = "/v2";
const MCP_PATH = "/mcp";

export interface ScopeMapRef {
  current: ScopeMap;
  reload(): void;
}

export function createScopeMapRef(file: string | null, initial?: ScopeMap): ScopeMapRef {
  const load = () => (file === null ? emptyScopeMap() : loadScopeMap(file));
  const ref: ScopeMapRef = {
    current: initial ?? load(),
    reload: () => {
      ref.current = load();
    },
  };

  return ref;
}

export interface Services {
  store: Store;
  scopeMap: ScopeMapRef;
  records: RecordService;
  proposals: ProposalService;
  recall: RecallService;
  settings: SettingsService;
  cleanerState: CleanerState;
  embedderHealth: EmbedderHealth;
}

export interface ServiceOptions {
  scopeMap?: ScopeMapRef;
  proposals?: ProposalOptions;
  now?: () => string;
}

export function createServices(store: Store, embedder: Embedder, options: ServiceOptions = {}): Services {
  const now = options.now ?? (() => new Date().toISOString());
  const records = new RecordService(store, embedder, now);
  const scopeMap = options.scopeMap ?? createScopeMapRef(null);
  const sharedScopeOf = (scope: string) => scopeMap.current.sharedScopeOf(scope);
  const profileOf = (scope: string) => scopeMap.current.profileOf(scope);

  return {
    store,
    scopeMap,
    records,
    proposals: new ProposalService(store, embedder, records, now, { sharedScopeOf, profileOf, ...options.proposals }),
    recall: new RecallService(store, embedder, options.proposals ?? {}, now, sharedScopeOf),
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

function reloadScopeMap(services: Services): { profiles: string[] } {
  try {
    services.scopeMap.reload();
  } catch (error) {
    throw new HttpError(400, (error as Error).message);
  }

  return { profiles: services.scopeMap.current.profiles };
}

interface ProfileCounts {
  profile: string;
  scopes: string[];
  records: number;
  proposals: number;
}

function profileWhere(scopes: string[]): string {
  return `scope IN (${scopes.map(sqlString).join(", ")})`;
}

function knownProfile(services: Services, name: string): string[] {
  if (name === DEFAULT_PROFILE) {
    throw new HttpError(400, "the default profile cannot be counted or deleted");
  }

  if (!services.scopeMap.current.profiles.includes(name)) {
    throw new HttpError(404, `profile ${name} not found`);
  }

  return services.scopeMap.current.scopesOf(name);
}

async function countProfile(services: Services, name: string, scopes: string[]): Promise<ProfileCounts> {
  const { records, legacy, proposals } = services.store;
  const where = profileWhere(scopes);
  const [current, archived, pending] = await Promise.all([
    records.countRows(where),
    legacy.countRows(where),
    proposals.countRows(where),
  ]);

  return { profile: name, scopes, records: current + archived, proposals: pending };
}

async function profileCounts(services: Services, name: string): Promise<ProfileCounts> {
  return countProfile(services, name, knownProfile(services, name));
}

async function deleteProfile(services: Services, name: string): Promise<ProfileCounts> {
  const scopes = knownProfile(services, name);
  const where = profileWhere(scopes);

  return services.store.withLock(async () => {
    const counts = await countProfile(services, name, scopes);
    const { records, legacy, proposals } = services.store;

    await records.delete(where);
    await legacy.delete(where);
    await proposals.delete(where);
    console.log(JSON.stringify({ event: "profile-deleted", ...counts }));

    return counts;
  });
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

  if (head === "scope-map" && id === "reload" && method === "POST" && parts.length === 2) {
    return [200, reloadScopeMap(services)];
  }

  if (head === "profiles" && id !== undefined) {
    if (method === "GET" && action === "count" && parts.length === 3) {
      return [200, await profileCounts(services, id)];
    }

    if (method === "DELETE" && parts.length === 2) {
      return [200, await deleteProfile(services, id)];
    }
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
        threadId: query.get("threadId"),
      }),
    ];
  }

  if (head === "records" && id !== undefined) {
    if (method === "GET" && parts.length === 2) {
      return [200, await services.records.read(id, query.get("threadId"))];
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
      const body = await readBody(req);

      return [200, await services.records.release(id, { threadId: threadIdOf(body) })];
    }

    if (method === "DELETE" && parts.length === 2) {
      const reason = secretDeletion(await readBody(req));

      return [200, await services.records.remove(id, reason)];
    }
  }

  throw new HttpError(404, "not found");
}

async function serveMcp(services: Services, req: IncomingMessage, res: ServerResponse): Promise<void> {
  if (req.method !== "POST") {
    res.setHeader("allow", "POST");

    throw new HttpError(405, "the MCP endpoint takes POST only");
  }

  await handleMcp(services, req, res, await readBody(req));
}

export interface HttpOptions {
  token?: string | null;
  mcp?: boolean;
}

export function readToken(file: string | null): string {
  if (file === null) {
    throw new Error("no token file configured: set TABUMCP_TOKEN_FILE");
  }

  let raw: string;

  try {
    raw = readFileSync(file, "utf8");
  } catch (error) {
    throw new Error(`cannot read token file ${file}: ${(error as Error).message}`);
  }

  const token = raw.trim();

  if (token === "") {
    throw new Error(`token file ${file} is empty`);
  }

  return token;
}

function digest(value: string): Buffer {
  return createHash("sha256").update(value).digest();
}

function assertAuthorized(req: IncomingMessage, token: string): void {
  if (req.headers.origin !== undefined) {
    throw new HttpError(403, "origin not allowed");
  }

  const match = /^Bearer (.+)$/i.exec(req.headers.authorization ?? "");

  if (match === null || !timingSafeEqual(digest(match[1] as string), digest(token))) {
    throw new HttpError(401, "missing or wrong bearer token");
  }
}

export function createHttpServer(services: Services, allowedHosts: string[], options: HttpOptions = {}): Server {
  const allowed = new Set(allowedHosts);
  const token = options.token ?? null;
  const mcp = options.mcp ?? false;

  return createServer(async (req, res) => {
    try {
      const host = req.headers.host ?? "";

      if (!allowed.has(host)) {
        throw new HttpError(403, "host not allowed");
      }

      if (token !== null) {
        assertAuthorized(req, token);
      }

      const url = new URL(req.url ?? "/", `http://${host}`);

      if (mcp && url.pathname.replace(/\/+$/, "") === MCP_PATH) {
        await serveMcp(services, req, res);

        return;
      }

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
