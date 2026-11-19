import { readFileSync } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";
import { hostname } from "node:os";

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

import { HttpError } from "./domain.ts";
import type { Proposal } from "./domain.ts";
import { EmbedError } from "./embed.ts";
import { CWD_REQUIRED, CwdError, resolveCwd } from "./gitContext.ts";
import type { CwdContext } from "./gitContext.ts";
import type { Services } from "./http.ts";
import { isReservedScope, sharedScopeOfProfile } from "./scopeMap.ts";

interface PropertySchema {
  type: "string" | "integer" | "array";
  description?: string;
  enum?: string[];
  maxLength?: number;
  minimum?: number;
  maximum?: number;
  minItems?: number;
  maxItems?: number;
  items?: { type: "string" };
}

interface ToolDefinition {
  name: string;
  title: string;
  description: string;
  inputSchema: {
    type: "object";
    properties: Record<string, PropertySchema>;
    required: string[];
  };
  annotations: {
    readOnlyHint: boolean;
    destructiveHint: boolean;
    idempotentHint: boolean;
    openWorldHint: boolean;
  };
}

type Args = Record<string, unknown>;

class ToolInputError extends Error {}

const KINDS = ["decision", "fact", "defect"];
const AUDIENCES = ["decider", "builder", "reviewer", "navigator"];

const CWD: PropertySchema = {
  type: "string",
  description: "your current working folder, absolute",
};

const text = (maxLength?: number, description?: string): PropertySchema => ({
  type: "string",
  ...(maxLength === undefined ? {} : { maxLength }),
  ...(description === undefined ? {} : { description }),
});

const readOnly = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: true };
const writes = { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: true };

export const TOOLS: ToolDefinition[] = [
  {
    name: "memory_propose",
    title: "Propose a memory",
    description:
      "Propose a memory. The server saves it at once and returns `recordId`; the cleaner reviews it later. `status` is `approved` for a new record, or `reinforced` when the server counted your proposal against an existing record and wrote nothing new, and then `recordId` is that record. Status `asked` means the note is held, not saved: `ask` carries the existing record and the reply to give, and you answer with `memory_answer` in the same turn. `kind`: fact, decision (needs `rejected`) or defect (needs `where`, `trigger`, `left_because`; `claim` is the symptom). `headline` is the claim in one sentence. No ticket ids, dates or file:line in `claim`. `closes` closes that defect at once. `supersedes` retires the notes this one replaces. `flags` and `neighbours` show what the checks found and which saved records are close.",
    inputSchema: {
      type: "object",
      properties: {
        kind: { type: "string", enum: KINDS },
        headline: text(100),
        claim: text(2000),
        rejected: text(2000),
        where: text(500),
        trigger: text(1000),
        left_because: text(1000),
        fix_hint: text(1000),
        impact: text(1000),
        audience: { type: "string", enum: AUDIENCES },
        stage: text(40),
        closes: text(),
        supersedes: {
          type: "array",
          items: { type: "string" },
          minItems: 1,
          maxItems: 5,
          description:
            "Ids of notes this one replaces, from `neighbours` or recall. They are retired when this note saves, same scope only. Restate anything from them that still holds, since each is replaced whole. Close defects with `closes`, not here.",
        },
        cwd: CWD,
      },
      required: ["kind", "headline", "claim", "cwd"],
    },
    annotations: writes,
  },
  {
    name: "memory_answer",
    title: "Answer a held memory",
    description:
      "Answer a held `memory_propose` (status `asked`). The held note must be in this project's profile. Pass `answers` (its `proposalId`) and either one merged note (`kind`, `headline`, `claim`, and `rejected` for a decision) that replaces both records, or `keep_both` with a one-line reason naming the fact in the existing note that your note is not about. Returns the resulting `status` and `recordId`.",
    inputSchema: {
      type: "object",
      properties: {
        answers: text(),
        keep_both: text(500),
        kind: { type: "string", enum: ["fact", "decision"] },
        headline: text(100),
        claim: text(2000),
        rejected: text(2000),
        cwd: CWD,
      },
      required: ["answers", "cwd"],
    },
    annotations: writes,
  },
  {
    name: "memory_recall",
    title: "Search memory",
    description:
      "Search saved memory for live rules, decisions and open defects that bear on your work. `task` says what you are working on, so search and scoring follow your work, not just the query words. The strongest results carry headline and claim; the rest carry the headline only, which states the rule. `memory_get` returns a record's why and rejected. Leave `limit` unset unless you need a wide sweep. Defaults to this project's scope; `scope` searches another scope in the same profile, `kind` filters, `audience` keeps records written for that role or for everyone.",
    inputSchema: {
      type: "object",
      properties: {
        query: text(500),
        task: text(
          undefined,
          "Your current work: `<area>: <what you are doing>`, e.g. `recall noise: setting the drop floor for weak rows`. Put the area first. Up to about 300 characters; longer is cut. Never paste a spec or a prompt.",
        ),
        scope: text(
          undefined,
          "Leave empty unless the user names one. Must be a scope in the current project's profile; a scope from another profile is refused.",
        ),
        kind: { type: "string", enum: KINDS },
        audience: { type: "string", enum: AUDIENCES },
        limit: { type: "integer", minimum: 1, maximum: 20 },
        cwd: CWD,
      },
      required: ["query", "task", "cwd"],
    },
    annotations: readOnly,
  },
  {
    name: "memory_get",
    title: "Read a memory",
    description:
      "Read one memory record by `id`, with its supersession lineage. The record must be in this project's profile; one from another profile is refused.",
    inputSchema: { type: "object", properties: { id: text(), cwd: CWD }, required: ["id", "cwd"] },
    annotations: readOnly,
  },
  {
    name: "memory_take",
    title: "Take a defect",
    description:
      "Mark an open defect as taken by this thread because you are folding its fix into current work. Returns the defect's status. A take left for 14 days reverts to open.",
    inputSchema: {
      type: "object",
      properties: { recordId: text(), cwd: CWD },
      required: ["recordId", "cwd"],
    },
    annotations: writes,
  },
  {
    name: "memory_release",
    title: "Release a defect",
    description:
      "Return a taken defect to open when the work that took it will not fix it. Returns the defect's status.",
    inputSchema: {
      type: "object",
      properties: { recordId: text(), cwd: CWD },
      required: ["recordId", "cwd"],
    },
    annotations: { ...writes, idempotentHint: true },
  },
];

function normalizeValue(value: unknown, schema: PropertySchema, name: string): unknown {
  if (schema.type === "integer") {
    return normalizeInteger(value, schema, name);
  }

  if (schema.type === "array") {
    return normalizeList(value, schema, name);
  }

  return normalizeString(value, schema, name);
}

function missingMessage(name: string): string {
  return name === "cwd" ? CWD_REQUIRED : `${name} is required`;
}

function normalizeString(value: unknown, schema: PropertySchema, name: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new ToolInputError(name === "cwd" ? CWD_REQUIRED : `${name} must be a non-empty string`);
  }

  const trimmed = value.trim();

  if (schema.maxLength !== undefined && trimmed.length > schema.maxLength) {
    throw new ToolInputError(`${name} must be ${schema.maxLength} characters or fewer`);
  }

  if (schema.enum !== undefined && !schema.enum.includes(trimmed)) {
    throw new ToolInputError(`${name} must be one of ${schema.enum.join(", ")}`);
  }

  return trimmed;
}

function normalizeInteger(value: unknown, schema: PropertySchema, name: string): number {
  const below = typeof value === "number" && schema.minimum !== undefined && value < schema.minimum;
  const above = typeof value === "number" && schema.maximum !== undefined && value > schema.maximum;

  if (typeof value !== "number" || !Number.isInteger(value) || below || above) {
    throw new ToolInputError(`${name} must be an integer from ${schema.minimum} to ${schema.maximum}`);
  }

  return value;
}

function normalizeList(value: unknown, schema: PropertySchema, name: string): string[] {
  if (!Array.isArray(value)) {
    throw new ToolInputError(`${name} must be a list`);
  }

  const tooFew = schema.minItems !== undefined && value.length < schema.minItems;
  const tooMany = schema.maxItems !== undefined && value.length > schema.maxItems;

  if (tooFew || tooMany) {
    throw new ToolInputError(`${name} must hold ${schema.minItems} to ${schema.maxItems} entries`);
  }

  return value.map((entry) => normalizeString(entry, { type: "string" }, `${name} entry`));
}

export function normalizeArgs(tool: ToolDefinition, raw: unknown): Args {
  const source = (typeof raw === "object" && raw !== null && !Array.isArray(raw) ? raw : {}) as Args;
  const args: Args = {};

  for (const name of tool.inputSchema.required) {
    if (source[name] === undefined || source[name] === null) {
      throw new ToolInputError(missingMessage(name));
    }
  }

  for (const [name, schema] of Object.entries(tool.inputSchema.properties)) {
    if (source[name] !== undefined && source[name] !== null) {
      args[name] = normalizeValue(source[name], schema, name);
    }
  }

  return args;
}

const str = (value: unknown): string | undefined => (typeof value === "string" ? value : undefined);

function proposalProblem(args: Args): string | null {
  if (args.closes !== undefined && args.supersedes !== undefined) {
    return "Pass `closes` or `supersedes`, not both: closing a defect and replacing notes are separate proposals.";
  }

  if (args.closes !== undefined) {
    return null;
  }

  if (args.kind === "decision" && args.rejected === undefined) {
    return "A decision needs `rejected`: the alternative that was ruled out.";
  }

  if (args.kind !== "defect") {
    return null;
  }

  const missing = ["where", "trigger", "left_because"].filter((field) => args[field] === undefined);

  return missing.length === 0 ? null : `A defect needs ${missing.map((field) => `\`${field}\``).join(", ")}.`;
}

function proposalFields(args: Args): Record<string, unknown> {
  const defect = ["claim", "where", "trigger", "left_because", "fix_hint", "impact"];
  const names = args.kind === "defect" ? defect : ["claim", "rejected"];
  const present = names.filter((name) => args[name] !== undefined);

  return Object.fromEntries(present.map((name) => [name, args[name]]));
}

function proposalBody(args: Args, context: CwdContext): Record<string, unknown> {
  return {
    kind: args.kind,
    headline: args.headline,
    fields: proposalFields(args),
    scope: context.scope,
    provenance: {
      threadId: "",
      threadTitle: "",
      projectId: context.projectId,
      branch: context.branch,
      machine: hostname(),
      link: "",
      excerpt: "",
      stage: str(args.stage) ?? "",
    },
    origin: "agent",
    ...(args.audience === undefined ? {} : { audience: args.audience }),
    ...(args.supersedes === undefined ? {} : { supersedes: args.supersedes }),
    ...(args.closes === undefined ? {} : { target: { recordId: args.closes, source: "v2", action: "close" } }),
  };
}

const noteText = (headline: string, claim: string, rejected: string | null | undefined) =>
  [headline, claim, ...(rejected ? [`Rejected: ${rejected}`] : [])].join("\n");

const WHY_RULE =
  "The merged note has no why field. Carry the concrete details from the existing note's why (commands, paths, numbers, config keys, conditions) into the claim, and leave out the account of how it was found, such as incident dates and lost round trips.";

function askInstruction(proposal: Proposal, args: Args): string {
  const existing = (proposal.ask as NonNullable<Proposal["ask"]>).existing;
  const why = existing.fields.why?.trim() ?? "";
  const incoming = noteText(str(args.headline) ?? "", str(args.claim) ?? "", str(args.rejected));

  return [
    "Your note was not saved yet. The store says it looks like the same thing as a note it already holds.",
    "",
    "EXISTING NOTE (already in the store):",
    noteText(existing.headline, existing.fields.claim, existing.fields.rejected),
    ...(why === "" ? [] : [`Why it was saved: ${why}`]),
    "",
    "YOUR NOTE (the one you just sent):",
    incoming,
    "",
    "Send back the one merged note that holds everything from both. It replaces both in the store.",
    "Keep every concrete detail from both notes: names, numbers, paths, config keys, dates, rejected options and the reasons they were rejected.",
    ...(why === "" ? [] : [WHY_RULE]),
    "Add nothing that is in neither note. No new advice, no inference, no framing of your own.",
    "Where the notes conflict, your note is the newer one and wins. Give the newer value and say that the older value was replaced.",
    "Write it as one fact or one decision, not as two notes joined together.",
    `Reply by calling memory_answer with answers: "${proposal.id}", kind, headline, and claim holding the merged note only: no preamble, no heading, no quotes, no explanation. For a decision, also pass rejected.`,
    "",
    "The only other reply is keep_both, and it is allowed only with a one-line reason that names the fact in the existing note that your note is not about. A reason that does not name such a fact is not accepted.",
    `Reply by calling memory_answer with answers: "${proposal.id}" and keep_both set to that one line, and nothing else.`,
  ].join("\n");
}

function askOf(proposal: Proposal, args: Args) {
  if (proposal.status !== "asked" || !proposal.ask) {
    return null;
  }

  const existing = proposal.ask.existing;

  return {
    proposalId: proposal.id,
    existing: {
      id: existing.id,
      kind: existing.kind,
      headline: existing.headline,
      claim: existing.fields.claim,
      rejected: existing.fields.rejected ?? null,
    },
    instruction: askInstruction(proposal, args),
  };
}

function proposeResult(proposal: Proposal, args: Args) {
  return {
    proposalId: proposal.id,
    status: proposal.status,
    recordId: proposal.resolution?.recordId ?? null,
    ask: askOf(proposal, args),
    flags: [...new Set(proposal.flags)],
    neighbours: proposal.neighbours.map(({ id, source, headline, similarity, relation }) => ({
      id,
      source,
      headline,
      similarity,
      relation: relation?.pick ?? null,
    })),
  };
}

function answerBody(args: Args): Record<string, unknown> {
  const noteGiven = ["kind", "headline", "claim", "rejected"].some((name) => args[name] !== undefined);

  if (args.keep_both !== undefined) {
    if (noteGiven) {
      throw new ToolInputError("Pass either `keep_both` or a merged note (`kind`, `headline`, `claim`), not both.");
    }

    return { action: "keep_both", reason: args.keep_both };
  }

  if (args.kind === undefined || args.headline === undefined || args.claim === undefined) {
    throw new ToolInputError("Pass `keep_both` with a reason, or a merged note with `kind`, `headline` and `claim`.");
  }

  if (args.kind === "decision" && args.rejected === undefined) {
    throw new ToolInputError("A decision needs `rejected`: the alternative that was ruled out.");
  }

  return {
    action: "fold",
    kind: args.kind,
    headline: args.headline,
    fields: { claim: args.claim, ...(args.rejected === undefined ? {} : { rejected: args.rejected }) },
  };
}

function takerOf(context: CwdContext): string {
  return `${context.projectId}@${context.branch}`;
}

async function requireLinkedRecordsInProfile(services: Services, context: CwdContext, args: Args): Promise<void> {
  const ids = [...(str(args.closes) === undefined ? [] : [args.closes as string]), ...((args.supersedes as string[] | undefined) ?? [])];

  for (const id of ids) {
    await requireRecordInProfile(services, context, id);
  }
}

async function propose(services: Services, args: Args) {
  const problem = proposalProblem(args);

  if (problem !== null) {
    throw new ToolInputError(problem);
  }

  const context = await resolveCwd(args.cwd, services.scopeMap.current);

  await requireLinkedRecordsInProfile(services, context, args);

  const proposal = await services.proposals.create(proposalBody(args, context));

  return proposeResult(proposal, args);
}

async function answer(services: Services, args: Args) {
  const body = answerBody(args);
  const context = await resolveCwd(args.cwd, services.scopeMap.current);

  requireProfile(services, context, (await services.proposals.get(args.answers as string)).scope);

  const proposal = await services.proposals.answer(args.answers as string, body);

  return {
    proposalId: proposal.id,
    status: proposal.status,
    recordId: proposal.resolution?.recordId ?? null,
  };
}

const OTHER_PROFILE = "that belongs to another profile, and memory cannot cross profiles";

function requireProfile(services: Services, context: CwdContext, scope: string): void {
  const inProfile = isReservedScope(scope)
    ? scope === sharedScopeOfProfile(context.profile)
    : services.scopeMap.current.profileOf(scope) === context.profile;

  if (!inProfile) {
    throw new ToolInputError(OTHER_PROFILE);
  }
}

async function requireRecordInProfile(services: Services, context: CwdContext, recordId: string): Promise<void> {
  requireProfile(services, context, (await services.records.get(recordId)).scope);
}

async function recall(services: Services, args: Args) {
  const context = await resolveCwd(args.cwd, services.scopeMap.current);
  const scope = str(args.scope) ?? context.scope;

  requireProfile(services, context, scope);

  return services.recall.recall({
    q: args.query as string,
    task: args.task as string,
    scope,
    kind: str(args.kind) ?? null,
    audience: str(args.audience) ?? null,
    limit: typeof args.limit === "number" ? args.limit : null,
    threadId: takerOf(context),
  });
}

async function get(services: Services, args: Args) {
  const context = await resolveCwd(args.cwd, services.scopeMap.current);

  await requireRecordInProfile(services, context, args.id as string);

  return services.records.read(args.id as string, null);
}

async function defectState(
  services: Services,
  context: CwdContext,
  recordId: string,
  action: () => Promise<unknown>,
) {
  await requireRecordInProfile(services, context, recordId);
  await action();

  const view = await services.records.get(recordId);

  return { recordId, defectStatus: view.defectStatus, takenBy: view.takenBy };
}

async function take(services: Services, args: Args) {
  const context = await resolveCwd(args.cwd, services.scopeMap.current);
  const taker = takerOf(context);
  const recordId = args.recordId as string;

  const act = () => services.records.take(recordId, { threadId: taker, link: taker });

  return defectState(services, context, recordId, act);
}

async function release(services: Services, args: Args) {
  const context = await resolveCwd(args.cwd, services.scopeMap.current);
  const recordId = args.recordId as string;

  const act = () => services.records.release(recordId, { threadId: takerOf(context) });

  return defectState(services, context, recordId, act);
}

type Handler = (services: Services, args: Args) => Promise<unknown>;

const HANDLERS: Record<string, Handler> = {
  memory_propose: propose,
  memory_answer: answer,
  memory_recall: recall,
  memory_get: get,
  memory_take: take,
  memory_release: release,
};

function failure(message: string) {
  return { isError: true, content: [{ type: "text" as const, text: message }] };
}

function success(result: unknown) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(result) }],
    structuredContent: result as Record<string, unknown>,
  };
}

export async function callTool(services: Services, name: string, raw: unknown) {
  const tool = TOOLS.find((candidate) => candidate.name === name);

  if (tool === undefined) {
    return failure(`unknown tool ${name}`);
  }

  try {
    return success(await (HANDLERS[name] as Handler)(services, normalizeArgs(tool, raw)));
  } catch (error) {
    return failure(messageOf(error));
  }
}

function messageOf(error: unknown): string {
  if (error instanceof ToolInputError || error instanceof CwdError || error instanceof HttpError) {
    return error.message;
  }

  if (error instanceof EmbedError) {
    return "embedding service unavailable";
  }

  console.error(error);

  return "internal error";
}

const MANIFEST = readFileSync(new URL("../package.json", import.meta.url), "utf8");
const VERSION = (JSON.parse(MANIFEST) as { version: string }).version;

function buildServer(services: Services): Server {
  const server = new Server({ name: "tabumcp", version: VERSION }, { capabilities: { tools: {} } });

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: TOOLS.map(({ name, title, description, inputSchema, annotations }) => ({
      name,
      title,
      description,
      inputSchema,
      annotations,
    })),
  }));
  server.setRequestHandler(CallToolRequestSchema, async (request) =>
    callTool(services, request.params.name, request.params.arguments),
  );

  return server;
}

export async function handleMcp(
  services: Services,
  req: IncomingMessage,
  res: ServerResponse,
  body: unknown,
): Promise<void> {
  const server = buildServer(services);
  const transport = new StreamableHTTPServerTransport({ enableJsonResponse: true });

  res.on("close", () => {
    void transport.close();
    void server.close();
  });

  await server.connect(transport);
  await transport.handleRequest(req, res, body);
}
