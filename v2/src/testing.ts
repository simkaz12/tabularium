import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { recordText } from "./domain.ts";
import type { JevClient, JevRequest, JevResponse, LegacyRecord, MemoryRecord, ProposalBody } from "./domain.ts";
import { HashEmbedder } from "./embed.ts";
import { createServices } from "./http.ts";
import type { ServiceOptions, Services } from "./http.ts";
import { insert, openStore } from "./store.ts";
import type { Store } from "./store.ts";

export const TEST_DIM = 64;

export interface Harness {
  store: Store;
  embedder: HashEmbedder;
  services: Services;
  clock: { value: string };
  cleanup(): void;
}

export async function makeHarness(options: ServiceOptions = {}): Promise<Harness> {
  const dir = mkdtempSync(join(tmpdir(), "tabularium-v2-"));
  const store = await openStore(dir, TEST_DIM);
  const embedder = new HashEmbedder(TEST_DIM);
  const clock = { value: "2026-11-01T10:00:00.000Z" };
  const services = createServices(store, embedder, { now: () => clock.value, ...options });

  return {
    store,
    embedder,
    services,
    clock,
    cleanup: () => rmSync(dir, { recursive: true, force: true }),
  };
}

export function proposalBody(overrides: Partial<ProposalBody> = {}): ProposalBody {
  return {
    kind: "fact",
    headline: "repl readline crashes on the history file",
    fields: {
      claim: "repl aborts with a readline error when the history file is unwritable",
    },
    scope: "project-a",
    provenance: {
      threadId: "thread-1",
      threadTitle: "Fix the test harness",
      projectId: "project-1",
      branch: "main",
      machine: "laptop",
      link: "t3-thread://thread-1",
      excerpt: "user: repl fails",
      stage: "",
    },
    origin: "agent",
    ...overrides,
  };
}

export function defectBody(overrides: Partial<ProposalBody> = {}): ProposalBody {
  return proposalBody({
    kind: "defect",
    headline: "the export job drops rows past the first page",
    fields: {
      claim: "the export job writes only the first 500 rows of a report",
      where: "app/Jobs/SyncReport.php",
      trigger: "export a report with more than 500 rows",
      leftBecause: "the fix needs an offset change outside the ticket",
    },
    ...overrides,
  });
}

export async function createApproved(
  harness: Harness,
  body: ProposalBody,
): Promise<MemoryRecord> {
  const proposal = await harness.services.proposals.create(body);

  return (await harness.services.records.get(proposal.resolution?.recordId as string)) as unknown as MemoryRecord;
}

export function createApprovedDefect(
  harness: Harness,
  overrides: Partial<ProposalBody> = {},
): Promise<MemoryRecord> {
  return createApproved(harness, defectBody(overrides));
}

function recordDraft(id: string, stamp: string): Omit<MemoryRecord, "vector"> {
  return {
    id,
    kind: "fact",
    headline: "seeded record",
    claim: "seeded claim",
    why: null,
    rejected: null,
    scope: "project-a",
    purpose: null,
    topic: null,
    audience: null,
    reader: null,
    where: null,
    trigger: null,
    leftBecause: null,
    fixHint: null,
    impact: null,
    defectStatus: null,
    takenThreadId: null,
    takenLink: null,
    takenAt: null,
    closedBy: null,
    closedAt: null,
    lifecycle: "live",
    supersededBy: null,
    supersessionReason: null,
    retiredReason: null,
    lifecycleNote: null,
    threadId: "",
    threadTitle: "",
    projectId: "",
    branch: "",
    machine: "",
    link: "",
    excerpt: "",
    stage: "",
    origin: "agent",
    legacyId: null,
    reinforcedCount: 0,
    lastReinforcedAt: null,
    createdAt: stamp,
    approvedAt: stamp,
  };
}

function legacyDraft(id: string, stamp: string): Omit<MemoryRecord, "vector"> {
  return { ...recordDraft(id, stamp), why: "seeded why" };
}

let sequence = 0;

function nextId(prefix: string): string {
  sequence += 1;

  return `${prefix}-${sequence}`;
}

export async function seedRecord(
  harness: Harness,
  overrides: Partial<MemoryRecord> = {},
): Promise<MemoryRecord> {
  const draft = { ...recordDraft(overrides.id ?? nextId("rec"), harness.clock.value), ...overrides };
  const vector = overrides.vector ?? (await harness.embedder.embed(recordText(draft)));
  const record: MemoryRecord = { ...draft, vector };

  await insert(harness.store.records, [record as unknown as Record<string, unknown>]);

  return record;
}

export async function seedLegacy(
  harness: Harness,
  overrides: Partial<LegacyRecord> = {},
): Promise<LegacyRecord> {
  const id = overrides.id ?? nextId("legacy");
  const draft = { ...legacyDraft(id, harness.clock.value), legacyId: id, ...overrides };
  const vector = overrides.vector ?? (await harness.embedder.embed(recordText(draft)));
  const legacy: LegacyRecord = {
    ...draft,
    vector,
    oldScope: draft.scope,
    oldSource: "v1",
    readCount: 0,
    originalCreatedAt: draft.createdAt,
    reviewed: false,
    collapsedIds: null,
    ...overrides,
  };

  await insert(harness.store.legacy, [legacy as unknown as Record<string, unknown>]);

  return legacy;
}

type WritableTable = "records" | "legacy" | "proposals" | "readLog";

function guardBuilder(builder: any, trip: () => void): any {
  return new Proxy(builder, {
    get(target, property) {
      const value = Reflect.get(target, property, target);

      if (typeof value !== "function") {
        return value;
      }

      if (property === "execute") {
        return (...args: unknown[]) => {
          trip();

          return value.apply(target, args);
        };
      }

      return (...args: unknown[]) => {
        const result = value.apply(target, args);

        return typeof result === "object" && result !== null && "execute" in result
          ? guardBuilder(result, trip)
          : result;
      };
    },
  });
}

export function failNthWrite(
  harness: Harness,
  name: WritableTable,
  nth: number,
  error: Error = new Error(`injected write failure on ${name}`),
): () => void {
  const table = harness.store[name] as any;
  const methods = ["mergeInsert", "add", "delete", "update"];
  const originals: Record<string, any> = {};
  let count = 0;

  const trip = () => {
    count += 1;

    if (count === nth) {
      throw error;
    }
  };

  for (const method of methods) {
    originals[method] = table[method];
  }

  table.mergeInsert = (...args: unknown[]) => guardBuilder(originals.mergeInsert.apply(table, args), trip);

  for (const method of ["add", "delete", "update"]) {
    table[method] = (...args: unknown[]) => {
      trip();

      return originals[method].apply(table, args);
    };
  }

  return () => {
    Object.assign(table, originals);
  };
}

export interface StubJev extends JevClient {
  calls: JevRequest[];
  failWith: Error | null;
}

type AnswerFactory = (question: any, request: JevRequest) => Record<string, unknown> | undefined;

export function choiceAnswer(choice: string, confidence = 0.9): Record<string, unknown> {
  return { type: "choice", choice, confidence, probabilities: { [choice]: confidence } };
}

function defaultAnswer(id: string, question: any, request: JevRequest): Record<string, unknown> {
  const state = request.state as Record<string, any>;

  if (question.type === "noul") {
    return { type: "noul", noul: 0.9 };
  }

  if (question.type === "score") {
    return {
      type: "score",
      score: 1,
      confidence: 0.9,
      legend: {},
      probabilities: { "0": 0.05, "1": 0.8, "2": 0.1, "3": 0.05 },
    };
  }

  const fixed: Record<string, string | undefined> = {
    kind: state.declared_kind,
    worth: "keep",
    perishable: "standing",
    breadth: "project_specific",
    scope: state.source_scope,
    purpose: "pitfall",
    reader: "lookup",
    audience: "builder",
    relation: "distinct",
    reason: "changed",
    contradiction: "coexist",
    new_covers_old: "no",
    same_subject: "no",
  };

  return choiceAnswer(fixed[id] ?? (Object.keys(question.criteria)[0] as string));
}

export function stubJev(overrides: Record<string, AnswerFactory> = {}): StubJev {
  const stub: StubJev = {
    calls: [],
    failWith: null,
    async systemOne(request) {
      stub.calls.push(request);

      if (stub.failWith !== null) {
        throw stub.failWith;
      }

      const answers: Record<string, unknown> = {};

      for (const [id, question] of Object.entries(request.questions)) {
        answers[id] = overrides[id]?.(question, request) ?? defaultAnswer(id, question, request);
      }

      return { model: request.model ?? "stub", answers, usage: {} } satisfies JevResponse;
    },
  };

  return stub;
}
