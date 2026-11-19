import { randomUUID } from "node:crypto";

import {
  AUDIENCES,
  EXTERNAL_ORIGINS,
  HEADLINE_MAX,
  HttpError,
  KINDS,
  ORIGINS,
  PROPOSAL_STATUSES,
  TARGET_ACTIONS,
  recordText,
} from "./domain.ts";
import type {
  Ask,
  FoldDraft,
  Audience,
  Checks,
  Fields,
  Kind,
  JevClient,
  JevSettings,
  LegacyRecord,
  MemoryRecord,
  Neighbour,
  Origin,
  Proposal,
  ProposalBody,
  ProposalStatus,
  ProposalTarget,
  Provenance,
  Reader,
  ReasonClass,
  Source,
  StoredRecord,
  Suggestion,
} from "./domain.ts";
import type { Mode } from "./config.ts";
import { cosine } from "./embed.ts";
import { ProposalChecker, ask as askJev, recordSummary, yesProbability } from "./jev/checks.ts";
import { NEW_COVERS_OLD_QUESTION, loadTopics } from "./jev/questions.ts";
import type { Embedder } from "./embed.ts";
import { liveFilter, mergeBySource } from "./recall.ts";
import { requireReasonClass, requireText, stripVector } from "./records.ts";
import type { NewRecord, PublicRecord, RecordService } from "./records.ts";
import {
  findById,
  hybridSearch,
  proposalToRow,
  rowToProposal,
  selectWhere,
  sqlString,
  upsert,
} from "./store.ts";
import type { Store } from "./store.ts";

export const NEIGHBOUR_COUNT = 3;
const DEFAULT_CONTAINMENT_FLOOR = 0.95;
const CONTAINED_KINDS: readonly Kind[] = ["fact", "decision"];
const DEFAULT_FOLD_SIMILARITY_FLOOR = 0.8;
const DEFAULT_FOLD_SUBJECT_FLOOR = 0.7;
const DEFAULT_FOLD_ASK_TIMEOUT_MS = 600_000;
const NEIGHBOUR_TEXT_CAP = 1500;

export interface ProposalOptions extends JevSettings {
  topicsPath?: string | null;
  containment?: boolean;
  containmentFloor?: number;
  foldAsk?: boolean;
  foldSimilarityFloor?: number;
  foldSubjectFloor?: number;
  foldAskTimeoutMs?: number;
  sharedScopeOf?: (scope: string) => string;
  profileOf?: (scope: string) => string;
  mode?: Mode;
}

export interface NeighbourCandidate extends Neighbour {
  text: string;
  pairText: string;
  kind: string;
  scope: string;
}

export interface ApproveInput {
  headline?: string;
  fields?: Partial<Fields>;
  audience?: Audience | null;
  scope?: string;
  purpose?: string | null;
  topic?: string | null;
  supersedes?: string[];
  reason?: string;
}

export type AnswerInput =
  | { action: "keep_both"; reason: string }
  | { action: "fold"; kind: Kind; headline: string; fields: Fields };

export interface AskCounts {
  fold: number;
  keep_both: number;
  unanswered: number;
  open: number;
}

export interface AskReport {
  counts: AskCounts;
  asks: Proposal[];
}

interface HeldProposal extends Proposal {
  ask: Ask;
}

interface AskTarget {
  neighbour: Neighbour;
  row: StoredRecord;
}

export interface CheckOutcome {
  checks: Checks;
  flags: string[];
  neighbours: Neighbour[];
  suggestion: Suggestion | null;
  purpose: string | null;
  topic: string | null;
  audience: Audience | null;
  reader: Reader | null;
}

function asObject(value: unknown, name: string): Record<string, unknown> {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }

  throw new HttpError(400, `${name} must be an object`);
}

function optionalString(value: unknown, name: string): string | null {
  if (value === undefined || value === null) {
    return null;
  }

  if (typeof value !== "string") {
    throw new HttpError(400, `${name} must be a string`);
  }

  return value;
}

function oneOf<T extends string>(value: unknown, allowed: readonly T[], name: string): T {
  if (typeof value === "string" && (allowed as readonly string[]).includes(value)) {
    return value as T;
  }

  throw new HttpError(400, `${name} must be one of ${allowed.join(", ")}`);
}

function parseHeadline(value: unknown): string {
  const headline = requireText(value, "headline");

  if (headline.length > HEADLINE_MAX) {
    throw new HttpError(400, `headline must be ${HEADLINE_MAX} characters or fewer`);
  }

  return headline;
}

const PROVENANCE_KEYS: Array<keyof Provenance> = [
  "threadId",
  "threadTitle",
  "projectId",
  "branch",
  "machine",
  "link",
  "excerpt",
  "stage",
];

function parseProvenance(value: unknown): Provenance {
  const source = value === undefined ? {} : asObject(value, "provenance");
  const provenance = {} as Provenance;

  for (const key of PROVENANCE_KEYS) {
    provenance[key] = optionalString(source[key], `provenance.${key}`) ?? "";
  }

  return provenance;
}

function parseTarget(value: unknown): ProposalTarget | null {
  if (value === undefined || value === null) {
    return null;
  }

  const target = asObject(value, "target");

  return {
    recordId: requireText(target.recordId, "target.recordId"),
    source: oneOf<Source>(target.source, ["v2", "legacy"], "target.source"),
    action: oneOf(target.action, TARGET_ACTIONS, "target.action"),
    replacementId: optionalString(target.replacementId, "target.replacementId"),
    recordHeadline: null,
    replacementHeadline: null,
    reason: optionalString(target.reason, "target.reason"),
  };
}

function firstOf(source: Record<string, unknown>, ...keys: string[]): unknown {
  for (const key of keys) {
    if (source[key] !== undefined && source[key] !== null) {
      return source[key];
    }
  }

  return undefined;
}

function blankToNull(value: string | null): string | null {
  return value === null || value.trim() === "" ? null : value.trim();
}

function parseAudience(value: unknown, name: string): Audience | null {
  if (value === undefined || value === null) {
    return null;
  }

  return oneOf(value, AUDIENCES, name);
}

function requireDefectField(source: Record<string, unknown>, name: string, ...keys: string[]): string {
  const value = firstOf(source, ...keys);

  if (typeof value !== "string" || value.trim() === "") {
    throw new HttpError(400, `${name} is required for a defect`);
  }

  return value.trim();
}

function parseFields(source: Record<string, unknown>, kind: Kind, target: ProposalTarget | null): Fields {
  const defect = kind === "defect" && target?.action !== "close";
  const claim = defect
    ? requireDefectField(source, "symptom", "claim", "symptom")
    : requireText(source.claim, "fields.claim");
  const fields: Fields = { claim, rejected: optionalString(source.rejected, "fields.rejected") };

  if (!defect) {
    return fields;
  }

  return {
    ...fields,
    where: requireDefectField(source, "where", "where"),
    trigger: requireDefectField(source, "trigger", "trigger"),
    leftBecause: requireDefectField(source, "left_because", "leftBecause", "left_because"),
    fixHint: blankToNull(optionalString(firstOf(source, "fixHint", "fix_hint"), "fields.fix_hint")),
    impact: blankToNull(optionalString(source.impact, "fields.impact")),
  };
}

const SUPERSEDES_MAX = 5;

function parseSupersedes(value: unknown): string[] | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }

  if (!Array.isArray(value)) {
    throw new HttpError(400, "supersedes must be a list of record ids");
  }

  const ids = [...new Set(value.map((entry, index) => requireText(entry, `supersedes[${index}]`)))];

  if (ids.length === 0 || ids.length > SUPERSEDES_MAX) {
    throw new HttpError(400, `supersedes must hold 1 to ${SUPERSEDES_MAX} record ids`);
  }

  return ids;
}

function guardTarget(target: ProposalTarget | null, internal: boolean | undefined): void {
  if (!internal && target !== null && target.action !== "close") {
    throw new HttpError(400, "target.action must be close on this route");
  }
}

export function parseProposalBody(raw: unknown, options: { internal?: boolean } = {}): ProposalBody {
  const body = asObject(raw, "body");
  const fields = asObject(body.fields, "fields");
  const allowed = options.internal ? ORIGINS : EXTERNAL_ORIGINS;
  const origin = body.origin === undefined ? "agent" : oneOf<Origin>(body.origin, allowed, "origin");
  const kind = oneOf<Kind>(body.kind, KINDS, "kind");
  const target = parseTarget(body.target);
  const supersedes = origin === "agent" ? parseSupersedes(body.supersedes) : undefined;

  guardTarget(target, options.internal);

  if (supersedes !== undefined && target?.action === "close") {
    throw new HttpError(400, "supersedes cannot be combined with a close target");
  }

  return {
    kind,
    headline: parseHeadline(body.headline),
    fields: parseFields(fields, kind, target),
    scope: requireText(body.scope, "scope"),
    provenance: parseProvenance(body.provenance),
    origin,
    target,
    ...(supersedes === undefined ? {} : { supersedes }),
    audience: parseAudience(body.audience, "audience"),
  };
}

function parseDefectEdits(source: Record<string, unknown>, target: Partial<Fields>): void {
  const required: Array<[keyof Fields, string[]]> = [
    ["where", ["where"]],
    ["trigger", ["trigger"]],
    ["leftBecause", ["leftBecause", "left_because"]],
  ];

  for (const [key, keys] of required) {
    const value = firstOf(source, ...keys);

    if (value !== undefined) {
      target[key] = requireText(value, `edits.fields.${keys[0]}`);
    }
  }

  for (const [key, keys] of [["fixHint", ["fixHint", "fix_hint"]], ["impact", ["impact"]]] as const) {
    const value = firstOf(source, ...keys);

    if (value !== undefined) {
      target[key] = blankToNull(optionalString(value, `edits.fields.${keys[0]}`));
    }
  }
}

export function parseApprove(raw: unknown): ApproveInput {
  const body = raw === undefined || raw === null ? {} : asObject(raw, "body");
  const edits = body.edits === undefined ? {} : asObject(body.edits, "edits");
  const input: ApproveInput = {};

  if (edits.headline !== undefined) {
    input.headline = parseHeadline(edits.headline);
  }

  if (edits.fields !== undefined) {
    const fields = asObject(edits.fields, "edits.fields");
    input.fields = {};

    if (fields.claim !== undefined) {
      input.fields.claim = requireText(fields.claim, "edits.fields.claim");
    }

    if (fields.rejected !== undefined) {
      input.fields.rejected = optionalString(fields.rejected, "edits.fields.rejected");
    }

    parseDefectEdits(fields, input.fields);
  }

  if (edits.audience !== undefined) {
    input.audience = parseAudience(edits.audience, "edits.audience");
  }

  if (edits.scope !== undefined) {
    input.scope = requireText(edits.scope, "edits.scope");
  }

  if (edits.purpose !== undefined) {
    input.purpose = optionalString(edits.purpose, "edits.purpose");
  }

  if (edits.topic !== undefined) {
    input.topic = optionalString(edits.topic, "edits.topic");
  }

  if (body.supersedes !== undefined) {
    if (!Array.isArray(body.supersedes)) {
      throw new HttpError(400, "supersedes must be an array of record ids");
    }

    input.supersedes = body.supersedes.map((entry) => requireText(entry, "supersedes entry"));
  }

  if (body.reason !== undefined && body.reason !== null) {
    input.reason = requireText(body.reason, "reason");
  }

  return input;
}

function provenanceOf(row: StoredRecord): Provenance {
  const provenance = {} as Provenance;

  for (const key of PROVENANCE_KEYS) {
    provenance[key] = row[key] ?? "";
  }

  return provenance;
}

export function parseAnswer(raw: unknown): AnswerInput {
  const body = asObject(raw, "body");
  const action = oneOf(body.action, ["fold", "keep_both"] as const, "action");

  if (action === "keep_both") {
    return { action, reason: requireText(body.reason, "reason") };
  }

  const kind = oneOf<Kind>(body.kind, CONTAINED_KINDS, "kind");

  return {
    action,
    kind,
    headline: parseHeadline(body.headline),
    fields: parseFields(asObject(body.fields, "fields"), kind, null),
  };
}

function sameFields(a: Fields, b: Fields): boolean {
  return a.claim === b.claim && (a.rejected ?? null) === (b.rejected ?? null);
}

export class ProposalService {
  private readonly store: Store;
  private readonly embedder: Embedder;
  private readonly records: RecordService;
  private readonly now: () => string;
  private readonly checker: ProposalChecker;
  private readonly containment: boolean;
  private readonly containmentFloor: number;
  private readonly foldAsk: boolean;
  private readonly foldSimilarityFloor: number;
  private readonly foldSubjectFloor: number;
  private readonly foldAskTimeoutMs: number;
  private readonly jev: JevClient | null;
  private readonly model: string;
  private readonly timeoutMs: number | undefined;
  private readonly sharedScopeOf: (scope: string) => string;
  private readonly profileOf: (scope: string) => string;
  private readonly mode: Mode;

  constructor(
    store: Store,
    embedder: Embedder,
    records: RecordService,
    now: () => string = () => new Date().toISOString(),
    options: ProposalOptions = {},
  ) {
    this.store = store;
    this.embedder = embedder;
    this.records = records;
    this.now = now;
    this.containment = options.containment ?? true;
    this.containmentFloor = options.containmentFloor ?? DEFAULT_CONTAINMENT_FLOOR;
    this.foldAsk = options.foldAsk ?? false;
    this.foldSimilarityFloor = options.foldSimilarityFloor ?? DEFAULT_FOLD_SIMILARITY_FLOOR;
    this.foldSubjectFloor = options.foldSubjectFloor ?? DEFAULT_FOLD_SUBJECT_FLOOR;
    this.foldAskTimeoutMs = options.foldAskTimeoutMs ?? DEFAULT_FOLD_ASK_TIMEOUT_MS;
    this.jev = options.jev ?? null;
    this.model = options.model ?? "jev-1.13.0";
    this.timeoutMs = options.timeoutMs;
    this.sharedScopeOf = options.sharedScopeOf ?? (() => "global");
    this.profileOf = options.profileOf ?? (() => "default");
    this.mode = options.mode ?? "t3";
    this.checker = new ProposalChecker({
      sharedScopeOf: this.sharedScopeOf,
      client: options.jev ?? null,
      model: this.model,
      topics: loadTopics(options.topicsPath),
      timeoutMs: options.timeoutMs,
      scopeLabels: (scope) => this.scopeLabels(scope),
    });
  }

  async create(raw: unknown): Promise<Proposal> {
    return this.createFrom(parseProposalBody(raw));
  }

  async createInternal(raw: unknown): Promise<Proposal> {
    return this.createFrom(parseProposalBody(raw, { internal: true }));
  }

  private async createFrom(parsed: ProposalBody): Promise<Proposal> {
    await this.assertClosable(parsed.target);

    const body = await this.withTargetHeadlines(parsed);
    const outcome = await this.assess(body);
    const proposal: Proposal = {
      ...body,
      id: randomUUID(),
      status: "pending",
      audience: body.audience ?? outcome.audience,
      reader: outcome.reader,
      purpose: outcome.purpose,
      topic: outcome.topic,
      checks: outcome.checks,
      flags: outcome.flags,
      neighbours: outcome.neighbours,
      suggestion: outcome.suggestion,
      createdAt: this.now(),
      resolvedAt: null,
      resolution: null,
      ask: null,
    };

    if (!this.appliesOnCreate(proposal)) {
      await upsert(this.store.proposals, [proposalToRow(proposal)]);

      return proposal;
    }

    return this.store.withLock(() => this.applyOnCreate(proposal));
  }

  private appliesOnCreate(proposal: Proposal): boolean {
    if (proposal.origin === "agent") {
      return true;
    }

    return this.mode === "standalone" && (proposal.origin === "cleaner" || proposal.origin === "promotion");
  }

  private async applyOnCreate(proposal: Proposal): Promise<Proposal> {
    if (proposal.supersedes !== undefined) {
      await this.assertSupersedable(proposal, proposal.supersedes);

      return this.applySuperseding(proposal, { supersedes: proposal.supersedes, reason: "changed" });
    }

    const suggestion = proposal.suggestion;

    if (suggestion?.action === "reinforce") {
      await this.reinforceOnto(proposal, suggestion.recordId);

      return this.get(proposal.id);
    }

    const covering = await this.coveringTarget(proposal);
    const flagged: Proposal =
      covering === null
        ? proposal
        : { ...proposal, flags: [...proposal.flags, `covers:${covering.neighbour.id}`] };
    const target = this.foldAsk ? (covering ?? (await this.askTarget(flagged))) : null;

    if (target !== null) {
      return this.hold(flagged, target);
    }

    return this.applySuperseding(flagged, {});
  }

  private async applySuperseding(proposal: Proposal, input: ApproveInput): Promise<Proposal> {
    const record = await this.apply(proposal, input);
    const approved: Proposal = {
      ...proposal,
      status: "approved",
      resolvedAt: this.now(),
      resolution: { recordId: record.id, reason: null },
    };

    await upsert(this.store.proposals, [proposalToRow(approved)]);

    return approved;
  }

  private async assertSupersedable(proposal: Proposal, ids: string[]): Promise<void> {
    for (const id of ids) {
      if (id === proposal.id) {
        throw new HttpError(400, `supersedes ${id} is the new note itself`);
      }

      const found = await this.records.locate(id);

      if (found === undefined) {
        throw new HttpError(404, `supersedes ${id} not found`);
      }

      if (found.row.lifecycle !== "live") {
        throw new HttpError(409, `supersedes ${id} is ${found.row.lifecycle}, not live`);
      }

      if (found.row.kind === "defect") {
        throw new HttpError(400, `supersedes ${id} is a defect; close it with closes`);
      }

      if (found.row.scope !== proposal.scope) {
        throw new HttpError(400, `supersedes ${id} is in scope ${found.row.scope}, not ${proposal.scope}`);
      }
    }
  }

  private async coveringTarget(proposal: Proposal): Promise<AskTarget | null> {
    if (!this.containment || proposal.target || !CONTAINED_KINDS.includes(proposal.kind)) {
      return null;
    }

    const covering = proposal.neighbours
      .filter((neighbour) => (neighbour.pair?.newCoversOld ?? 0) >= this.containmentFloor)
      .sort((a, b) => (b.pair?.newCoversOld ?? 0) - (a.pair?.newCoversOld ?? 0));

    return this.firstFoldable(proposal, covering);
  }

  private async askTarget(proposal: Proposal): Promise<AskTarget | null> {
    if (proposal.suggestion !== null || proposal.target || !CONTAINED_KINDS.includes(proposal.kind)) {
      return null;
    }

    const eligible = proposal.neighbours
      .filter((neighbour) => this.worthAsking(neighbour))
      .sort((a, b) => b.similarity - a.similarity);

    return this.firstFoldable(proposal, eligible);
  }

  private async firstFoldable(proposal: Proposal, neighbours: Neighbour[]): Promise<AskTarget | null> {
    for (const neighbour of neighbours) {
      const found = await this.records.locate(neighbour.id);

      if (
        found !== undefined &&
        found.row.lifecycle === "live" &&
        found.row.scope === proposal.scope &&
        CONTAINED_KINDS.includes(found.row.kind)
      ) {
        return { neighbour, row: found.row };
      }
    }

    return null;
  }

  private worthAsking(neighbour: Neighbour): boolean {
    const pick = neighbour.relation?.pick;

    return (
      neighbour.pair !== null &&
      neighbour.pair !== undefined &&
      pick !== undefined &&
      pick !== "distinct" &&
      neighbour.similarity >= this.foldSimilarityFloor &&
      neighbour.pair.sameSubject >= this.foldSubjectFloor
    );
  }

  private async hold(proposal: Proposal, target: AskTarget): Promise<Proposal> {
    const { neighbour, row } = target;
    const askedAt = this.now();
    const ask: Ask = {
      existing: {
        id: row.id,
        source: neighbour.source,
        kind: row.kind,
        headline: row.headline,
        fields: {
          claim: row.claim,
          ...(row.why ? { why: row.why } : {}),
          rejected: row.rejected,
        },
      },
      similarity: neighbour.similarity,
      sameSubject: neighbour.pair?.sameSubject ?? 0,
      newCoversOld: neighbour.pair?.newCoversOld ?? 0,
      existingIsFold: await this.isFold(row.id),
      askedAt,
      expiresAt: new Date(Date.parse(askedAt) + this.foldAskTimeoutMs).toISOString(),
      answer: null,
      answeredAt: null,
      reason: null,
      foldRecordId: null,
      fold: null,
      foldCoversOld: null,
      foldCoversNew: null,
    };
    const held: Proposal = { ...proposal, status: "asked", ask };

    await upsert(this.store.proposals, [proposalToRow(held)]);

    return held;
  }

  private async isFold(id: string): Promise<boolean> {
    const where = `supersededBy = ${sqlString(id)}`;
    const [current, archived] = await Promise.all([
      selectWhere<MemoryRecord>(this.store.records, where, 1),
      selectWhere<LegacyRecord>(this.store.legacy, where, 1),
    ]);

    return current.length + archived.length > 0;
  }

  private async assertClosable(target: ProposalTarget | null | undefined): Promise<void> {
    if (target?.action === "close") {
      await this.records.requireClosableDefect(target.recordId);
    }
  }

  private async withTargetHeadlines(body: ProposalBody): Promise<ProposalBody> {
    if (!body.target) {
      return body;
    }

    const [recordHeadline, replacementHeadline] = await Promise.all([
      this.headlineOf(body.target.recordId),
      this.headlineOf(body.target.replacementId),
    ]);

    return { ...body, target: { ...body.target, recordHeadline, replacementHeadline } };
  }

  private async headlineOf(id: string | null | undefined): Promise<string | null> {
    if (!id) {
      return null;
    }

    const found = await this.records.locate(id);

    return found?.row.headline ?? null;
  }

  private async assess(body: ProposalBody): Promise<CheckOutcome> {
    if (body.origin === "cleaner" || body.origin === "promotion" || body.target?.action === "close") {
      return this.assessTargeted(body);
    }

    const vector = await this.embedder.embed(recordText(this.textOf(body)));
    const candidates = await this.neighbours(body, vector);

    return this.checker.check(body, candidates);
  }

  private async assessTargeted(body: ProposalBody): Promise<CheckOutcome> {
    const found = body.target ? await this.records.locate(body.target.recordId) : undefined;

    return {
      checks: { status: "skipped", reason: "targeted proposal" },
      flags: [],
      neighbours: [],
      suggestion: null,
      purpose: found?.row.purpose ?? null,
      topic: found?.row.topic ?? null,
      audience: null,
      reader: null,
    };
  }

  private textOf(body: ProposalBody) {
    return { kind: body.kind, headline: body.headline, ...body.fields };
  }

  private async scopeLabels(declared: string): Promise<Array<[string, string | null]>> {
    const [current, archived] = await Promise.all([
      this.store.records.query().select(["scope"]).toArray(),
      this.store.legacy.query().select(["scope"]).toArray(),
    ]);
    const names = new Set<string>();
    const profile = this.profileOf(declared);

    for (const row of [...current, ...archived]) {
      if (typeof row.scope === "string" && row.scope !== "" && this.profileOf(row.scope) === profile) {
        names.add(row.scope);
      }
    }

    return [...names].sort().map((name) => [name, null]);
  }

  private async neighbours(body: ProposalBody, vector: number[]): Promise<NeighbourCandidate[]> {
    const query = recordText(this.textOf(body));
    const where = liveFilter(body.scope, null, null, this.sharedScopeOf(body.scope));
    const [current, archived] = await Promise.all([
      hybridSearch<MemoryRecord>(this.store.records, { query, vector, where, limit: NEIGHBOUR_COUNT }),
      hybridSearch<LegacyRecord>(this.store.legacy, { query, vector, where, limit: NEIGHBOUR_COUNT }),
    ]);
    const rows = new Map<string, StoredRecord>();

    for (const hit of [...current, ...archived]) {
      rows.set(hit.row.id, hit.row);
    }

    return mergeBySource(current, archived)
      .slice(0, NEIGHBOUR_COUNT)
      .map((result) => {
        const row = rows.get(result.id) as StoredRecord;

        return {
          id: result.id,
          source: result.source,
          headline: result.headline,
          similarity: cosine(vector, row.vector),
          relation: null,
          text: recordText(row),
          pairText: recordSummary(this.bodyOfRow(row)),
          kind: row.kind,
          scope: row.scope,
        };
      });
  }

  async list(status: string | null, since: string | null): Promise<Proposal[]> {
    const clauses: string[] = [];

    if (status !== null && status !== "all") {
      clauses.push(`status = ${sqlString(oneOf<ProposalStatus>(status, PROPOSAL_STATUSES, "status"))}`);
    }

    if (since !== null) {
      const parsed = new Date(since);

      if (Number.isNaN(parsed.getTime())) {
        throw new HttpError(400, "since must be an ISO timestamp");
      }

      clauses.push(`createdAt > ${sqlString(parsed.toISOString())}`);
    }

    const rows = await selectWhere<Record<string, unknown>>(
      this.store.proposals,
      clauses.length === 0 ? undefined : clauses.join(" AND "),
    );

    return rows.map(rowToProposal).sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }

  async get(id: string): Promise<Proposal> {
    const row = await findById<Record<string, unknown>>(this.store.proposals, id);

    if (row === undefined) {
      throw new HttpError(404, `proposal ${id} not found`);
    }

    return rowToProposal(row);
  }

  async answer(id: string, raw: unknown): Promise<Proposal> {
    const input = parseAnswer(raw);

    if (input.action === "keep_both") {
      return this.store.withLock(() => this.answerKeepBoth(id, input));
    }

    return this.answerFold(id, input);
  }

  async listAsks(since: string | null): Promise<AskReport> {
    const clauses = ["ask IS NOT NULL"];

    if (since !== null) {
      const parsed = new Date(since);

      if (Number.isNaN(parsed.getTime())) {
        throw new HttpError(400, "since must be an ISO timestamp");
      }

      clauses.push(`createdAt > ${sqlString(parsed.toISOString())}`);
    }

    const rows = await selectWhere<Record<string, unknown>>(this.store.proposals, clauses.join(" AND "));
    const asks = rows.map(rowToProposal).sort((a, b) => a.createdAt.localeCompare(b.createdAt));
    const counts: AskCounts = { fold: 0, keep_both: 0, unanswered: 0, open: 0 };

    for (const proposal of asks) {
      counts[proposal.ask?.answer ?? "open"] += 1;
    }

    return { counts, asks };
  }

  async expireAsks(now: string = this.now()): Promise<string[]> {
    const rows = await selectWhere<Record<string, unknown>>(this.store.proposals, "status = 'asked'");
    const cutoff = Date.parse(now);
    const due = rows
      .map(rowToProposal)
      .filter(
        (proposal) =>
          proposal.ask !== null &&
          (proposal.ask.answer !== null || Date.parse(proposal.ask.expiresAt) <= cutoff),
      );
    const expired: string[] = [];

    for (const proposal of due) {
      try {
        if (await this.store.withLock(() => this.expireOne(proposal.id, now))) {
          expired.push(proposal.id);
        }
      } catch (error) {
        console.error(error);
      }
    }

    return expired;
  }

  private async expireOne(id: string, now: string): Promise<boolean> {
    const held = await this.heldProposal(id);

    if (held.status !== "asked") {
      return false;
    }

    if (held.ask.foldRecordId !== null) {
      await this.finishFold(held);

      return true;
    }

    if (held.ask.answer !== null) {
      await this.finishHeld(held);

      return true;
    }

    if (Date.parse(held.ask.expiresAt) > Date.parse(now)) {
      return false;
    }

    await this.saveHeld(held, { answer: "unanswered", answeredAt: now, reason: null });

    return true;
  }

  private async heldProposal(id: string): Promise<HeldProposal> {
    const proposal = await this.get(id);

    if (proposal.ask === null) {
      throw new HttpError(409, `proposal ${id} is not an ask`);
    }

    return proposal as HeldProposal;
  }

  private async saveHeld(held: HeldProposal, outcome: Partial<Ask>): Promise<Proposal> {
    const intent: HeldProposal = { ...held, ask: { ...held.ask, ...outcome, heldRecordId: held.id } };

    await upsert(this.store.proposals, [proposalToRow(intent)]);

    return this.finishHeld(intent);
  }

  private async finishHeld(intent: HeldProposal): Promise<Proposal> {
    const record = await this.apply(intent, {});
    const approved: Proposal = {
      ...intent,
      status: "approved",
      resolvedAt: this.now(),
      resolution: { recordId: record.id, reason: null },
    };

    await upsert(this.store.proposals, [proposalToRow(approved)]);

    return approved;
  }

  private async answerKeepBoth(id: string, input: AnswerInput & { action: "keep_both" }): Promise<Proposal> {
    const held = await this.heldProposal(id);

    if (held.ask.answer !== null) {
      const replay = this.replayed(held, held.ask.answer === "keep_both" && held.ask.reason === input.reason);

      return held.status === "asked" ? this.finishHeld(held) : replay;
    }

    return this.saveHeld(held, { answer: "keep_both", answeredAt: this.now(), reason: input.reason });
  }

  private replayed(held: HeldProposal, matches: boolean): Proposal {
    if (!matches) {
      throw new HttpError(409, `proposal ${held.id} was already answered: ${held.ask.answer}`);
    }

    return held;
  }

  private replayedFold(held: HeldProposal, input: AnswerInput & { action: "fold" }): Proposal {
    const draft = held.ask.fold;
    const matches =
      held.ask.answer === "fold" &&
      draft !== null &&
      draft.kind === input.kind &&
      draft.headline === input.headline &&
      sameFields(draft.fields, input.fields);

    return this.replayed(held, matches);
  }

  private async answerFold(id: string, input: AnswerInput & { action: "fold" }): Promise<Proposal> {
    const first = await this.heldProposal(id);

    if (first.ask.answer !== null) {
      return this.settleReplay(first, input);
    }

    const body: ProposalBody = {
      kind: input.kind,
      headline: input.headline,
      fields: input.fields,
      scope: first.scope,
      provenance: first.provenance,
      origin: "agent",
      audience: first.audience,
    };
    const outcome = await this.checker.check(body, []);
    const folded = await this.store.withLock(() => this.writeFold(id, input, outcome));

    return folded.fresh ? this.recordCoverage(folded.proposal) : folded.proposal;
  }

  private async settleReplay(
    held: HeldProposal,
    input: AnswerInput & { action: "fold" },
  ): Promise<Proposal> {
    return this.store.withLock(async () => {
      const current = await this.heldProposal(held.id);
      const replay = this.replayedFold(current, input);

      return current.status === "asked" ? this.finishFold(current) : replay;
    });
  }

  private async writeFold(
    id: string,
    input: AnswerInput & { action: "fold" },
    outcome: CheckOutcome,
  ): Promise<{ proposal: HeldProposal; fresh: boolean }> {
    const held = await this.heldProposal(id);

    if (held.ask.answer !== null) {
      const replay = this.replayedFold(held, input) as HeldProposal;

      return { proposal: held.status === "asked" ? await this.finishFold(replay) : replay, fresh: false };
    }

    await this.requireLive(held.ask.existing.id);

    const draft: FoldDraft = {
      kind: input.kind,
      headline: input.headline,
      fields: input.fields,
      purpose: outcome.purpose ?? held.purpose,
      topic: outcome.topic ?? held.topic,
      audience: held.audience ?? outcome.audience,
      reader: outcome.reader ?? held.reader,
      checks: outcome.checks,
      flags: outcome.flags,
    };
    const intent: HeldProposal = {
      ...held,
      ask: { ...held.ask, answer: "fold", answeredAt: this.now(), foldRecordId: randomUUID(), fold: draft },
    };

    await upsert(this.store.proposals, [proposalToRow(intent)]);

    return { proposal: await this.finishFold(intent), fresh: true };
  }

  private async finishFold(held: HeldProposal): Promise<HeldProposal> {
    const draft = held.ask.fold;
    const foldId = held.ask.foldRecordId;

    if (draft === null || foldId === null) {
      throw new HttpError(409, `proposal ${held.id} has no fold to finish`);
    }

    const record: Proposal = {
      ...held,
      ...draft,
      id: foldId,
      status: "approved",
      neighbours: [],
      suggestion: null,
      target: null,
      ask: null,
    };
    const foldGone = await this.foldGone(foldId);
    const existingGone = !foldGone && (await this.existingGone(held.ask.existing.id, foldId));

    if (!foldGone) {
      await this.applyWrite(
        record,
        existingGone ? {} : { supersedes: [held.ask.existing.id], reason: "elaborated" },
      );
    }

    const done: HeldProposal = {
      ...held,
      status: "folded",
      resolvedAt: this.now(),
      resolution: { recordId: foldId, reason: null },
      ask: {
        ...held.ask,
        ...(existingGone ? { existingGone: true } : {}),
        ...(foldGone ? { foldGone: true } : {}),
      },
    };

    await upsert(this.store.proposals, [proposalToRow(done)]);

    return done;
  }

  private async foldGone(foldId: string): Promise<boolean> {
    const found = await this.records.locate(foldId);

    return found !== undefined && found.row.lifecycle !== "live";
  }

  private async existingGone(existingId: string, foldId: string): Promise<boolean> {
    const found = await this.records.locate(existingId);

    return found === undefined || (found.row.lifecycle !== "live" && found.row.supersededBy !== foldId);
  }

  private async recordCoverage(folded: HeldProposal): Promise<Proposal> {
    const foldId = folded.ask.foldRecordId as string;
    const stored = await this.records.locate(foldId);
    const existing = await this.records.locate(folded.ask.existing.id);

    if (stored === undefined || existing === undefined) {
      return folded;
    }

    const foldText = recordSummary(this.bodyOfRow(stored.row));
    const [coversOld, coversNew] = await Promise.all([
      this.coverage(recordSummary(this.bodyOfRow(existing.row)), foldText),
      this.coverage(recordSummary(folded), foldText),
    ]);

    return this.store.withLock(async () => {
      const current = await this.heldProposal(folded.id);
      const updated: HeldProposal = {
        ...current,
        ask: { ...current.ask, foldCoversOld: coversOld, foldCoversNew: coversNew },
      };

      await upsert(this.store.proposals, [proposalToRow(updated)]);

      return updated;
    });
  }

  private bodyOfRow(row: StoredRecord): ProposalBody {
    return {
      kind: row.kind,
      headline: row.headline,
      fields: { claim: row.claim, rejected: row.rejected },
      scope: row.scope,
      provenance: provenanceOf(row),
      origin: row.origin,
    };
  }

  private async coverage(existing: string, incoming: string): Promise<number | null> {
    if (this.jev === null) {
      return null;
    }

    try {
      const response = await askJev(
        this.jev,
        this.model,
        { existing: existing.slice(0, NEIGHBOUR_TEXT_CAP), incoming },
        [NEW_COVERS_OLD_QUESTION],
        this.timeoutMs,
      );

      return yesProbability(response, NEW_COVERS_OLD_QUESTION.id);
    } catch {
      return null;
    }
  }

  private async requirePending(id: string): Promise<Proposal> {
    const proposal = await this.get(id);

    if (proposal.status !== "pending") {
      throw new HttpError(409, `proposal ${id} is ${proposal.status}, not pending`);
    }

    return proposal;
  }

  private async resolve(
    proposal: Proposal,
    status: ProposalStatus,
    resolution: Proposal["resolution"],
  ): Promise<Proposal> {
    const resolved: Proposal = { ...proposal, status, resolvedAt: this.now(), resolution };

    await upsert(this.store.proposals, [proposalToRow(resolved)]);

    return resolved;
  }

  async reject(id: string, raw: unknown): Promise<Proposal> {
    const body = raw === undefined || raw === null ? {} : asObject(raw, "body");
    const reason = requireText(body.reason, "reason");

    return this.store.withLock(async () => {
      const proposal = await this.requirePending(id);

      return this.resolve(proposal, "rejected", { reason });
    });
  }

  async reinforce(id: string, raw: unknown): Promise<PublicRecord> {
    const body = asObject(raw, "body");
    const recordId = requireText(body.recordId, "recordId");

    return this.store.withLock(async () => {
      const proposal = await this.requirePending(id);

      return stripVector(await this.reinforceOnto(proposal, recordId));
    });
  }

  private async reinforceOnto(proposal: Proposal, recordId: string): Promise<StoredRecord> {
    const found = await this.records.require(recordId);

    if (found.source === "legacy") {
      return this.reinforceLegacy(proposal, found.row);
    }

    this.assertLive(found.row);
    await this.resolve(proposal, "reinforced", { recordId });

    return this.records.bumpReinforcement(recordId);
  }

  private async reinforceLegacy(proposal: Proposal, legacy: StoredRecord): Promise<StoredRecord> {
    if (legacy.lifecycle !== "live" && legacy.supersededBy !== proposal.id) {
      this.assertLive(legacy);
    }

    const copy = await this.records.insertRecord({
      id: proposal.id,
      kind: legacy.kind,
      headline: legacy.headline,
      claim: legacy.claim,
      why: legacy.why,
      rejected: legacy.rejected,
      scope: legacy.scope,
      purpose: legacy.purpose,
      topic: legacy.topic,
      audience: legacy.audience ?? null,
      reader: legacy.reader ?? null,
      provenance: provenanceOf(legacy),
      origin: "promotion",
      legacyId: legacy.id,
      vector: legacy.vector,
      reinforcedCount: 1,
      lastReinforcedAt: this.now(),
    });

    await this.records.retireLegacy(legacy.id, copy.id, `promoted to ${copy.id} by reinforcement`);
    await this.resolve(proposal, "reinforced", { recordId: copy.id });

    return copy;
  }

  private assertLive(row: StoredRecord): void {
    if (row.lifecycle !== "live") {
      throw new HttpError(409, `record ${row.id} is ${row.lifecycle}, not live`);
    }
  }

  async approve(id: string, raw: unknown): Promise<PublicRecord> {
    const input = parseApprove(raw);

    return this.store.withLock(async () => {
      const proposal = await this.requirePending(id);
      const outcome = await this.apply(proposal, input);

      await this.resolve(proposal, "approved", { recordId: outcome.id, reason: input.reason ?? null });

      return stripVector(outcome);
    });
  }

  private apply(proposal: Proposal, input: ApproveInput): Promise<StoredRecord> {
    const target = proposal.target;

    if (target?.action === "close") {
      return this.applyClose(proposal, target);
    }

    if (target?.action === "retire") {
      return this.applyRetire(target, input);
    }

    if (target?.action === "merge") {
      return this.applyMerge(target, input);
    }

    if (target?.action === "supersede" && target.replacementId) {
      return this.applySupersedeExisting(target, input);
    }

    return this.applyWrite(proposal, input);
  }

  private async applyClose(proposal: Proposal, target: ProposalTarget): Promise<StoredRecord> {
    const found = await this.records.require(target.recordId);

    if (found.row.defectStatus === "closed" && found.row.closedBy === proposal.id) {
      return found.row;
    }

    await this.records.requireClosableDefect(target.recordId);

    return this.records.markClosed(target.recordId, proposal.id, proposal.provenance.threadId ?? null);
  }

  private async applyRetire(target: ProposalTarget, input: ApproveInput): Promise<StoredRecord> {
    const reason = requireText(input.reason ?? target.reason, "reason");
    const found = await this.records.require(target.recordId);

    if (found.row.lifecycle === "retired" && found.row.retiredReason === reason) {
      return found.row;
    }

    return this.records.markRetired(target.recordId, reason);
  }

  private async applyMerge(target: ProposalTarget, input: ApproveInput): Promise<StoredRecord> {
    const keeperId = requireText(target.replacementId, "target.replacementId");

    if (keeperId === target.recordId) {
      throw new HttpError(400, "a record cannot be merged into itself");
    }

    const keeper = await this.requireLive(keeperId);
    const reason = input.reason ?? target.reason ?? `merged into ${keeperId}`;
    const found = await this.records.require(target.recordId);

    if (found.row.lifecycle === "retired" && found.row.retiredReason === reason) {
      return keeper;
    }

    await this.records.markRetired(target.recordId, reason);

    return keeper;
  }

  private async applySupersedeExisting(target: ProposalTarget, input: ApproveInput): Promise<StoredRecord> {
    const keeperId = requireText(target.replacementId, "target.replacementId");
    const reason = requireReasonClass(input.reason ?? target.reason);
    const keeper = await this.requireLive(keeperId);

    if (await this.isSettled(target.recordId, keeperId)) {
      return keeper;
    }

    await this.records.markSuperseded(target.recordId, keeperId, reason);

    return keeper;
  }

  private async requireLive(id: string): Promise<StoredRecord> {
    const found = await this.records.require(id);

    if (found.row.lifecycle !== "live") {
      throw new HttpError(409, `record ${id} is ${found.row.lifecycle}, not live`);
    }

    return found.row;
  }

  private async applyWrite(proposal: Proposal, input: ApproveInput): Promise<StoredRecord> {
    const target = proposal.target;
    const superseded = [...new Set(input.supersedes ?? (target?.action === "supersede" ? [target.recordId] : []))];
    const reason = superseded.length === 0 ? null : this.supersessionReason(proposal, input);
    const record = { ...this.newRecord(proposal, input), id: proposal.id };
    const unsettled: string[] = [];

    for (const oldId of superseded) {
      if (!(await this.isSettled(oldId, record.id))) {
        await this.requireLive(oldId);
        unsettled.push(oldId);
      }
    }

    const promoted = await this.promotedLegacy(target, record.id);
    const created = await this.records.insertRecord(
      promoted === null ? record : { ...record, why: promoted.why },
    );

    if (reason !== null) {
      for (const oldId of unsettled) {
        await this.records.markSuperseded(oldId, created.id, reason);
      }
    }

    if (promoted !== null) {
      await this.records.retireLegacy(promoted.id, created.id, `promoted to ${created.id}`);
    }

    return created;
  }

  private async promotedLegacy(
    target: ProposalTarget | null | undefined,
    byId: string,
  ): Promise<{ id: string; why: string | null } | null> {
    if (target?.action !== "promote") {
      return null;
    }

    if (!(await this.isSettled(target.recordId, byId))) {
      await this.requireLive(target.recordId);
    }

    const found = await this.records.require(target.recordId);

    return found.source === "legacy" ? { id: target.recordId, why: found.row.why } : null;
  }

  private async isSettled(id: string, byId: string): Promise<boolean> {
    const found = await this.records.require(id);

    return found.row.lifecycle !== "live" && found.row.supersededBy === byId;
  }

  private supersessionReason(proposal: Proposal, input: ApproveInput): ReasonClass {
    const fallback = proposal.suggestion?.reasonClass ?? proposal.target?.reason ?? undefined;

    return requireReasonClass(input.reason ?? fallback);
  }

  private newRecord(proposal: Proposal, input: ApproveInput): NewRecord {
    const fields = { ...proposal.fields, ...input.fields };
    const defect = proposal.kind === "defect";

    return {
      kind: proposal.kind,
      headline: input.headline ?? proposal.headline,
      claim: fields.claim,
      rejected: fields.rejected ?? null,
      scope: input.scope ?? proposal.scope,
      purpose: input.purpose === undefined ? proposal.purpose : input.purpose,
      topic: input.topic === undefined ? proposal.topic : input.topic,
      audience: input.audience === undefined ? proposal.audience : input.audience,
      reader: proposal.reader,
      where: defect ? (fields.where ?? null) : null,
      trigger: defect ? (fields.trigger ?? null) : null,
      leftBecause: defect ? (fields.leftBecause ?? null) : null,
      fixHint: defect ? (fields.fixHint ?? null) : null,
      impact: defect ? (fields.impact ?? null) : null,
      defectStatus: defect ? "open" : null,
      provenance: proposal.provenance,
      origin: proposal.origin,
      legacyId: proposal.target?.action === "promote" ? proposal.target.recordId : null,
    };
  }
}
