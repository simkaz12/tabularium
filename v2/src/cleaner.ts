import { createHash } from "node:crypto";
import { hostname } from "node:os";

import {
  CLEANER_DOUBT_CAP,
  CLEANER_DOUBT_READ_LIMIT,
  DEFAULT_CLEANER_CONFIDENCE_FLOOR,
  GATE_QUESTIONS,
} from "./config.ts";
import type { Mode } from "./config.ts";
import type { JevClient, LegacyRecord, MemoryRecord, Proposal, ProposalBody, ProposalTarget } from "./domain.ts";
import { recordText } from "./domain.ts";
import { MalformedAnswer, askContradiction, askRelevance } from "./jev/checks.ts";
import type { ContradictionResult } from "./jev/checks.ts";
import type { ProposalService } from "./proposals.ts";
import { SettingsService, StoreCleanerState } from "./settings.ts";
import type { CleanerState, RecordStamp } from "./settings.ts";
import { andAll, findById, rowToProposal, scopeClause, selectWhere, sqlString } from "./store.ts";
import type { Store } from "./store.ts";

const DAY_MS = 86_400_000;
const CATCH_UP_DELAY_MS = 30_000;
const MAX_CONSECUTIVE_JEV_FAILURES = 3;
const DOUBT_FIRST_RUN_DAYS = 7;

export interface CleanerOptions {
  store: Store;
  proposals: ProposalService;
  jev: JevClient | null;
  model: string;
  settings?: SettingsService;
  state?: CleanerState;
  timeoutMs?: number;
  now?: () => Date;
  cap?: number;
  doubtCap?: number;
  doubtReadLimit?: number;
  minAgeDays?: number;
  mergeFloor?: number;
  contradictionFloor?: number;
  maxJevCalls?: number;
  promotionReads?: number;
  promotionWindowDays?: number;
  rejectedCooldownDays?: number;
  sharedScopeOf?: (scope: string) => string;
  mode?: Mode;
  log?: (message: string) => void;
}

interface RunSettings {
  store: Store;
  proposals: ProposalService;
  state: CleanerState;
  jev: JevClient | null;
  model: string;
  timeoutMs: number | undefined;
  now: Date;
  cap: number;
  doubtCap: number;
  doubtReadLimit: number;
  doubtsSince: string;
  minAgeDays: number;
  mergeFloor: number;
  contradictionFloor: number;
  confidenceFloor: number;
  maxJevCalls: number;
  promotionReads: number;
  promotionWindowDays: number;
  sharedScopeOf: (scope: string) => string;
  mode: Mode;
  log: (message: string) => void;
}

export interface CleanerReport {
  proposed: Proposal[];
  candidates: number;
  jevCalls: number;
}

interface Pair {
  older: MemoryRecord;
  newer: MemoryRecord;
  similarity: number;
}

type SupersedeOutcome = "queued" | "settled" | "declined";

type PairVerdict =
  | { status: "unasked" }
  | { status: "remembered" }
  | { status: "failed" }
  | { status: "answered"; result: ContradictionResult };

export function nextRunAfter(from: Date, weekday: number | null = 0, hour = 3): Date {
  const next = new Date(from);

  next.setHours(hour, 0, 0, 0);

  while ((weekday !== null && next.getDay() !== weekday) || next.getTime() <= from.getTime()) {
    next.setDate(next.getDate() + 1);
    next.setHours(hour, 0, 0, 0);
  }

  return next;
}

export function previousRunAtOrBefore(from: Date, weekday: number | null = 0, hour = 3): Date {
  const previous = new Date(from);

  previous.setHours(hour, 0, 0, 0);

  while ((weekday !== null && previous.getDay() !== weekday) || previous.getTime() > from.getTime()) {
    previous.setDate(previous.getDate() - 1);
    previous.setHours(hour, 0, 0, 0);
  }

  return previous;
}

const GATE_LOW_CONFIDENCE_FLAGS = new Set(GATE_QUESTIONS.map((question) => `low_confidence:${question}`));

function isDoubtFlag(flag: string): boolean {
  return flag.startsWith("demote:") || flag.startsWith("shape_low:") || GATE_LOW_CONFIDENCE_FLAGS.has(flag);
}

export class Cleaner {
  private readonly store: Store;
  private readonly proposals: ProposalService;
  private readonly settingsService: SettingsService;
  private readonly state: CleanerState;
  private readonly jev: JevClient | null;
  private readonly model: string;
  private readonly timeoutMs: number | undefined;
  private readonly now: () => Date;
  private readonly cap: number;
  private readonly doubtCap: number;
  private readonly doubtReadLimit: number;
  private readonly minAgeDays: number;
  private readonly mergeFloor: number;
  private readonly contradictionFloor: number;
  private readonly maxJevCalls: number;
  private readonly promotionReads: number;
  private readonly promotionWindowDays: number;
  private readonly rejectedCooldownDays: number;
  private readonly sharedScopeOf: (scope: string) => string;
  private readonly mode: Mode;
  private readonly log: (message: string) => void;

  constructor(options: CleanerOptions) {
    this.store = options.store;
    this.proposals = options.proposals;
    this.jev = options.jev;
    this.model = options.model;
    this.timeoutMs = options.timeoutMs;
    this.now = options.now ?? (() => new Date());
    this.cap = options.cap ?? 5;
    this.doubtCap = options.doubtCap ?? CLEANER_DOUBT_CAP;
    this.doubtReadLimit = options.doubtReadLimit ?? CLEANER_DOUBT_READ_LIMIT;
    this.minAgeDays = options.minAgeDays ?? 30;
    this.mergeFloor = options.mergeFloor ?? 0.9;
    this.contradictionFloor = options.contradictionFloor ?? 0.75;
    this.maxJevCalls = options.maxJevCalls ?? 200;
    this.promotionReads = options.promotionReads ?? 3;
    this.promotionWindowDays = options.promotionWindowDays ?? 30;
    this.rejectedCooldownDays = options.rejectedCooldownDays ?? 90;
    this.sharedScopeOf = options.sharedScopeOf ?? (() => "global");
    this.mode = options.mode ?? "t3";
    this.log = options.log ?? ((message) => console.log(message));

    const stamp = () => this.now().toISOString();

    this.settingsService = options.settings ?? new SettingsService(this.store, stamp);
    this.state = options.state ?? new StoreCleanerState(this.store, stamp);
  }

  async run(): Promise<CleanerReport> {
    const settings = await this.settings();
    const run = new CleanerRun(settings, await this.blockedIds());

    if (!(await run.doubts())) {
      await this.advanceDoubtsCursor(settings.now);
    }

    await run.promotions();
    await run.duplicatesAndContradictions();
    await run.retirements();

    return run.report();
  }

  private async settings(): Promise<RunSettings> {
    return {
      store: this.store,
      proposals: this.proposals,
      state: this.state,
      jev: this.jev,
      model: this.model,
      timeoutMs: this.timeoutMs,
      now: this.now(),
      cap: this.cap,
      doubtCap: this.doubtCap,
      doubtReadLimit: this.doubtReadLimit,
      doubtsSince: await this.doubtsSince(),
      minAgeDays: this.minAgeDays,
      mergeFloor: this.mergeFloor,
      contradictionFloor: this.contradictionFloor,
      confidenceFloor: await this.confidenceFloor(),
      maxJevCalls: this.maxJevCalls,
      promotionReads: this.promotionReads,
      sharedScopeOf: this.sharedScopeOf,
      mode: this.mode,
      promotionWindowDays: this.promotionWindowDays,
      log: this.log,
    };
  }

  private async advanceDoubtsCursor(runStart: Date): Promise<void> {
    try {
      await this.state.setLastDoubtsAt(runStart.toISOString());
    } catch (error) {
      this.log(`cleaner: saving the doubts cursor failed, the next run re-reads this window: ${String(error)}`);
    }
  }

  private async doubtsSince(): Promise<string> {
    const now = this.now();
    const fallback = new Date(now.getTime() - DOUBT_FIRST_RUN_DAYS * DAY_MS).toISOString();

    try {
      return (await this.state.getLastDoubtsAt()) ?? fallback;
    } catch (error) {
      this.log(`cleaner: reading the doubts cursor failed, using ${DOUBT_FIRST_RUN_DAYS} days: ${String(error)}`);

      return fallback;
    }
  }

  private async confidenceFloor(): Promise<number> {
    try {
      return (await this.settingsService.getSettings()).cleanerConfidenceFloor;
    } catch (error) {
      this.log(`cleaner: reading the confidence floor failed, using the default: ${String(error)}`);

      return DEFAULT_CLEANER_CONFIDENCE_FLOOR;
    }
  }

  private async blockedIds(): Promise<Set<string>> {
    const cutoff = new Date(this.now().getTime() - this.rejectedCooldownDays * DAY_MS).toISOString();
    const rows = await selectWhere<{ targetRecordId: string | null; target: string | null }>(
      this.store.proposals,
      `status = 'pending' OR (status = 'rejected' AND resolvedAt >= ${sqlString(cutoff)})`,
    );
    const blocked = new Set<string>();

    for (const row of rows) {
      if (row.targetRecordId !== null) {
        blocked.add(row.targetRecordId);
      }

      const replacementId = replacementOf(row.target);

      if (replacementId !== null) {
        blocked.add(replacementId);
      }
    }

    return blocked;
  }
}

function replacementOf(raw: string | null): string | null {
  if (raw === null) {
    return null;
  }

  try {
    const parsed: unknown = JSON.parse(raw);
    const id = typeof parsed === "object" && parsed !== null ? (parsed as ProposalTarget).replacementId : null;

    return typeof id === "string" && id !== "" ? id : null;
  } catch {
    return null;
  }
}

function isOutage(error: unknown): boolean {
  if (error instanceof MalformedAnswer) {
    return false;
  }

  const status = typeof error === "object" && error !== null ? (error as { status?: unknown }).status : undefined;

  if (typeof status !== "number") {
    return true;
  }

  return status >= 500 || status === 401 || status === 403 || status === 429;
}

class CleanerRun {
  private readonly s: RunSettings;
  private readonly blocked: Set<string>;
  private readonly proposed: Proposal[] = [];
  private jevCalls = 0;
  private consecutiveFailures = 0;
  private candidateCount = 0;
  private doubtTiles = 0;

  constructor(settings: RunSettings, blocked: Set<string>) {
    this.s = settings;
    this.blocked = blocked;
  }

  report(): CleanerReport {
    return { proposed: this.proposed, candidates: this.candidateCount, jevCalls: this.jevCalls };
  }

  private full(): boolean {
    return this.proposed.length - this.doubtTiles >= this.s.cap;
  }

  private budgetLeft(): boolean {
    return this.s.jev !== null && this.jevCalls < this.s.maxJevCalls;
  }

  private jevDown(): boolean {
    return this.consecutiveFailures >= MAX_CONSECUTIVE_JEV_FAILURES;
  }

  private async callJev<T>(label: string, work: () => Promise<T>): Promise<T | null> {
    this.jevCalls += 1;

    try {
      const result = await work();

      this.consecutiveFailures = 0;

      return result;
    } catch (error) {
      this.s.log(`cleaner: jev check failed for ${label}: ${String(error)}`);

      if (!isOutage(error)) {
        this.consecutiveFailures = 0;

        return null;
      }

      this.consecutiveFailures += 1;

      if (this.jevDown()) {
        this.s.log(
          `cleaner: jev failed ${MAX_CONSECUTIVE_JEV_FAILURES} times in a row, ending the jev stages for this run`,
        );
      }

      return null;
    }
  }

  private anyBlocked(...ids: string[]): boolean {
    return ids.some((id) => this.blocked.has(id));
  }

  private async submit(body: ProposalBody, doubted = false): Promise<boolean> {
    const target = body.target as ProposalTarget;
    const ids = target.replacementId ? [target.recordId, target.replacementId] : [target.recordId];

    if ((!doubted && this.full()) || this.anyBlocked(...ids)) {
      return false;
    }

    try {
      this.proposed.push(await this.s.proposals.createInternal(body));
    } catch (error) {
      this.s.log(`cleaner: creating the ${target.action} tile for ${ids.join(" -> ")} failed: ${String(error)}`);

      return false;
    }

    for (const id of ids) {
      this.blocked.add(id);
    }

    if (doubted) {
      this.doubtTiles += 1;
    }

    return true;
  }

  private provenance(title: string) {
    return {
      threadId: "cleaner",
      threadTitle: title,
      projectId: "",
      branch: "",
      machine: hostname(),
      link: "",
      excerpt: "",
      stage: "",
    };
  }

  private bodyFor(record: MemoryRecord, target: ProposalTarget, note: string | null = null): ProposalBody {
    const day = this.s.now.toISOString().slice(0, 10);
    const claim = [record.claim, note].filter((part) => part !== null && part.trim() !== "").join("\n\n");

    return {
      kind: record.kind,
      headline: record.headline,
      fields: {
        claim,
        rejected: record.rejected,
        where: record.where,
        trigger: record.trigger,
        leftBecause: record.leftBecause,
        fixHint: record.fixHint,
        impact: record.impact,
      },
      scope: record.scope,
      provenance: this.provenance(`Cleaner run ${day}`),
      origin: "cleaner",
      target,
    };
  }

  async doubts(): Promise<boolean> {
    const rows = await selectWhere<Record<string, unknown>>(
      this.s.store.proposals,
      `status = 'approved' AND origin = 'agent' AND resolvedAt > ${sqlString(this.s.doubtsSince)}`,
      this.s.doubtReadLimit,
    );
    const newestFirst = rows
      .map(rowToProposal)
      .sort((a, b) => (b.resolvedAt ?? "").localeCompare(a.resolvedAt ?? ""));

    for (const proposal of newestFirst) {
      if (this.doubtTiles >= this.s.doubtCap) {
        return true;
      }

      await this.doubt(proposal);
    }

    return rows.length >= this.s.doubtReadLimit;
  }

  private async doubt(proposal: Proposal): Promise<void> {
    const record = await findById<MemoryRecord>(this.s.store.records, proposal.id);
    const doubts = proposal.flags.filter(isDoubtFlag);

    if (record === undefined || record.lifecycle !== "live" || this.blocked.has(record.id)) {
      return;
    }

    if ((await this.supersedeDoubt(proposal, record)) !== "declined") {
      return;
    }

    if (doubts.length > 0) {
      await this.submit(
        this.bodyFor(record, {
          recordId: record.id,
          source: "v2",
          action: "retire",
          reason: `jev doubted at propose: ${doubts.join(", ")}`,
        }),
        true,
      );
    }
  }

  private async supersedeDoubt(proposal: Proposal, record: MemoryRecord): Promise<SupersedeOutcome> {
    const suggestion = proposal.suggestion;

    if (suggestion?.action !== "supersede" || suggestion.reasonClass === null) {
      return "declined";
    }

    const table = suggestion.source === "legacy" ? this.s.store.legacy : this.s.store.records;
    const older = await findById<MemoryRecord>(table, suggestion.recordId);

    if (older === undefined) {
      return "declined";
    }

    if (older.lifecycle !== "live") {
      return older.supersededBy === record.id ? "settled" : "declined";
    }

    if (!keeperIsReachable({ older, newer: record, similarity: 0 }, this.s.sharedScopeOf)) {
      return "declined";
    }

    const queued = await this.submit(
      this.bodyFor(
        older,
        {
          recordId: older.id,
          source: suggestion.source,
          action: "supersede",
          replacementId: record.id,
          reason: suggestion.reasonClass,
        },
        `Cleaner: ${suggestion.reasonClass} (jev doubted at propose)`,
      ),
      true,
    );

    return queued ? "queued" : "declined";
  }

  async promotions(): Promise<void> {
    if (this.full()) {
      return;
    }

    const since = new Date(this.s.now.getTime() - this.s.promotionWindowDays * DAY_MS).toISOString();
    const reads = await selectWhere<{ resultId: string }>(
      this.s.store.readLog,
      `source = 'legacy' AND (via IS NULL OR via = 'recall') AND ts >= ${sqlString(since)}`,
      100_000,
    );
    const counts = new Map<string, number>();

    for (const read of reads) {
      counts.set(read.resultId, (counts.get(read.resultId) ?? 0) + 1);
    }

    const popular = [...counts.entries()]
      .filter(([, count]) => count >= this.s.promotionReads)
      .sort((a, b) => b[1] - a[1]);

    for (const [id, count] of popular) {
      if (this.full()) {
        return;
      }

      await this.promote(id, count);
    }
  }

  private async promote(id: string, count: number): Promise<void> {
    const legacy = await findById<LegacyRecord>(this.s.store.legacy, id);
    const promoted = await selectWhere<{ id: string }>(this.s.store.records, `legacyId = ${sqlString(id)}`, 1);

    if (legacy === undefined || legacy.lifecycle !== "live" || promoted.length > 0) {
      return;
    }

    const day = this.s.now.toISOString().slice(0, 10);

    await this.submit({
      kind: legacy.kind,
      headline: legacy.headline,
      fields: { claim: legacy.claim, rejected: legacy.rejected },
      scope: legacy.scope,
      provenance: this.provenance(`Promotion ${day}`),
      origin: "promotion",
      target: {
        recordId: legacy.id,
        source: "legacy",
        action: "promote",
        reason: `returned ${count} times in ${this.s.promotionWindowDays} days`,
      },
    });
  }

  private async candidates(): Promise<MemoryRecord[]> {
    const cutoff = new Date(this.s.now.getTime() - this.s.minAgeDays * DAY_MS).toISOString();
    const rows = await selectWhere<MemoryRecord>(
      this.s.store.records,
      `lifecycle = 'live' AND createdAt < ${sqlString(cutoff)}`,
      20_000,
    );

    this.candidateCount = rows.length;

    return rows;
  }

  async duplicatesAndContradictions(): Promise<void> {
    if (this.full()) {
      return;
    }

    const pairs = await this.pairs(await this.candidates());

    for (const pair of pairs) {
      if (this.full()) {
        return;
      }

      if (this.anyBlocked(pair.older.id, pair.newer.id)) {
        continue;
      }

      if (pair.similarity >= this.s.mergeFloor) {
        await this.nearDuplicate(pair);
      } else {
        await this.contradiction(pair);
      }
    }
  }

  private async pairs(candidates: MemoryRecord[]): Promise<Pair[]> {
    const eligible = new Map(candidates.map((record) => [record.id, record]));
    const seen = new Set<string>();
    const pairs: Pair[] = [];

    for (const record of candidates) {
      const where = andAll([
        "lifecycle = 'live'",
        scopeClause(record.scope, this.s.sharedScopeOf(record.scope)),
        `id != ${sqlString(record.id)}`,
      ]);
      const near = await this.s.store.records
        .query()
        .nearestTo(record.vector)
        .distanceType("cosine")
        .where(where as string)
        .limit(5)
        .toArray();

      for (const hit of near) {
        const other = eligible.get(hit.id as string);
        const similarity = 1 - Number(hit._distance);

        if (other === undefined || similarity < this.s.contradictionFloor) {
          continue;
        }

        const key = [record.id, other.id].sort().join("|");

        if (seen.has(key)) {
          continue;
        }

        seen.add(key);

        const pair = orderPair(record, other, similarity);

        if (keeperIsReachable(pair, this.s.sharedScopeOf)) {
          pairs.push(pair);
        }
      }
    }

    return pairs.sort((a, b) => b.similarity - a.similarity);
  }

  private async nearDuplicate(pair: Pair): Promise<void> {
    const verdict = await this.consult(pair);

    if (verdict.status === "failed") {
      return;
    }

    if (verdict.status === "answered" && this.replaces(verdict.result)) {
      await this.supersede(pair, verdict.result);

      return;
    }

    if (this.s.mode === "standalone" && !this.mergeable(verdict)) {
      return;
    }

    await this.merge(pair, verdict.status === "answered" ? verdict.result : null);
  }

  private mergeable(verdict: PairVerdict): boolean {
    return verdict.status === "answered" && verdict.result.pick !== "coexist";
  }

  private async contradiction(pair: Pair): Promise<void> {
    const verdict = await this.consult(pair);

    if (verdict.status === "answered" && this.replaces(verdict.result)) {
      await this.supersede(pair, verdict.result);
    }
  }

  private replaces(result: ContradictionResult): boolean {
    return result.pick === "replace" && result.confidence >= this.s.confidenceFloor;
  }

  private async consult(pair: Pair): Promise<PairVerdict> {
    const jev = this.s.jev;

    if (jev === null) {
      return { status: "unasked" };
    }

    const label = `${pair.older.id} | ${pair.newer.id}`;

    if (await this.pairRemembered(pair, label)) {
      return { status: "remembered" };
    }

    if (this.jevDown()) {
      return { status: "failed" };
    }

    if (!this.budgetLeft()) {
      return { status: "unasked" };
    }

    const result = await this.callJev(label, () =>
      askContradiction(jev, this.s.model, recordText(pair.older), recordText(pair.newer), this.s.timeoutMs),
    );

    if (result === null) {
      return { status: "failed" };
    }

    if (result.pick === "coexist") {
      await this.guarded(`remembering the coexist verdict for ${label}`, () =>
        this.s.state.rememberPair(stampOf(pair.older), stampOf(pair.newer)),
      );
    }

    return { status: "answered", result };
  }

  private async pairRemembered(pair: Pair, label: string): Promise<boolean> {
    try {
      return await this.s.state.isPairRemembered(stampOf(pair.older), stampOf(pair.newer));
    } catch (error) {
      this.s.log(`cleaner: reading the remembered verdict for ${label} failed: ${String(error)}`);

      return false;
    }
  }

  private async guarded(action: string, work: () => Promise<void>): Promise<void> {
    try {
      await work();
    } catch (error) {
      this.s.log(`cleaner: ${action} failed: ${String(error)}`);
    }
  }

  private async merge(pair: Pair, result: ContradictionResult | null): Promise<void> {
    const jev = result === null ? "" : `, jev ${result.choice} ${result.confidence.toFixed(2)}`;

    await this.submit(
      this.bodyFor(pair.older, {
        recordId: pair.older.id,
        source: "v2",
        action: "merge",
        replacementId: pair.newer.id,
        reason: `near-duplicate of ${pair.newer.id} (cosine ${pair.similarity.toFixed(2)}${jev})`,
      }),
    );
  }

  private async supersede(pair: Pair, result: ContradictionResult): Promise<void> {
    await this.submit(
      this.bodyFor(
        pair.older,
        {
          recordId: pair.older.id,
          source: "v2",
          action: "supersede",
          replacementId: pair.newer.id,
          reason: result.reasonClass as string,
        },
        `Cleaner: ${result.reasonClass} (jev confidence ${result.confidence.toFixed(2)})`,
      ),
    );
  }

  async retirements(): Promise<void> {
    if (this.full() || this.s.jev === null) {
      return;
    }

    const candidates = rotate(await this.candidates(), this.s.now);

    for (const record of candidates) {
      if (this.full() || this.jevDown() || !this.budgetLeft()) {
        return;
      }

      if (this.blocked.has(record.id)) {
        continue;
      }

      await this.assess(this.s.jev, record);
    }
  }

  private async assess(jev: JevClient, record: MemoryRecord): Promise<void> {
    const stamp = stampOf(record);

    if (await this.recordRemembered(stamp)) {
      return;
    }

    const verdict = await this.callJev(record.id, () =>
      askRelevance(jev, this.s.model, recordText(record), this.s.timeoutMs),
    );

    if (verdict === null) {
      return;
    }

    if (verdict.pick.startsWith("keep")) {
      await this.guarded(`remembering the keep verdict for ${record.id}`, () => this.s.state.rememberRecord(stamp));

      return;
    }

    if (verdict.pick === "shelve-lean" && verdict.confidence < this.s.confidenceFloor) {
      return;
    }

    await this.submit(
      this.bodyFor(record, {
        recordId: record.id,
        source: "v2",
        action: "retire",
        reason: `relevance ${verdict.pick} (confidence ${verdict.confidence.toFixed(2)})`,
      }),
    );
  }

  private async recordRemembered(stamp: RecordStamp): Promise<boolean> {
    try {
      return await this.s.state.isRecordRemembered(stamp);
    } catch (error) {
      this.s.log(`cleaner: reading the remembered verdict for ${stamp.id} failed: ${String(error)}`);

      return false;
    }
  }
}

function stampOf(record: MemoryRecord): RecordStamp {
  return { id: record.id, updatedAt: record.createdAt };
}

function orderPair(a: MemoryRecord, b: MemoryRecord, similarity: number): Pair {
  const aFirst = a.createdAt < b.createdAt || (a.createdAt === b.createdAt && a.id < b.id);

  return aFirst ? { older: a, newer: b, similarity } : { older: b, newer: a, similarity };
}

function keeperIsReachable(pair: Pair, sharedScopeOf: (scope: string) => string): boolean {
  return pair.newer.scope === pair.older.scope || pair.newer.scope === sharedScopeOf(pair.older.scope);
}

function rotate(records: MemoryRecord[], now: Date): MemoryRecord[] {
  const week = Math.floor(now.getTime() / (7 * DAY_MS));
  const keyed = records.map((record) => ({
    record,
    key: createHash("sha256").update(`${week}:${record.id}`).digest("hex"),
  }));

  return keyed.sort((a, b) => (a.key < b.key ? -1 : 1)).map((entry) => entry.record);
}

export interface CleanerSchedule {
  stop(): void;
}

export interface ScheduleOptions {
  now?: () => Date;
  log?: (message: string) => void;
  state?: Pick<CleanerState, "getLastRunAt" | "setLastRunAt">;
  catchUpDelayMs?: number;
}

export interface SlotOptions extends ScheduleOptions {
  weekday: number | null;
  hour: number;
  label?: string;
}

export const CLEANER_SLOT = { weekday: null, hour: 3, label: "cleaner" } as const satisfies SlotOptions;

export function scheduleAt(run: () => Promise<unknown>, options: SlotOptions): CleanerSchedule {
  const { weekday, hour } = options;
  const label = options.label ?? "cleaner";
  const now = options.now ?? (() => new Date());
  const log = options.log ?? ((message) => console.error(message));
  const state = options.state;
  const catchUpDelayMs = options.catchUpDelayMs ?? CATCH_UP_DELAY_MS;
  let timer: NodeJS.Timeout | null = null;
  let catchUpTimer: NodeJS.Timeout | null = null;
  let stopped = false;

  const execute = async () => {
    try {
      await run();
    } catch (error) {
      log(`${label} run failed: ${String(error)}`);

      return;
    }

    try {
      await state?.setLastRunAt(now().toISOString());
    } catch (error) {
      log(`${label}: recording the last run failed: ${String(error)}`);
    }
  };

  const arm = (from: Date) => {
    if (stopped) {
      return;
    }

    const due = nextRunAfter(from, weekday, hour);

    timer = setTimeout(async () => {
      await execute();

      arm(new Date(Math.max(now().getTime(), due.getTime() + 60_000)));
    }, due.getTime() - now().getTime());
    timer.unref();
  };

  const catchUp = async () => {
    if (state === undefined) {
      return;
    }

    const last = await state.getLastRunAt();

    if (last === null) {
      await state.setLastRunAt(now().toISOString());

      return;
    }

    if (stopped || previousRunAtOrBefore(now(), weekday, hour).getTime() <= Date.parse(last)) {
      return;
    }

    catchUpTimer = setTimeout(() => {
      void execute();
    }, catchUpDelayMs);
    catchUpTimer.unref();
  };

  arm(now());
  catchUp().catch((error) => log(`${label}: catch-up check failed: ${String(error)}`));

  return {
    stop() {
      stopped = true;

      if (timer !== null) {
        clearTimeout(timer);
      }

      if (catchUpTimer !== null) {
        clearTimeout(catchUpTimer);
      }
    },
  };
}
