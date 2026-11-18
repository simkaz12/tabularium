export type Kind = "fact" | "decision" | "defect";
export type Lifecycle = "live" | "superseded" | "retired" | "archived";
export type ReasonClass = "correction" | "changed" | "elaborated";
export type Origin = "agent" | "harvest" | "cleaner" | "promotion";
export type Source = "v2" | "legacy";
export type ProposalStatus = "pending" | "approved" | "rejected" | "reinforced" | "asked" | "folded";
export type AskAnswer = "fold" | "keep_both" | "unanswered";
export type TargetAction = "retire" | "supersede" | "merge" | "promote" | "close";
export type Audience = "decider" | "builder" | "reviewer" | "navigator";
export type Reader = "lookup" | "trap" | "ruling" | "none" | "unclear";
export type DefectStatus = "open" | "taken" | "closed";

export const KINDS: readonly Kind[] = ["fact", "decision", "defect"];
export const ORIGINS: readonly Origin[] = ["agent", "harvest", "cleaner", "promotion"];
export const EXTERNAL_ORIGINS: readonly Origin[] = ["agent"];
export const REASON_CLASSES: readonly ReasonClass[] = ["correction", "changed", "elaborated"];
export const TARGET_ACTIONS: readonly TargetAction[] = ["retire", "supersede", "merge", "promote", "close"];
export const AUDIENCES: readonly Audience[] = ["decider", "builder", "reviewer", "navigator"];
export const READERS: readonly Reader[] = ["lookup", "trap", "ruling", "none", "unclear"];
export const DEFECT_STATUSES: readonly DefectStatus[] = ["open", "taken", "closed"];
export const PROPOSAL_STATUSES: readonly ProposalStatus[] = [
  "pending",
  "approved",
  "rejected",
  "reinforced",
  "asked",
  "folded",
];
export const HEADLINE_MAX = 100;
export const GLOBAL_SCOPE = "global";

export interface Provenance {
  threadId: string;
  threadTitle: string;
  projectId: string;
  branch: string;
  machine: string;
  link: string;
  excerpt: string;
  stage: string;
}

export interface Fields {
  claim: string;
  why?: string | null;
  rejected?: string | null;
  where?: string | null;
  trigger?: string | null;
  leftBecause?: string | null;
  fixHint?: string | null;
  impact?: string | null;
}

export interface MemoryRecord {
  id: string;
  kind: Kind;
  headline: string;
  claim: string;
  why: string | null;
  rejected: string | null;
  scope: string;
  purpose: string | null;
  topic: string | null;
  audience: Audience | null;
  reader: Reader | null;
  where: string | null;
  trigger: string | null;
  leftBecause: string | null;
  fixHint: string | null;
  impact: string | null;
  defectStatus: DefectStatus | null;
  takenThreadId: string | null;
  takenLink: string | null;
  takenAt: string | null;
  closedBy: string | null;
  closedAt: string | null;
  lifecycle: Lifecycle;
  supersededBy: string | null;
  supersessionReason: ReasonClass | null;
  retiredReason: string | null;
  lifecycleNote: string | null;
  threadId: string;
  threadTitle: string;
  projectId: string;
  branch: string;
  machine: string;
  link: string;
  excerpt: string;
  stage: string;
  origin: Origin;
  legacyId: string | null;
  reinforcedCount: number;
  lastReinforcedAt: string | null;
  createdAt: string;
  approvedAt: string;
  vector: number[];
}

export interface LegacyRecord extends MemoryRecord {
  oldScope: string;
  oldSource: string;
  readCount: number;
  originalCreatedAt: string;
  reviewed: boolean;
  collapsedIds: string | null;
}

export type StoredRecord = MemoryRecord | LegacyRecord;

export interface ProposalTarget {
  recordId: string;
  source: Source;
  action: TargetAction;
  replacementId?: string | null;
  recordHeadline?: string | null;
  replacementHeadline?: string | null;
  reason?: string | null;
}

export interface ProposalBody {
  kind: Kind;
  headline: string;
  fields: Fields;
  scope: string;
  provenance: Provenance;
  origin: Origin;
  target?: ProposalTarget | null;
  supersedes?: string[];
  audience?: Audience | null;
  reader?: Reader | null;
}

export interface RelationVerdict {
  pick: string;
  confidence: number;
  probabilities: Record<string, number>;
  reasonClass: ReasonClass | null;
}

export interface PairVerdict {
  newCoversOld: number;
  sameSubject: number;
}

export interface Neighbour {
  id: string;
  source: Source;
  headline: string;
  similarity: number;
  relation: RelationVerdict | null;
  pair?: PairVerdict | null;
}

export interface Suggestion {
  action: "reinforce" | "supersede";
  recordId: string;
  source: Source;
  reasonClass: ReasonClass | null;
}

export interface Checks {
  status: "ok" | "unavailable" | "skipped";
  reason?: string;
  model?: string;
  wordingHashes?: Record<string, string>;
  answers?: Record<string, unknown>;
  relations?: Record<string, unknown>;
  pairs?: Record<string, unknown>;
}

export interface Resolution {
  reason?: string | null;
  recordId?: string | null;
}

export interface AskExisting {
  id: string;
  source: Source;
  kind: Kind;
  headline: string;
  fields: Fields;
}

export interface FoldDraft {
  kind: Kind;
  headline: string;
  fields: Fields;
  purpose: string | null;
  topic: string | null;
  audience: Audience | null;
  reader: Reader | null;
  checks: Checks;
  flags: string[];
}

export interface Ask {
  existing: AskExisting;
  similarity: number;
  sameSubject: number;
  newCoversOld: number;
  existingIsFold: boolean;
  askedAt: string;
  expiresAt: string;
  answer: AskAnswer | null;
  answeredAt: string | null;
  reason: string | null;
  foldRecordId: string | null;
  existingGone?: boolean;
  foldGone?: boolean;
  heldRecordId?: string | null;
  fold: FoldDraft | null;
  foldCoversOld: number | null;
  foldCoversNew: number | null;
}

export interface Proposal extends ProposalBody {
  id: string;
  status: ProposalStatus;
  audience: Audience | null;
  reader: Reader | null;
  purpose: string | null;
  topic: string | null;
  checks: Checks;
  flags: string[];
  neighbours: Neighbour[];
  suggestion: Suggestion | null;
  createdAt: string;
  resolvedAt: string | null;
  resolution: Resolution | null;
  ask: Ask | null;
}

export interface ReadLogEntry {
  id: string;
  ts: string;
  query: string;
  scope: string | null;
  resultId: string;
  source: Source;
  rank: number;
  score: number;
}

export interface JevRequest {
  state: unknown;
  questions: Record<string, unknown>;
  model?: string;
}

export interface JevResponse {
  model: string;
  answers: Record<string, any>;
  usage?: unknown;
}

export interface JevClient {
  systemOne(request: JevRequest, options?: { timeout?: number }): PromiseLike<JevResponse>;
}

export interface JevSettings {
  jev?: JevClient | null;
  model?: string;
  timeoutMs?: number;
}

export class HttpError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function recordText(record: {
  kind?: Kind;
  headline: string;
  claim: string;
  where?: string | null;
  trigger?: string | null;
}): string {
  const parts =
    record.kind === "defect"
      ? [record.headline, record.where ?? "", record.claim, record.trigger ?? ""]
      : [record.headline, record.claim];

  return parts.join("\n");
}
