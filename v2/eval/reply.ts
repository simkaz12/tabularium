import { HttpError } from "../src/domain.ts";
import type { Proposal } from "../src/domain.ts";
import { findById } from "../src/store.ts";
import { fallbackShape, foldWhy, readJsonl } from "./engine.ts";
import type { EvalHarness, PairRow } from "./engine.ts";

export const REPLAY_ARM = "opus-fold_default-in_task";

export type ReplyAction = "fold" | "keep_both" | "continue";
export type Replay = ReplyAction | "none";
export type EndState = "folded" | "both_live" | "no_reply" | "failed";

export interface AskReply {
  pair_id: string;
  arm: string;
  action: ReplyAction | null;
  note: string | null;
  reason: string | null;
  error: string | null;
}

export interface AnswerOutcome {
  replay: Replay;
  status: string;
  ask_answer: string | null;
  end_state: EndState;
  existing_live: boolean;
  existing_superseded_by_fold: boolean;
  fold_live: boolean;
  held_live: boolean;
  fold_covers_old: number | null;
  fold_covers_new: number | null;
  error: string | null;
}

export function loadReplies(rows: AskReply[], arm: string = REPLAY_ARM): Map<string, AskReply> {
  const replies = new Map<string, AskReply>();

  for (const row of rows) {
    if (row.arm === arm) {
      replies.set(row.pair_id, row);
    }
  }

  return replies;
}

export function readReplies(primary: string, extra: string | undefined, arm: string = REPLAY_ARM): Map<string, AskReply> {
  const more = extra === undefined ? [] : readJsonl<AskReply>(extra);

  return loadReplies([...more, ...readJsonl<AskReply>(primary)], arm);
}

export function replayOf(reply: AskReply | undefined): Replay {
  if (reply === undefined || reply.error !== null || reply.action === null) {
    return "none";
  }

  return reply.action;
}

async function isLive(harness: EvalHarness, recordId: string | null): Promise<boolean> {
  if (recordId === null) {
    return false;
  }

  const row = await findById<Record<string, unknown>>(harness.store.records, recordId);

  return row?.lifecycle === "live";
}

async function supersededBy(harness: EvalHarness, recordId: string): Promise<string | null> {
  const row = await findById<Record<string, unknown>>(harness.store.records, recordId);

  return (row?.supersededBy as string | null | undefined) ?? null;
}

async function expire(harness: EvalHarness, proposal: Proposal): Promise<Proposal> {
  const expiresAt = proposal.ask?.expiresAt ?? new Date().toISOString();

  await harness.judged.proposals.expireAsks(expiresAt);

  return harness.judged.proposals.get(proposal.id);
}

async function applyReply(
  harness: EvalHarness,
  proposal: Proposal,
  pair: PairRow,
  reply: AskReply,
  replay: Replay,
): Promise<Proposal> {
  if (replay === "fold") {
    const shape = fallbackShape(pair.kind, reply.note ?? "", `${pair.pair_id}-fold`);

    return harness.judged.proposals.answer(proposal.id, {
      action: "fold",
      kind: pair.kind,
      headline: shape.headline,
      fields: { claim: foldWhy(shape), rejected: shape.rejected },
    });
  }

  if (replay === "keep_both") {
    return harness.judged.proposals.answer(proposal.id, { action: "keep_both", reason: reply.reason ?? "" });
  }

  return expire(harness, proposal);
}

async function settle(
  harness: EvalHarness,
  replay: Replay,
  existingId: string,
  answered: Proposal,
): Promise<AnswerOutcome> {
  const recordId = answered.resolution?.recordId ?? null;
  const folded = replay === "fold";
  const foldId = folded ? recordId : null;
  const heldId = folded ? null : recordId;
  const existingLive = await isLive(harness, existingId);
  const bySuperseder = (await supersededBy(harness, existingId)) === foldId && foldId !== null;
  const foldLive = await isLive(harness, foldId);
  const heldLive = await isLive(harness, heldId);
  const foldEnds = !existingLive && bySuperseder && foldLive;
  const heldEnds = existingLive && heldLive;
  const ended = folded ? foldEnds : heldEnds;

  return {
    replay,
    status: answered.status,
    ask_answer: answered.ask?.answer ?? null,
    end_state: ended ? (folded ? "folded" : "both_live") : "failed",
    existing_live: existingLive,
    existing_superseded_by_fold: bySuperseder,
    fold_live: foldLive,
    held_live: heldLive,
    fold_covers_old: answered.ask?.foldCoversOld ?? null,
    fold_covers_new: answered.ask?.foldCoversNew ?? null,
    error: null,
  };
}

function failure(replay: Replay, proposal: Proposal, error: unknown): AnswerOutcome {
  const message = error instanceof HttpError ? `${error.status} ${error.message}` : String(error);

  return {
    replay,
    status: proposal.status,
    ask_answer: null,
    end_state: "failed",
    existing_live: false,
    existing_superseded_by_fold: false,
    fold_live: false,
    held_live: false,
    fold_covers_old: null,
    fold_covers_new: null,
    error: message,
  };
}

export interface Answered {
  outcome: AnswerOutcome;
  proposal: Proposal | null;
}

export async function answerAsked(
  harness: EvalHarness,
  proposal: Proposal,
  pair: PairRow,
  existingId: string,
  reply: AskReply | undefined,
): Promise<Answered> {
  const replay = replayOf(reply);

  if (replay === "none" || reply === undefined) {
    const expired = await expire(harness, proposal);
    const outcome = await settleExpired(harness, expired, existingId);

    return { outcome, proposal: expired };
  }

  try {
    const answered = await applyReply(harness, proposal, pair, reply, replay);

    return { outcome: await settle(harness, replay, existingId, answered), proposal: answered };
  } catch (error) {
    return { outcome: failure(replay, proposal, error), proposal: null };
  }
}

async function settleExpired(harness: EvalHarness, expired: Proposal, existingId: string): Promise<AnswerOutcome> {
  const settled = await settle(harness, "continue", existingId, expired);

  return { ...settled, replay: "none", end_state: settled.end_state === "failed" ? "failed" : "no_reply" };
}
