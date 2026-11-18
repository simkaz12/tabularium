import type { JevClient, JevResponse } from "../domain.ts";
import type { RecallCandidate, Reranker } from "../recall.ts";
import { ask, scoreOf } from "./checks.ts";
import { RERANK_QUESTION } from "./questions.ts";

export const RERANK_BATCH = 10;

export interface RerankOptions {
  client: JevClient;
  model: string;
  timeoutMs?: number;
  batchSize?: number;
  onError?: (error: unknown) => void;
}

function candidateState(query: string, candidate: RecallCandidate) {
  return {
    query,
    record: {
      kind: candidate.kind,
      headline: candidate.headline,
      claim: candidate.fields.claim,
      ...(candidate.fields.why ? { why: candidate.fields.why } : {}),
      rejected: candidate.fields.rejected,
      scope: candidate.scope,
    },
  };
}

export function orderByScore(
  candidates: readonly RecallCandidate[],
  scores: ReadonlyArray<number | null>,
): RecallCandidate[] {
  const slots: number[] = [];
  const scored: Array<{ candidate: RecallCandidate; index: number; rank: number }> = [];

  candidates.forEach((candidate, index) => {
    const score = scores[index];

    if (typeof score !== "number") {
      return;
    }

    slots.push(index);
    scored.push({ candidate: { ...candidate, score, jevScored: true }, index, rank: candidate.source === "v2" ? 0 : 1 });
  });

  scored.sort((a, b) => {
    if (b.candidate.score !== a.candidate.score) {
      return b.candidate.score - a.candidate.score;
    }

    if (a.rank !== b.rank) {
      return a.rank - b.rank;
    }

    return a.index - b.index;
  });

  const ordered = [...candidates];

  slots.forEach((slot, position) => {
    ordered[slot] = (scored[position] as { candidate: RecallCandidate }).candidate;
  });

  return ordered;
}

export function createJevReranker(options: RerankOptions): Reranker {
  const batchSize = options.batchSize ?? RERANK_BATCH;

  async function scoreOne(query: string, candidate: RecallCandidate): Promise<number> {
    const response: JevResponse = await ask(
      options.client,
      options.model,
      candidateState(query, candidate),
      [RERANK_QUESTION],
      options.timeoutMs,
    );

    return scoreOf(response, RERANK_QUESTION.id).score;
  }

  return async (query, candidates) => {
    if (candidates.length === 0) {
      return candidates;
    }

    const scores: Array<number | null> = [];
    const failures: unknown[] = [];

    for (let start = 0; start < candidates.length; start += batchSize) {
      const batch = candidates.slice(start, start + batchSize);
      const settled = await Promise.allSettled(batch.map((candidate) => scoreOne(query, candidate)));

      for (const result of settled) {
        if (result.status === "fulfilled") {
          scores.push(result.value);
        } else {
          scores.push(null);
          failures.push(result.reason);
        }
      }
    }

    for (const failure of failures) {
      options.onError?.(failure);
    }

    if (failures.length === candidates.length) {
      return null;
    }

    return orderByScore(candidates, scores);
  };
}
