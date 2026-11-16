import { CONFIDENCE_FLOOR, GATE_QUESTIONS } from "../config.ts";
import { AUDIENCES, READERS, REASON_CLASSES } from "../domain.ts";
import type {
  Audience,
  JevClient,
  JevResponse,
  Neighbour,
  PairVerdict,
  ProposalBody,
  Reader,
  ReasonClass,
  RelationVerdict,
  Suggestion,
} from "../domain.ts";
import type { CheckOutcome, NeighbourCandidate } from "../proposals.ts";
import {
  AUDIENCE_QUESTION,
  BREADTH_QUESTION,
  CLAIM_STANDING_QUESTION,
  CONTRADICTION_QUESTION,
  EXCERPT_GROUNDS_QUESTION,
  HEADLINE_FAITHFUL_QUESTION,
  KIND_QUESTION,
  NEW_COVERS_OLD_QUESTION,
  PERISHABLE_QUESTION,
  PURPOSE_QUESTION,
  READER_QUESTION,
  REASON_QUESTION,
  RELATION_QUESTION,
  RELEVANCE_QUESTION,
  REJECTED_ALTERNATIVE_QUESTION,
  SAME_SUBJECT_QUESTION,
  WORTH_QUESTION,
  questionPayload,
  scopeQuestion,
  topicQuestion,
  wordingHash,
} from "./questions.ts";
import type { QuestionSpec, TopicEntry } from "./questions.ts";

export const SHAPE_FLOOR = 0.5;
const NEIGHBOUR_CHAR_CAP = 1500;
const UNCLEAR = "unclear";
const SHAPE_QUESTIONS = [
  "headline_faithful",
  "rejected_names_alternative",
  "claim_standing",
  "excerpt_grounds",
];

export interface CheckerOptions {
  client: JevClient | null;
  model: string;
  topics: readonly TopicEntry[];
  timeoutMs?: number;
  scopeLabels: () => Promise<Array<[string, string | null]>>;
}

export interface ChoiceAnswer {
  choice: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export class MalformedAnswer extends Error {}

function answerOf(response: JevResponse, id: string): Record<string, any> {
  const answer = response.answers?.[id];

  if (typeof answer !== "object" || answer === null) {
    throw new MalformedAnswer(`missing answer for ${id}`);
  }

  return answer;
}

export function choiceOf(response: JevResponse, id: string): ChoiceAnswer {
  const answer = answerOf(response, id);

  if (typeof answer.choice !== "string" || typeof answer.confidence !== "number") {
    throw new MalformedAnswer(`answer for ${id} is not a choice`);
  }

  return {
    choice: answer.choice,
    confidence: answer.confidence,
    probabilities: (answer.probabilities ?? {}) as Record<string, number>,
  };
}

export function yesProbability(response: JevResponse, id: string): number {
  const answer = choiceOf(response, id);

  if (answer.choice !== "yes" && answer.choice !== "no") {
    throw new MalformedAnswer(`answer for ${id} is neither yes nor no`);
  }

  return answer.probabilities.yes ?? (answer.choice === "yes" ? answer.confidence : 1 - answer.confidence);
}

export function pairVerdict(response: JevResponse): PairVerdict {
  return {
    newCoversOld: yesProbability(response, NEW_COVERS_OLD_QUESTION.id),
    sameSubject: yesProbability(response, SAME_SUBJECT_QUESTION.id),
  };
}

export function noulOf(response: JevResponse, id: string): number {
  const answer = answerOf(response, id);

  if (typeof answer.noul !== "number") {
    throw new MalformedAnswer(`answer for ${id} is not a noul`);
  }

  return answer.noul;
}

export function scoreOf(response: JevResponse, id: string): { score: number; probabilities: Record<string, number> } {
  const answer = answerOf(response, id);

  if (typeof answer.score !== "number") {
    throw new MalformedAnswer(`answer for ${id} is not a score`);
  }

  return { score: answer.score, probabilities: (answer.probabilities ?? {}) as Record<string, number> };
}

export function buildQuestions(specs: readonly QuestionSpec[]): Record<string, unknown> {
  const questions: Record<string, unknown> = {};

  for (const spec of specs) {
    questions[spec.id] = questionPayload(spec);
  }

  return questions;
}

export function hashesOf(specs: readonly QuestionSpec[]): Record<string, string> {
  const hashes: Record<string, string> = {};

  for (const spec of specs) {
    hashes[spec.id] = wordingHash(spec);
  }

  return hashes;
}

export function ask(
  client: JevClient,
  model: string,
  state: unknown,
  specs: readonly QuestionSpec[],
  timeoutMs?: number,
): PromiseLike<JevResponse> {
  const options =
    timeoutMs === undefined ? undefined : { timeout: timeoutMs, signal: AbortSignal.timeout(timeoutMs) };

  return client.systemOne({ model, state, questions: buildQuestions(specs) }, options);
}

export function recordSummary(body: ProposalBody): string {
  const { fields } = body;
  const lines = [body.headline];

  if (fields.where) {
    lines.push(`Where: ${fields.where}`);
  }

  lines.push(fields.claim);

  if (fields.trigger) {
    lines.push(`Trigger: ${fields.trigger}`);
  }

  if (fields.rejected) {
    lines.push(`Rejected: ${fields.rejected}`);
  }

  return lines.join("\n");
}

function neighbourState(candidates: readonly NeighbourCandidate[]) {
  return candidates.map((candidate) => ({
    content: candidate.text.slice(0, NEIGHBOUR_CHAR_CAP),
    kind: candidate.kind,
    scope: candidate.scope,
  }));
}

export function computeFlags(
  body: ProposalBody,
  response: JevResponse,
  specIds: readonly string[],
): string[] {
  const flags: string[] = [];
  const kind = choiceOf(response, "kind").choice;
  const worth = choiceOf(response, "worth").choice;

  if (kind !== UNCLEAR && kind !== body.kind) {
    flags.push("kind_differs");
  }

  if (worth !== UNCLEAR && worth !== "keep") {
    flags.push("worth_not_keep");
  }

  if (specIds.includes("perishable")) {
    const perishable = choiceOf(response, "perishable").choice;

    if (perishable !== UNCLEAR && perishable !== "standing") {
      flags.push("perishable_not_standing");
    }
  }

  for (const id of SHAPE_QUESTIONS) {
    if (specIds.includes(id) && noulOf(response, id) < SHAPE_FLOOR) {
      flags.push(`shape_low:${id}`);
    }
  }

  return flags;
}

export function unclearFlags(response: JevResponse, specs: readonly QuestionSpec[]): string[] {
  return specs
    .filter((spec) => spec.type === "choice")
    .filter((spec) => choiceOf(response, spec.id).choice === UNCLEAR)
    .map((spec) => `unclear:${spec.id}`);
}

export function lowConfidenceFlags(response: JevResponse, specs: readonly QuestionSpec[]): string[] {
  return specs
    .filter((spec) => spec.type === "choice" && GATE_QUESTIONS.includes(spec.id))
    .map((spec) => ({ id: spec.id, answer: choiceOf(response, spec.id) }))
    .filter(({ answer }) => answer.choice !== UNCLEAR && answer.confidence < CONFIDENCE_FLOOR)
    .map(({ id }) => `low_confidence:${id}`);
}

const DEMOTING: Record<string, (choice: string) => boolean> = {
  kind: (choice) => choice === "neither",
  worth: (choice) => choice !== "keep" && choice !== UNCLEAR,
  perishable: (choice) => choice === "status" || choice === "sequencing",
  reader: (choice) => choice === "none",
};

export function demoteFlags(response: JevResponse, specIds: readonly string[]): string[] {
  const flags: string[] = [];

  for (const id of GATE_QUESTIONS) {
    if (!specIds.includes(id)) {
      continue;
    }

    const answer = choiceOf(response, id);

    if (answer.confidence >= CONFIDENCE_FLOOR && DEMOTING[id]?.(answer.choice)) {
      flags.push(`demote:${id}=${answer.choice}`);
    }
  }

  return flags;
}

function audienceOf(response: JevResponse): Audience | null {
  const { choice } = choiceOf(response, "audience");

  return (AUDIENCES as readonly string[]).includes(choice) ? (choice as Audience) : null;
}

function readerOf(response: JevResponse): Reader | null {
  const { choice } = choiceOf(response, "reader");

  return (READERS as readonly string[]).includes(choice) ? (choice as Reader) : null;
}

function labelOf(response: JevResponse, id: string): string | null {
  const { choice } = choiceOf(response, id);

  return choice === UNCLEAR ? null : choice;
}

function reasonClassOf(response: JevResponse): ReasonClass | null {
  try {
    const { choice } = choiceOf(response, "reason");

    return (REASON_CLASSES as readonly string[]).includes(choice) ? (choice as ReasonClass) : null;
  } catch {
    return null;
  }
}

function reasonConfidenceOf(response: JevResponse): number {
  try {
    return choiceOf(response, "reason").confidence;
  } catch {
    return 0;
  }
}

export function relationVerdict(response: JevResponse): RelationVerdict {
  const relation = choiceOf(response, "relation");

  return {
    pick: relation.choice,
    confidence: relation.confidence,
    probabilities: relation.probabilities,
    reasonClass: relation.choice === "replacement" ? reasonClassOf(response) : null,
  };
}

export function suggestionFrom(neighbours: readonly Neighbour[]): Suggestion | null {
  let best: Neighbour | null = null;

  for (const neighbour of neighbours) {
    const relation = neighbour.relation;

    if (relation?.pick !== "restatement" && relation?.pick !== "replacement") {
      continue;
    }

    if (best === null || relation.confidence > (best.relation?.confidence ?? 0)) {
      best = neighbour;
    }
  }

  if (best === null) {
    return null;
  }

  if (best.relation?.pick === "restatement") {
    return { action: "reinforce", recordId: best.id, source: best.source, reasonClass: null };
  }

  return {
    action: "supersede",
    recordId: best.id,
    source: best.source,
    reasonClass: best.relation?.reasonClass ?? null,
  };
}

export function neighbourFlags(neighbours: readonly Neighbour[]): string[] {
  return neighbours.flatMap((neighbour) => {
    const pick = neighbour.relation?.pick;

    return pick === "restatement" || pick === "replacement" ? [`${pick}:${neighbour.id}`] : [];
  });
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export class ProposalChecker {
  private readonly options: CheckerOptions;

  constructor(options: CheckerOptions) {
    this.options = options;
  }

  async check(body: ProposalBody, candidates: NeighbourCandidate[]): Promise<CheckOutcome> {
    const { client } = this.options;
    const plain = candidates.map(toNeighbour);

    if (client === null) {
      return unavailable(plain, "absent", "no assessor client configured");
    }

    const specs = await this.recordSpecs(body);
    const [recordResult, ...rest] = await Promise.allSettled([
      this.askRecord(client, body, candidates, specs),
      ...candidates.map((candidate) => this.askRelation(client, body, candidate)),
      ...candidates.map((candidate) => this.askPair(client, body, candidate)),
    ]);
    const relationResults = rest.slice(0, candidates.length);
    const pairResults = rest.slice(candidates.length);

    if (recordResult?.status !== "fulfilled") {
      const reason = recordResult?.status === "rejected" ? messageOf(recordResult.reason) : "no result";

      return unavailable(plain, "error", reason);
    }

    return this.assemble(body, specs, recordResult.value, plain, relationResults, pairResults);
  }

  private async recordSpecs(body: ProposalBody): Promise<QuestionSpec[]> {
    const labels = await this.options.scopeLabels();
    const specs = [KIND_QUESTION, WORTH_QUESTION];

    if (body.kind !== "defect") {
      specs.push(PERISHABLE_QUESTION);
    }

    specs.push(
      BREADTH_QUESTION,
      scopeQuestion(withDeclared(labels, body.scope)),
      PURPOSE_QUESTION,
      READER_QUESTION,
      AUDIENCE_QUESTION,
      HEADLINE_FAITHFUL_QUESTION,
      CLAIM_STANDING_QUESTION,
      EXCERPT_GROUNDS_QUESTION,
    );

    if (body.kind === "decision" && body.fields.rejected) {
      specs.push(REJECTED_ALTERNATIVE_QUESTION);
    }

    if (this.options.topics.length > 0) {
      specs.push(topicQuestion(this.options.topics));
    }

    return specs;
  }

  private async askRecord(
    client: JevClient,
    body: ProposalBody,
    candidates: readonly NeighbourCandidate[],
    specs: readonly QuestionSpec[],
  ): Promise<JevResponse> {
    const state: Record<string, unknown> = {
      text: recordSummary(body),
      declared_kind: body.kind,
      source_scope: body.scope,
      headline: body.headline,
      claim: body.fields.claim,
      rejected: body.fields.rejected ?? null,
      excerpt: body.provenance.excerpt,
    };

    if (body.kind === "defect") {
      state.where = body.fields.where ?? null;
      state.trigger = body.fields.trigger ?? null;
      state.left_because = body.fields.leftBecause ?? null;
    }

    if (candidates.length > 0) {
      state.neighbours = neighbourState(candidates);
    }

    return await ask(client, this.options.model, state, specs, this.options.timeoutMs);
  }

  private async askRelation(
    client: JevClient,
    body: ProposalBody,
    candidate: NeighbourCandidate,
  ): Promise<JevResponse> {
    const state = {
      existing: candidate.text.slice(0, NEIGHBOUR_CHAR_CAP),
      incoming: recordSummary(body),
      embedding_similarity: candidate.similarity,
    };

    return await ask(
      client,
      this.options.model,
      state,
      [RELATION_QUESTION, REASON_QUESTION],
      this.options.timeoutMs,
    );
  }

  private async askPair(
    client: JevClient,
    body: ProposalBody,
    candidate: NeighbourCandidate,
  ): Promise<JevResponse> {
    const state = {
      existing: candidate.pairText.slice(0, NEIGHBOUR_CHAR_CAP),
      incoming: recordSummary(body),
    };

    return await ask(
      client,
      this.options.model,
      state,
      [NEW_COVERS_OLD_QUESTION, SAME_SUBJECT_QUESTION],
      this.options.timeoutMs,
    );
  }

  private assemble(
    body: ProposalBody,
    specs: readonly QuestionSpec[],
    response: JevResponse,
    plain: Neighbour[],
    relationResults: Array<PromiseSettledResult<JevResponse>>,
    pairResults: Array<PromiseSettledResult<JevResponse>>,
  ): CheckOutcome {
    try {
      return this.finish(body, specs, response, plain, relationResults, pairResults);
    } catch (error) {
      return unavailable(plain, "error", messageOf(error));
    }
  }

  private finish(
    body: ProposalBody,
    specs: readonly QuestionSpec[],
    response: JevResponse,
    plain: Neighbour[],
    relationResults: Array<PromiseSettledResult<JevResponse>>,
    pairResults: Array<PromiseSettledResult<JevResponse>>,
  ): CheckOutcome {
    const specIds = specs.map((spec) => spec.id);
    const demotes = demoteFlags(response, specIds);
    const flags = [
      ...computeFlags(body, response, specIds),
      ...unclearFlags(response, specs),
      ...lowConfidenceFlags(response, specs),
      ...demotes,
    ];
    const relations: Record<string, unknown> = {};
    const pairs: Record<string, unknown> = {};

    if (body.kind === "decision" && !body.fields.rejected) {
      flags.push("rejected_missing");
    }

    const neighbours = plain.map((neighbour, index) => {
      const paired = this.withPair(neighbour, pairResults[index], pairs, flags);
      const result = relationResults[index];

      if (result?.status !== "fulfilled") {
        flags.push(`relation_unavailable:${neighbour.id}`);

        return paired;
      }

      relations[neighbour.id] = result.value.answers;

      return this.withRelation(paired, result.value, flags);
    });
    const topic = specIds.includes("topic") ? labelOf(response, "topic") : null;

    flags.push(...neighbourFlags(neighbours));

    return {
      checks: {
        status: "ok",
        model: response.model,
        wordingHashes: hashesOf([
          ...specs,
          RELATION_QUESTION,
          REASON_QUESTION,
          NEW_COVERS_OLD_QUESTION,
          SAME_SUBJECT_QUESTION,
        ]),
        answers: response.answers,
        relations,
        pairs,
      },
      flags,
      neighbours,
      suggestion: suggestionFrom(neighbours),
      purpose: labelOf(response, "purpose"),
      topic,
      audience: audienceOf(response),
      reader: readerOf(response),
    };
  }

  private withPair(
    neighbour: Neighbour,
    result: PromiseSettledResult<JevResponse> | undefined,
    pairs: Record<string, unknown>,
    flags: string[],
  ): Neighbour {
    if (result?.status !== "fulfilled") {
      flags.push(`pair_unavailable:${neighbour.id}`);

      return neighbour;
    }

    pairs[neighbour.id] = result.value.answers;

    try {
      return { ...neighbour, pair: pairVerdict(result.value) };
    } catch {
      flags.push(`pair_unavailable:${neighbour.id}`);

      return neighbour;
    }
  }

  private withRelation(neighbour: Neighbour, response: JevResponse, flags: string[]): Neighbour {
    try {
      const relation = relationVerdict(response);

      if (relation.pick === UNCLEAR) {
        flags.push(`unclear:relation:${neighbour.id}`);
      } else if (relation.confidence < CONFIDENCE_FLOOR) {
        flags.push(`low_confidence:relation:${neighbour.id}`);
      }

      if (relation.pick === "replacement") {
        flags.push(...reasonFlags(response, neighbour.id));
      }

      return { ...neighbour, relation };
    } catch {
      flags.push(`relation_unavailable:${neighbour.id}`);

      return neighbour;
    }
  }
}

function reasonFlags(response: JevResponse, neighbourId: string): string[] {
  try {
    if (choiceOf(response, "reason").choice === UNCLEAR) {
      return [`unclear:reason:${neighbourId}`];
    }
  } catch {
    return [`low_confidence:reason:${neighbourId}`];
  }

  return reasonConfidenceOf(response) < CONFIDENCE_FLOOR ? [`low_confidence:reason:${neighbourId}`] : [];
}

function withDeclared(
  labels: Array<[string, string | null]>,
  declared: string,
): Array<[string, string | null]> {
  const names = new Set(labels.map(([name]) => name));
  const merged = [...labels];

  for (const name of [declared, "global"]) {
    if (!names.has(name)) {
      merged.push([name, null]);
      names.add(name);
    }
  }

  return merged;
}

function toNeighbour(candidate: NeighbourCandidate): Neighbour {
  return {
    id: candidate.id,
    source: candidate.source,
    headline: candidate.headline,
    similarity: candidate.similarity,
    relation: null,
    pair: null,
  };
}

function unavailable(neighbours: Neighbour[], reason: string, detail: string): CheckOutcome {
  return {
    checks: { status: "unavailable", reason: `${reason}: ${detail}` },
    flags: ["jev_unavailable"],
    neighbours,
    suggestion: null,
    purpose: null,
    topic: null,
    audience: null,
    reader: null,
  };
}

export interface RelevanceVerdict {
  pick: string;
  confidence: number;
}

export interface ContradictionResult {
  pick: "coexist" | "replace";
  reasonClass: "correction" | "changed" | null;
  confidence: number;
  choice: string;
}

export async function askRelevance(
  client: JevClient,
  model: string,
  text: string,
  timeoutMs?: number,
): Promise<RelevanceVerdict> {
  const response = await ask(client, model, { text }, [RELEVANCE_QUESTION], timeoutMs);
  const { probabilities } = scoreOf(response, RELEVANCE_QUESTION.id);
  const grades = RELEVANCE_QUESTION.criteria as string[];
  const ranked = Object.entries(probabilities).sort(([, a], [, b]) => b - a);
  const top = ranked[0];
  const grade = top === undefined ? undefined : grades[Number(top[0])];

  if (top === undefined || grade === undefined) {
    throw new MalformedAnswer("relevance answer has no usable probabilities");
  }

  return { pick: grade, confidence: gradeConfidence(ranked.map(([, value]) => value)) };
}

function gradeConfidence(values: readonly number[]): number {
  const count = values.length;
  const peak = Math.max(...values);

  if (count <= 1) {
    return 1;
  }

  return Math.min(1, Math.max(0, (count * peak - 1) / (count - 1)));
}

export async function askContradiction(
  client: JevClient,
  model: string,
  existing: string,
  incoming: string,
  timeoutMs?: number,
): Promise<ContradictionResult> {
  const response = await ask(client, model, { existing, incoming }, [CONTRADICTION_QUESTION], timeoutMs);
  const answer = choiceOf(response, CONTRADICTION_QUESTION.id);

  if (answer.choice === "coexist") {
    return { pick: "coexist", reasonClass: null, confidence: answer.confidence, choice: answer.choice };
  }

  if (answer.choice === "correction" || answer.choice === "changed") {
    return { pick: "replace", reasonClass: answer.choice, confidence: answer.confidence, choice: answer.choice };
  }

  throw new MalformedAnswer(`unexpected contradiction label ${answer.choice}`);
}
