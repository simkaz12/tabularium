import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";

export type Criteria = Record<string, string | null> | string[] | null;

export interface QuestionSpec {
  id: string;
  type: "choice" | "score" | "noul";
  instructions: string;
  criteria: Criteria;
}

export interface TopicEntry {
  name: string;
  description: string | null;
}

function pyString(value: string): string {
  let out = '"';

  for (const char of value) {
    const code = char.codePointAt(0) as number;

    if (char === '"') {
      out += '\\"';
    } else if (char === "\\") {
      out += "\\\\";
    } else if (char === "\n") {
      out += "\\n";
    } else if (char === "\r") {
      out += "\\r";
    } else if (char === "\t") {
      out += "\\t";
    } else if (code < 0x20 || code > 0x7e) {
      out += pyEscape(code);
    } else {
      out += char;
    }
  }

  return `${out}"`;
}

function pyEscape(code: number): string {
  if (code > 0xffff) {
    const offset = code - 0x10000;
    const high = 0xd800 + (offset >> 10);
    const low = 0xdc00 + (offset & 0x3ff);

    return pyEscape(high) + pyEscape(low);
  }

  return `\\u${code.toString(16).padStart(4, "0")}`;
}

export function pyJsonDumps(value: unknown): string {
  if (value === null) {
    return "null";
  }

  if (typeof value === "string") {
    return pyString(value);
  }

  if (Array.isArray(value)) {
    return `[${value.map(pyJsonDumps).join(", ")}]`;
  }

  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([key, entry]) => `${pyString(key)}: ${pyJsonDumps(entry)}`);

    return `{${entries.join(", ")}}`;
  }

  return JSON.stringify(value);
}

export function wordingHash(spec: QuestionSpec): string {
  const payload = `${spec.instructions}\n${pyJsonDumps(spec.criteria)}`;

  return createHash("sha256").update(payload, "utf8").digest("hex").slice(0, 16);
}

export function questionPayload(spec: QuestionSpec): Record<string, unknown> {
  if (spec.type === "noul") {
    return { type: "noul", instructions: spec.instructions };
  }

  return { type: spec.type, instructions: spec.instructions, criteria: spec.criteria };
}

const UNCLEAR = "the text does not give enough to tell";

export const RELATION_QUESTION: QuestionSpec = {
  id: "relation",
  type: "choice",
  instructions:
    "How does the 'incoming' text relate to the 'existing' record in a long-term memory " +
    "store for a coding agent? 'existing' is the older record, already in the store; " +
    "'incoming' is the newer text being evaluated against it.",
  criteria: {
    restatement: "same fact, no new information; the store should reinforce the existing record",
    distinct: "a different fact that happens to be similar; both should be kept",
    follow_up:
      "same subject, a later step or a refinement that builds on the existing record " +
      "without contradicting it; both should be kept",
    replacement:
      "same subject but the existing record is now wrong or outdated; the incoming text " +
      "should supersede it",
    unclear: UNCLEAR,
  },
};

export const REASON_QUESTION: QuestionSpec = {
  id: "reason",
  type: "choice",
  instructions:
    "The 'relation' question above judged this a replacement. Why does the existing " +
    "record no longer hold?",
  criteria: {
    correction: "the existing record was wrong when it was written",
    changed: "the existing record was true when written and the world has since moved",
    elaborated:
      "the existing record is still true and the incoming text adds detail or precision",
    unclear: UNCLEAR,
  },
};

export const RELEVANCE_QUESTION: QuestionSpec = {
  id: "relevance",
  type: "score",
  instructions:
    "Should this record stay in a long-term memory store for a coding agent? " +
    "Shelve it if a reader could land on it cheaply by reading the code, docs or git " +
    "history; if it is false or irrelevant now; if it is a timeline event (task shipped, " +
    "PR merged, suite green, deployed); or if it is session narrative rather than a fact " +
    "the session produced. Keep it if it stands on its own: a convention, a gotcha, a " +
    "defect or absence the code does not show, a correction, a preference, an " +
    "infrastructure detail, or the reason something was done.",
  criteria: ["shelve-clear", "shelve-lean", "keep-lean", "keep-clear"],
};

export const CONTRADICTION_QUESTION: QuestionSpec = {
  id: "contradiction",
  type: "choice",
  instructions:
    "Two records from a long-term memory store for a coding agent. 'existing' is the " +
    "older record. 'incoming' is the newer record. Does the incoming record make the " +
    "existing one no longer true?",
  criteria: {
    coexist:
      "both records hold today; the newer one may add detail, precision or a separate " +
      "point, but the older one is still true",
    correction: "the older record was wrong when it was written; the newer one corrects it",
    changed:
      "the older record was true when written, but what it describes has since changed; " +
      "the newer one describes the present",
    unclear: UNCLEAR,
  },
};

export const WORTH_QUESTION: QuestionSpec = {
  id: "worth",
  type: "choice",
  instructions:
    "Should this 'text' be written to a long-term memory store for a coding agent? " +
    "When a 'neighbours' list is present it holds records the store already has that sit " +
    "closest to the text.",
  criteria: {
    keep:
      "stands on its own and a reader could not get it cheaply from the code, docs or git " +
      "history: a convention, a gotcha, a defect or absence the code does not show, a " +
      "correction, a preference or an infrastructure detail",
    derivable:
      "a reader could land on it cheaply by reading the code, the docs or the git " +
      "history, or it is a timeline event (task shipped, PR merged, suite green, " +
      "deployed) or session narrative",
    covered: "a record in 'neighbours' already says this; nothing new would be stored",
    unclear: UNCLEAR,
  },
};

export const KIND_QUESTION: QuestionSpec = {
  id: "kind",
  type: "choice",
  instructions:
    "What kind of record is this 'text' in a long-term memory store for a coding agent? " +
    "'declared_kind' is what the writer called it; judge the text itself.",
  criteria: {
    fact: "states how something is, was or behaves; a gotcha, a convention, a detail",
    decision:
      "a choice between live options that was taken, naming what was ruled out: what was " +
      "picked and why",
    defect:
      "names a bug, gap or missing behaviour that was seen and left in place, with where it " +
      "is and how it shows",
    neither:
      "a status update, a plan, a question, a task list, a requirement or design for " +
      "something not yet built, or narrative",
    unclear: UNCLEAR,
  },
};

export const PERISHABLE_QUESTION: QuestionSpec = {
  id: "perishable",
  type: "choice",
  instructions:
    "Will this 'text' still be true and useful once the piece of work it came from is " +
    "finished? It is a candidate record for a long-term memory store for a coding agent.",
  criteria: {
    standing:
      "holds regardless of the current task: a convention, a gotcha, a preference or an " +
      "infrastructure detail",
    status:
      "describes where the work stands right now: what is done, in progress, passing, " +
      "failing or deployed; true today and stale after the work moves on; a defect not " +
      "yet fixed counts as status",
    sequencing:
      "describes the order of steps, what to do next or what blocks what; only meaningful " +
      "while the work is under way",
    unclear: UNCLEAR,
  },
};

export const BREADTH_QUESTION: QuestionSpec = {
  id: "breadth",
  type: "choice",
  instructions:
    "Does this 'text' apply to the owner's work across every project, or only to the " +
    "project named in 'source_scope'? It is a candidate record for a long-term memory " +
    "store for a coding agent, and a record that is owner-wide would be stored where every " +
    "project reads it.",
  criteria: {
    owner_wide:
      "a working preference, a tooling or environment fact, or a rule that holds for the " +
      "owner in every repository, whatever the project",
    project_specific:
      "true of one project, its code, its tickets, its infrastructure or its conventions; " +
      "other projects would not need it",
    unclear: UNCLEAR,
  },
};

export function scopeQuestion(labels: ReadonlyArray<readonly [string, string | null]>): QuestionSpec {
  const criteria: Record<string, string> = {};

  for (const [name, description] of labels) {
    criteria[name] = description || name;
  }

  criteria.unclear = UNCLEAR;

  return {
    id: "scope",
    type: "choice",
    instructions:
      "Which scope should this 'text' be stored under in a long-term memory store for a " +
      "coding agent? Each label names a scope; pick the one the text belongs to.",
    criteria,
  };
}

export const HEADLINE_FAITHFUL_QUESTION: QuestionSpec = {
  id: "headline_faithful",
  type: "noul",
  instructions:
    "A candidate memory record has a 'headline' and a 'claim'. Does the 'headline' say what " +
    "the 'claim' says, without adding anything the 'claim' does not state and without " +
    "leaving out its main point?",
  criteria: null,
};

export const CLAIM_STANDING_QUESTION: QuestionSpec = {
  id: "claim_standing",
  type: "noul",
  instructions:
    "Does the 'claim' read as true without knowing which ticket, round, date, pull request, " +
    "commit or file line it came from? Those references are provenance and do not belong in " +
    "the claim.",
  criteria: null,
};

export const EXCERPT_GROUNDS_QUESTION: QuestionSpec = {
  id: "excerpt_grounds",
  type: "noul",
  instructions:
    "A candidate memory record has a 'claim' and an 'excerpt' quoted from the thread it came " +
    "from. Does the excerpt contain the basis for the claim, such as the owner's words, the " +
    "error text or the observed behaviour, rather than the agent's own summary?",
  criteria: null,
};

export const REJECTED_ALTERNATIVE_QUESTION: QuestionSpec = {
  id: "rejected_names_alternative",
  type: "noul",
  instructions:
    "A candidate memory record is a decision. Does its 'rejected' field name a real " +
    "alternative that was considered and not chosen, as opposed to a vague phrase, a " +
    "restatement of the choice made, or an option nobody would pick?",
  criteria: null,
};

export const PURPOSE_LABELS = [
  "constraint",
  "convention",
  "preference",
  "pitfall",
  "reference",
  "none",
] as const;

export const PURPOSE_QUESTION: QuestionSpec = {
  id: "purpose",
  type: "choice",
  instructions:
    "What job does this 'text' do for a coding agent that reads it later? It is a candidate " +
    "record for a long-term memory store.",
  criteria: {
    constraint: "a limit the agent must stay inside: a rule, a requirement or something forbidden",
    convention: "how work is normally done in this project: a naming, layout or process habit",
    preference: "what the owner likes or dislikes in how work is done or presented",
    pitfall: "a trap that costs time if the agent does not know it: a gotcha, a failure mode or a broken tool",
    reference: "a fact to look up: where something lives, how a system is wired, what a value is",
    none: "none of the above fits",
    unclear: UNCLEAR,
  },
};

export const READER_QUESTION: QuestionSpec = {
  id: "reader",
  type: "choice",
  instructions:
    "In what working situation would a coding agent in this scope meet this 'text' later? " +
    "Pick the one that fits.",
  criteria: {
    lookup:
      "the agent would ask a question this text answers (where something lives, how to run " +
      "or reach something, what a value is, why something is shaped this way) and could not " +
      "get the answer from the tree",
    trap:
      "the agent would not ask for it, but would hit the failure it names while doing " +
      "ordinary work in this scope, and needs it surfaced by proximity",
    ruling:
      "the agent would be about to choose between options and needs to know the choice was " +
      "already made and what was ruled out",
    none:
      "no working situation leads to this text; it belongs in a brief, a ticket, a " +
      "changelog, a pull request or the thread it came from",
    unclear: UNCLEAR,
  },
};

export const AUDIENCE_QUESTION: QuestionSpec = {
  id: "audience",
  type: "choice",
  instructions:
    "Which agent role has to find this 'text' while working? Pick the one that would search " +
    "for it or need it pushed at them.",
  criteria: {
    decider: "someone ruling on a design or a plan with the owner",
    builder: "someone writing or changing code against the project's standards",
    reviewer: "someone checking a diff against its brief and the project's review criteria",
    navigator:
      "someone mapping where things live, how to run and reach them, and what is already broken",
    unclear: UNCLEAR,
  },
};

export function topicQuestion(topics: readonly TopicEntry[]): QuestionSpec {
  const criteria: Record<string, string | null> = {};

  for (const topic of topics) {
    criteria[topic.name] = topic.description;
  }

  if (!("other" in criteria)) {
    criteria.other = "none of the listed topics fits";
  }

  criteria.unclear = UNCLEAR;

  return {
    id: "topic",
    type: "choice",
    instructions:
      "Which topic does this 'text' belong to? It is a candidate record for a long-term " +
      "memory store for a coding agent. Pick the single closest topic.",
    criteria,
  };
}

export const RERANK_QUESTION: QuestionSpec = {
  id: "rerank",
  type: "score",
  instructions:
    "How well does this 'record' answer the 'query'? The record comes from a long-term " +
    "memory store for a coding agent and the query is what the agent is looking for.",
  criteria: [
    "unrelated: the record has nothing to do with the query",
    "related: the record covers the same subject but does not answer the query",
    "partial: the record answers part of the query or answers it indirectly",
    "direct: the record answers the query",
  ],
};

export const FRAMING =
  "'existing' is the older record in a long-term memory store for a coding agent; " +
  "'incoming' is the newer text being evaluated against it.";

export const NEW_COVERS_OLD_QUESTION: QuestionSpec = {
  id: "new_covers_old",
  type: "choice",
  instructions:
    "Does the 'incoming' text state everything that the 'existing' record states? " +
    `${FRAMING} Rewording counts as stating it. Extra detail in the incoming text does not ` +
    "matter; only whether something in the existing record is missing from the incoming text.",
  criteria: {
    yes: "every fact, detail, reason and decision in the existing record also appears in the incoming text",
    no: "the existing record states at least one fact, detail, reason or decision that the incoming text leaves out or changes",
  },
};

export const SAME_SUBJECT_QUESTION: QuestionSpec = {
  id: "same_subject",
  type: "choice",
  instructions:
    "Are the 'existing' and 'incoming' notes both about the same single fact or decision? " +
    `${FRAMING} Sharing a topic is not enough; both must make points about one and the same ` +
    "fact or decision, so that a single note could hold both.",
  criteria: {
    yes: "both notes are about one and the same fact or decision, and one note could hold both",
    no: "the notes are about different facts or decisions, even if they share a topic",
  },
};

function topicName(entry: unknown): string | null {
  if (typeof entry === "string") {
    return entry.trim() === "" ? null : entry.trim();
  }

  if (typeof entry === "object" && entry !== null) {
    const record = entry as Record<string, unknown>;

    for (const key of ["name", "id", "slug", "topic", "label"]) {
      const value = record[key];

      if (typeof value === "string" && value.trim() !== "") {
        return value.trim();
      }
    }
  }

  return null;
}

function topicDescription(entry: unknown): string | null {
  if (typeof entry === "object" && entry !== null) {
    const value = (entry as Record<string, unknown>).description;

    return typeof value === "string" && value.trim() !== "" ? value.trim() : null;
  }

  return null;
}

export function parseTopics(raw: unknown): TopicEntry[] {
  const source =
    typeof raw === "object" && raw !== null && !Array.isArray(raw) && "topics" in raw
      ? (raw as { topics: unknown }).topics
      : raw;

  if (Array.isArray(source)) {
    return source.flatMap((entry) => {
      const name = topicName(entry);

      return name === null ? [] : [{ name, description: topicDescription(entry) }];
    });
  }

  if (typeof source === "object" && source !== null) {
    return Object.entries(source as Record<string, unknown>).map(([name, description]) => ({
      name,
      description: typeof description === "string" && description.trim() !== "" ? description : null,
    }));
  }

  throw new Error("topics file must hold an array or an object");
}

export function loadTopics(path: string | null | undefined): TopicEntry[] {
  if (!path || !existsSync(path)) {
    return [];
  }

  return parseTopics(JSON.parse(readFileSync(path, "utf8")));
}
