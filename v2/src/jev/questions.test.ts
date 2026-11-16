import { describe, expect, it } from "vitest";

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
  SAME_SUBJECT_QUESTION,
  WORTH_QUESTION,
  parseTopics,
  pyJsonDumps,
  scopeQuestion,
  topicQuestion,
  wordingHash,
} from "./questions.ts";

describe("wording hash", () => {
  it("keeps the v1 python hash for the relevance question", () => {
    expect(wordingHash(RELEVANCE_QUESTION)).toBe("803c0d1375c824da");
  });

  it("pins the wording of the choice questions that gained an unclear answer", () => {
    expect(wordingHash(RELATION_QUESTION)).toBe("63d30278a5301f37");
    expect(wordingHash(REASON_QUESTION)).toBe("5a464702df06c24a");
    expect(wordingHash(CONTRADICTION_QUESTION)).toBe("0ca7a23f1ff0a773");
    expect(wordingHash(WORTH_QUESTION)).toBe("7ec90dcbddc639d9");
    expect(wordingHash(KIND_QUESTION)).toBe("568b3e131db235f7");
    expect(wordingHash(PERISHABLE_QUESTION)).toBe("50e90fd57ccb5621");
    expect(wordingHash(BREADTH_QUESTION)).toBe("c5348ed2e237f911");
    expect(wordingHash(PURPOSE_QUESTION)).toBe("a76ada9126f20461");
    expect(wordingHash(scopeQuestion([["project-a", "Project-a project"], ["global", null]]))).toBe("a068e4fd6b4d69d5");
  });

  it("pins the reader gate wording", () => {
    expect(wordingHash(READER_QUESTION)).toBe("7b560e9845051046");
    expect(wordingHash(AUDIENCE_QUESTION)).toBe("2d7e414151387355");
    expect(wordingHash(HEADLINE_FAITHFUL_QUESTION)).toBe("32ad85f5fa5d36fb");
    expect(wordingHash(CLAIM_STANDING_QUESTION)).toBe("e95db5f42c56539d");
    expect(wordingHash(EXCERPT_GROUNDS_QUESTION)).toBe("99cae1effda50d96");
  });

  it("keeps the wording hashes of the pair questions the bench measured", () => {
    expect(wordingHash(NEW_COVERS_OLD_QUESTION)).toBe("4e0280a9a3f87ab8");
    expect(wordingHash(SAME_SUBJECT_QUESTION)).toBe("9554e5a25adcc29b");
  });

  it("gives every choice question an unclear answer", () => {
    const choices = [
      RELATION_QUESTION,
      REASON_QUESTION,
      CONTRADICTION_QUESTION,
      WORTH_QUESTION,
      KIND_QUESTION,
      PERISHABLE_QUESTION,
      BREADTH_QUESTION,
      PURPOSE_QUESTION,
      READER_QUESTION,
      AUDIENCE_QUESTION,
      scopeQuestion([["project-a", null]]),
      topicQuestion([{ name: "a", description: null }]),
    ];

    for (const question of choices) {
      expect(Object.keys(question.criteria as object)).toContain("unclear");
    }
  });

  it("defines both records in the contradiction instructions and drops the panel", () => {
    expect(CONTRADICTION_QUESTION.instructions).toContain("'existing'");
    expect(CONTRADICTION_QUESTION.instructions).toContain("'incoming'");
    expect(CONTRADICTION_QUESTION.instructions).not.toContain("panel");
  });

  it("dumps json like python with sorted keys and ascii escapes", () => {
    expect(pyJsonDumps({ b: null, a: ["x", "café"] })).toBe('{"a": ["x", "caf\\u00e9"], "b": null}');
  });
});

describe("topics", () => {
  it("accepts strings, objects and maps and appends other", () => {
    expect(parseTopics(["a", "b"]).map((topic) => topic.name)).toEqual(["a", "b"]);
    expect(parseTopics([{ id: "x", description: "the x" }])).toEqual([{ name: "x", description: "the x" }]);
    expect(parseTopics({ topics: { y: "the y" } })).toEqual([{ name: "y", description: "the y" }]);

    const question = topicQuestion([{ name: "a", description: null }]);
    expect(Object.keys(question.criteria as object)).toEqual(["a", "other", "unclear"]);
  });

  it("keeps an existing other entry", () => {
    const question = topicQuestion([{ name: "other", description: "misc" }]);

    expect(question.criteria).toMatchObject({ other: "misc" });
  });
});
