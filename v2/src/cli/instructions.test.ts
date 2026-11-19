import { describe, expect, it } from "vitest";

import { instructionBlock } from "./instructions.ts";

describe("instruction block", () => {
  it("stays short and tells agents to save a left defect, avoid duplicates and leave scope empty", () => {
    const block = instructionBlock();

    expect(block.split("\n").length).toBeLessThanOrEqual(18);
    expect(block).toContain("propose it as a `defect` before you finish");
    expect(block).toContain("don't propose it again");
    expect(block).toContain("`supersedes`");
    expect(block).toContain("never for a different claim about the same file or area");
    expect(block).toContain("In your final reply, briefly mention any defect or decision you saved");
    expect(block).toContain("Leave `scope` empty unless the user names one");
    expect(block).toContain("same turn the user rules, before you start the work");
    expect(block).toContain("`cwd`");
    expect(block).toContain("`memory_answer` in the same turn");
  });
});
