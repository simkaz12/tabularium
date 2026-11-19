export const MD_BEGIN = "<!-- tabumcp:begin -->";
export const MD_END = "<!-- tabumcp:end -->";
export const TOML_BEGIN = "# tabumcp:begin";
export const TOML_END = "# tabumcp:end";

export const INSTRUCTION_LINES = [
  "## TabuMCP memory",
  "",
  "- Pass your working folder as `cwd` on every memory call.",
  "- Call `memory_recall` before work in an area.",
  "- Leave `scope` empty unless the user names one. It only reaches scopes in this project's profile.",
  "- If recall already returned the same rule or decision, don't propose it again. If it changed, propose the new one with `supersedes`. Use `supersedes` only when the new note replaces the old note's claim, never for a different claim about the same file or area.",
  "- Propose a gotcha as a `fact` when you hit it: say what failed and what fixed it.",
  "- Propose a `decision` in the same turn the user rules, before you start the work, with `rejected` naming the alternative.",
  "- When you notice a bug you are not fixing, even outside your task (for example the user said \"fix X only\"), propose it as a `defect` before you finish: where it is, what breaks, what triggers it.",
  "- In your final reply, briefly mention any defect or decision you saved to memory.",
  "- Answer an `asked` result with `memory_answer` in the same turn.",
];

export function instructionBlock(): string {
  return [MD_BEGIN, ...INSTRUCTION_LINES, MD_END].join("\n");
}

function findBlock(text: string, begin: string, end: string): { start: number; stop: number } | null {
  const start = text.indexOf(begin);

  if (start === -1) {
    return null;
  }

  const endAt = text.indexOf(end, start);

  if (endAt === -1) {
    return null;
  }

  return { start, stop: endAt + end.length };
}

export function hasBlock(text: string, begin: string, end: string): boolean {
  return findBlock(text, begin, end) !== null;
}

export function upsertBlock(text: string, begin: string, end: string, block: string): string {
  const found = findBlock(text, begin, end);

  if (found !== null) {
    return text.slice(0, found.start) + block + text.slice(found.stop);
  }

  if (text === "") {
    return `${block}\n`;
  }

  return `${text}\n${block}\n`;
}

export function removeBlock(text: string, begin: string, end: string): string {
  const found = findBlock(text, begin, end);

  if (found === null) {
    return text;
  }

  let before = text.slice(0, found.start);
  let after = text.slice(found.stop);

  if (after.startsWith("\n")) {
    after = after.slice(1);
  }

  if (before.endsWith("\n")) {
    before = before.slice(0, -1);
  }

  return before + after;
}

export function outsideBlock(text: string, begin: string, end: string): string {
  const found = findBlock(text, begin, end);

  return found === null ? text : text.slice(0, found.start) + text.slice(found.stop);
}
