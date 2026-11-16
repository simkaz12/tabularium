import { describe, expect, it } from "vitest";

import type { JevClient, JevRequest } from "../src/domain.ts";
import { countingJev } from "./engine.ts";

describe("counting Jev wrapper", () => {
  it("passes the request options through to the inner client", async () => {
    const seen: Array<unknown> = [];
    const inner: JevClient = {
      async systemOne(_request, options) {
        seen.push(options);

        return { model: "m", answers: {} };
      },
    };
    const counting = countingJev(inner);
    const options = { timeout: 1234 };

    await counting.systemOne({} as JevRequest, options);

    expect(seen).toEqual([options]);
    expect(seen[0]).toBe(options);
    expect(counting.calls).toBe(1);
  });

  it("counts a rejected call as a failure and rethrows", async () => {
    const inner: JevClient = {
      async systemOne() {
        throw new Error("down");
      },
    };
    const counting = countingJev(inner);

    await expect(counting.systemOne({} as JevRequest)).rejects.toThrow("down");
    expect(counting.failures).toBe(1);
  });
});
