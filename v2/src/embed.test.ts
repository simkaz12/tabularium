import { createServer } from "node:http";
import type { AddressInfo, Socket } from "node:net";

import { describe, expect, it } from "vitest";

import { EmbedError, OllamaEmbedder } from "./embed.ts";

function embedderReturning(body: unknown, status = 200): OllamaEmbedder {
  const fetchImpl = (async () =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json" },
    })) as unknown as typeof fetch;

  return new OllamaEmbedder("http://embed.test/api/embed", "model", 3, fetchImpl);
}

describe("OllamaEmbedder", () => {
  it("returns the first embedding", async () => {
    const embedder = embedderReturning({ embeddings: [[1, 2, 3]] });

    expect(await embedder.embed("x")).toEqual([1, 2, 3]);
  });

  it.each([
    ["null", null],
    ["a string", "text"],
    ["a number", 7],
    ["an array", []],
    ["an empty object", {}],
    ["a non-array embeddings field", { embeddings: "nope" }],
    ["an empty embeddings list", { embeddings: [] }],
    ["a wrong-sized vector", { embeddings: [[1, 2]] }],
    ["an object element", { embeddings: [[1, 2, { valueOf: 0, toString: 0 }]] }],
    ["a string element", { embeddings: [[1, 2, "abc"]] }],
    ["a numeric string element", { embeddings: [[1, 2, "3"]] }],
    ["a null element", { embeddings: [[1, 2, null]] }],
    ["a boolean element", { embeddings: [[1, 2, true]] }],
    ["an array element", { embeddings: [[1, 2, [3]]] }],
  ])("throws EmbedError for %s", async (_name, body) => {
    await expect(embedderReturning(body).embed("x")).rejects.toBeInstanceOf(EmbedError);
  });

  it("throws EmbedError for a non-2xx status and for invalid JSON", async () => {
    await expect(embedderReturning({}, 500).embed("x")).rejects.toBeInstanceOf(EmbedError);

    const broken = (async () => new Response("not json", { status: 200 })) as unknown as typeof fetch;

    await expect(new OllamaEmbedder("http://embed.test", "model", 3, broken).embed("x")).rejects.toBeInstanceOf(
      EmbedError,
    );
  });

  it("throws EmbedError when a body element overflows to Infinity", async () => {
    const fetchImpl = (async () =>
      new Response('{"embeddings":[[1,2,1e400]]}', {
        status: 200,
        headers: { "content-type": "application/json" },
      })) as unknown as typeof fetch;

    await expect(new OllamaEmbedder("http://embed.test", "model", 3, fetchImpl).embed("x")).rejects.toBeInstanceOf(
      EmbedError,
    );
  });

  it("throws EmbedError when the server sends headers and then stalls the body", async () => {
    const sockets = new Set<Socket>();
    const server = createServer((_request, response) => {
      response.writeHead(200, { "content-type": "application/json" });
      response.write('{"embeddings":[[1,2');
    });
    server.on("connection", (socket) => {
      sockets.add(socket);
      socket.on("close", () => sockets.delete(socket));
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const { port } = server.address() as AddressInfo;
    const embedder = new OllamaEmbedder(`http://127.0.0.1:${port}/api/embed`, "model", 3, fetch, 200);
    const started = Date.now();

    try {
      await expect(embedder.embed("x")).rejects.toBeInstanceOf(EmbedError);
      expect(Date.now() - started).toBeLessThan(1000);
    } finally {
      for (const socket of sockets) {
        socket.destroy();
      }

      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });

  it("throws EmbedError when the server never answers within the timeout", async () => {
    const sockets = new Set<Socket>();
    const server = createServer(() => undefined);
    server.on("connection", (socket) => {
      sockets.add(socket);
      socket.on("close", () => sockets.delete(socket));
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const { port } = server.address() as AddressInfo;
    const embedder = new OllamaEmbedder(`http://127.0.0.1:${port}/api/embed`, "model", 3, fetch, 200);
    const started = Date.now();

    try {
      await expect(embedder.embed("x")).rejects.toBeInstanceOf(EmbedError);
      expect(Date.now() - started).toBeLessThan(1000);
    } finally {
      for (const socket of sockets) {
        socket.destroy();
      }

      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });
});
