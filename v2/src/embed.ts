import { createHash } from "node:crypto";

export interface Embedder {
  readonly dim: number;
  embed(text: string): Promise<number[]>;
}

const DEFAULT_EMBED_TIMEOUT_MS = 10_000;

export class EmbedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EmbedError";
  }
}

export class OllamaEmbedder implements Embedder {
  readonly dim: number;
  private readonly url: string;
  private readonly model: string;
  private readonly fetchImpl: typeof fetch;
  private readonly timeoutMs: number;

  constructor(
    url: string,
    model: string,
    dim: number,
    fetchImpl: typeof fetch = fetch,
    timeoutMs: number = DEFAULT_EMBED_TIMEOUT_MS,
  ) {
    this.url = url;
    this.model = model;
    this.dim = dim;
    this.fetchImpl = fetchImpl;
    this.timeoutMs = timeoutMs;
  }

  async embed(text: string): Promise<number[]> {
    let response: Response;

    try {
      response = await this.fetchImpl(this.url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: this.model, input: text }),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch (error) {
      throw new EmbedError(`embedding request failed: ${error instanceof Error ? error.message : String(error)}`);
    }

    if (!response.ok) {
      throw new EmbedError(`embedding request failed with status ${response.status}`);
    }

    let body: unknown;

    try {
      body = await response.json();
    } catch {
      throw new EmbedError("embedding response is not valid JSON");
    }

    const embeddings =
      typeof body === "object" && body !== null ? (body as { embeddings?: unknown }).embeddings : undefined;
    const first = Array.isArray(embeddings) ? embeddings[0] : undefined;

    if (!Array.isArray(first) || first.length !== this.dim || !first.every(isFiniteNumber)) {
      throw new EmbedError(`embedding response is not a ${this.dim}-dimension vector of finite numbers`);
    }

    return first;
  }
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export class HashEmbedder implements Embedder {
  readonly dim: number;

  constructor(dim: number) {
    this.dim = dim;
  }

  async embed(text: string): Promise<number[]> {
    const vector = new Array<number>(this.dim).fill(0);
    const tokens = text.toLowerCase().match(/[a-z0-9]+/g) ?? [];

    for (const token of tokens) {
      const digest = createHash("sha256").update(token).digest();
      const bucket = digest.readUInt32BE(0) % this.dim;
      vector[bucket] = (vector[bucket] ?? 0) + 1;
    }

    const norm = Math.hypot(...vector);

    if (norm === 0) {
      vector[0] = 1;
      return vector;
    }

    return vector.map((value) => value / norm);
  }
}

export function cosine(a: readonly number[], b: readonly number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let index = 0; index < a.length; index += 1) {
    const x = a[index] ?? 0;
    const y = b[index] ?? 0;
    dot += x * y;
    normA += x * x;
    normB += y * y;
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dot / Math.sqrt(normA * normB);
}
