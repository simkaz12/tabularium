export interface EmbedderChoice {
  name: string;
  dim: number;
  note: string;
}

export const DEFAULT_EMBED_MODEL = "nomic-embed-text";
export const DEFAULT_EMBED_DIM = 768;

const TUNED = "works, but duplicate-catching is tuned for nomic-embed-text";

export const EMBEDDERS: EmbedderChoice[] = [
  { name: DEFAULT_EMBED_MODEL, dim: 768, note: "recommended; duplicate-catching is tuned for it" },
  { name: "mxbai-embed-large", dim: 1024, note: TUNED },
  { name: "bge-m3", dim: 1024, note: `many languages; ${TUNED}` },
];

export function knownDim(model: string): number | null {
  return EMBEDDERS.find((choice) => choice.name === model)?.dim ?? null;
}
