import { homedir } from "node:os";
import { join } from "node:path";

import { fsCache, localReader } from "./cache-node.js";
import { DEFAULT_HOST, DEFAULT_REPO, DEFAULT_REVISION, type ShapesEnv, loadModel } from "./hub.js";
import { type ShapesModel } from "./model.js";
import type { Point, Shape } from "./shape.js";

export { createShapes, ShapesModel, type ShapesMeta } from "./model.js";
export { outline, type Point, type Shape, type ShapeKind } from "./shape.js";
export { type PreprocessConfig } from "./preprocess.js";
export { loadModel, type ShapesEnv, type FileCache } from "./hub.js";

/** Loading configuration. Mutate before the first call, or pass overrides to {@link load}. */
export const env: ShapesEnv = {
  host: DEFAULT_HOST,
  repo: DEFAULT_REPO,
  revision: DEFAULT_REVISION,
  allowRemote: true,
  useCache: true,
  cacheDir: process.env.SHAPES_CACHE_DIR ?? join(homedir(), ".cache", "shapes"),
  localModelPath: process.env.SHAPES_LOCAL_PATH,
  token: process.env.HF_TOKEN ?? process.env.HUGGING_FACE_HUB_TOKEN,
};

/** Loads the model: a local dir if configured, else the Hugging Face Hub (cached to disk). */
export async function load(options: Partial<ShapesEnv> = {}): Promise<ShapesModel> {
  const e = { ...env, ...options };
  const cache = e.useCache && e.cacheDir ? fsCache(e.cacheDir) : null;
  const readLocal = e.localModelPath ? localReader(e.localModelPath) : undefined;
  return loadModel(e, cache, readLocal);
}

let modelPromise: Promise<ShapesModel> | null = null;

/**
 * Recognizes a single stroke (ordered `[x, y]` points) as a clean {@link Shape},
 * or `null` if rejected. The model is loaded (and cached) lazily on first call.
 */
export async function recognize(points: Point[]): Promise<Shape | null> {
  if (!modelPromise) {
    modelPromise = load().catch((err) => {
      modelPromise = null;
      throw err;
    });
  }
  return (await modelPromise).recognize(points);
}

/** Clears the memoized model so the next {@link recognize} call re-reads `env`. */
export function reset(): void {
  modelPromise = null;
}
