import { initUsage, type UsageClient } from "@desert-ant-labs/desert-ant-web";

import { webCache } from "./cache-web.js";
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
};

let usage: UsageClient | null = null;

function instrument(model: ShapesModel, usageKey?: string): ShapesModel {
  const recognize = model.recognize.bind(model);
  model.recognize = (points) => {
    usage ??= initUsage({ key: usageKey });
    const shape = recognize(points);
    usage.recordCall();
    return shape;
  };
  return model;
}

/** Loads the model from the Hugging Face Hub (cached in Cache Storage). */
export async function load(options: Partial<ShapesEnv> & { usageKey?: string } = {}): Promise<ShapesModel> {
  const e = { ...env, ...options };
  return instrument(await loadModel(e, e.useCache ? webCache() : null), options.usageKey);
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
  // The model from load() is already instrumented, so this counts exactly once.
  return (await modelPromise).recognize(points);
}

/** Clears the memoized model so the next {@link recognize} call re-reads `env`. */
export function reset(): void {
  modelPromise = null;
}
