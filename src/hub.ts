import { createShapes, ShapesModel, type ShapesMeta } from "./model.js";

export const DEFAULT_HOST = "https://huggingface.co";
export const DEFAULT_REPO = "desert-ant-labs/shapes";
/** Pinned revision of the model repo. A tag (not a bare commit SHA) so it
 * survives history rewrites/squashes on the model repo. */
export const DEFAULT_REVISION = "v0.1.0";

const FILES = ["shapes.safetensors", "shapes_meta.json"] as const;

/** Resolution + caching configuration (mutate the exported `env` to change defaults). */
export interface ShapesEnv {
  /** Hugging Face host serving the model repo. */
  host: string;
  /** Model repo id, e.g. `"desert-ant-labs/shapes"`. */
  repo: string;
  /** Pinned revision (commit SHA, tag, or branch). */
  revision: string;
  /** Allow fetching from the Hugging Face Hub. Set `false` to require a local copy. */
  allowRemote: boolean;
  /** Cache downloaded files (filesystem on Node, Cache Storage in the browser). */
  useCache: boolean;
  /** Directory of pre-downloaded model files to use instead of the Hub (Node). */
  localModelPath?: string;
  /** Filesystem cache directory (Node). */
  cacheDir?: string;
  /** Optional Hugging Face access token (Node) — needed while the repo is private. */
  token?: string;
}

/** A key/value store for cached file bytes, keyed by resolve URL. */
export interface FileCache {
  get(key: string): Promise<Uint8Array | null>;
  put(key: string, data: Uint8Array): Promise<void>;
}

function resolveUrl(env: ShapesEnv, name: string): string {
  return `${env.host}/${env.repo}/resolve/${env.revision}/${name}`;
}

async function fetchFile(
  env: ShapesEnv,
  name: string,
  cache: FileCache | null,
  readLocal?: (name: string) => Promise<Uint8Array | null>,
): Promise<Uint8Array> {
  if (readLocal) {
    const local = await readLocal(name);
    if (local) return local;
  }
  const url = resolveUrl(env, name);
  if (cache && env.useCache) {
    const hit = await cache.get(url);
    if (hit) return hit;
  }
  if (!env.allowRemote) throw new Error(`shapes: ${name} unavailable locally and remote loading is disabled`);
  const res = await fetch(url, env.token ? { headers: { Authorization: `Bearer ${env.token}` } } : undefined);
  if (!res.ok) throw new Error(`shapes: failed to fetch ${name} from ${url} (${res.status} ${res.statusText})`);
  const data = new Uint8Array(await res.arrayBuffer());
  if (cache && env.useCache) {
    try {
      await cache.put(url, data);
    } catch {
      /* caching is best-effort */
    }
  }
  return data;
}

/** Resolves all model files (local dir → cache → Hub) and builds a {@link ShapesModel}. */
export async function loadModel(
  env: ShapesEnv,
  cache: FileCache | null,
  readLocal?: (name: string) => Promise<Uint8Array | null>,
): Promise<ShapesModel> {
  const [weights, meta] = await Promise.all(FILES.map((name) => fetchFile(env, name, cache, readLocal)));
  return createShapes({ weights, meta: JSON.parse(new TextDecoder().decode(meta)) as ShapesMeta });
}
