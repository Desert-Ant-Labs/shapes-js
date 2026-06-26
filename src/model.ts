import { fit } from "./fit.js";
import { preprocess, type PreprocessConfig } from "./preprocess.js";
import type { Point, Shape } from "./shape.js";

export type { Point, Shape, ShapeKind } from "./shape.js";
export { outline } from "./shape.js";

/** Parsed `shapes_meta.json`: classes, gates, preprocessing constants, dims. */
export interface ShapesMeta {
  classes: string[];
  gates: Record<string, { conf: number; resid: number }>;
  preprocess: PreprocessConfig;
  model: { width: number; heads: number; layers: number; in_channels: number; pool_factor: number };
}

class Tensor {
  readonly length: number;
  constructor(
    readonly shape: number[],
    private readonly floats: Float32Array | null,
    private readonly packed: Uint8Array | null,
    private readonly palette: Float32Array | null,
  ) {
    this.length = shape.reduce((a, b) => a * b, 1);
  }
  get(i: number): number {
    const f = this.floats;
    if (f) return f[i];
    const byte = this.packed![i >> 1];
    return this.palette![(i & 1) ? (byte >> 4) & 0xf : byte & 0xf];
  }
}

// Minimal safetensors reader: u64 little-endian header length, JSON header, then
// raw F32 tensors or packed 4-bit k-means palette tensors (`name` U8 indices +
// `name.palette` F32 centroids, with logical shape in __metadata__).
function parseSafetensors(bytes: Uint8Array): Map<string, Tensor> {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headerLen)));
  const dataStart = 8 + headerLen;
  const meta = header.__metadata__ ?? {};
  const out = new Map<string, Tensor>();
  const slice = (name: string): { shape: number[]; bytes: Uint8Array } => {
    const e = header[name];
    if (!e) throw new Error(`shapes: missing tensor ${name}`);
    const [a, b] = e.data_offsets;
    const buf = bytes.slice(dataStart + a, dataStart + b);
    return { shape: e.shape, bytes: buf };
  };
  const f32 = (name: string): Float32Array => {
    const s = slice(name);
    return new Float32Array(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength / 4);
  };
  for (const name of Object.keys(header)) {
    if (name === "__metadata__" || name.endsWith(".palette")) continue;
    if (header[name + ".palette"]) {
      const shape = String(meta["shape." + name]).split(",").map(Number);
      out.set(name, new Tensor(shape, null, slice(name).bytes, f32(name + ".palette")));
    } else {
      const s = slice(name);
      out.set(name, new Tensor(s.shape, new Float32Array(s.bytes.buffer, s.bytes.byteOffset, s.bytes.byteLength / 4), null, null));
    }
  }
  return out;
}

function erf(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-x * x));
  return x >= 0 ? y : -y;
}
const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2));

function layerNorm(v: Float32Array, w: Tensor, b: Tensor): Float32Array {
  const n = v.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += v[i];
  mean /= n;
  let varr = 0;
  for (let i = 0; i < n; i++) varr += (v[i] - mean) * (v[i] - mean);
  varr /= n;
  const inv = 1 / Math.sqrt(varr + 1e-5);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (v[i] - mean) * inv * w.get(i) + b.get(i);
  return out;
}

/** Loads the portable model and runs stroke recognition. Construct once and reuse. */
export class ShapesModel {
  private readonly t: Map<string, Tensor>;
  private readonly D: number;
  private readonly H: number;
  private readonly dh: number;
  private readonly scale: number;
  private readonly layers: number;

  constructor(weights: Uint8Array, readonly meta: ShapesMeta) {
    this.t = parseSafetensors(weights);
    this.D = meta.model.width;
    this.H = meta.model.heads;
    this.dh = this.D / this.H;
    this.scale = 1 / Math.sqrt(this.dh);
    this.layers = meta.model.layers;
  }

  private get(name: string): Tensor {
    const t = this.t.get(name);
    if (!t) throw new Error(`shapes: missing tensor ${name}`);
    return t;
  }

  /**
   * Recognize a single stroke (ordered `[x, y]` points). Returns the snapped
   * {@link Shape}, or `null` when the stroke is rejected or degenerate.
   */
  recognize(points: Point[]): Shape | null {
    const feats = preprocess(points, this.meta.preprocess);
    if (!feats) return null;
    const probs = this.classify(feats);
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    const kind = this.meta.classes[best];
    if (kind === "none") return null;
    const gate = this.meta.gates[kind];
    if (gate && probs[best] < gate.conf) return null;
    const { shape, residual } = fit(kind as Shape["type"], points);
    if (gate && residual > gate.resid) return null;
    return shape;
  }

  // Conv1d stem -> Transformer encoder -> mean-pool -> MLP head -> softmax.
  private classify(feats: number[][]): Float32Array {
    const N = feats.length;
    // channel-major input [in_channels][N]
    let h: Float32Array[] = [];
    const cin = this.meta.model.in_channels;
    for (let c = 0; c < cin; c++) {
      const row = new Float32Array(N);
      for (let t = 0; t < N; t++) row[t] = feats[t][c];
      h.push(row);
    }
    h = this.pool(this.relu(this.conv1d(h, "stem.conv1", cin, 32)));
    h = this.pool(this.relu(this.conv1d(h, "stem.conv2", 32, 64)));
    h = this.relu(this.conv1d(h, "stem.conv3", 64, 128));

    const T = h[0].length;
    const pe = this.get("encoder.pe");
    // tokens [T][D]
    const tok: Float32Array[] = [];
    for (let t = 0; t < T; t++) {
      const row = new Float32Array(this.D);
      for (let d = 0; d < this.D; d++) row[d] = h[d][t] + pe.get(t * this.D + d);
      tok.push(row);
    }
    let layer = tok;
    for (let l = 0; l < this.layers; l++) layer = this.encoderLayer(layer, l);

    const pooled = new Float32Array(this.D);
    for (const row of layer) for (let d = 0; d < this.D; d++) pooled[d] += row[d];
    for (let d = 0; d < this.D; d++) pooled[d] /= layer.length;

    const h0 = this.linear(pooled, "head.0", 64, this.D);
    for (let i = 0; i < 64; i++) h0[i] = Math.max(0, h0[i]);
    const logits = this.linear(h0, "head.3", this.meta.classes.length, 64);
    return softmax(logits);
  }

  private conv1d(input: Float32Array[], name: string, cin: number, cout: number): Float32Array[] {
    const W = this.get(name + ".weight");
    const B = this.get(name + ".bias");
    const T = input[0].length;
    const out: Float32Array[] = [];
    for (let o = 0; o < cout; o++) {
      const row = new Float32Array(T);
      const wbase = o * cin * 3;
      for (let t = 0; t < T; t++) {
        let acc = B.get(o);
        for (let c = 0; c < cin; c++) {
          const ib = input[c];
          const wb = wbase + c * 3;
          if (t - 1 >= 0) acc += W.get(wb) * ib[t - 1];
          acc += W.get(wb + 1) * ib[t];
          if (t + 1 < T) acc += W.get(wb + 2) * ib[t + 1];
        }
        row[t] = acc;
      }
      out.push(row);
    }
    return out;
  }

  private relu(x: Float32Array[]): Float32Array[] {
    for (const row of x) for (let i = 0; i < row.length; i++) if (row[i] < 0) row[i] = 0;
    return x;
  }

  private pool(x: Float32Array[]): Float32Array[] {
    const T2 = Math.floor(x[0].length / 2);
    return x.map((row) => {
      const out = new Float32Array(T2);
      for (let i = 0; i < T2; i++) out[i] = Math.max(row[2 * i], row[2 * i + 1]);
      return out;
    });
  }

  private linear(x: Float32Array, name: string, out: number, inDim: number): Float32Array {
    const W = this.get(name + ".weight");
    const B = this.get(name + ".bias");
    const y = new Float32Array(out);
    for (let o = 0; o < out; o++) {
      let acc = B.get(o);
      const base = o * inDim;
      for (let i = 0; i < inDim; i++) acc += W.get(base + i) * x[i];
      y[o] = acc;
    }
    return y;
  }

  private encoderLayer(tok: Float32Array[], l: number): Float32Array[] {
    const p = `encoder.enc.layers.${l}.`;
    const inW = this.get(p + "self_attn.in_proj_weight"); // [3D, D]
    const inB = this.get(p + "self_attn.in_proj_bias");
    const T = tok.length;
    const D = this.D, H = this.H, dh = this.dh;

    // q/k/v: [T][3D]
    const q: Float32Array[] = [], k: Float32Array[] = [], v: Float32Array[] = [];
    for (let t = 0; t < T; t++) {
      const qkv = new Float32Array(3 * D);
      for (let o = 0; o < 3 * D; o++) {
        let acc = inB.get(o);
        const base = o * D;
        for (let i = 0; i < D; i++) acc += inW.get(base + i) * tok[t][i];
        qkv[o] = acc;
      }
      q.push(qkv.subarray(0, D));
      k.push(qkv.subarray(D, 2 * D));
      v.push(qkv.subarray(2 * D, 3 * D));
    }

    // per-head scaled dot-product attention
    const attn: Float32Array[] = [];
    for (let t = 0; t < T; t++) attn.push(new Float32Array(D));
    for (let head = 0; head < H; head++) {
      const off = head * dh;
      for (let i = 0; i < T; i++) {
        const scores = new Float32Array(T);
        let mx = -Infinity;
        for (let j = 0; j < T; j++) {
          let s = 0;
          for (let d = 0; d < dh; d++) s += q[i][off + d] * k[j][off + d];
          s *= this.scale;
          scores[j] = s;
          if (s > mx) mx = s;
        }
        let sum = 0;
        for (let j = 0; j < T; j++) {
          scores[j] = Math.exp(scores[j] - mx);
          sum += scores[j];
        }
        const ai = attn[i];
        for (let j = 0; j < T; j++) {
          const a = scores[j] / sum;
          for (let d = 0; d < dh; d++) ai[off + d] += a * v[j][off + d];
        }
      }
    }

    // out projection + residual + norm1
    const outW = this.get(p + "self_attn.out_proj.weight");
    const outB = this.get(p + "self_attn.out_proj.bias");
    const n1w = this.get(p + "norm1.weight"), n1b = this.get(p + "norm1.bias");
    const x1: Float32Array[] = [];
    for (let t = 0; t < T; t++) {
      const a = new Float32Array(D);
      for (let o = 0; o < D; o++) {
        let acc = outB.get(o);
        const base = o * D;
        for (let i = 0; i < D; i++) acc += outW.get(base + i) * attn[t][i];
        a[o] = acc + tok[t][o];
      }
      x1.push(layerNorm(a, n1w, n1b));
    }

    // FFN + residual + norm2
    const l1w = this.get(p + "linear1.weight"), l1b = this.get(p + "linear1.bias");
    const l2w = this.get(p + "linear2.weight"), l2b = this.get(p + "linear2.bias");
    const ffn = l1b.length;
    const n2w = this.get(p + "norm2.weight"), n2b = this.get(p + "norm2.bias");
    const out: Float32Array[] = [];
    for (let t = 0; t < T; t++) {
      const hidden = new Float32Array(ffn);
      for (let o = 0; o < ffn; o++) {
        let acc = l1b.get(o);
        const base = o * D;
        for (let i = 0; i < D; i++) acc += l1w.get(base + i) * x1[t][i];
        hidden[o] = gelu(acc);
      }
      const y = new Float32Array(D);
      for (let o = 0; o < D; o++) {
        let acc = l2b.get(o);
        const base = o * ffn;
        for (let i = 0; i < ffn; i++) acc += l2w.get(base + i) * hidden[i];
        y[o] = acc + x1[t][o];
      }
      out.push(layerNorm(y, n2w, n2b));
    }
    return out;
  }
}

function softmax(logits: Float32Array): Float32Array {
  let mx = -Infinity;
  for (const x of logits) if (x > mx) mx = x;
  let sum = 0;
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] - mx);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

/** Creates a {@link ShapesModel} from raw buffers (lowest-level entry). */
export function createShapes(buffers: { weights: Uint8Array; meta: ShapesMeta }): ShapesModel {
  return new ShapesModel(buffers.weights, buffers.meta);
}
