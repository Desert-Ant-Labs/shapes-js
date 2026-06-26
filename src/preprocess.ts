import type { Point } from "./shape.js";

/** Frozen preprocessing constants (from the model's `shapes_meta.json`). */
export interface PreprocessConfig {
  spacing: number;
  dist_mean: number;
  dist_std: number;
  add_curvature: boolean;
}

const MIN_POINTS = 2;
const MIN_TOTAL_LENGTH = 1e-6;
const DEDUPE_EPSILON = 1e-9;

/**
 * Port of the shared `preprocess.py`: dedupe → normalize → arc-length resample →
 * `[dist, cos, sin]` features. Returns the `N × C` feature rows, or `null` for a
 * degenerate stroke (too few points / too small).
 */
export function preprocess(points: Point[], cfg: PreprocessConfig): number[][] | null {
  const cleaned = dedupe(points, DEDUPE_EPSILON);
  if (cleaned.length < MIN_POINTS || totalLength(cleaned) < MIN_TOTAL_LENGTH) return null;
  const normalized = normalize(cleaned);
  const resampled = resample(normalized, cfg.spacing);
  return features(resampled, cfg.dist_mean, cfg.dist_std);
}

function dedupe(points: Point[], epsilon: number): Point[] {
  if (points.length === 0) return [];
  const eps2 = epsilon * epsilon;
  const out: Point[] = [[points[0][0], points[0][1]]];
  for (let i = 1; i < points.length; i++) {
    const [px, py] = out[out.length - 1];
    const dx = points[i][0] - px;
    const dy = points[i][1] - py;
    if (dx * dx + dy * dy > eps2) out.push([points[i][0], points[i][1]]);
  }
  return out;
}

function totalLength(p: Point[]): number {
  let total = 0;
  for (let i = 1; i < p.length; i++) total += Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]);
  return total;
}

function normalize(p: Point[]): Point[] {
  let minX = p[0][0], maxX = p[0][0], minY = p[0][1], maxY = p[0][1];
  for (const [x, y] of p) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const longer = Math.max(maxX - minX, maxY - minY);
  const scale = longer > 0 ? 1 / longer : 1;
  return p.map(([x, y]) => [(x - cx) * scale, (y - cy) * scale] as Point);
}

function resample(p: Point[], spacing: number): Point[] {
  const n = p.length;
  if (n <= 1) return p.map(([x, y]) => [x, y] as Point);
  const out: Point[] = [[p[0][0], p[0][1]]];
  let [prevX, prevY] = p[0];
  let since = 0;
  let i = 1;
  while (i < n) {
    const dx = p[i][0] - prevX;
    const dy = p[i][1] - prevY;
    const segLen = Math.hypot(dx, dy);
    if (segLen <= 0) {
      prevX = p[i][0];
      prevY = p[i][1];
      i++;
      continue;
    }
    const needed = spacing - since;
    if (needed <= segLen) {
      const t = needed / segLen;
      prevX += dx * t;
      prevY += dy * t;
      out.push([prevX, prevY]);
      since = 0;
    } else {
      since += segLen;
      prevX = p[i][0];
      prevY = p[i][1];
      i++;
    }
  }
  const last = p[n - 1];
  const [rx, ry] = out[out.length - 1];
  if ((last[0] - rx) ** 2 + (last[1] - ry) ** 2 > 1e-18) out.push([last[0], last[1]]);
  return out;
}

function features(p: Point[], distMean: number, distStd: number): number[][] {
  const std = distStd > 0 ? distStd : 1;
  const out: number[][] = [];
  for (let i = 0; i < p.length; i++) {
    let dist = 0, cos = 0, sin = 0;
    if (i > 0) {
      const dx = p[i][0] - p[i - 1][0];
      const dy = p[i][1] - p[i - 1][1];
      dist = Math.hypot(dx, dy);
      if (dist > 0) {
        cos = dx / dist;
        sin = dy / dist;
      }
    }
    out.push([(dist - distMean) / std, cos, sin]);
  }
  return out;
}
