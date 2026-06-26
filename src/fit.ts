import type { Point, Shape, ShapeKind } from "./shape.js";

// Stage-2 geometric fitters + "smart shape" snapping (port of the Swift Fitter /
// Snapping). Each fitter returns a clean Shape and a normalized fit residual
// (RMS point-to-shape distance / bbox diagonal) used by the recognizer's gate.

type V = [number, number];

const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1]];
const add = (a: V, b: V): V => [a[0] + b[0], a[1] + b[1]];
const mul = (a: V, s: number): V => [a[0] * s, a[1] * s];
const dot = (a: V, b: V): number => a[0] * b[0] + a[1] * b[1];
const len = (a: V): number => Math.hypot(a[0], a[1]);
const rot = (v: V, ang: number): V => {
  const c = Math.cos(ang), s = Math.sin(ang);
  return [c * v[0] - s * v[1], s * v[0] + c * v[1]];
};

function centroid(p: V[]): V {
  let x = 0, y = 0;
  for (const q of p) {
    x += q[0];
    y += q[1];
  }
  return [x / p.length, y / p.length];
}

/** Larger/smaller eigenvalue + eigenvectors of the symmetric [[a,b],[b,c]]. */
function symEig(a: number, b: number, c: number): { v1: V; v2: V } {
  const tr = a + c;
  const disc = Math.hypot((a - c) / 2, b);
  const l1 = tr / 2 + disc;
  const vec = (l: number): V => {
    const v: V = Math.abs(b) > 1e-15 ? [l - c, b] : a >= c ? [1, 0] : [0, 1];
    const n = len(v);
    return n > 0 ? [v[0] / n, v[1] / n] : [1, 0];
  };
  const v1 = vec(l1);
  return { v1, v2: [-v1[1], v1[0]] };
}

function resampleUniform(p: V[], count: number): V[] {
  if (p.length <= 2) return p;
  const cum = [0];
  for (let i = 1; i < p.length; i++) cum.push(cum[i - 1] + len(sub(p[i], p[i - 1])));
  const total = cum[cum.length - 1];
  if (total <= 0) return p;
  const out: V[] = [];
  let j = 0;
  for (let i = 0; i < count; i++) {
    const target = (total * i) / (count - 1);
    while (j < p.length - 2 && cum[j + 1] < target) j++;
    const seg = cum[j + 1] - cum[j];
    const t = seg > 0 ? (target - cum[j]) / seg : 0;
    out.push(add(p[j], mul(sub(p[j + 1], p[j]), t)));
  }
  return out;
}

function bboxDiag(p: V[]): number {
  let lo: V = [p[0][0], p[0][1]];
  let hi: V = [p[0][0], p[0][1]];
  for (const q of p) {
    lo = [Math.min(lo[0], q[0]), Math.min(lo[1], q[1])];
    hi = [Math.max(hi[0], q[0]), Math.max(hi[1], q[1])];
  }
  const d = len(sub(hi, lo));
  return d > 0 ? d : 1;
}

function ptSegDist(q: V, a: V, b: V): number {
  const ab = sub(b, a);
  const l2 = dot(ab, ab);
  if (l2 === 0) return len(sub(q, a));
  const t = Math.max(0, Math.min(1, dot(sub(q, a), ab) / l2));
  return len(sub(q, add(a, mul(ab, t))));
}

function residual(stroke: V[], poly: V[], closed: boolean): number {
  const ring = closed ? [...poly, poly[0]] : poly;
  let sumSq = 0;
  for (const q of stroke) {
    let best = Infinity;
    for (let i = 0; i < ring.length - 1; i++) best = Math.min(best, ptSegDist(q, ring[i], ring[i + 1]));
    sumSq += best * best;
  }
  return Math.sqrt(sumSq / stroke.length) / bboxDiag(stroke);
}

function convexHull(points: V[]): V[] {
  const seen = new Set<string>();
  const pts: V[] = [];
  for (const p of points) {
    const key = `${Math.round(p[0] * 1e6)},${Math.round(p[1] * 1e6)}`;
    if (!seen.has(key)) {
      seen.add(key);
      pts.push([p[0], p[1]]);
    }
  }
  if (pts.length <= 2) return pts;
  pts.sort((a, b) => (a[0] === b[0] ? a[1] - b[1] : a[0] - b[0]));
  const cross = (o: V, a: V, b: V) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: V[] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper: V[] = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

// ---------------------------------------------------------------------------
// Per-kind fitters
// ---------------------------------------------------------------------------

function fitLine(pts: V[]): { shape: Shape; residual: number } {
  const c = centroid(pts);
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of pts) {
    const d = sub(p, c);
    sxx += d[0] * d[0];
    sxy += d[0] * d[1];
    syy += d[1] * d[1];
  }
  const dir = symEig(sxx, sxy, syy).v1;
  let lo = Infinity, hi = -Infinity;
  for (const p of pts) {
    const t = dot(sub(p, c), dir);
    lo = Math.min(lo, t);
    hi = Math.max(hi, t);
  }
  const a = add(c, mul(dir, lo));
  const b = add(c, mul(dir, hi));
  return { shape: { type: "line", from: a, to: b }, residual: residual(pts, [a, b], false) };
}

function fitRectangle(pts: V[]): { shape: Shape; residual: number } {
  const hull = convexHull(pts);
  if (hull.length < 3) return fitLine(pts);
  let best: { area: number; corners: V[] } | null = null;
  for (let i = 0; i < hull.length; i++) {
    const edge = sub(hull[(i + 1) % hull.length], hull[i]);
    const ang = Math.atan2(edge[1], edge[0]);
    let loX = Infinity, loY = Infinity, hiX = -Infinity, hiY = -Infinity;
    for (const h of hull) {
      const r = rot(h, -ang);
      loX = Math.min(loX, r[0]);
      loY = Math.min(loY, r[1]);
      hiX = Math.max(hiX, r[0]);
      hiY = Math.max(hiY, r[1]);
    }
    const area = (hiX - loX) * (hiY - loY);
    if (!best || area < best.area) {
      const cr: V[] = [[loX, loY], [hiX, loY], [hiX, hiY], [loX, hiY]];
      best = { area, corners: cr.map((p) => rot(p, ang)) };
    }
  }
  const corners = best!.corners;
  return { shape: { type: "rectangle", corners }, residual: residual(pts, corners, true) };
}

function fitTriangle(pts: V[]): { shape: Shape; residual: number } {
  let hull = convexHull(pts);
  if (hull.length < 3) return fitLine(pts);
  if (hull.length > 36) {
    const idx = Array.from({ length: 36 }, (_, i) => Math.round((i * (hull.length - 1)) / 35));
    const seen = new Set<number>();
    hull = idx.filter((k) => (seen.has(k) ? false : (seen.add(k), true))).map((k) => hull[k]);
  }
  let bestArea = -1;
  let tri: V[] = [hull[0], hull[1], hull[2]];
  for (let i = 0; i < hull.length; i++)
    for (let j = i + 1; j < hull.length; j++)
      for (let k = j + 1; k < hull.length; k++) {
        const ab = sub(hull[j], hull[i]);
        const ac = sub(hull[k], hull[i]);
        const area = Math.abs(ab[0] * ac[1] - ab[1] * ac[0]) * 0.5;
        if (area > bestArea) {
          bestArea = area;
          tri = [hull[i], hull[j], hull[k]];
        }
      }
  return { shape: { type: "triangle", vertices: tri }, residual: residual(pts, tri, true) };
}

function ellipseOutline(c: V, major: number, minor: number, rotation: number): V[] {
  const cc = Math.cos(rotation), ss = Math.sin(rotation);
  const out: V[] = [];
  for (let i = 0; i < 160; i++) {
    const t = (2 * Math.PI * i) / 160;
    const x = major * Math.cos(t), y = minor * Math.sin(t);
    out.push([c[0] + x * cc - y * ss, c[1] + x * ss + y * cc]);
  }
  return out;
}

function fitEllipse(pts: V[]): { shape: Shape; residual: number } {
  const c = centroid(pts);
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of pts) {
    const d = sub(p, c);
    sxx += d[0] * d[0];
    sxy += d[0] * d[1];
    syy += d[1] * d[1];
  }
  const u = symEig(sxx, sxy, syy).v1;
  const v: V = [-u[1], u[0]];
  let uLo = Infinity, uHi = -Infinity, vLo = Infinity, vHi = -Infinity;
  for (const p of pts) {
    const tu = dot(sub(p, c), u), tv = dot(sub(p, c), v);
    uLo = Math.min(uLo, tu);
    uHi = Math.max(uHi, tu);
    vLo = Math.min(vLo, tv);
    vHi = Math.max(vHi, tv);
  }
  const center = add(add(c, mul(u, (uLo + uHi) / 2)), mul(v, (vLo + vHi) / 2));
  const major = (uHi - uLo) / 2, minor = (vHi - vLo) / 2;
  const rotation = Math.atan2(u[1], u[0]);
  if (!(major > 0 && minor > 0 && isFinite(major) && isFinite(minor))) {
    const r = pts.reduce((acc, p) => acc + len(sub(p, c)), 0) / pts.length;
    return {
      shape: { type: "ellipse", center: c, semiMajor: r, semiMinor: r, rotation: 0 },
      residual: residual(pts, ellipseOutline(c, r, r, 0), true),
    };
  }
  return {
    shape: { type: "ellipse", center, semiMajor: major, semiMinor: minor, rotation },
    residual: residual(pts, ellipseOutline(center, major, minor, rotation), true),
  };
}

function starVertices(center: V, outer: number, inner: number, rotation: number): V[] {
  const out: V[] = [];
  for (let i = 0; i < 10; i++) {
    const a = rotation - Math.PI / 2 + (i * Math.PI) / 5;
    const r = i % 2 === 0 ? outer : inner;
    out.push([center[0] + r * Math.cos(a), center[1] + r * Math.sin(a)]);
  }
  return out;
}

function fitStar(pts: V[]): { shape: Shape; residual: number } {
  const center = centroid(pts);
  const radii = pts.map((p) => len(sub(p, center)));
  let sc = 0, ss = 0;
  for (let i = 0; i < pts.length; i++) {
    const a = Math.atan2(pts[i][1] - center[1], pts[i][0] - center[0]);
    const w = radii[i] * radii[i];
    sc += w * Math.cos(5 * a);
    ss += w * Math.sin(5 * a);
  }
  const rot0 = Math.atan2(ss, sc) / 5 + Math.PI / 2;
  const sorted = [...radii].sort((a, b) => a - b);
  const top = Math.max(1, Math.floor(sorted.length / 5));
  const outer = sorted.slice(sorted.length - top).reduce((a, b) => a + b, 0) / top;
  const inner = outer * 0.4;
  let best: { res: number; angle: number } = { res: Infinity, angle: rot0 };
  for (const angle of [rot0, rot0 + Math.PI / 5]) {
    const res = residual(pts, starVertices(center, outer, inner, angle), true);
    if (res < best.res) best = { res, angle };
  }
  return {
    shape: { type: "star", center, outerRadius: outer, innerRadius: inner, rotation: best.angle, pointCount: 5 },
    residual: best.res,
  };
}

// ---------------------------------------------------------------------------
// Snapping (regularize to clean axes / angles / circles / squares)
// ---------------------------------------------------------------------------

const LINE_AXIS_DEG = 5;
const ELLIPSE_CIRCLE_RATIO = 0.25;
const ELLIPSE_ROT_DEG = 15;
const RECT_SQUARE_RATIO = 0.25;
const RECT_ROT_DEG = 15;
const TRI_AXIS_DEG = 5;
const TRI_EQUI_RATIO = 0.25;
const TRI_ISO_RATIO = 0.25;

const snapIncrement = (a: number, incDeg: number) =>
  incDeg <= 0 ? a : Math.round(a / (incDeg * Math.PI / 180)) * (incDeg * Math.PI / 180);

function snapAxis(a: number, thrDeg: number): number | null {
  if (thrDeg <= 0) return null;
  const q = Math.PI / 2;
  const nearest = Math.round(a / q) * q;
  return Math.abs(a - nearest) <= (thrDeg * Math.PI) / 180 ? nearest : null;
}

function snap(shape: Shape): Shape {
  switch (shape.type) {
    case "line": {
      const a = shape.from, b = shape.to;
      const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
      const snapped = snapAxis(ang, LINE_AXIS_DEG);
      if (snapped == null) return shape;
      const mid: V = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      const half = len(sub(b, a)) / 2;
      const d: V = [Math.cos(snapped), Math.sin(snapped)];
      return { type: "line", from: sub(mid, mul(d, half)), to: add(mid, mul(d, half)) };
    }
    case "ellipse": {
      const hi = Math.max(shape.semiMajor, shape.semiMinor);
      const lo = Math.min(shape.semiMajor, shape.semiMinor);
      if (ELLIPSE_CIRCLE_RATIO > 0 && hi > 0 && lo / hi >= 1 - ELLIPSE_CIRCLE_RATIO) {
        const r = (shape.semiMajor + shape.semiMinor) / 2;
        return { ...shape, semiMajor: r, semiMinor: r, rotation: 0 };
      }
      return { ...shape, rotation: snapIncrement(shape.rotation, ELLIPSE_ROT_DEG) };
    }
    case "rectangle": {
      const p = shape.corners as V[];
      if (p.length !== 4) return shape;
      const center: V = [(p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4, (p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4];
      let w = len(sub(p[1], p[0]));
      let h = len(sub(p[3], p[0]));
      let ang = Math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0]);
      const hi = Math.max(w, h), lo = Math.min(w, h);
      if (RECT_SQUARE_RATIO > 0 && hi > 0 && lo / hi >= 1 - RECT_SQUARE_RATIO) {
        const s = (w + h) / 2;
        w = s;
        h = s;
      }
      ang = snapIncrement(ang, RECT_ROT_DEG);
      const hw = w / 2, hh = h / 2;
      const local: V[] = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
      return { type: "rectangle", corners: local.map((q) => add(center, rot(q, ang))) };
    }
    case "triangle":
      return snapTriangle(shape.vertices as V[]);
    case "star":
      return shape;
  }
}

const rel = (a: number, b: number) => {
  const m = Math.max(a, b);
  return m > 0 ? Math.abs(a - b) / m : 0;
};

function snapTriangle(verts: V[]): Shape {
  let v = verts.map((p) => [p[0], p[1]] as V);
  const c: V = [(v[0][0] + v[1][0] + v[2][0]) / 3, (v[0][1] + v[1][1] + v[2][1]) / 3];
  const lAB = len(sub(v[0], v[1])), lBC = len(sub(v[1], v[2])), lCA = len(sub(v[2], v[0]));
  const sides = [lAB, lBC, lCA];
  const mx = Math.max(...sides), mn = Math.min(...sides);
  if (TRI_EQUI_RATIO > 0 && mx > 0 && (mx - mn) / mx <= TRI_EQUI_RATIO) {
    const r = (len(sub(v[0], c)) + len(sub(v[1], c)) + len(sub(v[2], c))) / 3;
    let sx = 0, sy = 0;
    for (let i = 0; i < 3; i++) {
      const a = Math.atan2(v[i][1] - c[1], v[i][0] - c[0]) - (i * 2 * Math.PI) / 3;
      sx += Math.cos(a);
      sy += Math.sin(a);
    }
    const base = Math.atan2(sy, sx);
    v = [0, 1, 2].map((i) => [c[0] + r * Math.cos(base + (i * 2 * Math.PI) / 3), c[1] + r * Math.sin(base + (i * 2 * Math.PI) / 3)] as V);
  } else if (TRI_ISO_RATIO > 0) {
    const cand = [[0, 1, 2, lAB, lCA], [1, 0, 2, lAB, lBC], [2, 0, 1, lCA, lBC]] as const;
    let bi = 0;
    for (let i = 1; i < 3; i++) if (rel(cand[i][3], cand[i][4]) < rel(cand[bi][3], cand[bi][4])) bi = i;
    const best = cand[bi];
    if (rel(best[3], best[4]) <= TRI_ISO_RATIO) {
      const apex = v[best[0]];
      const avg = (best[3] + best[4]) / 2;
      const leg = (to: number): V => {
        const d = sub(v[to], apex);
        const n = len(d);
        return add(apex, mul(n > 0 ? [d[0] / n, d[1] / n] : d, avg));
      };
      v[best[1]] = leg(best[1]);
      v[best[2]] = leg(best[2]);
    }
  }
  // axis-align the longest edge
  const edges: [V, V][] = [[v[0], v[1]], [v[1], v[2]], [v[2], v[0]]];
  let longest = edges[0];
  for (const e of edges) if (len(sub(e[1], e[0])) > len(sub(longest[1], longest[0]))) longest = e;
  const ang = Math.atan2(longest[1][1] - longest[0][1], longest[1][0] - longest[0][0]);
  const snapped = snapAxis(ang, TRI_AXIS_DEG);
  if (snapped == null) return { type: "triangle", vertices: v };
  const cen: V = [(v[0][0] + v[1][0] + v[2][0]) / 3, (v[0][1] + v[1][1] + v[2][1]) / 3];
  const delta = snapped - ang;
  return { type: "triangle", vertices: v.map((p) => add(cen, rot(sub(p, cen), delta))) };
}

const FITTERS: Record<ShapeKind, (pts: V[]) => { shape: Shape; residual: number }> = {
  line: fitLine,
  rectangle: fitRectangle,
  triangle: fitTriangle,
  ellipse: fitEllipse,
  star: fitStar,
};

/** Fit `kind` to the raw stroke and snap it. Returns the clean shape + residual. */
export function fit(kind: ShapeKind, points: Point[]): { shape: Shape; residual: number } {
  const pts = resampleUniform(points as V[], 256);
  const { shape, residual: res } = FITTERS[kind](pts);
  return { shape: snap(shape), residual: res };
}
