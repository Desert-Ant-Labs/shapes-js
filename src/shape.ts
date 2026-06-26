/** A 2-D point, `[x, y]`, in the same coordinate space as the input stroke. */
export type Point = [number, number];

/** A recognized, fitted shape. */
export type Shape =
  | { type: "line"; from: Point; to: Point }
  | { type: "rectangle"; corners: Point[] }
  | { type: "triangle"; vertices: Point[] }
  | { type: "ellipse"; center: Point; semiMajor: number; semiMinor: number; rotation: number }
  | {
      type: "star";
      center: Point;
      outerRadius: number;
      innerRadius: number;
      rotation: number;
      pointCount: number;
    };

/** The semantic kinds a stroke can be recognized as. */
export type ShapeKind = Shape["type"];

/**
 * A closed (or, for a line, open) polyline outline of `shape`, suitable for
 * rendering. `samples` controls the smoothness of the ellipse.
 */
export function outline(shape: Shape, samples = 96): Point[] {
  switch (shape.type) {
    case "line":
      return [shape.from, shape.to];
    case "rectangle":
      return shape.corners;
    case "triangle":
      return shape.vertices;
    case "ellipse": {
      const [cx, cy] = shape.center;
      const c = Math.cos(shape.rotation);
      const s = Math.sin(shape.rotation);
      const out: Point[] = [];
      for (let i = 0; i < samples; i++) {
        const t = (2 * Math.PI * i) / samples;
        const x = shape.semiMajor * Math.cos(t);
        const y = shape.semiMinor * Math.sin(t);
        out.push([cx + x * c - y * s, cy + x * s + y * c]);
      }
      return out;
    }
    case "star": {
      const [cx, cy] = shape.center;
      const out: Point[] = [];
      const steps = shape.pointCount * 2;
      for (let i = 0; i < steps; i++) {
        const a = shape.rotation - Math.PI / 2 + (i * Math.PI) / shape.pointCount;
        const r = i % 2 === 0 ? shape.outerRadius : shape.innerRadius;
        out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
      }
      return out;
    }
  }
}
