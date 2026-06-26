import { createShapes, outline, type Point, type Shape, type ShapesModel } from "@desert-ant-labs/shapes/core";
import { useRef, useState } from "react";
import {
  createShapeId,
  DefaultToolbar,
  DrawToolbarItem,
  type Editor,
  EraserToolbarItem,
  getDefaultColorTheme,
  HandToolbarItem,
  type IndexKey,
  SelectToolbarItem,
  type TLComponents,
  type TLDrawShape,
  Tldraw,
} from "tldraw";
import "tldraw/tldraw.css";

// Restrict the toolbar to select, hand, pen (draw), and eraser.
const components: TLComponents = {
  Toolbar: () => (
    <DefaultToolbar>
      <SelectToolbarItem />
      <HandToolbarItem />
      <DrawToolbarItem />
      <EraserToolbarItem />
    </DefaultToolbar>
  ),
};

// The model files are self-hosted in /public so the demo needs no network/auth.
let modelPromise: Promise<ShapesModel> | null = null;
function getModel(): Promise<ShapesModel> {
  modelPromise ??= (async () => {
    const [weights, meta] = await Promise.all([
      fetch("/shapes.safetensors").then((r) => r.arrayBuffer()),
      fetch("/shapes_meta.json").then((r) => r.json()),
    ]);
    return createShapes({ weights: new Uint8Array(weights), meta });
  })();
  return modelPromise;
}

interface Preview {
  d: string;
  color: string;
}

export default function App() {
  const editorRef = useRef<Editor | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [empty, setEmpty] = useState(true);

  return (
    <div style={{ position: "fixed", inset: 0 }}>
      <Tldraw
        components={components}
        onMount={(editor) => {
          editorRef.current = editor;
          editor.setCurrentTool("draw");
          const update = () => setEmpty(editor.getCurrentPageShapeIds().size === 0);
          update();
          editor.store.listen(update);
          wireSnapping(editor, setPreview);
        }}
      />
      {empty && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 12,
            pointerEvents: "none",
            color: "#a1a1aa",
          }}
        >
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 20h9" />
            <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
          </svg>
          <span style={{ fontSize: 15, fontWeight: 500 }}>Draw a shape, pause, then lift to snap</span>
        </div>
      )}
      {preview && (
        <svg style={{ position: "absolute", inset: 0, pointerEvents: "none" }} width="100%" height="100%">
          <path d={preview.d} fill="none" stroke={preview.color} strokeWidth={3} strokeOpacity={0.5}
            strokeLinejoin="round" strokeLinecap="round" />
        </svg>
      )}
    </div>
  );
}

// iOS-style snapping on top of the standard tldraw draw tool + UI: while drawing,
// a brief pause shows a faded preview of the recognized shape; on lift the rough
// stroke is replaced by the clean one (one undoable step).
function wireSnapping(editor: Editor, setPreview: (p: Preview | null) => void) {
  const processed = new Set<string>();
  let drawing = false;
  let pauseTimer: ReturnType<typeof setTimeout> | null = null;
  const clearPause = () => {
    if (pauseTimer) clearTimeout(pauseTimer);
    pauseTimer = null;
  };

  window.addEventListener("pointerdown", () => {
    drawing = editor.getCurrentToolId() === "draw";
  }, true);
  window.addEventListener("pointermove", () => {
    if (!drawing) return;
    setPreview(null);
    clearPause();
    pauseTimer = setTimeout(() => void showPreview(editor, setPreview), 300);
  }, true);
  window.addEventListener("pointerup", () => {
    drawing = false;
    clearPause();
    setPreview(null);
    requestAnimationFrame(() => void commit(editor, processed));
  }, true);
}

function strokePoints(d: TLDrawShape): Point[] {
  const pts: Point[] = [];
  for (const seg of d.props.segments) for (const p of seg.points) pts.push([d.x + p.x, d.y + p.y]);
  return pts;
}

async function showPreview(editor: Editor, setPreview: (p: Preview | null) => void) {
  const live = editor.getCurrentPageShapes().find(
    (s): s is TLDrawShape => s.type === "draw" && !(s as TLDrawShape).props.isComplete && !s.meta?.snapped,
  );
  if (!live) return;
  const shape = await (await getModel()).recognize(strokePoints(live));
  if (!shape) return setPreview(null);
  const scr = densify(shape).map((p) => editor.pageToScreen({ x: p[0], y: p[1] }));
  const theme = getDefaultColorTheme({ isDarkMode: editor.user.getIsDarkMode() });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const color = (theme as any)[live.props.color]?.solid ?? theme.text;
  setPreview({ d: scr.map((p, i) => `${i ? "L" : "M"} ${p.x} ${p.y}`).join(" "), color });
}

async function commit(editor: Editor, processed: Set<string>) {
  for (const s of editor.getCurrentPageShapes()) {
    if (s.type !== "draw") continue;
    const d = s as TLDrawShape;
    if (!d.props.isComplete || d.meta?.snapped || processed.has(d.id)) continue;
    processed.add(d.id);

    const shape = await (await getModel()).recognize(strokePoints(d));
    if (!shape) continue;

    editor.run(() => {
      editor.deleteShape(d.id);
      createSnapped(editor, shape, d.props);
    });
  }
}

// Use native tldraw primitives where they match cleanly (line, rectangle,
// ellipse); a clean draw stroke for the rest (triangle, star), which tldraw's
// geo shapes don't represent faithfully.
function createSnapped(editor: Editor, shape: Shape, style: TLDrawShape["props"]) {
  const { color, size, fill, dash } = style;
  switch (shape.type) {
    case "line": {
      const [a, b] = [shape.from, shape.to];
      editor.createShape({
        type: "line",
        x: a[0],
        y: a[1],
        meta: { snapped: true },
        props: {
          color,
          size,
          dash,
          spline: "line",
          points: {
            a1: { id: "a1", index: "a1" as IndexKey, x: 0, y: 0 },
            a2: { id: "a2", index: "a2" as IndexKey, x: b[0] - a[0], y: b[1] - a[1] },
          },
        },
      });
      return;
    }
    case "rectangle": {
      const c = shape.corners;
      const center: Point = [
        (c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4,
        (c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4,
      ];
      const w = Math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1]);
      const h = Math.hypot(c[3][0] - c[0][0], c[3][1] - c[0][1]);
      createGeo(editor, "rectangle", center, w, h, Math.atan2(c[1][1] - c[0][1], c[1][0] - c[0][0]), style);
      return;
    }
    case "ellipse": {
      createGeo(editor, "ellipse", shape.center, shape.semiMajor * 2, shape.semiMinor * 2, shape.rotation, style);
      return;
    }
    default: {
      // triangle, star -> clean draw stroke tracing the outline
      const poly = densify(shape);
      const minX = Math.min(...poly.map((p) => p[0]));
      const minY = Math.min(...poly.map((p) => p[1]));
      editor.createShape<TLDrawShape>({
        id: createShapeId(),
        type: "draw",
        x: minX,
        y: minY,
        meta: { snapped: true },
        props: {
          color,
          size,
          fill,
          dash,
          segments: [{ type: "free", points: poly.map((p) => ({ x: p[0] - minX, y: p[1] - minY, z: 0.5 })) }],
          isClosed: true,
          isComplete: true,
        },
      });
    }
  }
}

// tldraw rotates a shape around its (x, y) origin, so place (x, y) such that the
// shape's center lands on `center`.
function createGeo(
  editor: Editor,
  geo: "rectangle" | "ellipse",
  center: Point,
  w: number,
  h: number,
  rotation: number,
  style: TLDrawShape["props"],
) {
  const c = Math.cos(rotation);
  const s = Math.sin(rotation);
  editor.createShape({
    type: "geo",
    x: center[0] - (c * (w / 2) - s * (h / 2)),
    y: center[1] - (s * (w / 2) + c * (h / 2)),
    rotation,
    meta: { snapped: true },
    props: { geo, w, h, color: style.color, size: style.size, fill: style.fill, dash: style.dash },
  });
}

// The clean outline has only a few vertices (4 for a rectangle); a tldraw draw
// shape spline-smooths its points, so densify the edges to keep them crisp.
function densify(shape: Shape): Point[] {
  let poly = outline(shape);
  if (shape.type !== "line") poly = [...poly, poly[0]];
  const xs = poly.map((p) => p[0]);
  const ys = poly.map((p) => p[1]);
  const diag = Math.hypot(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  const spacing = Math.max(2, diag / 120);
  const out: Point[] = [];
  for (let i = 0; i < poly.length - 1; i++) {
    const a = poly[i];
    const b = poly[i + 1];
    const steps = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / spacing));
    for (let j = 0; j < steps; j++) {
      const t = j / steps;
      out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
    }
  }
  out.push(poly[poly.length - 1]);
  return out;
}
