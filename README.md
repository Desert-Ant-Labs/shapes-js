# @desert-ant-labs/shapes

**On-device single-stroke shape recognition for Node and the browser. Pure JS, no ONNX or WASM runtime.**

On-device single-stroke shape recognition. Takes a hand-drawn stroke (a list of
`[x, y]` points) and returns a clean geometric shape, fully in-process, no
inference runtime. It does sketch recognition on raw digital ink: one stroke in,
one snapped vector shape out.

```ts
import { recognize } from "@desert-ant-labs/shapes";

const shape = await recognize(points); // points: [number, number][]
// { type: "ellipse", center: [120, 90], semiMajor: 64, semiMinor: 40, rotation: 0.2 }
// or null if the stroke isn't a recognizable shape
```

## Features

- Pure-JS inference (no ONNX/WASM runtime); recognition is a few ms
- Recognizes `line`, `rectangle`, `triangle`, `ellipse`, and `star` (rejects scribbles)
- Fits clean vector geometry and snaps it to axes, circles, squares, and 15° rotations
- Model (~0.2 MB, 4-bit palettized) is fetched from the Hugging Face Hub at a **pinned revision**, then
  cached, to the **filesystem** on Node and to **Cache Storage** in the browser, so
  it loads once and runs offline after

## Install

```bash
npm install @desert-ant-labs/shapes
```

## Importing

Pure ESM and fully tree-shakeable. The **same import works everywhere** (Node, bundlers,
browsers, and edge/worker runtimes); the right build is selected automatically:

```ts
import { recognize } from "@desert-ant-labs/shapes";
```

CommonJS consumers use dynamic import (`const { recognize } = await import("@desert-ant-labs/shapes")`);
native `require()` works on Node ≥ 22.12.

**Bring-your-own-bytes.** If you load the model files yourself, import the hub-free core
from `@desert-ant-labs/shapes/core`, only the inference engine, with zero network/filesystem code:

```ts
import { createShapes } from "@desert-ant-labs/shapes/core";

// weights is a Uint8Array (shapes.safetensors); meta is the parsed shapes_meta.json
const shapes = createShapes({ weights, meta });
shapes.recognize(points)?.type; // "triangle"
```

## Loading model

Model files (`shapes.safetensors`, `shapes_meta.json`) are fetched from the Hugging Face Hub
([`desert-ant-labs/shapes`](https://huggingface.co/desert-ant-labs/shapes)) at a pinned
revision and cached.

- **Node**: `recognize()` works zero-config; files cache under `~/.cache/shapes`. To run
  fully offline, ship the files yourself and point at a folder with `env.localModelPath`
  (or `SHAPES_LOCAL_PATH`).
- **Browser**: same API; files cache in Cache Storage.

```ts
import { env, load, recognize } from "@desert-ant-labs/shapes";

env.revision = "main";                 // or a commit SHA / tag
env.localModelPath = "./shapes-model"; // Node: use local files, skip the Hub

// or load an explicit instance (synchronous inference after it resolves)
const shapes = await load();
shapes.recognize(points)?.type;
```

## API

```ts
export function recognize(points: Point[]): Promise<Shape | null>;
export function load(options?: Partial<ShapesEnv>): Promise<ShapesModel>;
export function createShapes(buffers: { weights; meta }): ShapesModel; // raw buffers
export function outline(shape: Shape, samples?: number): Point[];       // renderable polyline
export const env: ShapesEnv;
export function reset(): void; // clear the memoized model so the next recognize() re-reads env

export type Point = [number, number];

export type Shape =
  | { type: "line"; from: Point; to: Point }
  | { type: "rectangle"; corners: Point[] }
  | { type: "triangle"; vertices: Point[] }
  | { type: "ellipse"; center: Point; semiMajor: number; semiMinor: number; rotation: number }
  | { type: "star"; center: Point; outerRadius: number; innerRadius: number; rotation: number; pointCount: number };
```

`recognize(points)` returns the snapped shape, or `null` when the stroke is rejected
(not a shape) or degenerate. `ShapesModel.recognize` is synchronous once loaded.

## Example

[`Examples/ShapesExample`](Examples/ShapesExample) is a tldraw canvas demo: draw a shape,
pause for the preview, then lift to snap on-device.

```bash
cd Examples/ShapesExample
npm install
npm run dev
```

## Model

Published at [`desert-ant-labs/shapes`](https://huggingface.co/desert-ant-labs/shapes) on Hugging Face.

## Other platforms

Same model, native on each platform:

- [`shapes-swift`](https://github.com/Desert-Ant-Labs/shapes-swift): Swift for iOS and macOS, with PencilKit stroke snapping
- [`shapes-kotlin`](https://github.com/Desert-Ant-Labs/shapes-kotlin): Kotlin for Android and the JVM
- Model weights and card: [`desert-ant-labs/shapes`](https://huggingface.co/desert-ant-labs/shapes)

## License

[Desert Ant Labs Source-Available License](https://license.desertant.ai/1.0). Free for
most apps; a commercial license is required at scale. Full terms are at the link.
Licensing: <licensing@desertant.ai>.
