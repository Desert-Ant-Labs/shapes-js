# ShapesExample

A [tldraw](https://tldraw.dev) canvas with iOS-style shape snapping: draw a rough
shape with the draw tool and it snaps to a clean one, on-device, with
[`@desert-ant-labs/shapes`](../../). Undo/redo and the rest of the tldraw UI work as usual.

```bash
# from the repo root, build the library once:
npm install && npm run build

# then run the example:
cd Examples/ShapesExample
npm install
npm run dev   # open the printed localhost URL
```

The model files (`shapes.safetensors`, `shapes_meta.json`) are served from `public/`,
so the demo runs fully offline with no Hugging Face auth.
