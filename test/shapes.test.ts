import assert from "node:assert/strict";
import test from "node:test";

import { env, recognize } from "../src/index.node.ts";

// Run offline against local model files when SHAPES_LOCAL_PATH is set; otherwise
// the model is fetched from the Hugging Face Hub (cached).
if (process.env.SHAPES_LOCAL_PATH) env.localModelPath = process.env.SHAPES_LOCAL_PATH;

function circle(n = 60): [number, number][] {
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / n;
    return [100 + 80 * Math.cos(t), 100 + 80 * Math.sin(t)] as [number, number];
  });
}

function line(n = 40): [number, number][] {
  return Array.from({ length: n + 1 }, (_, i) => [i * 5, i * 2] as [number, number]);
}

test("recognizes a circle as an ellipse", async () => {
  const shape = await recognize(circle());
  assert.equal(shape?.type, "ellipse");
});

test("recognizes a straight stroke as a line", async () => {
  const shape = await recognize(line());
  assert.equal(shape?.type, "line");
});

test("rejects a single point", async () => {
  assert.equal(await recognize([[1, 1]]), null);
});
