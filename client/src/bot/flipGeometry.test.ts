import { test } from "node:test";
import assert from "node:assert/strict";
import { contentPos } from "./flipGeometry";

test("a pure scroll leaves a slot's content position unchanged", () => {
  const before = contentPos({ left: 10, top: 300 }, { left: 0, top: 56, scrollLeft: 0, scrollTop: 0 });
  const after = contentPos({ left: 10, top: 200 }, { left: 0, top: 56, scrollLeft: 0, scrollTop: 100 });
  assert.deepEqual(after, before);
});

test("a real move still shows as a delta", () => {
  const host = { left: 0, top: 56, scrollLeft: 0, scrollTop: 40 };
  const a = contentPos({ left: 10, top: 300 }, host), b = contentPos({ left: 10, top: 380 }, host);
  assert.equal(b.top - a.top, 80);
});

test("no scroll host falls back to the viewport rect", () => {
  assert.deepEqual(contentPos({ left: 3, top: 4 }, null), { left: 3, top: 4 });
});
