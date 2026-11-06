import { test } from "node:test";
import assert from "node:assert/strict";
import { tierFor, gridStyle, emptyFlags, toggleIn, isPinned, clampRect, initialFloatRect, snapEdges, type LayoutFlags, type Template } from "./zoneModel";

type Id = "a" | "b" | "c";

test("tierFor: inclusive at each breakpoint; a non-number reads as the desktop tier", () => {
  assert.equal(tierFor(1920), "xl");
  assert.equal(tierFor(1919), "l");
  assert.equal(tierFor(1180), "l");
  assert.equal(tierFor(1179), "m");
  assert.equal(tierFor(768), "m");
  assert.equal(tierFor(767), "s");
  assert.equal(tierFor(0), "s");
  assert.equal(tierFor(NaN), "l");
  assert.equal(tierFor(Infinity), "l");
});

test("flags: a fresh set is empty and unshared; toggleIn adds then removes; held implies pinned", () => {
  const a = emptyFlags<Id>(), b = emptyFlags<Id>();
  assert.deepEqual(a, { hold: false, zoomed: null, floating: [], held: [], pinned: [] });
  assert.notEqual(a, b, "two callers never share one object");
  assert.notEqual(a.floating, b.floating);
  assert.deepEqual(toggleIn<Id>([], "a"), ["a"]);
  assert.deepEqual(toggleIn<Id>(["a", "b"], "a"), ["b"]);
  assert.deepEqual(toggleIn<Id>(["a"], "b"), ["a", "b"]);
  const f: LayoutFlags<Id> = { ...emptyFlags<Id>(), held: ["b"], pinned: ["c"] };
  assert.equal(isPinned(f, "b"), true, "held is pinned");
  assert.equal(isPinned(f, "c"), true);
  assert.equal(isPinned(f, "a"), false);
});

test("gridStyle: a custom template becomes the three grid-template-* strings, areas quoted row by row", () => {
  const t: Template = { columns: "200px minmax(0, 1fr)", rows: "auto 1fr", areas: ["top top", "side main"], scrolls: false };
  assert.deepEqual(gridStyle(t), { gridTemplateColumns: "200px minmax(0, 1fr)", gridTemplateRows: "auto 1fr", gridTemplateAreas: '"top top" "side main"' });
  assert.equal(gridStyle({ ...t, areas: ["only"] }).gridTemplateAreas, '"only"');
});

test("clampRect: the minimum size, never larger than the page, at least minVisible px stays on it", () => {
  const page = { w: 1000, h: 800 };
  assert.deepEqual(clampRect({ x: 0, y: 0, w: 100, h: 100 }, page), { x: 0, y: 0, w: 280, h: 160 });
  const big = clampRect({ x: 0, y: 0, w: 5000, h: 5000 }, page);
  assert.equal(big.w, 1000);
  assert.equal(big.h, 800);
  const left = clampRect({ x: -900, y: -50, w: 400, h: 300 }, page);
  assert.equal(left.x, 120 - 400);
  assert.equal(left.y, 0);
  const right = clampRect({ x: 2000, y: 2000, w: 400, h: 300 }, page);
  assert.equal(right.x, 1000 - 120);
  assert.equal(right.y, 800 - 120);
  assert.equal(clampRect({ x: 950, y: 0, w: 400, h: 300 }, page, 60).x, 940);
});

test("initialFloatRect: a quarter larger than the slot, at least 640×360, at most 78 % of the page, centred a little high", () => {
  const page = { w: 1600, h: 1000 };
  assert.deepEqual(initialFloatRect({ x: 10, y: 10, w: 800, h: 400 }, page), { x: 300, y: 230, w: 1000, h: 500 });
  assert.deepEqual(initialFloatRect({ x: 0, y: 0, w: 100, h: 100 }, page), { x: 480, y: 300, w: 640, h: 360 });
  const huge = initialFloatRect({ x: 0, y: 0, w: 3000, h: 3000 }, page);
  assert.equal(huge.w, page.w * 0.78);
  assert.equal(huge.h, page.h * 0.78);
  // a page smaller than the floor: clamped to the page, still centred
  assert.deepEqual(initialFloatRect({ x: 0, y: 0, w: 100, h: 100 }, { w: 500, h: 400 }), { x: 55, y: 24, w: 390, h: 312 });
});

test("snapEdges: within the threshold a window sticks to the page edge; beyond it nothing moves", () => {
  const page = { w: 1000, h: 800 };
  assert.deepEqual(snapEdges({ x: 11, y: 11, w: 300, h: 200 }, page), { x: 0, y: 0, w: 300, h: 200 });
  assert.deepEqual(snapEdges({ x: 12, y: 12, w: 300, h: 200 }, page), { x: 12, y: 12, w: 300, h: 200 });
  assert.deepEqual(snapEdges({ x: 1000 - 300 - 5, y: 800 - 200 + 3, w: 300, h: 200 }, page), { x: 700, y: 600, w: 300, h: 200 });
  assert.deepEqual(snapEdges({ x: 25, y: 300, w: 300, h: 200 }, page, 30), { x: 0, y: 300, w: 300, h: 200 });
});
