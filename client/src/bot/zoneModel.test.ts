import { test } from "node:test";
import assert from "node:assert/strict";
import { tierFor, TEMPLATES, gridStyle, toggleIn, isPinned, EMPTY_FLAGS, clampRect, initialFloatRect, snapEdges } from "./zoneModel";

test("tiers: big screen first, phone is its own tier", () => {
  assert.equal(tierFor(2560), "xl");
  assert.equal(tierFor(1440), "l");
  assert.equal(tierFor(1280), "l");
  assert.equal(tierFor(1024), "m");
  assert.equal(tierFor(768), "m");
  assert.equal(tierFor(390), "s");
  assert.equal(tierFor(NaN), "l");
});

test("every template row names the same number of columns as its track list", () => {
  for (const [tier, t] of Object.entries(TEMPLATES)) {
    const cols = t.columns.split(/\s+(?![^(]*\))/).length;
    for (const row of t.areas) assert.equal(row.split(" ").length, cols, `${tier}: "${row}"`);
    assert.ok(t.areas.some((r) => r.includes("centre")) && t.areas.some((r) => r.includes("timeline")));
  }
  assert.equal(TEMPLATES.l.scrolls, false, "desktop never scrolls the page");
  assert.equal(TEMPLATES.s.scrolls, true);
  assert.match(gridStyle("l").gridTemplateAreas, /^"kpi kpi kpi" "left centre right" "timeline timeline timeline"$/);
});

test("flags: toggling and the held-implies-pinned rule", () => {
  assert.deepEqual(toggleIn([], "book"), ["book"]);
  assert.deepEqual(toggleIn(["book"], "book"), []);
  assert.equal(isPinned({ ...EMPTY_FLAGS, held: ["book"] }, "book"), true);
  assert.equal(isPinned(EMPTY_FLAGS, "book"), false);
});

test("float rects stay on the page and snap to edges", () => {
  const page = { w: 1440, h: 844 };
  const r = initialFloatRect({ x: 300, y: 80, w: 700, h: 600 }, page);
  assert.ok(r.w <= page.w * 0.78 && r.h <= page.h * 0.78 && r.x >= 0 && r.y >= 0);
  const off = clampRect({ x: 1500, y: 900, w: 400, h: 300 }, page);
  assert.equal(off.x, page.w - 120);
  assert.equal(off.y, page.h - 120);
  assert.deepEqual(snapEdges({ x: 7, y: 836 - 300 + 5, w: 400, h: 300 }, page), { x: 0, y: page.h - 300, w: 400, h: 300 });
});
