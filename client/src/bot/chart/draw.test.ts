import { test } from "node:test";
import assert from "node:assert/strict";
import { deoverlap, gutterYs, withAlpha, geometry, GUTTER_R, AXIS_B, LABEL_H, readTheme, FALLBACK_THEME } from "./draw";

test("gutter labels are pushed apart by one label height and stay inside the pane", () => {
  const ys = deoverlap([100, 104, 101, 300], 320);
  const sorted = [...ys].sort((a, b) => a - b);
  for (let k = 1; k < sorted.length; k++) assert.ok(sorted[k] - sorted[k - 1] >= LABEL_H - 1e-9);
  assert.ok(Math.max(...ys) <= 320 - LABEL_H / 2 + 1e-9);
  assert.ok(Math.min(...ys) >= LABEL_H / 2 - 1e-9);
  // the input order is preserved (label k ↔ y k)
  assert.ok(ys[3] > ys[0]);
  assert.deepEqual(deoverlap([], 100), []);
  // crowded at the bottom edge: everything shifts up, still non-overlapping
  const crowded = deoverlap([318, 319, 320], 320);
  const s2 = [...crowded].sort((a, b) => a - b);
  assert.ok(s2[2] <= 320 - LABEL_H / 2 + 1e-9 && s2[1] <= s2[2] - LABEL_H + 1e-9);
});

test("withAlpha handles hex, short hex, rgb and rgba, and leaves unknown strings alone", () => {
  assert.equal(withAlpha("#4ade80", 0.5), "rgba(74, 222, 128, 0.5)");
  assert.equal(withAlpha("#fff", 0.2), "rgba(255, 255, 255, 0.2)");
  assert.equal(withAlpha("rgba(120, 165, 220, 0.16)", 0.9), "rgba(120, 165, 220, 0.9)");
  assert.equal(withAlpha("rgb(1,2,3)", 2), "rgba(1, 2, 3, 1)");
  assert.equal(withAlpha("var(--x)", 0.5), "var(--x)");
});

test("geometry reserves the price gutter and the time axis; the volume pane takes its fraction only when shown", () => {
  const g = geometry(600, 300, true);
  assert.equal(g.plotW, 600 - GUTTER_R);
  assert.equal(g.paneH, 300 - AXIS_B);
  assert.ok(g.volH > 0 && Math.abs(g.priceH + g.volH - g.paneH) < 1e-9);
  const noVol = geometry(600, 300, false);
  assert.equal(noVol.volH, 0);
  assert.equal(noVol.priceH, noVol.paneH);
  // never collapses to zero
  assert.ok(geometry(10, 10, true).plotW >= 1 && geometry(10, 10, true).paneH >= 1);
});

test("readTheme without a DOM returns the generic fallback, which restates none of the palette (the tokens live in index.css alone)", () => {
  assert.deepEqual(readTheme(null), FALLBACK_THEME);
  for (const [k, v] of Object.entries(FALLBACK_THEME)) if (k !== "font") assert.ok(!/#[0-9a-f]{3,8}/i.test(v), `${k} restates a hex colour`);
  assert.ok(FALLBACK_THEME.font.includes("monospace"));
});

test("gutterYs: the crosshair's pill is a pin — the level pills are pushed clear of it, it keeps its exact y, and nothing leaves the pane", () => {
  // a level pill sits right under the mouse: it moves one label height away; the pin's own y is not returned (it is drawn where it is)
  const pushed = gutterYs([200, 100], 200, 400);
  assert.equal(pushed.length, 2);
  assert.ok(Math.abs(pushed[0] - 200) >= LABEL_H, `the level under the pin moved: ${pushed[0]}`);
  assert.equal(pushed[1], 100);
  // pins near the bottom edge push the neighbours up, never below the pane
  const low = gutterYs([396, 394], 398, 400);
  assert.ok(low.every((v) => v <= 400 - LABEL_H / 2 && v >= LABEL_H / 2));
  assert.ok(low.every((v) => Math.abs(v - 398) >= LABEL_H - 1e-9), `pushed clear of the pin: ${low}`);
  // no pin: identical to the plain de-overlap
  assert.deepEqual(gutterYs([100, 104, 101, 300], null, 320), deoverlap([100, 104, 101, 300], 320));
});

