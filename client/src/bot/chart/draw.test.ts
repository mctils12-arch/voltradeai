import { test } from "node:test";
import assert from "node:assert/strict";
import { deoverlap, withAlpha, geometry, GUTTER_R, AXIS_B, LABEL_H, readTheme, FALLBACK_THEME } from "./draw";

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

test("readTheme without a DOM returns the DESIGN.md fallback tokens", () => {
  assert.deepEqual(readTheme(null), FALLBACK_THEME);
  assert.equal(FALLBACK_THEME.accent, "#4d9fff");
});
