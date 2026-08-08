import { test } from "node:test";
import assert from "node:assert/strict";
import { sectorChangePct } from "./sectorHeatmapGuards";

// ── the fabricated -100% class (2026-08-06 full-code-review finding,
// queued unfixed for several sessions until this repair) ──────────────────
// Pre-fix inline math: `const c = bar.c || 0; const pc = prev.c || c;
// const change = pc > 0 ? ((c - pc) / pc) * 100 : 0;` — when dailyBar has
// no real price yet (c undefined -> 0) but prevDailyBar carries yesterday's
// real close, pc = that real close, and (0 - pc) / pc * 100 == -100 every
// time, regardless of the sector's actual state.

test("no current price yet (dailyBar missing c), real prior close: returns null, never -100", () => {
  assert.equal(sectorChangePct({}, { c: 205.5 }), null);
  assert.equal(sectorChangePct(undefined, { c: 205.5 }), null);
});

test("dailyBar present but c is zero/non-finite: still null, not a fabricated crash", () => {
  assert.equal(sectorChangePct({ c: 0 }, { c: 100 }), null);
  assert.equal(sectorChangePct({ c: NaN }, { c: 100 }), null);
  assert.equal(sectorChangePct({ c: -1 }, { c: 100 }), null);
});

test("no prior close (prevDailyBar missing/zero): falls back to current price, 0% change", () => {
  assert.equal(sectorChangePct({ c: 50 }, {}), 0);
  assert.equal(sectorChangePct({ c: 50 }, undefined), 0);
  assert.equal(sectorChangePct({ c: 50 }, { c: 0 }), 0);
});

test("normal case: real percent change computed and rounded to 2dp", () => {
  assert.equal(sectorChangePct({ c: 102 }, { c: 100 }), 2);
  assert.equal(sectorChangePct({ c: 98 }, { c: 100 }), -2);
  assert.equal(sectorChangePct({ c: 100.333 }, { c: 100 }), 0.33);
});
