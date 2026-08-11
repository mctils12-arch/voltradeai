import { test } from "node:test";
import assert from "node:assert/strict";
import { demandCell, rankLedger, buildLedger } from "./demandProfile.mjs";

test("demandCell: deltas computed, absolute frame stats kept beside baseline", () => {
  const c = demandCell("terrain", "throttled4x",
    { medianMs: 40, p95Ms: 90, heapMB: 120, netKB: 500 },
    { medianMs: 95, p95Ms: 210, heapMB: 168.4, netKB: 2600, longTasks: 7, renderedFeatures: 0 });
  assert.equal(c.medianMs, 95);
  assert.equal(c.baselineMedianMs, 40);
  assert.equal(c.heapDeltaMB, 48.4);
  assert.equal(c.netDeltaKB, 2100);
  assert.equal(c.longTasks, 7);
});

test("demandCell: unmeasurable metrics stay null — never fabricated as 0", () => {
  const c = demandCell("alerts", "full", { medianMs: 30, heapMB: null }, { medianMs: 33, heapMB: null });
  assert.equal(c.heapDeltaMB, null);
  assert.equal(c.netDeltaKB, null);
});

test("rankLedger: most frame-expensive first; null-metric cells sink, not float", () => {
  const cells = [
    demandCell("cheap", "full", { medianMs: 30, heapMB: 100 }, { medianMs: 32, heapMB: 101 }),
    demandCell("heavy", "full", { medianMs: 30, heapMB: 100 }, { medianMs: 80, heapMB: 140 }),
    demandCell("unmeasured", "full", { medianMs: null }, { medianMs: null }),
  ];
  const ranked = rankLedger(cells).map((c) => c.layer);
  assert.deepEqual(ranked, ["heavy", "cheap", "unmeasured"]);
});

test("buildLedger: version-stamped, caps recorded (no silent truncation), ranked", () => {
  const l = buildLedger("1.0.621",
    [demandCell("a", "full", { medianMs: 30 }, { medianMs: 31 })],
    { layers: ["a"], skipped: ["b"], sampleSeconds: 3, tiers: ["full", "throttled4x"] });
  assert.equal(l.schema, "vt-demand-profile/1");
  assert.equal(l.version, "1.0.621");
  assert.deepEqual(l.caps.skipped, ["b"]);
  assert.match(l.note, /never assumed 0/);
});
