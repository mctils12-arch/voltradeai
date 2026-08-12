// perfHud.test.ts — the instrument that makes every later PR verifiable.
//
// The HUD is the only thing standing between "the moon looks fuzzy" and a
// measurement, so its thresholds and its arithmetic are tested rather than
// eyeballed. Includes the two Law-critical behaviours: an unready draw is
// always red (Law II.1 admits no healthy nonzero range) and the HUD's own
// teardown is complete (Law IV applies to the debug overlay too).

import { test } from "node:test";
import assert from "node:assert/strict";

import { FRAME, FrameLoop, type FrameHost, type Scheduler } from "./frameCore.ts";
import { HUD_SPARK_SAMPLES, hudRows, mountPerfHud, perfEnabled, sparkline } from "./perfHud.ts";
import {
  BYTES_PER_PIXEL,
  METRIC,
  addGauge,
  bump,
  formatBytes,
  getCounter,
  getGauge,
  resetGauge,
  resetMetrics,
  setGauge,
  snapshot,
  textureBytes,
} from "./perfMetrics.ts";

function baseInput(over: Partial<Parameters<typeof hudRows>[0]> = {}) {
  return {
    p50: 8,
    p95: 12,
    worst: 20,
    frames: 100,
    samples: 100,
    overBudget: 0,
    errorCount: 0,
    awakeSprings: 0,
    registrations: 3,
    paused: false,
    ...over,
  };
}

const rowFor = (rows: ReturnType<typeof hudRows>, label: string) => rows.find((r) => r.label === label)!;

// ── enablement ──────────────────────────────────────────────────────────────

test("perfEnabled recognises the documented forms and nothing else", () => {
  assert.equal(perfEnabled("?perf=1"), true);
  assert.equal(perfEnabled("perf=1"), true);
  assert.equal(perfEnabled("?a=b&perf=1&c=d"), true);
  assert.equal(perfEnabled("?perf"), true);
  assert.equal(perfEnabled("?perf=true"), true);
  assert.equal(perfEnabled("?perf=0"), false);
  assert.equal(perfEnabled("?performance=1"), false, "must not match a prefix");
  assert.equal(perfEnabled(""), false);
  assert.equal(perfEnabled("", "1"), true, "localStorage opt-in survives navigation");
  assert.equal(perfEnabled("", "0"), false);
});

// ── rows ────────────────────────────────────────────────────────────────────

test("frame-time rows colour against FRAME_BUDGET_MS", () => {
  resetMetrics();
  const ok = rowFor(hudRows(baseInput({ p95: FRAME.FRAME_BUDGET_MS })), "frame p95");
  assert.equal(ok.level, "ok", "exactly at budget is still ok");
  assert.equal(rowFor(hudRows(baseInput({ p95: 17 })), "frame p95").level, "warn");
  assert.equal(rowFor(hudRows(baseInput({ p95: 40 })), "frame p95").level, "bad");
});

test("an unready draw is ALWAYS red — Law II.1 has no healthy nonzero range", () => {
  resetMetrics();
  assert.equal(rowFor(hudRows(baseInput()), "unready draws").level, "ok");
  bump(METRIC.UNREADY_DRAWS);
  const row = rowFor(hudRows(baseInput()), "unready draws");
  assert.equal(row.value, "1");
  assert.equal(row.level, "bad", "one unready draw is a bug, not a warning");
  resetMetrics();
});

test("rows read the live gauges publishers set", () => {
  resetMetrics();
  setGauge(METRIC.DRAW_CALLS, 42);
  setGauge(METRIC.TEXTURES, 7);
  setGauge(METRIC.VRAM_BYTES, 200 * 1024 * 1024);
  setGauge(METRIC.IN_FLIGHT, 5);
  setGauge(METRIC.TILES_RESIDENT, 312);
  setGauge(METRIC.TILES_PENDING, 9);
  const rows = hudRows(baseInput());
  assert.equal(rowFor(rows, "draw calls").value, "42");
  assert.equal(rowFor(rows, "textures").value, "7");
  assert.equal(rowFor(rows, "vram est").value, "200 MB");
  assert.equal(rowFor(rows, "in flight").value, "5");
  assert.equal(rowFor(rows, "tiles").value, "312 res / 9 pend");
  resetMetrics();
});

test("in-flight above the streaming cap warns", () => {
  resetMetrics();
  setGauge(METRIC.IN_FLIGHT, 17); // STREAM.MAX_INFLIGHT is 8; 16 is 2× headroom
  assert.equal(rowFor(hudRows(baseInput()), "in flight").level, "warn");
  resetMetrics();
});

test("callback errors surface as their own red row, and only when nonzero", () => {
  resetMetrics();
  assert.equal(
    hudRows(baseInput()).some((r) => r.label === "cb errors"),
    false,
  );
  const row = rowFor(hudRows(baseInput({ errorCount: 3 })), "cb errors");
  assert.equal(row.value, "3");
  assert.equal(row.level, "bad");
});

test("a paused loop is visible in the HUD (a stopped loop looks like a fast one)", () => {
  resetMetrics();
  const row = rowFor(hudRows(baseInput({ paused: true })), "callbacks");
  assert.match(row.value, /paused/);
  assert.equal(row.level, "warn");
});

// ── sparkline ───────────────────────────────────────────────────────────────

test("sparkline maps frame times onto block glyphs and bounds its width", () => {
  assert.equal(sparkline([]), "");
  assert.equal(sparkline([0]), "▁");
  assert.equal(sparkline([999]), "█", "over-range clamps to full height");
  assert.equal(sparkline(new Array(200).fill(8)).length, HUD_SPARK_SAMPLES);
  const s = sparkline([0, 50]);
  assert.equal(s.length, 2);
  assert.ok(s[1] > s[0], "a slower frame draws taller");
});

// ── metrics store ───────────────────────────────────────────────────────────

test("gauges are last-write-wins; addGauge accumulates; resetGauge zeroes", () => {
  resetMetrics();
  setGauge("x", 5);
  assert.equal(getGauge("x"), 5);
  addGauge("x", 3);
  assert.equal(getGauge("x"), 8);
  resetGauge("x");
  assert.equal(getGauge("x"), 0);
  assert.equal(getGauge("never-set"), 0);
});

test("counters are monotonic and independent of gauges", () => {
  resetMetrics();
  bump("c");
  bump("c", 4);
  assert.equal(getCounter("c"), 5);
  assert.equal(getGauge("c"), 0, "a counter must not shadow a gauge of the same name");
  const snap = snapshot();
  assert.equal(snap.counters.c, 5);
  assert.equal(snap.gauges.c, undefined);
  resetMetrics();
  assert.equal(getCounter("c"), 0, "resetMetrics clears both stores for the leak harness");
});

// ── VRAM accounting ─────────────────────────────────────────────────────────

test("textureBytes matches the tile-budget arithmetic in the work order", () => {
  // 512² ETC1S at 4bpp = 128 KB; ×4/3 with the mip chain ≈ 175 KB.
  const flat = textureBytes(512, 512, "compressed4bpp");
  assert.equal(flat, 128 * 1024);
  const mipped = textureBytes(512, 512, "compressed4bpp", true);
  assert.ok(Math.abs(mipped - 174762) < 2, `${mipped} bytes`);
  // And the headline number: a 256MB mobile budget holds ~1,500 such tiles.
  assert.equal(Math.floor((256 * 1024 * 1024) / mipped), 1536);
});

test("textureBytes: RGBA8 is 8× a 4bpp compressed texture — the reason Law II.7 exists", () => {
  assert.equal(textureBytes(512, 512, "rgba8") / textureBytes(512, 512, "compressed4bpp"), 8);
  assert.equal(BYTES_PER_PIXEL.rgba8, 4);
  assert.equal(textureBytes(0, 512), 0);
  assert.equal(textureBytes(-1, -1), 0);
});

test("formatBytes is readable at every magnitude", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(-5), "0 B");
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(1536), "1.5 KB");
  assert.equal(formatBytes(200 * 1024 * 1024), "200 MB");
  assert.equal(formatBytes(2 * 1024 ** 3), "2.0 GB");
});

// ── mounting ────────────────────────────────────────────────────────────────

function headlessHost(): { host: FrameHost; frame(ms: number): void } {
  let clock = 0;
  let next = 1;
  const queue = new Map<number, (t: number) => void>();
  const scheduler: Scheduler = {
    request(cb) {
      const h = next++;
      queue.set(h, cb);
      return h;
    },
    cancel(h) {
      queue.delete(h);
    },
  };
  return {
    host: { now: () => clock, scheduler, visibility: { isHidden: () => false, subscribe: () => () => {} } },
    frame(ms) {
      clock += ms;
      const es = [...queue.values()];
      queue.clear();
      for (const cb of es) cb(clock);
    },
  };
}

test("mountPerfHud is a safe no-op with no DOM and when not requested", () => {
  const h = headlessHost();
  const loop = new FrameLoop(() => h.host);
  const handle = mountPerfHud(loop, { enabled: true }); // no document in node
  assert.equal(handle.element, null);
  assert.equal(loop.registrationCount, 0, "a disabled HUD must not register a frame callback");
  handle.dispose(); // must not throw
  const off = mountPerfHud(loop, { enabled: false });
  assert.equal(off.element, null);
  off.dispose();
  loop.dispose();
});
