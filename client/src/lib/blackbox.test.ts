// blackbox.test — the recorder's whole value is that it works when the page is
// killed without running any code, so the tests simulate exactly that: a boot
// that never calls bootComplete(), followed by another bootBegin().
import test from "node:test";
import assert from "node:assert/strict";
import {
  bootBegin, mark, bootComplete, shouldRunSafe, lastCrashReport, resetAll,
  TRAIL_MAX, KEYS, type Storage,
} from "./blackbox.ts";

function mem(): Storage & { dump: () => Record<string, string> } {
  const m = new Map<string, string>();
  return {
    getItem: (k) => (m.has(k) ? m.get(k)! : null),
    setItem: (k, v) => void m.set(k, v),
    removeItem: (k) => void m.delete(k),
    dump: () => Object.fromEntries(m),
  };
}

test("a clean boot leaves no crash signal for the next one", () => {
  const s = mem();
  const r1 = bootBegin(s, 1000, "b1");
  assert.equal(r1.prevCrashed, false);
  mark(s, 1000, 1100, "map-created");
  bootComplete(s);
  const r2 = bootBegin(s, 5000, "b2");
  assert.equal(r2.prevCrashed, false, "a completed boot must not look like a crash");
  assert.equal(r2.consecutive, 0);
  assert.equal(shouldRunSafe(r2), false, "no safe mode after a healthy run");
});

test("a boot killed WITHOUT running any code is detected by the next boot", () => {
  const s = mem();
  bootBegin(s, 1000, "b1");
  mark(s, 1000, 1200, "map-created");
  mark(s, 1000, 3400, "space-enter", { zoom: -0.06 });
  mark(s, 1000, 4900, "hold-zoom-out");
  // ...process killed here. No bootComplete(), no handler, nothing.
  const r = bootBegin(s, 9000, "b2");
  assert.equal(r.prevCrashed, true, "the marker left behind IS the crash signal");
  assert.equal(r.prevSurvivedMs, 8000, "survival time = now - previous startedAt");
  assert.equal(r.consecutive, 1);
  assert.ok(shouldRunSafe(r), "one dead boot is enough to stop trusting the config");
  // the evidence survives, in order, with the last action last
  assert.deepEqual(r.prevTrail.map((c) => c.step), ["map-created", "space-enter", "hold-zoom-out"]);
  assert.equal(r.prevTrail[1].zoom, -0.06, "breadcrumb payloads survive");
  const rep = lastCrashReport(s) as { survivedMs: number; trail: unknown[] };
  assert.equal(rep.survivedMs, 8000);
  assert.equal(rep.trail.length, 3, "report is retrievable for display/copy");
});

test("consecutive dead boots are counted — a loop is distinguishable from a one-off", () => {
  const s = mem();
  bootBegin(s, 0, "b1");                       // dies
  const r2 = bootBegin(s, 100, "b2");          // dies
  assert.equal(r2.consecutive, 1);
  const r3 = bootBegin(s, 200, "b3");
  assert.equal(r3.consecutive, 2, "second dead boot in a row");
  const r4 = bootBegin(s, 300, "b4");
  assert.equal(r4.consecutive, 3);
  // one healthy boot resets the streak
  bootComplete(s);
  const r5 = bootBegin(s, 400, "b5");
  assert.equal(r5.consecutive, 0, "recovery clears the streak");
  assert.equal(shouldRunSafe(r5), false);
});

test("this boot's trail starts empty, so a crash trail is never mixed with a live one", () => {
  const s = mem();
  bootBegin(s, 0, "b1");
  mark(s, 0, 10, "a");
  const r = bootBegin(s, 100, "b2");
  assert.deepEqual(r.prevTrail.map((c) => c.step), ["a"]);
  assert.deepEqual(JSON.parse(s.getItem(KEYS.TRAIL)!), [], "fresh boot, fresh trail");
});

test("the trail is ring-buffered and keeps the MOST RECENT steps", () => {
  const s = mem();
  bootBegin(s, 0, "b1");
  for (let i = 0; i < TRAIL_MAX + 25; i++) mark(s, 0, i, `step${i}`);
  const trail = JSON.parse(s.getItem(KEYS.TRAIL)!) as { step: string }[];
  assert.equal(trail.length, TRAIL_MAX, "bounded — it is written to storage constantly");
  assert.equal(trail[trail.length - 1].step, `step${TRAIL_MAX + 24}`,
    "the last thing before the crash is what matters, so keep the tail");
});

test("the recorder never throws, even when storage is hostile", () => {
  const broken: Storage = {
    getItem: () => { throw new Error("blocked"); },
    setItem: () => { throw new Error("full"); },
    removeItem: () => { throw new Error("nope"); },
  };
  // a recorder that throws inside a failing page destroys the evidence it exists
  // to collect, so every entry point must be inert under a hostile store
  assert.doesNotThrow(() => {
    const r = bootBegin(broken, 0, "b1");
    assert.equal(r.prevCrashed, false);
    mark(broken, 0, 1, "x");
    bootComplete(broken);
    shouldRunSafe(r);
    lastCrashReport(broken);
    resetAll(broken, ["vt-map-globe"]);
  });
});

test("resetAll clears recorder state AND the caller's preference keys", () => {
  const s = mem();
  bootBegin(s, 0, "b1");
  mark(s, 0, 5, "x");
  s.setItem("vt-map-globe", "globe");
  s.setItem("vt-terrain-exag", "3.0");
  resetAll(s, ["vt-map-globe", "vt-terrain-exag"]);
  for (const k of [KEYS.IN_PROGRESS, KEYS.TRAIL, KEYS.PREV, KEYS.STREAK, "vt-map-globe", "vt-terrain-exag"]) {
    assert.equal(s.getItem(k), null, `${k} cleared`);
  }
});
