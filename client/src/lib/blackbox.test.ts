// blackbox.test — the recorder's whole value is that it works when the page is
// killed without running any code, so the tests simulate exactly that: a boot
// that never calls bootComplete(), followed by another bootBegin().
import test from "node:test";
import assert from "node:assert/strict";
import {
  bootBegin, mark, bootComplete, shouldRunSafe, lastCrashReport, resetAll,
  heartbeat, closeCleanly,
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

test("a LATE death (healthy boot, then killed) is persisted with its last step and trail", () => {
  // the live gap this covers: a plane-click crash after a healthy boot left
  // report:null in the copied payload — the heartbeat saw it, nothing kept it
  const s = mem();
  bootBegin(s, 0, "b1");
  mark(s, 0, 500, "map-created");
  bootComplete(s);                                  // healthy
  mark(s, 0, 90_000, "plane-select", { icao: "8681b2" });
  heartbeat(s, 93_000, "plane-select");             // last beat before the kill
  // ...renderer killed here: no event, no handler, no pagehide
  const r = bootBegin(s, 200_000, "b2");
  assert.equal(r.prevCrashed, false, "the boot itself had completed");
  assert.equal(r.prevEndedAbruptly, true, "but the session never closed cleanly");
  assert.equal(r.prevAliveMs, 93_000);
  assert.equal(r.consecutive, 0, "late death must NOT trigger safe mode");
  assert.equal(shouldRunSafe(r), false);
  const rep = lastCrashReport(s) as { kind: string; aliveMs: number; lastStep: string; trail: { step: string }[] };
  assert.ok(rep, "the record is retrievable for the Copy button");
  assert.equal(rep.kind, "abrupt-end");
  assert.equal(rep.aliveMs, 93_000);
  assert.equal(rep.lastStep, "plane-select", "the last thing it did survives");
  assert.deepEqual(rep.trail.map((c) => c.step), ["map-created", "plane-select"],
    "the dead session's full breadcrumb trail survives");
});

test("a clean close leaves NO abrupt-end record", () => {
  const s = mem();
  bootBegin(s, 0, "b1");
  bootComplete(s);
  heartbeat(s, 5_000, "idle");
  closeCleanly(s);                                   // real navigation/close
  const r = bootBegin(s, 10_000, "b2");
  assert.equal(r.prevEndedAbruptly, false);
  assert.equal(lastCrashReport(s), null, "no phantom crash report");
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

test("closing the tab DURING startup is a navigation, not a crash (2026-08-08 field false positive)", () => {
  // the human's payload: healthy trail to first-idle, tab closed at 10.0s —
  // a hair before the ~10.2s healthy-idle mark — and the next visit booted
  // into safe mode with a crash banner. pagehide (→ closeCleanly) must
  // disarm the boot marker: real crashes never fire pagehide.
  const s = mem();
  bootBegin(s, 1000, "b1");
  mark(s, 1000, 3200, "first-idle");
  closeCleanly(s); // pagehide before bootComplete
  const r2 = bootBegin(s, 60_000, "b2");
  assert.equal(r2.prevCrashed, false, "graceful exit before the healthy mark is not a crash");
  assert.equal(r2.consecutive, 0);
  assert.equal(shouldRunSafe(r2), false, "no safe mode after a deliberate close");
});

test("a REAL startup death (no pagehide) is still detected after the false-positive fix", () => {
  const s = mem();
  bootBegin(s, 1000, "b1");
  mark(s, 1000, 1600, "map-create");
  // no closeCleanly, no bootComplete: GPU death / OOM kill runs nothing
  const r2 = bootBegin(s, 60_000, "b2");
  assert.equal(r2.prevCrashed, true, "real crashes must keep being detected");
  assert.equal(r2.consecutive, 1);
});

test("SOURCE RATCHET: the crash popup stays deleted; the silent capture stays wired (human directive 2026-08-08)", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(new URL("../pages/datamap.tsx", import.meta.url), "utf-8");
  assert.ok(!src.includes("Copy crash report"), "the startup crash popup came back — the human asked for it gone");
  assert.ok(!src.includes("Recovered from a crash"), "crash-banner JSX reappeared");
  assert.ok(src.includes('window.localStorage.setItem("vt-last-crash-payload"'),
    "silent crash-payload capture removed — the report must stay one paste away");
  assert.ok(src.includes("[VT CRASH REPORT]"), "console capture removed");
});
