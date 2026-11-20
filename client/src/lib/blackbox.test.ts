// blackbox.test — the recorder's whole value is that it works when the page is
// killed without running any code, so the tests simulate exactly that: a boot
// that never calls bootComplete(), followed by another bootBegin().
import test from "node:test";
import assert from "node:assert/strict";
import {
  bootBegin, mark, bootComplete, shouldRunSafe, lastCrashReport, resetAll,
  heartbeat, closeCleanly, standDown, armBoot,
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

// ── 2026-11-19: the marker is a statement about a MAP start ───────────────────
// Measured on production: a visit to pricing, to the AI Engine page or to the
// landing page, left normally, put the NEXT visit's map in reduced mode ("the
// last visit ended during startup"). The module that arms the marker runs on
// every page; only the mounted map ever disarmed it.

test("a visit that never starts the map does not accuse the next one", () => {
  const s = mem();
  const r1 = bootBegin(s, 1000, "b1");          // landing page: the module arms the marker
  assert.equal(r1.prevCrashed, false);
  assert.ok(s.getItem(KEYS.IN_PROGRESS), "armed at module evaluation, as before");
  standDown(s);                                  // page hidden / start-up window over, no map mounted
  assert.equal(s.getItem(KEYS.IN_PROGRESS), null);
  const r2 = bootBegin(s, 60_000, "b2");
  assert.equal(r2.prevCrashed, false, "a page that never started the map did not die during map startup");
  assert.equal(r2.consecutive, 0);
  assert.equal(shouldRunSafe(r2), false, "no reduced mode after a visit to another page");
  assert.equal(lastCrashReport(s), null, "and no crash report is fabricated for it");
});

test("opening the map later in the same visit arms the marker again: a death during THAT start is still caught", () => {
  const s = mem();
  bootBegin(s, 1000, "b1");
  standDown(s);                                  // 20 s on the AI Engine page
  armBoot(s, 1000, "b1");                        // then the map is opened inside the app
  mark(s, 1000, 31_000, "map-create");
  // ...renderer killed during the map's start: nothing runs
  const r2 = bootBegin(s, 90_000, "b2");
  assert.equal(r2.prevCrashed, true, "a real start-up death after a stand-down must be detected");
  assert.ok(shouldRunSafe(r2));
  assert.deepEqual(r2.prevTrail.map((c) => c.step), ["map-create"]);
});

test("armBoot does not replace a marker that is already armed (a direct load of the map)", () => {
  const s = mem();
  bootBegin(s, 1000, "b1");
  const before = s.getItem(KEYS.IN_PROGRESS);
  armBoot(s, 5000, "other");
  assert.equal(s.getItem(KEYS.IN_PROGRESS), before);
});

test("a healthy map start, then leaving the map inside the app, leaves nothing behind", () => {
  const s = mem();
  bootBegin(s, 1000, "b1");
  armBoot(s, 1000, "b1");
  bootComplete(s);                               // healthy
  heartbeat(s, 25_000, "healthy");               // still alive on the map
  closeCleanly(s);                               // the map unmounts: the user opened another tab of the app
  standDown(s);                                  // later the page is hidden, no map mounted
  const r2 = bootBegin(s, 90_000, "b2");
  assert.equal(r2.prevCrashed, false);
  assert.equal(r2.prevEndedAbruptly, false, "a heartbeat left by an unmounted map used to read as an abrupt end");
  assert.equal(shouldRunSafe(r2), false);
});

test("a REAL crash loop is not forgiven by a visit to another page in between", () => {
  const s = mem();
  bootBegin(s, 0, "b1");
  mark(s, 0, 600, "map-create");                 // dies during the map's start
  const r2 = bootBegin(s, 10_000, "b2");         // next visit: pricing only
  assert.ok(shouldRunSafe(r2), "b2 is owed a reduced start");
  standDown(s);                                  // ...and never starts the map
  const r3 = bootBegin(s, 20_000, "b3");         // the visit after that opens the map
  assert.equal(r3.prevCrashed, false, "b2 itself did not crash");
  assert.equal(r3.consecutive, 1, "the reduced start b2 never used is carried");
  assert.ok(shouldRunSafe(r3), "so the map still gets its reduced start after the real death");
  assert.ok(lastCrashReport(s), "and the report of that death is still readable");
  armBoot(s, 20_000, "b3");
  bootComplete(s);                               // the reduced start comes up healthy
  standDown(s);
  const r4 = bootBegin(s, 30_000, "b4");
  assert.equal(r4.consecutive, 0, "a healthy map start settles the debt");
  assert.equal(shouldRunSafe(r4), false);
});

test("a debt that was stood down and then settled in the SAME visit is not carried", () => {
  const s = mem();
  bootBegin(s, 0, "b1");                         // dies during the map's start
  bootBegin(s, 10_000, "b2");                    // owed a reduced start
  standDown(s);                                  // 20 s on another page: debt recorded
  assert.equal(s.getItem(KEYS.OWED), "1");
  armBoot(s, 10_000, "b2");                      // then the map is opened and comes up healthy
  bootComplete(s);
  assert.equal(s.getItem(KEYS.OWED), null);
  assert.equal(shouldRunSafe(bootBegin(s, 20_000, "b3")), false);
});

test("standDown and armBoot never throw on hostile storage; a corrupt debt reads as none", () => {
  const hostile: Storage = {
    getItem: () => { throw new Error("blocked"); },
    setItem: () => { throw new Error("full"); },
    removeItem: () => { throw new Error("blocked"); },
  };
  assert.doesNotThrow(() => { standDown(hostile); armBoot(hostile, 0, "x"); });
  const s = mem();
  s.setItem(KEYS.OWED, '"many"');
  assert.equal(bootBegin(s, 0, "b1").consecutive, 0);
  s.setItem(KEYS.OWED, "-3");
  assert.equal(bootBegin(s, 0, "b2").prevCrashed, true, "(b1 was never completed)");
});

test("SOURCE WIRING: every page is watched; the map arms the marker on mount and clears it on unmount", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(new URL("../pages/datamap.tsx", import.meta.url), "utf-8");
  const comp = src.indexOf("export default function DataMapPage()");
  assert.ok(comp > 0);
  const moduleLevel = src.slice(0, comp), component = src.slice(comp);
  // module level (runs on every page of the site): the page watcher, told whether the map is mounted
  assert.ok(moduleLevel.includes("watchBootMarker({ store: BOOT_STORE, mapMounted: () => BOOT_MAP_MOUNTED, windowMs: BOOT_HEALTHY_MS, win: window, doc: document });"),
    "the page watcher must be started for EVERY page, not only while the map is mounted");
  // the map: arm on mount, nothing left behind on unmount
  const at = component.indexOf("BOOT_MAP_MOUNTED = true;");
  assert.ok(at > 0, "the map never tells the watcher it is mounted");
  const fx = component.slice(at, at + 900);
  assert.ok(fx.includes("armBoot(BOOT_STORE, BOOT_STARTED_AT, BOOT_REPORT.bootId);"));
  assert.ok(fx.includes('bootComplete(BOOT_STORE); bmark("healthy"); }, BOOT_HEALTHY_MS)'));
  assert.match(fx, /BOOT_MAP_MOUNTED = false;\s+closeCleanly\(BOOT_STORE\);/);
  assert.ok(!component.includes('addEventListener("pagehide", clean)'), "the mounted-map-only pagehide listener came back");
});
