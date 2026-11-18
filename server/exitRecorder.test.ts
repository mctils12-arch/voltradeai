// REPAIR 2026-11-18: eight boots in 56 s during market hours and not one row
// saying why (server/exitRecorder.ts header). These tests replay that minute.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  createExitRecorder, parseState, describeUnclean, EXIT_RECORDER_FILE, UNCLEAN_EXIT, RESTART_BURST,
  ALIVE_INTERVAL_MS, BURST_BOOTS, BURST_WINDOW_MS, KEEP_MS, UNCLEAN_MAX, RAILWAY_MAX_RETRIES,
  type RecorderFs, type MemorySample,
} from "./exitRecorder";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const T0 = Date.parse("2026-11-18T13:48:35.000Z");   // the deploy boot of v1.0.1463

function world() {
  const files = new Map<string, string>();
  const rows: Array<{ action: string; detail: string }> = [];
  const clock = { t: T0 };
  const mem: MemorySample = { heapUsedMB: 1500, rssMB: 2100, cgroupUsageMB: 2700 };
  const fsx: RecorderFs & { failWrites: boolean } = {
    failWrites: false,
    readFileSync(p) { const v = files.get(p); if (v === undefined) throw new Error("ENOENT"); return v; },
    writeFileSync(p, d) { if (this.failWrites) throw new Error("ENOSPC"); files.set(p, d); },
    renameSync(a, b) { const v = files.get(a); if (v === undefined) throw new Error("ENOENT"); files.set(b, v); files.delete(a); },
  };
  const boot = (version = "1.0.1463") => createExitRecorder({
    file: "/data/voltrade/" + EXIT_RECORDER_FILE, fs: fsx, now: () => clock.t,
    audit: (action, detail) => { rows.push({ action, detail }); }, sample: () => ({ ...mem }), version, pid: 1,
  });
  return { files, rows, clock, mem, fsx, boot };
}

test("a first boot on an empty volume records itself and says nothing", () => {
  const w = world();
  const r = w.boot();
  assert.deepEqual(r.begin(), { unclean: null, boots10m: 1, burst: false });
  assert.equal(w.rows.length, 0);
  const st = r.status();
  assert.equal(st.status, "ok");
  assert.equal(st.boots_24h, 1);
  assert.equal(st.unclean_24h, 0);
  assert.equal(st.gates_top_level_status, false);
  assert.equal(st.recorder_faults, 0);
});

test("a deploy (SIGTERM) is a clean end: the next boot writes no UNCLEAN-EXIT row", () => {
  const w = world();
  const a = w.boot("1.0.1462"); a.begin();
  w.clock.t += 3600_000; a.tick();
  a.end("SIGTERM");
  w.clock.t += 6_000;
  const b = w.boot("1.0.1463");
  assert.equal(b.begin().unclean, null);
  assert.equal(w.rows.length, 0);
  assert.equal(b.status().boots_24h, 2);
});

test("an uncaught exception already has its own audit row: marked as an end, not as a kill", () => {
  const w = world();
  const a = w.boot(); a.begin(); a.end("uncaught exception");
  w.clock.t += 5_000;
  assert.equal(w.boot().begin().unclean, null);
});

test("a killed process is reported by the next boot with when, how long, which stage and its memory", () => {
  const w = world();
  const a = w.boot(); a.begin(); a.stage("bot_initialized"); a.stage("serving");
  // one hour and two minutes of ticks, then nothing: the process was killed at ~14:51:10Z
  for (let i = 0; i < 375; i++) { w.clock.t += ALIVE_INTERVAL_MS; a.tick(); }
  const lastAlive = w.clock.t;
  w.clock.t += 7_000;
  const found = w.boot().begin().unclean;
  assert.ok(found);
  assert.equal(found.lastAliveAt, new Date(lastAlive).toISOString());
  assert.equal(found.livedS, 3750);
  assert.equal(found.goneS, 7);
  assert.equal(found.stage, "serving");
  assert.deepEqual(found.sample, { heapUsedMB: 1500, rssMB: 2100, cgroupUsageMB: 2700 });
  assert.equal(w.rows.length, 1);
  assert.equal(w.rows[0].action, UNCLEAN_EXIT);
  assert.match(w.rows[0].detail, /no shutdown signal and no recorded exception/);
  assert.match(w.rows[0].detail, /3750s after its boot, at stage "serving"/);
  assert.match(w.rows[0].detail, /heap 1500 MB, rss 2100 MB, container 2700 MB/);
  assert.match(w.rows[0].detail, /nothing here can name the cause; Railway's deploy log/);
});

test("2026-11-18 replay: eight boots in 56 s — every killed one is reported and the burst is named from the third boot", () => {
  const w = world();
  const first = w.boot(); first.begin(); first.stage("serving");
  w.clock.t = Date.parse("2026-11-18T14:51:10.000Z"); first.tick();
  // the eight boots the audit log shows; the first reached bot_initialized, the next six died ~3 s in
  const boots = ["14:51:17", "14:51:30", "14:51:36", "14:51:41", "14:51:47", "14:51:53", "14:52:00", "14:52:13"];
  let last = first;
  boots.forEach((hms, i) => {
    w.clock.t = Date.parse(`2026-11-18T${hms}.000Z`);
    last = w.boot();
    const res = last.begin();
    assert.ok(res.unclean, `boot ${i + 1} found the killed process before it`);
    assert.equal(res.boots10m, i + 1);                       // the 13:48 deploy boot is outside the window
    assert.equal(res.burst, i + 1 >= BURST_BOOTS);
    if (i === 0) { w.clock.t += 5_000; last.stage("bot_initialized"); }
  });
  const unclean = w.rows.filter((r) => r.action === UNCLEAN_EXIT);
  const burst = w.rows.filter((r) => r.action === RESTART_BURST);
  assert.equal(unclean.length, 8);
  assert.equal(burst.length, 6);                              // boots 3..8
  assert.match(unclean[1].detail, /5s after its boot, at stage "bot_initialized"/);   // boot 1 died after init
  assert.match(unclean[2].detail, /0s after its boot, at stage "boot"/);              // boot 2 never got further
  assert.match(burst[5].detail, /^8 boots in the last 10 minutes \(8 of them after a process that was killed\)/);
  assert.match(burst[5].detail, new RegExp(`stops restarting after ${RAILWAY_MAX_RETRIES} failures in a row`));
  const st = last.status();
  assert.equal(st.status, "warning");
  assert.equal(st.burst, true);
  assert.equal(st.boots_10m, 8);
  assert.equal(st.unclean_24h, 8);
  assert.match(String(st.detail), /^8 boots in the last 10 minutes/);
  // eleven minutes later the burst is over but the kills still show for 24 h
  w.clock.t += 11 * 60_000; last.tick();
  const later = last.status();
  assert.equal(later.burst, false);
  assert.equal(later.status, "warning");
  assert.match(String(later.detail), /8 process\(es\) killed without a shutdown signal in the last 24h/);
  w.clock.t += KEEP_MS;
  assert.equal(last.status().status, "ok");
});

test("a night of deploys 14 minutes apart is not a burst", () => {
  const w = world();
  let r = w.boot(); r.begin();
  for (let i = 0; i < 18; i++) {
    r.end("SIGTERM"); w.clock.t += 14 * 60_000;
    r = w.boot(`1.0.${1440 + i}`);
    const res = r.begin();
    assert.equal(res.burst, false);
    assert.equal(res.unclean, null);
  }
  assert.equal(w.rows.length, 0);
  assert.equal(r.status().status, "ok");
  assert.equal(r.status().boots_24h, 19);
});

test("the history is bounded and old entries age out", () => {
  const w = world();
  let r = w.boot(); r.begin();
  for (let i = 0; i < UNCLEAN_MAX + 15; i++) { w.clock.t += BURST_WINDOW_MS + 1000; r = w.boot(); r.begin(); }
  const st = parseState(w.files.get("/data/voltrade/" + EXIT_RECORDER_FILE) ?? null);
  assert.equal(st.unclean.length, UNCLEAN_MAX);
  assert.ok(st.boots.length <= 200);
  w.clock.t += KEEP_MS + 1000;
  r = w.boot(); r.begin();
  const after = parseState(w.files.get("/data/voltrade/" + EXIT_RECORDER_FILE) ?? null);
  assert.equal(after.boots.length, 1);
  assert.equal(after.unclean.length, 1);                      // only the one this boot just found
});

test("a corrupt or foreign file is no record, and a failing disk is counted, never thrown", () => {
  const w = world();
  w.files.set("/data/voltrade/" + EXIT_RECORDER_FILE, "{not json");
  const r = w.boot();
  assert.deepEqual(r.begin(), { unclean: null, boots10m: 1, burst: false });
  assert.deepEqual(parseState('{"boot":{"at":"nonsense"},"boots":[1,"x"],"unclean":"no"}'), { v: 1, boot: null, boots: [], unclean: [] });
  w.fsx.failWrites = true;
  assert.doesNotThrow(() => { r.tick(); r.stage("serving"); r.end("SIGTERM"); });
  assert.equal(r.status().recorder_faults, 3);
  const thrower = createExitRecorder({
    file: "/x", fs: w.fsx, now: () => w.clock.t, audit: () => { throw new Error("db locked"); },
    sample: () => { throw new Error("no /proc"); }, version: "v", pid: 1,
  });
  assert.doesNotThrow(() => thrower.begin());
});

test("the row never guesses a cause", () => {
  const txt = describeUnclean({ bootAt: "a", version: "1", lastAliveAt: "b", livedS: 1, goneS: 2, stage: "boot", sample: null, detectedAt: "c" });
  assert.match(txt, /heap unknown, rss unknown, container unknown/);
  assert.doesNotMatch(txt, /out of memory|OOM|crash loop/i);
});

test("wiring: the orchestrator begins the recorder at boot, marks a requested stop first, and shows it on /api/health without gating", () => {
  const bot = fs.readFileSync(path.join(HERE, "bot.ts"), "utf8");
  assert.match(bot, /installCrashHandlers\(process, \{ audit \}\);[\s\S]{0,500}exitRecorder\.begin\(\);/);
  assert.match(bot, /process\.on\("uncaughtException", \(\) => exitRecorder\.end\("uncaught exception"\)\);/);
  assert.match(bot, /setInterval\(\(\) => exitRecorder\.tick\(\), ALIVE_INTERVAL_MS\)\.unref\(\);/);
  assert.match(bot, /async function gracefulShutdown\(signal: string\) \{\s*exitRecorder\.end\(signal\);/);
  assert.match(bot, /checks\.checks\.restarts = exitRecorder\.status\(\);/);
  const gate = fs.readFileSync(path.join(HERE, "healthGate.ts"), "utf8");
  assert.doesNotMatch(gate, /restarts/);                      // never a serving gate
});
