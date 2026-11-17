// The tracked tails' permanent, capped flight log (owner's instruction
// 2026-11-17: the bulk log of every aircraft is off, watched planes keep a
// full log, and that log has a cap). server/trackedPlaneLog.ts.
//   - every fix of a tracked plane is written ONCE; the dedupe and age rules;
//   - one file per plane per UTC month; a month rolls over on the fix's time;
//   - past months are gzipped, a late raw file is merged, nothing is lost;
//   - the per-plane cap and the total cap delete the OLDEST month first and
//     never the current month; a current month at the cap pauses the writer;
//   - low disk fails closed; the kill switch; untracked planes write nothing;
//   - a removed tail's log stays and still counts toward the total;
//   - the status block, the bus subscription, the restart rule, the wiring.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";
import { mapPointAircraft } from "./aircraftTiling";
import { publishFixes, type FixBatch } from "./aircraftFixBus";
import type { AircraftPoint } from "./datacoreArchive";
import type { TrackedPlane } from "./trackedPlanes";
import { fixFromLine } from "./aircraftTrackData";
import {
  TrackedPlaneLog, startTrackedPlaneLog, scanTrackedLog, readTrackedLogRows, trackedLogSummary, compressLogMonth,
  lateLinesNotHeld, trackedLogCaps, trackedLogEnabled, trackedLogDir, monthBoundsMs, _resetTrackedLogRangeCache,
  TRACKED_LOG_DIR, TRACKED_LOG_MAX_BYTES_PER_PLANE, TRACKED_LOG_MAX_BYTES_TOTAL, TRACKED_LOG_MIN_FREE_BYTES,
  TRACKED_LOG_NEW_FIX_MIN_MS, TRACKED_LOG_MAX_FIX_AGE_MS, TRACKED_LOG_MAX_FIX_AHEAD_MS,
} from "./trackedPlaneLog";

const here = path.dirname(fileURLToPath(import.meta.url));
const GiB = 1_073_741_824;
const MiB = 1024 * 1024;
/** 2026-11-17 15:00:10 UTC — a whole second, mid-month */
const T0 = Date.parse("2026-11-17T15:00:10Z");
const tmp = (tag = "log"): string => fs.mkdtempSync(path.join(os.tmpdir(), `vt-trackedlog-${tag}-`));

type Raw = Record<string, unknown>;
/** one readsb row with the fields adsb.lol sends for an airborne ADS-B aircraft */
const readsb = (hex: string, o: Raw = {}): Raw => ({
  hex, type: "adsb_icao", flight: "N843S   ", r: "N843S", t: "FA7X", alt_baro: 34000, alt_geom: 34525, gs: 459.3, track: 268.4,
  baro_rate: 1280, squawk: "4521", category: "A3", lat: 36.1234, lon: -83.9876, nic: 8, rc: 186, seen_pos: 1, version: 2,
  nic_baro: 1, nac_p: 10, nac_v: 2, sil: 3, sil_type: "perhour", gva: 2, sda: 2, mlat: [], tisb: [], messages: 51234, seen: 0.2, rssi: -9.1, ...o,
});
const mapped = (...rows: Raw[]): AircraftPoint[] => mapPointAircraft({ ac: rows }, "ac", "adsblol") as AircraftPoint[];
const plane = (reg: string, hex: string | null): TrackedPlane => ({ reg, added_at: T0 - 86_400_000, hex });

const A = "ab8c8e", B = "a42dee", C = "a0cafe";

function harness(o: {
  tracked?: TrackedPlane[]; env?: Record<string, string>; perPlane?: number; total?: number; free?: number | null; base?: string; t?: number;
} = {}) {
  const base = o.base ?? tmp();
  const clock = { t: o.t ?? T0 };
  const free = { v: o.free === undefined ? 50 * GiB : o.free };
  const tracked = { list: o.tracked ?? [plane("N843S", A)] };
  const log = new TrackedPlaneLog({
    baseDir: base, now: () => clock.t, env: (o.env ?? {}) as NodeJS.ProcessEnv, readFree: () => free.v,
    tracked: () => tracked.list, maxBytesPerPlane: o.perPlane, maxBytesTotal: o.total,
  });
  /** offer one fix received at `atMs` (its own time is 1 s earlier unless the row says otherwise) */
  const offer = (hex: string, atMs: number = clock.t, row: Raw = {}, b: Partial<FixBatch> = {}): number =>
    log.ingest({ provider: "adsblol", origin: "tracked", aircraft: mapped(readsb(hex, row)), fetchedAt: atMs, upstreamNowMs: atMs, ...b });
  /** `n` fixes two seconds apart, the clock moving with them */
  const fly = (hex: string, n: number): number => {
    let w = 0;
    for (let k = 0; k < n; k++) { clock.t += 2_000; w += offer(hex); }
    return w;
  };
  return { base, clock, free, tracked, log, offer, fly };
}

const monthPath = (base: string, hex: string, month: string, gz = false): string =>
  path.join(base, "aircraft_tracked", hex.toUpperCase(), `${month}.jsonl${gz ? ".gz" : ""}`);
function linesOf(fp: string): Raw[] {
  if (!fs.existsSync(fp)) return [];
  const buf = fs.readFileSync(fp);
  const text = fp.endsWith(".gz") ? zlib.gunzipSync(buf).toString() : buf.toString();
  return text.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l) as Raw);
}
const monthsOnDisk = (base: string, hex: string): string[] => {
  const dir = trackedLogDir(base, hex);
  return fs.existsSync(dir) ? fs.readdirSync(dir).sort() : [];
};

test("a tracked plane's fix is written ONCE, at its own time, in the shared archive's line format, with the provider", () => {
  const h = harness();
  assert.equal(h.offer(A), 1, "the tracked poller delivers the fix");
  assert.equal(h.offer(A), 0, "the viewport delivers the same fix again: not written twice");
  assert.equal(h.offer(A, T0, {}, { origin: "sweep-disc" }), 0, "... nor a third time from the world sweep");
  const fp = monthPath(h.base, A, "2026-11");
  const rows = linesOf(fp);
  assert.equal(rows.length, 1);
  const r = rows[0];
  assert.equal(r.t, T0 / 1000 - 1, "t is the fix's own time: provider clock minus the 1 s position age");
  assert.equal(r.ft, 1, "and the line says so");
  assert.equal(r.i, A);
  assert.equal(r.pv, "adsblol", "the provider that served the row");
  assert.equal(r.rg, "N843S");
  assert.equal(r.c, "N843S");
  assert.equal(r.al, 10363);
  assert.equal(r.ag, 10523, "geometric altitude, m");
  assert.equal(r.ak, "b");
  assert.equal(r.vr, 6.5);
  assert.equal(r.vs, "b");
  assert.equal(r.x, undefined, "no delta-encoded x block: every line stands alone");
  // the track reader's own decoder reads it
  const f = fixFromLine(r, null)!;
  assert.equal(f.t, T0 / 1000 - 1);
  assert.equal(f.t_src, "fix");
  assert.equal(f.alt_geom_m, 10523);
  assert.equal(f.alt_baro_m, 10363);
  assert.equal(f.vr_kind, "barometric");
  assert.equal(f.pv, "adsblol");
  assert.equal(path.basename(path.dirname(fp)), "AB8C8E", "one directory per aircraft, hex in upper case");
  assert.equal(h.log.counters.fixes_offered_total, 3);
  assert.equal(h.log.counters.lines_written_total, 1);
  assert.equal(h.log.counters.skipped_same_fix_total, 2);
  assert.equal(h.log.counters.bytes_written_total, fs.statSync(fp).size);
});

test("dedupe and age rules: at least 1 s newer, at most 10 min old, at most 60 s ahead; a late older fix is not written", () => {
  assert.equal(TRACKED_LOG_NEW_FIX_MIN_MS, 1_000);
  assert.equal(TRACKED_LOG_MAX_FIX_AGE_MS, 600_000);
  assert.equal(TRACKED_LOG_MAX_FIX_AHEAD_MS, 60_000);
  const h = harness({ tracked: [plane("N843S", A), plane("N786QS", B)] });
  assert.equal(h.offer(A, T0), 1);
  assert.equal(h.offer(A, T0 + 999), 0, "0.999 s newer: the same fix seen through a slightly different clock");
  assert.equal(h.offer(A, T0 + 1_000), 1, "1 s newer: a new fix");
  assert.equal(h.offer(A, T0 - 30_000), 0, "an older fix arriving late through a slower path");
  assert.equal(h.log.counters.skipped_same_fix_total, 2);
  // age: the fix's own time, not our receive time
  assert.equal(h.offer(B, T0, { seen_pos: 601 }), 0, "a 10 min 1 s old position");
  assert.equal(h.log.counters.skipped_stale_total, 1);
  assert.equal(h.offer(B, T0, { seen_pos: 599 }), 1, "9 min 59 s old: written, at its own time");
  assert.equal(linesOf(monthPath(h.base, B, "2026-11"))[0].t, T0 / 1000 - 599);
  assert.equal(h.offer(B, T0 + 61_500), 0, "a provider clock 60.5 s ahead of ours");
  assert.equal(h.log.counters.skipped_future_total, 1);
  assert.equal(h.offer(B, T0 + 59_000), 1);
  assert.equal(linesOf(monthPath(h.base, A, "2026-11")).length, 2);
  assert.equal(linesOf(monthPath(h.base, B, "2026-11")).length, 2);
});

test("a row with no position age is written at our receive time, labelled so; the batch's provider fills a missing one", () => {
  const h = harness();
  const [p] = mapped(readsb(A, { seen_pos: undefined }));
  p.provider = null;
  assert.equal(h.log.ingest({ provider: "adsbfi", origin: "viewport", aircraft: [p], fetchedAt: T0 }), 1);
  const [r] = linesOf(monthPath(h.base, A, "2026-11"));
  assert.equal(r.t, T0 / 1000);
  assert.equal(r.ft, undefined, "not the fix's own time: no ft");
  assert.equal(r.pv, "adsbfi", "the provider travels with the batch");
  assert.equal(fixFromLine(r, null)!.t_src, "unadjusted");
  assert.equal(h.log.counters.lines_receive_time_total, 1);
});

test("one file per plane per UTC month; the month is the FIX'S month, so the rollover is exact", () => {
  const h = harness({ tracked: [plane("N843S", A), plane("N786QS", B)], t: Date.parse("2026-11-30T23:59:58Z") });
  assert.equal(h.offer(A), 1); // fix at 23:59:57
  h.clock.t = Date.parse("2026-12-01T00:00:03Z");
  assert.equal(h.offer(A), 1); // fix at 00:00:02
  // plane B: received in December, but the fix itself is from November (10 s old)
  assert.equal(h.offer(B, h.clock.t, { seen_pos: 10 }), 1);
  h.clock.t += 5_000;
  assert.equal(h.offer(B), 1);
  assert.deepEqual(monthsOnDisk(h.base, A), ["2026-11.jsonl", "2026-12.jsonl"]);
  assert.deepEqual(monthsOnDisk(h.base, B), ["2026-11.jsonl", "2026-12.jsonl"]);
  assert.deepEqual(linesOf(monthPath(h.base, A, "2026-11")).map((r) => r.t), [Date.parse("2026-11-30T23:59:57Z") / 1000]);
  assert.deepEqual(linesOf(monthPath(h.base, A, "2026-12")).map((r) => r.t), [Date.parse("2026-12-01T00:00:02Z") / 1000]);
  assert.deepEqual(linesOf(monthPath(h.base, B, "2026-11")).map((r) => r.t), [Date.parse("2026-11-30T23:59:53Z") / 1000]);
  assert.deepEqual(monthBoundsMs("2026-12"), { start: Date.parse("2026-12-01T00:00:00Z"), end: Date.parse("2027-01-01T00:00:00Z") });
  const inv = scanTrackedLog(h.base);
  assert.deepEqual(inv.get(A)?.months.map((m) => m.month), ["2026-11", "2026-12"]);
  assert.equal(inv.get(A)?.bytes, fs.statSync(monthPath(h.base, A, "2026-11")).size + fs.statSync(monthPath(h.base, A, "2026-12")).size);
});

test("no thinning: a parked aircraft's fixes and a straight cruise are all kept, one line a fix", () => {
  const h = harness();
  let n = 0;
  for (let k = 0; k < 40; k++) { h.clock.t += 1_000; n += h.offer(A, h.clock.t, k < 20 ? { alt_baro: "ground", gs: 0 } : {}); }
  assert.equal(n, 40, "the shared archive would keep one ground fix per 5 min and one cruise fix per 75 s");
  assert.equal(linesOf(monthPath(h.base, A, "2026-11")).length, 40);
});

test("past months are gzipped an hour after they end; a late raw file is merged; no line is lost or doubled", async () => {
  const h = harness({ tracked: [plane("N843S", A), plane("N786QS", B)], t: Date.parse("2026-11-30T23:50:00Z") });
  assert.equal(h.fly(A, 3), 3);
  const novRaw = monthPath(h.base, A, "2026-11"), novGz = monthPath(h.base, A, "2026-11", true);
  const original = fs.readFileSync(novRaw, "utf8");
  h.clock.t = Date.parse("2026-12-01T00:20:00Z");
  assert.deepEqual(await h.log.maintain(), { compressed: 0, merged: 0, deleted: 0 }, "20 min after the month ended a fix could still land in it");
  assert.ok(fs.existsSync(novRaw));
  assert.equal(h.offer(A), 1); // December exists now
  h.clock.t = Date.parse("2026-12-01T01:00:01Z");
  assert.deepEqual(await h.log.maintain(), { compressed: 1, merged: 0, deleted: 0 });
  assert.equal(fs.existsSync(novRaw), false);
  assert.equal(zlib.gunzipSync(fs.readFileSync(novGz)).toString(), original, "byte for byte the lines that were written");
  assert.deepEqual(monthsOnDisk(h.base, A), ["2026-11.jsonl.gz", "2026-12.jsonl"], "the current month is never gzipped; no temp file left");

  // a LATE raw file beside the .gz: one line the .gz already holds (an interrupted pass), one new line
  const held = original.split("\n")[0];
  const fresh = JSON.stringify({ ...(JSON.parse(held) as Raw), t: Date.parse("2026-11-30T23:59:59Z") / 1000 });
  fs.writeFileSync(novRaw, `${held}\n${fresh}\n`);
  assert.deepEqual(await h.log.maintain(), { compressed: 0, merged: 1, deleted: 0 });
  assert.equal(fs.existsSync(novRaw), false);
  assert.equal(zlib.gunzipSync(fs.readFileSync(novGz)).toString(), `${original}${fresh}\n`, "the .gz's lines first, then the new one, once");
  assert.equal(h.log.counters.months_compressed_total, 1);
  assert.equal(h.log.counters.months_merged_total, 1);

  // a .gz that does not gunzip is never overwritten: both files stay, both are read
  fs.mkdirSync(trackedLogDir(h.base, B), { recursive: true });
  fs.writeFileSync(monthPath(h.base, B, "2026-10", true), "not gzip");
  fs.writeFileSync(monthPath(h.base, B, "2026-10"), `${held.replace(A, B)}\n`);
  assert.deepEqual(await h.log.maintain(), { compressed: 0, merged: 0, deleted: 0 });
  assert.deepEqual(monthsOnDisk(h.base, B), ["2026-10.jsonl", "2026-10.jsonl.gz"]);
  assert.equal(fs.readFileSync(monthPath(h.base, B, "2026-10", true), "utf8"), "not gzip");
  assert.equal(h.log.counters.months_left_beside_total, 1);

  // read-back: every line, oldest month first, through the gzip
  const got: number[] = [];
  const st = await readTrackedLogRows(h.base, A, {}, (row) => { got.push(row.t as number); });
  assert.equal(got.length, 5);
  assert.deepEqual(got, got.slice().sort((x, y) => x - y));
  assert.deepEqual({ months_held: st.months_held, files_read: st.files_read, files_failed: st.files_failed }, { months_held: 2, files_read: 2, files_failed: 0 });
});

test("gzip helpers: a merge too big for memory leaves both files; a raw file that grew meanwhile is kept for the next pass", async () => {
  const dir = path.join(tmp("gz"), "AB8C8E");
  fs.mkdirSync(dir, { recursive: true });
  const l1 = JSON.stringify({ t: 1, i: A }), l2 = JSON.stringify({ t: 2, i: A });
  assert.equal(lateLinesNotHeld(Buffer.from(`${l1}\n`), Buffer.from(`${l1}\n${l2}\n`)).toString(), `${l2}\n`);
  assert.equal(lateLinesNotHeld(Buffer.from(`${l1}\n${l2}\n`), Buffer.from(`${l1}\n`)).length, 0);
  fs.writeFileSync(path.join(dir, "2026-09.jsonl.gz"), zlib.gzipSync(`${l1}\n`));
  fs.writeFileSync(path.join(dir, "2026-09.jsonl"), `${l2}\n`);
  assert.equal(await compressLogMonth(dir, "2026-09", 4), "left_beside", "the existing .gz is over the merge limit");
  assert.deepEqual(fs.readdirSync(dir).sort(), ["2026-09.jsonl", "2026-09.jsonl.gz"]);
  assert.equal(await compressLogMonth(dir, "2026-09"), "merged");
  assert.equal(zlib.gunzipSync(fs.readFileSync(path.join(dir, "2026-09.jsonl.gz"))).toString(), `${l1}\n${l2}\n`);
  assert.deepEqual(fs.readdirSync(dir), ["2026-09.jsonl.gz"]);
  assert.equal(await compressLogMonth(dir, "2026-08"), "failed", "no such month: reported, nothing created");
  assert.deepEqual(fs.readdirSync(dir), ["2026-09.jsonl.gz"]);
});

test("per-plane cap: the OLDEST month goes first, the current month never; a current month at the cap pauses the writer", async () => {
  // three past months of 4 lines each, written by the real writer (uncapped)
  const base = tmp("cap");
  for (const m of ["2026-08", "2026-09", "2026-10"]) {
    const seed = harness({ base, t: Date.parse(`${m}-10T12:00:00Z`) });
    assert.equal(seed.fly(A, 4), 4);
  }
  const S = fs.statSync(monthPath(base, A, "2026-08")).size;
  const L = S / 4;
  assert.equal(fs.statSync(monthPath(base, A, "2026-10")).size, S, "every line is the same length in this fixture");
  const h = harness({ base, perPlane: 3 * S }); // exactly at the cap now
  assert.equal(h.fly(A, 1), 1, "one more line does not fit: August is deleted to make room");
  assert.deepEqual(monthsOnDisk(base, A), ["2026-09.jsonl", "2026-10.jsonl", "2026-11.jsonl"]);
  assert.equal(h.log.counters.months_deleted_plane_cap_total, 1);
  assert.equal(h.log.counters.bytes_deleted_total, S);
  assert.equal(h.fly(A, 3), 3, "back at the cap: 4 + 4 + 4 lines");
  assert.equal(h.fly(A, 4), 4, "September goes");
  assert.deepEqual(monthsOnDisk(base, A), ["2026-10.jsonl", "2026-11.jsonl"]);
  assert.equal(h.fly(A, 4), 4, "October goes");
  assert.deepEqual(monthsOnDisk(base, A), ["2026-11.jsonl"]);
  assert.equal(fs.statSync(monthPath(base, A, "2026-11")).size, 12 * L);
  // the current month alone is at the cap: writing pauses, the month is NOT deleted
  assert.equal(h.fly(A, 5), 0);
  assert.equal(h.log.counters.skipped_plane_cap_total, 5);
  assert.equal(h.log.counters.months_deleted_plane_cap_total, 3);
  assert.equal(linesOf(monthPath(base, A, "2026-11")).length, 12, "the current month is never deleted");
  assert.deepEqual(await h.log.maintain(), { compressed: 0, merged: 0, deleted: 0 });
  const paused = (await h.log.status()).by_reg.N843S;
  assert.equal(paused.paused, true);
  assert.match(paused.why || "", /current month alone is at the per-plane cap/);
  assert.equal(paused.paused_since_t, Math.round(h.clock.t / 1000));
  assert.equal(paused.cap_bytes, 3 * S);
  // the month turns: November can go now, and the plane is written again
  h.clock.t = Date.parse("2026-12-02T09:00:00Z");
  assert.equal(h.fly(A, 2), 2);
  assert.deepEqual(monthsOnDisk(base, A), ["2026-12.jsonl"]);
  const resumed = (await h.log.status()).by_reg.N843S;
  assert.equal(resumed.paused, false);
  assert.equal(resumed.why, null);
  assert.equal(resumed.months, 1);
});

test("total cap: the oldest month ACROSS planes goes first — a removed tail's log counts and can go; current months never", async () => {
  const base = tmp("total");
  // A: September; B: August; C (no longer tracked): July — 4 lines each
  for (const [hex, m] of [[A, "2026-09"], [B, "2026-08"], [C, "2026-07"]] as const) {
    const seed = harness({ base, tracked: [plane("X", hex)], t: Date.parse(`${m}-10T12:00:00Z`) });
    assert.equal(seed.fly(hex, 4), 4);
  }
  const S = fs.statSync(monthPath(base, A, "2026-09")).size;
  const h = harness({ base, tracked: [plane("N843S", A), plane("N786QS", B)], total: 3 * S, perPlane: 100 * S });
  const before = await h.log.status();
  assert.equal(before.total.bytes, 3 * S, "the removed tail's log is in the total");
  assert.deepEqual(before.total.untracked_logs.map((u) => [u.hex, u.months, u.bytes]), [[C, 1, S]]);
  assert.equal(before.total.cap_bytes_total, 3 * S);
  assert.equal(h.fly(A, 1), 1);
  assert.equal(fs.existsSync(trackedLogDir(base, C)), false, "July (the removed tail) was the oldest month anywhere: deleted, its empty directory removed");
  assert.equal(h.log.counters.months_deleted_total_cap_total, 1);
  assert.equal(h.fly(A, 3), 3);
  assert.equal(h.fly(B, 1), 1, "the next oldest is B's August");
  assert.deepEqual(monthsOnDisk(base, B), ["2026-11.jsonl"]);
  assert.deepEqual(monthsOnDisk(base, A), ["2026-09.jsonl", "2026-11.jsonl"]);
  assert.equal(h.fly(B, 3), 3);
  assert.equal(h.fly(B, 1), 1, "then A's September");
  assert.deepEqual(monthsOnDisk(base, A), ["2026-11.jsonl"]);
  assert.equal(h.fly(B, 3), 3, "4 + 8 lines of November: at the cap");
  // only current months are left and the total is reached: nothing more is written, nothing is deleted
  assert.equal(h.fly(A, 2), 0);
  assert.equal(h.fly(B, 2), 0);
  assert.equal(h.log.counters.skipped_total_cap_total, 4);
  assert.equal(h.log.counters.months_deleted_total_cap_total, 3);
  assert.equal(linesOf(monthPath(base, A, "2026-11")).length + linesOf(monthPath(base, B, "2026-11")).length, 12);
  const st = await h.log.status();
  assert.equal(st.total.bytes, 3 * S);
  assert.equal(st.total.over_total_cap, false);
  assert.match(st.by_reg.N786QS.why || "", /total cap/);
});

test("low disk fails CLOSED: under the shared 1 GiB floor, or unreadable, nothing is written and it is counted", async () => {
  assert.equal(TRACKED_LOG_MIN_FREE_BYTES, GiB);
  const h = harness({ free: GiB - 1 });
  assert.equal(h.fly(A, 2), 0);
  assert.equal(fs.existsSync(path.join(h.base, "aircraft_tracked")), false);
  assert.equal(h.log.counters.skipped_low_disk_total, 2);
  const st = await h.log.status();
  assert.equal(st.by_reg.N843S.paused, true);
  assert.match(st.by_reg.N843S.why || "", /free disk is under the 1 GiB floor/);
  assert.equal(st.total.free_bytes, GiB - 1);
  h.free.v = null; h.clock.t += 61_000; // the probe is re-read once a minute
  assert.equal(h.fly(A, 1), 0, "unreadable free space: closed, not open");
  assert.equal(h.log.counters.skipped_low_disk_total, 3);
  h.free.v = GiB; h.clock.t += 61_000;
  assert.equal(h.fly(A, 1), 1, "exactly the floor: written");
  assert.equal((await h.log.status()).by_reg.N843S.paused, false);
});

test("kill switch TRACKED_PLANE_LOG=0: the writer is off and counted; what is on disk still reads", async () => {
  assert.equal(trackedLogEnabled({} as NodeJS.ProcessEnv), true, "on unless switched off");
  for (const v of ["0", "false", "off", "no", " OFF "]) assert.equal(trackedLogEnabled({ TRACKED_PLANE_LOG: v } as NodeJS.ProcessEnv), false, v);
  for (const v of ["1", "true", "on", ""]) assert.equal(trackedLogEnabled({ TRACKED_PLANE_LOG: v } as NodeJS.ProcessEnv), true, v);
  const on = harness();
  assert.equal(on.fly(A, 2), 2);
  const off = harness({ base: on.base, env: { TRACKED_PLANE_LOG: "0" }, t: on.clock.t });
  assert.equal(off.fly(A, 3), 0);
  assert.equal(off.log.counters.skipped_disabled_total, 3);
  assert.equal(off.log.counters.lines_written_total, 0);
  assert.equal(linesOf(monthPath(on.base, A, "2026-11")).length, 2, "nothing added");
  let n = 0;
  await readTrackedLogRows(on.base, A, {}, () => { n++; });
  assert.equal(n, 2, "reads still work");
  const st = await off.log.status();
  assert.equal(st.enabled, false);
  assert.equal(st.by_reg.N843S.paused, true);
  assert.match(st.by_reg.N843S.why || "", /TRACKED_PLANE_LOG=0/);
  assert.equal(st.by_reg.N843S.months, 1);
});

test("untracked planes write nothing; a tracked tail whose hex is not known yet logs nothing, and that is counted", async () => {
  const h = harness({ tracked: [plane("N843S", A), plane("N786QS", null)] });
  assert.equal(h.offer(C), 0, "not on the tracked list");
  assert.equal(h.offer(B, T0, { r: "N999ZZ" }), 0);
  assert.equal(h.log.counters.fixes_offered_total, 0);
  assert.equal(h.log.counters.skipped_hex_unknown_total, 0);
  assert.equal(h.offer(B, T0, { r: "N786QS" }), 0, "the fix names a tracked tail, but the list has not learned its hex");
  assert.equal(h.log.counters.skipped_hex_unknown_total, 1);
  assert.equal(fs.existsSync(path.join(h.base, "aircraft_tracked")), false);
  const st = await h.log.status();
  assert.equal(st.by_reg.N786QS.hex, null);
  assert.equal(st.by_reg.N786QS.paused, true);
  assert.match(st.by_reg.N786QS.why || "", /hex of this tail is not known yet/);
  // the poller learns the hex (trackedPlanes.foldTrackedRows): from then on it is logged
  h.tracked.list = [plane("N843S", A), plane("N786QS", B.toUpperCase())];
  assert.equal(h.offer(B, T0 + 5_000, { r: "N786QS" }), 1);
  assert.deepEqual(monthsOnDisk(h.base, B), ["2026-11.jsonl"]);
  // an empty tracked list: nothing at all
  const none = harness({ tracked: [] });
  assert.equal(none.offer(A), 0);
  assert.equal(none.log.counters.fixes_offered_total, 0);
});

test("removing a tail keeps its log: nothing is deleted, nothing more is written, and it still counts", async () => {
  const h = harness({ tracked: [plane("N843S", A), plane("N786QS", B)] });
  assert.equal(h.fly(A, 3), 3);
  assert.equal(h.fly(B, 2), 2);
  const bytesA = fs.statSync(monthPath(h.base, A, "2026-11")).size;
  const bytesB = fs.statSync(monthPath(h.base, B, "2026-11")).size;
  h.tracked.list = [plane("N786QS", B)]; // DELETE /api/data/aircraft/tracked/N843S
  assert.equal(h.fly(A, 2), 0, "no longer tracked: not written");
  assert.equal(h.fly(B, 1), 1);
  assert.equal(fs.statSync(monthPath(h.base, A, "2026-11")).size, bytesA, "the log is still there, untouched");
  await h.log.maintain();
  assert.equal(fs.statSync(monthPath(h.base, A, "2026-11")).size, bytesA);
  const st = await h.log.status();
  assert.deepEqual(Object.keys(st.by_reg), ["N786QS"]);
  assert.equal(st.total.untracked_logs.length, 1);
  assert.equal(st.total.untracked_logs[0].hex, A);
  assert.equal(st.total.untracked_logs[0].bytes, bytesA);
  assert.equal(st.total.untracked_logs[0].first_fix_t, T0 / 1000 + 1);
  assert.equal(st.total.planes_with_log, 2);
  assert.ok(st.total.bytes > bytesA + bytesB, "the total counts the removed tail's log");
  assert.match(st.note, /does NOT delete its log/);
  assert.match(st.note, /still counts toward the total cap/);
  // tracked again: the log continues in the same file
  h.tracked.list = [plane("N786QS", B), plane("N843S", A)];
  assert.equal(h.fly(A, 1), 1);
  assert.equal(linesOf(monthPath(h.base, A, "2026-11")).length, 4);
});

test("after a restart the fix already on disk is not written a second time", () => {
  const first = harness();
  assert.equal(first.offer(A, T0), 1);
  const again = harness({ base: first.base }); // a new process: nothing in memory
  assert.equal(again.offer(A, T0), 0, "the last line of the month file says this fix is held");
  assert.equal(again.log.counters.skipped_same_fix_total, 1);
  assert.equal(again.offer(A, T0 + 2_000), 1);
  assert.deepEqual(linesOf(monthPath(first.base, A, "2026-11")).map((r) => r.t), [T0 / 1000 - 1, T0 / 1000 + 1]);
});

test("a month file whose last line was torn by a crash does not swallow the next fix", async () => {
  const first = harness();
  assert.equal(first.offer(A, T0), 1);
  const fp = monthPath(first.base, A, "2026-11");
  fs.appendFileSync(fp, '{"t":1794927615,"i":"ab8c8e","la":36.1'); // no closing brace, no newline
  const again = harness({ base: first.base, t: T0 + 60_000 });
  assert.equal(again.offer(A), 1);
  const text = fs.readFileSync(fp, "utf8").split("\n").filter((l) => l.length > 0);
  assert.equal(text.length, 3, "the torn line stays on its own line");
  assert.equal((JSON.parse(text[2]) as Raw).t, (T0 + 60_000) / 1000 - 1, "and the new fix is a whole line after it");
  const got: number[] = [];
  const st = await readTrackedLogRows(first.base, A, {}, (row) => { got.push(row.t as number); });
  assert.deepEqual(got, [T0 / 1000 - 1, (T0 + 60_000) / 1000 - 1]);
  assert.equal(st.lines_unparsed, 1, "the torn line is counted, not served");
});

test("status: months, bytes, first and last fix per tail (through a gzipped month), the caps, the rules in words", async () => {
  assert.equal(TRACKED_LOG_MAX_BYTES_PER_PLANE, 50 * MiB);
  assert.equal(TRACKED_LOG_MAX_BYTES_TOTAL, 500 * MiB);
  assert.deepEqual(trackedLogCaps({} as NodeJS.ProcessEnv), { perPlane: 50 * MiB, total: 500 * MiB });
  assert.deepEqual(trackedLogCaps({ TRACKED_PLANE_LOG_MAX_MB_PER_PLANE: "5", TRACKED_PLANE_LOG_MAX_MB_TOTAL: "20.5" } as NodeJS.ProcessEnv), { perPlane: 5 * MiB, total: 20.5 * MiB });
  assert.deepEqual(trackedLogCaps({ TRACKED_PLANE_LOG_MAX_MB_PER_PLANE: "0", TRACKED_PLANE_LOG_MAX_MB_TOTAL: "junk" } as NodeJS.ProcessEnv), { perPlane: 50 * MiB, total: 500 * MiB }, "a bad value never means no cap");
  _resetTrackedLogRangeCache();
  const oct = harness({ t: Date.parse("2026-10-05T08:00:00Z") });
  assert.equal(oct.fly(A, 3), 3);
  const h = harness({ base: oct.base, tracked: [plane("N843S", A), plane("N863QS", B)] });
  await h.log.maintain(); // October is gzipped
  assert.equal(h.fly(A, 2), 2);
  const st = await h.log.status();
  const a = st.by_reg.N843S;
  assert.equal(a.hex, A);
  assert.equal(a.months, 2);
  assert.equal(a.first_month, "2026-10");
  assert.equal(a.last_month, "2026-11");
  assert.equal(a.first_fix_t, Date.parse("2026-10-05T08:00:01Z") / 1000, "read through the gzip");
  assert.equal(a.last_fix_t, T0 / 1000 + 3);
  assert.equal(a.bytes, fs.statSync(monthPath(h.base, A, "2026-10", true)).size + fs.statSync(monthPath(h.base, A, "2026-11")).size);
  assert.equal(a.paused, false);
  assert.equal(a.cap_bytes, 50 * MiB);
  assert.deepEqual({ ...st.by_reg.N863QS, cap_bytes: 0 }, { hex: B, months: 0, first_month: null, last_month: null, bytes: 0, first_fix_t: null, last_fix_t: null, paused: false, why: null, paused_since_t: null, cap_bytes: 0 });
  assert.equal(st.total.bytes, a.bytes);
  assert.equal(st.total.cap_bytes_per_plane, 50 * MiB);
  assert.equal(st.total.cap_bytes_total, 500 * MiB);
  assert.equal(st.total.lines_written_total, 2);
  assert.match(st.storage, /aircraft_tracked\/<HEX>\/YYYY-MM\.jsonl/);
  assert.match(st.rule, /written once/);
  assert.match(st.cap_rule, /oldest month file goes first/);
  assert.match(st.cap_rule, /current UTC month is never deleted/);
  assert.deepEqual(await trackedLogSummary(h.base, A), { months: 2, first_month: "2026-10", last_month: "2026-11", bytes: a.bytes, first_fix_t: a.first_fix_t, last_fix_t: a.last_fix_t });
  assert.equal((await trackedLogSummary(h.base, C)).months, 0);
  assert.equal((await trackedLogSummary(h.base, "../etc")).months, 0, "only a 6-digit hex ever reaches a path");
});

test("the fix bus: every path's batch reaches the log once started, and none after stop()", () => {
  const base = tmp("bus");
  const now = Date.now();
  const { log, stop } = startTrackedPlaneLog({ baseDir: base, env: {} as NodeJS.ProcessEnv, readFree: () => 50 * GiB, tracked: () => [plane("N843S", A)] });
  try {
    publishFixes({ provider: "adsblol", origin: "fastlane", aircraft: mapped(readsb(A)), fetchedAt: now, upstreamNowMs: now });
    publishFixes({ provider: "adsblol", origin: "sweep-type", aircraft: mapped(readsb(A, { seen_pos: 0 }), readsb(C)), fetchedAt: now + 1_500, upstreamNowMs: now + 1_500 });
    assert.equal(log.counters.lines_written_total, 2, "the selected-aircraft lane and the world sweep both feed it");
  } finally {
    stop();
  }
  publishFixes({ provider: "adsblol", origin: "tracked", aircraft: mapped(readsb(A, { seen_pos: 0 })), fetchedAt: now + 9_000, upstreamNowMs: now + 9_000 });
  assert.equal(log.counters.lines_written_total, 2);
  assert.equal(fs.readdirSync(path.join(base, TRACKED_LOG_DIR)).join(), "AB8C8E");
});

test("the read-back window picks months and lines by the fix's own time", async () => {
  const base = tmp("read");
  for (const m of ["2026-09", "2026-10", "2026-11"]) assert.equal(harness({ base, t: Date.parse(`${m}-10T12:00:00Z`) }).fly(A, 3), 3);
  const got: number[] = [];
  const from = Date.parse("2026-10-10T12:00:02Z") / 1000, to = Date.parse("2026-10-10T12:00:04Z") / 1000;
  const st = await readTrackedLogRows(base, A, { fromSec: from, toSec: to, slackSec: 0 }, (row) => { got.push(row.t as number); });
  assert.deepEqual(got, [from + 1], "the fixes are at :01, :03 and :05 (1 s older than their receive times): only :03 is inside");
  assert.deepEqual({ held: st.months_held, inWindow: st.months_in_window, read: st.files_read }, { held: 3, inWindow: 1, read: 1 });
  let all = 0;
  await readTrackedLogRows(base, A, {}, () => { all++; });
  assert.equal(all, 9);
  let none = 0;
  const miss = await readTrackedLogRows(base, B, {}, () => { none++; });
  assert.equal(none, 0);
  assert.equal(miss.months_held, 0);
});

test("measured: bytes per line and per plane-month, from the writer's own output", () => {
  const h = harness();
  // one hour of flight at one fix every 5 s (the tracked poller's fast cadence), values changing as they do in flight
  let n = 0;
  for (let k = 0; k < 720; k++) {
    h.clock.t += 5_000;
    n += h.offer(A, h.clock.t, {
      lat: 36.1234 + k * 0.0101, lon: -83.9876 - k * 0.0113, alt_baro: 34000 + (k % 7) * 25, alt_geom: 34525 + (k % 5) * 25,
      gs: 459.3 + (k % 9) * 0.7, track: 268.4 + (k % 11) * 0.3, baro_rate: (k % 3) * 64 - 64, seen_pos: (k % 10) / 10,
    });
  }
  assert.equal(n, 720);
  const fp = monthPath(h.base, A, "2026-11");
  const raw = fs.statSync(fp).size;
  const gz = zlib.gzipSync(fs.readFileSync(fp)).length;
  const perLine = raw / 720;
  // the numbers the cap was sized against (REPORT: printed, and bounded here so a format change is noticed)
  console.log(`[tracked-log measured] ${perLine.toFixed(1)} bytes/line raw, ${(gz / 720).toFixed(1)} gzipped; one flight hour at 5 s = ${raw} B raw / ${gz} B gz`);
  assert.ok(perLine > 180 && perLine < 320, `a line is ${perLine} bytes`);
  assert.ok(gz < raw / 3, "a month compresses at least 3x once it is over");
  // 50 MiB holds at least 150 flight hours of the CURRENT month at this density
  assert.ok((50 * MiB) / raw > 150);
});

test("wiring: routes.ts starts the log once and serves it on GET /api/data/aircraft/tracked; the stream has a manifest", () => {
  const src = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.equal(src.split("startTrackedPlaneLog(").length - 1, 1);
  assert.ok(src.includes('import { startTrackedPlaneLog } from "./trackedPlaneLog"'));
  assert.ok(/startTrackedPlaneLog\(\{ tracked: \(\) => trackedPoller\.getRegistry\(\)\.planes \}\)/.test(src), "the same tracked list the poller owns");
  const handler = src.slice(src.indexOf('app.get("/api/data/aircraft/tracked"'), src.indexOf('app.post("/api/data/aircraft/tracked/:reg"'));
  assert.ok(handler.includes("trackedLog.log.status()"));
  assert.ok(handler.includes("log: logStatus ? logStatus.by_reg[p.reg]"), "a log block per tail");
  assert.ok(handler.includes("logStatus.total"), "and the total with the caps");
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "aircraft_tracked.json"), "utf8")) as Record<string, string>;
  assert.equal(manifest.stream, TRACKED_LOG_DIR);
  assert.equal(manifest.layout, "per_entity_dirs");
  assert.match(manifest.storage, /permanent, capped: 50 MiB per plane, 500 MiB total, oldest month first/);
  assert.match(manifest.license, /adsblol = ODbL 1\.0/);
  assert.match(manifest.license, /adsbfi \/ airplaneslive \/ opensky = personal \/ non-commercial/);
  assert.equal(manifest.schema_version, 5, "the aircraft stream's line schema");
});
