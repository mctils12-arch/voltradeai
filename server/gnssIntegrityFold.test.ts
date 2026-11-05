/**
 * gnssIntegrityFold battery (memory fix A2b, 2026-11-03). The property under
 * test is EXACTNESS: the incremental fold (per-closed-day cached partials,
 * only the open day re-read) must reproduce computeGnssIntegritySignal's
 * payload byte-for-byte — including distinct_airframes, which is a UNION
 * across days, and the even-sampler's truncation, which depends on the
 * whole day's file count. Fixtures deliberately span day and hour
 * boundaries, arrive out of time order, repeat airframes within and across
 * days, repeat identical lines, mix origins/bands/regions and exceed the
 * per-file budget. Hermetic: temp dirs only, no network.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import { computeGnssIntegritySignal, CANDIDATE_BBOX, CONTROL_BBOX, type GnssIntegritySignalSummary } from "./gnssIntegritySignal";
import { readGnssIntegrityDay } from "./gnssIntegrityQuery";
import {
  computeGnssIntegritySignalManaged, combineGnssDayPartials, airframeKey, gnssFoldMode, gnssFoldStats, setGnssFoldAuditWriter, GNSS_FOLD_DEFAULT,
  GNSS_FOLD_ENV, _resetGnssFoldForTests, _gnssFoldCacheForTests, _gnssFoldCacheKeyForTests,
  type GnssDayPartial,
} from "./gnssIntegrityFold";

const NOW = Date.parse("2026-08-14T12:00:00Z"); // 2026-08-14 is the OPEN day; 11..13 are closed
const DAYS = 4;
const LIMIT = 7; // per-day budget: a 3-file day gets ceil(7/3)=3 rows per file

const CAND = { la: 55, lo: 20 };      // Baltic candidate bbox
const NY = { la: 40.7, lo: -74 };     // control bbox
const PARIS = { la: 48, lo: 2 };      // control bbox
const TOKYO = { la: 35.6, lo: 139.7 }; // neither

function row(i: string, p: { la: number; lo: number }, t: number, al: number | null, ni?: number, extra: Record<string, unknown> = {}): string {
  const o: Record<string, unknown> = { t, i, la: p.la, lo: p.lo, pt: "adsb_icao", ...extra };
  if (al !== null) o.al = al;
  if (ni !== undefined) o.ni = ni;
  return JSON.stringify(o);
}

function writeHour(base: string, day: string, hh: string, lines: string[]): void {
  fs.writeFileSync(path.join(base, "aircraft", `${day}-${hh}.jsonl`), lines.join("\n") + "\n");
}

function gzipHour(base: string, day: string, hh: string): void {
  const fp = path.join(base, "aircraft", `${day}-${hh}.jsonl`);
  fs.writeFileSync(fp + ".gz", zlib.gzipSync(fs.readFileSync(fp)));
  fs.unlinkSync(fp);
}

function fixture(): string {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-gnss-fold-"));
  fs.mkdirSync(path.join(base, "aircraft"), { recursive: true });
  // 2026-08-11 — two hour files; airframe a1 repeats (out of order), a2's
  // reading flips to zero in the later hour, c2 is an identical duplicate
  // line, a3 has no ni (excluded), n1 is outside both regions.
  writeHour(base, "2026-08-11", "05", [
    row("a1", CAND, 500, 9000, 0), row("a2", CAND, 400, 9000, 8), row("a1", CAND, 450, 9000, 0),
    row("a3", CAND, 600, 9000), row("c1", NY, 100, 9000, 0), row("c2", PARIS, 90, 9000, 8),
    row("c2", PARIS, 90, 9000, 8), row("n1", TOKYO, 1, 9000, 0),
  ]);
  writeHour(base, "2026-08-11", "17", [
    row("a4", CAND, 2000, 9000, 0, { pt: "mlat", ml: ["nic", "rc"] }), row("a2", CAND, 1900, 9000, 0),
    row("c3", NY, 1800, 1000, 8), row("c4", PARIS, 1700, null, 0, { g: true }),
  ]);
  // 2026-08-12 — three hour files, so the per-file budget is 3: hour 00
  // holds 5 candidate rows (truncates the candidate side), hour 08 holds 4
  // control rows (truncates the control side), hour 23 repeats a1/c1 from
  // other days (the union-not-sum case for distinct_airframes).
  writeHour(base, "2026-08-12", "00", [
    row("b1", CAND, 50, 5000, 0), row("b2", CAND, 49, 5000, 8), row("b3", CAND, 48, 5000, 0),
    row("b4", CAND, 47, 5000, 8), row("b5", CAND, 46, 5000, 0),
  ]);
  writeHour(base, "2026-08-12", "08", [
    row("c5", NY, 800, 9000, 8), row("c6", NY, 790, 9000, 0), row("c7", PARIS, 780, 9000, 8), row("c8", PARIS, 770, 9000, 8),
  ]);
  writeHour(base, "2026-08-12", "23", [row("a1", CAND, 2300, 9000, 8), row("c1", NY, 2300, 9000, 8)]);
  // 2026-08-13 — one hour file.
  writeHour(base, "2026-08-13", "12", [row("a1", CAND, 1200, 9000, 0), row("c1", PARIS, 1200, 9000, 0), row("a5", CAND, 1100, 500, 8)]);
  // 2026-08-14 — the OPEN day: an early hour and the current hour.
  writeHour(base, "2026-08-14", "00", [row("a6", CAND, 10, 9000, 0), row("c9", NY, 10, 9000, 8)]);
  writeHour(base, "2026-08-14", "11", [row("a1", CAND, 1100, 9000, 0), row("c1", NY, 1100, 9000, 8)]);
  return base;
}

const full = (base: string, daily?: Parameters<typeof computeGnssIntegritySignal>[3]) =>
  computeGnssIntegritySignal(DAYS, base, NOW, daily, LIMIT);
const managed = (base: string, mode: "full" | "shadow" | "incremental", daily?: Parameters<typeof computeGnssIntegritySignal>[3]) =>
  computeGnssIntegritySignalManaged(DAYS, base, NOW, daily, { mode, perDayLimit: LIMIT });

async function captureConsoleError(fn: () => Promise<void>): Promise<string[]> {
  const orig = console.error;
  const lines: string[] = [];
  console.error = (...args: unknown[]) => { lines.push(args.map(String).join(" ")); };
  try { await fn(); } finally { console.error = orig; }
  return lines;
}

const cleanup = (base: string) => fs.rmSync(base, { recursive: true, force: true });

test("incremental == full exactly (cold and warm) across day/hour boundaries, out-of-order rows, duplicates and truncation", async () => {
  _resetGnssFoldForTests();
  const base = fixture();
  const ref = await full(base);
  // the fixture really exercises what it claims to
  assert.deepEqual(ref.freshness.candidate.days_read, ["2026-08-14", "2026-08-13", "2026-08-12", "2026-08-11"]);
  assert.equal(ref.freshness.candidate.truncated, true, "candidate side truncates on 2026-08-12 hour 00");
  assert.equal(ref.freshness.control.truncated, true, "control side truncates on 2026-08-12 hour 08");
  assert.ok(ref.bands.length > 0);

  const cold = await managed(base, "incremental");
  assert.deepEqual(cold, ref);
  assert.equal(JSON.stringify(cold), JSON.stringify(ref), "byte-identical API payload");
  let s = gnssFoldStats();
  assert.equal(s.mode, "incremental");
  assert.equal(s.lastCycleDayReads, 8, "cold cache: every day of both regions is read once");
  assert.equal(s.partialsCached, 8);
  assert.equal(typeof s.lastIncrementalMs, "number");

  const warm = await managed(base, "incremental");
  assert.deepEqual(warm, ref);
  s = gnssFoldStats();
  assert.equal(s.lastCycleDayReads, 0, "nothing changed on disk: no day is re-read");
  assert.equal(s.partialsCached, 8);
  cleanup(base);
});

test("open day grows (appended hour file + a new hour file): only the open day is re-read, result == a fresh full read", async () => {
  _resetGnssFoldForTests();
  const base = fixture();
  const before = await managed(base, "incremental");
  fs.appendFileSync(path.join(base, "aircraft", "2026-08-14-11.jsonl"), row("a7", CAND, 1130, 9000, 0) + "\n");
  writeHour(base, "2026-08-14", "12", [row("a8", CAND, 1200, 9000, 8), row("c1", PARIS, 1200, 9000, 0)]);
  const ref = await full(base);
  assert.ok(ref.freshness.candidate.rows_scanned > before.freshness.candidate.rows_scanned, "the appended rows are in the fresh sample");
  const inc = await managed(base, "incremental");
  assert.deepEqual(inc, ref);
  const s = gnssFoldStats();
  assert.equal(s.lastCycleDayReads, 2, "the open day, both regions — the three closed days came from cache");
  assert.equal(s.partialsCached, 8);
  cleanup(base);
});

test("a closed day's gzip rename (the ~2h compression pass) invalidates that day only and reproduces the full read exactly", async () => {
  _resetGnssFoldForTests();
  const base = fixture();
  await managed(base, "incremental");
  gzipHour(base, "2026-08-12", "08");
  const ref = await full(base);
  const inc = await managed(base, "incremental");
  assert.deepEqual(inc, ref);
  assert.equal(gnssFoldStats().lastCycleDayReads, 2, "2026-08-12 re-read for both regions; nothing else");
  cleanup(base);
});

test("a closed day evicted from disk goes missing in both folds and is filled from the daily fold identically", async () => {
  _resetGnssFoldForTests();
  const base = fixture();
  await managed(base, "incremental");
  for (const hh of ["05", "17"]) fs.unlinkSync(path.join(base, "aircraft", `2026-08-11-${hh}.jsonl`));
  const daily = {
    "2026-08-11": {
      candidate: [{ band: "cruise", origin: "broadcast", n_total: 300, n_zero: 30, distinct_airframes: 40 }],
      control: [{ band: "cruise", origin: "broadcast", n_total: 9000, n_zero: 90, distinct_airframes: 700 }],
    },
  } as never;
  const ref = await full(base, daily);
  assert.deepEqual(ref.freshness.candidate.days_from_daily_archive, ["2026-08-11"]);
  const inc = await managed(base, "incremental", daily);
  assert.deepEqual(inc, ref);
  assert.equal(gnssFoldStats().lastCycleDayReads, 2, "the evicted day's manifest changed -> re-read (and found missing) once per region");
  cleanup(base);
});

test("shadow mode serves the full result, compares exactly, records zero mismatches, and warms the cache for incremental", async () => {
  _resetGnssFoldForTests();
  const base = fixture();
  const ref = await full(base);
  let served: GnssIntegritySignalSummary | null = null;
  const errors = await captureConsoleError(async () => { served = await managed(base, "shadow"); });
  assert.deepEqual(served, ref);
  assert.equal(errors.filter((l) => l.includes("FOLD-MISMATCH")).length, 0);
  let s = gnssFoldStats();
  assert.equal(s.mode, "shadow");
  assert.equal(s.mismatches, 0);
  assert.equal(s.lastMismatchAt, null);
  assert.equal(s.lastCycleDayReads, 8, "shadow reads every day so both folds see the same bytes");
  assert.equal(s.partialsCached, 8);
  assert.equal(typeof s.lastFullMs, "number");
  assert.equal(typeof s.lastIncrementalMs, "number");
  assert.equal(s.lastCyclePeakRows, ref.freshness.candidate.rows_scanned + ref.freshness.control.rows_scanned,
    "shadow holds the whole window's rows, exactly like the full fold");

  const inc = await managed(base, "incremental");
  assert.deepEqual(inc, ref);
  s = gnssFoldStats();
  assert.equal(s.lastCycleDayReads, 0, "the shadow cycle's partials are reused as-is");
  cleanup(base);
});

test("shadow mode catches a corrupted cached partial: counts it, logs FOLD-MISMATCH gnss, writes the audit row, still serves the full result, then heals", async () => {
  _resetGnssFoldForTests();
  const base = fixture();
  const ref = await full(base);
  await managed(base, "incremental"); // warm
  const key = _gnssFoldCacheKeyForTests(base, CANDIDATE_BBOX, "2026-08-12");
  const cached = _gnssFoldCacheForTests().get(key)!;
  const cell = cached.cells.find((c) => c.band === "cruise" && c.origin === "broadcast")!;
  assert.ok(cell, "fixture has a candidate cruise/broadcast cell on 2026-08-12 (a1 in hour 23)");
  cell.n_zero += 1; // a wrong partial that would move the cruise verdict's candidate_k

  const auditRows: Array<[string, string]> = [];
  setGnssFoldAuditWriter((type, message) => { auditRows.push([type, message]); });
  let served: GnssIntegritySignalSummary | null = null;
  const errors = await captureConsoleError(async () => { served = await managed(base, "shadow"); });
  assert.deepEqual(served, ref, "the FULL result is what gets served in shadow mode");
  const s = gnssFoldStats();
  assert.equal(s.mismatches, 1);
  assert.equal(typeof s.lastMismatchAt, "string");
  // the wrong n_zero moves the cruise verdict (candidate_k) and, through
  // it, the top-level verdict — whichever key the payload order reaches
  // first is what the report names, with both values.
  assert.match(s.lastMismatchDetail!, /^\$\.(verdict|bands\[\d+\]\.candidate_k): .+ vs .+/, `detail names the differing path: ${s.lastMismatchDetail}`);
  assert.equal(errors.filter((l) => l.startsWith("FOLD-MISMATCH gnss: $.")).length, 1);
  assert.equal(auditRows.length, 1);
  assert.equal(auditRows[0][0], "FOLD-MISMATCH");
  assert.equal(auditRows[0][1], `gnss: ${s.lastMismatchDetail}`);

  // the fresh partial overwrote the corrupt one: the next cycle is clean
  const again = await managed(base, "shadow");
  assert.deepEqual(again, ref);
  assert.equal(gnssFoldStats().mismatches, 1);
  assert.deepEqual(await managed(base, "incremental"), ref);
  cleanup(base);
});

test("an audit writer that throws is contained and the mismatch is still counted", async () => {
  _resetGnssFoldForTests();
  const base = fixture();
  await managed(base, "incremental");
  const key = _gnssFoldCacheKeyForTests(base, CONTROL_BBOX, "2026-08-11");
  const cell = _gnssFoldCacheForTests().get(key)!.cells.find((c) => c.band === "cruise" && c.origin === "broadcast")!;
  cell.n_total += 5;
  setGnssFoldAuditWriter(() => { throw new Error("db closed"); });
  const errors = await captureConsoleError(async () => { await managed(base, "shadow"); });
  assert.equal(gnssFoldStats().mismatches, 1);
  assert.ok(errors.some((l) => l.includes("audit writer threw")));
  cleanup(base);
});

test("full mode IS computeGnssIntegritySignal and leaves the partial cache untouched", async () => {
  _resetGnssFoldForTests();
  const base = fixture();
  const ref = await full(base);
  const out = await managed(base, "full");
  assert.deepEqual(out, ref);
  const s = gnssFoldStats();
  assert.equal(s.mode, "full");
  assert.equal(s.partialsCached, 0);
  assert.equal(typeof s.lastFullMs, "number");
  assert.equal(s.lastIncrementalMs, null);
  assert.equal(s.lastCyclePeakRows, null, "uninstrumented path reports null, never a made-up number");
  assert.equal(s.lastCycleDayReads, 8);
  cleanup(base);
});

test("mode flag: unknown VOLTRADE_GNSS_FOLD is shadow; unset is the promoted default (incremental, 2026-11-05); full/incremental/shadow are honoured", () => {
  const prior = process.env[GNSS_FOLD_ENV];
  try {
    // 2026-11-05: the one changed expectation — unset was `shadow` until the
    // fold ran 24 h+ in shadow with zero mismatches; the promotion IS the change.
    delete process.env[GNSS_FOLD_ENV];
    assert.equal(gnssFoldMode(), "incremental");
    assert.equal(GNSS_FOLD_DEFAULT, "incremental");
    process.env[GNSS_FOLD_ENV] = "garbage";
    assert.equal(gnssFoldMode(), "shadow", "a typo never lands on a path nobody chose");
    process.env[GNSS_FOLD_ENV] = "shadow";
    assert.equal(gnssFoldMode(), "shadow", "spelled out, shadow is the rollback that restores the comparison");
    process.env[GNSS_FOLD_ENV] = "incremental";
    assert.equal(gnssFoldMode(), "incremental");
    process.env[GNSS_FOLD_ENV] = "FULL";
    assert.equal(gnssFoldMode(), "full");
  } finally {
    if (prior === undefined) delete process.env[GNSS_FOLD_ENV]; else process.env[GNSS_FOLD_ENV] = prior;
  }
});

test("no aircraft directory at all: every mode agrees on the honestly-empty summary", async () => {
  _resetGnssFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-gnss-fold-empty-"));
  const ref = await full(base);
  assert.equal(ref.verdict, "INCONCLUSIVE");
  assert.deepEqual(await managed(base, "incremental"), ref);
  assert.deepEqual(await managed(base, "shadow"), ref);
  assert.deepEqual(await managed(base, "full"), ref);
  assert.equal(gnssFoldStats().mismatches, 0);
  cleanup(base);
});

test("the incremental transient is one day of rows per region, not the window (shadow/full hold every sampled row)", async () => {
  _resetGnssFoldForTests();
  const base = fixture();
  await managed(base, "shadow");
  const shadowPeak = gnssFoldStats().lastCyclePeakRows!;
  _resetGnssFoldForTests();
  await managed(base, "incremental");
  const incPeak = gnssFoldStats().lastCyclePeakRows!;
  let maxDayRows = 0;
  for (const bbox of [CANDIDATE_BBOX, CONTROL_BBOX]) {
    let m = 0;
    for (const day of ["2026-08-14", "2026-08-13", "2026-08-12", "2026-08-11"]) {
      m = Math.max(m, (await readGnssIntegrityDay(day, bbox, base, LIMIT)).rows.length);
    }
    maxDayRows += m;
  }
  assert.ok(incPeak <= maxDayRows, `incremental peak ${incPeak} <= biggest day per region ${maxDayRows}`);
  assert.ok(incPeak < shadowPeak, `incremental peak ${incPeak} < window peak ${shadowPeak}`);
  cleanup(base);
});

test("airframeKey: distinct identities never share a key — case, length, non-hex and non-string forms stay apart; lowercase 6-hex compacts to its integer", () => {
  const ids: unknown[] = ["abcdef", "ABCDEF", "abcde", "abcdef0", "~abcdef", "123456", 123456, 0x123456, "000000", 0, true, "true"];
  const keys = ids.map(airframeKey);
  assert.equal(new Set(keys).size, ids.length, "injective on an adversarial set — a union over keys has the union-over-originals size");
  assert.equal(airframeKey("abcdef"), 0xabcdef);
  assert.equal(airframeKey("000000"), 0);
  assert.equal(airframeKey("ABCDEF"), "ABCDEF", "uppercase is a different original and stays a string");
  assert.equal(airframeKey("~abcdef"), "~abcdef", "TIS-B/anonymous addresses stay strings");
  assert.equal(typeof airframeKey(123456), "string", "a non-string original is tagged, never confused with an encoded hex string");
});

test("combineGnssDayPartials: airframe union is over keys — the same hex id on two days counts once, two ids differing only by case count twice", () => {
  const cell = (airframes: unknown[]): GnssDayPartial["cells"][number] =>
    ({ band: "cruise", origin: "broadcast", n_total: airframes.length, n_zero: 0, airframes: airframes.map(airframeKey) });
  const out = combineGnssDayPartials([
    { day: "2026-08-14", present: true, rows_scanned: 3, truncated: false, manifest: "m", cells: [cell(["abcdef", "ABCDEF", "~abcdef"])] },
    { day: "2026-08-13", present: true, rows_scanned: 2, truncated: false, manifest: "m", cells: [cell(["abcdef", "~abcdef"])] },
  ]);
  assert.equal(out.cells[0].distinct_airframes, 3);
  assert.equal(out.cells[0].n_total, 5);
});

test("combineGnssDayPartials: counts add, airframes UNION across days (never sum), day order and cell order are the pooled read's", () => {
  const p = (day: string, present: boolean, cells: GnssDayPartial["cells"], truncated = false): GnssDayPartial =>
    ({ day, present, rows_scanned: cells.reduce((s, c) => s + c.n_total, 0), truncated, cells, manifest: "m" });
  const out = combineGnssDayPartials([
    p("2026-08-14", true, [{ band: "cruise", origin: "broadcast", n_total: 2, n_zero: 1, airframes: ["x", "y"] }]),
    p("2026-08-13", false, []),
    p("2026-08-12", true, [
      { band: "low", origin: "ground", n_total: 1, n_zero: 1, airframes: ["q"] },
      { band: "cruise", origin: "broadcast", n_total: 3, n_zero: 0, airframes: ["y", "z"] },
    ], true),
  ]);
  assert.deepEqual(out.days_read, ["2026-08-14", "2026-08-12"]);
  assert.deepEqual(out.days_missing, ["2026-08-13"]);
  assert.equal(out.rows_scanned, 6);
  assert.equal(out.truncated, true);
  assert.deepEqual(out.cells, [
    { band: "cruise", origin: "broadcast", n_total: 5, n_zero: 1, distinct_airframes: 3 },
    { band: "low", origin: "ground", n_total: 1, n_zero: 1, distinct_airframes: 1 },
  ]);
});

test("heap report (informational): full vs incremental over a larger synthetic window", async () => {
  _resetGnssFoldForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-gnss-fold-heap-"));
  fs.mkdirSync(path.join(base, "aircraft"), { recursive: true });
  const days = ["2026-08-14", "2026-08-13", "2026-08-12", "2026-08-11", "2026-08-10", "2026-08-09"];
  for (const day of days) {
    for (const hh of ["03", "11", "19"]) {
      const lines: string[] = [];
      for (let k = 0; k < 400; k++) {
        const cand = k % 3 === 0;
        lines.push(row(`${day}-${hh}-${k % 150}`, cand ? CAND : (k % 2 ? NY : PARIS), 4000 - k, cand ? 9000 : 8000, k % 17 === 0 ? 0 : 8));
      }
      writeHour(base, day, hh, lines);
    }
  }
  const limit = 3000;
  const now = Date.parse("2026-08-14T12:00:00Z");
  const h0 = process.memoryUsage().heapUsed;
  const ref = await computeGnssIntegritySignal(6, base, now, undefined, limit);
  const h1 = process.memoryUsage().heapUsed;
  const cold = await computeGnssIntegritySignalManaged(6, base, now, undefined, { mode: "incremental", perDayLimit: limit });
  const h2 = process.memoryUsage().heapUsed;
  const warm = await computeGnssIntegritySignalManaged(6, base, now, undefined, { mode: "incremental", perDayLimit: limit });
  const h3 = process.memoryUsage().heapUsed;
  assert.deepEqual(cold, ref);
  assert.deepEqual(warm, ref);
  const s = gnssFoldStats();
  console.log(`[gnss-fold heap] rows/window=${ref.freshness.candidate.rows_scanned + ref.freshness.control.rows_scanned} ` +
    `full heapDelta=${((h1 - h0) / 1e6).toFixed(1)}MB incrementalCold heapDelta=${((h2 - h1) / 1e6).toFixed(1)}MB ` +
    `incrementalWarm heapDelta=${((h3 - h2) / 1e6).toFixed(1)}MB warmDayReads=${s.lastCycleDayReads} ` +
    `warmMs=${s.lastIncrementalMs} partials=${s.partialsCached} peakRowsIncremental=${s.lastCyclePeakRows}`);
  cleanup(base);
});
