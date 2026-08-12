/**
 * Hermetic tests for the GNSS integrity read/query path (Phase 3).
 * Runs via `npm run test:node` (tsx --test). No network, temp dirs only.
 */
import { test } from "node:test";
import assert from "node:assert";
import fs from "fs";
import os from "os";
import path from "path";
import {
  aggregateGnssIntegrity, bandForRow, inBbox, readGnssIntegrityWindow,
  LOW_ALT_MAX_M, MID_ALT_MAX_M,
} from "./gnssIntegrityQuery";

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "vt-gnss-"));

test("bandForRow: ground flag wins over altitude; unknown when altitude absent/non-finite", () => {
  assert.equal(bandForRow({ g: true, al: 9000 }), "ground");
  assert.equal(bandForRow({ al: LOW_ALT_MAX_M - 1 }), "low");
  assert.equal(bandForRow({ al: LOW_ALT_MAX_M }), "mid");
  assert.equal(bandForRow({ al: MID_ALT_MAX_M - 1 }), "mid");
  assert.equal(bandForRow({ al: MID_ALT_MAX_M }), "cruise");
  assert.equal(bandForRow({}), "unknown");
  assert.equal(bandForRow({ al: NaN }), "unknown");
});

test("inBbox: no bbox passes everything; a bbox excludes rows outside it and rows missing lat/lon", () => {
  const bbox = { lamin: 50, lamax: 60, lomin: 10, lomax: 30 };
  assert.ok(inBbox({ la: 55, lo: 20 }, undefined));
  assert.ok(inBbox({ la: 55, lo: 20 }, bbox));
  assert.ok(!inBbox({ la: 40, lo: 20 }, bbox), "outside lat range");
  assert.ok(!inBbox({ la: 55 }, bbox), "missing lon");
});

test("aggregateGnssIntegrity: rows with no ni field are excluded from every denominator", () => {
  const rows = [
    { i: "a", al: 8000, ni: 8, pt: "adsb_icao" },
    { i: "b", al: 8000, pt: "adsb_icao" }, // no ni — must not count
  ];
  const cells = aggregateGnssIntegrity(rows);
  assert.equal(cells.length, 1, "the ni-less row contributes no cell at all");
  assert.equal(cells[0].n_total, 1);
});

test("aggregateGnssIntegrity: numerator (nic==0) and denominator (nic present) never collapse into a bare rate", () => {
  const rows = [
    { i: "a", al: 8000, ni: 0, pt: "adsb_icao" },
    { i: "b", al: 8000, ni: 8, pt: "adsb_icao" },
    { i: "c", al: 8000, ni: 0, pt: "adsb_icao" },
  ];
  const cells = aggregateGnssIntegrity(rows);
  assert.equal(cells.length, 1);
  assert.deepEqual({ n_total: cells[0].n_total, n_zero: cells[0].n_zero }, { n_total: 3, n_zero: 2 });
});

test("aggregateGnssIntegrity: origin split keeps broadcast and mlat-derived nic==0 rows in separate cells", () => {
  // This is the finding the 2026-08-11 adversarial verification forced: a
  // pooled count would have silently mixed aircraft-reported degradation
  // with ground-computed (mlat) values.
  const rows = [
    { i: "a", al: 8000, ni: 0, pt: "adsb_icao" },              // broadcast
    { i: "b", al: 8000, ni: 0, pt: "mlat", ml: ["nic", "rc"] }, // ground
    { i: "c", al: 8000, ni: 0, pt: "tisb_other" },              // ground (tisb)
    { i: "d", al: 8000, ni: 5, pt: "mode_s" },                  // mode_s
    { i: "e", al: 8000, ni: 5 },                                // unknown (no pt)
  ];
  const cells = aggregateGnssIntegrity(rows);
  const byOrigin = Object.fromEntries(cells.map((c) => [c.origin, c]));
  assert.equal(byOrigin.broadcast.n_zero, 1);
  assert.equal(byOrigin.broadcast.n_total, 1);
  assert.equal(byOrigin.ground.n_zero, 2, "mlat + tisb both fold into 'ground', distinct from broadcast");
  assert.equal(byOrigin.ground.n_total, 2);
  assert.equal(byOrigin.mode_s.n_total, 1);
  assert.equal(byOrigin.unknown.n_total, 1);
});

test("aggregateGnssIntegrity: altitude gradient is separable — ground vs cruise never merge", () => {
  const rows = [
    { i: "a", g: true, ni: 0, pt: "mlat" },
    { i: "b", al: 9000, ni: 0, pt: "mlat" },   // cruise
    { i: "c", al: 9000, ni: 8, pt: "mlat" },   // cruise, non-zero
  ];
  const cells = aggregateGnssIntegrity(rows);
  const ground = cells.find((c) => c.band === "ground")!;
  const cruise = cells.find((c) => c.band === "cruise")!;
  assert.equal(ground.n_total, 1);
  assert.equal(ground.n_zero, 1);
  assert.equal(cruise.n_total, 2);
  assert.equal(cruise.n_zero, 1);
});

test("aggregateGnssIntegrity: bbox restricts the aggregate to one region, honestly excluding the rest", () => {
  const baltic = { i: "a", la: 55, lo: 20, al: 9000, ni: 0, pt: "mlat" };
  const control = { i: "b", la: 40, lo: -75, al: 9000, ni: 0, pt: "mlat" };
  const bbox = { lamin: 50, lamax: 60, lomin: 10, lomax: 30 };
  const cells = aggregateGnssIntegrity([baltic, control], bbox);
  assert.equal(cells.length, 1);
  assert.equal(cells[0].n_total, 1, "only the in-bbox row counted");
});

test("aggregateGnssIntegrity: distinct_airframes counts unique icao24s, not rows", () => {
  const rows = [
    { i: "a", al: 8000, ni: 0, pt: "adsb_icao" },
    { i: "a", al: 8000, ni: 0, pt: "adsb_icao" }, // same airframe, second fix
    { i: "b", al: 8000, ni: 0, pt: "adsb_icao" },
  ];
  const cells = aggregateGnssIntegrity(rows);
  assert.equal(cells[0].n_total, 3);
  assert.equal(cells[0].distinct_airframes, 2);
});

test("readGnssIntegrityWindow: unions multiple days, reports missing days honestly, never fabricates a read", async () => {
  const base = tmp();
  const dir = path.join(base, "aircraft");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "2026-08-11-00.jsonl"),
    '{"t":1,"i":"a","la":55,"lo":20,"al":9000,"ni":0,"pt":"mlat"}\n');
  fs.writeFileSync(path.join(dir, "2026-08-12-00.jsonl"),
    '{"t":2,"i":"b","la":55,"lo":20,"al":9000,"ni":8,"pt":"adsb_icao"}\n');
  const r = await readGnssIntegrityWindow(["2026-08-11", "2026-08-12", "2026-08-13"], undefined, base);
  assert.deepEqual(r.days_read.sort(), ["2026-08-11", "2026-08-12"]);
  assert.deepEqual(r.days_missing, ["2026-08-13"]);
  assert.equal(r.rows_scanned, 2);
  assert.equal(r.truncated, false);
  // rows come from two different origins (mlat vs adsb_icao broadcast), so
  // they land in two separate cruise cells by design — sum across origin.
  const cruiseCells = r.cells.filter((c) => c.band === "cruise");
  assert.equal(cruiseCells.reduce((s, c) => s + c.n_total, 0), 2);
  assert.equal(cruiseCells.reduce((s, c) => s + c.n_zero, 0), 1);
  fs.rmSync(base, { recursive: true, force: true });
});

test("readGnssIntegrityWindow: perDayLimit truncation is surfaced, not silently dropped", async () => {
  const base = tmp();
  const dir = path.join(base, "aircraft");
  fs.mkdirSync(dir, { recursive: true });
  const lines = Array.from({ length: 5 }, (_, i) =>
    `{"t":${i},"i":"a${i}","la":55,"lo":20,"al":9000,"ni":0,"pt":"mlat"}`).join("\n") + "\n";
  fs.writeFileSync(path.join(dir, "2026-08-11-00.jsonl"), lines);
  const r = await readGnssIntegrityWindow(["2026-08-11"], undefined, base, 2);
  assert.equal(r.truncated, true);
  assert.equal(r.rows_scanned, 2);
  fs.rmSync(base, { recursive: true, force: true });
});

// 2026-08-12: the live bug this fixes. Filed as NEXT in the 2026-08-11
// entry ("Phase 4 — an actual gate-2 statistical run") and hit on the
// first real attempt this session: `/api/diag/gnss_integrity` with a
// Baltic bbox returned zero cells against production while the identical
// call with no bbox returned rich data — not because the Baltic signal
// disappeared, but because readArchiveDay (the reader this probe used
// before this fix) exhausts the per-day row budget on the FIRST hour file
// it opens, and Baltic local-daytime traffic falls in later UTC hours.
test("readGnssIntegrityWindow: a bbox region whose traffic is concentrated in a LATER hour file is not starved out by an EARLIER hour file's volume", async () => {
  const base = tmp();
  const dir = path.join(base, "aircraft");
  fs.mkdirSync(dir, { recursive: true });
  // Hour 00 (quiet-Baltic UTC): 50 rows, all far outside the Baltic bbox.
  const hour00 = Array.from({ length: 50 }, (_, i) =>
    `{"t":${i},"i":"ny${i}","la":40.7,"lo":-74,"al":9000,"ni":8,"pt":"adsb_icao"}`).join("\n") + "\n";
  fs.writeFileSync(path.join(dir, "2026-08-11-00.jsonl"), hour00);
  // Hour 06 (Baltic daytime): 3 degraded-nic rows inside the bbox.
  const hour06 = Array.from({ length: 3 }, (_, i) =>
    `{"t":${i},"i":"bal${i}","la":55,"lo":20,"al":9000,"ni":0,"pt":"mlat"}`).join("\n") + "\n";
  fs.writeFileSync(path.join(dir, "2026-08-11-06.jsonl"), hour06);
  const balticBbox = { lamin: 53, lamax: 60, lomin: 17, lomax: 24 };
  // limit=10 is far smaller than hour00 alone (50 rows) — the old
  // file-by-file reader would truncate inside hour00 and never open
  // hour06 at all, silently returning zero Baltic cells.
  const r = await readGnssIntegrityWindow(["2026-08-11"], balticBbox, base, 10);
  const matched = r.cells.reduce((s, c) => s + c.n_total, 0);
  assert.equal(matched, 3, "all 3 Baltic rows from the later hour file must be counted");
  fs.rmSync(base, { recursive: true, force: true });
});
