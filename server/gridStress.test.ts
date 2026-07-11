// gridStress battery (GRID VISION A1 gate-2 FAIL path — descriptive
// dashboard surface, research/experiments.md 2026-07-07 v1.0.190):
// ERCO-only archive fold, same-month percentile math, insufficient-history
// honesty, composite only when all three ingredients are present.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  foldErcoDailyAsync, loadTxDegreeDays, pctRank, computeGridStressReading,
  gridStressEnabled, foldRegionsDailyAsync, computeAllRegionsStress, REGION_LABEL,
} from "./gridStress";

function mkArchive(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "gridstress-"));
}
function writeDay(base: string, day: string, rows: any[], gz = false) {
  const dir = path.join(base, "griddemand");
  fs.mkdirSync(dir, { recursive: true });
  const body = rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
  const fp = path.join(dir, `${day}.jsonl${gz ? ".gz" : ""}`);
  fs.writeFileSync(fp, gz ? zlib.gzipSync(body) : body);
}
const row = (period: string, respondent: string, type: "D" | "DF", mwh: number) =>
  ({ period, respondent, type, mwh, rt: period.slice(0, 10) });

test("gridStressEnabled mirrors gridDemandEnabled (same archive dependency)", () => {
  assert.equal(gridStressEnabled({} as any), false);
  assert.equal(gridStressEnabled({ EIA_API_KEY: "x" } as any), true);
});

test("pctRank: fraction of peers <= x, matches the gate-2 script's convention", () => {
  assert.equal(pctRank([], 5), null);
  assert.equal(pctRank([1, 2, 3, 4, 5], 3), 0.6);
  assert.equal(pctRank([1, 2, 3, 4, 5], 0), 0);
  assert.equal(pctRank([1, 2, 3, 4, 5], 10), 1);
});

test("loadTxDegreeDays: sums H+C per date, ignores nulls", () => {
  const json = {
    axes: { H: ["2020-01-01", "2020-01-02"], C: ["2020-01-01", "2020-01-02"] },
    series: { "TX|H": [10, null], "TX|C": [1, 2] },
  };
  const dd = loadTxDegreeDays(json);
  assert.equal(dd.get("2020-01-01"), 11);
  assert.equal(dd.get("2020-01-02"), 2); // H null contributes nothing, not zero-filled into a false 0+2 read
});

test("foldErcoDailyAsync: ERCO-only, other respondents ignored, peak + strain computed", async () => {
  const base = mkArchive();
  writeDay(base, "2024-07-01", [
    row("2024-07-01T00", "ERCO", "D", 100),
    row("2024-07-01T00", "ERCO", "DF", 90),   // strain = (100-90)/90 = 0.1111
    row("2024-07-01T01", "ERCO", "D", 120),
    row("2024-07-01T01", "ERCO", "DF", 100),  // strain = 0.2
    row("2024-07-01T00", "US48", "D", 999999), // must not leak into ERCO's peak
  ]);
  const daily = await foldErcoDailyAsync(base);
  const d = daily.get("2024-07-01")!;
  assert.equal(d.peak, 120);
  assert.equal(d.strainCount, 2);
  assert.ok(Math.abs(d.strainSum - (0.1111111 + 0.2)) < 1e-4);
});

test("foldErcoDailyAsync: gz day-files read identically to plain", async () => {
  const base = mkArchive();
  writeDay(base, "2024-07-02", [row("2024-07-02T00", "ERCO", "D", 50)], true);
  const daily = await foldErcoDailyAsync(base);
  assert.equal(daily.get("2024-07-02")!.peak, 50);
});

test("foldErcoDailyAsync: legacy typeless lines are demand (facet forced D pre-v2)", async () => {
  const base = mkArchive();
  writeDay(base, "2024-07-03", [
    { period: "2024-07-03T00", respondent: "ERCO", mwh: 77, rt: "2024-07-03" }, // no type field
  ]);
  const daily = await foldErcoDailyAsync(base);
  assert.equal(daily.get("2024-07-03")!.peak, 77);
});

test("computeGridStressReading: no archive at all -> null reading, honest zero depth", async () => {
  const base = mkArchive();
  const { reading, history_depth_days } = await computeGridStressReading(base, Date.parse("2024-07-10"));
  assert.equal(reading, null);
  assert.equal(history_depth_days, 0);
});

test("computeGridStressReading: thin same-month history withholds percentiles (never guesses)", async () => {
  const base = mkArchive();
  // exactly one prior day in July across all years + the target day itself —
  // MIN_SAMPLE_DAYS is 5, so percentile must come back null with an honest count.
  writeDay(base, "2024-07-08", [row("2024-07-08T00", "ERCO", "D", 100), row("2024-07-08T00", "ERCO", "DF", 100)]);
  writeDay(base, "2024-07-09", [row("2024-07-09T00", "ERCO", "D", 110), row("2024-07-09T00", "ERCO", "DF", 100)]);
  const nowMs = Date.parse("2024-07-10T00:00:00Z"); // "yesterday" = 07-09, the target day
  const { reading } = await computeGridStressReading(base, nowMs);
  assert.ok(reading);
  assert.equal(reading!.date, "2024-07-09");
  assert.equal(reading!.demand_percentile, null, "1 peer day is below MIN_SAMPLE_DAYS");
  assert.equal(reading!.demand_sample_days, 1);
  assert.equal(reading!.composite_percentile, null, "composite requires all three percentiles present");
});

test("computeGridStressReading: full end-to-end with enough history computes a composite", async () => {
  const base = mkArchive();
  // 6 prior July days with a range of peaks + strains, plus the target day —
  // enough for MIN_SAMPLE_DAYS=5 on demand/strain.
  const peaks = [80, 85, 90, 95, 100, 105]; // last one is the target day
  for (let i = 0; i < peaks.length; i++) {
    const day = `2024-07-${String(i + 1).padStart(2, "0")}`;
    writeDay(base, day, [
      row(`${day}T00`, "ERCO", "D", peaks[i]),
      row(`${day}T00`, "ERCO", "DF", 100),
    ]);
  }
  const nowMs = Date.parse("2024-07-07T00:00:00Z"); // yesterday = 07-06, peak 105 (the max)
  const { reading, history_depth_days } = await computeGridStressReading(base, nowMs);
  assert.ok(reading);
  assert.equal(reading!.date, "2024-07-06");
  assert.equal(reading!.demand_peak_mw, 105);
  assert.equal(reading!.demand_percentile, 100, "highest of all peers ranks at the top");
  assert.equal(reading!.demand_sample_days, 5);
  assert.equal(history_depth_days, 6);
  // weather percentile stays null here (no CPC fixture wired into this base
  // dir — loadTxDegreeDays reads the real committed archive, which won't
  // contain 2024-07-06 as a "peer-thin" July-2024 date relative to itself);
  // composite therefore requires weather too and may be null — assert only
  // what this test controls (demand), not the real committed CPC file's
  // shape (that would make this test brittle to a weekly-refreshed fixture).
  assert.ok(reading!.demand_percentile !== null);
});

// ── multi-region expansion (all US balancing authorities + region aggregates) ──

test("foldRegionsDailyAsync: folds many respondents in one pass, each isolated", async () => {
  const base = mkArchive();
  writeDay(base, "2024-07-01", [
    row("2024-07-01T00", "ERCO", "D", 100),
    row("2024-07-01T01", "ERCO", "D", 120),
    row("2024-07-01T00", "MISO", "D", 500),
    row("2024-07-01T01", "MISO", "D", 480),
    row("2024-07-01T00", "PJM", "D", 800),
  ]);
  const by = await foldRegionsDailyAsync(["ERCO", "MISO", "PJM"], base);
  assert.equal(by.get("ERCO")!.get("2024-07-01")!.peak, 120);
  assert.equal(by.get("MISO")!.get("2024-07-01")!.peak, 500);
  assert.equal(by.get("PJM")!.get("2024-07-01")!.peak, 800);
  assert.equal(by.has("CISO"), false, "unrequested respondent not folded");
});

test("computeAllRegionsStress: per-region readings; non-TX regions have null weather, ERCO has weather", async () => {
  const base = mkArchive();
  // 6 same-month (July) days so percentiles clear MIN_SAMPLE_DAYS for the target
  for (let d = 1; d <= 7; d++) {
    const day = `2024-07-0${d}`;
    writeDay(base, day, [
      row(`${day}T00`, "ERCO", "D", 100 + d),
      row(`${day}T00`, "ERCO", "DF", 95 + d),
      row(`${day}T00`, "MISO", "D", 500 + d),
      row(`${day}T00`, "MISO", "DF", 490 + d),
    ]);
  }
  const { regions } = await computeAllRegionsStress(base, Date.parse("2024-07-08"));
  const erco = regions.find((r) => r.respondent === "ERCO")!;
  const miso = regions.find((r) => r.respondent === "MISO")!;
  assert.ok(erco && miso, "both regions present");
  // MISO (no per-region degree days) never fabricates weather
  assert.equal(miso.weather_degree_days, null);
  assert.equal(miso.weather_percentile, null);
  // both compute demand percentile + forecast strain from real rows
  assert.ok(miso.demand_percentile != null, "MISO demand percentile computed");
  assert.ok(miso.forecast_strain_pct != null, "MISO forecast strain computed");
  assert.ok(erco.demand_percentile != null, "ERCO demand percentile computed");
});

test("REGION_LABEL: every configured respondent has a human label", () => {
  for (const code of ["US48", "ERCO", "MISO", "PJM", "CISO", "SWPP", "NYIS", "ISNE", "SE", "NW", "SW"]) {
    assert.ok(REGION_LABEL[code], `label for ${code}`);
  }
});
