// Edge P1 squeeze_gex stream: parse, registered rank definition, archive
// dedup/rollover, aggregate-only report, refresh failure handling.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseSqueezeCsv, trailingPctRank, archiveSqueezeRows, readAllSqueezeRows,
  gzipPastSqueezeYears, squeezeReport, refreshSqueezeGex, _resetSqueezeForTests, RANK_WINDOW,
} from "./squeezeGex";

// verbatim row shape from the live 2026-10-29 probe
const CSV = [
  "date,price,dix,gex",
  "2026-10-27,7722.6,0.4549768975156808,5515099082.650005",
  "2026-10-28,7743.98,0.46105040243405043,5605140552.230006",
  "garbage,1,2,3",
  "2026-10-29,7750,0.4,notanumber",
].join("\n") + "\n";

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "sqgex-"));

test("parseSqueezeCsv: verbatim shape, bad lines skipped, sorted", () => {
  const rows = parseSqueezeCsv(CSV, "T");
  assert.deepEqual(rows.map((r) => r.date), ["2026-10-27", "2026-10-28"]);
  assert.equal(rows[1].gex, 5605140552.230006);
  assert.deepEqual(parseSqueezeCsv("nope\n1,2"), []);
});

test("trailingPctRank: burn-in null, extremes 0/1, ties half, uses only past rows", () => {
  const v = [5, 1, 2, 3, 4];
  assert.deepEqual(trailingPctRank(v, 5), [null, null, null, null, 0.75]);
  assert.equal(trailingPctRank([1, 2, 3, 4, 5], 5)[4], 1);
  assert.equal(trailingPctRank([5, 4, 3, 2, 1], 5)[4], 0);
  assert.equal(trailingPctRank([2, 2, 2, 2, 2], 5)[4], 0.5);
  // appending a future value never changes an earlier rank
  assert.equal(trailingPctRank([...v, 100], 5)[4], trailingPctRank(v, 5)[4]);
  assert.equal(RANK_WINDOW, 252);
});

test("archive: dedups by date, reads back ascending, rolls past years to gz", () => {
  const d = tmp();
  const rows = parseSqueezeCsv("date,price,dix,gex\n2025-12-31,1,0.4,10\n2026-01-02,2,0.4,20\n", "T");
  assert.equal(archiveSqueezeRows(rows, d), 2);
  assert.equal(archiveSqueezeRows(rows, d), 0);
  assert.equal(gzipPastSqueezeYears(d, Date.parse("2026-06-01")), 1);
  assert.deepEqual(readAllSqueezeRows(d).map((r) => r.date), ["2025-12-31", "2026-01-02"]);
  // a late row for the rolled-over year lands and is still read
  assert.equal(archiveSqueezeRows(parseSqueezeCsv("date,price,dix,gex\n2025-12-30,1,0.4,9\n"), d), 1);
  assert.deepEqual(readAllSqueezeRows(d).map((r) => r.date), ["2025-12-30", "2025-12-31", "2026-01-02"]);
});

test("squeezeReport: aggregates only, burn-in rank null, negative-GEX flag", () => {
  const d = tmp();
  archiveSqueezeRows(parseSqueezeCsv("date,price,dix,gex\n2026-10-27,1,0.4,-5\n", "T"), d);
  const r = squeezeReport(d, Date.parse("2026-10-29T12:00:00Z"));
  assert.equal(r.rows, 1);
  assert.equal(r.latest_rank, null);
  assert.equal(r.latest_gex_negative, true);
  assert.equal(r.latest_age_days, 2);
  assert.ok(!JSON.stringify(r).includes('"gex"'), "no raw gex value in the report");
});

test("refreshSqueezeGex: success archives; HTTP error and empty body archive nothing and flag failure", async () => {
  const d = tmp(); _resetSqueezeForTests();
  const ok = async () => ({ ok: true, status: 200, text: async () => CSV });
  assert.equal(await refreshSqueezeGex(ok as any, d), 2);
  assert.equal(squeezeReport(d).last_refresh_ok, true);
  const bad = async () => ({ ok: false, status: 503, text: async () => "" });
  assert.equal(await refreshSqueezeGex(bad as any, d), 0);
  assert.equal(squeezeReport(d).last_refresh_ok, false);
  const empty = async () => ({ ok: true, status: 200, text: async () => "<html>" });
  assert.equal(await refreshSqueezeGex(empty as any, d), 0);
  assert.equal(readAllSqueezeRows(d).length, 2);
});
