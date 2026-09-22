// gnssIntegrityDaily battery: pins the permanent-archive fix found this
// session (research/experiments.md's dated entry) — datacoreArchive's
// generic rollup drops the nic/pos_type/altitude fields the gnss_integrity_
// adsb root reads, capping its usable depth at RAW_RETENTION_DAYS no matter
// how long the system runs. Mirrors fleetUtilization.test.ts's own
// preserveWeeklyBeforeRollup battery (fold-then-survive-deletion,
// idempotent-before-deletion, additive-across-ticks), adapted for this
// module's band x origin cell shape instead of per-owner flight deltas.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  preserveGnssIntegrityDailyBeforeRollup, loadGnssIntegrityDailyArchive,
  _gnssDailyArchivePathForTests,
} from "./gnssIntegrityDaily";

const MON = Date.parse("2026-06-29T00:00:00Z") / 1000; // a Monday, seconds
const hourFile = (ms: number) => new Date(ms).toISOString().slice(0, 13).replace("T", "-") + ".jsonl";

// Baltic candidate bbox (53,60,17,24) — one point well inside it.
const CANDIDATE_PT = { la: 55, lo: 20 };
// Control bbox (35,55,-80,10) — one point well inside it (Paris-ish),
// deliberately outside the candidate bbox too.
const CONTROL_PT = { la: 48, lo: 2 };
// Outside BOTH bboxes (Tokyo-ish) — must never contaminate either cell set.
const NEITHER_PT = { la: 35.6, lo: 139.7 };

function setup(prefix: string) {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  fs.mkdirSync(path.join(base, "aircraft"), { recursive: true });
  return base;
}

test("preserveGnssIntegrityDailyBeforeRollup folds an aged-out hour file into per-band/origin cells for BOTH candidate and control bboxes, survives deletion, and does not double-count if called again before deletion", async () => {
  const base = setup("gnss-daily-");
  const oldMs = (MON + 10 * 3600) * 1000;
  const t0 = Math.floor(oldMs / 1000);
  const fname = hourFile(oldMs);
  const rows = [
    // candidate region, cruise, broadcast, nic==0 -> the jamming signature
    { t: t0, i: "aaa111", la: CANDIDATE_PT.la, lo: CANDIDATE_PT.lo, al: 10000, ni: 0, pt: "adsb_icao" },
    // candidate region, cruise, broadcast, nic==1 (healthy reading, same cell denominator)
    { t: t0 + 60, i: "aaa112", la: CANDIDATE_PT.la, lo: CANDIDATE_PT.lo, al: 10000, ni: 1, pt: "adsb_icao" },
    // control region, cruise, broadcast, nic==1 (healthy — the control baseline)
    { t: t0 + 120, i: "bbb221", la: CONTROL_PT.la, lo: CONTROL_PT.lo, al: 10000, ni: 1, pt: "adsb_icao" },
    // outside both bboxes — must not appear in either cell set
    { t: t0 + 180, i: "ccc331", la: NEITHER_PT.la, lo: NEITHER_PT.lo, al: 10000, ni: 0, pt: "adsb_icao" },
    // candidate region but no `ni` field at all — excluded from every denominator (absence is not zero)
    { t: t0 + 240, i: "aaa113", la: CANDIDATE_PT.la, lo: CANDIDATE_PT.lo, al: 10000, pt: "adsb_icao" },
  ];
  fs.writeFileSync(path.join(base, "aircraft", fname), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  const nowMs = oldMs + 31 * 86400_000; // past the default 30-day retention

  const r1 = await preserveGnssIntegrityDailyBeforeRollup(base, nowMs);
  assert.equal(r1.filesFolded, 1);
  assert.equal(r1.daysTouched, 1);

  // idempotent: raw file still present (rollup hasn't deleted it yet) -> no-op
  const r2 = await preserveGnssIntegrityDailyBeforeRollup(base, nowMs);
  assert.equal(r2.filesFolded, 0, "already-folded file is not re-processed");

  // simulate the generic rollup deleting the raw file afterward
  fs.unlinkSync(path.join(base, "aircraft", fname));

  const days = loadGnssIntegrityDailyArchive(base);
  const day = new Date(oldMs).toISOString().slice(0, 10);
  assert.ok(days[day], "day entry survives raw-file deletion");
  const cCell = days[day].candidate.find((c) => c.band === "cruise" && c.origin === "broadcast");
  assert.ok(cCell, "candidate cruise/broadcast cell present");
  assert.equal(cCell!.n_total, 2, "the no-ni row is excluded from the denominator");
  assert.equal(cCell!.n_zero, 1);
  assert.equal(cCell!.distinct_airframes, 2);

  const ctrlCell = days[day].control.find((c) => c.band === "cruise" && c.origin === "broadcast");
  assert.ok(ctrlCell, "control cruise/broadcast cell present");
  assert.equal(ctrlCell!.n_total, 1);
  assert.equal(ctrlCell!.n_zero, 0, "control baseline row was a healthy nic==1 reading");

  // the outside-both-bboxes row must never inflate either cell set
  const candidateTotal = days[day].candidate.reduce((s, c) => s + c.n_total, 0);
  const controlTotal = days[day].control.reduce((s, c) => s + c.n_total, 0);
  assert.equal(candidateTotal, 2);
  assert.equal(controlTotal, 1);
});

test("preserveGnssIntegrityDailyBeforeRollup: a day spanning two rollup ticks accumulates additively, not overwritten", async () => {
  const base = setup("gnss-daily2-");
  const day1Ms = (MON + 10 * 3600) * 1000;
  const day2Ms = day1Ms + 3600_000; // one hour later, same calendar day, separate hour file

  const f1 = hourFile(day1Ms);
  fs.writeFileSync(path.join(base, "aircraft", f1),
    JSON.stringify({ t: Math.floor(day1Ms / 1000), i: "aaa111", la: CANDIDATE_PT.la, lo: CANDIDATE_PT.lo, al: 10000, ni: 0, pt: "adsb_icao" }) + "\n");
  const r1 = await preserveGnssIntegrityDailyBeforeRollup(base, day1Ms + 31 * 86400_000);
  assert.equal(r1.filesFolded, 1);
  fs.unlinkSync(path.join(base, "aircraft", f1));

  const f2 = hourFile(day2Ms);
  fs.writeFileSync(path.join(base, "aircraft", f2),
    JSON.stringify({ t: Math.floor(day2Ms / 1000), i: "aaa112", la: CANDIDATE_PT.la, lo: CANDIDATE_PT.lo, al: 10000, ni: 0, pt: "adsb_icao" }) + "\n");
  const r2 = await preserveGnssIntegrityDailyBeforeRollup(base, day2Ms + 31 * 86400_000);
  assert.equal(r2.filesFolded, 1);
  fs.unlinkSync(path.join(base, "aircraft", f2));

  const day = new Date(day1Ms).toISOString().slice(0, 10);
  const days = loadGnssIntegrityDailyArchive(base);
  const cCell = days[day].candidate.find((c) => c.band === "cruise" && c.origin === "broadcast");
  assert.equal(cCell!.n_total, 2, "both hour files' rows accumulate in the same day across two preserve calls, not overwritten");
  assert.equal(cCell!.n_zero, 2);
});

test("preserveGnssIntegrityDailyBeforeRollup: a file not yet past retention is left alone (no-op)", async () => {
  const base = setup("gnss-daily3-");
  const recentMs = Date.now() - 5 * 86400_000; // 5 days old, well within the 30-day default
  fs.writeFileSync(path.join(base, "aircraft", hourFile(recentMs)),
    JSON.stringify({ t: Math.floor(recentMs / 1000), i: "aaa111", la: CANDIDATE_PT.la, lo: CANDIDATE_PT.lo, al: 10000, ni: 0, pt: "adsb_icao" }) + "\n");
  const r = await preserveGnssIntegrityDailyBeforeRollup(base, Date.now());
  assert.equal(r.filesFolded, 0);
  assert.deepEqual(loadGnssIntegrityDailyArchive(base), {});
});

test("preserveGnssIntegrityDailyBeforeRollup is a safe no-op with no aircraft directory or no eligible files", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "gnss-daily4-"));
  const r = await preserveGnssIntegrityDailyBeforeRollup(base, Date.now());
  assert.equal(r.filesFolded, 0);
  assert.equal(r.daysTouched, 0);
});

test("loadGnssIntegrityDailyArchive returns an honest empty object when no archive file exists yet", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "gnss-daily5-"));
  assert.deepEqual(loadGnssIntegrityDailyArchive(base), {});
  assert.ok(!fs.existsSync(_gnssDailyArchivePathForTests(base)));
});

test("preserveGnssIntegrityDailyBeforeRollup survives a corrupt/truncated hour file rather than crashing", async () => {
  const base = setup("gnss-daily6-");
  const oldMs = (MON + 10 * 3600) * 1000;
  const good = Buffer.from(JSON.stringify({ t: Math.floor(oldMs / 1000), i: "aaa111", la: CANDIDATE_PT.la, lo: CANDIDATE_PT.lo, al: 10000, ni: 0, pt: "adsb_icao" }) + "\n");
  // Write a plain (non-gz) file with a .gz extension so decompression fails immediately.
  fs.writeFileSync(path.join(base, "aircraft", hourFile(oldMs) + ".gz"), good);
  const r = await preserveGnssIntegrityDailyBeforeRollup(base, oldMs + 31 * 86400_000);
  // The file is still counted as processed (best-effort: never re-attempted forever),
  // and no crash/throw propagates — days may be empty since nothing parsed.
  assert.equal(r.filesFolded, 1);
});
