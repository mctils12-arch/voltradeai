// EIA-930 grid generation-by-fuel-type battery: key gate, envelope parse,
// bracket-encoding + key-never-logged pins, observation-day day-files,
// respondent|period|fueltype dedup across fetches and restarts, and the
// negative-storage-value case that distinguishes this module's
// data-quality bound from gridDemand.ts's floor-at-zero one.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  gridGenerationEnabled, parseGeneration, generationUrl, fetchGeneration, archiveGeneration,
  refreshGeneration, latestGeneration, RESPONDENTS, HOURS_PER_FETCH,
  computeGenerationStats, readArchivedGeneration, _resetGridGenerationForTests,
} from "./gridGeneration";

// Mirrors the live EIA v2 shape verified 2026-09-09 (real EIA_API_KEY probe):
// value arrives as a STRING of MWh.
const ROW = (period: string, respondent: string, fueltype: string, value: string) => ({
  period, respondent, "respondent-name": "x", fueltype, "type-name": "x",
  value, "value-units": "megawatthours",
});
const ENVELOPE = (rows: unknown[]) => ({ response: { total: rows.length, data: rows } });

test("key gate: disabled without EIA_API_KEY; fetch returns [] keyless", async () => {
  assert.equal(gridGenerationEnabled({} as any), false);
  assert.equal(gridGenerationEnabled({ EIA_API_KEY: "x" } as any), true);
  let called = 0;
  const spy = async () => { called++; return { ok: true, status: 200, text: async () => "{}" }; };
  assert.deepEqual(await fetchGeneration(spy as any, {} as any, 0, 0), []);
  assert.equal(called, 0, "no key -> no network calls at all");
});

test("url: brackets encoded, key encoded, no type facet (fueltype rides free), bounded window", () => {
  const u = generationUrl("US48", "se cret");
  assert.ok(u.includes("facets%5Brespondent%5D%5B%5D=US48"), "brackets must be pre-encoded");
  assert.ok(!u.includes("facets["), "no raw brackets");
  assert.ok(u.includes("se%20cret"), "key URL-encoded");
  assert.ok(!u.includes("fueltype"), "fuel type is not a facet — the API returns every code per hour");
  assert.ok(u.includes(`length=${HOURS_PER_FETCH * 20}`));
});

test("parseGeneration: string MWh -> number, bad periods dropped, missing fueltype dropped", () => {
  const obs = parseGeneration(ENVELOPE([
    ROW("2026-09-08T21", "US48", "NG", "244668"),
    ROW("2026-09-08T21", "US48", "UES", "-18"),          // storage charging: legitimately negative
    ROW("2026-09-08T20", "US48", "SUN", ""),
    { period: "garbage", respondent: "US48", fueltype: "NG", value: "1" },
    { period: "2026-09-08T19", respondent: "CISO", value: "5" },  // no fueltype
  ]), "2026-09-09");
  assert.equal(obs.length, 3);
  assert.equal(obs[0].mwh, 244668);
  assert.equal(obs[1].fueltype, "UES");
  assert.equal(obs[1].mwh, -18, "storage negatives parse through unchanged");
  assert.equal(obs[2].mwh, null, "empty value stays null, never zero");
  assert.deepEqual(parseGeneration(null, "x"), []);
});

test("archive: respondent|period|fueltype dedup; hours land in their OBSERVATION day-file", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "gridgen-"));
  const obs = parseGeneration(ENVELOPE([
    ROW("2026-09-08T23", "US48", "NG", "1"),
    ROW("2026-09-08T23", "US48", "COL", "2"),   // same hour, different fuel type — not a dup
    ROW("2026-09-09T00", "US48", "NG", "3"),    // next UTC day
  ]), "2026-09-09");
  assert.equal(archiveGeneration(obs, base), 3);
  assert.equal(archiveGeneration(obs, base), 0, "same respondent+period+fueltype never re-archives");
  const dir = path.join(base, "gridgeneration");
  assert.ok(fs.existsSync(path.join(dir, "2026-09-08.jsonl")), "hour 23 in its own day");
  assert.ok(fs.existsSync(path.join(dir, "2026-09-09.jsonl")), "hour 00 in the next day");
  const lines = fs.readFileSync(path.join(dir, "2026-09-08.jsonl"), "utf8").trim().split("\n");
  assert.equal(lines.length, 2, "both fuel types for the same hour are distinct rows");
});

test("refresh sweep: one call per respondent, per-respondent fuel-mix cached", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "gridgen-"));
  let calls = 0;
  const ok = async (url: string) => {
    calls++;
    const resp = decodeURIComponent(url).match(/respondent\]\[\]=(\w+)/)?.[1] || "X";
    return { ok: true, status: 200,
             text: async () => JSON.stringify(ENVELOPE([
               ROW("2026-09-08T21", resp, "NG", "100"),
               ROW("2026-09-08T21", resp, "COL", "50"),
               ROW("2026-09-08T20", resp, "NG", "90"),   // older hour, same respondent
             ])) };
  };
  await refreshGeneration(ok as any, { EIA_API_KEY: "k" } as any, Date.parse("2026-09-08T22:00:00Z"), base, 0);
  assert.equal(calls, RESPONDENTS.length);
  const hit = latestGeneration();
  assert.ok(hit);
  assert.equal(hit!.stats.length, RESPONDENTS.length);
  assert.ok(hit!.stats.every((s) => s.latest_period === "2026-09-08T21" && s.hours_in_window === 2),
    "hours_in_window counts distinct periods, not rows");
  assert.ok(hit!.stats.every((s) => s.total_mwh === 150), "total sums only the latest-period rows");
  assert.ok(hit!.stats.every((s) => s.fuel_mix[0].fueltype === "NG" && s.fuel_mix[0].latest_mwh === 100),
    "fuel mix sorted by MWh desc");
});

test("data-quality gate: implausible generation rows are quarantined, storage negatives are NOT", async () => {
  const os2 = await import("node:os"); const p2 = await import("node:path");
  const base = fs.mkdtempSync(p2.join(os2.tmpdir(), "gdqgen-"));
  const good    = { period: "2026-09-09T00", respondent: "ERCO", fueltype: "NG", mwh: 30000, rt: "2026-09-09" };
  const storageNeg = { period: "2026-09-09T01", respondent: "ERCO", fueltype: "BAT", mwh: -500, rt: "2026-09-09" }; // legit
  const tooNeg  = { period: "2026-09-09T02", respondent: "ERCO", fueltype: "UES", mwh: -99999, rt: "2026-09-09" }; // implausible
  const huge    = { period: "2026-09-09T03", respondent: "ERCO", fueltype: "NG", mwh: 9_999_999, rt: "2026-09-09" }; // absurd
  const n = archiveGeneration([good, storageNeg, tooNeg, huge], base);
  assert.equal(n, 2, "the plausible positive row and the plausible storage-negative row are archived; the two absurd rows are quarantined");
});

// ── cold-cache-no-disk-backfill fix thread (research/open_questions.md's
// module audit table) — a cold boot or a live EIA outage (every respondent
// request failing/erroring) left /api/data/gridgeneration warming_up
// forever despite a real per-day archive already on disk. A/B-verified
// against the pre-fix shape: the old refresh only ever wrote `cache` inside
// `if (obs.length)`, so an all-respondents-empty sweep on a cold boot left
// `cache` permanently null even with archived days sitting right there.
// Same shape as gridDemand.ts's own thread entry, this module's direct
// sibling.

test("computeGenerationStats: pure aggregation, reused for both a live sweep and a backfilled archive day", () => {
  const obs = parseGeneration(ENVELOPE([
    ROW("2026-09-08T21", "US48", "NG", "244668"),
    ROW("2026-09-08T21", "US48", "COL", "50000"),
    ROW("2026-09-08T20", "US48", "NG", "200000"),
  ]), "2026-09-09");
  const stats = computeGenerationStats(obs);
  assert.equal(stats.length, 1);
  assert.equal(stats[0].respondent, "US48");
  assert.equal(stats[0].latest_period, "2026-09-08T21");
  assert.equal(stats[0].total_mwh, 294668);
  assert.equal(stats[0].hours_in_window, 2);
  assert.equal(stats[0].fuel_mix[0].fueltype, "NG");
});

test("readArchivedGeneration: returns the most recent archived day's raw obs, not a merged multi-day window", () => {
  _resetGridGenerationForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "gridgeneration-"));
  const day1 = parseGeneration(ENVELOPE([ROW("2026-09-07T10", "US48", "NG", "100")]), "2026-09-07");
  const day2 = parseGeneration(ENVELOPE([ROW("2026-09-08T10", "US48", "NG", "200")]), "2026-09-08");
  archiveGeneration(day1, base);
  archiveGeneration(day2, base);
  const obs = readArchivedGeneration(base, Date.parse("2026-09-09T00:00:00Z"));
  assert.ok(obs.length > 0);
  assert.ok(obs.every((o) => o.period.startsWith("2026-09-08")), "walks back to the newest day with an archive file, not a blend of both days");
});

test("readArchivedGeneration: gzipped days are read too, and an empty archive returns []", () => {
  _resetGridGenerationForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "gridgeneration-"));
  assert.deepEqual(readArchivedGeneration(base, Date.parse("2026-09-09T00:00:00Z")), [], "no archive on disk at all");
  const day = parseGeneration(ENVELOPE([ROW("2026-09-08T10", "US48", "NG", "200")]), "2026-09-08");
  archiveGeneration(day, base);
  const dir = path.join(base, "gridgeneration");
  fs.readdirSync(dir).forEach((f) => {
    if (!f.endsWith(".jsonl")) return;
    const fp = path.join(dir, f);
    fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
    fs.unlinkSync(fp);
  });
  const obs = readArchivedGeneration(base, Date.parse("2026-09-09T00:00:00Z"));
  assert.equal(obs.length, day.length);
});

test("refresh: a cold cache backfills from disk when every respondent request fails, aggregated exactly like a live result", async () => {
  _resetGridGenerationForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "gridgeneration-"));
  const day = parseGeneration(ENVELOPE([
    ROW("2026-09-08T21", "US48", "NG", "244668"),
    ROW("2026-09-08T21", "US48", "COL", "50000"),
  ]), "2026-09-08");
  archiveGeneration(day, base);
  assert.equal(latestGeneration(), null, "pre-fix baseline: cache starts cold");

  const allFail = async () => ({ ok: false, status: 500, text: async () => "" });
  await refreshGeneration(allFail as any, { EIA_API_KEY: "k" } as any,
                          Date.parse("2026-09-09T12:00:00Z"), base, 0);
  const hit = latestGeneration();
  assert.ok(hit, "cold cache backfilled from the on-disk archive instead of staying warming_up forever");
  assert.equal(hit!.stats.length, 1);
  assert.equal(hit!.stats[0].respondent, "US48");
  assert.equal(hit!.stats[0].total_mwh, 294668, "backfilled rows flow through the same computeGenerationStats aggregation as a live result");
});

test("refresh: an already-good cache is never clobbered by a transient all-respondents-empty sweep", async () => {
  _resetGridGenerationForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "gridgeneration-"));
  const ok = async () => ({ ok: true, status: 200,
    text: async () => JSON.stringify(ENVELOPE([ROW("2026-09-08T10", "US48", "NG", "500")])) });
  await refreshGeneration(ok as any, { EIA_API_KEY: "k" } as any,
                          Date.parse("2026-09-08T12:00:00Z"), base, 0);
  const first = latestGeneration();
  assert.ok(first);

  const allFail = async () => ({ ok: false, status: 500, text: async () => "" });
  await refreshGeneration(allFail as any, { EIA_API_KEY: "k" } as any,
                          Date.parse("2026-09-08T18:00:00Z"), base, 0);
  assert.deepEqual(latestGeneration(), first, "a transient all-failed sweep with a good cache already in hand leaves it untouched, not overwritten by an archive re-read");
});

test("refresh: cold cache with nothing archived either stays honestly null, never fabricates a result", async () => {
  _resetGridGenerationForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "gridgeneration-"));
  const allFail = async () => ({ ok: false, status: 500, text: async () => "" });
  await refreshGeneration(allFail as any, { EIA_API_KEY: "k" } as any,
                          Date.parse("2026-09-09T12:00:00Z"), base, 0);
  assert.equal(latestGeneration(), null, "no archive and no live result means warming_up is the honest state, not a fabricated one");
});
