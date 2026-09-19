// EIA-930 grid demand battery (DATACORE MAXIMUS Phase 0): key gate,
// envelope parse, bracket-encoding + key-never-logged pins, observation-day
// day-files, respondent|period dedup across fetches and restarts.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  gridDemandEnabled, parseDemand, demandUrl, fetchDemand, archiveDemand,
  refreshDemand, latestDemand, RESPONDENTS, HOURS_PER_FETCH,
  computeDemandStats, readArchivedDemand, _resetGridDemandForTests,
} from "./gridDemand";

// Mirrors the live EIA v2 shape verified 2026-07-06 (DEMO_KEY probe):
// value arrives as a STRING of MWh.
const ROW = (period: string, respondent: string, value: string) => ({
  period, respondent, "respondent-name": "x", type: "D", "type-name": "Demand",
  value, "value-units": "megawatthours",
});
const ENVELOPE = (rows: any[]) => ({ response: { total: rows.length, data: rows } });

test("key gate: disabled without EIA_API_KEY; fetch returns [] keyless", async () => {
  assert.equal(gridDemandEnabled({} as any), false);
  assert.equal(gridDemandEnabled({ EIA_API_KEY: "x" } as any), true);
  let called = 0;
  const spy = async () => { called++; return { ok: true, status: 200, text: async () => "{}" }; };
  assert.deepEqual(await fetchDemand(spy as any, {} as any, 0, 0), []);
  assert.equal(called, 0, "no key -> no network calls at all");
});

test("url: brackets encoded, key encoded, D+DF facets, bounded window", () => {
  const u = demandUrl("US48", "se cret");
  assert.ok(u.includes("facets%5Brespondent%5D%5B%5D=US48"), "brackets must be pre-encoded");
  assert.ok(!u.includes("facets["), "no raw brackets");
  assert.ok(u.includes("se%20cret"), "key URL-encoded");
  assert.ok(u.includes("facets%5Btype%5D%5B%5D=D&"), "demand series");
  assert.ok(u.includes("facets%5Btype%5D%5B%5D=DF"), "day-ahead forecast rides the same call (v2)");
  // window doubled deliberately: two series x the same 48h coverage
  assert.ok(u.includes(`length=${HOURS_PER_FETCH * 2}`));
});

test("archive v2: legacy typeless lines seed as D; DF same hour is a distinct event", () => {
  // MUST run before any other archiveDemand call in this file — the
  // seed pass fires once per process, and this test exercises it.
  // distinct respondent+period from every other test in this file — the
  // seen-set is process-global and spans baseDirs
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "grid-"));
  const dir = path.join(base, "griddemand");
  fs.mkdirSync(dir, { recursive: true });
  // pre-v2 archived line: no type field (the facet forced D back then)
  fs.writeFileSync(path.join(dir, "2026-07-05.jsonl"),
    JSON.stringify({ period: "2026-07-05T23", respondent: "ERCO", mwh: 1, rt: "2026-07-06" }) + "\n");
  const obs = parseDemand(ENVELOPE([
    ROW("2026-07-05T23", "ERCO", "1"),                       // dup of the legacy D line
    { ...ROW("2026-07-05T23", "ERCO", "2"), type: "DF", "type-name": "Day-ahead demand forecast" },
  ]), "2026-07-06");
  assert.equal(obs.length, 2);
  assert.equal(archiveDemand(obs, base), 1, "legacy line blocks the D dup; DF is fresh");
  const lines = fs.readFileSync(path.join(dir, "2026-07-05.jsonl"), "utf8").trim().split("\n");
  assert.equal(lines.length, 2);
  assert.equal(JSON.parse(lines[1]).type, "DF");
});

test("parseDemand: string MWh -> number, bad periods dropped, D+DF kept, others dropped", () => {
  const obs = parseDemand(ENVELOPE([
    ROW("2026-07-06T21", "US48", "678730"),
    ROW("2026-07-06T20", "US48", ""),
    { period: "garbage", respondent: "US48", type: "D", value: "1" },
    { period: "2026-07-06T19", respondent: "CISO", type: "NG", value: "5" },
    { ...ROW("2026-07-06T21", "US48", "690000"), type: "DF" },
  ]), "2026-07-06");
  assert.equal(obs.length, 3);
  assert.equal(obs[0].mwh, 678730);
  assert.equal(obs[0].type, "D");
  assert.equal(obs[1].mwh, null, "empty value stays null, never zero");
  assert.equal(obs[2].type, "DF", "day-ahead forecast rows kept and tagged");
  assert.deepEqual(parseDemand(null, "x"), []);
});

test("archive: respondent|period dedup; hours land in their OBSERVATION day-file", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "grid-"));
  const obs = parseDemand(ENVELOPE([
    ROW("2026-07-06T23", "US48", "1"),
    ROW("2026-07-07T00", "US48", "2"),   // next UTC day
  ]), "2026-07-07");
  assert.equal(archiveDemand(obs, base), 2);
  assert.equal(archiveDemand(obs, base), 0, "same hours never re-archive");
  const dir = path.join(base, "griddemand");
  assert.ok(fs.existsSync(path.join(dir, "2026-07-06.jsonl")), "hour 23 in its own day");
  assert.ok(fs.existsSync(path.join(dir, "2026-07-07.jsonl")), "hour 00 in the next day");
});

test("refresh sweep: one call per respondent, per-respondent stats cached", async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "grid-"));
  let calls = 0;
  const ok = async (url: string) => {
    calls++;
    const resp = decodeURIComponent(url).match(/respondent\]\[\]=(\w+)/)?.[1] || "X";
    return { ok: true, status: 200,
             text: async () => JSON.stringify(ENVELOPE([
               ROW("2026-07-06T21", resp, "100"), ROW("2026-07-06T20", resp, "90"),
               { ...ROW("2026-07-06T21", resp, "110"), type: "DF" },
               // day-ahead: a FUTURE-hour forecast exists but is partial at the
               // aggregate level (live-probed) — it must never be the readout
               { ...ROW("2026-07-06T23", resp, "7"), type: "DF" },
             ])) };
  };
  await refreshDemand(ok as any, { EIA_API_KEY: "k" } as any, Date.parse("2026-07-06T22:00:00Z"), base, 0);
  assert.equal(calls, RESPONDENTS.length);
  const hit = latestDemand();
  assert.ok(hit);
  assert.equal(hit!.stats.length, RESPONDENTS.length);
  assert.ok(hit!.stats.every((s) => s.latest_period === "2026-07-06T21" && s.hours_in_window === 2),
    "hours_in_window counts DEMAND rows only — meaning unchanged by v2");
  assert.ok(hit!.stats.every((s) => s.latest_mwh === 100 && s.latest_forecast_mwh === 110),
    "forecast is the SAME-hour DF, never the newest (future, partial) DF row");
});

test("backfill: opt-in gate, oldest-first years, pagination, done-marker single pass", async () => {
  const { gridDemandBackfillEnabled, gridDemandBackfillIfEnabled, backfillUrl, SEED_WINDOW_DAYS, seedFileInWindow } =
    await import("./gridDemand");
  assert.equal(gridDemandBackfillEnabled({} as any), false, "OFF by default (R8 lesson)");
  assert.equal(gridDemandBackfillEnabled({ GRID_DEMAND_BACKFILL: "1" } as any), true);

  const u = backfillUrl("ERCO", "k", 2019, 5000);
  assert.ok(u.includes("start=2019-01-01T00") && u.includes("end=2019-12-31T23"), "year-windowed");
  assert.ok(u.includes("direction%5D=asc"), "ascending walk");
  assert.ok(u.includes("offset=5000") && u.includes("length=5000"), "EIA v2 max page");
  assert.ok(u.includes("facets%5Btype%5D%5B%5D=DF"), "both series backfilled");

  const base = fs.mkdtempSync(path.join(os.tmpdir(), "grid-"));
  const years: number[] = [];
  let calls = 0;
  const fake = async (url: string) => {
    calls++;
    const y = Number(url.match(/start=(\d{4})/)?.[1]);
    years.push(y);
    // small page (< BACKFILL_PAGE) -> one page per respondent-year
    return { ok: true, status: 200,
             text: async () => JSON.stringify(ENVELOPE([
               { ...ROW(`${y}-03-01T05`, "US48", "77"), type: "D" },
               { ...ROW(`${y}-03-01T05`, "US48", "80"), type: "DF" },
             ])) };
  };
  const env = { GRID_DEMAND_BACKFILL: "1", EIA_API_KEY: "k" } as any;
  const now = Date.parse("2020-06-01T00:00:00Z"); // two-year walk: 2019, 2020
  await gridDemandBackfillIfEnabled(fake as any, env, now, base, 0);
  assert.equal(calls, 2 * RESPONDENTS.length, "one page per respondent per year");
  assert.ok(years.slice(0, RESPONDENTS.length).every((y) => y === 2019), "oldest year first");
  const dir = path.join(base, "griddemand");
  assert.ok(fs.existsSync(path.join(dir, "backfill_done.json")), "done-marker written");
  // rows landed in observation day-files, gz'd at end of pass (2019+2020 both old vs now)
  assert.ok(fs.existsSync(path.join(dir, "2019-03-01.jsonl.gz")), "backfilled day gz'd immediately");
  const marker = JSON.parse(fs.readFileSync(path.join(dir, "backfill_done.json"), "utf8"));
  assert.ok(marker.rows_archived >= 4, "D+DF rows for both years archived");
  assert.ok(marker.note.includes("delete this marker"), "re-run contract stated");

  calls = 0;
  await gridDemandBackfillIfEnabled(fake as any, env, now, base, 0);
  assert.equal(calls, 0, "done-marker makes the pass single-shot");

  // seed-window bound: heap protection after a deep backfill
  assert.ok(seedFileInWindow("2026-07-01.jsonl", Date.parse("2026-07-07")));
  assert.ok(!seedFileInWindow("2019-03-01.jsonl.gz", Date.parse("2026-07-07")),
    `files older than ${SEED_WINDOW_DAYS}d never seed the in-memory set`);
});

test("data-quality gate: implausible demand rows are quarantined, not archived", async () => {
  const os = await import("node:os"); const p2 = await import("node:path");
  const base = fs.mkdtempSync(p2.join(os.tmpdir(), "gdq-"));
  const good = { period: "2026-07-11T00", respondent: "ERCO", type: "D" as const, mwh: 60000, rt: "2026-07-11" };
  const neg  = { period: "2026-07-11T01", respondent: "ERCO", type: "D" as const, mwh: -5, rt: "2026-07-11" };     // impossible
  const huge = { period: "2026-07-11T02", respondent: "ERCO", type: "D" as const, mwh: 9_999_999, rt: "2026-07-11" }; // absurd
  const n = archiveDemand([good, neg, huge], base);
  assert.equal(n, 1, "only the plausible row is archived; the negative + absurd rows are quarantined");
});

// ── cold-cache-no-disk-backfill fix thread (research/open_questions.md's
// module audit table) — a cold boot or a live EIA outage (every respondent
// request failing/erroring) left /api/data/griddemand warming_up forever
// despite a real per-day archive already on disk. A/B-verified against the
// pre-fix shape: the old refresh only ever wrote `cache` inside `if
// (obs.length)`, so an all-respondents-empty sweep on a cold boot left
// `cache` permanently null even with archived days sitting right there.

test("computeDemandStats: pure aggregation, reused for both a live sweep and a backfilled archive day", () => {
  const obs = parseDemand({
    response: {
      data: [
        ROW("2026-07-06T21", "US48", "678730"),
        ROW("2026-07-06T20", "US48", "600000"),
        { ...ROW("2026-07-06T21", "US48", "690000"), type: "DF" },
      ],
    },
  }, "2026-07-07");
  const stats = computeDemandStats(obs);
  assert.equal(stats.length, 1);
  assert.equal(stats[0].respondent, "US48");
  assert.equal(stats[0].latest_period, "2026-07-06T21");
  assert.equal(stats[0].latest_mwh, 678730);
  assert.equal(stats[0].latest_forecast_mwh, 690000);
  assert.equal(stats[0].hours_in_window, 2);
});

test("readArchivedDemand: returns the most recent archived day's raw obs, not a merged multi-day window", () => {
  _resetGridDemandForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "griddemand-"));
  const day1 = parseDemand(ENVELOPE([ROW("2026-07-05T10", "US48", "100")]), "2026-07-05");
  const day2 = parseDemand(ENVELOPE([ROW("2026-07-06T10", "US48", "200")]), "2026-07-06");
  archiveDemand(day1, base);
  archiveDemand(day2, base);
  const obs = readArchivedDemand(base, Date.parse("2026-07-07T00:00:00Z"));
  assert.ok(obs.length > 0);
  assert.ok(obs.every((o) => o.period.startsWith("2026-07-06")), "walks back to the newest day with an archive file, not a blend of both days");
});

test("readArchivedDemand: gzipped days are read too, and an empty archive returns []", () => {
  _resetGridDemandForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "griddemand-"));
  assert.deepEqual(readArchivedDemand(base, Date.parse("2026-07-07T00:00:00Z")), [], "no archive on disk at all");
  const day = parseDemand(ENVELOPE([ROW("2026-07-06T10", "US48", "200")]), "2026-07-06");
  archiveDemand(day, base);
  const dir = path.join(base, "griddemand");
  fs.readdirSync(dir).forEach((f) => {
    if (!f.endsWith(".jsonl")) return;
    const fp = path.join(dir, f);
    fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
    fs.unlinkSync(fp);
  });
  const obs = readArchivedDemand(base, Date.parse("2026-07-07T00:00:00Z"));
  assert.equal(obs.length, day.length);
});

test("refresh: a cold cache backfills from disk when every respondent request fails, aggregated exactly like a live result", async () => {
  _resetGridDemandForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "griddemand-"));
  const day = parseDemand(ENVELOPE([
    ROW("2026-07-06T21", "US48", "678730"),
    ROW("2026-07-06T20", "US48", "600000"),
  ]), "2026-07-06");
  archiveDemand(day, base);
  assert.equal(latestDemand(), null, "pre-fix baseline: cache starts cold");

  const allFail = async () => ({ ok: false, status: 500, text: async () => "" });
  await refreshDemand(allFail as any, { EIA_API_KEY: "k" } as any,
                      Date.parse("2026-07-07T12:00:00Z"), base, 0);
  const hit = latestDemand();
  assert.ok(hit, "cold cache backfilled from the on-disk archive instead of staying warming_up forever");
  assert.equal(hit!.stats.length, 1);
  assert.equal(hit!.stats[0].respondent, "US48");
  assert.equal(hit!.stats[0].latest_mwh, 678730, "backfilled rows flow through the same computeDemandStats aggregation as a live result");
});

test("refresh: an already-good cache is never clobbered by a transient all-respondents-empty sweep", async () => {
  _resetGridDemandForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "griddemand-"));
  const ok = async () => ({ ok: true, status: 200,
    text: async () => JSON.stringify(ENVELOPE([ROW("2026-07-06T10", "US48", "500")])) });
  await refreshDemand(ok as any, { EIA_API_KEY: "k" } as any,
                      Date.parse("2026-07-06T12:00:00Z"), base, 0);
  const first = latestDemand();
  assert.ok(first);

  const allFail = async () => ({ ok: false, status: 500, text: async () => "" });
  await refreshDemand(allFail as any, { EIA_API_KEY: "k" } as any,
                      Date.parse("2026-07-06T18:00:00Z"), base, 0);
  assert.deepEqual(latestDemand(), first, "a transient all-failed sweep with a good cache already in hand leaves it untouched, not overwritten by an archive re-read");
});

test("refresh: cold cache with nothing archived either stays honestly null, never fabricates a result", async () => {
  _resetGridDemandForTests();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "griddemand-"));
  const allFail = async () => ({ ok: false, status: 500, text: async () => "" });
  await refreshDemand(allFail as any, { EIA_API_KEY: "k" } as any,
                      Date.parse("2026-07-07T12:00:00Z"), base, 0);
  assert.equal(latestDemand(), null, "no archive and no live result means warming_up is the honest state, not a fabricated one");
});
