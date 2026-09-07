/**
 * portdwell_weekly_snapshot.ts — captures completed-week snapshots into
 * `datacore/port_dwell_weekly.json`, the durable accumulator
 * `server/portDwellWeekly.ts` defines (see that module's header for the
 * full rationale: raw retention rolls forward with "now" and can never
 * accumulate on its own, so GATE 2's weekly series has to be captured one
 * week at a time, before each week ages out of the rolling window).
 *
 * PREFERRED PATH (added 2026-09-07, follow-up to server/portDwellCapture.ts,
 * v1.0.859): first merges in whatever `server/portDwellCapture.ts`'s own
 * Tier-3 in-process job has already captured, via the cheap, read-only
 * `/api/diag/portdwell_weekly_captured` probe -- a small JSON-file read, no
 * fold, no proxy-timeout exposure regardless of how large the vessel
 * archive grows. That module's own header named this exact migration as a
 * deliberately deferred follow-up ("Migrating that script to prefer the
 * cheap probe is deliberately left for a follow-up session") -- this is it.
 *
 * FALLBACK PATH (the original mechanism, now only exercised for weeks the
 * server hasn't captured yet -- e.g. a fresh deploy before the Tier-3 job
 * has caught up, or historical backfill from before that job existed):
 * calls the older `/api/diag/portdwell_window` probe once per still-missing
 * week, `end` pinned to that week's own end (the fixed archive-start-
 * anchored grid `portDwellWeekly.ts` defines, not "now"), `hours=168`. Two
 * conditions SKIP a week rather than persisting it degraded: a
 * `coverage_caveat` (part of the window falls outside current raw
 * retention -- once retention rolls past a week for good, its true history
 * is permanently unreachable) and `isDegenerateAllZeroRead` (every port
 * reads zero completed AND zero ongoing visits despite vessels being seen
 * -- the signature of either a broken reader or a week fully swallowed by
 * an archive-feed outage; confirmed live for week index 5, 2026-08-07..
 * 08-14, which overlaps the already-diagnosed 2026-08-05..08-12
 * aisstream.io outage almost entirely). Either way, writing a silently-zero
 * number that looks identical to a genuine quiet week would poison every
 * future statistic computed over this series. Per the 2026-09-06
 * measurement (research/experiments.md, same date): this fold is CPU-bound
 * and its cost does not amortize with window size, so this fallback path
 * can still hit the platform's proxy timeout for a dense/recent week --
 * exactly the failure mode the preferred path above exists to avoid for
 * every week the server gets to first.
 *
 * The one-off "how far does raw retention reach" boundary check below asks
 * for `hours=1` rather than the historical default of 168 -- the field it
 * actually reads (`raw_vessel_archive_from`) is archive-wide metadata
 * (`oldestRawHour`, server/bot.ts's `portdwell_window` case), unaffected by
 * the requested window size, so there is no reason to pay for a 168h fold
 * (the single most expensive window this script could request, since it is
 * always the most data-dense) just to read it.
 *
 * Idempotent and safe to run every session: a week already in the file is
 * never re-fetched or overwritten (`missingWeekIndices`/`mergeWeeklySnapshot`
 * /`mergeCapturedWeeks`).
 *
 * Usage: DIAG_TOKEN=... npx tsx scripts/portdwell_weekly_snapshot.ts [prodBaseUrl]
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  weekBounds, lastCompletedWeekIndex, missingWeekIndices, mergeWeeklySnapshot,
  extractWeeklySnapshot, isDegenerateAllZeroRead, ARCHIVE_START_MS, WeeklySnapshot,
} from "../server/portDwellWeekly.ts";
import type { PortDwellStats } from "../server/portDwell.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
const OUT_FILE = path.join(here, "..", "datacore", "port_dwell_weekly.json");

const BASE = process.argv[2] || process.env.VOLTRADE_PROD_URL || "https://voltradeai-production.up.railway.app";
const TOKEN = process.env.DIAG_TOKEN;

function loadExisting(): WeeklySnapshot[] {
  try {
    return JSON.parse(fs.readFileSync(OUT_FILE, "utf8"));
  } catch {
    return [];
  }
}

interface PortdwellWindowResponse {
  stats: PortDwellStats;
  coverage_caveat: string | null;
  raw_vessel_archive_from: string | null;
}

// One shared fetch helper (one AbortSignal.timeout cast site, not one per
// call site) -- serves both the raw-retention boundary check and each
// per-week fallback capture below. `end` omitted means "the current rolling
// window". `hours` defaults to 168 (a real week, needed for a genuine
// per-week capture); the boundary check overrides it to 1 (see module
// header -- only `raw_vessel_archive_from` is read from that call, and it
// does not vary with the requested window size).
async function fetchWindow(endMs?: number, hours = 168): Promise<PortdwellWindowResponse> {
  const endParam = endMs != null ? `&end=${new Date(endMs).toISOString()}` : "";
  const url = `${BASE}/api/diag/portdwell_window?hours=${hours}${endParam}&token=${TOKEN}`;
  const r = await fetch(url, { signal: AbortSignal.timeout(180000) as unknown as AbortSignal });
  if (!r.ok) throw new Error(`${url} -> ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const body = await r.json();
  const { coverage_caveat, raw_vessel_archive_from, ...stats } = body;
  return {
    stats: stats as PortDwellStats,
    coverage_caveat: coverage_caveat ?? null,
    raw_vessel_archive_from: raw_vessel_archive_from ?? null,
  };
}

// Cheap, read-only passthrough of server/portDwellCapture.ts's own Tier-3
// in-process capture state -- a small JSON-file read server-side, no fold,
// no proxy-timeout risk regardless of archive size. Preferred source for
// any week the server has already captured.
async function fetchCapturedWeeks(): Promise<WeeklySnapshot[]> {
  const url = `${BASE}/api/diag/portdwell_weekly_captured?token=${TOKEN}`;
  const r = await fetch(url, { signal: AbortSignal.timeout(30000) as unknown as AbortSignal });
  if (!r.ok) throw new Error(`${url} -> ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const body = await r.json();
  return Array.isArray(body.weeks) ? (body.weeks as WeeklySnapshot[]) : [];
}

/** Pure: folds each server-captured week into `existing` via the same
 *  never-overwrite merge every other capture path in this file already
 *  uses (`mergeWeeklySnapshot` -- a week already present is never replaced,
 *  and the result is always sorted by week_index). Order of `captured`
 *  does not matter. No fs/network here, so this is unit-testable without a
 *  live server. */
export function mergeCapturedWeeks(existing: WeeklySnapshot[], captured: WeeklySnapshot[]): WeeklySnapshot[] {
  return captured.reduce((acc, snap) => mergeWeeklySnapshot(acc, snap), existing);
}

async function main() {
  if (!TOKEN) {
    console.log(JSON.stringify({ verdict: "ERROR", error: "DIAG_TOKEN not set in environment" }));
    process.exitCode = 1;
    return;
  }

  const nowMs = Date.now();
  let existing = loadExisting();

  // PREFERRED PATH: merge in whatever the server's own Tier-3 in-process
  // job has already captured (cheap, no fold) before attempting anything
  // expensive. A stale deployed server without this probe yet (or any
  // other fetch failure) degrades gracefully to the fallback loop only --
  // this call adding nothing is not an error condition.
  const beforeServerMerge = new Set(existing.map((s) => s.week_index));
  let weeksFromServerCapture: number[] = [];
  try {
    const capturedFromServer = await fetchCapturedWeeks();
    existing = mergeCapturedWeeks(existing, capturedFromServer);
    weeksFromServerCapture = capturedFromServer
      .map((s) => s.week_index)
      .filter((idx) => !beforeServerMerge.has(idx));
    if (weeksFromServerCapture.length > 0) {
      console.error(`[portdwell_weekly_snapshot] merged ${weeksFromServerCapture.length} week(s) from the server's own Tier-3 capture state: ${weeksFromServerCapture.join(", ")}`);
    }
  } catch (e: unknown) {
    console.error(`[portdwell_weekly_snapshot] server-captured-weeks probe unavailable, falling back to the per-week loop only: ${e instanceof Error ? e.message : String(e)}`);
  }

  // FALLBACK PATH boundary check: ask the live probe how far raw retention
  // currently reaches (hours=1 -- see module header), then find the
  // earliest week whose full 7 days sit at or after that boundary -- never
  // assume week 0 (archive start) is reachable; it almost never is.
  const boundary = await fetchWindow(undefined, 1);
  const rawArchiveFromMs: number | null = boundary.raw_vessel_archive_from ? Date.parse(boundary.raw_vessel_archive_from) : null;
  const earliestIndex = rawArchiveFromMs != null
    ? Math.ceil((rawArchiveFromMs - ARCHIVE_START_MS) / (7 * 24 * 3600_000))
    : 0;

  const candidates = missingWeekIndices(existing, Math.max(earliestIndex, 0), nowMs);
  const captured: number[] = [];
  const skipped: Array<{ week_index: number; reason: string }> = [];

  for (const idx of candidates) {
    const week = weekBounds(idx);
    let resp: PortdwellWindowResponse;
    try {
      resp = await fetchWindow(week.endMs);
    } catch (e: unknown) {
      skipped.push({ week_index: idx, reason: `fetch error: ${e instanceof Error ? e.message : String(e)}` });
      continue;
    }
    if (resp.coverage_caveat) {
      skipped.push({ week_index: idx, reason: "coverage_caveat present (partial raw coverage for this week)" });
      continue;
    }
    if (isDegenerateAllZeroRead(resp.stats)) {
      skipped.push({
        week_index: idx,
        reason: `all-zero read despite ${resp.stats.vessels_seen} vessels_seen -- likely a feed outage or reader defect, not a real quiet week`,
      });
      continue;
    }
    const snap = extractWeeklySnapshot(resp.stats, week, Date.now());
    existing = mergeWeeklySnapshot(existing, snap);
    captured.push(idx);
    console.error(`[portdwell_weekly_snapshot] captured week ${idx} (${snap.week_start} -> ${snap.week_end})`);
  }

  if (captured.length > 0 || weeksFromServerCapture.length > 0) {
    fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true });
    fs.writeFileSync(OUT_FILE, JSON.stringify(existing, null, 2) + "\n");
  }

  console.log(JSON.stringify({
    raw_vessel_archive_from: boundary.raw_vessel_archive_from,
    earliest_attemptable_week_index: earliestIndex,
    last_completed_week_index: lastCompletedWeekIndex(nowMs),
    weeks_captured_from_server_state: weeksFromServerCapture,
    weeks_captured_this_run: captured,
    weeks_skipped_this_run: skipped,
    total_weeks_in_file: existing.length,
    file: path.relative(path.join(here, ".."), OUT_FILE),
  }, null, 2));
}

// Guarded (2026-09-07, added alongside the new `mergeCapturedWeeks` export
// and its test file): this module is now `import`-ed by
// portdwell_weekly_snapshot.test.ts for the pure helper above, and an
// unguarded top-level `main()` call fired live network requests against
// production on every test run -- confirmed live before this guard was
// added (a real 502 from prod on both live calls this test run made; no
// write happened only because both calls failed, not because anything
// prevented the attempt). Same guard convention as every other
// scripts/*.ts file with a sibling .test.ts (wikiattention_gate1.ts,
// finra_shortvol_gate2.ts, gdelt_fires_gate2.ts).
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => {
    console.error(JSON.stringify({ verdict: "ERROR", error: e?.message || String(e) }));
    process.exitCode = 1;
  });
}
