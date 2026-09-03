/**
 * portdwell_weekly_snapshot.ts — captures completed-week snapshots into
 * `datacore/port_dwell_weekly.json`, the durable accumulator
 * `server/portDwellWeekly.ts` defines (see that module's header for the
 * full rationale: raw retention rolls forward with "now" and can never
 * accumulate on its own, so GATE 2's weekly series has to be captured one
 * week at a time, before each week ages out of the rolling window).
 *
 * Calls the existing `/api/diag/portdwell_window` probe (read-only, token-
 * gated, unchanged) once per missing week, `end` pinned to that week's own
 * end (the fixed archive-start-anchored grid `portDwellWeekly.ts` defines,
 * not "now"), `hours=168`. Two conditions SKIP a week rather than persisting
 * it degraded: a `coverage_caveat` (part of the window falls outside current
 * raw retention -- once retention rolls past a week for good, its true
 * history is permanently unreachable) and `isDegenerateAllZeroRead` (every
 * port reads zero completed AND zero ongoing visits despite vessels being
 * seen -- the signature of either a broken reader or a week fully swallowed
 * by an archive-feed outage; confirmed live this session for week index 5,
 * 2026-08-07..08-14, which overlaps the already-diagnosed 2026-08-05..08-12
 * aisstream.io outage almost entirely). Either way, writing a silently-zero
 * number that looks identical to a genuine quiet week would poison every
 * future statistic computed over this series.
 *
 * Idempotent and safe to run every session: a week already in the file is
 * never re-fetched or overwritten (`missingWeekIndices`/`mergeWeeklySnapshot`).
 *
 * Usage: DIAG_TOKEN=... npx tsx scripts/portdwell_weekly_snapshot.ts [prodBaseUrl]
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
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
// per-week capture below. `end` omitted means "the current rolling window",
// used only to read `raw_vessel_archive_from`.
async function fetchWindow(endMs?: number): Promise<PortdwellWindowResponse> {
  const endParam = endMs != null ? `&end=${new Date(endMs).toISOString()}` : "";
  const url = `${BASE}/api/diag/portdwell_window?hours=168${endParam}&token=${TOKEN}`;
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

async function main() {
  if (!TOKEN) {
    console.log(JSON.stringify({ verdict: "ERROR", error: "DIAG_TOKEN not set in environment" }));
    process.exitCode = 1;
    return;
  }

  const nowMs = Date.now();
  let existing = loadExisting();

  // Earliest week worth attempting: ask the live probe (current rolling
  // 168h window) how far raw retention currently reaches, then find the
  // earliest week whose full 7 days sit at or after that boundary -- never
  // assume week 0 (archive start) is reachable; it almost never is.
  const boundary = await fetchWindow();
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

  if (captured.length > 0) {
    fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true });
    fs.writeFileSync(OUT_FILE, JSON.stringify(existing, null, 2) + "\n");
  }

  console.log(JSON.stringify({
    raw_vessel_archive_from: boundary.raw_vessel_archive_from,
    earliest_attemptable_week_index: earliestIndex,
    last_completed_week_index: lastCompletedWeekIndex(nowMs),
    weeks_captured_this_run: captured,
    weeks_skipped_this_run: skipped,
    total_weeks_in_file: existing.length,
    file: path.relative(path.join(here, ".."), OUT_FILE),
  }, null, 2));
}

main().catch((e) => {
  console.error(JSON.stringify({ verdict: "ERROR", error: e?.message || String(e) }));
  process.exitCode = 1;
});
