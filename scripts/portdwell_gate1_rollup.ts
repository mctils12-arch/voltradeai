/**
 * portdwell_gate1_rollup.ts — GATE 1 (DATA) for port_dwell_maritime_transit,
 * attempted via the rollup-summary reader (server/portDwell.ts's
 * portPresenceFromRollup/summarizeRollupPresence, this same session).
 *
 * BACKGROUND (research/open_questions.md "PORT DWELL ANALYTICS"): the
 * 2026-08-19 GATE 1 session found the raw-archive pipeline
 * (computePortDwellAsync/detectVisits, RAW_RETENTION_DAYS=30) could not
 * reach July 2026 — every raw hour file that old had already been rolled
 * into a coarse per-day summary and deleted, under the SHORTER 7-day
 * retention that applied before PR #760 (2026-08-11) widened it to 30. That
 * session declared the specific "July 2026 vs Port of LA's published July
 * TEU figure" comparison "CLOSED AS PERMANENTLY UNATTAINABLE via this data
 * path" and filed the alternative as NEXT: "build a rollup-summary-format
 * reader." Never attempted until this session.
 *
 * WHAT CHANGED THIS SESSION that makes this runnable: (1) confirmed live
 * that `vessels_tracks/<day>.jsonl.gz` rollup summaries genuinely exist back
 * to at least 2026-07-15 (`/api/diag/archive?stream=vessels_tracks&day=...`)
 * — the raw data was never lost, only folded; (2) found the existing
 * `/api/diag/archive` bbox filter only matched top-level lat/lon (point
 * rows), so a bbox-scoped `stream=vessels_tracks` query matched ZERO rows
 * — fixed in datacoreArchive.ts's new `rowInBbox` (now also matches a
 * rollup row's own `[minLa,minLo,maxLa,maxLo]` bbox array on overlap);
 * (3) built portPresenceFromRollup/summarizeRollupPresence — a coarse,
 * DAY-granularity port-presence detector over rollup rows (see their own
 * header comments in server/portDwell.ts for the full honesty accounting:
 * lower-bound on true dwell length, and can OVER-count call count when a
 * day's ~50-point subsample happens to miss the one in-fence point).
 *
 * TRUTH SOURCE (already gathered by the 2026-08-19 session, re-stated here
 * verbatim, not re-fetched): Port of LA's July 2026 news release
 * (portoflosangeles.org, `news_081826_july_cargo`) reports 960,464 TEUs for
 * July — second-busiest July on record, following an even stronger June
 * (>1M TEUs) — and 6,083,067 TEUs year-to-date through July, +1.8% YoY.
 *
 * PRE-REGISTERED BAR (matches the 2026-08-19 entry's own framing, which
 * this script is completing, not redesigning): TEU counts are container
 * throughput, not vessel-call/presence counts, so an exact reconciliation
 * was never the right bar. The falsifiable claim: July 2026's rollup-
 * derived port_la/port_lb activity should read as a HIGH-ACTIVITY month —
 * same order of magnitude as a FRESH raw-pipeline baseline pulled this same
 * session for a recent, uncontroversial week — not anomalously low. A
 * result an order of magnitude below the fresh baseline (after normalizing
 * for the ~4.3x week-to-month scaling) is the FAIL condition; same order of
 * magnitude is PASS. This is a coarse magnitude check, consistent with the
 * coarseness this script's own detector honestly carries — not a precise
 * reconciliation.
 *
 * Usage: DIAG_TOKEN=... npx tsx scripts/portdwell_gate1_rollup.ts [prodBaseUrl]
 * Read-only against production; prints a JSON verdict to stdout. The
 * result is recorded in research/experiments.md + datacore/signal_ladder.json
 * by the session that runs it, not by this script.
 */
import { portsFromSites, summarizeRollupPresence, type PortDef, type RollupTrackRow } from "../server/portDwell.ts";
import datacoreSites from "../datacore/sites/strategic_sites.json";

const BASE = process.argv[2] || process.env.VOLTRADE_PROD_URL || "https://voltradeai-production.up.railway.app";
const TOKEN = process.env.DIAG_TOKEN;

// Generous box around both LA-basin ports (radius_km=5 geofences sit well
// inside it) — see server/portDwell.ts's own PortDef list via portsFromSites.
const LA_BASIN_BBOX = "33.68,33.82,-118.35,-118.15";

const JULY_DAYS: string[] = [];
for (let d = 1; d <= 31; d++) JULY_DAYS.push(`2026-07-${String(d).padStart(2, "0")}`);

async function fetchRollupDay(day: string): Promise<RollupTrackRow[]> {
  const url = `${BASE}/api/diag/archive?stream=vessels_tracks&day=${day}&bbox=${LA_BASIN_BBOX}&limit=5000&token=${TOKEN}`;
  const r = await fetch(url, { signal: AbortSignal.timeout(30000) as any });
  if (!r.ok) throw new Error(`${url} -> ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const body = await r.json();
  return (body.rows || []) as RollupTrackRow[];
}

async function fetchFreshRawBaseline(): Promise<any> {
  // A FRESH raw-pipeline reading (current rolling 168h) for the same ports,
  // pulled this same session rather than reused from the 2026-08-19 entry's
  // now-stale mid-August figure, so the comparison baseline is contemporary.
  const url = `${BASE}/api/diag/portdwell_window?hours=168&token=${TOKEN}`;
  const r = await fetch(url, { signal: AbortSignal.timeout(30000) as any });
  if (!r.ok) throw new Error(`${url} -> ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return r.json();
}

async function main() {
  if (!TOKEN) {
    console.log(JSON.stringify({ verdict: "ERROR", error: "DIAG_TOKEN not set in environment" }));
    process.exitCode = 1;
    return;
  }

  const ports: PortDef[] = portsFromSites((datacoreSites as any).sites || [])
    .filter((p) => p.id === "port_la" || p.id === "port_lb");

  const byVessel = new Map<string, RollupTrackRow[]>();
  let truncatedDays = 0;
  for (const day of JULY_DAYS) {
    const rows = await fetchRollupDay(day);
    for (const row of rows) {
      if (!byVessel.has(row.i)) byVessel.set(row.i, []);
      byVessel.get(row.i)!.push(row);
    }
    console.error(`[portdwell_gate1_rollup] ${day}: ${rows.length} in-bbox rollup rows`);
  }
  // Each vessel's rows must be day-ascending — the archive read above walks
  // JULY_DAYS in order, so push() already left them sorted; assert it
  // instead of re-sorting blind (a silent re-sort could hide an upstream
  // ordering bug the way findGasFlareCandidates' own sessions warn about).
  for (const [mmsi, rows] of byVessel) {
    for (let i = 1; i < rows.length; i++) {
      if (rows[i].d < rows[i - 1].d) throw new Error(`rows for vessel ${mmsi} are not day-ascending`);
    }
  }

  const julySummary = summarizeRollupPresence(byVessel, ports, JULY_DAYS[JULY_DAYS.length - 1]);
  const freshBaseline = await fetchFreshRawBaseline();

  console.log(JSON.stringify({
    window: { days: JULY_DAYS, count: JULY_DAYS.length, truncated_days: truncatedDays },
    unique_vessels_tracked_in_bbox: byVessel.size,
    july_rollup_summary: julySummary,
    fresh_raw_baseline_168h: freshBaseline,
    truth_source: "Port of LA news release news_081826_july_cargo (portoflosangeles.org): " +
      "960,464 TEUs July 2026 (2nd-busiest July on record), 6,083,067 TEUs YTD through July (+1.8% YoY). " +
      "Gathered by the 2026-08-19 GATE 1 session, restated here verbatim, not re-fetched this session.",
    bar: "TEU (container throughput) vs vessel-presence count are different units -- order-of-magnitude " +
      "comparison against the fresh 168h raw baseline (scaled ~4.3x week-to-month) is the falsifiable bar, " +
      "not exact reconciliation. See this script's own header for the full framing.",
  }, null, 2));
}

main().catch((e) => {
  console.error(JSON.stringify({ verdict: "ERROR", error: e?.message || String(e) }));
  process.exitCode = 1;
});
