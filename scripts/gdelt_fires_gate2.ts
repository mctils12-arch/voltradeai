/**
 * gdelt_fires_gate2.ts — GATE 2 (SIGNAL layer) for the gdelt_facility_events
 * root's own stated hypothesis (`server/gdeltEvents.ts` docstring, HYPOTHESIS
 * comment, "gate 2, not attempted" since the stream shipped 2026-07-05):
 * do geo-tagged unrest/strike bursts near a tracked facility serve as a real
 * ALERT TRIGGER — do they precede a detection from one of our OWN sensors
 * (this run: NASA FIRMS active-fire) at that same facility more than chance?
 * No trading is involved (gate 2, not gate 3/4/5).
 *
 * ROOT CONTEXT: `research/experiments.md`'s 2026-08-20 GDELT-facility-events
 * session ("closing the LAST shipped-data-no-UI gap") named this exact
 * retrospective as its own NEXT item (3): "The gate-2 hypothesis this feed
 * exists for (unrest burst -> own-sensor confirmation) has never been
 * attempted; the archive has been recording since 2026-07-05, so a
 * retrospective burst-vs-sensor study is now possible." This script is that
 * study.
 *
 * INFRA PRECONDITION (shipped same session, own logical change): the
 * `/api/diag/archive` probe's `fires` stream is a GLOBAL VIIRS feed
 * (`FIRMS_AREA = "-180,-90,180,90"` in nasaFirms.ts) — every real day
 * checked live returns `truncated:true` at the probe's 5,000-row cap, so an
 * unfiltered read would hand back an arbitrary first-N-in-file-order slice
 * of WORLDWIDE fire detections, not a representative sample near our 16 US
 * facilities. `readArchiveDay` (datacoreArchive.ts) and the `archive` probe
 * (server/bot.ts) gained an optional inline `bbox` rowFilter this same
 * session for exactly this reason; this script is its first real caller.
 * GDELT itself needs no such filter — `parseGdeltExport` already discards
 * any event that doesn't match a facility bbox at ARCHIVE-WRITE time
 * (`gdeltEvents.ts`'s `nearestFacility`), so every archived gdelt row is
 * already facility-adjacent and daily counts (128-556 in spot checks) never
 * approach the cap.
 *
 * METHODOLOGY, PRE-REGISTERED BEFORE THIS SCRIPT WAS RUN AGAINST REAL DATA
 * (Reasoning Standard #10 — state the prior, then update):
 *
 *   - Facilities: the 16 sites in `datacore/sites/strategic_sites.json`
 *     (same catalogue gdeltEvents.ts's own `nearestFacility()` and
 *     firesFacilities.ts's `firesNearFacilities()` already use) — 3
 *     tank farms (Cushing OK), 4 steel mills, 9 ports.
 *   - Radius: 25 km, REUSING the existing live `/api/data/fires-near-
 *     facilities` cross-tie's own default radius (`server/routes.ts`) —
 *     not a new tuned parameter chosen after seeing this script's results.
 *     Distance is recomputed exactly via `haversineKm` (reused from
 *     `server/firesFacilities.ts`, not reimplemented — EDGE DOCTRINE #3)
 *     against each row's own lat/lon; the archived GDELT `site` field is
 *     NOT trusted directly, since its ingest-time match uses a wider
 *     +-0.5-degree box (up to ~55 km at these latitudes) than this test's
 *     25 km bar.
 *   - "Unrest day": a facility-day with >=1 real (25 km-verified) GDELT
 *     event. No count/severity threshold beyond "at least one" — picking a
 *     burst-count threshold AFTER looking at the data would be exactly the
 *     kind of post-hoc tuning Reasoning Standard #4 warns against; "any
 *     qualifying event" is the simplest, least-tunable definition and is
 *     fixed here before the fetch runs.
 *   - "Hit": a FIRMS detection within the same 25 km radius on the unrest
 *     day itself through GATE2_WINDOW_DAYS (3, pre-registered) days after.
 *   - Date range: GDELT archive starts 2026-07-05; FIRMS archive starts
 *     2026-07-04. Every evaluated day must have a FULL, uncensored
 *     GATE2_WINDOW_DAYS-ahead FIRMS window available, so the usable range is
 *     [2026-07-05, latest_fires_day - GATE2_WINDOW_DAYS] for BOTH the unrest-day
 *     group and the control group — the same censoring rule applied to
 *     both, so neither is biased toward "no hit found" by missing data.
 *     `main()` defaults to `--end 2026-08-17` (FIRMS confirmed live through
 *     2026-08-20 at authorship time; 2026-08-20 - 3 = 2026-08-17) but
 *     accepts an override once more days have accumulated.
 *   - Test: pooled one-tailed exact binomial test (`statsUtils.
 *     binomialUpperTailP`, reused — not reimplemented) — is the unrest-day
 *     hit rate (k of n) elevated above the CONTROL rate (the same
 *     facilities' own non-unrest-day hit rate, same window, same range) by
 *     more than chance? Bar: p < 0.05, one-tailed, ONE pooled test (not the
 *     stricter 0.01 the gnss_integrity script used for a multi-band scan —
 *     this is a single pre-registered comparison, not several).
 *   - insufficient_n floor: n (unrest-days) < 5 reports `insufficient_n:
 *     true` rather than fabricating a p-value on a handful of days — same
 *     floor every other gate-2 script in this repo uses.
 *   - PER-CATEGORY breakdown (tank_farm / steel_mill / port) ALSO reported,
 *     stratified the same way, pre-registered alongside the pooled result
 *     (not added after seeing which category "worked") — because a steel
 *     mill's ROUTINE industrial heat (blast furnaces, flares) is a
 *     plausible FIRMS false-positive source unrelated to any unrest event,
 *     which could inflate BOTH that category's control rate and its
 *     unrest-day rate together; the pooled number alone could hide that
 *     (Simpson's-paradox risk) if unrest days cluster unevenly by category.
 *
 * PRIOR, STATED BEFORE RUNNING: expect NO significant pooled elevation.
 * `gdeltEvents.ts`'s own docstring already says CAMEO captures strikes/
 * protests/unrest well but NOT clean industrial accidents, and FIRMS is a
 * THERMAL sensor — the two measure different physical phenomena except in
 * the minority of violent/riot events that produce a fire (arson,
 * sabotage). A clean negative result is the EXPECTED, useful outcome here:
 * it tells datacore whether "GDELT unrest -> go check FIRMS" is a real
 * verification prompt or not. A steel-mill-only "pass" would be LESS
 * credible than a port-only one, not more — it is the category most likely
 * to show a false elevation from a pre-existing high heat baseline
 * unrelated to unrest, exactly the failure mode the per-category
 * breakdown above exists to catch.
 *
 * HONEST LIMITATIONS:
 *   - Small facility count (16) and short archive (~6 weeks at authorship)
 *     — a null result here is evidence at this sample size, not proof the
 *     hypothesis is false at scale; a future re-run as the archive deepens
 *     should be treated as the real confirmatory test.
 *   - VIIRS 375m NRT detects large/hot thermal anomalies — it will miss a
 *     small fire, an indoor incident, or a fire already out by its ~3h-12h
 *     latency revisit. A miss here is not proof no fire occurred.
 *   - This is gate 2 (statistical discrimination), not gate 1 (external
 *     ground truth for either feed individually) — both GDELT and FIRMS
 *     already ship as RAW (gate-0) overlays per the RAW-vs-SIGNAL rule;
 *     this script tests whether JOINING them produces a real cross-tie
 *     signal, which is a separate question from either feed's own
 *     individual accuracy.
 *
 * STATUS: NOT YET RUN AGAINST LIVE DATA (honest constraint, not a
 * shortcut — Reasoning Standard #10 applies to code, not just numbers).
 * A live smoke test against production THIS session, over a 3-day window,
 * proved the `fires` fetch's new `bbox` param has no effect yet — the
 * first row back from a ~0.35deg box around a Cushing OK facility was a
 * detection in Libya. That is expected, not a bug in this script: the
 * `bbox` param and the `readArchiveDay` rowFilter it depends on are THIS
 * SAME PR's own change, and production has not deployed it yet. Running
 * this script against the OLD deployed endpoint would silently truncate
 * every facility-day fire query to an arbitrary global slice and produce
 * a fabricated-looking but meaningless verdict — worse than not running
 * it at all. The pure functions above are fully unit-tested on synthetic
 * data (`gdelt_fires_gate2.test.ts`) and the fetch/join plumbing is ready;
 * a future session should confirm `server_version` bumped past this PR
 * (via `/api/data/layers` or `/api/health`) and then run:
 *   DIAG_TOKEN=... npx tsx scripts/gdelt_fires_gate2.ts
 * and log the real pooled/by-category verdict in experiments.md and (if
 * gate 2 passes) `datacore/signal_ladder.json`.
 */
import { pathToFileURL } from "url";
import { binomialUpperTailP } from "./statsUtils";
import { haversineKm } from "../server/firesFacilities";
import strategicSites from "../datacore/sites/strategic_sites.json";

export const GATE2_RADIUS_KM = 25;
export const GATE2_WINDOW_DAYS = 3;
export const GATE2_MIN_UNREST_DAYS = 5;
export const GATE2_P_BAR = 0.05;

export interface Facility { id: string; name: string; lat: number; lon: number; category?: string }

interface RawStrategicSite { id: string; name: string; lat: number; lon: number; category?: string }

export function loadFacilities(): Facility[] {
  const data = strategicSites as { sites?: RawStrategicSite[] };
  const sites = data.sites || [];
  return sites
    .filter((s) => Number.isFinite(s?.lat) && Number.isFinite(s?.lon))
    .map((s) => ({ id: s.id, name: s.name, lat: s.lat, lon: s.lon, category: s.category || "unknown" }));
}

/** GDELT's own archived "day" column is YYYYMMDD (parseGdeltExport, COL.day) — normalize to YYYY-MM-DD. */
export function gdeltDayToIso(day: string): string | null {
  const m = /^(\d{4})(\d{2})(\d{2})$/.exec(String(day || ""));
  return m ? `${m[1]}-${m[2]}-${m[3]}` : null;
}

export function addDaysIso(day: string, n: number): string {
  const d = new Date(`${day}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}

/** Inclusive [start, end] list of YYYY-MM-DD days. */
export function dayRange(start: string, end: string): string[] {
  const out: string[] = [];
  for (let d = start; d <= end; d = addDaysIso(d, 1)) out.push(d);
  return out;
}

/**
 * For each facility, the set of days (YYYY-MM-DD) with >=1 row within
 * radiusKm. `rows` must already carry a normalized `day` field — callers
 * normalize GDELT's YYYYMMDD and FIRMS' native YYYY-MM-DD before calling.
 * Pure; recomputes exact distance rather than trusting any upstream bbox.
 */
export function nearbyDaysByFacility(
  rows: Array<{ lat: number; lon: number; day: string }>,
  facilities: Facility[],
  radiusKm = GATE2_RADIUS_KM,
): Map<string, Set<string>> {
  const out = new Map<string, Set<string>>();
  for (const f of facilities) out.set(f.id, new Set());
  for (const r of rows) {
    if (!Number.isFinite(r.lat) || !Number.isFinite(r.lon) || !r.day) continue;
    for (const f of facilities) {
      if (haversineKm(f.lat, f.lon, r.lat, r.lon) <= radiusKm) out.get(f.id)!.add(r.day);
    }
  }
  return out;
}

/** True if fireDays contains any day in [day, day + windowDays] inclusive. */
export function hitWithinWindow(fireDays: Set<string>, day: string, windowDays = GATE2_WINDOW_DAYS): boolean {
  for (let i = 0; i <= windowDays; i++) {
    if (fireDays.has(addDaysIso(day, i))) return true;
  }
  return false;
}

export interface DayRow { facility: string; category: string; day: string; unrest: boolean; hit: boolean }

/** evalDays must already be censored to days with a full forward window observable in the fire archive. */
export function buildDayRows(
  facilities: Facility[],
  unrestDaysByFacility: Map<string, Set<string>>,
  fireDaysByFacility: Map<string, Set<string>>,
  evalDays: string[],
  windowDays = GATE2_WINDOW_DAYS,
): DayRow[] {
  const rows: DayRow[] = [];
  for (const f of facilities) {
    const unrestDays = unrestDaysByFacility.get(f.id) || new Set<string>();
    const fireDays = fireDaysByFacility.get(f.id) || new Set<string>();
    for (const day of evalDays) {
      rows.push({
        facility: f.id, category: f.category || "unknown", day,
        unrest: unrestDays.has(day),
        hit: hitWithinWindow(fireDays, day, windowDays),
      });
    }
  }
  return rows;
}

export interface Gate2Verdict {
  n: number; k: number; control_n: number; control_k: number;
  control_rate: number; expected_under_null: number; p_value: number;
  elevated: boolean; insufficient_n: boolean;
}

export function verdictFromRows(rows: DayRow[], minN = GATE2_MIN_UNREST_DAYS, pBar = GATE2_P_BAR): Gate2Verdict {
  const unrestRows = rows.filter((r) => r.unrest);
  const controlRows = rows.filter((r) => !r.unrest);
  const n = unrestRows.length;
  const k = unrestRows.filter((r) => r.hit).length;
  const control_n = controlRows.length;
  const control_k = controlRows.filter((r) => r.hit).length;
  const control_rate = control_n > 0 ? control_k / control_n : NaN;
  const insufficient_n = n < minN;
  const p_value = insufficient_n || !Number.isFinite(control_rate)
    ? NaN
    : binomialUpperTailP(k, n, control_rate);
  const elevated = !insufficient_n && Number.isFinite(p_value) && p_value < pBar && (k / n) > control_rate;
  return {
    n, k, control_n, control_k, control_rate,
    expected_under_null: Number.isFinite(control_rate) ? control_rate * n : NaN,
    p_value, elevated, insufficient_n,
  };
}

export function computeGate2(rows: DayRow[], minN = GATE2_MIN_UNREST_DAYS, pBar = GATE2_P_BAR): {
  pooled: Gate2Verdict; by_category: Record<string, Gate2Verdict>;
} {
  const pooled = verdictFromRows(rows, minN, pBar);
  const categories = Array.from(new Set(rows.map((r) => r.category))).sort();
  const by_category: Record<string, Gate2Verdict> = {};
  for (const cat of categories) {
    by_category[cat] = verdictFromRows(rows.filter((r) => r.category === cat), minN, pBar);
  }
  return { pooled, by_category };
}

// ── Fetch (diag archive probe) ──────────────────────────────────────────

type ArchiveRow = Record<string, unknown>;

async function fetchArchiveDay(stream: string, day: string, bbox?: string): Promise<{ rows: ArchiveRow[]; truncated: boolean }> {
  const token = process.env.DIAG_TOKEN;
  if (!token) throw new Error("DIAG_TOKEN env var required (same token used by every other /api/diag/* probe)");
  const base = process.env.VOLTRADE_BASE_URL ?? "https://voltradeai.com";
  const bboxQ = bbox ? `&bbox=${encodeURIComponent(bbox)}` : "";
  const url = `${base}/api/diag/archive?stream=${stream}&day=${day}&limit=5000${bboxQ}&token=${token}`;
  const r = await fetch(url, { signal: AbortSignal.timeout(30_000) as any });
  if (!r.ok) throw new Error(`archive diag ${stream} ${day}: HTTP ${r.status}`);
  const j = (await r.json()) as { rows?: ArchiveRow[]; truncated?: boolean };
  return { rows: j.rows || [], truncated: Boolean(j.truncated) };
}

async function mapWithConcurrency<T, R>(items: T[], concurrency: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i]);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
  return out;
}

/**
 * A single CONUS-wide bbox for the `fires` fetch was tried first and
 * REJECTED live: VIIRS detections across the whole continental US during
 * an active fire season exceed the diag probe's 5,000-row cap on EVERY
 * day checked (confirmed: 6/6 days, even after the CONUS bbox filter
 * shipped this session) — the same truncation-bias problem one bbox
 * level up. Fetching per-FACILITY instead (a ~0.35deg box, comfortably
 * containing the full 25km radius circle at every site's latitude) keeps
 * each query's matching-row count small (a handful of detections near
 * one point, not thousands across a continent), which is what actually
 * fixes the truncation risk rather than just moving it.
 */
function facilityBboxStr(f: Facility, padDeg = 0.35): string {
  const lonPad = padDeg / Math.max(0.15, Math.cos((f.lat * Math.PI) / 180));
  return `${(f.lat - padDeg).toFixed(4)},${(f.lat + padDeg).toFixed(4)},${(f.lon - lonPad).toFixed(4)},${(f.lon + lonPad).toFixed(4)}`;
}

async function main() {
  const args = process.argv.slice(2);
  const arg = (flag: string, dflt: string) => {
    const i = args.indexOf(flag);
    return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
  };
  const gdeltStart = arg("--start", "2026-07-05"); // GDELT archive start (server/gdeltEvents.ts docstring)
  const evalEnd = arg("--end", "2026-08-17"); // last day with a full GATE2_WINDOW_DAYS-ahead FIRMS window, at authorship
  const firesEnd = addDaysIso(evalEnd, GATE2_WINDOW_DAYS);
  const concurrency = Number(arg("--concurrency", "10"));

  const facilities = loadFacilities();
  const gdeltDays = dayRange(gdeltStart, evalEnd);
  const fireDays = dayRange(gdeltStart, firesEnd);
  console.log(`gdelt_fires_gate2: ${facilities.length} facilities, gdelt days ${gdeltDays.length} [${gdeltStart}..${evalEnd}], fire days ${fireDays.length} [${gdeltStart}..${firesEnd}] x ${facilities.length} facility-bbox calls each, radius=${GATE2_RADIUS_KM}km window=${GATE2_WINDOW_DAYS}d`);

  const gdeltResults = await mapWithConcurrency(gdeltDays, concurrency, (d) => fetchArchiveDay("gdelt", d));
  const gdeltTruncatedDays = gdeltDays.filter((_, i) => gdeltResults[i].truncated);
  const gdeltRows = gdeltResults.flatMap((r) =>
    r.rows
      .map((row) => ({ lat: Number(row.lat), lon: Number(row.lon), day: gdeltDayToIso(String(row.day ?? "")) }))
      .filter((row): row is { lat: number; lon: number; day: string } => row.day !== null),
  );

  const fireTasks: Array<{ day: string; facility: Facility }> = [];
  for (const day of fireDays) for (const f of facilities) fireTasks.push({ day, facility: f });
  const fireTaskResults = await mapWithConcurrency(fireTasks, concurrency, (t) => fetchArchiveDay("fires", t.day, facilityBboxStr(t.facility)));
  const fireRowsByFacility = new Map<string, Array<{ lat: number; lon: number; day: string }>>();
  for (const f of facilities) fireRowsByFacility.set(f.id, []);
  let fireTaskTruncatedCount = 0;
  fireTaskResults.forEach((r, i) => {
    if (r.truncated) fireTaskTruncatedCount++;
    const bucket = fireRowsByFacility.get(fireTasks[i].facility.id)!;
    for (const row of r.rows) bucket.push({ lat: Number(row.lat), lon: Number(row.lon), day: String(row.acq_date ?? "") });
  });
  const fireRowCount = Array.from(fireRowsByFacility.values()).reduce((s, a) => s + a.length, 0);

  console.log(`fetched gdelt rows=${gdeltRows.length} (truncated days: ${JSON.stringify(gdeltTruncatedDays)}), fires rows=${fireRowCount} across ${fireTasks.length} facility-day calls (${fireTaskTruncatedCount} truncated)`);
  if (gdeltTruncatedDays.length) console.warn("WARNING: gdelt hit the 5000-row cap on some days — counts below may undercount unrest days");
  if (fireTaskTruncatedCount) console.warn(`WARNING: ${fireTaskTruncatedCount} facility-day fire queries hit the 5000-row cap even at ~0.35deg — counts below may undercount fire hits at that specific facility/day`);

  const unrestByFacility = nearbyDaysByFacility(gdeltRows, facilities, GATE2_RADIUS_KM);
  const fireByFacility = new Map<string, Set<string>>();
  for (const f of facilities) {
    const m = nearbyDaysByFacility(fireRowsByFacility.get(f.id) || [], [f], GATE2_RADIUS_KM);
    fireByFacility.set(f.id, m.get(f.id) || new Set());
  }
  const rows = buildDayRows(facilities, unrestByFacility, fireByFacility, gdeltDays, GATE2_WINDOW_DAYS);
  const { pooled, by_category } = computeGate2(rows);

  console.log("\nPOOLED:", JSON.stringify(pooled, null, 2));
  console.log("\nBY CATEGORY:");
  for (const [cat, v] of Object.entries(by_category)) {
    console.log(`  ${cat}:`, JSON.stringify(v));
  }
  console.log("\nUnrest days per facility:");
  for (const f of facilities) {
    const days = Array.from(unrestByFacility.get(f.id) || []).sort();
    if (days.length) console.log(`  ${f.id} (${f.category}): ${days.length} day(s) — ${days.join(", ")}`);
  }
  const verdict = pooled.insufficient_n
    ? "INSUFFICIENT_N"
    : pooled.elevated ? "GATE_2_PASS (pooled elevation)" : "GATE_2_FAIL (no significant elevation)";
  console.log(`\nGATE-2 VERDICT (bar: p<${GATE2_P_BAR}, one-tailed, pooled unrest-day hit rate vs. same-facility non-unrest control rate): ${verdict}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
