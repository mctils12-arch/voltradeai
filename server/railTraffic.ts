/**
 * railTraffic.ts — STB EP724 weekly rail carload archive, RAW /data view
 * (API only this PR — no /data client page yet, same incremental
 * sequencing as un_comtrade/plant-operations/eu-macro: archive+API
 * first, a dedicated client page a documented follow-up).
 *
 * Source: datacore/rail/ep724_carloads.json (scripts/stb_rail.py,
 * session-run ~weekly — STB's own EP724 consolidated workbook, keyless,
 * never a live Railway poll, same "seeded pattern" as jodiOil.ts/
 * un_comtrade). 7 Class I railroads (BNSF/CN/CP-CPKC/CSX/KCS/NS/UP),
 * weekly carloads across 22 commodity categories, 2017-03 onward.
 *
 * LADDER STATUS (RAW OVERLAYS vs SIGNALS, CLAUDE.md): GATE 1 (DATA)
 * PASSED 2026-09-27 — our 7-railroad system sum reconciles against TWO
 * independent FRED/BTS-AAR national series (RAILFRTCARLOADS for the 20
 * non-intermodal commodities, RAILFRTINTERMODAL for Containers+Trailers)
 * within a stable multiplicative offset (mean ratio 1.58/1.26, cv
 * 0.10/0.11 across 111 overlapping months each — see
 * scripts/rail_traffic_gate1.py and datacore/signal_ladder.json's
 * rail_ep724_carload_traffic entry for the pre-registered bar, the exact
 * numbers, and the interchange-double-count explanation for why our own
 * multi-railroad sum runs above, not below, the once-counted national
 * total). GATE 2 (SIGNAL) NOT attempted — this view is RAW, self-reported
 * carload counts only, no predictive claim; the manifest's own filed
 * hypothesis ("carload deltas by commodity lead rail earnings + industrial
 * regime") stays gate-locked.
 */
import railArchive from "../datacore/rail/ep724_carloads.json";

export const RAIL_TRAFFIC_GATE_NOTE =
  "GATE 1 (data) PASSED 2026-09-27 — our 7-Class-I-railroad system sum " +
  "reconciles against FRED/BTS-AAR's independent national carload and " +
  "intermodal series within a stable offset (see scripts/rail_traffic_gate1.py). " +
  "GATE 2 (signal) not attempted — shown here as RAW, self-reported weekly " +
  "carload counts only, no predictive claim.";

const CARLOAD_MEASURE = "Weekly Carloads By 22 Commodity Categories";
const INTERMODAL_VARIABLES = new Set(["Containers", "Trailers"]);

interface RailArchiveFile {
  built: string;
  source: string;
  attribution: string;
  selection: string;
  n_weeks: number;
  n_series: number;
  weeks: string[];
  series: Record<string, (number | null)[]>;
}

export interface RailCommodityRow {
  variable: string;
  latestWeekCarloads: number;
  priorWeekCarloads: number | null;
  weekOverWeekDeltaPct: number | null;
  isIntermodal: boolean;
}

export interface RailRailroadRow {
  railroad: string;
  latestWeekCarloads: number;
}

export interface RailTrafficView {
  kind: "raw";
  predictive: false;
  source: string;
  attribution: string;
  license: string;
  archiveBuiltAt: string;
  latestWeek: string;
  priorWeek: string | null;
  weeksArchived: number;
  railroadsReporting: number;
  systemLatestWeekCarloads: number;
  systemIntermodalLatestWeekCarloads: number;
  systemNonIntermodalLatestWeekCarloads: number;
  commodities: RailCommodityRow[];
  railroads: RailRailroadRow[];
  note: string;
}

function parseKey(key: string): { railroad: string; measure: string; variable: string } {
  const [railroad, measure, variable] = key.split("|");
  return { railroad, measure, variable };
}

/** Builds a system-wide (all 7 railroads summed) snapshot of the most
 *  recently archived week, split into per-commodity and per-railroad
 *  rows. A commodity/railroad not reported for the latest week is
 *  skipped entirely, never zero-filled (same "skip, never zero"
 *  convention unComtradeView/jodiOilStocksView already use for a
 *  government source that reports on its own irregular schedule). */
export function railTrafficView(
  doc: RailArchiveFile = railArchive as unknown as RailArchiveFile,
): RailTrafficView {
  const weeks = doc.weeks;
  const latestIdx = weeks.length - 1;
  const priorIdx = latestIdx - 1;
  const latestWeek = weeks[latestIdx];
  const priorWeek = priorIdx >= 0 ? weeks[priorIdx] : null;

  const byVariable = new Map<string, { latest: number; latestReported: boolean; prior: number; priorReported: boolean }>();
  const byRailroad = new Map<string, number>();
  const reportingRailroads = new Set<string>();

  for (const [key, vals] of Object.entries(doc.series)) {
    const { railroad, measure, variable } = parseKey(key);
    if (measure !== CARLOAD_MEASURE) continue;

    const latestVal = vals[latestIdx];
    if (latestVal != null) {
      reportingRailroads.add(railroad);
      byRailroad.set(railroad, (byRailroad.get(railroad) ?? 0) + latestVal);
      const cur = byVariable.get(variable) ?? { latest: 0, latestReported: false, prior: 0, priorReported: false };
      cur.latest += latestVal;
      cur.latestReported = true;
      byVariable.set(variable, cur);
    }

    const priorVal = priorIdx >= 0 ? vals[priorIdx] : null;
    if (priorVal != null) {
      const cur = byVariable.get(variable) ?? { latest: 0, latestReported: false, prior: 0, priorReported: false };
      cur.prior += priorVal;
      cur.priorReported = true;
      byVariable.set(variable, cur);
    }
  }

  const commodities: RailCommodityRow[] = Array.from(byVariable.entries())
    .filter(([, v]) => v.latestReported)
    .map(([variable, v]) => ({
      variable,
      latestWeekCarloads: Math.round(v.latest),
      priorWeekCarloads: v.priorReported ? Math.round(v.prior) : null,
      weekOverWeekDeltaPct: v.priorReported && v.prior !== 0
        ? Math.round(((v.latest - v.prior) / v.prior) * 1000) / 10
        : null,
      isIntermodal: INTERMODAL_VARIABLES.has(variable),
    }))
    .sort((a, b) => b.latestWeekCarloads - a.latestWeekCarloads);

  const systemLatestWeekCarloads = commodities.reduce((s, c) => s + c.latestWeekCarloads, 0);
  const systemIntermodalLatestWeekCarloads = commodities
    .filter((c) => c.isIntermodal)
    .reduce((s, c) => s + c.latestWeekCarloads, 0);

  const railroads: RailRailroadRow[] = Array.from(byRailroad.entries())
    .map(([railroad, latestWeekCarloads]) => ({ railroad, latestWeekCarloads: Math.round(latestWeekCarloads) }))
    .sort((a, b) => b.latestWeekCarloads - a.latestWeekCarloads);

  return {
    kind: "raw",
    predictive: false,
    source: doc.source,
    attribution: doc.attribution,
    license: "US government work (Surface Transportation Board), public domain — attribution requested",
    archiveBuiltAt: doc.built,
    latestWeek,
    priorWeek,
    weeksArchived: weeks.length,
    railroadsReporting: reportingRailroads.size,
    systemLatestWeekCarloads: Math.round(systemLatestWeekCarloads),
    systemIntermodalLatestWeekCarloads: Math.round(systemIntermodalLatestWeekCarloads),
    systemNonIntermodalLatestWeekCarloads: Math.round(systemLatestWeekCarloads - systemIntermodalLatestWeekCarloads),
    commodities,
    railroads,
    note: RAIL_TRAFFIC_GATE_NOTE,
  };
}
