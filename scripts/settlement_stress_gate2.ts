/**
 * settlement_stress_gate2.ts — GATE 2 (SIGNAL) readiness check for the
 * settlement-stress composite (server/settlementStress.ts): does
 * threshold-list persistence x FTD delta x short-vol percentile predict
 * forward returns? Named "gate-locked" in secFtd.ts's and
 * finraShortVolume.ts's own module headers; datacore/signal_ladder.json's
 * `sec_ftd` entry (2026-09-21) filed it as "a real, filed, but
 * NOT-YET-ATTEMPTED gate-2 hypothesis" for a future session — this is
 * that session's attempt.
 *
 * COMPILED FINDING THIS SCRIPT EXISTS TO ENCODE (EDGE DOCTRINE #3 — never
 * re-derive this by reasoning a second time): the composite's daily
 * archive (datacore/settlementstress/, one JSONL file per calendar day)
 * RE-EMITS one listing episode on every day it stays on the Reg SHO
 * threshold list, so raw row/event count vastly overstates the number of
 * INDEPENDENT observations available for a forward-return test. Reading
 * the live archive this session (2026-09-22, all 44 archived dates
 * 2026-06-25..2026-08-31 via `/api/diag/archive`) found 33 raw rows that
 * collapse to exactly 8 independent episodes (one per symbol per
 * continuous listing run) — e.g. ALKEF alone contributed 9 of the 33 raw
 * rows as its persistence_days climbed 11->26 across six weeks, one
 * ongoing episode, not nine events. A naive session counting "33 events"
 * and running a per-row t-test would silently commit exactly the
 * pseudo-replication error Reasoning Standard #4 warns about (the same
 * shape statsUtils.ts's day-clustering already fixed for occ_options_volume
 * — here the non-independence is symbol-level and multi-week, not
 * day-level).
 *
 * SECOND FINDING, equally load-bearing: of those 8 independent episodes,
 * 7 are FOREIGN ADRs/ordinary shares (CFRUY, ALKEF, FSNUY, LLKKF, BAYRY,
 * WEGZY, TGOPY — Swiss/Australian/German/Brazilian/UK names) and only 1
 * (NBND) is a plain US common stock. This is a plausible structural
 * confound, not dismissed by assumption: Reg SHO threshold-list
 * persistence for foreign ADRs is a well-documented artifact of
 * cross-border settlement mechanics (local market holidays, custodian/
 * depositary-bank delays converting local shares into ADRs) — a reason
 * for FTD/threshold persistence that has NOTHING to do with short-squeeze
 * economics (Reasoning Standard #5: "who is on the other side, and why
 * hasn't it been arbitraged" has an uncomfortable answer here — nobody;
 * it may just be clearing-system plumbing). Pooling ADR and domestic
 * episodes into one test would let that confound silently drive the
 * result in either direction. This script therefore classifies every
 * episode (isForeignListing, keyed off FINRA's own security-name
 * conventions, not invented) and REQUIRES a minimum count of DOMESTIC
 * episodes specifically before treating any correlation test as
 * meaningful — pooling foreign-listing episodes in would launder the
 * confound into the headline number instead of isolating it.
 *
 * PRE-REGISTERED READINESS BAR (stated before this session read the
 * numbers below in the file's own history — MIN_DOMESTIC_EPISODES was
 * fixed by asking "how many independent clusters do this repo's other
 * gate-2 day-clustered tests use", not backfit to whatever the count
 * turned out to be): MIN_DOMESTIC_EPISODES=20, matching the ~16-30
 * sample-cluster counts occ_volume_gate2.ts/finra_shortvol_gate2.ts use
 * as their own minimum viable n for a day-clustered t-test to have any
 * realistic power against a moderate effect size. Below that, any
 * measured correlation (positive, negative, or null) is not informative
 * and this script refuses to render a PASS/FAIL verdict — it reports
 * READY/WAITING only, the same posture scripts/ladder_readiness_check.py
 * uses for calendar-based triggers.
 *
 * EXPLORATORY, NON-GATING SECTION: when run without --dry, this script
 * additionally prints each episode's own forward return at +5/+20 trading
 * days (entry = the episode's first archived date, Yahoo adjusted close,
 * fetchYahooDaily/toSeries/fwdReturn reused verbatim from
 * occ_volume_gate2.ts — no lookahead, Reasoning Standard #7) purely as a
 * descriptive, publicly-loggable data point for future sessions to track
 * as the sample grows. This is explicitly NOT a statistical test and MUST
 * NOT be read as a gate verdict at n=8 — printed under its own banner
 * for exactly that reason.
 *
 * Session-run: `DIAG_TOKEN=... npx tsx scripts/settlement_stress_gate2.ts [--dry] [prodBaseUrl]`
 * --dry skips the archive fetch entirely and only prints the pure-function
 * unit-testable behavior would need live data to exercise otherwise.
 * Touches no runtime path; result goes in research/experiments.md +
 * datacore/signal_ladder.json.
 */
import { pathToFileURL } from "url";
import { fetchYahooDaily, toSeries, fwdReturn, meanStd } from "./occ_volume_gate2";

export interface CompositeRow {
  date: string;
  symbol: string;
  name: string;
  persistence_days: number;
  ftd_qty: number;
  ftd_delta: number;
  short_ratio: number;
  short_vol_percentile: number;
  composite_score: number;
}

export interface Episode {
  symbol: string;
  name: string;
  foreign: boolean;
  startDate: string;
  endDate: string;
  observations: number;
  entryScore: number;
  peakAbsScore: number;
}

export const MAX_EPISODE_GAP_DAYS = 10;
export const MIN_DOMESTIC_EPISODES = 20;

/** FINRA's own security-name conventions for ADRs/foreign ordinary shares
 *  (verified against this session's 7 live examples: "Sponsored ADR",
 *  "Unsponsored ADR", "American Depositary Shares", "Ordinary Shares" all
 *  appear verbatim; NBND's "NetBrands Corp. Common Stock" matches none).
 *  A data-driven heuristic off the catalog's own text, not a ticker-suffix
 *  guess (ALKEF/LLKKF end in F, not the "Y" ADR-suffix folklore). */
export function isForeignListing(name: string): boolean {
  return /\bADR\b|American Depositary|Depositary (Shares|Receipt)|Ordinary Shares/i.test(name);
}

function daysBetween(a: string, b: string): number {
  return Math.round((Date.parse(`${b}T00:00:00Z`) - Date.parse(`${a}T00:00:00Z`)) / 86_400_000);
}

/** Collapses raw daily composite rows into independent episodes: all
 *  observations of one symbol are the SAME episode as long as consecutive
 *  archived dates are no more than `maxGapDays` apart (a listing that goes
 *  quiet for longer than that is treated as having ended and, if it later
 *  reappears, as a genuinely new episode). Pure function — no I/O — so
 *  this is fully unit-testable on synthetic rows. Input order does not
 *  matter; output is sorted by episode start date. */
export function collapseEpisodes(rows: CompositeRow[], maxGapDays = MAX_EPISODE_GAP_DAYS): Episode[] {
  const bySymbol = new Map<string, CompositeRow[]>();
  for (const r of rows) {
    if (!bySymbol.has(r.symbol)) bySymbol.set(r.symbol, []);
    bySymbol.get(r.symbol)!.push(r);
  }
  const episodes: Episode[] = [];
  for (const obsAll of bySymbol.values()) {
    const obs = [...obsAll].sort((a, b) => a.date.localeCompare(b.date));
    let cur: CompositeRow[] = [];
    for (const row of obs) {
      if (cur.length && daysBetween(cur[cur.length - 1].date, row.date) > maxGapDays) {
        episodes.push(finalizeEpisode(cur));
        cur = [];
      }
      cur.push(row);
    }
    if (cur.length) episodes.push(finalizeEpisode(cur));
  }
  episodes.sort((a, b) => a.startDate.localeCompare(b.startDate));
  return episodes;
}

function finalizeEpisode(obs: CompositeRow[]): Episode {
  return {
    symbol: obs[0].symbol,
    name: obs[0].name,
    foreign: isForeignListing(obs[0].name),
    startDate: obs[0].date,
    endDate: obs[obs.length - 1].date,
    observations: obs.length,
    entryScore: obs[0].composite_score,
    peakAbsScore: Math.max(...obs.map((o) => Math.abs(o.composite_score))),
  };
}

export interface Readiness {
  ready: boolean;
  domestic: number;
  foreign: number;
  detail: string;
}

/** GATE 2 readiness verdict — READY/WAITING only, never PASS/FAIL: this
 *  script measures whether a meaningful test is even possible yet, it
 *  does not run one. Gated on DOMESTIC episodes only (see module header's
 *  ADR-confound finding) — foreign-listing episodes are counted and
 *  reported but deliberately excluded from the readiness count so the
 *  confound cannot be diluted away by pooling. */
export function assessReadiness(episodes: Episode[]): Readiness {
  const domestic = episodes.filter((e) => !e.foreign).length;
  const foreign = episodes.filter((e) => e.foreign).length;
  const ready = domestic >= MIN_DOMESTIC_EPISODES;
  return {
    ready,
    domestic,
    foreign,
    detail: `${domestic} domestic / ${foreign} foreign-listing independent episodes `
      + `(need >=${MIN_DOMESTIC_EPISODES} domestic before a gate-2 correlation test is `
      + `meaningful — foreign-listing episodes are excluded from this count, not just `
      + `down-weighted, per the ADR-settlement-mechanics confound in this file's header)`,
  };
}

// ── Live fetch (production diag probe — no direct volume access from a
//    fresh session sandbox; mirrors scripts/portdwell_weekly_snapshot.ts's
//    DIAG_TOKEN pattern) ───────────────────────────────────────────────
const DRY = process.argv.includes("--dry");
const BASE = process.argv.find((a, i) => i >= 2 && !a.startsWith("--")) || process.env.VOLTRADE_PROD_URL || "https://voltradeai.com";
const TOKEN = process.env.DIAG_TOKEN;

async function fetchArchiveStats(): Promise<{ oldest: string; newest: string } | null> {
  const r = await fetch(`${BASE}/api/data/archive/stats`, { signal: AbortSignal.timeout(20000) as unknown as AbortSignal });
  if (!r.ok) return null;
  const body = await r.json();
  const s = body?.kinds?.settlementstress;
  if (!s?.oldest || !s?.newest) return null;
  // Both fields are `${date}.jsonl[.gz]` for this stream (no non-date marker
  // files, unlike e.g. finrashortvol's "backfill_done.json").
  const clean = (f: string) => f.replace(/\.jsonl(\.gz)?$/, "");
  return { oldest: clean(s.oldest), newest: clean(s.newest) };
}

function* dateRange(start: string, end: string): Generator<string> {
  let d = new Date(`${start}T00:00:00Z`);
  const endMs = new Date(`${end}T00:00:00Z`).getTime();
  while (d.getTime() <= endMs) {
    yield d.toISOString().slice(0, 10);
    d = new Date(d.getTime() + 86_400_000);
  }
}

async function fetchDay(day: string): Promise<CompositeRow[]> {
  const url = `${BASE}/api/diag/archive?stream=settlementstress&day=${day}&token=${TOKEN}`;
  const r = await fetch(url, { signal: AbortSignal.timeout(20000) as unknown as AbortSignal });
  if (!r.ok) return [];
  const body = await r.json();
  return Array.isArray(body?.rows) ? (body.rows as CompositeRow[]) : [];
}

async function main() {
  if (DRY) {
    console.log("--dry: skipping live archive fetch. Pure functions (collapseEpisodes/isForeignListing/assessReadiness) are exercised by settlement_stress_gate2.test.ts.");
    return;
  }
  if (!TOKEN) {
    console.log(JSON.stringify({ verdict: "ERROR", error: "DIAG_TOKEN not set in environment" }));
    process.exitCode = 1;
    return;
  }

  const stats = await fetchArchiveStats();
  if (!stats) {
    console.log(JSON.stringify({ verdict: "ERROR", error: "settlementstress archive not found via /api/data/archive/stats" }));
    process.exitCode = 1;
    return;
  }
  console.log(`settlementstress archive: ${stats.oldest} .. ${stats.newest}`);

  const rows: CompositeRow[] = [];
  let daysFetched = 0;
  for (const day of dateRange(stats.oldest, stats.newest)) {
    const dayRows = await fetchDay(day);
    if (dayRows.length) rows.push(...dayRows);
    daysFetched++;
    await new Promise((res) => setTimeout(res, 150)); // polite spacing vs the platform's own proxy
  }
  console.log(`fetched ${daysFetched} calendar days, ${rows.length} raw composite rows`);

  const episodes = collapseEpisodes(rows);
  console.log(`\ncollapsed to ${episodes.length} independent episodes:`);
  for (const e of episodes) {
    console.log(`  ${e.symbol}${e.foreign ? " [foreign]" : " [domestic]"}: ${e.startDate}..${e.endDate} `
      + `(${e.observations} obs, entryScore=${e.entryScore}, peakAbsScore=${e.peakAbsScore.toFixed(2)})`);
  }

  const readiness = assessReadiness(episodes);
  console.log(`\n=== GATE 2 READINESS (not a PASS/FAIL verdict) ===`);
  console.log(readiness.ready ? "READY" : "WAITING", "—", readiness.detail);

  console.log(`\n=== EXPLORATORY forward returns (NON-GATING — n=${episodes.length}, informational only) ===`);
  const firstMs = new Date(`${episodes[0]?.startDate ?? stats.oldest}T00:00:00Z`).getTime() / 1000;
  const lastMs = new Date(`${stats.newest}T00:00:00Z`).getTime() / 1000;
  const startSec = firstMs - 5 * 86400;
  const endSec = lastMs + 45 * 86400;
  const spySeries = toSeries(await fetchYahooDaily("SPY", startSec, endSec).catch(() => new Map()));
  const ret5: number[] = [];
  const ret20: number[] = [];
  for (const e of episodes) {
    let series;
    try {
      series = toSeries(await fetchYahooDaily(e.symbol, startSec, endSec));
    } catch {
      console.log(`  ${e.symbol}: price fetch failed, skipped`);
      continue;
    }
    const r5 = fwdReturn(series, e.startDate, 5);
    const r20 = fwdReturn(series, e.startDate, 20);
    const spy5 = fwdReturn(spySeries, e.startDate, 5);
    const spy20 = fwdReturn(spySeries, e.startDate, 20);
    if (r5 != null) ret5.push(r5);
    if (r20 != null) ret20.push(r20);
    console.log(`  ${e.symbol} entry=${e.startDate}: +5d=${fmtPct(r5)} (SPY ${fmtPct(spy5)})  +20d=${fmtPct(r20)} (SPY ${fmtPct(spy20)})`);
    await new Promise((res) => setTimeout(res, 250));
  }
  if (ret5.length) console.log(`\n  mean +5d across ${ret5.length} priced episodes: ${(meanStd(ret5).mean * 100).toFixed(2)}%`);
  if (ret20.length) console.log(`  mean +20d across ${ret20.length} priced episodes: ${(meanStd(ret20).mean * 100).toFixed(2)}%`);
  console.log(`\n  NOT A GATE VERDICT — n=${episodes.length} is far below MIN_DOMESTIC_EPISODES=${MIN_DOMESTIC_EPISODES}; printed for tracking only.`);
}

function fmtPct(x: number | null): string {
  return x == null ? "n/a" : `${(x * 100).toFixed(2)}%`;
}

// Entrypoint guard (ESM has no require.main) — same fix class as
// finra_shortvol_gate2.ts's own header note: without this, importing
// collapseEpisodes/isForeignListing from a test file would also re-trigger
// this file's live network fetch as an unwanted import side effect.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
