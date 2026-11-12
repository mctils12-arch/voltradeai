/**
 * githubGate2.ts — the weekly in-process job that runs the pre-registered
 * GATE 2 (SIGNAL) test of the github_org_engineering_momentum root, and the
 * read-only status behind the `github_gate2` diag probe. Research plumbing:
 * it schedules, measures and reports; nothing it produces is read by any
 * trading path or surfaced on /data.
 *
 * WHAT IT TESTS (unchanged from scripts/github_activity_gate2.ts, whose
 * header is the pre-registration of record — every rule below is that
 * script's, moved here so the production server can run it; the script now
 * imports these functions and prints the same evaluation):
 *   - weekly total per org = mergedPRs + commits (uniqueActorsSample excluded);
 *   - signal = week-over-week PERCENT change in that total, per ticker against
 *     its OWN prior archived week (never a raw-count delta);
 *   - cross-sectional TOP / MID / BOTTOM terciles per week; a week with fewer
 *     than MIN_PANEL_FOR_WEEK valid deltas is dropped, never bucketed unevenly;
 *   - entry = the latest trading day on or before the week's weekEnd; forward
 *     return = close-to-close +5 trading days (gating) and +20 (informational);
 *   - cluster = the WEEK; gating test = one-sample cluster-mean t-test on each
 *     week's (TOP mean − BOTTOM mean) +5d spread, exact df-correct critical value;
 *   - READINESS BAR MIN_CLUSTERS = 15 usable (+5d-realized) weeks: below it
 *     the result is WAITING, never PASS/FAIL (Reasoning Standard #4);
 *   - PASS = pooled TOP > MID > BOTTOM ordering holds AND the cluster test
 *     survives at 5%; otherwise FAIL.
 *
 * WHY A JOB AND NOT A SESSION: the readiness condition (n >= 15) is met the
 * first Sunday after 2026-11-15; the only thing between the archive and the
 * registered verdict was a session typing `npx tsx scripts/github_activity_gate2.ts`
 * (EDGE DOCTRINE #3: the second occurrence of a computation becomes code the
 * bot runs free). The production image ships dist/ only (frozen Dockerfile —
 * no scripts/ tree, KNOWN BROKEN #44), so the test runs IN-PROCESS from the
 * bundle: the archive is read directly (githubOrgActivity.lookupGithubOrgHistory,
 * the same reader behind the public /history route the script fetched), and
 * forward returns come from Alpaca daily bars (adjustment=all like the price
 * panel; the feed from resolveAlpacaFeed — delayed_sip by default, the
 * subscription-permitted form, ALPACA_DATA_FEED overrides) instead of the
 * script's Yahoo adjusted closes. That is the ONE measurement-input difference between a
 * hand run and the job; the rule, the bar and the evaluation are the same
 * code. FIRST SUFFICIENT RUN IS THE REGISTERED VERDICT: latest.json carries
 * `first_sufficient_run` forward unchanged (earningsQoq.ts's convention) —
 * later weekly runs are informational, and a hand run of the script is
 * informational too.
 *
 * WHEN: Sundays 19:00-20:00 ET start — after earningsQoq's 18:00-19:00 and
 * done well before the 20:15 panel build, so the Sunday long-job lane stays
 * serialized even though this job is seconds long (15 symbols, one paged
 * multi-symbol bars request). At most one attempt per ISO week; re-runs for a
 * new code version or a result older than 6 days. No Python, no subprocess.
 * Result files are KB-sized under DATA_DIR/github_gate2 — derived statistics
 * only: per-week bucket means and the test, never a price (Alpaca data licence).
 * FLAG: VOLTRADE_GITHUB_GATE2, default ON; present env var wins.
 */
import fs from "fs";
import path from "path";
import { beat, zonedParts } from "./jobHeartbeat";
import { WATCHLIST, lookupGithubOrgHistory, type GithubActivityRecord } from "./githubOrgActivity";
import { isoWeekKey, type RunOutcome } from "./walkforward";
import { resolveAlpacaFeed } from "./alpacaFeed";
import { clusterMeanTTest, survivesAtCrit005, type ClusterTTestResult } from "../scripts/statsUtils";
import { version as pkgVersion } from "../package.json";

export const GITHUB_GATE2_SUBDIR = "github_gate2";
export const GITHUB_GATE2_SCHEMA = "github_gate2/1";
export const GITHUB_GATE2_TIMEOUT_MS = 15 * 60_000;
export const GITHUB_GATE2_TICK_MS = 30 * 60_000;
/** Weekly re-read of a growing archive (informational after the first sufficient run). */
export const GITHUB_GATE2_STALE_DAYS = 6;
export const GITHUB_GATE2_WINDOW_START_MIN = 19 * 60;
export const GITHUB_GATE2_WINDOW_END_MIN = 20 * 60;
/** Sanity margin only: the job writes KB-sized results; the volume hovers ~1 GB free. */
export const GITHUB_GATE2_MIN_FREE_BYTES = 64_000_000;
/** Archived weeks read per ticker — the archive is ~17 weeks old (2026-07-20); 90 is the route's own cap. */
export const GITHUB_GATE2_HISTORY_WEEKS = 90;
export const GITHUB_GATE2_REGISTERED_NOT_BEFORE = "2026-11-15";
const ET = "America/New_York";
const DAY_MS = 86_400_000;
const YMD_RE = /^\d{4}-\d{2}-\d{2}$/;

// ── the pre-registered rule (moved verbatim from scripts/github_activity_gate2.ts) ──

export const MIN_CLUSTERS = 15;
export const MIN_PANEL_FOR_WEEK = 12; // out of 15 — allow a couple of missing orgs before dropping the week

export interface OrgWeek {
  ticker: string;
  weekStart: string;
  weekEnd: string;
  total: number | null; // mergedPRs + commits, null if either field missing
}

export function toWeeklyTotal(rec: { weekStart: string; weekEnd: string; mergedPRs: number | null; commits: number | null }): OrgWeek {
  const total = rec.mergedPRs != null && rec.commits != null ? rec.mergedPRs + rec.commits : null;
  return { ticker: "", weekStart: rec.weekStart, weekEnd: rec.weekEnd, total };
}

export interface WeekDelta {
  ticker: string;
  weekEnd: string;
  pctChange: number;
}

/** Week-over-week percent change per ticker, computed against each
 *  ticker's own prior archived week only (never cross-ticker). Pure
 *  function over already-fetched per-ticker series (weeks sorted
 *  ascending by weekStart, one series per ticker). */
export function computeWeeklyDeltas(seriesByTicker: Map<string, OrgWeek[]>): WeekDelta[] {
  const out: WeekDelta[] = [];
  for (const [ticker, weeks] of seriesByTicker) {
    for (let i = 1; i < weeks.length; i++) {
      const prev = weeks[i - 1].total;
      const cur = weeks[i].total;
      if (prev == null || cur == null || prev === 0) continue;
      out.push({ ticker, weekEnd: weeks[i].weekEnd, pctChange: (cur - prev) / prev });
    }
  }
  return out;
}

export interface WeekBucket {
  weekEnd: string;
  top: string[];
  mid: string[];
  bottom: string[];
}

/** Cross-sectional TOP/MID/BOTTOM terciles for one week's deltas. Returns
 *  null if fewer than MIN_PANEL_FOR_WEEK tickers have a valid delta that
 *  week (dropped entirely, never bucketed unevenly). */
export function bucketWeek(weekEnd: string, deltas: WeekDelta[]): WeekBucket | null {
  const rows = deltas.filter((d) => d.weekEnd === weekEnd);
  if (rows.length < MIN_PANEL_FOR_WEEK) return null;
  const sorted = [...rows].sort((a, b) => b.pctChange - a.pctChange);
  const third = Math.floor(sorted.length / 3);
  return {
    weekEnd,
    top: sorted.slice(0, third).map((r) => r.ticker),
    mid: sorted.slice(third, sorted.length - third).map((r) => r.ticker),
    bottom: sorted.slice(sorted.length - third).map((r) => r.ticker),
  };
}

/** A daily adjusted-close series: dates ascending, one close per date.
 *  Structurally compatible with scripts/occ_volume_gate2.ts's `Series`
 *  (Yahoo) and with what fetchAlpacaCloses builds (Alpaca). */
export interface CloseSeries { dates: string[]; closes: number[] }

/** Latest date in `series` on or before `onOrBefore` (ISO yyyy-mm-dd
 *  string compare is safe since both are zero-padded). Null if the
 *  series has no date that early. */
export function nearestTradingDayOnOrBefore(series: { dates: string[] }, onOrBefore: string): string | null {
  let best: string | null = null;
  for (const d of series.dates) {
    if (d <= onOrBefore && (best == null || d > best)) best = d;
  }
  return best;
}

/** Close-to-close return entering at `entryDate` (an exact series date) and
 *  exiting `horizon` bars later; null when the window is not fully in the
 *  past yet (no lookahead: a week is scored only once its +h bar exists). Same
 *  rule as scripts/occ_volume_gate2.ts fwdReturn, over a CloseSeries. */
export function forwardCloseReturn(series: CloseSeries, entryDate: string, horizon: number): number | null {
  const idx = series.dates.indexOf(entryDate);
  if (idx < 0) return null;
  const fwdIdx = idx + horizon;
  if (fwdIdx >= series.dates.length) return null;
  const entry = series.closes[idx];
  const fwd = series.closes[fwdIdx];
  if (!entry || !Number.isFinite(entry) || !Number.isFinite(fwd)) return null;
  return fwd / entry - 1;
}

export interface WeekResult {
  weekEnd: string;
  topMean5: number | null;
  midMean5: number | null;
  bottomMean5: number | null;
  spread5: number | null; // top - bottom, this week's cluster value
  topMean20: number | null;
  midMean20: number | null;
  bottomMean20: number | null;
}

export function meanOf(xs: (number | null)[]): number | null {
  const clean = xs.filter((x): x is number => x != null);
  return clean.length ? clean.reduce((a, b) => a + b, 0) / clean.length : null;
}

export type GateVerdict = "PASS" | "FAIL" | "WAITING";

export interface GithubGate2Evaluation {
  verdict: GateVerdict;
  ready: boolean;
  n_clusters: number;
  min_clusters: number;
  panel: { tickers_with_history: number; watchlist: number; weekly_deltas: number; candidate_weeks: number; bucketed_weeks: number; dropped_weeks: number; tickers_priced: number };
  pooledTop5: number | null;
  pooledMid5: number | null;
  pooledBottom5: number | null;
  orderingHolds: boolean;
  clusterTest: ClusterTTestResult | null;
  survivesAtCrit005: boolean;
  results: WeekResult[];
}

/** The whole pre-registered evaluation, pure: archive series in, price series
 *  in, verdict out. Shared by the production job (Alpaca prices) and the
 *  session script (Yahoo prices) so the rule has exactly one home. */
export function evaluateGithubGate2(seriesByTicker: Map<string, OrgWeek[]>, priceSeries: Map<string, CloseSeries>, watchlistSize = WATCHLIST.length): GithubGate2Evaluation {
  const deltas = computeWeeklyDeltas(seriesByTicker);
  const weekEnds = Array.from(new Set(deltas.map((d) => d.weekEnd))).sort();
  const buckets = weekEnds.map((w) => bucketWeek(w, deltas)).filter((b): b is WeekBucket => b != null);

  const bucketFwd = (tickers: string[], weekEnd: string, horizon: number): (number | null)[] =>
    tickers.map((t) => {
      const s = priceSeries.get(t);
      if (!s) return null;
      const entry = nearestTradingDayOnOrBefore(s, weekEnd);
      return entry ? forwardCloseReturn(s, entry, horizon) : null;
    });

  const results: WeekResult[] = buckets.map((b) => {
    const topMean5 = meanOf(bucketFwd(b.top, b.weekEnd, 5));
    const midMean5 = meanOf(bucketFwd(b.mid, b.weekEnd, 5));
    const bottomMean5 = meanOf(bucketFwd(b.bottom, b.weekEnd, 5));
    return {
      weekEnd: b.weekEnd,
      topMean5, midMean5, bottomMean5,
      spread5: topMean5 != null && bottomMean5 != null ? topMean5 - bottomMean5 : null,
      topMean20: meanOf(bucketFwd(b.top, b.weekEnd, 20)),
      midMean20: meanOf(bucketFwd(b.mid, b.weekEnd, 20)),
      bottomMean20: meanOf(bucketFwd(b.bottom, b.weekEnd, 20)),
    };
  });

  const usable5 = results.filter((r) => r.spread5 != null);
  const ready = usable5.length >= MIN_CLUSTERS;
  const spreads = usable5.map((r) => r.spread5 as number);
  const clusterTest = spreads.length ? clusterMeanTTest(spreads) : null;
  const pooledTop5 = meanOf(usable5.map((r) => r.topMean5));
  const pooledMid5 = meanOf(usable5.map((r) => r.midMean5));
  const pooledBottom5 = meanOf(usable5.map((r) => r.bottomMean5));
  const orderingHolds = pooledTop5 != null && pooledMid5 != null && pooledBottom5 != null && pooledTop5 > pooledMid5 && pooledMid5 > pooledBottom5;
  const significant = clusterTest ? survivesAtCrit005(clusterTest) : false;
  const verdict: GateVerdict = ready ? (orderingHolds && significant ? "PASS" : "FAIL") : "WAITING";
  return {
    verdict, ready, n_clusters: usable5.length, min_clusters: MIN_CLUSTERS,
    panel: {
      tickers_with_history: seriesByTicker.size, watchlist: watchlistSize, weekly_deltas: deltas.length,
      candidate_weeks: weekEnds.length, bucketed_weeks: buckets.length, dropped_weeks: weekEnds.length - buckets.length,
      tickers_priced: priceSeries.size,
    },
    pooledTop5, pooledMid5, pooledBottom5, orderingHolds, clusterTest, survivesAtCrit005: significant, results,
  };
}

// ── inputs: the archive and Alpaca daily bars ────────────────────────────────

/** Per-ticker weekly series from the archive (ascending by weekStart), one
 *  entry per watchlist ticker that has at least one archived week. */
export function readArchiveSeries(baseDir?: string, weeks = GITHUB_GATE2_HISTORY_WEEKS): Map<string, OrgWeek[]> {
  const out = new Map<string, OrgWeek[]>();
  for (const org of WATCHLIST) {
    const recs: GithubActivityRecord[] = lookupGithubOrgHistory(org.ticker, weeks, baseDir);
    if (!recs.length) continue;
    out.set(org.ticker, recs.map((rec) => ({ ...toWeeklyTotal(rec), ticker: org.ticker })));
  }
  return out;
}

export type JsonGetter = (url: string) => Promise<unknown>;

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);

/** Daily adjusted closes for `symbols` from Alpaca's multi-symbol bars endpoint
 *  (adjustment=all like the price panel; `feed` from resolveAlpacaFeed — never
 *  hardcoded, alpacaFeed.test.ts's ratchet), all pages, into ascending
 *  CloseSeries per symbol. A symbol Alpaca returns nothing for is omitted (the
 *  evaluation then scores it as unpriced), never zero-filled. */
export async function fetchAlpacaCloses(get: JsonGetter, dataBase: string, symbols: readonly string[], start: string, end: string, feed: string, maxPages = 20): Promise<Map<string, CloseSeries>> {
  const closes = new Map<string, Map<string, number>>();
  let token: string | null = null;
  let pages = 0;
  do {
    const q = new URLSearchParams({ symbols: symbols.join(","), timeframe: "1Day", adjustment: "all", start, end, limit: "10000", feed });
    if (token) q.set("page_token", token);
    const body = await get(`${dataBase}/v2/stocks/bars?${q.toString()}`);
    const bars = isObj(body) && isObj(body.bars) ? body.bars : {};
    for (const [sym, arr] of Object.entries(bars)) {
      if (!Array.isArray(arr)) continue;
      const m = closes.get(sym) ?? new Map<string, number>();
      for (const b of arr) {
        if (!isObj(b)) continue;
        const d = typeof b.t === "string" ? b.t.slice(0, 10) : "";
        const c = typeof b.c === "number" && Number.isFinite(b.c) ? b.c : null;
        if (YMD_RE.test(d) && c !== null) m.set(d, c);
      }
      closes.set(sym, m);
    }
    token = isObj(body) && typeof body.next_page_token === "string" && body.next_page_token ? body.next_page_token : null;
    pages++;
  } while (token && pages < maxPages);
  const out = new Map<string, CloseSeries>();
  for (const [sym, m] of closes) {
    if (!m.size) continue;
    const dates = Array.from(m.keys()).sort();
    out.set(sym, { dates, closes: dates.map((d) => m.get(d) as number) });
  }
  return out;
}

/** Alpaca GET with the server's data-API credentials; one attempt, 20 s cap. */
export function alpacaJsonGetter(env: Record<string, string | undefined>, fetchImpl: typeof fetch = fetch): JsonGetter {
  const headers = { "APCA-API-KEY-ID": env.ALPACA_KEY || "", "APCA-API-SECRET-KEY": env.ALPACA_SECRET || "" };
  return async (url: string) => {
    const r = await fetchImpl(url, { headers, signal: AbortSignal.timeout(20_000) });
    if (!r.ok) throw new Error(`alpaca HTTP ${r.status}`);
    return r.json();
  };
}

/** The bars window the archive needs: a week before the earliest archived
 *  weekEnd through the day before `nowMs` (daily bars; yesterday is final). */
export function barsWindow(seriesByTicker: Map<string, OrgWeek[]>, nowMs: number): { start: string; end: string } | null {
  const ends = Array.from(seriesByTicker.values()).flat().map((w) => w.weekEnd).filter((d) => YMD_RE.test(d)).sort();
  if (!ends.length) return null;
  const start = new Date(Date.parse(`${ends[0]}T00:00:00Z`) - 7 * DAY_MS).toISOString().slice(0, 10);
  const end = zonedParts(nowMs - DAY_MS, ET).date;
  return { start, end };
}

// ── schedule ────────────────────────────────────────────────────────────────

export function githubGate2Flag(env: Record<string, string | undefined>): { value: string; source: "env" | "default"; on: boolean } {
  const raw = env.VOLTRADE_GITHUB_GATE2;
  const value = raw === undefined ? "1" : String(raw);
  const v = value.trim().toLowerCase();
  return { value, source: raw === undefined ? "default" : "env", on: v === "1" || v === "true" || v === "on" || v === "yes" };
}

export function githubGate2Root(env: Record<string, string | undefined> = process.env): string {
  return path.join(env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp"), GITHUB_GATE2_SUBDIR);
}

/** Sunday, 19:00-20:00 ET. */
export function inGithubGate2Window(nowMs: number): boolean {
  const p = zonedParts(nowMs, ET);
  return p.weekday === "Sun" && p.minutes >= GITHUB_GATE2_WINDOW_START_MIN && p.minutes < GITHUB_GATE2_WINDOW_END_MIN;
}

export interface GithubGate2Latest { code_version: string | null; generated_at: string | null }

export type GithubGate2Decision =
  | { action: "idle"; reason: string }
  | { action: "skip" | "run"; reason: string; week: string };

/** Pure: what this tick should do. */
export function decideGithubGate2(nowMs: number, s: {
  enabled: boolean; latest: GithubGate2Latest | null; lastAttemptWeek: string; codeVersion: string; freeBytes: number | null;
}): GithubGate2Decision {
  if (!s.enabled) return { action: "idle", reason: "VOLTRADE_GITHUB_GATE2 off" };
  if (!inGithubGate2Window(nowMs)) return { action: "idle", reason: "outside Sunday 19:00-20:00 ET" };
  const week = isoWeekKey(zonedParts(nowMs, ET).date);
  if (s.lastAttemptWeek === week) return { action: "idle", reason: `already attempted ${week}` };
  if (s.freeBytes !== null && s.freeBytes < GITHUB_GATE2_MIN_FREE_BYTES) {
    return { action: "skip", week, reason: `only ${Math.floor(s.freeBytes / 1e6)} MB free on the volume (< ${GITHUB_GATE2_MIN_FREE_BYTES / 1e6} MB)` };
  }
  const l = s.latest;
  if (l && l.code_version === s.codeVersion && l.generated_at) {
    const age = nowMs - Date.parse(l.generated_at);
    if (Number.isFinite(age) && age < GITHUB_GATE2_STALE_DAYS * DAY_MS) {
      return { action: "skip", week, reason: `fresh: code ${l.code_version}, ${Math.floor(age / DAY_MS)}d old` };
    }
  }
  return { action: "run", week, reason: l ? `code ${l.code_version ?? "?"} -> ${s.codeVersion}, last ${l.generated_at ?? "?"}` : "no result yet" };
}

// ── result files ────────────────────────────────────────────────────────────

function readJson(file: string): Record<string, unknown> | null {
  try {
    const v = JSON.parse(fs.readFileSync(file, "utf8"));
    return isObj(v) ? v : null;
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[github_gate2] unreadable", file, e instanceof Error ? e.message : e);
    return null;
  }
}
const str = (v: unknown): string | null => (typeof v === "string" ? v : null);
const obj = (v: unknown): Record<string, unknown> => (isObj(v) ? v : {});

export function githubGate2Latest(root: string): GithubGate2Latest | null {
  const l = readJson(path.join(root, "latest.json"));
  if (!l) return null;
  return { code_version: str(l.code_version), generated_at: str(l.generated_at) };
}

export interface FirstSufficientRun { generated_at: string; code_version: string; verdict: GateVerdict; n_clusters: number; t: number | null; df: number | null; orderingHolds: boolean }

/** The `first_sufficient_run` block to persist: a prior one is carried forward
 *  UNCHANGED (the first sufficient look is the registered verdict); a WAITING
 *  evaluation never creates one. Pure. */
export function carryFirstSufficientRun(prev: Record<string, unknown> | null, ev: GithubGate2Evaluation, generatedAt: string, codeVersion: string): FirstSufficientRun | null {
  const existing = prev && isObj(prev.first_sufficient_run) ? (prev.first_sufficient_run as unknown as FirstSufficientRun) : null;
  if (existing) return existing;
  if (ev.verdict === "WAITING") return null;
  return {
    generated_at: generatedAt, code_version: codeVersion, verdict: ev.verdict, n_clusters: ev.n_clusters,
    t: ev.clusterTest ? ev.clusterTest.t : null, df: ev.clusterTest ? ev.clusterTest.df : null, orderingHolds: ev.orderingHolds,
  };
}

/** Round every per-week mean so the file holds statistics, not re-derivable prices. */
function roundResult(r: WeekResult): WeekResult {
  const r6 = (v: number | null) => (v == null ? null : Math.round(v * 1e6) / 1e6);
  return { weekEnd: r.weekEnd, topMean5: r6(r.topMean5), midMean5: r6(r.midMean5), bottomMean5: r6(r.bottomMean5), spread5: r6(r.spread5), topMean20: r6(r.topMean20), midMean20: r6(r.midMean20), bottomMean20: r6(r.bottomMean20) };
}

export interface GithubGate2Deps {
  env: Record<string, string | undefined>;
  root: string;
  archiveBaseDir?: string;
  get?: JsonGetter;
  dataBase?: string;
  now?: () => number;
  codeVersion?: string;
}

/** Run the gate once: archive -> Alpaca closes -> evaluation -> latest.json /
 *  last_run.json. Never throws; the outcome is what the heartbeat records. */
export async function runGithubGate2(deps: GithubGate2Deps): Promise<RunOutcome> {
  const now = deps.now ?? Date.now;
  const codeVersion = deps.codeVersion ?? pkgVersion;
  const startedMs = now();
  const generatedAt = new Date(startedMs).toISOString();
  const writeLastRun = (status: RunOutcome["status"], detail: string, extra: Record<string, unknown> = {}) => {
    try {
      fs.mkdirSync(deps.root, { recursive: true });
      fs.writeFileSync(path.join(deps.root, "last_run.json"), JSON.stringify({ schema: GITHUB_GATE2_SCHEMA, generated_at: generatedAt, code_version: codeVersion, status, detail, elapsed_ms: now() - startedMs, ...extra }, null, 2));
    } catch (e) {
      console.error("[github_gate2] last_run.json write failed:", e instanceof Error ? e.message : e);
    }
  };
  try {
    if (!deps.env.ALPACA_KEY || !deps.env.ALPACA_SECRET) {
      const detail = "ALPACA_KEY/ALPACA_SECRET not set — no price source for forward returns";
      writeLastRun("skipped", detail);
      return { status: "skipped", detail };
    }
    const seriesByTicker = readArchiveSeries(deps.archiveBaseDir);
    const window = barsWindow(seriesByTicker, startedMs);
    if (!window) {
      const detail = "no archived github_activity weeks found";
      writeLastRun("skipped", detail);
      return { status: "skipped", detail };
    }
    const get = deps.get ?? alpacaJsonGetter(deps.env);
    const dataBase = deps.dataBase || "https://data.alpaca.markets";
    const feed = resolveAlpacaFeed(deps.env.ALPACA_DATA_FEED);
    const priceSeries = await fetchAlpacaCloses(get, dataBase, Array.from(seriesByTicker.keys()), window.start, window.end, feed);
    const ev = evaluateGithubGate2(seriesByTicker, priceSeries);
    const prev = readJson(path.join(deps.root, "latest.json"));
    const first = carryFirstSufficientRun(prev, ev, generatedAt, codeVersion);
    const latest = {
      schema: GITHUB_GATE2_SCHEMA, generated_at: generatedAt, code_version: codeVersion,
      source: { archive: "github_activity (datacoreArchive)", prices: `alpaca /v2/stocks/bars adjustment=all feed=${feed}`, bars_window: window },
      root: "github_org_engineering_momentum", registered_not_before: GITHUB_GATE2_REGISTERED_NOT_BEFORE,
      verdict: ev.verdict, ready: ev.ready, n_clusters: ev.n_clusters, min_clusters: ev.min_clusters, panel: ev.panel,
      pooled5: { top: ev.pooledTop5, mid: ev.pooledMid5, bottom: ev.pooledBottom5, orderingHolds: ev.orderingHolds },
      cluster_test: ev.clusterTest, survives_at_crit005: ev.survivesAtCrit005,
      per_week: ev.results.map(roundResult),
      first_sufficient_run: first,
      verdict_rule: `WAITING below MIN_CLUSTERS=${MIN_CLUSTERS} usable +5d weeks; PASS = pooled TOP>MID>BOTTOM AND cluster-mean t survives the df-correct 5% critical value; else FAIL. first_sufficient_run is the registered verdict`,
      caveat: "pooled 15-org panel; the root's own prior expects structure for maybe a third of it — a per-ticker split after the fact would be fishing and is not computed here",
      elapsed_ms: now() - startedMs,
    };
    fs.mkdirSync(deps.root, { recursive: true });
    fs.writeFileSync(path.join(deps.root, "latest.json"), JSON.stringify(latest, null, 2));
    const t = ev.clusterTest ? ev.clusterTest.t.toFixed(3) : "n/a";
    const detail = `verdict ${ev.verdict}: ${ev.n_clusters}/${MIN_CLUSTERS} usable weeks, ${ev.panel.tickers_priced}/${ev.panel.tickers_with_history} priced, t=${t}, ordering=${ev.orderingHolds}`;
    writeLastRun("ok", detail, { verdict: ev.verdict, n_clusters: ev.n_clusters });
    return { status: "ok", detail };
  } catch (e) {
    const detail = e instanceof Error ? e.message : String(e);
    writeLastRun("error", detail);
    return { status: "error", detail };
  }
}

/** Read-only status for the "github_gate2" diag probe: headline blocks only. */
export function githubGate2Status(env: Record<string, string | undefined> = process.env, root = githubGate2Root(env)): Record<string, unknown> {
  const l = readJson(path.join(root, "latest.json"));
  const flag = githubGate2Flag(env);
  return {
    probe: "github_gate2",
    flag: { value: flag.value, source: flag.source, on: flag.on },
    schedule: `Sunday ${GITHUB_GATE2_WINDOW_START_MIN / 60}:00-${GITHUB_GATE2_WINDOW_END_MIN / 60}:00 ET start, once per ISO week, in-process, budget ${GITHUB_GATE2_TIMEOUT_MS / 60_000} min`,
    registered_not_before: GITHUB_GATE2_REGISTERED_NOT_BEFORE,
    code_version: pkgVersion,
    latest: l ? {
      schema: str(l.schema), generated_at: str(l.generated_at), code_version: str(l.code_version), source: obj(l.source),
      verdict: str(l.verdict), ready: l.ready ?? null, n_clusters: l.n_clusters ?? null, min_clusters: l.min_clusters ?? null, panel: obj(l.panel),
      pooled5: obj(l.pooled5), cluster_test: l.cluster_test ?? null, survives_at_crit005: l.survives_at_crit005 ?? null,
      per_week: Array.isArray(l.per_week) ? l.per_week : null,
      first_sufficient_run: l.first_sufficient_run ?? null, verdict_rule: str(l.verdict_rule), caveat: str(l.caveat), elapsed_ms: l.elapsed_ms ?? null,
    } : null,
    last_run: readJson(path.join(root, "last_run.json")),
  };
}

let booted = false;

/** Weekly tick (unref'd; first tick one interval after boot). One run at a time. */
export function bootGithubGate2(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  let lastAttemptWeek = "";
  const tick = async () => {
    if (running) return;
    const root = githubGate2Root(env);
    let free: number | null = null;
    try { const s = fs.statfsSync(root.startsWith("/data") ? "/data" : path.dirname(root)); free = Number(s.bavail) * Number(s.bsize); } catch { free = null; }
    const d = decideGithubGate2(Date.now(), {
      enabled: githubGate2Flag(env).on, latest: githubGate2Latest(root), lastAttemptWeek, codeVersion: pkgVersion, freeBytes: free,
    });
    if (d.action === "idle") return;
    lastAttemptWeek = d.week;
    if (d.action === "skip") {
      beat("github_gate2_weekly", "finish", "skipped", d.reason);
      return;
    }
    running = true;
    beat("github_gate2_weekly", "start");
    try {
      const budget = new Promise<RunOutcome>((resolve) => setTimeout(() => resolve({ status: "error", detail: `exceeded the ${GITHUB_GATE2_TIMEOUT_MS / 60_000}-minute budget` }), GITHUB_GATE2_TIMEOUT_MS).unref?.());
      const out = await Promise.race([runGithubGate2({ env, root }), budget]);
      beat("github_gate2_weekly", "finish", out.status === "error" ? "error" : out.status === "ok" ? "ok" : "skipped", `${out.status}: ${out.detail}`);
    } catch (e) {
      beat("github_gate2_weekly", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  setInterval(() => { void tick(); }, GITHUB_GATE2_TICK_MS).unref?.();
}
