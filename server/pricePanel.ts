/**
 * pricePanel.ts — survivorship-REDUCED daily US-equity price panel built from
 * Alpaca's free historical bars (human order 2026-10-28: "Check whether
 * Alpaca's free bars include delisted tickers since 2010. If they do, build
 * the price panel from them. No paid data.").
 *
 * WHAT THE PROBE FOUND (bars_coverage, prod v1.0.1203, research/
 * price_panel_coverage.md): sip history starts 2016-01-04 (iex 2018-11-01),
 * so 2010-2015 is NOT available. 12 of 13 post-2016 delistings probed reach
 * their event under the default symbol mapping; the 4 pre-2016 cases return
 * nothing. /v2/assets lists only 3 of the 17 probed delisted names as
 * inactive, so the asset list alone UNDER-ENUMERATES delistings: the
 * universe is supplemented from Alpaca's own corporate-actions feed
 * (merger acquirees, name changes, worthless removals).
 *
 * WHAT IT BUILDS (DATA_DIR/price_panel/<build_id>/; storage_config.py
 * PRICE_PANEL_DIR names the same directory for Python readers):
 *   universe.jsonl     one symbol per line: status, asset ids, event, reuse flag
 *   bars/bNNNNN.jsonl.gz  one line per (symbol, mapping mode, date segment),
 *                      columnar + delta-encoded (encodeSegment / decodeSegment)
 *   manifest.json      as_of, source, feed, counts, progress checkpoint, caveats
 *   ../current.json    pointer to the last COMPLETE build (swapped on completion)
 * Identity = asset id(s) + date segment, never the ticker alone: a series is
 * split wherever consecutive bars are > SEGMENT_GAP_DAYS apart, and a symbol
 * that is both active and inactive (ticker reuse, 236 such symbols live) is
 * also fetched with mapping disabled (asof=-) so the new company is not lost.
 * Default-mapping series of delisted names are clipped after the corporate-
 * action event date when one is known (successor / OTC continuation);
 * otherwise kept and flagged.
 *
 * HOW IT RUNS: a nightly job (JOB_REGISTRY "price_panel_build"), ticking
 * every 15 min, working only 20:15-04:00 ET (Tier 2 scans run 04:00-20:00 ET
 * every day, so the bot's own Alpaca usage is lowest then). One request in
 * flight, >= 600 ms apart (<= 100/min of the 200/min quota), 60 s back-off on
 * 429. Resumable: the manifest checkpoint advances after every batch of
 * BATCH_SYMBOLS symbols. Memory-bounded: one batch (<= 50 symbols x ~2.7k
 * bars) in memory at a time, gzip is async, nothing runs in the Python
 * daemon. Disk-bounded: stops before free space on the volume falls under
 * PRICE_PANEL_MIN_FREE_BYTES (archives keep priority: an archive gap never
 * refills) and at PRICE_PANEL_MAX_BYTES. Never started by a boot; never on
 * the trading loop. GET only — no order path is referenced.
 *
 * FLAG: VOLTRADE_PRICE_PANEL, default ON (human-ordered build); when the env
 * var is PRESENT it wins, like regime_engine.flag_raw ("0"/"off"/"" disable).
 *
 * CAVEATS (also written into every manifest): pre-2016 history and pre-2016
 * delistings are invisible; default-mapping series can run past an event
 * when no corporate action dates it; OTC continuation after an exchange
 * delisting is usually absent, so delisting returns must be assumed by the
 * backtest; adjustment=all back-adjusts with today's knowledge (the raw
 * close is recoverable from the per-segment factor runs); prices are for
 * internal research only — never redistributed through /data or /api/v1.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { promisify } from "util";
import type { FetchLike } from "./optionsEntitlements";
import { beat, zonedParts } from "./jobHeartbeat";
import { vendorMsgClass } from "./barsCoverage";

const gzip = promisify(zlib.gzip);

// ── constants ────────────────────────────────────────────────────────────────
export const PRICE_PANEL_SUBDIR = "price_panel";
export const PANEL_START = "2016-01-01";
export const BATCH_SYMBOLS = 50;
export const MIN_INTERVAL_MS = 600;
export const BACKOFF_429_MS = 60_000;
export const MAX_429_IN_A_ROW = 5;
export const RETRY_OTHER_MS = 10_000;
export const MAX_RETRIES_OTHER = 3;
export const CALL_TIMEOUT_MS = 30_000;
export const SEGMENT_GAP_DAYS = 30;
/** On a batch 403, retry with an end date up to this many days earlier before bisecting. */
export const END_FALLBACK_DAYS = 3;
// 2026-10-29 (human-ordered "lower the panel disk floor so it finishes"): was
// 1.5 GiB, which paused the first build at 62% (288/466 batches, ~208 MB on
// disk) with 1.53 GiB free. 1.25 GiB still leaves the archives their 1 GiB
// margin plus 256 MiB; the remaining ~130 MB fits. The 30-day rebuild briefly
// holds two builds (~2 x 340 MB) — grow the volume before then (wishlist
// VOLUME CAPACITY). Env VOLTRADE_PRICE_PANEL_MIN_FREE_BYTES still overrides.
export const PRICE_PANEL_MIN_FREE_BYTES = 1.25 * 1_073_741_824;
export const PRICE_PANEL_MAX_BYTES = 600 * 1_048_576;
export const PRICE_PANEL_REFRESH_DAYS = 30;
export const WINDOW_START_MIN = 20 * 60 + 15;   // = the heartbeat dueMin, so every session counts for that night
export const WINDOW_END_MIN = 4 * 60;
export const TICK_MS = 15 * 60_000;
const ET = "America/New_York";
const DAY_MS = 86_400_000;
const SYMBOL_RE = /^[A-Z][A-Z0-9.-]{0,11}$/;
const YMD_RE = /^\d{4}-\d{2}-\d{2}$/;
const EXCLUDED_EXCHANGES = new Set(["OTC", "CRYPTO"]);
const CORP_TYPES = "cash_merger,stock_merger,stock_and_cash_merger,name_change,worthless_removal";

export const CAVEATS = [
  "History starts 2016-01-04 (Alpaca free sip); 2010-2015 is not in this panel and pre-2016 delistings are invisible.",
  "Survivorship-REDUCED, not free: the universe is /v2/assets (active+inactive) plus corporate-action symbols; a delisted name that appears in neither is missing.",
  "Identity is asset id(s) + date segment, never the ticker alone; segments split at gaps > 30 days; reuse_suspect symbols carry a second, unmapped (asof=-) series.",
  "Default-mapping series are clipped after a dated corporate action; without one (e.g. bankruptcies moving to OTC) they are kept and flagged inactive_no_event_date.",
  "Post-delisting OTC prices are usually absent: the backtest must assume a delisting return and state it.",
  "Prices are adjustment=all (splits+dividends, back-adjusted with today's knowledge); raw close = c * factor run (field f).",
  "Alpaca market data: internal research use only — never redistributed through /data or /api/v1.",
];

// ── flag ─────────────────────────────────────────────────────────────────────
/** Env var PRESENT wins (any value; "0"/"off"/"" disable); absent = default ON. */
export function pricePanelFlag(env: Record<string, string | undefined>): { value: string; source: "env" | "default"; on: boolean } {
  const raw = env.VOLTRADE_PRICE_PANEL;
  const value = raw === undefined ? "1" : String(raw);
  const v = value.trim().toLowerCase();
  return { value, source: raw === undefined ? "default" : "env", on: v === "1" || v === "true" || v === "on" || v === "yes" };
}

export function pricePanelEnabled(env: Record<string, string | undefined>): boolean {
  return pricePanelFlag(env).on && !!(env.ALPACA_KEY && env.ALPACA_SECRET);
}

export function pricePanelRoot(env: Record<string, string | undefined> = process.env): string {
  return path.join(env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp"), PRICE_PANEL_SUBDIR);
}

/** Last bar date the free plan will serve: the day before the ET calendar date.
 *  NOT the UTC day before: the build window opens 20:15 ET = 00:15 UTC, so UTC
 *  "yesterday" is the current ET date there, and Alpaca answers 403
 *  "recent SIP data" for it (first live night, 2026-10-29: HTTP 403, batch 0/466). */
export function panelAsOf(nowMs: number): string {
  return ymd(Date.parse(`${zonedParts(nowMs, ET).date}T00:00:00Z`) - DAY_MS);
}

/** 20:15-04:00 ET, every night. */
export function inBuildWindow(nowMs: number): boolean {
  const m = zonedParts(nowMs, ET).minutes;
  return m >= WINDOW_START_MIN || m < WINDOW_END_MIN;
}

// ── universe ─────────────────────────────────────────────────────────────────
export interface AssetLite { id: string; symbol: string; exchange: string; status: "active" | "inactive" }
export interface CorpEvent { symbol: string; kind: "merger" | "name_change" | "worthless_removal"; date: string; successor: string | null }
export interface UniverseEntry {
  symbol: string;
  status: "active" | "inactive" | "corp_action";
  ids: string[];
  /** Earliest dated corporate action on/after PANEL_START that ends the symbol's old company. */
  event: { kind: CorpEvent["kind"]; date: string } | null;
  reuse_suspect: boolean;
}

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const str = (v: unknown) => (typeof v === "string" ? v : "");

export function parseAssets(body: unknown, status: "active" | "inactive"): AssetLite[] {
  if (!Array.isArray(body)) return [];
  const out: AssetLite[] = [];
  for (const a of body) {
    if (!isObj(a)) continue;
    const symbol = str(a.symbol).toUpperCase();
    if (!SYMBOL_RE.test(symbol)) continue;
    out.push({ id: str(a.id).slice(0, 64), symbol, exchange: str(a.exchange).toUpperCase(), status });
  }
  return out;
}

/** Alpaca v1beta1 corporate actions: { corporate_actions: { cash_mergers: [...], ... }, next_page_token }. */
export function parseCorporateActions(body: unknown): { events: CorpEvent[]; next: string | null } {
  const ca = isObj(body) && isObj(body.corporate_actions) ? body.corporate_actions : {};
  const events: CorpEvent[] = [];
  const add = (symbol: string, kind: CorpEvent["kind"], date: string, successor: string) => {
    const s = symbol.toUpperCase();
    if (SYMBOL_RE.test(s) && YMD_RE.test(date)) events.push({ symbol: s, kind, date, successor: successor ? successor.toUpperCase() : null });
  };
  const list = (k: string) => (Array.isArray(ca[k]) ? (ca[k] as unknown[]).filter(isObj) : []);
  for (const k of ["cash_mergers", "stock_mergers", "stock_and_cash_mergers"]) {
    for (const e of list(k)) add(str(e.acquiree_symbol), "merger", str(e.effective_date) || str(e.process_date), str(e.acquirer_symbol));
  }
  for (const e of list("name_changes")) add(str(e.old_symbol), "name_change", str(e.process_date), str(e.new_symbol));
  for (const e of list("worthless_removals")) add(str(e.symbol), "worthless_removal", str(e.process_date), "");
  const next = isObj(body) && typeof body.next_page_token === "string" && body.next_page_token ? body.next_page_token : null;
  return { events, next };
}

export function buildUniverse(assets: AssetLite[], events: CorpEvent[]): { entries: UniverseEntry[]; excluded: number } {
  const by = new Map<string, { active: AssetLite[]; inactive: AssetLite[] }>();
  let excluded = 0;
  for (const a of assets) {
    if (EXCLUDED_EXCHANGES.has(a.exchange)) { excluded++; continue; }
    const g = by.get(a.symbol) ?? { active: [], inactive: [] };
    g[a.status].push(a);
    by.set(a.symbol, g);
  }
  const firstEvent = new Map<string, CorpEvent>();
  for (const e of events) {
    if (e.date < PANEL_START) continue;
    const prev = firstEvent.get(e.symbol);
    if (!prev || e.date < prev.date) firstEvent.set(e.symbol, e);
  }
  const symbols = new Set<string>([...by.keys(), ...firstEvent.keys()]);
  const entries: UniverseEntry[] = [...symbols].sort().map((symbol) => {
    const g = by.get(symbol) ?? { active: [], inactive: [] };
    const ids = [...g.active, ...g.inactive].map((a) => a.id).filter(Boolean);
    const ev = firstEvent.get(symbol);
    return {
      symbol,
      status: g.active.length ? "active" : g.inactive.length ? "inactive" : "corp_action",
      ids,
      event: ev ? { kind: ev.kind, date: ev.date } : null,
      reuse_suspect: (g.active.length > 0 && g.inactive.length > 0) || ids.length > 1 || (g.active.length > 0 && !!ev),
    };
  });
  return { entries, excluded };
}

// ── series encoding ──────────────────────────────────────────────────────────
export interface Row { d: string; o: number; h: number; l: number; c: number; v: number; rc: number | null }

export interface EncodedSegment {
  d0: string; n: number; k: number;
  dd: number[]; c: number[]; o: number[]; h: number[]; l: number[]; v: number[];
  /** Raw-close factor runs: [row index, raw/adj ratio] whenever the ratio changes. */
  f: Array<[number, number]>;
}

const dayNum = (d: string) => Math.round(Date.parse(`${d}T00:00:00Z`) / DAY_MS);
const dayStr = (n: number) => new Date(n * DAY_MS).toISOString().slice(0, 10);

/** Price scale exponent: >= 4 significant digits at the series minimum, >= 1e-4 absolute. */
export function scaleFor(rows: Row[]): number {
  let min = Infinity;
  for (const r of rows) for (const p of [r.o, r.h, r.l, r.c]) if (p > 0 && p < min) min = p;
  if (!Number.isFinite(min)) return 4;
  return Math.min(10, Math.max(4, Math.ceil(4 - Math.log10(min))));
}

export function encodeSegment(rows: Row[]): EncodedSegment {
  const k = scaleFor(rows);
  const m = 10 ** k;
  const q = (x: number) => Math.round(x * m);
  const out: EncodedSegment = { d0: rows[0].d, n: rows.length, k, dd: [], c: [], o: [], h: [], l: [], v: [], f: [] };
  let prevDay = dayNum(rows[0].d), prevC = 0, lastRatio = NaN;
  rows.forEach((r, i) => {
    const day = dayNum(r.d), c = q(r.c);
    out.dd.push(day - prevDay); prevDay = day;
    out.c.push(c - prevC); prevC = c;
    out.o.push(q(r.o) - c); out.h.push(q(r.h) - c); out.l.push(q(r.l) - c);
    out.v.push(Math.round(r.v));
    if (r.rc !== null && r.c > 0) {
      const ratio = Number((r.rc / r.c).toPrecision(10));
      if (!Number.isFinite(lastRatio) || Math.abs(ratio / lastRatio - 1) > 1e-6) { out.f.push([i, ratio]); lastRatio = ratio; }
    }
  });
  return out;
}

export function decodeSegment(e: EncodedSegment): Array<Row & { rc: number | null }> {
  const m = 10 ** e.k;
  const rows: Array<Row & { rc: number | null }> = [];
  let day = dayNum(e.d0), c = 0, fi = 0, ratio: number | null = null;
  for (let i = 0; i < e.n; i++) {
    day += e.dd[i]; c += e.c[i];
    while (fi < e.f.length && e.f[fi][0] <= i) { ratio = e.f[fi][1]; fi++; }
    const cc = c / m;
    rows.push({ d: dayStr(day), o: (c + e.o[i]) / m, h: (c + e.h[i]) / m, l: (c + e.l[i]) / m, c: cc, v: e.v[i], rc: ratio === null ? null : cc * ratio });
  }
  return rows;
}

/** Split at gaps > SEGMENT_GAP_DAYS (a reused or relisted ticker is a different series). */
export function splitSegments(rows: Row[]): Row[][] {
  const segs: Row[][] = [];
  for (const r of rows) {
    const cur = segs[segs.length - 1];
    if (cur && dayNum(r.d) - dayNum(cur[cur.length - 1].d) <= SEGMENT_GAP_DAYS) cur.push(r);
    else segs.push([r]);
  }
  return segs;
}

export interface PanelLine extends EncodedSegment {
  s: string; mode: "default" | "raw"; seg: number; seg_start: string; seg_end: string;
  ids: string[]; st: UniverseEntry["status"]; ev: UniverseEntry["event"]; flags: string[];
}

/** Clip + segment + flag one symbol's rows into panel lines. */
export function linesFor(u: UniverseEntry, mode: "default" | "raw", rowsIn: Row[]): { lines: PanelLine[]; clipped: number } {
  let rows = rowsIn;
  let clipped = 0;
  const flags: string[] = [];
  if (u.reuse_suspect) flags.push("reuse_suspect");
  if (mode === "default" && u.event) {
    if (u.status === "active" && !u.reuse_suspect) flags.push("event_but_active_unclipped");
    else {
      const keep = rows.filter((r) => r.d <= u.event!.date);
      clipped = rows.length - keep.length;
      if (clipped) flags.push("clipped_after_event");
      rows = keep;
    }
  }
  if (u.status !== "active" && !u.event) flags.push("inactive_no_event_date");
  const lines = splitSegments(rows).map((seg, i) => {
    const segFlags = [...flags];
    if (mode === "raw" && u.event && seg[0].d > u.event.date) segFlags.push("post_event_segment");
    return {
      s: u.symbol, mode, seg: i, seg_start: seg[0].d, seg_end: seg[seg.length - 1].d,
      ids: u.ids, st: u.status, ev: u.event, flags: segFlags, ...encodeSegment(seg),
    };
  });
  return { lines, clipped };
}

// ── manifest ─────────────────────────────────────────────────────────────────
export interface Manifest {
  version: 1;
  build_id: string;
  status: "building" | "complete";
  started_at: string;
  completed_at: string | null;
  as_of: string;
  source: "alpaca";
  feed: "sip";
  adjustment: "all";
  start: string;
  asof_modes: { default: "all symbols"; raw: "reuse_suspect symbols (asof=-)" };
  universe: {
    total: number; active: number; inactive: number; corp_action_only: number; excluded_otc: number;
    reuse_suspect: number; with_event: number; corp_actions_http: number | null; corp_action_events: number;
  };
  progress: { batches_total: number; next_batch: number; requests: number; retries: number; sessions: number };
  counts: {
    symbols_with_bars: number; delisted_with_bars: number; lines: number; bars: number; clipped_bars: number;
    bytes: number; earliest_bar: string | null;
    /** Symbols Alpaca refused (HTTP 400/404/422 even alone) — skipped, never retried forever. */
    skipped_symbols: number;
    /** Of skipped_symbols, how many Alpaca answered 403 even alone. */
    skipped_403?: number;
    /** Batches that only succeeded with an earlier end date (403 fallback). */
    end_fallbacks?: number;
    /** Active symbols whose default-mapping series ended > 30 days before as_of (a reused ticker mapped to the OLD company) — refetched unmapped. */
    stale_active: number;
  };
  blocked: string | null;
  last_error: string | null;
  /** Last rejected request: "HTTP 403 (recent_sip_not_permitted)" — status + whitelisted vendor-message class only. */
  last_rejection?: string | null;
  /** Last 403 specifically (the 2026-10-29 failure class), kept even when a later 422 follows. */
  last_403?: string | null;
  caveats: string[];
}

export function writeJsonAtomic(file: string, data: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(data));
  fs.renameSync(tmp, file);
}

function readJson<T>(file: string): T | null {
  try { return JSON.parse(fs.readFileSync(file, "utf8")) as T; } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[pricePanel] unreadable", file, e instanceof Error ? e.message : e);
    return null;
  }
}

// ── session ──────────────────────────────────────────────────────────────────
export interface SessionDeps {
  env: Record<string, string | undefined>;
  fetch: FetchLike;
  root: string;
  now?: () => number;
  sleep?: (ms: number) => Promise<void>;
  freeBytes?: (dir: string) => number | null;
  inWindow?: (ms: number) => boolean;
  tradingBase?: string;
  dataBase?: string;
  minIntervalMs?: number;
  batchSymbols?: number;
}

export type SessionOutcome =
  | { status: "complete" | "paused_window" | "fresh" | "disabled"; detail: string }
  | { status: "blocked"; detail: string }
  | { status: "error"; detail: string };

class StopSession extends Error {
  constructor(public outcome: SessionOutcome) { super(outcome.detail); }
}

/** A request Alpaca rejects as malformed (400/404/422): retrying cannot help. */
class Rejected extends Error {
  constructor(public http: number, public cls: string | null) { super(`HTTP ${http}${cls ? ` (${cls})` : ""}`); }
}

export function readFreeBytes(dir: string): number | null {
  try {
    const s = fs.statfsSync(dir);
    return Number(s.bavail) * Number(s.bsize);
  } catch (e) {
    console.error("[pricePanel] statfs failed:", e instanceof Error ? e.message : e);
    return null;
  }
}

function dirBytes(dir: string): number {
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) n += fs.statSync(path.join(dir, f)).size;
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.error("[pricePanel] dirBytes:", e instanceof Error ? e.message : e);
  }
  return n;
}

/** One paced GET; 429 / 5xx / network retried; anything else stops the session. */
function makeGetter(d: Required<Pick<SessionDeps, "fetch" | "now" | "sleep" | "inWindow">> & { minIntervalMs: number }, headers: Record<string, string>, m: Manifest) {
  let lastStart = 0;
  return async (url: string): Promise<unknown> => {
    let n429 = 0, nOther = 0;
    for (;;) {
      if (!d.inWindow(d.now())) throw new StopSession({ status: "paused_window", detail: "build window closed (resumes 20:00 ET)" });
      const wait = lastStart + d.minIntervalMs - d.now();
      if (wait > 0) await d.sleep(wait);
      lastStart = d.now();
      m.progress.requests++;
      let status = -1; // -1 = network error, timeout, or an unreadable body: retryable
      let body: unknown = null;
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), CALL_TIMEOUT_MS);
      try {
        const r = await d.fetch(url, { method: "GET", headers, signal: ctl.signal });
        body = await r.json();
        status = r.status;
      } catch (e) {
        status = -1;
        body = e instanceof Error ? e.message : null;
      } finally {
        clearTimeout(timer);
      }
      if (status === 200) return body;
      // 403 included (2026-10-29 live): a batch Alpaca refuses must reach the
      // end-date fallback and the bisect, not end the night.
      if (status === 400 || status === 403 || status === 404 || status === 422) throw new Rejected(status, vendorMsgClass(body));
      if (status === 429) {
        if (++n429 > MAX_429_IN_A_ROW) throw new StopSession({ status: "error", detail: `rate limited ${n429}x in a row` });
        m.progress.retries++;
        await d.sleep(BACKOFF_429_MS);
        continue;
      }
      if (status === -1 || status >= 500) {
        if (++nOther > MAX_RETRIES_OTHER) throw new StopSession({ status: "error", detail: `HTTP ${status === -1 ? "network/timeout" : status} after ${MAX_RETRIES_OTHER} retries` });
        m.progress.retries++;
        await d.sleep(RETRY_OTHER_MS);
        continue;
      }
      const cls = vendorMsgClass(body);
      throw new StopSession({ status: "error", detail: `HTTP ${status}${cls ? ` (${cls})` : ""}` });
    }
  };
}

type Getter = (url: string) => Promise<unknown>;

/** Multi-symbol daily bars, all pages, into per-symbol Row lists. */
export async function fetchBatchBars(get: Getter, dataBase: string, symbols: string[], end: string, adjustment: "all" | "raw", unmapped: boolean): Promise<Map<string, Array<{ d: string; o: number; h: number; l: number; c: number; v: number }>>> {
  const out = new Map<string, Array<{ d: string; o: number; h: number; l: number; c: number; v: number }>>();
  let token: string | null = null;
  do {
    const q = new URLSearchParams({ symbols: symbols.join(","), timeframe: "1Day", adjustment, start: PANEL_START, end, limit: "10000", feed: "sip" });
    if (unmapped) q.set("asof", "-");
    if (token) q.set("page_token", token);
    const body = await get(`${dataBase}/v2/stocks/bars?${q.toString()}`);
    const bars = isObj(body) && isObj(body.bars) ? body.bars : {};
    for (const [sym, arr] of Object.entries(bars)) {
      if (!Array.isArray(arr)) continue;
      const list = out.get(sym) ?? [];
      for (const b of arr) {
        if (!isObj(b)) continue;
        const d = str(b.t).slice(0, 10);
        const nums = [b.o, b.h, b.l, b.c, b.v].map((x) => (typeof x === "number" && Number.isFinite(x) ? x : NaN));
        if (!YMD_RE.test(d) || nums.some((x) => Number.isNaN(x))) continue;
        list.push({ d, o: nums[0], h: nums[1], l: nums[2], c: nums[3], v: nums[4] });
      }
      out.set(sym, list);
    }
    token = isObj(body) && typeof body.next_page_token === "string" && body.next_page_token ? body.next_page_token : null;
  } while (token);
  return out;
}

const ymd = (ms: number) => new Date(ms).toISOString().slice(0, 10);

type BarMap = Map<string, Array<{ d: string; o: number; h: number; l: number; c: number; v: number }>>;

/** Per-batch fallback bookkeeping, folded into the manifest only when the batch commits. */
interface BatchStats { skipped: number; skipped403: number; endFallbacks: number; rejected403Leaves: number; lastRejection: string | null; last403: string | null }
interface FetchCtx {
  get: Getter; dataBase: string; unmapped: boolean; stats: BatchStats;
  /** Called when an earlier end date succeeded (lets a not-yet-started build lower its as_of). */
  onEndFallback: (end: string) => void;
}

function note(stats: BatchStats, e: Rejected): void {
  stats.lastRejection = e.message;
  if (e.http === 403) stats.last403 = e.message;
}

async function fetchPair(ctx: FetchCtx, symbols: string[], end: string): Promise<{ adj: BarMap; rawc: BarMap }> {
  return {
    adj: await fetchBatchBars(ctx.get, ctx.dataBase, symbols, end, "all", ctx.unmapped),
    rawc: await fetchBatchBars(ctx.get, ctx.dataBase, symbols, end, "raw", ctx.unmapped),
  };
}

/** Adjusted + raw bars for a batch. A rejected batch (400/403/404/422) is
 *  handled, never fatal for the night:
 *   1. 403 only: retry with an end date 1..END_FALLBACK_DAYS earlier (the free
 *      plan refuses "recent SIP data": first live night, 2026-10-29);
 *   2. bisect the batch, so one bad ticker costs ~2·log2(n) requests;
 *   3. a symbol rejected alone is skipped and counted.
 *  The caller stops the session when EVERY symbol of a batch was 403'd alone
 *  (systemic: the plan/keys, not the tickers). */
async function fetchWithFallback(ctx: FetchCtx, symbols: string[], end: string, allowEndFallback = true): Promise<{ adj: BarMap; rawc: BarMap }> {
  let rejected: Rejected;
  try {
    return await fetchPair(ctx, symbols, end);
  } catch (e) {
    if (!(e instanceof Rejected)) throw e;
    rejected = e;
    note(ctx.stats, e);
  }
  if (rejected.http === 403 && allowEndFallback) {
    for (let k = 1; k <= END_FALLBACK_DAYS; k++) {
      const earlier = ymd(Date.parse(`${end}T00:00:00Z`) - k * DAY_MS);
      try {
        const r = await fetchPair(ctx, symbols, earlier);
        ctx.stats.endFallbacks++;
        ctx.onEndFallback(earlier);
        return r;
      } catch (e) {
        if (!(e instanceof Rejected)) throw e;
        note(ctx.stats, e);
        if (e.http !== 403) {
          // The earlier end got past the 403; what is left is a bad ticker
          // (e.g. 422): bisect at the end date that works.
          ctx.stats.endFallbacks++;
          ctx.onEndFallback(earlier);
          end = earlier;
          rejected = e;
          break;
        }
      }
    }
  }
  if (symbols.length === 1) {
    ctx.stats.skipped++;
    if (rejected.http === 403) { ctx.stats.skipped403++; ctx.stats.rejected403Leaves++; }
    return { adj: new Map(), rawc: new Map() };
  }
  const mid = Math.ceil(symbols.length / 2);
  const adj: BarMap = new Map(), rawc: BarMap = new Map();
  for (const part of [symbols.slice(0, mid), symbols.slice(mid)]) {
    const r = await fetchWithFallback(ctx, part, end, false);
    for (const [k, v] of r.adj) adj.set(k, v);
    for (const [k, v] of r.rawc) rawc.set(k, v);
  }
  return { adj, rawc };
}

/**
 * Run one build session: continue the in-progress build (or start one when
 * none is current/fresh), until done, the window closes, the disk guard
 * trips, or the API fails. Every batch is checkpointed.
 */
export async function runPanelSession(deps: SessionDeps): Promise<SessionOutcome> {
  if (!pricePanelEnabled(deps.env)) return { status: "disabled", detail: "VOLTRADE_PRICE_PANEL off or Alpaca keys unset" };
  const d = {
    fetch: deps.fetch, now: deps.now ?? Date.now,
    sleep: deps.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms))),
    inWindow: deps.inWindow ?? inBuildWindow, minIntervalMs: deps.minIntervalMs ?? MIN_INTERVAL_MS,
  };
  const freeBytes = deps.freeBytes ?? readFreeBytes;
  const minFree = Number(deps.env.VOLTRADE_PRICE_PANEL_MIN_FREE_BYTES) || PRICE_PANEL_MIN_FREE_BYTES;
  const dataBase = deps.dataBase || "https://data.alpaca.markets";
  const tradingBase = deps.tradingBase || "https://paper-api.alpaca.markets";
  const batchSymbols = deps.batchSymbols ?? BATCH_SYMBOLS;
  const root = deps.root;
  fs.mkdirSync(root, { recursive: true });

  const current = readJson<{ build_id: string; completed_at: string }>(path.join(root, "current.json"));
  let building = readJson<{ build_id: string }>(path.join(root, "building.json"));
  const nowMs = d.now();
  if (!building && current && nowMs - Date.parse(current.completed_at) < PRICE_PANEL_REFRESH_DAYS * DAY_MS) {
    return { status: "fresh", detail: `build ${current.build_id} complete ${current.completed_at}` };
  }
  const free0 = freeBytes(root);
  if (free0 !== null && free0 < minFree) {
    return { status: "blocked", detail: `disk: free ${Math.round(free0 / 1_048_576)} MiB < floor ${Math.round(minFree / 1_048_576)} MiB` };
  }

  const headers = { "APCA-API-KEY-ID": deps.env.ALPACA_KEY || "", "APCA-API-SECRET-KEY": deps.env.ALPACA_SECRET || "" };
  let dir: string;
  let m: Manifest;
  if (building && readJson<Manifest>(path.join(root, building.build_id, "manifest.json"))) {
    dir = path.join(root, building.build_id);
    m = readJson<Manifest>(path.join(dir, "manifest.json"))!;
  } else {
    const buildId = `b${ymd(nowMs).replace(/-/g, "")}`;
    dir = path.join(root, buildId);
    m = {
      version: 1, build_id: buildId, status: "building", started_at: new Date(nowMs).toISOString(), completed_at: null,
      as_of: panelAsOf(nowMs), source: "alpaca", feed: "sip", adjustment: "all", start: PANEL_START,
      asof_modes: { default: "all symbols", raw: "reuse_suspect symbols (asof=-)" },
      universe: { total: 0, active: 0, inactive: 0, corp_action_only: 0, excluded_otc: 0, reuse_suspect: 0, with_event: 0, corp_actions_http: null, corp_action_events: 0 },
      progress: { batches_total: 0, next_batch: 0, requests: 0, retries: 0, sessions: 0 },
      counts: { symbols_with_bars: 0, delisted_with_bars: 0, lines: 0, bars: 0, clipped_bars: 0, bytes: 0, earliest_bar: null, skipped_symbols: 0, skipped_403: 0, end_fallbacks: 0, stale_active: 0 },
      blocked: null, last_error: null, caveats: CAVEATS,
    };
    building = { build_id: buildId };
  }
  m.progress.sessions++;
  m.blocked = null;
  // Repair a build started with an as_of the free plan refuses (b20261029 was
  // created 20:59 ET with UTC-yesterday = the ET date itself). Safe only while
  // no batch has been written: a started build keeps one as_of throughout.
  const safeAsOf = panelAsOf(nowMs);
  if (m.progress.next_batch === 0 && m.as_of > safeAsOf) m.as_of = safeAsOf;
  const save = () => writeJsonAtomic(path.join(dir, "manifest.json"), m);
  const get = makeGetter(d, headers, m);

  try {
    // Phase 1: universe (once per build).
    const uniFile = path.join(dir, "universe.jsonl");
    if (!fs.existsSync(uniFile)) {
      const assets = [
        ...parseAssets(await get(`${tradingBase}/v2/assets?status=active&asset_class=us_equity`), "active"),
        ...parseAssets(await get(`${tradingBase}/v2/assets?status=inactive&asset_class=us_equity`), "inactive"),
      ];
      const events: CorpEvent[] = [];
      let caHttp: number | null = 200;
      const lastYear = Number(m.as_of.slice(0, 4));
      try {
        for (let y = Number(PANEL_START.slice(0, 4)); y <= lastYear; y++) {
          let token: string | null = null;
          do {
            const q = new URLSearchParams({ types: CORP_TYPES, start: `${y}-01-01`, end: `${y}-12-31`, limit: "1000" });
            if (token) q.set("page_token", token);
            const p = parseCorporateActions(await get(`${dataBase}/v1beta1/corporate-actions?${q.toString()}`));
            events.push(...p.events);
            token = p.next;
          } while (token);
        }
      } catch (e) {
        // Corporate actions SUPPLEMENT the universe; their failure must not stop the
        // build. A closed window still pauses the session (rethrown).
        if (e instanceof StopSession && e.outcome.status === "paused_window") throw e;
        caHttp = null;
        m.last_error = `corporate actions unavailable: ${e instanceof Error ? e.message : String(e)}`.slice(0, 200);
      }
      const { entries, excluded } = buildUniverse(assets, events);
      fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(`${uniFile}.tmp`, entries.map((e) => JSON.stringify(e)).join("\n") + "\n");
      fs.renameSync(`${uniFile}.tmp`, uniFile);
      m.universe = {
        total: entries.length,
        active: entries.filter((e) => e.status === "active").length,
        inactive: entries.filter((e) => e.status === "inactive").length,
        corp_action_only: entries.filter((e) => e.status === "corp_action").length,
        excluded_otc: excluded,
        reuse_suspect: entries.filter((e) => e.reuse_suspect).length,
        with_event: entries.filter((e) => e.event).length,
        corp_actions_http: caHttp,
        corp_action_events: events.length,
      };
      m.progress.batches_total = Math.ceil(entries.length / batchSymbols);
      writeJsonAtomic(path.join(root, "building.json"), building);
      save();
    }
    const universe = fs.readFileSync(uniFile, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l) as UniverseEntry);

    // Phase 2: bars, one checkpointed batch at a time.
    fs.mkdirSync(path.join(dir, "bars"), { recursive: true });
    while (m.progress.next_batch < m.progress.batches_total) {
      const free = freeBytes(root);
      if (free !== null && free < minFree) throw new StopSession({ status: "blocked", detail: `disk: free ${Math.round(free / 1_048_576)} MiB < floor ${Math.round(minFree / 1_048_576)} MiB` });
      if (m.counts.bytes > PRICE_PANEL_MAX_BYTES) throw new StopSession({ status: "blocked", detail: `panel size cap ${Math.round(PRICE_PANEL_MAX_BYTES / 1_048_576)} MiB reached` });
      const b = m.progress.next_batch;
      const batch = universe.slice(b * batchSymbols, (b + 1) * batchSymbols);
      const lines: PanelLine[] = [];
      let clippedBars = 0, withBars = 0, delistedWithBars = 0;
      const end = m.as_of < panelAsOf(d.now()) ? m.as_of : panelAsOf(d.now());
      const stats: BatchStats = { skipped: 0, skipped403: 0, endFallbacks: 0, rejected403Leaves: 0, lastRejection: null, last403: null };
      const ctxFor = (unmapped: boolean): FetchCtx => ({
        get, dataBase, unmapped, stats,
        onEndFallback: (earlier) => { if (m.progress.next_batch === 0 && m.counts.lines === 0 && earlier < m.as_of) m.as_of = earlier; },
      });
      const staleBefore = ymd(Date.parse(`${m.as_of}T00:00:00Z`) - SEGMENT_GAP_DAYS * DAY_MS);
      const runMode = async (mode: "default" | "raw", entries: UniverseEntry[]): Promise<Set<string>> => {
        const stale = new Set<string>();
        if (!entries.length) return stale;
        const { adj, rawc } = await fetchWithFallback(ctxFor(mode === "raw"), entries.map((u) => u.symbol), m.as_of < end ? m.as_of : end);
        for (const u of entries) {
          const a = adj.get(u.symbol);
          if (!a || !a.length) continue;
          const rc = new Map((rawc.get(u.symbol) ?? []).map((r) => [r.d, r.c]));
          const rows: Row[] = a.map((r) => ({ ...r, rc: rc.get(r.d) ?? null }));
          const res = linesFor(u, mode, rows);
          if (!res.lines.length) continue;
          if (mode === "default" && u.status === "active" && !u.reuse_suspect && rows[rows.length - 1].d < staleBefore) {
            stale.add(u.symbol);
            for (const l of res.lines) l.flags.push("stale_active");
          }
          lines.push(...res.lines);
          clippedBars += res.clipped;
          if (mode === "default") {
            withBars++;
            if (u.status !== "active") delistedWithBars++;
          }
          const first = res.lines[0].seg_start;
          if (!m.counts.earliest_bar || first < m.counts.earliest_bar) m.counts.earliest_bar = first;
        }
        return stale;
      };
      const stale = await runMode("default", batch);
      if (stats.lastRejection) m.last_rejection = stats.lastRejection;
      if (stats.last403) m.last_403 = stats.last403;
      if (batch.length > 1 && stats.rejected403Leaves === batch.length) {
        // Every symbol refused even alone, after the end-date fallback: the
        // plan/keys, not the tickers. Stop (nothing committed for this batch)
        // rather than skip the whole universe 50 symbols at a time.
        throw new StopSession({ status: "error", detail: `systemic ${stats.lastRejection ?? "HTTP 403"}: every symbol of batch ${b} refused` });
      }
      // Unmapped (asof=-) pass: reused tickers, plus active names the default
      // mapping tied to an older company (S -> Sprint, live 2026-10-28).
      await runMode("raw", batch.filter((u) => u.reuse_suspect || stale.has(u.symbol)));
      const shard = path.join(dir, "bars", `b${String(b).padStart(5, "0")}.jsonl.gz`);
      if (lines.length) {
        const buf = await gzip(Buffer.from(lines.map((l) => JSON.stringify(l)).join("\n") + "\n"));
        await fs.promises.writeFile(`${shard}.tmp`, buf);
        await fs.promises.rename(`${shard}.tmp`, shard);
        m.counts.bytes += buf.length;
      }
      m.counts.lines += lines.length;
      m.counts.bars += lines.reduce((s, l) => s + l.n, 0);
      m.counts.clipped_bars += clippedBars;
      m.counts.symbols_with_bars += withBars;
      m.counts.delisted_with_bars += delistedWithBars;
      m.counts.stale_active += stale.size;
      m.counts.skipped_symbols += stats.skipped;
      m.counts.skipped_403 = (m.counts.skipped_403 ?? 0) + stats.skipped403;
      m.counts.end_fallbacks = (m.counts.end_fallbacks ?? 0) + stats.endFallbacks;
      if (stats.lastRejection) m.last_rejection = stats.lastRejection;
      if (stats.last403) m.last_403 = stats.last403;
      m.progress.next_batch = b + 1;
      save();
    }

    // Complete: swap the pointer, drop the previous build.
    m.status = "complete";
    m.completed_at = new Date(d.now()).toISOString();
    save();
    const prev = current?.build_id;
    writeJsonAtomic(path.join(root, "current.json"), { build_id: m.build_id, completed_at: m.completed_at });
    fs.rmSync(path.join(root, "building.json"), { force: true });
    if (prev && prev !== m.build_id) fs.rmSync(path.join(root, prev), { recursive: true, force: true });
    return { status: "complete", detail: `${m.counts.symbols_with_bars} symbols (${m.counts.delisted_with_bars} not active), ${m.counts.bars} bars, ${Math.round(m.counts.bytes / 1_048_576)} MiB` };
  } catch (e) {
    const outcome: SessionOutcome = e instanceof StopSession ? e.outcome : { status: "error", detail: e instanceof Error ? e.message : String(e) };
    if (outcome.status === "blocked") m.blocked = outcome.detail.slice(0, 200);
    if (outcome.status === "error") m.last_error = outcome.detail.slice(0, 200);
    if (fs.existsSync(dir)) save();
    return { ...outcome, detail: `${outcome.detail} (batch ${m.progress.next_batch}/${m.progress.batches_total})` };
  }
}

// ── status (diag probe) ─────────────────────────────────────────────────────
/** Read-only status for the "price_panel" diag probe: manifests only, never bars. */
export function pricePanelStatus(env: Record<string, string | undefined> = process.env, root = pricePanelRoot(env)): Record<string, unknown> {
  const current = readJson<{ build_id: string; completed_at: string }>(path.join(root, "current.json"));
  const building = readJson<{ build_id: string }>(path.join(root, "building.json"));
  const pick = (id: string | undefined) => {
    if (!id || !/^b\d{8}$/.test(id)) return null;
    const m = readJson<Manifest>(path.join(root, id, "manifest.json"));
    if (!m) return null;
    return {
      build_id: m.build_id, status: m.status, started_at: m.started_at, completed_at: m.completed_at, as_of: m.as_of,
      source: m.source, feed: m.feed, adjustment: m.adjustment, start: m.start, universe: m.universe,
      progress: m.progress, counts: m.counts, blocked: m.blocked, last_error: m.last_error, last_rejection: m.last_rejection ?? null, last_403: m.last_403 ?? null,
      bytes_on_disk: dirBytes(path.join(root, id, "bars")),
    };
  };
  const flag = pricePanelFlag(env);
  return {
    probe: "price_panel",
    flag: { value: flag.value, source: flag.source, on: flag.on },
    keys: !!(env.ALPACA_KEY && env.ALPACA_SECRET),
    current: pick(current?.build_id),
    building: pick(building?.build_id),
    free_bytes: readFreeBytes(fs.existsSync(root) ? root : path.dirname(root)),
    min_free_bytes: Number(env.VOLTRADE_PRICE_PANEL_MIN_FREE_BYTES) || PRICE_PANEL_MIN_FREE_BYTES,
    caveats: CAVEATS,
  };
}

// ── scheduler ────────────────────────────────────────────────────────────────
let booted = false;

/** Nightly tick (every 15 min, unref'd; first tick one TICK_MS after boot —
 *  a boot never starts a build by itself). One session at a time; a session
 *  works only inside the 20:00-04:00 ET window. */
export function bootPricePanel(env: NodeJS.ProcessEnv = process.env): void {
  if (booted) return;
  booted = true;
  let running = false;
  let lastIdleBeatNight = "";
  let lastAttemptNight = "";
  const tick = async () => {
    if (running || !pricePanelEnabled(env)) return;
    const now = Date.now();
    if (!inBuildWindow(now)) return;
    // A night runs 20:15 ET on day D to 04:00 on D+1; key it by D.
    const night = zonedParts(now - WINDOW_END_MIN * 60_000, ET).date;
    running = true;
    try {
      const root = pricePanelRoot(env);
      const pending = fs.existsSync(path.join(root, "building.json"));
      const current = readJson<{ completed_at: string }>(path.join(root, "current.json"));
      const fresh = !pending && !!current && now - Date.parse(current.completed_at) < PRICE_PANEL_REFRESH_DAYS * DAY_MS;
      if (fresh) {
        // One idle beat per night keeps the daily heartbeat green.
        if (lastIdleBeatNight !== night) {
          beat("price_panel_build", "finish", "skipped", `complete ${current!.completed_at}; next refresh after ${PRICE_PANEL_REFRESH_DAYS}d`);
          lastIdleBeatNight = night;
        }
        return;
      }
      // Blocked / errored / paused: one attempt per night, never a 15-minute retry loop.
      if (lastAttemptNight === night) return;
      lastAttemptNight = night;
      beat("price_panel_build", "start");
      const out = await runPanelSession({ env, fetch: fetch as unknown as FetchLike, root });
      if (out.status === "error") beat("price_panel_build", "finish", "error", out.detail);
      else beat("price_panel_build", "finish", out.status === "complete" ? "ok" : "skipped", `${out.status}: ${out.detail}`);
      if (out.status === "complete") { lastIdleBeatNight = night; lastAttemptNight = ""; }
    } catch (e) {
      beat("price_panel_build", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  setInterval(() => { void tick(); }, TICK_MS).unref?.();
}
