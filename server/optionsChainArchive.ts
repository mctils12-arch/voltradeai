/**
 * optionsChainArchive.ts — free daily options-chain snapshots (approved
 * options-data decision 2026-07-04: "start the free Alpaca chain archive
 * now regardless" — forward-only history that can never be back-bought).
 *
 * FEED HONESTY: Alpaca paper accounts serve feed=indicative — quotes are
 * INDICATIVE, not consolidated NBBO. Every record and the manifest carry
 * that label; the Databento pull (real consolidated BBO) is the
 * ground-truth complement for 2013→present.
 *
 * VOLUME CONTROL (position-archive lesson — budget stated up front):
 * ≤120 underlyings × (exp ≤60 days, strikes ±20% of spot) ≈ 100–250
 * contracts each ≈ ≤30k compact records/day ≈ 3–5MB/day raw JSONL,
 * gzipped by the existing compressOldHours/rollup machinery. One snapshot
 * per trading day, taken after the close.
 *
 * DAY LABEL + IDEMPOTENCY ([REPAIR 2026-10-27], audit §2.4 defect 1): the
 * ET clock used `hour12:false`, which Node 20 (the production image)
 * renders as hour "24" at midnight — 00:40 ET parsed as minute 1480 ≥
 * 16:15, so the run fired just after midnight and labeled the PREVIOUS
 * session's closing quotes with the new day (file D held D-1's close on
 * every sampled prod day 2026-10-20..27). Now: the clock is `hourCycle:
 * "h23"` with a `% 24` belt, the run is confined to a post-close window on
 * an NYSE trading day (16:30–18:30 ET), the file label is that ET trading
 * date (never the UTC date), and a day is written through a staging file
 * plus a per-day `_captured/<day>.json` sidecar, so a restart mid-capture
 * never appends a second copy of the day.
 *
 * Pure helpers exported for tests; network + scheduling at the bottom,
 * FIRMS-poller pattern (self-scheduling, unref'd, key-gated).
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";
import { beat } from "./jobHeartbeat";
import { resolveAlpacaFeed, alpacaErrorBody } from "./alpacaFeed";

export const CHAIN_MAX_UNDERLYINGS = 120;
export const CHAIN_MAX_DTE_DAYS = 60;
export const CHAIN_STRIKE_BAND = 0.20; // ±20% of spot
/** Capture window on a trading day, in ET minutes. 16:30, not 16:00: SPY/
 *  QQQ/IWM options trade until 16:15 ET and the indicative feed's trades
 *  lag ~15 minutes, so 16:30 is the first moment the session is final for
 *  every pinned underlying (half days close at 13:00, so the window is
 *  post-close there too). 18:30 ET is still before 00:00 UTC in both EDT
 *  (22:30Z) and EST (23:30Z), so a capture never straddles the UTC-day
 *  boundary that the derived readers' write-once caches (chainIv /
 *  chainCspEntry) key on, and a restart inside the window retries the SAME
 *  session's closing quotes. A day the process misses entirely is lost —
 *  never captured late under the wrong label. */
export const CHAIN_RUN_AFTER_ET_MINUTES = 16 * 60 + 30; // 16:30 ET
export const CHAIN_RUN_BEFORE_ET_MINUTES = 18 * 60 + 30; // 18:30 ET (exclusive)
/** Scheduler tick: a stat per tick, short enough to land inside the window. */
export const CHAIN_TICK_MS = 5 * 60_000;

export function chainEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.ALPACA_KEY && env.ALPACA_SECRET);
}

/** The index chains every day, ahead of everything else ([PIPELINE]
 *  2026-10-20): 76 archived days held NO SPY / QQQ / IWM rows because the
 *  CSP cache appends its anchors AFTER the ranked candidates, beyond the cap
 *  — research test 2's index control (the variance risk premium of the
 *  index vs single names) was unmeasurable. Three of 120 slots. */
export const CHAIN_ALWAYS: readonly string[] = ["SPY", "QQQ", "IWM"];

/** Underlyings for today's snapshot: the pinned index names, then any open
 *  position underlyings (spot unknown → band check skipped for them), then
 *  the CSP-universe cache tuples [symbol, price, volume, dollarVol] (spot
 *  rides along free). */
export function underlyingsFor(
  cacheJson: any, positionSymbols: string[] = [], cap = CHAIN_MAX_UNDERLYINGS,
): Array<{ sym: string; spot: number | null }> {
  const out = new Map<string, number | null>();
  for (const s of CHAIN_ALWAYS) out.set(s, null);
  for (const s of positionSymbols) {
    const sym = String(s || "").toUpperCase();
    if (sym && sym.length <= 5) out.set(sym, null);
  }
  const cands = Array.isArray(cacheJson?.candidates) ? cacheJson.candidates : [];
  for (const c of cands) {
    const sym = String(c?.[0] || "").toUpperCase();
    if (!sym) continue;
    const spot = Number(c?.[1]);
    const known = Number.isFinite(spot) && spot > 0 ? spot : null;
    // a pinned or position name later in the cache still takes the cache's spot
    // (the strike band needs it); the cap only limits NEW names, so the scan
    // never stops early
    if (out.has(sym)) { if (out.get(sym) === null && known !== null) out.set(sym, known); continue; }
    if (out.size < cap) out.set(sym, known);
  }
  return [...out.entries()].slice(0, cap).map(([sym, spot]) => ({ sym, spot }));
}

/** Parse an OCC option symbol: AAPL261218C00200000 →
 *  {exp: Date(UTC), type, strike}. Null on anything malformed. */
export function parseOcc(occ: string): { exp: number; type: "C" | "P"; strike: number } | null {
  const m = /^[A-Z]{1,6}(\d{2})(\d{2})(\d{2})([CP])(\d{8})$/.exec(String(occ || "").replace(/\s+/g, ""));
  if (!m) return null;
  const [, yy, mm, dd, cp, k] = m;
  const exp = Date.UTC(2000 + Number(yy), Number(mm) - 1, Number(dd));
  const strike = Number(k) / 1000;
  if (!Number.isFinite(exp) || !Number.isFinite(strike) || strike <= 0) return null;
  return { exp, type: cp as "C" | "P", strike };
}

/** Contract filter: expiry within CHAIN_MAX_DTE_DAYS, strike within the
 *  band when spot is known (unknown spot → keep; better complete than
 *  silently narrow for position underlyings). */
export function withinFilters(occ: string, spot: number | null, nowMs: number): boolean {
  const p = parseOcc(occ);
  if (!p) return false;
  const dte = (p.exp - nowMs) / 86_400_000;
  if (dte < -1 || dte > CHAIN_MAX_DTE_DAYS) return false;
  if (spot != null && spot > 0) {
    const lo = spot * (1 - CHAIN_STRIKE_BAND), hi = spot * (1 + CHAIN_STRIKE_BAND);
    if (p.strike < lo || p.strike > hi) return false;
  }
  return true;
}

export interface ChainRecord {
  s: string;            // OCC symbol
  u: string;            // underlying
  t: number;            // snapshot epoch ms
  bid: number | null;
  ask: number | null;
  bsz: number | null;
  asz: number | null;
  iv: number | null;
  delta: number | null;
  /** Open interest. NOT AVAILABLE on this feed: the Alpaca indicative
   *  snapshot carries no openInterest (0 of 5,000 sampled prod rows, audit
   *  2026-10-27 §2.4 defect 3). Kept for schema continuity; always null in
   *  practice. Real OI needs OCC/OPRA-grade data (wishlist). */
  oi: number | null;
  // ── schema v2 (2026-10-27, audit §10.2: GEX / P-C volume inputs) ──
  /** greeks.gamma (Alpaca-computed from the indicative quote, not an
   *  exchange value). Absent on older rows. */
  gamma?: number | null;
  /** Contract volume for the labelled session = dailyBar.v when dailyBar is
   *  dated that ET session; 0 when the latest dailyBar is an EARLIER session
   *  (no prints today on this feed); null when no dailyBar at all. Indicative
   *  trades lag ~15 min, hence the 16:30 ET capture. Absent on older rows. */
  vol?: number | null;
  /** Underlying latest trade price at capture time (stock snapshot,
   *  ALPACA_DATA_FEED / default delayed_sip = 15-min delayed consolidated
   *  tape; may be an extended-hours print, NOT the official close). null
   *  when the stock snapshot call failed. Absent on older rows. */
  spot?: number | null;
  feed: "indicative";   // NEVER dropped — this is not NBBO
}

/** Session volume from an option snapshot's dailyBar (see ChainRecord.vol). */
export function sessionVolume(dailyBar: unknown, nowMs: number): number | null {
  if (!dailyBar || typeof dailyBar !== "object") return null;
  const bar = dailyBar as { v?: unknown; t?: unknown };
  const v = Number(bar.v);
  if (!Number.isFinite(v) || v < 0) return null;
  const barMs = Date.parse(String(bar.t ?? ""));
  if (!Number.isFinite(barMs)) return null; // undated bar: cannot attribute to a session
  const barDay = etClock(barMs).etDay, today = etClock(nowMs).etDay;
  if (barDay === today) return v;
  return barDay < today ? 0 : null;
}

/** Compact one Alpaca snapshot entry. Null when there is no quote at all
 *  (a chain row with no market is noise, not history). */
export function compactContract(occ: string, u: string, snap: any, nowMs: number, spot: number | null = null): ChainRecord | null {
  const q = snap?.latestQuote;
  const bid = Number(q?.bp), ask = Number(q?.ap);
  if (!q || (!Number.isFinite(bid) && !Number.isFinite(ask))) return null;
  const g = snap?.greeks;
  const n = (v: any) => (v !== null && Number.isFinite(Number(v)) ? Number(v) : null);
  return {
    s: occ, u, t: nowMs,
    bid: n(q?.bp), ask: n(q?.ap), bsz: n(q?.bs), asz: n(q?.as),
    iv: n(snap?.impliedVolatility), delta: n(g?.delta), oi: n(snap?.openInterest),
    gamma: n(g?.gamma), vol: sessionVolume(snap?.dailyBar, nowMs),
    spot: spot != null && Number.isFinite(spot) && spot > 0 ? spot : null,
    feed: "indicative",
  };
}

export function chainDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "optionchains");
}

/** The published day-file for an ET trading date. */
export function chainDayFile(day: string, baseDir?: string): string {
  return path.join(chainDir(baseDir), `${day}.jsonl`);
}
/** In-progress capture. Lives under a "_" subdir so no reader
 *  (listChainDays' `^YYYY-MM-DD.jsonl` regex, archiveDayFiles'
 *  `startsWith(day)` fallback, archiveStats' isFile count) ever sees a
 *  half-written day. */
export function chainStagingFile(day: string, baseDir?: string): string {
  return path.join(chainDir(baseDir), "_staging", `${day}.jsonl`);
}
/** Completion sidecar: its existence is what makes a day "captured". */
export function chainCapturedFile(day: string, baseDir?: string): string {
  return path.join(chainDir(baseDir), "_captured", `${day}.json`);
}

/** Append records to a day-file (default: the published file for the ET
 *  trading date of nowMs — never the UTC date; 20:30 ET is already
 *  tomorrow in UTC). Returns records written. */
export function archiveChainRecords(
  records: ChainRecord[], baseDir?: string, nowMs = Date.now(),
  fp: string = chainDayFile(etClock(nowMs).etDay, baseDir),
): number {
  if (!records.length) return 0;
  fs.mkdirSync(path.dirname(fp), { recursive: true });
  fs.appendFileSync(fp, records.map((r) => JSON.stringify(r)).join("\n") + "\n");
  return records.length;
}

// hourCycle "h23", NOT hour12:false: on Node 20 (node:20-slim in prod)
// hour12:false resolves to an h24 clock and midnight formats as "24" (the
// audit reproduced it with npx node@20: 2026-10-21T04:46Z -> "24":"46");
// Node 22 prints "00", which is why local tests never saw it.
const ET_FMT = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York", weekday: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
});
const ET_DAY_FMT = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit",
});

/** Minutes after ET midnight from formatToParts output. `% 24` backs up
 *  hourCycle: an hour of "24" (h24 runtimes) is midnight, not minute 1440 —
 *  the exact misparse that fired the run at 00:40 ET. */
export function etMinutesFromParts(parts: ReadonlyArray<{ type: string; value: string }>): number {
  const get = (t: string) => parts.find((p) => p.type === t)?.value || "";
  const h = parseInt(get("hour"), 10), m = parseInt(get("minute"), 10);
  if (!Number.isFinite(h) || !Number.isFinite(m)) return NaN;
  return (h % 24) * 60 + m;
}

/** ET weekday ("Mon".."Sun"), minutes after ET midnight, ET calendar date. */
export function etClock(nowMs: number): { dow: string; mins: number; etDay: string } {
  const parts = ET_FMT.formatToParts(nowMs);
  const dow = parts.find((p) => p.type === "weekday")?.value || "";
  const dp = ET_DAY_FMT.formatToParts(nowMs);
  const g = (t: string) => dp.find((p) => p.type === t)?.value || "";
  return { dow, mins: etMinutesFromParts(parts), etDay: `${g("year")}-${g("month")}-${g("day")}` };
}

// ── Market holidays ([REPAIR 2026-07-05, audit defect #4c]: the archiver
// ran on July 3 — a full NYSE closure — burning ~120 API calls to archive
// stale Thursday quotes as a fresh day). Source of truth stays
// market_calendar.py (FROZEN, read-only here — parsed, never duplicated,
// so December's year-add flows through automatically). Parse failure
// degrades to weekend-only gating, never to a crash. ──────────────────────

export function parseMarketHolidays(pyText: string, setPrefix: "MARKET_HOLIDAYS" | "HALF_DAYS" = "MARKET_HOLIDAYS"): Set<string> {
  const out = new Set<string>();
  const m = pyText.match(new RegExp(`${setPrefix}_\\d{4}\\s*=\\s*\\{([\\s\\S]*?)\\}`, "g")) || [];
  for (const block of m) {
    const dates = Array.from(block.matchAll(/date\((\d{4}),\s*(\d{1,2}),\s*(\d{1,2})\)/g));
    for (const d of dates) {
      out.add(`${d[1]}-${d[2].padStart(2, "0")}-${d[3].padStart(2, "0")}`);
    }
  }
  return out;
}

let holidayCache: Set<string> | null = null;
export function marketHolidays(): Set<string> {
  if (holidayCache) return holidayCache;
  try {
    holidayCache = parseMarketHolidays(fs.readFileSync(path.resolve(process.cwd(), "market_calendar.py"), "utf8"));
  } catch {
    holidayCache = new Set(); // degrade to weekend-only gating
  }
  return holidayCache;
}

/** 1:00 PM ET early closes (HALF_DAYS_<year> in market_calendar.py) — used by
 *  server/jobHeartbeat.ts's market-hours window. Same degrade-to-empty rule. */
let halfDayCache: Set<string> | null = null;
export function marketHalfDays(): Set<string> {
  if (halfDayCache) return halfDayCache;
  try {
    halfDayCache = parseMarketHolidays(fs.readFileSync(path.resolve(process.cwd(), "market_calendar.py"), "utf8"), "HALF_DAYS");
  } catch {
    halfDayCache = new Set();
  }
  return halfDayCache;
}

/** Once per ET trading day, inside the post-close window. lastRunDay is the
 *  ET trading date already captured (the caller derives it from the
 *  `_captured/` sidecar). The returned etDay is the capture's label. */
export function shouldRunNow(lastRunDay: string | null, nowMs = Date.now(), holidays: Set<string> = marketHolidays()): { run: boolean; etDay: string } {
  const { dow, mins, etDay } = etClock(nowMs);
  const run = dow !== "Sat" && dow !== "Sun" && !holidays.has(etDay) &&
    mins >= CHAIN_RUN_AFTER_ET_MINUTES && mins < CHAIN_RUN_BEFORE_ET_MINUTES &&
    lastRunDay !== etDay;
  return { run, etDay };
}

/** Is a record timestamp a valid post-close capture FOR `day`? (Pre-fix
 *  files were written ~00:40 ET and carry the previous session.) */
export function isPostCloseCaptureFor(day: string, tMs: number): boolean {
  if (!Number.isFinite(tMs)) return false;
  const c = etClock(tMs);
  return c.etDay === day && c.mins >= CHAIN_RUN_AFTER_ET_MINUTES && c.mins < CHAIN_RUN_BEFORE_ET_MINUTES;
}

/** First record's `t` of a plain .jsonl day-file (null if none/unreadable). */
function firstRecordT(fp: string): number | null {
  let fd: number | null = null;
  try {
    fd = fs.openSync(fp, "r");
    const buf = Buffer.alloc(4096);
    const n = fs.readSync(fd, buf, 0, buf.length, 0);
    const line = buf.subarray(0, n).toString("utf8").split("\n")[0];
    const t = Number(JSON.parse(line)?.t);
    return Number.isFinite(t) ? t : null;
  } catch {
    return null; // unreadable/partial first line: treat as not a valid capture
  } finally {
    if (fd !== null) fs.closeSync(fd);
  }
}

/** Idempotency check for an ET trading date: captured iff the sidecar
 *  exists — or (crash between publish and sidecar) the published day-file's
 *  first record is a post-close capture for that day, in which case the
 *  sidecar is repaired. A pre-fix midnight file does NOT count. */
export function isDayCaptured(day: string, baseDir?: string): boolean {
  if (fs.existsSync(chainCapturedFile(day, baseDir))) return true;
  const fp = chainDayFile(day, baseDir);
  const t = fs.existsSync(fp) ? firstRecordT(fp) : null;
  if (t !== null && isPostCloseCaptureFor(day, t)) {
    writeCapturedSidecar(day, { day, captured_at: new Date(t).toISOString(), repaired: true }, baseDir);
    return true;
  }
  return false;
}

function writeCapturedSidecar(day: string, body: Record<string, unknown>, baseDir?: string): void {
  const fp = chainCapturedFile(day, baseDir);
  fs.mkdirSync(path.dirname(fp), { recursive: true });
  fs.writeFileSync(fp, JSON.stringify(body));
}

/** Publish a finished staging file as the day's file. A pre-existing
 *  published file for the same label can only be a pre-fix capture (it
 *  failed isDayCaptured): it holds the PREVIOUS session's close, so it is
 *  moved — never deleted — to `_superseded/` and logged. Then the staging
 *  file is renamed into place (atomic on one filesystem) and the sidecar
 *  is written last. */
export function publishStagedDay(day: string, summary: Record<string, unknown>, baseDir?: string): void {
  const staged = chainStagingFile(day, baseDir);
  const fp = chainDayFile(day, baseDir);
  for (const old of [fp, `${fp}.gz`]) {
    if (!fs.existsSync(old)) continue;
    const dest = path.join(chainDir(baseDir), "_superseded", path.basename(old));
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.renameSync(old, fs.existsSync(dest) ? `${dest}.${Date.now()}` : dest);
    console.warn(`[optionchains] ${path.basename(old)} was a pre-fix capture (not post-close for ${day}) — moved to _superseded/`);
  }
  if (fs.existsSync(staged)) fs.renameSync(staged, fp);
  writeCapturedSidecar(day, summary, baseDir);
}

// ── Full-chain pagination ([REPAIR 2026-10-27], audit §2.4 defect 2) ───────
// Cause of the truncated index chains: the request carried NO server-side
// expiry/strike bounds, pages come back in OCC-symbol order (nearest expiry
// first), and the loop stopped at 5 pages × 1000. SPY on 2026-10-26 held
// 4,151 rows across 15 expiries ending 2026-12-04 (~39 DTE) — every expiry
// past the 5,000th contract (the 2026-12-18 monthly included) was never
// fetched. Fix: ask Alpaca for exactly the archive's documented scope
// (expiry ≤ CHAIN_MAX_DTE_DAYS = 60 calendar days; strikes ±20% when spot is
// known), and follow next_page_token until it is exhausted, bounded only by
// a safety cap that is LOGGED when hit (never silently short again).
export const CHAIN_PAGE_LIMIT = 1000; // Alpaca's max page size
/** Safety cap per underlying: 40 × 1000 contracts. A ≤60-DTE SPY chain is
 *  on the order of 10-20k contracts (daily expiries), so the cap is ~2×
 *  headroom; reaching it marks the underlying truncated in the run summary. */
export const CHAIN_MAX_PAGES = 40;

const ymdPlusDays = (ymd: string, days: number): string => {
  const [y, m, d] = ymd.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d + days)).toISOString().slice(0, 10);
};

/** Server-side bounds for one underlying, matching withinFilters: expiries
 *  from the ET capture date (0DTE kept, as before) to +CHAIN_MAX_DTE_DAYS;
 *  strike band only when spot is known (rounded outward to the cent so the
 *  exact client-side band, which still runs, is the binding one). */
export function chainQueryWindow(nowMs: number, spot: number | null): {
  expGte: string; expLte: string; strikeGte?: string; strikeLte?: string;
} {
  const day = etClock(nowMs).etDay;
  const w: { expGte: string; expLte: string; strikeGte?: string; strikeLte?: string } =
    { expGte: day, expLte: ymdPlusDays(day, CHAIN_MAX_DTE_DAYS) };
  if (spot != null && spot > 0) {
    w.strikeGte = (Math.floor(spot * (1 - CHAIN_STRIKE_BAND) * 100) / 100).toFixed(2);
    w.strikeLte = (Math.ceil(spot * (1 + CHAIN_STRIKE_BAND) * 100) / 100).toFixed(2);
  }
  return w;
}

export function snapshotUrl(
  underlying: string, pageToken?: string,
  w?: { expGte: string; expLte: string; strikeGte?: string; strikeLte?: string },
): string {
  let url = `https://data.alpaca.markets/v1beta1/options/snapshots/${encodeURIComponent(underlying)}?feed=indicative&limit=${CHAIN_PAGE_LIMIT}`;
  if (w) {
    url += `&expiration_date_gte=${w.expGte}&expiration_date_lte=${w.expLte}`;
    if (w.strikeGte && w.strikeLte) url += `&strike_price_gte=${w.strikeGte}&strike_price_lte=${w.strikeLte}`;
  }
  return pageToken ? `${url}&page_token=${encodeURIComponent(pageToken)}` : url;
}

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; json(): Promise<any> }>;

/** Fetch + filter + compact one underlying's FULL in-scope chain. Follows
 *  next_page_token to exhaustion; `truncated` is true only when the safety
 *  cap stopped it with pages still outstanding (or the API repeated a
 *  token, which would otherwise loop). */
export async function fetchChainPages(
  u: string, spot: number | null, env: NodeJS.ProcessEnv = process.env,
  fetchImpl: FetchFn = fetch as any, nowMs = Date.now(), maxPages = CHAIN_MAX_PAGES,
  /** capture-time underlying price stored on every record (null = unknown;
   *  `spot` above may be the CSP cache's stale price, used only for the band) */
  recordSpot: number | null = null,
): Promise<{ records: ChainRecord[]; pages: number; truncated: boolean }> {
  const headers = { "APCA-API-KEY-ID": env.ALPACA_KEY || "", "APCA-API-SECRET-KEY": env.ALPACA_SECRET || "" };
  const w = chainQueryWindow(nowMs, spot);
  const records: ChainRecord[] = [];
  const seen = new Set<string>();
  let token: string | undefined;
  let pages = 0;
  while (pages < maxPages) {
    const r = await fetchImpl(snapshotUrl(u, token, w), { headers });
    if (!r.ok) throw new Error(`alpaca chain ${u}: HTTP ${r.status}`);
    const d = await r.json();
    pages++;
    for (const [occ, snap] of Object.entries(d?.snapshots || {})) {
      if (!withinFilters(occ, spot, nowMs)) continue;
      const rec = compactContract(occ, u, snap, nowMs, recordSpot);
      if (rec) records.push(rec);
    }
    token = d?.next_page_token || undefined;
    if (!token) return { records, pages, truncated: false };
    if (seen.has(token)) break; // API repeated a page token: stop, flag it
    seen.add(token);
  }
  console.error(`[optionchains] ${u}: chain TRUNCATED after ${pages} pages (cap ${maxPages}) — later expiries missing`);
  return { records, pages, truncated: true };
}

/** Underlying spot at capture time for every chain underlying, in ONE
 *  batched stock-snapshot call: latestTrade.p, else the latestQuote mid.
 *  Feed = ALPACA_DATA_FEED (default delayed_sip, a 15-min delayed
 *  consolidated tape; the account's SIP real-time is 403). Any failure →
 *  empty map (records carry spot:null, the band falls back to the cache
 *  spot); logged, never thrown — spot must not cost the day's chains. */
export async function fetchSpots(
  symbols: string[], env: NodeJS.ProcessEnv = process.env, fetchImpl: FetchFn = fetch as any,
): Promise<Map<string, number>> {
  const out = new Map<string, number>();
  if (!symbols.length) return out;
  const feed = resolveAlpacaFeed(env.ALPACA_DATA_FEED);
  const url = `https://data.alpaca.markets/v2/stocks/snapshots?symbols=${symbols.map(encodeURIComponent).join(",")}&feed=${encodeURIComponent(feed)}`;
  try {
    const r = await fetchImpl(url, { headers: { "APCA-API-KEY-ID": env.ALPACA_KEY || "", "APCA-API-SECRET-KEY": env.ALPACA_SECRET || "" } });
    const body = await r.json();
    const err = alpacaErrorBody(r.status, body);
    if (err) { console.error(`[optionchains] spot snapshot failed (${feed}): ${err}`); return out; }
    for (const [sym, s] of Object.entries((body || {}) as Record<string, any>)) {
      const p = Number(s?.latestTrade?.p);
      const bp = Number(s?.latestQuote?.bp), ap = Number(s?.latestQuote?.ap);
      const px = Number.isFinite(p) && p > 0 ? p
        : Number.isFinite(bp) && Number.isFinite(ap) && bp > 0 && ap >= bp ? (bp + ap) / 2 : NaN;
      if (Number.isFinite(px)) out.set(sym.toUpperCase(), px);
    }
  } catch (e) {
    console.error("[optionchains] spot snapshot failed:", e instanceof Error ? e.message : String(e));
  }
  return out;
}

/** Records only (see fetchChainPages for the truncation flag). */
export async function fetchChainForUnderlying(
  u: string, spot: number | null, env: NodeJS.ProcessEnv = process.env,
  fetchImpl: FetchFn = fetch as any, nowMs = Date.now(),
): Promise<ChainRecord[]> {
  return (await fetchChainPages(u, spot, env, fetchImpl, nowMs)).records;
}

// ── runtime state + boot (FIRMS pattern) ────────────────────────────────────
// The old `.last_run_day` marker is no longer read: the pre-fix midnight run
// wrote the CURRENT ET date into it, which would have blocked the first
// correct post-close capture. `_captured/<day>.json` replaces it.

function universeCachePath(): string {
  // storage_config.py: DATA_DIR is /data/voltrade on the volume, /tmp local
  for (const p of ["/data/voltrade/voltrade_csp_universe_cache.json", "/tmp/voltrade_csp_universe_cache.json"]) {
    if (fs.existsSync(p)) return p;
  }
  return "";
}

export interface ChainRunOpts {
  baseDir?: string;
  /** pause between underlyings (politeness on the data API); tests pass 0 */
  sleepMs?: number;
  /** CSP-universe cache JSON; default reads it from the volume */
  cacheJson?: unknown;
}

/** One capture of the chain universe into the day's STAGING file (labelled
 *  with the ET trading date of nowMs). The staging file is truncated first,
 *  so a capture interrupted by a restart is redone whole, never appended to
 *  twice. Publishing is the caller's step (captureTradingDay). */
export async function runDailyChainArchive(
  env: NodeJS.ProcessEnv = process.env, fetchImpl: FetchFn = fetch as any,
  positionSymbols: string[] = [], nowMs = Date.now(), opts: ChainRunOpts = {},
): Promise<{ day: string; underlyings: number; contracts: number; failures: number; truncated: string[]; spots: number }> {
  let cache: any = opts.cacheJson ?? null;
  if (opts.cacheJson === undefined) {
    const cp = universeCachePath();
    if (cp) { try { cache = JSON.parse(fs.readFileSync(cp, "utf8")); } catch {} }
  }
  const day = etClock(nowMs).etDay;
  const staged = chainStagingFile(day, opts.baseDir);
  fs.mkdirSync(path.dirname(staged), { recursive: true });
  fs.writeFileSync(staged, ""); // restart-safe: a leftover partial stage is discarded
  const sleepMs = opts.sleepMs ?? 350;
  const unders = underlyingsFor(cache, positionSymbols);
  let contracts = 0;
  let failures = 0;
  const truncated: string[] = [];
  // capture-time spot (stored on every record; also the strike band's
  // centre — the CSP cache's price is up to a session stale and is null for
  // the pinned index names, which is why SPY's band never applied)
  const spots = await fetchSpots(unders.map((x) => x.sym), env, fetchImpl);
  for (const { sym, spot } of unders) {
    try {
      const live = spots.get(sym) ?? null;
      const got = await fetchChainPages(sym, live ?? spot, env, fetchImpl, nowMs, CHAIN_MAX_PAGES, live);
      if (got.truncated) truncated.push(sym);
      contracts += archiveChainRecords(got.records, opts.baseDir, nowMs, staged);
    } catch (e: any) {
      failures++;
      console.error(`[optionchains] ${sym}:`, e?.message || e);
    }
    if (sleepMs > 0) await new Promise((r) => setTimeout(r, sleepMs)); // stay polite on the data API
  }
  console.log(`[optionchains] ${day}: staged ${contracts} contracts across ${unders.length} underlyings (${failures} failed, ${truncated.length} truncated${truncated.length ? `: ${truncated.join(",")}` : ""})`);
  return { day, underlyings: unders.length, contracts, failures, truncated, spots: spots.size };
}

/** Capture the ET trading date of nowMs exactly once: skip when already
 *  captured (sidecar, or a valid post-close published file), else stage,
 *  then publish + write the sidecar only if the run was not a total
 *  failure (defect #4a semantics). Returns what happened. */
export async function captureTradingDay(
  env: NodeJS.ProcessEnv = process.env, fetchImpl: FetchFn = fetch as any,
  positionSymbols: string[] = [], nowMs = Date.now(), opts: ChainRunOpts = {},
): Promise<{ status: "skipped" | "published" | "unclaimed"; day: string; underlyings: number; contracts: number; failures: number; truncated: string[]; spots: number }> {
  const day = etClock(nowMs).etDay;
  if (isDayCaptured(day, opts.baseDir)) return { status: "skipped", day, underlyings: 0, contracts: 0, failures: 0, truncated: [], spots: 0 };
  const r = await runDailyChainArchive(env, fetchImpl, positionSymbols, nowMs, opts);
  if (!shouldClaimDay(r)) {
    const staged = chainStagingFile(day, opts.baseDir);
    if (fs.existsSync(staged)) fs.unlinkSync(staged);
    console.error(`[optionchains] total failure (${r.failures}/${r.underlyings}) — ${day} NOT claimed, retrying next tick`);
    return { status: "unclaimed", ...r };
  }
  publishStagedDay(day, {
    day, captured_at: new Date(nowMs).toISOString(), label: "ET trading date",
    underlyings: r.underlyings, contracts: r.contracts, failures: r.failures, feed: "indicative",
    scope: { max_dte_days: CHAIN_MAX_DTE_DAYS, strike_band: CHAIN_STRIKE_BAND, max_pages: CHAIN_MAX_PAGES },
    truncated: r.truncated, // underlyings whose chain hit the page cap — incomplete, by name
    schema_version: 2, // + gamma, vol, spot per record (oi stays null: not in the indicative feed)
    spots_known: r.spots, spot_feed: resolveAlpacaFeed(env.ALPACA_DATA_FEED),
  }, opts.baseDir);
  return { status: "published", ...r };
}

/** [REPAIR 2026-07-05, audit defect #4a] Should the day be marked done?
 *  Claim ONLY after a run that wasn't a total failure — a crashed process
 *  or all-underlyings-failed run leaves the marker unwritten so the next
 *  hourly tick retries the SAME day instead of losing it forever (the old
 *  claim-before-run lost the trading day on any crash). An empty universe
 *  claims (nothing to archive ≠ failure); partial failures claim (the
 *  day's data is on disk, per-symbol errors were logged). */
export function shouldClaimDay(r: { underlyings: number; contracts: number; failures: number }): boolean {
  if (r.underlyings === 0) return true;
  return r.failures < r.underlyings;
}

/** [REPAIR 2026-07-05, audit defect #4b] The manifest promised .jsonl(.gz)
 *  but nothing ever gzipped this dir — ~3-5MB/day accumulated raw. Same
 *  2-day lifecycle as every other stream archive. */
export function gzipOldChainDays(baseDir?: string, nowMs?: number): number {
  const dir = chainDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      if (now - Date.parse(f.slice(0, 10)) < 2 * 86400_000) continue;
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

let booted = false;
/** 5-minute check; captures once per trading day inside the 16:30–18:30 ET
 *  window (the old hourly tick could not reliably land in a 2h window and
 *  never needed to, because the misparsed clock made "after 16:15" true
 *  all evening and again at midnight). */
export function bootChainArchive(
  env: NodeJS.ProcessEnv = process.env,
  getPositions: () => string[] = () => [],
  intervalMs = CHAIN_TICK_MS,
): void {
  if (booted || !chainEnabled(env)) return;
  booted = true;
  let running = false; // in-memory guard replaces claim-before-run (defect #4a)
  const tick = async () => {
    if (running) return;
    try {
      const now = Date.now();
      const today = etClock(now).etDay;
      const { run } = shouldRunNow(isDayCaptured(today) ? today : null, now);
      if (!run) return;
      running = true;
      beat("options_chain_archive", "start");
      // claim AFTER success: a crash/total failure leaves no sidecar, so the
      // next tick inside the window retries the same trading day
      const cap = await captureTradingDay(env, fetch as any, getPositions(), now);
      if (cap.status === "unclaimed") {
        beat("options_chain_archive", "finish", "error", `${cap.day}: total failure ${cap.failures}/${cap.underlyings} underlyings — day not claimed`);
      } else {
        beat("options_chain_archive", "finish", "ok", `${cap.day} ${cap.status}: ${cap.contracts} contracts, ${cap.underlyings} underlyings, ${cap.failures} failed`);
      }
      try { gzipOldChainDays(); } catch {}
    } catch (e: any) {
      console.error("[optionchains] daily run failed (day not claimed):", e?.message || e);
      beat("options_chain_archive", "finish", "error", `daily run failed (day not claimed): ${e?.message || e}`);
    } finally {
      running = false;
    }
  };
  tick();
  setInterval(tick, intervalMs).unref?.();
}
