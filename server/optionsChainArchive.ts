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
 * Pure helpers exported for tests; network + scheduling at the bottom,
 * FIRMS-poller pattern (self-scheduling, unref'd, key-gated).
 */
import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";

export const CHAIN_MAX_UNDERLYINGS = 120;
export const CHAIN_MAX_DTE_DAYS = 60;
export const CHAIN_STRIKE_BAND = 0.20; // ±20% of spot
export const CHAIN_RUN_AFTER_ET_MINUTES = 16 * 60 + 15; // 16:15 ET

export function chainEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(env.ALPACA_KEY && env.ALPACA_SECRET);
}

/** Underlyings for today's snapshot: CSP-universe cache tuples
 *  [symbol, price, volume, dollarVol] (spot rides along free) + any open
 *  position underlyings (spot unknown → band check skipped for them). */
export function underlyingsFor(
  cacheJson: any, positionSymbols: string[] = [], cap = CHAIN_MAX_UNDERLYINGS,
): Array<{ sym: string; spot: number | null }> {
  const out = new Map<string, number | null>();
  for (const s of positionSymbols) {
    const sym = String(s || "").toUpperCase();
    if (sym && sym.length <= 5) out.set(sym, null);
  }
  const cands = Array.isArray(cacheJson?.candidates) ? cacheJson.candidates : [];
  for (const c of cands) {
    if (out.size >= cap) break;
    const sym = String(c?.[0] || "").toUpperCase();
    const spot = Number(c?.[1]);
    if (sym && !out.has(sym)) out.set(sym, Number.isFinite(spot) && spot > 0 ? spot : null);
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
  oi: number | null;
  feed: "indicative";   // NEVER dropped — this is not NBBO
}

/** Compact one Alpaca snapshot entry. Null when there is no quote at all
 *  (a chain row with no market is noise, not history). */
export function compactContract(occ: string, u: string, snap: any, nowMs: number): ChainRecord | null {
  const q = snap?.latestQuote;
  const bid = Number(q?.bp), ask = Number(q?.ap);
  if (!q || (!Number.isFinite(bid) && !Number.isFinite(ask))) return null;
  const g = snap?.greeks;
  const n = (v: any) => (Number.isFinite(Number(v)) ? Number(v) : null);
  return {
    s: occ, u, t: nowMs,
    bid: n(q?.bp), ask: n(q?.ap), bsz: n(q?.bs), asz: n(q?.as),
    iv: n(snap?.impliedVolatility), delta: n(g?.delta), oi: n(snap?.openInterest),
    feed: "indicative",
  };
}

export function chainDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "optionchains");
}

/** Append records to the day-file. Returns records written. */
export function archiveChainRecords(records: ChainRecord[], baseDir?: string, nowMs = Date.now()): number {
  if (!records.length) return 0;
  const dir = chainDir(baseDir);
  fs.mkdirSync(dir, { recursive: true });
  const day = new Date(nowMs).toISOString().slice(0, 10);
  const fp = path.join(dir, `${day}.jsonl`);
  fs.appendFileSync(fp, records.map((r) => JSON.stringify(r)).join("\n") + "\n");
  return records.length;
}

const ET_FMT = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York", weekday: "short", hour: "2-digit", minute: "2-digit", hour12: false,
});

/** Once per ET weekday, after the close (16:15 ET). lastRunDay is the ET
 *  date string of the last completed run (persisted by the caller). */
export function shouldRunNow(lastRunDay: string | null, nowMs = Date.now()): { run: boolean; etDay: string } {
  const parts = ET_FMT.formatToParts(nowMs);
  const get = (t: string) => parts.find((p) => p.type === t)?.value || "";
  const dow = get("weekday");
  const mins = parseInt(get("hour"), 10) * 60 + parseInt(get("minute"), 10);
  const etDay = new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York" }).format(nowMs); // YYYY-MM-DD
  const run = dow !== "Sat" && dow !== "Sun" && mins >= CHAIN_RUN_AFTER_ET_MINUTES && lastRunDay !== etDay;
  return { run, etDay };
}

export function snapshotUrl(underlying: string, pageToken?: string): string {
  const base = `https://data.alpaca.markets/v1beta1/options/snapshots/${encodeURIComponent(underlying)}?feed=indicative&limit=1000`;
  return pageToken ? `${base}&page_token=${encodeURIComponent(pageToken)}` : base;
}

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; json(): Promise<any> }>;

/** Fetch + filter + compact one underlying's chain (paginated, ≤5 pages). */
export async function fetchChainForUnderlying(
  u: string, spot: number | null, env: NodeJS.ProcessEnv = process.env,
  fetchImpl: FetchFn = fetch as any, nowMs = Date.now(),
): Promise<ChainRecord[]> {
  const headers = { "APCA-API-KEY-ID": env.ALPACA_KEY || "", "APCA-API-SECRET-KEY": env.ALPACA_SECRET || "" };
  const out: ChainRecord[] = [];
  let token: string | undefined;
  for (let page = 0; page < 5; page++) {
    const r = await fetchImpl(snapshotUrl(u, token), { headers });
    if (!r.ok) throw new Error(`alpaca chain ${u}: HTTP ${r.status}`);
    const d = await r.json();
    for (const [occ, snap] of Object.entries(d?.snapshots || {})) {
      if (!withinFilters(occ, spot, nowMs)) continue;
      const rec = compactContract(occ, u, snap, nowMs);
      if (rec) out.push(rec);
    }
    token = d?.next_page_token || undefined;
    if (!token) break;
  }
  return out;
}

// ── runtime state + boot (FIRMS pattern) ────────────────────────────────────
const LAST_RUN_PATH = () => path.join(chainDir(), ".last_run_day");

function readLastRunDay(): string | null {
  try { return fs.readFileSync(LAST_RUN_PATH(), "utf8").trim() || null; } catch { return null; }
}
function writeLastRunDay(day: string) {
  try { fs.mkdirSync(chainDir(), { recursive: true }); fs.writeFileSync(LAST_RUN_PATH(), day); } catch {}
}

function universeCachePath(): string {
  // storage_config.py: DATA_DIR is /data/voltrade on the volume, /tmp local
  for (const p of ["/data/voltrade/voltrade_csp_universe_cache.json", "/tmp/voltrade_csp_universe_cache.json"]) {
    if (fs.existsSync(p)) return p;
  }
  return "";
}

export async function runDailyChainArchive(
  env: NodeJS.ProcessEnv = process.env, fetchImpl: FetchFn = fetch as any,
  positionSymbols: string[] = [], nowMs = Date.now(),
): Promise<{ underlyings: number; contracts: number }> {
  let cache: any = null;
  const cp = universeCachePath();
  if (cp) { try { cache = JSON.parse(fs.readFileSync(cp, "utf8")); } catch {} }
  const unders = underlyingsFor(cache, positionSymbols);
  let contracts = 0;
  for (const { sym, spot } of unders) {
    try {
      const recs = await fetchChainForUnderlying(sym, spot, env, fetchImpl, nowMs);
      contracts += archiveChainRecords(recs, undefined, nowMs);
    } catch (e: any) {
      console.error(`[optionchains] ${sym}:`, e?.message || e);
    }
    await new Promise((r) => setTimeout(r, 350)); // stay polite on the data API
  }
  console.log(`[optionchains] archived ${contracts} contracts across ${unders.length} underlyings`);
  return { underlyings: unders.length, contracts };
}

let booted = false;
/** Hourly check; runs once per trading day after 16:15 ET. */
export function bootChainArchive(
  env: NodeJS.ProcessEnv = process.env,
  getPositions: () => string[] = () => [],
  intervalMs = 60 * 60_000,
): void {
  if (booted || !chainEnabled(env)) return;
  booted = true;
  const tick = async () => {
    try {
      const { run, etDay } = shouldRunNow(readLastRunDay());
      if (!run) return;
      writeLastRunDay(etDay); // claim first: a slow run must not double-fire
      await runDailyChainArchive(env, fetch as any, getPositions());
    } catch (e: any) {
      console.error("[optionchains] daily run failed:", e?.message || e);
    }
  };
  tick();
  setInterval(tick, intervalMs).unref?.();
}
