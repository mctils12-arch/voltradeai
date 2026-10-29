/**
 * optionsOiArchive.ts — daily open-interest sidecar for the index option
 * chains (Edge P1 gate 1, research/edge_preregistration.md).
 *
 * WHY: the chain archive's `oi` is always null (the indicative snapshot has
 * no openInterest), but Alpaca's Trading API `GET /v2/options/contracts`
 * carries real `open_interest` + `open_interest_date` (options_entitlements
 * probe, master_status 2026-10-29). Gate 1 of the dealer-gamma signal needs
 * our own GEX = sum(gamma x OI x 100 x spot) to be cross-checked against
 * SqueezeMetrics on >= 40 overlapping days — and OI history can never be
 * back-filled, so recording starts now (BUILD-FIRST: accumulation).
 *
 * SCOPE: CHAIN_ALWAYS (SPY/QQQ/IWM) only, expiry <= CHAIN_MAX_DTE_DAYS —
 * the index names the GEX proxy needs, ~tens of thousands of rows/day.
 * `open_interest_date` is stored per row: OCC publishes OI as of the PRIOR
 * session close, so the row's own date, not the file label, is the truth.
 *
 * Lives under optionchains/oi/ (a subdir: listChainDays / archiveStats only
 * read top-level day files, so no existing reader sees it). Internal
 * research use only; never served.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { beat } from "./jobHeartbeat";
import {
  chainDir, chainEnabled, etClock, shouldRunNow, chainQueryWindow, CHAIN_ALWAYS,
  CHAIN_MAX_PAGES, CHAIN_TICK_MS,
} from "./optionsChainArchive";

export const OI_PAGE_LIMIT = 10_000; // Alpaca contracts endpoint max page size

export interface OiRecord {
  s: string;            // OCC symbol
  u: string;            // underlying
  t: number;            // capture epoch ms
  oi: number | null;    // open interest (contracts)
  oid: string | null;   // open_interest_date reported by Alpaca (YYYY-MM-DD)
}

interface ContractRow { symbol?: unknown; open_interest?: unknown; open_interest_date?: unknown }
interface ContractsPage { option_contracts?: ContractRow[]; next_page_token?: string }
type FetchFn = (url: string, init?: { headers: Record<string, string> }) => Promise<{ ok: boolean; status: number; json(): Promise<unknown> }>;

export function oiDir(baseDir?: string): string { return path.join(chainDir(baseDir), "oi"); }
export function oiDayFile(day: string, baseDir?: string): string { return path.join(oiDir(baseDir), `${day}.jsonl`); }
export function isOiCaptured(day: string, baseDir?: string): boolean {
  return fs.existsSync(oiDayFile(day, baseDir)) || fs.existsSync(`${oiDayFile(day, baseDir)}.gz`);
}

export function contractsUrl(u: string, expGte: string, expLte: string, pageToken?: string, base = "https://paper-api.alpaca.markets"): string {
  const url = `${base}/v2/options/contracts?underlying_symbols=${encodeURIComponent(u)}&expiration_date_gte=${expGte}&expiration_date_lte=${expLte}&limit=${OI_PAGE_LIMIT}`;
  return pageToken ? `${url}&page_token=${encodeURIComponent(pageToken)}` : url;
}

/** Compact one Alpaca contract row; null when it has no symbol. OI arrives as
 *  a string ("12345") or absent — absent stays null, never 0 (unknown ≠ none). */
export function compactOi(row: ContractRow | null | undefined, u: string, nowMs: number): OiRecord | null {
  const s = String(row?.symbol || "");
  if (!s) return null;
  const raw = row?.open_interest;
  const oi = raw !== null && raw !== undefined && raw !== "" && Number.isFinite(Number(raw)) ? Number(raw) : null;
  const d = typeof row?.open_interest_date === "string" && /^\d{4}-\d{2}-\d{2}$/.test(row.open_interest_date) ? row.open_interest_date : null;
  return { s, u, t: nowMs, oi, oid: d };
}

export async function fetchOiForUnderlying(
  u: string, env: NodeJS.ProcessEnv = process.env, fetchImpl: FetchFn = fetch as unknown as FetchFn,
  nowMs = Date.now(), maxPages = CHAIN_MAX_PAGES,
): Promise<{ records: OiRecord[]; pages: number; truncated: boolean }> {
  const headers = { "APCA-API-KEY-ID": env.ALPACA_KEY || "", "APCA-API-SECRET-KEY": env.ALPACA_SECRET || "" };
  const w = chainQueryWindow(nowMs, null);
  const records: OiRecord[] = [];
  const seen = new Set<string>();
  let token: string | undefined;
  let pages = 0;
  while (pages < maxPages) {
    const r = await fetchImpl(contractsUrl(u, w.expGte, w.expLte, token), { headers });
    if (!r.ok) throw new Error(`alpaca contracts ${u}: HTTP ${r.status}`);
    const d = (await r.json()) as ContractsPage | null;
    pages++;
    for (const row of d?.option_contracts || []) {
      const rec = compactOi(row, u, nowMs);
      if (rec) records.push(rec);
    }
    token = d?.next_page_token || undefined;
    if (!token) return { records, pages, truncated: false };
    if (seen.has(token)) break;
    seen.add(token);
  }
  console.error(`[optionoi] ${u}: contracts TRUNCATED after ${pages} pages (cap ${maxPages})`);
  return { records, pages, truncated: true };
}

/** Capture the ET day exactly once. Staged then renamed, so a crash never
 *  leaves a half file that isOiCaptured would accept. Any underlying failing
 *  aborts the day (retried next tick): a partial day would silently bias the
 *  own-GEX sum low. */
export async function captureOpenInterest(
  env: NodeJS.ProcessEnv = process.env, fetchImpl: FetchFn = fetch as unknown as FetchFn, nowMs = Date.now(),
  baseDir?: string, sleepMs = 350,
): Promise<{ status: "skipped" | "published" | "failed"; day: string; rows: number; withOi: number; truncated: string[] }> {
  const day = etClock(nowMs).etDay;
  if (isOiCaptured(day, baseDir)) return { status: "skipped", day, rows: 0, withOi: 0, truncated: [] };
  const all: OiRecord[] = [];
  const truncated: string[] = [];
  try {
    for (const u of CHAIN_ALWAYS) {
      const got = await fetchOiForUnderlying(u, env, fetchImpl, nowMs);
      if (got.truncated) truncated.push(u);
      for (const r of got.records) all.push(r);
      if (sleepMs > 0) await new Promise((r) => setTimeout(r, sleepMs));
    }
  } catch (e) {
    console.error("[optionoi] capture failed (day not claimed):", e instanceof Error ? e.message : String(e));
    return { status: "failed", day, rows: 0, withOi: 0, truncated };
  }
  if (!all.length) return { status: "failed", day, rows: 0, withOi: 0, truncated };
  const fp = oiDayFile(day, baseDir);
  fs.mkdirSync(path.dirname(fp), { recursive: true });
  const tmp = `${fp}.tmp`;
  fs.writeFileSync(tmp, all.map((r) => JSON.stringify(r)).join("\n") + "\n");
  fs.renameSync(tmp, fp);
  const withOi = all.filter((r) => r.oi !== null).length;
  console.log(`[optionoi] ${day}: ${all.length} contracts (${withOi} with OI) across ${CHAIN_ALWAYS.length} underlyings`);
  return { status: "published", day, rows: all.length, withOi, truncated };
}

/** Same 2-day lifecycle as every other stream archive. */
export function gzipOldOiDays(baseDir?: string, nowMs = Date.now()): number {
  const dir = oiDir(baseDir);
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      if (nowMs - Date.parse(f.slice(0, 10)) < 2 * 86400_000) continue;
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch (e) {
    console.error("[optionoi] gzip rollover failed:", e instanceof Error ? e.message : String(e));
  }
  return n;
}

let booted = false;
/** Same post-close window and NYSE-day gate as the chain archive. */
export function bootOiArchive(env: NodeJS.ProcessEnv = process.env, intervalMs = CHAIN_TICK_MS): void {
  if (booted || !chainEnabled(env)) return;
  booted = true;
  let running = false;
  const tick = async () => {
    if (running) return;
    try {
      const now = Date.now();
      const today = etClock(now).etDay;
      if (!shouldRunNow(isOiCaptured(today) ? today : null, now).run) return;
      running = true;
      beat("options_oi_archive", "start");
      const cap = await captureOpenInterest(env, fetch as unknown as FetchFn, now);
      if (cap.status === "failed") beat("options_oi_archive", "finish", "error", `${cap.day}: capture failed — day not claimed`);
      else beat("options_oi_archive", "finish", "ok", `${cap.day} ${cap.status}: ${cap.rows} contracts, ${cap.withOi} with OI`);
      gzipOldOiDays();
    } catch (e) {
      beat("options_oi_archive", "finish", "error", `tick failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally { running = false; }
  };
  tick();
  setInterval(tick, intervalMs).unref?.();
}
