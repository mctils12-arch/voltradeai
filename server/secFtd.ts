/**
 * secFtd.ts — SEC fails-to-deliver (CNS fails), half-month files
 * (DATACORE MAXIMUS census build #6; contract live-verified 2026-07-07
 * by the source workup — every quirk below was probed, not assumed).
 *
 * Source: sec.gov cnsfails{YYYYMM}{a|b}.zip. Public domain (US
 * government work) — the one census top-10 source that is resale-safe.
 *
 * VERIFIED CONTRACT:
 *  - URL pattern is NON-UNIFORM: the standard path
 *    /files/data/fails-deliver-data/ covers most of 2017-06b→present,
 *    but 202605a lives at /files/data/other/fails-deliver-data/ TODAY,
 *    and 202308b needed a `_0` suffix there. This module tries the
 *    verified fallback chain; if every variant 404s the period is
 *    honestly "not published or moved" (the index page is the deep
 *    fallback, filed for the backfill task, not the poller).
 *  - Half-month boundary is 1st-14th ("a") / 15th-EOM ("b") — NOT the
 *    15/16 folklore (verified across 3 months). "a" publishes ~EOM of
 *    the same month; "b" ~the 15th of the following month.
 *  - File: one pipe-delimited .txt per zip, header
 *    SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE,
 *    LF (current) or CRLF (2004-era), pure ASCII, no quoting.
 *  - TWO trailer lines ("Trailer record count N" / "Trailer total
 *    quantity of shares M") — both verified exact against the data;
 *    used here as a free integrity checksum: mismatch = refuse file.
 *  - PRICE is either \d+\.\d{2} or the literal "." (~0.6% of rows;
 *    100% of 2004-era rows) → null, never zero. DESCRIPTION is
 *    hard-truncated at 30 chars with trailing spaces → rstrip.
 *  - QUANTITY is the aggregate net CNS fail BALANCE on that settlement
 *    date — a LEVEL, not a daily flow. CUSIP field can be a CINS
 *    (letter prefix, ~9% of rows) — never validate numeric.
 *  - Default UA → 403; SEC etiquette UA required (house EDGAR UA).
 *
 * HYPOTHESIS (gate-locked): raw FTD spikes are maximally crowded; the
 * edge is the settlement-stress composite (threshold persistence x FTD
 * x daily short volume — census #4/#6 joint), never FTD alone.
 *
 * ZIP handling: dependency-free single-member reader (central-directory
 * walk, stored + deflate) — SEC zips are single-.txt, stable format;
 * build-don't-buy applies to dependencies too.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const SEC_UA = { "User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)" };

// Verified fallback chain (in order). {p} = cnsfails{YYYYMM}{a|b}
const URL_VARIANTS = [
  (p: string) => `https://www.sec.gov/files/data/fails-deliver-data/${p}.zip`,
  (p: string) => `https://www.sec.gov/files/data/other/fails-deliver-data/${p}.zip`,
  (p: string) => `https://www.sec.gov/files/data/other/fails-deliver-data/${p}_0.zip`,
];

export interface FtdRow {
  date: string;           // ISO settlement date
  cusip: string;          // 9-char CUSIP or CINS (letter prefix = foreign)
  symbol: string;
  qty: number;            // aggregate net fail BALANCE (level, not flow)
  name: string;           // rstripped, source-truncated at 30 chars
  price: number | null;   // null when source prints "."
  rt: string;             // capture date
}

// ── Minimal single-member ZIP reader (stored + deflate) ─────────────────────

/** Extract the first .txt member via the central directory. Throws on
 *  anything unexpected — a malformed archive must never parse as data. */
export function unzipSingleText(buf: Buffer): string {
  // EOCD: scan back for signature 0x06054b50 (comment can pad the tail)
  let eocd = -1;
  const scanFrom = Math.max(0, buf.length - 65557);
  for (let i = buf.length - 22; i >= scanFrom; i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error("zip: EOCD not found");
  const entries = buf.readUInt16LE(eocd + 10);
  let off = buf.readUInt32LE(eocd + 16); // central directory offset
  for (let e = 0; e < entries; e++) {
    if (buf.readUInt32LE(off) !== 0x02014b50) throw new Error("zip: bad central directory entry");
    const method = buf.readUInt16LE(off + 10);
    const csize = buf.readUInt32LE(off + 20);
    const fnLen = buf.readUInt16LE(off + 28);
    const extraLen = buf.readUInt16LE(off + 30);
    const commentLen = buf.readUInt16LE(off + 32);
    const localOff = buf.readUInt32LE(off + 42);
    const name = buf.toString("utf8", off + 46, off + 46 + fnLen);
    if (name.toLowerCase().endsWith(".txt")) {
      if (buf.readUInt32LE(localOff) !== 0x04034b50) throw new Error("zip: bad local header");
      const lfnLen = buf.readUInt16LE(localOff + 26);
      const lextraLen = buf.readUInt16LE(localOff + 28);
      const dataStart = localOff + 30 + lfnLen + lextraLen;
      const data = buf.subarray(dataStart, dataStart + csize);
      if (method === 0) return data.toString("utf8");
      if (method === 8) return zlib.inflateRawSync(data).toString("utf8");
      throw new Error(`zip: unsupported compression method ${method}`);
    }
    off += 46 + fnLen + extraLen + commentLen;
  }
  throw new Error("zip: no .txt member");
}

// ── Parser (trailer-checksummed) ─────────────────────────────────────────────

const HEADER = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE";

/** Parse one half-month file. Returns [] when the header is wrong or
 *  EITHER trailer checksum disagrees (truncation/corruption guard —
 *  both trailers verified exact against the data in the workup). */
export function parseFtd(text: string, rt: string): FtdRow[] {
  if (!text) return [];
  const lines = text.split(/\r?\n/).filter((l) => l.length);
  if (!lines.length || lines[0].trim() !== HEADER) {
    console.error(`[datacore] secftd: unexpected header: ${(lines[0] || "").slice(0, 80)}`);
    return [];
  }
  const out: FtdRow[] = [];
  let trailerCount: number | null = null;
  let trailerQty: number | null = null;
  for (const line of lines.slice(1)) {
    if (!line.includes("|")) {
      const mc = line.match(/^Trailer record count\s+(\d+)/);
      if (mc) { trailerCount = parseInt(mc[1], 10); continue; }
      const mq = line.match(/^Trailer total quantity of shares\s+(\d+)/);
      if (mq) { trailerQty = parseInt(mq[1], 10); continue; }
      continue; // unknown non-delimited line — ignored, trailers still enforced
    }
    const p = line.split("|");
    if (p.length !== 6) continue;
    const [rawDate, cusip, symbol, rawQty, rawName, rawPrice] = p;
    if (!/^\d{8}$/.test(rawDate)) continue;
    const qty = parseInt(rawQty, 10);
    if (!Number.isFinite(qty)) continue;
    out.push({
      date: `${rawDate.slice(0, 4)}-${rawDate.slice(4, 6)}-${rawDate.slice(6, 8)}`,
      cusip,
      symbol,
      qty,
      name: rawName.replace(/\s+$/, ""),
      price: rawPrice === "." ? null : (Number.isFinite(parseFloat(rawPrice)) ? parseFloat(rawPrice) : null),
      rt,
    });
  }
  if (trailerCount == null || trailerCount !== out.length) {
    console.error(`[datacore] secftd: trailer count ${trailerCount} vs parsed ${out.length} — refusing file`);
    return [];
  }
  const qtySum = out.reduce((a, r) => a + r.qty, 0);
  if (trailerQty != null && trailerQty !== qtySum) {
    console.error(`[datacore] secftd: trailer qty ${trailerQty} vs summed ${qtySum} — refusing file`);
    return [];
  }
  return out;
}

// ── Periods (half-month publication units) ──────────────────────────────────

/** Newest-first candidate periods at time `now` — e.g. "202606b",
 *  "202606a", "202605b"… The newest one or two typically 404 until
 *  their publish day (EOM for "a", ~15th next month for "b") — an
 *  expected honest no-op, not an error. */
export function candidatePeriods(nowMs: number, count = 5): string[] {
  const out: string[] = [];
  const d = new Date(nowMs);
  let y = d.getUTCFullYear(), m = d.getUTCMonth() + 1;
  let half: "a" | "b" = d.getUTCDate() >= 15 ? "b" : "a";
  while (out.length < count) {
    out.push(`${y}${String(m).padStart(2, "0")}${half}`);
    if (half === "b") half = "a";
    else { half = "b"; m--; if (m === 0) { m = 12; y--; } }
  }
  return out;
}

// ── Fetch (verified URL fallback chain) ──────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; arrayBuffer(): Promise<ArrayBuffer> }>;

/** Fetch one period's rows. [] = not published anywhere on the chain
 *  (honest no-op); null = transport error (retry next poll). */
export async function fetchFtdPeriod(
  period: string,
  fetchImpl: FetchFn = fetch as any,
  nowMs?: number,
): Promise<FtdRow[] | null> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  let sawTransportError = false;
  for (const variant of URL_VARIANTS) {
    const url = variant(`cnsfails${period}`);
    try {
      const r = await fetchImpl(url, { headers: SEC_UA, signal: AbortSignal.timeout(60000) as any });
      if (r.status === 404) continue; // try next variant
      if (r.status === 403) {
        console.error(`[datacore] secftd ${period}: 403 — UA rejected or fair-access throttle`);
        return null;
      }
      if (!r.ok) {
        console.error(`[datacore] secftd ${period} ${url} -> ${r.status}`);
        sawTransportError = true;
        continue;
      }
      const buf = Buffer.from(await r.arrayBuffer());
      return parseFtd(unzipSingleText(buf), rt);
    } catch (e: any) {
      console.error(`[datacore] secftd ${period}:`, e?.message || e);
      sawTransportError = true;
    }
  }
  return sawTransportError ? null : [];
}

// ── Archive (period-level dedup; one gz file per half-month) ────────────────

const archivedPeriods = new Set<string>();
let seeded = false;

function ftdDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "secftd");
}

function seedSeen(baseDir?: string): void {
  if (seeded) return;
  try {
    for (const f of fs.readdirSync(ftdDir(baseDir))) {
      const m = f.match(/^(\d{6}[ab])\.jsonl\.gz$/);
      if (m) archivedPeriods.add(m[1]);
    }
  } catch {}
  seeded = true;
}

export function isFtdPeriodArchived(period: string, baseDir?: string): boolean {
  seedSeen(baseDir);
  return archivedPeriods.has(period);
}

/** ~58k rows ≈ 3.5MB plain → gz-on-write (~700KB). */
export function archiveFtdPeriod(period: string, rows: FtdRow[], baseDir?: string): number {
  if (!rows.length) return 0;
  seedSeen(baseDir);
  if (archivedPeriods.has(period)) return 0;
  try {
    const dir = ftdDir(baseDir);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, `${period}.jsonl.gz`),
      zlib.gzipSync(rows.map((r) => JSON.stringify(r)).join("\n") + "\n"));
    archivedPeriods.add(period);
    return rows.length;
  } catch (e: any) {
    console.error(`[datacore] secftd archive ${period}:`, e?.message || e);
    return 0;
  }
}

export function _resetFtdForTests(): void {
  archivedPeriods.clear();
  seeded = false;
  cache = null;
  polling = false;
}

// ── Summary (poll-time only) ─────────────────────────────────────────────────

export const FTD_TOP_CAP = 15;
export const FTD_QTY_FLOOR = 100_000; // shares — leaderboard floor, stated

export interface FtdSummary {
  period: string;
  settlement_dates: number;
  newest_date: string;
  rows: number;
  top_fails: Array<{ symbol: string; name: string; qty: number; price: number | null }>;
  qty_floor: number;
  top_cap: number;
}

export function summarizeFtd(period: string, rows: FtdRow[]): FtdSummary | null {
  if (!rows.length) return null;
  const dates = new Set(rows.map((r) => r.date));
  const newest = Array.from(dates).sort().pop()!;
  const top = rows
    .filter((r) => r.date === newest && r.qty >= FTD_QTY_FLOOR)
    .sort((a, b) => b.qty - a.qty)
    .slice(0, FTD_TOP_CAP)
    .map((r) => ({ symbol: r.symbol, name: r.name, qty: r.qty, price: r.price }));
  return {
    period,
    settlement_dates: dates.size,
    newest_date: newest,
    rows: rows.length,
    top_fails: top,
    qty_floor: FTD_QTY_FLOOR,
    top_cap: FTD_TOP_CAP,
  };
}

let cache: { at: number; summary: FtdSummary } | null = null;
let polling = false;

export function latestFtd() {
  return cache;
}

/** Read one archived period back (poller context only). */
export function readFtdPeriod(period: string, baseDir?: string): FtdRow[] {
  try {
    const text = zlib.gunzipSync(fs.readFileSync(path.join(ftdDir(baseDir), `${period}.jsonl.gz`))).toString("utf8");
    const out: FtdRow[] = [];
    for (const line of text.split("\n")) {
      if (!line) continue;
      try { out.push(JSON.parse(line)); } catch {}
    }
    return out;
  } catch { return []; }
}

export async function refreshFtd(
  fetchImpl: FetchFn = fetch as any,
  nowMs?: number,
  baseDir?: string,
): Promise<void> {
  const now = nowMs ?? Date.now();
  try {
    seedSeen(baseDir);
    let newestHandled = false;
    for (const period of candidatePeriods(now)) {
      if (archivedPeriods.has(period)) {
        if (!newestHandled) {
          if (cache?.summary.period !== period) {
            const s = summarizeFtd(period, readFtdPeriod(period, baseDir));
            if (s) cache = { at: Date.now(), summary: s };
          }
          newestHandled = true;
        }
        continue;
      }
      const rows = await fetchFtdPeriod(period, fetchImpl, now);
      if (rows === null || rows.length === 0) continue; // retry-later or not published
      archiveFtdPeriod(period, rows, baseDir);
      if (!newestHandled) {
        const s = summarizeFtd(period, rows);
        if (s) cache = { at: Date.now(), summary: s };
        newestHandled = true;
      }
      await new Promise((res) => setTimeout(res, 500)); // SEC fair access
    }
  } catch (e: any) {
    console.error("[datacore] secftd refresh:", e?.message || e);
  }
}

/** Files land ~EOM ("a") and ~15th ("b") — a 12h poll catches both
 *  within half a day. Eager boot. History backfill (2004→present,
 *  ~370MB gz total — a volume-budget decision) is a SEPARATE filed
 *  task, never the boot path. */
export function bootFtdPoll(intervalMs = 12 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshFtd().catch((e) => console.error("[datacore] secftd boot:", e?.message || e));
  setInterval(() => { refreshFtd(); }, intervalMs).unref?.();
}
