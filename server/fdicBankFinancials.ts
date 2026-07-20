/**
 * fdicBankFinancials.ts — FDIC per-bank quarterly financials snapshot
 * (BUILD ORDER 6 #3 v2 — the documented follow-up to fdicBanks.ts's v1
 * failures-only stream, filed 2026-07-06, built 2026-07-20).
 *
 * Source: FDIC Bank Data API /banks/financials — same host as failures
 * (api.fdic.gov; the documented banks.data.fdic.gov 301s there, per
 * fdicBanks.ts's probe finding). Keyless JSON. Live-verified 2026-07-20:
 * the most recent published quarter (REPDTE) covers 4,352 institutions;
 * stable CERT-ordered pagination confirmed (offset+limit, sort_by=CERT);
 * a >10000 `limit` 400s ("Number must be less than or equal to 10000")
 * so pagination is implemented even though today's whole quarter fits
 * on one page. REPDTE arrives as YYYYMMDD (normalized to ISO).
 *
 * v2 SCOPE — one IMMUTABLE snapshot per quarter. Published financials
 * for a REPDTE never change after the fact, so a file already archived
 * for a given REPDTE is never re-fetched (unlike fdicfailures' rolling
 * event-identity dedup, this is a presence check on the quarter itself).
 *
 * HYPOTHESIS (gate-locked, RAW only — same regional-bank-stress family
 * as fdicfailures, joinable on CERT): deposits_uninsured_k / deposits_k
 * spikes and nonperforming_pct increases at SMALL regional banks precede
 * failures/stress — EDGE DOCTRINE #2 (whales can't fish small regional
 * banks; nobody trades a single-branch community bank's balance sheet).
 * This session builds the ARCHIVE only. Gate 1 = cross-check a known-
 * failed bank's last snapshot before failure against its FDIC failure
 * record (fdicfailures, joined on CERT). Gate 2 = deposit/asset-quality
 * deltas vs forward KRE/regional-bank returns — NEEDS >=2 archived
 * quarters to compute any delta at all; not available from one snapshot.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const FINANCIALS_URL = "https://api.fdic.gov/banks/financials";
const FIELDS = "CERT,NAME,REPDTE,STALP,CITY,ASSET,DEP,DEPDOM,DEPINS,DEPUNINS,EQ,NETINC,ROA,ROE,NPERFV";
/** FDIC API hard cap; verified live 2026-07-20 (limit=20000 -> 400 validate:too_big). */
export const FINANCIALS_PAGE_LIMIT = 10000;
/** Safety backstop so a pagination bug can't loop forever against a live API. */
const MAX_PAGES = 50;

export interface BankFinancialsSnapshot {
  cert: number | null;              // FDIC certificate number (join key to fdicfailures)
  name: string;
  repdte: string;                   // YYYY-MM-DD, normalized from REPDTE (YYYYMMDD)
  state: string | null;             // STALP
  city: string | null;
  assets_k: number | null;          // ASSET, $ thousands as published
  deposits_k: number | null;        // DEP
  deposits_domestic_k: number | null;   // DEPDOM
  deposits_insured_k: number | null;    // DEPINS
  deposits_uninsured_k: number | null;  // DEPUNINS — the run-risk field
  equity_k: number | null;          // EQ
  net_income_k: number | null;      // NETINC
  roa_pct: number | null;           // ROA
  roe_pct: number | null;           // ROE
  nonperforming_pct: number | null; // NPERFV — nonperforming assets / total assets
  rt: string;                       // as-fetched UTC date
}

const num = (v: any): number | null => {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
};

/** "20260331" -> "2026-03-31"; returns null on garbage. */
export function normalizeRepdte(v: any): string | null {
  const m = String(v ?? "").match(/^(\d{4})(\d{2})(\d{2})$/);
  if (!m) return null;
  return `${m[1]}-${m[2]}-${m[3]}`;
}

/** FDIC envelope ({data:[{data:{...}}]}) -> snapshots. */
export function parseFinancials(json: any, rt: string): BankFinancialsSnapshot[] {
  const data = json?.data;
  if (!Array.isArray(data)) return [];
  const out: BankFinancialsSnapshot[] = [];
  for (const wrap of data) {
    const r = wrap?.data ?? wrap;
    if (!r || typeof r !== "object") continue;
    const repdte = normalizeRepdte(r.REPDTE);
    const name = r.NAME;
    if (!repdte || !name) continue;
    out.push({
      cert: num(r.CERT),
      name: String(name),
      repdte,
      state: r.STALP != null ? String(r.STALP) : null,
      city: r.CITY != null ? String(r.CITY) : null,
      assets_k: num(r.ASSET),
      deposits_k: num(r.DEP),
      deposits_domestic_k: num(r.DEPDOM),
      deposits_insured_k: num(r.DEPINS),
      deposits_uninsured_k: num(r.DEPUNINS),
      equity_k: num(r.EQ),
      net_income_k: num(r.NETINC),
      roa_pct: num(r.ROA),
      roe_pct: num(r.ROE),
      nonperforming_pct: num(r.NPERFV),
      rt,
    });
  }
  return out;
}

// ── Fetch ────────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

/** Finds the most recently published REPDTE (e.g. "20260331"), or null on failure. */
export async function fetchLatestRepdte(fetchImpl: FetchFn = fetch as any): Promise<string | null> {
  const url = `${FINANCIALS_URL}?limit=1&sort_by=REPDTE&sort_order=DESC&fields=REPDTE`;
  try {
    const r = await fetchImpl(url, {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(30000) as any,
    });
    if (!r.ok) {
      console.error(`[datacore] fdicbankfinancials latest-repdte -> ${r.status}`);
      return null;
    }
    const json = JSON.parse(await r.text());
    const raw = json?.data?.[0]?.data?.REPDTE;
    return raw != null ? String(raw) : null;
  } catch (e: any) {
    console.error("[datacore] fdicbankfinancials latest-repdte:", e?.message || e);
    return null;
  }
}

/** Pulls every institution's financials for one quarter, paginated (stable CERT-ordered). */
export async function fetchFinancialsSnapshot(
  repdte: string,
  fetchImpl: FetchFn = fetch as any,
  nowMs?: number,
): Promise<BankFinancialsSnapshot[]> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const out: BankFinancialsSnapshot[] = [];
  let offset = 0;
  for (let page = 0; page < MAX_PAGES; page++) {
    const url = `${FINANCIALS_URL}?filters=${encodeURIComponent(`REPDTE:${repdte}`)}` +
      `&limit=${FINANCIALS_PAGE_LIMIT}&offset=${offset}&sort_by=CERT&sort_order=ASC` +
      `&fields=${encodeURIComponent(FIELDS)}`;
    try {
      const r = await fetchImpl(url, {
        headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
        signal: AbortSignal.timeout(30000) as any,
      });
      if (!r.ok) {
        console.error(`[datacore] fdicbankfinancials -> ${r.status} (offset ${offset})`);
        break;
      }
      const json = JSON.parse(await r.text());
      const page_records = parseFinancials(json, rt);
      out.push(...page_records);
      const total = num(json?.meta?.total ?? json?.totals?.count) ?? out.length;
      offset += FINANCIALS_PAGE_LIMIT;
      if (page_records.length < FINANCIALS_PAGE_LIMIT || offset >= total) break;
    } catch (e: any) {
      console.error("[datacore] fdicbankfinancials:", e?.message || e);
      break;
    }
  }
  return out;
}

// ── Archive (one immutable file per REPDTE quarter — never re-fetched once present) ──

function financialsDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "fdicbankfinancials");
}

function snapshotFilePaths(repdte: string, baseDir?: string): { plain: string; gz: string } {
  const dir = financialsDir(baseDir);
  return { plain: path.join(dir, `${repdte}.jsonl`), gz: path.join(dir, `${repdte}.jsonl.gz`) };
}

/** True if this quarter is already archived (plain or gzipped) — the dedup check. */
export function snapshotArchived(repdte: string, baseDir?: string): boolean {
  const { plain, gz } = snapshotFilePaths(repdte, baseDir);
  return fs.existsSync(plain) || fs.existsSync(gz);
}

function readSnapshotText(repdte: string, baseDir?: string): string | null {
  const { plain, gz } = snapshotFilePaths(repdte, baseDir);
  try {
    if (fs.existsSync(gz)) return zlib.gunzipSync(fs.readFileSync(gz)).toString("utf8");
    if (fs.existsSync(plain)) return fs.readFileSync(plain, "utf8");
    return null;
  } catch {
    return null;
  }
}

/** Row count of an already-archived quarter (0 if absent/unreadable) — cheap re-read on boot. */
export function readArchivedCount(repdte: string, baseDir?: string): number {
  const text = readSnapshotText(repdte, baseDir);
  return text == null ? 0 : text.split("\n").filter(Boolean).length;
}

/** Full records for an already-archived quarter (read-on-demand — not kept resident in memory). */
export function readArchivedSnapshot(repdte: string, baseDir?: string): BankFinancialsSnapshot[] {
  const text = readSnapshotText(repdte, baseDir);
  if (text == null) return [];
  const out: BankFinancialsSnapshot[] = [];
  for (const line of text.split("\n")) {
    if (!line) continue;
    try { out.push(JSON.parse(line)); } catch {}
  }
  return out;
}

/** Writes a full quarter's snapshot, gzipped immediately (publication is final — never appended to). */
export function archiveSnapshot(repdte: string, records: BankFinancialsSnapshot[], baseDir?: string): number {
  if (!records.length) return 0;
  const dir = financialsDir(baseDir);
  const { gz } = snapshotFilePaths(repdte, baseDir);
  try {
    fs.mkdirSync(dir, { recursive: true });
    const body = records.map((r) => JSON.stringify(r)).join("\n") + "\n";
    fs.writeFileSync(gz, zlib.gzipSync(Buffer.from(body, "utf8")));
    return records.length;
  } catch (e: any) {
    console.error("[datacore] fdicbankfinancials archive:", e?.message || e);
    return 0;
  }
}

// ── Cache + poll ────────────────────────────────────────────────────────────

interface SnapshotMeta { at: number; repdte: string; count: number }
let cache: SnapshotMeta | null = null;
let polling = false;

export function latestFinancialsMeta(): SnapshotMeta | null {
  return cache;
}

/**
 * Checks for a newer published quarter (cheap 1-record probe); pulls +
 * archives the full snapshot only when the REPDTE isn't already on disk.
 * A quiet quarter (no new REPDTE yet) costs one small request.
 */
export async function refreshFinancials(fetchImpl: FetchFn = fetch as any, nowMs?: number,
                                        baseDir?: string): Promise<void> {
  try {
    const repdte = await fetchLatestRepdte(fetchImpl);
    if (!repdte) return;
    const isoRepdte = normalizeRepdte(repdte) ?? repdte;
    if (!snapshotArchived(isoRepdte, baseDir)) {
      const records = await fetchFinancialsSnapshot(repdte, fetchImpl, nowMs);
      if (records.length) archiveSnapshot(isoRepdte, records, baseDir);
      cache = { at: Date.now(), repdte: isoRepdte, count: records.length };
    } else if (!cache || cache.repdte !== isoRepdte) {
      cache = { at: Date.now(), repdte: isoRepdte, count: readArchivedCount(isoRepdte, baseDir) };
    }
  } catch (e: any) {
    console.error("[datacore] fdicbankfinancials refresh:", e?.message || e);
  }
}

/** Quarterly-cadence source — a daily probe is a single cheap request until
 *  a new REPDTE actually appears. Eager boot. */
export function bootFinancialsPoll(intervalMs = 24 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshFinancials();
  setInterval(() => { refreshFinancials(); }, intervalMs).unref?.();
}
