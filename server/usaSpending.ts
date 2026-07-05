/**
 * usaSpending.ts — USAspending.gov federal contract-award pipeline.
 * Stream #4 of the DATA STREAM EXPANSION build order (hypothesis + ladder
 * filed in research/open_questions.md BEFORE the build; endpoint behavior
 * live-verified 2026-07-05 — see the research annex there).
 *
 * HYPOTHESIS (gate 2, not attempted here): large award/market-cap ratios
 * move SMALL caps with a lag; the filing feed beats news wires — for
 * CIVILIAN agencies only. DoD/USACE publish ~90 DAYS late (verified), so
 * gate-2 work must exclude or separately cohort DoD rows. rt (as-seen
 * date) is the ONLY honest event date for returns math — action_date is
 * the signature date, not publication.
 *
 * EXPLICIT CAP (stated, never silent): contracts A-D with
 * |Transaction Amount| >= $25,000 — measured to keep 99.74% of positive
 * obligated dollars in ~20% of rows (2026-06-30, n=6,691). Deobligations
 * (negative) kept symmetrically. Because the API's award_amounts filter
 * applies to the award's LIFETIME total (verified trap), the floor is
 * applied client-side via two sorted passes with early stop.
 *
 * TICKER MAPPING (gate 1's hard part; precision-first, NEVER fuzzy):
 * exact normalized-name match against SEC company_tickers.json, then —
 * for large unmatched rows — the award-detail endpoint's FPDS-reported
 * parent (NEVER the recipient-profile endpoint: its parent data is
 * vintage-less and demonstrably wrong). Unmatched rows archive with
 * tkr:null and are SKIPPED by consumers. Every resolved UEI lands in a
 * persistent cache — resolved forever (BUILD-FIRST #2).
 *
 * LICENSE: US-government work, free incl. commercial; attribution
 * "USAspending.gov, U.S. Department of the Treasury". RESTRICTION:
 * legacy DUNS numbers are D&B-proprietary — this stream stores UEI ONLY.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export const CONTRACT_FLOOR_USD = 25_000;
export const PARENT_LOOKUP_MIN_USD = 1_000_000;
export const PARENT_LOOKUPS_PER_CYCLE = 150;
const PAGE_LIMIT = 100;
const MAX_PAGES_PER_PASS = 120; // ~12k rows/pass; early stop hits far sooner

export interface ContractTxn {
  aid: string;          // generated_internal_id (award key)
  piid: string | null;
  mod: string | null;   // "0" = new award
  ad: string | null;    // action_date (signature date — never the event date)
  amt: number;          // federal_action_obligation for this action
  r: string | null;     // recipient name as reported (raw, always kept)
  uei: string | null;   // SAM UEI (never DUNS)
  rid: string | null;
  pn: string | null;    // parent recipient name (award-detail, when fetched)
  puei: string | null;
  tkr: string | null;   // matched ticker or null (null = unmatched, skipped)
  mm: "cache" | "name" | "parent" | null; // match method (gate-1 audit key)
  mname: string | null; // the exact name string that matched
  ag: string | null;    // awarding toptier agency
  sub: string | null;
  naics: string | null;
  psc: string | null;
  atype: string | null;
  desc: string | null;  // truncated 300 chars
  rt: string;           // as-seen UTC date — the honest event date
}

// ── Name normalization (precision-first) ────────────────────────────────────

const SUFFIX_TOKENS = new Set([
  "THE", "INC", "INCORPORATED", "CORP", "CORPORATION", "LLC", "L.L.C", "CO",
  "COMPANY", "LTD", "LIMITED", "LP", "L.P", "PLC", "HOLDINGS", "HOLDING",
  "GROUP", "INTERNATIONAL", "INTL",
]);

/** Uppercase, strip punctuation and corporate suffix tokens. Returns "" when
 *  the residue is too short to match safely (<5 chars and <2 tokens) — a
 *  too-aggressive strip must yield NO match, never a wrong one. */
export function normalizeCompanyName(s: string | null | undefined): string {
  if (!s) return "";
  const tokens = String(s).toUpperCase().replace(/[^A-Z0-9 ]+/g, " ").split(/\s+/)
    .filter((t) => t && !SUFFIX_TOKENS.has(t));
  const out = tokens.join(" ").trim();
  if (tokens.length < 2 && out.length < 5) return "";
  return out;
}

/** Builds normalized-name -> ticker from SEC company_tickers.json
 *  ({"0":{cik_str,ticker,title},...}). Collisions drop BOTH names —
 *  ambiguity must never guess. */
export function buildTickerMap(json: any): Map<string, string> {
  const m = new Map<string, string>();
  const collided = new Set<string>();
  for (const row of Object.values(json || {}) as any[]) {
    const key = normalizeCompanyName(row?.title);
    if (!key || !row?.ticker) continue;
    if (m.has(key) && m.get(key) !== row.ticker) { collided.add(key); continue; }
    m.set(key, row.ticker);
  }
  collided.forEach((k) => m.delete(k));
  return m;
}

// ── Transaction parsing (field names are the API's display names) ──────────

export function parseTxnRow(row: any, rt: string): ContractTxn {
  return {
    aid: String(row["generated_internal_id"] || ""),
    piid: row["Award ID"] ?? null,
    mod: row["Mod"] != null ? String(row["Mod"]) : null,
    ad: row["Action Date"] ?? null,
    amt: Number(row["Transaction Amount"] ?? 0),
    r: row["Recipient Name"] ?? null,
    uei: row["Recipient UEI"] ?? null,
    rid: row["recipient_id"] ?? null,
    pn: null, puei: null, tkr: null, mm: null, mname: null,
    ag: row["Awarding Agency"] ?? null,
    sub: row["Awarding Sub Agency"] ?? null,
    naics: row["NAICS"]?.code ?? null,
    psc: row["PSC"]?.code ?? null,
    atype: row["Award Type"] ?? null,
    desc: row["Transaction Description"] ? String(row["Transaction Description"]).slice(0, 300) : null,
    rt,
  };
}

// ── Fetch (injectable; two sorted passes with early stop at the floor) ─────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)", "Content-Type": "application/json" };
const TXN_URL = "https://api.usaspending.gov/api/v2/search/spending_by_transaction/";
const AWARD_URL = "https://api.usaspending.gov/api/v2/awards/";

const TXN_FIELDS = [
  "Award ID", "Mod", "Recipient Name", "Recipient UEI", "recipient_id",
  "Action Date", "Transaction Amount", "Transaction Description",
  "Award Type", "Awarding Agency", "Awarding Sub Agency", "NAICS", "PSC",
  "generated_internal_id",
];

async function postJson(fetchImpl: FetchFn, url: string, body: any): Promise<any> {
  const r = await fetchImpl(url, {
    method: "POST", headers: UA, body: JSON.stringify(body),
    signal: AbortSignal.timeout(20000) as any,
  });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return JSON.parse(await r.text());
}

/** One sorted pagination pass; stops when the floor is crossed (results are
 *  monotone in Transaction Amount under the sort). */
async function passAboveFloor(
  fetchImpl: FetchFn, start: string, end: string, order: "desc" | "asc",
  rt: string, delayMs: number,
): Promise<ContractTxn[]> {
  const out: ContractTxn[] = [];
  for (let page = 1; page <= MAX_PAGES_PER_PASS; page++) {
    const d = await postJson(fetchImpl, TXN_URL, {
      filters: {
        time_period: [{ start_date: start, end_date: end, date_type: "last_modified_date" }],
        award_type_codes: ["A", "B", "C", "D"],
      },
      fields: TXN_FIELDS,
      limit: PAGE_LIMIT, page, sort: "Transaction Amount", order,
    });
    const rows: any[] = d?.results || [];
    let crossed = false;
    for (const row of rows) {
      const amt = Number(row["Transaction Amount"] ?? 0);
      const past = order === "desc" ? amt < CONTRACT_FLOOR_USD : amt > -CONTRACT_FLOOR_USD;
      if (past) { crossed = true; break; }
      out.push(parseTxnRow(row, rt));
    }
    if (crossed || !d?.page_metadata?.hasNext) break;
    if (delayMs > 0) await new Promise((res) => setTimeout(res, delayMs));
  }
  return out;
}

export async function fetchRecentContracts(
  fetchImpl: FetchFn = fetch as any, nowMs?: number, delayMs = 150,
): Promise<ContractTxn[]> {
  const now = nowMs ?? Date.now();
  const rt = new Date(now).toISOString().slice(0, 10);
  const start = new Date(now - 2 * 86400_000).toISOString().slice(0, 10);
  const end = rt;
  // positives descending (stop at +floor), then negatives ascending
  // (stop at -floor) — deobligations kept symmetrically without paging
  // through the ~80% of rows below the floor.
  const pos = await passAboveFloor(fetchImpl, start, end, "desc", rt, delayMs);
  const neg = await passAboveFloor(fetchImpl, start, end, "asc", rt, delayMs);
  const seen = new Set<string>();
  return [...pos, ...neg].filter((t) => {
    const k = `${t.aid}|${t.mod}|${t.amt}`;
    if (!t.aid || seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

/** Award-detail parent (the ONLY trustworthy parent source — the FPDS
 *  record on the award itself; the recipient-profile endpoint aggregates
 *  vintage-less SAM history and returns provably wrong parents). */
export async function fetchAwardParent(
  aid: string, fetchImpl: FetchFn = fetch as any,
): Promise<{ pn: string | null; puei: string | null }> {
  const r = await fetchImpl(`${AWARD_URL}${encodeURIComponent(aid)}/`, {
    headers: UA, signal: AbortSignal.timeout(15000) as any,
  });
  if (!r.ok) throw new Error(`award ${aid} -> ${r.status}`);
  const d = JSON.parse(await r.text());
  return {
    pn: d?.recipient?.parent_recipient_name ?? null,
    puei: d?.recipient?.parent_recipient_uei ?? null,
  };
}

// ── Ticker resolution (cache -> name -> parent; never fuzzy) ───────────────

export interface UeiCache { [uei: string]: { tkr: string; mm: string; at: string } }

export function resolveTickers(
  txns: ContractTxn[], tickerMap: Map<string, string>, ueiCache: UeiCache,
): { resolved: ContractTxn[]; needParent: ContractTxn[] } {
  const needParent: ContractTxn[] = [];
  for (const t of txns) {
    if (t.uei && ueiCache[t.uei]) {
      t.tkr = ueiCache[t.uei].tkr; t.mm = "cache"; t.mname = null;
      continue;
    }
    const key = normalizeCompanyName(t.r);
    const hit = key ? tickerMap.get(key) : undefined;
    if (hit) {
      t.tkr = hit; t.mm = "name"; t.mname = key;
      if (t.uei) ueiCache[t.uei] = { tkr: hit, mm: "name", at: t.rt };
    } else if (Math.abs(t.amt) >= PARENT_LOOKUP_MIN_USD) {
      needParent.push(t);
    }
  }
  return { resolved: txns, needParent };
}

export function applyParentMatch(
  t: ContractTxn, pn: string | null, puei: string | null,
  tickerMap: Map<string, string>, ueiCache: UeiCache,
): void {
  t.pn = pn; t.puei = puei;
  if (puei && ueiCache[puei]) {
    t.tkr = ueiCache[puei].tkr; t.mm = "parent"; t.mname = null;
  } else {
    const key = normalizeCompanyName(pn);
    const hit = key ? tickerMap.get(key) : undefined;
    if (hit) {
      t.tkr = hit; t.mm = "parent"; t.mname = key;
      if (t.uei) ueiCache[t.uei] = { tkr: hit, mm: "parent", at: t.rt };
    }
  }
}

// ── Archive + persistent UEI cache ──────────────────────────────────────────

const archivedKeys = new Set<string>();
let seeded = false;

function usaDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "usaspending");
}

export function loadUeiCache(baseDir?: string): UeiCache {
  try { return JSON.parse(fs.readFileSync(path.join(usaDir(baseDir), "ueimap.json"), "utf8")); }
  catch { return {}; }
}

export function saveUeiCache(cache: UeiCache, baseDir?: string): void {
  try {
    fs.mkdirSync(usaDir(baseDir), { recursive: true });
    fs.writeFileSync(path.join(usaDir(baseDir), "ueimap.json"), JSON.stringify(cache));
  } catch {}
}

function seedSeen(dir: string, nowMs: number): void {
  for (let i = 0; i < 3; i++) {
    const day = new Date(nowMs - i * 86400_000).toISOString().slice(0, 10);
    for (const fp of [path.join(dir, `${day}.jsonl`), path.join(dir, `${day}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { const o = JSON.parse(line); archivedKeys.add(`${o.aid}|${o.mod}|${o.amt}`); } catch {}
      }
    }
  }
}

/** Dedup by (aid,mod,amt): an FPDS correction restating the same action
 *  with a NEW amount appends as a new row with its own rt — the vintage
 *  record, same discipline as fredMacro. */
export function archiveContractTxns(txns: ContractTxn[], baseDir?: string, nowMs?: number): number {
  const dir = usaDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = txns.filter((t) => t.aid && !archivedKeys.has(`${t.aid}|${t.mod}|${t.amt}`));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((t) => JSON.stringify(t)).join("\n") + "\n");
    fresh.forEach((t) => archivedKeys.add(`${t.aid}|${t.mod}|${t.amt}`));
    if (archivedKeys.size > 300_000) archivedKeys.clear(); // bound memory
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] usaspending archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldUsaDays(baseDir?: string, nowMs?: number): number {
  const dir = usaDir(baseDir);
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

// ── Cache + poll loop ───────────────────────────────────────────────────────

let cache: { at: number; txns: ContractTxn[] } | null = null;
let polling = false;

export function latestContracts(): { at: number; txns: ContractTxn[] } | null {
  return cache;
}

const SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json";
let tickerMapCache: { at: number; map: Map<string, string> } | null = null;

async function getTickerMap(fetchImpl: FetchFn): Promise<Map<string, string>> {
  if (tickerMapCache && Date.now() - tickerMapCache.at < 24 * 3600_000) return tickerMapCache.map;
  const r = await fetchImpl(SEC_TICKERS_URL, {
    headers: { "User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)" },
    signal: AbortSignal.timeout(20000) as any,
  });
  if (!r.ok) throw new Error(`company_tickers ${r.status}`);
  const map = buildTickerMap(JSON.parse(await r.text()));
  tickerMapCache = { at: Date.now(), map };
  return map;
}

export async function refreshContractsCache(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<void> {
  try {
    const txns = await fetchRecentContracts(fetchImpl, nowMs);
    const tickerMap = await getTickerMap(fetchImpl);
    const ueiCache = loadUeiCache();
    const { needParent } = resolveTickers(txns, tickerMap, ueiCache);
    // Parent lookups: only large unmatched rows, hard per-cycle budget —
    // fair-access politeness; unresolved rows simply stay tkr:null.
    for (const t of needParent.slice(0, PARENT_LOOKUPS_PER_CYCLE)) {
      try {
        const { pn, puei } = await fetchAwardParent(t.aid, fetchImpl);
        applyParentMatch(t, pn, puei, tickerMap, ueiCache);
      } catch {}
      await new Promise((res) => setTimeout(res, 150));
    }
    saveUeiCache(ueiCache);
    if (txns.length || !cache) cache = { at: Date.now(), txns };
    try { archiveContractTxns(txns, undefined, nowMs); } catch {}
    try { gzipOldUsaDays(undefined, nowMs); } catch {}
  } catch (e: any) {
    console.error("[datacore] usaspending refresh:", e?.message || e);
  }
}

/** 6h poll — USAspending loads FPDS nightly; extra cycles catch late DoD
 *  publication and corrections (each lands as a new vintage row). */
export function bootContractsPoll(intervalMs = 6 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshContractsCache();
  setInterval(() => { refreshContractsCache(); }, intervalMs).unref?.();
}
