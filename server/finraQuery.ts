/**
 * finraQuery.ts — FINRA Query API cluster, part 1 of 2: the two
 * settlement-stress datasets (DATACORE MAXIMUS census build #4,
 * contract verified live 2026-07-07 by the API workup — every quirk
 * below was probed, not assumed):
 *
 *  - consolidatedShortInterest: semi-monthly per-symbol short interest
 *    (2017-12-29 →, 204+ settlement-date partitions, ~22k records each),
 *    days-to-cover precomputed by FINRA.
 *  - thresholdList: daily Reg SHO threshold names, OTC side (2016-01-04 →,
 *    ~17 records/day).
 *
 * The ATS venue summaries (weekly/monthly/blocks — 66k-210k records per
 * week) are a SEPARATE follow-up build with their own volume budget.
 *
 * VERIFIED CONTRACT (api.finra.org, keyless):
 *  - POST /data/group/otcMarket/name/<ds> with JSON body is REQUIRED for
 *    server-side filters; GET filter params are silently ignored.
 *  - Accept: application/json → array of objects (avoids the verified
 *    GET-CSV corruption: quoteValues defaults false and commas inside
 *    issueName break rows).
 *  - Pagination: limit max 5000 (silently clamped above), zero-based
 *    offset; the `record-total` response header is the count primitive;
 *    HTTP 204 (not 200) = zero matching records.
 *  - Partitions: GET /partitions/group/otcMarket/name/<ds> lists
 *    available partition values NEWEST FIRST — the archiver's "what
 *    dates exist" primitive. A listed partition can still be EMPTY
 *    (monthlySummary 2014-2016 precedent) — treat 204 as honest no-op.
 *  - Result order without sortFields is not guaranteed, so a partition
 *    is archived only when rows fetched === record-total (truncation +
 *    shuffle guard, the CNMS trailer-count analog).
 *
 * HYPOTHESIS (gate-locked): threshold-list persistence x FTD x daily
 *    short volume = a settlement-stress composite nobody computes;
 *    days-to-cover extremes on small caps join 13F/Form 4 clusters.
 *    Nothing is claimed until ladder gate 2.
 *
 * License: FINRA data, free with attribution ("FINRA Query API").
 * No deep backfill in the boot path (R8 crash-loop lesson): history
 * capture is env-gated FINRA_QUERY_BACKFILL=1, off by default.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const API = "https://api.finra.org";
const GROUP = "otcMarket";
const UA = "voltradeai-datacore/1.0 (+https://voltradeai.com)";
export const PAGE_LIMIT = 5000;
const MAX_PAGES = 12; // 60k records — far above either dataset's per-partition size

type HeadersLike = { get(name: string): string | null };
type QueryFetchFn = (url: string, init?: any) => Promise<{
  ok: boolean; status: number; headers: HeadersLike; text(): Promise<string>;
}>;

// ── Query primitives ─────────────────────────────────────────────────────────

export interface QueryPage {
  status: number;
  recordTotal: number | null; // from the record-total header
  rows: any[];
}

/** One POST page. 204 → {rows:[], recordTotal:0}. Non-OK logs the body
 *  VERBATIM (the error contract echoes the request — that text is the
 *  diagnostic) and returns null for retry-next-poll. */
export async function postQueryPage(
  dataset: string,
  body: Record<string, any>,
  fetchImpl: QueryFetchFn = fetch as any,
): Promise<QueryPage | null> {
  try {
    const r = await fetchImpl(`${API}/data/group/${GROUP}/name/${dataset}`, {
      method: "POST",
      headers: {
        "User-Agent": UA,
        "Accept": "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(45000) as any,
    });
    if (r.status === 204) return { status: 204, recordTotal: 0, rows: [] };
    const text = await r.text();
    if (!r.ok) {
      console.error(`[datacore] finraquery ${dataset} -> ${r.status}: ${text.slice(0, 300)}`);
      return null;
    }
    const rt = r.headers?.get("record-total");
    const rows = JSON.parse(text);
    if (!Array.isArray(rows)) {
      console.error(`[datacore] finraquery ${dataset}: non-array response`);
      return null;
    }
    return { status: r.status, recordTotal: rt != null ? parseInt(rt, 10) : null, rows };
  } catch (e: any) {
    console.error(`[datacore] finraquery ${dataset}:`, e?.message || e);
    return null;
  }
}

/** Partition values for a dataset, NEWEST FIRST, single-key partitions
 *  flattened to their value (e.g. "2026-06-15"). Null on transport error. */
export async function fetchPartitions(
  dataset: string,
  fetchImpl: QueryFetchFn = fetch as any,
): Promise<string[] | null> {
  try {
    const r = await fetchImpl(`${API}/partitions/group/${GROUP}/name/${dataset}`, {
      headers: { "User-Agent": UA, "Accept": "application/json" },
      signal: AbortSignal.timeout(30000) as any,
    });
    const text = await r.text();
    if (!r.ok) {
      console.error(`[datacore] finraquery partitions ${dataset} -> ${r.status}: ${text.slice(0, 200)}`);
      return null;
    }
    const doc = JSON.parse(text);
    const parts = (doc?.availablePartitions || [])
      .map((p: any) => (Array.isArray(p?.partitions) ? p.partitions[0] : null))
      .filter((v: any): v is string => typeof v === "string" && v.length > 0);
    return parts;
  } catch (e: any) {
    console.error(`[datacore] finraquery partitions ${dataset}:`, e?.message || e);
    return null;
  }
}

/** Every row of one partition (EQUAL filter + pagination). Returns null
 *  unless rows collected === record-total — result order without
 *  sortFields is not guaranteed, so a partial read may not be a clean
 *  prefix; only a complete, count-verified set is archive-worthy. */
export async function fetchPartitionRows(
  dataset: string,
  field: string,
  value: string,
  fetchImpl: QueryFetchFn = fetch as any,
): Promise<any[] | null> {
  const rows: any[] = [];
  let expected: number | null = null;
  for (let page = 0; page < MAX_PAGES; page++) {
    const res = await postQueryPage(dataset, {
      limit: PAGE_LIMIT,
      offset: page * PAGE_LIMIT,
      compareFilters: [{ compareType: "EQUAL", fieldName: field, fieldValue: value }],
    }, fetchImpl);
    if (res === null) return null;
    if (res.recordTotal != null) expected = res.recordTotal;
    rows.push(...res.rows);
    if (expected == null) {
      // no count header — accept only a short (single, non-full) page
      if (res.rows.length < PAGE_LIMIT) return rows;
      console.error(`[datacore] finraquery ${dataset} ${value}: full page without record-total header — refusing unverifiable read`);
      return null;
    }
    if (rows.length >= expected) break;
    await new Promise((res2) => setTimeout(res2, 300)); // polite spacing
  }
  if (expected != null && rows.length !== expected) {
    console.error(`[datacore] finraquery ${dataset} ${value}: got ${rows.length} of record-total ${expected} — refusing incomplete partition`);
    return null;
  }
  return rows;
}

// ── Archive (partition-level dedup; one file per partition value) ────────────

const DIRS = {
  shortInterest: "finrashortinterest",
  threshold: "finrathreshold",
} as const;

const archived: Record<string, Set<string>> = {
  [DIRS.shortInterest]: new Set<string>(),
  [DIRS.threshold]: new Set<string>(),
};
const seeded: Record<string, boolean> = {};

function dirPath(dirName: string, baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), dirName);
}

function seedSeen(dirName: string, baseDir?: string): void {
  if (seeded[dirName]) return;
  try {
    for (const f of fs.readdirSync(dirPath(dirName, baseDir))) {
      const m = f.match(/^(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$/);
      if (m) archived[dirName].add(m[1]);
    }
  } catch {}
  seeded[dirName] = true;
}

export function isPartitionArchived(dirName: string, value: string, baseDir?: string): boolean {
  seedSeen(dirName, baseDir);
  return archived[dirName].has(value);
}

/** Write one partition's rows. Short-interest files gz on write (~22k
 *  rows ≈ 3MB plain); threshold days stay plain (~17 rows). Rows carry
 *  an `rt` capture date for vintage honesty — FINRA can revise a
 *  settlement date (revisionFlag field); we keep the first capture and
 *  the manifest says so. */
export function archivePartition(
  dirName: string,
  value: string,
  rows: any[],
  rt: string,
  baseDir?: string,
): number {
  if (!rows.length) return 0;
  seedSeen(dirName, baseDir);
  if (archived[dirName].has(value)) return 0;
  try {
    const dir = dirPath(dirName, baseDir);
    fs.mkdirSync(dir, { recursive: true });
    const payload = rows.map((r) => JSON.stringify({ ...r, rt })).join("\n") + "\n";
    if (dirName === DIRS.shortInterest) {
      fs.writeFileSync(path.join(dir, `${value}.jsonl.gz`), zlib.gzipSync(payload));
    } else {
      fs.writeFileSync(path.join(dir, `${value}.jsonl`), payload);
    }
    archived[dirName].add(value);
    return rows.length;
  } catch (e: any) {
    console.error(`[datacore] finraquery archive ${dirName}/${value}:`, e?.message || e);
    return 0;
  }
}

export function _resetFinraQueryForTests(): void {
  for (const k of Object.keys(archived)) archived[k].clear();
  for (const k of Object.keys(seeded)) delete seeded[k];
  siCache = null;
  polling = false;
}

// ── Summaries (computed at poll time — never on the request path) ───────────

export const SI_TOP_CAP = 15;
export const SI_ADV_FLOOR = 100_000;       // shares/day — keeps illiquid junk out of top days-to-cover
export const SI_POSITION_FLOOR = 1_000_000; // shares — floor for the change leaderboard
// Percent change on a near-zero base explodes (live 2026-07-07: NVRI
// +259,847,500% from a ~0 previous position). Both endpoints of the
// ratio need a floor for the leaderboard to mean anything.
export const SI_PREV_FLOOR = 100_000;       // shares — previous-position floor for change%

export interface ShortInterestSummary {
  settlement_date: string;
  records: number;
  top_days_to_cover: Array<{ symbol: string; name: string; days_to_cover: number; short_qty: number; adv: number }>;
  top_change_pct: Array<{ symbol: string; name: string; change_pct: number; short_qty: number }>;
  adv_floor: number;
  position_floor: number;
  top_cap: number;
}

export function summarizeShortInterest(rows: any[]): ShortInterestSummary | null {
  if (!rows.length) return null;
  const dtc: ShortInterestSummary["top_days_to_cover"] = [];
  const chg: ShortInterestSummary["top_change_pct"] = [];
  for (const r of rows) {
    const symbol = r.symbolCode, name = r.issueName || "";
    if (!symbol) continue;
    const d = Number(r.daysToCoverQuantity), adv = Number(r.averageDailyVolumeQuantity);
    const q = Number(r.currentShortPositionQuantity), c = Number(r.changePercent);
    const prev = Number(r.previousShortPositionQuantity);
    if (Number.isFinite(d) && Number.isFinite(adv) && adv >= SI_ADV_FLOOR) {
      dtc.push({ symbol, name, days_to_cover: d, short_qty: q || 0, adv });
    }
    if (Number.isFinite(c) && Number.isFinite(q) && q >= SI_POSITION_FLOOR
        && Number.isFinite(prev) && prev >= SI_PREV_FLOOR) {
      chg.push({ symbol, name, change_pct: c, short_qty: q });
    }
  }
  dtc.sort((a, b) => b.days_to_cover - a.days_to_cover);
  chg.sort((a, b) => b.change_pct - a.change_pct);
  return {
    settlement_date: rows[0].settlementDate,
    records: rows.length,
    top_days_to_cover: dtc.slice(0, SI_TOP_CAP),
    top_change_pct: chg.slice(0, SI_TOP_CAP),
    adv_floor: SI_ADV_FLOOR,
    position_floor: SI_POSITION_FLOOR,
    top_cap: SI_TOP_CAP,
  };
}

export interface ThresholdSummary {
  trade_date: string;
  count: number;
  symbols: Array<{ symbol: string; name: string; market: string }>;
}

export function summarizeThreshold(rows: any[]): ThresholdSummary | null {
  if (!rows.length) return null;
  return {
    trade_date: rows[0].tradeDate,
    count: rows.length,
    symbols: rows.map((r) => ({
      symbol: r.issueSymbolIdentifier || "",
      name: r.issueName || "",
      market: r.marketCategoryDescription || r.marketClassCode || "",
    })),
  };
}

let siCache: {
  at: number;
  si: ShortInterestSummary | null;
  threshold: ThresholdSummary | null;
} | null = null;
let polling = false;

export function latestFinraSi() {
  return siCache;
}

// ── Refresh (partition diff → archive → summarize newest) ───────────────────

/** Windows: newest N partitions checked per cycle. Small on purpose —
 *  history capture is the env-gated backfill below, never the boot path
 *  (R8 crash-loop lesson). */
export async function refreshFinraQuery(
  fetchImpl: QueryFetchFn = fetch as any,
  nowMs?: number,
  baseDir?: string,
  siWindow = 3,
  thWindow = 6,
): Promise<void> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  try {
    // consolidated short interest — semi-monthly
    const siParts = await fetchPartitions("consolidatedShortInterest", fetchImpl);
    if (siParts && siParts.length) {
      for (const p of siParts.slice(0, siWindow)) {
        if (isPartitionArchived(DIRS.shortInterest, p, baseDir)) continue;
        const rows = await fetchPartitionRows("consolidatedShortInterest", "settlementDate", p, fetchImpl);
        if (rows === null) continue;      // transport/incomplete — retry next poll
        if (rows.length === 0) continue;  // listed-but-empty partition (verified possible)
        archivePartition(DIRS.shortInterest, p, rows, rt, baseDir);
      }
      // newest ARCHIVED partition feeds the cache (survives restarts)
      const newest = siParts.find((p) => isPartitionArchived(DIRS.shortInterest, p, baseDir));
      if (newest && siCache?.si?.settlement_date !== newest) {
        const rows = readPartition(DIRS.shortInterest, newest, baseDir);
        const s = summarizeShortInterest(rows);
        if (s) siCache = { at: Date.now(), si: s, threshold: siCache?.threshold ?? null };
      }
    }
    // threshold list — daily, tiny
    const thParts = await fetchPartitions("thresholdList", fetchImpl);
    if (thParts && thParts.length) {
      for (const p of thParts.slice(0, thWindow)) {
        if (isPartitionArchived(DIRS.threshold, p, baseDir)) continue;
        const rows = await fetchPartitionRows("thresholdList", "tradeDate", p, fetchImpl);
        if (rows === null || rows.length === 0) continue;
        archivePartition(DIRS.threshold, p, rows, rt, baseDir);
      }
      const newest = thParts.find((p) => isPartitionArchived(DIRS.threshold, p, baseDir));
      if (newest && siCache?.threshold?.trade_date !== newest) {
        const s = summarizeThreshold(readPartition(DIRS.threshold, newest, baseDir));
        if (s) siCache = { at: Date.now(), si: siCache?.si ?? null, threshold: s };
      }
    }
  } catch (e: any) {
    console.error("[datacore] finraquery refresh:", e?.message || e);
  }
}

/** Read one archived partition back (gz or plain). Poller context only. */
export function readPartition(dirName: string, value: string, baseDir?: string): any[] {
  const dir = dirPath(dirName, baseDir);
  for (const fp of [path.join(dir, `${value}.jsonl.gz`), path.join(dir, `${value}.jsonl`)]) {
    let text: string | null = null;
    try {
      text = fp.endsWith(".gz")
        ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
        : fs.readFileSync(fp, "utf8");
    } catch { continue; }
    const out: any[] = [];
    for (const line of text.split("\n")) {
      if (!line) continue;
      try { out.push(JSON.parse(line)); } catch {}
    }
    return out;
  }
  return [];
}

// ── History backfill (env-gated, OFF by default — R8 lesson) ────────────────
// Full depth: SI 204 partitions x ~20k rows (~80MB gz total);
// threshold 2,635 daily partitions x ~20 rows (~8MB total, but 2,635
// request-pairs). Opt-in via FINRA_QUERY_BACKFILL=1; runs at most one
// pass (done-marker), gzips as it lands (nothing accumulates plain).

export function backfillEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return env.FINRA_QUERY_BACKFILL === "1";
}

const doneMarker = (baseDir?: string) => path.join(dirPath(DIRS.shortInterest, baseDir), "backfill_done.json");

export async function backfillIfEnabled(
  fetchImpl: QueryFetchFn = fetch as any,
  nowMs?: number,
  baseDir?: string,
  env: NodeJS.ProcessEnv = process.env,
): Promise<void> {
  if (!backfillEnabled(env)) return;
  if (fs.existsSync(doneMarker(baseDir))) return;
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  console.log("[datacore] finraquery backfill: starting full-history capture (SI + threshold)");
  let siN = 0, thN = 0;
  const siParts = (await fetchPartitions("consolidatedShortInterest", fetchImpl)) || [];
  for (const p of siParts) {
    if (isPartitionArchived(DIRS.shortInterest, p, baseDir)) continue;
    const rows = await fetchPartitionRows("consolidatedShortInterest", "settlementDate", p, fetchImpl);
    if (rows && rows.length) siN += archivePartition(DIRS.shortInterest, p, rows, rt, baseDir);
    await new Promise((res) => setTimeout(res, 500));
  }
  const thParts = (await fetchPartitions("thresholdList", fetchImpl)) || [];
  for (const p of thParts) {
    if (isPartitionArchived(DIRS.threshold, p, baseDir)) continue;
    const rows = await fetchPartitionRows("thresholdList", "tradeDate", p, fetchImpl);
    if (rows && rows.length) thN += archivePartition(DIRS.threshold, p, rows, rt, baseDir);
    await new Promise((res) => setTimeout(res, 300));
  }
  try {
    fs.writeFileSync(doneMarker(baseDir), JSON.stringify({
      done_rt: new Date().toISOString(), si_rows: siN, threshold_rows: thN,
    }));
  } catch {}
  console.log(`[datacore] finraquery backfill done: ${siN} SI rows, ${thN} threshold rows`);
}

/** SI publishes semi-monthly (~T+9), threshold nightly — 6h poll catches
 *  both same-day. Eager boot; backfill only on explicit env opt-in. */
export function bootFinraQueryPoll(intervalMs = 6 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshFinraQuery().then(() => backfillIfEnabled())
    .catch((e) => console.error("[datacore] finraquery boot:", e?.message || e));
  setInterval(() => { refreshFinraQuery(); }, intervalMs).unref?.();
}
