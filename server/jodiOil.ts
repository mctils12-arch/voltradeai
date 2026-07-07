/**
 * jodiOil.ts — JODI Oil World Database, Primary series (closing stock
 * levels) — DATA CENSUS section 1 #1, filed 2026-07-06, built 2026-07-07.
 *
 * Source: jodidata.org world_Primary_CSV.zip — a single-entry ZIP
 * (verified live 2026-07-07: 23.3MB compressed -> 282.7MB CSV, deflate,
 * one file "NewProcedure_Primary_CSV.csv"). Keyless static download,
 * monthly release (~19th, ~2-month lag), history back to 2002-01.
 * Columns verified live: REF_AREA,TIME_PERIOD,ENERGY_PRODUCT,
 * FLOW_BREAKDOWN,UNIT_MEASURE,OBS_VALUE,ASSESSMENT_CODE. "Primary"
 * covers CRUDEOIL/OTHERCRUDE/NGL/TOTCRUDE only (refined products live
 * in a separate world_Secondary_CSV.zip — a documented follow-up, not
 * built here). Free use with acknowledgment (JODI terms).
 *
 * FILTER (verified live 2026-07-07 against the full real file, all
 * 6,872,400 rows): kept to FLOW_BREAKDOWN=CLOSTLV (closing stock
 * levels) x UNIT_MEASURE=CONVBBL, JODI's own cross-country-comparable
 * conversion (vs. the native-unit columns KBBL/KL/KTONS, which mix
 * incompatible reporting units across countries) — exactly what a
 * comparability signal needs. Every CLOSTLV row carries a CONVBBL
 * field, but only ~42% (57,973/137,448 archived rows, live-counted)
 * carry an actual reported number; the rest are honestly null (that
 * area/product/month was never reported to JODI) — OBS_VALUE also
 * uses 'N/A' and '..' as no-data tokens alongside '-'/'x', all four
 * verified live and all four fall through to null via the same
 * Number.isFinite(parseFloat(...)) check, not a placeholder allowlist.
 * This cuts 6.87M source rows to 137,448 archived rows.
 *
 * HYPOTHESIS (gate-locked, RAW only): non-OECD crude/product stock
 * builds (Saudi/UAE/India etc.) are invisible in EIA's US-only weekly
 * release and thin out Brent structure signals that US-only inventory
 * models miss — a direct gate-1 ground-truth partner for the
 * tank-shadow satellite root (same physical quantity, independent
 * measurement path). Gate 1 = cross-check JODI's own reported levels
 * for a known country/month against JODI's published bulletin; gate 2
 * = JODI stock-build/draw vs forward Brent-structure or product-crack
 * moves, regime-split.
 *
 * ENGINEERING: the source is a full-history snapshot every fetch (no
 * incremental query exists) — downloading + inflating + parsing 283MB
 * is real cost we do NOT want on every poll. A cheap HEAD probe
 * (Last-Modified + ETag) runs on the poll cadence; the expensive
 * fetch+parse only fires when those change from the stored marker,
 * which JODI's own ~monthly cadence makes a ~12x/year event. Archived
 * per TIME_PERIOD (one JSONL file per YYYY-MM): periods within the
 * last RECENT_MONTHS are rewritten on every real refresh (JODI revises
 * recent months); older periods are archived once and never rewritten
 * (treated as final, then gzipped) — bounds both parse-time disk I/O
 * and the gzip churn to the handful of months that can still move.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { unzipSync } from "fflate";
import { archiveBaseDir } from "./datacoreArchive";

const ZIP_URL = "https://www.jodidata.org/_resources/files/downloads/oil-data/world_Primary_CSV.zip";
const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };
const KEPT_FLOW = "CLOSTLV";
const KEPT_UNIT = "CONVBBL";
/** Periods within this many months of "now" are still revisable. */
export const RECENT_MONTHS = 6;

export interface JodiRow {
  ref_area: string;        // JODI 2-letter area code
  time_period: string;     // YYYY-MM
  energy_product: string;  // CRUDEOIL | OTHERCRUDE | NGL | TOTCRUDE
  stock: number | null;    // OBS_VALUE on the CONVBBL basis, as published ('-'/'x' -> null)
  assessment_code: string | null; // JODI ASSESSMENT_CODE, passed through unverified
  rt: string;              // as-seen UTC date
}

export interface JodiMeta {
  etag: string | null;
  last_modified: string | null;
}

// ── CSV parse (manual line-scan on the raw Buffer — the 283MB decompressed
//    text is unavoidable, but scanning bytes and decoding only kept lines
//    avoids also materializing a 6.87M-element split() array of every row) ──

export function parseJodiCsv(buf: Buffer, rt: string): JodiRow[] {
  const out: JodiRow[] = [];
  const NL = 10, CR = 13;
  const len = buf.length;
  let start = 0, headerSkipped = false;
  while (start < len) {
    let end = buf.indexOf(NL, start);
    if (end === -1) end = len;
    let lineEnd = end;
    if (lineEnd > start && buf[lineEnd - 1] === CR) lineEnd--;
    if (!headerSkipped) { headerSkipped = true; start = end + 1; continue; }
    if (lineEnd > start) {
      // cheap pre-filter before decoding+splitting: both kept tokens must be substrings
      const line = buf.toString("latin1", start, lineEnd);
      if (line.indexOf(KEPT_FLOW) !== -1 && line.indexOf(KEPT_UNIT) !== -1) {
        const c = line.split(",");
        if (c.length === 7 && c[3] === KEPT_FLOW && c[4] === KEPT_UNIT) {
          const v = c[5];
          const n = v === "-" || v === "x" || v === "" ? null : parseFloat(v);
          out.push({
            ref_area: c[0],
            time_period: c[1],
            energy_product: c[2],
            stock: n != null && Number.isFinite(n) ? n : null,
            assessment_code: c[6] || null,
            rt,
          });
        }
      }
    }
    start = end + 1;
  }
  return out;
}

// ── Fetch ────────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{
  ok: boolean; status: number;
  headers?: { get(name: string): string | null };
  arrayBuffer(): Promise<ArrayBuffer>;
}>;

/** Cheap HEAD probe — Last-Modified/ETag only, no body transfer. */
export async function probeJodiMeta(fetchImpl: FetchFn = fetch as any): Promise<JodiMeta | null> {
  try {
    const r = await fetchImpl(ZIP_URL, {
      method: "HEAD", headers: UA, signal: AbortSignal.timeout(15000) as any,
    });
    if (!r.ok) return null;
    return {
      etag: r.headers?.get("etag") ?? null,
      last_modified: r.headers?.get("last-modified") ?? null,
    };
  } catch {
    return null;
  }
}

export function metaUnchanged(a: JodiMeta | null, b: JodiMeta | null): boolean {
  if (!a || !b) return false;
  if (!a.etag && !a.last_modified) return false; // nothing to compare against -> treat as changed
  return a.etag === b.etag && a.last_modified === b.last_modified;
}

/** Full download + inflate + parse — the expensive path, only invoked
 *  when the HEAD probe indicates the remote actually changed. */
export async function fetchJodiRows(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<JodiRow[]> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const r = await fetchImpl(ZIP_URL, { headers: UA, signal: AbortSignal.timeout(90000) as any });
  if (!r.ok) throw new Error(`world_Primary_CSV.zip -> ${r.status}`);
  const files = unzipSync(new Uint8Array(await r.arrayBuffer()));
  const name = Object.keys(files)[0];
  if (!name) throw new Error("empty JODI zip");
  return parseJodiCsv(Buffer.from(files[name]), rt);
}

// ── Archive (one JSONL file per TIME_PERIOD; recent months revisable) ───────

function jodiDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "jodioil");
}

function metaPath(baseDir?: string): string {
  return path.join(jodiDir(baseDir), "_meta.json");
}

export function readStoredMeta(baseDir?: string): JodiMeta | null {
  try { return JSON.parse(fs.readFileSync(metaPath(baseDir), "utf8")); } catch { return null; }
}

export function writeStoredMeta(meta: JodiMeta, baseDir?: string): void {
  try {
    fs.mkdirSync(jodiDir(baseDir), { recursive: true });
    fs.writeFileSync(metaPath(baseDir), JSON.stringify(meta));
  } catch (e: any) {
    console.error("[datacore] jodioil meta write:", e?.message || e);
  }
}

export function isRecentPeriod(period: string, nowMs: number): boolean {
  const m = period.match(/^(\d{4})-(\d{2})$/);
  if (!m) return false;
  const periodMs = Date.UTC(Number(m[1]), Number(m[2]) - 1, 1);
  const now = new Date(nowMs);
  const cutoffMs = Date.UTC(now.getUTCFullYear(), now.getUTCMonth() - RECENT_MONTHS, 1);
  return periodMs >= cutoffMs;
}

/** Writes each period's rows; recent periods are always rewritten (JODI
 *  revises them), older periods only if not already archived. Older
 *  periods are gzipped immediately since they will not be touched again.
 *  Returns the number of period-files written. */
export function archiveJodiRows(rows: JodiRow[], baseDir?: string, nowMs?: number): number {
  if (!rows.length) return 0;
  const now = nowMs ?? Date.now();
  const dir = jodiDir(baseDir);
  const byPeriod = new Map<string, JodiRow[]>();
  for (const r of rows) {
    if (!byPeriod.has(r.time_period)) byPeriod.set(r.time_period, []);
    byPeriod.get(r.time_period)!.push(r);
  }
  let written = 0;
  try { fs.mkdirSync(dir, { recursive: true }); } catch {}
  byPeriod.forEach((periodRows: JodiRow[], period: string) => {
    const plainPath = path.join(dir, `${period}.jsonl`);
    const gzPath = `${plainPath}.gz`;
    const recent = isRecentPeriod(period, now);
    if (!recent && (fs.existsSync(plainPath) || fs.existsSync(gzPath))) return;
    try {
      fs.writeFileSync(plainPath, periodRows.map((r) => JSON.stringify(r)).join("\n") + "\n");
      written++;
      if (!recent) {
        fs.writeFileSync(gzPath, zlib.gzipSync(fs.readFileSync(plainPath)));
        fs.unlinkSync(plainPath);
      }
    } catch (e: any) {
      console.error(`[datacore] jodioil archive ${period}:`, e?.message || e);
    }
  });
  return written;
}

export function readArchivedPeriod(period: string, baseDir?: string): JodiRow[] {
  const dir = jodiDir(baseDir);
  for (const fp of [path.join(dir, `${period}.jsonl`), path.join(dir, `${period}.jsonl.gz`)]) {
    let text: string;
    try {
      text = fp.endsWith(".gz")
        ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
        : fs.readFileSync(fp, "utf8");
    } catch { continue; }
    const out: JodiRow[] = [];
    for (const line of text.split("\n")) {
      if (!line) continue;
      try { out.push(JSON.parse(line)); } catch {}
    }
    return out;
  }
  return [];
}

export function listArchivedPeriods(baseDir?: string): string[] {
  const dir = jodiDir(baseDir);
  try {
    const periods = new Set<string>();
    for (const f of fs.readdirSync(dir)) {
      const m = f.match(/^(\d{4}-\d{2})\.jsonl(\.gz)?$/);
      if (m) periods.add(m[1]);
    }
    return Array.from(periods).sort();
  } catch {
    return [];
  }
}

// ── Cache + poll ────────────────────────────────────────────────────────────

let cache: { at: number; period: string; rows: JodiRow[]; periods_archived: number } | null = null;
let polling = false;

export function latestJodi() {
  return cache;
}

/** On restart with no remote change, rebuild the cache from disk (same
 *  honesty rule as cftcCot/finraShortVolume) rather than re-downloading. */
export async function refreshJodi(fetchImpl: FetchFn = fetch as any, nowMs?: number,
                                  baseDir?: string): Promise<void> {
  try {
    const remoteMeta = await probeJodiMeta(fetchImpl);
    const storedMeta = readStoredMeta(baseDir);
    const periods = listArchivedPeriods(baseDir);
    if (remoteMeta && metaUnchanged(remoteMeta, storedMeta) && periods.length) {
      if (!cache) {
        const newest = periods[periods.length - 1];
        cache = { at: Date.now(), period: newest, rows: readArchivedPeriod(newest, baseDir), periods_archived: periods.length };
      }
      return;
    }
    const rows = await fetchJodiRows(fetchImpl, nowMs);
    if (!rows.length) return;
    archiveJodiRows(rows, baseDir, nowMs);
    if (remoteMeta) writeStoredMeta(remoteMeta, baseDir);
    const newestPeriod = rows.reduce((mx, r) => (r.time_period > mx ? r.time_period : mx), "");
    cache = {
      at: Date.now(),
      period: newestPeriod,
      rows: rows.filter((r) => r.time_period === newestPeriod),
      periods_archived: listArchivedPeriods(baseDir).length,
    };
  } catch (e: any) {
    console.error("[datacore] jodioil refresh:", e?.message || e);
  }
}

/** Monthly source, ~2-month publish lag — the HEAD probe is nearly free
 *  and runs daily; the expensive 283MB parse only fires when Last-
 *  Modified/ETag actually changes (~once/month). Eager boot per KNOWN
 *  BROKEN #9 precedent. */
export function bootJodiPoll(intervalMs = 24 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshJodi();
  setInterval(() => { refreshJodi(); }, intervalMs).unref?.();
}
