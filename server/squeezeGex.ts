/**
 * squeezeGex.ts — SqueezeMetrics' public DIX.csv (date, price, dix, gex: their
 * dark-pool index and dealer-gamma ESTIMATE), READ IN MEMORY on a 6h cadence,
 * plus the pre-registered trailing-252 percentile rank R. Edge P1
 * (research/edge_preregistration.md § P1; gate-2 PASS 2026-10-28,
 * research/edge_p1_dealer_gamma_result.md).
 *
 * NOTHING OF THEIR DATA IS STORED ([REPAIR] 2026-11-06, research/wishlist.md
 * "SQUEEZEMETRICS DATA TERMS"). Their Terms of Service grant no licence and say
 * users may not "systematically retrieve data … to create or compile … a
 * collection, compilation, database, or directory without written
 * permission". From 2026-10-29 to 2026-11-06 this module compiled the file
 * into a daily archive (<archive>/squeeze_gex/<year>.jsonl). Now each poll
 * fetches the file — which itself carries the full daily history, 2011-05-02
 * onward (3,903 rows, checked live 2026-11-06) — computes the dealer-gamma
 * filter's inputs from that history in memory, and drops the rows. The one
 * file written is the filter's DERIVED summary (SQUEEZE_SUMMARY_FILE: latest
 * date, row count, rank R, the GEX sign, the fetch time — no SqueezeMetrics
 * value or row), which dealer_gamma.py reads. The old archive directory is
 * deleted once, at boot, and the deletion is an audit row.
 * RE-ENABLE (only with SqueezeMetrics' written permission): restore archiveSqueezeRows from git history (removed 2026-11-06) and delete removeSqueezeArchive.
 *
 * INTERNAL ONLY: no /data route, no /api/v1 surface; the diag probe is a
 * status (dates, counts, R, refresh health), and the streams inventory never
 * peeks this stream (its manifest sets "peek": false).
 *
 * R is bit-for-bit the registered definition: percentile rank of G_t in
 * the trailing 252 rows ending at t, inclusive, (below + 0.5*ties_excl_self)
 * / 251; null for the first 251 rows. Mirrors trailing_pct_rank() in
 * scripts/edge_p1_dealer_gamma.py (tests pin the same fixtures).
 */

import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";

export const SQUEEZE_URL = "https://squeezemetrics.com/monitor/static/DIX.csv";
export const RANK_WINDOW = 252;
export const SQUEEZE_SOURCE = "SqueezeMetrics DIX.csv (published estimate, not ours)";
/** The derived summary, at <DATA_DIR>/ (dealer_gamma.SUMMARY_FILE reads it). */
export const SQUEEZE_SUMMARY_FILE = "voltrade_squeeze_gex_summary.json";

export interface SqueezeRow {
  date: string;       // YYYY-MM-DD, the SPX session the values describe
  price: number;      // SPX close
  dix: number;
  gex: number;        // $ gamma per 1% move (estimate)
}

/** `date,price,dix,gex` -> rows sorted by date; malformed lines skipped,
 *  never filled. Duplicate dates keep the last occurrence. */
export function parseSqueezeCsv(csv: string): SqueezeRow[] {
  const lines = csv.split(/\r?\n/);
  if (!lines.length || !lines[0].trim().toLowerCase().startsWith("date,")) return [];
  const byDate = new Map<string, SqueezeRow>();
  for (const line of lines.slice(1)) {
    if (!line.trim()) continue;
    const c = line.split(",");
    if (c.length < 4 || !/^\d{4}-\d{2}-\d{2}$/.test(c[0])) continue;
    const price = parseFloat(c[1]), dix = parseFloat(c[2]), gex = parseFloat(c[3]);
    if (![price, dix, gex].every(Number.isFinite)) continue;
    byDate.set(c[0], { date: c[0], price, dix, gex });
  }
  return Array.from(byDate.values()).sort((a, b) => a.date.localeCompare(b.date));
}

/** Trailing percentile rank, uses only values[<= t]. null until a full window. */
export function trailingPctRank(values: number[], window = RANK_WINDOW): (number | null)[] {
  const out: (number | null)[] = [];
  for (let t = 0; t < values.length; t++) {
    if (t < window - 1) { out.push(null); continue; }
    const v = values[t];
    let below = 0, ties = 0;
    for (let i = t - window + 1; i <= t; i++) {
      if (values[i] < v) below++;
      else if (values[i] === v) ties++;
    }
    out.push((below + 0.5 * (ties - 1)) / (window - 1));
  }
  return out;
}

// ── The derived summary: the only thing written ──────────────────────────────

export interface SqueezeSummary {
  v: 1;
  source: string;
  fetchedAt: string;           // when the file was read for this summary
  asOf: string | null;         // the file's latest row (the SPX session it describes)
  firstDate: string | null;
  rows: number;                // dated rows in the file
  rank252: number | null;      // R of the latest row; null for fewer than 252 rows
  rankRows: number;            // rows with a defined R
  gexNegative: boolean | null; // sign of the latest GEX (the value itself is not kept)
}

/** The filter's inputs from the file's own history (pure). The same numbers
 *  the archive path derived from the same rows: R of the last row looks back
 *  over the last 252 rows only, so it is computed on those. */
export function summarizeSqueeze(rows: SqueezeRow[], fetchedAt: string): SqueezeSummary {
  const last = rows[rows.length - 1];
  const rank252 = rows.length >= RANK_WINDOW
    ? trailingPctRank(rows.slice(-RANK_WINDOW).map((r) => r.gex))[RANK_WINDOW - 1]
    : null;
  return {
    v: 1,
    source: SQUEEZE_SOURCE,
    fetchedAt,
    asOf: last?.date ?? null,
    firstDate: rows[0]?.date ?? null,
    rows: rows.length,
    rank252,
    rankRows: Math.max(0, rows.length - RANK_WINDOW + 1),
    gexNegative: last ? last.gex < 0 : null,
  };
}

/** <DATA_DIR>/SQUEEZE_SUMMARY_FILE: Python's storage_config.DATA_DIR as this
 *  process sees it (the rule dealerGamma.dealerGammaDataDir and
 *  streamsInventory.defaultVolumeFiles use). */
export function squeezeSummaryPath(env: NodeJS.ProcessEnv = process.env): string {
  return path.join(env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp"), SQUEEZE_SUMMARY_FILE);
}

/** Atomic (tmp + rename), so the filter never reads a half-written file. */
function writeSqueezeSummary(fp: string, summary: SqueezeSummary): void {
  fs.mkdirSync(path.dirname(fp), { recursive: true });
  const tmp = `${fp}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(summary) + "\n");
  fs.renameSync(tmp, fp);
}

export function readSqueezeSummary(fp = squeezeSummaryPath()): SqueezeSummary | null {
  try {
    const parsed = JSON.parse(fs.readFileSync(fp, "utf8"));
    return parsed && parsed.v === 1 ? (parsed as SqueezeSummary) : null;
  } catch {
    return null; // absent until the first successful read of the file
  }
}

// ── One-time boot cleanup of the old archive ─────────────────────────────────

export interface SqueezeArchiveCleanup {
  at: string;
  path: string;
  removed: boolean;
  files: number;
  bytes: number;
  error: string | null;
}

/** Deletes <archive>/squeeze_gex — the copy of SqueezeMetrics' file compiled
 *  2026-10-29..11-06, recomputable from the live file — and nothing else (a
 *  symlink there is removed as a link; its target is never touched). null
 *  when it is not there, i.e. every boot after the first. */
export function removeSqueezeArchive(baseDir?: string, nowMs = Date.now()): SqueezeArchiveCleanup | null {
  const dir = path.join(baseDir || archiveBaseDir(), "squeeze_gex");
  const at = new Date(nowMs).toISOString();
  let st: fs.Stats;
  try {
    st = fs.lstatSync(dir);
  } catch (e) {
    if ((e as NodeJS.ErrnoException)?.code === "ENOENT") return null;
    return { at, path: dir, removed: false, files: 0, bytes: 0, error: e instanceof Error ? e.message : String(e) };
  }
  let files = 0, bytes = 0;
  try {
    if (st.isDirectory()) {
      for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
        if (!e.isFile()) continue;
        files++;
        try { bytes += fs.statSync(path.join(dir, e.name)).size; } catch { continue; }
      }
      fs.rmSync(dir, { recursive: true, force: true });
    } else {
      fs.unlinkSync(dir);
    }
    return { at, path: dir, removed: true, files, bytes, error: null };
  } catch (e) {
    return { at, path: dir, removed: false, files, bytes, error: e instanceof Error ? e.message : String(e) };
  }
}

export type SqueezeAudit = (type: string, message: string) => void;
let lastCleanup: SqueezeArchiveCleanup | null = null;

/** The boot step: remove the old archive if it is there and say so — console
 *  AND the audit sink (SQUEEZE-ARCHIVE-REMOVED, or SQUEEZE-ARCHIVE-ERROR with
 *  the next boot retrying). Nothing to remove -> nothing logged. */
export function runSqueezeArchiveCleanup(audit?: SqueezeAudit, baseDir?: string, nowMs = Date.now()): SqueezeArchiveCleanup | null {
  const r = removeSqueezeArchive(baseDir, nowMs);
  if (!r) return null;
  lastCleanup = r;
  const type = r.removed ? "SQUEEZE-ARCHIVE-REMOVED" : "SQUEEZE-ARCHIVE-ERROR";
  const message = r.removed
    ? `deleted ${r.path} (${r.files} file(s), ${r.bytes} bytes): SqueezeMetrics' DIX.csv is no longer archived (their terms bar compiling it without written permission); the dealer-gamma filter reads the live file's own history in memory, so no input is lost`
    : `could not delete ${r.path}: ${r.error} — the next boot retries; nothing new is written there`;
  if (r.removed) console.log(`[datacore] squeeze_gex ${type}: ${message}`);
  else console.error(`[datacore] squeeze_gex ${type}: ${message}`);
  if (audit) {
    try { audit(type, message); } catch (e) { console.error("[datacore] squeeze_gex audit sink threw:", e instanceof Error ? e.message : e); }
  }
  return r;
}

// ── Status (diag probe squeeze_gex) ──────────────────────────────────────────

export interface SqueezeReport {
  probe: "squeeze_gex";
  stored: false;                   // nothing of SqueezeMetrics' data is kept
  rows: number;                    // dated rows in the file last read
  first_date: string | null;
  latest_date: string | null;
  latest_age_days: number | null;
  latest_gex_negative: boolean | null;
  latest_rank: number | null;      // R_t; null inside the 251-row burn-in
  rank_rows: number;               // rows with a defined R
  fetched_at: string | null;       // when the file was last read
  last_refresh_ok: boolean | null; // this process's last poll
  last_error: string | null;
  archive_cleanup: SqueezeArchiveCleanup | null; // this boot's deletion of the old archive, if it ran
  note: string;
}

let latest: SqueezeSummary | null = null;
let lastRefreshOk: boolean | null = null;
let lastError: string | null = null;

/** The latest summary this process computed, else the one on disk (a restart
 *  reports the last read until its own first poll lands). */
export function squeezeReport(nowMs = Date.now(), summaryPath = squeezeSummaryPath()): SqueezeReport {
  const s = latest ?? readSqueezeSummary(summaryPath);
  return {
    probe: "squeeze_gex",
    stored: false,
    rows: s?.rows ?? 0,
    first_date: s?.firstDate ?? null,
    latest_date: s?.asOf ?? null,
    latest_age_days: s?.asOf ? Math.floor((nowMs - Date.parse(s.asOf + "T00:00:00Z")) / 86_400_000) : null,
    latest_gex_negative: s?.gexNegative ?? null,
    latest_rank: s?.rank252 ?? null,
    rank_rows: s?.rankRows ?? 0,
    fetched_at: s?.fetchedAt ?? null,
    last_refresh_ok: lastRefreshOk,
    last_error: lastError,
    archive_cleanup: lastCleanup,
    note: "status only: SqueezeMetrics' file is read in memory every 6h and not stored (their terms); the dealer-gamma filter reads the derived summary (rank, date, sign) — no raw value is kept or served",
  };
}

// ── Fetch + poll ─────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: unknown) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
let polling = false;

export function _resetSqueezeForTests(): void { latest = null; lastRefreshOk = null; lastError = null; lastCleanup = null; polling = false; }

function refreshFailed(why: string): null {
  console.error(`[datacore] squeeze_gex: ${why}`);
  lastRefreshOk = false;
  lastError = why.slice(0, 300);
  return null;
}

/** One poll: fetch the file, summarize its own history in memory, write the
 *  summary. A failed poll writes nothing (the last summary stands and ages:
 *  the filter goes inactive once it is stale). */
export async function refreshSqueezeGex(fetchImpl: FetchFn = fetch as any, nowMs = Date.now(), summaryPath = squeezeSummaryPath()): Promise<SqueezeSummary | null> {
  try {
    const r = await fetchImpl(SQUEEZE_URL, {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(30000) as any,
    });
    if (!r.ok) return refreshFailed(`HTTP ${r.status}`);
    const rows = parseSqueezeCsv(await r.text());
    if (!rows.length) return refreshFailed("no parseable rows");
    const summary = summarizeSqueeze(rows, new Date(nowMs).toISOString());
    try {
      writeSqueezeSummary(summaryPath, summary);
    } catch (e) {
      return refreshFailed(`summary not written: ${e instanceof Error ? e.message : String(e)}`);
    }
    latest = summary;
    lastRefreshOk = true;
    lastError = null;
    return summary;
  } catch (e) {
    return refreshFailed(`refresh: ${e instanceof Error ? e.message : String(e)}`);
  }
}

/** Source publishes after the close; a 6h poll lands the new row within hours.
 *  The old archive is deleted first (once; logged through `audit`). */
export function bootSqueezeGexPoll(opts: { intervalMs?: number; audit?: SqueezeAudit } = {}): void {
  if (polling) return;
  polling = true;
  runSqueezeArchiveCleanup(opts.audit);
  refreshSqueezeGex().catch((e) => console.error("[datacore] squeeze_gex boot:", e?.message || e));
  setInterval(() => { refreshSqueezeGex(); }, opts.intervalMs ?? 6 * 60 * 60_000).unref?.();
}
