/**
 * settlementStress.ts — SETTLEMENT-STRESS COMPOSITE.
 *
 * Queued 2026-07-07 in wishlist.md's DATACORE MAXIMUS block as soon as its
 * three ingredients landed (census builds #4/#6): "ingredients all
 * recording (finrathreshold + secftd + finrashortvol) ... needs the
 * env-gated backfills for depth — design can precede data." Built this
 * session — a pure downstream JOIN over three already-archived, already-
 * independently-verified free streams:
 *   - finrathreshold (server/finraQuery.ts):   daily Reg SHO threshold list
 *   - finrashortvol  (server/finraShortVolume.ts): daily short-sale volume
 *   - secftd         (server/secFtd.ts):        half-month SEC fails-to-deliver
 *
 * HYPOTHESIS (unchanged from the census #4/#6 gate-lock, restated here):
 * threshold-list persistence x FTD balance DELTA x short-vol ratio
 * percentile identifies settlement stress none of the three sources shows
 * alone. PRIOR (stated before this build, Reasoning Standard #10): the
 * three source populations are large and mostly disjoint, so the daily
 * three-way intersection will be a short (likely single/low-double-digit)
 * watchlist, not a broad screen — if the intersection turns out empty for
 * weeks running, that is itself a finding (either genuinely rare overlap,
 * or a join-key mismatch worth re-checking).
 *
 * ROOT VALIDATION LADDER: this module is GATE-1 feature construction only.
 * composite_score is a ranking aid for the eventual gate-2 correlation
 * test (composite vs. forward N-day returns, vs. a same-universe random-
 * entry base rate per Reasoning Standard #3) — NOT a claimed trading
 * signal, not wired into deep_score/any order path, and not surfaced on
 * /data (RAW OVERLAYS vs SIGNALS: an unvalidated derived score is neither
 * — it stays internal until it clears gate 2). A future session should
 * read the archived datacore/settlementstress/ JSONL files directly once
 * enough dated history has accumulated, the same way every other
 * rejected-decision shadow bucket in this codebase is eventually read.
 *
 * ZERO INCREMENTAL COST: no network calls of its own — it only joins
 * files its three ingredient pollers already wrote locally. Runs off the
 * same boot-poll cadence as the rest of the datacore battery.
 */
import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";
import { readPartition } from "./finraQuery";
import { readArchivedDay, FLOOR_TOTAL_VOL } from "./finraShortVolume";
import { readFtdPeriod, type FtdRow } from "./secFtd";

const THRESHOLD_DIR = "finrathreshold";
const OUT_DIR = "settlementstress";

export interface CompositeRow {
  date: string;
  symbol: string;
  name: string;
  persistence_days: number;
  ftd_qty: number;
  ftd_delta: number;
  short_ratio: number;
  short_vol_percentile: number;
  composite_score: number;
}

function outDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), OUT_DIR);
}
function threshDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), THRESHOLD_DIR);
}

/** Archived finrathreshold trade dates, ascending. Gaps are possible (poll
 *  cadence, not a fixed daily guarantee) — persistence below counts only
 *  what's actually archived, never assumes a missing day was "still on
 *  the list." */
export function listThresholdDates(baseDir?: string): string[] {
  let files: string[];
  try { files = fs.readdirSync(threshDir(baseDir)); } catch { return []; }
  return files
    .map((f) => f.match(/^(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$/))
    .filter((m): m is RegExpMatchArray => !!m)
    .map((m) => m[1])
    .sort();
}

/** Consecutive-day persistence per symbol as of `asOf`, counted backward
 *  over archived threshold dates <= asOf. Only symbols present ON asOf
 *  get an entry. An archive gap breaks the streak (honest undercount,
 *  never fabricated). */
export function thresholdPersistence(
  asOf: string,
  baseDir?: string,
): Map<string, { persistence_days: number; name: string }> {
  const dates = listThresholdDates(baseDir).filter((d) => d <= asOf);
  const perDate: Array<Set<string>> = [];
  const names = new Map<string, string>();
  for (const d of dates) {
    const rows = readPartition(THRESHOLD_DIR, d, baseDir);
    const syms = new Set<string>();
    for (const r of rows) {
      const sym = r.issueSymbolIdentifier;
      if (!sym) continue;
      syms.add(sym);
      if (!names.has(sym)) names.set(sym, r.issueName || "");
    }
    perDate.push(syms);
  }
  const out = new Map<string, { persistence_days: number; name: string }>();
  if (!dates.length || dates[dates.length - 1] !== asOf) return out;
  const today = perDate[perDate.length - 1];
  for (const sym of Array.from(today)) {
    let count = 0;
    for (let i = perDate.length - 1; i >= 0; i--) {
      if (perDate[i].has(sym)) count++;
      else break;
    }
    out.set(sym, { persistence_days: count, name: names.get(sym) || "" });
  }
  return out;
}

/** Half-month period string (YYYYMM + a|b) covering an ISO date — mirrors
 *  secFtd.ts's own verified boundary: 1st-14th = 'a', 15th-EOM = 'b'. */
export function periodForDate(iso: string): string {
  const [y, m, d] = iso.split("-");
  return `${y}${m}${Number(d) <= 14 ? "a" : "b"}`;
}

/** The preceding half-month period string. */
export function prevPeriod(period: string): string {
  const y = Number(period.slice(0, 4));
  const m = Number(period.slice(4, 6));
  const half = period.slice(6);
  if (half === "b") return `${period.slice(0, 6)}a`;
  const pm = m === 1 ? 12 : m - 1;
  const py = m === 1 ? y - 1 : y;
  return `${py}${String(pm).padStart(2, "0")}b`;
}

/** Per-symbol newest-settlement-date FTD qty within a period (qty is a
 *  BALANCE/level per secFtd.ts — the newest date in the period is the
 *  period's current reading for that symbol). */
function newestQtyPerSymbol(rows: FtdRow[]): Map<string, number> {
  const newest = new Map<string, { date: string; qty: number }>();
  for (const r of rows) {
    const cur = newest.get(r.symbol);
    if (!cur || r.date > cur.date) newest.set(r.symbol, { date: r.date, qty: r.qty });
  }
  const out = new Map<string, number>();
  newest.forEach((v, k) => out.set(k, v.qty));
  return out;
}

/** FTD balance + delta vs. the prior half-month for the period covering
 *  `asOf`. Empty map if the covering period isn't archived yet (never
 *  interpolated). A missing PRIOR period is treated as a 0 balance
 *  (early-history limitation — the delta is then just the current
 *  balance — stated here, not silently assumed away). */
export function ftdDeltas(asOf: string, baseDir?: string): Map<string, { qty: number; delta: number }> {
  const period = periodForDate(asOf);
  const curRows = readFtdPeriod(period, baseDir);
  if (!curRows.length) return new Map();
  const prevRows = readFtdPeriod(prevPeriod(period), baseDir);
  const curQty = newestQtyPerSymbol(curRows);
  const prevQty = newestQtyPerSymbol(prevRows);
  const out = new Map<string, { qty: number; delta: number }>();
  curQty.forEach((qty, sym) => out.set(sym, { qty, delta: qty - (prevQty.get(sym) ?? 0) }));
  return out;
}

/** Short-vol ratio + cross-sectional percentile (0-100, 100=highest ratio)
 *  among eligible symbols on `date` — reuses finrashortvol's own
 *  liquidity floor so "eligible" means the same thing in both streams. */
export function shortVolPercentiles(
  date: string,
  baseDir?: string,
): Map<string, { short_ratio: number; percentile: number }> {
  const rows = readArchivedDay(date, baseDir);
  const eligible = rows.filter((r) => r.total_vol >= FLOOR_TOTAL_VOL && r.total_vol > 0);
  const ratios = eligible.map((r) => ({ symbol: r.symbol, ratio: r.short_vol / r.total_vol }));
  ratios.sort((a, b) => a.ratio - b.ratio);
  const n = ratios.length;
  const out = new Map<string, { short_ratio: number; percentile: number }>();
  ratios.forEach((r, i) => {
    out.set(r.symbol, {
      short_ratio: Math.round(r.ratio * 10000) / 10000,
      percentile: n > 1 ? Math.round((i / (n - 1)) * 100) : 100,
    });
  });
  return out;
}

/** Joint-extremity composite for one date: the intersection of symbols
 *  present in ALL THREE archives that day (on the threshold list AND
 *  carrying an archived FTD balance for the covering period AND an
 *  eligible short-vol reading). composite_score = persistence_days x
 *  (short_vol_percentile/100) x sign(ftd_delta)*log1p(|ftd_delta|) — a
 *  ranking aid only; see module docstring re: gate 1 vs gate 2. */
export function computeComposite(date: string, baseDir?: string): CompositeRow[] {
  const persistence = thresholdPersistence(date, baseDir);
  if (!persistence.size) return [];
  const ftd = ftdDeltas(date, baseDir);
  if (!ftd.size) return [];
  const svPct = shortVolPercentiles(date, baseDir);
  if (!svPct.size) return [];
  const out: CompositeRow[] = [];
  persistence.forEach((p, sym) => {
    const f = ftd.get(sym);
    const sv = svPct.get(sym);
    if (!f || !sv) return; // gate-1 join: score only symbols present in all three
    const ftdMag = Math.sign(f.delta) * Math.log1p(Math.abs(f.delta));
    out.push({
      date,
      symbol: sym,
      name: p.name,
      persistence_days: p.persistence_days,
      ftd_qty: f.qty,
      ftd_delta: f.delta,
      short_ratio: sv.short_ratio,
      short_vol_percentile: sv.percentile,
      composite_score: Math.round(p.persistence_days * (sv.percentile / 100) * ftdMag * 100) / 100,
    });
  });
  out.sort((a, b) => Math.abs(b.composite_score) - Math.abs(a.composite_score));
  return out;
}

// ── Archive (date-level dedup; small file, plain JSONL) ─────────────────────

const archivedDates = new Set<string>();
let seeded = false;

function seedSeen(baseDir?: string): void {
  if (seeded) return;
  try {
    for (const f of fs.readdirSync(outDir(baseDir))) {
      const m = f.match(/^(\d{4}-\d{2}-\d{2})\.jsonl$/);
      if (m) archivedDates.add(m[1]);
    }
  } catch {}
  seeded = true;
}

export function isCompositeArchived(date: string, baseDir?: string): boolean {
  seedSeen(baseDir);
  return archivedDates.has(date);
}

/** Archive one date's composite rows (may be an empty array — a genuine
 *  "no three-way overlap today" finding is worth recording, not hidden).
 *  Returns rows written; 0 if the date is already archived. */
export function archiveComposite(date: string, rows: CompositeRow[], baseDir?: string): number {
  seedSeen(baseDir);
  if (archivedDates.has(date)) return 0;
  try {
    const dir = outDir(baseDir);
    fs.mkdirSync(dir, { recursive: true });
    const payload = rows.length ? rows.map((r) => JSON.stringify(r)).join("\n") + "\n" : "";
    fs.writeFileSync(path.join(dir, `${date}.jsonl`), payload);
    archivedDates.add(date);
    return rows.length;
  } catch (e: any) {
    console.error(`[datacore] settlementstress archive ${date}:`, e?.message || e);
    return 0;
  }
}

export function _resetSettlementStressForTests(): void {
  archivedDates.clear();
  seeded = false;
}

/** One refresh pass: for every archived threshold date not yet processed
 *  (oldest first, capped per call so a poll never does unbounded work),
 *  attempt the join. A date is marked archived (and thus never retried)
 *  ONLY when its short-vol day AND covering FTD period are both already
 *  archived by their own pollers — otherwise it's left for the next pass
 *  once those catch up. Returns the list of dates newly archived. */
export function refreshSettlementStress(baseDir?: string, maxDates = 30): string[] {
  const done: string[] = [];
  const candidates = listThresholdDates(baseDir).filter((d) => !isCompositeArchived(d, baseDir));
  for (const date of candidates.slice(0, maxDates)) {
    const svRows = readArchivedDay(date, baseDir);
    const ftdRows = readFtdPeriod(periodForDate(date), baseDir);
    if (!svRows.length || !ftdRows.length) continue; // ingredients not caught up yet — retry later
    const rows = computeComposite(date, baseDir);
    archiveComposite(date, rows, baseDir);
    done.push(date);
  }
  return done;
}

let intervalHandle: ReturnType<typeof setInterval> | null = null;

/** Boot-poll entry point (mirrors bootFtdPoll/bootShortVolPoll/
 *  bootFinraQueryPoll's pattern in server/routes.ts). Runs after those
 *  three so this pass usually finds fresh ingredient data waiting; no
 *  network calls of its own either way. */
export function bootSettlementStressPoll(intervalMs = 6 * 60 * 60_000): void {
  const run = () => {
    try {
      const done = refreshSettlementStress();
      if (done.length) console.log(`[datacore] settlementstress: archived ${done.length} date(s)`);
    } catch (e: any) {
      console.error("[datacore] settlementstress poll:", e?.message || e);
    }
  };
  run();
  if (intervalHandle) clearInterval(intervalHandle);
  intervalHandle = setInterval(run, intervalMs);
}
