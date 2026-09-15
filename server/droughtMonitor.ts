/**
 * droughtMonitor.ts — US Drought Monitor weekly severity (BUILD ORDER 2 #5,
 * 2026-07-05; access live-probed keyless the same day). Attribution is
 * REQUIRED by the source: "U.S. Drought Monitor (NDMC/USDA/NOAA)" — the
 * USDM is produced by the National Drought Mitigation Center, USDA, and
 * NOAA and is free for any use with credit.
 *
 * HYPOTHESIS (gate-locked in the build order — archive + RAW only):
 * drought-severity DELTAS over the corn/soy/wheat belt lead ag commodity
 * moves and food-producer margins by weeks. AOIs: CONUS + eight ag/water
 * states (belt states by FIPS; the API takes FIPS for states, "us" for
 * CONUS — probed: state abbreviations return empty, FIPS works).
 *
 * D0-D4 are CUMULATIVE area percentages per USDM convention (D0 includes
 * everything drier). One DERIVED field, labeled: dsci = D0+D1+D2+D3+D4,
 * the published Drought Severity and Coverage Index (0-500). Maps are
 * weekly (Tuesday data, Thursday release) and final on publish — dedup by
 * aoi|map_date.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export const DROUGHT_AOIS: Array<{ key: string; api: string; aoi: string }> = [
  { key: "CONUS", api: "USStatistics", aoi: "us" },
  { key: "IA", api: "StateStatistics", aoi: "19" },
  { key: "IL", api: "StateStatistics", aoi: "17" },
  { key: "NE", api: "StateStatistics", aoi: "31" },
  { key: "KS", api: "StateStatistics", aoi: "20" },
  { key: "MN", api: "StateStatistics", aoi: "27" },
  { key: "TX", api: "StateStatistics", aoi: "48" },
  { key: "OK", api: "StateStatistics", aoi: "40" },
  { key: "CA", api: "StateStatistics", aoi: "06" },
];

export interface DroughtRec {
  aoi: string;           // CONUS | state abbreviation
  map_date: string;      // YYYY-MM-DD (Tuesday of the map)
  none: number;
  d0: number;            // cumulative: share in D0 or worse
  d1: number;
  d2: number;
  d3: number;
  d4: number;
  dsci: number;          // DERIVED: d0+d1+d2+d3+d4 (USDM DSCI, 0-500)
  valid_start: string | null;
  valid_end: string | null;
  rt: string;            // as-seen UTC date
}

const pct = (v: any): number | null => {
  const n = typeof v === "number" ? v : parseFloat(v);
  return Number.isFinite(n) && n >= 0 && n <= 100 ? n : null;
};

/** One AOI's JSON array -> records. Rows with any malformed percentage are
 *  dropped whole (a partial severity row would corrupt the DSCI).
 *
 *  LABEL HONESTY (live-caught 2026-07-05): the aoi=us endpoint returns BOTH
 *  "CONUS" and "Total" (US incl. AK/HI/PR) rows per week — a row is kept
 *  only when its OWN label matches the requested AOI; mismatches drop,
 *  they are never relabeled. */
export function parseDrought(json: any, aoiKey: string, rt: string): DroughtRec[] {
  const out: DroughtRec[] = [];
  for (const r of Array.isArray(json) ? json : []) {
    const ownLabel = r?.areaOfInterest ?? r?.stateAbbreviation;
    if (typeof ownLabel === "string" && ownLabel !== aoiKey) continue;
    const md = typeof r?.mapDate === "string" ? r.mapDate.slice(0, 10) : null;
    if (!md) continue;
    const vals = [pct(r?.none), pct(r?.d0), pct(r?.d1), pct(r?.d2), pct(r?.d3), pct(r?.d4)];
    if (vals.some((v) => v == null)) continue;
    const [none, d0, d1, d2, d3, d4] = vals as number[];
    out.push({
      aoi: aoiKey,
      map_date: md,
      none, d0, d1, d2, d3, d4,
      dsci: +(d0 + d1 + d2 + d3 + d4).toFixed(2),
      valid_start: typeof r?.validStart === "string" ? r.validStart.slice(0, 10) : null,
      valid_end: typeof r?.validEnd === "string" ? r.validEnd.slice(0, 10) : null,
      rt,
    });
  }
  return out;
}

// ── Fetch (one request per AOI, polite spacing) ─────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

function windowDates(nowMs: number): { start: string; end: string } {
  const fmt = (ms: number) => {
    const d = new Date(ms);
    return `${d.getUTCMonth() + 1}/${d.getUTCDate()}/${d.getUTCFullYear()}`;
  };
  return { start: fmt(nowMs - 70 * 86400_000), end: fmt(nowMs) };
}

export async function fetchDrought(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<DroughtRec[]> {
  const now = nowMs ?? Date.now();
  const rt = new Date(now).toISOString().slice(0, 10);
  const { start, end } = windowDates(now);
  const out: DroughtRec[] = [];
  let failures = 0;
  for (const a of DROUGHT_AOIS) {
    const url = `https://usdmdataservices.unl.edu/api/${a.api}/GetDroughtSeverityStatisticsByAreaPercent` +
      `?aoi=${a.aoi}&startdate=${encodeURIComponent(start)}&enddate=${encodeURIComponent(end)}&statisticsType=1`;
    try {
      const r = await fetchImpl(url, {
        headers: {
          "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)",
          "Accept": "application/json",
        },
        signal: AbortSignal.timeout(20000) as any,
      });
      if (!r.ok) throw new Error(`-> ${r.status}`);
      out.push(...parseDrought(JSON.parse(await r.text()), a.key, rt));
    } catch (e: any) {
      failures++;
      console.error(`[datacore] drought ${a.key}:`, e?.message || e);
    }
    await new Promise((res) => setTimeout(res, 300)); // polite spacing
  }
  if (failures === DROUGHT_AOIS.length) throw new Error("drought: every AOI failed");
  return out;
}

// ── Archive (dedup aoi|map_date — maps are final on publish) ───────────────

const archivedKeys = new Set<string>();
let seeded = false;

function droughtDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "drought");
}

const keyOf = (r: DroughtRec) => `${r.aoi}|${r.map_date}`;

function seedSeen(dir: string, nowMs: number): void {
  // weekly data over a 70-day fetch window -> seed ~11 weeks of day-files
  for (let i = 0; i < 78; i++) {
    const d = new Date(nowMs - i * 86400_000).toISOString().slice(0, 10);
    for (const fp of [path.join(dir, `${d}.jsonl`), path.join(dir, `${d}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { archivedKeys.add(keyOf(JSON.parse(line))); } catch {}
      }
    }
  }
}

export function archiveDrought(recs: DroughtRec[], baseDir?: string, nowMs?: number): number {
  const dir = droughtDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = recs.filter((r) => !archivedKeys.has(keyOf(r)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((r) => JSON.stringify(r)).join("\n") + "\n");
    fresh.forEach((r) => archivedKeys.add(keyOf(r)));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] drought archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldDroughtDays(baseDir?: string, nowMs?: number): number {
  const dir = droughtDir(baseDir);
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

/** Archived map_dates, newest first (jsonl or jsonl.gz only) — same bounded
 *  read pattern as appStoreRankings.ts's/fdaEvents.ts's own
 *  listArchived*Dates, needed for refreshDroughtCache's
 *  cold-cache-no-disk-backfill restore below. */
export function listArchivedDroughtDates(baseDir?: string, limit = 90): string[] {
  let files: string[];
  try { files = fs.readdirSync(droughtDir(baseDir)); } catch { return []; }
  return files
    .map((f) => f.match(/^(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$/))
    .filter((m): m is RegExpMatchArray => !!m)
    .map((m) => m[1])
    .sort()
    .reverse()
    .slice(0, limit);
}

/** Read one archived day back (plain or gz). */
export function readArchivedDroughtDay(iso: string, baseDir?: string): DroughtRec[] {
  const dir = droughtDir(baseDir);
  for (const fp of [path.join(dir, `${iso}.jsonl`), path.join(dir, `${iso}.jsonl.gz`)]) {
    let text: string | null = null;
    try {
      text = fp.endsWith(".gz")
        ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
        : fs.readFileSync(fp, "utf8");
    } catch { continue; }
    const out: DroughtRec[] = [];
    for (const line of text.split("\n")) {
      if (!line) continue;
      try { out.push(JSON.parse(line)); } catch { continue; }
    }
    return out;
  }
  return [];
}

// ── Cache + poll ────────────────────────────────────────────────────────────

let cache: { at: number; drought: DroughtRec[] } | null = null;
let polling = false;

export function latestDrought() {
  return cache;
}

/** Test-only: clears the module-level cache/dedup state so a test can
 *  exercise a genuine cold-boot scenario (same shape as fdaEvents.ts's own
 *  `_resetFdaForTests`) instead of inheriting state from an earlier test in
 *  this same process. */
export function _resetDroughtForTests(): void {
  cache = null;
  archivedKeys.clear();
  seeded = false;
}

export async function refreshDroughtCache(fetchImpl: FetchFn = fetch as any, nowMs?: number, baseDir?: string): Promise<void> {
  // fetchDrought THROWS (not just returns []) when every AOI fails —
  // caught here, not left to the outer swallow, so a total-outage cycle
  // still reaches the archive/restore logic below instead of exiting
  // early (unlike fdaEvents.ts's fetchFdaEvents, which never throws; this
  // module's own per-AOI failure count intentionally does, see fetchDrought
  // above, so the two siblings need slightly different plumbing here even
  // though the cache-restore contract itself is identical).
  let drought: DroughtRec[] = [];
  try {
    drought = await fetchDrought(fetchImpl, nowMs);
  } catch (e: any) {
    console.error("[datacore] drought refresh:", e?.message || e);
  }
  if (drought.length) cache = { at: Date.now(), drought };
  try { archiveDrought(drought, baseDir, nowMs); } catch {}
  try { gzipOldDroughtDays(baseDir, nowMs); } catch {}
  // COLD-CACHE-NO-DISK-BACKFILL FIX (2026-09-15, same shape as
  // occVolume.ts/cftcCot.ts/appStoreRankings.ts/fdaEvents.ts's own
  // "!cache && archived" branch): the pre-fix `if (drought.length ||
  // !cache) cache = ...` set cache to an EMPTY, non-warming_up result the
  // moment a live fetch returned zero rows on a cold boot (every AOI
  // request failing/timing out at once) — silently masking a real
  // transport failure as "checked, nothing pending" instead of restoring
  // the real multi-week archive already on disk. Only ever fires when the
  // live fetch produced nothing AND cache is still unset; a live fetch
  // that succeeds always wins, and a transient empty fetch during steady
  // state (cache already populated) leaves the existing cache alone, both
  // unchanged from before.
  if (!cache) {
    const dates = listArchivedDroughtDates(baseDir, 1);
    if (dates.length) {
      const restored = readArchivedDroughtDay(dates[0], baseDir);
      if (restored.length) cache = { at: Date.now(), drought: restored };
    }
  }
}

/** 24h poll — the map is weekly (Thursday release); dedup makes the daily
 *  check a cheap no-op most days. Eager boot per KNOWN BROKEN #9. */
export function bootDroughtPoll(intervalMs = 24 * 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshDroughtCache();
  setInterval(() => { refreshDroughtCache(); }, intervalMs).unref?.();
}
