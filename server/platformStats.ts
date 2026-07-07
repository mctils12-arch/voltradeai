/**
 * platformStats.ts — honest, self-updating platform counters for the
 * landing hero (hero-refinements directive 2026-07-05). NOTHING here is
 * hardcoded to today's numbers: layers come from the live registry,
 * streams from registry + what the archive is actually recording, and
 * observations are REAL line counts across the archive (the previous
 * hero summed a `samples` field prod never returned — hence the dash).
 *
 * Counting 80+ MB of JSONL(.gz) per request would be rude; the count runs
 * at most once per TTL window and is cached (stale value served while a
 * refresh runs — stale beats spinner, per the house rule).
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import readline from "readline";
import { archiveBaseDir } from "./datacoreArchive";
import { repoDataPath } from "./repoFiles";

/** Archive dir -> registry layer id, where the names differ. */
export const DIR_LAYER_MAP: Record<string, string> = {
  filings: "insider",
  earnings8k: "earnings",
};
/** Operational streams — recorded, but not data products; never counted. */
export const OPERATIONAL_DIRS = new Set(["waitlist", "apiusage"]);

/** Distinct ingestion streams = every LIVE registry layer (each live layer
 *  is a live feed/surface: weather fields, radar, port dwell, plants, …)
 *  plus archive streams that have no map layer of their own (e.g. option
 *  chains). Grows automatically as layers and archives are added. */
export function streamsRecording(liveLayerIds: string[], archiveDirs: string[]): number {
  const mapped = new Set(liveLayerIds);
  let extra = 0;
  for (const d of archiveDirs) {
    if (OPERATIONAL_DIRS.has(d) || d.endsWith("_tracks")) continue;
    if (!mapped.has(DIR_LAYER_MAP[d] || d)) extra++;
  }
  return liveLayerIds.length + extra;
}

export function listArchiveDirs(base = archiveBaseDir()): string[] {
  try {
    return fs.readdirSync(base, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => e.name)
      .filter((n) => {
        try {
          return fs.readdirSync(path.join(base, n)).some((f) => f.endsWith(".jsonl") || f.endsWith(".jsonl.gz"));
        } catch { return false; }
      });
  } catch { return []; }
}

/** Count newline-terminated records in one archive file (gz-aware,
 *  streaming — never loads a file into memory). */
export function countFileLines(fp: string): Promise<number> {
  return new Promise((resolve) => {
    try {
      let stream: NodeJS.ReadableStream = fs.createReadStream(fp);
      if (fp.endsWith(".gz")) stream = stream.pipe(zlib.createGunzip());
      const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
      let n = 0;
      rl.on("line", (l) => { if (l.trim()) n++; });
      rl.on("close", () => resolve(n));
      stream.on("error", () => resolve(0));
    } catch { resolve(0); }
  });
}

/** Real total archived records across all non-operational streams. */
export async function countObservations(base = archiveBaseDir()): Promise<number> {
  let total = 0;
  for (const dir of listArchiveDirs(base)) {
    if (OPERATIONAL_DIRS.has(dir) || dir.endsWith("_tracks")) continue;
    const dp = path.join(base, dir);
    let files: string[] = [];
    try { files = fs.readdirSync(dp).filter((f) => f.endsWith(".jsonl") || f.endsWith(".jsonl.gz")); } catch {}
    for (const f of files) total += await countFileLines(path.join(dp, f));
  }
  return total;
}

// ── cached assembly ─────────────────────────────────────────────────────────
const TTL_MS = 10 * 60_000;
let cache: { at: number; observations: number } | null = null;
let inflight: Promise<number> | null = null;

/** [REPAIR 2026-07-05, audit defect #9] Sentinel-2 readings are git-side
 *  (session-run pipeline; Railway never runs it) — a stall was invisible.
 *  Surfacing the newest reading date + age here makes staleness a finding
 *  every DAILY health check and dashboard can see. Reads the bundled repo
 *  file; parse failure returns nulls, never throws. */
// [REPAIR 2026-07-07] repoDataPath, not bare cwd: the production image
// has no working-tree datacore/ — this default silently resolved to a
// missing file on Railway, so prod served sentinel2_last_reading:null
// since this surface shipped. The dist/datacore copy (script/build.ts)
// is the runtime source there.
export function sentinel2Freshness(nowMs = Date.now(), fp = repoDataPath("datacore/sentinel2/readings.jsonl")): { last: string | null; age_days: number | null } {
  try {
    const lines = fs.readFileSync(fp, "utf8").trim().split("\n");
    let last: string | null = null;
    for (const line of lines) {
      try {
        const d = JSON.parse(line)?.date;
        if (d && (!last || d > last)) last = d;
      } catch {}
    }
    if (!last) return { last: null, age_days: null };
    return { last, age_days: Math.floor((nowMs - Date.parse(last)) / 86400_000) };
  } catch {
    return { last: null, age_days: null };
  }
}

export async function platformStats(
  layers: Array<{ id: string; status: string }>,
  base = archiveBaseDir(),
  nowMs = Date.now(),
): Promise<{ layers_live: number; layers_total: number; streams_recording: number; observations: number; observations_as_of: string | null; sentinel2_last_reading: string | null; sentinel2_reading_age_days: number | null }> {
  const live = layers.filter((l) => l.status === "live").map((l) => l.id);
  const dirs = listArchiveDirs(base);
  if (!cache || nowMs - cache.at > TTL_MS) {
    if (!inflight) {
      inflight = countObservations(base).then((n) => {
        cache = { at: Date.now(), observations: n };
        inflight = null;
        return n;
      });
    }
    if (!cache) await inflight; // first call: wait; later calls: serve stale
  }
  const s2 = sentinel2Freshness(nowMs);
  return {
    layers_live: live.length,
    layers_total: layers.length,
    streams_recording: streamsRecording(live, dirs),
    observations: cache?.observations ?? 0,
    observations_as_of: cache ? new Date(cache.at).toISOString() : null,
    sentinel2_last_reading: s2.last,
    sentinel2_reading_age_days: s2.age_days,
  };
}

/** test hook */
export function _resetPlatformStatsCache() { cache = null; inflight = null; }
