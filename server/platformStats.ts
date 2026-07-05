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

export async function platformStats(
  layers: Array<{ id: string; status: string }>,
  base = archiveBaseDir(),
  nowMs = Date.now(),
): Promise<{ layers_live: number; layers_total: number; streams_recording: number; observations: number; observations_as_of: string | null }> {
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
  return {
    layers_live: live.length,
    layers_total: layers.length,
    streams_recording: streamsRecording(live, dirs),
    observations: cache?.observations ?? 0,
    observations_as_of: cache ? new Date(cache.at).toISOString() : null,
  };
}

/** test hook */
export function _resetPlatformStatsCache() { cache = null; inflight = null; }
