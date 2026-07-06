/**
 * streamsInventory.ts — DATACORE MAXIMUS Phase 4: the streams inventory
 * aggregate behind the /data "Streams" tab.
 *
 * Design (2026-07-06): join the STATIC manifest envelope
 * (datacore/manifests/*.json — source, license, attribution, started,
 * cadence, confidence_model) with DYNAMIC facts scanned from the archive
 * directories (file count, bytes, newest-file age, latest-record peek).
 * The aggregator enumerates the manifest directory at runtime, so every
 * manifested stream appears in the inventory mechanically — no hardcoded
 * stream list to fall stale. That property is pinned by the
 * inventory-coverage ratchet in streamsInventory.test.ts (Phase 5).
 *
 * Event-loop rule: the request path serves a cache only; the scan (which
 * gunzips newest files for peeks) runs on a timer + eager boot, fully
 * async, one stream at a time.
 *
 * HEALTH HONESTY: "health" is DERIVED from newest-file age vs. a cadence
 * keyword heuristic (a weekly stream 3 days quiet is healthy; an hourly
 * stream 3 days quiet is stale). The raw age is always reported alongside
 * so the derivation can be second-guessed. Streams with no archive dir
 * report "no-data" (key-gated, warming, or never landed) — never hidden.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import { promisify } from "util";
import { archiveBaseDir } from "./datacoreArchive";

const gunzip = promisify(zlib.gunzip);

// process.cwd(), NOT import.meta.url: the server ships as a CJS bundle
// (dist/index.cjs) where import.meta doesn't survive esbuild. Runtime
// repo-file access via cwd is the proven pattern (platformStats
// sentinel2Freshness, audit defect #9 repair).
export const MANIFEST_DIR_DEFAULT = path.resolve(process.cwd(), "datacore", "manifests");

// Peek safety: never load more than this many compressed bytes for the
// latest-record preview — honesty over completeness on jumbo files.
const PEEK_MAX_BYTES = 8 * 1024 * 1024;
const PEEK_CHARS = 700;

export interface StreamInventoryEntry {
  stream: string;
  // static (manifest envelope)
  source: string;
  attribution: string;
  license: string;
  started: string;
  cadence: string;
  hypothesis: string; // confidence_model — what this stream is FOR
  storage: string;
  written_by: string;
  // dynamic (archive scan)
  files: number;
  bytes: number;
  newest_file: string | null;
  age_hours: number | null; // newest file mtime → now
  // records in the NEWEST file only — full-archive record counts already
  // run in platformStats (10-min TTL); duplicating that scan here every
  // 5 min would double the volume read for a number files/bytes proxies.
  records_newest_file: number | null;
  latest_peek: string | null; // last record of newest file, truncated
  peek_note: string | null; // why a peek is absent, when it is
  health: "live" | "recent" | "stale" | "no-data";
  health_note: string;
}

function readManifests(manifestDir: string): Record<string, any> {
  const out: Record<string, any> = {};
  let files: string[] = [];
  try {
    files = fs.readdirSync(manifestDir).filter((f) => f.endsWith(".json"));
  } catch {
    return out;
  }
  for (const f of files) {
    try {
      out[f.replace(/\.json$/, "")] = JSON.parse(fs.readFileSync(path.join(manifestDir, f), "utf8"));
    } catch {
      // a corrupt manifest still gets an inventory row (name only) — the
      // envelope test will fail CI; the inventory must not hide it
      out[f.replace(/\.json$/, "")] = { stream: f.replace(/\.json$/, ""), _corrupt: true };
    }
  }
  return out;
}

// Cadence keyword → stale threshold in hours. Derived-health heuristic
// (see header); the default is deliberately generous — false "stale" on a
// slow stream is noise, false "live" on a dead one is a lie.
function staleThresholdHours(cadence: string): number {
  const c = (cadence || "").toLowerCase();
  if (/annual|year/.test(c)) return 400 * 24;
  if (/quarter/.test(c)) return 110 * 24;
  if (/month/.test(c)) return 40 * 24;
  if (/week/.test(c)) return 10 * 24;
  if (/daily|overnight|day/.test(c)) return 3 * 24;
  if (/hour|minute|min\b|live|real-?time|continuous|\d+\s*h\b/.test(c)) return 26;
  return 8 * 24; // unknown cadence: generous
}

function deriveHealth(ageHours: number | null, cadence: string): { health: StreamInventoryEntry["health"]; note: string } {
  if (ageHours === null) {
    return { health: "no-data", note: "no archive files yet — key-gated, warming up, or first write pending" };
  }
  const stale = staleThresholdHours(cadence);
  if (ageHours <= Math.min(stale, 26)) return { health: "live", note: `newest file ${ageHours.toFixed(1)}h old` };
  if (ageHours <= stale) return { health: "recent", note: `newest file ${ageHours.toFixed(1)}h old (within cadence ~${Math.round(stale)}h)` };
  return { health: "stale", note: `newest file ${ageHours.toFixed(1)}h old — exceeds cadence-derived threshold ~${Math.round(stale)}h` };
}

async function peekNewest(dir: string, newest: string | null): Promise<{ peek: string | null; records: number | null; note: string | null }> {
  if (!newest) return { peek: null, records: null, note: "no files" };
  const fp = path.join(dir, newest);
  try {
    const st = await fs.promises.stat(fp);
    if (st.size > PEEK_MAX_BYTES) {
      return { peek: null, records: null, note: `peek skipped: newest file ${(st.size / 1e6).toFixed(1)}MB > ${(PEEK_MAX_BYTES / 1e6).toFixed(0)}MB cap` };
    }
    let buf = await fs.promises.readFile(fp);
    if (newest.endsWith(".gz")) buf = (await gunzip(buf)) as Buffer;
    const text = buf.toString("utf8");
    const lines = text.split("\n").filter((l) => l.trim());
    const records = lines.length;
    if (!records) return { peek: null, records: 0, note: "newest file empty" };
    const line = lines[lines.length - 1].trim();
    // parse to prove it's a record; re-serialize truncated for display
    try {
      JSON.parse(line);
      return { peek: line.length > PEEK_CHARS ? line.slice(0, PEEK_CHARS) + "…" : line, records, note: null };
    } catch {
      return { peek: null, records, note: "newest file tail is not valid JSON (partial write in progress?)" };
    }
  } catch (e: any) {
    return { peek: null, records: null, note: `peek failed: ${e?.code || e?.message || "unknown"}` };
  }
}

export async function buildStreamsInventory(
  baseDir?: string,
  manifestDir?: string,
  now: number = Date.now(),
): Promise<{ time: number; count: number; streams: StreamInventoryEntry[] }> {
  const base = baseDir || archiveBaseDir();
  const manifests = readManifests(manifestDir || MANIFEST_DIR_DEFAULT);
  const streams: StreamInventoryEntry[] = [];
  for (const name of Object.keys(manifests).sort()) {
    const m = manifests[name];
    const dir = path.join(base, name);
    let files = 0, bytes = 0, newest: string | null = null, newestMtime = 0;
    try {
      for (const f of (await fs.promises.readdir(dir)).sort()) {
        let st;
        try { st = await fs.promises.stat(path.join(dir, f)); } catch { continue; }
        if (!st.isFile()) continue;
        files++; bytes += st.size;
        newest = f; // sorted ascending; name order == time order for day/hour files
        newestMtime = st.mtimeMs;
      }
    } catch {
      // dir missing → no-data row (never hidden)
    }
    const ageHours = newest ? (now - newestMtime) / 3.6e6 : null;
    const { health, note } = deriveHealth(ageHours, m.cadence || "");
    const { peek, records, note: peekNote } = files > 0
      ? await peekNewest(dir, newest)
      : { peek: null, records: null, note: "no files" };
    streams.push({
      stream: name,
      source: String(m.source || "").slice(0, 500),
      attribution: String(m.attribution || ""),
      license: String(m.license || "").slice(0, 300),
      started: String(m.started || ""),
      cadence: String(m.cadence || "").slice(0, 300),
      hypothesis: String(m.confidence_model || "").slice(0, 500),
      storage: String(m.storage || "").slice(0, 300),
      written_by: String(m.written_by || "").slice(0, 200),
      files, bytes,
      newest_file: newest,
      age_hours: ageHours === null ? null : Math.round(ageHours * 10) / 10,
      records_newest_file: records,
      latest_peek: peek,
      peek_note: peekNote,
      health,
      health_note: note,
    });
  }
  return { time: now, count: streams.length, streams };
}

// ── cache + poller (event-loop rule: routes serve the cache only) ───────────
let cache: { time: number; count: number; streams: StreamInventoryEntry[] } | null = null;
let building = false;
let timer: ReturnType<typeof setInterval> | null = null;

export function getStreamsInventoryCached() {
  return cache; // null until first scan lands → route reports warming_up
}

async function refresh(): Promise<void> {
  if (building) return;
  building = true;
  try {
    cache = await buildStreamsInventory();
  } catch (e: any) {
    console.error(`[streamsInventory] scan failed: ${e?.message || e}`);
  } finally {
    building = false;
  }
}

export function bootStreamsInventoryPoll(intervalMs = 5 * 60 * 1000): void {
  if (timer) return;
  void refresh(); // eager boot
  timer = setInterval(() => void refresh(), intervalMs);
  (timer as any).unref?.();
}

export function _resetStreamsInventoryForTests(): void {
  cache = null;
  building = false;
  if (timer) { clearInterval(timer); timer = null; }
}
