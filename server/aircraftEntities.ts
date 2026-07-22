/**
 * aircraftEntities.ts — aircraft entity-continuity spine v1 (GIP Part 3b;
 * BUILD ORDER 2 #1, 2026-07-05). Two halves:
 *
 * 1. ARCHIVED-HEX ENUMERATION: stream the aircraft position archive
 *    (gz-aware, never loads a file into memory) and reduce it to one row
 *    per distinct icao24 — first/last seen, point count, callsigns seen
 *    (capped), most recent type designator. This is the join key list the
 *    session-side FAA build script consumes; it is NOT the spine itself.
 *    The scan is expensive on a big archive, so it runs at most once per
 *    TTL window and is cached (stale served while a refresh runs).
 *
 * 2. SPINE SERVING: the committed derived artifact
 *    datacore/aircraft/entity_spine.json (built session-side by
 *    scripts/build_entity_spine.py from the FAA Releasable registry —
 *    ONLY archived hexes, never the full 300k-row dump) is served per-hex,
 *    merged with live archive stats when the cache is warm. Missing
 *    artifact degrades gracefully to spine: null on every hex.
 *
 * Every spine record carries the GIP evidence envelope: source, retrieved
 * date, and match basis (exact Mode S hex) — inferences carry evidence,
 * never bare facts.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import readline from "readline";
import { archiveBaseDir } from "./datacoreArchive";
// [REPAIR 2026-07-05] The runtime Docker image copies dist/ but NOT
// datacore/ (frozen Dockerfile) — a runtime disk read of the artifact
// returned null on Railway forever while working locally. Static import
// bundles it into dist like every other datacore JSON (layers, sites,
// powerplants). The fp-based disk path below remains for tests only.
import bundledSpine from "../datacore/aircraft/entity_spine.json";

export interface HexSummary {
  i: string;            // icao24 (lowercase hex)
  n: number;            // archived points
  f: number;            // first seen (epoch s)
  l: number;            // last seen (epoch s)
  c?: string[];         // callsigns seen (max 5)
  ty?: string;          // most recent ICAO type designator, when broadcast
}

export interface SpineRecord {
  n_number: string;
  owner: string;
  registrant_type: string | null;
  mfr: string | null;
  model: string | null;
  year_mfr: number | null;
  cert_issue: string | null;
  expiration: string | null;
  evidence: { source: string; retrieved: string; match: string };
}

const CALLSIGN_CAP = 5;

/** Reduce one archive line into the per-hex accumulator. Exported for tests. */
export function foldLine(acc: Map<string, HexSummary>, line: string): void {
  let r: any;
  try { r = JSON.parse(line); } catch { return; }
  if (!r || typeof r.i !== "string" || !r.i || typeof r.t !== "number") return;
  const hex = r.i.toLowerCase();
  let s = acc.get(hex);
  if (!s) {
    s = { i: hex, n: 0, f: r.t, l: r.t };
    acc.set(hex, s);
  }
  s.n++;
  if (r.t < s.f) s.f = r.t;
  if (r.t > s.l) s.l = r.t;
  if (r.c) {
    if (!s.c) s.c = [];
    if (!s.c.includes(r.c) && s.c.length < CALLSIGN_CAP) s.c.push(r.c);
  }
  if (r.ty && r.t >= s.l) s.ty = r.ty;
}

function foldFile(acc: Map<string, HexSummary>, fp: string): Promise<void> {
  return new Promise((resolve) => {
    try {
      let stream: NodeJS.ReadableStream = fs.createReadStream(fp);
      if (fp.endsWith(".gz")) stream = stream.pipe(zlib.createGunzip());
      const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
      rl.on("line", (l) => { if (l.trim()) foldLine(acc, l); });
      rl.on("close", () => resolve());
      stream.on("error", () => resolve());
      // readline.Interface re-emits a piped-in stream's error on ITSELF too
      // (separate from the stream.on("error", ...) above) — unlistened,
      // that crashes the whole process on a truncated/corrupt .gz. See
      // datacoreArchive.ts's streamJsonlLines for the full writeup.
      rl.on("error", () => resolve());
    } catch { resolve(); }
  });
}

/** Full scan of the aircraft archive -> per-hex summaries. */
export async function distinctArchivedAircraft(base = archiveBaseDir()): Promise<HexSummary[]> {
  const dir = path.join(base, "aircraft");
  let files: string[] = [];
  try {
    files = fs.readdirSync(dir).filter((f) => f.endsWith(".jsonl") || f.endsWith(".jsonl.gz"));
  } catch { return []; }
  const acc = new Map<string, HexSummary>();
  for (const f of files.sort()) await foldFile(acc, path.join(dir, f));
  return Array.from(acc.values()).sort((a, b) => b.n - a.n);
}

// ── cached enumeration ──────────────────────────────────────────────────────
const TTL_MS = 6 * 60 * 60_000;
let cache: { at: number; hexes: HexSummary[] } | null = null;
let inflight: Promise<HexSummary[]> | null = null;

export async function archivedAircraftCached(base = archiveBaseDir(), nowMs = Date.now()): Promise<{ hexes: HexSummary[]; as_of: string }> {
  if (!cache || nowMs - cache.at > TTL_MS) {
    if (!inflight) {
      inflight = distinctArchivedAircraft(base).then((hexes) => {
        cache = { at: Date.now(), hexes };
        inflight = null;
        return hexes;
      });
    }
    if (!cache) await inflight; // first call waits; later calls serve stale
  }
  return { hexes: cache?.hexes ?? [], as_of: cache ? new Date(cache.at).toISOString() : "" };
}

export function _resetAircraftEntityCache() { cache = null; inflight = null; spine = undefined; }

// ── spine artifact ──────────────────────────────────────────────────────────
let spine: Record<string, SpineRecord> | null | undefined;

export function loadEntitySpine(fp?: string): Record<string, SpineRecord> | null {
  if (spine !== undefined) return spine;
  if (fp === undefined) {
    // production path: the bundled artifact (see import note above)
    const doc = bundledSpine as any;
    spine = doc?.entities && typeof doc.entities === "object" ? doc.entities : null;
    return spine ?? null;
  }
  try {
    const doc = JSON.parse(fs.readFileSync(fp, "utf8"));
    spine = doc?.entities && typeof doc.entities === "object" ? doc.entities : null;
  } catch {
    spine = null; // missing/corrupt test file — every lookup degrades to null
  }
  return spine ?? null;
}

export function entityForHex(hex: string, fp?: string): SpineRecord | null {
  const s = loadEntitySpine(fp);
  if (!s) return null;
  return s[hex.toLowerCase()] || null;
}
