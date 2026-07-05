/**
 * cbpBorderWait.ts — CBP land-border wait times (commercial + passenger)
 * (BUILD ORDER 5 #5, filed + probed 2026-07-05).
 *
 * Source: bwt.cbp.gov/api/bwtnew — keyless JSON, 83 crossings (US
 * government work, public domain; attribution "CBP Border Wait Times").
 * The RSS path named at filing turned out to serve an HTML SPA (a
 * status-200 probe is not a body probe — lesson recorded); bwtnew is the
 * real feed.
 *
 * LOCALE HONESTY: the API localizes STATUS STRINGS by serving region
 * (our probe egress got Spanish — "Abierto", "Sin demora"; prod may get
 * English). Parsing therefore keys ONLY on locale-independent fields
 * (numeric delay_minutes / lanes_open, port identifiers, structure);
 * status strings are archived VERBATIM as published, never translated
 * or matched against.
 *
 * Fills the ROAD leg of the freight-proxy gap (sea=AIS, rail=STB,
 * air=ADS-B). HYPOTHESIS (gate-locked): sustained commercial wait
 * anomalies at top freight crossings (Laredo, Otay Mesa, El Paso) lead
 * border-dependent logistics/intermodal volumes. Prior ~20%.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export type LaneClass = "commercial_standard" | "commercial_FAST" | "passenger_standard";

export interface BorderWaitObs {
  port_number: string;
  port_name: string;
  crossing_name: string;
  border: string;          // as published (LOCALIZED by serving region — join on port_number)
  lane: LaneClass;
  status: string | null;   // as published, verbatim (localized)
  delay_min: number | null;
  lanes_open: number | null;
  update_time: string | null; // as published ("En 2:00 pm CDT" / "At 2:00 pm CDT")
  src_date: string | null;    // feed's own date/time stamps
  src_time: string | null;
  rt: string;              // as-seen UTC timestamp
}

const num = (v: any): number | null => {
  if (v == null || v === "") return null;
  const n = parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
};

const str = (v: any): string | null => (v == null || v === "" ? null : String(v));

function laneObs(rec: any, lane: LaneClass, block: any, rt: string): BorderWaitObs | null {
  if (!block || typeof block !== "object") return null;
  return {
    port_number: String(rec.port_number),
    port_name: String(rec.port_name || ""),
    crossing_name: String(rec.crossing_name || ""),
    border: String(rec.border || ""),
    lane,
    status: str(block.operational_status),
    delay_min: num(block.delay_minutes),
    lanes_open: num(block.lanes_open),
    update_time: str(block.update_time),
    src_date: str(rec.date),
    src_time: str(rec.time),
    rt,
  };
}

/** bwtnew JSON -> flattened per-lane-class observations. Records without
 *  a port_number are dropped; lane blocks that are absent contribute
 *  nothing (structure varies by crossing — many have no commercial
 *  lanes at all, which is real, not missing). */
export function parseBorderWaits(json: any, rt: string): BorderWaitObs[] {
  if (!Array.isArray(json)) return [];
  const out: BorderWaitObs[] = [];
  for (const rec of json) {
    if (!rec || !rec.port_number) continue;
    const cv = rec.commercial_vehicle_lanes || {};
    const pv = rec.passenger_vehicle_lanes || {};
    for (const o of [
      laneObs(rec, "commercial_standard", cv.standard_lanes, rt),
      laneObs(rec, "commercial_FAST", cv.FAST_lanes, rt),
      laneObs(rec, "passenger_standard", pv.standard_lanes, rt),
    ]) {
      if (o) out.push(o);
    }
  }
  return out;
}

// ── Fetch ────────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

const API = "https://bwt.cbp.gov/api/bwtnew";

export async function fetchBorderWaits(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<BorderWaitObs[] | null> {
  const rt = new Date(nowMs ?? Date.now()).toISOString();
  try {
    const r = await fetchImpl(API, {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(25000) as any,
    });
    if (!r.ok) {
      console.error(`[datacore] cbpborderwait -> ${r.status}`);
      return null;
    }
    return parseBorderWaits(JSON.parse(await r.text()), rt);
  } catch (e: any) {
    console.error("[datacore] cbpborderwait:", e?.message || e);
    return null;
  }
}

// ── Archive (observation-identity dedup — only CHANGES land on disk) ────────

const archivedKeys = new Set<string>();
let seeded = false;

function bwtDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "cbpborderwait");
}

const keyOf = (o: BorderWaitObs) =>
  `${o.port_number}|${o.crossing_name}|${o.lane}|${o.delay_min}|${o.lanes_open}|${o.status}|${o.update_time}`;

function seedSeen(dir: string, nowMs: number): void {
  for (let i = 0; i < 5; i++) {
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

export function archiveBorderWaits(obs: BorderWaitObs[], baseDir?: string, nowMs?: number): number {
  const dir = bwtDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = obs.filter((o) => !archivedKeys.has(keyOf(o)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.appendFileSync(path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`),
                      fresh.map((o) => JSON.stringify(o)).join("\n") + "\n");
    fresh.forEach((o) => archivedKeys.add(keyOf(o)));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] cbpborderwait archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldBorderWaitDays(baseDir?: string, nowMs?: number): number {
  const dir = bwtDir(baseDir);
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

// ── Cache + poll ────────────────────────────────────────────────────────────

let cache: { at: number; obs: BorderWaitObs[] } | null = null;
let polling = false;

export function latestBorderWaits() {
  return cache;
}

export async function refreshBorderWaits(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<void> {
  try {
    const obs = await fetchBorderWaits(fetchImpl, nowMs);
    if (obs === null) return; // transport error — keep last snapshot
    cache = { at: Date.now(), obs };
    archiveBorderWaits(obs, undefined, nowMs);
    gzipOldBorderWaitDays(undefined, nowMs);
  } catch (e: any) {
    console.error("[datacore] cbpborderwait refresh:", e?.message || e);
  }
}

/** Waits move intra-day at commercial crossings — hourly poll; the
 *  change-only dedup keeps quiet hours nearly free. Eager boot per
 *  KNOWN BROKEN #9. */
export function bootBorderWaitPoll(intervalMs = 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshBorderWaits();
  setInterval(() => { refreshBorderWaits(); }, intervalMs).unref?.();
}
