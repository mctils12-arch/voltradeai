// LARGE METEORS (NASA/JPL CNEOS fireballs) — archiver + snapshot for the
// /data "Large meteors (NASA)" layer (human-approved mockup 2026-08-13:
// plain naming, direction streaks where NASA publishes the velocity
// vector, viewer-local times, per-event coverage links).
//
// SOURCE: https://ssd-api.jpl.nasa.gov/fireball.api (US Government sensors,
// public domain). Bolides — big meteors that exploded in the atmosphere —
// published AFTER the fact, a few dozen per year worldwide. RAW overlay
// (no predictive claim, source attributed) per the RAW-vs-SIGNALS rule.
//
// ACCUMULATION (BUILD-FIRST rule 2): the API serves history, but our store
// merges every poll into a volume-persisted archive anyway — independence
// from upstream availability costs one small JSON file, and the store is
// the layer's serving path (the API is polled 4×/day, never per-request).
//
// DIRECTION: CNEOS publishes the pre-entry velocity vector (vx,vy,vz, km/s,
// geocentric Earth-fixed) for roughly half the events. The ground-track
// heading is the projection of that vector onto the local east/north axes
// at the event point. Events without a vector get NO heading — the client
// draws no streak for them (never an invented direction).

import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";

export const METEORS_URL = "https://ssd-api.jpl.nasa.gov/fireball.api?limit=500&vel-comp=true";
export const METEORS_POLL_MS = 6 * 3_600_000; // 4×/day — events land ~weekly
/** newest-event age above which the layer must label itself quiet, not
 *  broken (events are sparse: 30 days without one is NORMAL). */
export const METEORS_QUIET_AFTER_DAYS = 60;

export interface MeteorEvent {
  /** unix seconds of peak brightness (CNEOS date, UTC). */
  t: number;
  date: string;          // "YYYY-MM-DD hh:mm:ss" UTC, verbatim from CNEOS
  la: number;
  lo: number;
  /** total radiated (flash) energy, 1e10 J, as published. */
  e: number;
  /** impact (blast) energy, kt TNT, as published. */
  imp: number;
  /** peak-brightness altitude km, or null when not published. */
  alt: number | null;
  /** entry speed km/s, or null when not published. */
  vel: number | null;
  /** ground-track heading degrees (direction of travel), or null when the
   *  velocity vector wasn't published — no streak then. */
  hdg: number | null;
}

/** Ground-track heading (deg, 0=N, 90=E) from the geocentric Earth-fixed
 *  velocity components at (laDeg, loDeg): project v onto local east/north. */
export function groundHeading(
  vx: number, vy: number, vz: number, laDeg: number, loDeg: number,
): number {
  const lam = (loDeg * Math.PI) / 180;
  const phi = (laDeg * Math.PI) / 180;
  const ve = -Math.sin(lam) * vx + Math.cos(lam) * vy;
  const vn = -Math.sin(phi) * Math.cos(lam) * vx - Math.sin(phi) * Math.sin(lam) * vy + Math.cos(phi) * vz;
  return ((Math.atan2(ve, vn) * 180) / Math.PI + 360) % 360;
}

/** Parse the CNEOS payload (fields-array format) into events. Rows without
 *  coordinates are dropped (they exist — energy-only detections); every
 *  other missing field stays null, never invented. */
export function parseFireballs(json: any): MeteorEvent[] {
  const fields: string[] = Array.isArray(json?.fields) ? json.fields : [];
  const rows: any[] = Array.isArray(json?.data) ? json.data : [];
  if (!fields.length || !rows.length) return [];
  const out: MeteorEvent[] = [];
  for (const row of rows) {
    const r: Record<string, any> = {};
    fields.forEach((f, i) => { r[f] = row[i]; });
    if (r.lat == null || r.lon == null || !r.date) continue;
    const la = Number(r.lat) * (r["lat-dir"] === "S" ? -1 : 1);
    const lo = Number(r.lon) * (r["lon-dir"] === "W" ? -1 : 1);
    if (!Number.isFinite(la) || !Number.isFinite(lo)) continue;
    const tMs = Date.parse(String(r.date).replace(" ", "T") + "Z");
    if (!Number.isFinite(tMs)) continue;
    let hdg: number | null = null;
    if (r.vx != null && r.vy != null && r.vz != null) {
      const vx = Number(r.vx), vy = Number(r.vy), vz = Number(r.vz);
      if (Number.isFinite(vx) && Number.isFinite(vy) && Number.isFinite(vz)) {
        hdg = Math.round(groundHeading(vx, vy, vz, la, lo) * 10) / 10;
      }
    }
    out.push({
      t: Math.floor(tMs / 1000),
      date: String(r.date),
      la, lo,
      e: Number(r.energy) || 0,
      imp: Number(r["impact-e"]) || 0,
      alt: r.alt != null && Number.isFinite(Number(r.alt)) ? Number(r.alt) : null,
      vel: r.vel != null && Number.isFinite(Number(r.vel)) ? Number(r.vel) : null,
      hdg,
    });
  }
  out.sort((a, b) => b.t - a.t);
  return out;
}

/** date+position identifies a physical event across polls. */
export const eventKey = (e: MeteorEvent): string => `${e.date}|${e.la}|${e.lo}`;

/** Merge fresh events into the store (newer poll wins on key collision —
 *  CNEOS occasionally revises energies). Sorted newest-first. */
export function mergeEvents(existing: MeteorEvent[], fresh: MeteorEvent[]): MeteorEvent[] {
  const byKey = new Map<string, MeteorEvent>();
  for (const e of existing) byKey.set(eventKey(e), e);
  for (const e of fresh) byKey.set(eventKey(e), e);
  return Array.from(byKey.values()).sort((a, b) => b.t - a.t);
}

export function storePath(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "meteors.json");
}

export function loadStore(baseDir?: string): MeteorEvent[] {
  try {
    const d = JSON.parse(fs.readFileSync(storePath(baseDir), "utf-8"));
    return Array.isArray(d?.events) ? d.events : [];
  } catch { return []; }
}

export function saveStore(events: MeteorEvent[], baseDir?: string): void {
  const p = storePath(baseDir);
  try {
    fs.mkdirSync(path.dirname(p), { recursive: true });
    fs.writeFileSync(p, JSON.stringify({ saved_at: Date.now(), events }));
  } catch (e: any) {
    console.error("[meteors] save:", e?.message || e);
  }
}

export interface MeteorsSnapshot {
  events: MeteorEvent[];
  fetched_at: number | null;   // unix ms of the last successful poll
  last_error: string | null;
  with_direction: number;
}

/** Boot the poller. Store-first: serves the archived events immediately
 *  even if the first poll fails (Freshness Law: cached state first). */
export function startMeteorsPoller(deps: {
  fetchImpl: typeof fetch;
  baseDir?: string;
  onCycle?: (count: number) => void;
}): { getSnapshot: () => MeteorsSnapshot; stop: () => void } {
  let events = loadStore(deps.baseDir);
  let fetchedAt: number | null = null;
  let lastError: string | null = null;
  let stopped = false;
  const tick = async () => {
    if (stopped) return;
    try {
      const res = await deps.fetchImpl(METEORS_URL, {
        headers: { "User-Agent": "voltradeai-datacore/1.0 (mctils12@gmail.com)" },
        signal: AbortSignal.timeout(30000) as any,
      });
      if (!res.ok) throw new Error(`CNEOS ${res.status}`);
      const fresh = parseFireballs(await res.json());
      if (fresh.length) {
        events = mergeEvents(events, fresh);
        saveStore(events, deps.baseDir);
      }
      fetchedAt = Date.now();
      lastError = null;
      deps.onCycle?.(events.length);
    } catch (e: any) {
      lastError = String(e?.message || e);
      console.error("[meteors] poll:", lastError);
    }
  };
  const timer = setInterval(() => { void tick(); }, METEORS_POLL_MS);
  (timer as any).unref?.();
  void tick();
  return {
    getSnapshot: () => ({
      events,
      fetched_at: fetchedAt,
      last_error: lastError,
      with_direction: events.filter((e) => e.hdg != null).length,
    }),
    stop: () => { stopped = true; clearInterval(timer); },
  };
}
