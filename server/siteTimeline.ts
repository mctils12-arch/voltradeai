/**
 * siteTimeline.ts — Everything Graph R1: per-strategic-site event timeline
 * (BUILD ORDER 3 #5, [PRODUCT], 2026-07-05). The first user-visible
 * cross-stream join, composed ENTIRELY from archives we already record:
 * NWS alerts, FIRMS fire detections, USGS gauge readings, and our own
 * aircraft/vessel position density near each site. Raw overlay
 * composition — every element is already labeled at its source stream, no
 * predictive claim, no ladder gate (standing raw-vs-signals rule).
 *
 * One scan pass computes ALL sites' timelines together (the expensive
 * part is reading the position archives — reading them 16x per site
 * would be absurd), cached 6h with stale-served refresh like the other
 * archive-derived surfaces.
 *
 * HONESTY: density counts are ARCHIVED-POINT counts under adaptive
 * thinning (near-site traffic is sampled at full resolution by design —
 * see datacoreArchive.aircraftIntervalMs — so near-site day-over-day
 * comparisons are fair); absent days are absent, not zero; the window is
 * the raw-retention week the archive guarantees.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import readline from "readline";
import { archiveBaseDir } from "./datacoreArchive";

export interface SiteRef { id: string; name: string; lat: number; lon: number }

export interface TimelineEvent {
  t: string;                 // ISO date (or datetime where the source has it)
  kind: "alert" | "fire" | "gauge";
  label: string;
  severity?: string | null;  // alerts
  value?: number | null;     // gauges / fire confidence-count
}

export interface SiteTimeline {
  site: string;
  events: TimelineEvent[];   // newest first, capped (stated in payload)
  density: Record<string, { a: number; v: number }>; // day -> archived points near site
}

export const RADIUS_KM = 50;
export const WINDOW_DAYS = 7;
export const EVENTS_CAP = 12;

const kmBetween = (aLat: number, aLon: number, bLat: number, bLon: number) => {
  const R = 6371, dLat = ((bLat - aLat) * Math.PI) / 180, dLon = ((bLon - aLon) * Math.PI) / 180;
  const s = Math.sin(dLat / 2) ** 2 +
    Math.cos((aLat * Math.PI) / 180) * Math.cos((bLat * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
};

function lastDays(nowMs: number, n = WINDOW_DAYS): string[] {
  const out: string[] = [];
  for (let i = 0; i < n; i++) out.push(new Date(nowMs - i * 86400_000).toISOString().slice(0, 10));
  return out;
}

function readJsonlDay(dir: string, day: string): any[] {
  const out: any[] = [];
  for (const fp of [path.join(dir, `${day}.jsonl`), path.join(dir, `${day}.jsonl.gz`)]) {
    let text: string | null = null;
    try {
      text = fp.endsWith(".gz")
        ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
        : fs.readFileSync(fp, "utf8");
    } catch { continue; }
    for (const line of text.split("\n")) {
      if (!line) continue;
      try { out.push(JSON.parse(line)); } catch {}
    }
  }
  return out;
}

/** Fold the event-stream archives (small day-files) for all sites. */
export function foldEventStreams(sites: SiteRef[], base: string, nowMs: number): Map<string, TimelineEvent[]> {
  const out = new Map<string, TimelineEvent[]>(sites.map((s) => [s.id, []]));
  const days = lastDays(nowMs);
  for (const day of days) {
    for (const rec of readJsonlDay(path.join(base, "nwsalerts"), day)) {
      if (rec.lat == null || rec.lon == null) continue; // zone-only alerts have no geo — excluded, per stream honesty
      for (const s of sites) {
        if (kmBetween(rec.lat, rec.lon, s.lat, s.lon) <= RADIUS_KM) {
          out.get(s.id)!.push({ t: rec.sent || day, kind: "alert", label: rec.event, severity: rec.severity ?? null });
        }
      }
    }
    for (const rec of readJsonlDay(path.join(base, "fires"), day)) {
      if (rec.lat == null || rec.lon == null) continue; // FireDetection shape (nasaFirms.ts)
      for (const s of sites) {
        if (kmBetween(rec.lat, rec.lon, s.lat, s.lon) <= RADIUS_KM) {
          out.get(s.id)!.push({
            t: rec.acq_date || day, kind: "fire",
            label: `Fire detection (${rec.confidence || "unknown"} confidence)`,
            value: rec.frp ?? null,
          });
        }
      }
    }
    for (const rec of readJsonlDay(path.join(base, "usgswater"), day)) {
      if (rec.lat == null || rec.lon == null) continue;
      for (const s of sites) {
        if (kmBetween(rec.lat, rec.lon, s.lat, s.lon) <= RADIUS_KM) {
          out.get(s.id)!.push({ t: rec.d || day, kind: "gauge", label: rec.name || rec.site, value: rec.v ?? null });
        }
      }
    }
  }
  return out;
}

/** Stream one position archive kind, incrementing per-site per-day counts. */
function foldPositions(kind: "aircraft" | "vessels", field: "a" | "v", sites: SiteRef[],
                       density: Map<string, Record<string, { a: number; v: number }>>,
                       base: string, days: Set<string>): Promise<void> {
  const dir = path.join(base, kind);
  let files: string[] = [];
  try {
    files = fs.readdirSync(dir).filter((f) =>
      (f.endsWith(".jsonl") || f.endsWith(".jsonl.gz")) && days.has(f.slice(0, 10)));
  } catch { return Promise.resolve(); }
  const chain = files.sort().map((f) => () => new Promise<void>((resolve) => {
    try {
      let stream: NodeJS.ReadableStream = fs.createReadStream(path.join(dir, f));
      if (f.endsWith(".gz")) stream = stream.pipe(zlib.createGunzip());
      const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
      rl.on("line", (l) => {
        if (!l.trim()) return;
        let r: any;
        try { r = JSON.parse(l); } catch { return; }
        if (r?.la == null || r?.lo == null || typeof r.t !== "number") return;
        const day = new Date(r.t * 1000).toISOString().slice(0, 10);
        for (const s of sites) {
          if (kmBetween(r.la, r.lo, s.lat, s.lon) <= RADIUS_KM) {
            const d = density.get(s.id)!;
            const cell = d[day] || (d[day] = { a: 0, v: 0 });
            cell[field]++;
          }
        }
      });
      rl.on("close", () => resolve());
      stream.on("error", () => resolve());
      // readline.Interface re-emits a piped-in stream's error on ITSELF too
      // (separate from stream.on("error", ...) above) — unlistened, that
      // crashes the whole process on a truncated/corrupt .gz. See
      // datacoreArchive.ts's streamJsonlLines for the full writeup.
      rl.on("error", () => resolve());
    } catch { resolve(); }
  }));
  return chain.reduce((p, next) => p.then(next), Promise.resolve());
}

export async function buildSiteTimelines(sites: SiteRef[], base = archiveBaseDir(), nowMs = Date.now()): Promise<Map<string, SiteTimeline>> {
  const events = foldEventStreams(sites, base, nowMs);
  const density = new Map<string, Record<string, { a: number; v: number }>>(sites.map((s) => [s.id, {}]));
  const days = new Set(lastDays(nowMs));
  await foldPositions("aircraft", "a", sites, density, base, days);
  await foldPositions("vessels", "v", sites, density, base, days);
  const out = new Map<string, SiteTimeline>();
  for (const s of sites) {
    const ev = (events.get(s.id) || []).sort((a, b) => (a.t < b.t ? 1 : -1)).slice(0, EVENTS_CAP);
    out.set(s.id, { site: s.id, events: ev, density: density.get(s.id)! });
  }
  return out;
}

// ── cached assembly ─────────────────────────────────────────────────────────
const TTL_MS = 6 * 60 * 60_000;
let cache: { at: number; timelines: Map<string, SiteTimeline> } | null = null;
let inflight: Promise<Map<string, SiteTimeline>> | null = null;

export async function siteTimelineCached(sites: SiteRef[], base = archiveBaseDir(), nowMs = Date.now()):
    Promise<{ timelines: Map<string, SiteTimeline>; as_of: string }> {
  if (!cache || nowMs - cache.at > TTL_MS) {
    if (!inflight) {
      inflight = buildSiteTimelines(sites, base, nowMs).then((timelines) => {
        cache = { at: Date.now(), timelines };
        inflight = null;
        return timelines;
      });
    }
    if (!cache) await inflight;
  }
  return { timelines: cache?.timelines ?? new Map(), as_of: cache ? new Date(cache.at).toISOString() : "" };
}

export function _resetSiteTimelineCache() { cache = null; inflight = null; }
