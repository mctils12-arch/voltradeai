/**
 * faaStatus.ts — FAA national airspace status: ground stops, ground-delay
 * programs, arrival/departure delays, closures
 * (BUILD ORDER 5 #4, filed + probed 2026-07-05).
 *
 * Source: nasstatus.faa.gov/api/airport-status-information — keyless XML,
 * US government work, public domain; attribution "FAA National Airspace
 * System Status". Shape verified live 2026-07-05 (a thunderstorm day:
 * ground stops DCA/BWI, GDPs JFK/LGA/PHL/EWR made a real fixture).
 *
 * XML parsing follows the edgarForm4.ts precedent: the document is flat,
 * machine-generated, and DTD-backed — regex tag extraction with the same
 * helper shapes, no XML dependency added.
 *
 * ARCHIVE SEMANTICS: the feed is a rolling snapshot, not an event log.
 * Dedup is by EVENT IDENTITY (type|airport|reason|numbers) — a delay
 * program that persists across polls archives ONCE; when its numbers
 * change, the new state appends (vintage-style). End/clear times are NOT
 * in the feed, so event durations are lower bounds from our own capture
 * times — stated in the manifest, never inferred.
 *
 * HYPOTHESIS (gate-locked): sustained delay-program frequency at cargo
 * hubs (MEM, SDF, CVG, ANC) is a quarter-to-date cost-pressure indicator
 * for parcel carriers. RAW archive + route only; the map layer needs an
 * airport-coordinate table and is deliberately a follow-up.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export interface FaaEvent {
  type: "ground_stop" | "ground_delay" | "delay" | "closure";
  airport: string;
  reason: string | null;
  avg: string | null;        // GDP average delay, as published ("2 hours and 30 minutes")
  max: string | null;
  min: string | null;
  trend: string | null;      // arrival/departure delays
  direction: string | null;  // "Arrival" | "Departure" for type=delay
  end_time: string | null;   // ground stops publish a local end time
  reopen: string | null;     // closures
  update_time: string;       // feed's own Update_Time
  rt: string;                // as-seen UTC timestamp
}

// Unlike the EDGAR helpers, tag names here collide by prefix
// (<Delay> vs <Delay_type>) — the name must be followed by whitespace
// or ">" so a prefix never swallows its longer sibling.
const tagContent = (block: string, tag: string): string | null => {
  const m = block.match(new RegExp(`<${tag}(?:\\s[^>]*)?>([\\s\\S]*?)<\\/${tag}>`, "i"));
  return m ? m[1].trim() : null;
};

const allBlocks = (xml: string, tag: string): string[] => {
  const out: string[] = [];
  const re = new RegExp(`<${tag}(?:\\s[^>]*)?>([\\s\\S]*?)<\\/${tag}>`, "gi");
  let m: RegExpExecArray | null;
  while ((m = re.exec(xml))) out.push(m[1]);
  return out;
};

export function parseFaaStatus(xml: string, rt: string): FaaEvent[] {
  if (!xml || !xml.includes("<AIRPORT_STATUS_INFORMATION>")) return [];
  const update = tagContent(xml, "Update_Time") || "";
  const out: FaaEvent[] = [];
  const base = { avg: null, max: null, min: null, trend: null, direction: null, end_time: null, reopen: null };

  for (const b of allBlocks(xml, "Program")) {
    const airport = tagContent(b, "ARPT");
    if (!airport) continue;
    out.push({ ...base, type: "ground_stop", airport, reason: tagContent(b, "Reason"),
               end_time: tagContent(b, "End_Time"), update_time: update, rt });
  }
  for (const b of allBlocks(xml, "Ground_Delay")) {
    const airport = tagContent(b, "ARPT");
    if (!airport) continue;
    out.push({ ...base, type: "ground_delay", airport, reason: tagContent(b, "Reason"),
               avg: tagContent(b, "Avg"), max: tagContent(b, "Max"), update_time: update, rt });
  }
  for (const b of allBlocks(xml, "Delay")) {
    const airport = tagContent(b, "ARPT");
    if (!airport) continue;
    const ad = b.match(/<Arrival_Departure[^>]*Type="([^"]*)"[^>]*>([\s\S]*?)<\/Arrival_Departure>/i);
    out.push({
      ...base, type: "delay", airport, reason: tagContent(b, "Reason"),
      direction: ad ? ad[1] : null,
      min: ad ? tagContent(ad[2], "Min") : null,
      max: ad ? tagContent(ad[2], "Max") : null,
      trend: ad ? tagContent(ad[2], "Trend") : null,
      update_time: update, rt,
    });
  }
  for (const b of allBlocks(xml, "Airport_Closure_List")) {
    for (const a of allBlocks(b, "Airport")) {
      const airport = tagContent(a, "ARPT");
      if (!airport) continue;
      out.push({ ...base, type: "closure", airport, reason: tagContent(a, "Reason"),
                 reopen: tagContent(a, "Reopen"), update_time: update, rt });
    }
  }
  return out;
}

// ── Fetch ────────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

const API = "https://nasstatus.faa.gov/api/airport-status-information";

export async function fetchFaaStatus(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<FaaEvent[] | null> {
  const rt = new Date(nowMs ?? Date.now()).toISOString();
  try {
    const r = await fetchImpl(API, {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(20000) as any,
    });
    if (!r.ok) {
      console.error(`[datacore] faastatus -> ${r.status}`);
      return null;
    }
    return parseFaaStatus(await r.text(), rt);
  } catch (e: any) {
    console.error("[datacore] faastatus:", e?.message || e);
    return null;
  }
}

// ── Archive (event-identity dedup; a persisting program archives once) ──────

const archivedKeys = new Set<string>();
let seeded = false;

function faaDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "faastatus");
}

const keyOf = (e: FaaEvent) =>
  `${e.type}|${e.airport}|${e.reason}|${e.avg}|${e.max}|${e.min}|${e.end_time}|${e.reopen}`;

function seedSeen(dir: string, nowMs: number): void {
  for (let i = 0; i < 10; i++) {
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

export function archiveFaaEvents(events: FaaEvent[], baseDir?: string, nowMs?: number): number {
  const dir = faaDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = events.filter((e) => !archivedKeys.has(keyOf(e)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.appendFileSync(path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`),
                      fresh.map((e) => JSON.stringify(e)).join("\n") + "\n");
    fresh.forEach((e) => archivedKeys.add(keyOf(e)));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] faastatus archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldFaaDays(baseDir?: string, nowMs?: number): number {
  const dir = faaDir(baseDir);
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

let cache: { at: number; update_time: string; events: FaaEvent[] } | null = null;
let polling = false;

export function latestFaaStatus() {
  return cache;
}

export async function refreshFaaStatus(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<void> {
  try {
    const events = await fetchFaaStatus(fetchImpl, nowMs);
    if (events === null) return; // transport error — keep last snapshot
    // an empty NAS (no programs anywhere) is a real, publishable state
    cache = { at: Date.now(), update_time: events[0]?.update_time || "", events };
    archiveFaaEvents(events, undefined, nowMs);
    gzipOldFaaDays(undefined, nowMs);
  } catch (e: any) {
    console.error("[datacore] faastatus refresh:", e?.message || e);
  }
}

/** Delay programs are intraday phenomena — 15-min poll (96 light
 *  requests/day) so program churn lands in the archive. Eager boot per
 *  KNOWN BROKEN #9. */
export function bootFaaPoll(intervalMs = 15 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshFaaStatus();
  setInterval(() => { refreshFaaStatus(); }, intervalMs).unref?.();
}
