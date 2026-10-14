// LADD REGISTRY (2026-10-14, human-reported): which aircraft are in the FAA
// LADD privacy program, so FAA-sourced data about them is never served.
//
// WHY: the owner of a LADD aircraft has asked the FAA to limit public display
// of its flight data, and FAA data feeds (SWIM SFDPS here) carry that
// obligation to the consumer. The FAA-filed flight plan of N843S (a LADD
// Falcon 7X) was being served on /api/data/aircraft/plan and could reach
// /api/data/procedures/flight. Live ADS-B positions are NOT FAA data (they are
// received directly from the aircraft by community receivers) and are not
// affected; only FAA-sourced fields are withheld.
//
// SOURCE: adsb.lol's /v2/ladd global scope (already polled every 120s by
// globalScopes.ts). It lists LADD aircraft that are transmitting NOW, so the
// registry REMEMBERS every hex/callsign it has seen for LADD_MEMORY_MS and
// persists them, so an aircraft parked at the gate with a filed plan is still
// known from earlier flights. Gap, stated honestly: a LADD aircraft never yet
// seen airborne by adsb.lol is unknown here. The authoritative FAA LADD
// industry list needs a separate FAA registration (wishlist.md).

import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";

/** how long one sighting in the LADD scope keeps an aircraft listed */
export const LADD_MEMORY_MS = 90 * 24 * 3600_000;
/** persisted at most this often (the scope is polled every 120s) */
export const LADD_SAVE_MIN_MS = 10 * 60_000;

const normHex = (h: string) => String(h || "").trim().toLowerCase();
const normCs = (c: string) => String(c || "").trim().toUpperCase();

export class LaddRegistry {
  private hexes = new Map<string, number>();
  private callsigns = new Map<string, number>();
  private dirty = false;
  private lastSave = 0;

  constructor(private opts: { file?: string | null; now?: () => number } = {}) {
    this.load();
  }

  private now(): number { return (this.opts.now ?? Date.now)(); }

  /** Record aircraft seen in the LADD scope this cycle. */
  note(rows: Array<{ icao24?: string; callsign?: string }>): void {
    const t = this.now();
    let added = false;
    for (const r of rows) {
      const h = normHex(r.icao24 ?? "");
      if (h) { if (!this.hexes.has(h)) added = true; this.hexes.set(h, t); }
      const c = normCs(r.callsign ?? "");
      if (c) { if (!this.callsigns.has(c)) added = true; this.callsigns.set(c, t); }
    }
    if (rows.length) this.dirty = true; // last-seen times moved
    this.prune(t);
    // a NEW member is saved at once (it must survive a restart); refreshing
    // known members' last-seen times is throttled
    if (added || (this.dirty && t - this.lastSave >= LADD_SAVE_MIN_MS)) this.save();
  }

  /** True when this aircraft (by hex OR callsign) was seen in the LADD scope
   *  within LADD_MEMORY_MS. Either identifier alone is enough. */
  isLadd(hex: string | null | undefined, callsign?: string | null): boolean {
    const t = this.now();
    const fresh = (seen: number | undefined) => seen != null && t - seen <= LADD_MEMORY_MS;
    return (!!hex && fresh(this.hexes.get(normHex(hex))))
      || (!!callsign && fresh(this.callsigns.get(normCs(callsign))));
  }

  get size(): { hexes: number; callsigns: number } {
    return { hexes: this.hexes.size, callsigns: this.callsigns.size };
  }

  private prune(t: number): void {
    for (const m of [this.hexes, this.callsigns]) {
      for (const [k, seen] of m) if (t - seen > LADD_MEMORY_MS) { m.delete(k); this.dirty = true; }
    }
  }

  private load(): void {
    const f = this.opts.file;
    if (!f) return;
    try {
      const d = JSON.parse(fs.readFileSync(f, "utf8")) as { hexes?: Record<string, number>; callsigns?: Record<string, number> };
      for (const [k, v] of Object.entries(d.hexes ?? {})) if (Number.isFinite(v)) this.hexes.set(k, v);
      for (const [k, v] of Object.entries(d.callsigns ?? {})) if (Number.isFinite(v)) this.callsigns.set(k, v);
      this.prune(this.now());
    } catch (e: unknown) {
      // a missing file is the normal first boot; anything else is worth a line
      if ((e as { code?: string })?.code !== "ENOENT") {
        console.error("[ladd] registry load failed:", e instanceof Error ? e.message : String(e));
      }
    }
  }

  private save(): void {
    const f = this.opts.file;
    this.lastSave = this.now();
    this.dirty = false;
    if (!f) return;
    try {
      fs.mkdirSync(path.dirname(f), { recursive: true });
      const tmp = `${f}.tmp`;
      fs.writeFileSync(tmp, JSON.stringify({
        hexes: Object.fromEntries(this.hexes), callsigns: Object.fromEntries(this.callsigns),
      }));
      fs.renameSync(tmp, f);
    } catch (e: unknown) {
      this.dirty = true; // retry on the next cycle
      console.error("[ladd] registry save failed:", e instanceof Error ? e.message : String(e));
    }
  }
}

let shared: LaddRegistry | null = null;

/** The process-wide registry, persisted next to the aircraft archive. */
export function laddRegistry(): LaddRegistry {
  if (!shared) shared = new LaddRegistry({ file: path.join(archiveBaseDir(), "ladd_registry.json") });
  return shared;
}

/** Neutral wording shown whenever FAA-sourced data is withheld. It does not
 *  say whether a plan exists, which would itself leak flight activity. */
export const LADD_WITHHELD_NOTE =
  "This aircraft is in the FAA LADD privacy program, so FAA-sourced flight data (filed plans) is not shown. Live position comes from community ADS-B receivers, not the FAA.";
