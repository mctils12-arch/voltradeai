// COVERAGE GAP-FILL (2026-10-14, human report: "some planes show up on the
// ground as on the ground, but I didn't see my plane do this"). adsb.lol, our
// primary provider, has NO receiver with line of sight to some airport
// surfaces. Measured live at CLT the same day: adsb.lol 0 aircraft on the
// ground within 4 nm; adsb.fi 9 on the ground (airline gates + the bizjet
// ramp); control at ATL: adsb.lol 28 on the ground. The provider chain in
// fetchDiscs only falls through to adsb.fi when adsb.lol FAILS, so an
// adsb.lol answer that is merely BLIND to a surface never reached the map.
//
// What this adds, nothing else:
//   - a zoomed-in viewport (ONE disc of radius <= GAPFILL_MAX_RADIUS_NM)
//     also asks adsb.fi for the same disc and merges only the hexes the
//     primary answer lacks (primary rows always win on overlap);
//   - the selected-aircraft fast lane asks adsb.fi for a hex adsb.lol
//     returned nothing for (aircraftFastLane.ts).
// Every gap-filled row keeps provider "adsbfi" provenance, so the snapshot's
// lawful-only mode and the monetization tripwire (providerCompliance.ts —
// adsb.fi is NON-COMMERCIAL) see it exactly as before. Politeness: adsb.fi
// asks for at most 1 request/second; one shared token bucket covers both
// paths here, and an empty bucket just skips the fill (counted).

import { mapPointAircraft, type DiscPlan } from "./aircraftTiling";
import { publishFixes } from "./aircraftFixBus";

/** viewports wider than this (one disc) are not gap-filled */
export const GAPFILL_MAX_RADIUS_NM = 60;
/** adsb.fi's published limit is 1 req/s; stay under it across both paths */
export const GAPFILL_MIN_INTERVAL_MS = 1_100;
export const GAPFILL_TIMEOUT_MS = 6_000;
export const GAPFILL_PROVIDER = "adsbfi";

export const adsbFiDiscUrl = (lat: number, lon: number, rNm: number): string =>
  `https://opendata.adsb.fi/api/v2/lat/${lat.toFixed(3)}/lon/${lon.toFixed(3)}/dist/${Math.max(1, Math.round(rNm))}`;
export const adsbFiHexUrl = (hex: string): string => `https://opendata.adsb.fi/api/v2/hex/${hex}`;

/** Pure: does this refresh qualify for a gap-fill? One disc, zoomed in, and
 *  adsb.fi was not already the provider that answered. */
export function shouldGapFill(plan: Pick<DiscPlan, "centers" | "radiusNm">, providersUsed: string[]): boolean {
  return plan.centers.length === 1
    && plan.radiusNm <= GAPFILL_MAX_RADIUS_NM
    && !providersUsed.some((p) => /adsb\.fi/i.test(p));
}

/** Pure: primary rows unchanged, plus extra rows whose hex the primary lacks. */
export function mergeMissing<T extends { icao24?: string | null }>(primary: T[], extra: T[]): { merged: T[]; added: number } {
  const have = new Set(primary.map((a) => String(a.icao24 || "").toLowerCase()).filter(Boolean));
  const add = extra.filter((a) => {
    const h = String(a.icao24 || "").toLowerCase();
    if (!h || have.has(h)) return false;
    have.add(h);
    return true;
  });
  return { merged: add.length ? [...primary, ...add] : primary, added: add.length };
}

export interface GapFillStats {
  disc_requests: number; hex_requests: number; rows_added: number;
  skipped_rate: number; errors: number; last_error: string | null;
}

export interface GapFiller {
  /** viewport: the primary rows plus whatever adsb.fi sees that they lack */
  fillDisc<T extends { icao24?: string | null }>(plan: Pick<DiscPlan, "centers" | "radiusNm">, primary: T[], providersUsed: string[]): Promise<{ aircraft: T[]; added: number }>;
  /** fast lane: adsb.fi's rows for one hex (mapped), [] when none / limited */
  fetchHex(hex: string): Promise<any[]>;
  stats(): GapFillStats;
}

/** Ops kill switch: AIRCRAFT_GAPFILL=0 turns both paths off (no adsb.fi
 *  requests beyond the existing failure fallback). */
export function gapFillEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return String(env.AIRCRAFT_GAPFILL ?? "1").trim() !== "0";
}

export function createGapFiller(deps: {
  fetchImpl?: typeof fetch;
  now?: () => number;
  headers?: Record<string, string>;
  timeoutMs?: number;
  enabled?: boolean;
} = {}): GapFiller {
  const enabled = deps.enabled ?? gapFillEnabled();
  const fetchImpl = deps.fetchImpl ?? fetch;
  const now = deps.now ?? Date.now;
  const headers = deps.headers ?? { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };
  const timeoutMs = deps.timeoutMs ?? GAPFILL_TIMEOUT_MS;
  let lastAt = -Infinity;
  const s: GapFillStats = { disc_requests: 0, hex_requests: 0, rows_added: 0, skipped_rate: 0, errors: 0, last_error: null };

  const takeSlot = (): boolean => {
    const t = now();
    if (t - lastAt < GAPFILL_MIN_INTERVAL_MS) { s.skipped_rate++; return false; }
    lastAt = t;
    return true;
  };
  const get = async (url: string): Promise<any | null> => {
    try {
      const r = await fetchImpl(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
      if (!r.ok) throw new Error(`adsb.fi ${r.status}`);
      return await r.json();
    } catch (e) {
      s.errors++;
      s.last_error = e instanceof Error ? e.message : String(e);
      // logged on the 1st and every 50th failure: loud, not a flood
      if (s.errors === 1 || s.errors % 50 === 0) console.error(`[gap-fill] adsb.fi (x${s.errors}):`, s.last_error);
      return null;
    }
  };

  return {
    async fillDisc(plan, primary, providersUsed) {
      if (!enabled || !shouldGapFill(plan, providersUsed) || !takeSlot()) return { aircraft: primary, added: 0 };
      s.disc_requests++;
      const c = plan.centers[0];
      const raw = await get(adsbFiDiscUrl(c.lat, c.lon, plan.radiusNm));
      if (!raw) return { aircraft: primary, added: 0 };
      const mapped = mapPointAircraft(raw, "aircraft", GAPFILL_PROVIDER);
      const { merged, added } = mergeMissing(primary, mapped as typeof primary);
      s.rows_added += added;
      if (added) {
        // the snapshot (and so the fast lane + plan fill) learn these too;
        // no `disc`, so the sweep scheduler gets no coverage credit for them
        publishFixes({
          provider: GAPFILL_PROVIDER, origin: "gapfill",
          aircraft: (merged as unknown[]).slice(primary.length) as never[], fetchedAt: now(),
          upstreamNowMs: typeof raw?.now === "number" ? raw.now : null,
        });
      }
      return { aircraft: merged, added };
    },
    async fetchHex(hex) {
      if (!enabled || !takeSlot()) return [];
      s.hex_requests++;
      const raw = await get(adsbFiHexUrl(hex));
      if (!raw) return [];
      const mapped = mapPointAircraft(raw, "ac", GAPFILL_PROVIDER)
        .filter((a: { icao24?: unknown }) => String(a.icao24 || "").toLowerCase() === hex);
      s.rows_added += mapped.length;
      return mapped;
    },
    stats: () => ({ ...s }),
  };
}

let shared: GapFiller | null = null;
export function gapFiller(): GapFiller {
  if (!shared) shared = createGapFiller();
  return shared;
}
