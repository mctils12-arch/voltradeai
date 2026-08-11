// aircraftTiling.ts — server-side viewport TILING for the live-aircraft feed.
//
// HUMAN DIRECTIVE 2026-07-16: "I want it to render all of them, not the
// radius of 250nm — the whole visible range on the map." The ceiling is
// upstream: adsb.lol (and the fallback chain) only expose a point query
// (/v2/point/{lat}/{lon}/{radius}) with a hard 250nm radius cap, so a wide
// viewport can never be served by ONE request. This module covers the
// requested bbox with a MINIMAL grid of 250nm-max discs, fetched through
// the existing provider chain, deduped by hex id, and reported HONESTLY:
// the envelope always states how many discs were fetched, how many a full
// cover would need, and whether the bbox was actually covered.
//
// POLITENESS (these are free community feeds — the same discipline as the
// single-query era, multiplied carefully, never blindly):
//   - MAX_DISCS_PER_REFRESH = 8: one refresh never fires more than 8
//     upstream requests. 8 discs at 250nm cover roughly a large-US-region
//     viewport (~1400 x 700 nm); beyond that the response says "viewport
//     too large — showing central region" instead of hammering a free
//     community API with dozens of calls for a zoomed-out globe.
//   - Disc #1 resolves ALONE first (full provider chain, exactly the old
//     single-query behavior). Only after it succeeds do discs 2..N launch,
//     staggered DISC_SPACING_MS apart — so a dead provider is discovered
//     ONCE and enters backoff before the fan-out, never 8 times over.
//   - The per-provider exponential backoff, the 30s shared snapshot cache
//     and the in-flight dedup slot in routes.ts all still apply; tiling
//     multiplies a single cache-miss refresh, not every visitor poll.
//
// The module is pure/injectable (fetch, sleep, backoff hooks) so tests
// never touch the real APIs — same pattern as liveDelta.ts / viewport.ts.

/** adsb.lol / airplanes.live / adsb.fi point-query hard radius cap (nm). */
export const DISC_RADIUS_MAX_NM = 250;
/** Politeness cap: max upstream point-queries per snapshot refresh. */
export const MAX_DISCS_PER_REFRESH = 8;
/** Stagger between sub-query launches (ms) — spreads the burst. */
export const DISC_SPACING_MS = 175;
/** Legacy per-request floor (nm) — tiny viewports still ask for a useful disc. */
export const DISC_RADIUS_MIN_NM = 50;

const NM_PER_DEG_LAT = 60;
/** Square-grid spacing that GUARANTEES cover: a disc of radius r covers the
 *  s x s cell centered on it iff s <= r*sqrt(2) (half-diagonal = r). This is
 *  the minimal square grid with zero uncovered gaps; hex packing would save
 *  ~10% of discs but has fiddly edge cases — correctness of cover wins. */
const GRID_SPACING_NM = DISC_RADIUS_MAX_NM * Math.SQRT2;

export interface DiscCenter { lat: number; lon: number }

export interface DiscPlan {
  /** Query centers, ordered center-of-bbox first (best coverage if later
   *  discs fail; disc #1 doubles as the provider-chain probe). */
  centers: DiscCenter[];
  /** One radius for all discs in the plan (uniform grid cells). */
  radiusNm: number;
  /** Discs a FULL cover of the bbox would take (may exceed the cap). */
  needed: number;
  /** Radius a single legacy point-query would need for this bbox (nm). */
  neededNm: number;
  /** The politeness cap the plan was built under. */
  cappedAt: number;
  /** True iff the planned centers fully cover the requested bbox. */
  bboxCovered: boolean;
}

const clampLat = (v: number) => Math.min(85, Math.max(-85, v));
const clampLon = (v: number) => Math.min(180, Math.max(-180, v));

/**
 * Plan the minimal set of point-query discs covering [lamin..lamax] x
 * [lomin..lomax], capped at maxDiscs. Geometry is the same local
 * approximation the legacy single-query used (1 deg lat = 60nm; lon scaled
 * by cos(center lat), floored at 0.1) — honest at viewport scales, and the
 * coverage tests use the identical metric. No antimeridian wrap (the route
 * clamps lon to [-180,180], matching current behavior).
 *
 * maxDiscs=1 reproduces the LEGACY behavior exactly: one disc at the bbox
 * center, radius min(250, max(50, neededNm)).
 */
export function planDiscs(
  lamin: number, lamax: number, lomin: number, lomax: number,
  maxDiscs: number = MAX_DISCS_PER_REFRESH,
): DiscPlan {
  // Normalize (route clamps but does not order).
  const laLo = Math.min(lamin, lamax), laHi = Math.max(lamin, lamax);
  const loLo = Math.min(lomin, lomax), loHi = Math.max(lomin, lomax);
  const cap = Math.max(1, Math.floor(maxDiscs));
  const clat = (laLo + laHi) / 2;
  const clon = (loLo + loHi) / 2;
  const cosLat = Math.max(0.1, Math.cos((clat * Math.PI) / 180));
  const latSpanNm = (laHi - laLo) * NM_PER_DEG_LAT;
  const lonSpanNm = (loHi - loLo) * NM_PER_DEG_LAT * cosLat;
  const neededNm = Math.ceil(Math.hypot(latSpanNm, lonSpanNm) / 2);

  // One disc suffices — identical to the legacy single point-query.
  if (neededNm <= DISC_RADIUS_MAX_NM) {
    return {
      centers: [{ lat: clat, lon: clon }],
      radiusNm: Math.max(DISC_RADIUS_MIN_NM, neededNm),
      needed: 1, neededNm, cappedAt: cap, bboxCovered: true,
    };
  }

  // Full-cover grid dimensions at the guaranteed-cover spacing.
  const rows = Math.max(1, Math.ceil(latSpanNm / GRID_SPACING_NM));
  const cols = Math.max(1, Math.ceil(lonSpanNm / GRID_SPACING_NM));
  const needed = rows * cols;

  let useRows = rows, useCols = cols, covered = true;
  if (needed > cap) {
    // Cap bites: pick the r x c sub-grid (product <= cap) that maximizes
    // covered cells, tie-broken toward the bbox's own aspect ratio, and
    // center it on the viewport — "showing central region", honestly.
    covered = false;
    let bestR = 1, bestC = Math.min(cols, cap);
    for (let r = 1; r <= Math.min(rows, cap); r++) {
      const c = Math.min(cols, Math.floor(cap / r));
      if (c < 1) continue;
      const cur = r * c, best = bestR * bestC;
      const want = rows / cols;
      if (cur > best || (cur === best && Math.abs(r / c - want) < Math.abs(bestR / bestC - want))) {
        bestR = r; bestC = c;
      }
    }
    useRows = bestR; useCols = bestC;
  }

  // Cell size: dimensions fully covered shrink cells to fit the span
  // (radius shrinks with them); capped dimensions keep full-spacing cells.
  const cellLatNm = useRows === rows ? latSpanNm / rows : GRID_SPACING_NM;
  const cellLonNm = useCols === cols ? lonSpanNm / cols : GRID_SPACING_NM;
  const radiusNm = Math.min(
    DISC_RADIUS_MAX_NM,
    Math.max(DISC_RADIUS_MIN_NM, Math.ceil(Math.hypot(cellLatNm, cellLonNm) / 2)),
  );
  const latStepDeg = cellLatNm / NM_PER_DEG_LAT;
  const lonStepDeg = cellLonNm / (NM_PER_DEG_LAT * cosLat);

  const centers: DiscCenter[] = [];
  for (let i = 0; i < useRows; i++) {
    for (let j = 0; j < useCols; j++) {
      centers.push({
        lat: clampLat(clat + (i - (useRows - 1) / 2) * latStepDeg),
        lon: clampLon(clon + (j - (useCols - 1) / 2) * lonStepDeg),
      });
    }
  }
  // Center-first order: disc #1 is the most central (provider probe +
  // best single-disc coverage if the fan-out partially fails).
  centers.sort((a, b) =>
    Math.hypot(a.lat - clat, (a.lon - clon) * cosLat) -
    Math.hypot(b.lat - clat, (b.lon - clon) * cosLat));

  return { centers, radiusNm, needed, neededNm, cappedAt: cap, bboxCovered: covered };
}

/** One upstream in the fallback chain (URL builder per disc center). */
export interface DiscProvider {
  key: string;
  label: string;
  /** JSON array key in the response ("ac" for readsb, "aircraft" for adsb.fi). */
  arr: string;
  url: (lat: number, lon: number, radiusNm: number) => string;
}

/** readsb point-query record -> the unified aircraft shape the client and
 *  the position archive consume (extracted verbatim from routes.ts so every
 *  disc maps identically and the mapping is unit-testable). */
export function mapPointAircraft(raw: any, arrKey: string): any[] {
  return ((raw?.[arrKey] as any[]) || []).slice(0, 10000).map((a: any) => {
    // alt_baro drops out of readsb rows for stretches while alt_geom (GNSS
    // altitude — also a real broadcast field) stays present; a null here
    // NaN-gapped the 3D track curtain at the live end even though the plane
    // was clearly airborne (2026-07-31 video). Baro first (the aviation
    // convention every consumer expects), geometric as the fallback — both
    // are broadcast measurements, never an estimate.
    const onGround = a.alt_baro === "ground";
    const altFt = onGround ? null
      : typeof a.alt_baro === "number" ? a.alt_baro
      : typeof a.alt_geom === "number" ? a.alt_geom
      : null;
    return {
      icao24: a.hex,
      callsign: String(a.flight || "").trim(),
      // `r` is the aircraft REGISTRATION (tail number, e.g. N8667D) in the
      // readsb schema — it was mislabeled origin_country from 2026-07-03
      // until 2026-08-08 and displayed as "Country" on the flight card
      // (identity repair, plane-tracking T1). Country now comes from the
      // icao24 hex allocation client-side (lib/air/planeIdentity).
      registration: a.r || "",
      lon: a.lon,
      lat: a.lat,
      altitude_m: altFt == null ? null : Math.round(altFt * 0.3048),
      on_ground: onGround,
      velocity_ms: a.gs == null ? null : Math.round(a.gs * 0.5144),
      heading: a.track ?? null,
      type: a.t || null,                    // ICAO type designator (e.g. B738, C172)
      category: a.category || null,          // ADS-B emitter category (A1..A7)
    };
  }).filter((a: any) => a.lat != null && a.lon != null);
}

/** Merge per-disc aircraft lists, deduping by icao24 (discs overlap by
 *  construction — the grid guarantees cover via overlap). First occurrence
 *  wins: lists arrive center-disc first, and within one refresh all discs
 *  are the same snapshot generation, so there is no "fresher" copy to
 *  prefer. Records without an id can't be deduped and are kept. */
export function dedupeByHex(lists: any[][]): any[] {
  const seen = new Set<string>();
  const out: any[] = [];
  for (const list of lists) {
    for (const a of list) {
      const id = a?.icao24;
      if (id) {
        if (seen.has(id)) continue;
        seen.add(id);
      }
      out.push(a);
    }
  }
  return out;
}

export interface TiledFetchDeps {
  providers: DiscProvider[];
  headers?: Record<string, string>;
  fetchImpl?: typeof fetch;
  backoffActive?: (key: string) => boolean;
  backoffBump?: (key: string) => void;
  backoffClear?: (key: string) => void;
  spacingMs?: number;
  timeoutMs?: number;
  sleep?: (ms: number) => Promise<void>;
}

export interface TiledFetchResult {
  aircraft: any[];
  /** Distinct provider labels that served discs this refresh. */
  sources: string[];
  /** Discs that actually returned data (<= plan.centers.length). */
  discsOk: number;
  errs: string[];
}

/**
 * Fetch every disc in the plan through the provider chain.
 *
 * Disc #1 is fetched ALONE first (exactly the legacy chain walk) — if the
 * whole chain fails for it, the refresh throws (stale-beats-spinner in the
 * route takes over) and, crucially, the failing providers are already in
 * backoff BEFORE any fan-out, so an outage costs one chain walk, not N.
 * Discs 2..N then launch staggered `spacingMs` apart and run concurrently;
 * individual failures degrade coverage honestly instead of failing the
 * whole refresh.
 */
export async function fetchDiscs(plan: DiscPlan, deps: TiledFetchDeps): Promise<TiledFetchResult> {
  const {
    providers,
    headers = {},
    fetchImpl = fetch,
    backoffActive = () => false,
    backoffBump = () => {},
    backoffClear = () => {},
    spacingMs = DISC_SPACING_MS,
    timeoutMs = 12000,
    sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms)),
  } = deps;
  const errs: string[] = [];
  const sources = new Set<string>();

  async function oneDisc(c: DiscCenter): Promise<any[] | null> {
    for (const p of providers) {
      if (backoffActive(p.key)) { errs.push(`${p.key} in backoff`); continue; }
      try {
        const r = await fetchImpl(p.url(c.lat, c.lon, plan.radiusNm), {
          headers, signal: AbortSignal.timeout(timeoutMs),
        });
        if (!r.ok) throw new Error(`${p.key} ${r.status}`);
        const raw: any = await r.json();
        backoffClear(p.key);
        sources.add(p.label);
        return mapPointAircraft(raw, p.arr);
      } catch (e: any) {
        backoffBump(p.key);
        const cause = e?.cause?.code || e?.cause?.message || "";
        errs.push(`${p.key}: ${e?.message}${cause ? ` (${cause})` : ""}`);
      }
    }
    return null;
  }

  const first = await oneDisc(plan.centers[0]);
  if (first == null) throw new Error(errs.join(" | ") || "all providers failed");

  const rest = await Promise.all(
    plan.centers.slice(1).map((c, i) =>
      (async () => { await sleep(spacingMs * (i + 1)); return oneDisc(c); })()),
  );
  const ok = [first, ...rest].filter((x): x is any[] => x != null);
  return { aircraft: dedupeByHex(ok), sources: Array.from(sources), discsOk: ok.length, errs };
}

export interface TilingCoverage {
  /** Discs that returned data this refresh. */
  discs: number;
  /** Discs a full cover of the requested bbox needs. */
  needed: number;
  /** The politeness cap (upstream point-queries per refresh). */
  cappedAt: number;
  /** True iff the WHOLE requested bbox was covered this refresh. */
  bboxCovered: boolean;
}

/**
 * The honest coverage half of the response envelope. Keeps the existing
 * client contract (`coverage: "full"|"partial"` + human `coverage_note`)
 * and adds structured `tiling` metadata so the layers panel can state true
 * coverage ("N discs / full viewport" or "viewport too large — central
 * region"). `legacyMode` preserves the old single-query note verbatim for
 * bbox-less callers.
 */
export function tilingEnvelope(plan: DiscPlan, discsOk: number, legacyMode: boolean): {
  coverage: "full" | "partial";
  coverage_note?: string;
  tiling: TilingCoverage;
} {
  const bboxCovered = plan.bboxCovered && discsOk >= plan.centers.length;
  let coverage_note: string | undefined;
  if (!plan.bboxCovered) {
    coverage_note = legacyMode
      ? `feed covers ~250nm around view center (viewport needs ~${plan.neededNm}nm) — zoom in for full coverage`
      : `viewport too large — showing central region (${plan.centers.length} of ${plan.needed} 250nm coverage discs; politeness cap ${plan.cappedAt} per refresh) — zoom in for full coverage`;
  } else if (discsOk < plan.centers.length) {
    coverage_note = `partial refresh: ${discsOk} of ${plan.centers.length} coverage discs answered — uncovered area retries next poll`;
  }
  return {
    coverage: bboxCovered ? "full" : "partial",
    ...(coverage_note ? { coverage_note } : {}),
    tiling: { discs: discsOk, needed: plan.needed, cappedAt: plan.cappedAt, bboxCovered },
  };
}
