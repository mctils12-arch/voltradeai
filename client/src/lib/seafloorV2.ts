// GEBCO v2 seafloor loader + TID confidence decode — EARTH TWIN E2 v2
// (research/earth_twin_program.md V4: GEBCO 15-arcsec self-tiled pipeline +
// "UNCERTAINTY, EXACTLY AS DEMANDED … the seafloor_confidence overlay —
// measured vs interpolated seafloor, visually distinct").
//
// DATA PATH: scripts/gebco_seafloor_tiles.py fetches GEBCO grid subsets from
// GEBCO's own subsetting service and packs them into PMTiles v3 archives under
// client/public/tiles/ (served by express static with range requests — the
// committed-pmtiles precedent from the power-grid layers):
//   - seafloor_gebco_<region>.pmtiles — TERRARIUM-encoded depth (same encoding
//     as the live seafloor v1 layer; client/src/lib/bathymetry.ts's ramp
//     applies unchanged, including its land/no-data transparency at 0 m).
//   - seafloor_tid_<region>.pmtiles — the RAW GEBCO TID source-type code
//     terrarium-encoded as "elevation" (code 11 decodes as 11). Styling stays
//     HERE, client-side, derived from datacore/gebco/tid_decode.json — the
//     tiles carry data, never baked-in colors.
//   - seafloor_gebco_<region>.provenance.json — attribution, true extent,
//     resolution, sampling methods, measured per-group TID shares. Any wiring
//     MUST surface attribution + the not-for-navigation disclaimer.
//
// ONE SOURCE OF TRUTH: datacore/gebco/tid_decode.json carries the VERBATIM
// official GEBCO_2026 TID table (codes, definitions, GEBCO's own grouping:
// Land / Direct measurements / Indirect measurements / Unknown). This module
// derives the map expression AND the legend from that one file — the decode
// can never drift between the pipeline, the map, and its legend. NEVER invent,
// merge, or re-group codes here: mirror official documentation only.
//
// RENDER CAVEAT (documented, not hidden): raster upsampling may interpolate
// between neighboring pixels' code values. The color expression below maps
// every gap BETWEEN documented code runs to TRANSPARENT, so cross-group
// interpolation renders as a transparent seam rather than a wrong class —
// except the direct(10-17)↔unknown(70-72) adjacency, whose interpolation path
// crosses the indirect band and can tint a thin fringe. Prefer
// raster-resampling "nearest" where the layer type supports it. (The shipped
// mariana region contains no unknown-group cells; see its provenance.)

import tidDecodeJson from "../../../datacore/gebco/tid_decode.json";

export type TidGroup = "land" | "direct" | "indirect" | "unknown";

export interface TidCodeInfo {
  group: TidGroup;
  /** VERBATIM definition from the official GEBCO documentation. */
  definition: string;
}

interface TidDecodeFile {
  dataset: string;
  source: {
    documentationUrl: string;
    documentationFetched: string;
    basketDocument: string;
    attribution: string;
    termsOfUse: string;
  };
  nodataValue: number;
  groups: Record<TidGroup, { label: string; meaning: string }>;
  codes: Record<string, TidCodeInfo>;
}

const decode = tidDecodeJson as unknown as TidDecodeFile;

/** Verbatim official decode table, keyed by numeric TID code. */
export const TID_DECODE: ReadonlyMap<number, TidCodeInfo> = new Map(
  Object.entries(decode.codes).map(([k, v]) => [Number(k), v]),
);

/** GEBCO's documented grid nodata value (source grid, not our tiles). */
export const TID_GRID_NODATA = decode.nodataValue;

/** Our tiles' out-of-coverage marker: NOT a TID code; renders transparent. */
export const TID_NO_DATA = -1;

/** Required attribution (GEBCO terms of use), verbatim from the decode file. */
export const GEBCO_ATTRIBUTION = decode.source.attribution;

/** The GEBCO terms' own disclaimer — surface with every v2 seafloor display. */
export const GEBCO_NOT_FOR_NAVIGATION =
  "Indicative only — NOT for navigation or any purpose involving safety at sea (GEBCO terms of use).";

/**
 * Confidence classes = GEBCO's OWN documented grouping, with display colors
 * (colors are our render choice; labels/meanings come from the shared table).
 * Land is transparent — the overlay is about the seafloor, and painting land
 * would claim a reading that isn't there.
 */
export const TID_CONFIDENCE: Record<
  TidGroup,
  { label: string; meaning: string; color: string }
> = {
  land: { ...decode.groups.land, color: "rgba(0,0,0,0)" },
  direct: { ...decode.groups.direct, color: "#2dd4a7" },
  indirect: { ...decode.groups.indirect, color: "#f0a63c" },
  unknown: { ...decode.groups.unknown, color: "#9b87c9" },
};

/** Group of a decoded TID code; null for no-data or undocumented values. */
export function tidGroupOf(code: number): TidGroup | null {
  return TID_DECODE.get(code)?.group ?? null;
}

/** Terrarium decode (AWS Terrain Tiles encoding, used by both v2 tile sets). */
export function terrariumDecode(r: number, g: number, b: number): number {
  return r * 256 + g + b / 256 - 32768;
}

const TRANSPARENT = "rgba(0,0,0,0)";

/**
 * Contiguous runs of documented codes sharing a group, ascending — derived
 * from the table so a future official revision flows through automatically.
 */
export function tidCodeRuns(): { start: number; end: number; group: TidGroup }[] {
  const codes = Array.from(TID_DECODE.keys()).sort((a, b) => a - b);
  const runs: { start: number; end: number; group: TidGroup }[] = [];
  for (const c of codes) {
    const g = TID_DECODE.get(c)!.group;
    const last = runs[runs.length - 1];
    if (last && last.group === g && c === last.end + 1) last.end = c;
    else runs.push({ start: c, end: c, group: g });
  }
  return runs;
}

/**
 * MapLibre `color-relief-color` expression over the TID tiles' decoded
 * "elevation" (= raw TID code): each documented code run gets its group's
 * confidence color; everything else — no-data (-1), gaps between runs,
 * values past the last run — is TRANSPARENT (never a guessed class).
 *
 * INTERPOLATE form, not ["step"]: maplibre v5.24 VALIDATES a step
 * expression here but silently renders NOTHING (root-caused 2026-07-16 —
 * getPaintProperty returned the step paint verbatim while zero pixels
 * drew; swapping the identical classes to interpolate form painted
 * immediately; probe chain in experiments.md). This was exactly the
 * failure mode the #497 wiring session's own hypothesis predicted ("if
 * the whole region renders one uniform color, suspect the step
 * expression"). TID codes are integers, so knife-edge transitions at
 * ±0.02 around each documented run keep EXACT class semantics — no real
 * code value can land inside a transition, and land/gaps stay absent.
 */
export function tidConfidenceColorRelief(): unknown[] {
  const expr: unknown[] = ["interpolate", ["linear"], ["elevation"]];
  for (const run of tidCodeRuns()) {
    const color = TID_CONFIDENCE[run.group].color;
    if (color === TRANSPARENT) continue; // land: absent, exactly as documented
    expr.push(run.start - 0.02, TRANSPARENT);
    if (run.end > run.start) {
      expr.push(run.start, color, run.end, color);
    } else {
      expr.push(run.start, color); // single-code run — one stop, no duplicates
    }
    expr.push(run.end + 0.02, TRANSPARENT);
  }
  return expr;
}

/**
 * Legend rows for the confidence overlay — same table, same colors as the map
 * expression (the iconDataURL one-source-of-truth principle). Land is omitted
 * (transparent on the map = absent from the legend).
 */
export function tidConfidenceLegend(): {
  group: TidGroup;
  label: string;
  meaning: string;
  color: string;
}[] {
  return (["direct", "indirect", "unknown"] as TidGroup[]).map((g) => ({
    group: g,
    ...TID_CONFIDENCE[g],
  }));
}

// ── committed regional assets ────────────────────────────────────────────────

export interface SeafloorV2Region {
  name: string;
  /** raster-dem source URL (pmtiles protocol over express static). */
  demUrl: string;
  /** TID-code raster-dem source URL. */
  tidUrl: string;
  /** provenance sidecar (attribution, extent, measured TID shares). */
  provenanceUrl: string;
  tileSize: 256;
  encoding: "terrarium";
  minzoom: number;
  maxzoom: number;
  /** true coverage bbox [W, S, E, N] — outside it the layer is transparent. */
  bbox: [number, number, number, number];
}

/**
 * Regions with committed v2 assets. Coverage honesty: these are REGIONAL
 * subsets (the pipeline scales by adding fetch bboxes); everywhere else the
 * v1 ETOPO1 layer remains the global fallback.
 */
export const SEAFLOOR_V2_REGIONS: readonly SeafloorV2Region[] = [
  {
    name: "mariana",
    demUrl: "/tiles/seafloor_gebco_mariana.pmtiles",
    tidUrl: "/tiles/seafloor_tid_mariana.pmtiles",
    provenanceUrl: "/tiles/seafloor_gebco_mariana.provenance.json",
    tileSize: 256,
    encoding: "terrarium",
    minzoom: 0,
    maxzoom: 9,
    bbox: [138, 7, 146.0000000007, 15.0000000007],
  },
];
