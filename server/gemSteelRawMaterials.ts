/**
 * gemSteelRawMaterials.ts — GEM "Global Iron and Steel Tracker" companion
 * release, national met-coal/iron-ore production-consumption balance sheet
 * (RAW-DATA overlay, CLAUDE.md RAW-vs-SIGNAL surface rule): a catalogued
 * national accounting figure, no predictive claim.
 *
 * Source: datacore/gem/steel_raw_materials.json, already ingested by
 * scripts/gem_ingest.py from GEM's "Production-Consumption of Met Coal &
 * Iron Ore by Steel Industry" release (CC BY 4.0, per the release's own
 * Copyright sheet). STATIC reference dataset, same seeded pattern as the
 * rest of the GEM family (gemIronSteelPlants.ts / gemChemicals.ts /
 * gemIronOreMines.ts): GEM ships a new release ~2x/year and a human
 * re-runs the ingest script on delivery — no boot-poll loop.
 *
 * Genuinely different SHAPE from the rest of the GEM family (the
 * open_questions.md GEM-suite backlog note that named this file explicitly
 * called it "a country-level balance sheet ... a future choropleth
 * candidate, not a point layer"): the release's `country_balance` array is
 * a COUNTRY-level accounting balance sheet, not a per-facility row with
 * coordinates, so it has no lat/lon to place a symbol at. Rendered instead
 * as a country-choropleth fill, joined by NAME against the already-
 * vendored Natural Earth 1:110m admin0 boundary set
 * (datacore/boundaries/ne_110m_admin0.json, public domain — the same file
 * server/countryLookup.ts's point-in-polygon reverse lookup already uses;
 * reused here rather than fetching or vendoring a second copy of world
 * boundaries, EDGE DOCTRINE #3: compile once, reuse forever).
 *
 * NAME JOIN: GEM's country names and Natural Earth's 110m admin0 names
 * disagree on spelling/form for a real, finite set of countries (GEM
 * "Czech Republic" vs NE "Czechia", GEM "United States" vs NE "United
 * States of America", etc.) — COUNTRY_NAME_ALIASES below is that finite,
 * hand-verified map (live-diffed this session: all 252 GEM
 * `country_balance` rows against all 177 NE admin0 features), not a fuzzy
 * matcher. Everything left unmatched after aliasing is a genuine
 * micro-state/territory absent from the 110m boundary set at this
 * resolution (Monaco, Singapore, Hong Kong, most small island states,
 * ...) — real data with no polygon to shade, not a bug; it still ships in
 * the flat `balances` array for a table view, just never appears on the
 * choropleth fill.
 *
 * "Global" is GEM's own world-aggregate row (Country: "Global"), not a
 * real country — excluded from both the geometry join and the flat
 * balances array; a per-country choropleth including a "Global" polygon
 * fill would be a fabricated claim.
 *
 * ECOLOGICAL-FALLACY-STYLE GUARD (mirrors cdcCancer.ts's county guard, one
 * level up at the country scale): every admin0 geometry keeps its shape
 * whether or not GEM's release reports a value for it — `has_data:false`,
 * never a dropped polygon (a hole would read as "zero", not "not
 * reported") and never a fabricated number. A reported 0 (most countries
 * genuinely mine no iron ore) is a real fact and stays 0, kept distinct
 * from a null/not-reported field by the choropleth's own NO-DATA color
 * band on the client side.
 */
import fs from "fs";
import { repoDataPath } from "./repoFiles";
import admin0Json from "../datacore/boundaries/ne_110m_admin0.json";

export interface CountryBalance {
  country: string;
  metCoalMinedTtpa: number | null;
  ironOreMinedTtpa: number | null;
  metCoalConsumedPigIronTtpa: number | null;
  ironOreConsumedPigIronTtpa: number | null;
  ironOreConsumedDriTtpa: number | null;
  ironOreConsumedTotalTtpa: number | null;
  pigIronProducedTtpa: number | null;
  driProducedTtpa: number | null;
}

function toNumOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}
function toStrOrNull(v: unknown): string | null {
  if (v == null) return null;
  const s = String(v).trim();
  return s.length > 0 ? s : null;
}

// GEM's own world-aggregate row — not a real country, see this module's header.
const GLOBAL_AGGREGATE_SENTINEL = "global";

/** Normalizes the release's raw per-row columns into the clean schema.
 *  Drops rows with no country name and the "Global" world-aggregate row —
 *  nothing to place on a per-country map. A legitimately reported 0 stays
 *  0; anything non-numeric (including GEM's own "unknown" sentinel, seen
 *  in the consumption/production columns of the current release) degrades
 *  to null, never guessed. */
export function normalizeCountryBalance(rows: Record<string, unknown>[]): CountryBalance[] {
  const out: CountryBalance[] = [];
  for (const r of rows || []) {
    const country = toStrOrNull(r["Country"]);
    if (!country || country.toLowerCase() === GLOBAL_AGGREGATE_SENTINEL) continue;
    out.push({
      country,
      metCoalMinedTtpa: toNumOrNull(r["Met coal mined (ttpa)"]),
      ironOreMinedTtpa: toNumOrNull(r["Iron ore mined (ttpa)"]),
      metCoalConsumedPigIronTtpa: toNumOrNull(r["Met coal consumed by pig iron production (ttpa)"]),
      ironOreConsumedPigIronTtpa: toNumOrNull(r["Iron ore consumed by pig iron production (ttpa)"]),
      ironOreConsumedDriTtpa: toNumOrNull(r["Iron ore consumed by DRI production (ttpa)"]),
      ironOreConsumedTotalTtpa: toNumOrNull(r["Total iron ore consumed by pig iron and DRI production (ttpa)"]),
      pigIronProducedTtpa: toNumOrNull(r["Pig iron produced (ttpa)"]),
      driProducedTtpa: toNumOrNull(r["DRI produced (ttpa)"]),
    });
  }
  return out;
}

export interface CountryBalanceResult {
  release: string | null;
  attribution: string;
  license: string;
  balances: CountryBalance[];
}

/** Reads datacore/gem/steel_raw_materials.json. Never throws: a missing/
 *  corrupt file degrades to null, matching every other gemXxx.ts loader's
 *  fetch-failure-degrades precedent, so the route can serve an honest
 *  "unavailable" instead of crashing the process. */
export function loadGemSteelRawMaterials(
  fp: string = repoDataPath("datacore/gem/steel_raw_materials.json"),
): CountryBalanceResult | null {
  try {
    const raw = JSON.parse(fs.readFileSync(fp, "utf8"));
    const prov = raw.provenance || {};
    return {
      release: toStrOrNull(prov.release),
      attribution: typeof prov.attribution === "string" ? prov.attribution : "Global Energy Monitor",
      license: typeof prov.license === "string" ? prov.license : "CC BY 4.0",
      balances: normalizeCountryBalance(raw.country_balance || []),
    };
  } catch {
    return null;
  }
}

// In-memory cache — same rationale as every other gemXxx.ts `cached`:
// static reference data, parse once per process lifetime.
let cached: CountryBalanceResult | null | undefined;

export function cachedGemSteelRawMaterials(): CountryBalanceResult | null {
  if (cached === undefined) cached = loadGemSteelRawMaterials();
  return cached;
}

/** Test-only: clears the module cache so a test can inject a different fp. */
export function _resetGemSteelRawMaterialsCacheForTests(): void {
  cached = undefined;
}

// ── country-choropleth geometry join ─────────────────────────────────────

/** GEM country name -> Natural Earth 1:110m admin0 feature name, for the
 *  finite set of countries where the two datasets spell the same country
 *  differently. Hand-verified against a live diff of both files this
 *  session (see this module's header) — NOT a fuzzy/normalized matcher, so
 *  a future GEM release renaming a country here simply drops out of the
 *  choropleth join (has_data:false) instead of silently matching the wrong
 *  polygon. */
export const COUNTRY_NAME_ALIASES: Record<string, string> = {
  "Bosnia and Herzegovina": "Bosnia and Herz.",
  "Central African Republic": "Central African Rep.",
  "Czech Republic": "Czechia",
  "DR Congo": "Dem. Rep. Congo",
  "Dominican Republic": "Dominican Rep.",
  "Equatorial Guinea": "Eq. Guinea",
  Eswatini: "eSwatini",
  "Falkland Islands": "Falkland Is.",
  "Republic of the Congo": "Congo",
  "South Sudan": "S. Sudan",
  "Solomon Islands": "Solomon Is.",
  "The Gambia": "Gambia",
  Türkiye: "Turkey",
  "United States": "United States of America",
  "Western Sahara": "W. Sahara",
};

interface Admin0Feature {
  type: "Feature";
  properties: { name: string; iso3: string };
  geometry: GeoJSON.Geometry;
}
interface Admin0Collection {
  type: "FeatureCollection";
  features: Admin0Feature[];
}

/** Joins the admin0 boundary FeatureCollection with the country balance
 *  sheet by name (through COUNTRY_NAME_ALIASES). Every input geometry
 *  keeps its shape — has_data:false + all-null fields on a genuine miss,
 *  never dropped (see this module's header, ecological-fallacy-style
 *  guard). Pure — testable on small fixtures, independent of the real
 *  177-feature boundary file. */
export function joinCountryChoropleth(
  admin0Geo: Admin0Collection,
  balances: CountryBalance[],
): GeoJSON.FeatureCollection {
  const byName = new Map(balances.map((b) => [COUNTRY_NAME_ALIASES[b.country] ?? b.country, b]));
  return {
    type: "FeatureCollection",
    features: admin0Geo.features.map((f) => {
      const name = f.properties?.name ?? null;
      const b = name ? byName.get(name) ?? null : null;
      return {
        type: "Feature",
        geometry: f.geometry,
        properties: {
          name,
          iso3: f.properties?.iso3 ?? null,
          has_data: Boolean(b),
          met_coal_mined_ttpa: b?.metCoalMinedTtpa ?? null,
          iron_ore_mined_ttpa: b?.ironOreMinedTtpa ?? null,
          met_coal_consumed_pig_iron_ttpa: b?.metCoalConsumedPigIronTtpa ?? null,
          iron_ore_consumed_pig_iron_ttpa: b?.ironOreConsumedPigIronTtpa ?? null,
          iron_ore_consumed_dri_ttpa: b?.ironOreConsumedDriTtpa ?? null,
          iron_ore_consumed_total_ttpa: b?.ironOreConsumedTotalTtpa ?? null,
          pig_iron_produced_ttpa: b?.pigIronProducedTtpa ?? null,
          dri_produced_ttpa: b?.driProducedTtpa ?? null,
        },
      } as unknown as GeoJSON.Feature;
    }),
  };
}

export interface SteelRawMaterialsGeo {
  geo: GeoJSON.FeatureCollection;
  matched: number;
  unmatched: number;
}

let cachedGeo: SteelRawMaterialsGeo | null | undefined;

/** Cached join of the balance sheet onto admin0 geometry. Returns null
 *  only when the underlying balance-sheet artifact itself failed to load
 *  (matches cachedGemSteelRawMaterials's own degrade contract) — the
 *  admin0 boundary file is a static import, not a runtime read, so it
 *  cannot independently fail at serve time. */
export function cachedSteelRawMaterialsGeoJSON(): SteelRawMaterialsGeo | null {
  if (cachedGeo !== undefined) return cachedGeo;
  const balances = cachedGemSteelRawMaterials();
  if (!balances) {
    cachedGeo = null;
    return null;
  }
  const geo = joinCountryChoropleth(admin0Json as unknown as Admin0Collection, balances.balances);
  const matched = geo.features.filter((f) => (f.properties as any)?.has_data).length;
  cachedGeo = { geo, matched, unmatched: geo.features.length - matched };
  return cachedGeo;
}

/** Test-only: clears the geometry-join cache. */
export function _resetSteelRawMaterialsGeoCacheForTests(): void {
  cachedGeo = undefined;
}
