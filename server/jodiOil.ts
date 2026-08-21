/**
 * jodiOil.ts — JODI World oil closing-stock levels, /data client view.
 *
 * Source: datacore/jodi/primary_stocks.json (scripts/jodi_oil.py, session-run
 * monthly after JODI's ~19th release; never a live Railway poll — same
 * "seeded pattern" as military_installations.json / gemMethane.ts). Ships
 * TOTCRUDE (closing stock levels, thousand barrels) for whichever of the
 * 96 reporting areas last published — CRUDEOIL/NGL/OTHERCRUDE also exist in
 * the underlying archive but are out of scope for this v1 view (one logical
 * change; can be added later if there's a concrete use).
 *
 * LADDER STATUS (RAW OVERLAYS vs SIGNALS, CLAUDE.md): GATE 1 (data) PASSED
 * 2026-08-06 — reconciles against EIA within 1.2% via
 * scripts/jodi_eia_reconcile.py. GATE 2 (signal) KILLED same day —
 * scripts/jodi_gate2_test.py pre-registered that a non-OECD stock-BUILD
 * composite predicts a negative BNO/USO forward return; none of the 4
 * pre-registered comparisons cleared even an uncorrected 0.05 bar (see
 * research/open_questions.md, datacore/signal_ladder.json id
 * "jodi_oil_stocks"). The gate-1 RAW archive stands unaffected — this view
 * surfaces it as self-reported levels only, no predictive claim, exactly the
 * posture research/experiments.md's kill entry states.
 *
 * Per-country staleness varies a lot and is NOT smoothed over: some areas
 * (e.g. UAE) stopped reporting TOTCRUDE years before the archive's overall
 * latest_period — every row carries its OWN period, not a shared "as of"
 * date, so the UI can be honest about it row by row.
 */
import jodiPrimaryStocks from "../datacore/jodi/primary_stocks.json";

export const JODI_PRODUCT = "TOTCRUDE" as const;

export const JODI_GATE_NOTE =
  "GATE 1 (data) PASSED — JODI's closing-stock levels reconcile against EIA " +
  "within 1.2% (scripts/jodi_eia_reconcile.py). GATE 2 (signal) KILLED " +
  "2026-08-06 — a pre-registered non-OECD stock-build composite found no " +
  "significant forward-return signal in BNO/USO. Shown here as RAW, " +
  "self-reported closing-stock levels only — no predictive claim.";

// ISO 3166-1 alpha-2 -> name, limited to the 96 REF_AREA codes JODI's own
// primary database ships (verified against datacore/jodi/primary_stocks.json
// at build time by jodiOil.test.ts — never a general-purpose country table).
export const JODI_AREA_NAMES: Record<string, string> = {
  AE: "United Arab Emirates", AO: "Angola", AR: "Argentina", AT: "Austria",
  AU: "Australia", AZ: "Azerbaijan", BB: "Barbados", BE: "Belgium",
  BG: "Bulgaria", BH: "Bahrain", BN: "Brunei Darussalam", BO: "Bolivia",
  BR: "Brazil", BZ: "Belize", CA: "Canada", CH: "Switzerland", CL: "Chile",
  CO: "Colombia", CR: "Costa Rica", CU: "Cuba", CY: "Cyprus", CZ: "Czechia",
  DE: "Germany", DK: "Denmark", DO: "Dominican Republic", DZ: "Algeria",
  EC: "Ecuador", EE: "Estonia", ES: "Spain", FI: "Finland", FR: "France",
  GA: "Gabon", GB: "United Kingdom", GD: "Grenada", GQ: "Equatorial Guinea",
  GR: "Greece", GT: "Guatemala", GY: "Guyana", HN: "Honduras",
  HR: "Croatia", HT: "Haiti", HU: "Hungary", ID: "Indonesia", IE: "Ireland",
  IN: "India", IQ: "Iraq", IR: "Iran", IS: "Iceland", IT: "Italy",
  JM: "Jamaica", JP: "Japan", KR: "South Korea", KW: "Kuwait",
  KZ: "Kazakhstan", LT: "Lithuania", LU: "Luxembourg", LV: "Latvia",
  LY: "Libya", MA: "Morocco", MK: "North Macedonia", MM: "Myanmar",
  MX: "Mexico", NG: "Nigeria", NI: "Nicaragua", NL: "Netherlands",
  NO: "Norway", NZ: "New Zealand", OM: "Oman", PA: "Panama", PE: "Peru",
  PG: "Papua New Guinea", PH: "Philippines", PL: "Poland", PT: "Portugal",
  PY: "Paraguay", QA: "Qatar", RO: "Romania", RS: "Serbia", RU: "Russia",
  SA: "Saudi Arabia", SE: "Sweden", SI: "Slovenia", SK: "Slovakia",
  SR: "Suriname", SV: "El Salvador", TH: "Thailand", TJ: "Tajikistan",
  TN: "Tunisia", TR: "Turkey", TT: "Trinidad and Tobago", TW: "Taiwan",
  UA: "Ukraine", US: "United States", UY: "Uruguay", VE: "Venezuela",
  ZA: "South Africa",
};

export interface JodiCountryRow {
  area: string;
  name: string;
  period: string;
  levelKbbl: number;
  assessmentCode: string;
  priorPeriod: string | null;
  priorLevelKbbl: number | null;
  deltaKbbl: number | null;
  deltaPct: number | null;
}

export interface JodiOilStocksView {
  kind: "raw";
  predictive: false;
  source: string;
  attribution: string;
  license: string;
  product: typeof JODI_PRODUCT;
  archiveLatestPeriod: string;
  seriesCount: number;
  countriesReporting: number;
  rows: JodiCountryRow[];
  note: string;
}

// The data file itself carries no license field (that lives in
// datacore/manifests/jodi.json); the exact wording is restated here
// verbatim rather than read from a second file.
const JODI_LICENSE = "free with acknowledgment (JODI data are publicly available for use with attribution)";

type JodiPoint = [string, number, string];
interface JodiSeries { points: JodiPoint[]; n: number; first: string; last: string }
interface JodiPrimaryStocksFile {
  source: string; attribution: string;
  latest_period: string; series_count: number;
  series: Record<string, JodiSeries>;
}

/** Builds the TOTCRUDE closing-stock-level view: latest reported point per
 *  country, sorted by level descending. Never zero-fills a gap — a country
 *  with no TOTCRUDE series at all is simply absent from `rows`, matching
 *  the source script's own "skip, never zero" rule for missing periods. */
export function jodiOilStocksView(
  doc: JodiPrimaryStocksFile = jodiPrimaryStocks as unknown as JodiPrimaryStocksFile,
): JodiOilStocksView {
  const rows: JodiCountryRow[] = [];
  for (const [key, s] of Object.entries(doc.series)) {
    const [area, product] = key.split("|");
    if (product !== JODI_PRODUCT) continue;
    const pts = s.points;
    if (!pts || !pts.length) continue;
    const last = pts[pts.length - 1];
    const prior = pts.length > 1 ? pts[pts.length - 2] : null;
    const level = last[1];
    const priorLevel = prior ? prior[1] : null;
    rows.push({
      area,
      name: JODI_AREA_NAMES[area] || area,
      period: last[0],
      levelKbbl: level,
      assessmentCode: last[2],
      priorPeriod: prior ? prior[0] : null,
      priorLevelKbbl: priorLevel,
      deltaKbbl: priorLevel != null ? Math.round((level - priorLevel) * 10) / 10 : null,
      deltaPct: priorLevel ? Math.round(((level - priorLevel) / priorLevel) * 1000) / 10 : null,
    });
  }
  rows.sort((a, b) => b.levelKbbl - a.levelKbbl);
  return {
    kind: "raw",
    predictive: false,
    source: doc.source,
    attribution: doc.attribution,
    license: JODI_LICENSE,
    product: JODI_PRODUCT,
    archiveLatestPeriod: doc.latest_period,
    seriesCount: doc.series_count,
    countriesReporting: rows.length,
    rows,
    note: JODI_GATE_NOTE,
  };
}
