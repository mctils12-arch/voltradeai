/**
 * entityJoin.ts — ORBITAL program workstream e-entity (cross-tie (b) in
 * research/orbital_program.md): join satellites into the Everything Graph
 * via operator -> company -> ticker, so a company lookup can show its
 * orbital footprint beside its jets / vessels / plants.
 *
 * This is the CLIENT-SIDE resolver over datacore/orbital/operators.json (a
 * curated, WebSearch-verified seed table, same discipline as
 * datacore/entity_map.json). It performs NO collection and asserts NO
 * predictive claim — it maps a messy SATCAT operator/owner STRING to the
 * public company + ticker where one exists.
 *
 * HONESTY RULE (hard, from the charter + orbital_program.md): a wrong
 * ticker is worse than an honest "unmapped". The normalizer is deliberately
 * CONSERVATIVE — it does exact matching against a curated variant set after
 * light normalization, and returns null ("unmapped") rather than a fuzzy
 * guess. Its honest failure mode is therefore a HIGH UNMAPPED RATE on the
 * full real SATCAT (thousands of tiny/foreign/one-off operators we never
 * seeded) — never a FALSE mapping. That trade is intentional.
 *
 * The trading value of this join is SPECULATIVE and slow (constellation
 * size moves over months); the real justification is STRUCTURAL / SHOWCASE
 * (Everything-Graph connective tissue) — stated plainly per the program.
 */
import operatorsData from "../../../../datacore/orbital/operators.json";

export type OperatorCategory =
  | "launch-provider"
  | "comms"
  | "earth-observation"
  | "navigation"
  | "defense"
  | "gov"
  | "other";

export interface OperatorEntry {
  company: string;
  operator_name_variants: string[];
  ticker: string | null;
  exchange: string | null;
  category: OperatorCategory;
  country: string;
  notes: string;
}

interface OperatorsFile {
  asof: string;
  operators: OperatorEntry[];
}

/** Result of a successful resolve — a projection of the matched entry plus
 *  the provenance of HOW it matched (which raw variant), so a downstream
 *  surface can show the join honestly. */
export interface ResolvedOperator {
  company: string;
  ticker: string | null;
  exchange: string | null;
  category: OperatorCategory;
  country: string;
  notes: string;
  /** true when the operator is real but has no public equity (private/gov)
   *  — distinguishes an honest "mapped, no ticker" from an "unmapped" null. */
  is_public: boolean;
  /** the normalized key that matched — provenance of the join */
  matched_on: string;
  asof: string;
}

const FILE = operatorsData as unknown as OperatorsFile;
export const OPERATORS: OperatorEntry[] = FILE.operators ?? [];
export const OPERATORS_ASOF: string = FILE.asof ?? "unknown";

/** Corporate-FORM suffix tokens stripped during normalization. Deliberately
 *  limited to unambiguous legal-form words — NOT descriptor words like
 *  "communications" / "technologies" / "satellite", which stay so that
 *  distinct companies never collapse together. Matching leans on the
 *  explicit variant list, not aggressive descriptor stripping. */
const FORM_SUFFIX_TOKENS = new Set([
  "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
  "LLC", "LLP", "LP", "LTD", "LIMITED", "SA", "SAS", "PLC", "GMBH",
  "AG", "NV", "BV", "PBC", "AB", "OY", "THE", "GROUP", "HOLDINGS",
]);

/**
 * Normalize a messy operator/owner string to a canonical comparison key:
 *   1. drop parenthetical asides  "... (SpaceX)" -> "..."
 *   2. uppercase; replace every non-alphanumeric run with a single space
 *   3. drop corporate-FORM suffix tokens (INC/LLC/CORP/...)
 *   4. collapse whitespace, trim
 * Exported for testing and for callers that want to pre-key their own data.
 */
export function normalizeOperator(raw: string | null | undefined): string {
  if (!raw) return "";
  const noParen = String(raw).replace(/\([^)]*\)/g, " ");
  const tokens = noParen
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, " ")
    .trim()
    .split(" ")
    .filter((t) => t.length > 0 && !FORM_SUFFIX_TOKENS.has(t));
  return tokens.join(" ");
}

// Build the normalized-variant -> entries index once at module load. Uses an
// array per key so a future accidental collision is representable (and
// resolvable by country) rather than silently overwriting — honesty over
// convenience. The seed is authored collision-free.
const INDEX: Map<string, OperatorEntry[]> = (() => {
  const idx = new Map<string, OperatorEntry[]>();
  for (const entry of OPERATORS) {
    const keys = new Set<string>();
    for (const v of entry.operator_name_variants || []) {
      const k = normalizeOperator(v);
      if (k) keys.add(k);
    }
    // also index the company name itself (minus any parenthetical)
    const ck = normalizeOperator(entry.company);
    if (ck) keys.add(ck);
    for (const k of Array.from(keys)) {
      if (!idx.has(k)) idx.set(k, []);
      const bucket = idx.get(k)!;
      if (!bucket.includes(entry)) bucket.push(entry);
    }
  }
  return idx;
})();

function project(entry: OperatorEntry, matchedOn: string): ResolvedOperator {
  return {
    company: entry.company,
    ticker: entry.ticker,
    exchange: entry.exchange,
    category: entry.category,
    country: entry.country,
    notes: entry.notes,
    is_public: entry.ticker != null,
    matched_on: matchedOn,
    asof: OPERATORS_ASOF,
  };
}

/**
 * Resolve a SATCAT operator/owner string to its company + public ticker.
 * Returns null for an UNMAPPED operator (honest gap) rather than a guess.
 *
 * `country` is an optional disambiguation hint only: it is used solely to
 * break a tie IF a normalized string maps to more than one seeded entry
 * (the seed is authored so this never happens today). A country hint NEVER
 * turns a clean single match into a null — SATCAT country codes are noisy.
 */
export function resolveOperator(
  satcatOperator: string | null | undefined,
  country?: string | null,
): ResolvedOperator | null {
  const key = normalizeOperator(satcatOperator);
  if (!key) return null;
  const hits = INDEX.get(key);
  if (!hits || hits.length === 0) return null;
  if (hits.length === 1) return project(hits[0], key);

  // Rare collision path: prefer entries whose country loosely matches the
  // hint; if that leaves exactly one, take it, else stay honest and null.
  if (country) {
    const c = country.trim().toUpperCase();
    const narrowed = hits.filter(
      (e) => e.country.toUpperCase().includes(c) || c.includes(e.country.toUpperCase()),
    );
    if (narrowed.length === 1) return project(narrowed[0], key);
  }
  return null;
}

/** A satellite already carrying either a resolved ticker or a raw operator
 *  string (plus anything else the caller wants — NORAD id, name, ...). */
export interface JoinedSat {
  operator?: string | null;
  ticker?: string | null;
  country?: string | null;
  [k: string]: unknown;
}

/** Resolve one sat's owning ticker: an explicit `ticker` wins; otherwise
 *  resolve its `operator` string. Returns null when unmapped or private/gov. */
function tickerOf(sat: JoinedSat): string | null {
  if (sat.ticker != null && String(sat.ticker).trim() !== "") return String(sat.ticker).toUpperCase();
  if (sat.operator) return resolveOperator(sat.operator, sat.country ?? undefined)?.ticker ?? null;
  return null;
}

/**
 * The orbital footprint of one company: every satellite in `joinedSats`
 * whose owning company resolves to `ticker`. Ticker match is
 * case-insensitive. Sats that are unmapped or owned by a private/gov entity
 * (null ticker) are simply excluded — never mis-attributed.
 */
export function orbitalFootprintByTicker<T extends JoinedSat>(
  joinedSats: T[],
  ticker: string,
): T[] {
  const want = String(ticker).trim().toUpperCase();
  if (!want) return [];
  return joinedSats.filter((s) => tickerOf(s) === want);
}

/**
 * Group a satellite set by owning ticker in one pass. Sats with no public
 * ticker (unmapped, private, or government) are collected under a single
 * honest `null` key so nothing is silently dropped from the accounting.
 */
export function groupOrbitalFootprint<T extends JoinedSat>(
  joinedSats: T[],
): Map<string | null, T[]> {
  const out = new Map<string | null, T[]>();
  for (const s of joinedSats) {
    const key = tickerOf(s);
    if (!out.has(key)) out.set(key, []);
    out.get(key)!.push(s);
  }
  return out;
}
