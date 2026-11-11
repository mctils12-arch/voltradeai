/**
 * signalProducts.ts — the Signals PRODUCT catalog behind `#/data/validated-signals`
 * and `GET /api/data/signal-products` (PLATFORM INTEGRATION PROGRAM: the
 * "Signals — validated signal products" nav leaf, which sat disabled as
 * "soon" from 2026-07-11 until this module while two gate-2-passed signals
 * were already live on /data and /api/v1).
 *
 * WHAT THIS IS: a composition over three things that already exist —
 *   (1) datacore/signal_ladder.json (via signalLadder.ts) decides WHICH
 *       roots are validated signals: `isValidatedSignal` = status ends in
 *       "_pass" AND current_gate >= 2. This module never upgrades a status.
 *   (2) SIGNAL_PRODUCTS (below) is the hand-maintained WIRING table: for
 *       each validated root, where its live board, preview JSON, keyed v1
 *       endpoint and LICENSE_MARKS entry live. Facts only — every string is
 *       copied from the owning module's own header/caveats, never inferred.
 *   (3) The live summaries the existing routes already serve (the GNSS
 *       eager-poller cache, the per-request wiki board) are reduced to one
 *       headline + freshness stamp each. No new computation, no new poller.
 *
 * HONESTY CONTRACT (RAW OVERLAYS vs SIGNALS, CLAUDE.md): a root appears as a
 * PRODUCT here only when the ladder says gate 2 passed. A validated root
 * with no wiring row is NOT hidden — it is listed with `live.state ==
 * "unwired"` so the gap is visible on the page, and signalProducts.test.ts
 * pins `unwired` to [] against the real ladder so CI goes red the moment a
 * new root passes gate 2 without its product surface. Gate 2 is statistical
 * discrimination, not a backtested rule: every product row carries
 * `tradeable: false` until a gate-3 pass is recorded in the ladder.
 */
import { isValidatedSignal, type LadderRootView } from "./signalLadder";
import { LICENSE_MARKS } from "./apiProduct";
import type { GnssIntegritySignalSummary } from "./gnssIntegritySignal";
import type { WikiAttentionSignalSummary } from "./wikiAttentionSignal";

export interface SignalProductSpec {
  root_id: string;
  /** Short product name shown on the card (the ladder `name` is the research title). */
  product: string;
  /** One sentence: what the signal measures. Copied from the owning module's header. */
  measures: string;
  /** The exact channel gate 2 validated — never wider than the ladder note. */
  validated_channel: string;
  /** How the live board refreshes (from the serving route, not a guess). */
  cadence: string;
  data_route: string;   // no-token preview JSON
  v1_path: string;      // keyed /api/v1 endpoint
  detail_route: string; // #/data/... live board
  license_key: keyof typeof LICENSE_MARKS | string;
}

export const SIGNAL_PRODUCTS: readonly SignalProductSpec[] = [
  {
    root_id: "gnss_integrity_adsb",
    product: "GNSS interference — Baltic corridor",
    measures:
      "Per-altitude-band GPS position-integrity degradation over the Gdańsk–Bornholm–Gotland corridor, from our own " +
      "broadcast-origin ADS-B archive: a one-tailed exact binomial test per band, the candidate region's nic==0 " +
      "(zero containment) rate against a NY+Paris control region's own observed rate as the null, at p<0.01.",
    validated_channel:
      "Elevation at cruise and mid altitude only (line-of-sight to a ground jammer), not at low/ground — the " +
      "pre-registered pattern; a hit everywhere is scored FAIL. Gate 1 is PARTIAL: DTU Space's Bornholm RF station " +
      "corroborates the phenomenon and region, not the exact sample days.",
    cadence: "Rolling 21-day window over the aircraft archive, recomputed every 10 minutes by the server's own poller.",
    data_route: "/api/data/gnss-integrity-signal",
    v1_path: "/api/v1/data/gnss-integrity-signal",
    detail_route: "#/data/gnss-integrity",
    license_key: "data/gnss-integrity-signal",
  },
  {
    root_id: "wikimedia_pageviews_attention",
    product: "Attention spike → forward volume",
    measures:
      "A Wikipedia pageview spike (z ≥ 2.0 against the article's own trailing baseline of up to 90 days, at least " +
      "20 baseline days) on a curated 23-ticker company-article seed, read from our own daily pageviews archive.",
    validated_channel:
      "Small/mid-cap tickers only: elevated forward trading VOLUME at 1, 3 and 5 days, net of a same-day-or-prior-day " +
      "SEC 8-K (the news-free control), Bonferroni-corrected across a 10-cell family at alpha=0.005 (study 2026-09-04). " +
      "No volatility or directional-price claim — the study found none. Gate 3 (a cost-net entry/exit rule) was " +
      "attempted 2026-09-05 and did not pass.",
    cadence: "Computed per request from the daily pageviews archive (served with a 5-minute cache).",
    data_route: "/api/data/wiki-attention-signal",
    v1_path: "/api/v1/data/wiki-attention-signal",
    detail_route: "#/data/wiki-attention-signal",
    license_key: "data/wiki-attention-signal",
  },
];

export type LiveState = "live" | "warming_up" | "unavailable" | "unwired";

export interface LiveReading {
  state: LiveState;
  /** ISO stamp of the reading the headline describes; null when nothing has been computed. */
  generated_at: string | null;
  /** One line a customer can read without opening the board. */
  headline: string;
}

/** Reduce the GNSS poller cache to a headline. `null` = the first archive scan is still running. */
export function gnssReading(s: GnssIntegritySignalSummary | null | undefined): LiveReading {
  if (!s) return { state: "warming_up", generated_at: null, headline: "first archive scan in progress" };
  const elevated = s.bands.filter((b) => b.elevated).map((b) => b.band);
  const expected = s.bands.filter((b) => b.expected_to_elevate).map((b) => b.band);
  const days = s.freshness?.candidate?.days_read?.length ?? 0;
  const headline =
    `${s.verdict} — ${elevated.length}/${expected.length} expected bands elevated` +
    (elevated.length ? ` (${elevated.join(", ")})` : "") +
    ` over ${days} archive day${days === 1 ? "" : "s"}`;
  return { state: "live", generated_at: s.generated_at, headline };
}

/** Reduce the wiki attention board to a headline. `null` = the computation threw this request. */
export function wikiReading(s: WikiAttentionSignalSummary | null | undefined): LiveReading {
  if (!s) return { state: "unavailable", generated_at: null, headline: "live board unavailable this request" };
  const total = s.tickers.length;
  const withZ = s.tickers.filter((t) => t.z_score != null).length;
  const headline =
    `${s.spike_count} of ${withZ} scored tickers spiking (z ≥ ${s.z_threshold})` +
    (withZ < total ? ` · ${total - withZ} without enough baseline` : "");
  return { state: "live", generated_at: s.generated_at, headline };
}

export interface SignalProductRow extends SignalProductSpec {
  name: string;
  category: string;
  status: string;
  current_gate: number;
  last_update_date: string;
  source_ref: string;
  tradeable: false;
  license: { resell: "ok" | "share-alike" | "conditional" | "unknown"; attribution: string; note: string };
  live: LiveReading;
}

export interface InValidationRow {
  id: string;
  name: string;
  category: string;
  status: string;
  current_gate: number;
  last_update_date: string;
  note: string;
}

export const GATE_DEFINITIONS: readonly { gate: number; name: string; test: string }[] = [
  { gate: 1, name: "DATA", test: "the reading is verified against an external truth source before anything downstream" },
  { gate: 2, name: "SIGNAL", test: "predictive power is measured statistically, out of sample, with no trading involved" },
  { gate: 3, name: "LOGIC", test: "entry/exit rules are backtested by ablation against the validated signal" },
  { gate: 4, name: "SIZING", test: "results are compared against equal-weight to isolate allocation effects" },
  { gate: 5, name: "EXECUTION", test: "the fills tracker's slippage gap isolates fill quality from decision quality" },
];

export interface SignalProductsPayload {
  generated_at: string;
  /** The ladder snapshot's own compile stamp — the products list is only as current as it. */
  ladder_compiled: string;
  products: SignalProductRow[];
  /** Validated roots the ladder knows about that have no product wiring yet (must be empty — see test). */
  unwired: string[];
  /** gate2_pending roots: in validation, NOT products. Notes truncated for the card. */
  in_validation: InValidationRow[];
  gate_definitions: typeof GATE_DEFINITIONS;
  honesty: string;
}

export const IN_VALIDATION_NOTE_CHARS = 280;

const HONESTY =
  "A signal is listed here only after ladder gate 2 (SIGNAL): statistical discrimination on out-of-sample data, " +
  "no trading involved. None of these has passed gate 3 (a backtested entry/exit rule), so nothing on this page is " +
  "a trade recommendation and the bot trades none of it. Every row links to the ladder record its claim comes from.";

function truncateNote(note: string, max = IN_VALIDATION_NOTE_CHARS): string {
  if (note.length <= max) return note;
  const cut = note.slice(0, max);
  const lastSpace = cut.lastIndexOf(" ");
  return `${cut.slice(0, lastSpace > max * 0.6 ? lastSpace : max).trimEnd()}…`;
}

/**
 * Pure composition. `live` is keyed by root_id; a validated+wired root with
 * no entry gets "unavailable" (the route forgot to pass it), never a
 * fabricated reading.
 */
export function buildSignalProducts(
  roots: readonly LadderRootView[],
  ladderCompiled: string,
  live: Readonly<Record<string, LiveReading | undefined>>,
  now: number = Date.now(),
): SignalProductsPayload {
  const specById = new Map(SIGNAL_PRODUCTS.map((s) => [s.root_id, s]));
  const products: SignalProductRow[] = [];
  const unwired: string[] = [];
  const in_validation: InValidationRow[] = [];

  for (const r of roots) {
    if (isValidatedSignal(r.status, r.current_gate)) {
      const spec = specById.get(r.id);
      if (!spec) { unwired.push(r.id); continue; }
      const mark = LICENSE_MARKS[spec.license_key];
      products.push({
        ...spec,
        name: r.name,
        category: r.category,
        status: r.status,
        current_gate: r.current_gate,
        last_update_date: r.last_update_date,
        source_ref: r.source_ref,
        tradeable: false,
        license: mark
          ? { resell: mark.resell, attribution: mark.attribution, note: mark.license }
          : { resell: "unknown", attribution: "", note: `no LICENSE_MARKS entry for ${spec.license_key}` },
        live: live[r.id] ?? { state: "unavailable", generated_at: null, headline: "no live reading supplied" },
      });
    } else if (r.status === "gate2_pending") {
      in_validation.push({
        id: r.id, name: r.name, category: r.category, status: r.status, current_gate: r.current_gate,
        last_update_date: r.last_update_date, note: truncateNote(r.note),
      });
    }
  }

  // Furthest gate first, then the ladder's own last-update recency, then name — deterministic.
  products.sort((a, b) => b.current_gate - a.current_gate || b.last_update_date.localeCompare(a.last_update_date) || a.product.localeCompare(b.product));
  in_validation.sort((a, b) => b.last_update_date.localeCompare(a.last_update_date) || a.name.localeCompare(b.name));

  return {
    generated_at: new Date(now).toISOString(),
    ladder_compiled: ladderCompiled,
    products,
    unwired,
    in_validation,
    gate_definitions: GATE_DEFINITIONS,
    honesty: HONESTY,
  };
}
