/**
 * gnssIntegritySignal.ts — the FIRST candidate root to reach the
 * "gated-SIGNAL /data surface" milestone described in
 * research/open_questions.md's gnss_integrity_adsb entry (2026-08-13
 * NEXT note (c): "if both hold, this becomes the first candidate root
 * queued for a RAW-overlay + gated-SIGNAL /data surface per the
 * SPINOUT-READY DATA LAYER rule"). GATE 2 (SIGNAL) passed and
 * strengthened across two re-runs (2026-08-13 at 2 days, 2026-08-15 at 4
 * days — datacore/signal_ladder.json, current_gate 2, status gate2_pass);
 * GATE 1 is PARTIAL, not full (DTU Space's Bornholm RF station
 * independently corroborates the phenomenon/region, not the exact
 * sample days) — every caveat this module emits mirrors that honestly,
 * per PREMIUM EXPERIENCE STANDARD (c): "premium presentation of wrong
 * numbers is fraud with good typography."
 *
 * CANONICAL LOCATION for the band-verdict statistics (moved here from
 * scripts/gnss_integrity_gate2.ts, which now imports these instead of
 * carrying its own copy — EDGE DOCTRINE #3, one definition, not two
 * drifting copies). The script keeps its own role: an ad-hoc CLI runner
 * against arbitrary days/bboxes via the token-gated diag endpoint, for a
 * human or session re-verifying a specific historical claim. This module
 * is the LIVE, public, no-token path: it reads the archive directly
 * (same privacy contract as gnssIntegrityQuery.ts's own diag probe — only
 * band x origin AGGREGATE COUNTS ever leave this module, no per-row
 * lat/lon, tail number, or callsign) over a rolling recent window, so
 * `/api/data/gnss-integrity-signal` always answers with the freshest
 * accumulated evidence instead of a hardcoded date range going stale.
 */
import { binomialUpperTailP } from "../scripts/statsUtils";
import { readGnssIntegrityWindow, type Bbox } from "./gnssIntegrityQuery";

export interface DiagCell {
  band: string;
  origin: string;
  n_total: number;
  n_zero: number;
  distinct_airframes: number;
}

export const SIG_THRESHOLD = 0.01;
export const EXPECTED_ELEVATED_BANDS = ["cruise", "mid"] as const;
export const EXPECTED_NULL_BANDS = ["low", "ground"] as const;

export interface BandVerdict {
  band: string;
  candidate_k: number;
  candidate_n: number;
  control_rate: number;
  expected_under_null: number;
  p_value: number;
  elevated: boolean; // observed rate > control rate AND p < SIG_THRESHOLD
  expected_to_elevate: boolean; // per the pre-registered physical hypothesis
}

/** Only broadcast-origin cells carry a genuine "the aircraft told us its
 *  own GPS is degraded" reading (see gnssIntegrityQuery.ts's own header)
 *  — ground/mode_s/unknown origin cells are aggregator-derived and
 *  excluded from this test. */
function broadcastOnly(cells: DiagCell[]): DiagCell[] {
  return cells.filter((c) => c.origin === "broadcast");
}

/**
 * Pure comparison: for each band present in BOTH the candidate and control
 * cell sets, test whether the candidate's zero-rate is elevated beyond
 * chance under the control's own observed rate as the null.
 */
export function evaluateBands(candidateCells: DiagCell[], controlCells: DiagCell[]): BandVerdict[] {
  const candidate = broadcastOnly(candidateCells);
  const control = broadcastOnly(controlCells);
  const controlByBand = new Map(control.map((c) => [c.band, c]));
  const verdicts: BandVerdict[] = [];
  for (const c of candidate) {
    const ctrl = controlByBand.get(c.band);
    if (!ctrl || ctrl.n_total === 0 || c.n_total === 0) continue;
    const controlRate = ctrl.n_zero / ctrl.n_total;
    const expected = c.n_total * controlRate;
    const pValue = binomialUpperTailP(c.n_zero, c.n_total, controlRate);
    verdicts.push({
      band: c.band,
      candidate_k: c.n_zero,
      candidate_n: c.n_total,
      control_rate: controlRate,
      expected_under_null: expected,
      p_value: pValue,
      elevated: c.n_zero > expected && pValue < SIG_THRESHOLD,
      expected_to_elevate: (EXPECTED_ELEVATED_BANDS as readonly string[]).includes(c.band),
    });
  }
  return verdicts.sort((a, b) => a.band.localeCompare(b.band));
}

/**
 * Overall gate-2 verdict per the pre-registered bar: PASS requires (a) at
 * least one EXPECTED-elevated band actually shows significant elevation,
 * AND (b) no EXPECTED-null band shows significant elevation (a hit
 * everywhere is a data-artifact pattern, not a targeted signature — see
 * this file's header). Anything else is INCONCLUSIVE, never silently
 * rounded up.
 */
export function gate2Verdict(verdicts: BandVerdict[]): "PASS" | "FAIL" | "INCONCLUSIVE" {
  const expectedElevatedHit = verdicts.some((v) => v.expected_to_elevate && v.elevated);
  const unexpectedNullBandElevated = verdicts.some(
    (v) => !v.expected_to_elevate && (EXPECTED_NULL_BANDS as readonly string[]).includes(v.band) && v.elevated,
  );
  if (unexpectedNullBandElevated) return "FAIL"; // artifact pattern, not a targeted signature
  if (expectedElevatedHit) return "PASS";
  return "INCONCLUSIVE";
}

// Pre-registered regions (research/open_questions.md 2026-08-11 item 1):
// Baltic candidate = documented Russian GPS jamming corridor; control =
// NY + Paris, ordinary GPS reception, no known jamming activity.
export const CANDIDATE_BBOX: Bbox = { lamin: 53, lamax: 60, lomin: 17, lomax: 24 };
export const CANDIDATE_LABEL = "Baltic corridor (Gdańsk–Bornholm–Gotland)";
export const CONTROL_BBOX: Bbox = { lamin: 35, lamax: 55, lomin: -80, lomax: 10 };
export const CONTROL_LABEL = "NY + Paris control region";

// The archive writer that carries integrity fields (nic/pos_type) only
// started 2026-08-11T23:19Z (research/open_questions.md) — no day before
// this has any integrity data to read, so the rolling window never
// reaches further back than this regardless of how large maxDays is.
export const WRITER_LIVE_SINCE = "2026-08-11";

const DAY_MS = 86_400_000;

/** Last `maxDays` calendar days (UTC, most-recent-first) not earlier than
 *  `since` — the same "trailing window, floor at a known start date"
 *  idiom used by apiKeyAccounts.ts's usage-history days. `now` is
 *  injectable for deterministic tests. */
export function recentDays(maxDays: number, since: string = WRITER_LIVE_SINCE, now: number = Date.now()): string[] {
  const days: string[] = [];
  for (let i = 0; i < maxDays; i++) {
    const d = new Date(now - i * DAY_MS).toISOString().slice(0, 10);
    if (d < since) break;
    days.push(d);
  }
  return days;
}

export interface GnssIntegritySignalSummary {
  kind: "signal";
  root_id: "gnss_integrity_adsb";
  generated_at: string;
  gate: { current_gate: 2; status: "gate2_pass" };
  verdict: "PASS" | "FAIL" | "INCONCLUSIVE";
  bands: BandVerdict[];
  region: {
    candidate_bbox: [number, number, number, number];
    candidate_label: string;
    control_bbox: [number, number, number, number];
    control_label: string;
  };
  freshness: {
    writer_live_since: string;
    candidate: { days_read: string[]; days_missing: string[]; rows_scanned: number; truncated: boolean };
    control: { days_read: string[]; days_missing: string[]; rows_scanned: number; truncated: boolean };
  };
  methodology_note: string;
  caveats: string[];
  license: { source: string; note: string };
}

const METHODOLOGY_NOTE =
  "Broadcast-origin ADS-B rows only (excludes mlat/multilateration-derived rows, which reflect the ground " +
  "receiver network's own position solver, not the aircraft's GPS). One-tailed exact binomial test per " +
  "altitude band: is the candidate region's rate of nic==0 (zero position-integrity containment) elevated " +
  "beyond chance versus the control region's own observed rate as the null, at p<0.01. Pre-registered " +
  "expectation: elevation at cruise/mid altitude (line-of-sight to a ground jammer), NOT at low/ground " +
  "(both regions have ordinary near-surface GPS reception) — a hit everywhere would look like a data " +
  "artifact, not a targeted interference signature, and is scored as a FAIL, not a stronger pass.";

const CAVEATS = [
  "GATE 1 is PARTIAL, not a full pass: DTU Space's Tein RF monitoring station on Bornholm independently " +
    "confirms GNSS jamming/spoofing is real, ongoing, and elevated in this exact corridor throughout 2026 " +
    "(30 incidents 2026 to date vs. 16 in all of 2025) — but that source is not day-exact, so it corroborates " +
    "the phenomenon and region, not this specific sample's specific dates.",
  "Small, growing sample: the archive has carried integrity fields only since 2026-08-11 — every day in " +
    "the window below is necessarily recent. The effect has held and strengthened across two re-runs so far; " +
    "treat it as durable only as the window keeps growing without reversing.",
  "Not tradeable: this is GATE 2 (statistical discrimination), not gate 3 (LOGIC/backtested entry-exit) — " +
    "no position sizing or trading decision is made from this signal.",
];

const LICENSE_NOTE =
  "Aircraft archive sourced from adsb.lol (ODbL 1.0, primary, monetization-lawful) with adsb.fi and " +
  "airplanes.live as non-commercial-licensed fallbacks (server/providerCompliance.ts enforces this stays " +
  "true — no billing is active on this feature). Any future SOLD surface of this signal must derive from " +
  "adsb.lol data alone, per the MONETIZATION TRIPWIRE license condition on this root.";

/**
 * Live computation: reads the aircraft archive over a rolling window
 * (capped at 21 days, mirroring the diag probe's own cap) for the
 * candidate and control regions, evaluates band verdicts, and packages
 * the result with full freshness/provenance/caveat metadata. No per-row
 * PII of any kind is included — only aggregate band x origin counts and
 * the derived statistics.
 */
export async function computeGnssIntegritySignal(
  maxDays = 21, baseDir?: string, now: number = Date.now(),
): Promise<GnssIntegritySignalSummary> {
  const days = recentDays(maxDays, WRITER_LIVE_SINCE, now);
  const [candidate, control] = await Promise.all([
    readGnssIntegrityWindow(days, CANDIDATE_BBOX, baseDir),
    readGnssIntegrityWindow(days, CONTROL_BBOX, baseDir),
  ]);
  const bands = evaluateBands(candidate.cells as unknown as DiagCell[], control.cells as unknown as DiagCell[]);
  return {
    kind: "signal",
    root_id: "gnss_integrity_adsb",
    generated_at: new Date(now).toISOString(),
    gate: { current_gate: 2, status: "gate2_pass" },
    verdict: gate2Verdict(bands),
    bands,
    region: {
      candidate_bbox: [CANDIDATE_BBOX.lamin, CANDIDATE_BBOX.lamax, CANDIDATE_BBOX.lomin, CANDIDATE_BBOX.lomax],
      candidate_label: CANDIDATE_LABEL,
      control_bbox: [CONTROL_BBOX.lamin, CONTROL_BBOX.lamax, CONTROL_BBOX.lomin, CONTROL_BBOX.lomax],
      control_label: CONTROL_LABEL,
    },
    freshness: {
      writer_live_since: WRITER_LIVE_SINCE,
      candidate: {
        days_read: candidate.days_read, days_missing: candidate.days_missing,
        rows_scanned: candidate.rows_scanned, truncated: candidate.truncated,
      },
      control: {
        days_read: control.days_read, days_missing: control.days_missing,
        rows_scanned: control.rows_scanned, truncated: control.truncated,
      },
    },
    methodology_note: METHODOLOGY_NOTE,
    caveats: CAVEATS,
    license: { source: "adsb.lol (ODbL 1.0) + adsb.fi/airplanes.live (non-commercial fallbacks)", note: LICENSE_NOTE },
  };
}
