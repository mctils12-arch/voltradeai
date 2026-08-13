/**
 * gnss_integrity_gate2.ts — GATE 2 (SIGNAL layer) for the gnss_integrity
 * root: does the ADS-B integrity-field passthrough (nic/nac_p/.../pos_type,
 * archived since v1.0.662) discriminate a known-real GPS-interference
 * region from an unaffected control region, with no trading involved?
 *
 * ROOT CONTEXT: research/open_questions.md's 2026-08-11 "5 Bilawal-derived
 * build candidates" entry, item 1, is where this hypothesis, its band/
 * origin methodology, and its Baltic-vs-control bboxes were PRE-REGISTERED
 * (stated before this script existed or ran): broadcast-origin rows only
 * (91.7-96.4% of raw nic==0 rows in the anomalous region carry `mlat`,
 * meaning readsb's ground network computed nic/rc for that row rather than
 * the aircraft broadcasting it — counting those would misattribute a
 * ground-station artifact to the aircraft's own GPS), altitude-band
 * stratified (the physical line-of-sight signature: interference should
 * show at cruise/mid, NOT at low/ground, where sky visibility to a jammer
 * differs), Baltic bbox 53,60,17,24 (documented Russian GPS jamming
 * region) vs. NY+Paris control bbox 35,55,-80,10.
 *
 * Phase 1 (ground-truth validation) and Phase 2 (archive writer) shipped
 * 2026-08-11 (v1.0.662); Phase 3 (the /api/diag/gnss_integrity read path)
 * shipped later the same day. TWO subsequent sessions (2026-08-12, own
 * PRs #803/#800) attempted the actual Phase 4 statistical run and BOTH
 * found and fixed infrastructure blockers in the archive reader instead of
 * producing a result (temporal starvation, then geographic/bbox
 * starvation — full traces in experiments.md). This script is the first
 * to run against the fixed reader (server_version 1.0.688, confirmed live
 * before this script was written) and is written so the comparison is
 * REUSABLE — any future region/date range/root with the same
 * "rare-event rate in a small region vs. a control" shape reuses this
 * exact harness rather than a one-off curl (EDGE DOCTRINE #3: compile
 * knowledge into code, never re-derive by hand a second time).
 *
 * PRE-REGISTERED CRITERIA (this script formalizes, not invents, the
 * 2026-08-11 entry's stated methodology):
 *   - Statistical method: one-tailed exact binomial test per band
 *     (statsUtils.binomialUpperTailP) — is the candidate region's
 *     broadcast-origin nic==0 count, given its sample size, higher than
 *     chance under the CONTROL region's own observed zero-rate as the
 *     null rate? This is deliberately NOT a two-proportion z-test: with
 *     small candidate-region counts (tens of rows per band) the exact
 *     binomial tail is the correct, non-approximated test.
 *   - Significance bar: p < 0.01 (one-tailed), pre-stated here before
 *     this run, stricter than the conventional 0.05 given how many
 *     band/origin combinations exist to informally eyeball (Reasoning
 *     Standard #4 — discount for the number of things that could be
 *     "found" by chance).
 *   - EXPECTED pattern, stated in advance per the 2026-08-11 physical
 *     hypothesis: cruise and mid bands should show elevation; low and
 *     ground bands should NOT (both regions have ordinary GPS reception
 *     near the surface) — a hit that shows elevation everywhere,
 *     including where the physical hypothesis says it shouldn't, is
 *     LESS credible, not more (it looks like a data artifact, not a
 *     targeted jamming signature).
 *   - Direction is not free either: only an ELEVATED candidate rate
 *     counts as a pass. A SUPPRESSED candidate rate is not evidence for
 *     the interference hypothesis and is reported as-is, not reframed.
 *
 * HONEST LIMITATIONS, stated up front:
 *   - This is gate 2 (statistical discrimination), not gate 1 (external
 *     ground truth) — gate 1 for this root would be cross-referencing
 *     against an independent jamming-report source (e.g. gpsjam.org,
 *     OPSGROUP NOTAMs) for the same dates/region, which this script does
 *     NOT do. A gate-2 pass here is evidence the signal is REAL and
 *     discriminating, not yet formal proof it matches a specific
 *     documented jamming event.
 *   - Small absolute sample: the archive only carries integrity fields
 *     since the Phase-2 writer merged (2026-08-11 23:19 UTC) — every day
 *     used here is necessarily recent and thin. A pass today should be
 *     re-run as more days accumulate, not treated as final.
 */
import { pathToFileURL } from "url";
import { binomialUpperTailP } from "./statsUtils";

export interface DiagCell {
  band: string;
  origin: string;
  n_total: number;
  n_zero: number;
  distinct_airframes: number;
}

export interface DiagResponse {
  cells: DiagCell[];
  days_read: string[];
  days_missing: string[];
  rows_scanned: number;
  truncated: boolean;
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
 *  own GPS is degraded" reading (see file header) — ground/mode_s/unknown
 *  origin cells are aggregator-derived and excluded from this test. */
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
 * file header). Anything else is INCONCLUSIVE, never silently rounded up.
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

async function fetchDiag(bbox: string, days: string): Promise<DiagResponse> {
  const token = process.env.DIAG_TOKEN;
  if (!token) throw new Error("DIAG_TOKEN env var required (same token used by every other /api/diag/* probe)");
  const base = process.env.VOLTRADE_BASE_URL ?? "https://voltradeai.com";
  const url = `${base}/api/diag/gnss_integrity?days=${encodeURIComponent(days)}&bbox=${encodeURIComponent(bbox)}&limit=50000&token=${token}`;
  const r = await fetch(url, { signal: AbortSignal.timeout(30_000) as any });
  if (!r.ok) throw new Error(`gnss_integrity diag ${bbox} ${days}: HTTP ${r.status}`);
  return (await r.json()) as DiagResponse;
}

async function main() {
  const days = process.argv[2] ?? "2026-08-11,2026-08-12";
  const candidateBbox = process.argv[3] ?? "53,60,17,24"; // Baltic
  const controlBbox = process.argv[4] ?? "35,55,-80,10"; // NY+Paris

  console.log(`gnss_integrity gate-2 run: days=${days} candidate=${candidateBbox} control=${controlBbox}`);
  const [candidate, control] = await Promise.all([
    fetchDiag(candidateBbox, days),
    fetchDiag(controlBbox, days),
  ]);
  console.log(`candidate rows_scanned=${candidate.rows_scanned} truncated=${candidate.truncated} days_missing=${JSON.stringify(candidate.days_missing)}`);
  console.log(`control   rows_scanned=${control.rows_scanned} truncated=${control.truncated} days_missing=${JSON.stringify(control.days_missing)}`);

  const verdicts = evaluateBands(candidate.cells, control.cells);
  console.log("\nband        k      n   control_rate  expected  p_value      elevated  expected_to_elevate");
  for (const v of verdicts) {
    console.log(
      `${v.band.padEnd(10)}  ${String(v.candidate_k).padStart(4)}  ${String(v.candidate_n).padStart(5)}  ` +
      `${(v.control_rate * 100).toFixed(3).padStart(9)}%  ${v.expected_under_null.toFixed(2).padStart(8)}  ` +
      `${v.p_value < 1e-6 ? "<1e-6" : v.p_value.toFixed(6)}      ${String(v.elevated).padEnd(8)}  ${v.expected_to_elevate}`,
    );
  }
  const verdict = gate2Verdict(verdicts);
  console.log(`\nGATE-2 VERDICT (bar: significant elevation in an EXPECTED-elevated band, none in an EXPECTED-null band): ${verdict}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
