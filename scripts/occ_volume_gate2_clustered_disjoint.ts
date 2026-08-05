/**
 * occ_volume_gate2_clustered_disjoint.ts — LADDER PATH STEP (2) for the
 * OCC customer put-heavy skew REVERSAL candidate (research/open_questions.md,
 * filed 2026-08-03, day-clustered step (1) done 2026-08-04: +20d SURVIVES
 * clustering at 5% (t=-2.961 vs critical 2.131), +5d does not — a SPLIT
 * verdict, not a clean pass or kill).
 *
 * Per that entry's own pre-stated step (2): "test on a DISJOINT sample
 * window (e.g. 2026-05 through the most recent >20-trading-day-old date)
 * as true out-of-sample replication — the 2026-01..04-22 window used
 * [in step (1)] must not be reused as its own confirmation."
 *
 * IDENTICAL METHODOLOGY, NOT A NEW VARIANT (do not re-fish): imports
 * `runClusteredDayTest` directly from occ_volume_gate2_clustered.ts so the
 * fetch/bucket/cluster-test logic (same FLOOR=8000, BUCKET_SIZE=40,
 * HORIZONS=[5,20], same clusterMeanTTest/tCrit005 day-clustered stats) is
 * shared code, not a re-implementation that could silently drift — same
 * reuse pattern as cftc_tff_tlt_disjoint_replication.py's reuse of
 * cftc_tff_gate2_test.py's fetch/HAC machinery. Only the DAY LIST changes.
 *
 * WINDOW CHOICE: 9 weekly Wednesdays, 2026-05-06 through 2026-07-01 —
 * chosen, not fished, for two constraints stated before running:
 *   (a) DISJOINT: starts 2 weeks after the original window's last day
 *       (2026-04-22), sharing zero sample days with step (1)'s window
 *       (verified programmatically, see occ_volume_gate2_clustered_
 *       disjoint.test.ts).
 *   (b) FULLY REALIZED: this script runs 2026-08-05; the last sample day
 *       (2026-07-01) is 35 calendar days / ~24 trading days before today,
 *       comfortably clearing the +20-trading-day forward-return horizon
 *       with a buffer (the original script's own convention reaches for
 *       >100 calendar days of margin where the sample window allows it —
 *       here the constraint is "as much history as fits before today
 *       while staying disjoint," so the buffer is smaller but still
 *       positive and stated honestly, not silently thin).
 * A larger disjoint sample was not available: the candidate was only
 * filed 2026-08-03, so no window between the original's end (2026-04-22)
 * and today has more than ~9 realized weekly days without reusing dates
 * still too recent for +20d to have occurred.
 *
 * PRE-STATED PASS BAR (unchanged from step (1), Reasoning Standard #10):
 * if the +20d horizon's SURVIVES flag is true again on this disjoint
 * window, the candidate proceeds to step (3) (IV-selection confound
 * control) before any `gate2_pass` promotion. If SURVIVES flips to false
 * here, this is a FAILED replication — the TLT momentum candidate found
 * the identical outcome on its own disjoint window 2026-08-05, and
 * REASONING STANDARD #2 says a real effect should not vanish on a
 * genuinely different sample of days.
 *
 * Session-run: `npx tsx scripts/occ_volume_gate2_clustered_disjoint.ts`
 * Result goes in research/open_questions.md + research/experiments.md,
 * never into any runtime path or signal_ladder.json gate-2 status — this
 * script touches no production state.
 */
import { pathToFileURL } from "url";
import { weeklySampleDays } from "./occ_volume_gate2";
import { runClusteredDayTest } from "./occ_volume_gate2_clustered";

export const DISJOINT_SAMPLE_START = "2026-05-06"; // Wednesday, 2 weeks after the original window's last day (2026-04-22)
export const DISJOINT_SAMPLE_WEEKS = 9; // through 2026-07-01

async function main() {
  const days = weeklySampleDays(DISJOINT_SAMPLE_START, DISJOINT_SAMPLE_WEEKS);
  const horizonResults = await runClusteredDayTest(days, "occ_volume_gate2_clustered_disjoint");

  console.log(`\n=== STEP (2) VERDICT vs step (1) (original window, 2026-01-07..04-22) ===`);
  console.log(`step (1) result: +5d SURVIVES=false, +20d SURVIVES=true (t=-2.961, crit=2.131)`);
  for (const r of horizonResults) {
    console.log(`disjoint window +${r.horizon}d: mean spread=${(r.result.mean * 100).toFixed(3)}%  t=${r.result.t.toFixed(3)}  df=${r.result.df}  crit=${r.crit.toFixed(3)}  SURVIVES=${r.survives}`);
  }
  const h20 = horizonResults.find((r) => r.horizon === 20);
  if (!h20) {
    console.log(`\nNo +20d result computed (insufficient qualifying days/tickers) — cannot judge step (2), report as a data gap, not a kill.`);
  } else if (h20.survives) {
    console.log(`\nREPLICATES: +20d SURVIVES on this disjoint window too. Proceed to LADDER PATH STEP (3) (IV-selection confound control) before any gate2_pass promotion — do not promote on this result alone.`);
  } else {
    console.log(`\nFAILS REPLICATION: +20d does NOT survive on this disjoint window (the one horizon that survived step (1) does not hold out-of-sample). Per this candidate's own pre-stated ladder path, this candidate should be marked killed in datacore/signal_ladder.json terms — the original step (1) result does not generalize beyond the sample it was found on.`);
  }
}

// Entrypoint guard (ESM has no require.main) — this file's own constants
// are imported by occ_volume_gate2_clustered_disjoint.test.ts; without
// this guard, that import would ALSO trigger a full live OCC+Yahoo fetch
// as an unwanted side effect (the exact bug this session found and fixed
// in occ_volume_gate2.ts / occ_volume_gate2_clustered.ts).
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
