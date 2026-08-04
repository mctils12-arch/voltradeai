/**
 * crop_conditions_gate1.ts — GATE 1 (DATA layer) for the
 * crop_conditions_usda_nass root: do our archived/served NASS QuickStats
 * CONDITION rows reconcile against USDA NASS's own PUBLISHED Crop
 * Progress report for the same week?
 *
 * cropConditions.ts's header has stated since 2026-07-06 (v1.0.164):
 * "Gate 1 = values vs the published Crop Progress report" — never run
 * (research/open_questions.md, datacore/signal_ladder.json both noted
 * the gap; most recently flagged again in the 2026-08-03 occ_options_
 * volume gate-2 session's NEXT list).
 *
 * TRUTH SOURCE (hand-verified this session via direct download of the
 * actual government release, not an AI summary of it): USDA NASS's
 * official "Crop Progress" text bulletin, ISSN 1948-3007, released
 * 2026-08-03 4pm ET covering week ending 2026-08-02 —
 * https://esmis.nal.usda.gov/sites/default/release-files/796007/prog3126.txt
 * (esmis.nal.usda.gov is the current home of the former
 * usda.library.cornell.edu Crop Progress archive; confirmed via a
 * direct 301 redirect, not a guess). The "18 States" national summary
 * rows in that file (Corn Condition table, Soybean Condition table) are
 * QuickStats's OWN compiled national figures for the same underlying
 * survey — the strongest available ground truth for this pipeline
 * (there is no third, independent source of the same weekly condition
 * survey; QuickStats and the bulletin are two publications of the same
 * NASS estimate, same discipline as eia930_grid_demand's gate-1 note).
 *
 * TRUTH VALUES (hand-recorded from the downloaded .txt, "18 States"
 * row, percent, order Very poor/Poor/Fair/Good/Excellent):
 *   Corn:      4, 10, 25, 48, 13
 *   Soybeans:  2,  7, 28, 52, 11
 *
 * METHOD: fetch the LIVE production route `/api/data/crop-conditions`
 * (the same cache cropConditions.ts's own poller writes via
 * refreshConditions -> parseConditions against the real NASS_API_KEY in
 * Railway — this session's own container has no NASS_API_KEY, so
 * re-hitting QuickStats directly is not possible; reading what
 * production already served, through the unmodified `item`/`pct`
 * fields the real parser produced, is the honest substitute and avoids
 * re-implementing the parse), compare each of the 10 (commodity, class)
 * cells against the hand-recorded truth above.
 *
 * PASS = exact match (0 pp diff) on every cell for week_ending
 * 2026-08-02. Percent-of-farmland-surveyed figures are integers in the
 * source bulletin, so "close" is not a meaningful outcome here — any
 * nonzero diff is a real parser or query defect.
 *
 * Usage: npx tsx scripts/crop_conditions_gate1.ts [prodUrl]
 * Session-run: prints a JSON verdict to stdout. Result goes in
 * research/experiments.md + datacore/signal_ladder.json, never into any
 * production code path — this script touches no runtime state.
 */

const PROD_URL = process.argv[2] || "https://voltradeai.com/api/data/crop-conditions";

const WEEK_ENDING = "2026-08-02";

// Hand-recorded from prog3126.txt's "18 States" national summary rows,
// verified by direct curl of the government release this session.
const TRUTH: Record<string, Record<string, number>> = {
  CORN: { "VERY POOR": 4, POOR: 10, FAIR: 25, GOOD: 48, EXCELLENT: 13 },
  SOYBEANS: { "VERY POOR": 2, POOR: 7, FAIR: 28, GOOD: 52, EXCELLENT: 11 },
};

interface Row {
  commodity: string;
  week_ending: string;
  item: string;
  pct: number | null;
}

function classFromItem(item: string): string | null {
  const m = item.match(/PCT\s+(VERY POOR|POOR|FAIR|GOOD|EXCELLENT)\b/i);
  return m ? m[1].toUpperCase() : null;
}

async function main() {
  const r = await fetch(PROD_URL, { signal: AbortSignal.timeout(20000) as any });
  if (!r.ok) throw new Error(`${PROD_URL} -> ${r.status}: ${(await r.text()).slice(0, 300)}`);
  const body = await r.json();
  const rows: Row[] = body?.rows || [];
  if (body?.latest_week !== WEEK_ENDING) {
    console.log(JSON.stringify({
      verdict: "SKIPPED",
      reason: `production latest_week is ${body?.latest_week}, this script's hand-recorded truth is only for ${WEEK_ENDING} — re-run with an updated TRUTH table once a new week ships`,
    }, null, 2));
    return;
  }

  const diffs: any[] = [];
  let checked = 0;
  for (const commodity of Object.keys(TRUTH)) {
    for (const cls of Object.keys(TRUTH[commodity])) {
      const truth = TRUTH[commodity][cls];
      const row = rows.find((x) => x.commodity === commodity && x.week_ending === WEEK_ENDING && classFromItem(x.item) === cls);
      checked++;
      const got = row?.pct ?? null;
      if (got !== truth) diffs.push({ commodity, cls, truth, got, found_row: !!row });
    }
  }

  const verdict = diffs.length === 0 && checked === 10 ? "PASS" : "FAIL";
  console.log(JSON.stringify({
    verdict,
    week_ending: WEEK_ENDING,
    cells_checked: checked,
    cells_matched: checked - diffs.length,
    diffs,
    truth_source: "https://esmis.nal.usda.gov/sites/default/release-files/796007/prog3126.txt (18 States national summary rows)",
    prod_url: PROD_URL,
  }, null, 2));
}

main().catch((e) => {
  console.error(JSON.stringify({ verdict: "ERROR", error: e?.message || String(e) }));
  process.exitCode = 1;
});
