/**
 * gate1_eia930_revision_probe.ts — ROOT VALIDATION LADDER gate 1 evidence
 * for `server/gridDemand.ts` (EIA-930 grid demand). See
 * `server/gridDemandRevision.ts`'s header for why an independent re-draw
 * of EIA's own API is the honestly-available ground-truth check here.
 *
 * Fetches the live 48h/12-respondent/D+DF window TWICE, `--wait-min`
 * apart (default 20, well past the ~4-minute anecdotal check this script
 * replaces), using the real production `fetchDemand()` unchanged, then
 * reports revision statistics via `computeRevisionStats`.
 *
 * OUTPUT: prints a JSON report to stdout. As with the sibling
 * gate1_usaspending_ticker_check.ts precedent, this script produces
 * evidence, not the verdict — the ladder-status update is a human/agent
 * judgment step recorded afterward in research/open_questions.md and
 * research/experiments.md.
 *
 * Usage: npx tsx scripts/gate1_eia930_revision_probe.ts [wait-min]
 * Requires EIA_API_KEY in the environment.
 */
import { fetchDemand } from "../server/gridDemand";
import { computeRevisionStats } from "../server/gridDemandRevision";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function main() {
  if (!process.env.EIA_API_KEY) {
    console.error("EIA_API_KEY not set — nothing to probe.");
    process.exit(1);
  }
  const waitMin = Number(process.argv[2] || 20);

  const draw1At = Date.now();
  const draw1 = await fetchDemand();
  console.error(`draw1: ${draw1.length} rows at ${new Date(draw1At).toISOString()}`);

  console.error(`waiting ${waitMin} min before draw2...`);
  await sleep(waitMin * 60_000);

  const draw2At = Date.now();
  const draw2 = await fetchDemand();
  console.error(`draw2: ${draw2.length} rows at ${new Date(draw2At).toISOString()}`);

  const report = computeRevisionStats(draw1, draw1At, draw2, draw2At);
  console.log(JSON.stringify(report, null, 2));
}

main().catch((e) => { console.error(e); process.exit(1); });
