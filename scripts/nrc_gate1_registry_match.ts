/**
 * nrc_gate1_registry_match.ts — GATE 1 (DATA layer) for the
 * nrc_outage_reports root: does our parsed NRC Power Reactor Status feed
 * reconcile against the registry it will join to for gate 2 (unit ->
 * plant, MW, operator)?
 *
 * open_questions.md has stated the criterion since 2026-07-03 ("DATA
 * gate = NRC report parse matches registry units"); signal_ladder.json
 * marked it `gate1_pending` with "no build or gate-1 run found in the
 * record" as of the 2026-08-04 crop_conditions gate-1 session's NEXT
 * queue. UNLIKE the crop_conditions/occ_volume gate-1 checks (which
 * compare our output against an INDEPENDENT third-party published
 * bulletin), this gate is a reconciliation between two things we
 * control: the real parser (nrcReactorStatus.ts, imported here — not
 * reimplemented) run against the REAL, LIVE NRC feed, and the REAL
 * registry file on disk. That makes it mechanical and immediate — no
 * production round-trip or accumulated history needed, same class as
 * the earthquake/utility-exposure hypotheses' "gate 1 = proximity join
 * reconciles against known plant coordinates (mechanical, no new data
 * needed)" precedent in open_questions.md.
 *
 * PASS BAR (stated before running, REASONING STANDARD #10): every live
 * NRC unit name resolves to a registry plant (matchRate 1.0), AND every
 * registry plant absent from the match is a documented, independently
 * verified retirement/shutdown (EXPECTED_REGISTRY_ONLY) — a partial
 * match with only unexplained gaps would be a FAIL requiring either a
 * new alias or a genuine data-quality finding, not silently accepted.
 *
 * Usage: npx tsx scripts/nrc_gate1_registry_match.ts
 * Session-run: prints a JSON verdict to stdout. Result goes in
 * research/experiments.md + datacore/signal_ladder.json, never into any
 * production code path — this script touches no runtime state.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { fetchReactorStatus, matchToRegistry, latestDay } from "../server/nrcReactorStatus";

const here = path.dirname(fileURLToPath(import.meta.url));

async function main() {
  const rows = await fetchReactorStatus();
  if (!rows.length) {
    console.log(JSON.stringify({ verdict: "ERROR", reason: "fetchReactorStatus returned zero rows (network or source-shape issue)" }, null, 2));
    process.exitCode = 1;
    return;
  }
  const day = latestDay(rows);
  const nrcUnits = rows.map((r) => r.unit);

  const registryPath = path.join(here, "..", "datacore", "powerplants", "us_power_plants.json");
  const registry = JSON.parse(fs.readFileSync(registryPath, "utf8"));
  const nuclearPlants: string[] = registry.plants
    .filter((p: any[]) => p[2] === "nuclear")
    .map((p: any[]) => p[0]);

  const result = matchToRegistry(nrcUnits, nuclearPlants);
  const verdict = result.unmatched.length === 0 && result.unexpectedRegistryGaps.length === 0 ? "PASS" : "FAIL";

  console.log(JSON.stringify({
    verdict,
    report_date: day,
    nrc_units_checked: new Set(nrcUnits).size,
    registry_plants_checked: nuclearPlants.length,
    matched: result.matched,
    match_rate: result.matchRate,
    unmatched_nrc_units: result.unmatched,
    documented_expected_gaps: result.expectedGaps,
    unexpected_registry_gaps: result.unexpectedRegistryGaps,
    source: `https://www.nrc.gov/reading-rm/doc-collections/event-status/reactor-status/${day.slice(0, 4)}/${day.slice(0, 4)}PowerStatus.txt`,
  }, null, 2));
  if (verdict !== "PASS") process.exitCode = 1;
}

main().catch((e) => {
  console.error(JSON.stringify({ verdict: "ERROR", error: e?.message || String(e) }));
  process.exitCode = 1;
});
