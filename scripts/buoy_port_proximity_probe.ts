/**
 * buoy_port_proximity_probe.ts — GATE 1 (mechanical) check for the
 * sea-state/shipping-lane hypothesis (research/open_questions.md, MARINE/
 * BUOY HAZARD-ADJACENT HYPOTHESES): does a real NDBC buoy exist near each
 * of the 9 imagery-verified port terminals at all?
 *
 * Runs the REAL, unmodified fetchBuoys() against the live NDBC feed and the
 * REAL portsFromSites() against the REAL strategic_sites.json registry —
 * both already-shipped, already-verified modules, not reimplemented here.
 *
 * PASS BAR (stated before running, REASONING STANDARD #10): every port
 * finds at least one reporting buoy within MAX_JOIN_KM. A port coming back
 * unmatched is not automatically a bug (NDBC coverage is real and uneven —
 * some approaches genuinely have no nearby station), but it must be
 * reported honestly, not silently dropped, so a future gate-2 session
 * knows which ports have no buoy-based sea-state feature available.
 *
 * Usage: npx tsx scripts/buoy_port_proximity_probe.ts
 * Session-run: prints a JSON report to stdout. Result goes in
 * research/experiments.md + datacore/signal_ladder.json; this script
 * touches no runtime/trading state.
 */
import { fetchBuoys } from "../server/ndbcBuoys";
import { portsFromSites } from "../server/portDwell";
import { buoyPortProximityReport, hasWaveHeight } from "../server/buoyPortProximity";
import datacoreSites from "../datacore/sites/strategic_sites.json";

async function main() {
  const ports = portsFromSites((datacoreSites as any).sites || []);
  const buoys = await fetchBuoys();
  if (!buoys.length) {
    console.log(JSON.stringify({ verdict: "ERROR", reason: "fetchBuoys returned zero rows (network or source-shape issue)" }, null, 2));
    process.exitCode = 1;
    return;
  }
  // Two views, deliberately: "any station nearby" answers whether NDBC
  // covers the approach at all; "wave-reporting station nearby" answers the
  // question the sea-state hypothesis actually needs. A prior run found
  // these DIVERGE (every port's nearest station is a non-wave harbor
  // gauge) — reporting both keeps that gap visible instead of collapsing
  // it into one number.
  const anyStation = buoyPortProximityReport(ports, buoys);
  const waveOnly = buoyPortProximityReport(ports, buoys, undefined, undefined, hasWaveHeight);
  const verdict = waveOnly.unmatched_ports.length === 0 ? "PASS"
    : waveOnly.ports_matched > 0 ? "PARTIAL" : "FAIL";

  console.log(JSON.stringify({
    verdict,
    ports_checked: ports.length,
    buoy_stations_in_feed: buoys.length,
    wave_reporting_stations_in_feed: buoys.filter(hasWaveHeight).length,
    any_station: anyStation,
    wave_reporting_station: waveOnly,
  }, null, 2));
}

main().catch((e) => {
  console.error(JSON.stringify({ verdict: "ERROR", error: e?.message || String(e) }));
  process.exitCode = 1;
});
