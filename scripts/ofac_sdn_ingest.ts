/**
 * ofac_sdn_ingest.ts — GATE 1 (DATA) reference-list build for the
 * shadow-fleet signal (research/open_questions.md "SHADOW-FLEET SIGNAL").
 * Fetches the live OFAC SDN.XML, parses it via
 * server/shadowFleetReference.ts (dependency-free, unit-tested against a
 * fixed fixture — see shadowFleetReference.test.ts), and writes the
 * MMSI-joinable vessel subset to datacore/ofac_sdn_vessels.json.
 *
 * This is a periodic snapshot builder (same shape as scripts/
 * dtcc_swaps_gate1.ts / scripts/gem_ingest.py), not a runtime poller — the
 * SDN list changes (additions/delistings) on Treasury's own schedule, not
 * ours, and the shadow-fleet detectors this file's output feeds (server/
 * shadowFleet.ts) run against archive windows measured in hours/days, so a
 * reference list re-fetched every few weeks by a future session is
 * accurate enough. Re-running this script overwrites the previous
 * snapshot; `fetched_at`/`record_count` in the output let a reader see how
 * stale it is.
 *
 * Run: npx tsx scripts/ofac_sdn_ingest.ts
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { fetchOfacSdnXml, parseSdnVesselEntries } from "../server/shadowFleetReference";

const here = path.dirname(fileURLToPath(import.meta.url));

async function main() {
  console.log(`Fetching OFAC SDN.XML ...`);
  const xml = await fetchOfacSdnXml();
  console.log(`Fetched ${xml.length} bytes. Parsing ...`);
  const vessels = parseSdnVesselEntries(xml);
  console.log(`${vessels.length} MMSI-joinable vessel entries extracted.`);

  const out = {
    _doc:
      "GATE 1 (DATA) reference list for the shadow-fleet signal " +
      "(research/open_questions.md 'SHADOW-FLEET SIGNAL'): publicly " +
      "documented sanctioned vessels, keyed by MMSI so they can be " +
      "joined against our own AIS archive's gap/loiter detections " +
      "(server/shadowFleet.ts). Built by scripts/ofac_sdn_ingest.ts. " +
      "This is a REFERENCE cohort for a statistical enrichment test " +
      "(server/shadowFleetGate1.ts) — presence on this list is not, by " +
      "itself, a claim about any vessel's current activity.",
    source_url: "https://www.treasury.gov/ofac/downloads/sdn.xml",
    source_name: "OFAC (US Treasury) Specially Designated Nationals (SDN) list",
    license: "US government work product; public domain, no reuse restriction.",
    fetched_at: new Date().toISOString(),
    record_count: vessels.length,
    vessels,
  };

  const outPath = path.join(here, "..", "datacore", "ofac_sdn_vessels.json");
  fs.writeFileSync(outPath, JSON.stringify(out, null, 2) + "\n");
  console.log(`Wrote ${outPath} (${vessels.length} vessels).`);
}

main().catch((e) => {
  console.error("ofac_sdn_ingest failed:", e?.message || e);
  process.exit(1);
});
