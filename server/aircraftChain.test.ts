// Source-level regression tests for the aircraft provider chain
// (mirrors test_audit_critical.py's source-check style: the chain lives
// inside registerRoutes' closure, so behavior is pinned at the source).
//
// 2026-07-03 (human decision): OpenSky removed from the chain — Railway
// egress rejects it even with OAuth creds AND its license requires a
// written agreement for operational use (requested; reinstate if granted).
// adsb.lol (ODbL 1.0) is primary; airplanes.live is fallback and is
// licensed NON-COMMERCIAL — the monetization tripwire in wishlist.md
// depends on this ordering staying deliberate.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const src = fs.readFileSync(path.join(here, "routes.ts"), "utf8");

test("OpenSky is fully removed from the runtime chain (no dead 12s attempt)", () => {
  assert.ok(!src.includes("opensky-network.org/api"), "OpenSky states API still fetched");
  assert.ok(!src.includes("auth.opensky-network.org"), "OpenSky OAuth endpoint still present");
  assert.ok(!src.includes("OPENSKY_CLIENT_ID"), "OpenSky creds still read");
  assert.ok(!src.includes('backoffActive("opensky")'), "OpenSky backoff branch still present");
});

test("chain is three deep and adsb.lol (ODbL) leads", () => {
  const adsblol = src.indexOf("api.adsb.lol/v2/point");
  const airplaneslive = src.indexOf("api.airplanes.live/v2/point");
  const adsbfi = src.indexOf("opendata.adsb.fi/api/v2");
  assert.ok(adsblol > 0, "adsb.lol provider missing");
  assert.ok(airplaneslive > 0, "airplanes.live provider missing");
  assert.ok(adsbfi > 0, "adsb.fi provider missing (third leg, human directive 2026-07-03)");
  assert.ok(adsblol < airplaneslive && airplaneslive < adsbfi,
    "order must be adsb.lol (ODbL, monetization-lawful) -> airplanes.live -> adsb.fi (both non-commercial)");
});

test("coverage note no longer promises OpenSky credentials", () => {
  assert.ok(!src.includes("add OpenSky credentials"), "stale coverage note");
});

test("layers.json attribution matches the live chain", () => {
  const layers = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "layers.json"), "utf8"));
  const aircraft = layers.layers.find((l: any) => l.id === "aircraft");
  assert.ok(aircraft.source.includes("adsb.lol"), "aircraft layer attribution stale");
  assert.ok(!aircraft.source.includes("OpenSky"), "aircraft layer still attributes OpenSky");
});
