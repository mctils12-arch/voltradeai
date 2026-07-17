// ENTITY DOSSIER v2 (W5) — dossier.ts fixture tests. buildDossier is a pure
// function over an injected graph/contracts/sites (mirrors entityGraph.ts's
// own every-IO-source-overridable convention) so these never touch the real
// archive or datacore registries.
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildDossier, NEAREST_SITES_CAP, CONTRACTS_CAP, HAZARD_CAP, HAZARD_RADIUS_KM_DEFAULT, HAZARD_RADIUS_KM_MAX } from "./dossier";
import type { FloodZoneResult } from "./femaFlood";
import type { EverythingGraph, GraphNode, GraphEdge, SiteRow } from "./entityGraph";
import type { ContractTxn } from "./usaSpending";
import type { SuperfundSite } from "./superfund";
import type { WaterViolator } from "./waterViolators";
import type { PfasSystem } from "./pfas";

const node = (over: Partial<GraphNode>): GraphNode => ({
  id: "x", type: "company", label: "x", attrs: {}, ...over,
});
const edge = (over: Partial<GraphEdge>): GraphEdge => ({
  type: "operates", from: "a", to: "b", source: "test", confidence: "high",
  first_seen: null, last_seen: null, attrs: {}, ...over,
});

function graphFixture(nodes: GraphNode[], edges: GraphEdge[]): EverythingGraph {
  return {
    kind: "raw", built_at: 12345,
    counts: { nodes: nodes.length, edges: edges.length, company: 0, person: 0, facility: 0, vessel: 0, institution: 0, insider_of: 0, operates: 0, calls_at: 0, owns: 0 },
    nodes, edges, caveat: "test graph",
  };
}

const sites: SiteRow[] = [
  { id: "cushing_hub", name: "Cushing Oil Hub", category: "tank_farm", lat: 35.94, lon: -96.75, operator: "Enbridge" },
  { id: "la_port", name: "Port of LA", category: "port", lat: 33.73, lon: -118.26, operator: "City of LA (POLA)" },
  { id: "far_away", name: "Far Away Site", category: "tank_farm", lat: 61.2, lon: -149.9 },
];

const contract = (over: Partial<ContractTxn>): ContractTxn => ({
  aid: "aid1", piid: null, mod: "0", ad: "2026-07-01", amt: 50000, r: "Enbridge Inc", uei: null,
  rid: null, pn: null, puei: null, tkr: "ENB", mm: "name", mname: "ENBRIDGE", ag: "DOE", sub: null,
  naics: null, psc: null, atype: "D", desc: null, rt: "2026-07-01", ...over,
});

test("resolves a facility entity, returns its own hops-1 neighborhood, and anchors nearest_sites on its own coordinates", () => {
  const facility = node({ id: "facility:site:cushing_hub", type: "facility", label: "Cushing Oil Hub", attrs: { lat: 35.94, lon: -96.75 } });
  const company = node({ id: "company:ENB", type: "company", label: "ENB", attrs: { tickers: ["ENB"] } });
  const g = graphFixture([facility, company], [edge({ type: "operates", from: "company:ENB", to: "facility:site:cushing_hub" })]);
  const out = buildDossier(g, { entity: "cushing_hub" }, { sites });
  assert.equal(out.identity?.id, "facility:site:cushing_hub");
  assert.equal(out.graph?.nodes.length, 2);
  assert.equal(out.built_at, 12345);
  // self-site excluded from its own nearest list
  assert.ok(!out.nearest_sites.some((s) => s.id === "cushing_hub"));
  assert.equal(out.nearest_sites[0].id, "la_port"); // closer than far_away
});

test("ticker linkage reaches contracts through an operates edge (facility -> company), not just a direct company match", () => {
  const facility = node({ id: "facility:site:cushing_hub", type: "facility", label: "Cushing Oil Hub", attrs: { lat: 35.94, lon: -96.75 } });
  const company = node({ id: "company:ENB", type: "company", label: "ENB", attrs: { tickers: ["ENB"] } });
  const g = graphFixture([facility, company], [edge({ type: "operates", from: "company:ENB", to: "facility:site:cushing_hub" })]);
  const contracts = [contract({ tkr: "ENB", rt: "2026-07-01" }), contract({ tkr: "XOM", rt: "2026-07-02" })];
  const out = buildDossier(g, { entity: "cushing_hub" }, { contracts, sites });
  assert.equal(out.contracts.length, 1);
  assert.equal(out.contracts[0].tkr, "ENB");
});

test("a CIK-fallback company node (no filed ticker) never matches a contract by ticker", () => {
  const cikCo = node({ id: "company:cik:0000999", type: "company", label: "No Ticker Co", attrs: { cik: "0000999", ticker_known: false } });
  const g = graphFixture([cikCo], []);
  const out = buildDossier(g, { entity: "cik:0000999" }, { contracts: [contract({ tkr: "ENB" })], sites });
  assert.equal(out.contracts.length, 0);
});

test("contracts are capped, sorted newest-first, and the cap is surfaced honestly (never a silent truncation)", () => {
  const company = node({ id: "company:ENB", type: "company", label: "ENB", attrs: {} });
  const g = graphFixture([company], []);
  const many = Array.from({ length: CONTRACTS_CAP + 3 }, (_, i) =>
    contract({ tkr: "ENB", rt: `2026-07-${String(i + 1).padStart(2, "0")}` }));
  const out = buildDossier(g, { entity: "ENB" }, { contracts: many, sites });
  assert.equal(out.contracts.length, CONTRACTS_CAP);
  assert.equal(out.contracts_capped, true);
  assert.equal(out.contracts[0].rt, `2026-07-${String(CONTRACTS_CAP + 3).padStart(2, "0")}`); // newest first
});

test("unresolvable entity degrades honestly: no identity/graph, but lat/lon still drives nearest_sites", () => {
  const g = graphFixture([], []);
  const out = buildDossier(g, { entity: "not-a-real-entity", lat: 35.9, lon: -96.7 }, { sites });
  assert.equal(out.identity, null);
  assert.equal(out.graph, null);
  assert.equal(out.nearest_sites[0].id, "cushing_hub");
});

test("lat/lon-only mode (no entity) works for point-based clicks — e.g. an aircraft, which is not a graph node", () => {
  const g = graphFixture([], []);
  const out = buildDossier(g, { lat: 33.7, lon: -118.2 }, { contracts: [contract({})], sites });
  assert.equal(out.identity, null);
  assert.equal(out.contracts.length, 0); // no resolved ticker -> no contract match, not "all contracts"
  assert.equal(out.nearest_sites[0].id, "la_port");
});

test("cold-start (graph still building, null) degrades to nearest_sites-only, never throws", () => {
  const out = buildDossier(null, { entity: "ENB", lat: 35.9, lon: -96.7 }, { sites });
  assert.equal(out.built_at, null);
  assert.equal(out.identity, null);
  assert.equal(out.graph, null);
  assert.equal(out.nearest_sites.length > 0, true);
});

test("nearest_sites is capped at NEAREST_SITES_CAP and sorted by distance", () => {
  const manySites: SiteRow[] = Array.from({ length: NEAREST_SITES_CAP + 5 }, (_, i) => ({
    id: `s${i}`, name: `Site ${i}`, category: "tank_farm", lat: 35.9 + i * 0.1, lon: -96.7,
  }));
  const out = buildDossier(null, { lat: 35.9, lon: -96.7 }, { sites: manySites });
  assert.equal(out.nearest_sites.length, NEAREST_SITES_CAP);
  assert.equal(out.nearest_sites[0].id, "s0"); // exact anchor match, distance 0
  for (let i = 1; i < out.nearest_sites.length; i++) {
    assert.ok(out.nearest_sites[i].km >= out.nearest_sites[i - 1].km);
  }
});

test("hops param is clamped to [0,3]", () => {
  const facility = node({ id: "facility:site:cushing_hub", type: "facility", label: "Cushing Oil Hub", attrs: { lat: 35.94, lon: -96.75 } });
  const g = graphFixture([facility], []);
  const out = buildDossier(g, { entity: "cushing_hub", hops: 99 }, { sites });
  assert.equal(out.query.hops, 3);
  const out2 = buildDossier(g, { entity: "cushing_hub", hops: -5 }, { sites });
  assert.equal(out2.query.hops, 0);
});

// LOCATION DOSSIER hazard cross-join (research/location_context_engine.md)
const ANCHOR = { lat: 35.94, lon: -96.75 }; // Cushing Oil Hub, same fixture anchor as nearest_sites tests

const superfundSite = (over: Partial<SuperfundSite>): SuperfundSite => ({
  name: "Test NPL Site", epa_id: "OK0001", hrs_score: 50, status: "NPL Site",
  state: "OK", city: "Cushing", county: "Payne", lat: ANCHOR.lat + 0.02, lon: ANCHOR.lon,
  listed: "1990-01-01", ...over,
});
const violator = (over: Partial<WaterViolator>): WaterViolator => ({
  name: "Test Facility", id: "OK0000001", city: "Cushing", state: "OK",
  lat: ANCHOR.lat + 0.02, lon: ANCHOR.lon, permit: "Major", snc: "SNC", qtrs: 9, actions: 1, ...over,
});
const pfasSystem = (over: Partial<PfasSystem>): PfasSystem => ({
  pwsid: "OK0000001", name: "Test Water System", state: "OK",
  lat: ANCHOR.lat + 0.02, lon: ANCHOR.lon, population_served: 5000,
  n_analytes_detected: 2,
  detections: [
    { contaminant: "PFOA", units: "µg/L", mrl: 0.004, max_value: 0.01, first_detected: "1/1/2024", last_detected: "6/1/2024", n_detections: 2 },
    { contaminant: "PFOS", units: "µg/L", mrl: 0.004, max_value: 0.02, first_detected: "1/1/2024", last_detected: "6/1/2024", n_detections: 2 },
  ],
  ...over,
});

test("no anchor coordinates (no entity resolved, no lat/lon passed) -> hazards is null, not empty sections", () => {
  const g = graphFixture([], []);
  const out = buildDossier(g, { entity: "not-a-real-entity" }, {});
  assert.equal(out.hazards, null);
});

test("hazard categories cross-join within radius_km of the anchor, sorted nearest-first, distances rounded", () => {
  const g = graphFixture([], []);
  const near = superfundSite({ name: "Near Site", lat: ANCHOR.lat + 0.01, lon: ANCHOR.lon }); // ~1.1km
  const mid = superfundSite({ name: "Mid Site", lat: ANCHOR.lat + 0.03, lon: ANCHOR.lon }); // ~3.3km
  const far = superfundSite({ name: "Far Site", epa_id: "FAR", lat: ANCHOR.lat + 5, lon: ANCHOR.lon }); // ~555km, outside default 50km
  const out = buildDossier(g, { entity: "no-match", lat: ANCHOR.lat, lon: ANCHOR.lon }, { superfund: [far, mid, near] });
  assert.ok(out.hazards);
  assert.equal(out.hazards!.radius_km, HAZARD_RADIUS_KM_DEFAULT);
  assert.equal(out.hazards!.superfund.ready, true);
  assert.equal(out.hazards!.superfund.total_within, 2); // near + mid only, far excluded
  assert.equal(out.hazards!.superfund.hits[0].label, "Near Site");
  assert.equal(out.hazards!.superfund.hits[1].label, "Mid Site");
  assert.ok(out.hazards!.superfund.hits[0].km < out.hazards!.superfund.hits[1].km);
});

test("hazard source not passed at all degrades to ready:false, not a false-clean 0", () => {
  const g = graphFixture([], []);
  const out = buildDossier(g, { lat: ANCHOR.lat, lon: ANCHOR.lon }, { superfund: [superfundSite({})] });
  assert.ok(out.hazards);
  assert.equal(out.hazards!.superfund.ready, true);
  assert.equal(out.hazards!.water_violators.ready, false); // waterViolators opt never passed
  assert.equal(out.hazards!.water_violators.total_within, 0);
  assert.equal(out.hazards!.water_violators.hits.length, 0);
});

test("hazard source passed as empty array (cache warm, genuinely nothing nearby) is ready:true with total_within 0", () => {
  const g = graphFixture([], []);
  const out = buildDossier(g, { lat: ANCHOR.lat, lon: ANCHOR.lon }, { waterViolators: [] });
  assert.equal(out.hazards!.water_violators.ready, true);
  assert.equal(out.hazards!.water_violators.total_within, 0);
});

test("hazard hits are capped at HAZARD_CAP, but total_within reports the true count and capped is honestly flagged", () => {
  const g = graphFixture([], []);
  const many = Array.from({ length: HAZARD_CAP + 4 }, (_, i) =>
    violator({ id: `v${i}`, name: `Violator ${i}`, lat: ANCHOR.lat + i * 0.001, lon: ANCHOR.lon }));
  const out = buildDossier(g, { lat: ANCHOR.lat, lon: ANCHOR.lon }, { waterViolators: many });
  assert.equal(out.hazards!.water_violators.total_within, HAZARD_CAP + 4);
  assert.equal(out.hazards!.water_violators.hits.length, HAZARD_CAP);
  assert.equal(out.hazards!.water_violators.capped, true);
});

test("pfas hazard category cross-joins by distance and carries analyte detail, same contract as the other hazard sources", () => {
  const g = graphFixture([], []);
  const near = pfasSystem({ pwsid: "near", name: "Near System", lat: ANCHOR.lat + 0.01, lon: ANCHOR.lon });
  const far = pfasSystem({ pwsid: "far", name: "Far System", lat: ANCHOR.lat + 5, lon: ANCHOR.lon });
  const out = buildDossier(g, { lat: ANCHOR.lat, lon: ANCHOR.lon }, { pfas: [far, near] });
  assert.ok(out.hazards);
  assert.equal(out.hazards!.pfas.ready, true);
  assert.equal(out.hazards!.pfas.total_within, 1);
  assert.equal(out.hazards!.pfas.hits[0].label, "Near System");
  assert.equal(out.hazards!.pfas.hits[0].detail.analytes_detected, 2);
  assert.deepEqual(out.hazards!.pfas.hits[0].detail.top, ["PFOA 0.01µg/L", "PFOS 0.02µg/L"]);
});

test("pfas hazard source not passed at all degrades to ready:false, matching every other hazard source's contract", () => {
  const g = graphFixture([], []);
  const out = buildDossier(g, { lat: ANCHOR.lat, lon: ANCHOR.lon }, { superfund: [superfundSite({})] });
  assert.equal(out.hazards!.pfas.ready, false);
  assert.equal(out.hazards!.pfas.total_within, 0);
});

// FEMA flood zone point lookup (research/location_context_engine.md hazard
// layer #3) — pre-fetched by routes.ts, injected as a plain value so
// buildDossier stays pure/sync like every other hazard source here.
const floodZone = (over: Partial<FloodZoneResult>): FloodZoneResult => ({
  zone: "AE", subtype: null, sfha: true, base_flood_elevation_ft: 12,
  meaning: "1% annual-chance flood zone...", source_citation: "22071C_STUDY1",
  source: "FEMA NFHL", ready: true, ...over,
});

test("flood_zone passes through opts.floodZone unchanged when an anchor exists", () => {
  const g = graphFixture([], []);
  const fz = floodZone({ zone: "VE", sfha: true });
  const out = buildDossier(g, { lat: ANCHOR.lat, lon: ANCHOR.lon }, { floodZone: fz });
  assert.deepEqual(out.hazards!.flood_zone, fz);
});

test("flood_zone defaults to ready:false/zone:null (not looked up) when opts.floodZone isn't passed, never crashes", () => {
  const g = graphFixture([], []);
  const out = buildDossier(g, { lat: ANCHOR.lat, lon: ANCHOR.lon }, {});
  assert.equal(out.hazards!.flood_zone.ready, false);
  assert.equal(out.hazards!.flood_zone.zone, null);
});

test("flood_zone with ready:true and zone:null (genuinely outside NFHL's mapped footprint) is preserved, not collapsed to 'not looked up'", () => {
  const g = graphFixture([], []);
  const fz = floodZone({ zone: null, subtype: null, sfha: null, base_flood_elevation_ft: null, meaning: null, source_citation: null, ready: true });
  const out = buildDossier(g, { lat: ANCHOR.lat, lon: ANCHOR.lon }, { floodZone: fz });
  assert.equal(out.hazards!.flood_zone.ready, true);
  assert.equal(out.hazards!.flood_zone.zone, null);
});

test("radius_km param is honored and clamped to [1, HAZARD_RADIUS_KM_MAX]", () => {
  const g = graphFixture([], []);
  const mid = superfundSite({ name: "Mid Site", lat: ANCHOR.lat + 0.03, lon: ANCHOR.lon }); // ~3.3km
  const tight = buildDossier(g, { lat: ANCHOR.lat, lon: ANCHOR.lon, radius_km: 1 }, { superfund: [mid] });
  assert.equal(tight.hazards!.radius_km, 1);
  assert.equal(tight.hazards!.superfund.total_within, 0); // 3.3km outside a 1km radius
  const huge = buildDossier(g, { lat: ANCHOR.lat, lon: ANCHOR.lon, radius_km: 99999 }, { superfund: [mid] });
  assert.equal(huge.hazards!.radius_km, HAZARD_RADIUS_KM_MAX);
});
