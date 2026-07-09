// ENTITY DOSSIER v2 (W5) — dossier.ts fixture tests. buildDossier is a pure
// function over an injected graph/contracts/sites (mirrors entityGraph.ts's
// own every-IO-source-overridable convention) so these never touch the real
// archive or datacore registries.
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildDossier, NEAREST_SITES_CAP, CONTRACTS_CAP } from "./dossier";
import type { EverythingGraph, GraphNode, GraphEdge, SiteRow } from "./entityGraph";
import type { ContractTxn } from "./usaSpending";

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
    counts: { nodes: nodes.length, edges: edges.length, company: 0, person: 0, facility: 0, vessel: 0, insider_of: 0, operates: 0, calls_at: 0 },
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
