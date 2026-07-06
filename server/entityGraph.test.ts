// Everything Graph v1 build step 2 (datacore/EVERYTHING_GRAPH.md) —
// entityGraph.ts fixture tests. Every IO source is injected so these run
// against small deterministic fixtures, never the real repo registries or
// archive (mirrors the shadowFleet/portDwell baseDir-injectable pattern).
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildGraph, neighborhood, resolveEntityId } from "./entityGraph";
import type { Form4Filing } from "./edgarForm4";
import type { PortVisit } from "./portDwell";

const sites = [
  { id: "cushing_hub", name: "Cushing Oil Hub", category: "tank_farm", lat: 35.94, lon: -96.75, operator: "Enbridge" },
  { id: "la_port", name: "Port of LA", category: "port", lat: 33.73, lon: -118.26, operator: "City of LA (POLA)" },
];

const plants: Array<[string, number, string, string, number, number, 0 | 1]> = [
  ["Big Coal Plant", 3000, "coal", "Georgia Power Co", 33.06, -83.8, 1],
  ["Small Solar Farm", 50, "solar", "Some Rural Co-op", 40.0, -90.0, 0],
];

const entities = [
  { operator: "Enbridge", ticker: "ENB", confidence: "high" as const, note: "pipeline operator", parent: "Enbridge Inc." },
  { operator: "Georgia Power Co", ticker: "SO", confidence: "high" as const, note: "regulated utility subsidiary", parent: "Southern Company" },
  { operator: "City of LA (POLA)", ticker: null, confidence: "unmapped" as const, note: "municipal port authority, no public equity" },
  // deliberately no entry for "Some Rural Co-op" — an unmapped operator not
  // even present in the map (distinct from an explicit "unmapped" row).
];

function form4(overrides: Partial<Form4Filing>): Form4Filing {
  return {
    issuerCik: "0000001", issuerName: "Steel Co", issuerTradingSymbol: "STLD",
    periodOfReport: "2026-07-01",
    owners: [{ cik: "0000123", name: "Jane Insider", isDirector: true, isOfficer: false, isTenPercentOwner: false, officerTitle: null }],
    transactions: [{ table: "nonDerivative", securityTitle: "Common", transactionDate: "2026-07-01", transactionCode: "P", kind: "open_market_buy", shares: 100, pricePerShare: 50, acquiredDisposedCode: "A", sharesOwnedAfter: 500 }],
    accession: "0000001-26-000001", filedAt: "2026-07-01T12:00:00Z", cik: "0000123", indexUrl: "https://example.test",
    ...overrides,
  };
}

const visit = (over: Partial<PortVisit>): PortVisit => ({
  mmsi: "111111111", portId: "la_port", firstSeen: 1000, lastSeen: 1000 + 3600 * 10,
  dwellHours: 10, points: 5, ongoing: false, ...over,
});

test("facility nodes: every site + plant becomes a node, lat/lon intrinsic (no separate located_at edge)", async () => {
  const g = await buildGraph({ sites, plants, entities, form4Filings: [], portVisitsByPort: new Map() });
  assert.equal(g.counts.facility, sites.length + plants.length);
  const cushing = g.nodes.find((n) => n.id === "facility:site:cushing_hub");
  assert.ok(cushing);
  assert.equal(cushing!.attrs.lat, 35.94);
  assert.equal(cushing!.attrs.imagery_verified, true);
  const plant0 = g.nodes.find((n) => n.id === "facility:plant:0");
  assert.ok(plant0);
  assert.equal(plant0!.attrs.owner, "Georgia Power Co");
  assert.equal((g.edges as Array<{ type: string }>).some((e) => e.type === "located_at"), false);
});

test("operates edges: only entity_map entries with a ticker produce an edge; unmapped/absent operators are honest gaps", async () => {
  const g = await buildGraph({ sites, plants, entities, form4Filings: [], portVisitsByPort: new Map() });
  assert.equal(g.counts.operates, 2); // Enbridge->cushing_hub, Georgia Power->plant:0
  const enbridgeEdge = g.edges.find((e) => e.type === "operates" && e.to === "facility:site:cushing_hub");
  assert.ok(enbridgeEdge);
  assert.equal(enbridgeEdge!.from, "company:ENB");
  assert.equal(enbridgeEdge!.confidence, "high");
  assert.equal(enbridgeEdge!.attrs.parent, "Enbridge Inc.");
  // City of LA (POLA) is explicitly unmapped, and "Some Rural Co-op" isn't
  // in entity_map at all — neither produces an edge.
  assert.equal(g.edges.some((e) => e.to === "facility:site:la_port" && e.type === "operates"), false);
  assert.equal(g.edges.some((e) => e.to === "facility:plant:1" && e.type === "operates"), false);
  const company = g.nodes.find((n) => n.id === "company:ENB");
  assert.ok(company);
  assert.equal(company!.type, "company");
});

test("insider_of edges aggregate across multiple filings for the same person/company pair", async () => {
  const filings = [
    form4({ filedAt: "2026-06-01T00:00:00Z", transactions: [{ table: "nonDerivative", securityTitle: "Common", transactionDate: "2026-06-01", transactionCode: "P", kind: "open_market_buy", shares: 10, pricePerShare: 40, acquiredDisposedCode: "A", sharesOwnedAfter: 100 }] }),
    form4({ filedAt: "2026-07-01T00:00:00Z", transactions: [{ table: "nonDerivative", securityTitle: "Common", transactionDate: "2026-07-01", transactionCode: "S", kind: "open_market_sale", shares: 5, pricePerShare: 55, acquiredDisposedCode: "D", sharesOwnedAfter: 95 }] }),
  ];
  const g = await buildGraph({ sites: [], plants: [], entities: [], form4Filings: filings, portVisitsByPort: new Map() });
  assert.equal(g.counts.insider_of, 1); // one person, one company, two filings -> one aggregated edge
  const edge = g.edges.find((e) => e.type === "insider_of")!;
  assert.equal(edge.from, "person:0000123");
  assert.equal(edge.to, "company:STLD");
  assert.equal(edge.attrs.filing_count, 2);
  assert.equal(edge.first_seen, "2026-06-01T00:00:00Z");
  assert.equal(edge.last_seen, "2026-07-01T00:00:00Z");
  assert.equal(edge.attrs.last_transaction_kind, "open_market_sale"); // most recent filing wins
  assert.deepEqual(edge.attrs.roles, ["director"]);
  assert.equal(edge.confidence, "high");
});

test("insider_of falls back to a CIK-keyed company id when no ticker is filed, and never collides with a real ticker id", async () => {
  const filings = [form4({ issuerTradingSymbol: null, issuerCik: "0000999", issuerName: "No Ticker Co" })];
  const g = await buildGraph({ sites: [], plants: [], entities: [], form4Filings: filings, portVisitsByPort: new Map() });
  const company = g.nodes.find((n) => n.type === "company" && n.attrs.cik === "0000999");
  assert.ok(company);
  assert.equal(company!.id, "company:cik:0000999");
  assert.equal(company!.attrs.ticker_known, false);
});

test("calls_at edges: only ports (site category 'port') get vessel edges; visit_count and median dwell aggregate correctly", async () => {
  const visitsByPort = new Map<string, PortVisit[]>([
    ["la_port", [visit({ dwellHours: 4, lastSeen: 5000 }), visit({ dwellHours: 12, lastSeen: 9000 }), visit({ dwellHours: 8, lastSeen: 7000 })]],
  ]);
  const g = await buildGraph({ sites, plants: [], entities: [], form4Filings: [], portVisitsByPort: visitsByPort });
  assert.equal(g.counts.calls_at, 1); // one vessel, one port -> one aggregated edge
  const edge = g.edges.find((e) => e.type === "calls_at")!;
  assert.equal(edge.to, "facility:site:la_port");
  assert.equal(edge.attrs.visit_count, 3);
  assert.equal(edge.attrs.median_dwell_h, 8); // sorted [4,8,12] -> median 8
  assert.equal(edge.confidence, "high");
  const vessel = g.nodes.find((n) => n.type === "vessel");
  assert.ok(vessel);
  assert.equal(vessel!.id, "vessel:111111111");
});

test("calls_at ignores visits assigned to a portId that isn't a real port site (defensive — should never happen upstream)", async () => {
  const visitsByPort = new Map<string, PortVisit[]>([["not_a_real_port", [visit({ portId: "not_a_real_port" })]]]);
  const g = await buildGraph({ sites, plants: [], entities: [], form4Filings: [], portVisitsByPort: visitsByPort });
  assert.equal(g.counts.calls_at, 0);
});

test("resolveEntityId + neighborhood: bare ticker resolves, hops=1 returns direct neighbors only", async () => {
  const g = await buildGraph({ sites, plants, entities, form4Filings: [], portVisitsByPort: new Map() });
  const id = resolveEntityId(g, "ENB");
  assert.equal(id, "company:ENB");
  const nbhd = neighborhood(g, id!, 1);
  assert.ok(nbhd.nodes.some((n) => n.id === "facility:site:cushing_hub"));
  assert.equal(nbhd.nodes.some((n) => n.id === "company:SO"), false); // 2 hops away, not included at hops=1
  assert.equal(nbhd.edges.length, 1);
});

test("resolveEntityId returns null for an unknown entity", async () => {
  const g = await buildGraph({ sites, plants, entities, form4Filings: [], portVisitsByPort: new Map() });
  assert.equal(resolveEntityId(g, "NOPE_NOT_A_REAL_ENTITY"), null);
});

test("counts block is internally consistent with the emitted node/edge arrays", async () => {
  const filings = [form4({})];
  const visitsByPort = new Map<string, PortVisit[]>([["la_port", [visit({})]]]);
  const g = await buildGraph({ sites, plants, entities, form4Filings: filings, portVisitsByPort: visitsByPort });
  assert.equal(g.counts.nodes, g.nodes.length);
  assert.equal(g.counts.edges, g.edges.length);
  assert.equal(g.counts.company + g.counts.person + g.counts.facility + g.counts.vessel, g.nodes.length);
  assert.equal(g.counts.insider_of + g.counts.operates + g.counts.calls_at, g.edges.length);
});

test("graph is marked RAW with a caveat naming provenance/staleness caveats", () => {
  return buildGraph({ sites: [], plants: [], entities: [], form4Filings: [], portVisitsByPort: new Map() }).then((g) => {
    assert.equal(g.kind, "raw");
    assert.ok(g.caveat.toLowerCase().includes("no predictive claim"));
  });
});
