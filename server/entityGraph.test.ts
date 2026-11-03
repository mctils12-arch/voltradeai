// Everything Graph v1 build step 2 (datacore/EVERYTHING_GRAPH.md) —
// entityGraph.ts fixture tests. Every IO source is injected so these run
// against small deterministic fixtures, never the real repo registries or
// archive (mirrors the shadowFleet/portDwell baseDir-injectable pattern).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  buildGraph, neighborhood, resolveEntityId,
  buildGraphManaged, distillGemOwnership, loadGemOwnershipDistilledCached, graphPollMode, graphPollStats,
  setGraphPollAuditWriter, GRAPH_POLL_ENV, _resetGraphPollForTests, _graphPollGemCacheForTests,
  type EverythingGraph, type GemOwnershipFile,
} from "./entityGraph";
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
  const g = await buildGraph({ sites, plants, entities, form4Filings: [], portVisitsByPort: new Map(), gemOwnership: null });
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
  const g = await buildGraph({ sites, plants, entities, form4Filings: [], portVisitsByPort: new Map(), gemOwnership: null });
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
  const g = await buildGraph({ sites: [], plants: [], entities: [], form4Filings: filings, portVisitsByPort: new Map(), gemOwnership: null });
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
  const g = await buildGraph({ sites: [], plants: [], entities: [], form4Filings: filings, portVisitsByPort: new Map(), gemOwnership: null });
  const company = g.nodes.find((n) => n.type === "company" && n.attrs.cik === "0000999");
  assert.ok(company);
  assert.equal(company!.id, "company:cik:0000999");
  assert.equal(company!.attrs.ticker_known, false);
});

test("calls_at edges: only ports (site category 'port') get vessel edges; visit_count and median dwell aggregate correctly", async () => {
  const visitsByPort = new Map<string, PortVisit[]>([
    ["la_port", [visit({ dwellHours: 4, lastSeen: 5000 }), visit({ dwellHours: 12, lastSeen: 9000 }), visit({ dwellHours: 8, lastSeen: 7000 })]],
  ]);
  const g = await buildGraph({ sites, plants: [], entities: [], form4Filings: [], portVisitsByPort: visitsByPort, gemOwnership: null });
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
  const g = await buildGraph({ sites, plants: [], entities: [], form4Filings: [], portVisitsByPort: visitsByPort, gemOwnership: null });
  assert.equal(g.counts.calls_at, 0);
});

test("resolveEntityId + neighborhood: bare ticker resolves, hops=1 returns direct neighbors only", async () => {
  const g = await buildGraph({ sites, plants, entities, form4Filings: [], portVisitsByPort: new Map(), gemOwnership: null });
  const id = resolveEntityId(g, "ENB");
  assert.equal(id, "company:ENB");
  const nbhd = neighborhood(g, id!, 1);
  assert.ok(nbhd.nodes.some((n) => n.id === "facility:site:cushing_hub"));
  assert.equal(nbhd.nodes.some((n) => n.id === "company:SO"), false); // 2 hops away, not included at hops=1
  assert.equal(nbhd.edges.length, 1);
});

test("resolveEntityId returns null for an unknown entity", async () => {
  const g = await buildGraph({ sites, plants, entities, form4Filings: [], portVisitsByPort: new Map(), gemOwnership: null });
  assert.equal(resolveEntityId(g, "NOPE_NOT_A_REAL_ENTITY"), null);
});

test("counts block is internally consistent with the emitted node/edge arrays", async () => {
  const filings = [form4({})];
  const visitsByPort = new Map<string, PortVisit[]>([["la_port", [visit({})]]]);
  const g = await buildGraph({ sites, plants, entities, form4Filings: filings, portVisitsByPort: visitsByPort, gemOwnership: null });
  assert.equal(g.counts.nodes, g.nodes.length);
  assert.equal(g.counts.edges, g.edges.length);
  assert.equal(g.counts.company + g.counts.person + g.counts.facility + g.counts.vessel, g.nodes.length);
  assert.equal(g.counts.insider_of + g.counts.operates + g.counts.calls_at + g.counts.owns, g.edges.length);
});

test("graph is marked RAW with a caveat naming provenance/staleness caveats", () => {
  return buildGraph({ sites: [], plants: [], entities: [], form4Filings: [], portVisitsByPort: new Map(), gemOwnership: null }).then((g) => {
    assert.equal(g.kind, "raw");
    assert.ok(g.caveat.toLowerCase().includes("no predictive claim"));
  });
});

// ── owns edges (GEM Global Energy Ownership Tracker CIK join) ──────────────
const gemEntities = [
  { "Entity ID": "E1", "Full Name": "Big Utility Corp", "US SEC Central Index Key": "0000001305" },
  { "Entity ID": "E2", "Full Name": "Mega Holdings Inc", "US SEC Central Index Key": "0001432290" },
  { "Entity ID": "E3", "Full Name": "No CIK Government Fund", "Entity Type": "state", "Headquarters Country": "United States" }, // no CIK field at all
  { "Entity ID": "E4", "Full Name": "Not Found Co", "US SEC Central Index Key": "not found", "Entity Type": "legal entity", "Headquarters Country": "Canada" },
  { "Entity ID": "E5", "Full Name": "Untracked Foreign Parent" }, // neither end of its edge below resolves to a CIK
  { "Entity ID": "E6", "Full Name": "Untracked Private Holdco" },
];
const gemEdges = [
  { "Subject Entity ID": "E1", "Subject Entity Name": "Big Utility Corp", "Interested Party ID": "E2", "Interested Party Name": "Mega Holdings Inc", "% Share of Ownership": 12.6, "Share Imputed?": "known value, not imputed", "Data Source URL": "https://example.test/src" },
  { "Subject Entity ID": "E1", "Subject Entity Name": "Big Utility Corp", "Interested Party ID": "E3", "Interested Party Name": "No CIK Government Fund", "% Share of Ownership": 40 }, // owner has no CIK -> CIK-anchored on the subject side, owner becomes an institution node
  { "Subject Entity ID": "E4", "Subject Entity Name": "Not Found Co", "Interested Party ID": "E2", "Interested Party Name": "Mega Holdings Inc", "% Share of Ownership": 5 }, // subject CIK is the literal "not found" sentinel -> CIK-anchored on the owner side, subject becomes an institution node
  { "Subject Entity ID": "E6", "Subject Entity Name": "Untracked Private Holdco", "Interested Party ID": "E5", "Interested Party Name": "Untracked Foreign Parent", "% Share of Ownership": 100 }, // NEITHER end has a CIK -> no anchor, still an honest gap, skipped
];
const gemFixture = { provenance: { attribution: "Global Energy Monitor", license: "CC BY 4.0", release: "test", built_at: "2026-07-07T03:06:32Z" }, entities: gemEntities, entity_edges: gemEdges };

test("owns edges: an edge is kept whenever AT LEAST ONE end resolves to a real CIK; the neither-end-CIK case is still an honest gap", async () => {
  const g = await buildGraph({ sites: [], plants: [], entities: [], form4Filings: [], portVisitsByPort: new Map(), gemOwnership: gemFixture, cikTickerMap: new Map() });
  assert.equal(g.counts.owns, 3); // E2->E1 (both CIK), E3->E1 (owner uncertain), E2->E4 (subject uncertain); E5->E6 dropped
  const ownsEdges = g.edges.filter((e) => e.type === "owns");
  assert.equal(ownsEdges.some((e) => e.from === "institution:E5" || e.to === "institution:E6"), false);

  const bothCik = ownsEdges.find((e) => e.from === "company:cik:0001432290" && e.to === "company:cik:0000001305")!;
  assert.ok(bothCik);
  assert.equal(bothCik.attrs.share_pct, 12.6);
  assert.equal(bothCik.confidence, "high");
  assert.equal(bothCik.source.includes("Global Energy Monitor"), true);
  assert.equal(bothCik.first_seen, "2026-07-07T03:06:32Z"); // GEM's own release date, not entity_map.json's
  assert.equal(bothCik.last_seen, "2026-07-07T03:06:32Z");

  const ownerIsInstitution = ownsEdges.find((e) => e.to === "company:cik:0000001305" && e.from === "institution:E3")!;
  assert.ok(ownerIsInstitution);
  assert.equal(ownerIsInstitution.attrs.share_pct, 40);
  assert.equal(ownerIsInstitution.confidence, "medium"); // no "Share Imputed?" field -> not the known-value string

  const subjectIsInstitution = ownsEdges.find((e) => e.from === "company:cik:0001432290" && e.to === "institution:E4")!;
  assert.ok(subjectIsInstitution);
  assert.equal(subjectIsInstitution.attrs.share_pct, 5);
});

test("owns edges: the non-CIK side of a CIK-anchored edge becomes an honestly-typed institution node, never a company node", async () => {
  const g = await buildGraph({ sites: [], plants: [], entities: [], form4Filings: [], portVisitsByPort: new Map(), gemOwnership: gemFixture, cikTickerMap: new Map() });
  assert.equal(g.counts.institution, 2); // E3 (state) + E4 (legal entity, "not found" CIK) — E5/E6 never enter the graph
  assert.equal(g.nodes.some((n) => n.id === "institution:E5" || n.id === "institution:E6"), false);

  const state = g.nodes.find((n) => n.id === "institution:E3")!;
  assert.equal(state.type, "institution");
  assert.equal(state.label, "No CIK Government Fund");
  assert.equal(state.attrs.gem_entity_type, "state");
  assert.equal(state.attrs.headquarters_country, "United States");
  assert.equal(state.attrs.cik_verified, false);
  assert.equal(g.nodes.some((n) => n.type === "company" && n.id === "company:E3"), false); // never mistyped as company

  const legalEntity = g.nodes.find((n) => n.id === "institution:E4")!;
  assert.equal(legalEntity.attrs.gem_entity_type, "legal entity");
  assert.equal(legalEntity.attrs.headquarters_country, "Canada");
});

test("owns edges: a resolvable CIK->ticker mapping lands the CIK-verified node on the SAME id the EDGAR/insider join would use", async () => {
  const cikTickerMap = new Map([["1432290", "MEGA"]]); // getCikTickerMap keys are un-padded, per sec8kEarnings.ts
  const g = await buildGraph({ sites: [], plants: [], entities: [], form4Filings: [], portVisitsByPort: new Map(), gemOwnership: gemFixture, cikTickerMap });
  const edge = g.edges.find((e) => e.type === "owns" && e.attrs.share_pct === 12.6)!;
  assert.equal(edge.from, "company:MEGA");
  const node = g.nodes.find((n) => n.id === "company:MEGA");
  assert.ok(node);
  assert.equal(node!.attrs.ticker_known, true);
});

test("owns edges: null gemOwnership (disk read unavailable/skipped) yields zero owns edges, never throws", async () => {
  const g = await buildGraph({ sites: [], plants: [], entities: [], form4Filings: [], portVisitsByPort: new Map(), gemOwnership: null });
  assert.equal(g.counts.owns, 0);
  assert.equal(g.counts.institution, 0);
  assert.equal(g.edges.some((e) => e.type === "owns"), false);
});

// ── memory fix A2b (2026-11-03): the distilled GEM slice, its file-identity cache, and the three-mode managed poll ──
// Property under test: the graph built from the cached distilled slice is
// byte-identical to the graph built from a raw parse of the same release —
// node order, edge order, dedupe, institution typing, share/imputed/url
// passthrough — and shadow mode proves it on every poll.

const NOW_MS = Date.parse("2026-11-03T12:00:00Z");
const fixtureInputs = {
  sites, plants, entities, form4Filings: [form4({})], portVisitsByPort: new Map<string, PortVisit[]>([["la_port", [visit({})]]]), nowMs: NOW_MS,
};

async function captureConsoleError(fn: () => Promise<void>): Promise<string[]> {
  const orig = console.error;
  const lines: string[] = [];
  console.error = (...args: unknown[]) => { lines.push(args.map(String).join(" ")); };
  try { await fn(); } finally { console.error = orig; }
  return lines;
}

function gemTempFile(gem: unknown): { dir: string; fp: string; write: (g: unknown) => void } {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vt-gem-cache-"));
  const fp = path.join(dir, "ownership.json.gz");
  const write = (g: unknown) => fs.writeFileSync(fp, zlib.gzipSync(JSON.stringify(g)));
  write(gem);
  return { dir, fp, write };
}

test("distilled GEM build == raw GEM build, byte for byte, with and without a CIK->ticker map", async () => {
  for (const cikTickerMap of [new Map<string, string>(), new Map([["1432290", "MEGA"], ["1305", "BIGU"]])]) {
    const raw = await buildGraph({ ...fixtureInputs, gemOwnership: gemFixture, cikTickerMap });
    const viaDistilled = await buildGraph({ ...fixtureInputs, gemDistilled: distillGemOwnership(gemFixture), cikTickerMap });
    assert.deepEqual(viaDistilled, raw);
    assert.equal(JSON.stringify(viaDistilled), JSON.stringify(raw), "identical /api/data/graph payload");
    assert.equal(raw.counts.owns, 3);
  }
});

test("distill keeps the anchoring/dedupe semantics exactly: duplicate entity rows, the 'not found' sentinel, unlisted ids, duplicate and self edges", async () => {
  const dupEntities = [
    { "Entity ID": "D1", "Full Name": "Dup Co", "US SEC Central Index Key": "0000000777", "Entity Type": "legal entity", "Headquarters Country": "Norway" },
    { "Entity ID": "D1", "Full Name": "Dup Co (amended)", "Entity Type": "state body" }, // later duplicate WITHOUT a CIK: type re-set, cik must survive
    { "Entity ID": "D2", "Full Name": "Sentinel Co", "US SEC Central Index Key": "not found", "Entity Type": "arrangement" },
    { "Entity ID": "D3", "Full Name": "Owner Co", "US SEC Central Index Key": "0000000888" },
    { "Entity ID": "D4", "Full Name": "Never Referenced", "Entity Type": "person", "Headquarters Country": "France" },
  ];
  const dupEdges = [
    { "Subject Entity ID": "D1", "Subject Entity Name": "Dup Co", "Interested Party ID": "D3", "Interested Party Name": "Owner Co", "% Share of Ownership": 10 },
    { "Subject Entity ID": "D1", "Subject Entity Name": "Dup Co", "Interested Party ID": "D3", "Interested Party Name": "Owner Co", "% Share of Ownership": 99 }, // duplicate pair: first wins at build time
    { "Subject Entity ID": "D2", "Subject Entity Name": "Sentinel Co", "Interested Party ID": "D3", "Interested Party Name": "Owner Co" },
    { "Subject Entity ID": "D9", "Subject Entity Name": "Unlisted Target", "Interested Party ID": "D3", "Interested Party Name": "Owner Co", "% Share of Ownership": "n/a" as never }, // D9 appears in no entity row
    { "Subject Entity ID": "D3", "Subject Entity Name": "Owner Co", "Interested Party ID": "D3", "Interested Party Name": "Owner Co" }, // self edge: dropped at build time
    { "Subject Entity ID": "D4", "Subject Entity Name": "Never Referenced", "Interested Party ID": "D2", "Interested Party Name": "Sentinel Co" }, // neither end has a CIK: not anchored
  ];
  const gem: GemOwnershipFile = { provenance: { attribution: "GEM", license: "CC BY 4.0", release: "t", built_at: "2026-01-01T00:00:00Z" }, entities: dupEntities, entity_edges: dupEdges };
  const d = distillGemOwnership(gem);
  assert.equal(d.builtDate, "2026-01-01T00:00:00Z");
  assert.equal(d.cikByEntity.get("D1"), "0000000777", "a later CIK-less duplicate never unsets the CIK");
  assert.equal(d.typeByEntity.get("D1"), "state body", "last duplicate wins for type (Map.set semantics)");
  assert.equal(d.hqByEntity.get("D1"), "Norway", "hq set by the first row stays when the later row has none");
  assert.equal(d.cikByEntity.has("D2"), false, "'not found' is not a CIK");
  assert.equal(d.rows.length, 5, "every anchored row is kept in order; dedupe and the self-edge drop happen at build time");
  assert.equal(d.typeByEntity.has("D4"), false, "an entity no anchored row references is pruned");
  assert.equal(d.hqByEntity.has("D4"), false);
  assert.equal(d.typeByEntity.has("D9"), false);
  for (const cikTickerMap of [new Map<string, string>(), new Map([["777", "DUPC"]])]) {
    const raw = await buildGraph({ ...fixtureInputs, gemOwnership: gem, cikTickerMap });
    const viaDistilled = await buildGraph({ ...fixtureInputs, gemDistilled: d, cikTickerMap });
    assert.deepEqual(viaDistilled, raw);
    const owns = raw.edges.filter((e) => e.type === "owns");
    assert.equal(owns.length, 3, "D3->D1 (first duplicate), D3->D2, D3->D9; self edge and the unanchored edge dropped");
    assert.equal(owns[0].attrs.share_pct, 10, "first duplicate pair wins");
    assert.equal(owns[2].attrs.share_pct, null, "a non-numeric share is null in both builds");
    const d9 = raw.nodes.find((n) => n.id === "institution:D9")!;
    assert.equal(d9.attrs.gem_entity_type, "unknown entity");
    assert.equal(d9.attrs.headquarters_country, null);
    assert.equal(raw.nodes.some((n) => n.id === "institution:D4"), false);
  }
});

test("loadGemOwnershipDistilledCached: one parse per (mtime, size); a touched or rewritten file reloads; corrupt caches null; missing drops the cache", () => {
  _resetGraphPollForTests();
  const { dir, fp, write } = gemTempFile(gemFixture);
  const a = loadGemOwnershipDistilledCached(fp)!;
  const b = loadGemOwnershipDistilledCached(fp)!;
  assert.equal(a, b, "same object: no re-parse while mtime+size are unchanged");
  let s = graphPollStats();
  assert.equal(s.gem.reloads, 1);
  assert.equal(s.gem.cached, true);
  assert.equal(s.gem.anchoredRows, 3);
  assert.equal(s.gem.cikEntities, 2);
  assert.equal(s.gem.sizeBytes, fs.statSync(fp).size);
  assert.equal(s.partialsCached, 1);
  // same bytes, newer mtime -> reload (a deploy rewriting the file in place is never served stale)
  const later = new Date(Date.now() + 5_000);
  fs.utimesSync(fp, later, later);
  const c = loadGemOwnershipDistilledCached(fp)!;
  assert.notEqual(c, a);
  assert.equal(graphPollStats().gem.reloads, 2);
  // different content -> reload, and the new content is what is served
  write({ ...gemFixture, entity_edges: gemFixture.entity_edges.slice(0, 1) });
  assert.equal(loadGemOwnershipDistilledCached(fp)!.rows.length, 1);
  assert.equal(graphPollStats().gem.reloads, 3);
  // unreadable file -> null, cached under its stat identity (no re-parse storm), nothing held
  fs.writeFileSync(fp, "not gzip at all");
  assert.equal(loadGemOwnershipDistilledCached(fp), null);
  assert.equal(loadGemOwnershipDistilledCached(fp), null);
  s = graphPollStats();
  assert.equal(s.gem.reloads, 4);
  assert.equal(s.gem.cached, true);
  assert.equal(s.partialsCached, 0);
  // missing -> null and the cache is dropped
  fs.unlinkSync(fp);
  assert.equal(loadGemOwnershipDistilledCached(fp), null);
  assert.equal(graphPollStats().gem.cached, false);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("buildGraphManaged: full, incremental and shadow all equal buildGraph on a file-backed GEM; shadow records zero mismatches and parses the slice once", async () => {
  _resetGraphPollForTests();
  const { dir, fp } = gemTempFile(gemFixture);
  const inputs = { ...fixtureInputs, gemPath: fp, cikTickerMap: new Map([["1432290", "MEGA"]]) };
  const ref = await buildGraph({ ...inputs, gemOwnership: gemFixture });
  assert.equal(ref.counts.owns, 3);
  assert.deepEqual(await buildGraphManaged(inputs, "full"), ref);
  assert.equal(graphPollStats().gem.reloads, 0, "full mode never touches the cache");
  assert.deepEqual(await buildGraphManaged(inputs, "incremental"), ref);
  let shadowOut: EverythingGraph | null = null;
  const errors = await captureConsoleError(async () => { shadowOut = await buildGraphManaged(inputs, "shadow"); });
  assert.deepEqual(shadowOut, ref);
  assert.equal(errors.filter((l) => l.includes("FOLD-MISMATCH")).length, 0);
  const s = graphPollStats();
  assert.equal(s.mode, "shadow");
  assert.equal(s.polls, 3);
  assert.equal(s.mismatches, 0);
  assert.equal(s.lastMismatchAt, null);
  assert.equal(typeof s.lastFullMs, "number");
  assert.equal(typeof s.lastIncrementalMs, "number");
  assert.equal(typeof s.lastSharedInputsMs, "number");
  assert.equal(s.gem.reloads, 1, "the slice was parsed once across the incremental and shadow polls");
  assert.equal(s.partialsCached, 1);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("shadow catches a corrupted distilled cache: FOLD-MISMATCH graph logged + audit row, the full build is served, the slice is dropped and reloaded next poll", async () => {
  _resetGraphPollForTests();
  const { dir, fp } = gemTempFile(gemFixture);
  const inputs = { ...fixtureInputs, gemPath: fp, cikTickerMap: new Map<string, string>() };
  const ref = await buildGraph({ ...inputs, gemOwnership: gemFixture });
  await buildGraphManaged(inputs, "incremental"); // warm the slice
  _graphPollGemCacheForTests()!.distilled!.rows[0].share = 99; // corrupt the 12.6% E2->E1 row
  const auditRows: Array<[string, string]> = [];
  setGraphPollAuditWriter((type, message) => { auditRows.push([type, message]); });
  let served: EverythingGraph | null = null;
  const errors = await captureConsoleError(async () => { served = await buildGraphManaged(inputs, "shadow"); });
  assert.deepEqual(served, ref, "the FULL build is what gets served in shadow mode");
  const s = graphPollStats();
  assert.equal(s.mismatches, 1);
  assert.equal(typeof s.lastMismatchAt, "string");
  assert.match(s.lastMismatchDetail!, /^\$\.edges\[\d+\]\.attrs\.share_pct: 12\.6 vs 99$/);
  assert.equal(errors.filter((l) => l.startsWith("FOLD-MISMATCH graph: $.edges[")).length, 1);
  assert.deepEqual(auditRows, [["FOLD-MISMATCH", `graph: ${s.lastMismatchDetail}`]]);
  assert.equal(s.gem.cached, false, "a mismatch drops the cached slice so the next poll re-derives it");
  assert.deepEqual(await buildGraphManaged(inputs, "shadow"), ref);
  assert.equal(graphPollStats().mismatches, 1, "a reloaded slice compares clean again");
  assert.equal(graphPollStats().gem.reloads, 2);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("an audit writer that throws is contained and the mismatch is still counted", async () => {
  _resetGraphPollForTests();
  const { dir, fp } = gemTempFile(gemFixture);
  const inputs = { ...fixtureInputs, gemPath: fp, cikTickerMap: new Map<string, string>() };
  await buildGraphManaged(inputs, "incremental");
  _graphPollGemCacheForTests()!.distilled!.rows.pop();
  setGraphPollAuditWriter(() => { throw new Error("db closed"); });
  const errors = await captureConsoleError(async () => { await buildGraphManaged(inputs, "shadow"); });
  assert.equal(graphPollStats().mismatches, 1);
  assert.ok(errors.some((l) => l.includes("audit writer threw")));
  fs.rmSync(dir, { recursive: true, force: true });
});

test("mode flag: unset or unknown VOLTRADE_GRAPH_POLL is shadow; full/incremental are honoured", () => {
  const prior = process.env[GRAPH_POLL_ENV];
  try {
    delete process.env[GRAPH_POLL_ENV];
    assert.equal(graphPollMode(), "shadow");
    process.env[GRAPH_POLL_ENV] = "nonsense";
    assert.equal(graphPollMode(), "shadow");
    process.env[GRAPH_POLL_ENV] = "incremental";
    assert.equal(graphPollMode(), "incremental");
    process.env[GRAPH_POLL_ENV] = " Full ";
    assert.equal(graphPollMode(), "full");
  } finally {
    if (prior === undefined) delete process.env[GRAPH_POLL_ENV]; else process.env[GRAPH_POLL_ENV] = prior;
  }
});

test("missing GEM file: every mode yields the same zero-owns graph and nothing is cached", async () => {
  _resetGraphPollForTests();
  const inputs = { ...fixtureInputs, gemPath: path.join(os.tmpdir(), "vt-gem-does-not-exist", "ownership.json.gz"), cikTickerMap: new Map<string, string>() };
  const ref = await buildGraph({ ...inputs, gemOwnership: null });
  assert.equal(ref.counts.owns, 0);
  assert.deepEqual(await buildGraphManaged(inputs, "full"), ref);
  assert.deepEqual(await buildGraphManaged(inputs, "incremental"), ref);
  assert.deepEqual(await buildGraphManaged(inputs, "shadow"), ref);
  const s = graphPollStats();
  assert.equal(s.mismatches, 0);
  assert.equal(s.gem.cached, false);
});

test("heap report: the distilled slice of a synthetic 40k-entity / 40k-edge release is a small fraction of the raw parse (and of the live file when present)", () => {
  const entitiesBig: GemOwnershipFile["entities"] = [];
  for (let i = 0; i < 40_000; i++) {
    entitiesBig.push({
      "Entity ID": `E${i}`, "Full Name": `Entity number ${i} with a reasonably long legal name LLC`,
      "Entity Type": i % 7 ? "legal entity" : "state", "Headquarters Country": "United States",
      ...(i % 100 === 0 ? { "US SEC Central Index Key": String(1_000_000 + i).padStart(10, "0") } : {}),
      // the live release carries ~11 columns per entity; pad to a comparable row weight
      ...({ "Name": `Entity ${i}`, "Legal Entity Type": "corporation", "Registration Country": "United States",
        "Registration Subdivision": "Delaware", "Headquarters Subdivision": "Texas",
        "Global Legal Entity Identifier Index": `LEI${String(i).padStart(17, "0")}`, "PermID: Refinitiv Permanent Identifier": `${4_000_000_000 + i}` } as object),
    });
  }
  const edgesBig: GemOwnershipFile["entity_edges"] = [];
  for (let j = 0; j < 40_000; j++) {
    const ownerIdx = j % 20 === 0 ? (j % 400) * 100 : (j * 7 + 1) % 40_000; // every 20th edge is anchored on a CIK owner (~5%)
    edgesBig.push({
      "Subject Entity ID": `E${(j * 3 + 1) % 40_000}`, "Subject Entity Name": `Entity number ${(j * 3 + 1) % 40_000} with a reasonably long legal name LLC`,
      "Interested Party ID": `E${ownerIdx}`, "Interested Party Name": `Entity number ${ownerIdx} with a reasonably long legal name LLC`,
      "% Share of Ownership": (j % 97) + 0.5, "Data Source URL": `https://example.test/source/${j}`, "Share Imputed?": j % 2 ? "known value, not imputed" : "imputed",
    });
  }
  const big: GemOwnershipFile = { provenance: { attribution: "synthetic", license: "n/a", release: "synthetic", built_at: "2026-11-03T00:00:00Z" }, entities: entitiesBig, entity_edges: edgesBig };
  const rawBytes = JSON.stringify(big).length;
  const h0 = process.memoryUsage().heapUsed;
  const d = distillGemOwnership(big);
  const h1 = process.memoryUsage().heapUsed;
  const serialized = JSON.stringify({ ...d, cikByEntity: Array.from(d.cikByEntity), typeByEntity: Array.from(d.typeByEntity), hqByEntity: Array.from(d.hqByEntity) });
  assert.ok(d.rows.length >= 2_000 && d.rows.length < 3_000, `anchored rows ${d.rows.length} ~ 5% of the edges`);
  assert.equal(d.cikByEntity.size, 400);
  assert.ok(serialized.length < rawBytes * 0.05, `distilled ${serialized.length}B must be under 5% of the raw ${rawBytes}B`);
  let live = "live file not present";
  const livePath = path.resolve(process.cwd(), "datacore/gem/ownership.json.gz");
  if (fs.existsSync(livePath)) {
    const g0 = process.memoryUsage().heapUsed;
    const text = zlib.gunzipSync(fs.readFileSync(livePath)).toString("utf8");
    const raw = JSON.parse(text) as GemOwnershipFile;
    const g1 = process.memoryUsage().heapUsed;
    const ld = distillGemOwnership(raw);
    const liveSerialized = JSON.stringify({ ...ld, cikByEntity: Array.from(ld.cikByEntity), typeByEntity: Array.from(ld.typeByEntity), hqByEntity: Array.from(ld.hqByEntity) });
    live = `live: inflated=${(text.length / 1e6).toFixed(1)}MB rawParseHeapDelta=${((g1 - g0) / 1e6).toFixed(1)}MB entities=${raw.entities.length} edges=${raw.entity_edges.length} ` +
      `-> distilled rows=${ld.rows.length} cik=${ld.cikByEntity.size} type=${ld.typeByEntity.size} hq=${ld.hqByEntity.size} serialized=${(liveSerialized.length / 1e3).toFixed(0)}KB`;
    assert.ok(liveSerialized.length < text.length * 0.02, "the live release distils to under 2% of its inflated size");
  }
  console.log(`[graph heap] synthetic raw=${(rawBytes / 1e6).toFixed(1)}MB distilled=${(serialized.length / 1e3).toFixed(0)}KB rows=${d.rows.length} distillHeapDelta=${((h1 - h0) / 1e6).toFixed(1)}MB | ${live}`);
});
