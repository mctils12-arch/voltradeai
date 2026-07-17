/**
 * entityGraph.ts — the Everything Graph, build step 2
 * (datacore/EVERYTHING_GRAPH.md; open_questions.md MAP V2 ROADMAP R5).
 *
 * Joins entities we already collect, keyed by whatever identifier each
 * source speaks natively (Form 4 -> CIK, market -> ticker, sites/plants ->
 * facility id, AIS -> MMSI, GEM ownership -> CIK) into one node/edge graph.
 * This module performs NO new collection and NO trading-logic import
 * (datacore boundary, per CLAUDE.md SPINOUT-READY DATA LAYER) — it is a pure
 * join over archives and git-versioned registries the platform already has.
 *
 * RAW vs SIGNAL: the graph asserts relationships with provenance
 * (RAW, per the design doc's ground rule 2). No edge here is a predictive
 * claim; any future anomaly/interpretation layer on top is SIGNAL-class and
 * ladder-gated separately.
 *
 * v1 storage decision (design doc): pure builder + in-memory cache, no
 * database — recomputed from archives/registries on every refresh, so
 * losing the cache loses nothing. Evolution trigger (sqlite) is stated in
 * the design doc, not needed at today's scale (~10k nodes estimated).
 *
 * Entity types: company (ticker), person (CIK), facility (site/plant id),
 * vessel (MMSI). `located_at` is NOT a separate edge in this build: the
 * design doc itself notes facility lat/lon is "intrinsic" to the facility
 * node, so it ships as a facility node attribute rather than inventing a
 * "geo" node type the v1 entity-type table never defines — a deliberate,
 * documented deviation from the relationship table's literal edge list,
 * not an omission.
 *
 * Every edge carries {source, confidence, first_seen, last_seen} per the
 * design doc's envelope-alignment note.
 */
import fs from "fs";
import zlib from "zlib";
import sitesJson from "../datacore/sites/strategic_sites.json";
import plantsJson from "../datacore/powerplants/us_power_plants.json";
import entityMapJson from "../datacore/entity_map.json";
import { readFilingHistory, type Form4Filing } from "./edgarForm4";
import { foldPortVisitsAsync, portsFromSites, type PortDef, type PortVisit } from "./portDwell";
import { repoDataPath } from "./repoFiles";
import { getCikTickerMap } from "./sec8kEarnings";

export type GraphNodeType = "company" | "person" | "facility" | "vessel";
export type GraphEdgeType = "insider_of" | "operates" | "calls_at" | "owns";

export interface GraphNode {
  id: string;
  type: GraphNodeType;
  label: string;
  attrs: Record<string, any>;
}

export interface GraphEdge {
  type: GraphEdgeType;
  from: string;
  to: string;
  source: string;
  confidence: "high" | "medium" | "low" | "unverified";
  first_seen: string | null;
  last_seen: string | null;
  attrs: Record<string, any>;
}

export interface EverythingGraph {
  kind: "raw";
  built_at: number;
  counts: {
    nodes: number; edges: number;
    company: number; person: number; facility: number; vessel: number;
    insider_of: number; operates: number; calls_at: number; owns: number;
  };
  nodes: GraphNode[];
  edges: GraphEdge[];
  caveat: string;
}

export interface SiteRow {
  id: string; name: string; category: string; lat: number; lon: number;
  operator?: string; relevance?: string;
}
type PlantRow = [name: string, capacity_mw: number, fuel: string, owner: string, lat: number, lon: number, verified: 0 | 1];
interface EntityMapEntry {
  operator: string; ticker: string | null;
  confidence: "high" | "medium" | "unmapped"; note: string; parent?: string;
}

const companyId = (ticker: string) => `company:${ticker.toUpperCase()}`;
const personId = (cik: string) => `person:${cik}`;
const siteFacilityId = (id: string) => `facility:site:${id}`;
const plantFacilityId = (idx: number) => `facility:plant:${idx}`;
const vesselId = (mmsi: string) => `vessel:${mmsi}`;

/** Facility nodes from the sites + plants registries (no time dimension —
 *  these are git-versioned reference data, not archived observations). */
function buildFacilityNodes(sites: SiteRow[], plants: PlantRow[]): GraphNode[] {
  const out: GraphNode[] = [];
  for (const s of sites) {
    out.push({
      id: siteFacilityId(s.id), type: "facility", label: s.name,
      attrs: {
        kind: "site", category: s.category, lat: s.lat, lon: s.lon,
        operator: s.operator ?? null, imagery_verified: true, // all 16 sites verified per strategic_sites.json _doc
      },
    });
  }
  plants.forEach((p, idx) => {
    out.push({
      id: plantFacilityId(idx), type: "facility", label: p[0],
      attrs: {
        kind: "plant", capacity_mw: p[1], fuel: p[2], owner: p[3],
        lat: p[4], lon: p[5], imagery_verified: p[6] === 1,
      },
    });
  });
  return out;
}

/** operates edges: company -> facility, from entity_map.json's hand-verified
 *  operator/owner -> ticker mapping. Only entries with a ticker (confidence
 *  high/medium) produce an edge; "unmapped" entries are honest gaps per the
 *  entity_map's own coverage-honesty rule and are skipped here too. */
function buildOperatesEdges(
  sites: SiteRow[], plants: PlantRow[], entities: EntityMapEntry[], builtDate: string,
): { edges: GraphEdge[]; companyNodes: Map<string, GraphNode> } {
  const byOperator = new Map<string, EntityMapEntry>();
  for (const e of entities) if (e.ticker) byOperator.set(e.operator, e);

  const edges: GraphEdge[] = [];
  const companyNodes = new Map<string, GraphNode>();
  const ensureCompany = (entry: EntityMapEntry): string => {
    const id = companyId(entry.ticker!);
    if (!companyNodes.has(id)) {
      companyNodes.set(id, {
        id, type: "company", label: entry.ticker!,
        attrs: { parent: entry.parent ?? null, tickers: [entry.ticker] },
      });
    }
    return id;
  };

  for (const s of sites) {
    if (!s.operator) continue;
    const entry = byOperator.get(s.operator);
    if (!entry) continue;
    edges.push({
      type: "operates", from: ensureCompany(entry), to: siteFacilityId(s.id),
      source: "datacore/entity_map.json + datacore/sites/strategic_sites.json",
      confidence: entry.confidence as "high" | "medium", first_seen: builtDate, last_seen: builtDate,
      attrs: { provenance: s.operator, parent: entry.parent ?? null },
    });
  }
  plants.forEach((p, idx) => {
    const owner = p[3];
    if (!owner) return;
    const entry = byOperator.get(owner);
    if (!entry) return;
    edges.push({
      type: "operates", from: ensureCompany(entry), to: plantFacilityId(idx),
      source: "datacore/entity_map.json + datacore/powerplants/us_power_plants.json",
      confidence: entry.confidence as "high" | "medium", first_seen: builtDate, last_seen: builtDate,
      attrs: { provenance: owner, parent: entry.parent ?? null },
    });
  });
  return { edges, companyNodes };
}

/** insider_of edges: person(CIK) -> company, aggregated across every
 *  archived Form 4 filing for that owner/issuer pair. Issuer identity
 *  prefers the filed ticker; filings without one (rare — some issuers omit
 *  issuerTradingSymbol) fall back to a CIK-keyed company id, labeled
 *  distinctly so it's never confused with a real ticker downstream. */
function buildInsiderEdges(filings: Form4Filing[]): { edges: GraphEdge[]; personNodes: Map<string, GraphNode>; companyNodes: Map<string, GraphNode> } {
  const personNodes = new Map<string, GraphNode>();
  const companyNodes = new Map<string, GraphNode>();
  const agg = new Map<string, {
    from: string; to: string; filingCount: number;
    firstSeen: string; lastSeen: string; lastTransactionKind: string | null;
    roles: Set<string>;
  }>();

  for (const f of filings) {
    if (!f.issuerCik) continue;
    const issuerId = f.issuerTradingSymbol ? companyId(f.issuerTradingSymbol) : `company:cik:${f.issuerCik}`;
    if (!companyNodes.has(issuerId)) {
      companyNodes.set(issuerId, {
        id: issuerId, type: "company",
        label: f.issuerTradingSymbol || f.issuerName || issuerId,
        attrs: {
          cik: f.issuerCik, name: f.issuerName ?? null,
          ticker_known: Boolean(f.issuerTradingSymbol),
        },
      });
    }
    const primaryKind =
      f.transactions.find((t) => t.table === "nonDerivative")?.kind ??
      f.transactions[0]?.kind ?? null;
    const filedAt = f.filedAt ?? "";

    for (const owner of f.owners) {
      const pId = personId(owner.cik);
      // last-seen filing wins the display name (SEC owner names are stable
      // but this keeps the node fresh if a name is ever amended).
      const existing = personNodes.get(pId);
      if (!existing || filedAt >= (existing.attrs.lastSeenFiledAt || "")) {
        personNodes.set(pId, { id: pId, type: "person", label: owner.name, attrs: { cik: owner.cik, lastSeenFiledAt: filedAt } });
      }
      const key = `${pId}|${issuerId}`;
      const roles: string[] = [];
      if (owner.isDirector) roles.push("director");
      if (owner.isOfficer) roles.push(owner.officerTitle ? `officer:${owner.officerTitle}` : "officer");
      if (owner.isTenPercentOwner) roles.push("10pct_owner");
      let a = agg.get(key);
      if (!a) {
        a = { from: pId, to: issuerId, filingCount: 0, firstSeen: filedAt, lastSeen: filedAt, lastTransactionKind: primaryKind, roles: new Set() };
        agg.set(key, a);
      }
      a.filingCount++;
      if (!a.firstSeen || filedAt < a.firstSeen) a.firstSeen = filedAt;
      if (!a.lastSeen || filedAt > a.lastSeen) { a.lastSeen = filedAt; a.lastTransactionKind = primaryKind; }
      roles.forEach((r) => a!.roles.add(r));
    }
  }

  const edges: GraphEdge[] = [];
  agg.forEach((a) => {
    edges.push({
      type: "insider_of", from: a.from, to: a.to,
      source: "SEC EDGAR Form 4 filings archive",
      confidence: "high", // SEC filing of record — this is not an inference
      first_seen: a.firstSeen || null, last_seen: a.lastSeen || null,
      attrs: { roles: Array.from(a.roles).sort(), filing_count: a.filingCount, last_transaction_kind: a.lastTransactionKind },
    });
  });
  return { edges, personNodes, companyNodes };
}

/** calls_at edges: vessel -> facility(port site), aggregated from our own
 *  AIS-derived port-visit detections (portDwell.ts). Ports are the subset
 *  of sites with category "port" — the only site coordinates in the repo
 *  with a verified geofence radius. */
function buildCallsAtEdges(visitsByPort: Map<string, PortVisit[]>, ports: PortDef[]): { edges: GraphEdge[]; vesselNodes: Map<string, GraphNode> } {
  const vesselNodes = new Map<string, GraphNode>();
  const byVesselPort = new Map<string, PortVisit[]>();
  visitsByPort.forEach((visits, portId) => {
    for (const v of visits) {
      const key = `${v.mmsi}|${portId}`;
      if (!byVesselPort.has(key)) byVesselPort.set(key, []);
      byVesselPort.get(key)!.push(v);
    }
  });
  const edges: GraphEdge[] = [];
  byVesselPort.forEach((visits: PortVisit[], key: string) => {
    const [mmsi, portId] = key.split("|");
    if (!ports.some((p) => p.id === portId)) return;
    const vId = vesselId(mmsi);
    const name = visits.find((v) => v.name)?.name;
    if (!vesselNodes.has(vId)) vesselNodes.set(vId, { id: vId, type: "vessel", label: name || mmsi, attrs: { mmsi } });
    const completed = visits.filter((v) => !v.ongoing).map((v) => v.dwellHours).sort((a, b) => a - b);
    const median = completed.length ? completed[Math.floor(completed.length / 2)] : null;
    const lastCall = Math.max(...visits.map((v) => v.lastSeen));
    edges.push({
      type: "calls_at", from: vId, to: siteFacilityId(portId),
      source: "Derived from our own AIS position archive (terrestrial coverage)",
      confidence: "high", // direct AIS observation, not an inference
      first_seen: new Date(Math.min(...visits.map((v) => v.firstSeen)) * 1000).toISOString(),
      last_seen: new Date(lastCall * 1000).toISOString(),
      attrs: { visit_count: visits.length, median_dwell_h: median, ongoing: visits.some((v) => v.ongoing) },
    });
  });
  return { edges, vesselNodes };
}

interface GemOwnershipEntity {
  "Entity ID": string;
  "Full Name": string;
  "US SEC Central Index Key"?: string;
}
interface GemOwnershipEdgeRow {
  "Subject Entity ID": string; "Subject Entity Name": string;
  "Interested Party ID": string; "Interested Party Name": string;
  "% Share of Ownership"?: number;
  "Data Source URL"?: string;
  "Share Imputed?"?: string;
}
export interface GemOwnershipFile {
  provenance: { attribution: string; license: string; release: string; built_at?: string };
  entities: GemOwnershipEntity[];
  entity_edges: GemOwnershipEdgeRow[];
}

/** Reads datacore/gem/ownership.json.gz (Global Energy Monitor Global Energy
 *  Ownership Tracker, CC BY 4.0 — scripts/gem_suite_ingest.py). Never throws:
 *  a missing/corrupt file degrades to null (no owns edges), matching the
 *  fetch-failure-degrades pattern getCikTickerMap already uses below, since
 *  a graph rebuild must never fail wholesale over one optional join source. */
function loadGemOwnership(fp: string = repoDataPath("datacore/gem/ownership.json.gz")): GemOwnershipFile | null {
  try {
    return JSON.parse(zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8"));
  } catch {
    return null;
  }
}

/** owns edges: company(owner) -> company(owned), from GEM's Global Energy
 *  Ownership Tracker entity_edges. Only pairs where BOTH the subject
 *  (owned) and interested party (owner) resolve to a real US SEC CIK are
 *  included — GEM also tracks governments, private holders, and foreign
 *  entities with no CIK, which this v1 slice honestly excludes rather than
 *  mistype them as `company` nodes (that broader join is a documented
 *  follow-up, see research/open_questions.md). Node ids reuse the exact
 *  same ticker-preferred / `company:cik:<CIK>` fallback scheme
 *  buildInsiderEdges uses (via the same `getCikTickerMap` resolver), so a
 *  GEM-tracked company lands on the SAME node the EDGAR pipeline already
 *  populates rather than a duplicate. */
function buildOwnershipEdges(
  gem: GemOwnershipFile | null, cikTicker: Map<string, string>,
): { edges: GraphEdge[]; companyNodes: Map<string, GraphNode> } {
  const edges: GraphEdge[] = [];
  const companyNodes = new Map<string, GraphNode>();
  if (!gem) return { edges, companyNodes };
  // GEM's own release date, not entity_map.json's builtDate (a different
  // registry with its own vintage) — first_seen/last_seen must reflect
  // the actual source this edge came from.
  const builtDate: string = gem.provenance?.built_at ?? "unknown";

  const cikByEntity = new Map<string, string>();
  for (const e of gem.entities) {
    const cik = e["US SEC Central Index Key"];
    if (cik && cik !== "not found") cikByEntity.set(e["Entity ID"], cik);
  }

  const ensureCompany = (entityId: string, name: string): string => {
    const cik = cikByEntity.get(entityId)!;
    const ticker = cikTicker.get(String(Number(cik)));
    const id = ticker ? companyId(ticker) : `company:cik:${cik}`;
    if (!companyNodes.has(id)) {
      companyNodes.set(id, { id, type: "company", label: ticker || name, attrs: { cik, name, ticker_known: Boolean(ticker) } });
    }
    return id;
  };

  const seen = new Set<string>();
  for (const row of gem.entity_edges) {
    if (!cikByEntity.has(row["Subject Entity ID"]) || !cikByEntity.has(row["Interested Party ID"])) continue;
    const fromId = ensureCompany(row["Interested Party ID"], row["Interested Party Name"]);
    const toId = ensureCompany(row["Subject Entity ID"], row["Subject Entity Name"]);
    if (fromId === toId) continue;
    const key = `${fromId}|${toId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const share = row["% Share of Ownership"];
    edges.push({
      type: "owns", from: fromId, to: toId,
      source: "Global Energy Monitor Global Energy Ownership Tracker (CC BY 4.0)",
      confidence: row["Share Imputed?"] === "known value, not imputed" ? "high" : "medium",
      first_seen: builtDate, last_seen: builtDate,
      attrs: { share_pct: typeof share === "number" ? share : null, imputed: row["Share Imputed?"] ?? null, source_url: row["Data Source URL"] ?? null },
    });
  }
  return { edges, companyNodes };
}

export interface BuildGraphOpts {
  baseDir?: string; nowMs?: number; portWindowHours?: number;
  sites?: SiteRow[]; plants?: PlantRow[]; entities?: EntityMapEntry[];
  form4Filings?: Form4Filing[];
  portVisitsByPort?: Map<string, PortVisit[]>;
  /** GEM ownership file; explicit null/object overrides the disk read
   *  (tests never hit disk or network). Omit to load the real file. */
  gemOwnership?: GemOwnershipFile | null;
  /** CIK -> ticker map; overrides the live SEC company_tickers.json fetch
   *  (tests never hit the network). Omit to use getCikTickerMap(). */
  cikTickerMap?: Map<string, string>;
}

/** Builds the full v1 graph. Every IO source is overridable so tests run
 *  against small deterministic fixtures instead of the real archive/repo
 *  files (mirrors the shadowFleet/portDwell baseDir-injectable pattern). */
export async function buildGraph(opts: BuildGraphOpts = {}): Promise<EverythingGraph> {
  const sites: SiteRow[] = opts.sites ?? ((sitesJson as any).sites || []);
  const plants: PlantRow[] = opts.plants ?? ((plantsJson as any).plants || []);
  const entities: EntityMapEntry[] = opts.entities ?? ((entityMapJson as any).entities || []);
  const builtDate: string = (entityMapJson as any).built || "unknown";

  const form4Filings = opts.form4Filings ?? readFilingHistory(30, opts.baseDir, opts.nowMs);

  const ports = portsFromSites(sites);
  let portVisitsByPort = opts.portVisitsByPort;
  if (!portVisitsByPort) {
    const { visitsByPort } = await foldPortVisitsAsync(ports, opts.portWindowHours ?? 168, opts.baseDir, opts.nowMs);
    portVisitsByPort = visitsByPort;
  }

  const facilityNodes = buildFacilityNodes(sites, plants);
  const { edges: operatesEdges, companyNodes: operatesCompanies } = buildOperatesEdges(sites, plants, entities, builtDate);
  const { edges: insiderEdges, personNodes, companyNodes: insiderCompanies } = buildInsiderEdges(form4Filings);
  const { edges: callsAtEdges, vesselNodes } = buildCallsAtEdges(portVisitsByPort, ports);
  const gemOwnership = opts.gemOwnership !== undefined ? opts.gemOwnership : loadGemOwnership();
  const cikTickerMap = opts.cikTickerMap ?? (gemOwnership ? await getCikTickerMap() : new Map<string, string>());
  const { edges: ownsEdges, companyNodes: ownsCompanies } = buildOwnershipEdges(gemOwnership, cikTickerMap);

  const companyNodes = new Map<string, GraphNode>(operatesCompanies);
  insiderCompanies.forEach((n, id) => { if (!companyNodes.has(id)) companyNodes.set(id, n); });
  ownsCompanies.forEach((n, id) => { if (!companyNodes.has(id)) companyNodes.set(id, n); });

  const nodes: GraphNode[] = [
    ...Array.from(companyNodes.values()), ...Array.from(personNodes.values()),
    ...facilityNodes, ...Array.from(vesselNodes.values()),
  ];
  const edges: GraphEdge[] = [...operatesEdges, ...insiderEdges, ...callsAtEdges, ...ownsEdges];

  return {
    kind: "raw",
    built_at: opts.nowMs ?? Date.now(),
    counts: {
      nodes: nodes.length, edges: edges.length,
      company: companyNodes.size, person: personNodes.size, facility: facilityNodes.length, vessel: vesselNodes.size,
      insider_of: insiderEdges.length, operates: operatesEdges.length, calls_at: callsAtEdges.length, owns: ownsEdges.length,
    },
    nodes, edges,
    caveat: "RAW graph — every edge asserts a relationship with provenance " +
      "(source/confidence/first_seen/last_seen), no predictive claim. " +
      "operates edges are static reference data (datacore/entity_map.json, " +
      `built ${builtDate}); unmapped operators are honest gaps, never guessed ` +
      "tickers. insider_of reflects only the last 30 days of archived Form 4 " +
      "filings. calls_at reflects our own terrestrial-AIS archive only " +
      "(coverage gaps and archive age both understate true call counts). " +
      "owns reflects Global Energy Monitor's Global Energy Ownership Tracker " +
      "(CC BY 4.0) restricted to pairs where BOTH ends resolve to a real US " +
      "SEC CIK — government/private/foreign owners without a CIK are an " +
      "honest gap, not a guessed identifier.",
  };
}

/** BFS neighborhood of one entity, both edge directions, up to `hops` away.
 *  `entity` may be a full node id (company:STLD) or a bare ticker/MMSI/CIK —
 *  resolved case-insensitively against company tickers first, then any node
 *  id suffix, for a friendlier query param. */
export function resolveEntityId(graph: EverythingGraph, entity: string): string | null {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  if (byId.has(entity)) return entity;
  const asCompany = companyId(entity);
  if (byId.has(asCompany)) return asCompany;
  const lower = entity.toLowerCase();
  const hit = graph.nodes.find((n) => n.id.toLowerCase() === lower || n.id.toLowerCase().endsWith(`:${lower}`));
  return hit?.id ?? null;
}

export function neighborhood(graph: EverythingGraph, entityId: string, hops = 1): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  if (!byId.has(entityId)) return { nodes: [], edges: [] };
  const adjacency = new Map<string, GraphEdge[]>();
  for (const e of graph.edges) {
    if (!adjacency.has(e.from)) adjacency.set(e.from, []);
    if (!adjacency.has(e.to)) adjacency.set(e.to, []);
    adjacency.get(e.from)!.push(e);
    adjacency.get(e.to)!.push(e);
  }
  const visitedNodes = new Set<string>([entityId]);
  const visitedEdges = new Set<GraphEdge>();
  let frontier = [entityId];
  for (let h = 0; h < Math.max(0, hops); h++) {
    const next: string[] = [];
    for (const id of frontier) {
      for (const e of adjacency.get(id) || []) {
        visitedEdges.add(e);
        const other = e.from === id ? e.to : e.from;
        if (!visitedNodes.has(other)) { visitedNodes.add(other); next.push(other); }
      }
    }
    frontier = next;
  }
  return {
    nodes: Array.from(visitedNodes).map((id) => byId.get(id)!).filter(Boolean),
    edges: Array.from(visitedEdges),
  };
}

// ── shared eager-poll cache ─────────────────────────────────────────────────
// Lifted out of server/routes.ts (was a private per-route closure) so a
// second consumer (server/dossier.ts, W5) can read the SAME 15-min-fresh
// graph instead of triggering its own independent 168h-AIS-fold rebuild —
// the exact "R4/R5 hazard" (per analyst.ts's tool-exclusion note) this
// module was already careful to avoid for its one existing caller.
// Behavior preserved exactly: non-blocking, warming_up until the first
// build completes (never blocks a request on the expensive rebuild).
let graphCache: { at: number; graph: EverythingGraph } | null = null;
let graphRefreshing = false;

/** Synchronous peek — null until the first background build completes. */
export function cachedGraphSync(): EverythingGraph | null {
  return graphCache?.graph ?? null;
}

export function bootGraphPoll(intervalMs = 15 * 60_000): void {
  const refresh = async () => {
    if (graphRefreshing) return;
    graphRefreshing = true;
    try {
      graphCache = { at: Date.now(), graph: await buildGraph() };
    } catch (e: any) {
      console.error("[datacore] entityGraph refresh:", e?.message || e);
    } finally {
      graphRefreshing = false;
    }
  };
  refresh();
  setInterval(refresh, intervalMs).unref?.();
}

export function _resetGraphCache() { graphCache = null; graphRefreshing = false; }
