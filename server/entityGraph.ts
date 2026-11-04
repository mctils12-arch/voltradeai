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
import { performance } from "perf_hooks";
import sitesJson from "../datacore/sites/strategic_sites.json";
import { LAZY_TABLES } from "./lazyAssets"; // us_power_plants.json: 1.2MB, read on first use (A7)
import entityMapJson from "../datacore/entity_map.json";
import { readFilingHistory, type Form4Filing } from "./edgarForm4";
import { foldPortVisitsAsync, portsFromSites, type PortDef, type PortVisit } from "./portDwell";
import { repoDataPath } from "./repoFiles";
import { getCikTickerMap } from "./sec8kEarnings";
import { firstDifference, parseFoldMode, type AuditWriter, type FoldMode } from "./foldShadow";

export type GraphNodeType = "company" | "person" | "facility" | "vessel" | "institution";
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
    company: number; person: number; facility: number; vessel: number; institution: number;
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
const institutionId = (gemEntityId: string) => `institution:${gemEntityId}`;

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
  "Entity Type"?: string;
  "Headquarters Country"?: string;
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

/** The slice of a GEM ownership release that the owns-edge builder actually
 *  reads — the per-entity CIK / type / HQ lookups plus the CIK-anchored edge
 *  rows, in file order and NOT yet deduplicated (dedupe depends on the
 *  CIK->ticker map, which is resolved at build time, not here). Live file,
 *  measured 2026-11-03: 3.8 MB gz / 36.5 MB inflated, 26,250 entities and
 *  24,351 edges -> 277 CIK entities + 1,403 anchored rows. This slice is
 *  what the 15-minute poll keeps between polls (loadGemOwnershipDistilledCached
 *  below, memory fix A2b) instead of gunzipping + parsing the whole release
 *  every time it rebuilds the graph. */
export interface GemOwnershipDistilled {
  builtDate: string;
  cikByEntity: Map<string, string>;
  typeByEntity: Map<string, string>;
  hqByEntity: Map<string, string>;
  rows: GemAnchoredRow[];
}
export interface GemAnchoredRow {
  subjectId: string; subjectName: string; ownerId: string; ownerName: string;
  share?: number; imputed?: string; url?: string;
}

/** First half of the former buildOwnershipEdges, its rules verbatim: the
 *  entity lookups and the "at least one end has a real CIK" anchoring test
 *  (rationale in buildOwnershipEdgesFromDistilled). The type/HQ lookups are
 *  then pruned to the ids an anchored row can reference — the only ids the
 *  second half ever looks up — which is where the size falls away. */
export function distillGemOwnership(gem: GemOwnershipFile): GemOwnershipDistilled {
  // GEM's own release date, not entity_map.json's builtDate (a different
  // registry with its own vintage) — first_seen/last_seen must reflect
  // the actual source this edge came from.
  const builtDate: string = gem.provenance?.built_at ?? "unknown";

  const cikByEntity = new Map<string, string>();
  const typeByEntity = new Map<string, string>();
  const hqByEntity = new Map<string, string>();
  for (const e of gem.entities) {
    const cik = e["US SEC Central Index Key"];
    if (cik && cik !== "not found") cikByEntity.set(e["Entity ID"], cik);
    typeByEntity.set(e["Entity ID"], e["Entity Type"] || "unknown entity");
    if (e["Headquarters Country"]) hqByEntity.set(e["Entity ID"], e["Headquarters Country"]);
  }

  const rows: GemAnchoredRow[] = [];
  const referenced = new Set<string>();
  for (const row of gem.entity_edges) {
    const subjectId = row["Subject Entity ID"];
    const ownerId = row["Interested Party ID"];
    if (!cikByEntity.has(subjectId) && !cikByEntity.has(ownerId)) continue; // no CIK anchor at all -> honest gap
    referenced.add(subjectId);
    referenced.add(ownerId);
    rows.push({
      subjectId, subjectName: row["Subject Entity Name"], ownerId, ownerName: row["Interested Party Name"],
      share: row["% Share of Ownership"], imputed: row["Share Imputed?"], url: row["Data Source URL"],
    });
  }
  for (const id of Array.from(typeByEntity.keys())) if (!referenced.has(id)) typeByEntity.delete(id);
  for (const id of Array.from(hqByEntity.keys())) if (!referenced.has(id)) hqByEntity.delete(id);
  return { builtDate, cikByEntity, typeByEntity, hqByEntity, rows };
}

/** owns edges: (company|institution)(owner) -> (company|institution)(owned),
 *  from GEM's Global Energy Ownership Tracker entity_edges. An edge is kept
 *  whenever AT LEAST ONE end resolves to a real US SEC CIK — that CIK-mapped
 *  side anchors the edge into a node our EDGAR/insider pipeline already
 *  knows, so it is never a join between two entities we have zero context
 *  for (the far larger neither-end-CIK pool, ~93% of GEM's edges, stays
 *  excluded — it would explode node count with entities this platform
 *  cannot otherwise place). The non-CIK side — a government, state body,
 *  private/foreign parent, or unincorporated arrangement — is NOT mistyped
 *  as a `company` node (v1 CIK-only-both-ends behavior, superseded here):
 *  it becomes a distinct `institution` node, typed honestly from GEM's own
 *  `Entity Type` field (state / state body / legal entity / arrangement /
 *  person / unknown entity), so a reader can never mistake an unverified
 *  foreign or state owner for an SEC-reporting company. (Design decided in
 *  research/open_questions.md's 2026-07-17 EVERYTHING-GRAPH filing: a new
 *  node type over a looser `cik_verified` flag on `company`.) CIK-mapped
 *  node ids reuse the exact same ticker-preferred / `company:cik:<CIK>`
 *  fallback scheme buildInsiderEdges uses (via `getCikTickerMap`), so a
 *  GEM-tracked company lands on the SAME node the EDGAR pipeline already
 *  populates rather than a duplicate. Takes the distilled slice (2026-11-03,
 *  memory fix A2b) — the anchoring filter already ran in distillGemOwnership;
 *  node creation order, self-edge drop and first-wins dedupe are unchanged. */
function buildOwnershipEdgesFromDistilled(
  gem: GemOwnershipDistilled | null, cikTicker: Map<string, string>,
): { edges: GraphEdge[]; companyNodes: Map<string, GraphNode>; institutionNodes: Map<string, GraphNode> } {
  const edges: GraphEdge[] = [];
  const companyNodes = new Map<string, GraphNode>();
  const institutionNodes = new Map<string, GraphNode>();
  if (!gem) return { edges, companyNodes, institutionNodes };
  const { builtDate, cikByEntity, typeByEntity, hqByEntity } = gem;

  const ensureNode = (entityId: string, name: string): string => {
    const cik = cikByEntity.get(entityId);
    if (cik) {
      const ticker = cikTicker.get(String(Number(cik)));
      const id = ticker ? companyId(ticker) : `company:cik:${cik}`;
      if (!companyNodes.has(id)) {
        companyNodes.set(id, { id, type: "company", label: ticker || name, attrs: { cik, name, ticker_known: Boolean(ticker) } });
      }
      return id;
    }
    const id = institutionId(entityId);
    if (!institutionNodes.has(id)) {
      institutionNodes.set(id, {
        id, type: "institution", label: name,
        attrs: {
          gem_entity_id: entityId, gem_entity_type: typeByEntity.get(entityId) ?? "unknown entity",
          headquarters_country: hqByEntity.get(entityId) ?? null, cik_verified: false,
        },
      });
    }
    return id;
  };

  const seen = new Set<string>();
  for (const row of gem.rows) {
    const fromId = ensureNode(row.ownerId, row.ownerName);
    const toId = ensureNode(row.subjectId, row.subjectName);
    if (fromId === toId) continue;
    const key = `${fromId}|${toId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const share = row.share;
    edges.push({
      type: "owns", from: fromId, to: toId,
      source: "Global Energy Monitor Global Energy Ownership Tracker (CC BY 4.0)",
      confidence: row.imputed === "known value, not imputed" ? "high" : "medium",
      first_seen: builtDate, last_seen: builtDate,
      attrs: { share_pct: typeof share === "number" ? share : null, imputed: row.imputed ?? null, source_url: row.url ?? null },
    });
  }
  return { edges, companyNodes, institutionNodes };
}

export interface BuildGraphOpts {
  baseDir?: string; nowMs?: number; portWindowHours?: number;
  sites?: SiteRow[]; plants?: PlantRow[]; entities?: EntityMapEntry[];
  form4Filings?: Form4Filing[];
  portVisitsByPort?: Map<string, PortVisit[]>;
  /** GEM ownership file; explicit null/object overrides the disk read
   *  (tests never hit disk or network). Omit to load the real file. */
  gemOwnership?: GemOwnershipFile | null;
  /** The distilled GEM slice (memory fix A2b, 2026-11-03): an explicit
   *  null/object wins over both `gemOwnership` and the disk read. Production
   *  reaches it through loadGemOwnershipDistilledCached in buildGraphManaged;
   *  tests and the shadow comparison pass it directly. */
  gemDistilled?: GemOwnershipDistilled | null;
  /** Path of the GEM ownership .json.gz (test seam; default: the repo file). */
  gemPath?: string;
  /** CIK -> ticker map; overrides the live SEC company_tickers.json fetch
   *  (tests never hit the network). Omit to use getCikTickerMap(). */
  cikTickerMap?: Map<string, string>;
}

/** Every non-GEM input, resolved exactly as buildGraph always did — once.
 *  Split out (2026-11-03, memory fix A2b) so the shadow poll can hand the
 *  SAME resolved inputs to both of its builds: the two graphs then differ
 *  only through the GEM path under comparison, and the 168h AIS fold (the
 *  expensive input) runs once per poll, never twice. */
export async function resolveGraphInputs(opts: BuildGraphOpts): Promise<{
  sites: SiteRow[]; plants: PlantRow[]; entities: EntityMapEntry[];
  form4Filings: Form4Filing[]; portVisitsByPort: Map<string, PortVisit[]>;
}> {
  const sites: SiteRow[] = opts.sites ?? ((sitesJson as any).sites || []);
  const plants: PlantRow[] = opts.plants ?? ((LAZY_TABLES.usPowerPlants.get() as any).plants || []);
  const entities: EntityMapEntry[] = opts.entities ?? ((entityMapJson as any).entities || []);
  const form4Filings = opts.form4Filings ?? readFilingHistory(30, opts.baseDir, opts.nowMs);
  let portVisitsByPort = opts.portVisitsByPort;
  if (!portVisitsByPort) {
    const { visitsByPort } = await foldPortVisitsAsync(portsFromSites(sites), opts.portWindowHours ?? 168, opts.baseDir, opts.nowMs);
    portVisitsByPort = visitsByPort;
  }
  return { sites, plants, entities, form4Filings, portVisitsByPort };
}

/** Builds the full v1 graph. Every IO source is overridable so tests run
 *  against small deterministic fixtures instead of the real archive/repo
 *  files (mirrors the shadowFleet/portDwell baseDir-injectable pattern). */
export async function buildGraph(opts: BuildGraphOpts = {}): Promise<EverythingGraph> {
  const { sites, plants, entities, form4Filings, portVisitsByPort } = await resolveGraphInputs(opts);
  const builtDate: string = (entityMapJson as any).built || "unknown";
  const ports = portsFromSites(sites);

  const facilityNodes = buildFacilityNodes(sites, plants);
  const { edges: operatesEdges, companyNodes: operatesCompanies } = buildOperatesEdges(sites, plants, entities, builtDate);
  const { edges: insiderEdges, personNodes, companyNodes: insiderCompanies } = buildInsiderEdges(form4Filings);
  const { edges: callsAtEdges, vesselNodes } = buildCallsAtEdges(portVisitsByPort, ports);
  // GEM source precedence: a pinned distilled slice; else an injected raw
  // file, distilled here; else the FULL path — gunzip + parse the whole
  // release on every call, which is what `full` mode keeps and `incremental`
  // mode replaces with loadGemOwnershipDistilledCached (buildGraphManaged).
  const gemDistilled: GemOwnershipDistilled | null =
    opts.gemDistilled !== undefined ? opts.gemDistilled
      : opts.gemOwnership !== undefined ? (opts.gemOwnership ? distillGemOwnership(opts.gemOwnership) : null)
        : distillOrNull(loadGemOwnership(opts.gemPath));
  const cikTickerMap = opts.cikTickerMap ?? (gemDistilled ? await getCikTickerMap() : new Map<string, string>());
  const { edges: ownsEdges, companyNodes: ownsCompanies, institutionNodes } = buildOwnershipEdgesFromDistilled(gemDistilled, cikTickerMap);

  const companyNodes = new Map<string, GraphNode>(operatesCompanies);
  insiderCompanies.forEach((n, id) => { if (!companyNodes.has(id)) companyNodes.set(id, n); });
  ownsCompanies.forEach((n, id) => { if (!companyNodes.has(id)) companyNodes.set(id, n); });

  const nodes: GraphNode[] = [
    ...Array.from(companyNodes.values()), ...Array.from(personNodes.values()),
    ...facilityNodes, ...Array.from(vesselNodes.values()), ...Array.from(institutionNodes.values()),
  ];
  const edges: GraphEdge[] = [...operatesEdges, ...insiderEdges, ...callsAtEdges, ...ownsEdges];

  return {
    kind: "raw",
    built_at: opts.nowMs ?? Date.now(),
    counts: {
      nodes: nodes.length, edges: edges.length,
      company: companyNodes.size, person: personNodes.size, facility: facilityNodes.length, vessel: vesselNodes.size,
      institution: institutionNodes.size,
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
      "(CC BY 4.0) restricted to edges with AT LEAST ONE end resolving to a " +
      "real US SEC CIK; the other end is a `company` node when it is also " +
      "CIK-verified, or an honestly-typed `institution` node (state, state " +
      "body, private/foreign entity, arrangement) when it is not — never a " +
      "guessed ticker or a company node with an unverified identity.",
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
      graphCache = { at: Date.now(), graph: await buildGraphManaged() };
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

// ── managed poll: distilled GEM cache + three-mode shadow (memory fix A2b, 2026-11-03) ──
// WHAT THIS FIXES. Every 15-minute poll gunzipped and JSON.parsed the whole
// GEM ownership release (3.8 MB gz -> 36.5 MB of text -> a transient parse
// tree of roughly that size again) to extract 1,403 anchored rows and 277
// CIK lookups — a periodic heap spike on the same event loop the trading
// tier runs on, for a git-versioned file that changes only on deploy.
// The distilled slice (GemOwnershipDistilled) is kept instead, keyed by the
// file's mtime + size so a rewritten file still reloads; it is small enough
// (<1 MB for the live release) to hold permanently, which is why there is
// no idle TTL: the raw 36 MB is never retained in any mode.
//
// MODES (env VOLTRADE_GRAPH_POLL, parsed by foldShadow.ts): `full` =
// buildGraph() as before; `shadow` (DEFAULT) = resolve the shared inputs
// once (Form 4 history, the 168h AIS port fold), assemble the graph BOTH
// ways — raw-parse-per-poll vs the cached slice — deep-compare with
// built_at excluded, log any difference as FOLD-MISMATCH graph (console.error
// + audit row) and SERVE THE FULL BUILD; `incremental` = the cached slice
// only. Sharing the resolved inputs is what makes the comparison exact: the
// AIS fold's output moves as the archive grows, so two independent
// buildGraph() calls would differ for reasons that are not bugs.
// Rollback: VOLTRADE_GRAPH_POLL=full. The /api/data/graph payload is
// byte-identical across modes (one builder, one edge order).

export type GraphPollMode = FoldMode;
export const GRAPH_POLL_ENV = "VOLTRADE_GRAPH_POLL";

export function graphPollMode(): GraphPollMode {
  return parseFoldMode(process.env[GRAPH_POLL_ENV]);
}

const distillOrNull = (gem: GemOwnershipFile | null): GemOwnershipDistilled | null => (gem ? distillGemOwnership(gem) : null);

interface GemDistilledCacheEntry {
  fp: string;
  mtimeMs: number;
  size: number;
  distilled: GemOwnershipDistilled | null;
  loadedAt: number;
  loadMs: number;
}
let gemDistilledCache: GemDistilledCacheEntry | null = null;

export interface GraphPollStats {
  mode: GraphPollMode;
  /** datasets held between polls: 1 while the distilled GEM slice is resident, else 0 */
  partialsCached: number;
  lastFullMs: number | null;
  lastIncrementalMs: number | null;
  /** shadow mode: the shared inputs' share of both numbers above (Form 4 read + 168h AIS fold) */
  lastSharedInputsMs: number | null;
  mismatches: number;
  lastMismatchAt: string | null;
  lastMismatchDetail: string | null;
  polls: number;
  lastPollAt: string | null;
  gem: {
    cached: boolean; mtimeMs: number | null; sizeBytes: number | null;
    anchoredRows: number | null; cikEntities: number | null; loadMs: number | null; reloads: number;
  };
}

interface PollStatsState {
  mode: GraphPollMode; lastFullMs: number | null; lastIncrementalMs: number | null; lastSharedInputsMs: number | null;
  mismatches: number; lastMismatchAt: string | null; lastMismatchDetail: string | null;
  polls: number; lastPollAt: string | null; gemReloads: number;
}
const freshPollStats = (): PollStatsState => ({
  mode: graphPollMode(), lastFullMs: null, lastIncrementalMs: null, lastSharedInputsMs: null,
  mismatches: 0, lastMismatchAt: null, lastMismatchDetail: null, polls: 0, lastPollAt: null, gemReloads: 0,
});
let pollStats: PollStatsState = freshPollStats();
let graphAuditWriter: AuditWriter | null = null;

/** Snapshot for the health diagnostic. `mode` is the mode the LAST poll ran
 *  in (the env's current value before any poll). */
export function graphPollStats(): GraphPollStats {
  const c = gemDistilledCache;
  const { gemReloads, ...rest } = pollStats;
  return {
    ...rest,
    partialsCached: c && c.distilled ? 1 : 0,
    gem: {
      cached: c !== null, mtimeMs: c?.mtimeMs ?? null, sizeBytes: c?.size ?? null,
      anchoredRows: c?.distilled?.rows.length ?? null, cikEntities: c?.distilled?.cikByEntity.size ?? null,
      loadMs: c ? Math.round(c.loadMs) : null, reloads: gemReloads,
    },
  };
}

/** Wire the persistent audit log (routes.ts passes the same audit_log insert
 *  the compliance tripwire uses). Null unwires. */
export function setGraphPollAuditWriter(w: AuditWriter | null): void {
  graphAuditWriter = w;
}

/** The distilled GEM slice, parsed once per (path, mtime, size). A changed
 *  file — rewritten in place or touched — reloads; a missing file returns
 *  null AND drops the cache (nothing stale is ever served); an unreadable
 *  file caches null under its stat identity so a corrupt deploy is not
 *  re-parsed every poll (the raw loader returns null for it too). */
export function loadGemOwnershipDistilledCached(fp: string = repoDataPath("datacore/gem/ownership.json.gz")): GemOwnershipDistilled | null {
  let st: fs.Stats;
  try {
    st = fs.statSync(fp);
  } catch {
    gemDistilledCache = null;
    return null;
  }
  const hit = gemDistilledCache;
  if (hit && hit.fp === fp && hit.mtimeMs === st.mtimeMs && hit.size === st.size) return hit.distilled;
  const t0 = performance.now();
  const distilled = distillOrNull(loadGemOwnership(fp));
  gemDistilledCache = { fp, mtimeMs: st.mtimeMs, size: st.size, distilled, loadedAt: Date.now(), loadMs: performance.now() - t0 };
  pollStats.gemReloads++;
  return distilled;
}

/** The incremental side's GEM source: a pinned distilled slice stays; an
 *  injected raw fixture is distilled (fixture-driven tests); otherwise the
 *  cached slice from disk — the production path. */
function withDistilledGem(opts: BuildGraphOpts): BuildGraphOpts {
  if (opts.gemDistilled !== undefined) return opts;
  if (opts.gemOwnership !== undefined) return { ...opts, gemDistilled: distillOrNull(opts.gemOwnership) };
  return { ...opts, gemDistilled: loadGemOwnershipDistilledCached(opts.gemPath) };
}

function recordGraphMismatch(detail: string): void {
  pollStats.mismatches++;
  pollStats.lastMismatchAt = new Date().toISOString();
  pollStats.lastMismatchDetail = detail;
  // Drop the cached slice so the next poll re-derives it from the file; a
  // mismatch that persists across reloads is a logic fault, flagged every poll.
  gemDistilledCache = null;
  console.error(`FOLD-MISMATCH graph: ${detail}`);
  if (!graphAuditWriter) return;
  try {
    graphAuditWriter("FOLD-MISMATCH", `graph: ${detail}`.slice(0, 500));
  } catch (e: unknown) {
    console.error("[entityGraph] audit writer threw:", (e as Error)?.message || e);
  }
}

const roundMs = (t0: number) => Math.round((performance.now() - t0) * 10) / 10;

/** The managed build bootGraphPoll runs: same graph as buildGraph, dispatching
 *  on VOLTRADE_GRAPH_POLL (see the section header). */
export async function buildGraphManaged(opts: BuildGraphOpts = {}, mode: GraphPollMode = graphPollMode()): Promise<EverythingGraph> {
  pollStats.mode = mode;
  pollStats.polls++;
  pollStats.lastPollAt = new Date().toISOString();
  const t0 = performance.now();
  if (mode === "full") {
    const g = await buildGraph(opts);
    pollStats.lastFullMs = roundMs(t0);
    return g;
  }
  if (mode === "incremental") {
    const g = await buildGraph(withDistilledGem(opts));
    pollStats.lastIncrementalMs = roundMs(t0);
    return g;
  }
  // shadow: one resolution of the shared inputs, two assemblies, exact compare, serve full
  const nowMs = opts.nowMs ?? Date.now();
  const shared: BuildGraphOpts = { ...opts, ...(await resolveGraphInputs({ ...opts, nowMs })), nowMs };
  const sharedMs = roundMs(t0);
  const t1 = performance.now();
  const full = await buildGraph(shared);
  const fullMs = roundMs(t1);
  const t2 = performance.now();
  const incremental = await buildGraph(withDistilledGem(shared));
  const incMs = roundMs(t2);
  pollStats.lastSharedInputsMs = sharedMs;
  pollStats.lastFullMs = Math.round((sharedMs + fullMs) * 10) / 10;
  pollStats.lastIncrementalMs = Math.round((sharedMs + incMs) * 10) / 10;
  const diff = firstDifference({ ...full, built_at: 0 }, { ...incremental, built_at: 0 });
  if (diff) recordGraphMismatch(diff);
  return full;
}

/** Test seams: fresh stats, no cached slice, no audit writer. */
export function _resetGraphPollForTests(): void {
  pollStats = freshPollStats();
  gemDistilledCache = null;
  graphAuditWriter = null;
}

export function _graphPollGemCacheForTests(): GemDistilledCacheEntry | null {
  return gemDistilledCache;
}
