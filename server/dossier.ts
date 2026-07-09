/**
 * dossier.ts — ENTITY DOSSIER v2 (ANALYST CONSOLE charter W5,
 * research/console_charter.md). "Click anything -> one panel: identity
 * (spine), cross-layer history (W4/Everything Graph), related filings/
 * contracts (EDGAR/USAspending streams), nearest strategic sites, ticker
 * linkage where it exists."
 *
 * This module performs NO new collection and duplicates NO existing join —
 * it composes three already-built, already-cached primitives:
 *   - server/entityGraph.ts's Everything Graph (identity + cross-layer
 *     neighborhood + ticker linkage + insider filings, all already joined
 *     as insider_of/operates/calls_at edges with provenance)
 *   - server/usaSpending.ts's ticker-resolved USAspending contract feed
 *     (the one genuinely NEW cross-stream join this file adds: contracts
 *     were not previously joined into the graph)
 *   - datacore/sites/strategic_sites.json + the shared haversineKm helper
 *     (server/firesFacilities.ts) for "nearest strategic sites"
 *
 * Pure function over injected data (mirrors buildGraph's own
 * every-IO-source-overridable test convention) — no fs/network access
 * here, so tests run against small deterministic fixtures.
 */
import type { EverythingGraph, GraphNode, GraphEdge, SiteRow } from "./entityGraph";
import { resolveEntityId, neighborhood } from "./entityGraph";
import type { ContractTxn } from "./usaSpending";
import { haversineKm } from "./firesFacilities";
import sitesJson from "../datacore/sites/strategic_sites.json";

export const NEAREST_SITES_CAP = 5;
export const CONTRACTS_CAP = 10;
export const DEFAULT_HOPS = 2;

export interface DossierParams {
  entity?: string | null;
  lat?: number | null;
  lon?: number | null;
  hops?: number;
}

export interface NearestSite {
  id: string;
  name: string;
  category: string;
  km: number;
}

export interface DossierIdentity {
  id: string;
  type: GraphNode["type"];
  label: string;
  attrs: Record<string, any>;
}

export interface DossierResult {
  kind: "raw";
  built_at: number | null;
  query: { entity: string | null; lat: number | null; lon: number | null; hops: number };
  identity: DossierIdentity | null;
  graph: { nodes: GraphNode[]; edges: GraphEdge[] } | null;
  contracts: ContractTxn[];
  contracts_capped: boolean;
  nearest_sites: NearestSite[];
  caveat: string;
}

const tickerOf = (n: GraphNode): string | null => {
  if (n.type !== "company") return null;
  if (n.id.startsWith("company:cik:")) return null; // CIK-fallback id — no real ticker known
  return n.id.slice("company:".length);
};

export function buildDossier(
  graph: EverythingGraph | null,
  params: DossierParams,
  opts: { contracts?: ContractTxn[] | null; sites?: SiteRow[] } = {},
): DossierResult {
  const hops = Math.max(0, Math.min(3, params.hops ?? DEFAULT_HOPS));
  const entityQ = params.entity?.trim() || null;
  let identity: DossierIdentity | null = null;
  let graphNbhd: { nodes: GraphNode[]; edges: GraphEdge[] } | null = null;
  let anchorLat = params.lat ?? null;
  let anchorLon = params.lon ?? null;
  let resolvedId: string | null = null;

  if (graph && entityQ) {
    resolvedId = resolveEntityId(graph, entityQ);
    if (resolvedId) {
      const node = graph.nodes.find((n) => n.id === resolvedId) ?? null;
      if (node) {
        identity = { id: node.id, type: node.type, label: node.label, attrs: node.attrs };
        graphNbhd = neighborhood(graph, resolvedId, hops);
        if (typeof node.attrs?.lat === "number" && typeof node.attrs?.lon === "number") {
          anchorLat = node.attrs.lat;
          anchorLon = node.attrs.lon;
        }
      }
    }
  }

  // Ticker linkage: the resolved entity itself (if a company) plus every
  // company node in its neighborhood — this is what makes "related
  // contracts" possible for a facility (site/plant) whose OWN node is not
  // a company but whose operates edge reaches one.
  const tickers = new Set<string>();
  if (graph && identity) {
    const selfNode = graph.nodes.find((n) => n.id === identity!.id);
    const t = selfNode ? tickerOf(selfNode) : null;
    if (t) tickers.add(t);
  }
  graphNbhd?.nodes.forEach((n) => {
    const t = tickerOf(n);
    if (t) tickers.add(t);
  });

  const contractsAll = opts.contracts ?? [];
  const matched = tickers.size
    ? contractsAll.filter((c) => c.tkr && tickers.has(c.tkr))
    : [];
  matched.sort((a, b) => (b.rt || "").localeCompare(a.rt || ""));
  const contracts = matched.slice(0, CONTRACTS_CAP);

  const sites: SiteRow[] = opts.sites ?? ((sitesJson as any).sites || []);
  let nearestSites: NearestSite[] = [];
  if (anchorLat != null && anchorLon != null && Number.isFinite(anchorLat) && Number.isFinite(anchorLon)) {
    nearestSites = sites
      .filter((s) => `facility:site:${s.id}` !== resolvedId)
      .map((s) => ({ id: s.id, name: s.name, category: s.category, km: haversineKm(anchorLat!, anchorLon!, s.lat, s.lon) }))
      .sort((a, b) => a.km - b.km)
      .slice(0, NEAREST_SITES_CAP);
  }

  return {
    kind: "raw",
    built_at: graph?.built_at ?? null,
    query: { entity: entityQ, lat: params.lat ?? null, lon: params.lon ?? null, hops },
    identity,
    graph: graphNbhd,
    contracts,
    contracts_capped: matched.length > contracts.length,
    nearest_sites: nearestSites,
    caveat: "RAW composition — identity/graph from the Everything Graph " +
      "(operates/insider_of/calls_at edges, each with its own source/" +
      "confidence/first_seen/last_seen), contracts from USAspending.gov " +
      "matched by resolved ticker only (never fuzzy; DoD/USACE awards " +
      "publish ~90 days late), nearest_sites by straight-line distance to " +
      `up to ${NEAREST_SITES_CAP} of our ${sites.length}-site imagery-` +
      "verified registry. No predictive claim anywhere in this payload.",
  };
}
