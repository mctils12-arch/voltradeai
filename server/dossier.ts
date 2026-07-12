/**
 * dossier.ts — ENTITY DOSSIER v2 (ANALYST CONSOLE charter W5,
 * research/console_charter.md). "Click anything -> one panel: identity
 * (spine), cross-layer history (W4/Everything Graph), related filings/
 * contracts (EDGAR/USAspending streams), nearest strategic sites, ticker
 * linkage where it exists."
 *
 * This module performs NO new collection and duplicates NO existing join —
 * it composes already-built, already-cached primitives:
 *   - server/entityGraph.ts's Everything Graph (identity + cross-layer
 *     neighborhood + ticker linkage + insider filings, all already joined
 *     as insider_of/operates/calls_at edges with provenance)
 *   - server/usaSpending.ts's ticker-resolved USAspending contract feed
 *     (the one genuinely NEW cross-stream join this file adds: contracts
 *     were not previously joined into the graph)
 *   - datacore/sites/strategic_sites.json + the shared haversineKm helper
 *     (server/firesFacilities.ts) for "nearest strategic sites"
 *   - the LOCATION DOSSIER hazard cross-join (research/location_context_engine.md
 *     — "the layer of all layers"): superfund.ts / waterViolators.ts /
 *     quake_history.json / nuclear_tests.json, each queried within radius_km
 *     of the same anchor point and reported RAW (count + nearest N per
 *     category, never a risk score)
 *
 * Pure function over injected data (mirrors buildGraph's own
 * every-IO-source-overridable test convention) — no fs/network access
 * here, so tests run against small deterministic fixtures.
 */
import type { EverythingGraph, GraphNode, GraphEdge, SiteRow } from "./entityGraph";
import { resolveEntityId, neighborhood } from "./entityGraph";
import type { ContractTxn } from "./usaSpending";
import { haversineKm } from "./firesFacilities";
import type { SuperfundSite } from "./superfund";
import type { WaterViolator } from "./waterViolators";
import sitesJson from "../datacore/sites/strategic_sites.json";

export const NEAREST_SITES_CAP = 5;
export const CONTRACTS_CAP = 10;
export const DEFAULT_HOPS = 2;

// LOCATION DOSSIER hazard cross-join (research/location_context_engine.md's
// "layer of all layers" — queued NEXT after the hazards-wave sessions that
// shipped superfund.ts/waterViolators.ts/quake_history.json/nuclear_tests.json
// as standalone RAW layers). Each hazard category is capped independently so
// one dense category (25k water violators) never crowds out a sparse one
// (1,840 superfund sites) in the response. RAW/FACTUAL only — this reports
// "N records within R km", never a risk score (that would be an unvalidated
// SIGNAL — see the file's honesty rails).
export const HAZARD_RADIUS_KM_DEFAULT = 50;
export const HAZARD_RADIUS_KM_MAX = 200;
export const HAZARD_CAP = 10;

export interface QuakeRow { d: string; pl: string; lat: number; lon: number; dep?: number | null; m?: number }
export interface NuclearTestRow { d: string; c: string; n: string; lat: number; lon: number; kt?: number | null }

export interface HazardHit {
  id: string;
  label: string;
  km: number;
  detail: Record<string, any>;
}

export interface HazardSection {
  radius_km: number;
  total_within: number;
  capped: boolean;
  hits: HazardHit[];
  /** false when the underlying layer's cache hasn't finished its first
   *  load yet (superfund/water-violators poll asynchronously on boot) —
   *  total_within is honestly 0 in that case, not "confirmed clear". */
  ready: boolean;
}

export interface DossierHazards {
  radius_km: number;
  superfund: HazardSection;
  water_violators: HazardSection;
  quakes: HazardSection;
  nuclear_tests: HazardSection;
  caveat: string;
}

export interface DossierParams {
  entity?: string | null;
  lat?: number | null;
  lon?: number | null;
  hops?: number;
  radius_km?: number;
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
  hazards: DossierHazards | null;
  caveat: string;
}

/** Sort every item in `items` by distance from the anchor, keep the ones
 *  inside `radiusKm`, cap the returned list at HAZARD_CAP, and report the
 *  true within-radius count separately so a capped list is never mistaken
 *  for the whole picture. Pure — no I/O. */
function nearbyHazards<T>(
  items: T[] | null, anchorLat: number, anchorLon: number, radiusKm: number,
  coordsOf: (item: T) => { lat: number; lon: number },
  toHit: (item: T, km: number) => HazardHit,
): HazardSection {
  if (!items) return { radius_km: radiusKm, total_within: 0, capped: false, hits: [], ready: false };
  const within: HazardHit[] = [];
  for (const item of items) {
    const { lat, lon } = coordsOf(item);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    const km = haversineKm(anchorLat, anchorLon, lat, lon);
    if (km <= radiusKm) within.push(toHit(item, km));
  }
  within.sort((a, b) => a.km - b.km);
  return {
    radius_km: radiusKm,
    total_within: within.length,
    capped: within.length > HAZARD_CAP,
    hits: within.slice(0, HAZARD_CAP),
    ready: true,
  };
}

const tickerOf = (n: GraphNode): string | null => {
  if (n.type !== "company") return null;
  if (n.id.startsWith("company:cik:")) return null; // CIK-fallback id — no real ticker known
  return n.id.slice("company:".length);
};

export function buildDossier(
  graph: EverythingGraph | null,
  params: DossierParams,
  opts: {
    contracts?: ContractTxn[] | null;
    sites?: SiteRow[];
    superfund?: SuperfundSite[] | null;
    waterViolators?: WaterViolator[] | null;
    quakes?: QuakeRow[] | null;
    nuclearTests?: NuclearTestRow[] | null;
  } = {},
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

  let hazards: DossierHazards | null = null;
  if (anchorLat != null && anchorLon != null && Number.isFinite(anchorLat) && Number.isFinite(anchorLon)) {
    const radiusKm = Math.max(1, Math.min(HAZARD_RADIUS_KM_MAX, params.radius_km ?? HAZARD_RADIUS_KM_DEFAULT));
    const anchor = { lat: anchorLat, lon: anchorLon };
    hazards = {
      radius_km: radiusKm,
      superfund: nearbyHazards(opts.superfund ?? null, anchor.lat, anchor.lon, radiusKm,
        (s) => ({ lat: s.lat, lon: s.lon }),
        (s, km) => ({ id: s.epa_id ?? s.name, label: s.name, km: Math.round(km * 10) / 10,
          detail: { status: s.status, hrs_score: s.hrs_score, listed: s.listed } })),
      water_violators: nearbyHazards(opts.waterViolators ?? null, anchor.lat, anchor.lon, radiusKm,
        (v) => ({ lat: v.lat, lon: v.lon }),
        (v, km) => ({ id: v.id, label: v.name, km: Math.round(km * 10) / 10,
          detail: { permit: v.permit, snc: v.snc, qtrs_noncompliant: v.qtrs } })),
      quakes: nearbyHazards(opts.quakes ?? null, anchor.lat, anchor.lon, radiusKm,
        (q) => ({ lat: q.lat, lon: q.lon }),
        (q, km) => ({ id: `${q.d}:${q.pl}`, label: q.pl, km: Math.round(km * 10) / 10,
          detail: { date: q.d, magnitude: q.m ?? null, depth_km: q.dep ?? null } })),
      nuclear_tests: nearbyHazards(opts.nuclearTests ?? null, anchor.lat, anchor.lon, radiusKm,
        (t) => ({ lat: t.lat, lon: t.lon }),
        (t, km) => ({ id: `${t.d}:${t.n}`, label: t.n, km: Math.round(km * 10) / 10,
          detail: { date: t.d, country: t.c, yield_kt: t.kt ?? null } })),
      caveat: "RAW cross-join within radius_km of the clicked point, each category from its own "
        + "already-validated layer (superfund.ts/waterViolators.ts/quake_history.json/nuclear_tests.json — "
        + "see each layer's own route for full source/date/data-quality-gate detail). total_within is the "
        + `true count inside the radius; hits is capped at ${HAZARD_CAP} nearest, capped=true means more `
        + "exist. A count of nearby records is a FACT, not a risk score — no impact/safety claim is made "
        + "or implied here (that would be an unvalidated SIGNAL).",
    };
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
    hazards,
    caveat: "RAW composition — identity/graph from the Everything Graph " +
      "(operates/insider_of/calls_at edges, each with its own source/" +
      "confidence/first_seen/last_seen), contracts from USAspending.gov " +
      "matched by resolved ticker only (never fuzzy; DoD/USACE awards " +
      "publish ~90 days late), nearest_sites by straight-line distance to " +
      `up to ${NEAREST_SITES_CAP} of our ${sites.length}-site imagery-` +
      "verified registry. No predictive claim anywhere in this payload.",
  };
}
