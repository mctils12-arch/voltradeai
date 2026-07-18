import { lazy, memo, Suspense, useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { Layers as LayersIcon, Info, X, Minus, Plane, Ship, MapPin, Satellite, FileText, Zap, TrainFront, Maximize2, Minimize2, Mountain, CloudRain, Thermometer, Wind, Flame, TrendingUp, Share2, Database as DatabaseIcon, Globe as GlobeIcon, Map as FlatMapIcon, MessageSquareText, Moon, CloudFog, Leaf, Droplets, Factory, ChevronLeft, ChevronRight, Clock, ThermometerSun, Activity, Waves, Eye, Scale, Anchor, TreePine, Gauge, Shield } from "lucide-react";
// Static CSS import: without maplibre's stylesheet loaded BEFORE the map
// constructs, maplibre mis-measures the container (300px fallback canvas) and
// its controls render unpositioned. The JS stays dynamically imported below.
import "maplibre-gl/dist/maplibre-gl.css";
import {
  registerIcons, classifyAircraft, classifyVessel, velocityEndpoint, iconDataURL,
  AIRCRAFT_ICON, VESSEL_ICON, SITE_ICON, AIRCRAFT_CLASS_LABEL, VESSEL_CLASS_LABEL,
  POWER_FUEL_ICON, POWER_FUEL_COLOR, POWER_FUEL_LABEL, FIRE_CONFIDENCE_COLOR,
  EIA_FUEL_TO_CANON, EIA_FUEL_LABEL, quakeMagnitudeColor,
  classifyNukeTest, NUKE_CLASS_ICON, NUKE_CLASS_LABEL, NUKE_COUNTRY_COLOR,
  radiationBandColor, RADIATION_BANDS, RADIATION_CPM_COLOR, inesColor, NUKE_FACILITY_COLOR,
  PFAS_COUNT_BANDS,
} from "@/lib/mapIcons";
import { decodePurpose, decodeType, testingAgency, yieldContext, blastRadiusKm } from "@/lib/nukeCodes";
import FilingsView from "./filings";
import EarningsView from "./earnings";
import ShortVolView from "./shortvol";
import AttentionView from "./attention";
import CotView from "./cot";
import GraphView from "./graph";
import StreamsView from "./streams";
import GridStressView from "./gridstress";
// W6 ANALYST pane (console charter): lazy chunk — a closed pane loads no
// analyst code at all (zero-cost-when-off spirit) and never polls.
const AnalystPane = lazy(() => import("@/components/AnalystPane"));
import type { AnalystMapCommand } from "@/components/AnalystPane";
// W3 TIME SCRUBBER (console charter): lazy chunk, same zero-cost-when-off
// spirit — a closed panel loads no code and issues no requests.
const TimeScrubber = lazy(() => import("@/components/TimeScrubber"));
import { mmsiFlag } from "@/lib/mmsiFlag";
import { skyForRenderer } from "@/lib/globeAtmosphere";
import {
  OCEAN_BASEMAP_SOURCE_ID, OCEAN_BASEMAP_LAYER_ID,
  oceanBasemapSource, oceanBasemapFallbackSource, oceanBasemapLayer,
} from "@/lib/oceanBasemap";
// ORBITAL program O2 (research/orbital_program.md): live satellites on the
// globe. GP elements are client-fetched from CelesTrak (the browser is NOT
// firewalled from CelesTrak the way Railway is — charter DATA-PATH SPLIT),
// SGP4 runs off-thread in a Web Worker, and the population draws as
// GPU-instanced points. REAL positions only — SGP4 near-earth + SDP4 deep space
// and are skipped + COUNTED, never fabricated.
import { SatLayer } from "@/lib/orbital/satLayer";
import { fetchGp, fetchSatcat, type GpRecord, type SatcatRecord } from "@/lib/orbital/tle";
// ORBITAL O5-2b (human directive: the 3D rendering shows ON THE WORLD MAP,
// not a side viewer): the followed satellite resolves to a lit, tumbling
// class-representative form drawn at its live position on the globe.
import { SatModelLayer } from "@/lib/orbital/modelLayer";
import { ArcLayer, defaultWallRamp } from "@/lib/orbital/arcLayer";
import { sampleOrbitArc, ARC_GAP } from "@/lib/orbital/orbitArc";
import { selectMiniSats, formsFromSatcat, MINI_MAX_CAM_KM } from "@/lib/orbital/miniSelect";
import type { FormKind } from "@/lib/orbital/model3d";
import { raanColor } from "@/lib/orbital/orbitArc";
import { groupMask, maskCount, applyGroupSentinel, spreadIndices, SAT_GROUPS, collapseStationComplexes, isStationComplex } from "@/lib/orbital/satFind";
import { readSatAt } from "@/lib/orbital/satBuffer";
import { mercatorToSphere } from "@/lib/orbital/occlusion";
import { nightPolygon } from "@/lib/celestial/ephemeris";
import { SatFinder } from "@/components/SatFinder";
import { classFormNamed, formLabel } from "@/lib/orbital/model3d";
import { loadRealModel, realModelLabel } from "@/lib/orbital/realMesh";
import { AIRLINE_PRESETS, applyAirlineFilter } from "@/lib/air/airFilter";
// EARTH TWIN E4-1 (identity before models): SATCAT metadata + the curated
// operator→ticker map turn a clicked point into "small payload, CubeSat-class
// size, owned by X, launched Y" — formatting lives in lib/orbital/identity
// (pure, tested); resolveOperator stays conservative (null = honest unmapped).
import { satelliteIdentityLines, nameStemForOperator, buildNoradIndex } from "@/lib/orbital/identity";
// ORBITAL O5 slice 1 (human directive: click a satellite → it remains in
// focus): pure follow math + the focus-ring geometry live in lib/orbital/
// follow; this file owns the camera easing and the stop conditions.
import { followTarget } from "@/lib/orbital/follow";
// EARTH TWIN E3 (true-altitude aircraft): 3D heading-oriented silhouettes at
// real baro altitude take over from the 2D icons at AIR_3D_MIN_ZOOM.
import { AirLayer, buildAircraftInstances, pickNearestAircraft, pickNearestAircraftScreen, AIR_3D_MIN_ZOOM } from "@/lib/air/airLayer";
// DEAD-RECKONING GLIDE (2026-07-18 "planes stopped moving"): between-poll
// extrapolation along the BROADCAST track/speed, capped then frozen — the
// satellite SMOOTH SKY honesty model applied to the 15s aircraft poll.
import { MAX_AIR_GLIDE_SEC, AIR_GLIDE_2D_MIN_ZOOM, AIR_GLIDE_STEP_MS, glideDegPerSec, airGlideDtSec } from "@/lib/air/airGlide";
// SESSION BREADCRUMBS (2026-07-18 "the data is cut off"): while a plane's
// card is open, each live poll appends its REAL fix so the 3D trail +
// altitude curtain reach the plane's CURRENT position instead of ending at
// the last archived sample (1-5 min behind at cruise).
import { pushCrumb, mergeTrackWithCrumbs, type Crumb, type TrackPoint } from "@/lib/air/breadcrumbs";
import type { SatcatWorkerOutbound } from "@/lib/orbital/satcatWorker";
import type { GpWorkerOutbound } from "@/lib/orbital/gpWorker";
import { resolveOperator } from "@/lib/orbital/entityJoin";
import type { SatWorkerOutbound } from "@/lib/orbital/satWorker";
import { pickNearestSatellite, pickNearestSatelliteScreen, pixelToleranceToMercUnits } from "@/lib/orbital/pick";
import { lonLatToMercator } from "@/lib/orbital/satBuffer";
import { epochAgeDays } from "@/lib/orbital/propagate";
import { siteCoverageReport, coverageQueryAllowed } from "@/lib/orbital/siteQuery";
import { STARLINK_MIN_ELEV_DEG } from "@/lib/orbital/geometry";
// EARTH TWIN A1 (E0-2, research/earth_twin_program.md): camera-altitude LOD
// envelopes — the registry (layers.json v2 `lod` block) declares at which
// camera altitudes a layer exists; the LOD director math lives in @/lib/lod
// (pure, tested). orbital_sats is the first consumer: the population fades
// out near the ground and pauses its worker (zero cost), returning on zoom
// out — a render choice, always reversible, surfaced on-panel, never silent.
import { cameraAltitudeKmFromMap, zoomForCameraAltitudeKm, lodOpacity, type LodEnvelope } from "@/lib/lod";
// EARTH TWIN E2-1 ("drain the ocean" v1): the bathymetry depth palette — one
// source of truth shared by the map's color-relief ramp and the legend chips.
import { BATHYMETRY_STOPS, bathymetryColorRelief } from "@/lib/bathymetry";
// PERF session #2 (user-reported lag/freezes): pure, tested guards for the
// live-points tick pipeline — vector-build gating below visibility,
// count quantization so the render bail engages, redundant-refetch skip.
import { shouldBuildVectors, quantizeLiveCount, fetchFootprint, needsRefetch, type FetchFootprint } from "@/lib/livePoints";
// EARTH TWIN E1: the global time axis — the Time Machine publishes, the
// dated GIBS layers subscribe (one scrubber moves the whole world).
import { getTimeAxis, subscribeTimeAxis, gibsDateForAxis, setTimeAxis, formatAxisInstant } from "@/lib/timeAxis";
// Reliability (BUG 1): single-shot layers (sites, powerplants, boundaries,
// orbital_sats) had no fetch timeout and no retry — one stalled/failed request
// left them spinning or dead until a manual toggle. runResilientLoad adds a hard
// timeout + auto-retry backoff so a transient blip self-heals.
import { runResilientLoad } from "@/lib/resilientLoad";
// worldview_globe.md Phase G2: shared NASA GIBS raster-layer factory. G2a
// (night lights) is the first consumer; G2b-h reuse this same helper.
import { gibsTileUrl, gibsDefaultDate, gibsStepDate, gibsIsLatestAvailable, gibsLatestScanTime } from "@/lib/gibs";
// Reliability (BUG 4): six hand-rolled layers stacked click/hover listeners
// across toggle cycles. attachLayerInteractions binds them with named handlers
// and returns a detach() the effect cleanup calls — no more stacking.
import { attachLayerInteractions } from "@/lib/mapInteractions";
import { formatPortDetail } from "@/lib/portDetail";
import { fmtKm, fmtMetersSmall, fmtMetersPerSec, fmtKmh, fmtCelsius, fmtMeters, getUnits, setUnits, subscribeUnits } from "@/lib/units";
// EARTH TWIN E2 v2 wiring (research/earth_twin_program.md RESUME STATE
// 2026-07-16): GEBCO TID measured-vs-predicted seafloor confidence — the
// decode table, color expression, and legend all derive from the SAME
// datacore/gebco/tid_decode.json (one source of truth, see the module's
// own header for the render caveat on cross-group interpolation).
import { SEAFLOOR_V2_REGIONS, tidConfidenceColorRelief, tidConfidenceLegend, GEBCO_ATTRIBUTION, GEBCO_NOT_FOR_NAVIGATION } from "@/lib/seafloorV2";

// Satellite GP element cache (live-tracking stability). CelesTrak's `active`
// group is ~6.6 MB / ~16k objects and CelesTrak RATE-LIMITS repeated pulls, so
// re-fetching on every Satellites toggle failed into a "retrying" loop. Elements
// change only ~every 2h, so cache them for the session and reuse on toggle —
// one fetch per page load, instant re-enable, no rate-limit. Module-scoped so it
// survives the effect's mount/unmount cycles (lost only on a full page reload).
let orbitalGpCache: { at: number; gp: GpRecord[] } | null = null;
const ORBITAL_GP_TTL_MS = 2 * 60 * 60_000; // 2h — CelesTrak's GP refresh cadence
// EARTH TWIN A1: fallback camera-altitude envelope for orbital_sats when the
// registry entry predates v2 (mid-deploy older registry). The registry's own
// `lod` block is the source of truth and overrides this.
const ORBITAL_LOD_FALLBACK = { camMinKm: 100, fadeBandKm: 150 };
// EARTH TWIN E4-1: SATCAT identity catalog — module cache mirroring
// orbitalGpCache (survives effect unmounts; one download per day at most —
// catalog metadata changes slowly and CelesTrak rate-limits). Fetched in the
// BACKGROUND when the satellites layer enables, never blocking the layer;
// a click before it lands gets an honest "still downloading" line.
// E4-2 (perf): the ~6 MB CSV parse measured ~300 ms main-thread block on
// desktop (~1 s mobile) — fetch+parse now run in a one-shot worker
// (satcatWorker.ts) and the lookup index builds in frame-sized chunks
// (buildNoradIndex); the direct fetch remains only as a degrade-safe
// fallback if Worker construction itself fails.
let satcatByNorad: Map<number, SatcatRecord> | null = null;
let satcatFetchedAt = 0;
let satcatState: "loading" | "ready" | "error" = "error";
let satcatInflight: Promise<void> | null = null;
const SATCAT_TTL_MS = 24 * 60 * 60_000;
// SYMBOLS NOT DOTS (human-directed 2026-07-15): shape code per catalogued
// object type — a dot now MEANS "not yet identified". Index-aligned to gp.
function shapeCodesFromSatcat(gp: GpRecord[], byNorad: Map<number, SatcatRecord>): Float32Array {
  const codes = new Float32Array(gp.length); // default 0 = dot
  for (let i = 0; i < gp.length; i++) {
    const t = byNorad.get(gp[i].noradId)?.objectType;
    codes[i] = t === "PAYLOAD" ? 1 : t === "ROCKET BODY" ? 2 : t === "DEBRIS" ? 3 : 0;
  }
  return codes;
}
// PERF (SCALE queue item 3): GP fetch+parse off the main thread — the 6.6MB
// res.json() + parseGp of ~16k records froze the map 150-500ms at satellite
// enable. Same worker shape as SATCAT below; the resilient-load's abort
// signal terminates the worker (timeout semantics preserved); Worker
// construction failure falls back to the main-thread path (degrade, never
// break).
function fetchGpOffThread(group: string, signal?: AbortSignal): Promise<GpRecord[]> {
  return new Promise((resolve, reject) => {
    let worker: Worker;
    try {
      worker = new Worker(
        new URL("../lib/orbital/gpWorker.ts", import.meta.url),
        { type: "module" },
      );
    } catch {
      fetchGp(group, signal ? ((url: string) => fetch(url, { signal }) as any) : undefined).then(resolve, reject);
      return;
    }
    const done = (fn: () => void) => { try { worker.terminate(); } catch {} fn(); };
    const onAbort = () => done(() => reject(new DOMException("aborted", "AbortError")));
    signal?.addEventListener("abort", onAbort, { once: true });
    worker.onmessage = (ev: MessageEvent<GpWorkerOutbound>) => {
      signal?.removeEventListener("abort", onAbort);
      const m = ev.data;
      if (m.type === "rows") done(() => resolve(m.rows));
      else done(() => reject(new Error(m.message)));
    };
    worker.onerror = () => {
      signal?.removeEventListener("abort", onAbort);
      done(() => reject(new Error("GP worker failed")));
    };
    worker.postMessage({ type: "fetch", group });
  });
}

function fetchSatcatRowsOffThread(): Promise<SatcatRecord[]> {
  return new Promise((resolve, reject) => {
    let worker: Worker;
    try {
      worker = new Worker(
        new URL("../lib/orbital/satcatWorker.ts", import.meta.url),
        { type: "module" },
      );
    } catch (e) {
      // No Worker support / bundler edge: fall back to the main-thread path
      // (slower but functional — degrade, never break).
      fetchSatcat().then(resolve, reject);
      return;
    }
    const done = (fn: () => void) => { try { worker.terminate(); } catch {} fn(); };
    worker.onmessage = (ev: MessageEvent<SatcatWorkerOutbound>) => {
      const m = ev.data;
      if (m.type === "rows") done(() => resolve(m.rows));
      else done(() => reject(new Error(m.message)));
    };
    worker.onerror = () => done(() => reject(new Error("satcat worker failed")));
    worker.postMessage({ type: "fetch" });
  });
}
function ensureSatcat(): Promise<void> {
  if (satcatByNorad && Date.now() - satcatFetchedAt < SATCAT_TTL_MS) return Promise.resolve();
  if (satcatInflight) return satcatInflight;
  satcatState = "loading";
  satcatInflight = fetchSatcatRowsOffThread()
    .then(async (rows) => {
      // An HTTP error page (CelesTrak 403/5xx) parses to [] — NEVER cache
      // that as "ready" or identity stays dead for the whole TTL (review
      // finding, session #1). Empty = failure; the old cache (if any) keeps
      // serving clicks.
      if (!rows.length) throw new Error("empty SATCAT response");
      satcatByNorad = await buildNoradIndex(rows); // chunked — never blocks a frame
      satcatFetchedAt = Date.now();
      satcatState = "ready";
    })
    .catch(() => { satcatState = satcatByNorad ? "ready" : "error"; }) // honest absent; stale-but-real cache still counts
    .finally(() => { satcatInflight = null; });
  return satcatInflight;
}
// Baked-in build version — compared against the registry's server_version
// to detect open-tab skew (old bundle + fresh registry = layer rows the
// bundle has no wiring for; the 2026-07-04 production toggle desync).
import { version as CLIENT_VERSION } from "../../../package.json";

/**
 * /data — the data-intelligence map (v2).
 *
 * DESIGN.md-governed: full-viewport at every width, collapsible controls,
 * alive on first load, designed error/stale/partial-coverage states, theme
 * tokens only, PERFORMANCE BUDGET (WebGL symbol layers — no per-marker DOM;
 * smooth at 10k+ features; stale-with-timestamp beats spinner).
 * Verified by `npm run visual` (layout + perf checks) at 390/768/1440.
 *
 * SPINOUT-READY rules: overlay data flows only through /api/data/* (base
 * imagery tiles are the documented scoped exception). RAW vs SIGNAL labels.
 */

interface LayerMeta {
  id: string;
  name: string;
  kind: "raw" | "signal";
  status: "live" | "awaiting_key" | "planned" | "down";
  source: string;
  description: string;
  // registry-native panel placement + cost budget (GIP Part 4 UI SCALABILITY
  // item, BUILD ORDER 4 #2): optional so the visual-harness fixture (and any
  // older cached registry mid-deploy) keeps working via the LAYER_GROUP/
  // "light" fallbacks below — additive fields, no breaking migration.
  group?: string;
  costTier?: "light" | "moderate" | "heavy";
}

type RuntimeStatus = "off" | "loading" | "active" | "error" | "awaiting_key";

interface Detail {
  kind: "site" | "aircraft" | "vessel" | "powerplant" | "substation" | "transmission" | "train" | "fire" | "gauge" | "alert" | "satellite" | "coverage" | "quake" | "buoy" | "place" | "superfund" | "nuketest" | "waterviolator" | "pfas" | "radiation" | "nukeaccident" | "nukefacility" | "port" | "celestial" | "military_installation";
  title: string;
  subtitle: string;
  body: string;
  /** Optional external source link shown in the card footer (e.g. the
   *  military-installations source_url — its citable provenance). */
  sourceUrl?: string;
  trailId?: string;      // archive id for the trail (aircraft icao24 / mmsi)
  trailKind?: "aircraft" | "vessels" | "trains";
  trailNote?: string;
  /** Epoch seconds of the newest archived position — drives the freshness
   *  chip so trail gaps (coverage/sampling) are distinguishable from a
   *  stale feed ([REPAIR 2026-07-05]: trails were a static snapshot). */
  trailLastT?: number;
  /** FAA-registry identity line (entity spine, exact Mode S hex match) —
   *  arrives async after the card opens; absent for non-US hexes. */
  owner?: string;
  /** Everything Graph R1: 7-day cross-stream events + own-archive traffic
   *  density near a strategic site — arrives async after the card opens. */
  timeline?: {
    events: Array<{ t: string; kind: string; label: string; severity?: string | null; value?: number | null }>;
    density: Record<string, { a: number; v: number }>;
  };
  /** External profile/photo pages — LINK OUT only, never embedded
   *  (third-party photo copyright). */
  links?: { label: string; href: string }[];
  /** ENTITY DOSSIER v2 (ANALYST CONSOLE W5) match key — NOT trailId/title
   *  (those are reused by other async enrichments above); a fresh key per
   *  click means a rapid re-click can never let a stale dossier response
   *  land on the wrong card. */
  dossierKey?: string;
  dossier?: DossierPayload;
}

/** Mirrors server/dossier.ts's DossierResult (loosely typed — this is a
 *  display payload, not a contract either side needs to enforce). */
interface DossierPayload {
  identity: { id: string; type: string; label: string; attrs: Record<string, any> } | null;
  graph: {
    nodes: Array<{ id: string; type: string; label: string; attrs: Record<string, any> }>;
    edges: Array<{ type: string; from: string; to: string; confidence: string; attrs: Record<string, any> }>;
  } | null;
  contracts: Array<{ r: string | null; tkr: string | null; amt: number; ag: string | null; rt: string }>;
  contracts_capped: boolean;
  nearest_sites: Array<{ id: string; name: string; category: string; km: number }>;
  hazards: {
    radius_km: number;
    superfund: DossierHazardSection;
    water_violators: DossierHazardSection;
    pfas: DossierHazardSection;
    quakes: DossierHazardSection;
    nuclear_tests: DossierHazardSection;
    flood_zone: DossierFloodZone;
  } | null;
}

/** Mirrors server/dossier.ts's HazardSection. */
interface DossierHazardSection {
  total_within: number;
  capped: boolean;
  ready: boolean;
  hits: Array<{ id: string; label: string; km: number; detail: Record<string, any> }>;
}

/** Mirrors server/femaFlood.ts's FloodZoneResult — a point-in-polygon
 *  lookup AT the anchor, not a radius list (see DossierHazardSection
 *  above for the "N nearby" shape this is deliberately NOT). */
interface DossierFloodZone {
  zone: string | null;
  subtype: string | null;
  sfha: boolean | null;
  base_flood_elevation_ft: number | null;
  meaning: string | null;
  source_citation: string | null;
  ready: boolean;
}

const IMAGERY_TILES =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const IMAGERY_ATTRIB = "© Esri, Maxar, Earthstar Geographics";

// ── W1 GLOBE MODE (ANALYST CONSOLE charter, 2026-07-07) ──
// MapLibre v5 native globe projection is the default view; the library
// itself interpolates globe→mercator as you zoom in, so close-in work is
// unaffected. The choice is a lasting preference (localStorage, unlike the
// per-session fullscreen flag). Read by both the map bootstrap (initial
// style) and the toggle state so the two can never disagree.
const GLOBE_PREF_KEY = "vt-map-globe";
const readGlobePref = (): boolean => {
  try {
    const v = window.localStorage.getItem(GLOBE_PREF_KEY);
    if (v === "1") return true;
    if (v === "0") return false;
  } catch {}
  return true; // charter default: the cinematic globe
};

// Harness kill switch (v2.4 ZERO-COST-WHEN-OFF assertion): with
// vt-layers-all-off set, only the base imagery mounts — the harness then
// asserts NO layer-data API calls fire and measures interactive time.
const HIST_MIN_YEAR = 1900;
const HIST_MAX_YEAR = 2026;
const ALL_OFF = typeof window !== "undefined" && window.sessionStorage?.getItem("vt-layers-all-off") === "1";
const DEFAULT_ON: Record<string, boolean> = ALL_OFF
  ? { imagery: true }
  : { imagery: true, places: true, aircraft: true, sites: true, insider: true, earnings: true, shortvol: true, attention: true, cot: true, powerplants: true, trains: true, shadowstats: true, portdwell: true, graph: true };

// Layer panel v2 (2026-07-04): with 7+ layers the flat list stopped scaling —
// collapsible groups keep the panel scannable as layers keep arriving.
const PANEL_GROUPS = [
  { id: "base", label: "Base" },
  { id: "live", label: "Live tracking" },
  { id: "facilities", label: "Facilities" },
  { id: "hazards", label: "Hazards & environment" },
  { id: "grid", label: "Power grid — US (by state)" },
  { id: "grid_ca", label: "Power grid — Canada (by province)" },
  { id: "environmental", label: "Environmental" },
  { id: "filings", label: "Filings & flows" },
  { id: "graph", label: "Everything Graph" },
  { id: "signals", label: "Signals — coming soon" },
] as const;
// SMAP L4 root-zone soil moisture lands ~6 days behind real time; default the
// scrubber a conservative 7 days back so it never opens on a guaranteed-blank
// tile (verified live 2026-07-08: 07-01 and 07-02 both carry data, 07-07 does
// not). Passed to gibsDefaultDate/gibsIsLatestAvailable as latencyDays.
const SOIL_LATENCY_DAYS = 7;
const LAYER_GROUP: Record<string, string> = {
  imagery: "base", terrain: "base", seafloor: "base", seafloor_confidence: "base", daynight: "base", weather: "base",
  weather_temp: "base", weather_wind: "base", boundaries: "base", boundaries_admin1: "base", places: "base",
  celestial_paths: "base",
  aircraft: "live", vessels: "live", trains: "live",
  sites: "facilities", powerplants: "facilities", nukefacilities: "facilities", military_installations: "facilities",
  superfund: "hazards", nucleartests: "hazards", quakehistory: "hazards", waterviolators: "hazards",
  radiation: "hazards", nukeaccidents: "hazards", floodzones: "hazards", pfas: "hazards",
  fires: "environmental", surfacewater: "environmental", forest: "environmental",
  nightlights: "environmental",
  aerosol: "environmental",
  vegetation: "environmental",
  soilmoisture: "environmental",
  no2: "environmental",
  firetemp: "environmental",
  rivergauges: "environmental",
  alerts: "environmental",
  earthquakes: "environmental",
  buoys: "environmental",
  biomass: "environmental",
  insider: "filings", earnings: "filings", shortvol: "filings", attention: "filings", cot: "filings", shadowstats: "filings", portdwell: "filings",
  graph: "graph",
  powergrid: "facilities",
  powergrid_hifld: "facilities", powergrid_hifld_sub: "facilities", powergrid_hifld_plants: "facilities",
  powergrid_al: "grid", powergrid_ak: "grid", powergrid_az: "grid", powergrid_ar: "grid", powergrid_ca: "grid", powergrid_co: "grid",
  powergrid_ct: "grid", powergrid_de: "grid", powergrid_dc: "grid", powergrid_fl: "grid", powergrid_ga: "grid", powergrid_hi: "grid",
  powergrid_id: "grid", powergrid_il: "grid", powergrid_in: "grid", powergrid_ia: "grid", powergrid_ks: "grid", powergrid_ky: "grid",
  powergrid_la: "grid", powergrid_me: "grid", powergrid_md: "grid", powergrid_ma: "grid", powergrid_mi: "grid", powergrid_mn: "grid",
  powergrid_ms: "grid", powergrid_mo: "grid", powergrid_mt: "grid", powergrid_ne: "grid", powergrid_nv: "grid", powergrid_nh: "grid",
  powergrid_nj: "grid", powergrid_nm: "grid", powergrid_ny: "grid", powergrid_nc: "grid", powergrid_nd: "grid", powergrid_oh: "grid",
  powergrid_ok: "grid", powergrid_or: "grid", powergrid_pa: "grid", powergrid_ri: "grid", powergrid_sc: "grid", powergrid_sd: "grid",
  powergrid_tn: "grid", powergrid_tx: "grid", powergrid_ut: "grid", powergrid_vt: "grid", powergrid_va: "grid", powergrid_wa: "grid",
  powergrid_wv: "grid", powergrid_wi: "grid", powergrid_wy: "grid",
  powergrid_canada: "facilities",
  powergrid_ca_ab: "grid_ca", powergrid_ca_bc: "grid_ca", powergrid_ca_mb: "grid_ca",
  powergrid_ca_nb: "grid_ca", powergrid_ca_nl: "grid_ca", powergrid_ca_nt: "grid_ca",
  powergrid_ca_ns: "grid_ca", powergrid_ca_nu: "grid_ca", powergrid_ca_on: "grid_ca",
  powergrid_ca_pe: "grid_ca", powergrid_ca_qc: "grid_ca", powergrid_ca_sk: "grid_ca",
  powergrid_ca_yt: "grid_ca",
  orbital_sats: "live",
};

// Military-installations operator-nation palette (human directive: colour by
// operator nation, ONE MUTED reference palette — NO red / threat styling,
// this is reference geography not a threat board). Named nations get a fixed
// muted hue; everything else hashes into the same muted ramp so the map reads
// as an atlas, never an ops board. Unattributed (offshore/unresolved) = grey.
const MILITARY_NATION_PALETTE: Record<string, string> = {
  "United States of America": "#6f8fb0", // muted steel blue
  "China": "#9a8bb0",                    // muted mauve
  "Russia": "#8a9aa8",                   // muted slate
  "United Kingdom": "#7fa08f",           // muted sage
  "Taiwan": "#b0a074",                   // muted ochre
  "India": "#a68f7a",                    // muted tan
  "France": "#7d94ad",                   // muted blue-grey
  "Germany": "#8fa9a0",                  // muted teal-grey
  "Japan": "#a89bab",                    // muted lilac-grey
};
const MILITARY_NATION_RAMP = ["#8a94a0", "#8f9c88", "#9a8f9c", "#94a0a6", "#a09488", "#889aa0", "#9c9488"];
function militaryNationTint(nation?: string | null): string {
  if (!nation) return "#7c8794"; // unattributed — neutral grey
  if (MILITARY_NATION_PALETTE[nation]) return MILITARY_NATION_PALETTE[nation];
  let h = 0;
  for (let i = 0; i < nation.length; i++) h = (h * 31 + nation.charCodeAt(i)) & 0xffff;
  return MILITARY_NATION_RAMP[h % MILITARY_NATION_RAMP.length];
}

// GRID VISION national rollout — one OSM-derived PMTiles per state (built by
// scripts/build_power_tiles.sh, committed under client/public/tiles/). The
// `powergrid` master toggle shows ALL available states; each `powergrid_<code>`
// toggles one. Add a row here + a layers.json entry + drop the .pmtiles in to
// light up a new state. Provenance = osm-verified (raw OSM); ML-detected towers
// layer on top later as a separate provenance tier.
const POWER_STATES = [
  { code: "al", name: "Alabama", file: "power_al.pmtiles" },
  { code: "ak", name: "Alaska", file: "power_ak.pmtiles" },
  { code: "az", name: "Arizona", file: "power_az.pmtiles" },
  { code: "ar", name: "Arkansas", file: "power_ar.pmtiles" },
  { code: "ca", name: "California", file: "power_ca.pmtiles" },
  { code: "co", name: "Colorado", file: "power_co.pmtiles" },
  { code: "ct", name: "Connecticut", file: "power_ct.pmtiles" },
  { code: "de", name: "Delaware", file: "power_de.pmtiles" },
  { code: "dc", name: "District of Columbia", file: "power_dc.pmtiles" },
  { code: "fl", name: "Florida", file: "power_fl.pmtiles" },
  { code: "ga", name: "Georgia", file: "power_ga.pmtiles" },
  { code: "hi", name: "Hawaii", file: "power_hi.pmtiles" },
  { code: "id", name: "Idaho", file: "power_id.pmtiles" },
  { code: "il", name: "Illinois", file: "power_il.pmtiles" },
  { code: "in", name: "Indiana", file: "power_in.pmtiles" },
  { code: "ia", name: "Iowa", file: "power_ia.pmtiles" },
  { code: "ks", name: "Kansas", file: "power_ks.pmtiles" },
  { code: "ky", name: "Kentucky", file: "power_ky.pmtiles" },
  { code: "la", name: "Louisiana", file: "power_la.pmtiles" },
  { code: "me", name: "Maine", file: "power_me.pmtiles" },
  { code: "md", name: "Maryland", file: "power_md.pmtiles" },
  { code: "ma", name: "Massachusetts", file: "power_ma.pmtiles" },
  { code: "mi", name: "Michigan", file: "power_mi.pmtiles" },
  { code: "mn", name: "Minnesota", file: "power_mn.pmtiles" },
  { code: "ms", name: "Mississippi", file: "power_ms.pmtiles" },
  { code: "mo", name: "Missouri", file: "power_mo.pmtiles" },
  { code: "mt", name: "Montana", file: "power_mt.pmtiles" },
  { code: "ne", name: "Nebraska", file: "power_ne.pmtiles" },
  { code: "nv", name: "Nevada", file: "power_nv.pmtiles" },
  { code: "nh", name: "New Hampshire", file: "power_nh.pmtiles" },
  { code: "nj", name: "New Jersey", file: "power_nj.pmtiles" },
  { code: "nm", name: "New Mexico", file: "power_nm.pmtiles" },
  { code: "ny", name: "New York", file: "power_ny.pmtiles" },
  { code: "nc", name: "North Carolina", file: "power_nc.pmtiles" },
  { code: "nd", name: "North Dakota", file: "power_nd.pmtiles" },
  { code: "oh", name: "Ohio", file: "power_oh.pmtiles" },
  { code: "ok", name: "Oklahoma", file: "power_ok.pmtiles" },
  { code: "or", name: "Oregon", file: "power_or.pmtiles" },
  { code: "pa", name: "Pennsylvania", file: "power_pa.pmtiles" },
  { code: "ri", name: "Rhode Island", file: "power_ri.pmtiles" },
  { code: "sc", name: "South Carolina", file: "power_sc.pmtiles" },
  { code: "sd", name: "South Dakota", file: "power_sd.pmtiles" },
  { code: "tn", name: "Tennessee", file: "power_tn.pmtiles" },
  { code: "tx", name: "Texas", file: "power_tx.pmtiles" },
  { code: "ut", name: "Utah", file: "power_ut.pmtiles" },
  { code: "vt", name: "Vermont", file: "power_vt.pmtiles" },
  { code: "va", name: "Virginia", file: "power_va.pmtiles" },
  { code: "wa", name: "Washington", file: "power_wa.pmtiles" },
  { code: "wv", name: "West Virginia", file: "power_wv.pmtiles" },
  { code: "wi", name: "Wisconsin", file: "power_wi.pmtiles" },
  { code: "wy", name: "Wyoming", file: "power_wy.pmtiles" },
] as const;
// Canada — OSM community power grid per province/territory (ODbL). Codes are
// prefixed "ca_" so they never collide with US state codes (US California is
// already "ca" / power_ca.pmtiles). Same voltage-classed rendering as the
// US states; national roll-up served from power_canada.pmtiles.
const CANADA_PROVINCES = [
  { code: "ca_ab", name: "Alberta", file: "power_ca_ab.pmtiles" },
  { code: "ca_bc", name: "British Columbia", file: "power_ca_bc.pmtiles" },
  { code: "ca_mb", name: "Manitoba", file: "power_ca_mb.pmtiles" },
  { code: "ca_nb", name: "New Brunswick", file: "power_ca_nb.pmtiles" },
  { code: "ca_nl", name: "Newfoundland and Labrador", file: "power_ca_nl.pmtiles" },
  { code: "ca_nt", name: "Northwest Territories", file: "power_ca_nt.pmtiles" },
  { code: "ca_ns", name: "Nova Scotia", file: "power_ca_ns.pmtiles" },
  { code: "ca_nu", name: "Nunavut", file: "power_ca_nu.pmtiles" },
  { code: "ca_on", name: "Ontario", file: "power_ca_on.pmtiles" },
  { code: "ca_pe", name: "Prince Edward Island", file: "power_ca_pe.pmtiles" },
  { code: "ca_qc", name: "Quebec", file: "power_ca_qc.pmtiles" },
  { code: "ca_sk", name: "Saskatchewan", file: "power_ca_sk.pmtiles" },
  { code: "ca_yt", name: "Yukon", file: "power_ca_yt.pmtiles" },
] as const;
// [REPAIR R15 2026-07-07] LAYER_GROUP doubles as the CLIENT-WIRED
// declaration: the panel marks any live registry id missing from it
// "reload to enable" (the honest mid-deploy state) — so an id that IS
// wired but missing here renders PERMANENTLY un-enableable (powergrid
// shipped v1.0.166 with wiring but no entry; stuck for a day). The
// wiring ratchet (server/layersWiring.test.ts) now fails CI whenever a
// non-signal/non-planned registry layer is absent from this map.
// registry-native grouping (BUILD ORDER 4 #2): datacore/layers.json now
// carries `group` per layer directly — a future pipeline can slot a new
// layer into a panel group by editing the registry alone, no datamap.tsx
// code change required. LAYER_GROUP above stays as the fallback for the
// visual-harness fixture and any registry response from an older deploy
// mid-rollout that predates the field.
const groupOf = (l: LayerMeta): string =>
  l.group || (l.kind === "signal" || l.status === "planned" ? "signals" : LAYER_GROUP[l.id] || "live");

// per-layer relative client cost (network + render), registry-native
// (BUILD ORDER 4 #2 cost-budget item). Unlabeled layers (fixture, stale
// registry) default to "light" — never overclaims load.
const COST_WEIGHT: Record<string, number> = { light: 1, moderate: 2, heavy: 4 };
const costWeightOf = (l: LayerMeta): number => COST_WEIGHT[l.costTier || "light"] ?? 1;
// groups shown expanded by default; any group id NOT in this set (including
// every group a future registry update introduces) defaults COLLAPSED —
// inverted from the old hardcoded collapsed-list so growth is safe by
// construction instead of by remembering to update a second list.
const OPEN_GROUPS_BY_DEFAULT = new Set(["base", "live"]);
// per-group DOM row cap (BUILD ORDER 4 #2 panel-scale item): an open group
// renders at most this many rows before collapsing the rest behind a
// "show all" control — bounds panel DOM cost per group regardless of how
// large the registry grows, without needing a windowed-scroll virtualizer.
// Verified against today's registry: no group exceeds this (zero visual
// change); see scripts/visual_check.mjs's layer-scale battery for the
// measured behavior at 50/100/200 synthetic layers.
const GROUP_ROW_CAP = 12;

// altitude → tint for aircraft icons (SDF icon-color)
const ALT_COLOR: any = ["case",
  ["get", "ground"], "#6680a0",
  ["<", ["coalesce", ["get", "alt"], 99999], 3000], "#fbb24c",
  "#4d9fff"];

const VESSEL_COLOR: Record<string, string> = {
  tanker: "#fbb24c", cargo: "#4ade80", passenger: "#c084fc",
  fishing: "#7cc4ff", tug: "#b3c2d8", other: "#4ade80",
};


// Legend entry that renders THE ACTUAL registry shape the map draws
// (iconDataURL rasterizes the same ImageData registerIcons feeds maplibre).
// data-vt-icon is the parity hook the harness checks in both directions.
// Debug fps readout (sprint W2 2026-07-17): rAF cadence over 1s windows —
// measures the page's real achievable frame rate including main-thread jank.
// Rendered ONLY behind the debug flag (?fps=1 or localStorage vt-fps=1), so
// the rAF loop never runs for normal users (zero-cost-when-off).
function FpsChip() {
  const [fps, setFps] = useState(0);
  useEffect(() => {
    let frames = 0, last = performance.now(), raf = 0;
    const loop = (t: number) => {
      frames++;
      if (t - last >= 1000) { setFps(Math.round((frames * 1000) / (t - last))); frames = 0; last = t; }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);
  return <div className="vt-fps-chip" aria-label="Frames per second (debug)">{fps} fps</div>;
}

function LegendIcon({ icon, color, label }: { icon: string; color: string; label: string }) {
  return (
    <span className="vt-legend-item" data-vt-icon={icon}>
      <img src={iconDataURL(icon, color)} width={15} height={15} alt="" aria-hidden />
      {label}
    </span>
  );
}

// LEGEND v3 (legend directive 2026-07-04): symbol entries render the SAME
// registry shapes the map draws (iconDataURL — one shared icon source;
// DESIGN.md legend rule). Sections mirror the panel groups, entries appear
// ONLY while their layer is on, and the block collapses as one unit so it
// never fights the panel for space. Color-only chips are color MEANINGS
// (altitude tints, raster ramps), not symbols — chips by design.
//
// React memo boundary (SCALE program S1(d), queued since 2026-07-15): this
// section depends only on layer-toggle/date/sat-selector state, never on
// live position ticks (aircraft/vessel/satellite WebSocket updates repaint
// the map every second-ish but never touch these props) — wrapped in
// React.memo so those high-frequency parent re-renders no longer force this
// ~330-line JSX tree to re-diff. All props below are either primitive state,
// stable useCallback/useState-setter identities, or stable ref objects, so
// memo's default shallow comparison is correct without a custom comparator.
interface LegendPanelProps {
  legendOpen: boolean;
  setLegendOpen: React.Dispatch<React.SetStateAction<boolean>>;
  enabled: Record<string, boolean>;
  airFilter: string | null;
  setAirFilter: React.Dispatch<React.SetStateAction<string | null>>;
  nightlightsDate: string;
  aerosolDate: string;
  vegetationDate: string;
  soilmoistureDate: string;
  no2Date: string;
  firetempScanTime: string | null;
  tempUnitF: boolean;
  windArrows: boolean;
  orbitalGpRef: React.RefObject<GpRecord[] | null>;
  gpVersion: number;
  satGroup: string | null;
  satGroupCount: number | null;
  satGroupOrbits: boolean;
  satArcInfo: { shown: number; total: number } | null;
  applySatGroup: (key: string | null) => void;
  setSatGroupOrbits: React.Dispatch<React.SetStateAction<boolean>>;
  /** parent's findSat — ends the old focus/card, clears an excluding group
   *  filter, then focuses (one path for search hits + group members). */
  onFindSat: (index: number) => void;
  seafloorConfShares: Record<string, Record<string, number>>;
}

const LegendPanel = memo(function LegendPanel({
  legendOpen, setLegendOpen, enabled, airFilter, setAirFilter,
  nightlightsDate, aerosolDate, vegetationDate, soilmoistureDate, no2Date,
  firetempScanTime, tempUnitF, windArrows, orbitalGpRef, gpVersion,
  satGroup, satGroupCount, satGroupOrbits, satArcInfo, applySatGroup,
  setSatGroupOrbits, onFindSat,
  seafloorConfShares,
}: LegendPanelProps) {
  return (
    <div className="vt-legend" data-vt-legend>
      <button className="vt-legend-head" aria-expanded={legendOpen}
              onClick={() => setLegendOpen((v) => !v)}>
        <span className={`vt-layer-group-chev${legendOpen ? "" : " closed"}`}>▾</span>
        <span>Legend</span>
      </button>
      {legendOpen && (
        <div className="vt-legend-body">
          {(enabled.aircraft || enabled.vessels || enabled.trains) && (
            <div className="vt-legend-sec">
              <div className="vt-legend-sec-head">Live Tracking</div>
              <div className="vt-legend-items">
                {enabled.aircraft && (
                  <>
                    <LegendIcon icon={AIRCRAFT_ICON.jet} color="#4d9fff" label="Jet" />
                    <LegendIcon icon={AIRCRAFT_ICON.turboprop} color="#4d9fff" label="Turboprop / Piston" />
                    <LegendIcon icon={AIRCRAFT_ICON.helicopter} color="#4d9fff" label="Helicopter" />
                    <LegendIcon icon={AIRCRAFT_ICON.unknown} color="#4d9fff" label="Unclassified Aircraft" />
                    <span className="vt-legend-chip"><i style={{ background: "#4d9fff" }} /> Cruise</span>
                    <span className="vt-legend-chip"><i style={{ background: "#fbb24c" }} /> Low Altitude</span>
                    <span className="vt-legend-chip"><i style={{ background: "#6680a0" }} /> On Ground</span>
                    <span className="vt-legend-note">zoom in (z8+): planes become 3D silhouettes at their real altitude, with a drop-line to the ground — shape = broadcast aircraft class (light / airliner / heavy / fast / rotor), color = altitude band; tilt the map to see them fly above the terrain. Click a plane: its archived track draws in 3D with a solid altitude curtain to the ground, colored by the same bands</span>
                    {/* O6-4: operator filter — broadcast callsign prefixes (ICAO
                        telephony codes), one airline's sky at a time */}
                    <div className="vt-satfinder-groups" style={{ width: "100%", marginTop: 4 }}>
                      {AIRLINE_PRESETS.map((p) => (
                        <button key={p.key}
                                className={`vt-satfinder-chip${airFilter === p.prefix ? " vt-satfinder-chip-on" : ""}`}
                                onClick={() => setAirFilter(airFilter === p.prefix ? null : p.prefix)}>
                          {p.label}
                        </button>
                      ))}
                    </div>
                    {airFilter && (
                      <span className="vt-legend-note">showing only callsigns {airFilter}* (broadcast flight IDs — the operator's ICAO code; charters/GA under other callsigns won't match)</span>
                    )}
                  </>
                )}
                {enabled.vessels && (
                  <>
                    <LegendIcon icon={VESSEL_ICON.tanker} color={VESSEL_COLOR.tanker} label="Tanker" />
                    <LegendIcon icon={VESSEL_ICON.cargo} color={VESSEL_COLOR.cargo} label="Cargo" />
                    <LegendIcon icon={VESSEL_ICON.passenger} color={VESSEL_COLOR.passenger} label="Passenger" />
                    <LegendIcon icon={VESSEL_ICON.fishing} color={VESSEL_COLOR.fishing} label="Fishing / Tug / Small" />
                  </>
                )}
                {enabled.trains && <LegendIcon icon="vt-train" color="#2dd4bf" label="Train" />}
              </div>
            </div>
          )}
          {(enabled.sites || enabled.powerplants || enabled.powergrid_hifld_plants || enabled.powergrid_hifld_sub) && (
            <div className="vt-legend-sec">
              <div className="vt-legend-sec-head">Facilities</div>
              <div className="vt-legend-items">
                {enabled.sites && (
                  <>
                    <LegendIcon icon={SITE_ICON.port} color="#4ade80" label="Port" />
                    <LegendIcon icon={SITE_ICON.tank_farm} color="#fbb24c" label="Tank Farm" />
                    <LegendIcon icon={SITE_ICON.steel_mill} color="#ff5a6e" label="Steel Mill" />
                  </>
                )}
                {(enabled.powerplants || enabled.powergrid_hifld_plants) && Object.keys(POWER_FUEL_ICON).map((fuel) => (
                  <LegendIcon key={fuel} icon={POWER_FUEL_ICON[fuel]}
                              color={POWER_FUEL_COLOR[fuel]}
                              label={`${POWER_FUEL_LABEL[fuel]} Plant`} />
                ))}
                {enabled.powergrid_hifld_sub && (
                  <LegendIcon icon="vt-substation" color="#fbbf24" label="Substation" />
                )}
              </div>
            </div>
          )}
          {(enabled.fires || enabled.surfacewater || enabled.forest || enabled.nightlights || enabled.aerosol || enabled.vegetation || enabled.soilmoisture || enabled.no2 || enabled.firetemp || enabled.biomass || enabled.rivergauges || enabled.alerts || enabled.earthquakes || enabled.buoys) && (
            <div className="vt-legend-sec">
              <div className="vt-legend-sec-head">Environmental</div>
              <div className="vt-legend-items">
                {enabled.fires && (
                  <>
                    <LegendIcon icon="vt-fire" color={FIRE_CONFIDENCE_COLOR.high} label="Fire — High Confidence" />
                    <LegendIcon icon="vt-fire" color={FIRE_CONFIDENCE_COLOR.nominal} label="Fire — Nominal" />
                    <LegendIcon icon="vt-fire" color={FIRE_CONFIDENCE_COLOR.low} label="Fire — Low Confidence" />
                  </>
                )}
                {enabled.rivergauges && <LegendIcon icon="vt-gauge" color="#4d9fff" label="River Gauge (USGS)" />}
                {enabled.earthquakes && (
                  <>
                    <LegendIcon icon="vt-quake" color="#8bc34a" label="Quake M2.5-4" />
                    <LegendIcon icon="vt-quake" color="#ffd23f" label="Quake M4-5" />
                    <LegendIcon icon="vt-quake" color="#ff8c42" label="Quake M5-6" />
                    <LegendIcon icon="vt-quake" color="#ff3b3b" label="Quake M6+" />
                  </>
                )}
                {enabled.buoys && <LegendIcon icon="vt-buoy" color="#22d3ee" label="Ocean Buoy (NDBC)" />}
                {enabled.alerts && (
                  <>
                    {([["Extreme", "#ff3b3b"], ["Severe", "#ff8c42"], ["Moderate", "#ffd23f"], ["Minor", "#4d9fff"]] as const)
                      .map(([t, c]) => (
                        <span key={t} className="vt-legend-chip"><i style={{ background: c }} /> {t} Alert</span>
                      ))}
                  </>
                )}
                {enabled.surfacewater && (
                  <>
                    {([["Rare", "#ffcccc"], ["Seasonal", "#8683ff"], ["Permanent", "#0000ff"]] as const)
                      .map(([t, c]) => (
                        <span key={t} className="vt-legend-chip"><i style={{ background: c }} /> {t} Water</span>
                      ))}
                    <span className="vt-legend-note">(1984–2021, JRC GSW)</span>
                  </>
                )}
                {enabled.forest && (
                  <>
                    <span className="vt-legend-chip"><i style={{ background: "#2e7d32" }} /> Forest Extent</span>
                    <span className="vt-legend-note">(2020 10m, JRC GFC2020)</span>
                  </>
                )}
                {enabled.nightlights && (
                  <>
                    <span className="vt-legend-chip"><i style={{ background: "#ffe873" }} /> Night Lights Radiance</span>
                    <span className="vt-legend-note">(daily, NASA GIBS/VIIRS — {nightlightsDate})</span>
                  </>
                )}
                {enabled.aerosol && (
                  <>
                    <span className="vt-legend-chip"><i style={{ background: "#c8842a" }} /> Aerosol Optical Depth</span>
                    <span className="vt-legend-note">(daily, NASA GIBS/MODIS — {aerosolDate})</span>
                  </>
                )}
                {enabled.vegetation && (
                  <>
                    <span className="vt-legend-chip"><i style={{ background: "#2e9e4a" }} /> Vegetation (NDVI)</span>
                    <span className="vt-legend-note">(8-day, NASA GIBS/VIIRS — {vegetationDate})</span>
                  </>
                )}
                {enabled.soilmoisture && (
                  <>
                    <span className="vt-legend-chip"><i style={{ background: "#2b6cb0" }} /> Soil Moisture (SMAP)</span>
                    <span className="vt-legend-note">(~6-day lag, NASA GIBS/SMAP — {soilmoistureDate})</span>
                  </>
                )}
                {enabled.no2 && (
                  <>
                    <span className="vt-legend-chip"><i style={{ background: "#e53e3e" }} /> NO₂ (TROPOMI)</span>
                    <span className="vt-legend-note">(daily, NASA GIBS/Sentinel-5P — {no2Date})</span>
                  </>
                )}
                {enabled.firetemp && (
                  <>
                    <span className="vt-legend-chip"><i style={{ background: "#ff7a1a" }} /> Fire/Hotspot Temp (GOES-East)</span>
                    <span className="vt-legend-note">(~10-min, NASA GIBS/ABI — {firetempScanTime ? `${firetempScanTime} UTC` : "latest"})</span>
                  </>
                )}
                {enabled.biomass && (
                  <>
                    <span className="vt-legend-chip"><i style={{ background: "#6b8e23" }} /> Biomass Density (GEDI)</span>
                    <span className="vt-legend-note">(static, 2019-04 to 2023-03 mean, NASA GIBS/GEDI L4B)</span>
                  </>
                )}
              </div>
            </div>
          )}
          {enabled.radiation && (
            <div className="vt-legend-sec">
              <div className="vt-legend-sec-head">Ambient Radiation</div>
              <div className="vt-legend-items">
                <LegendIcon icon="vt-radiation" color="#eef3fb" label="Gamma Monitor (exact location)" />
                <span className="vt-legend-chip"><i style={{ background: "rgba(74,222,128,0.35)", borderRadius: "50%" }} /> US Monitor — City-Area Circle</span>
                {RADIATION_BANDS.map((b) => (
                  <span key={b.label} className="vt-legend-chip"><i style={{ background: b.color }} /> {b.label}</span>
                ))}
                <span className="vt-legend-chip"><i style={{ background: RADIATION_CPM_COLOR }} /> CPM-only Station</span>
                <span className="vt-legend-note">(bands = display buckets, not thresholds; US circles span roughly the city's area — exact addresses not public)</span>
              </div>
            </div>
          )}
          {enabled.pfas && (
            <div className="vt-legend-sec">
              <div className="vt-legend-sec-head">PFAS Drinking-Water Detections</div>
              <div className="vt-legend-items">
                <LegendIcon icon="vt-pfas" color="#eef3fb" label="Water System (service-area centroid)" />
                {PFAS_COUNT_BANDS.map((b) => (
                  <span key={b.label} className="vt-legend-chip"><i style={{ background: b.color }} /> {b.label}</span>
                ))}
                <span className="vt-legend-note">(color = count of distinct PFAS compounds detected, a fact — not a concentration or health-risk tier; EPA UCMR 5, 2023-2025 monitoring)</span>
              </div>
            </div>
          )}
          {enabled.nukefacilities && (
            <div className="vt-legend-sec">
              <div className="vt-legend-sec-head">Nuclear Facilities</div>
              <div className="vt-legend-items">
                <LegendIcon icon="vt-nukefacility" color="#eef3fb" label="Fuel-cycle / Production Site" />
                {Object.entries(NUKE_FACILITY_COLOR).map(([cat, col]) => (
                  <span key={cat} className="vt-legend-chip"><i style={{ background: col }} /> {cat}</span>
                ))}
              </div>
            </div>
          )}
          {enabled.military_installations && (
            <div className="vt-legend-sec">
              <div className="vt-legend-sec-head">Military installations</div>
              <div className="vt-legend-items">
                {["United States of America", "China", "Russia", "United Kingdom", "France", "India", "Other"].map((n) => (
                  <LegendIcon key={n} icon="vt-military"
                    color={n === "Other" ? militaryNationTint("zzz-other") : militaryNationTint(n)}
                    label={n === "United States of America" ? "United States" : n} />
                ))}
                <span className="vt-legend-note">colour = operator nation (reference palette, not a threat board) · shield symbols at low zoom, boundaries at high zoom · Officially published installation locations. Sources: US DoD open data, OpenStreetMap contributors (© OpenStreetMap contributors), and cited government publications. Reference geography only — current as of retrieval date; not operational information.</span>
              </div>
            </div>
          )}
          {enabled.nukeaccidents && (
            <div className="vt-legend-sec">
              <div className="vt-legend-sec-head">Nuclear Accidents</div>
              <div className="vt-legend-items">
                <LegendIcon icon="vt-meltdown" color="#eef3fb" label="Accident / Incident Site" />
                <span className="vt-legend-chip"><i style={{ background: inesColor(7) }} /> INES 6–7 Major/Serious</span>
                <span className="vt-legend-chip"><i style={{ background: inesColor(4) }} /> INES 4–5 Accident</span>
                <span className="vt-legend-chip"><i style={{ background: inesColor(2) }} /> INES 1–3 Incident</span>
                <span className="vt-legend-chip"><i style={{ background: inesColor(null) }} /> Unrated (no INES catalogued)</span>
              </div>
            </div>
          )}
          {enabled.nucleartests && (
            <div className="vt-legend-sec">
              <div className="vt-legend-sec-head">Nuclear Tests</div>
              <div className="vt-legend-items">
                {(Object.keys(NUKE_CLASS_ICON) as Array<keyof typeof NUKE_CLASS_ICON>).map((k) => (
                  <LegendIcon key={k} icon={NUKE_CLASS_ICON[k]} color="#eef3fb" label={NUKE_CLASS_LABEL[k]} />
                ))}
                {Object.entries(NUKE_COUNTRY_COLOR).map(([c, col]) => (
                  <span key={c} className="vt-legend-chip"><i style={{ background: col }} /> {
                    ({ USA: "USA", USSR: "USSR", FRANCE: "France", UK: "UK", CHINA: "China", INDIA: "India", PAKIST: "Pakistan" } as Record<string, string>)[c] || c
                  }</span>
                ))}
                <span className="vt-legend-chip"><i style={{ background: "rgba(253,224,71,0.6)" }} /> 5-psi Blast Estimate Ring</span>
                <span className="vt-legend-note">(ring = Glasstone–Dolan blast estimate, not fallout)</span>
              </div>
            </div>
          )}
          {(enabled.weather_temp || (enabled.weather_wind && windArrows)) && (
            <div className="vt-legend-sec">
              <div className="vt-legend-sec-head">Fields</div>
              <div className="vt-legend-items">
                {enabled.weather_wind && windArrows && (
                  <LegendIcon icon="vt-wind-arrow" color="#eef3fb" label="Wind (Direction + kt)" />
                )}
                {enabled.weather_temp && (
                  <>
                    {([["-40", "#821692"], ["-20", "#208CEC"], ["0", "#23DDDD"],
                       ["10", "#C2FF28"], ["20", "#FFF028"], ["30+", "#FC8014"]] as const)
                      .map(([t, c]) => (
                        <span key={t} className="vt-legend-chip"><i style={{ background: c }} /> {tempUnitF ? `${Math.round(Number(t.replace("+", "")) * 9 / 5 + 32)}${t.includes("+") ? "+" : ""}°F` : `${t}°C`}</span>
                      ))}
                    <span className="vt-legend-note">(approx — amplified for dark basemap)</span>
                  </>
                )}
              </div>
            </div>
          )}
          {enabled["orbital_sats"] && (
            <div className="vt-legend-sec">
              <div className="vt-legend-sec-head">Orbital</div>
              <SatFinder
                gp={orbitalGpRef.current}
                gpVersion={gpVersion}
                activeGroup={satGroup}
                groupCount={satGroupCount}
                orbitsOn={satGroupOrbits}
                arcInfo={satArcInfo}
                onFind={onFindSat}
                onGroup={applySatGroup}
                onOrbits={setSatGroupOrbits}
              />
              <div className="vt-legend-items">
                <span className="vt-legend-chip"><i style={{ background: "#4d9fff" }} /> LEO</span>
                <span className="vt-legend-chip"><i style={{ background: "#ffb840" }} /> MEO</span>
                <span className="vt-legend-chip"><i style={{ background: "#d973ff" }} /> GEO</span>
                <span className="vt-legend-chip">shape = type: ▣ payload · ▮ rocket body · ◆ debris · ● not yet identified</span>
                <span className="vt-legend-note">the FULL catalog live — near-earth SGP4 + deep-space SDP4 (GPS/GLONASS/Galileo/GEO comms included) · zoom below ~{fmtKm(MINI_MAX_CAM_KM)} camera altitude: the nearest CATALOGUED satellites render as 3D class forms (unidentified stay dots) · click one to identify + FOLLOW it — it zooms in, shows its full SGP4 orbit track, and keeps flying while you pan anywhere; drag frees the camera, the card's ✕ ends the focus · click empty ground for Starlink coverage there · fades out by city zoom (LOD) — zoom out to bring the sky back</span>
              </div>
            </div>
          )}
          {enabled.daynight && (
            <div className="vt-legend-sec">
              <div className="vt-legend-sec-head">Day/Night</div>
              <div className="vt-legend-items">
                <span className="vt-legend-chip"><i style={{ background: "#020617" }} /> night side (right now)</span>
                <span className="vt-legend-note">terminator: computed ephemeris (Meeus low-precision) — display-grade, recomputed each minute; no feed. The Sun, Moon (real phase) and planets render in the sky itself — always on, real astronomy-engine ephemeris at true apparent size; tilt toward the horizon they're above to see them</span>
              </div>
            </div>
          )}
          {enabled.celestial_paths && (
            <div className="vt-legend-sec">
              <div className="vt-legend-sec-head">Celestial paths</div>
              <div className="vt-legend-items">
                <span className="vt-legend-chip"><i style={{ background: "#f2d980" }} /> ecliptic</span>
                <span className="vt-legend-chip">body-colored lines = Moon/planet sky tracks</span>
                <span className="vt-legend-note">real ephemeris (astronomy-engine), sky frozen at the current sidereal time; segments below the horizon fade out — reference lines, not observations</span>
              </div>
            </div>
          )}
          {enabled.seafloor && (
            <div className="vt-legend-sec">
              <div className="vt-legend-sec-head">Seafloor depth</div>
              <div className="vt-legend-items">
                {/* chips render from the SAME stops the map ramp uses
                    (lib/bathymetry — one source of truth), shallow → deep */}
                {[...BATHYMETRY_STOPS].reverse().map((s) => (
                  <span key={s.elevM} className="vt-legend-chip">
                    <i style={{ background: s.color }} /> {s.label}{s.depthM > 0 ? ` ~${fmtMeters(s.depthM)}` : ""}
                  </span>
                ))}
                <span className="vt-legend-note">chips decode OUR depth tint (drawn at low opacity); the dominant coloring + ridge texture beneath is GEBCO_2024 shaded relief (15 arc-sec) with its own depth palette — ship soundings + satellite-gravity interpolation; indicative depths, not for navigation · land shows GEBCO's hypsometric tint while drained · turn on Terrain (3D relief) to make the basins physically sink</span>
              </div>
            </div>
          )}
          {enabled.seafloor_confidence && (
            <div className="vt-legend-sec">
              <div className="vt-legend-sec-head">Seafloor mapping confidence</div>
              <div className="vt-legend-items">
                {/* rows + colors from the SAME GEBCO TID decode table the
                    map's color-relief expression uses (lib/seafloorV2) —
                    they can never drift apart */}
                {tidConfidenceLegend().map((row) => (
                  <span key={row.group} className="vt-legend-chip">
                    <i style={{ background: row.color }} /> {row.label}
                  </span>
                ))}
                {SEAFLOOR_V2_REGIONS.map((r) => {
                  const shares = seafloorConfShares[r.name];
                  return (
                    <span key={r.name} className="vt-legend-note">
                      {r.name}: {shares
                        ? Object.entries(shares).map(([g, v]) => `${Math.round(v * 1000) / 10}% ${g}`).join(" · ")
                        : "measured shares loading…"}
                    </span>
                  );
                })}
                <span className="vt-legend-note">{GEBCO_ATTRIBUTION}</span>
                <span className="vt-legend-note">{GEBCO_NOT_FOR_NAVIGATION} Regional coverage only — the dashed border draws the data's true extent; everywhere else is transparent (no data), never a guessed class.</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

export default function DataMapPage() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<any>(null);
  const glRef = useRef<any>(null);
  const sinceRef = useRef<Record<string, string>>({});
  // EARTH TWIN E3 follow-up: hover tooltip for the 3D aircraft silhouettes
  // (altitude label on hover — deferred from the E3 polish slice). Updated
  // imperatively (direct style/text writes, no setState) so cursor movement
  // never triggers a React re-render of this large component.
  const airHoverTipRef = useRef<HTMLDivElement | null>(null);
  // ORBITAL O2: the satellite worker + GPU layer live across renders so the
  // enable/disable effect can tear both down cleanly (zero-cost-when-off).
  const satWorkerRef = useRef<Worker | null>(null);
  const placesDetach = useRef<(() => void) | null>(null);
  const satLayerRef = useRef<SatLayer | null>(null);
  // ORBITAL O3 (click-to-identify): the GP array the worker was initialized
  // with, kept index-aligned to the layer's position buffer per satWorker.ts's
  // INDEX ALIGNMENT contract — the picking effect resolves a click to an
  // index into this same array.
  const orbitalGpRef = useRef<GpRecord[] | null>(null);
  // EARTH TWIN A1: the registry-declared camera-altitude envelope for
  // orbital_sats, kept in a ref so the O2 effect's move handler always reads
  // the freshest registry value WITHOUT re-running (and tearing down) the
  // whole effect when the registry fetch lands.
  const orbitalLodRef = useRef<LodEnvelope | null>(null);
  // ORBITAL O5 slice 1: the followed satellite (buffer index is stable —
  // the worker keeps ONE slot per GP record forever). null = not following.
  // O6-1 follow v2 + lock modes (human-refined): 'sat' pins the camera to
  // the CRAFT — rotate/tilt/zoom freely around it, it stays centered until
  // unpressed (drag never releases it); 'ground' pins the nadir and a drag
  // releases to free camera; null = free camera, focus persists until ✕.
  const satFollowRef = useRef<{ index: number; noradId: number; name: string | null; lockMode: "sat" | "ground" | null } | null>(null);
  const satArcLayerRef = useRef<ArcLayer | null>(null);
  // O6 aircraft 3D trail (human directive: "see the line from the airport
  // and how it came up to alt") — reuses the generic ArcLayer; fed from the
  // ARCHIVED track (real altitudes; unknown-altitude points = honest gaps).
  const airTrail3dRef = useRef<ArcLayer | null>(null);
  const stopSatFocusRef = useRef<(() => void) | null>(null);
  // O6-3 find & group: search focuses via the same path as a click; a group
  // filters the sky by writing the layer's own sentinel into non-members.
  const focusSatByIndexRef = useRef<((index: number) => void) | null>(null);
  const satGroupMaskRef = useRef<Uint8Array | null>(null);
  const satGroupInfoRef = useRef<{ label: string; count: number } | null>(null);
  // O8 (live report 2026-07-18): the last RAW worker tick buffer, BEFORE the
  // group sentinel copy. The worker always propagates the WHOLE sky (the
  // group chip is a display-only filter), so retaining the raw tick lets
  // (a) a chip change re-derive the layer buffer THIS frame instead of
  // waiting up to a full 1s tick, and (b) search/coverage read the full
  // catalog while the display stays filtered. Cost: one extra retained
  // Float32Array only while a filter is active (~340KB at 12k objects).
  const satRawPosRef = useRef<Float32Array | null>(null);
  const satRepushRef = useRef<(() => void) | null>(null);
  const [satGroup, setSatGroup] = useState<string | null>(null);
  const [satGroupOrbits, setSatGroupOrbits] = useState(false);
  const [satGroupCount, setSatGroupCount] = useState<number | null>(null);
  const [satArcInfo, setSatArcInfo] = useState<{ shown: number; total: number } | null>(null);
  const [gpVersion, setGpVersion] = useState(0);
  // O6-4: aircraft operator filter (broadcast callsign prefix, e.g. AAL*)
  const [airFilter, setAirFilter] = useState<string | null>(null);
  // O6 follow tools (human-requested): re-center lock, zoom-on-sat, ground-
  // spot marker — a minimizable cluster shown only while following.
  const [satFollowing, setSatFollowing] = useState(false);
  const [satLockMode, setSatLockMode] = useState<"sat" | "ground" | null>(null);
  const [satToolsMin, setSatToolsMin] = useState(false);
  const [showNadir, setShowNadir] = useState(false);
  const showNadirRef = useRef(false);
  useEffect(() => { showNadirRef.current = showNadir; }, [showNadir]);
  // O6 tools drag (human-requested round 7: "you cant move them around"):
  // the WHOLE cluster is the drag surface — the tiny grip glyph alone was
  // undiscoverable. Buttons stay buttons via the closest() guard; pointer
  // capture goes on the cluster so the drag survives leaving it. Direct
  // style mutation so a drag never re-renders the whole page component.
  const satToolsRef = useRef<HTMLDivElement | null>(null);
  const satToolsDrag = useRef<{ dx: number; dy: number } | null>(null);
  const onToolsDown = useCallback((e: React.PointerEvent) => {
    if ((e.target as Element).closest("button")) return; // buttons stay buttons
    const el = satToolsRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    satToolsDrag.current = { dx: e.clientX - r.left, dy: e.clientY - r.top };
    el.setPointerCapture?.(e.pointerId);
    e.preventDefault();
  }, []);
  const onToolsMove = useCallback((e: React.PointerEvent) => {
    const el = satToolsRef.current;
    const d = satToolsDrag.current;
    if (!el || !d) return;
    el.style.left = `${Math.max(4, Math.min(window.innerWidth - 70, e.clientX - d.dx))}px`;
    el.style.top = `${Math.max(48, Math.min(window.innerHeight - 46, e.clientY - d.dy))}px`;
    el.style.bottom = "auto";
    el.style.transform = "none"; // resting spot centers via translateX
  }, []);
  const onToolsUp = useCallback(() => { satToolsDrag.current = null; }, []);
  // INSPECT IS THE MAP (human, third repetition 2026-07-18: "i want that to
  // be part of the system not a separate thing so you can inspect as it
  // moves around the earth and see it on the map"). The O7 separate
  // free-orbit scene (lib/orbital/inspectScene) is RETIRED — the map-native
  // follow already orbits the craft itself (setCenterElevation puts the
  // camera center AT the craft: rotate/tilt orbit it, zoom approaches it)
  // over the live map, with the always-on celestial sky supplying the real
  // Sun/Moon context toward the horizon. inspectCraft() below is the one
  // affordance kept from the old chip: a single ease into the close-orbit
  // framing — same camera model, no mode switch, the Earth keeps moving
  // underneath. What the map camera structurally cannot do (look fully
  // AWAY from the ground, pitch past the horizon) stays honestly out of
  // scope rather than living in a disconnected scene the human rejected.
  const inspectCraft = useCallback(() => {
    const map = mapRef.current;
    const f = satFollowRef.current;
    if (!map || !f) return;
    const t = followTarget(satLayerRef.current?.getPositions() ?? null, f.index);
    if (!t) return;
    // ensure the craft-centered lock so the ease orbits the craft, not ground
    if (f.lockMode !== "sat") { f.lockMode = "sat"; setSatLockMode("sat"); }
    const altKmNow = t.altKm;
    // close-orbit framing from altitude (same guarantee as focus framing:
    // the camera never dives inside the craft's orbit shell)
    const zoom = altKmNow < 3000 ? 6.5
      : zoomForCameraAltitudeKm(altKmNow * 1.8, t.latDeg,
          map.getCanvas()?.height ?? 900,
          (map as any).getVerticalFieldOfView?.(), 65);
    map.easeTo({ center: [t.lonDeg, t.latDeg], zoom, pitch: 65, duration: 1400 });
  }, []);
  // CONTINUOUS SPACE FRAME (human-approved 2026-07-18 — the third "no
  // separate scenes" directive, same precedent as INSPECT IS THE MAP above):
  // the O6-7 separate solar-system scene (lib/celestial/solarView) is
  // RETIRED. Past the globe's zoom floor the LIVE MAP CANVAS itself becomes
  // the Earth inside lib/celestial/spaceFrame — it keeps rendering (layers,
  // terminator and all) while a per-frame CSS pose (translate/scale/opacity,
  // computed by the frame) shrinks it into a true-scale star-space frame;
  // when Earth's disc drops under the fade band a live-shaded impostor
  // (real terminator) crossfades in. Zooming back through the seam hands the
  // camera to MapLibre at the zoom floor — the exact reverse of entry, no
  // chip, no flash, no easeTo. Real ephemeris positions and sizes
  // everywhere; the camera does the compressing.
  const spaceActiveRef = useRef(false);
  const spaceHandleRef = useRef<import("@/lib/celestial/spaceFrame").SpaceFrameHandle | null>(null);
  const spaceCleanupRef = useRef<(() => void) | null>(null);
  const [spaceActive, setSpaceActive] = useState(false);
  const [atZoomFloor, setAtZoomFloor] = useState(false);
  const exitSpaceRef = useRef<() => void>(() => {});
  const exitSpace = useCallback(() => {
    if (!spaceActiveRef.current) return;
    spaceActiveRef.current = false;
    setSpaceActive(false);
    const container = mapContainer.current;
    const handle = spaceHandleRef.current;
    spaceHandleRef.current = null;
    spaceCleanupRef.current?.();
    spaceCleanupRef.current = null;
    // dispose releases the map canvas pose (applyEarthAnchor(null) resets
    // transform/opacity) — at the seam that is a visual no-op: scale is
    // already 1, opacity already 1
    try { handle?.dispose(); } catch {}
    container?.classList.remove("vt-space-active");
    const map = mapRef.current;
    if (map) {
      for (const h of ["scrollZoom", "dragPan", "dragRotate", "doubleClickZoom", "touchZoomRotate", "keyboard"] as const) {
        try { (map as any)[h]?.enable(); } catch {}
      }
      // NO easeTo: the frame lands exactly at the zoom floor, above the
      // hemisphere you flew home over — the next wheel-in keeps zooming.
      // Re-entry hysteresis is the wheel accumulator itself (≥60 deltaY).
    }
  }, []);
  useEffect(() => { exitSpaceRef.current = exitSpace; }, [exitSpace]);
  const enterSpace = useCallback(async (entry?: { nudgeDeltaY?: number; flyOut?: boolean }) => {
    const map = mapRef.current;
    const container = mapContainer.current;
    if (!map || !container || spaceActiveRef.current) return;
    spaceActiveRef.current = true;
    try {
      // lazy: the map bundle grows by nothing until a user actually leaves Earth
      const { mountSpaceFrame } = await import("@/lib/celestial/spaceFrame");
      // the seam reads Earth's disc from the GLOBE — flip mercator first
      try {
        if (((map as any).getProjection?.() || {}).type !== "globe" && typeof (map as any).setProjection === "function") {
          (map as any).setProjection({ type: "globe" });
        }
      } catch {}
      // the anchor assumes the map's own north-up, top-down drawing (the
      // frame's axis-referenced camera keeps roll at 0 — tested contract)
      try { if (map.getBearing() !== 0 || map.getPitch() !== 0) map.jumpTo({ bearing: 0, pitch: 0 }); } catch {}
      const mapCanvas = container.querySelector(".maplibregl-canvas") as HTMLElement | null;
      const axis = getTimeAxis();
      const handle = mountSpaceFrame(container, {
        timeMs: axis.mode === "historical" ? axis.atMs : Date.now(),
        getMapSeam: () => {
          const c = map.getCenter();
          return { zoom: map.getZoom(), minZoom: map.getMinZoom(), centerLatDeg: c.lat, centerLonDeg: c.lng };
        },
        applyEarthAnchor: (a) => {
          if (!mapCanvas) return;
          if (a === null) {
            // release (dispose): hand the untouched canvas back to MapLibre
            mapCanvas.style.transform = "";
            mapCanvas.style.opacity = "";
            mapCanvas.style.transformOrigin = "";
            return;
          }
          if (!a.visible) { mapCanvas.style.opacity = "0"; return; }
          mapCanvas.style.transformOrigin = "center center";
          mapCanvas.style.transform =
            `translate(${a.dxPx}px, ${a.dyPx}px) rotate(${a.rollDeg}deg) scale(${a.scale})`;
          mapCanvas.style.opacity = String(a.opacity);
        },
        // hemisphere alignment: the live map always shows Earth's face that
        // looks at the camera (throttled inside the frame; ±85 pre-clamped)
        recenterMap: (latDeg, lonDeg) => { try { map.jumpTo({ center: [lonDeg, latDeg] }); } catch {} },
        onExitToMap: () => exitSpaceRef.current(),
      });
      spaceHandleRef.current = handle;
      (window as any).__vtSpace = handle; // harness seam (prod-inert, like __vtMap)
      // while active: the time axis drives the ephemeris (the scrubber is a
      // planetarium), units repaint labels; live mode ticks 60s (Moon ≈ 0.5'/min)
      const offAxis = subscribeTimeAxis(() => {
        const a = getTimeAxis();
        try { handle.setTime(a.mode === "historical" ? a.atMs : Date.now()); } catch {}
      });
      const offUnits = subscribeUnits(() => { try { handle.render(); } catch {} });
      const iv = window.setInterval(() => {
        if (getTimeAxis().mode === "live") { try { handle.setTime(Date.now()); } catch {} }
      }, 60_000);
      spaceCleanupRef.current = () => {
        offAxis(); offUnits(); window.clearInterval(iv);
        try { delete (window as any).__vtSpace; } catch {}
      };
      container.classList.add("vt-space-active");
      for (const h of ["scrollZoom", "dragPan", "dragRotate", "doubleClickZoom", "touchZoomRotate", "keyboard"] as const) {
        try { (map as any)[h]?.disable(); } catch {}
      }
      setSpaceActive(true);
      // continuity: the triggering gesture's momentum carries into the frame
      // (wheel path), or the chip's continuous outward fly (touch/a11y path)
      if (entry?.nudgeDeltaY) handle.nudgeZoom(entry.nudgeDeltaY);
      if (entry?.flyOut) handle.flyOut();
    } catch {
      // degrade, never break: the map stays fully usable
      spaceActiveRef.current = false;
      spaceCleanupRef.current?.();
      spaceCleanupRef.current = null;
      try { container.classList.remove("vt-space-active"); } catch {}
    }
  }, []);
  // O5-2b: the on-map 3D form layer for the followed satellite (one instance,
  // same lifecycle as satLayerRef).
  const satModelLayerRef = useRef<SatModelLayer | null>(null);
  // Pillar-6 cross-tie cache: generating capacity near each river gauge, keyed
  // by USGS site, populated when the rivergauges layer loads so the gauge-click
  // detail can surface the exposed plants without a network round-trip on click.
  const riverPlantsRef = useRef<Record<string, any>>({});
  const [layers, setLayers] = useState<LayerMeta[]>([]);
  const [enabled, setEnabled] = useState<Record<string, boolean>>(DEFAULT_ON);
  const [runtime, setRuntime] = useState<Record<string, { status: RuntimeStatus; count?: number; note?: string }>>({});
  const [mapReady, setMapReady] = useState(false);
  // v2.4 load-perf: heavy default-on layers (registry data, pollers) mount
  // AFTER the map's first idle so the base map + aircraft win the initial
  // network/CPU contention. Safety timeout keeps a tile-erroring session
  // from never mounting them.
  const [mapSettled, setMapSettled] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);
  // W1 always-on celestial sky: handle + readiness (the paths toggle effect
  // must re-run once the async mount lands).
  const celestialRef = useRef<any>(null);
  const [celestialReady, setCelestialReady] = useState(false);
  // W2 debug flag — evaluated once; when false the FpsChip (and its rAF
  // loop) never mounts.
  const [fpsDebug] = useState<boolean>(() => {
    try {
      return new URLSearchParams(window.location.search).has("fps") ||
        window.localStorage?.getItem("vt-fps") === "1";
    } catch { return false; }
  });
  const [panelOpen, setPanelOpen] = useState<boolean>(() =>
    typeof window !== "undefined" ? window.innerWidth >= 768 : true);
  // Legend v3: collapsible as one unit so it never fights the panel for
  // space — open on desktop, collapsed on phone by default.
  // history time machine (shared): scrub 1900 -> now across every history
  // layer (nuclear tests 1945-98, earthquakes 1900+). Filters run on the GPU
  // so dragging is glitch-free (no refetch — one setFilter per tick per layer).
  const [histYear, setHistYear] = useState(HIST_MAX_YEAR);
  const [histPlay, setHistPlay] = useState(false);
  const [legendOpen, setLegendOpen] = useState<boolean>(() =>
    typeof window !== "undefined" ? window.innerWidth >= 768 : true);
  const [showRawInfo, setShowRawInfo] = useState(false);
  const [detail, setDetail] = useState<Detail | null>(null);
  const applySatGroup = useCallback((key: string | null) => {
    const gp = orbitalGpRef.current;
    const mask = key && gp ? groupMask(gp, key) : null;
    const count = mask ? maskCount(mask) : null;
    // O8 TAKE-ME-THERE (live report 2026-07-18: the ISS chip "does not send
    // you to the satellite and it's hard to find"): a group that resolves to
    // exactly ONE catalog object (ISS after the station collapse) is a
    // shortcut, not a filter — do what searching it does: end the old
    // focus/card and any active filter, then focus + follow + zoom. Multi-
    // object groups keep the filter behavior (and the finder lists their
    // members as the click path).
    if (mask && count === 1 && focusSatByIndexRef.current) {
      setSatGroup(null);
      setSatGroupOrbits(false);
      setSatGroupCount(null);
      satGroupMaskRef.current = null;
      satGroupInfoRef.current = null;
      satRepushRef.current?.(); // clear any prior filter THIS frame — the focus below reads the buffer
      stopSatFocusRef.current?.();
      setDetail(null);
      focusSatByIndexRef.current(mask.indexOf(1));
      return;
    }
    setSatGroup(key);
    setSatGroupOrbits(false);
    satGroupMaskRef.current = mask;
    setSatGroupCount(count);
    satGroupInfoRef.current = mask && key
      ? { label: SAT_GROUPS.find((g) => g.key === key)?.label ?? key, count: count ?? 0 }
      : null;
    // O8: the filter applies/clears THIS frame (re-derives the layer buffer
    // from the retained raw tick) — not on the next 1Hz worker tick.
    satRepushRef.current?.();
  }, []);
  // O6-3 SatFinder entrance (search hits + group-member list), hardened by
  // two live-report fixes: (2026-07-16) searching while an old focus card
  // was open left the STALE card up — hard-swap: end the old focus, drop
  // the old card, then focus fresh. (2026-07-18) SEARCH OVERRIDES FILTER:
  // the finder searches the FULL catalog, so a hit OUTSIDE the active group
  // chip clears the filter first — otherwise the target's buffer slot is
  // sentineled and the focus dead-ends as "no live position" while the old
  // object stays on screen (the reported bug). Clearing goes through
  // applySatGroup, whose instant repush restores the slot before focusSat
  // reads the buffer.
  const findSat = useCallback((index: number) => {
    stopSatFocusRef.current?.();
    setDetail(null);
    const gmask = satGroupMaskRef.current;
    if (gmask && !gmask[index]) applySatGroup(null);
    focusSatByIndexRef.current?.(index);
  }, [applySatGroup]);
  // O6 minimize: collapse the card to a pill (focus keeps running); a NEW
  // detail always restores the full card so fresh clicks are never hidden.
  const [detailMin, setDetailMin] = useState(false);
  // O6 round 6: the card is DRAGGABLE by its header (human: the card must
  // never block the flight track you're inspecting). Direct style mutation
  // (tools-cluster precedent); a NEW detail resets to the default spot.
  const detailCardRef = useRef<HTMLDivElement | null>(null);
  const detailDrag = useRef<{ dx: number; dy: number } | null>(null);
  const onCardHeadDown = useCallback((e: React.PointerEvent) => {
    if ((e.target as Element).closest("button")) return; // buttons stay buttons
    const el = detailCardRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    detailDrag.current = { dx: e.clientX - r.left, dy: e.clientY - r.top };
    (e.target as Element).setPointerCapture?.(e.pointerId);
    e.preventDefault();
  }, []);
  const onCardHeadMove = useCallback((e: React.PointerEvent) => {
    const el = detailCardRef.current;
    const d = detailDrag.current;
    if (!el || !d) return;
    el.style.left = `${Math.max(0, Math.min(window.innerWidth - 80, e.clientX - d.dx))}px`;
    el.style.top = `${Math.max(0, Math.min(window.innerHeight - 60, e.clientY - d.dy))}px`;
    el.style.right = "auto";
    el.style.bottom = "auto";
  }, []);
  const onCardHeadUp = useCallback(() => { detailDrag.current = null; }, []);
  useEffect(() => {
    setDetailMin(false);
    const el = detailCardRef.current;
    if (el) { el.style.left = ""; el.style.top = ""; el.style.right = ""; el.style.bottom = ""; }
  }, [detail?.title, detail?.kind]);
  // Full filings view (#/data/filings) — overlay on top of the map page so
  // the map stays mounted; hash-driven so it deep-links and back-buttons.
  const [filingsOpen, setFilingsOpen] = useState(() => window.location.hash === "#/data/filings");
  // Full earnings-language view (#/data/earnings) — same overlay pattern.
  const [earningsOpen, setEarningsOpen] = useState(() => window.location.hash === "#/data/earnings");
  // Full FINRA short-volume view (#/data/short-volume) — same overlay pattern.
  const [shortvolOpen, setShortvolOpen] = useState(() => window.location.hash === "#/data/short-volume");
  // Full Wikipedia attention view (#/data/attention) — same overlay pattern.
  const [attentionOpen, setAttentionOpen] = useState(() => window.location.hash === "#/data/attention");
  // Full CFTC COT view (#/data/cot) — same overlay pattern.
  const [cotOpen, setCotOpen] = useState(() => window.location.hash === "#/data/cot");
  // Everything Graph full view (#/data/graph) — same overlay pattern.
  const [graphOpen, setGraphOpen] = useState(() => window.location.hash === "#/data/graph");
  // Streams inventory (#/data/streams) — same overlay pattern (Phase 4).
  const [streamsOpen, setStreamsOpen] = useState(() => window.location.hash === "#/data/streams");
  // Grid-stress descriptive reading (#/data/grid-stress) — same overlay
  // pattern (GRID VISION A1 gate-2 FAIL path product, 2026-07-07).
  const [gridStressOpen, setGridStressOpen] = useState(() => window.location.hash === "#/data/grid-stress");
  // v2.3: groups beyond the first fold start collapsed — the panel stays
  // scannable and everything below is one visible tap away. Derived from
  // PANEL_GROUPS + OPEN_GROUPS_BY_DEFAULT (BUILD ORDER 4 #2) instead of a
  // second hardcoded list: today's result is identical
  // ({base: false, live: false, facilities: true, environmental: true,
  // filings: true, signals: true}) but any group PANEL_GROUPS gains later
  // defaults collapsed automatically, closing the "forgot to add it to the
  // collapsed list" gap that let a big new group dump its full DOM by default.
  const [groupCollapsed, setGroupCollapsed] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(PANEL_GROUPS.map((g) => [g.id, !OPEN_GROUPS_BY_DEFAULT.has(g.id)])));
  // per-group "show all" override for the GROUP_ROW_CAP progressive-
  // disclosure cap (BUILD ORDER 4 #2 panel-scale item) — starts empty
  // (every group capped) until the user explicitly expands one past 12 rows.
  const [groupShowAll, setGroupShowAll] = useState<Record<string, boolean>>({});
  // W6 ANALYST pane — closed by default at every width (a chat panel is a
  // deliberate act, never a permanent overlay); no persistence: the pane's
  // session history lives inside the lazy chunk instead.
  const [analystOpen, setAnalystOpen] = useState(false);
  // W3 TIME SCRUBBER panel — same "deliberate act, never a permanent overlay"
  // rule as the analyst pane; closed by default, no persistence.
  const [timescrubOpen, setTimescrubOpen] = useState(false);
  // v2.3 fullscreen map mode — nav hidden via a body class; remembered per
  // session; the map needs a resize after the container jumps.
  const [fullscreen, setFullscreen] = useState<boolean>(() => {
    try { return sessionStorage.getItem("vt-map-fs") === "1"; } catch { return false; }
  });
  useEffect(() => {
    document.body.classList.toggle("vt-map-fullscreen", fullscreen);
    try { sessionStorage.setItem("vt-map-fs", fullscreen ? "1" : "0"); } catch {}
    const t = window.setTimeout(() => { try { mapRef.current?.resize(); } catch {} }, 60);
    return () => {
      window.clearTimeout(t);
      document.body.classList.remove("vt-map-fullscreen");
    };
  }, [fullscreen]);
  // ── W1 globe/flat projection toggle (console charter) ──
  // globeSupport: "ok" once the loaded maplibre exposes setProjection;
  // "unavailable" if it doesn't OR a projection call throws at runtime
  // (GPU/WebGL constraint) — then the map stays mercator and the toggle
  // renders disabled with the reason in its title. Never a broken map,
  // never a silent failure.
  const [globeOn, setGlobeOn] = useState<boolean>(readGlobePref);
  const [globeSupport, setGlobeSupport] = useState<"unknown" | "ok" | "unavailable">("unknown");
  // Style presets (worldview-globe G1): switch the BASE look on the one globe —
  // real-first geographic identities, no tactical FLIR/NVG. Persisted per browser.
  const [mapPreset, setMapPreset] = useState<string>(() => {
    try { return window.localStorage.getItem("vt-map-preset") || "natural"; } catch { return "natural"; }
  });
  useEffect(() => {
    try { window.localStorage.setItem(GLOBE_PREF_KEY, globeOn ? "1" : "0"); } catch {}
    const map = mapRef.current;
    if (!map || !mapReady || globeSupport !== "ok") return;
    try {
      // normalize: getProjection() is undefined until a projection was set
      const cur = (map.getProjection?.() || {}).type || "mercator";
      const want = globeOn ? "globe" : "mercator";
      // zero-cost-when-off: with the flat preference the bootstrap style
      // carries no projection and cur===want — no projection API work at all.
      if (cur !== want) map.setProjection({ type: want });
    } catch {
      setGlobeSupport("unavailable");
      setGlobeOn(false);
    }
  }, [globeOn, mapReady, globeSupport]);

  // Style-preset base look (worldview-globe G1). Switches the BASE imagery on the
  // one globe — real-first geographic presets, never a tactical filter:
  //   natural  = Esri World Imagery (default)
  //   night    = NASA VIIRS Black Marble (Earth at night — city lights)
  //   terrain  = Esri imagery + auto-enable the 3D relief layer
  //   minimal  = no imagery (clean dark astro base + boundaries for reference)
  // Only base-imagery visibility changes here; all data layers are untouched.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    try { window.localStorage.setItem("vt-map-preset", mapPreset); } catch {}
    try {
      const BLACK_MARBLE = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_Black_Marble/default/2016-01-01/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png";
      if (mapPreset === "night") {
        if (!map.getSource("blackmarble")) {
          map.addSource("blackmarble", {
            type: "raster", tiles: [BLACK_MARBLE], tileSize: 256, maxzoom: 8,
            attribution: "Earth at Night — VIIRS Black Marble · NASA GIBS/ESDIS (public domain)",
          } as any);
        }
        if (!map.getLayer("blackmarble")) {
          map.addLayer({ id: "blackmarble", type: "raster", source: "blackmarble" } as any, "imagery");
        }
      } else {
        if (map.getLayer("blackmarble")) map.removeLayer("blackmarble");
        if (map.getSource("blackmarble")) map.removeSource("blackmarble");
      }
      // Base imagery hidden for night (black marble shows through) + minimal
      // (clean dark base shows through); visible for natural + terrain.
      const imageryVisible = mapPreset === "natural" || mapPreset === "terrain";
      if (map.getLayer("imagery")) {
        map.setLayoutProperty("imagery", "visibility", imageryVisible ? "visible" : "none");
      }
      // Terrain preset auto-enables the 3D relief + boundaries give minimal a
      // reference frame — set once, the user can still toggle afterwards.
      if (mapPreset === "terrain") setEnabled((s) => (s.terrain ? s : { ...s, terrain: true }));
      if (mapPreset === "minimal") setEnabled((s) => (s.boundaries ? s : { ...s, boundaries: true }));
    } catch { /* base swap failed — data layers unaffected, map stays alive */ }
  }, [mapPreset, mapReady]);

  const [descOpen, setDescOpen] = useState<Record<string, boolean>>({});
  // worldview_globe.md G2a: night-lights time-scrubber state. Defaults to
  // the charter's "yesterday" — GIBS daily layers never carry today's data.
  const [nightlightsDate, setNightlightsDate] = useState<string>(() => gibsDefaultDate(Date.now()));
  // worldview_globe.md G2c: aerosol optical depth time-scrubber state (same
  // yesterday-default rule as night lights — GIBS daily layers never carry
  // same-day data).
  const [aerosolDate, setAerosolDate] = useState<string>(() => gibsDefaultDate(Date.now()));
  // worldview_globe.md G2e: vegetation health (NDVI) time-scrubber state. Same
  // yesterday-default; NDVI is an 8-day composite, so a yesterday request
  // returns the current composite (verified non-blank over land at build time).
  const [vegetationDate, setVegetationDate] = useState<string>(() => gibsDefaultDate(Date.now()));
  // worldview_globe.md G2d: root-zone soil moisture (SMAP). Unlike the daily/
  // composite layers above, SMAP L4 lands ~6 days back, so the default steps
  // back a conservative 7 days (SOIL_LATENCY_DAYS) rather than "yesterday" —
  // otherwise the scrubber would default to a guaranteed-blank tile.
  const [soilmoistureDate, setSoilmoistureDate] = useState<string>(() => gibsDefaultDate(Date.now(), SOIL_LATENCY_DAYS));
  // worldview_globe.md G2g: tropospheric NO2 column (TROPOMI). Daily, ~1-day
  // lag like the other daily layers — the charter's "genuinely differentiated"
  // layer (industrial/traffic combustion throughput nowcast).
  const [no2Date, setNo2Date] = useState<string>(() => gibsDefaultDate(Date.now()));
  // ── EARTH TWIN E1: GLOBAL TIME AXIS — one clock moves the whole world.
  // The Time Machine panel publishes the axis (lib/timeAxis); every dated
  // GIBS layer above follows it to ITS OWN honest ceiling (latency-aware —
  // SMAP snaps ~7 days back, dailies to yesterday), and returning to LIVE
  // restores each layer's default. Per-layer scrubbers still work as manual
  // overrides afterward. firetemp is deliberately NOT wired: it is sub-daily
  // latest-scan-only (no dated archive endpoint on this map yet — honest gap,
  // charter A3 sub-daily work). React bails cheaply on identical dates. ──
  useEffect(() => {
    const apply = () => {
      const axis = getTimeAxis();
      const now = Date.now();
      setNightlightsDate(gibsDateForAxis(axis, now).dateISO);
      setAerosolDate(gibsDateForAxis(axis, now).dateISO);
      setVegetationDate(gibsDateForAxis(axis, now).dateISO);
      setNo2Date(gibsDateForAxis(axis, now).dateISO);
      setSoilmoistureDate(gibsDateForAxis(axis, now, SOIL_LATENCY_DAYS).dateISO);
    };
    return subscribeTimeAxis(apply);
  }, []);
  // EARTH TWIN E1 remainder: a persistent LIVE/HISTORICAL indicator OUTSIDE
  // the Time Machine panel — the panel is small and can scroll out of view
  // (mobile especially), and dated imagery layers silently show the past
  // while the rest of the map looks unchanged, so the mode needs its own
  // always-visible chip, not just the panel's inline date line.
  const [historicalAtMs, setHistoricalAtMs] = useState<number | null>(() => {
    const axis = getTimeAxis();
    return axis.mode === "historical" ? axis.atMs : null;
  });
  useEffect(() => {
    const apply = () => {
      const axis = getTimeAxis();
      setHistoricalAtMs(axis.mode === "historical" ? axis.atMs : null);
    };
    return subscribeTimeAxis(apply);
  }, []);
  // worldview_globe.md G2b: GOES-East fire/hotspot brightness temperature.
  // Genuinely sub-daily (~10-min, irregular scan gaps) — no day-granularity
  // scrubber like the layers above; always requests GIBS's own "default"
  // (freshest available scan) and separately reads the real scan timestamp
  // via gibsLatestScanTime for an honest freshness note. null = not yet
  // fetched or fetch failed — rendered as "scan time unknown", never guessed.
  const [firetempScanTime, setFiretempScanTime] = useState<string | null>(null);
  // worldview_globe.md G2h: GEDI L4B aboveground biomass density — a
  // genuinely STATIC composite (single 2019-04–2023-03 mission-life mean,
  // no daily/8-day cadence, no scrubber), unlike every dated layer above.
  // ── weather-upgrade (2026-07-04): registry-native FIELD layer controls ──
  // Field layers (registry flag `field: true`) get a per-layer opacity
  // slider; default 60% so the basemap + live layers stay visible beneath
  // (directive: the field is context, never a curtain).
  const FIELD_MAP_LAYER: Record<string, string> = {
    weather: "weather-radar", weather_temp: "wx-temp_new", weather_wind: "wx-wind_new",
    surfacewater: "gsw-occurrence", forest: "jrc-forest", nightlights: "gibs-nightlights",
    aerosol: "gibs-aerosol",
    vegetation: "gibs-vegetation",
    soilmoisture: "gibs-soilmoisture",
    no2: "gibs-no2",
    firetemp: "gibs-firetemp",
    biomass: "gibs-biomass",
    floodzones: "fema-floodzones",
    // E2 v2: the slider drives the GEBCO shaded-relief raster (the dominant
    // visual); the legend-bearing depth tint above it stays at a fixed low
    // opacity (see the seafloor effect)
    seafloor: OCEAN_BASEMAP_LAYER_ID,
  };
  // Most field layers are raster (paint prop "raster-opacity"); layer types
  // with their own opacity prop override here.
  const FIELD_OPACITY_PROP: Record<string, string> = {};
  // Per-layer opacity DEFAULTS: fields default to 60% (context, never a
  // curtain); the drained ocean is a basemap swap, not a context field —
  // it defaults to full and the slider still blends it.
  // seafloor_confidence defaults high (not the standard 60% "context" fade):
  // the confidence classes ARE the content, not a backdrop — a faint tint
  // would defeat the honesty-display purpose of the layer.
  const FIELD_OPACITY_DEFAULT: Record<string, number> = { seafloor: 100, seafloor_confidence: 85 };
  const [fieldOpacity, setFieldOpacityState] = useState<Record<string, number>>(() => {
    try { return JSON.parse(sessionStorage.getItem("vt-field-opacity") || "{}"); } catch { return {}; }
  });
  const opacityOf = (id: string) => fieldOpacity[id] ?? FIELD_OPACITY_DEFAULT[id] ?? 60;
  const setFieldOpacity = (id: string, v: number) => {
    setFieldOpacityState((s) => {
      const next = { ...s, [id]: v };
      try { sessionStorage.setItem("vt-field-opacity", JSON.stringify(next)); } catch {}
      return next;
    });
    try {
      if (id === "seafloor_confidence") {
        // multi-region layer (today: Mariana only) — apply to every committed
        // region's TID layer so SEAFLOOR_V2_REGIONS growing needs no new code
        // here, only a new array entry + pipeline run.
        for (const r of SEAFLOOR_V2_REGIONS) {
          mapRef.current?.setPaintProperty(`seafloor-confidence-relief-${r.name}`, "color-relief-opacity", v / 100);
        }
      } else {
        mapRef.current?.setPaintProperty(FIELD_MAP_LAYER[id], FIELD_OPACITY_PROP[id] ?? "raster-opacity", v / 100);
      }
    } catch {}
  };
  // Wind vectors + temperature labels — sampled point grid (HONEST: OWM
  // tiles carry no vector data; numbers come from point samples, arrows
  // never denser than the sampling — the note shows real spacing).
  const [windArrows, setWindArrows] = useState(true);
  const [tempLabels, setTempLabels] = useState(false);
  // site-wide unit system (imperial|metric; lib/units.ts). useSyncExternalStore
  // keeps panel JSX (legends, chips, the toggle itself) re-rendering on change;
  // click handlers read the live pref at call time through the fmt* helpers.
  const unitSystem = useSyncExternalStore(subscribeUnits, getUnits, getUnits);
  // weather temp labels follow the global pref by default; its own °F/°C
  // chip still overrides for the map-label layer specifically.
  const [tempUnitF, setTempUnitF] = useState(() => getUnits() === "imperial");
  useEffect(() => { setTempUnitF(unitSystem === "imperial"); }, [unitSystem]);
  const [wxGrid, setWxGrid] = useState<any>(null);
  useEffect(() => {
    const onHash = () => {
      setFilingsOpen(window.location.hash === "#/data/filings");
      setEarningsOpen(window.location.hash === "#/data/earnings");
      setShortvolOpen(window.location.hash === "#/data/short-volume");
      setAttentionOpen(window.location.hash === "#/data/attention");
      setCotOpen(window.location.hash === "#/data/cot");
      setGraphOpen(window.location.hash === "#/data/graph");
      setStreamsOpen(window.location.hash === "#/data/streams");
      setGridStressOpen(window.location.hash === "#/data/grid-stress");
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // v2.4 eternal-spinner rule (DESIGN.md): every status change is
  // timestamped; a watchdog upgrades any bare "loading" older than 30s to a
  // designed retrying note so no spinner ever lives unexplained.
  const statusAtRef = useRef<Record<string, number>>({});
  const setStatus = useCallback((id: string, status: RuntimeStatus, count?: number, note?: string) => {
    statusAtRef.current[id] = Date.now();
    setRuntime(s => {
      // No-op bail ([REPAIR 2026-07-05] map perf): five default layers
      // poll on 15-60s ticks and re-reported identical status every tick —
      // each write re-rendered the whole page (18 panel rows + legend),
      // contending with the map's rAF. Identical payload → same state
      // object → React skips the render.
      const prev = s[id];
      if (prev && prev.status === status && prev.count === count && prev.note === note) return s;
      return { ...s, [id]: { status, count, note } };
    });
  }, []);
  useEffect(() => {
    const iv = window.setInterval(() => {
      setRuntime((s) => {
        let changed = false;
        const next: typeof s = { ...s };
        for (const [id, rt] of Object.entries(s)) {
          if (rt.status === "loading" && !rt.note &&
              Date.now() - (statusAtRef.current[id] || 0) > 30_000) {
            next[id] = { ...rt, note: "no response in 30s — still retrying automatically" };
            changed = true;
          }
        }
        return changed ? next : s;
      });
    }, 10_000);
    return () => window.clearInterval(iv);
  }, []);

  // Layer registry (datacore boundary) + open-tab version-skew detection
  const [versionSkew, setVersionSkew] = useState<string | null>(null);
  useEffect(() => {
    fetch("/api/data/layers")
      .then(r => r.json())
      .then(d => {
        setLayers(Array.isArray(d.layers) ? d.layers : []);
        if (d.server_version && d.server_version !== CLIENT_VERSION) {
          setVersionSkew(String(d.server_version));
        }
      })
      .catch(() => setLayers([]));
  }, []);

  // Map bootstrap (maplibre JS lazy-loaded to keep the main bundle lean)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const maplibregl = (await import("maplibre-gl")).default;
        if (cancelled || !mapContainer.current || mapRef.current) return;
        glRef.current = maplibregl;
        // pmtiles:// protocol (single static file on our origin, range
        // requests — powers the OSM grid layer; registration is idempotent)
        try {
          const { Protocol } = await import("pmtiles");
          maplibregl.addProtocol("pmtiles", new Protocol().tile);
        } catch {}
        // W1 globe: feature-detect on the loaded library (an older bundle
        // or constrained runtime without setProjection degrades to flat
        // with the toggle disabled), then bake the initial projection into
        // the style so the first paint is already in the preferred mode.
        const canGlobe = typeof (maplibregl.Map.prototype as any).setProjection === "function";
        setGlobeSupport(canGlobe ? "ok" : "unavailable");
        const startGlobe = canGlobe && readGlobePref();
        const map = new maplibregl.Map({
          container: mapContainer.current,
          // round 8 realism: allow the low Google-Earth-style camera — the
          // default 60° cap makes real 3D terrain read like a tilted map
          maxPitch: 80,
          style: {
            version: 8,
            ...(startGlobe ? { projection: { type: "globe" } } : {}),
            glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
            sources: {
              imagery: { type: "raster", tiles: [IMAGERY_TILES], tileSize: 256, attribution: IMAGERY_ATTRIB },
            },
            layers: [
              { id: "bg", type: "background", paint: { "background-color": "#050a13" } },
              { id: "imagery", type: "raster", source: "imagery" },
            ],
          },
          center: [-96.77, 37.5],
          zoom: 3.6,
          attributionControl: { compact: true } as any,
          keyboard: true,
        });
        // v2.4 control occlusion: zoom lives bottom-LEFT — the layers panel
        // (right side, full-height allowance) can never cover it at any
        // width. Self-see asserts non-occlusion mechanically.
        // Compass + pitch indicator (worldview-globe G0b): shows heading, resets
        // north on click, and the needle tilts to visualize pitch — the
        // orientation cue the 3D globe/terrain view needs. Clean Google-Earth-
        // style nav, our styling (index.css .maplibregl-ctrl-compass*).
        map.addControl(new maplibregl.NavigationControl({ showCompass: true, showZoom: true, visualizePitch: true }), "bottom-left");
        map.addControl(new maplibregl.ScaleControl({ unit: "imperial" }), "bottom-left");
        mapRef.current = map;
        // Perf-harness hook (scripts/visual_check.mjs drives pans through this).
        (window as any).__vtMap = map;
        let readyFired = false;
        const ready = () => {
          if (cancelled || readyFired) return;
          readyFired = true;
          window.clearInterval(stylePoll);
          try { map.resize(); } catch {}
          try { registerIcons(map); } catch {}
          setMapReady(true);
          // v2.4 deferred mount: heavy default-on layers wait for the first
          // post-ready idle (base map + aircraft win the initial contention);
          // 4s failsafe so tile errors can't starve them forever.
          map.once("idle", () => { if (!cancelled) setMapSettled(true); });
          window.setTimeout(() => { if (!cancelled) setMapSettled(true); }, 4000);
        };
        map.once("load", ready);
        map.once("idle", ready);
        const stylePoll = window.setInterval(() => {
          try { if (map.isStyleLoaded()) ready(); } catch {}
        }, 400);
        // Failsafe: a hostile network (tiles resetting forever) must degrade
        // to a usable map with layer-level error states, never a dead page.
        window.setTimeout(ready, 8000);
        map.on("error", (e: any) => {
          if (readyFired) return;
          if (e?.error?.message && /style/i.test(e.error.message)) setMapError(e.error.message);
        });
      } catch (e: any) {
        setMapError(e?.message || "Map failed to load");
      }
    })();
    return () => {
      cancelled = true;
      try { delete (window as any).__vtMap; } catch {}
      try { mapRef.current?.remove(); mapRef.current = null; } catch {}
    };
  }, []);

  // Escape closes card / tooltip / (phone) panel
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      // space frame: Escape = fly home — a continuous flight back through
      // the seam (the frame exits itself on landing), never a scene cut
      if (spaceActiveRef.current) { try { spaceHandleRef.current?.flyHome(); } catch {} return; }
      setDetail(null);
      clearTrail();
      setShowRawInfo(false);
      setAnalystOpen(false); // DESIGN.md: Escape closes panels/popovers
      setTimescrubOpen(false);
      if (window.innerWidth < 768) setPanelOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // CONTINUOUS ZOOM SEAM: at the zoom floor further wheel-out is inert to
  // MapLibre — read the raw gesture and hand the SAME motion straight to the
  // space frame (≥60 accumulated deltaY inside 400ms windows ≈ one real
  // notch, so "keep zooming out" simply keeps going; trackpad micro-jitter
  // never triggers). The accumulated delta is forwarded as the frame's first
  // zoom impulse — no dead notch at the boundary. The ☉ chip (shown while
  // atZoomFloor) is the touch/a11y path and triggers the same continuous
  // outward fly, never a scene swap.
  useEffect(() => {
    if (!mapReady) return;
    const map = mapRef.current;
    const el = mapContainer.current;
    if (!map || !el) return;
    const updFloor = () => { try { setAtZoomFloor(map.getZoom() <= map.getMinZoom() + 0.05); } catch {} };
    updFloor();
    map.on("zoom", updFloor);
    map.on("zoomend", updFloor);
    let acc = 0;
    let timer: number | null = null;
    const onWheel = (e: WheelEvent) => {
      if (spaceActiveRef.current || e.deltaY <= 0) return;
      try { if (map.getZoom() > map.getMinZoom() + 0.05) { acc = 0; return; } } catch { return; }
      acc += e.deltaY;
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(() => { acc = 0; }, 400);
      if (acc >= 60) { const carry = acc; acc = 0; void enterSpace({ nudgeDeltaY: carry }); }
    };
    el.addEventListener("wheel", onWheel, { capture: true, passive: true });
    return () => {
      try { map.off("zoom", updFloor); map.off("zoomend", updFloor); } catch {}
      el.removeEventListener("wheel", onWheel, { capture: true } as any);
      if (timer) window.clearTimeout(timer);
    };
  }, [mapReady, enterSpace]);

  // SESSION BREADCRUMBS state (2026-07-18): the followed aircraft's live
  // fixes this session (display-only; the archive stays the recorded
  // truth), and the last fetched archived track so a fresh crumb can
  // repaint the merged trail WITHOUT hitting the track endpoint.
  const airCrumbsRef = useRef<{ id: string | null; crumbs: Crumb[] }>({ id: null, crumbs: [] });
  const archivedTrackRef = useRef<{ kind: string; id: string; raw: TrackPoint[] } | null>(null);
  // The followed plane's latest REAL fix + its broadcast dead-reckoning
  // rate + the receipt anchor — lets the curtain tail meet the plane where
  // it is DRAWN between polls (the same glide the plane renders with).
  const airFollowLiveRef = useRef<{ id: string; fix: Crumb; vel: { dLon: number; dLat: number } | null; anchorMs: number } | null>(null);

  const clearTrail = () => {
    const map = mapRef.current;
    if (!map) return;
    try {
      if (map.getLayer("trail-line")) map.removeLayer("trail-line");
      if (map.getSource("trail")) map.removeSource("trail");
    } catch {}
    try { airTrail3dRef.current?.setArcs(null); airTrail3dRef.current?.setWalls(null); } catch {}
    // NOTE: archivedTrackRef is deliberately NOT cleared here. paintTrack's
    // first-paint setup branch calls clearTrail before adding the source —
    // clearing the cache there wiped the archived points showTrail had just
    // stored, so every follow-tick repainted crumbs-only until the 30s
    // refresh (probe-caught: ArcLayer held 2 quads instead of ~30). The
    // cache is read only under a {kind,id} match while that card is open —
    // a stale entry can never be misused.
  };

  /** Paint/refresh the trail layers from a (possibly crumb-extended) point
   *  list — extracted from showTrail so a live breadcrumb can repaint
   *  without a refetch. Returns the newest position time. */
  const paintTrack = (kind: "aircraft" | "vessels" | "trains", raw: TrackPoint[]): number | undefined => {
    const map = mapRef.current;
    if (!map) return undefined;
    try {
      const pts = raw.map((p) => [p.lo, p.la]);
      const lastT = raw.length ? raw[raw.length - 1].t : undefined;
      const feature = {
        type: "Feature", geometry: { type: "LineString", coordinates: pts }, properties: {},
      } as any;
      const existing = map.getSource("trail") as any;
      if (existing && pts.length >= 2) {
        existing.setData(feature); // live append — no remove/re-add flicker
      } else {
        clearTrail();
        if (pts.length >= 2) {
          map.addSource("trail", { type: "geojson", data: feature });
          map.addLayer({
            id: "trail-line", type: "line", source: "trail",
            paint: { "line-color": "#7cc4ff", "line-width": 2, "line-opacity": 0.8, "line-dasharray": [1, 1.5] },
          });
        }
      }
      // O6: the 3D TRACK — the same archived points at their REAL recorded
      // altitude (climb-out from the airport reads as a rising line). Points
      // without a broadcast altitude become honest GAPS in the 3D line (the
      // flat dashed trail above still shows the position). Aircraft only —
      // vessels/trains live at the surface.
      if (kind === "aircraft") {
        try {
          // terrain match: the DEM mesh is exaggerated 1.3x — the track uses
          // the SAME factor so it flies over the mountains it really flew
          // over (the aircraft 3D layer's setAltScale precedent).
          const altScale = map.getTerrain() ? 1.3 : 1;
          const p3 = new Float32Array(raw.length * 3);
          for (let i = 0; i < raw.length; i++) {
            const m = lonLatToMercator(raw[i].lo, raw[i].la);
            const al = (raw[i] as any).al;
            p3[i * 3] = m.x;
            p3[i * 3 + 1] = m.y;
            p3[i * 3 + 2] = al == null ? ARC_GAP : Math.max(0, al) * altScale;
          }
          // CURTAIN v2 (the human's "plane with the 3d terrain and altitude
          // path" ask): a SOLID VeloViewer-style wall from the ground up to
          // the flight path via ArcLayer.setWalls — replaces the ribbed
          // vertical lines. Colored by REAL altitude through the same AIR
          // band stops as the aircraft silhouettes (defaultWallRamp,
          // test-pinned to airLayer BAND_COLORS): the ramp unscales the
          // terrain exaggeration so 3000m stays cruise-blue regardless of
          // the 1.3x mesh factor.
          let arcs = airTrail3dRef.current;
          if (!arcs) {
            arcs = new ArcLayer({ id: "aircraft-trail-3d" });
            airTrail3dRef.current = arcs;
          }
          if (!map.getLayer("aircraft-trail-3d")) map.addLayer(arcs);
          arcs.setArcs(raw.length >= 2 ? [
            { pts: p3, color: [0.49, 0.77, 1.0, 0.9] },
          ] : null);
          arcs.setWalls(raw.length >= 2 ? [{ pts: p3 }] : null,
            { ramp: (alt) => defaultWallRamp(alt / altScale) });
        } catch { /* the flat trail still works */ }
      } else {
        try { airTrail3dRef.current?.setArcs(null); airTrail3dRef.current?.setWalls(null); } catch {}
      }
      (window as any).__vtTrailLen = pts.length; // harness ratchet reads this
      return lastT;
    } catch { return undefined; }
  };

  /** Compose the followed aircraft's display track — archived history +
   *  session crumbs + a dead-reckoned TAIL to where the plane is DRAWN
   *  right now (the same broadcast-velocity glide the plane itself renders
   *  with, same MAX_AIR_GLIDE_SEC cap; altitude held at the last broadcast
   *  value — vertical rate isn't in the feed, never invented) — and paint
   *  it. Snaps to the real fix on every poll. Without the tail the curtain
   *  would end up to a glide-cap behind the moving plane at high zoom —
   *  the reported "data is cut off" gap. */
  const paintFollowedTrail = () => {
    const fid = airCrumbsRef.current.id;
    if (!fid) return;
    const at = archivedTrackRef.current;
    const base = at && at.kind === "aircraft" && at.id === fid ? at.raw : [];
    let track = mergeTrackWithCrumbs(base, airCrumbsRef.current.crumbs);
    const lv = airFollowLiveRef.current;
    if (lv && lv.id === fid && lv.vel) {
      const dt = airGlideDtSec(performance.now(), lv.anchorMs);
      if (dt > 0) {
        track = track.concat({
          lo: lv.fix.lo + lv.vel.dLon * dt,
          la: lv.fix.la + lv.vel.dLat * dt,
          al: lv.fix.al, t: lv.fix.t + dt,
        });
      }
    }
    paintTrack("aircraft", track);
  };

  /** Fetch the archived track, merge the session's live breadcrumbs for the
   *  followed aircraft (mergeTrackWithCrumbs — archived history untouched,
   *  crumbs only extend past its newest sample), and paint/refresh the
   *  trail. On refresh the existing geojson source is UPDATED via setData
   *  (no layer churn). Returns the note + newest position time so the card
   *  can show live freshness. ([REPAIR 2026-07-05]: the trail was fetched
   *  ONCE at selection and never again — a static snapshot while the
   *  aircraft kept moving; see the refresh effect below.) */
  const showTrail = async (kind: "aircraft" | "vessels" | "trains", id: string):
      Promise<{ note: string; lastT?: number }> => {
    if (!mapRef.current) return { note: "" };
    try {
      const r = await fetch(`/api/data/track/${kind}/${encodeURIComponent(id)}`);
      const d = await r.json();
      const raw = (d.points || []) as TrackPoint[];
      archivedTrackRef.current = { kind, id, raw };
      const followed = kind === "aircraft" && airCrumbsRef.current.id === id;
      const merged = followed ? mergeTrackWithCrumbs(raw, airCrumbsRef.current.crumbs) : raw;
      let lastT: number | undefined;
      if (followed) {
        paintFollowedTrail(); // includes the dead-reckoned tail to the drawn plane
        lastT = merged.length ? merged[merged.length - 1].t : undefined;
      } else {
        lastT = paintTrack(kind, merged);
      }
      const liveN = merged.length - raw.length;
      return {
        note: d.note || (merged.length
          ? `${raw.length} archived positions (our own feed history, sampled ~1-5 min at cruise)` +
            (liveN > 0 ? ` + ${liveN} live fixes this session` : "")
          : ""),
        lastT,
      };
    } catch { return { note: "trail unavailable" }; }
  };

  /** ENTITY DOSSIER v2 (ANALYST CONSOLE charter W5) — fetch identity +
   *  cross-layer Everything-Graph neighborhood + ticker-matched USAspending
   *  contracts + nearest strategic sites for the just-clicked entity, and
   *  patch it onto the open card. `entityId` is a graph-resolvable id
   *  (facility/company/vessel) when the clicked kind has one; null for kinds
   *  the graph doesn't model yet (aircraft, trains, fires, gauges, alerts,
   *  quakes, buoys) —
   *  those still get `nearest_sites` from lat/lon alone, which is why lat/lon
   *  is always sent even when entityId is present. Matched back via
   *  `dossierKey`, not trailId/title (see the Detail interface note). Fails
   *  silently like every other async card enrichment (owner/timeline above) —
   *  a dossier that never arrives just leaves that section absent. */
  const fetchDossier = async (dossierKey: string, entityId: string | null, lat: number, lon: number) => {
    try {
      const qs = new URLSearchParams({ lat: String(lat), lon: String(lon) });
      if (entityId) qs.set("entity", entityId);
      const r = await fetch(`/api/data/dossier?${qs.toString()}`);
      if (!r.ok) return;
      const d = await r.json();
      setDetail((prev) => (prev && prev.dossierKey === dossierKey ? { ...prev, dossier: d } : prev));
    } catch {}
  };

  // Live trail refresh — while a track-bearing card is open, re-pull the
  // archived track every 30s so the trail extends as new positions land
  // (the archive tick appends every few minutes; 30s keeps the popup
  // honest without hammering the tiny track endpoint).
  const detailTrailId = detail?.trailId;
  const detailTrailKind = detail?.trailKind;
  // Breadcrumb ownership follows the open card: a fresh aircraft card
  // starts an empty session buffer for that hex; closing the card (or
  // following a non-aircraft) drops it — crumbs never outlive the follow.
  useEffect(() => {
    if (detailTrailKind === "aircraft" && detailTrailId) {
      if (airCrumbsRef.current.id !== detailTrailId) {
        airCrumbsRef.current = { id: detailTrailId, crumbs: [] };
        airFollowLiveRef.current = null;
      }
    } else {
      airCrumbsRef.current = { id: null, crumbs: [] };
      airFollowLiveRef.current = null;
    }
  }, [detailTrailId, detailTrailKind]);
  useEffect(() => {
    if (!detailTrailId || !detailTrailKind) return;
    const iv = setInterval(async () => {
      const { note, lastT } = await showTrail(detailTrailKind, detailTrailId);
      setDetail((prev) => prev && prev.trailId === detailTrailId
        ? { ...prev, trailNote: note || prev.trailNote, trailLastT: lastT ?? prev.trailLastT }
        : prev);
    }, 30_000);
    return () => clearInterval(iv);
  }, [detailTrailId, detailTrailKind]);

  // 10s ticker so the freshness age in the open card counts up between
  // refreshes instead of freezing at its fetch-time value.
  const [freshTick, setFreshTick] = useState(0);
  useEffect(() => {
    if (!detailTrailId) return;
    const iv = setInterval(() => setFreshTick((n) => n + 1), 10_000);
    return () => clearInterval(iv);
  }, [detailTrailId]);

  const formatAge = (epochSec?: number): string | null => {
    if (!epochSec) return null;
    const s = Math.max(0, Math.floor(Date.now() / 1000 - epochSec));
    if (s < 90) return `${s}s ago`;
    if (s < 5400) return `${Math.round(s / 60)}m ago`;
    return `${Math.round(s / 3600)}h ago`;
  };

  // ── imagery toggle ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    try { map.setLayoutProperty("imagery", "visibility", enabled.imagery ? "visible" : "none"); } catch {}
    // IMAGERY METADATA honesty (DESIGN.md 2026-07-04): the rule says
    // display capture dates WHERE AVAILABLE — available since the
    // identify endpoint was verified (census §3 #9, DATACORE MAXIMUS
    // Phase 3a); the live readout below owns the on-map chip and this
    // panel note points to it.
    if (enabled.imagery) setStatus("imagery", "active", undefined, "capture date of the view centre shown on-map (bottom-left); dates vary within a view");
    else setStatus("imagery", "off");
  }, [enabled.imagery, mapReady, setStatus]);

  // ── Phase 3a: live viewport capture-date readout (2026-07-07) ──
  // Esri World_Imagery identify at the VIEW CENTRE on every settle
  // (debounced moveend), picking the metadata level that spans the
  // current zoom. Honesty: low zooms have no dated metadata (verified:
  // TerraColor NextGen carries no DATE) → "date unknown at this zoom",
  // never a fabricated or stale-implying value. Esri terms reading
  // (census §3 #9): a recency check displayed on the imagery it
  // describes — client-side only, nothing archived, no API route.
  const [imageryDate, setImageryDate] = useState<{ label: string; known: boolean }>(
    { label: "capture date: checking…", known: false });
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady || !enabled.imagery) return;
    let timer: number | undefined;
    let ctrl: AbortController | null = null;
    let gone = false;
    const lookup = async () => {
      try {
        ctrl?.abort();
        ctrl = new AbortController();
        const c = map.getCenter();
        const b = map.getBounds();
        const zoom = Math.round(map.getZoom());
        const el = map.getContainer();
        const url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/identify" +
          `?geometry=${c.lng.toFixed(5)},${c.lat.toFixed(5)}&geometryType=esriGeometryPoint&sr=4326&tolerance=1` +
          `&mapExtent=${b.getWest()},${b.getSouth()},${b.getEast()},${b.getNorth()}` +
          `&imageDisplay=${el.clientWidth || 400},${el.clientHeight || 400},96&layers=all&returnGeometry=false&f=json`;
        const r = await fetch(url, { signal: ctrl.signal });
        const d = await r.json();
        const hit = (d?.results || [])
          .map((x: any) => x?.attributes || {})
          .find((a: any) => {
            const raw = a["DATE (YYYYMMDD)"];
            if (!raw || !/^\d{8}$/.test(String(raw))) return false;
            const lo = parseInt(a.MinMapLevel, 10), hi = parseInt(a.MaxMapLevel, 10);
            return !(Number.isFinite(lo) && zoom < lo) && !(Number.isFinite(hi) && zoom > hi);
          });
        if (gone) return;
        if (hit) {
          const raw = String(hit["DATE (YYYYMMDD)"]);
          const iso = `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6, 8)}`;
          const src = hit.SOURCE ? ` · ${hit.SOURCE}` : "";
          setImageryDate({ label: `imagery at centre: ${iso}${src}`, known: true });
        } else {
          setImageryDate({ label: "capture date unknown at this zoom", known: false });
        }
      } catch {
        // transport/abort: keep a known value; never fabricate one
        if (!gone) setImageryDate((v) => (v.known ? v : { label: "capture date unknown", known: false }));
      }
    };
    const onMove = () => { window.clearTimeout(timer); timer = window.setTimeout(lookup, 1200); };
    map.on("moveend", onMove);
    onMove();
    return () => {
      gone = true;
      map.off("moveend", onMove);
      window.clearTimeout(timer);
      ctrl?.abort();
    };
  }, [mapReady, enabled.imagery]);

  // ── terrain mesh + hillshade + drained-ocean mesh (RAW). ONE effect owns
  // map.setTerrain so the terrain and seafloor toggles can never race over
  // the mesh. Sources: Mapterhorn terrarium DEM (land; geospatial Tier-1(a),
  // licensing register 2026-07-04: commercial-OK, © Mapterhorn attribution
  // via TileJSON) and the AWS Terrain Tiles terrarium set (ETOPO1 bathymetry
  // baked in) for the drain — real soundings + satellite-gravity
  // interpolation, global coverage, labeled as an estimate.
  // Round 8 realism (human: "real 3d terrain… like the example pic; the
  // [drain] toggle — use radar-mapped ocean data"):
  //  · drain ON → the mesh deforms from the bathymetric DEM, so ocean basins
  //    PHYSICALLY sink instead of only tinting (land relief rides along from
  //    the same set — SRTM-class there; Mapterhorn resumes when drain is off)
  //  · the stylized blue-dark hillshade paints ONLY on the dark bases (night/
  //    minimal) where it is the sole relief cue — over photo imagery it made
  //    real mountains look like a tinted map, so imagery presets now show
  //    the photo draped on the true displacement + a sky/fog horizon
  // exaggeration 1.3 everywhere: the aircraft layer + 3D trails match their
  // altitudes to this exact constant (setAltScale coupling). Degrade-safe:
  // any failure keeps the base map alive. ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    const imageryVisible = mapPreset === "natural" || mapPreset === "terrain";
    const meshSource = enabled.seafloor ? "ocean-terrain-dem" : enabled.terrain ? "terrain-dem" : null;
    try {
      if (enabled.terrain && !map.getSource("terrain-dem")) {
        map.addSource("terrain-dem", {
          type: "raster-dem",
          url: "https://tiles.mapterhorn.com/tilejson.json",
          encoding: "terrarium",
        } as any);
      }
      if (enabled.seafloor && !map.getSource("ocean-terrain-dem")) {
        // own source for the MESH — the seafloor tint keeps its separate
        // seafloor-dem source (same tiles, shared HTTP cache): a source
        // consumed by setTerrain is treated specially by MapLibre (session
        // #1 finding), so mesh and paint never share one
        map.addSource("ocean-terrain-dem", {
          type: "raster-dem",
          tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"],
          encoding: "terrarium",
          tileSize: 256,
          maxzoom: 15,
          attribution: "Bathymetry: NOAA ETOPO1 · Terrain Tiles (Mapzen, AWS Open Data)",
        } as any);
      }
      map.setTerrain(meshSource ? ({ source: meshSource, exaggeration: 1.3 } as any) : null);
    } catch {
      if (enabled.terrain) setStatus("terrain", "error");
    }
    // hillshade: rebuild each pass (source may swap with the drain) — dark
    // bases only; inserted beneath the lowest data layer so shading never
    // covers markers or velocity vectors
    try { if (map.getLayer("terrain-hillshade")) map.removeLayer("terrain-hillshade"); } catch {}
    if (enabled.terrain && !imageryVisible) {
      try {
        const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
        map.addLayer({
          id: "terrain-hillshade", type: "hillshade", source: meshSource ?? "terrain-dem",
          paint: {
            "hillshade-exaggeration": 0.45,
            "hillshade-shadow-color": "rgba(5,10,19,0.9)",
            "hillshade-highlight-color": "rgba(238,243,251,0.25)",
            "hillshade-accent-color": "rgba(77,159,255,0.15)",
          },
        } as any, firstMarker?.id);
      } catch {}
    }
    // sky/atmosphere — ONE always-on setSky (lib/globeAtmosphere, verified
    // against the installed v5.24 shaders): the globe limb glow is
    // hardcoded Rayleigh/Mie physics gated only by atmosphere-blend
    // (zoom-faded, pass skipped at 0 — free at street zooms), and the
    // horizon/fog colors only render on the mercator/pitched side, so a
    // single spec covers the hero globe AND tilted terrain with no
    // conditional churn. Presentation, not data: the rim is a physical
    // render of Earth's atmosphere, not a measurement. ADAPTIVE TIER:
    // software GL (SwiftShader/llvmpipe — VMs, the perf harness) can't
    // afford the scattering pass (~156ms/frame measured) → those
    // renderers get the sky with atmosphere-blend pinned 0.
    try {
      const glc: any = map.getCanvas().getContext("webgl2") || map.getCanvas().getContext("webgl");
      const dbg = glc?.getExtension?.("WEBGL_debug_renderer_info");
      const renderer = glc ? String(glc.getParameter(dbg ? dbg.UNMASKED_RENDERER_WEBGL : glc.RENDERER) ?? "") : "";
      (map as any).setSky?.(skyForRenderer(renderer) as any);
    } catch {}
    if (!enabled.terrain) setStatus("terrain", "off");
    else setStatus("terrain", "active", undefined, enabled.seafloor
      ? "3D relief from the drained-ocean DEM — basins sink for real; land is SRTM-class while the drain is on (© Mapterhorn set resumes when it's off)"
      : imageryVisible
        ? "true 3D relief — imagery draped on the Copernicus GLO-30 mesh + sky horizon (© Mapterhorn); tilt the map to see it"
        : "3D relief + hillshade — Copernicus GLO-30 + national DEMs (© Mapterhorn)");
    // keep the map lean when off (mesh + hillshade already detached above)
    if (!enabled.terrain) { try { if (map.getSource("terrain-dem")) map.removeSource("terrain-dem"); } catch {} }
    if (!enabled.seafloor) { try { if (map.getSource("ocean-terrain-dem")) map.removeSource("ocean-terrain-dem"); } catch {} }
  }, [enabled.terrain, enabled.seafloor, mapPreset, mapReady, setStatus]);

  // ── seafloor bathymetry (RAW; EARTH TWIN E2-1 — "drain the ocean" v1,
  // research/earth_twin_program.md V4). NOAA ETOPO1 ocean depths via the open
  // Terrain Tiles bucket (terrarium raster-dem with bathymetry baked in at
  // every zoom — verified live: the z0 tile's imagery-sources header names
  // ETOPO1), drawn as a color-relief depth tint that is TRANSPARENT at and
  // above sea level: toggling swaps the sea surface for seafloor relief while
  // land imagery stays untouched. Its OWN raster-dem source — never reuses
  // terrain-dem (that source is land-only and feeds setTerrain; MapLibre
  // treats terrain-owned sources specially) and never calls setTerrain.
  // E2 v2 QUALITY (human's Google Earth reference): the ridge texture is the
  // GEBCO_2024 shaded-relief WMS raster (15 arc-sec — 4x the old ETOPO1
  // hillshade), drawn UNDER the legend-bearing depth tint (fixed 0.25) so
  // the legend chips stay the one source of truth for OUR palette; NOAA
  // ETOPO 2022 hillshade swaps in if the GEBCO WMS errors (degrade, never
  // break). Licensing + fetch evidence: research/ocean_quality_notes.md.
  // HONESTY: soundings + satellite-gravity interpolation, indicative, not
  // navigational; on-screen colors are dominated by GEBCO's own depth
  // palette; LAND shows GEBCO's hypsometric tint while drained (a raster
  // cannot be elevation-masked); per-cell TID confidence is the charter's
  // next E2 slice. E2 v2 DEPTH BLEND (2026-07-16, the WIRING RECIPE's other
  // half, left unclaimed by the seafloor_confidence PR): each
  // SEAFLOOR_V2_REGIONS entry's OWN native-15-arc-sec `demUrl` is layered
  // ON TOP of the v1 global ETOPO1 tint using the SAME bathymetryColorRelief()
  // ramp (one legend, two resolutions) — the pmtiles archive only carries
  // tiles inside its committed bbox, so outside the region there is no tile
  // to draw and the v1 relief beneath shows through unchanged (never a
  // guessed blend, never a visible seam beyond the bbox edge). Own
  // source/layer ids per region so toggling "seafloor" adds/removes both
  // resolutions atomically; nothing here touches seafloor_confidence's TID
  // sources. raster-resampling "nearest" is deliberately NOT set — this is a
  // depth-color ramp already smoothed by bathymetryColorRelief()'s stops
  // (unlike the TID class raster, cross-pixel interpolation here can't
  // produce a wrong discrete class, only a slightly softer gradient). ──
  const oceanBasemapErrRef = useRef<((e: any) => void) | null>(null);
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.seafloor) {
      try {
        if (oceanBasemapErrRef.current) { map.off("error", oceanBasemapErrRef.current); oceanBasemapErrRef.current = null; }
        for (const r of SEAFLOOR_V2_REGIONS) {
          if (map.getLayer(`seafloor-relief-v2-${r.name}`)) map.removeLayer(`seafloor-relief-v2-${r.name}`);
          if (map.getSource(`seafloor-dem-v2-${r.name}`)) map.removeSource(`seafloor-dem-v2-${r.name}`);
        }
        if (map.getLayer("seafloor-relief")) map.removeLayer("seafloor-relief");
        if (map.getLayer(OCEAN_BASEMAP_LAYER_ID)) map.removeLayer(OCEAN_BASEMAP_LAYER_ID);
        if (map.getSource("seafloor-dem")) map.removeSource("seafloor-dem");
        if (map.getSource(OCEAN_BASEMAP_SOURCE_ID)) map.removeSource(OCEAN_BASEMAP_SOURCE_ID);
      } catch {}
      setStatus("seafloor", "off");
      return;
    }
    try {
      if (!map.getSource("seafloor-dem")) {
        map.addSource("seafloor-dem", {
          type: "raster-dem",
          tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"],
          encoding: "terrarium",
          tileSize: 256,
          maxzoom: 15,
          attribution: "Bathymetry: NOAA ETOPO1 · Terrain Tiles (Mapzen, AWS Open Data)",
        } as any);
      }
      if (!map.getSource(OCEAN_BASEMAP_SOURCE_ID)) {
        map.addSource(OCEAN_BASEMAP_SOURCE_ID, oceanBasemapSource() as any);
      }
      if (!map.getLayer("seafloor-relief")) {
        // under all marker layers, same anchor rule as every raster overlay —
        // and DETERMINISTICALLY below terrain-hillshade when terrain is on
        // (both share the firstMarker anchor, so without this the stacking
        // depended on toggle order; review finding, session #1)
        const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
        const beforeId = map.getLayer("terrain-hillshade") ? "terrain-hillshade" : firstMarker?.id;
        map.addLayer({
          id: "seafloor-relief", type: "color-relief", source: "seafloor-dem",
          paint: {
            "color-relief-color": bathymetryColorRelief(),
            "color-relief-opacity": 0.25,
          },
        } as any, beforeId);
        // GEBCO shaded relief directly BELOW the tint; the opacity slider
        // drives this raster (FIELD_MAP_LAYER), default 100
        map.addLayer(oceanBasemapLayer(opacityOf("seafloor")) as any, "seafloor-relief");
      }
      // v2 regional depth blend: native-resolution GEBCO DEM per committed
      // region, stacked ABOVE the v1 global tint (same anchor rule, added
      // after so it lands on top) — same ramp, so the only visible change
      // inside a region's bbox is sharper relief, never a different palette.
      {
        const firstMarkerV2 = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
        const beforeIdV2 = map.getLayer("terrain-hillshade") ? "terrain-hillshade" : firstMarkerV2?.id;
        for (const r of SEAFLOOR_V2_REGIONS) {
          const sid = `seafloor-dem-v2-${r.name}`;
          const lid = `seafloor-relief-v2-${r.name}`;
          if (!map.getSource(sid)) {
            map.addSource(sid, {
              type: "raster-dem",
              url: `pmtiles://${window.location.origin}${r.demUrl}`,
              encoding: r.encoding,
              tileSize: r.tileSize,
              minzoom: r.minzoom,
              maxzoom: r.maxzoom,
              attribution: GEBCO_ATTRIBUTION,
            } as any);
          }
          if (!map.getLayer(lid)) {
            map.addLayer({
              id: lid, type: "color-relief", source: sid,
              paint: {
                "color-relief-color": bathymetryColorRelief(),
                "color-relief-opacity": 0.25,
              },
            } as any, beforeIdV2);
          }
        }
      }
      if (!oceanBasemapErrRef.current) {
        // one-shot degrade to the NOAA public-domain hillshade if the GEBCO
        // WMS errors — the status note stays honest about which source is live
        const onErr = (e: any) => {
          if (e?.sourceId !== OCEAN_BASEMAP_SOURCE_ID) return;
          try { map.off("error", onErr); } catch {}
          if (oceanBasemapErrRef.current === onErr) oceanBasemapErrRef.current = null;
          try {
            if (map.getLayer(OCEAN_BASEMAP_LAYER_ID)) map.removeLayer(OCEAN_BASEMAP_LAYER_ID);
            if (map.getSource(OCEAN_BASEMAP_SOURCE_ID)) map.removeSource(OCEAN_BASEMAP_SOURCE_ID);
            map.addSource(OCEAN_BASEMAP_SOURCE_ID, oceanBasemapFallbackSource() as any);
            map.addLayer(oceanBasemapLayer(opacityOf("seafloor")) as any, "seafloor-relief");
            setStatus("seafloor", "active", undefined,
              "ocean drained — NOAA ETOPO 2022 hillshade (fallback; GEBCO WMS unreachable) + depth tint + 3D basins; soundings + gravity interpolation, not navigational");
          } catch {}
        };
        oceanBasemapErrRef.current = onErr;
        map.on("error", onErr);
      }
      setStatus("seafloor", "active", undefined,
        `ocean drained — GEBCO_2024 shaded relief (15 arc-sec; real soundings + satellite-gravity interpolation, not navigational) + depth tint + 3D basins · land shows GEBCO's tint while drained · native-resolution GEBCO_2026 depth blend over ${SEAFLOOR_V2_REGIONS.map((r) => r.name).join(", ")} (elsewhere the global ETOPO1 relief above applies)`);
    } catch {
      setStatus("seafloor", "error");
    }
  }, [enabled.seafloor, mapReady, setStatus]);

  // Measured TID group shares per region — read from each region's own
  // provenance sidecar (never a hardcoded quote): "the shipped Mariana demo
  // region measures 65.9% direct / 34.1% predicted — computed from the
  // data, never quoted" (layers.json's own description of this layer).
  const [seafloorConfShares, setSeafloorConfShares] = useState<Record<string, Record<string, number>>>({});

  // ── seafloor mapping confidence (RAW; EARTH TWIN E2 v2 wiring —
  // research/earth_twin_program.md RESUME STATE 2026-07-16: "the honesty-as-
  // hero moment"). GEBCO_2026 TID grid, per-cell measured (direct soundings/
  // lidar/seismic) vs predicted (satellite-gravity/interpolated) vs unknown —
  // colors + legend come from the ONE decode table (lib/seafloorV2, mirrors
  // datacore/gebco/tid_decode.json verbatim; no re-grouping here). Own
  // raster-dem source per committed region (today: Mariana Trench only —
  // SEAFLOOR_V2_REGIONS is the pipeline's scale-by-region list; looping it
  // means a future region needs zero code here, only a new array entry + a
  // pipeline run) — never touches the depth "seafloor" layer's sources, so
  // the two toggles can never race over shared state. Coverage is regional
  // and said so on the panel + legend; nothing outside a committed bbox
  // renders (pmtiles returns no tile there — absence, never a guessed
  // class). ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.seafloor_confidence) {
      try {
        for (const r of SEAFLOOR_V2_REGIONS) {
          const lid = `seafloor-confidence-relief-${r.name}`;
          const sid = `seafloor-tid-${r.name}`;
          if (map.getLayer(lid)) map.removeLayer(lid);
          if (map.getLayer(`${lid}-extent`)) map.removeLayer(`${lid}-extent`);
          if (map.getSource(sid)) map.removeSource(sid);
          if (map.getSource(`${sid}-extent`)) map.removeSource(`${sid}-extent`);
        }
      } catch {}
      setStatus("seafloor_confidence", "off");
      return;
    }
    try {
      const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
      for (const r of SEAFLOOR_V2_REGIONS) {
        const lid = `seafloor-confidence-relief-${r.name}`;
        const sid = `seafloor-tid-${r.name}`;
        if (!map.getSource(sid)) {
          map.addSource(sid, {
            type: "raster-dem",
            url: `pmtiles://${window.location.origin}${r.tidUrl}`,
            encoding: r.encoding,
            tileSize: r.tileSize,
          } as any);
        }
        if (!map.getLayer(lid)) {
          map.addLayer({
            id: lid, type: "color-relief", source: sid,
            paint: {
              "color-relief-color": tidConfidenceColorRelief(),
              "color-relief-opacity": opacityOf("seafloor_confidence") / 100,
            },
          } as any, firstMarker?.id);
        }
        // CLOSED COVERAGE BORDER (human-directed 2026-07-16: "just have a
        // closed border of the area") — the region's true bbox drawn as a
        // dashed ring, so partial coverage is visible on the map itself,
        // not only stated in panel/legend text.
        const [w, s, e, n] = r.bbox;
        if (!map.getSource(`${sid}-extent`)) {
          map.addSource(`${sid}-extent`, {
            type: "geojson",
            data: {
              type: "Feature", properties: { region: r.name },
              geometry: { type: "LineString", coordinates: [[w, s], [e, s], [e, n], [w, n], [w, s]] },
            },
          } as any);
        }
        if (!map.getLayer(`${lid}-extent`)) {
          map.addLayer({
            id: `${lid}-extent`, type: "line", source: `${sid}-extent`,
            paint: {
              "line-color": "#8ab8ff", "line-width": 1.4,
              "line-dasharray": [2, 2], "line-opacity": 0.8,
            },
          } as any);
        }
      }
      setStatus("seafloor_confidence", "active", undefined,
        `regional coverage only (${SEAFLOOR_V2_REGIONS.map((r) => r.name).join(", ")}) — everywhere else renders transparent, never a guessed reading · ${GEBCO_NOT_FOR_NAVIGATION}`);
    } catch {
      setStatus("seafloor_confidence", "error");
    }
    // measured shares for the legend — fetched from the pipeline's own
    // provenance sidecar, one per region, never hardcoded (see the state
    // declaration above)
    let gone = false;
    for (const r of SEAFLOOR_V2_REGIONS) {
      fetch(r.provenanceUrl).then((res) => res.json()).then((p) => {
        if (gone) return;
        const shares = p?.tid?.group_share_of_covered_cells;
        if (shares) setSeafloorConfShares((s) => ({ ...s, [r.name]: shares }));
      }).catch(() => {});
    }
    return () => { gone = true; };
  }, [enabled.seafloor_confidence, mapReady, setStatus]);

  // ── surface water (RAW; JRC Global Surface Water v2021 — atlas-parity
  // layer 1, licensing per open_questions ATLAS PARITY: free with EC
  // JRC/Google attribution + Pekel et al. 2016 citation. STATIC dataset —
  // 1984–2021 occurrence, stated in the status note per the imagery-date
  // honesty rule. Tiles direct from the JRC public bucket: zero server
  // cost, zero key. field:true — opacity slider inherited.) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.surfacewater) {
      try {
        if (map.getLayer("gsw-occurrence")) map.removeLayer("gsw-occurrence");
        if (map.getSource("gsw-occurrence")) map.removeSource("gsw-occurrence");
      } catch {}
      setStatus("surfacewater", "off");
      return;
    }
    try {
      if (!map.getSource("gsw-occurrence")) {
        map.addSource("gsw-occurrence", {
          type: "raster",
          tiles: ["https://storage.googleapis.com/global-surface-water/tiles2021/occurrence/{z}/{x}/{y}.png"],
          tileSize: 256, maxzoom: 13,
          attribution: "Surface water © EC JRC/Google",
        } as any);
      }
      if (!map.getLayer("gsw-occurrence")) {
        const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
        map.addLayer({
          id: "gsw-occurrence", type: "raster", source: "gsw-occurrence",
          paint: { "raster-opacity": opacityOf("surfacewater") / 100 },
        } as any, firstMarker?.id);
      }
      setStatus("surfacewater", "active", undefined,
        "water occurrence 1984–2021 (static, v2021) · EC JRC/Google, Pekel et al. 2016");
    } catch {
      setStatus("surfacewater", "error");
    }
  }, [enabled.surfacewater, mapReady, setStatus]);

  // ── forest cover (RAW; JRC Global Forest Cover 2020 — atlas-parity
  // layer 2, licensing per open_questions ATLAS PARITY: CC BY 4.0, tiles
  // via the GFW public tile API. STATIC 2020 vintage stated per the
  // imagery-date honesty rule. Zero server cost, zero key. field:true —
  // opacity slider inherited.) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.forest) {
      try {
        if (map.getLayer("jrc-forest")) map.removeLayer("jrc-forest");
        if (map.getSource("jrc-forest")) map.removeSource("jrc-forest");
      } catch {}
      setStatus("forest", "off");
      return;
    }
    try {
      if (!map.getSource("jrc-forest")) {
        map.addSource("jrc-forest", {
          type: "raster",
          tiles: ["https://tiles.globalforestwatch.org/jrc_global_forest_cover/latest/dynamic/{z}/{x}/{y}.png"],
          tileSize: 256, maxzoom: 12,
          attribution: "Forest cover © EC JRC (CC BY 4.0), tiles by GFW",
        } as any);
      }
      if (!map.getLayer("jrc-forest")) {
        const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
        map.addLayer({
          id: "jrc-forest", type: "raster", source: "jrc-forest",
          paint: { "raster-opacity": opacityOf("forest") / 100 },
        } as any, firstMarker?.id);
      }
      setStatus("forest", "active", undefined,
        "forest extent 2020, 10m (static) · EC JRC GFC2020 · tiles via Global Forest Watch");
    } catch {
      setStatus("forest", "error");
    }
  }, [enabled.forest, mapReady, setStatus]);

  // ── night lights (RAW; worldview_globe.md Phase G2a — the first NASA
  // GIBS layer, via the shared gibs.ts factory. DATED daily layer (unlike
  // surfacewater/forest above): re-adds the source/layer whenever the
  // scrubbed date changes, which is fine at human click cadence (no
  // polling). Tile access verified live 2026-07-08 (Level8 is the correct
  // TileMatrixSet for this layer; Level9 is explicitly rejected by GIBS,
  // not a network fluke — see gibs.ts header). field:true — opacity slider
  // inherited; the date scrubber is this layer's own extra control,
  // rendered next to it the same way weather_temp/weather_wind add theirs.) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.nightlights) {
      try {
        if (map.getLayer("gibs-nightlights")) map.removeLayer("gibs-nightlights");
        if (map.getSource("gibs-nightlights")) map.removeSource("gibs-nightlights");
      } catch {}
      setStatus("nightlights", "off");
      return;
    }
    try {
      if (map.getLayer("gibs-nightlights")) map.removeLayer("gibs-nightlights");
      if (map.getSource("gibs-nightlights")) map.removeSource("gibs-nightlights");
      const url = gibsTileUrl(
        { layer: "VIIRS_SNPP_DayNightBand_At_Sensor_Radiance", tileMatrixSet: "GoogleMapsCompatible_Level8", ext: "png" },
        nightlightsDate,
      );
      map.addSource("gibs-nightlights", {
        type: "raster", tiles: [url], tileSize: 256, maxzoom: 8,
        attribution: "Night lights radiance · VIIRS/SNPP · NASA GIBS/ESDIS (public domain)",
      } as any);
      const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
      map.addLayer({
        id: "gibs-nightlights", type: "raster", source: "gibs-nightlights",
        paint: { "raster-opacity": opacityOf("nightlights") / 100 },
      } as any, firstMarker?.id);
      setStatus("nightlights", "active", undefined,
        `radiance for ${nightlightsDate} (UTC) · NASA GIBS/ESDIS — some dates/areas may render blank ` +
        `(daylight side of the terminator, sensor gaps); step back a day if so`);
    } catch {
      setStatus("nightlights", "error");
    }
  }, [enabled.nightlights, nightlightsDate, mapReady, setStatus]);

  // ── aerosol optical depth (RAW; worldview_globe.md Phase G2c — NASA GIBS
  // via the shared gibs.ts factory, same DATED-daily pattern as night lights.
  // Tile access verified live 2026-07-08 against GIBS GetCapabilities:
  // MODIS_Combined_Value_Added_AOD is PNG at GoogleMapsCompatible_Level6
  // (max native zoom 6); a real yesterday tile pixel-checked non-blank (21%
  // coverage over land). field:true — opacity slider inherited; date scrubber
  // is this layer's own extra control, rendered next to it like night lights. ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.aerosol) {
      try {
        if (map.getLayer("gibs-aerosol")) map.removeLayer("gibs-aerosol");
        if (map.getSource("gibs-aerosol")) map.removeSource("gibs-aerosol");
      } catch {}
      setStatus("aerosol", "off");
      return;
    }
    try {
      if (map.getLayer("gibs-aerosol")) map.removeLayer("gibs-aerosol");
      if (map.getSource("gibs-aerosol")) map.removeSource("gibs-aerosol");
      const url = gibsTileUrl(
        { layer: "MODIS_Combined_Value_Added_AOD", tileMatrixSet: "GoogleMapsCompatible_Level6", ext: "png" },
        aerosolDate,
      );
      map.addSource("gibs-aerosol", {
        type: "raster", tiles: [url], tileSize: 256, maxzoom: 6,
        attribution: "Aerosol optical depth · MODIS (Terra+Aqua) · NASA GIBS/ESDIS (public domain)",
      } as any);
      const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
      map.addLayer({
        id: "gibs-aerosol", type: "raster", source: "gibs-aerosol",
        paint: { "raster-opacity": opacityOf("aerosol") / 100 },
      } as any, firstMarker?.id);
      setStatus("aerosol", "active", undefined,
        `aerosol optical depth for ${aerosolDate} (UTC) · NASA GIBS/ESDIS — blank over cloud/glint/` +
        `high-albedo areas is the retrieval's honest coverage; step back a day if a region is empty`);
    } catch {
      setStatus("aerosol", "error");
    }
  }, [enabled.aerosol, aerosolDate, mapReady, setStatus]);

  // ── vegetation health (RAW; worldview_globe.md Phase G2e — NASA GIBS via
  // the shared gibs.ts factory, same DATED pattern. VIIRS_SNPP_NDVI_8Day is
  // an 8-DAY COMPOSITE, PNG at GoogleMapsCompatible_Level8; access + non-blank
  // land coverage verified live 2026-07-08 (US/N.America yesterday tile 41%
  // coverage — vegetation is land-only, so ocean tiles are legitimately
  // transparent). field:true opacity slider + own date scrubber, like above. ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.vegetation) {
      try {
        if (map.getLayer("gibs-vegetation")) map.removeLayer("gibs-vegetation");
        if (map.getSource("gibs-vegetation")) map.removeSource("gibs-vegetation");
      } catch {}
      setStatus("vegetation", "off");
      return;
    }
    try {
      if (map.getLayer("gibs-vegetation")) map.removeLayer("gibs-vegetation");
      if (map.getSource("gibs-vegetation")) map.removeSource("gibs-vegetation");
      const url = gibsTileUrl(
        { layer: "VIIRS_SNPP_NDVI_8Day", tileMatrixSet: "GoogleMapsCompatible_Level8", ext: "png" },
        vegetationDate,
      );
      map.addSource("gibs-vegetation", {
        type: "raster", tiles: [url], tileSize: 256, maxzoom: 8,
        attribution: "Vegetation index (NDVI, 8-day) · VIIRS/SNPP · NASA GIBS/ESDIS (public domain)",
      } as any);
      const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
      map.addLayer({
        id: "gibs-vegetation", type: "raster", source: "gibs-vegetation",
        paint: { "raster-opacity": opacityOf("vegetation") / 100 },
      } as any, firstMarker?.id);
      setStatus("vegetation", "active", undefined,
        `NDVI 8-day composite ending ${vegetationDate} (UTC) · NASA GIBS/ESDIS — greener = denser/` +
        `healthier vegetation; ocean and barren areas are legitimately blank (land-only index)`);
    } catch {
      setStatus("vegetation", "error");
    }
  }, [enabled.vegetation, vegetationDate, mapReady, setStatus]);

  // ── root-zone soil moisture (RAW; worldview_globe.md Phase G2d — NASA GIBS
  // SMAP L4 via the shared gibs.ts factory. PNG at GoogleMapsCompatible_Level6.
  // ~6-day processing lag (SOIL_LATENCY_DAYS), so the date defaults 7 days back
  // rather than "yesterday" — verified live 2026-07-08 (07-01/07-02 carry data,
  // 07-07 does not). Land-only, like NDVI; ocean tiles legitimately blank. ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.soilmoisture) {
      try {
        if (map.getLayer("gibs-soilmoisture")) map.removeLayer("gibs-soilmoisture");
        if (map.getSource("gibs-soilmoisture")) map.removeSource("gibs-soilmoisture");
      } catch {}
      setStatus("soilmoisture", "off");
      return;
    }
    try {
      if (map.getLayer("gibs-soilmoisture")) map.removeLayer("gibs-soilmoisture");
      if (map.getSource("gibs-soilmoisture")) map.removeSource("gibs-soilmoisture");
      const url = gibsTileUrl(
        { layer: "SMAP_L4_Analyzed_Root_Zone_Soil_Moisture", tileMatrixSet: "GoogleMapsCompatible_Level6", ext: "png" },
        soilmoistureDate,
      );
      map.addSource("gibs-soilmoisture", {
        type: "raster", tiles: [url], tileSize: 256, maxzoom: 6,
        attribution: "Root-zone soil moisture · SMAP L4 · NASA GIBS/ESDIS (public domain)",
      } as any);
      const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
      map.addLayer({
        id: "gibs-soilmoisture", type: "raster", source: "gibs-soilmoisture",
        paint: { "raster-opacity": opacityOf("soilmoisture") / 100 },
      } as any, firstMarker?.id);
      setStatus("soilmoisture", "active", undefined,
        `SMAP L4 root-zone soil moisture for ${soilmoistureDate} (UTC, ~6-day lag) · NASA GIBS/ESDIS — ` +
        `wetter soils darker/bluer; ocean is legitimately blank (land soil index)`);
    } catch {
      setStatus("soilmoisture", "error");
    }
  }, [enabled.soilmoisture, soilmoistureDate, mapReady, setStatus]);

  // ── tropospheric NO2 (RAW; worldview_globe.md Phase G2g — NASA GIBS TROPOMI
  // via the shared gibs.ts factory. Daily, PNG at GoogleMapsCompatible_Level6;
  // access + non-blank field verified live 2026-07-08 (yesterday tile 100% over
  // N.America — a continuous column-density field: ocean legitimately LOW/dark,
  // industrial zones/ports HIGH. The charter's "genuinely differentiated" layer:
  // NO2 as a real-time combustion/throughput nowcast). ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.no2) {
      try {
        if (map.getLayer("gibs-no2")) map.removeLayer("gibs-no2");
        if (map.getSource("gibs-no2")) map.removeSource("gibs-no2");
      } catch {}
      setStatus("no2", "off");
      return;
    }
    try {
      if (map.getLayer("gibs-no2")) map.removeLayer("gibs-no2");
      if (map.getSource("gibs-no2")) map.removeSource("gibs-no2");
      const url = gibsTileUrl(
        { layer: "TROPOMI_L2_Nitrogen_Dioxide_Tropospheric_Column", tileMatrixSet: "GoogleMapsCompatible_Level6", ext: "png" },
        no2Date,
      );
      map.addSource("gibs-no2", {
        type: "raster", tiles: [url], tileSize: 256, maxzoom: 6,
        attribution: "Tropospheric NO₂ column · Sentinel-5P/TROPOMI · NASA GIBS/ESDIS (public domain)",
      } as any);
      const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
      map.addLayer({
        id: "gibs-no2", type: "raster", source: "gibs-no2",
        paint: { "raster-opacity": opacityOf("no2") / 100 },
      } as any, firstMarker?.id);
      setStatus("no2", "active", undefined,
        `tropospheric NO₂ column for ${no2Date} (UTC) · Sentinel-5P/TROPOMI via NASA GIBS/ESDIS — ` +
        `redder = more NO₂ (industry/traffic); satellite swath gaps and cloud can leave stripes/blanks`);
    } catch {
      setStatus("no2", "error");
    }
  }, [enabled.no2, no2Date, mapReady, setStatus]);

  // ── fire/hotspot brightness temperature (RAW; worldview_globe.md Phase
  // G2b — NASA GIBS GOES-East ABI via the shared gibs.ts factory. Access +
  // non-blank field verified live 2026-07-10 (z=0 tile 99% non-transparent —
  // a continuous full-disk brightness field, not just discrete hotspots).
  // Genuinely different cadence class from every G2 layer above (~10-min,
  // irregular scan gaps, `GoogleMapsCompatible_Level7` max native zoom) —
  // no day scrubber; always requests GIBS's "default" (freshest scan) and
  // re-fetches every REFRESH_MS to pick up the next scan, since the server
  // marks every response no-store (verified: never safe to trust a stale
  // cached tile). COMPLEMENTS, does not replace, the existing NASA FIRMS
  // point-detection `fires` layer above: FIRMS gives discrete confirmed-fire
  // detections (~3h latency), this gives a continuous heat-intensity field
  // at ~10-min cadence — the fires×facilities cross-tie hypothesis already
  // filed against FIRMS (#388) is unchanged; this is a complementary visual,
  // not a new signal path. ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.firetemp) {
      try {
        if (map.getLayer("gibs-firetemp")) map.removeLayer("gibs-firetemp");
        if (map.getSource("gibs-firetemp")) map.removeSource("gibs-firetemp");
      } catch {}
      setStatus("firetemp", "off");
      return;
    }
    const FIRETEMP_SPEC = { layer: "GOES-East_ABI_FireTemp", tileMatrixSet: "GoogleMapsCompatible_Level7", ext: "png" as const };
    const REFRESH_MS = 5 * 60 * 1000; // ~10-min product cadence; poll at 2x rate
    let cancelled = false;
    const paint = () => {
      if (cancelled) return;
      try {
        if (map.getLayer("gibs-firetemp")) map.removeLayer("gibs-firetemp");
        if (map.getSource("gibs-firetemp")) map.removeSource("gibs-firetemp");
        const url = gibsTileUrl(FIRETEMP_SPEC, "default");
        map.addSource("gibs-firetemp", {
          type: "raster", tiles: [url], tileSize: 256, maxzoom: 7,
          attribution: "Fire/hotspot brightness temperature · GOES-East ABI · NASA GIBS/ESDIS (public domain)",
        } as any);
        const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
        map.addLayer({
          id: "gibs-firetemp", type: "raster", source: "gibs-firetemp",
          paint: { "raster-opacity": opacityOf("firetemp") / 100 },
        } as any, firstMarker?.id);
        setStatus("firetemp", "active", undefined,
          `brightness temperature, most recent GOES-East scan · NASA GIBS/ESDIS — brighter/hotter pixels ` +
          `indicate fire/thermal anomalies; coverage is GOES-East's fixed Americas+Atlantic scan domain, ` +
          `not global (blank tiles outside it are the honest domain edge, not an error)`);
      } catch {
        if (!cancelled) setStatus("firetemp", "error");
      }
    };
    const refreshScanTime = () => {
      gibsLatestScanTime(FIRETEMP_SPEC).then((t) => { if (!cancelled) setFiretempScanTime(t); });
    };
    paint();
    refreshScanTime();
    const timer = setInterval(() => { paint(); refreshScanTime(); }, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [enabled.firetemp, mapReady, setStatus]);

  // ── biomass (RAW; worldview_globe.md Phase G2h — NASA GIBS GEDI L4B
  // aboveground biomass density via the shared gibs.ts factory). Genuinely
  // STATIC unlike every G2 layer above: GEDI L4B is a single mission-life
  // mean composite (2019-04 through 2023-03), not a daily/8-day product, so
  // there is no scrubber and no refresh interval — it mounts once and never
  // changes. Requests GIBS's own "default" token rather than hardcoding the
  // 2019-04-18 date the current identifier advertises, so this survives a
  // future GIBS-side rollover to a newer mission-life composite without a
  // code change (same "default" pattern as firetemp above). Access + real
  // (non-fabricated) field verified live this session: Amazon basin tile
  // 95% non-transparent with plausible biomass-density coloring, Pacific-
  // NW forest 99% non-transparent, open ocean ~0.04% non-transparent
  // (legitimately blank — GEDI is a land vegetation-structure product). ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.biomass) {
      try {
        if (map.getLayer("gibs-biomass")) map.removeLayer("gibs-biomass");
        if (map.getSource("gibs-biomass")) map.removeSource("gibs-biomass");
      } catch {}
      setStatus("biomass", "off");
      return;
    }
    try {
      if (map.getLayer("gibs-biomass")) map.removeLayer("gibs-biomass");
      if (map.getSource("gibs-biomass")) map.removeSource("gibs-biomass");
      const url = gibsTileUrl(
        { layer: "GEDI_ISS_L4B_Aboveground_Biomass_Density_Mean_201904-202303", tileMatrixSet: "GoogleMapsCompatible_Level7", ext: "png" },
        "default",
      );
      map.addSource("gibs-biomass", {
        type: "raster", tiles: [url], tileSize: 256, maxzoom: 7,
        attribution: "Aboveground biomass density (GEDI L4B, 2019-04 to 2023-03 mean) · NASA GIBS/ESDIS (public domain)",
      } as any);
      const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
      map.addLayer({
        id: "gibs-biomass", type: "raster", source: "gibs-biomass",
        paint: { "raster-opacity": opacityOf("biomass") / 100 },
      } as any, firstMarker?.id);
      setStatus("biomass", "active", undefined,
        "aboveground biomass density · GEDI L4B mission-life mean (2019-04 to 2023-03), static — " +
        "NASA GIBS/ESDIS — denser/greener shading = more standing forest carbon; GEDI is spaceborne " +
        "LiDAR limited to roughly ±51.6° latitude and land vegetation only, so poles, ocean, and " +
        "non-forested areas are legitimately blank, not a gap");
    } catch {
      setStatus("biomass", "error");
    }
  }, [enabled.biomass, mapReady, setStatus]);

  // ── day/night terminator (RAW; EARTH TWIN O6-7 tier 1 — COMPUTED
  // EPHEMERIS, no feed: Meeus low-precision series, display-grade accuracy
  // stated on the layer; shade recomputes every minute). SPRINT W1
  // (human-directed 2026-07-17): the cartoon ☀️/🌗 emoji DOM markers are
  // REMOVED — the real Sun/Moon/planets now render in the always-on
  // celestial sky (next effect), at true apparent size/position from real
  // ephemeris. The terminator shade stays: it is the on-map "night side
  // right now" readout. ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.daynight) {
      setStatus("daynight", "off");
      return;
    }
    let timer: number | undefined;
    const update = () => {
      try {
        const feat = nightPolygon(Date.now());
        const src: any = map.getSource("daynight");
        if (src) src.setData(feat as any);
        else {
          map.addSource("daynight", { type: "geojson", data: feat as any });
          const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
          map.addLayer({
            id: "daynight-shade", type: "fill", source: "daynight",
            paint: { "fill-color": "#020617", "fill-opacity": 0.38 },
          } as any, firstMarker?.id);
        }
        setStatus("daynight", "active", undefined,
          "computed ephemeris (display-grade) — shaded side is night NOW · the realistic Sun/Moon/planets render in the sky itself (always on, astronomy-engine)");
      } catch { /* style mid-swap — next tick retries */ }
    };
    update();
    timer = window.setInterval(update, 60_000);
    return () => {
      window.clearInterval(timer);
      try {
        if (map.getLayer("daynight-shade")) map.removeLayer("daynight-shade");
        if (map.getSource("daynight")) map.removeSource("daynight");
      } catch {}
    };
  }, [enabled.daynight, mapReady, setStatus]);

  // ── ALWAYS-ON celestial sky (SPRINT W1, human-directed 2026-07-17:
  // "real scale sun moon and planets just always on that look real").
  // client/src/lib/celestial/celestialSky.ts — real astronomy-engine
  // ephemeris, true apparent angular sizes (0.35° render floor for planets,
  // stated in the lib), real moon phase + earthshine, sun flare that grows
  // as the look direction approaches the sun. Observer = map center; the
  // sky aligns to the map camera each frame (bearing/pitch/fov), so bodies
  // appear when the camera tilts toward the horizon they're above. Bundle
  // note: dynamic import keeps astronomy-engine in its own chunk. GL
  // failure latches inside the lib and the page continues. ──
  useEffect(() => {
    const map = mapRef.current;
    const container = mapContainer.current;
    if (!map || !mapReady || !container) return;
    let disposed = false;
    let handle: any = null;
    (async () => {
      const { mountCelestialSky } = await import("@/lib/celestial/celestialSky");
      if (disposed) return;
      handle = mountCelestialSky(container, {
        getView: () => {
          const c = map.getCenter();
          // maplibre stores the vertical fov on the transform; normalize
          // radians vs degrees defensively (API differs across versions)
          const rawFov = (map as any).transform?.fov ?? 36.87;
          return {
            timeMs: Date.now(),
            observerLatDeg: c.lat,
            observerLonDeg: c.lng,
            lookAzDeg: map.getBearing(),
            lookElDeg: map.getPitch() - 90, // pitch 0 = straight down
            fovDeg: rawFov > 3.2 ? rawFov : (rawFov * 180) / Math.PI,
          };
        },
      });
      celestialRef.current = handle;
      (window as any).__vtCelestial = handle; // harness seam (prod-inert, like __vtMap)
      setCelestialReady(true);
    })();
    return () => {
      disposed = true;
      try { handle?.dispose(); } catch {}
      celestialRef.current = null;
      setCelestialReady(false);
      try { delete (window as any).__vtCelestial; } catch {}
    };
  }, [mapReady]);

  // ── celestial paths toggle (SPRINT W1: "you could turn on the path of
  // everything") — ecliptic + Moon/planet sky tracks on the same sky
  // projection, real ephemeris, default OFF. Pure GPU-line render inside
  // the sky canvas; zero cost while off. ──
  useEffect(() => {
    const h = celestialRef.current;
    if (!celestialReady || !h) { if (!enabled.celestial_paths) setStatus("celestial_paths", "off"); return; }
    h.setPathsVisible(!!enabled.celestial_paths);
    if (enabled.celestial_paths) {
      setStatus("celestial_paths", "active", undefined,
        "ecliptic + Moon/planet sky tracks — real ephemeris (astronomy-engine), frozen sidereal frame; segments below the horizon fade out");
    } else {
      setStatus("celestial_paths", "off");
    }
  }, [enabled.celestial_paths, celestialReady, setStatus]);

  // ── satellites (RAW; ORBITAL program O2 — live GP elements client-fetched
  // from CelesTrak, SGP4 propagated off-thread in a Web Worker, drawn as
  // GPU-instanced points on the globe with LEO/MEO/GEO altitude shells. HEAVY
  // + off by default → zero-cost-when-off: the worker + layer only exist while
  // the toggle is on. REAL positions only — since the SDP4 port (O6-5) the
// deep-space population (GEO comms,
  // MEO nav) need SDP4 the near-earth kernel lacks, so they are skipped and
  // COUNTED in the status note, never faked. __vtOrbitalGpFixture is a
  // prod-inert test seam so the render path is verifiable offline. ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;

    const teardown = () => {
      try { satWorkerRef.current?.postMessage({ type: "stop" }); } catch {}
      try { satWorkerRef.current?.terminate(); } catch {}
      satWorkerRef.current = null;
      try {
        const layer = satLayerRef.current;
        if (layer && map.getLayer(layer.id)) map.removeLayer(layer.id);
      } catch {}
      satLayerRef.current = null;
      orbitalGpRef.current = null;
      satFollowRef.current = null;
      try {
        const model = satModelLayerRef.current;
        if (model && map.getLayer(model.id)) map.removeLayer(model.id);
      } catch {}
      satModelLayerRef.current = null;
      try {
        const arcs = satArcLayerRef.current;
        if (arcs && map.getLayer(arcs.id)) map.removeLayer(arcs.id);
      } catch {}
      satArcLayerRef.current = null;
      stopSatFocusRef.current = null;
      // O6-3: masks are index-aligned to THIS gp load — never survive it
      satGroupMaskRef.current = null;
      satGroupInfoRef.current = null;
      satRawPosRef.current = null;
      satRepushRef.current = null;
      setSatGroup(null);
      setSatGroupOrbits(false);
      setSatGroupCount(null);
      try {
        // legacy ground-point ring (pre-O6-ring-fix sessions) — clean up if present
        if (map.getLayer("sat-focus-ring")) map.removeLayer("sat-focus-ring");
        if (map.getSource("sat-focus")) map.removeSource("sat-focus");
        if (map.getLayer("sat-nadir-pt")) map.removeLayer("sat-nadir-pt");
        if (map.getSource("sat-nadir")) map.removeSource("sat-nadir");
      } catch {}
      setSatFollowing(false);
    };

    // ── ORBITAL O5 slice 1: FOLLOW — each worker tick re-centers on the
    // followed object's fresh SGP4 position and moves the focus ring.
    // Camera eases 800ms (under the 1s tick) so tracking reads smooth;
    // easing is skipped while the map is already moving (a user gesture or
    // a running ease — dragstart below is the explicit hand-back). ──
    const followTick = () => {
      const f = satFollowRef.current;
      if (!f) return;
      const t = followTarget(satLayerRef.current?.getPositions() ?? null, f.index);
      // O5-2b + O6 ring fix: the 3D form AND the focus ring both ride the
      // anchor (rendered at true altitude — no more ground-point parallax);
      // null sentinel tick = both hidden, never a guessed placement.
      satModelLayerRef.current?.setAnchor(
        t ? { mercX: t.mercX, mercY: t.mercY, altMeters: t.altKm * 1000 } : null);
      // O6 tools: the GROUND-SPOT marker — the exact point the object is
      // passing over right now (its nadir; a real geodetic fact).
      try {
        const nsrc: any = map.getSource("sat-nadir");
        if (showNadirRef.current && t) {
          const fc = { type: "FeatureCollection", features: [{ type: "Feature", geometry: { type: "Point", coordinates: [t.lonDeg, t.latDeg] }, properties: {} }] };
          if (nsrc) nsrc.setData(fc);
          else {
            map.addSource("sat-nadir", { type: "geojson", data: fc as any });
            map.addLayer({
              id: "sat-nadir-pt", type: "circle", source: "sat-nadir",
              paint: {
                "circle-radius": 5, "circle-color": "#ffd166",
                "circle-stroke-width": 2, "circle-stroke-color": "#0b1220",
                "circle-pitch-alignment": "map",
              },
            });
          }
        } else if (nsrc) {
          nsrc.setData({ type: "FeatureCollection", features: [] });
        }
      } catch { /* marker is chrome */ }
      if (!t) return; // sentinel this tick — ring cleared, camera stays put
      // O6-1: camera tracks only while locked — after a drag the focus
      // persists (arc + moving model) from anywhere on the globe.
      if (!f.lockMode) {
        // lock just released → hand the camera back to normal ground use
        try { (map as any).setCenterClampedToGround?.(true); (map as any).setCenterElevation?.(0); } catch {}
        return;
      }
      try {
        if (!map.isMoving()) {
          // MAP-NATIVE CRAFT ORBIT (human round 10, verbatim: "I dont want
          // it to be a separate thing it need to be part of the map"): the
          // camera CENTER itself is un-clamped from the ground and given
          // the craft's real altitude (maplibre v5 setCenterElevation +
          // setCenterClampedToGround(false)) — the center point IS the
          // craft, in the real map scene over real imagery. Every native
          // gesture now does the right thing with zero custom math:
          // rotate/tilt ORBIT the craft, zoom approaches THE CRAFT (no
          // more diving to the terrain under it — the old snap-back), and
          // there is no offset/feedback loop left to jerk or wander (this
          // replaces round 9's h·tan(pitch) analytic ground-center, which
          // was the best a ground-clamped camera could do). 'ground' mode
          // stays clamped and pins the nadir (watch the ground track).
          if (f.lockMode === "sat") {
            (map as any).setCenterClampedToGround?.(false);
            (map as any).setCenterElevation?.(t.altKm * 1000);
          } else {
            (map as any).setCenterClampedToGround?.(true);
            (map as any).setCenterElevation?.(0);
          }
          // linear + near-tick duration: the chase reads as one continuous
          // motion instead of hop-pause-hop (the "moves all around" feel)
          map.easeTo({ center: [t.lonDeg, t.latDeg], duration: 800, easing: (x: number) => x });
        }
      } catch {}
    };
    const stopFollow = () => {
      if (!satFollowRef.current) return;
      // ticks stop with the follow — restore the ground-clamped camera NOW
      try { (map as any).setCenterClampedToGround?.(true); (map as any).setCenterElevation?.(0); } catch {}
      satFollowRef.current = null;
      satModelLayerRef.current?.setAnchor(null); // model + ring vanish with the follow
      satArcLayerRef.current?.setArcs(null); // the one-object orbit arc goes with it
      setSatFollowing(false); // tools cluster goes with the focus
      setSatLockMode(null);
      try { (map.getSource("sat-nadir") as any)?.setData({ type: "FeatureCollection", features: [] }); } catch {}
    };
    // O6-1: the card's X (and layer teardown) end the focus completely
    stopSatFocusRef.current = stopFollow;
    // user drag = the user takes the CAMERA back; the focus itself persists
    // ("you can move away from the sat … until you press the X")
    const releaseCamera = () => {
      const f = satFollowRef.current;
      // 'sat' lock survives drags by design ("stay centered … until
      // unpressed"); only the ground lock hands the camera back on drag.
      if (f && f.lockMode === "ground") { f.lockMode = null; setSatLockMode(null); }
    };
    map.on("dragstart", releaseCamera);
    // D3 FIX (filed v1.0.370, instrumented 2026-07-18): MapLibre's
    // dragstart intermittently never fires for a real canvas drag — 2/24
    // drags in instrumentation produced pointerdown with ZERO map gesture
    // events, leaving the ground lock stuck fighting the user (the
    // reported "doesn't work to follow" feel). Detect the pan gesture at
    // the DOM level — primary-button press + >4px movement with the
    // button still held — independent of the handler pipeline. Click
    // semantics preserved (no movement = no release); rotate (right/
    // ctrl-drag) keeps the lock as before.
    const d3Canvas = map.getCanvas();
    let d3Down: { x: number; y: number } | null = null;
    const d3PointerDown = (e: PointerEvent) => {
      if (e.button === 0 && !e.ctrlKey) d3Down = { x: e.clientX, y: e.clientY };
    };
    const d3PointerMove = (e: PointerEvent) => {
      if (!d3Down) return;
      if (!(e.buttons & 1)) { d3Down = null; return; } // button no longer held
      if (Math.hypot(e.clientX - d3Down.x, e.clientY - d3Down.y) > 4) {
        d3Down = null;
        releaseCamera();
      }
    };
    const d3PointerUp = () => { d3Down = null; };
    d3Canvas.addEventListener("pointerdown", d3PointerDown);
    d3Canvas.addEventListener("pointermove", d3PointerMove);
    d3Canvas.addEventListener("pointerup", d3PointerUp);
    const d3Detach = () => {
      d3Canvas.removeEventListener("pointerdown", d3PointerDown);
      d3Canvas.removeEventListener("pointermove", d3PointerMove);
      d3Canvas.removeEventListener("pointerup", d3PointerUp);
    };

    if (!enabled["orbital_sats"]) {
      teardown();
      setStatus("orbital_sats", "off");
      return;
    }

    setStatus("orbital_sats", "loading", undefined, "fetching orbital elements (CelesTrak)…");

    // ── EARTH TWIN A1 (E0-2): camera-altitude LOD envelope. The satellite
    // population exists at globe/regional zooms and fades out approaching the
    // street (the registry's lod block says where); while fully hidden the
    // worker is STOPPED (zero propagation + zero draw cost) and the panel
    // says so — LOD is reversible by zoom and never a silent drop. ──
    let lodPaused = false;
    let lodLastOpacity = -1; // sentinel: first applyLod() always applies
    let lastCounts = { shown: 0, skipped: 0 };
    // O8: the last tick's full meta, so a filter change can re-push the
    // retained raw buffer with honest counts without waiting for a tick.
    let lastMeta: { shown: number; deepSpaceSkipped: number; invalidSkipped: number } | null = null;
    // O6-2: per-index class forms (null until SATCAT lands); minis appear
    // in the close-zoom band without a click — catalogued classes only.
    let miniForms: (FormKind | null)[] | null = null;
    let miniCount = 0;
    const updateMinis = () => {
      const modelLayer = satModelLayerRef.current;
      const layer = satLayerRef.current;
      if (!modelLayer || !layer) return;
      const camKm = cameraAltitudeKmFromMap(map);
      const inBand = camKm != null && camKm <= MINI_MAX_CAM_KM && layer.getGlobalOpacity() > 0 && !!miniForms;
      if (!inBand) {
        if (miniCount) { miniCount = 0; modelLayer.setMinis(null); }
        return;
      }
      try {
        const b = map.getBounds();
        const sw = lonLatToMercator(b.getWest(), Math.max(-85, b.getSouth()));
        const ne = lonLatToMercator(b.getEast(), Math.min(85, b.getNorth()));
        const ctr = map.getCenter();
        const c = lonLatToMercator(ctr.lng, Math.max(-85, Math.min(85, ctr.lat)));
        // co-location thinning at ~34px (a mini's own size); the FOCUSED
        // object's spot is owned by the big model (docked craft bug)
        const followT = satFollowRef.current
          ? followTarget(layer.getPositions(), satFollowRef.current.index)
          : null;
        const minis = selectMiniSats(
          layer.getPositions(), miniForms,
          { minX: sw.x, maxX: ne.x, minY: ne.y, maxY: sw.y, cx: c.x, cy: c.y },
          undefined, satFollowRef.current?.index ?? -1,
          pixelToleranceToMercUnits(34, map.getZoom() ?? 0),
          followT ? { x: followT.mercX, y: followT.mercY } : null,
        );
        miniCount = minis.length;
        modelLayer.setMinis(minis);
      } catch { /* minis are chrome — never break the tick */ }
    };
    const publishOrbitalStatus = () => {
      if (lodPaused) {
        const minKm = orbitalLodRef.current?.camMinKm ?? ORBITAL_LOD_FALLBACK.camMinKm;
        setStatus("orbital_sats", "active", lastCounts.shown,
          `hidden at this zoom (LOD) — returns above ~${fmtKm(minKm)} camera altitude; propagation paused, nothing lost`);
      } else {
        const grp = satGroupInfoRef.current;
        setStatus("orbital_sats", "active", grp ? grp.count : lastCounts.shown,
          grp
            ? `filtered to ${grp.label} — ${grp.count.toLocaleString()} of ${lastCounts.shown.toLocaleString()} live shown (clear the chip to see the whole sky)`
            : `${lastCounts.shown.toLocaleString()} live (near-earth SGP4 + deep-space SDP4)${lastCounts.skipped ? ` · ${lastCounts.skipped.toLocaleString()} not rendered (incomplete/decayed elements)` : ""}`);
      }
    };
    // O8 (live report 2026-07-18): re-derive the layer's display buffer from
    // the retained RAW tick + the CURRENT group mask, immediately. Called by
    // applySatGroup so activating/clearing a chip takes effect this frame —
    // critically, clearing a filter restores every slot BEFORE a follow-up
    // focusSat() reads the buffer (the search-overrides-filter path). No
    // setTickTime: positions are from the same physics tick, so the shader's
    // velocity glide anchor stays honest.
    const repushPositions = () => {
      const layer = satLayerRef.current;
      const raw = satRawPosRef.current;
      if (!layer || !raw || !lastMeta) return;
      const gmask = satGroupMaskRef.current;
      layer.updatePositions(gmask ? applyGroupSentinel(raw, gmask) : raw, lastMeta);
      publishOrbitalStatus();
    };
    satRepushRef.current = repushPositions;
    const applyLod = () => {
      const layer = satLayerRef.current;
      if (!layer) return;
      const camKm = cameraAltitudeKmFromMap(map); // null → lodOpacity fails OPEN (visible)
      const op = lodOpacity(orbitalLodRef.current ?? ORBITAL_LOD_FALLBACK, camKm);
      if (op === lodLastOpacity) return;
      lodLastOpacity = op;
      layer.setGlobalOpacity(op);
      if (op <= 0 && !lodPaused) {
        lodPaused = true;
        satWorkerRef.current?.postMessage({ type: "stop" }); // pause: gp stays loaded
        stopFollow(); // an invisible object is never silently "followed"
        publishOrbitalStatus();
      } else if (op > 0 && lodPaused) {
        lodPaused = false;
        satWorkerRef.current?.postMessage({ type: "start", hz: 1 }); // resume: ticks immediately
        publishOrbitalStatus();
      }
      updateMinis(); // zooming through the band updates minis between ticks
    };
    map.on("move", applyLod);
    map.on("resize", applyLod); // rotate/resize changes camera altitude without a move event

    // Resilient fetch+init: a CelesTrak stall/blip now retries automatically with
    // backoff instead of leaving the layer dead until a manual toggle (BUG 1). The
    // timeout signal is threaded into fetchGp's fetchImpl so a hung request aborts.
    const stopLoad = runResilientLoad(
      async (signal) => {
        const fixture = (window as any).__vtOrbitalGpFixture;
        let gp: GpRecord[];
        if (Array.isArray(fixture) && fixture.length) {
          gp = fixture as GpRecord[];
        } else if (orbitalGpCache && Date.now() - orbitalGpCache.at < ORBITAL_GP_TTL_MS) {
          gp = orbitalGpCache.gp; // reuse cached elements — toggling never re-hits CelesTrak
        } else {
          // PERF: fetched + parsed in a one-shot worker (150-500ms main-thread
          // freeze removed); the resilient-load signal still aborts/times out.
          gp = await fetchGpOffThread("active", signal);
          if (gp.length) orbitalGpCache = { at: Date.now(), gp }; // cache for the session
        }
        if (signal.aborted) return;
        if (!gp.length) throw new Error("no orbital elements returned");
        // one physical station = ONE object (human-directed 2026-07-16):
        // ISS/CSS module entries collapse to the core-module keeper BEFORE
        // the worker/ref split, so buffer indices stay aligned everywhere
        gp = collapseStationComplexes(gp);
        orbitalGpRef.current = gp; // index-aligned to the worker's buffer — picking reads this
        setGpVersion((v) => v + 1); // O6-3: SatFinder can search now
        if (satWorkerRef.current) return; // already initialized — don't double-add

        const layer = new SatLayer({ id: "orbital_sats" });
        satLayerRef.current = layer;
        map.addLayer(layer);
        // O5-2b: the focused-satellite 3D form layer sits above the points
        // (draws nothing until a follow sets its anchor + form)
        const modelLayer = new SatModelLayer({ id: "orbital_sat_model" });
        satModelLayerRef.current = modelLayer;
        map.addLayer(modelLayer);
        // O6-1: the focused object's real orbit track (zero cost until set)
        const arcLayer = new ArcLayer({ id: "orbital_arcs" });
        satArcLayerRef.current = arcLayer;
        map.addLayer(arcLayer);

        const worker = new Worker(
          new URL("../lib/orbital/satWorker.ts", import.meta.url),
          { type: "module" },
        );
        satWorkerRef.current = worker;
        worker.onmessage = (ev: MessageEvent<SatWorkerOutbound>) => {
          const m = ev.data;
          if (m.type === "positions") {
            // PERF (scale_program.md item (c)): worker ticks at hz=1 below —
            // tickIntervalSec=1 lets updatePositions skip the forced repaint
            // when the accumulated worst-case ground-track motion is still
            // sub-pixel at the current camera (default globe view is the
            // common case). If the tick rate here ever changes, this must
            // change with it.
            // O6-3: an active group filter hides non-members via the layer's
            // own sentinel semantics (copy — the worker buffer stays whole).
            // O8: the raw tick is retained so filter changes / search /
            // coverage can read the whole sky between ticks (satRawPosRef).
            let posBuf = new Float32Array(m.buf);
            satRawPosRef.current = posBuf;
            lastMeta = { shown: m.shown, deepSpaceSkipped: m.deepSpaceSkipped, invalidSkipped: m.invalidSkipped };
            const gmask = satGroupMaskRef.current;
            if (gmask) posBuf = applyGroupSentinel(posBuf, gmask);
            satLayerRef.current?.updatePositions(posBuf, {
              shown: m.shown,
              deepSpaceSkipped: m.deepSpaceSkipped,
              invalidSkipped: m.invalidSkipped,
            }, 1);
            // SMOOTH SKY (human: "make them move smoothly … still use the
            // data we get"): anchor the velocity glide at this tick — the
            // shader slides every sat along its REAL SGP4 velocity between
            // 1Hz exact-physics ticks (measured 0.23m/1s vs true
            // propagation; capped 2.5s so a stale worker holds, never lies)
            satLayerRef.current?.setTickTime();
            lastCounts = { shown: m.shown, skipped: m.deepSpaceSkipped + m.invalidSkipped };
            publishOrbitalStatus(); // formats the LOD-paused note when applicable
            followTick(); // O5: keep the followed satellite centered + ringed
            updateMinis(); // O6-2: auto-3D forms in the close-zoom band
          }
        };
        worker.postMessage({ type: "init", gp });
        worker.postMessage({ type: "start", hz: 1 });
        applyLod(); // a page opened already deep-zoomed pauses immediately
        // E4-1: identity catalog in the background, non-blocking — and once
        // it lands, identified objects stop being dots (SYMBOLS NOT DOTS):
        // shape = catalogued type, color = orbit class, dot = unidentified.
        ensureSatcat().then(() => {
          const g = orbitalGpRef.current;
          if (g && satcatByNorad) {
            satLayerRef.current?.setShapeCodes(shapeCodesFromSatcat(g, satcatByNorad));
            miniForms = formsFromSatcat(g, satcatByNorad); // O6-2 minis unlock
          }
        }).catch(() => {});
      },
      (failures) => setStatus("orbital_sats", "error", undefined,
        failures === 0 ? "could not reach CelesTrak — retrying automatically…" : "still retrying automatically…"),
      // The ~6.6 MB `active` fetch needs headroom on slow links (default 15s was
      // too tight and aborted mid-download → the "retrying" the user reported).
      { timeoutMs: 45_000 },
    );

    return () => { map.off("move", applyLod); map.off("resize", applyLod); map.off("dragstart", releaseCamera); d3Detach(); stopLoad(); teardown(); };
  }, [enabled["orbital_sats"], mapReady, setStatus]);

  // EARTH TWIN A1: keep the orbital LOD envelope in sync with the fetched
  // registry (source of truth; ORBITAL_LOD_FALLBACK covers older registries).
  useEffect(() => {
    const entry = layers.find((l) => l.id === "orbital_sats") as any;
    orbitalLodRef.current = entry?.lod ?? null;
  }, [layers]);

  // ── O6-3: GROUP ORBITS — the real SGP4 tracks of the filtered
  // constellation, RAAN-colored so planes read apart, evenly sampled under
  // GROUP_ARC_CAP with the cap DISCLOSED in the panel. A live follow's own
  // arc takes the layer over; toggling here rebuilds. ──
  useEffect(() => {
    const arcs = satArcLayerRef.current;
    if (!arcs || !mapReady) return;
    if (!satGroup || !satGroupOrbits) {
      setSatArcInfo(null);
      if (!satFollowRef.current) arcs.setArcs(null);
      return;
    }
    // O8 (live report 2026-07-18, "the track feature looks to be gone"):
    // while a follow is live, its single amber arc owns the layer (focusSat
    // set it) — do NOT rebuild group arcs over it. `satFollowing` in the
    // deps re-runs this effect when the follow ENDS, so closing the card
    // RESTORES the group orbits instead of leaving the orbits toggle ON
    // with zero tracks rendered (the old dead state: stopFollow cleared
    // arcs and nothing ever rebuilt them until the toggle was flicked).
    if (satFollowRef.current) return;
    const gp = orbitalGpRef.current;
    const mask = satGroupMaskRef.current;
    if (!gp || !mask) return;
    const members: number[] = [];
    for (let i = 0; i < mask.length; i++) if (mask[i]) members.push(i);
    // sort by RAAN so the capped sample covers EVERY orbital plane evenly
    // (live feedback: index-order sampling read as "a fraction of the orbits")
    members.sort((a, b) => (gp[a].raan ?? 0) - (gp[b].raan ?? 0));
    const chosen = spreadIndices(members);
    const now = Date.now();
    const list: { pts: Float32Array; color: [number, number, number, number] }[] = [];
    for (const idx of chosen) {
      const pts = sampleOrbitArc(gp[idx], now, 121);
      if (pts) list.push({ pts, color: raanColor(gp[idx].raan) });
    }
    arcs.setArcs(list.length ? list : null);
    setSatArcInfo({ shown: list.length, total: members.length });
  }, [satGroup, satGroupOrbits, mapReady, gpVersion, satFollowing]);

  // ── satellite click-to-identify (ORBITAL O3; research/orbital_program.md's
  // "O3 picking" recipe). SatLayer is a raw custom WebGL layer with no
  // MapLibre queryRenderedFeatures hit-testing (see satLayer.ts's PICKING
  // note), so this is a plain map-wide click listener + CPU nearest-point
  // search over the layer's own position buffer via pick.ts, gated to a tight
  // screen-pixel tolerance so it only fires on near-exact clicks and doesn't
  // steal clicks meant for other (properly feature-scoped) layers. Only
  // registered while orbital_sats is enabled. A miss (nothing within pixel
  // tolerance) is now a Starlink COVERAGE query at the clicked ground point
  // (O7; see siteQuery.ts) instead of a no-op — GPS DOP, the charter's other
  // O7 example, is NOT wired here: GPS is MEO and this catalog's near-earth-
  // only SGP4 kernel never gives it a valid position, so querying it would
  // silently report permanent "no coverage" (a modeling gap, not a finding). ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady || !enabled["orbital_sats"]) return;

    const PICK_TOLERANCE_PX = 12; // touch-friendly; was 8
    const ORBIT_CLASS_NAME = ["LEO", "MEO", "GEO"];

    const onClick = (e: any) => {
      const layer = satLayerRef.current;
      const gp = orbitalGpRef.current;
      if (!layer || !gp || !gp.length) return;
      // EARTH TWIN A1: while the LOD envelope has the layer fully hidden, its
      // worker is paused and the position buffer is STALE — an invisible
      // satellite must not be clickable, and a coverage report from stale
      // positions would be dishonest. The whole handler goes dormant.
      if (layer.getGlobalOpacity() <= 0) return;
      const positions = layer.getPositions();
      if (!positions) return;

      const clickLL = map.unproject(e.point);
      const clickMerc = lonLatToMercator(clickLL.lng, clickLL.lat);
      // O6 PICK FIX: in globe mode, pick in SCREEN space with the exact
      // frame matrix the shader used — a MEO object renders displaced from
      // its ground point by its altitude, and ground-mercator picking was
      // selecting whatever LEO object's nadir sat under the cursor.
      const globeMatrix = layer.getGlobeProjection();
      const canvas = map.getCanvas();
      const hit = globeMatrix
        ? pickNearestSatelliteScreen(
            positions, layer.getStride(), gp, globeMatrix,
            e.point.x, e.point.y, canvas.clientWidth || 1, canvas.clientHeight || 1,
            PICK_TOLERANCE_PX, layer.getGlobeCamera())
        : pickNearestSatellite(
        positions, layer.getStride(), gp, clickMerc.x, clickMerc.y,
        pixelToleranceToMercUnits(PICK_TOLERANCE_PX, map.getZoom()),
        // globe mode: exclude satellites the far-side cull has hidden — a
        // click at the limb must not select an invisible object (null in
        // mercator view, where nothing is culled).
        layer.getGlobeCamera(),
      );
      if (!hit) {
        // O6-1: empty ground NO LONGER releases the focus — the directive is
        // explicit: the focus lives until the card's ✕ (drag already freed
        // the camera). Fall through to the coverage report only.
        // FEATURE CLICKS OWN THEIR POPUPS: only fall through to the coverage
        // report on genuinely empty ground. The basemap is raster-only, so any
        // rendered vector feature under the cursor belongs to a data layer
        // (port markers, plants, alerts, …) — its handler owns the click, and
        // the coverage card must not paint over it (live bug: clicking the LA
        // port marker showed "Starlink coverage" instead of the port).
        let atPoint: unknown[] = [];
        try { atPoint = map.queryRenderedFeatures(e.point) ?? []; } catch { /* style mid-swap */ }
        if (!coverageQueryAllowed(atPoint)) return;
        // O7 STARLINK COVERAGE: reuse this tick's already-propagated buffer
        // to answer "does Starlink cover this ground point right now" rather
        // than leaving the click a pure no-op.
        // O8 (live report 2026-07-18, replaces the O6-3 instruction card):
        // a group chip must not dead-end this click. The DISPLAY buffer is
        // sentineled to the group, but the worker always propagates the
        // whole sky (the filter is a display-only copy) — so with a chip
        // active the query reads the retained RAW tick buffer instead: one
        // click, a real numeric answer, the map stays filtered, and the
        // card states the bypass.
        const grpInfo = satGroupInfoRef.current;
        const covBuf = grpInfo ? satRawPosRef.current ?? positions : positions;
        const { visible, totalModeled } = siteCoverageReport(
          covBuf, layer.getStride(), gp, clickLL.lat, clickLL.lng,
        );
        const nearest = visible.slice(0, 5);
        setDetail({
          kind: "coverage",
          title: "Starlink coverage",
          subtitle: `${clickLL.lat.toFixed(2)}°, ${clickLL.lng.toFixed(2)}°`,
          body: [
            totalModeled === 0
              ? "No Starlink elements in the current CelesTrak fetch had a valid position this tick."
              : visible.length === 0
                ? `0 of ${totalModeled} modeled Starlinks currently cover this point (>=${STARLINK_MIN_ELEV_DEG}° elevation mask) — a real gap right now, not missing data.`
                : `${visible.length} of ${totalModeled} modeled Starlinks currently cover this point (>=${STARLINK_MIN_ELEV_DEG}° elevation mask).`,
            ...nearest.map((s) =>
              `${s.name ?? `NORAD ${s.noradId}`}: ${s.elevationDeg.toFixed(1)}° elevation, ${fmtKm(s.rangeKm)} range`),
            grpInfo
              ? `Sky is filtered to ${grpInfo.label} right now — this query bypassed the filter and read the full live catalog (the map stays filtered as you set it).`
              : null,
            "Geometric visibility only (published ~25° user-terminal mask) — no link-budget, beam-steering, or cell-capacity model; real SGP4 position, no predictive claim.",
          ].filter(Boolean).join("\n"),
        });
        return;
      }

      focusSat(hit.index);
    };

    // ── O6-3: one focus path for BOTH entrances — a map click (above) and a
    // search hit (SatFinder → focusSatByIndexRef). Position/class come from
    // the live buffer; an object with no live position this tick (deep-space,
    // group-filtered) still gets its identity card, honestly un-followed. ──
    const focusSat = (index: number) => {
      const layer = satLayerRef.current;
      const gp = orbitalGpRef.current;
      const g = gp?.[index];
      if (!layer || !g) return;
      const positions = layer.getPositions();
      const t = followTarget(positions ?? null, index);
      const s = positions ? readSatAt(positions, index) : null;
      const cls = t && s ? (ORBIT_CLASS_NAME[s.classCode] ?? "unknown") : "no live position";
      const altKm = s && t ? s.altMeters / 1000 : null;
      const ageDays = epochAgeDays(g.epoch, Date.now());
      // E4-1 identity: SATCAT row (may still be downloading — honest line) +
      // conservative operator resolve (owner code first, then the
      // constellation name stem; null = honestly unmapped, never guessed).
      if (satcatState === "error") ensureSatcat(); // failed earlier → retry on demand (this card stays honest, the next click enriches)
      const sc = satcatByNorad?.get(g.noradId) ?? null;
      const op = sc
        ? resolveOperator(sc.owner, sc.country) ??
          resolveOperator(nameStemForOperator(g.name ?? sc.name), sc.country)
        : null;
      // ── ORBITAL O5 slice 1 + O6-1: FOLLOW — only with an honest live
      // position this tick; the focus persists until the card's ✕. ──
      if (t && s) {
        satFollowRef.current = { index, noradId: g.noradId, name: g.name ?? null, lockMode: "sat" };
        setSatFollowing(true); // shows the follow-tools cluster
        setSatLockMode("sat");
        // phone (live report round 9): the layers panel covers ~half the
        // viewport — a correctly-centered craft hides behind it. Locking
        // on closes the panel so the real center is visible (same gesture
        // Escape already uses on phones).
        if (window.innerWidth < 768) setPanelOpen(false);
        // the REAL orbit track for this one object — one full period,
        // SGP4-propagated (gaps honest, never bridged). Amber = focus ring.
        try {
          const arcPts = sampleOrbitArc(g, Date.now());
          satArcLayerRef.current?.setArcs(arcPts ? [{ pts: arcPts, color: [1.0, 0.82, 0.4, 0.85] }] : null);
        } catch { satArcLayerRef.current?.setArcs(null); }
        // zoom to FRAME the object — by its ALTITUDE, not just class (live
        // bug: Cluster II's ~119,000 km apogee is unframeable at the GEO
        // zoom; super-high objects need the camera to back OUT so the
        // craft appears above the globe). ALWAYS center.
        try {
          const altKmNow = s.altMeters / 1000;
          const cur = map.getZoom() ?? 0;
          // FRAMING FROM ALTITUDE (live report 2026-07-18: far/GEO clicks
          // zoomed PAST the craft — camera inside the orbit shell, craft
          // behind it). Camera parks at 2.3x the craft's altitude: in front
          // of the camera at ANY orbit height. LEO keeps a close floor so
          // low craft still read big.
          const frameZoom = zoomForCameraAltitudeKm(
            Math.max(altKmNow * 2.3, 900), t.latDeg,
            map.getCanvas()?.height ?? 900,
            (map as any).getVerticalFieldOfView?.(), map.getPitch?.());
          const zoom = altKmNow < 3000 ? Math.max(cur, Math.min(frameZoom, 4.3))
            : Math.max(Math.min(cur, frameZoom), Math.min(frameZoom, 1.0));
          map.easeTo({ center: [t.lonDeg, t.latDeg], zoom, duration: 1200 });
        } catch {}
        // O5-2b: the on-map 3D form — ONLY when the catalog knows the class
        // (unknown class = honest ring-only follow, never a guessed spacecraft)
        const focusForm = sc ? classFormNamed(sc.objectType, sc.rcsSize, g.name ?? sc.name) : null;
        satModelLayerRef.current?.setForm(focusForm);
        // O5-3b: REAL model where a verified public asset exists. Cleared
        // FIRST so the previous target's model never rides this orbit.
        satModelLayerRef.current?.setRealMesh(null);
        // name passed through: ISS MODULE entries (e.g. "ISS (UNITY)"
        // 25575) resolve to the station model — live report 2026-07-16
        if (realModelLabel(g.noradId, g.name ?? sc?.name)) {
          const wantNorad = g.noradId;
          loadRealModel(wantNorad, g.name ?? sc?.name).then((mesh) => {
            if (!mesh) return; // fetch/decode failed → representative form stays
            if (satFollowRef.current?.noradId !== wantNorad) return;
            satModelLayerRef.current?.setRealMesh(mesh);
          });
        }
        // O6 ring fix: seed the altitude-anchored ring/model IMMEDIATELY
        // (the next worker tick takes over) — no geojson ground-point ring.
        satModelLayerRef.current?.setAnchor({ mercX: t.mercX, mercY: t.mercY, altMeters: s.altMeters });
      }
      const realLabel = realModelLabel(g.noradId, g.name ?? sc?.name);
      setDetail({
        kind: "satellite",
        title: g.name || `NORAD ${g.noradId}`,
        subtitle: `${sc?.objectType === "ROCKET BODY" ? "Rocket body · " : sc?.objectType === "DEBRIS" ? "Debris · " : ""}${cls}${altKm != null ? ` · ${fmtKm(altKm)} altitude` : ""}`,
        body: [
          `NORAD catalog ID: ${g.noradId}`,
          ...satelliteIdentityLines(sc, op, satcatState),
          g.meanMotion != null ? `Orbital period: ${(1440 / g.meanMotion).toFixed(1)} min` : null,
          g.inclination != null ? `Inclination: ${g.inclination.toFixed(1)}°` : null,
          ageDays != null ? `Element set age: ${ageDays.toFixed(1)} days (orbit uncertainty grows with age)` : null,
          t
            ? "FOLLOWING — the camera tracks this object as it moves (updates each second). Drag to look elsewhere: the focus and orbit track stay until you close this card (✕)."
            : "No live position this tick (incomplete or decayed elements, or filtered out by a group chip) — identity only, nothing followed.",
          t ? "Orbit track shown: one full period, SGP4-propagated from the epoch elements — the real path, not a drawn ellipse." : null,
          t && realLabel
            ? `On-map 3D: ${realLabel}.`
            : t && sc ? `On-map 3D: ${formLabel(classFormNamed(sc.objectType, sc.rcsSize, g.name ?? sc.name))}.` : null,
          isStationComplex(g.name)
            ? "One station, one object: modules cataloged separately (e.g. UNITY, NAUKA) are collapsed into this single entry — visiting vehicles stay separate."
            : null,
          "RAW catalog data (CelesTrak GP + SATCAT), SGP4-propagated — real position, no predictive claim.",
        ].filter(Boolean).join("\n"),
        links: [{
          label: "CelesTrak catalog entry",
          href: `https://celestrak.org/satcat/table-satcat.php?NORAD_CAT_ID=${g.noradId}`,
        }],
      });
    };
    focusSatByIndexRef.current = focusSat; // SatFinder's entrance

    map.on("click", onClick);
    return () => { map.off("click", onClick); focusSatByIndexRef.current = null; };
  }, [enabled["orbital_sats"], mapReady, setDetail]);

  // ── country borders (RAW; Natural Earth 1:110m admin-0, PUBLIC DOMAIN —
  // atlas-parity layer 3. Self-hosted datacore compile served by our own
  // API: zero external dependency. Fetched ONLY on enable (zero-cost-
  // when-off); generalized-resolution honesty in the status note.) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.boundaries) {
      try {
        if (map.getLayer("ne-boundaries")) map.removeLayer("ne-boundaries");
        if (map.getSource("ne-boundaries")) map.removeSource("ne-boundaries");
      } catch {}
      setStatus("boundaries", "off");
      return;
    }
    setStatus("boundaries", "loading");
    return runResilientLoad(
      async (signal) => {
        const r = await fetch("/api/data/boundaries", { signal });
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (signal.aborted) return;
        if (!map.getSource("ne-boundaries")) {
          map.addSource("ne-boundaries", { type: "geojson", data: d as any } as any);
        }
        if (!map.getLayer("ne-boundaries")) {
          const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
          map.addLayer({
            id: "ne-boundaries", type: "line", source: "ne-boundaries",
            paint: {
              "line-color": "rgba(179,194,216,0.55)",
              "line-width": ["interpolate", ["linear"], ["zoom"], 2, 0.7, 8, 1.6],
            },
          } as any, firstMarker?.id);
        }
        setStatus("boundaries", "active", d?.features?.length,
          "1:110m generalized (Natural Earth, public domain) — reference, not survey-grade");
      },
      (failures) => setStatus("boundaries", "error", undefined,
        failures === 0 ? "load failed — retrying automatically…" : "still retrying automatically…"),
    );
  }, [enabled.boundaries, mapReady, setStatus]);

  // ── state/province borders (RAW; Natural Earth 1:50m admin-1 lines,
  // PUBLIC DOMAIN — sprint W3 2026-07-17). Same self-hosted pattern as
  // country borders, one level down: thinner, dimmer, and faded out below
  // zoom ~3 where 581 sub-national lines are just noise on a globe view. ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.boundaries_admin1) {
      try {
        if (map.getLayer("ne-admin1")) map.removeLayer("ne-admin1");
        if (map.getSource("ne-admin1")) map.removeSource("ne-admin1");
      } catch {}
      setStatus("boundaries_admin1", "off");
      return;
    }
    setStatus("boundaries_admin1", "loading");
    return runResilientLoad(
      async (signal) => {
        const r = await fetch("/api/data/boundaries_admin1", { signal });
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (signal.aborted) return;
        if (!map.getSource("ne-admin1")) {
          map.addSource("ne-admin1", { type: "geojson", data: d as any } as any);
        }
        if (!map.getLayer("ne-admin1")) {
          const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
          map.addLayer({
            id: "ne-admin1", type: "line", source: "ne-admin1", minzoom: 2,
            paint: {
              // dimmer + thinner than country borders (visual hierarchy:
              // admin-0 reads first), fading in across zoom 2→4
              "line-color": "rgba(179,194,216,0.38)",
              "line-width": ["interpolate", ["linear"], ["zoom"], 3, 0.5, 8, 1.1],
              "line-opacity": ["interpolate", ["linear"], ["zoom"], 2, 0, 3, 0.55, 4, 1],
            },
          } as any, firstMarker?.id);
        }
        setStatus("boundaries_admin1", "active", d?.features?.length,
          "1:50m generalized (Natural Earth, public domain) — first-level subdivisions worldwide; reference, not survey-grade");
      },
      (failures) => setStatus("boundaries_admin1", "error", undefined,
        failures === 0 ? "load failed — retrying automatically…" : "still retrying automatically…"),
    );
  }, [enabled.boundaries_admin1, mapReady, setStatus]);

  // ── power grid (RAW; OSM power features © OpenStreetMap contributors,
  // ODbL — DATACORE MAXIMUS Phase 2 TX pilot. Single 16MB PMTiles on our
  // origin (range requests, zero-cost-when-off: the file is fetched only
  // when the layer is on). VOLTAGE HONESTY: lines whose voltage tag is
  // missing or unparseable (multi-value "138000;69000") render as a
  // distinct dashed class — never hidden. Zoom gates per the grid build
  // order keep low-zoom vertex counts phone-safe.) ──
  // stable key over the master + every per-state grid flag, so the effect below
  // re-runs on any grid toggle without hardcoding one dependency per state.
  const powerGridKey = (enabled.powergrid ? "M" : "") + (enabled.powergrid_hifld ? "H" : "") +
    (enabled.powergrid_hifld_sub ? "S" : "") + (enabled.powergrid_hifld_plants ? "P" : "") +
    (enabled.powergrid_canada ? "C" : "") +
    POWER_STATES.map((s) => (enabled[`powergrid_${s.code}`] ? s.code : "")).join("") +
    CANADA_PROVINCES.map((p) => (enabled[`powergrid_${p.code}`] ? p.code : "")).join("");
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    // voltage as a number; -1 = missing/unparseable (to-number fallback)
    const V = ["to-number", ["get", "voltage"], -1] as any;
    const isLine = ["match", ["get", "power"], ["line", "minor_line", "cable"], true, false] as any;
    const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle"].includes(l.type));
    const add = (def: any) => { if (!map.getLayer(def.id)) map.addLayer(def, firstMarker?.id); };
    // click/hover handlers to detach on cleanup (mapInteractions contract —
    // callers MUST detach or handlers stack across toggle cycles).
    const gridDetach: Array<() => void> = [];
    // fuel-driven icon + tint for HIFLD plants, built from the EIA-code tables
    // so plant markers read as their generation type (solar/wind/gas/...) and
    // tint to the same palette as the legend and the WRI plants layer.
    const fuelIcon = ["match", ["get", "fuel"],
      ...Object.entries(EIA_FUEL_TO_CANON).flatMap(([code, canon]) => [code, POWER_FUEL_ICON[canon]]),
      POWER_FUEL_ICON.other] as any;
    const fuelColor = ["match", ["get", "fuel"],
      ...Object.entries(EIA_FUEL_TO_CANON).flatMap(([code, canon]) => [code, POWER_FUEL_COLOR[canon]]),
      POWER_FUEL_COLOR.other] as any;
    const STATUS_LABEL: Record<string, string> = {
      OP: "Operating", SB: "Standby", OS: "Out of service", RE: "Retired",
      PL: "Planned", CN: "Cancelled", TS: "Under construction",
    };

    // add the 6 voltage/element sublayers for one grid source (national or a state)
    const addGrid = (src: string, file: string) => {
      if (!map.getSource(src)) {
        map.addSource(src, {
          type: "vector",
          url: `pmtiles://${window.location.origin}/tiles/${file}`,
          attribution: "© OpenStreetMap contributors, ODbL",
        } as any);
      }
      add({ id: `${src}-substation`, type: "fill", source: src, "source-layer": "power",
            minzoom: 9, filter: ["==", ["get", "power"], "substation"],
            paint: { "fill-color": "rgba(250,204,21,0.14)", "fill-outline-color": "rgba(250,204,21,0.6)" } });
      add({ id: `${src}-plant`, type: "fill", source: src, "source-layer": "power",
            minzoom: 9, filter: ["==", ["get", "power"], "plant"],
            paint: { "fill-color": "rgba(74,222,128,0.10)", "fill-outline-color": "rgba(74,222,128,0.5)" } });
      add({ id: `${src}-unknown`, type: "line", source: src, "source-layer": "power",
            minzoom: 8, filter: ["all", isLine, ["<", V, 0]],
            paint: { "line-color": "rgba(216,180,254,0.55)", "line-width": 0.9, "line-dasharray": [2, 2] } });
      add({ id: `${src}-low`, type: "line", source: src, "source-layer": "power",
            minzoom: 11, filter: ["all", isLine, [">=", V, 0], ["<", V, 100000]],
            paint: { "line-color": "rgba(148,163,184,0.55)", "line-width": 0.8 } });
      add({ id: `${src}-mv`, type: "line", source: src, "source-layer": "power",
            minzoom: 6, filter: ["all", isLine, [">=", V, 100000], ["<", V, 230000]],
            paint: { "line-color": "rgba(56,189,248,0.75)",
                     "line-width": ["interpolate", ["linear"], ["zoom"], 6, 0.8, 12, 2] } });
      add({ id: `${src}-hv`, type: "line", source: src, "source-layer": "power",
            filter: ["all", isLine, [">=", V, 230000]],
            paint: { "line-color": "rgba(250,204,21,0.9)",
                     "line-width": ["interpolate", ["linear"], ["zoom"], 3, 1, 12, 3] } });
    };
    const removeGrid = (src: string) => {
      try {
        [`${src}-hv`, `${src}-mv`, `${src}-low`, `${src}-unknown`, `${src}-substation`, `${src}-plant`]
          .forEach((id) => { if (map.getLayer(id)) map.removeLayer(id); });
        if (map.getSource(src)) map.removeSource(src);
      } catch {}
    };

    // "US Power Grid (all states)" master -> ONE efficient national tile (not 51
    // simultaneous state sources). Whole US grid in a single range-served file.
    if (enabled.powergrid) {
      try {
        setStatus("powergrid", "loading");
        addGrid("powergrid_us", "power_us.pmtiles");
        setStatus("powergrid", "active", undefined,
          "Entire US grid — all 50 states + DC (OSM, ODbL): voltage-classed; dashed = voltage untagged (never hidden); overview fidelity");
      } catch { setStatus("powergrid", "error"); }
    } else {
      removeGrid("powergrid_us");
      setStatus("powergrid", "off");
    }

    // "Canada Power Grid (all provinces)" master -> ONE national tile merged
    // from all 13 provinces/territories (OSM, ODbL). Same voltage-classed
    // rendering as the US master.
    if (enabled.powergrid_canada) {
      try {
        setStatus("powergrid_canada", "loading");
        addGrid("powergrid_canada_src", "power_canada.pmtiles");
        setStatus("powergrid_canada", "active", undefined,
          "Entire Canada grid — all 13 provinces & territories (OSM, ODbL): voltage-classed; dashed = voltage untagged (never hidden); overview fidelity");
      } catch { setStatus("powergrid_canada", "error"); }
    } else {
      removeGrid("powergrid_canada_src");
      setStatus("powergrid_canada", "off");
    }

    // HIFLD — AUTHORITATIVE national transmission lines (DHS / Oak Ridge National
    // Lab; EIA defers to it). Public-domain surveyed vector data, 69–765 kV — not
    // ML, no generalization problem. Lines carry no `power` tag (all transmission),
    // so classify by voltage only; no substation/plant here. Voltage normalized
    // kV->V at ingest so the same class thresholds apply.
    const hSrc = "powergrid_hifld";
    const hIDS = [`${hSrc}-hv`, `${hSrc}-mv`, `${hSrc}-low`, `${hSrc}-unknown`];
    if (enabled.powergrid_hifld) {
      try {
        setStatus("powergrid_hifld", "loading");
        if (!map.getSource(hSrc)) {
          map.addSource(hSrc, {
            type: "vector",
            url: `pmtiles://${window.location.origin}/tiles/power_hifld.pmtiles`,
            attribution: "HIFLD — U.S. DHS / Oak Ridge National Laboratory (public domain)",
          } as any);
        }
        add({ id: `${hSrc}-unknown`, type: "line", source: hSrc, "source-layer": "power",
              minzoom: 4, filter: ["<", V, 0],
              paint: { "line-color": "rgba(216,180,254,0.7)", "line-width": 0.9, "line-dasharray": [2, 2] } });
        add({ id: `${hSrc}-low`, type: "line", source: hSrc, "source-layer": "power",
              minzoom: 6, filter: ["all", [">=", V, 0], ["<", V, 100000]],
              paint: { "line-color": "rgba(148,163,184,0.7)", "line-width": 0.8 } });
        add({ id: `${hSrc}-mv`, type: "line", source: hSrc, "source-layer": "power",
              minzoom: 4, filter: ["all", [">=", V, 100000], ["<", V, 230000]],
              paint: { "line-color": "rgba(56,189,248,0.85)",
                       "line-width": ["interpolate", ["linear"], ["zoom"], 4, 0.7, 12, 2.2] } });
        add({ id: `${hSrc}-hv`, type: "line", source: hSrc, "source-layer": "power",
              filter: [">=", V, 230000],
              paint: { "line-color": "rgba(250,204,21,1)",
                       "line-width": ["interpolate", ["linear"], ["zoom"], 3, 1.1, 12, 3.2] } });
        gridDetach.push(attachLayerInteractions(map,
          [`${hSrc}-hv`, `${hSrc}-mv`, `${hSrc}-low`, `${hSrc}-unknown`], (e: any) => {
            const f = e.features?.[0]; if (!f) return;
            const v = Number(f.properties.voltage);
            const vl = (isFinite(v) && v > 0) ? `${Math.round(v / 1000).toLocaleString()} kV` : "voltage untagged";
            setDetail({
              kind: "transmission",
              title: "Transmission line",
              subtitle: vl,
              body: "Part of the authoritative U.S. transmission tier (69–765 kV).\n\n" +
                    "HIFLD authoritative (U.S. DHS / Oak Ridge National Laboratory, public domain).",
            });
          }));
        setStatus("powergrid_hifld", "active", undefined,
          "US transmission — HIFLD authoritative (DHS / Oak Ridge Nat'l Lab, public domain): 69–765 kV, voltage-classed; dashed = voltage untagged — tap a line for its voltage");
      } catch { setStatus("powergrid_hifld", "error"); }
    } else {
      try { hIDS.forEach((id) => { if (map.getLayer(id)) map.removeLayer(id); }); if (map.getSource(hSrc)) map.removeSource(hSrc); } catch {}
      setStatus("powergrid_hifld", "off");
    }

    // HIFLD — AUTHORITATIVE national substations (75,328 points, ≥69 kV; DHS/ORNL,
    // public domain). Rendered as circle markers, sized by zoom.
    const sSrc = "powergrid_hifld_sub";
    if (enabled.powergrid_hifld_sub) {
      try {
        setStatus(sSrc, "loading");
        if (!map.getSource(sSrc)) {
          map.addSource(sSrc, {
            type: "vector",
            url: `pmtiles://${window.location.origin}/tiles/power_hifld_sub.pmtiles`,
            attribution: "HIFLD — U.S. DHS / Oak Ridge National Laboratory (public domain)",
          } as any);
        }
        if (!map.getLayer(`${sSrc}-pt`)) {
          map.addLayer({
            id: `${sSrc}-pt`, type: "symbol", source: sSrc, "source-layer": "subs", minzoom: 6,
            layout: {
              "icon-image": "vt-substation",
              "icon-size": ["interpolate", ["linear"], ["zoom"], 6, 0.4, 10, 0.62, 14, 0.85],
              "icon-allow-overlap": false,       // declutter at low zoom (collision cull)
            },
            paint: {
              "icon-color": "#fbbf24",
              "icon-halo-color": "rgba(5,10,19,0.92)", "icon-halo-width": 1.1,
            },
          } as any, firstMarker?.id);
        }
        gridDetach.push(attachLayerInteractions(map, `${sSrc}-pt`, (e: any) => {
          const f = e.features?.[0]; if (!f) return; const p = f.properties;
          const kv = (p.maxvolt != null && p.maxvolt !== "") ? `${Number(p.maxvolt).toLocaleString()} kV max` : "voltage n/a";
          setDetail({
            kind: "substation",
            title: p.name || "Substation",
            subtitle: `${kv}${p.type ? ` · ${p.type}` : ""}`,
            body: `${p.state ? `State: ${p.state}\n` : ""}` +
                  `\nHIFLD authoritative (U.S. DHS / Oak Ridge National Laboratory, public domain).`,
          });
        }));
        setStatus(sSrc, "active", undefined,
          "US substations — HIFLD authoritative (DHS / Oak Ridge Nat'l Lab, public domain): 75,328 sites, ≥69 kV — tap for details");
      } catch { setStatus(sSrc, "error"); }
    } else {
      try { if (map.getLayer(`${sSrc}-pt`)) map.removeLayer(`${sSrc}-pt`); if (map.getSource(sSrc)) map.removeSource(sSrc); } catch {}
      setStatus(sSrc, "off");
    }

    // HIFLD — AUTHORITATIVE national power plants (11,810 generating stations;
    // DHS/ORNL, sourced from EIA-860, public domain). Rendered as FUEL-TYPE
    // SDF icons (solar/wind/gas/coal/hydro/nuclear/oil/biomass/other), tinted to
    // the shared fuel palette — the generation type readable at a glance. Click
    // any plant for name, fuel, capacity (MW), operator, and status.
    const pSrc = "powergrid_hifld_plants";
    if (enabled.powergrid_hifld_plants) {
      try {
        setStatus(pSrc, "loading");
        if (!map.getSource(pSrc)) {
          map.addSource(pSrc, {
            type: "vector",
            url: `pmtiles://${window.location.origin}/tiles/power_hifld_plants.pmtiles`,
            attribution: "HIFLD — U.S. DHS / Oak Ridge National Laboratory (public domain), from EIA-860",
          } as any);
        }
        if (!map.getLayer(`${pSrc}-pt`)) {
          map.addLayer({
            id: `${pSrc}-pt`, type: "symbol", source: pSrc, "source-layer": "plants", minzoom: 3,
            layout: {
              // fuel-type silhouette (solar/wind/gas/coal/hydro/nuclear/oil/
              // biomass/other) — the generation type readable at a glance
              "icon-image": fuelIcon,
              // grows with zoom; ≥500 MW plants render a touch larger
              "icon-size": ["interpolate", ["linear"], ["zoom"],
                3, ["case", [">=", ["coalesce", ["to-number", ["get", "mw"], 0], 0], 500], 0.42, 0.3],
                8, ["case", [">=", ["coalesce", ["to-number", ["get", "mw"], 0], 0], 500], 0.72, 0.5],
                13, ["case", [">=", ["coalesce", ["to-number", ["get", "mw"], 0], 0], 500], 1.0, 0.72]],
              "icon-allow-overlap": false,       // declutter at low zoom (collision cull)
            },
            paint: {
              "icon-color": fuelColor,
              "icon-halo-color": "rgba(5,10,19,0.92)", "icon-halo-width": 1.2,
            },
          } as any, firstMarker?.id);
        }
        gridDetach.push(attachLayerInteractions(map, `${pSrc}-pt`, (e: any) => {
          const f = e.features?.[0]; if (!f) return; const p = f.properties;
          const canon = EIA_FUEL_TO_CANON[p.fuel] || "other";
          const fuelLabel = EIA_FUEL_LABEL[p.fuel] || POWER_FUEL_LABEL[canon] || p.fuel || "Unknown fuel";
          const mw = (p.mw != null && p.mw !== "") ? `${Number(p.mw).toLocaleString()} MW` : "capacity n/a";
          // dossier parity fix (2026-07-14, filed PRODUCT-DEBT 2026-07-09): this
          // layer never called fetchDossier — every other clickable layer does
          // (aircraft/trains/fires/gauges/quakes/buoys all pass entityId:null +
          // lat/lon for "nearest_sites"+hazards enrichment, per fetchDossier's own
          // doc comment). HIFLD plants have no entity-graph node of their own yet
          // (entityGraph.ts's facility:plant:N ids are built from the WRI GPPD
          // array only — see research/open_questions.md PRODUCT-DEBT entry for
          // why a full consolidation onto HIFLD needs that migrated first), so
          // entityId stays null here rather than guessing a WRI index; lat/lon
          // alone still surfaces the location-dossier hazard/nearest-site section.
          const dossierKey = `plant_hifld:${e.lngLat?.lat},${e.lngLat?.lng}:${Date.now()}`;
          setDetail({
            kind: "powerplant",
            title: p.name || "Power plant",
            subtitle: `${fuelLabel} · ${mw}`,
            body: `${p.operator ? `Operator: ${p.operator}\n` : ""}` +
                  `${p.state ? `State: ${p.state}\n` : ""}` +
                  `${p.status ? `Status: ${STATUS_LABEL[p.status] || p.status}\n` : ""}` +
                  // per-plant position honesty (HIFLD VAL_METHOD; mirrors the WRI
                  // layer's convention) — user report: a wind-plant icon sat on a
                  // house; registry-reported points can mark the plant's
                  // registered address rather than the equipment itself.
                  `${p.val === "IMAGERY" ? "Position imagery-verified (HIFLD).\n"
                    : p.val === "IMAGERY/OTHER" ? "Position verified against imagery/other sources (HIFLD).\n"
                    : p.val === "OTHER" ? "Position verified by a non-imagery source (HIFLD).\n"
                    : "Position registry-reported (EIA-860), unverified — the point may mark the plant's registered address rather than the equipment.\n"}` +
                  `\nHIFLD authoritative (U.S. DHS / Oak Ridge National Laboratory, public domain; from EIA-860).`,
            dossierKey,
          });
          fetchDossier(dossierKey, null, e.lngLat?.lat, e.lngLat?.lng);
        }));
        setStatus(pSrc, "active", undefined,
          "US power plants — HIFLD authoritative (DHS / Oak Ridge Nat'l Lab, public domain; EIA-860): 11,810 stations, fuel-typed icons — tap for details");
      } catch { setStatus(pSrc, "error"); }
    } else {
      try { if (map.getLayer(`${pSrc}-pt`)) map.removeLayer(`${pSrc}-pt`); if (map.getSource(pSrc)) map.removeSource(pSrc); } catch {}
      setStatus(pSrc, "off");
    }

    // individual per-state layers (each its own toggle, independent of the master)
    POWER_STATES.forEach((st) => {
      const src = `powergrid_${st.code}`;
      if (!enabled[src]) { removeGrid(src); setStatus(src, "off"); return; }
      try {
        setStatus(src, "loading");
        addGrid(src, st.file);
        setStatus(src, "active", undefined,
          `${st.name} — OSM community grid (ODbL): voltage-classed; dashed = voltage untagged (never hidden); no CEII/underground detail`);
      } catch { setStatus(src, "error"); }
    });
    // Canada per-province/territory layers (each its own toggle, like the US states)
    CANADA_PROVINCES.forEach((pr) => {
      const src = `powergrid_${pr.code}`;
      if (!enabled[src]) { removeGrid(src); setStatus(src, "off"); return; }
      try {
        setStatus(src, "loading");
        addGrid(src, pr.file);
        setStatus(src, "active", undefined,
          `${pr.name} — OSM community grid (ODbL): voltage-classed; dashed = voltage untagged (never hidden)`);
      } catch { setStatus(src, "error"); }
    });
    // scales to N states: re-run when the master OR any per-state grid flag flips
    // (derived key below), so adding a state needs no new dependency wiring.
    // Detach every click/hover handler on cleanup so they don't stack across
    // toggle cycles (mapInteractions contract — BUG 4 discipline).
    return () => { gridDetach.forEach((d) => d()); };
  }, [powerGridKey, mapReady, setStatus]);

  // ── weather radar (RAW; NOAA nowCOAST WMS — geospatial Tier-1(b), licensing
  // register 2026-07-04: public domain, no key, US-only. Honest gap stated in
  // the registry: no free lawful GLOBAL radar exists; global temp/wind fields
  // activate later if the human sets the OpenWeatherMap key. Tiles refresh on
  // a 5-min bucket — radar mosaics update every ~4-10 min upstream.) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.weather) {
      try {
        if (map.getLayer("weather-radar")) map.removeLayer("weather-radar");
        if (map.getSource("weather-radar")) map.removeSource("weather-radar");
      } catch {}
      setStatus("weather", "off");
      return;
    }
    const radarTiles = (bucket: number) =>
      "https://nowcoast.noaa.gov/geoserver/observations/weather_radar/ows" +
      "?service=WMS&version=1.3.0&request=GetMap&layers=base_reflectivity_mosaic" +
      "&styles=&format=image/png&transparent=true&crs=EPSG:3857" +
      "&bbox={bbox-epsg-3857}&width=256&height=256&_t=" + bucket;
    const bucketNow = () => Math.floor(Date.now() / 300_000);
    try {
      if (!map.getSource("weather-radar")) {
        map.addSource("weather-radar", {
          type: "raster", tiles: [radarTiles(bucketNow())], tileSize: 256,
          attribution: "NOAA/NWS radar (nowCOAST)",
        } as any);
        const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
        map.addLayer({
          id: "weather-radar", type: "raster", source: "weather-radar",
          paint: { "raster-opacity": opacityOf("weather") / 100 },
        } as any, firstMarker?.id);
      }
      setStatus("weather", "active", undefined, "US radar mosaic (NOAA nowCOAST) · refreshes ~5 min · US-only — no free lawful global radar exists");
    } catch {
      setStatus("weather", "error");
    }
    const iv = window.setInterval(() => {
      try {
        const src: any = map.getSource("weather-radar");
        if (src?.setTiles) src.setTiles([radarTiles(bucketNow())]);
      } catch {}
    }, 300_000);
    return () => { window.clearInterval(iv); };
  }, [enabled.weather, mapReady, setStatus]);

  // ── places & labels (RAW discovery layer; our OWN vector tile built from the
  // GeoNames gazetteer, CC-BY — countries, states, cities/towns, ISLANDS, seas
  // & bays as named points. Zoom-tiered via a per-feature rank `r` (label shows
  // once zoom >= r) so the world reads at every scale without clutter. Click a
  // label to see what it is. Ownable + storeable (unlike Google labels); this
  // is "know what you're looking at" for discovery. Photorealistic 3D imagery
  // of a place stays the separate Google 3D Tiles path at deep zoom.) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    const IDS = ["places-water", "places-admin", "places-city", "places-dot"];
    if (!enabled.places) {
      try {
        IDS.forEach((id) => { if (map.getLayer(id)) map.removeLayer(id); });
        if (map.getSource("places")) map.removeSource("places");
      } catch {}
      setStatus("places", "off");
      return;
    }
    try {
      if (!map.getSource("places")) {
        map.addSource("places", {
          type: "vector",
          url: `pmtiles://${window.location.origin}/tiles/places.pmtiles`,
          attribution: "Place data © GeoNames (CC BY 4.0)",
        } as any);
      }
      const showByZoom = ["<=", ["to-number", ["get", "r"], 9], ["zoom"]] as any;
      // small dot for populated places (cities/towns), so a place reads even
      // before its label declutters in
      if (!map.getLayer("places-dot")) {
        map.addLayer({
          id: "places-dot", type: "circle", source: "places", "source-layer": "places",
          filter: ["all", showByZoom, ["==", ["get", "kind"], "city"]],
          paint: {
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 3, 1.2, 10, 3],
            "circle-color": "rgba(226,232,240,0.9)",
            "circle-stroke-color": "rgba(8,12,20,0.9)", "circle-stroke-width": 0.6,
          },
        } as any);
      }
      const labelBase = {
        type: "symbol", source: "places", "source-layer": "places",
        layout: {
          "text-field": ["get", "name"],
          "text-font": ["Open Sans Semibold"],
          "text-size": ["interpolate", ["linear"], ["zoom"],
            2, ["match", ["get", "kind"], "country", 13, "water", 11, 9],
            8, ["match", ["get", "kind"], "country", 16, "water", 13, 12]],
          "text-max-width": 8,
          "text-offset": [0, 0.6],
          "text-allow-overlap": false,
        },
      };
      const paintFor = (color: string, halo = "rgba(3,7,13,0.95)") => ({
        "text-color": color, "text-halo-color": halo, "text-halo-width": 1.3,
      });
      // water (italic-ish blue), admin (countries/states), cities — three passes
      // so styling differs by kind; each gated by the zoom rank.
      if (!map.getLayer("places-water")) {
        map.addLayer({ id: "places-water", ...labelBase,
          filter: ["all", showByZoom, ["==", ["get", "kind"], "water"]],
          paint: paintFor("rgba(125,211,252,0.95)") } as any);
      }
      if (!map.getLayer("places-admin")) {
        map.addLayer({ id: "places-admin", ...labelBase,
          filter: ["all", showByZoom, ["match", ["get", "kind"], ["country", "state"], true, false]],
          paint: paintFor("rgba(248,250,252,1)") } as any);
      }
      if (!map.getLayer("places-city")) {
        map.addLayer({ id: "places-city", ...labelBase,
          filter: ["all", showByZoom, ["match", ["get", "kind"], ["city", "island"], true, false]],
          paint: paintFor("rgba(226,232,240,0.98)") } as any);
      }
      // click any label -> "what is this place"
      placesDetach.current?.();
      placesDetach.current = attachLayerInteractions(map, ["places-city", "places-admin", "places-water"], (e: any) => {
        const f = e.features?.[0]; if (!f) return; const p = f.properties;
        const KIND: Record<string, string> = { country: "Country", state: "State / province", city: "City / town", island: "Island", water: "Sea / bay" };
        setDetail({
          kind: "place",
          title: p.name || "Place",
          subtitle: `${KIND[p.kind] || p.kind}${p.cc ? ` · ${p.cc}` : ""}`,
          body: `${p.pop && Number(p.pop) > 0 ? `Population: ${Number(p.pop).toLocaleString()}\n` : ""}` +
                `\nPlace reference — GeoNames (CC BY 4.0). Names/coordinates as gazetteered, not survey-grade.`,
        });
      });
      setStatus("places", "active", undefined,
        "Places & labels (GeoNames, CC BY) — countries, cities, islands, seas; zoom in for more, tap a name to see what it is");
    } catch { setStatus("places", "error"); }
    return () => { placesDetach.current?.(); placesDetach.current = null; };
  }, [enabled.places, mapReady, setStatus]);

  // ── sampled weather grid: fetch when arrows or labels are wanted;
  // refetch on pan (debounced) + 10-min interval; stale beats spinner ──
  useEffect(() => {
    const map = mapRef.current;
    const want = (enabled.weather_wind && windArrows) || (enabled.weather_temp && tempLabels);
    if (!want || !map || !mapReady) { setWxGrid(null); return; }
    let stop = false;
    let debounce: number | undefined;
    const load = async () => {
      try {
        const b = map.getBounds();
        const r = await fetch(`/api/data/weather/grid?bbox=${b.getSouth().toFixed(2)},${b.getWest().toFixed(2)},${b.getNorth().toFixed(2)},${b.getEast().toFixed(2)}`);
        if (!r.ok) return;
        const d = await r.json();
        if (!stop) setWxGrid(d);
      } catch {}
    };
    const onMove = () => { window.clearTimeout(debounce); debounce = window.setTimeout(load, 600); };
    load();
    map.on("moveend", onMove);
    const iv = window.setInterval(load, 10 * 60_000);
    return () => { stop = true; window.clearInterval(iv); window.clearTimeout(debounce); try { map.off("moveend", onMove); } catch {} };
  }, [enabled.weather_wind, enabled.weather_temp, windArrows, tempLabels, mapReady]);

  // ── wind arrows layer (static grid — redraws on pan; no animation by
  // design: phone budget over spectacle) ──
  useEffect(() => {
    const map = mapRef.current;
    const on = enabled.weather_wind && windArrows && wxGrid?.points?.length;
    if (!map || !mapReady) return;
    if (!on) {
      try {
        if (map.getLayer("wx-wind-arrows")) map.removeLayer("wx-wind-arrows");
        if (map.getSource("wx-grid-wind")) map.removeSource("wx-grid-wind");
      } catch {}
      return;
    }
    const fc = {
      type: "FeatureCollection",
      features: (wxGrid.points as any[])
        .filter((p) => p.wd != null && p.ws != null)
        .map((p) => ({
          type: "Feature",
          geometry: { type: "Point", coordinates: [p.lo, p.la] },
          properties: {
            rot: (p.wd + 180) % 360,             // OWM reports FROM-direction; arrow points TO
            kts: Math.round((p.ws || 0) * 1.944),
            sz: Math.max(0.45, Math.min(1.0, 0.45 + (p.ws || 0) / 25)),
          },
        })),
    };
    try {
      const src: any = map.getSource("wx-grid-wind");
      if (src) src.setData(fc as any);
      else {
        map.addSource("wx-grid-wind", { type: "geojson", data: fc as any } as any);
        map.addLayer({
          id: "wx-wind-arrows", type: "symbol", source: "wx-grid-wind",
          layout: {
            "icon-image": "vt-wind-arrow",
            "icon-rotate": ["get", "rot"],       // always-numeric (MapLibre lesson)
            "icon-size": ["get", "sz"],
            // Arrow + kt text are ONE unit, fully outside the collision
            // pass in BOTH directions (never hidden, never reserving
            // space). REPAIR 2026-07-05: temp value-labels sample the
            // same grid points and, sitting higher in the style, won
            // placement — the arrow/kt pair got collision-split (arrows
            // vanished, orphaned kt survived). Density stays bounded by
            // the server-side sampled grid, so opting out of declutter
            // is safe; the temp label offset (see wx-temp-labels) does
            // the visual dodge at shared points.
            "icon-allow-overlap": true,
            "icon-ignore-placement": true,
            "text-field": ["concat", ["to-string", ["get", "kts"]], " kt"],
            "text-font": ["Open Sans Semibold"],
            "text-size": 9.5,
            "text-offset": [0, 1.3],
            "text-anchor": "top",
            "text-allow-overlap": true,
            "text-ignore-placement": true,
          },
          paint: {
            "icon-color": "#eef3fb",
            "icon-halo-color": "rgba(5,10,19,0.95)",
            "icon-halo-width": 1.2,
            "text-color": "#b3c2d8",
            "text-halo-color": "rgba(5,10,19,0.95)",
            "text-halo-width": 1.1,
          },
        } as any);
      }
    } catch {}
  }, [wxGrid, enabled.weather_wind, windArrows, mapReady]);

  // ── temperature value labels (°F default, °C toggle) ──
  useEffect(() => {
    const map = mapRef.current;
    const on = enabled.weather_temp && tempLabels && wxGrid?.points?.length;
    if (!map || !mapReady) return;
    if (!on) {
      try {
        if (map.getLayer("wx-temp-labels")) map.removeLayer("wx-temp-labels");
        if (map.getSource("wx-grid-temp")) map.removeSource("wx-grid-temp");
      } catch {}
      return;
    }
    const fc = {
      type: "FeatureCollection",
      features: (wxGrid.points as any[])
        .filter((p) => p.tc != null)
        .map((p) => ({
          type: "Feature",
          geometry: { type: "Point", coordinates: [p.lo, p.la] },
          properties: { lbl: tempUnitF ? `${Math.round(p.tc * 9 / 5 + 32)}°F` : `${Math.round(p.tc)}°C` },
        })),
    };
    try {
      const src: any = map.getSource("wx-grid-temp");
      if (src) src.setData(fc as any);
      else {
        map.addSource("wx-grid-temp", { type: "geojson", data: fc as any } as any);
        map.addLayer({
          id: "wx-temp-labels", type: "symbol", source: "wx-grid-temp",
          layout: {
            "text-field": ["get", "lbl"],
            "text-font": ["Open Sans Semibold"],
            "text-size": 11,
            // Shares grid points with the wind arrows: temp reads ABOVE
            // the point, arrow + kt read at/below it (kt offset [0,1.3]).
            // Still declutters against ITSELF at low zoom (allow-overlap
            // stays false) — the arrows layer ignores placement, so the
            // two never contend.
            "text-anchor": "bottom",
            "text-offset": [0, -1.2],
            "text-allow-overlap": false,
          },
          paint: {
            "text-color": "#eef3fb",
            "text-halo-color": "rgba(5,10,19,0.95)",
            "text-halo-width": 1.4,
          },
        } as any);
      }
    } catch {}
  }, [wxGrid, enabled.weather_temp, tempLabels, tempUnitF, mapReady]);

  // ── OpenWeatherMap GLOBAL weather fields (temp/wind) — Tier-1(b) global
  // half, unblocked when the human set OPENWEATHERMAP_KEY (2026-07-04).
  // Tiles come through OUR proxy (/api/data/wxtile) so the key stays
  // server-side and the 60-calls/min free budget is shared-cache bounded.
  // FRESH-KEY RULE (human directive): while OWM activates a new key (~2h,
  // upstream 401), the status is "activating" -> shown as LOADING with a
  // retry note and re-probed every 10 min — never an error state. ──
  useEffect(() => {
    const map = mapRef.current;
    const FIELDS: Array<{ id: "weather_temp" | "weather_wind"; owm: string }> = [
      { id: "weather_temp", owm: "temp_new" },
      { id: "weather_wind", owm: "wind_new" },
    ];
    const anyOn = FIELDS.some((f) => enabled[f.id]);
    const removeField = (f: { id: string; owm: string }) => {
      try {
        if (map?.getLayer(`wx-${f.owm}`)) map.removeLayer(`wx-${f.owm}`);
        if (map?.getSource(`wx-${f.owm}`)) map.removeSource(`wx-${f.owm}`);
      } catch {}
    };
    for (const f of FIELDS) if (!enabled[f.id]) { removeField(f); setStatus(f.id, "off"); }
    if (!anyOn) return;
    if (!map || !mapReady) return;
    let stop = false;
    const probe = async () => {
      for (const f of FIELDS) if (enabled[f.id]) setStatus(f.id, "loading");
      try {
        const r = await fetch("/api/data/weather/global/status");
        const d = await r.json();
        if (stop) return;
        for (const f of FIELDS) {
          if (!enabled[f.id]) continue;
          if (d.status === "ok") {
            if (!map.getSource(`wx-${f.owm}`)) {
              map.addSource(`wx-${f.owm}`, {
                type: "raster",
                tiles: [`/api/data/wxtile/${f.owm}/{z}/{x}/{y}`],
                tileSize: 256, maxzoom: 7,
                attribution: "Weather data © OpenWeatherMap",
              } as any);
              const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
              map.addLayer({
                id: `wx-${f.owm}`, type: "raster", source: `wx-${f.owm}`,
                // registry-native field opacity (default 60%): tiles arrive
                // alpha-amplified from the proxy; the slider owns the blend.
                paint: { "raster-opacity": opacityOf(f.id) / 100 },
              } as any, firstMarker?.id);
            }
            setStatus(f.id, "active", undefined, "global field · Weather data © OpenWeatherMap");
          } else if (d.status === "awaiting_key") {
            removeField(f);
            setStatus(f.id, "awaiting_key");
          } else if (d.status === "activating") {
            // fresh-key delay is NOT an error: keep it in loading with the
            // retry note; the 10-min interval below re-probes.
            removeField(f);
            setStatus(f.id, "loading", undefined, d.note);
          } else {
            removeField(f);
            setStatus(f.id, "error", undefined, d.note || "upstream error — retrying");
          }
        }
      } catch {
        if (!stop) for (const f of FIELDS) if (enabled[f.id]) setStatus(f.id, "error");
      }
    };
    probe();
    const iv = window.setInterval(probe, 10 * 60_000);
    return () => { stop = true; window.clearInterval(iv); };
  }, [enabled.weather_temp, enabled.weather_wind, mapReady, setStatus]);

  // ── generic live-points wiring (aircraft + vessels share the machinery) ──
  const wireLivePoints = useCallback((opts: {
    id: "aircraft" | "vessels";
    intervalMs: number;
    toFeatures: (d: any) => any[];
    toVectors?: (d: any) => any[];
    onClick: (props: any, lngLat: any) => void;
    iconLayout: any;
    iconPaint: any;
    /** E3 LOD hand-off: cap the 2D symbol layer at this zoom (the 3D
     *  silhouette layer takes over above it — one representation at a time). */
    iconMaxZoom?: number;
    /** E3: fresh-snapshot hook — the caller feeds side renderers (the 3D
     *  aircraft layer) from the same payload the symbols use. */
    onData?: (d: any) => void;
    /** O6-4: display-side payload transform (operator/callsign filtering).
     *  Applied AFTER the delta cursor is recorded (the raw feed stays the
     *  delta unit) and before features/vectors/onData/status — one filter,
     *  every renderer. Must return an honest count/coverage_note. */
    transformData?: (d: any) => any;
    /** Low-zoom render decimation ([REPAIR 2026-07-05] perf 3/3): below
     *  splitZoom draw only features with rank < keepFraction (rank is a
     *  stable per-feature hash set by toFeatures). RENDER-side only — the
     *  source always holds every feature (harness data-richness guard
     *  pins >=9500 in source at 1440), and zooming past splitZoom shows
     *  everything. At the default z3.6 view, 10k overlapping icons were
     *  pure overdraw. */
    lowZoom?: { splitZoom: number; keepFraction: number };
    /** DEAD-RECKONING GLIDE (2026-07-18, "planes stopped moving"): between
     *  polls, re-ship the source at stepMs with each row extrapolated along
     *  its BROADCAST velocity (velOf; null = frozen — never a guessed
     *  vector), capped at maxSec then FROZEN (honesty cap — see airGlide).
     *  Runs only in [minZoom, iconMaxZoom) where the motion is visible and
     *  the icons own the map (above, the 3D layer glides in-shader), and
     *  only over rows within the padded viewport — the setData payload
     *  stays a few hundred features, not the full 10k snapshot. Rebuilds
     *  THROUGH toFeatures/toVectors via withRows, so glided symbols and
     *  velocity whiskers can never drift apart. */
    glide?: {
      rows: (d: any) => any[];
      withRows: (d: any, rows: any[]) => any;
      velOf: (row: any) => { dLon: number; dLat: number } | null;
      minZoom: number;
      maxSec: number;
      stepMs: number;
    };
  }) => {
    const map = mapRef.current;
    const { id } = opts;
    let stop = false;
    const srcId = id, layerId = `${id}-sym`, loLayerId = `${id}-sym-lo`, vecSrc = `${id}-vec`, vecLayer = `${id}-veclines`;
    // O6 toggle fix (live report: "click it on and off, the second time the
    // data doesn't load"): every fresh wire starts from a clean slate — no
    // inherited delta cursor, and the FIRST fetch bypasses the browser's
    // HTTP cache (Cache-Control max-age landed in v1.0.340; a cached
    // unchanged/stale body must never decide a fresh mount).
    delete sinceRef.current[id];
    let firstFetch = true;

    // Named handlers so teardown can map.off() them — listeners are keyed
    // by layerId string and SURVIVE layer removal; without off(), each
    // toggle cycle stacked another set (N clicks -> N detail cards + N
    // trail fetches). ([REPAIR 2026-07-05] map perf/correctness.)
    const onClickLayer = (e: any) => {
      const f = e.features?.[0];
      if (f) opts.onClick(f.properties, e.lngLat);
    };
    const onEnter = () => { map.getCanvas().style.cursor = "pointer"; };
    const onLeave = () => { map.getCanvas().style.cursor = ""; };

    const teardown = () => {
      stop = true;
      // Clear the delta cursor: a re-enabled layer must fetch a FULL
      // snapshot. With a stale ?since= the server answers {unchanged:true}
      // and the early return skips addSource/addLayer entirely — the layer
      // stays absent until upstream data changes (toggle-repair 2026-07-04,
      // caught by the toggle-consistency battery).
      delete sinceRef.current[id];
      try {
        for (const l of [layerId, loLayerId]) {
          map.off("click", l, onClickLayer);
          map.off("mouseenter", l, onEnter);
          map.off("mouseleave", l, onLeave);
        }
      } catch {}
      try {
        if (map.getLayer(layerId)) map.removeLayer(layerId);
        if (map.getLayer(loLayerId)) map.removeLayer(loLayerId);
        if (map.getLayer(vecLayer)) map.removeLayer(vecLayer);
        if (map.getSource(srcId)) map.removeSource(srcId);
        if (map.getSource(vecSrc)) map.removeSource(vecSrc);
      } catch {}
    };

    // PERF session #2: last successful fetch's coverage + payload, for the
    // redundant-refetch skip and the lazy vector build (see lib/livePoints).
    let lastFetch: FetchFootprint | null = null;
    let lastPayload: any = null;
    let vectorsCurrent = false;
    // glide state: when the current payload's positions were received, the
    // last dt actually shipped (lets the stepper stop once frozen at the
    // cap instead of re-shipping identical frames forever), and whether the
    // last step shipped an empty set (skip repeat empty setDatas — measured
    // at ~2x idle frame cost on an empty viewport under SwiftShader).
    let glideAnchor: number | null = null;
    let lastGlideDt = -1;
    let lastGlideEmpty = false;

    // add-or-update the velocity-vector source/layer from a payload — shared
    // by the tick path (zoom high enough) and the zoomend lazy build.
    const applyVectors = (d: any) => {
      if (!opts.toVectors) return;
      const vfc = { type: "FeatureCollection", features: opts.toVectors(d) };
      const vsrc: any = map.getSource(vecSrc);
      if (vsrc) {
        vsrc.setData(vfc);
      } else {
        map.addSource(vecSrc, { type: "geojson", data: vfc as any });
        map.addLayer({
          id: vecLayer, type: "line", source: vecSrc,
          minzoom: 6,   // vectors are 2px noise at continent zooms and double
                        // the draw load — appear once you zoom into a region
          paint: { "line-color": ["get", "color"], "line-width": 1, "line-opacity": 0.45 },
        }, layerId);
      }
      vectorsCurrent = true;
    };

    const load = async () => {
      // Hidden-tab gate ([REPAIR 2026-07-05] map perf): a backgrounded /data
      // tab kept polling aircraft 4x/min. Skip while hidden; the
      // visibilitychange listener below refreshes immediately on return
      // (stale-with-timestamp already covers the gap honestly).
      if (document.hidden) return;
      // PERF session #2: never rebuild mid-gesture — a tick landing during a
      // drag stacked a full parse+setData onto the busiest frames; the
      // moveend debounce below already reloads at settle.
      try { if (map.isMoving()) return; } catch {}
      try {
        const b = map.getBounds();
        const since = sinceRef.current[id] || "";
        const q = `lamin=${b.getSouth().toFixed(2)}&lamax=${b.getNorth().toFixed(2)}&lomin=${b.getWest().toFixed(2)}&lomax=${b.getEast().toFixed(2)}${since ? `&since=${since}` : ""}`;
        const r = await fetch(`/api/data/${id}?${q}`, firstFetch ? { cache: "reload" } : undefined);
        firstFetch = false;
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (stop) return;
        if (d.enabled === false) { setStatus(id, "awaiting_key"); return; }
        if (d.unchanged) {
          // delta says nothing new — but if the layer isn't mounted yet the
          // cursor is stale (fresh mount): drop it so the next tick fetches
          // the full snapshot instead of staying invisible forever.
          if (!map.getSource(srcId)) delete sinceRef.current[id];
          return;
        }
        if (d.time != null) sinceRef.current[id] = String(d.time);
        // O6-4: display-side transform (operator filter) — after the delta
        // cursor, before every renderer reads the payload.
        const dd = opts.transformData ? opts.transformData(d) : d;
        // record what this fetch covered (redundant-refetch skip) + keep the
        // payload for the lazy vector build on zoom-in
        try {
          const c = map.getCenter();
          lastFetch = fetchFootprint(c.lat, c.lng, map.getZoom(), b.getNorth(), b.getSouth(), b.getEast(), b.getWest());
        } catch {}
        lastPayload = dd;
        glideAnchor = performance.now(); // fresh REAL positions — glide restarts from truth
        lastGlideDt = -1;

        // Honest feed states (DESIGN.md): partial coverage + staleness shown.
        let note: string | undefined;
        if (dd.coverage === "partial" && dd.coverage_note) note = dd.coverage_note;
        if (dd.filter_note) note = dd.filter_note; // O6-4 operator filter disclosure
        if (dd.stale) note = `stale — data as of ${new Date(dd.stale_at || Date.now()).toLocaleTimeString()}`;

        const features = opts.toFeatures(dd);
        const fc = { type: "FeatureCollection", features };
        const src: any = map.getSource(srcId);
        if (src) {
          src.setData(fc);
        } else {
          map.addSource(srcId, { type: "geojson", data: fc as any });
          const lz = opts.lowZoom;
          map.addLayer({
            id: layerId, type: "symbol", source: srcId,
            ...(lz ? { minzoom: lz.splitZoom } : {}),
            ...(opts.iconMaxZoom ? { maxzoom: opts.iconMaxZoom } : {}),
            layout: opts.iconLayout, paint: opts.iconPaint,
          });
          if (lz) {
            // same source, same styling — just fewer drawn at continent zoom
            map.addLayer({
              id: loLayerId, type: "symbol", source: srcId,
              maxzoom: lz.splitZoom,
              filter: ["<", ["get", "rank"], lz.keepFraction],
              layout: opts.iconLayout, paint: opts.iconPaint,
            });
          }
          for (const l of lz ? [layerId, loLayerId] : [layerId]) {
            map.on("click", l, onClickLayer);
            map.on("mouseenter", l, onEnter);
            map.on("mouseleave", l, onLeave);
          }
        }
        // PERF session #2: the vector pass (a second full iteration + a
        // second structured-clone setData over up to 10k records) only runs
        // when the vec layer can actually be seen; below that the payload is
        // kept and the zoomend handler builds lazily on the way in.
        if (opts.toVectors && shouldBuildVectors(map.getZoom())) applyVectors(dd);
        else vectorsCurrent = false;
        try { opts.onData?.(dd); } catch {} // E3 side renderers — never break the tick
        // Quantized count (nearest-25 above 500, display-only): the exact
        // count jiggled ±dozens per tick, defeating setStatus's no-op bail
        // and re-rendering the entire page component every fresh snapshot.
        setStatus(id, "active", quantizeLiveCount(dd.count ?? features.length), note);
      } catch {
        if (!stop) setStatus(id, "error", undefined, "feed error — backing off, retrying");
      }
    };
    load();
    const iv = window.setInterval(load, opts.intervalMs);
    // GLIDE stepper (~3.3Hz): dead-reckoned setData between polls. Skips
    // whenever it could not be seen (hidden tab, mid-gesture — symbols ride
    // the camera transform anyway, outside the visible-glide zoom band) or
    // could not be honest (no payload, frozen at the cap). Downstream chain
    // (REASONING STANDARD): setData → geojson source re-tile of a few
    // hundred viewport rows (~1-3ms worker-side, measured in the drive);
    // the delta-poll cursor is untouched (server-time based) and the next
    // real payload rebuilds from truth, snapping the glide to zero.
    let glideIv: number | undefined;
    if (opts.glide) {
      const g = opts.glide;
      glideIv = window.setInterval(() => {
        if (stop || document.hidden || glideAnchor == null || !lastPayload) return;
        try { if (map.isMoving()) return; } catch {}
        const z = map.getZoom();
        if (!(z >= g.minZoom)) return;                       // sub-pixel motion — pure cost
        if (opts.iconMaxZoom != null && z >= opts.iconMaxZoom) return; // 3D silhouettes own it
        const dt = Math.min((performance.now() - glideAnchor) / 1000, g.maxSec);
        if (dt <= 0 || dt === lastGlideDt) return;           // frozen at the honesty cap
        lastGlideDt = dt;
        const src: any = map.getSource(srcId);
        if (!src) return;
        let s: number, n2: number, w2: number, e2: number;
        try {
          const b = map.getBounds();
          const mLat = (b.getNorth() - b.getSouth()) * 0.3;  // 30% margin: pans inside
          const mLon = (b.getEast() - b.getWest()) * 0.3;    // coverage refill next step
          s = b.getSouth() - mLat; n2 = b.getNorth() + mLat;
          w2 = b.getWest() - mLon; e2 = b.getEast() + mLon;
        } catch { return; }
        // bounds unwrap past ±180 — test lon and its ±360 aliases so a
        // viewport straddling the antimeridian never DROPS the wrapped side
        const inLon = (lo: number) => (lo >= w2 && lo <= e2) || (lo + 360 >= w2 && lo + 360 <= e2) || (lo - 360 >= w2 && lo - 360 <= e2);
        const glided: any[] = [];
        for (const row of g.rows(lastPayload)) {
          if (row.lat == null || row.lon == null || row.lat < s || row.lat > n2 || !inLon(row.lon)) continue;
          const v = g.velOf(row);
          glided.push(v ? { ...row, lon: row.lon + v.dLon * dt, lat: row.lat + v.dLat * dt } : row);
        }
        if (glided.length === 0 && lastGlideEmpty) return; // nothing to move, nothing to clear
        lastGlideEmpty = glided.length === 0;
        const gliddedPayload = g.withRows(lastPayload, glided);
        src.setData({ type: "FeatureCollection", features: opts.toFeatures(gliddedPayload) });
        if (opts.toVectors && shouldBuildVectors(z)) {
          const vsrc: any = map.getSource(vecSrc);
          if (vsrc) vsrc.setData({ type: "FeatureCollection", features: opts.toVectors(gliddedPayload) });
        }
      }, g.stepMs);
    }
    // Trailing debounce ([REPAIR 2026-07-05] map perf): bare moveend fired a
    // full fetch + 10k-feature rebuild on EVERY camera settle — each wheel
    // step during a zoom was a fetch. Same 400ms pattern the wx-grid effect
    // already used.
    let moveDebounce: number | undefined;
    const onMove = () => {
      window.clearTimeout(moveDebounce);
      moveDebounce = window.setTimeout(() => {
        // PERF session #2: a jitter pan inside the last fetch's served
        // coverage re-downloaded the SAME 250nm circle under a new cache
        // key (full parse + rebuild for nothing). Skip until the camera
        // meaningfully leaves coverage; the interval still polls (cheap
        // `unchanged` answers), so data never ages past one interval.
        try {
          const c = map.getCenter();
          const nb = map.getBounds();
          const next = fetchFootprint(c.lat, c.lng, map.getZoom(), nb.getNorth(), nb.getSouth(), nb.getEast(), nb.getWest());
          if (!needsRefetch(lastFetch, next)) return;
        } catch {}
        void load();
      }, 400);
    };
    map.on("moveend", onMove);
    // Lazy vector build: crossing into vector-visible zoom between ticks
    // builds from the kept payload instead of waiting up to a full interval.
    const onZoomEnd = () => {
      if (vectorsCurrent || !lastPayload || !opts.toVectors) return;
      if (!shouldBuildVectors(map.getZoom())) return;
      try { applyVectors(lastPayload); } catch {}
    };
    map.on("zoomend", onZoomEnd);
    const onVisible = () => { if (!document.hidden) load(); };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      teardown();
      window.clearInterval(iv);
      if (glideIv != null) window.clearInterval(glideIv);
      window.clearTimeout(moveDebounce);
      document.removeEventListener("visibilitychange", onVisible);
      try { map.off("moveend", onMove); } catch {}
      try { map.off("zoomend", onZoomEnd); } catch {}
    };
  }, [setStatus]);

  // ── live aircraft (RAW; WebGL symbols, heading-rotated, class icons;
  // EARTH TWIN E3: at/above AIR_3D_MIN_ZOOM the icons hand off to 3D
  // heading-oriented silhouettes at REAL baro altitude — with pitch +
  // terrain, planes fly above the ground instead of being painted on it) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.aircraft) {
      if (airHoverTipRef.current) airHoverTipRef.current.style.display = "none";
      try {
        if (map.getLayer("aircraft-sym")) map.removeLayer("aircraft-sym");
        if (map.getLayer("aircraft-sym-lo")) map.removeLayer("aircraft-sym-lo");
        if (map.getLayer("aircraft-veclines")) map.removeLayer("aircraft-veclines");
        if (map.getLayer("aircraft-3d")) map.removeLayer("aircraft-3d");
        if (map.getSource("aircraft")) map.removeSource("aircraft");
        if (map.getSource("aircraft-vec")) map.removeSource("aircraft-vec");
      } catch {}
      setStatus("aircraft", "off");
      return;
    }
    setStatus("aircraft", "loading");
    // stable per-aircraft hash for the low-zoom decimation filter — by
    // icao24 so an aircraft never flickers in/out across refreshes
    const rankOf = (s: string) => {
      let h = 0;
      for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
      return (Math.abs(h) % 100) / 100;
    };

    // E3: the 3D silhouette layer — draws nothing below the hand-off zoom or
    // before the first payload; instances rebuilt from each fresh snapshot.
    const airLayer = new AirLayer({ id: "aircraft-3d" });
    try { map.addLayer(airLayer); } catch {}
    (window as any).__vtAir = airLayer; // harness seam (prod-inert, like __vtMap)
    let airRows: any[] = []; // index-aligned to the instance buffer (picking)

    const wire = () => wireLivePoints({
      id: "aircraft",
      intervalMs: 15_000,
      lowZoom: { splitZoom: 4.5, keepFraction: 0.35 },
      // E3: icons cap at the hand-off zoom; the 3D silhouettes take over
      iconMaxZoom: AIR_3D_MIN_ZOOM,
      // O6-4: operator filter — one pass feeds symbols, vectors AND the 3D
      // layer; the note discloses exactly what the filter dropped.
      transformData: (d: any) => applyAirlineFilter(d, airFilter),
      onData: (d: any) => {
        const built = buildAircraftInstances(d.aircraft || []);
        airRows = built.rows;
        airLayer.setInstances(built.inst, built.groups);
        airLayer.setTickTime(); // glide anchor: these positions are true NOW
        // match the terrain mesh's vertical exaggeration so a plane above a
        // peak stays above the exaggerated peak (never-intersect-mountains)
        try { airLayer.setAltScale(map.getTerrain() ? 1.3 : 1); } catch {}
        // SESSION BREADCRUMB: while this plane's card is open, append its
        // fresh REAL fix and repaint the merged trail + curtain so they
        // reach the CURRENT position (the archive lags 1-5 min at cruise).
        // pushCrumb dedupes the server-cache window (same snapshot time).
        try {
          const fid = airCrumbsRef.current.id;
          if (fid) {
            const live = (d.aircraft || []).find((x: any) => x.icao24 === fid);
            if (live && live.lat != null && live.lon != null) {
              const t = Number.isFinite(d.time) ? Number(d.time) : Date.now() / 1000;
              const fix: Crumb = {
                lo: live.lon, la: live.lat,
                al: live.on_ground ? 0 : (live.altitude_m ?? null), t,
              };
              // tail anchor: the same fresh-fix instant the plane's own
              // glide anchors at (setTickTime above)
              airFollowLiveRef.current = {
                id: fid, fix, anchorMs: performance.now(),
                vel: glideDegPerSec(live.lat, live.heading, live.velocity_ms, live.on_ground),
              };
              const before = airCrumbsRef.current.crumbs;
              const after = pushCrumb(before, fix);
              if (after !== before) {
                airCrumbsRef.current.crumbs = after;
                setDetail(prev => prev && prev.trailId === fid ? { ...prev, trailLastT: t } : prev);
              }
              paintFollowedTrail(); // fresh fix and/or re-anchored tail
            }
          }
        } catch { /* trail continuity must never break the tick */ }
      },
      // 2D glide: same dead-reckoning the 3D shader applies (one honesty
      // model, two renderers) — icons stop jumping poll-to-poll at z5.5-8.
      glide: {
        rows: (d: any) => d.aircraft || [],
        withRows: (d: any, rows: any[]) => ({ ...d, aircraft: rows }),
        velOf: (a: any) => glideDegPerSec(a.lat, a.heading, a.velocity_ms, a.on_ground),
        minZoom: AIR_GLIDE_2D_MIN_ZOOM,
        maxSec: MAX_AIR_GLIDE_SEC,
        stepMs: AIR_GLIDE_STEP_MS,
      },
      toFeatures: (d) => (d.aircraft || []).map((a: any) => {
        const cls = classifyAircraft(a.type, a.category);
        return {
          type: "Feature",
          geometry: { type: "Point", coordinates: [a.lon, a.lat] },
          properties: {
            icon: AIRCRAFT_ICON[cls], cls,
            rank: rankOf(String(a.icao24 || a.callsign || "")),
            heading: a.heading ?? 0,
            callsign: a.callsign || a.icao24, icao24: a.icao24,
            country: a.origin_country, type: a.type || "",
            alt: a.altitude_m, ground: !!a.on_ground,
            kts: a.velocity_ms == null ? null : Math.round(a.velocity_ms * 1.944),
          },
        };
      }),
      toVectors: (d) => (d.aircraft || [])
        .filter((a: any) => a.heading != null && !a.on_ground)
        .map((a: any) => ({
          type: "Feature",
          geometry: { type: "LineString", coordinates: [[a.lon, a.lat], velocityEndpoint(a.lat, a.lon, a.heading, a.velocity_ms)] },
          properties: { color: a.on_ground ? "#6680a0" : (a.altitude_m != null && a.altitude_m < 3000 ? "#fbb24c" : "#4d9fff") },
        })),
      iconLayout: {
        "icon-image": ["get", "icon"],
        // Zoom-interpolated: SwiftShader/mid-range-GPU profiling (M4,
        // 2026-07-03) showed the 10k-icon layer is FILL-RATE bound at
        // global zooms — smaller icons where they're dense cut drawn
        // pixels ~60% and roughly halve the layer's frame cost.
        "icon-size": ["interpolate", ["linear"], ["zoom"], 2, 0.32, 5, 0.42, 7, 0.55],
        "icon-rotate": ["get", "heading"],
        "icon-rotation-alignment": "map",
        "icon-allow-overlap": true,
        "icon-ignore-placement": true,
      },
      iconPaint: { "icon-color": ALT_COLOR, "icon-opacity": 0.95 },
      onClick: (p: any, lngLat: any) => { void onAircraftClickProps(p, lngLat); },
    });
    // one card handler for BOTH renderers (2D symbol clicks + 3D picks)
    const onAircraftClickProps = async (p: any, lngLat: any) => {
        const cls = AIRCRAFT_CLASS_LABEL[(p.cls || "unknown") as keyof typeof AIRCRAFT_CLASS_LABEL] || "Aircraft";
        const alt = p.ground === true || p.ground === "true" ? "on ground" : (p.alt != null ? fmtMeters(p.alt) : "alt unknown");
        const dossierKey = `aircraft:${p.icao24}:${Date.now()}`;
        setDetail({
          kind: "aircraft",
          title: `✈ ${p.callsign}`,
          subtitle: `${cls}${p.type ? ` · ${p.type}` : ""} · ${p.country || "—"}`,
          body: `${alt}${p.kts ? ` · ${p.kts} kts` : ""} · hdg ${Math.round(p.heading || 0)}°\n` +
                `Route/flight-plan data unavailable — filed plans are a paid source (wishlist); ` +
                `trail is our own archived feed history — the 3D line + solid ground curtain climb at the RECORDED altitude, colored by the same low/cruise bands as the planes (gaps where altitude wasn't broadcast). ` +
                `Archived history is sampled every 1-5 min, so straight segments join real recorded fixes (never smoothed into invented curves); while this card is open the newest segment extends LIVE at the ~15s feed cadence.`,
          trailId: p.icao24, trailKind: "aircraft", dossierKey,
          links: [
            { label: "Photos/registry (Planespotters)", href: `https://www.planespotters.net/hex/${String(p.icao24 || "").toUpperCase()}` },
            { label: "Live track (adsb.lol)", href: `https://adsb.lol/?icao=${p.icao24}` },
          ],
        });
        // entity spine enrichment (BUILD ORDER 2 #1): owner/model from the
        // FAA registry by exact Mode S hex. Non-US hexes have no entity —
        // the card simply shows nothing extra, never a guess.
        fetch(`/api/data/aircraft/entity/${p.icao24}`)
          .then((r) => (r.ok ? r.json() : null))
          .then((d) => {
            const e = d?.entity;
            if (!e?.owner) return;
            const bits = [e.owner, [e.mfr, e.model].filter(Boolean).join(" "), e.year_mfr].filter(Boolean);
            setDetail(prev => prev && prev.trailId === p.icao24
              ? { ...prev, owner: `${bits.join(" · ")} — ${e.n_number}, FAA registry` } : prev);
          })
          .catch(() => {});
        // Aircraft aren't Everything Graph nodes yet (see research/
        // open_questions.md) — entity=null, but lat/lon alone still surfaces
        // nearest strategic sites.
        fetchDossier(dossierKey, null, lngLat?.lat, lngLat?.lng);
        const { note, lastT } = await showTrail("aircraft", p.icao24);
        setDetail(prev => prev && prev.trailId === p.icao24 ? { ...prev, trailNote: note, trailLastT: lastT } : prev);
    };
    const stopWire = wire();
    // E3 picking above the hand-off zoom: custom layers have no
    // queryRenderedFeatures (the satellite-picking precedent) — CPU nearest
    // over the instance buffer; any real rendered feature keeps priority.
    // PICK FIX (live bug: at 45° tilt the plane renders displaced by its
    // altitude — ground-mercator picking never registered the click).
    // Globe mode projects instances with the frame matrix (screen-space);
    // mercator mode keeps the ground pick.
    const pickAir = (e: any, tolPx: number): number => {
      // pick at the GLIDED position — the pixels are dead-reckoned up to
      // MAX_AIR_GLIDE_SEC ahead of the poll positions (many px at z8+)
      const dtSec = airLayer.getGlideDtSec();
      const matrix = airLayer.getGlobeProjection();
      if (matrix) {
        const canvas = map.getCanvas();
        return pickNearestAircraftScreen(
          airLayer.getInstances(), matrix, e.point.x, e.point.y,
          canvas.clientWidth || 1, canvas.clientHeight || 1, tolPx, dtSec);
      }
      const ll = map.unproject(e.point);
      const merc = lonLatToMercator(ll.lng, ll.lat);
      return pickNearestAircraft(airLayer.getInstances(), merc.x, merc.y, pixelToleranceToMercUnits(tolPx, map.getZoom()), dtSec);
    };
    const onAir3dClick = (e: any) => {
      if (map.getZoom() < AIR_3D_MIN_ZOOM) return;
      let atPoint: unknown[] = [];
      try { atPoint = map.queryRenderedFeatures(e.point) ?? []; } catch {}
      if (atPoint.length > 0) return; // a feature-scoped handler owns this click
      const ll = map.unproject(e.point);
      const idx = pickAir(e, 12);
      if (idx < 0) return;
      const a = airRows[idx];
      if (!a) return;
      void onAircraftClickProps({
        cls: classifyAircraft(a.type, a.category),
        callsign: a.callsign || a.icao24, icao24: a.icao24,
        country: a.origin_country, type: a.type || "",
        alt: a.altitude_m, ground: !!a.on_ground, heading: a.heading ?? 0,
        kts: a.velocity_ms == null ? null : Math.round(a.velocity_ms * 1.944),
      }, ll);
    };
    map.on("click", onAir3dClick);
    // altitude-on-hover (E3 follow-up, deferred from the original polish
    // slice): same CPU-nearest pick + precedence rule as the click handler
    // above, rAF-throttled so a fast mousemove stream never re-picks more
    // than once per frame. Writes the tooltip DOM node directly (no
    // setState) — see airHoverTipRef's declaration for why.
    let hoverFrame: number | null = null;
    const hideHoverTip = () => { if (airHoverTipRef.current) airHoverTipRef.current.style.display = "none"; };
    const onAir3dMove = (e: any) => {
      if (hoverFrame != null) return;
      hoverFrame = requestAnimationFrame(() => {
        hoverFrame = null;
        const tip = airHoverTipRef.current;
        if (!tip) return;
        if (map.getZoom() < AIR_3D_MIN_ZOOM) { hideHoverTip(); return; }
        let atPoint: unknown[] = [];
        try { atPoint = map.queryRenderedFeatures(e.point) ?? []; } catch {}
        if (atPoint.length > 0) { hideHoverTip(); return; } // a feature-scoped layer owns this pixel
        const idx = pickAir(e, 10);
        const a = idx >= 0 ? airRows[idx] : null;
        if (!a) { hideHoverTip(); return; }
        const alt = a.on_ground ? "on ground" : (a.altitude_m != null ? fmtMeters(a.altitude_m) : "alt unknown");
        tip.textContent = `${a.callsign || a.icao24 || "aircraft"} · ${alt}`;
        const oe = e.originalEvent as MouseEvent | undefined;
        tip.style.left = `${(oe?.clientX ?? e.point.x) + 14}px`;
        tip.style.top = `${(oe?.clientY ?? e.point.y) + 14}px`;
        tip.style.display = "block";
      });
    };
    map.on("mousemove", onAir3dMove);
    map.on("mouseout", hideHoverTip);
    // 3D glide low-rate repaint (~3.3Hz): re-evaluates u_dtSec between polls
    // at z8+ where the silhouettes own the map; the layer upgrades itself to
    // per-frame self-repaint at close zooms (shouldGlidePerFrame) and stops
    // at the honesty cap. No-op below the hand-off or with no planes. The
    // same tick keeps the followed plane's curtain tail meeting the drawn
    // (glided) plane — a <1ms wall rebuild, only while a card is open.
    const glideRepaintIv = window.setInterval(() => {
      airLayer.glideRepaintTick();
      try { if (!document.hidden && airFollowLiveRef.current) paintFollowedTrail(); } catch {}
    }, AIR_GLIDE_STEP_MS);
    return () => {
      stopWire();
      window.clearInterval(glideRepaintIv);
      try { map.off("click", onAir3dClick); } catch {}
      try { map.off("mousemove", onAir3dMove); } catch {}
      try { map.off("mouseout", hideHoverTip); } catch {}
      if (hoverFrame != null) cancelAnimationFrame(hoverFrame);
      hideHoverTip();
      try { delete (window as any).__vtAir; } catch {}
      try { if (map.getLayer("aircraft-3d")) map.removeLayer("aircraft-3d"); } catch {}
    };
  }, [enabled.aircraft, mapReady, wireLivePoints, setStatus, airFilter]);

  // ── live vessels (RAW; class icons + heading, destination from AIS) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    const meta = layers.find(l => l.id === "vessels");
    if (meta && meta.status === "awaiting_key") setStatus("vessels", "awaiting_key");
    if (!enabled.vessels) {
      try {
        if (map.getLayer("vessels-sym")) map.removeLayer("vessels-sym");
        if (map.getLayer("vessels-veclines")) map.removeLayer("vessels-veclines");
        if (map.getSource("vessels")) map.removeSource("vessels");
        if (map.getSource("vessels-vec")) map.removeSource("vessels-vec");
      } catch {}
      if (!meta || meta.status !== "awaiting_key") setStatus("vessels", "off");
      return;
    }
    setStatus("vessels", "loading");
    return wireLivePoints({
      id: "vessels",
      intervalMs: 20_000,
      toFeatures: (d) => (d.vessels || []).map((v: any) => {
        const cls = classifyVessel(v.shiptype);
        return {
          type: "Feature",
          geometry: { type: "Point", coordinates: [v.lon, v.lat] },
          properties: {
            icon: VESSEL_ICON[cls], cls, color: VESSEL_COLOR[cls],
            heading: v.cog ?? 0,
            name: v.name, mmsi: v.mmsi,
            kts: v.sog == null ? null : Math.round(v.sog),
            destination: v.destination || "",
          },
        };
      }),
      toVectors: (d) => (d.vessels || [])
        .filter((v: any) => v.cog != null && (v.sog || 0) > 0.5)
        .map((v: any) => ({
          type: "Feature",
          geometry: { type: "LineString", coordinates: [[v.lon, v.lat], velocityEndpoint(v.lat, v.lon, v.cog, (v.sog || 0) * 0.5144, 0.25)] },
          properties: { color: VESSEL_COLOR[classifyVessel(v.shiptype)] },
        })),
      iconLayout: {
        "icon-image": ["get", "icon"],
        // same fill-rate treatment as aircraft (M4 profile)
        "icon-size": ["interpolate", ["linear"], ["zoom"], 2, 0.3, 5, 0.38, 7, 0.5],
        "icon-rotate": ["get", "heading"],
        "icon-rotation-alignment": "map",
        "icon-allow-overlap": true,
        "icon-ignore-placement": true,
      },
      iconPaint: { "icon-color": ["get", "color"], "icon-opacity": 0.95 },
      onClick: async (p, lngLat) => {
        const cls = VESSEL_CLASS_LABEL[(p.cls || "other") as keyof typeof VESSEL_CLASS_LABEL] || "Vessel";
        const flag = mmsiFlag(p.mmsi);
        const dossierKey = `vessel:${p.mmsi}:${Date.now()}`;
        setDetail({
          kind: "vessel",
          title: `⚓ ${p.name}`,
          subtitle: `${cls} · MMSI ${p.mmsi}${flag ? ` · ${flag}` : ""}`,
          body: `${p.kts != null ? `${p.kts} kts · ` : ""}hdg ${Math.round(p.heading || 0)}°` +
                `${p.destination ? `\nDestination (AIS-broadcast): ${p.destination}` : "\nDestination: not broadcast"}`,
          trailId: p.mmsi, trailKind: "vessels", dossierKey,
          links: [
            { label: "MarineTraffic", href: `https://www.marinetraffic.com/en/ais/details/ships/mmsi:${p.mmsi}` },
            { label: "VesselFinder", href: `https://www.vesselfinder.com/vessels/details/${p.mmsi}` },
          ],
        });
        // vessel:<mmsi> resolves to a real Everything Graph node only if this
        // vessel has an archived port call (calls_at edge) — otherwise the
        // dossier degrades to nearest_sites-only, honestly, not an error.
        fetchDossier(dossierKey, `vessel:${p.mmsi}`, lngLat?.lat, lngLat?.lng);
        const { note, lastT } = await showTrail("vessels", p.mmsi);
        setDetail(prev => prev && prev.trailId === p.mmsi ? { ...prev, trailNote: note, trailLastT: lastT } : prev);
      },
    });
  }, [enabled.vessels, mapReady, layers, wireLivePoints, setStatus]);

  // ── strategic sites (RAW; opens the detail card) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.sites) {
      try {
        if (map.getLayer("sites-icons")) map.removeLayer("sites-icons");
        if (map.getSource("sites")) map.removeSource("sites");
      } catch {}
      setStatus("sites", "off");
      return;
    }
    setStatus("sites", "loading");
    let detach = () => {};
    const stopLoad = runResilientLoad(
      async (signal) => {
        const r = await fetch("/api/data/sites", { signal });
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (signal.aborted) return;
        if (!d.sites) throw new Error("no sites in response");
        if (map.getSource("sites")) return;
        const colors: Record<string, string> = {};
        const catLabels: Record<string, string> = {};
        Object.entries(d.categories || {}).forEach(([k, v]: any) => { colors[k] = v.color; catLabels[k] = v.label; });
        map.addSource("sites", { type: "geojson", data: {
          type: "FeatureCollection",
          features: d.sites.map((s: any) => ({
            type: "Feature",
            geometry: { type: "Point", coordinates: [s.lon, s.lat] },
            properties: {
              id: s.id,
              name: s.name, category: catLabels[s.category] || s.category,
              operator: s.operator, relevance: s.relevance,
              color: colors[s.category] || "#4d9fff",
              icon: SITE_ICON[s.category] || "vt-tank",
            },
          })),
        } as any });
        // Category silhouettes (anchor / tank cluster / factory), same SDF
        // system as aircraft/vessel classes — per-feature icon + icon-color,
        // upright (sites don't rotate), dark halo for imagery contrast.
        map.addLayer({
          id: "sites-icons", type: "symbol", source: "sites",
          layout: {
            "icon-image": ["get", "icon"],
            "icon-size": ["interpolate", ["linear"], ["zoom"], 3, 0.55, 8, 0.85],
            "icon-allow-overlap": true,
            "icon-ignore-placement": true,
          },
          paint: {
            "icon-color": ["get", "color"],
            "icon-halo-color": "rgba(5,10,19,0.95)",
            "icon-halo-width": 1.4,
          },
        });
        detach = attachLayerInteractions(map, "sites-icons", (e: any) => {
          const f = e.features?.[0];
          if (!f) return;
          const dossierKey = `site:${f.properties.id}:${Date.now()}`;
          setDetail({
            kind: "site",
            title: f.properties.name,
            subtitle: `${f.properties.category} · ${f.properties.operator}`,
            body: f.properties.relevance,
            dossierKey,
          });
          // Everything Graph R1: async 7-day cross-stream timeline; any
          // failure just leaves the section absent — the card never degrades
          if (f.properties.id) {
            fetch(`/api/data/site-timeline/${f.properties.id}`)
              .then((r) => (r.ok ? r.json() : null))
              .then((d) => {
                if (!d) return;
                setDetail((prev) => prev && prev.title === f.properties.name
                  ? { ...prev, timeline: { events: d.events || [], density: d.density || {} } } : prev);
              })
              .catch(() => {});
            // ENTITY DOSSIER v2 (W5): sites are facility:site:<id> nodes —
            // resolveEntityId's bare-id suffix match finds them directly.
            fetchDossier(dossierKey, f.properties.id, e.lngLat?.lat, e.lngLat?.lng);
          }
        });
        setStatus("sites", "active", d.sites.length);
      },
      (failures) => setStatus("sites", "error", undefined,
        failures === 0 ? "load failed — retrying automatically…" : "still retrying automatically…"),
    );
    return () => { stopLoad(); detach(); };
  }, [enabled.sites, mapReady, setStatus]);

  // ── US power plants (RAW; static reference data, WRI GPPD CC BY 4.0) ──
  // ~9.8k plants: maplibre native clustering keeps low zooms legible and
  // cheap on phones (DESIGN.md performance budget — clustering is
  // client-side, the server serves one cached static JSON). Unclustered
  // points render fuel-type SDF silhouettes with per-feature tint.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.powerplants) {
      try {
        for (const l of ["pp-points", "pp-cluster-count", "pp-clusters"]) if (map.getLayer(l)) map.removeLayer(l);
        if (map.getSource("powerplants")) map.removeSource("powerplants");
      } catch {}
      setStatus("powerplants", "off");
      return;
    }
    if (!mapSettled) { setStatus("powerplants", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("powerplants", "loading");
    let detach = () => {};
    const stopLoad = runResilientLoad(
      async (signal) => {
        const r = await fetch("/api/data/powerplants", { signal });
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (signal.aborted) return;
        if (!d.plants) throw new Error("no plants in response");
        if (map.getSource("powerplants")) return;
        map.addSource("powerplants", {
          type: "geojson",
          cluster: true, clusterMaxZoom: 7, clusterRadius: 50,
          data: {
            type: "FeatureCollection",
            features: d.plants.map(([name, mw, fuel, owner, lat, lon, verified]: [string, number, string, string, number, number, number], idx: number) => ({
              type: "Feature",
              geometry: { type: "Point", coordinates: [lon, lat] },
              properties: {
                name, mw, fuel, owner, verified: verified === 1,
                // W5 entity dossier: entityGraph.ts builds plantFacilityId(idx)
                // from this SAME datacore/powerplants/us_power_plants.json
                // array, unfiltered/unsorted — the index is stable as long as
                // both this route and entityGraph.ts read the file verbatim
                // (verified true of both today).
                plantId: idx,
                icon: POWER_FUEL_ICON[fuel] || "vt-power",
                color: POWER_FUEL_COLOR[fuel] || "#6680a0",
              },
            })),
          } as any,
        });
        map.addLayer({
          id: "pp-clusters", type: "circle", source: "powerplants",
          filter: ["has", "point_count"],
          paint: {
            "circle-color": "rgba(77,159,255,0.28)",
            "circle-stroke-color": "rgba(124,196,255,0.85)",
            "circle-stroke-width": 1.4,
            "circle-radius": ["step", ["get", "point_count"], 12, 25, 16, 100, 21, 500, 27],
          },
        });
        map.addLayer({
          id: "pp-cluster-count", type: "symbol", source: "powerplants",
          filter: ["has", "point_count"],
          layout: {
            "text-field": ["get", "point_count_abbreviated"],
            "text-font": ["Open Sans Semibold"],
            "text-size": 11,
            "text-allow-overlap": true,
          },
          paint: { "text-color": "#eef3fb" },
        });
        map.addLayer({
          id: "pp-points", type: "symbol", source: "powerplants",
          filter: ["!", ["has", "point_count"]],
          layout: {
            "icon-image": ["get", "icon"],
            "icon-size": ["interpolate", ["linear"], ["zoom"], 6, 0.5, 10, 0.8],
            "icon-allow-overlap": true,
            "icon-ignore-placement": true,
          },
          paint: {
            "icon-color": ["get", "color"],
            "icon-halo-color": "rgba(5,10,19,0.95)",
            "icon-halo-width": 1.3,
          },
        });
        const detachClusters = attachLayerInteractions(map, "pp-clusters", (e: any) => {
          const f = e.features?.[0];
          if (!f) return;
          const src: any = map.getSource("powerplants");
          src.getClusterExpansionZoom(f.properties.cluster_id, (err: any, zoom: number) => {
            if (!err) map.easeTo({ center: f.geometry.coordinates, zoom: zoom + 0.3 });
          });
        });
        const detachPoints = attachLayerInteractions(map, "pp-points", (e: any) => {
          const f = e.features?.[0];
          if (!f) return;
          const p = f.properties;
          const dossierKey = `powerplant:${p.plantId}:${Date.now()}`;
          setDetail({
            kind: "powerplant",
            title: p.name,
            subtitle: `${POWER_FUEL_LABEL[p.fuel] || p.fuel} · ${Number(p.mw).toLocaleString()} MW`,
            body: `${p.owner ? `Operator: ${p.owner}\n` : ""}` +
                  `${p.verified === true || p.verified === "true"
                     ? "Position imagery-verified.\n"
                     : "Position approximate (registry-reported — GPPD/EIA-860).\n"}` +
                  // wind-centroid caveat (position audit 2026-07-18): a CORRECT
                  // wind-farm point is the farm centroid — turbines spread over
                  // km around it, so imagery at the exact marker can honestly
                  // show empty ground (the human's screenshot case, distinct
                  // from the 4 genuinely-wrong coords the audit fixed).
                  `${p.fuel === "wind"
                     ? "Wind farms span many turbines — this marker is the farm centroid; the nearest turbine may be a few km away.\n"
                     : ""}` +
                  `Static reference data — WRI GPPD v1.3.0 (CC BY 4.0) + EIA-860.`,
            dossierKey,
          });
          fetchDossier(dossierKey, `facility:plant:${p.plantId}`, e.lngLat?.lat, e.lngLat?.lng);
        });
        detach = () => { detachClusters(); detachPoints(); };
        setStatus("powerplants", "active", d.count ?? d.plants.length,
          `top ${d.verified_count ?? 100} by MW imagery-verified · rest approximate`);
      },
      (failures) => setStatus("powerplants", "error", undefined,
        failures === 0 ? "load failed — retrying automatically…" : "still retrying automatically…"),
    );
    return () => { stopLoad(); detach(); };
  }, [enabled.powerplants, mapReady, mapSettled, setStatus]);

  // ── EPA Superfund NPL sites (RAW/FACTUAL hazard layer; U.S. EPA SEMS, public
  // domain — first Location Context Engine hazard layer. Points colored by NPL
  // status; every site passed the server-side data-quality gate. Facts only —
  // location/status/HRS score — never a risk claim about a specific property. ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.superfund) {
      try {
        if (map.getLayer("superfund-pts")) map.removeLayer("superfund-pts");
        if (map.getSource("superfund")) map.removeSource("superfund");
      } catch {}
      setStatus("superfund", "off");
      return;
    }
    if (!mapSettled) { setStatus("superfund", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("superfund", "loading");
    let detach = () => {};
    const stopLoad = runResilientLoad(
      async (signal) => {
        const r = await fetch("/api/data/superfund", { signal });
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (signal.aborted) return;
        if (!Array.isArray(d.sites)) throw new Error("no sites in response");
        if (d.warming_up) { setStatus("superfund", "loading", undefined, "warming up — EPA fetch in progress"); return; }
        if (map.getSource("superfund")) return;
        map.addSource("superfund", {
          type: "geojson",
          data: {
            type: "FeatureCollection",
            features: d.sites.map((s: any) => ({
              type: "Feature",
              geometry: { type: "Point", coordinates: [s.lon, s.lat] },
              properties: s,
            })),
          } as any,
          attribution: "U.S. EPA Superfund (SEMS/NPL), public domain",
        } as any);
        map.addLayer({
          id: "superfund-pts", type: "circle", source: "superfund", minzoom: 3,
          paint: {
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 3, 2.2, 8, 5, 13, 8],
            // color by NPL status: active red, proposed orange, deleted gray
            "circle-color": ["match", ["get", "status"],
              "NPL Site", "#ef4444",
              "Proposed NPL Site", "#fb923c",
              "Deleted NPL Site", "#94a3b8",
              "Partial NPL Deletion", "#fbbf24",
              "#a78bfa"],
            "circle-opacity": 0.85,
            "circle-stroke-color": "rgba(8,12,20,0.9)", "circle-stroke-width": 0.6,
          },
        } as any);
        detach = attachLayerInteractions(map, "superfund-pts", (e: any) => {
          const f = e.features?.[0]; if (!f) return; const p = f.properties;
          setDetail({
            kind: "superfund",
            title: p.name || "Superfund site",
            subtitle: `${p.status}${p.hrs_score != null && p.hrs_score !== "" ? ` · HRS ${Number(p.hrs_score).toFixed(1)}` : ""}`,
            body: `${[p.city, p.county && `${p.county} County`, p.state].filter(Boolean).join(", ")}\n` +
                  `${p.listed ? `Listed: ${p.listed}\n` : ""}` +
                  `${p.epa_id ? `EPA ID: ${p.epa_id}\n` : ""}` +
                  `\nEPA Superfund NPL (SEMS, public domain). Factual site record — location + status + Hazard Ranking System score as published; not a risk claim about any specific property.`,
          });
        });
        const h = d.health;
        setStatus("superfund", "active", d.sites.length,
          `U.S. EPA Superfund NPL — ${d.sites.length.toLocaleString()} sites${h?.suspect ? ` (${h.suspect} quarantined by the data-quality gate)` : ""} · ${h?.freshness || "public domain"}`);
      },
      (failures) => setStatus("superfund", "error", undefined,
        failures === 0 ? "load failed — retrying automatically…" : "still retrying automatically…"),
    );
    return () => { stopLoad(); detach(); };
  }, [enabled.superfund, mapReady, mapSettled, setStatus]);

  // ── PFAS drinking-water detections (RAW/FACTUAL hazard layer; EPA UCMR 5,
  // public domain — location_context_engine.md's queued PFAS item. Droplet-
  // with-chemical-ring SYMBOLS (not bare dots, per the symbols directive)
  // tinted by the COUNT of distinct PFAS compounds detected at that system —
  // a factual count, never a concentration/risk tier. Location is the
  // system's service-area centroid, an approximation (stated in the card).
  // Static artifact (server/pfas.ts), rebuilt session-side — no boot poll. ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.pfas) {
      try {
        if (map.getLayer("pfas-pts")) map.removeLayer("pfas-pts");
        if (map.getSource("pfas")) map.removeSource("pfas");
      } catch {}
      setStatus("pfas", "off");
      return;
    }
    if (!mapSettled) { setStatus("pfas", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("pfas", "loading");
    let detach = () => {};
    const stopLoad = runResilientLoad(
      async (signal) => {
        const r = await fetch("/api/data/pfas", { signal });
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (signal.aborted) return;
        if (!Array.isArray(d.systems)) throw new Error("no systems in response");
        if (map.getSource("pfas")) return;
        map.addSource("pfas", {
          type: "geojson",
          data: {
            type: "FeatureCollection",
            features: d.systems.map((s: any) => ({
              type: "Feature",
              geometry: { type: "Point", coordinates: [s.lon, s.lat] },
              properties: s,
            })),
          } as any,
          attribution: "U.S. EPA UCMR 5, public domain",
        } as any);
        map.addLayer({
          id: "pfas-pts", type: "symbol", source: "pfas", minzoom: 3,
          layout: {
            "icon-image": "vt-pfas",
            "icon-size": ["interpolate", ["linear"], ["zoom"], 3, 0.3, 8, 0.5, 13, 0.7],
            "icon-allow-overlap": false,
          },
          paint: {
            "icon-color": ["step", ["get", "n_analytes_detected"],
              "#fde047",
              2, "#fb923c",
              4, "#f87171"],
            "icon-halo-color": "rgba(8,12,20,0.9)", "icon-halo-width": 1.1,
          },
        } as any);
        detach = attachLayerInteractions(map, "pfas-pts", (e: any) => {
          const f = e.features?.[0]; if (!f) return; const p = f.properties;
          const dets: any[] = typeof p.detections === "string" ? JSON.parse(p.detections) : (p.detections || []);
          const lines = dets.map((det: any) =>
            `  ${det.contaminant}: max ${det.max_value} ${det.units} (${det.n_detections} detection${det.n_detections === 1 ? "" : "s"}, last ${det.last_detected})`);
          setDetail({
            kind: "pfas",
            title: p.name || "Water system",
            subtitle: `${p.n_analytes_detected} PFAS compound${p.n_analytes_detected === 1 ? "" : "s"} detected${p.population_served ? ` · serves ~${Number(p.population_served).toLocaleString()} people` : ""}`,
            body: `${lines.join("\n")}\n\n` +
                  `Location is this system's service-area CENTROID (an approximation, not the exact intake). ` +
                  `EPA UCMR 5 monitoring (2023-2025 cycle), public domain. NOT an MCL-exceedance or health-risk ` +
                  `claim — PFAS MCL compliance monitoring under the 2024 NPDWR doesn't begin until 2027-2029, ` +
                  `and UCMR5 predates that compliance data. Marker color is a display bucket of the detected-` +
                  `compound COUNT (see legend), not a concentration or risk threshold.`,
          });
        });
        const h = d.health;
        setStatus("pfas", "active", d.systems.length,
          `EPA UCMR 5 — ${d.systems.length.toLocaleString()} water systems with a PFAS detection${h?.suspect ? ` (${h.suspect} quarantined by the data-quality gate)` : ""} · ${h?.freshness || "public domain"}`);
      },
      (failures) => setStatus("pfas", "error", undefined,
        failures === 0 ? "load failed — retrying automatically…" : "still retrying automatically…"),
    );
    return () => { stopLoad(); detach(); };
  }, [enabled.pfas, mapReady, mapSettled, setStatus]);

  // ── FEMA flood hazard zones (RAW; location_context_engine.md hazard layer
  // #3 — "the closest direct Zillow parallel"). Raster overlay rendered LIVE
  // by FEMA's own public ArcGIS MapServer (hazards.fema.gov, CORS-open,
  // verified live) via MapLibre's `{bbox-epsg-3857}` template token — the
  // same "zero server cost, tiles from someone else's public service"
  // pattern as surfacewater/forest above, no data archived by us for the
  // overlay itself (the click-anywhere dossier's flood_zone field is the
  // separate server-side point lookup, server/femaFlood.ts). FEMA's own
  // scale limit (minScale ~1:36k) means this legitimately renders blank
  // until zoomed to roughly property level — stated in the status note, not
  // a bug. field:true — opacity slider inherited. ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.floodzones) {
      try {
        if (map.getLayer("fema-floodzones")) map.removeLayer("fema-floodzones");
        if (map.getSource("fema-floodzones")) map.removeSource("fema-floodzones");
      } catch {}
      setStatus("floodzones", "off");
      return;
    }
    try {
      if (!map.getSource("fema-floodzones")) {
        map.addSource("fema-floodzones", {
          type: "raster",
          tiles: [
            "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/export?" +
            "bbox={bbox-epsg-3857}&bboxSR=102100&imageSR=102100&size=256,256&format=png32&" +
            "transparent=true&layers=show:28&f=image",
          ],
          tileSize: 256, maxzoom: 16,
          attribution: "Flood hazard zones © FEMA (public domain)",
        } as any);
      }
      if (!map.getLayer("fema-floodzones")) {
        const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
        map.addLayer({
          id: "fema-floodzones", type: "raster", source: "fema-floodzones",
          paint: { "raster-opacity": opacityOf("floodzones") / 100 },
        } as any, firstMarker?.id);
      }
      setStatus("floodzones", "active", undefined,
        "FEMA National Flood Hazard Layer · rendered live by FEMA's own map service — only visible " +
        "zoomed to roughly property level (FEMA's own scale limit, not a bug); click anywhere for the " +
        "exact zone/SFHA status at that point");
    } catch {
      setStatus("floodzones", "error");
    }
  }, [enabled.floodzones, mapReady, setStatus]);

  // ── nuclear tests time machine (RAW/FACTUAL; SIPRI/Johnston archive catalog,
  // 1945-1998 — 2,027 located tests; 24 unlocated quarantined server-side.
  // The "Lucy" scrub: a year slider drives a GPU filter (["<=",["get","y"],
  // year]) over one static geojson source — no refetch per tick, so dragging
  // through five decades is smooth. Tests IN the selected year pulse larger.
  // SYMBOL DIRECTIVE 2026-07-12: emplacement silhouettes (airburst / surface-
  // tower / water / underground shaft) instead of dots, country tint kept,
  // yield-scaled size — what KIND of shot it was reads at a glance. Blast
  // rings are 5-psi ESTIMATES (Glasstone & Dolan cube-root scaling, 0.47 km ×
  // ∛kt, surface-burst figure) drawn as true ground-distance polygons — an
  // honest labeled estimate of severe-blast reach, NEVER fallout modeling
  // (fallout depends on weather/burial the catalog doesn't carry). Buried
  // shots get no ring: contained. Facts only otherwise. ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.nucleartests) {
      try {
        for (const l of ["nuke-ring-fill", "nuke-ring-year", "nuke-pts", "nuke-year"]) if (map.getLayer(l)) map.removeLayer(l);
        for (const s of ["nucleartests", "nucleartests-rings"]) if (map.getSource(s)) map.removeSource(s);
      } catch {}
      setStatus("nucleartests", "off");
      return;
    }
    if (!mapSettled) { setStatus("nucleartests", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("nucleartests", "loading");
    let detach = () => {};
    const stopLoad = runResilientLoad(
      async (signal) => {
        const r = await fetch("/api/data/nucleartests", { signal });
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (signal.aborted || !Array.isArray(d.tests)) throw new Error("no tests in response");
        if (map.getSource("nucleartests")) return;
        map.addSource("nucleartests", {
          type: "geojson",
          data: {
            type: "FeatureCollection",
            features: d.tests.map((t: any) => ({
              type: "Feature", geometry: { type: "Point", coordinates: [t.lon, t.lat] },
              properties: { ...t, cls: classifyNukeTest(t.t) },
            })),
          } as any,
          attribution: "SIPRI / Johnston archive nuclear test catalog",
        } as any);
        // 5-psi blast-radius rings as true ground-distance polygons (48-gon,
        // equirectangular — radii are a few km, distortion negligible). Only
        // above-ground/water shots with a catalogued yield get one.
        const ringFeatures: any[] = [];
        for (const t of d.tests) {
          const rkm = blastRadiusKm(t.kt, t.t);
          if (!rkm) continue;
          const dLat = rkm / 111.32;
          const dLon = rkm / (111.32 * Math.max(0.15, Math.cos((t.lat * Math.PI) / 180)));
          const ring: [number, number][] = [];
          for (let i = 0; i <= 48; i++) {
            const a = (i * 2 * Math.PI) / 48;
            ring.push([t.lon + dLon * Math.cos(a), t.lat + dLat * Math.sin(a)]);
          }
          ringFeatures.push({
            type: "Feature", geometry: { type: "Polygon", coordinates: [ring] },
            properties: { y: t.y, kt: t.kt },
          });
        }
        map.addSource("nucleartests-rings", {
          type: "geojson",
          data: { type: "FeatureCollection", features: ringFeatures } as any,
        } as any);
        // cumulative faint blast footprints up to the selected year…
        map.addLayer({
          id: "nuke-ring-fill", type: "fill", source: "nucleartests-rings",
          filter: ["<=", ["get", "y"], histYear],
          paint: { "fill-color": "rgba(253,224,71,0.05)", "fill-outline-color": "rgba(253,224,71,0.18)" },
        } as any);
        // …and the selected year's rings bright
        map.addLayer({
          id: "nuke-ring-year", type: "line", source: "nucleartests-rings",
          filter: ["==", ["get", "y"], histYear],
          paint: { "line-color": "#fde047", "line-width": 1.6, "line-opacity": 0.85 },
        } as any);
        // emplacement silhouette, country tint, yield-scaled size
        const kt = ["coalesce", ["to-number", ["get", "kt"], 0], 0] as any;
        const sizeAt = (base: number) =>
          ["min", base * 2.4, ["+", base, ["/", ["sqrt", kt], 90 / base]]] as any;
        map.addLayer({
          id: "nuke-pts", type: "symbol", source: "nucleartests",
          filter: ["<=", ["get", "y"], histYear],
          layout: {
            "icon-image": ["match", ["get", "cls"],
              "air", NUKE_CLASS_ICON.air, "water", NUKE_CLASS_ICON.water,
              "underground", NUKE_CLASS_ICON.underground, NUKE_CLASS_ICON.ground],
            "icon-size": ["interpolate", ["linear"], ["zoom"], 2, sizeAt(0.34), 8, sizeAt(0.62)],
            "icon-allow-overlap": true,      // history must not decimate under collision
          },
          paint: {
            "icon-color": ["match", ["get", "c"],
              "USA", NUKE_COUNTRY_COLOR.USA, "USSR", NUKE_COUNTRY_COLOR.USSR,
              "FRANCE", NUKE_COUNTRY_COLOR.FRANCE, "UK", NUKE_COUNTRY_COLOR.UK,
              "CHINA", NUKE_COUNTRY_COLOR.CHINA, "INDIA", NUKE_COUNTRY_COLOR.INDIA,
              "PAKIST", NUKE_COUNTRY_COLOR.PAKIST, "#94a3b8"],
            "icon-halo-color": "rgba(8,12,20,0.9)", "icon-halo-width": 1.1,
          },
        } as any);
        // tests detonated IN the selected year: bright pulse ring on top
        map.addLayer({
          id: "nuke-year", type: "circle", source: "nucleartests",
          filter: ["==", ["get", "y"], histYear],
          paint: {
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 2, 11, 8, 20] as any,
            "circle-color": "rgba(0,0,0,0)",
            "circle-stroke-color": "#fde047", "circle-stroke-width": 2,
          },
        } as any);
        detach = attachLayerInteractions(map, "nuke-pts", (e: any) => {
          const f = e.features?.[0]; if (!f) return; const t = f.properties;
          const CTRY: Record<string, string> = { USA: "United States", USSR: "Soviet Union", UK: "United Kingdom", FRANCE: "France", CHINA: "China", INDIA: "India", PAKIST: "Pakistan" };
          const rkm = blastRadiusKm(t.kt, t.t);
          setDetail({
            kind: "nuketest",
            title: t.n && t.n !== "NA" ? t.n : "Nuclear test",
            subtitle: `${CTRY[t.c] || t.c} · ${t.d}${t.kt ? ` · ${Number(t.kt).toLocaleString()} kt` : ""}`,
            body: `${t.r ? `Site: ${t.r}\n` : ""}` +
                  `Conducted by: ${testingAgency(t.c, t.y)}\n\n` +
                  `How it was fired: ${decodeType(t.t)}.\n` +
                  `Why (catalog purpose): ${decodePurpose(t.p)}.\n\n` +
                  `Yield: ${yieldContext(t.kt)}\n` +
                  `${rkm ? `Ring on map: ~${fmtKm(rkm, 1)} severe-blast (5 psi) radius ESTIMATE — Glasstone & Dolan cube-root scaling from the catalogued yield. An estimate of blast reach, not fallout: fallout depends on weather and burst height the catalog doesn't record.\n` :
                          `No ring on map: ${Number(t.kt) > 0 ? "buried shot — blast contained underground" : "yield not catalogued, so no radius is estimated"}.\n`}` +
                  `\nSource: the "Nuclear Explosions 1945–1998" catalog (Bergkvist & Ferm, Swedish Defence Research Establishment FOA / SIPRI) — the standard open historical record of all known tests: who, when, where, yield, emplacement and stated purpose. Locations and yields as catalogued (yields are the catalog's upper estimates).`,
          });
        });
        setStatus("nucleartests", "active", d.count,
          `${d.count.toLocaleString()} tests 1945–1998 (${d.quarantined} unlocated, quarantined) — symbols = how fired (air/surface/water/underground), color = country, ring = 5-psi blast estimate. Drag the year bar to travel`);
      },
      (failures) => setStatus("nucleartests", "error", undefined,
        failures === 0 ? "load failed — retrying automatically…" : "still retrying automatically…"),
    );
    return () => { stopLoad(); detach(); };
    // note: histYear is applied by the cheap setFilter effect below, not here —
    // the source/layers mount once, the filter moves per tick.
  }, [enabled.nucleartests, mapReady, mapSettled, setStatus]);

  // ── ambient radiation monitors (RAW; observed gamma dose-rate readings
  // from four national networks — BfS DE, Health Canada, STUK/FMI FI, EPA
  // RadNet US. Trefoil symbols tinted by dose DISPLAY band (bucket edges are
  // presentation, stated in the legend — not health thresholds); CPM-only
  // stations neutral-tinted, never converted to dose. US markers are
  // city-approximate and say so. No interpolation, no modeling. ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.radiation) {
      try {
        for (const l of ["radiation-pt", "radiation-area-fill", "radiation-area-line"]) if (map.getLayer(l)) map.removeLayer(l);
        for (const s of ["radiation", "radiation-areas"]) if (map.getSource(s)) map.removeSource(s);
      } catch {}
      setStatus("radiation", "off");
      return;
    }
    if (!mapSettled) { setStatus("radiation", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("radiation", "loading");
    let detach = () => {};
    const stopLoad = runResilientLoad(
      async (signal) => {
        const r = await fetch("/api/data/radiation", { signal });
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (signal.aborted) return;
        if (d.warming_up) throw new Error("radiation feed warming up");
        if (!Array.isArray(d.stations) || !d.stations.length) throw new Error("no stations");
        if (map.getSource("radiation")) return;
        // EXACT-location stations (DE/CA/FI) keep the trefoil pin; US RadNet
        // stations are city-approximate, so a pin would fake precision — they
        // render instead as a translucent circle spanning roughly the CITY's
        // land area (rkm = equivalent-circle radius from the Census gazetteer):
        // "the instrument is somewhere in this shaded area."
        const exact = d.stations.filter((s: any) => s.approx !== true);
        const approx = d.stations.filter((s: any) => s.approx === true);
        map.addSource("radiation", {
          type: "geojson",
          data: {
            type: "FeatureCollection",
            features: exact.map((s: any) => ({
              type: "Feature", geometry: { type: "Point", coordinates: [s.lon, s.lat] },
              properties: { ...s, band: radiationBandColor(s.value, s.unit) },
            })),
          } as any,
          attribution: "BfS · Health Canada · STUK/FMI · EPA RadNet",
        } as any);
        // geodesic 48-gon per approx station (true ground distance, same
        // approach as the nuclear blast rings; radii are 2-25 km so
        // equirectangular distortion is negligible)
        map.addSource("radiation-areas", {
          type: "geojson",
          data: {
            type: "FeatureCollection",
            features: approx.map((s: any) => {
              const rkm = Number(s.rkm) > 0 ? Number(s.rkm) : 6;
              const dLat = rkm / 111.32;
              const dLon = rkm / (111.32 * Math.max(0.15, Math.cos((s.lat * Math.PI) / 180)));
              const ring: [number, number][] = [];
              for (let i = 0; i <= 48; i++) {
                const a = (i * 2 * Math.PI) / 48;
                ring.push([s.lon + dLon * Math.cos(a), s.lat + dLat * Math.sin(a)]);
              }
              return {
                type: "Feature", geometry: { type: "Polygon", coordinates: [ring] },
                properties: { ...s, band: radiationBandColor(s.value, s.unit) },
              };
            }),
          } as any,
          attribution: "EPA RadNet (public domain)",
        } as any);
        map.addLayer({
          id: "radiation-area-fill", type: "fill", source: "radiation-areas",
          paint: { "fill-color": ["get", "band"], "fill-opacity": 0.16 },
        } as any);
        map.addLayer({
          id: "radiation-area-line", type: "line", source: "radiation-areas",
          paint: { "line-color": ["get", "band"], "line-opacity": 0.65, "line-width": 1.4, "line-dasharray": [3, 2] },
        } as any);
        map.addLayer({
          id: "radiation-pt", type: "symbol", source: "radiation",
          layout: {
            "icon-image": "vt-radiation",
            "icon-size": ["interpolate", ["linear"], ["zoom"], 2, 0.32, 8, 0.55, 12, 0.75],
            "icon-allow-overlap": false,
          },
          paint: {
            "icon-color": ["get", "band"],
            "icon-halo-color": "rgba(8,12,20,0.9)", "icon-halo-width": 1.1,
          },
        } as any);
        const NETWORK_LABEL: Record<string, string> = {
          "bfs-de": "Germany — BfS ODL network (DL-DE-BY-2.0)",
          "hc-ca": "Canada — Health Canada Fixed Point Surveillance (Open Government Licence – Canada)",
          "stuk-fi": "Finland — STUK via FMI open data (CC BY 4.0)",
          "radnet-us": "United States — EPA RadNet (public domain)",
        };
        // station reporting cadence, per each network's own documentation —
        // shown so "measured N min ago" reads as normal, not stale
        const NETWORK_CADENCE: Record<string, string> = {
          "bfs-de": "this network's stations report hourly",
          "hc-ca": "this network's stations report about every 15 minutes",
          "stuk-fi": "this network's stations report every 10 minutes",
          "radnet-us": "this network's monitors report several times a day",
        };
        const onRadiationClick = (e: any) => {
          const f = e.features?.[0]; if (!f) return; const s = f.properties;
          const v = Number(s.value);
          const val = s.unit === "uSv/h"
            ? `${v.toFixed(3)} µSv/h gamma dose rate`
            : `${v.toLocaleString()} counts/min gamma (this monitor publishes count rates, not dose — we never convert)`;
          // NOT live-streaming: each card states measured-at (relative + absolute),
          // the station's own cadence, and the platform's refresh cycle.
          let when = "Measurement time not published for this reading.";
          if (s.time) {
            const ageMin = Math.max(0, Math.round((Date.now() - Date.parse(s.time)) / 60000));
            const ago = ageMin < 60 ? `${ageMin} min ago`
              : ageMin < 2880 ? `${(ageMin / 60).toFixed(1)} h ago`
              : `${Math.round(ageMin / 1440)} days ago`;
            when = `Measured ${ago} (${s.time}) — not a live stream: ${NETWORK_CADENCE[s.network] || "stations report periodically"}, and this layer refreshes on a ~3-hour cycle.`;
          }
          // factual comparison against the typical natural background range —
          // banding of the measured value, no health interpretation
          let compare = "";
          if (s.unit === "uSv/h" && Number.isFinite(v)) {
            compare = v < 0.05
              ? `How this compares: BELOW the typical natural background range (0.05–0.3 µSv/h) — nothing unusual; low-geology sites and some instruments simply read low.`
              : v <= 0.3
              ? `How this compares: WITHIN the typical natural background range (0.05–0.3 µSv/h) — the normal gamma from soil minerals and cosmic rays.`
              : v <= 1.0
              ? `How this compares: ABOVE the typical background range (0.05–0.3 µSv/h). Elevated background is common over granite and at altitude; this is the measured value, not an alert.`
              : `How this compares: WELL ABOVE the typical background range (0.05–0.3 µSv/h) — a reading this high is rare; consult the source network's own site for context. We report the published value only.`;
          } else if (s.unit === "cpm") {
            compare = `How this compares: count rates (CPM) are instrument-specific and can't be compared to the µSv/h background range — compare this station against its own history on EPA's RadNet site.`;
          }
          setDetail({
            kind: "radiation",
            title: s.name,
            subtitle: val,
            body: `${s.approx === true || s.approx === "true" ? `LOCATION IS APPROXIMATE: the shaded circle spans roughly this CITY's land area — the EPA publishes RadNet locations as "City, ST" only, so the instrument is somewhere within the circle; its exact address is not public. The reading is real.\n\n` : ""}` +
                  `${when}\n\n` +
                  `${compare}\n\n` +
                  `Network: ${NETWORK_LABEL[s.network] || s.network}\n` +
                  `Observed reading from the network's own published feed — no interpolation, no modeling, no health claim. Marker color is a display bucket of the measured value (see legend), not a threshold.`,
          });
        };
        const d1 = attachLayerInteractions(map, "radiation-pt", onRadiationClick);
        const d2 = attachLayerInteractions(map, "radiation-area-fill", onRadiationClick);
        detach = () => { d1(); d2(); };
        const nets = d.networks || {};
        setStatus("radiation", "active", d.stations.length,
          `${d.stations.length.toLocaleString()} monitors — DE ${nets["bfs-de"] ?? 0} · CA ${nets["hc-ca"] ?? 0} · FI ${nets["stuk-fi"] ?? 0} · US ${nets["radnet-us"] ?? 0}. US monitors draw as translucent city-area circles (exact addresses not public); pins elsewhere are exact. Observed readings, no modeling`);
      },
      (failures) => setStatus("radiation", "error", undefined,
        failures === 0 ? "load failed — retrying automatically…" : "still retrying automatically…"),
    );
    return () => { stopLoad(); detach(); };
  }, [enabled.radiation, mapReady, mapSettled, setStatus]);

  // ── nuclear fuel-cycle & production facilities (RAW/FACTUAL; Wikidata CC0,
  // 67 curated sites: enrichment, reprocessing, waste repositories, test-site
  // facilities, weapons-complex — building+trefoil symbol, category-tinted.
  // Power plants/reactors excluded (the plant layers cover them). Static. ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.nukefacilities) {
      try {
        if (map.getLayer("nukefac-pt")) map.removeLayer("nukefac-pt");
        if (map.getSource("nukefacilities")) map.removeSource("nukefacilities");
      } catch {}
      setStatus("nukefacilities", "off");
      return;
    }
    if (!mapSettled) { setStatus("nukefacilities", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("nukefacilities", "loading");
    let detach = () => {};
    const stopLoad = runResilientLoad(
      async (signal) => {
        const r = await fetch("/api/data/nukefacilities", { signal });
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (signal.aborted || !Array.isArray(d.facilities)) throw new Error("no facilities");
        if (map.getSource("nukefacilities")) return;
        map.addSource("nukefacilities", {
          type: "geojson",
          data: {
            type: "FeatureCollection",
            features: d.facilities.map((f: any) => ({
              type: "Feature", geometry: { type: "Point", coordinates: [f.lon, f.lat] },
              properties: { ...f, tint: NUKE_FACILITY_COLOR[f.cat] || "#94a3b8" },
            })),
          } as any,
          attribution: "Wikidata (CC0 1.0)",
        } as any);
        map.addLayer({
          id: "nukefac-pt", type: "symbol", source: "nukefacilities",
          layout: {
            "icon-image": "vt-nukefacility",
            "icon-size": ["interpolate", ["linear"], ["zoom"], 2, 0.42, 8, 0.75],
            "icon-allow-overlap": true,
          },
          paint: {
            "icon-color": ["get", "tint"],
            "icon-halo-color": "rgba(8,12,20,0.9)", "icon-halo-width": 1.2,
          },
        } as any);
        detach = attachLayerInteractions(map, "nukefac-pt", (e: any) => {
          const f = e.features?.[0]; if (!f) return; const p = f.properties;
          setDetail({
            kind: "nukefacility",
            title: p.n || "Nuclear facility",
            subtitle: `${p.cat}${p.country ? ` · ${p.country}` : ""}`,
            body: `What this marker is: a nuclear fuel-cycle or production FACILITY as catalogued in Wikidata — ` +
                  `${p.cat === "Enrichment plant" ? "a plant that enriches uranium for fuel or weapons." :
                     p.cat === "Reprocessing site" ? "a site that chemically reprocesses spent nuclear fuel." :
                     p.cat === "Waste repository" ? "a radioactive-waste storage/disposal site." :
                     p.cat === "Test site" ? "a nuclear test SITE facility/region — individual detonations live in the Nuclear Tests layer." :
                     p.cat === "Weapons-complex site" ? "a nuclear-weapons design, production, or assembly site." :
                     "a nuclear facility whose Wikidata classing is generic."}\n` +
                  `\nDistinct from power plants (atom icons in the plant layers), radiation monitors (bare trefoil), ` +
                  `and accident sites (warning triangle).\n` +
                  `\nSource: Wikidata (CC0), curated 2026-07-12 — power-plant-classed items excluded, English label + ` +
                  `valid coordinates required; 4 US weapons-complex sites resolved individually (the class tree misses them). ` +
                  `Provenance: wikidata.org/wiki/${p.qid}. Locations as catalogued — no activity, output, or risk claims.`,
          });
        });
        setStatus("nukefacilities", "active", d.count,
          `${d.count} fuel-cycle & production facilities (Wikidata CC0, curated) — building-trefoil = facility, color = category; plants/reactors live in the power-plant layers`);
      },
      (failures) => setStatus("nukefacilities", "error", undefined,
        failures === 0 ? "load failed — retrying automatically…" : "still retrying automatically…"),
    );
    return () => { stopLoad(); detach(); };
  }, [enabled.nukefacilities, mapReady, mapSettled, setStatus]);

  // ── Military installations (RAW; STATIC REFERENCE GEOGRAPHY, human-specced
  // 2026-07-17). Officially published installation locations only — ~3,024
  // named OSM military=base sites (US bases included) + any cited government
  // publications; DoD authoritative overlay pending. Rendered polygons at
  // high zoom, centroid markers at low zoom (declutter). Colour by operator
  // nation, ONE MUTED reference palette — deliberately NO red / threat
  // styling: this is reference geography, not a threat board. DEFAULT OFF
  // (heavy). NO CROSS-TIES: this layer is NEVER joined to the live aircraft/
  // vessel layers or any correlation feature — the popup carries name/nation/
  // branch/type/source only, no timeline block (human's standing rule). ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    const clear = () => {
      try {
        for (const l of ["mil-inst-fill", "mil-inst-outline", "mil-inst-pt"]) if (map.getLayer(l)) map.removeLayer(l);
        for (const s of ["mil-inst-poly", "mil-inst-centroid"]) if (map.getSource(s)) map.removeSource(s);
      } catch {}
    };
    if (!enabled.military_installations) { clear(); setStatus("military_installations", "off"); return; }
    if (!mapSettled) { setStatus("military_installations", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("military_installations", "loading");
    let detach = () => {};
    const stopLoad = runResilientLoad(
      async (signal) => {
        const r = await fetch("/api/data/military_installations", { signal });
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (signal.aborted || !Array.isArray(d.installations)) throw new Error("no installations");
        if (map.getSource("mil-inst-poly") || map.getSource("mil-inst-centroid")) return;
        const withTint = (f: any) => ({ ...f, tint: militaryNationTint(f.operator_nation) });
        // polygons — high-zoom boundaries
        const polys = d.installations.filter((f: any) => f.geometry?.type === "Polygon" || f.geometry?.type === "MultiPolygon");
        map.addSource("mil-inst-poly", {
          type: "geojson",
          data: { type: "FeatureCollection", features: polys.map((f: any) => ({
            type: "Feature", geometry: f.geometry, properties: withTint(f),
          })) } as any,
          attribution: "© OpenStreetMap contributors",
        } as any);
        // centroids — low-zoom markers for EVERY installation (declutter)
        map.addSource("mil-inst-centroid", {
          type: "geojson",
          data: { type: "FeatureCollection", features: d.installations
            .filter((f: any) => Array.isArray(f.centroid))
            .map((f: any) => ({ type: "Feature", geometry: { type: "Point", coordinates: f.centroid }, properties: withTint(f) })) } as any,
          attribution: "© OpenStreetMap contributors",
        } as any);
        map.addLayer({
          id: "mil-inst-fill", type: "fill", source: "mil-inst-poly", minzoom: 7,
          paint: { "fill-color": ["get", "tint"], "fill-opacity": 0.22 },
        } as any);
        map.addLayer({
          id: "mil-inst-outline", type: "line", source: "mil-inst-poly", minzoom: 7,
          paint: { "line-color": ["get", "tint"], "line-width": 1.1, "line-opacity": 0.7 },
        } as any);
        map.addLayer({
          // SYMBOL not dot (human 2026-07-17: "can we get a symbol instead of
          // the dot — i like the outline and shade box, don't change that"):
          // shield/crossed-swords SDF, nation tint. Polygon fill+outline
          // layers above are untouched.
          id: "mil-inst-pt", type: "symbol", source: "mil-inst-centroid",
          layout: {
            "icon-image": "vt-military",
            "icon-size": ["interpolate", ["linear"], ["zoom"], 2, 0.34, 6, 0.5, 10, 0.62],
            "icon-allow-overlap": false,       // declutter at low zoom (collision cull)
          },
          paint: {
            "icon-color": ["get", "tint"],
            "icon-halo-color": "rgba(8,12,20,0.9)", "icon-halo-width": 1.1,
            // fade the low-zoom symbols out as the polygons take over at high zoom
            "icon-opacity": ["interpolate", ["linear"], ["zoom"], 7, 0.9, 9, 0.35],
          },
        } as any);
        const onClick = (e: any) => {
          const f = e.features?.[0]; if (!f) return; const p = f.properties;
          setDetail({
            kind: "military_installation",
            title: p.name || "Military installation",
            subtitle: `${p.operator_nation || "nation unattributed"}${p.type && p.type !== "other" ? ` · ${p.type}` : ""}`,
            // NO timeline / cross-tie block — this layer is static reference
            // geography and is NEVER correlated with live tracking (human rule).
            body: `Military installation (reference geography).\n` +
                  `Nation: ${p.operator_nation || "unattributed (offshore / unresolved at source resolution)"}\n` +
                  `Branch/service: ${p.branch || "not specified in source"}\n` +
                  `Type: ${p.type}\n` +
                  `Status: ${p.status}\n\n` +
                  `Officially published installation location — reference geography only, current as of ${p.source_retrieved_date}; ` +
                  `not operational information. Source: ${p.source}.`,
            sourceUrl: p.source_url,
          } as any);
        };
        const d1 = attachLayerInteractions(map, "mil-inst-pt", onClick);
        const d2 = attachLayerInteractions(map, "mil-inst-fill", onClick);
        detach = () => { d1(); d2(); };
        setStatus("military_installations", "active", d.count,
          `${d.count} named installations — © OpenStreetMap contributors + US DoD open data + cited government publications · colour = operator nation (reference palette, not a threat board) · reference geography, not operational information`);
      },
      (failures) => setStatus("military_installations", "error", undefined,
        failures === 0 ? "load failed — retrying automatically…" : "still retrying automatically…"),
    );
    return () => { stopLoad(); detach(); };
  }, [enabled.military_installations, mapReady, mapSettled, setStatus]);

  // ── nuclear accidents & radiological incidents (RAW/FACTUAL; Wikidata CC0,
  // 46 curated events through a quality gate. Hazard-triangle-trefoil symbols
  // tinted by official INES level where catalogued (never inferred; unrated =
  // gray). A MELTDOWN reads differently from a TEST at a glance — the symbol
  // directive's canonical case. Static history, mounts once. ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.nukeaccidents) {
      try {
        if (map.getLayer("nukeacc-pt")) map.removeLayer("nukeacc-pt");
        if (map.getSource("nukeaccidents")) map.removeSource("nukeaccidents");
      } catch {}
      setStatus("nukeaccidents", "off");
      return;
    }
    if (!mapSettled) { setStatus("nukeaccidents", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("nukeaccidents", "loading");
    let detach = () => {};
    const stopLoad = runResilientLoad(
      async (signal) => {
        const r = await fetch("/api/data/nukeaccidents", { signal });
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (signal.aborted || !Array.isArray(d.events)) throw new Error("no events");
        if (map.getSource("nukeaccidents")) return;
        map.addSource("nukeaccidents", {
          type: "geojson",
          data: {
            type: "FeatureCollection",
            features: d.events.map((ev: any) => ({
              type: "Feature", geometry: { type: "Point", coordinates: [ev.lon, ev.lat] },
              properties: { ...ev, tint: inesColor(ev.ines) },
            })),
          } as any,
          attribution: "Wikidata (CC0 1.0)",
        } as any);
        map.addLayer({
          id: "nukeacc-pt", type: "symbol", source: "nukeaccidents",
          layout: {
            "icon-image": "vt-meltdown",
            // severity reads in size too: INES 6-7 render larger
            "icon-size": ["interpolate", ["linear"], ["zoom"],
              2, ["case", [">=", ["coalesce", ["get", "ines"], 0], 6], 0.55, 0.4],
              8, ["case", [">=", ["coalesce", ["get", "ines"], 0], 6], 0.95, 0.7]],
            "icon-allow-overlap": true,
          },
          paint: {
            "icon-color": ["get", "tint"],
            "icon-halo-color": "rgba(8,12,20,0.9)", "icon-halo-width": 1.2,
          },
        } as any);
        detach = attachLayerInteractions(map, "nukeacc-pt", (e: any) => {
          const f = e.features?.[0]; if (!f) return; const ev = f.properties;
          const ines = Number(ev.ines);
          const inesLine = Number.isFinite(ines) && ines >= 1
            ? `INES level ${ines} of 7 — the IAEA's official severity scale (1–3 incidents, 4–7 accidents; 7 = major accident like Chernobyl/Fukushima).`
            : "No official INES level is catalogued for this event — shown unrated, never guessed.";
          setDetail({
            kind: "nukeaccident",
            title: ev.n || "Nuclear event",
            subtitle: `${ev.d || "date not catalogued"}${ev.loc ? ` · ${ev.loc}` : ""}`,
            body: `${inesLine}\n` +
                  `\nWhat this marker is: the catalogued site of a nuclear or radiological accident/incident — ` +
                  `reactor accidents, criticality events, contamination releases and lost-source incidents. ` +
                  `Distinct from the nuclear TESTS layer (deliberate detonations, mushroom-cloud symbols).\n` +
                  `\nSource: Wikidata (CC0), curated 2026-07-12 through a quality gate (English label + date-or-INES ` +
                  `evidence required, duplicates merged, 3 majors resolved individually). Provenance: wikidata.org/wiki/${ev.qid}. ` +
                  `Facts as catalogued — no radiation, damage, or risk modeling.`,
          });
        });
        setStatus("nukeaccidents", "active", d.count,
          `${d.count} accidents & incidents 1949–2024 (Wikidata CC0, curated) — triangle-trefoil = nuclear event, color = official INES level, gray = unrated`);
      },
      (failures) => setStatus("nukeaccidents", "error", undefined,
        failures === 0 ? "load failed — retrying automatically…" : "still retrying automatically…"),
    );
    return () => { stopLoad(); detach(); };
  }, [enabled.nukeaccidents, mapReady, mapSettled, setStatus]);

  // ── earthquake history (RAW; USGS ComCat M6+ since 1900, public domain —
  // 14,492 events compiled through the data-quality gate. Shares the history
  // time bar: accumulate to the year, the year's quakes pulse. The LIVE quakes
  // layer covers the present separately.) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.quakehistory) {
      try {
        for (const l of ["qh-pts", "qh-year"]) if (map.getLayer(l)) map.removeLayer(l);
        if (map.getSource("quakehistory")) map.removeSource("quakehistory");
      } catch {}
      setStatus("quakehistory", "off");
      return;
    }
    if (!mapSettled) { setStatus("quakehistory", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("quakehistory", "loading");
    let detach = () => {};
    const stopLoad = runResilientLoad(
      async (signal) => {
        const r = await fetch("/api/data/quakehistory", { signal });
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (signal.aborted || !Array.isArray(d.quakes)) throw new Error("no quakes in response");
        if (map.getSource("quakehistory")) return;
        map.addSource("quakehistory", {
          type: "geojson",
          data: {
            type: "FeatureCollection",
            features: d.quakes.map((q: any) => ({
              type: "Feature", geometry: { type: "Point", coordinates: [q.lon, q.lat] },
              properties: q,
            })),
          } as any,
          attribution: "USGS ANSS ComCat (public domain)",
        } as any);
        const mag = ["coalesce", ["get", "m"], 6] as any;
        const radius = ["interpolate", ["linear"], ["zoom"],
          2, ["-", mag, 4.6], 8, ["*", 2.2, ["-", mag, 4]]] as any;
        map.addLayer({
          id: "qh-pts", type: "circle", source: "quakehistory",
          filter: ["<=", ["get", "y"], histYear],
          paint: {
            "circle-radius": radius,
            "circle-color": ["interpolate", ["linear"], mag,
              6, "#facc15", 7, "#fb923c", 8, "#ef4444", 9, "#b91c1c"],
            "circle-opacity": 0.4,
            "circle-stroke-color": "rgba(8,12,20,0.8)", "circle-stroke-width": 0.4,
          },
        } as any);
        map.addLayer({
          id: "qh-year", type: "circle", source: "quakehistory",
          filter: ["==", ["get", "y"], histYear],
          paint: {
            "circle-radius": ["+", 3, radius] as any,
            "circle-color": "rgba(0,0,0,0)",
            "circle-stroke-color": "#f87171", "circle-stroke-width": 1.8,
          },
        } as any);
        detach = attachLayerInteractions(map, "qh-pts", (e: any) => {
          const f = e.features?.[0]; if (!f) return; const q = f.properties;
          setDetail({
            kind: "quake",
            title: q.pl || "Earthquake",
            subtitle: `M${q.m} · ${q.d}`,
            body: `${q.dep != null ? `Depth: ${fmtKm(q.dep)}\n` : ""}` +
                  `\nUSGS ANSS ComCat (public domain) — historical catalog M6+ since 1900; the live quakes layer covers the present.`,
          });
        });
        setStatus("quakehistory", "active", d.count,
          `${d.count.toLocaleString()} quakes M6+ ${d.min_year}–${d.max_year} — drag the year bar to travel`);
      },
      (failures) => setStatus("quakehistory", "error", undefined,
        failures === 0 ? "load failed — retrying automatically…" : "still retrying automatically…"),
    );
    return () => { stopLoad(); detach(); };
  }, [enabled.quakehistory, mapReady, mapSettled, setStatus]);

  // ── CWA water violators (RAW/FACTUAL; EPA ECHO, public domain — active
  // facilities >8 of last 12 quarters in Clean Water Act noncompliance. The
  // "factories' water quality" layer: EPA's own compliance records, colored by
  // violation kind. Current snapshot (not time-scrubbed). ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.waterviolators) {
      try {
        if (map.getLayer("wv-pts")) map.removeLayer("wv-pts");
        if (map.getSource("waterviolators")) map.removeSource("waterviolators");
      } catch {}
      setStatus("waterviolators", "off");
      return;
    }
    if (!mapSettled) { setStatus("waterviolators", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("waterviolators", "loading");
    let detach = () => {};
    const stopLoad = runResilientLoad(
      async (signal) => {
        const r = await fetch("/api/data/waterviolators", { signal });
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (signal.aborted || !Array.isArray(d.violators)) throw new Error("no violators in response");
        if (d.warming_up) { setStatus("waterviolators", "loading", undefined, "warming up — EPA ECHO fetch in progress"); return; }
        if (map.getSource("waterviolators")) return;
        map.addSource("waterviolators", {
          type: "geojson",
          data: {
            type: "FeatureCollection",
            features: d.violators.map((v: any) => ({
              type: "Feature", geometry: { type: "Point", coordinates: [v.lon, v.lat] },
              properties: v,
            })),
          } as any,
          attribution: "U.S. EPA ECHO / NPDES (public domain)",
        } as any);
        map.addLayer({
          id: "wv-pts", type: "circle", source: "waterviolators", minzoom: 4,
          paint: {
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, 1.6, 9, 4.5, 13, 7],
            // effluent (actual discharge) violations red; reporting failures
            // amber; schedule/other violet; none-current slate
            "circle-color": ["case",
              ["in", "Effluent", ["coalesce", ["get", "snc"], ""]], "#ef4444",
              ["in", "Report", ["coalesce", ["get", "snc"], ""]], "#f59e0b",
              ["in", "Schedule", ["coalesce", ["get", "snc"], ""]], "#a78bfa",
              "#64748b"],
            "circle-opacity": 0.75,
            "circle-stroke-color": "rgba(8,12,20,0.85)", "circle-stroke-width": 0.5,
          },
        } as any);
        detach = attachLayerInteractions(map, "wv-pts", (e: any) => {
          const f = e.features?.[0]; if (!f) return; const v = f.properties;
          setDetail({
            kind: "waterviolator",
            title: v.name || "Facility",
            subtitle: `${v.qtrs}/12 quarters in CWA noncompliance${v.snc ? ` · ${v.snc}` : ""}`,
            body: `${[v.city, v.state].filter(Boolean).join(", ")}\n` +
                  `${v.permit ? `Permit: ${v.permit} (${v.id})\n` : `NPDES ID: ${v.id}\n`}` +
                  `${v.actions ? `Formal enforcement actions: ${v.actions}\n` : ""}` +
                  `\nEPA ECHO Clean Water Act compliance record (public domain) — facts about permits and violations as EPA publishes them; not a water-safety claim about any location.`,
          });
        });
        const h = d.health;
        setStatus("waterviolators", "active", d.violators.length,
          `EPA ECHO — ${d.violators.length.toLocaleString()} facilities >8/12 quarters in CWA noncompliance${h?.suspect ? ` (${h.suspect} quarantined)` : ""}`);
      },
      (failures) => setStatus("waterviolators", "error", undefined,
        failures === 0 ? "load failed — retrying automatically…" : "still retrying automatically…"),
    );
    return () => { stopLoad(); detach(); };
  }, [enabled.waterviolators, mapReady, mapSettled, setStatus]);

  // year scrub -> GPU filter update on every history layer (cheap; no source churn)
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    try {
      if (map.getLayer("nuke-pts")) map.setFilter("nuke-pts", ["<=", ["get", "y"], histYear]);
      if (map.getLayer("nuke-year")) map.setFilter("nuke-year", ["==", ["get", "y"], histYear]);
      if (map.getLayer("nuke-ring-fill")) map.setFilter("nuke-ring-fill", ["<=", ["get", "y"], histYear]);
      if (map.getLayer("nuke-ring-year")) map.setFilter("nuke-ring-year", ["==", ["get", "y"], histYear]);
      if (map.getLayer("qh-pts")) map.setFilter("qh-pts", ["<=", ["get", "y"], histYear]);
      if (map.getLayer("qh-year")) map.setFilter("qh-year", ["==", ["get", "y"], histYear]);
    } catch {}
  }, [histYear, enabled.nucleartests, enabled.quakehistory, mapReady]);

  // play: advance one year every 500ms, loop at the end
  useEffect(() => {
    if (!histPlay || (!enabled.nucleartests && !enabled.quakehistory)) return;
    const iv = window.setInterval(() => {
      setHistYear((y) => (y >= HIST_MAX_YEAR ? HIST_MIN_YEAR : y + 1));
    }, 500);
    return () => window.clearInterval(iv);
  }, [histPlay, enabled.nucleartests, enabled.quakehistory]);

  // ── live trains (RAW; Finland Digitraffic CC BY 4.0 + Norway Entur NLOD;
  // per-source status from the server keeps coverage labeling honest) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.trains) {
      try {
        if (map.getLayer("trains-icons")) map.removeLayer("trains-icons");
        if (map.getSource("trains")) map.removeSource("trains");
      } catch {}
      setStatus("trains", "off");
      return;
    }
    if (!mapSettled) { setStatus("trains", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("trains", "loading");
    let stop = false;
    let detach = () => {};
    const load = async () => {
      try {
        const r = await fetch("/api/data/trains");
        const d = await r.json();
        if (stop || !d.trains) return;
        const fc = {
          type: "FeatureCollection",
          features: d.trains.map((t: any) => ({
            type: "Feature",
            geometry: { type: "Point", coordinates: [t.lon, t.lat] },
            properties: {
              id: t.id, country: t.country, label: t.label || t.id,
              speed: t.speed_kmh,
              // numeric always: Entur publishes bearing, Digitraffic doesn't
              // (those render upright at 0)
              bearing: t.bearing ?? 0,
            },
          })),
        };
        const src: any = map.getSource("trains");
        if (src) src.setData(fc);
        else {
          map.addSource("trains", { type: "geojson", data: fc as any });
          map.addLayer({
            id: "trains-icons", type: "symbol", source: "trains",
            layout: {
              "icon-image": "vt-train",
              "icon-size": ["interpolate", ["linear"], ["zoom"], 3, 0.4, 8, 0.7],
              "icon-rotate": ["get", "bearing"],
              "icon-rotation-alignment": "map",
              "icon-allow-overlap": true,
              "icon-ignore-placement": true,
            },
            paint: {
              "icon-color": "#2dd4bf",
              "icon-halo-color": "rgba(5,10,19,0.95)",
              "icon-halo-width": 1.3,
            },
          });
          detach = attachLayerInteractions(map, "trains-icons", async (e: any) => {
            const f = e.features?.[0];
            if (!f) return;
            const p = f.properties;
            const dossierKey = `train:${p.id}:${Date.now()}`;
            setDetail({
              kind: "train",
              title: `${p.label}`,
              subtitle: `${p.country === "FI" ? "Finland · Digitraffic (CC BY 4.0)" : "Norway · Entur (NLOD)"}`,
              body: `${p.speed != null && p.speed !== "null" ? `Speed: ${fmtKmh(Number(p.speed))}\n` : ""}Live passenger-rail position, shown as received.`,
              trailId: p.id, trailKind: "trains", dossierKey,
            });
            // Trains aren't Everything Graph nodes — lat/lon-only dossier.
            fetchDossier(dossierKey, null, e.lngLat?.lat, e.lngLat?.lng);
            const { note, lastT } = await showTrail("trains", p.id);
            setDetail(prev => prev && prev.trailId === p.id ? { ...prev, trailNote: note, trailLastT: lastT } : prev);
          });
        }
        const per = (d.sources || []).map((s: any) => `${s.country} ${s.status === "ok" ? s.count : s.status}`).join(" · ");
        setStatus("trains", "active", d.count, per || undefined);
      } catch {
        if (!stop) setStatus("trains", "error");
      }
    };
    load();
    // hidden-tab gate ([REPAIR 2026-07-05] map perf) — refresh resumes on
    // the next tick after the tab returns; server cache covers the gap
    const iv = window.setInterval(() => { if (!document.hidden) load(); }, 30_000);
    return () => { stop = true; window.clearInterval(iv); detach(); };
  }, [enabled.trains, mapReady, mapSettled, setStatus]);

  // ── active fires (RAW; NASA FIRMS/LANCE VIIRS 375m NRT — Tier-1(c)
  // geospatial root, key-gated exactly like vessels. No free history exists
  // upstream, so the server archives every fetch from day one. NOT FOR
  // SAFETY-OF-LIFE USE per the LANCE data-use disclaimer, restated in every
  // detection's detail card, not just the layer description.) ──
  useEffect(() => {
    const map = mapRef.current;
    const meta = layers.find(l => l.id === "fires");
    if (meta && meta.status === "awaiting_key") setStatus("fires", "awaiting_key");
    if (!enabled.fires) {
      try {
        if (map?.getLayer("fires-sym")) map.removeLayer("fires-sym");
        if (map?.getSource("fires")) map.removeSource("fires");
      } catch {}
      if (!meta || meta.status !== "awaiting_key") setStatus("fires", "off");
      return;
    }
    if (!map || !mapReady) return;
    setStatus("fires", "loading");
    let stop = false;
    let detach = () => {};
    const load = async () => {
      try {
        // SCALE S1: bound the served payload to the current viewport (the
        // NASA FIRMS feed is global — during an active fire season the
        // uncapped world set can run into the tens of thousands, so a
        // viewport-first cut is both cheaper AND more useful than an
        // arbitrary head-of-array slice; server/nasaFirms.ts still caps as a
        // last-resort safety net and states it honestly, never inside `count`).
        const b = map.getBounds();
        const bbox = `${b.getWest().toFixed(2)},${b.getSouth().toFixed(2)},${b.getEast().toFixed(2)},${b.getNorth().toFixed(2)}`;
        const r = await fetch(`/api/data/fires?bbox=${bbox}`);
        const d = await r.json();
        if (stop) return;
        if (d.enabled === false) { setStatus("fires", "awaiting_key"); return; }
        if (d.warming_up) { setStatus("fires", "loading", 0, "warming up — first poll can take a few minutes"); return; }
        const fc = {
          type: "FeatureCollection",
          features: (d.fires || []).map((f: any) => ({
            type: "Feature",
            geometry: { type: "Point", coordinates: [f.lon, f.lat] },
            properties: {
              color: FIRE_CONFIDENCE_COLOR[f.confidence] || FIRE_CONFIDENCE_COLOR.nominal,
              confidence: f.confidence, brightness: f.brightness, frp: f.frp,
              acq_date: f.acq_date, acq_time: f.acq_time, satellite: f.satellite,
              daynight: f.daynight || "",
            },
          })),
        };
        const src: any = map.getSource("fires");
        if (src) {
          src.setData(fc as any);
        } else {
          map.addSource("fires", { type: "geojson", data: fc as any });
          map.addLayer({
            id: "fires-sym", type: "symbol", source: "fires",
            layout: {
              "icon-image": "vt-fire",
              "icon-size": ["interpolate", ["linear"], ["zoom"], 2, 0.28, 6, 0.5],
              "icon-allow-overlap": true,
              "icon-ignore-placement": true,
            },
            paint: { "icon-color": ["get", "color"], "icon-opacity": 0.9 },
          });
          detach = attachLayerInteractions(map, "fires-sym", (e: any) => {
            const f = e.features?.[0];
            if (!f) return;
            const p = f.properties;
            const dossierKey = `fire:${e.lngLat?.lat},${e.lngLat?.lng}:${Date.now()}`;
            setDetail({
              kind: "fire",
              title: "Active fire detection",
              subtitle: `${p.confidence} confidence · ${p.satellite}${p.daynight ? ` · ${p.daynight === "D" ? "day" : "night"}` : ""}`,
              body: `Detected ${p.acq_date} ${String(p.acq_time).padStart(4, "0")} UTC` +
                    `${p.brightness != null ? `\nBrightness: ${Math.round(p.brightness)} K` : ""}` +
                    `${p.frp != null ? `\nFire radiative power: ${p.frp} MW` : ""}\n\n` +
                    `NASA FIRMS/LANCE — for informational purposes only, NOT for safety-of-life use.`,
              links: [{ label: "NASA FIRMS map", href: "https://firms.modaps.eosdis.nasa.gov/map/" }],
              dossierKey,
            });
            // Fires aren't Everything Graph nodes — lat/lon-only dossier
            // (nearest strategic sites is the useful part here).
            fetchDossier(dossierKey, null, e.lngLat?.lat, e.lngLat?.lng);
          });
        }
        const baseNote = "NASA FIRMS/LANCE · VIIRS 375m · ~3h latency · not for safety-of-life use";
        const note = d.capped
          ? `showing ${d.count.toLocaleString()} of ${d.count_before_cap.toLocaleString()} in view (capped) · ${baseNote}`
          : baseNote;
        setStatus("fires", "active", d.count ?? (d.fires || []).length, note);
      } catch {
        if (!stop) setStatus("fires", "error");
      }
    };
    load();
    const iv = window.setInterval(load, 15 * 60_000);
    return () => { stop = true; window.clearInterval(iv); detach(); };
  }, [enabled.fires, mapReady, layers, setStatus]);

  // ── USGS river gauges (RAW; stream #6 surface — 14 barge-corridor
  // gauges, raw stage/discharge as published; the low-water SIGNAL stays
  // gate-2-locked). Off by default (reference layer; initial-load budget). ──
  useEffect(() => {
    const map = mapRef.current;
    if (!enabled.rivergauges) {
      try {
        if (map?.getLayer("rivergauges-sym")) map.removeLayer("rivergauges-sym");
        if (map?.getSource("rivergauges")) map.removeSource("rivergauges");
      } catch {}
      setStatus("rivergauges", "off");
      return;
    }
    if (!map || !mapReady) return;
    setStatus("rivergauges", "loading");
    let stop = false;
    let detach = () => {};
    const load = async () => {
      try {
        const r = await fetch("/api/data/rivergauges");
        const d = await r.json();
        if (stop) return;
        if (d.warming_up) { setStatus("rivergauges", "loading", 0, "warming up — first poll can take a minute"); return; }
        // Pillar-6 cross-tie: fetch the generating capacity near each gauge once
        // per load and index it by site, so the gauge-click detail can name the
        // exposed plants (RAW proximity join — see /api/data/plants-near-rivergauges).
        fetch("/api/data/plants-near-rivergauges")
          .then((pr) => pr.json())
          .then((pd) => {
            if (stop || !pd || !Array.isArray(pd.gauges)) return;
            const bySite: Record<string, any> = {};
            for (const g of pd.gauges) bySite[g.site] = g;
            riverPlantsRef.current = bySite;
          })
          .catch(() => {});
        const fc = {
          type: "FeatureCollection",
          features: (d.gauges || []).filter((g: any) => g.lat != null && g.lon != null).map((g: any) => ({
            type: "Feature",
            geometry: { type: "Point", coordinates: [g.lon, g.lat] },
            properties: {
              site: g.site, name: g.name, param: g.param, v: g.v, q: g.q, d: g.d,
            },
          })),
        };
        const src: any = map.getSource("rivergauges");
        if (src) {
          src.setData(fc as any);
        } else {
          map.addSource("rivergauges", { type: "geojson", data: fc as any });
          map.addLayer({
            id: "rivergauges-sym", type: "symbol", source: "rivergauges",
            layout: {
              "icon-image": "vt-gauge",
              "icon-size": ["interpolate", ["linear"], ["zoom"], 3, 0.5, 8, 0.8],
              "icon-allow-overlap": true,
              "icon-ignore-placement": true,
            },
            paint: {
              "icon-color": "#4d9fff",
              "icon-halo-color": "rgba(5,10,19,0.95)",
              "icon-halo-width": 1.3,
            },
          });
          detach = attachLayerInteractions(map, "rivergauges-sym", (e: any) => {
            const f = e.features?.[0];
            if (!f) return;
            const p = f.properties;
            const unit = p.param === "00060" ? "ft³/s (discharge)" : "ft (gage height)";
            // Pillar-6 cross-tie surface: generating capacity within 50 km of this
            // gauge (RAW proximity join). No predictive claim — this is the exposed
            // capacity a low-water reading would name, not a forecast that it will.
            const xt = riverPlantsRef.current[p.site];
            let exposure = "";
            if (xt && xt.plant_count > 0) {
              const fuels = Object.entries(xt.capacity_by_fuel || {})
                .sort((a: any, b: any) => b[1] - a[1])
                .slice(0, 4)
                .map(([fuel, mw]: any) => `${fuel} ${Math.round(mw).toLocaleString()} MW`)
                .join(", ");
              const top = (xt.plants || []).slice(0, 3)
                .map((pl: any) => `${pl.name} (${pl.capacity_mw.toLocaleString()} MW ${pl.fuel}, ${fmtKm(pl.distance_km)})`)
                .join("\n  ");
              exposure =
                `\n\nGenerating capacity within ${fmtKm(50)}: ${xt.plant_count} plants, ` +
                `${xt.total_capacity_mw.toLocaleString()} MW\n  ${fuels}` +
                `${top ? `\nNearest:\n  ${top}` : ""}` +
                `\nRAW proximity join (plants near this reach), NOT confirmed water-intake ` +
                `dependence — low-water → operator-risk is validation-gated.`;
            }
            const dossierKey = `gauge:${p.site}:${Date.now()}`;
            setDetail({
              kind: "gauge",
              title: `≈ ${p.name || p.site}`,
              subtitle: `USGS ${p.site}`,
              body: `${p.v} ${unit}\nObserved ${p.d}\n` +
                    `${p.q === "P" ? "PROVISIONAL — subject to revision" : p.q === "A" ? "Approved reading" : ""}\n\n` +
                    `Data courtesy U.S. Geological Survey. Raw readings only — ` +
                    `low-water interpretation is validation-gated.` + exposure,
              links: [{ label: "USGS monitoring page", href: `https://waterdata.usgs.gov/monitoring-location/${p.site}/` }],
              dossierKey,
            });
            // Gauges aren't Everything Graph nodes — lat/lon-only dossier
            // (the plants-near-rivergauges exposure above already covers the
            // power-plant cross-tie; nearest_sites here is a broader net).
            fetchDossier(dossierKey, null, e.lngLat?.lat, e.lngLat?.lng);
          });
        }
        setStatus("rivergauges", "active", fc.features.length, "USGS NWIS · provisional readings revise");
      } catch {
        if (!stop) setStatus("rivergauges", "error");
      }
    };
    load();
    // hourly refresh, hidden-tab gated (matches the server's 1h poll)
    const iv = window.setInterval(() => { if (!document.hidden) load(); }, 60 * 60_000);
    return () => { stop = true; window.clearInterval(iv); detach(); };
  }, [enabled.rivergauges, mapReady, setStatus]);

  // ── NWS severe-weather alerts (RAW overlay — official warnings as-is,
  // colored by CAP severity; zone-only alerts are counted, not drawn). ──
  useEffect(() => {
    const map = mapRef.current;
    if (!enabled.alerts) {
      try {
        if (map?.getLayer("alerts-line")) map.removeLayer("alerts-line");
        if (map?.getLayer("alerts-fill")) map.removeLayer("alerts-fill");
        if (map?.getSource("alerts")) map.removeSource("alerts");
      } catch {}
      setStatus("alerts", "off");
      return;
    }
    if (!map || !mapReady) return;
    setStatus("alerts", "loading");
    let stop = false;
    let detach = () => {};
    const SEV_COLOR: any = ["match", ["get", "severity"],
      "Extreme", "#ff3b3b", "Severe", "#ff8c42", "Moderate", "#ffd23f", "Minor", "#4d9fff",
      "#9aa4b2"];
    const load = async () => {
      try {
        const r = await fetch("/api/data/alerts");
        const d = await r.json();
        if (stop) return;
        if (d.warming_up) { setStatus("alerts", "loading", 0, "warming up — first poll can take a minute"); return; }
        const fc = {
          type: "FeatureCollection",
          features: (d.alerts || []).map((a: any) => ({
            type: "Feature",
            geometry: { type: "Polygon", coordinates: a.rings },
            properties: { id: a.id, event: a.event, severity: a.severity || "Unknown", area: a.area, ends: a.ends },
          })),
        };
        const src: any = map.getSource("alerts");
        if (src) {
          src.setData(fc as any);
        } else {
          map.addSource("alerts", { type: "geojson", data: fc as any });
          map.addLayer({
            id: "alerts-fill", type: "fill", source: "alerts",
            paint: { "fill-color": SEV_COLOR, "fill-opacity": 0.18 },
          });
          map.addLayer({
            id: "alerts-line", type: "line", source: "alerts",
            paint: { "line-color": SEV_COLOR, "line-width": 1.2, "line-opacity": 0.8 },
          });
          detach = attachLayerInteractions(map, "alerts-fill", (e: any) => {
            const f = e.features?.[0];
            if (!f) return;
            const p = f.properties;
            const dossierKey = `alert:${p.id}:${Date.now()}`;
            setDetail({
              kind: "alert",
              title: `⚠ ${p.event}`,
              subtitle: `NWS · ${p.severity} severity`,
              body: `${p.area || ""}\n${p.ends ? `Ends ${p.ends}` : ""}\n\n` +
                    `Official National Weather Service alert, displayed as published. ` +
                    `Not for safety-of-life use — see weather.gov for authoritative guidance.`,
              links: [{ label: "weather.gov alerts", href: "https://www.weather.gov/alerts" }],
              dossierKey,
            });
            // Alerts aren't Everything Graph nodes — lat/lon-only dossier,
            // anchored on the clicked point (alert polygons can span states;
            // "nearest sites" is honest for the click point, not the whole
            // polygon's centroid).
            fetchDossier(dossierKey, null, e.lngLat?.lat, e.lngLat?.lng);
          });
        }
        const note = d.zone_only ? `NWS · ${d.zone_only} zone-coded alerts not drawn` : "NWS api.weather.gov";
        setStatus("alerts", "active", fc.features.length, note);
      } catch {
        if (!stop) setStatus("alerts", "error");
      }
    };
    load();
    // 5-min refresh, hidden-tab gated (server polls upstream every 10)
    const iv = window.setInterval(() => { if (!document.hidden) load(); }, 5 * 60_000);
    return () => { stop = true; window.clearInterval(iv); detach(); };
  }, [enabled.alerts, mapReady, setStatus]);

  // ── USGS earthquakes (RAW; M2.5+, global, rolling 24h — sized/colored by
  // magnitude, no predictive claim). Off by default (reference layer;
  // initial-load budget, same precedent as rivergauges/alerts). ──
  useEffect(() => {
    const map = mapRef.current;
    if (!enabled.earthquakes) {
      try {
        if (map?.getLayer("earthquakes-sym")) map.removeLayer("earthquakes-sym");
        if (map?.getSource("earthquakes")) map.removeSource("earthquakes");
      } catch {}
      setStatus("earthquakes", "off");
      return;
    }
    if (!map || !mapReady) return;
    setStatus("earthquakes", "loading");
    let stop = false;
    let detach = () => {};
    const load = async () => {
      try {
        const r = await fetch("/api/data/earthquakes");
        const d = await r.json();
        if (stop) return;
        if (d.warming_up) { setStatus("earthquakes", "loading", 0, "warming up — first poll can take a minute"); return; }
        const fc = {
          type: "FeatureCollection",
          features: (d.quakes || []).filter((q: any) => q.lat != null && q.lon != null).map((q: any) => ({
            type: "Feature",
            geometry: { type: "Point", coordinates: [q.lon, q.lat] },
            properties: {
              id: q.id, mag: q.mag, place: q.place, depth: q.depth, time: q.time,
              tsunami: !!q.tsunami, sig: q.sig, magType: q.magType, status: q.status,
              type: q.type, url: q.url, color: quakeMagnitudeColor(q.mag),
            },
          })),
        };
        const src: any = map.getSource("earthquakes");
        if (src) {
          src.setData(fc as any);
        } else {
          map.addSource("earthquakes", { type: "geojson", data: fc as any });
          map.addLayer({
            id: "earthquakes-sym", type: "symbol", source: "earthquakes",
            layout: {
              "icon-image": "vt-quake",
              "icon-size": ["interpolate", ["linear"], ["get", "mag"], 2.5, 0.32, 4, 0.48, 6, 0.7, 8, 0.95],
              "icon-allow-overlap": true,
              "icon-ignore-placement": true,
            },
            paint: { "icon-color": ["get", "color"], "icon-opacity": 0.92 },
          });
          detach = attachLayerInteractions(map, "earthquakes-sym", (e: any) => {
            const f = e.features?.[0];
            if (!f) return;
            const p = f.properties;
            const dossierKey = `quake:${p.id}:${Date.now()}`;
            setDetail({
              kind: "quake",
              title: `M${p.mag != null ? Number(p.mag).toFixed(1) : "?"} — ${p.place || "unknown location"}`,
              subtitle: `${p.time ? new Date(p.time).toUTCString() : "time unknown"}${p.depth != null ? ` · ${fmtKm(p.depth)} deep` : ""}`,
              body: `${p.type !== "earthquake" ? `Reported type: ${p.type}\n` : ""}` +
                    `${p.status ? `Review status: ${p.status}\n` : ""}` +
                    `${p.magType ? `Magnitude type: ${p.magType}\n` : ""}` +
                    `${p.tsunami ? "⚠ TSUNAMI ADVISORY ISSUED\n" : ""}` +
                    `\nUSGS Earthquake Hazards Program — official feed displayed as-is, ` +
                    `not for safety-of-life use.`,
              links: p.url ? [{ label: "USGS event page", href: p.url }] : [],
              dossierKey,
            });
            // Quakes aren't Everything Graph nodes — lat/lon-only dossier
            // (nearest strategic sites is the useful part here).
            fetchDossier(dossierKey, null, e.lngLat?.lat, e.lngLat?.lng);
          });
        }
        setStatus("earthquakes", "active", fc.features.length, "USGS · M2.5+ · rolling 24h · not for safety-of-life use");
      } catch {
        if (!stop) setStatus("earthquakes", "error");
      }
    };
    load();
    // 2-min refresh, hidden-tab gated (matches server's max-age=120)
    const iv = window.setInterval(() => { if (!document.hidden) load(); }, 2 * 60_000);
    return () => { stop = true; window.clearInterval(iv); detach(); };
  }, [enabled.earthquakes, mapReady, setStatus]);

  // ── NOAA NDBC ocean buoys (RAW; ~889 stations worldwide, latest obs —
  // wave height/period, wind, pressure, temps, no predictive claim). Off by
  // default (reference layer; initial-load budget, same precedent as
  // rivergauges/alerts/earthquakes). ──
  useEffect(() => {
    const map = mapRef.current;
    if (!enabled.buoys) {
      try {
        if (map?.getLayer("buoys-sym")) map.removeLayer("buoys-sym");
        if (map?.getSource("buoys")) map.removeSource("buoys");
      } catch {}
      setStatus("buoys", "off");
      return;
    }
    if (!map || !mapReady) return;
    setStatus("buoys", "loading");
    let stop = false;
    let detach = () => {};
    const load = async () => {
      try {
        const r = await fetch("/api/data/buoys");
        const d = await r.json();
        if (stop) return;
        if (d.warming_up) { setStatus("buoys", "loading", 0, "warming up — first poll can take a minute"); return; }
        const fc = {
          type: "FeatureCollection",
          features: (d.buoys || []).filter((b: any) => b.lat != null && b.lon != null).map((b: any) => ({
            type: "Feature",
            geometry: { type: "Point", coordinates: [b.lon, b.lat] },
            properties: {
              station: b.station, time: b.time, waveHeight: b.waveHeight,
              dominantPeriod: b.dominantPeriod, windSpeed: b.windSpeed, windDir: b.windDir,
              pressure: b.pressure, pressureTendency: b.pressureTendency,
              airTemp: b.airTemp, waterTemp: b.waterTemp,
            },
          })),
        };
        const src: any = map.getSource("buoys");
        if (src) {
          src.setData(fc as any);
        } else {
          map.addSource("buoys", { type: "geojson", data: fc as any });
          map.addLayer({
            id: "buoys-sym", type: "symbol", source: "buoys",
            layout: {
              "icon-image": "vt-buoy",
              "icon-size": ["interpolate", ["linear"], ["zoom"], 2, 0.4, 8, 0.7],
              "icon-allow-overlap": true,
              "icon-ignore-placement": true,
            },
            paint: { "icon-color": "#22d3ee", "icon-halo-color": "rgba(5,10,19,0.95)", "icon-halo-width": 1.3 },
          });
          detach = attachLayerInteractions(map, "buoys-sym", (e: any) => {
            const f = e.features?.[0];
            if (!f) return;
            const p = f.properties;
            const fmt = (v: any, unit: string, digits = 1) => v == null ? "no data" : `${Number(v).toFixed(digits)} ${unit}`;
            const dossierKey = `buoy:${p.station}:${Date.now()}`;
            setDetail({
              kind: "buoy",
              title: `Buoy ${p.station}`,
              subtitle: p.time ? new Date(p.time).toUTCString() : "time unknown",
              // hPa stays in both systems (NDBC/NWS marine convention — units.ts)
              body: `Wave height: ${fmtMetersSmall(p.waveHeight)}\n` +
                    `Dominant period: ${fmt(p.dominantPeriod, "s", 0)}\n` +
                    `Wind speed: ${fmtMetersPerSec(p.windSpeed)}\n` +
                    `Pressure: ${fmt(p.pressure, "hPa", 0)}${p.pressureTendency != null ? ` (${p.pressureTendency > 0 ? "+" : ""}${p.pressureTendency.toFixed(1)} hPa/3h)` : ""}\n` +
                    `Air / water temp: ${fmtCelsius(p.airTemp)} / ${fmtCelsius(p.waterTemp)}\n\n` +
                    `NOAA National Data Buoy Center — raw station reading, missing sensors shown as no data.`,
              links: [{ label: "NDBC station page", href: `https://www.ndbc.noaa.gov/station_page.php?station=${p.station}` }],
              dossierKey,
            });
            // Buoys aren't Everything Graph nodes — lat/lon-only dossier.
            fetchDossier(dossierKey, null, e.lngLat?.lat, e.lngLat?.lng);
          });
        }
        setStatus("buoys", "active", fc.features.length, "NOAA NDBC · latest observations");
      } catch {
        if (!stop) setStatus("buoys", "error");
      }
    };
    load();
    // 5-min refresh, hidden-tab gated (matches server's max-age=300)
    const iv = window.setInterval(() => { if (!document.hidden) load(); }, 5 * 60_000);
    return () => { stop = true; window.clearInterval(iv); detach(); };
  }, [enabled.buoys, mapReady, setStatus]);

  // ── dark-ship RAW statistics (non-geospatial; derived from our own AIS
  // archive — counts only, per-vessel claims stay ladder-gated) ──
  const [shadowStats, setShadowStats] = useState<any>(null);
  useEffect(() => {
    if (!enabled.shadowstats) { setStatus("shadowstats", "off"); setShadowStats(null); return; }
    if (!mapSettled) { setStatus("shadowstats", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("shadowstats", "loading");
    let stop = false;
    const load = async () => {
      try {
        const r = await fetch("/api/data/shadowstats");
        const d = await r.json();
        if (stop) return;
        setShadowStats(d);
        setStatus("shadowstats", "active", d.gap_events,
          `${d.window_hours}h: ${d.gap_events} gaps · ${d.identity_candidates} identity cand. · ${d.loiter_events} loiter (${d.vessels_seen} vessels archived)`);
      } catch {
        if (!stop) setStatus("shadowstats", "error");
      }
    };
    load();
    const iv = window.setInterval(load, 10 * 60_000);
    return () => { stop = true; window.clearInterval(iv); };
  }, [enabled.shadowstats, mapSettled, setStatus]);

  // ── port dwell RAW statistics (fusion directive 2026-07-04) — per-port
  // arrivals/dwell from our own AIS archive, rendered as small text labels
  // under the 9 imagery-verified port sites plus a per-port panel note.
  // Anomaly FLAGS only (3x median, thin-history suppressed); the congestion
  // SIGNAL stays ladder-gated. ──
  const [dwellStats, setDwellStats] = useState<any>(null);
  useEffect(() => {
    const map = mapRef.current;
    if (!enabled.portdwell) {
      try {
        if (map?.getLayer("portdwell-labels")) map.removeLayer("portdwell-labels");
        if (map?.getSource("portdwell")) map.removeSource("portdwell");
      } catch {}
      setStatus("portdwell", "off"); setDwellStats(null);
      return;
    }
    if (!map || !mapReady) return;
    if (!mapSettled) { setStatus("portdwell", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("portdwell", "loading");
    let stop = false;
    let detachClicks: (() => void) | null = null;
    const load = async () => {
      try {
        const r = await fetch("/api/data/portdwell");
        const d = await r.json();
        if (stop) return;
        setDwellStats(d);
        const fc = {
          type: "FeatureCollection",
          features: (d.ports || []).map((p: any) => ({
            type: "Feature",
            geometry: { type: "Point", coordinates: [p.lon, p.lat] },
            properties: {
              label: `${String(p.name).replace(/^Port of /, "")}\n` +
                     `${p.in_port_now} in port · ` +
                     (p.dwell_median_h != null ? `med ${p.dwell_median_h}h` : `${p.visits_completed} calls`),
              // full per-port stats ride in the feature so the click card
              // (attached ONCE below) always shows the latest setData payload
              name: p.name,
              in_port_now: p.in_port_now,
              visits_completed: p.visits_completed,
              unique_vessels: p.unique_vessels,
              dwell_median_h: p.dwell_median_h,
              dwell_p90_h: p.dwell_p90_h,
              dwell_max_h: p.dwell_max_h,
              anomaly_count: p.anomaly_count,
              anomaly_examples: JSON.stringify(p.anomaly_examples || []),
              window_hours: d.window_hours,
              caveat: d.caveat, // server's own honesty text — pass through verbatim
            },
          })),
        };
        const src: any = map.getSource("portdwell");
        if (src) src.setData(fc as any);
        else {
          map.addSource("portdwell", { type: "geojson", data: fc as any });
          map.addLayer({
            id: "portdwell-labels", type: "symbol", source: "portdwell",
            layout: {
              "text-field": ["get", "label"],
              "text-font": ["Open Sans Semibold"],
              "text-size": 10,
              "text-anchor": "top",
              "text-offset": [0, 1.4],
            },
            paint: {
              "text-color": "#4ade80",
              "text-halo-color": "rgba(5,10,19,0.95)",
              "text-halo-width": 1.3,
            },
          });
          // click card (user report: clicking a port marker showed the
          // Starlink coverage fallback because ports had NO handler at all).
          // Attached once per layer creation, detached in cleanup — never
          // per-load (BUG 4: stacked anonymous handlers).
          detachClicks = attachLayerInteractions(map, "portdwell-labels", (e: any) => {
            const p = e.features?.[0]?.properties; if (!p) return;
            setDetail({ kind: "port", ...formatPortDetail(p) });
          });
        }
        setStatus("portdwell", "active", d.visits_completed,
          `${Math.round(d.window_hours / 24)}d: ${d.visits_completed} completed calls · ${d.in_port_now} in port now · ` +
          `${d.anomaly_count} anomaly flag${d.anomaly_count === 1 ? "" : "s"} (${d.vessels_seen} vessels archived)`);
      } catch {
        if (!stop) setStatus("portdwell", "error");
      }
    };
    load();
    const iv = window.setInterval(load, 10 * 60_000);
    return () => { stop = true; window.clearInterval(iv); detachClicks?.(); };
  }, [enabled.portdwell, mapReady, mapSettled, setStatus]);

  // ── SEC EDGAR Form 4 insider transactions (RAW; non-geospatial — no
  // markers, an inline list inside the layer panel instead) ──
  useEffect(() => {
    if (!enabled.insider) { setStatus("insider", "off"); return; }
    if (!mapSettled) { setStatus("insider", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("insider", "loading");
    let stop = false;
    const load = async () => {
      try {
        const r = await fetch("/api/data/insider");
        const d = await r.json();
        if (stop) return;
        if (d.warming_up) { setStatus("insider", "loading", 0, "warming up — first poll can take a minute"); return; }
        setStatus("insider", "active", d.count ?? (d.filings || []).length);
      } catch {
        if (!stop) setStatus("insider", "error", undefined, "feed error — retrying");
      }
    };
    load();
    // 300s ([REPAIR 2026-07-05] map perf): this poll only renders a panel
    // COUNT — the actual feed lives in the overlay view; the server caches
    // upstream at 15-min anyway. Hidden-tab gated inside load's fetch cost
    // via the interval guard.
    const iv = window.setInterval(() => { if (!document.hidden) load(); }, 300_000);
    return () => { stop = true; window.clearInterval(iv); };
  }, [enabled.insider, mapSettled, setStatus]);

  // ── SEC EDGAR 8-K Item 2.02 earnings-language releases (RAW;
  // non-geospatial — same inline-panel-row + full-view pattern as insider) ──
  useEffect(() => {
    if (!enabled.earnings) { setStatus("earnings", "off"); return; }
    if (!mapSettled) { setStatus("earnings", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("earnings", "loading");
    let stop = false;
    const load = async () => {
      try {
        const r = await fetch("/api/data/earnings-language");
        const d = await r.json();
        if (stop) return;
        if (d.warming_up) { setStatus("earnings", "loading", 0, "warming up — first poll can take a minute"); return; }
        setStatus("earnings", "active", d.count ?? (d.filings || []).length);
      } catch {
        if (!stop) setStatus("earnings", "error", undefined, "feed error — retrying");
      }
    };
    load();
    // 300s + hidden-gate — same rationale as the insider poll above.
    const iv = window.setInterval(() => { if (!document.hidden) load(); }, 300_000);
    return () => { stop = true; window.clearInterval(iv); };
  }, [enabled.earnings, mapSettled, setStatus]);

  // ── FINRA daily short-sale volume (RAW; non-geospatial — same
  // inline-panel-row + full-view pattern as insider/earnings). The
  // underlying data is a once-per-trading-day batch (server itself only
  // refetches every 6h), so this poll exists to refresh the panel's
  // symbol-count badge, not to chase intraday freshness — 300s matches
  // the sibling layers rather than inventing a slower one-off cadence. ──
  useEffect(() => {
    if (!enabled.shortvol) { setStatus("shortvol", "off"); return; }
    if (!mapSettled) { setStatus("shortvol", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("shortvol", "loading");
    let stop = false;
    const load = async () => {
      try {
        const r = await fetch("/api/data/short-volume");
        const d = await r.json();
        if (stop) return;
        if (d.warming_up) { setStatus("shortvol", "loading", 0, "warming up — first poll can take a minute"); return; }
        setStatus("shortvol", "active", d.summary?.symbols);
      } catch {
        if (!stop) setStatus("shortvol", "error", undefined, "feed error — retrying");
      }
    };
    load();
    const iv = window.setInterval(() => { if (!document.hidden) load(); }, 300_000);
    return () => { stop = true; window.clearInterval(iv); };
  }, [enabled.shortvol, mapSettled, setStatus]);

  // ── Wikipedia pageviews attention proxy (RAW; non-geospatial — same
  // inline-panel-row + full-view pattern as insider/earnings/shortvol).
  // BUILD ORDER 5 #3 pipeline shipped API-only 2026-07-05; this is its
  // UI follow-up (the "next lowest-effort UI gap" the build-order note
  // named). Server polls every 12h; this poll only refreshes the panel's
  // ticker-count badge, same cadence convention as the sibling layers. ──
  useEffect(() => {
    if (!enabled.attention) { setStatus("attention", "off"); return; }
    if (!mapSettled) { setStatus("attention", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("attention", "loading");
    let stop = false;
    const load = async () => {
      try {
        const r = await fetch("/api/data/attention");
        const d = await r.json();
        if (stop) return;
        if (d.warming_up) { setStatus("attention", "loading", 0, "warming up — first poll can take a minute"); return; }
        setStatus("attention", "active", d.count);
      } catch {
        if (!stop) setStatus("attention", "error", undefined, "feed error — retrying");
      }
    };
    load();
    const iv = window.setInterval(() => { if (!document.hidden) load(); }, 300_000);
    return () => { stop = true; window.clearInterval(iv); };
  }, [enabled.attention, mapSettled, setStatus]);

  // ── CFTC Commitments of Traders, disaggregated (RAW; non-geospatial —
  // same inline-panel-row + full-view pattern as insider/earnings/shortvol/
  // attention). BUILD ORDER 5 #2 pipeline shipped API-only 2026-07-05; this
  // is its UI follow-up (the last remaining item from that build-order's
  // own UI-status note). Server polls every 12h; this poll only refreshes
  // the panel's market-count badge, same cadence convention as the
  // sibling layers. ──
  useEffect(() => {
    if (!enabled.cot) { setStatus("cot", "off"); return; }
    if (!mapSettled) { setStatus("cot", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("cot", "loading");
    let stop = false;
    const load = async () => {
      try {
        const r = await fetch("/api/data/cot");
        const d = await r.json();
        if (stop) return;
        if (d.warming_up) { setStatus("cot", "loading", 0, "warming up — first poll can take a minute"); return; }
        setStatus("cot", "active", d.count);
      } catch {
        if (!stop) setStatus("cot", "error", undefined, "feed error — retrying");
      }
    };
    load();
    const iv = window.setInterval(() => { if (!document.hidden) load(); }, 300_000);
    return () => { stop = true; window.clearInterval(iv); };
  }, [enabled.cot, mapSettled, setStatus]);

  // ── Everything Graph (RAW join over insiders/facilities/vessels; non-
  // geospatial — same inline-panel-row + full-view pattern as insider/
  // earnings/shortvol). Server rebuilds every 15 min; this poll only reads
  // the cached counts-only summary, never a full entity dump. ──
  useEffect(() => {
    if (!enabled.graph) { setStatus("graph", "off"); return; }
    if (!mapSettled) { setStatus("graph", "loading", undefined, "queued — mounts after the map settles"); return; }
    setStatus("graph", "loading");
    let stop = false;
    const load = async () => {
      try {
        const r = await fetch("/api/data/graph");
        const d = await r.json();
        if (stop) return;
        if (d.warming_up) { setStatus("graph", "loading", 0, "first graph build in progress — retry shortly"); return; }
        setStatus("graph", "active", d.counts?.nodes, d.counts ? `${d.counts.edges.toLocaleString()} connections` : undefined);
      } catch {
        if (!stop) setStatus("graph", "error", undefined, "feed error — retrying");
      }
    };
    load();
    const iv = window.setInterval(() => { if (!document.hidden) load(); }, 300_000);
    return () => { stop = true; window.clearInterval(iv); };
  }, [enabled.graph, mapSettled, setStatus]);

  // ── panel helpers ──
  const layerIcon = (id: string) =>
    id === "imagery" ? <Satellite size={15} /> :
    id === "terrain" ? <Mountain size={15} /> :
    id === "seafloor" ? <Anchor size={15} /> :
    id === "seafloor_confidence" ? <Gauge size={15} /> :
    id === "weather" ? <CloudRain size={15} /> :
    id === "weather_temp" ? <Thermometer size={15} /> :
    id === "weather_wind" ? <Wind size={15} /> :
    id === "aircraft" ? <Plane size={15} /> :
    id === "vessels" ? <Ship size={15} /> :
    id === "sites" ? <MapPin size={15} /> :
    id === "powerplants" ? <Zap size={15} /> :
    id === "military_installations" ? <Shield size={15} /> :
    id === "trains" ? <TrainFront size={15} /> :
    id === "fires" ? <Flame size={15} /> :
    id === "nightlights" ? <Moon size={15} /> :
    id === "aerosol" ? <CloudFog size={15} /> :
    id === "vegetation" ? <Leaf size={15} /> :
    id === "soilmoisture" ? <Droplets size={15} /> :
    id === "no2" ? <Factory size={15} /> :
    id === "firetemp" ? <ThermometerSun size={15} /> :
    id === "biomass" ? <TreePine size={15} /> :
    id === "earthquakes" ? <Activity size={15} /> :
    id === "buoys" ? <Waves size={15} /> :
    id === "attention" ? <Eye size={15} /> :
    id === "cot" ? <Scale size={15} /> :
    id === "insider" || id === "earnings" ? <FileText size={15} /> :
    id === "shortvol" ? <TrendingUp size={15} /> :
    id === "graph" ? <Share2 size={15} /> : <LayersIcon size={15} />;

  const statusFor = (l: LayerMeta): { dot: string; text: string; note?: string } => {
    const rt = runtime[l.id];
    if (l.status === "planned") return { dot: "var(--text-tertiary)", text: "coming soon" };
    // health-aware registry override ([REPAIR 2026-07-05] audit #2): a feed
    // the server knows is down says so — never "off", never a dead toggle
    // pretending the layer is fine
    if (l.status === "down") return { dot: "var(--accent-red)", text: "feed down", note: (l as any).status_note || "source outage — auto-recovers" };
    if (l.status === "awaiting_key" || rt?.status === "awaiting_key") return { dot: "var(--accent-orange)", text: "awaiting API key" };
    if (rt?.status === "error") return { dot: "var(--accent-red)", text: rt.note || "feed error — retrying" };
    // v2.4 eternal-spinner rule: loading always carries its note (the OWM
    // "activating" retry note was being dropped here — the production defect).
    if (rt?.status === "loading") return { dot: "var(--accent-orange)", text: "loading…", note: rt.note };
    if (rt?.status === "active") {
      const c = rt.count;
      const unit = l.id === "sites" ? "sites" : l.id === "insider" ? "filings" : l.id === "earnings" ? "releases" : l.id === "shortvol" ? "symbols" : l.id === "powerplants" ? "plants" : l.id === "trains" ? "trains" : l.id === "shadowstats" ? "gap events" : l.id === "portdwell" ? "port calls" : l.id === "fires" ? "detections" : l.id === "graph" ? "entities" : l.id === "earthquakes" ? "quakes" : l.id === "buoys" ? "stations" : l.id === "attention" ? "tickers" : l.id === "cot" ? "markets" : l.id;
      return { dot: "var(--accent-green)", text: c != null ? `${c.toLocaleString()} ${unit}` : "active", note: rt.note };
    }
    return { dot: "var(--text-tertiary)", text: "off" };
  };

  const toggleable = (l: LayerMeta) => l.status === "live";

  // ── W6 ANALYST map commands (console charter): the pane executes the
  // server-validated commands through THIS callback so toggle_layer drives
  // the exact same `enabled` state the layer panel's switches use (no
  // parallel state) and fly_to uses the live map ref. Returns the
  // human-readable note the chat renders ("→ flew to …"). Honesty: a layer
  // this bundle can't toggle (awaiting key, planned, unwired mid-deploy)
  // says so instead of flipping a switch that paints nothing.
  const runAnalystMapCommand = useCallback((cmd: AnalystMapCommand): string => {
    if (cmd?.command === "fly_to" && Number.isFinite(cmd.lat) && Number.isFinite(cmd.lon)) {
      try {
        mapRef.current?.flyTo({
          center: [cmd.lon, cmd.lat],
          ...(cmd.zoom != null ? { zoom: cmd.zoom } : {}),
        });
      } catch { return `fly_to failed — map not ready`; }
      return `flew to ${cmd.lat!.toFixed(2)}, ${cmd.lon!.toFixed(2)}${cmd.zoom != null ? ` (z${cmd.zoom})` : ""}`;
    }
    if (cmd?.command === "toggle_layer" && typeof cmd.layer === "string" && typeof cmd.on === "boolean") {
      const id = cmd.layer, on = cmd.on;
      const meta = layers.find((l) => l.id === id);
      if (!meta) return `layer '${id}' is not in this page's registry — not toggled`;
      // same unwired guard as the panel row (open-tab skew, [REPAIR R15])
      const unwired = !(id in LAYER_GROUP) && meta.kind !== "signal" && meta.status !== "planned";
      if (!toggleable(meta) || unwired)
        return `layer '${meta.name}' can't be toggled (${unwired ? "reload to enable" : meta.status.replace("_", " ")})`;
      setEnabled((s) => ({ ...s, [id]: on }));
      return `turned ${on ? "on" : "off"} ${meta.name}`;
    }
    return "unrecognized map command — ignored";
  }, [layers]);

  // active-cost-budget advisory (BUILD ORDER 4 #2): sums the registry-native
  // costTier of every currently-active layer. Today's default-on set
  // (imagery+aircraft+sites+insider+earnings+shortvol+powerplants+trains+
  // shadowstats+portdwell) weighs 14 — at the "light" ceiling, so the badge
  // stays silent by default (zero visual regression); it only surfaces once
  // a user (or a future larger registry's defaults) actually loads enough
  // heavy layers to matter.
  const activeCostWeight = layers.reduce(
    (sum, l) => sum + (enabled[l.id] && toggleable(l) ? costWeightOf(l) : 0), 0);
  const costLoad: "light" | "moderate" | "heavy" =
    activeCostWeight <= 14 ? "light" : activeCostWeight <= 26 ? "moderate" : "heavy";

  const renderLayerRow = (l: LayerMeta) => {
    const st = statusFor(l);
    const on = !!enabled[l.id] && toggleable(l);
    const descIsOpen = !!descOpen[l.id];
    // Open-tab skew guard: a live layer id this bundle has NO wiring for
    // (registry newer than the running code) must not render a
    // functional-looking toggle — pill would flip while the label stays
    // "off" and nothing paints (the 2026-07-04 production desync).
    const unwired = !(l.id in LAYER_GROUP) && l.kind !== "signal" && l.status !== "planned";
    return (
      <div key={l.id}>
        <div className={`vt-layer-row${toggleable(l) ? "" : " vt-layer-row-disabled"}`} data-vt-layer={l.id}
             data-vt-rt={runtime[l.id]?.status || "none"} data-vt-since={statusAtRef.current[l.id] || 0}>
          <span className="vt-layer-ic">{layerIcon(l.id)}</span>
          <span className="vt-layer-name">
            <button className="vt-layer-namebtn" aria-expanded={descIsOpen}
                    aria-label={`About ${l.name}`}
                    onClick={() => setDescOpen((s) => ({ ...s, [l.id]: !s[l.id] }))}>
              {l.name} <Info size={11} aria-hidden style={{ opacity: 0.55 }} />
            </button>
            <span className={`vt-kind-badge ${l.kind}`}>{l.kind === "raw" ? "RAW" : "SIGNAL"}</span>
            <span className="vt-layer-status">
              <i style={{ background: st.dot }} /> {unwired ? "reload to enable" : st.text}
            </span>
            {unwired && <span className="vt-layer-covnote">site updated — reload the page to enable this new layer</span>}
            {!unwired && st.note && <span className="vt-layer-covnote">{st.note}</span>}
          </span>
          <button
            role="switch"
            aria-checked={on}
            aria-label={`Toggle ${l.name}`}
            disabled={!toggleable(l) || unwired}
            className={`vt-switch${on ? " on" : ""}`}
            onClick={() => setEnabled(s => ({ ...s, [l.id]: !s[l.id] }))}
          >
            <i />
          </button>
        </div>
        {descIsOpen && (
          <div className="vt-layer-desc" role="note">
            {l.description}
            <span className="vt-layer-desc-src">Source: {l.source}</span>
          </div>
        )}
        {(l as any).field && on && (
          <div className="vt-field-controls" role="group" aria-label={`${l.name} display controls`}>
            <label className="vt-field-slider">
              <span>intensity {opacityOf(l.id)}%</span>
              <input
                type="range" min={0} max={100} step={5}
                value={opacityOf(l.id)}
                aria-label={`${l.name} opacity`}
                onChange={(e) => setFieldOpacity(l.id, Number(e.target.value))}
              />
            </label>
            {l.id === "weather_wind" && (
              <label className="vt-field-check">
                <input type="checkbox" checked={windArrows} onChange={() => setWindArrows(!windArrows)} />
                <span>arrows (direction + kt)</span>
              </label>
            )}
            {l.id === "weather_temp" && (
              <>
                <label className="vt-field-check">
                  <input type="checkbox" checked={tempLabels} onChange={() => setTempLabels(!tempLabels)} />
                  <span>value labels</span>
                </label>
                {tempLabels && (
                  <button className="vt-field-unit" onClick={() => setTempUnitF(!tempUnitF)}
                          aria-label="Toggle temperature unit">
                    {tempUnitF ? "°F" : "°C"}
                  </button>
                )}
              </>
            )}
            {(l.id === "weather_wind" && windArrows) || (l.id === "weather_temp" && tempLabels) ? (
              <span className="vt-field-note">
                {wxGrid?.note || "sampling grid…"}
              </span>
            ) : null}
            {l.id === "nightlights" && (
              <div className="vt-gibs-scrubber" role="group" aria-label="Night lights date">
                <button
                  aria-label="Previous day"
                  onClick={() => setNightlightsDate((d) => gibsStepDate(d, -1))}
                >
                  <ChevronLeft size={13} />
                </button>
                <span className="vt-gibs-scrubber-date">{nightlightsDate} UTC</span>
                <button
                  aria-label="Next day"
                  disabled={gibsIsLatestAvailable(nightlightsDate, Date.now())}
                  onClick={() => setNightlightsDate((d) => gibsStepDate(d, 1))}
                >
                  <ChevronRight size={13} />
                </button>
              </div>
            )}
            {l.id === "aerosol" && (
              <div className="vt-gibs-scrubber" role="group" aria-label="Aerosol optical depth date">
                <button
                  aria-label="Previous day"
                  onClick={() => setAerosolDate((d) => gibsStepDate(d, -1))}
                >
                  <ChevronLeft size={13} />
                </button>
                <span className="vt-gibs-scrubber-date">{aerosolDate} UTC</span>
                <button
                  aria-label="Next day"
                  disabled={gibsIsLatestAvailable(aerosolDate, Date.now())}
                  onClick={() => setAerosolDate((d) => gibsStepDate(d, 1))}
                >
                  <ChevronRight size={13} />
                </button>
              </div>
            )}
            {l.id === "vegetation" && (
              <div className="vt-gibs-scrubber" role="group" aria-label="Vegetation NDVI date">
                <button
                  aria-label="Previous day"
                  onClick={() => setVegetationDate((d) => gibsStepDate(d, -1))}
                >
                  <ChevronLeft size={13} />
                </button>
                <span className="vt-gibs-scrubber-date">{vegetationDate} UTC</span>
                <button
                  aria-label="Next day"
                  disabled={gibsIsLatestAvailable(vegetationDate, Date.now())}
                  onClick={() => setVegetationDate((d) => gibsStepDate(d, 1))}
                >
                  <ChevronRight size={13} />
                </button>
              </div>
            )}
            {l.id === "soilmoisture" && (
              <div className="vt-gibs-scrubber" role="group" aria-label="Soil moisture date">
                <button
                  aria-label="Previous day"
                  onClick={() => setSoilmoistureDate((d) => gibsStepDate(d, -1))}
                >
                  <ChevronLeft size={13} />
                </button>
                <span className="vt-gibs-scrubber-date">{soilmoistureDate} UTC</span>
                <button
                  aria-label="Next day"
                  disabled={gibsIsLatestAvailable(soilmoistureDate, Date.now(), SOIL_LATENCY_DAYS)}
                  onClick={() => setSoilmoistureDate((d) => gibsStepDate(d, 1))}
                >
                  <ChevronRight size={13} />
                </button>
              </div>
            )}
            {l.id === "no2" && (
              <div className="vt-gibs-scrubber" role="group" aria-label="NO2 date">
                <button
                  aria-label="Previous day"
                  onClick={() => setNo2Date((d) => gibsStepDate(d, -1))}
                >
                  <ChevronLeft size={13} />
                </button>
                <span className="vt-gibs-scrubber-date">{no2Date} UTC</span>
                <button
                  aria-label="Next day"
                  disabled={gibsIsLatestAvailable(no2Date, Date.now())}
                  onClick={() => setNo2Date((d) => gibsStepDate(d, 1))}
                >
                  <ChevronRight size={13} />
                </button>
              </div>
            )}
            {l.id === "firetemp" && (
              // No day scrubber (10-min irregular cadence — always "default"/
              // freshest). Freshness comes from the real layer-time-actual
              // header, never a guess; "unknown" until the first fetch lands.
              <span className="vt-field-note">
                {firetempScanTime ? `scan: ${firetempScanTime} UTC` : "live · ~10-min cadence · scan time unknown"}
              </span>
            )}
          </div>
        )}
        {l.id === "shadowstats" && on && shadowStats && shadowStats.loiter_events > 0 && (
          <div className="vt-layer-desc" role="note">
            {Object.entries(shadowStats.loiter_by_zone as Record<string, number>)
              .filter(([, n]) => n > 0)
              .map(([zid, n]) => {
                const z = (shadowStats.zones || []).find((x: any) => x.id === zid);
                return `${z?.name || zid}: ${n} loitering`;
              }).join(" · ")}
          </div>
        )}
        {l.id === "portdwell" && on && dwellStats && (dwellStats.ports || []).length > 0 && (
          <div className="vt-layer-desc" role="note">
            {(dwellStats.ports as any[])
              .filter((p) => p.in_port_now > 0 || p.visits_completed > 0)
              .map((p) => `${String(p.name).replace(/^Port of /, "")}: ${p.in_port_now} in port` +
                          (p.dwell_median_h != null ? ` · med ${p.dwell_median_h}h` : ""))
              .join(" · ") || "no port calls in window yet (archive accumulating)"}
          </div>
        )}
        {l.id === "insider" && on && (
          // v2.3: the filings FEED does not belong inside a layer-toggle
          // sidebar — it lives in the full view; the panel keeps one button.
          <div style={{ padding: "0 14px" }}>
            <button className="vt-filings-openfull"
                    onClick={() => { window.location.hash = "#/data/filings"; setFilingsOpen(true); }}>
              Open filings view — history, filters, SEC links →
            </button>
          </div>
        )}
        {l.id === "earnings" && on && (
          // Same pattern as insider: a scrolling text feed doesn't belong in
          // a layer-toggle sidebar — the panel keeps one button to the view.
          <div style={{ padding: "0 14px" }}>
            <button className="vt-filings-openfull"
                    onClick={() => { window.location.hash = "#/data/earnings"; setEarningsOpen(true); }}>
              Open earnings language view — full releases, SEC links →
            </button>
          </div>
        )}
        {l.id === "shortvol" && on && (
          // Same pattern as insider/earnings: a per-symbol ratio table +
          // search doesn't belong in a layer-toggle sidebar.
          <div style={{ padding: "0 14px" }}>
            <button className="vt-filings-openfull"
                    onClick={() => { window.location.hash = "#/data/short-volume"; setShortvolOpen(true); }}>
              Open short-volume view — ticker lookup, top movers →
            </button>
          </div>
        )}
        {l.id === "attention" && on && (
          // Same pattern as insider/earnings/shortvol: a ticker search +
          // trend table doesn't belong in a layer-toggle sidebar.
          <div style={{ padding: "0 14px" }}>
            <button className="vt-filings-openfull"
                    onClick={() => { window.location.hash = "#/data/attention"; setAttentionOpen(true); }}>
              Open attention view — ticker lookup, ranked panel →
            </button>
          </div>
        )}
        {l.id === "cot" && on && (
          // Same pattern as insider/earnings/shortvol/attention: a market
          // search + weekly positioning table doesn't belong in a
          // layer-toggle sidebar.
          <div style={{ padding: "0 14px" }}>
            <button className="vt-filings-openfull"
                    onClick={() => { window.location.hash = "#/data/cot"; setCotOpen(true); }}>
              Open COT view — market search, ranked panel →
            </button>
          </div>
        )}
        {l.id === "graph" && on && (
          // Same pattern as insider/earnings/shortvol: an entity-search +
          // connections view doesn't belong in a layer-toggle sidebar.
          <div style={{ padding: "0 14px" }}>
            <button className="vt-filings-openfull"
                    onClick={() => { window.location.hash = "#/data/graph"; setGraphOpen(true); }}>
              Open Everything Graph — entity search, connections →
            </button>
          </div>
        )}
      </div>
    );
  };

  // shared by every named PANEL_GROUPS entry AND the unknown-group catch-all
  // below — GROUP_ROW_CAP progressive disclosure lives here once, not
  // duplicated per call site.
  const renderPanelGroup = (id: string, label: string, members: LayerMeta[]) => {
    if (members.length === 0) return null;
    const onCount = members.filter((l) => !!enabled[l.id] && toggleable(l)).length;
    // groups outside PANEL_GROUPS (the "_more" catch-all) have no entry in
    // groupCollapsed's initial state — default those to collapsed, same
    // rule OPEN_GROUPS_BY_DEFAULT already applies to every named group.
    const isCollapsed = groupCollapsed[id] !== undefined ? groupCollapsed[id] : !OPEN_GROUPS_BY_DEFAULT.has(id);
    const showAll = members.length <= GROUP_ROW_CAP || !!groupShowAll[id];
    const shown = showAll ? members : members.slice(0, GROUP_ROW_CAP);
    return (
      <div key={id} className="vt-layer-group">
        <button className="vt-layer-group-head" aria-expanded={!isCollapsed}
                onClick={() => setGroupCollapsed((s) => ({ ...s, [id]: !s[id] }))}>
          <span className={`vt-layer-group-chev${isCollapsed ? " closed" : ""}`}>▾</span>
          <span>{label}</span>
          <span className="vt-layer-group-count">{onCount}/{members.length} on</span>
        </button>
        {!isCollapsed && (
          <>
            {shown.map((l) => renderLayerRow(l))}
            {!showAll && (
              <button className="vt-layer-showmore"
                      onClick={() => setGroupShowAll((s) => ({ ...s, [id]: true }))}>
                + {members.length - GROUP_ROW_CAP} more — show all
              </button>
            )}
          </>
        )}
      </div>
    );
  };

  return (
    <div className="vt-map-page" data-vt-map>
      {filingsOpen && (
        <FilingsView onBack={() => { window.location.hash = "#/data"; setFilingsOpen(false); }} />
      )}
      {earningsOpen && (
        <EarningsView onBack={() => { window.location.hash = "#/data"; setEarningsOpen(false); }} />
      )}
      {shortvolOpen && (
        <ShortVolView onBack={() => { window.location.hash = "#/data"; setShortvolOpen(false); }} />
      )}
      {attentionOpen && (
        <AttentionView onBack={() => { window.location.hash = "#/data"; setAttentionOpen(false); }} />
      )}
      {cotOpen && (
        <CotView onBack={() => { window.location.hash = "#/data"; setCotOpen(false); }} />
      )}
      {graphOpen && (
        <GraphView onBack={() => { window.location.hash = "#/data"; setGraphOpen(false); }} />
      )}
      {streamsOpen && (
        <StreamsView onBack={() => { window.location.hash = "#/data"; setStreamsOpen(false); }} />
      )}
      {gridStressOpen && (
        <GridStressView onBack={() => { window.location.hash = "#/data"; setGridStressOpen(false); }} />
      )}

      {/* EARTH TWIN E1 remainder: persistent LIVE/HISTORICAL badge, outside
          the Time Machine panel — see the historicalAtMs effect above.
          "Back to live" mirrors the panel's own close behavior (its unmount
          cleanup already resets the axis) so there is one path that resets
          both the panel's replay state and the axis, never two that could
          drift apart; the direct setTimeAxis call is a defensive fallback
          for the (currently unreachable) case of historical mode without
          the panel open. */}
      {historicalAtMs !== null && (
        <div className="vt-time-axis-badge" data-testid="time-axis-badge" role="status">
          <Clock size={13} />
          <span>HISTORICAL — {formatAxisInstant(historicalAtMs)}</span>
          <button
            className="vt-time-axis-badge-live"
            onClick={() => {
              if (timescrubOpen) setTimescrubOpen(false);
              else setTimeAxis({ mode: "live" });
            }}
          >
            Back to live
          </button>
        </div>
      )}

      {/* EARTH TWIN E3 follow-up: 3D aircraft hover tooltip (altitude on
          hover). Always mounted (so the ref is live before the aircraft
          effect ever fires) but hidden by default; the aircraft effect
          below writes text/position/display directly, never via setState —
          pointer-events:none keeps it from stealing map interaction. */}
      <div ref={airHoverTipRef} className="vt-air-hover-tip" style={{ display: "none" }} />

      {/* Phase 3a imagery capture-date chip (DESIGN.md imagery-honesty
          rule: display dates where available; unknown states stay loud).
          Hidden while the space frame owns the viewport — a capture-date
          for a shrinking-globe map reads as noise; returns at the seam. */}
      {enabled.imagery && !spaceActive && (
        <div className="vt-imagery-date-chip" data-testid="imagery-date" role="status"
             title="Capture date of the Esri World Imagery at the view centre — dates vary within a view and by zoom level">
          {imageryDate.label}
        </div>
      )}

      {/* v2.3 fullscreen: hide the site nav for a full-viewport map */}
      <button className="vt-map-fs-btn" data-vt-fullscreen
              aria-label={fullscreen ? "Exit fullscreen map" : "Fullscreen map"}
              aria-pressed={fullscreen}
              onClick={() => setFullscreen((v) => !v)}>
        {fullscreen ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
      </button>
      {/* W1 globe/flat projection toggle — same control family as the
          fullscreen button above it (icon shows the action it performs,
          matching the fullscreen convention). Disabled-with-reason when
          the runtime can't do globe; the map stays flat, never broken. */}
      {(() => {
        const globeActive = globeOn && globeSupport !== "unavailable";
        return (
          <button className="vt-map-globe-btn" data-vt-globe
                  aria-label={globeActive ? "Switch to flat map projection" : "Switch to 3D globe projection"}
                  aria-pressed={globeActive}
                  disabled={globeSupport === "unavailable"}
                  title={globeSupport === "unavailable"
                    ? "3D globe unavailable on this device (the map runtime lacks globe projection support) — flat map shown"
                    : globeActive ? "Switch to flat map" : "Switch to 3D globe"}
                  onClick={() => setGlobeOn((v) => !v)}>
            {globeActive ? <FlatMapIcon size={18} /> : <GlobeIcon size={18} />}
          </button>
        );
      })()}
      {/* W6 ANALYST toggle — third button in the top-left control column
          (same 44px family as fullscreen/globe above it). The pane itself
          is a lazy chunk: closed = zero analyst code loaded, ever. */}
      <button className="vt-map-analyst-btn" data-vt-analyst
              aria-label={analystOpen ? "Close analyst" : "Open analyst"}
              aria-pressed={analystOpen}
              title="Analyst — ask questions answered strictly from the platform's own data"
              onClick={() => setAnalystOpen((v) => !v)}>
        <MessageSquareText size={18} />
      </button>
      {analystOpen && (
        <Suspense fallback={
          <div className="vt-analyst-panel vt-analyst-panel-boot" data-vt-analyst-panel
               role="dialog" aria-label="Analyst loading">
            <div className="vt-map-skeleton-shimmer" />
            <span>Loading analyst…</span>
          </div>
        }>
          <AnalystPane onClose={() => setAnalystOpen(false)} onMapCommand={runAnalystMapCommand} />
        </Suspense>
      )}
      {/* W3 TIME SCRUBBER toggle — fourth button in the top-left control
          column. Lazy chunk: closed = zero time-scrubber code loaded, ever. */}
      <button className="vt-map-timescrub-btn" data-vt-timescrub
              aria-label={timescrubOpen ? "Close time machine" : "Open time machine"}
              aria-pressed={timescrubOpen}
              title="Time Machine — scrub our own recorded archives across the past week"
              onClick={() => setTimescrubOpen((v) => !v)}>
        <Clock size={18} />
      </button>
      {timescrubOpen && (
        <Suspense fallback={
          <div className="vt-timescrub-panel vt-timescrub-panel-boot" data-vt-timescrub-panel
               role="dialog" aria-label="Time machine loading">
            <div className="vt-map-skeleton-shimmer" />
            <span>Loading time machine…</span>
          </div>
        }>
          <TimeScrubber map={mapRef.current} onClose={() => setTimescrubOpen(false)} />
        </Suspense>
      )}
      {/* Nuclear-tests time machine bar — appears with the layer; the "Lucy"
          scrub. Range input drives a GPU filter, so dragging is smooth. */}
      {(enabled.nucleartests || enabled.quakehistory) && (
        <div className="vt-nuke-timebar" role="group" aria-label="Nuclear test year scrubber">
          <button className="vt-preset-pill" aria-label={histPlay ? "Pause" : "Play"}
                  onClick={() => setHistPlay((v) => !v)}>
            {histPlay ? "❚❚" : "▶"}
          </button>
          <input
            type="range" min={HIST_MIN_YEAR} max={HIST_MAX_YEAR} step={1} value={histYear}
            aria-label="Test year"
            onChange={(e) => { setHistPlay(false); setHistYear(Number(e.target.value)); }}
          />
          <span className="vt-nuke-year">{histYear}</span>
        </div>
      )}
      {/* Style presets (worldview-globe G1) — real-first geographic looks,
          bottom-center segmented control. No tactical filters. Hidden while
          the space frame owns the viewport (a style switcher for a
          shrinking-globe map is meta-noise; returns at the seam). */}
      {!spaceActive && (
      <div className="vt-preset-switch" role="group" aria-label="Map style preset">
        {([
          ["natural", "Natural"],
          ["night", "Night"],
          ["terrain", "Terrain"],
          ["minimal", "Minimal"],
        ] as const).map(([id, label]) => (
          <button
            key={id}
            className={`vt-preset-pill${mapPreset === id ? " vt-preset-pill-on" : ""}`}
            aria-pressed={mapPreset === id}
            onClick={() => setMapPreset(id)}
          >
            {label}
          </button>
        ))}
      </div>
      )}

      <div ref={mapContainer} className="vt-map-canvas" />
      {fpsDebug && <FpsChip />}
      {atZoomFloor && !spaceActive && mapReady && (
        <button className="vt-space-chip" onClick={() => void enterSpace({ flyOut: true })}
                title="Keep zooming out — one continuous flight into the true-scale solar system; the live map is the Earth. Zoom back in (or Escape) to return">
          ☉ Solar system
        </button>
      )}

      {!mapReady && !mapError && (
        <div className="vt-map-skeleton" aria-label="Map loading">
          <div className="vt-map-skeleton-shimmer" />
          <span>Loading map…</span>
        </div>
      )}
      {mapError && (
        <div className="vt-map-skeleton">
          <span style={{ color: "var(--accent-red)" }}>{mapError}</span>
        </div>
      )}

      {/* Layers control — top-right; collapsed by default on phone */}
      <div className="vt-map-controls">
        {!panelOpen ? (
          <button className="vt-map-fab" aria-label="Open layers panel" onClick={() => setPanelOpen(true)}>
            <LayersIcon size={19} />
          </button>
        ) : (
          <div className="vt-layer-panel" role="region" aria-label="Map layers">
            <div className="vt-layer-panel-head">
              <span>
                <LayersIcon size={14} /> Layers
                {costLoad !== "light" && (
                  <span className="vt-cost-note" data-tier={costLoad} role="status"
                        title={`Active layer cost budget: ${activeCostWeight} (heavy layers weigh 4x, moderate 2x, light 1x)`}>
                    {costLoad} load
                  </span>
                )}
              </span>
              <span style={{ display: "inline-flex", gap: 2 }}>
                <button className="vt-icon-btn" aria-label="About data labels"
                        onClick={() => setShowRawInfo(v => !v)}>
                  <Info size={15} />
                </button>
                <button className="vt-icon-btn" aria-label="Collapse layers panel" onClick={() => setPanelOpen(false)}>
                  <X size={15} />
                </button>
              </span>
            </div>
            {showRawInfo && (
              <div className="vt-layer-info-tip" role="note">
                <b>RAW DATA</b> layers show sources as-is with attribution — no
                predictive claim. <b>SIGNAL</b> layers appear only after
                statistical validation (ladder gate 2). Coverage limits are
                stated per layer (terrestrial AIS has mid-ocean gaps; ADS-B
                follows receiver density).
              </div>
            )}
            {versionSkew && (
              <div className="vt-skew-note" role="status">
                Site updated to v{versionSkew} (you're on v{CLIENT_VERSION}) —
                reload the page to enable the newest layers.
              </div>
            )}
            {/* Site-wide unit system (human directive 2026-07-13): every
                measurement in cards/panels renders through lib/units.ts.
                Open cards keep their units until reopened. */}
            <div className="vt-preset-switch" role="group" aria-label="Unit system"
                 style={{ position: "static", transform: "none", margin: "6px 10px 2px", justifyContent: "flex-start", alignItems: "center" }}>
              <span className="vt-streams-launch-sub" style={{ marginRight: 6 }}>Units</span>
              {([["imperial", "mi · °F"], ["metric", "km · °C"]] as const).map(([id, label]) => (
                <button key={id}
                        className={`vt-preset-pill${unitSystem === id ? " vt-preset-pill-on" : ""}`}
                        aria-pressed={unitSystem === id}
                        onClick={() => setUnits(id)}>
                  {label}
                </button>
              ))}
            </div>
            {/* Streams inventory launcher (Phase 4, 2026-07-06): the archive
                census is page-wide, not layer-scoped, so it launches from the
                panel top — "nothing ships invisible". */}
            <button type="button" className="vt-streams-launch" data-vt-streams-launch
                    onClick={() => { window.location.hash = "#/data/streams"; setStreamsOpen(true); }}>
              <DatabaseIcon size={13} /> Streams inventory
              <span className="vt-streams-launch-sub">every archived stream · health &amp; freshness</span>
            </button>
            {/* Grid-stress launcher (GRID VISION A1 gate-2 FAIL path, 2026-07-07):
                TX/ERCOT-specific, not a spatial layer, so it launches from the
                panel top like Streams inventory rather than joining the map's
                on/off layer list. */}
            <button type="button" className="vt-streams-launch" data-vt-gridstress-launch
                    onClick={() => { window.location.hash = "#/data/grid-stress"; setGridStressOpen(true); }}>
              <Zap size={13} /> Grid stress (TX) — descriptive only
              <span className="vt-streams-launch-sub">gate-2 FAILED · non-predictive reading</span>
            </button>
            {PANEL_GROUPS.map((g) => renderPanelGroup(g.id, g.label, layers.filter((l) => groupOf(l) === g.id)))}
            {/* SELF-SEE FOR UNKNOWN GROUPS (BUILD ORDER 4 #2 — caught live by
                the layer-scale synthetic harness, not assumed): a registry-
                native `group` value the client doesn't have a PANEL_GROUPS
                label for must still render its layers somewhere — silently
                dropping them would be exactly the self-see violation
                DESIGN.md already treats as build-failing for known layers.
                Renders once, appended after the named groups, defaulting
                collapsed like any group not in OPEN_GROUPS_BY_DEFAULT. */}
            {(() => {
              const known = new Set(PANEL_GROUPS.map((g) => g.id));
              const orphans = layers.filter((l) => !known.has(groupOf(l)));
              return orphans.length ? renderPanelGroup("_more", "More", orphans) : null;
            })()}
            <LegendPanel
              legendOpen={legendOpen} setLegendOpen={setLegendOpen}
              enabled={enabled} airFilter={airFilter} setAirFilter={setAirFilter}
              nightlightsDate={nightlightsDate} aerosolDate={aerosolDate}
              vegetationDate={vegetationDate} soilmoistureDate={soilmoistureDate}
              no2Date={no2Date} firetempScanTime={firetempScanTime}
              tempUnitF={tempUnitF} windArrows={windArrows}
              orbitalGpRef={orbitalGpRef} gpVersion={gpVersion}
              satGroup={satGroup} satGroupCount={satGroupCount}
              satGroupOrbits={satGroupOrbits} satArcInfo={satArcInfo}
              applySatGroup={applySatGroup} setSatGroupOrbits={setSatGroupOrbits}
              onFindSat={findSat} seafloorConfShares={seafloorConfShares}
            />
          </div>
        )}
      </div>

      {/* Detail card — side card on desktop, bottom sheet on phone */}
      {satFollowing && (
        // O6 follow tools (human-requested): minimizable cluster — re-lock
        // the camera on the object, zoom in/out AROUND it, toggle the exact
        // ground spot it's passing over.
        <div ref={satToolsRef} className={`vt-sat-tools${satToolsMin ? " vt-sat-tools-min" : ""}`}
             title="Drag anywhere to move"
             onPointerDown={onToolsDown} onPointerMove={onToolsMove}
             onPointerUp={onToolsUp} onPointerCancel={onToolsUp}>
          <span className="vt-sat-tools-grip" aria-hidden>⠿</span>
          <button className="vt-icon-btn" aria-label={satToolsMin ? "Expand satellite tools" : "Minimize satellite tools"}
                  onClick={() => setSatToolsMin(!satToolsMin)}>
            {satToolsMin ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
          </button>
          {!satToolsMin && (
            <>
              <button className={`vt-satfinder-chip${satLockMode === "sat" ? " vt-satfinder-chip-on" : ""}`}
                      title="Pin the camera to the CRAFT — rotate/tilt/zoom orbit it while it stays centered, until unpressed"
                      onClick={() => {
                        const f = satFollowRef.current;
                        const next = satLockMode === "sat" ? null : "sat";
                        if (f) f.lockMode = next;
                        setSatLockMode(next);
                      }}>
                ◉ sat lock
              </button>
              <button className={`vt-satfinder-chip${satLockMode === "ground" ? " vt-satfinder-chip-on" : ""}`}
                      title="Pin the camera to the point on the ground below it (a drag releases)"
                      onClick={() => {
                        const f = satFollowRef.current;
                        const next = satLockMode === "ground" ? null : "ground";
                        if (f) f.lockMode = next;
                        setSatLockMode(next);
                      }}>
                ⌖ ground lock
              </button>
              <button className="vt-satfinder-chip" title="Zoom in on the object"
                      onClick={() => {
                        const t = followTarget(satLayerRef.current?.getPositions() ?? null, satFollowRef.current?.index ?? -1);
                        const m = mapRef.current;
                        if (t && m) try { m.easeTo({ center: [t.lonDeg, t.latDeg], zoom: Math.min((m.getZoom() ?? 0) + 1, 9), duration: 500 }); } catch {}
                      }}>
                +
              </button>
              <button className="vt-satfinder-chip" title="Zoom out"
                      onClick={() => {
                        const t = followTarget(satLayerRef.current?.getPositions() ?? null, satFollowRef.current?.index ?? -1);
                        const m = mapRef.current;
                        if (t && m) try { m.easeTo({ center: [t.lonDeg, t.latDeg], zoom: Math.max((m.getZoom() ?? 0) - 1, 1.2), duration: 500 }); } catch {}
                      }}>
                −
              </button>
              <button className={`vt-satfinder-chip${showNadir ? " vt-satfinder-chip-on" : ""}`}
                      title="Mark the exact ground point the object is passing over"
                      onClick={() => setShowNadir(!showNadir)}>
                ⌖ ground spot
              </button>
              <button className="vt-satfinder-chip"
                      title="Ease into close orbit around the craft — on the live map: rotate to circle it, tilt toward the horizon for the real Sun/Moon, zoom approaches the craft itself"
                      onClick={inspectCraft}>
                ⟳ inspect
              </button>
            </>
          )}
        </div>
      )}
      {detail && detailMin && (
        // O6 minimize (human-requested): the card collapses to a pill so the
        // globe shows through — the focus/follow keeps running underneath;
        // click the pill to restore, ✕ still ends everything.
        <div ref={detailCardRef} className="vt-site-card vt-site-card-min" role="dialog" aria-label={detail.title}
             style={{ cursor: "grab", touchAction: "none" }}
             onPointerDown={onCardHeadDown} onPointerMove={onCardHeadMove}
             onPointerUp={onCardHeadUp} onPointerCancel={onCardHeadUp}>
          <span className="vt-card-grip" aria-hidden>⠿</span>
          <button className="vt-site-card-restore" onClick={() => setDetailMin(false)}
                  aria-label="Restore details">
            {detail.title}
          </button>
          <button className="vt-icon-btn" aria-label="Close details"
                  onClick={() => { setDetail(null); setDetailMin(false); clearTrail(); stopSatFocusRef.current?.(); }}>
            <X size={15} />
          </button>
        </div>
      )}
      {detail && !detailMin && (
        <div ref={detailCardRef} className="vt-site-card" role="dialog" aria-label={detail.title}>
          <div className="vt-site-card-head" style={{ cursor: "grab", touchAction: "none" }}
               title="Drag to move"
               onPointerDown={onCardHeadDown} onPointerMove={onCardHeadMove}
               onPointerUp={onCardHeadUp} onPointerCancel={onCardHeadUp}>
            <span className="vt-card-grip" aria-hidden>⠿</span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="vt-site-card-title">{detail.title}</div>
              <div className="vt-site-card-cat">{detail.subtitle}</div>
            </div>
            <button className="vt-icon-btn" aria-label="Minimize details"
                    onClick={() => setDetailMin(true)}>
              <Minus size={17} />
            </button>
            <button className="vt-icon-btn" aria-label="Close details"
                    onClick={() => { setDetail(null); clearTrail(); stopSatFocusRef.current?.(); }}>
              <X size={17} />
            </button>
          </div>
          <p className="vt-site-card-body" style={{ whiteSpace: "pre-line" }}>{detail.body}</p>
          {detail.sourceUrl && (
            <a className="vt-site-card-link" href={detail.sourceUrl} target="_blank" rel="noopener noreferrer">
              Source ↗
            </a>
          )}
          {detail.owner && (
            <p className="vt-site-card-trail">Registered: {detail.owner}</p>
          )}
          {detail.timeline && (
            <div>
              <p className="vt-site-card-trail" style={{ fontWeight: 600 }}>Past 7 days within {fmtKm(50)} (own archives):</p>
              {detail.timeline.events.length === 0 && (
                <p className="vt-site-card-trail">No archived alerts, fire detections, or gauge readings.</p>
              )}
              {detail.timeline.events.slice(0, 5).map((ev, i) => (
                <p key={i} className="vt-site-card-trail">
                  {ev.kind === "alert" ? "⚠" : ev.kind === "fire" ? "▲" : "≈"} {ev.label}
                  {ev.severity ? ` · ${ev.severity}` : ""} · {String(ev.t).slice(0, 10)}
                </p>
              ))}
              {(() => {
                const days = Object.keys(detail.timeline.density).sort();
                if (!days.length) return null;
                const a = days.reduce((s, d) => s + (detail.timeline!.density[d].a || 0), 0);
                const v = days.reduce((s, d) => s + (detail.timeline!.density[d].v || 0), 0);
                return (
                  <p className="vt-site-card-trail">
                    Traffic near site: {a.toLocaleString()} aircraft + {v.toLocaleString()} vessel archived points
                    over {days.length} day{days.length > 1 ? "s" : ""}
                  </p>
                );
              })()}
            </div>
          )}
          {detail.dossier && (() => {
            const dos = detail.dossier!;
            const companyNode = dos.graph?.nodes.find((n) => n.type === "company")
              ?? (dos.identity?.type === "company" ? dos.identity : null);
            const edges = dos.graph?.edges || [];
            const insiderEdges = edges.filter((e) => e.type === "insider_of");
            const callsAtEdges = edges.filter((e) => e.type === "calls_at");
            // Defensive `?? []` (not just `dos.graph?.edges`): a malformed or
            // not-yet-shaped response (e.g. a test harness's generic {}
            // fallback for an unfixtured route) must never crash the card.
            const contracts = dos.contracts ?? [];
            const nearestSites = dos.nearest_sites ?? [];
            // LOCATION DOSSIER hazard cross-join (research/location_context_engine.md) —
            // only categories whose layer cache was actually warm (`ready`) render, so
            // a cold cache shows nothing rather than a false "0 nearby" all-clear.
            const hazardCats: Array<{ key: string; label: string; section: DossierHazardSection | undefined }> = [
              { key: "superfund", label: "EPA Superfund (NPL) site", section: dos.hazards?.superfund },
              { key: "water_violators", label: "EPA Clean Water Act chronic violator", section: dos.hazards?.water_violators },
              { key: "pfas", label: "PFAS-detecting water system (EPA UCMR 5)", section: dos.hazards?.pfas },
              { key: "quakes", label: "historical M6+ earthquake", section: dos.hazards?.quakes },
              { key: "nuclear_tests", label: "historical nuclear test", section: dos.hazards?.nuclear_tests },
            ].filter((c) => c.section?.ready && c.section.total_within > 0);
            // flood_zone is a POINT lookup, not a radius list — render only
            // once ready (mirrors the hazardCats ready-gate above); a
            // ready:true zone:null result (outside NFHL's footprint) is
            // still real content worth showing, not withheld like an
            // empty hazardCats section would be.
            const floodZone = dos.hazards?.flood_zone;
            const showFloodZone = Boolean(floodZone?.ready);
            const hasContent = Boolean(companyNode) || insiderEdges.length > 0 || callsAtEdges.length > 0
              || contracts.length > 0 || nearestSites.length > 0 || hazardCats.length > 0 || showFloodZone;
            if (!hasContent) return null;
            return (
              <div>
                {companyNode && (
                  <p className="vt-site-card-trail" style={{ fontWeight: 600 }}>
                    Linked ticker: {companyNode.label} (Everything Graph — RAW, no predictive claim)
                  </p>
                )}
                {insiderEdges.length > 0 && (
                  <p className="vt-site-card-trail">
                    {insiderEdges.length} insider{insiderEdges.length > 1 ? "s" : ""} on file
                    (SEC Form 4, our own 30-day archive)
                  </p>
                )}
                {callsAtEdges.length > 0 && (
                  <p className="vt-site-card-trail">
                    {callsAtEdges.reduce((s, e) => s + (e.attrs?.visit_count || 0), 0)} archived port call(s) —
                    our own AIS archive
                  </p>
                )}
                {contracts.length > 0 && (
                  <div>
                    <p className="vt-site-card-trail" style={{ fontWeight: 600 }}>
                      Related federal contracts (USAspending):
                    </p>
                    {contracts.slice(0, 5).map((c, i) => (
                      <p key={i} className="vt-site-card-trail">
                        {c.r || c.tkr || "recipient n/a"} · ${Math.round(c.amt).toLocaleString()}
                        {c.ag ? ` · ${c.ag}` : ""} · {c.rt}
                      </p>
                    ))}
                    {dos.contracts_capped && (
                      <p className="vt-site-card-trail">…more not shown (capped)</p>
                    )}
                  </div>
                )}
                {nearestSites.length > 0 && (
                  <div>
                    <p className="vt-site-card-trail" style={{ fontWeight: 600 }}>Nearest strategic sites:</p>
                    {nearestSites.map((s) => (
                      <p key={s.id} className="vt-site-card-trail">{s.name} · {fmtKm(s.km)}</p>
                    ))}
                  </div>
                )}
                {hazardCats.length > 0 && (
                  <div>
                    <p className="vt-site-card-trail" style={{ fontWeight: 600 }}>
                      Nearby hazard records ({fmtKm(dos.hazards!.radius_km)} radius — facts only, no risk claim):
                    </p>
                    {hazardCats.map((c) => (
                      <div key={c.key}>
                        <p className="vt-site-card-trail">
                          {c.section!.total_within} {c.label}{c.section!.total_within > 1 ? "s" : ""} nearby
                        </p>
                        {c.section!.hits.slice(0, 3).map((h) => (
                          <p key={h.id} className="vt-site-card-trail" style={{ paddingLeft: 8 }}>
                            {h.label} · {fmtKm(h.km, 1)}
                          </p>
                        ))}
                        {c.section!.capped && (
                          <p className="vt-site-card-trail" style={{ paddingLeft: 8 }}>…more not shown (capped)</p>
                        )}
                      </div>
                    ))}
                  </div>
                )}
                {showFloodZone && (
                  <div>
                    <p className="vt-site-card-trail" style={{ fontWeight: 600 }}>FEMA flood zone at this point:</p>
                    {floodZone!.zone ? (
                      <>
                        <p className="vt-site-card-trail">
                          Zone {floodZone!.zone}{floodZone!.subtype ? ` (${floodZone!.subtype})` : ""}
                          {floodZone!.sfha != null ? ` · ${floodZone!.sfha ? "Special Flood Hazard Area" : "not a Special Flood Hazard Area"}` : ""}
                        </p>
                        {floodZone!.meaning && (
                          <p className="vt-site-card-trail" style={{ paddingLeft: 8 }}>{floodZone!.meaning}</p>
                        )}
                        {floodZone!.base_flood_elevation_ft != null && (
                          <p className="vt-site-card-trail" style={{ paddingLeft: 8 }}>
                            Base flood elevation: {floodZone!.base_flood_elevation_ft} ft
                          </p>
                        )}
                      </>
                    ) : (
                      <p className="vt-site-card-trail">Outside FEMA's mapped NFHL footprint (unstudied, not a low-risk claim).</p>
                    )}
                  </div>
                )}
              </div>
            );
          })()}
          {detail.links && detail.links.length > 0 && (
            <div className="vt-site-card-links">
              {detail.links.map((l) => (
                <a key={l.href} href={l.href} target="_blank" rel="noreferrer">
                  {l.label} ↗
                </a>
              ))}
            </div>
          )}
          {detail.trailNote && (
            <p className="vt-site-card-trail">
              Trail: {detail.trailNote}
              {detail.trailLastT && (
                <span className="vt-trail-freshness" data-testid="trail-freshness" data-tick={freshTick}>
                  {" · "}last position {formatAge(detail.trailLastT)}
                  {" (trail extends live; gaps = coverage/sampling, not necessarily staleness)"}
                </span>
              )}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
