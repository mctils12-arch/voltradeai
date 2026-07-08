import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { Layers as LayersIcon, Info, X, Plane, Ship, MapPin, Satellite, FileText, Zap, TrainFront, Maximize2, Minimize2, Mountain, CloudRain, Thermometer, Wind, Flame, TrendingUp, Share2, Database as DatabaseIcon, Globe as GlobeIcon, Map as FlatMapIcon, MessageSquareText, Moon, ChevronLeft, ChevronRight } from "lucide-react";
// Static CSS import: without maplibre's stylesheet loaded BEFORE the map
// constructs, maplibre mis-measures the container (300px fallback canvas) and
// its controls render unpositioned. The JS stays dynamically imported below.
import "maplibre-gl/dist/maplibre-gl.css";
import {
  registerIcons, classifyAircraft, classifyVessel, velocityEndpoint, iconDataURL,
  AIRCRAFT_ICON, VESSEL_ICON, SITE_ICON, AIRCRAFT_CLASS_LABEL, VESSEL_CLASS_LABEL,
  POWER_FUEL_ICON, POWER_FUEL_COLOR, POWER_FUEL_LABEL, FIRE_CONFIDENCE_COLOR,
} from "@/lib/mapIcons";
import FilingsView from "./filings";
import EarningsView from "./earnings";
import ShortVolView from "./shortvol";
import GraphView from "./graph";
import StreamsView from "./streams";
import GridStressView from "./gridstress";
// W6 ANALYST pane (console charter): lazy chunk — a closed pane loads no
// analyst code at all (zero-cost-when-off spirit) and never polls.
const AnalystPane = lazy(() => import("@/components/AnalystPane"));
import type { AnalystMapCommand } from "@/components/AnalystPane";
import { mmsiFlag } from "@/lib/mmsiFlag";
// ORBITAL program O2 (research/orbital_program.md): live satellites on the
// globe. GP elements are client-fetched from CelesTrak (the browser is NOT
// firewalled from CelesTrak the way Railway is — charter DATA-PATH SPLIT),
// SGP4 runs off-thread in a Web Worker, and the population draws as
// GPU-instanced points. REAL positions only — deep-space objects need SDP4
// and are skipped + COUNTED, never fabricated.
import { SatLayer } from "@/lib/orbital/satLayer";
import { fetchGp, type GpRecord } from "@/lib/orbital/tle";
import type { SatWorkerOutbound } from "@/lib/orbital/satWorker";
// Reliability (BUG 1): single-shot layers (sites, powerplants, boundaries,
// orbital_sats) had no fetch timeout and no retry — one stalled/failed request
// left them spinning or dead until a manual toggle. runResilientLoad adds a hard
// timeout + auto-retry backoff so a transient blip self-heals.
import { runResilientLoad } from "@/lib/resilientLoad";
// worldview_globe.md Phase G2: shared NASA GIBS raster-layer factory. G2a
// (night lights) is the first consumer; G2b-h reuse this same helper.
import { gibsTileUrl, gibsDefaultDate, gibsStepDate, gibsIsLatestAvailable } from "@/lib/gibs";
// Reliability (BUG 4): six hand-rolled layers stacked click/hover listeners
// across toggle cycles. attachLayerInteractions binds them with named handlers
// and returns a detach() the effect cleanup calls — no more stacking.
import { attachLayerInteractions } from "@/lib/mapInteractions";

// Satellite GP element cache (live-tracking stability). CelesTrak's `active`
// group is ~6.6 MB / ~16k objects and CelesTrak RATE-LIMITS repeated pulls, so
// re-fetching on every Satellites toggle failed into a "retrying" loop. Elements
// change only ~every 2h, so cache them for the session and reuse on toggle —
// one fetch per page load, instant re-enable, no rate-limit. Module-scoped so it
// survives the effect's mount/unmount cycles (lost only on a full page reload).
let orbitalGpCache: { at: number; gp: GpRecord[] } | null = null;
const ORBITAL_GP_TTL_MS = 2 * 60 * 60_000; // 2h — CelesTrak's GP refresh cadence
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
  kind: "site" | "aircraft" | "vessel" | "powerplant" | "train" | "fire" | "gauge" | "alert";
  title: string;
  subtitle: string;
  body: string;
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
const ALL_OFF = typeof window !== "undefined" && window.sessionStorage?.getItem("vt-layers-all-off") === "1";
const DEFAULT_ON: Record<string, boolean> = ALL_OFF
  ? { imagery: true }
  : { imagery: true, aircraft: true, sites: true, insider: true, earnings: true, shortvol: true, powerplants: true, trains: true, shadowstats: true, portdwell: true, graph: true };

// Layer panel v2 (2026-07-04): with 7+ layers the flat list stopped scaling —
// collapsible groups keep the panel scannable as layers keep arriving.
const PANEL_GROUPS = [
  { id: "base", label: "Base" },
  { id: "live", label: "Live tracking" },
  { id: "facilities", label: "Facilities" },
  { id: "environmental", label: "Environmental" },
  { id: "filings", label: "Filings & flows" },
  { id: "graph", label: "Everything Graph" },
  { id: "signals", label: "Signals — coming soon" },
] as const;
const LAYER_GROUP: Record<string, string> = {
  imagery: "base", terrain: "base", weather: "base",
  weather_temp: "base", weather_wind: "base", boundaries: "base",
  aircraft: "live", vessels: "live", trains: "live",
  sites: "facilities", powerplants: "facilities",
  fires: "environmental", surfacewater: "environmental", forest: "environmental",
  nightlights: "environmental",
  rivergauges: "environmental",
  alerts: "environmental",
  insider: "filings", earnings: "filings", shortvol: "filings", shadowstats: "filings", portdwell: "filings",
  graph: "graph",
  powergrid: "facilities",
  orbital_sats: "live",
};
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
function LegendIcon({ icon, color, label }: { icon: string; color: string; label: string }) {
  return (
    <span className="vt-legend-item" data-vt-icon={icon}>
      <img src={iconDataURL(icon, color)} width={15} height={15} alt="" aria-hidden />
      {label}
    </span>
  );
}

export default function DataMapPage() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<any>(null);
  const glRef = useRef<any>(null);
  const sinceRef = useRef<Record<string, string>>({});
  // ORBITAL O2: the satellite worker + GPU layer live across renders so the
  // enable/disable effect can tear both down cleanly (zero-cost-when-off).
  const satWorkerRef = useRef<Worker | null>(null);
  const satLayerRef = useRef<SatLayer | null>(null);
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
  const [panelOpen, setPanelOpen] = useState<boolean>(() =>
    typeof window !== "undefined" ? window.innerWidth >= 768 : true);
  // Legend v3: collapsible as one unit so it never fights the panel for
  // space — open on desktop, collapsed on phone by default.
  const [legendOpen, setLegendOpen] = useState<boolean>(() =>
    typeof window !== "undefined" ? window.innerWidth >= 768 : true);
  const [showRawInfo, setShowRawInfo] = useState(false);
  const [detail, setDetail] = useState<Detail | null>(null);
  // Full filings view (#/data/filings) — overlay on top of the map page so
  // the map stays mounted; hash-driven so it deep-links and back-buttons.
  const [filingsOpen, setFilingsOpen] = useState(() => window.location.hash === "#/data/filings");
  // Full earnings-language view (#/data/earnings) — same overlay pattern.
  const [earningsOpen, setEarningsOpen] = useState(() => window.location.hash === "#/data/earnings");
  // Full FINRA short-volume view (#/data/short-volume) — same overlay pattern.
  const [shortvolOpen, setShortvolOpen] = useState(() => window.location.hash === "#/data/short-volume");
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
  // ── weather-upgrade (2026-07-04): registry-native FIELD layer controls ──
  // Field layers (registry flag `field: true`) get a per-layer opacity
  // slider; default 60% so the basemap + live layers stay visible beneath
  // (directive: the field is context, never a curtain).
  const FIELD_MAP_LAYER: Record<string, string> = {
    weather: "weather-radar", weather_temp: "wx-temp_new", weather_wind: "wx-wind_new",
    surfacewater: "gsw-occurrence", forest: "jrc-forest", nightlights: "gibs-nightlights",
  };
  const [fieldOpacity, setFieldOpacityState] = useState<Record<string, number>>(() => {
    try { return JSON.parse(sessionStorage.getItem("vt-field-opacity") || "{}"); } catch { return {}; }
  });
  const opacityOf = (id: string) => fieldOpacity[id] ?? 60;
  const setFieldOpacity = (id: string, v: number) => {
    setFieldOpacityState((s) => {
      const next = { ...s, [id]: v };
      try { sessionStorage.setItem("vt-field-opacity", JSON.stringify(next)); } catch {}
      return next;
    });
    try { mapRef.current?.setPaintProperty(FIELD_MAP_LAYER[id], "raster-opacity", v / 100); } catch {}
  };
  // Wind vectors + temperature labels — sampled point grid (HONEST: OWM
  // tiles carry no vector data; numbers come from point samples, arrows
  // never denser than the sampling — the note shows real spacing).
  const [windArrows, setWindArrows] = useState(true);
  const [tempLabels, setTempLabels] = useState(false);
  const [tempUnitF, setTempUnitF] = useState(true);
  const [wxGrid, setWxGrid] = useState<any>(null);
  useEffect(() => {
    const onHash = () => {
      setFilingsOpen(window.location.hash === "#/data/filings");
      setEarningsOpen(window.location.hash === "#/data/earnings");
      setShortvolOpen(window.location.hash === "#/data/short-volume");
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
      setDetail(null);
      clearTrail();
      setShowRawInfo(false);
      setAnalystOpen(false); // DESIGN.md: Escape closes panels/popovers
      if (window.innerWidth < 768) setPanelOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const clearTrail = () => {
    const map = mapRef.current;
    if (!map) return;
    try {
      if (map.getLayer("trail-line")) map.removeLayer("trail-line");
      if (map.getSource("trail")) map.removeSource("trail");
    } catch {}
  };

  /** Fetch the archived track and paint/refresh the trail. On refresh the
   *  existing geojson source is UPDATED via setData (no layer churn).
   *  Returns the note + newest position time so the card can show live
   *  freshness. ([REPAIR 2026-07-05]: the trail was fetched ONCE at
   *  selection and never again — a static snapshot while the aircraft
   *  kept moving; see the refresh effect below.) */
  const showTrail = async (kind: "aircraft" | "vessels" | "trains", id: string):
      Promise<{ note: string; lastT?: number }> => {
    const map = mapRef.current;
    if (!map) return { note: "" };
    try {
      const r = await fetch(`/api/data/track/${kind}/${encodeURIComponent(id)}`);
      const d = await r.json();
      const raw = (d.points || []) as Array<{ lo: number; la: number; t?: number }>;
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
      (window as any).__vtTrailLen = pts.length; // harness ratchet reads this
      return {
        note: d.note || (pts.length ? `${pts.length} archived positions (our own feed history)` : ""),
        lastT,
      };
    } catch { return { note: "trail unavailable" }; }
  };

  // Live trail refresh — while a track-bearing card is open, re-pull the
  // archived track every 30s so the trail extends as new positions land
  // (the archive tick appends every few minutes; 30s keeps the popup
  // honest without hammering the tiny track endpoint).
  const detailTrailId = detail?.trailId;
  const detailTrailKind = detail?.trailKind;
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

  // ── terrain hillshade (RAW; Mapterhorn terrarium DEM — geospatial Tier-1(a),
  // licensing register 2026-07-04: commercial-OK, © Mapterhorn attribution via
  // TileJSON. Also wires the raster-dem source R4's 3D terrain will reuse.
  // Default OFF: the imagery base already carries visual relief; hillshade is
  // an opt-in accent inserted UNDER all marker layers.) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.terrain) {
      try { map.setTerrain(null); } catch {}
      try {
        if (map.getLayer("terrain-hillshade")) map.removeLayer("terrain-hillshade");
        if (map.getSource("terrain-dem")) map.removeSource("terrain-dem");
      } catch {}
      setStatus("terrain", "off");
      return;
    }
    try {
      if (!map.getSource("terrain-dem")) {
        map.addSource("terrain-dem", {
          type: "raster-dem",
          url: "https://tiles.mapterhorn.com/tilejson.json",
          encoding: "terrarium",
        } as any);
      }
      // REAL 3D relief (worldview-globe upgrade): deform the base mesh from the
      // same DEM so mountains rise and valleys sink — a physical globe, not a
      // flat sphere — especially with pitch. exaggeration 1.3 reads premium
      // without cartoonish spikes. Hillshade below stays for shading detail on
      // the raised mesh. Guarded: if a device/projection can't do terrain the
      // catch keeps the base map alive (degrade, never break).
      map.setTerrain({ source: "terrain-dem", exaggeration: 1.3 } as any);
      if (!map.getLayer("terrain-hillshade")) {
        // insert beneath the lowest data layer (symbol/circle/line) so
        // shading never covers markers or velocity vectors
        const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle", "line"].includes(l.type));
        map.addLayer({
          id: "terrain-hillshade", type: "hillshade", source: "terrain-dem",
          paint: {
            "hillshade-exaggeration": 0.45,
            "hillshade-shadow-color": "rgba(5,10,19,0.9)",
            "hillshade-highlight-color": "rgba(238,243,251,0.25)",
            "hillshade-accent-color": "rgba(77,159,255,0.15)",
          },
        } as any, firstMarker?.id);
      }
      setStatus("terrain", "active", undefined, "3D relief + hillshade — Copernicus GLO-30 + national DEMs (© Mapterhorn)");
    } catch {
      setStatus("terrain", "error");
    }
  }, [enabled.terrain, mapReady, setStatus]);

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

  // ── satellites (RAW; ORBITAL program O2 — live GP elements client-fetched
  // from CelesTrak, SGP4 propagated off-thread in a Web Worker, drawn as
  // GPU-instanced points on the globe with LEO/MEO/GEO altitude shells. HEAVY
  // + off by default → zero-cost-when-off: the worker + layer only exist while
  // the toggle is on. REAL positions only — deep-space objects (GEO comms,
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
    };

    if (!enabled["orbital_sats"]) {
      teardown();
      setStatus("orbital_sats", "off");
      return;
    }

    setStatus("orbital_sats", "loading", undefined, "fetching orbital elements (CelesTrak)…");

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
          gp = await fetchGp("active", (url: string) => fetch(url, { signal }) as any);
          if (gp.length) orbitalGpCache = { at: Date.now(), gp }; // cache for the session
        }
        if (signal.aborted) return;
        if (!gp.length) throw new Error("no orbital elements returned");
        if (satWorkerRef.current) return; // already initialized — don't double-add

        const layer = new SatLayer({ id: "orbital_sats" });
        satLayerRef.current = layer;
        map.addLayer(layer);

        const worker = new Worker(
          new URL("../lib/orbital/satWorker.ts", import.meta.url),
          { type: "module" },
        );
        satWorkerRef.current = worker;
        worker.onmessage = (ev: MessageEvent<SatWorkerOutbound>) => {
          const m = ev.data;
          if (m.type === "positions") {
            satLayerRef.current?.updatePositions(new Float32Array(m.buf), {
              shown: m.shown,
              deepSpaceSkipped: m.deepSpaceSkipped,
              invalidSkipped: m.invalidSkipped,
            });
            const skipped = m.deepSpaceSkipped + m.invalidSkipped;
            setStatus("orbital_sats", "active", m.shown,
              `${m.shown.toLocaleString()} live · ${skipped.toLocaleString()} not rendered (deep-space, needs SDP4)`);
          }
        };
        worker.postMessage({ type: "init", gp });
        worker.postMessage({ type: "start", hz: 1 });
      },
      (failures) => setStatus("orbital_sats", "error", undefined,
        failures === 0 ? "could not reach CelesTrak — retrying automatically…" : "still retrying automatically…"),
      // The ~6.6 MB `active` fetch needs headroom on slow links (default 15s was
      // too tight and aborted mid-download → the "retrying" the user reported).
      { timeoutMs: 45_000 },
    );

    return () => { stopLoad(); teardown(); };
  }, [enabled["orbital_sats"], mapReady, setStatus]);

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

  // ── power grid (RAW; OSM power features © OpenStreetMap contributors,
  // ODbL — DATACORE MAXIMUS Phase 2 TX pilot. Single 16MB PMTiles on our
  // origin (range requests, zero-cost-when-off: the file is fetched only
  // when the layer is on). VOLTAGE HONESTY: lines whose voltage tag is
  // missing or unparseable (multi-value "138000;69000") render as a
  // distinct dashed class — never hidden. Zoom gates per the grid build
  // order keep low-zoom vertex counts phone-safe.) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    const IDS = ["powergrid-hv", "powergrid-mv", "powergrid-low", "powergrid-unknown", "powergrid-substation", "powergrid-plant"];
    if (!enabled.powergrid) {
      try {
        IDS.forEach((id) => { if (map.getLayer(id)) map.removeLayer(id); });
        if (map.getSource("powergrid")) map.removeSource("powergrid");
      } catch {}
      setStatus("powergrid", "off");
      return;
    }
    try {
      setStatus("powergrid", "loading");
      if (!map.getSource("powergrid")) {
        map.addSource("powergrid", {
          type: "vector",
          url: `pmtiles://${window.location.origin}/tiles/power_tx.pmtiles`,
          attribution: "© OpenStreetMap contributors, ODbL",
        } as any);
      }
      // voltage as a number; -1 = missing/unparseable (to-number fallback)
      const V = ["to-number", ["get", "voltage"], -1] as any;
      const isLine = ["match", ["get", "power"], ["line", "minor_line", "cable"], true, false] as any;
      const firstMarker = (map.getStyle().layers || []).find((l: any) => ["symbol", "circle"].includes(l.type));
      const add = (def: any) => { if (!map.getLayer(def.id)) map.addLayer(def, firstMarker?.id); };
      add({ id: "powergrid-substation", type: "fill", source: "powergrid", "source-layer": "power",
            minzoom: 9, filter: ["==", ["get", "power"], "substation"],
            paint: { "fill-color": "rgba(250,204,21,0.14)", "fill-outline-color": "rgba(250,204,21,0.6)" } });
      add({ id: "powergrid-plant", type: "fill", source: "powergrid", "source-layer": "power",
            minzoom: 9, filter: ["==", ["get", "power"], "plant"],
            paint: { "fill-color": "rgba(74,222,128,0.10)", "fill-outline-color": "rgba(74,222,128,0.5)" } });
      add({ id: "powergrid-unknown", type: "line", source: "powergrid", "source-layer": "power",
            minzoom: 8, filter: ["all", isLine, ["<", V, 0]],
            paint: { "line-color": "rgba(216,180,254,0.55)", "line-width": 0.9, "line-dasharray": [2, 2] } });
      add({ id: "powergrid-low", type: "line", source: "powergrid", "source-layer": "power",
            minzoom: 11, filter: ["all", isLine, [">=", V, 0], ["<", V, 100000]],
            paint: { "line-color": "rgba(148,163,184,0.55)", "line-width": 0.8 } });
      add({ id: "powergrid-mv", type: "line", source: "powergrid", "source-layer": "power",
            minzoom: 6, filter: ["all", isLine, [">=", V, 100000], ["<", V, 230000]],
            paint: { "line-color": "rgba(56,189,248,0.75)",
                     "line-width": ["interpolate", ["linear"], ["zoom"], 6, 0.8, 12, 2] } });
      add({ id: "powergrid-hv", type: "line", source: "powergrid", "source-layer": "power",
            filter: ["all", isLine, [">=", V, 230000]],
            paint: { "line-color": "rgba(250,204,21,0.9)",
                     "line-width": ["interpolate", ["linear"], ["zoom"], 3, 1, 12, 3] } });
      setStatus("powergrid", "active", undefined,
        "TEXAS PILOT — OSM community data (ODbL): voltage-classed; dashed = voltage untagged (never hidden); no CEII/underground detail");
    } catch {
      setStatus("powergrid", "error");
    }
  }, [enabled.powergrid, mapReady, setStatus]);

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
    /** Low-zoom render decimation ([REPAIR 2026-07-05] perf 3/3): below
     *  splitZoom draw only features with rank < keepFraction (rank is a
     *  stable per-feature hash set by toFeatures). RENDER-side only — the
     *  source always holds every feature (harness data-richness guard
     *  pins >=9500 in source at 1440), and zooming past splitZoom shows
     *  everything. At the default z3.6 view, 10k overlapping icons were
     *  pure overdraw. */
    lowZoom?: { splitZoom: number; keepFraction: number };
  }) => {
    const map = mapRef.current;
    const { id } = opts;
    let stop = false;
    const srcId = id, layerId = `${id}-sym`, loLayerId = `${id}-sym-lo`, vecSrc = `${id}-vec`, vecLayer = `${id}-veclines`;

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

    const load = async () => {
      // Hidden-tab gate ([REPAIR 2026-07-05] map perf): a backgrounded /data
      // tab kept polling aircraft 4x/min. Skip while hidden; the
      // visibilitychange listener below refreshes immediately on return
      // (stale-with-timestamp already covers the gap honestly).
      if (document.hidden) return;
      try {
        const b = map.getBounds();
        const since = sinceRef.current[id] || "";
        const q = `lamin=${b.getSouth().toFixed(2)}&lamax=${b.getNorth().toFixed(2)}&lomin=${b.getWest().toFixed(2)}&lomax=${b.getEast().toFixed(2)}${since ? `&since=${since}` : ""}`;
        const r = await fetch(`/api/data/${id}?${q}`);
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

        // Honest feed states (DESIGN.md): partial coverage + staleness shown.
        let note: string | undefined;
        if (d.coverage === "partial" && d.coverage_note) note = d.coverage_note;
        if (d.stale) note = `stale — data as of ${new Date(d.stale_at || Date.now()).toLocaleTimeString()}`;

        const features = opts.toFeatures(d);
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
        if (opts.toVectors) {
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
        }
        setStatus(id, "active", d.count ?? features.length, note);
      } catch {
        if (!stop) setStatus(id, "error", undefined, "feed error — backing off, retrying");
      }
    };
    load();
    const iv = window.setInterval(load, opts.intervalMs);
    // Trailing debounce ([REPAIR 2026-07-05] map perf): bare moveend fired a
    // full fetch + 10k-feature rebuild on EVERY camera settle — each wheel
    // step during a zoom was a fetch. Same 400ms pattern the wx-grid effect
    // already used.
    let moveDebounce: number | undefined;
    const onMove = () => {
      window.clearTimeout(moveDebounce);
      moveDebounce = window.setTimeout(load, 400);
    };
    map.on("moveend", onMove);
    const onVisible = () => { if (!document.hidden) load(); };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      teardown();
      window.clearInterval(iv);
      window.clearTimeout(moveDebounce);
      document.removeEventListener("visibilitychange", onVisible);
      try { map.off("moveend", onMove); } catch {}
    };
  }, [setStatus]);

  // ── live aircraft (RAW; WebGL symbols, heading-rotated, class icons) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.aircraft) {
      try {
        if (map.getLayer("aircraft-sym")) map.removeLayer("aircraft-sym");
        if (map.getLayer("aircraft-sym-lo")) map.removeLayer("aircraft-sym-lo");
        if (map.getLayer("aircraft-veclines")) map.removeLayer("aircraft-veclines");
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
    return wireLivePoints({
      id: "aircraft",
      intervalMs: 15_000,
      lowZoom: { splitZoom: 4.5, keepFraction: 0.35 },
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
      onClick: async (p) => {
        const cls = AIRCRAFT_CLASS_LABEL[(p.cls || "unknown") as keyof typeof AIRCRAFT_CLASS_LABEL] || "Aircraft";
        const alt = p.ground === true || p.ground === "true" ? "on ground" : (p.alt != null ? `${p.alt} m` : "alt unknown");
        setDetail({
          kind: "aircraft",
          title: `✈ ${p.callsign}`,
          subtitle: `${cls}${p.type ? ` · ${p.type}` : ""} · ${p.country || "—"}`,
          body: `${alt}${p.kts ? ` · ${p.kts} kts` : ""} · hdg ${Math.round(p.heading || 0)}°\n` +
                `Route/flight-plan data unavailable — filed plans are a paid source (wishlist); ` +
                `trail below is our own archived feed history.`,
          trailId: p.icao24, trailKind: "aircraft",
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
        const { note, lastT } = await showTrail("aircraft", p.icao24);
        setDetail(prev => prev && prev.trailId === p.icao24 ? { ...prev, trailNote: note, trailLastT: lastT } : prev);
      },
    });
  }, [enabled.aircraft, mapReady, wireLivePoints, setStatus]);

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
      onClick: async (p) => {
        const cls = VESSEL_CLASS_LABEL[(p.cls || "other") as keyof typeof VESSEL_CLASS_LABEL] || "Vessel";
        const flag = mmsiFlag(p.mmsi);
        setDetail({
          kind: "vessel",
          title: `⚓ ${p.name}`,
          subtitle: `${cls} · MMSI ${p.mmsi}${flag ? ` · ${flag}` : ""}`,
          body: `${p.kts != null ? `${p.kts} kts · ` : ""}hdg ${Math.round(p.heading || 0)}°` +
                `${p.destination ? `\nDestination (AIS-broadcast): ${p.destination}` : "\nDestination: not broadcast"}`,
          trailId: p.mmsi, trailKind: "vessels",
          links: [
            { label: "MarineTraffic", href: `https://www.marinetraffic.com/en/ais/details/ships/mmsi:${p.mmsi}` },
            { label: "VesselFinder", href: `https://www.vesselfinder.com/vessels/details/${p.mmsi}` },
          ],
        });
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
          setDetail({
            kind: "site",
            title: f.properties.name,
            subtitle: `${f.properties.category} · ${f.properties.operator}`,
            body: f.properties.relevance,
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
            features: d.plants.map(([name, mw, fuel, owner, lat, lon, verified]: [string, number, string, string, number, number, number]) => ({
              type: "Feature",
              geometry: { type: "Point", coordinates: [lon, lat] },
              properties: {
                name, mw, fuel, owner, verified: verified === 1,
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
          setDetail({
            kind: "powerplant",
            title: p.name,
            subtitle: `${POWER_FUEL_LABEL[p.fuel] || p.fuel} · ${Number(p.mw).toLocaleString()} MW`,
            body: `${p.owner ? `Operator: ${p.owner}\n` : ""}` +
                  `${p.verified === true || p.verified === "true"
                     ? "Position imagery-verified.\n"
                     : "Position approximate (registry-reported — GPPD/EIA-860).\n"}` +
                  `Static reference data — WRI GPPD v1.3.0 (CC BY 4.0) + EIA-860.`,
          });
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
            setDetail({
              kind: "train",
              title: `${p.label}`,
              subtitle: `${p.country === "FI" ? "Finland · Digitraffic (CC BY 4.0)" : "Norway · Entur (NLOD)"}`,
              body: `${p.speed != null && p.speed !== "null" ? `Speed: ${p.speed} km/h\n` : ""}Live passenger-rail position, shown as received.`,
              trailId: p.id, trailKind: "trains",
            });
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
            setDetail({
              kind: "fire",
              title: "Active fire detection",
              subtitle: `${p.confidence} confidence · ${p.satellite}${p.daynight ? ` · ${p.daynight === "D" ? "day" : "night"}` : ""}`,
              body: `Detected ${p.acq_date} ${String(p.acq_time).padStart(4, "0")} UTC` +
                    `${p.brightness != null ? `\nBrightness: ${Math.round(p.brightness)} K` : ""}` +
                    `${p.frp != null ? `\nFire radiative power: ${p.frp} MW` : ""}\n\n` +
                    `NASA FIRMS/LANCE — for informational purposes only, NOT for safety-of-life use.`,
              links: [{ label: "NASA FIRMS map", href: "https://firms.modaps.eosdis.nasa.gov/map/" }],
            });
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
            setDetail({
              kind: "gauge",
              title: `≈ ${p.name || p.site}`,
              subtitle: `USGS ${p.site}`,
              body: `${p.v} ${unit}\nObserved ${p.d}\n` +
                    `${p.q === "P" ? "PROVISIONAL — subject to revision" : p.q === "A" ? "Approved reading" : ""}\n\n` +
                    `Data courtesy U.S. Geological Survey. Raw readings only — ` +
                    `low-water interpretation is validation-gated.`,
              links: [{ label: "USGS monitoring page", href: `https://waterdata.usgs.gov/monitoring-location/${p.site}/` }],
            });
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
            setDetail({
              kind: "alert",
              title: `⚠ ${p.event}`,
              subtitle: `NWS · ${p.severity} severity`,
              body: `${p.area || ""}\n${p.ends ? `Ends ${p.ends}` : ""}\n\n` +
                    `Official National Weather Service alert, displayed as published. ` +
                    `Not for safety-of-life use — see weather.gov for authoritative guidance.`,
              links: [{ label: "weather.gov alerts", href: "https://www.weather.gov/alerts" }],
            });
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
    return () => { stop = true; window.clearInterval(iv); };
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
    id === "weather" ? <CloudRain size={15} /> :
    id === "weather_temp" ? <Thermometer size={15} /> :
    id === "weather_wind" ? <Wind size={15} /> :
    id === "aircraft" ? <Plane size={15} /> :
    id === "vessels" ? <Ship size={15} /> :
    id === "sites" ? <MapPin size={15} /> :
    id === "powerplants" ? <Zap size={15} /> :
    id === "trains" ? <TrainFront size={15} /> :
    id === "fires" ? <Flame size={15} /> :
    id === "nightlights" ? <Moon size={15} /> :
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
      const unit = l.id === "sites" ? "sites" : l.id === "insider" ? "filings" : l.id === "earnings" ? "releases" : l.id === "shortvol" ? "symbols" : l.id === "powerplants" ? "plants" : l.id === "trains" ? "trains" : l.id === "shadowstats" ? "gap events" : l.id === "portdwell" ? "port calls" : l.id === "fires" ? "detections" : l.id === "graph" ? "entities" : l.id;
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
      {graphOpen && (
        <GraphView onBack={() => { window.location.hash = "#/data"; setGraphOpen(false); }} />
      )}
      {streamsOpen && (
        <StreamsView onBack={() => { window.location.hash = "#/data"; setStreamsOpen(false); }} />
      )}
      {gridStressOpen && (
        <GridStressView onBack={() => { window.location.hash = "#/data"; setGridStressOpen(false); }} />
      )}

      {/* Phase 3a imagery capture-date chip (DESIGN.md imagery-honesty
          rule: display dates where available; unknown states stay loud) */}
      {enabled.imagery && (
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
      {/* Style presets (worldview-globe G1) — real-first geographic looks,
          bottom-center segmented control. No tactical filters. */}
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

      <div ref={mapContainer} className="vt-map-canvas" />

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
            {/* LEGEND v3 (legend directive 2026-07-04): symbol entries render
                the SAME registry shapes the map draws (iconDataURL — one
                shared icon source; DESIGN.md legend rule). Sections mirror
                the panel groups, entries appear ONLY while their layer is on,
                and the block collapses as one unit so it never fights the
                panel for space. Color-only chips are color MEANINGS (altitude
                tints, raster ramps), not symbols — chips by design. */}
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
                  {(enabled.sites || enabled.powerplants) && (
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
                        {enabled.powerplants && Object.keys(POWER_FUEL_ICON).map((fuel) => (
                          <LegendIcon key={fuel} icon={POWER_FUEL_ICON[fuel]}
                                      color={POWER_FUEL_COLOR[fuel]}
                                      label={`${POWER_FUEL_LABEL[fuel]} Plant`} />
                        ))}
                      </div>
                    </div>
                  )}
                  {(enabled.fires || enabled.surfacewater || enabled.forest || enabled.nightlights || enabled.rivergauges || enabled.alerts) && (
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
                      <div className="vt-legend-items">
                        <span className="vt-legend-chip"><i style={{ background: "#4d9fff" }} /> LEO satellite</span>
                        <span className="vt-legend-chip"><i style={{ background: "#ffb840" }} /> MEO satellite</span>
                        <span className="vt-legend-chip"><i style={{ background: "#d973ff" }} /> GEO satellite</span>
                        <span className="vt-legend-note">live SGP4 · deep-space (GEO/MEO nav) needs SDP4 — counted, not drawn</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Detail card — side card on desktop, bottom sheet on phone */}
      {detail && (
        <div className="vt-site-card" role="dialog" aria-label={detail.title}>
          <div className="vt-site-card-head">
            <div>
              <div className="vt-site-card-title">{detail.title}</div>
              <div className="vt-site-card-cat">{detail.subtitle}</div>
            </div>
            <button className="vt-icon-btn" aria-label="Close details"
                    onClick={() => { setDetail(null); clearTrail(); }}>
              <X size={17} />
            </button>
          </div>
          <p className="vt-site-card-body" style={{ whiteSpace: "pre-line" }}>{detail.body}</p>
          {detail.owner && (
            <p className="vt-site-card-trail">Registered: {detail.owner}</p>
          )}
          {detail.timeline && (
            <div>
              <p className="vt-site-card-trail" style={{ fontWeight: 600 }}>Past 7 days within 50 km (own archives):</p>
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
