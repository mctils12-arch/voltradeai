import { useCallback, useEffect, useRef, useState } from "react";
import { Layers as LayersIcon, Info, X, Plane, Ship, MapPin, Satellite, FileText, Zap, TrainFront, Maximize2, Minimize2 } from "lucide-react";
// Static CSS import: without maplibre's stylesheet loaded BEFORE the map
// constructs, maplibre mis-measures the container (300px fallback canvas) and
// its controls render unpositioned. The JS stays dynamically imported below.
import "maplibre-gl/dist/maplibre-gl.css";
import {
  registerIcons, classifyAircraft, classifyVessel, velocityEndpoint,
  AIRCRAFT_ICON, VESSEL_ICON, SITE_ICON, AIRCRAFT_CLASS_LABEL, VESSEL_CLASS_LABEL,
  POWER_FUEL_ICON, POWER_FUEL_COLOR, POWER_FUEL_LABEL,
} from "@/lib/mapIcons";
import FilingsView from "./filings";
import { mmsiFlag } from "@/lib/mmsiFlag";

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
  status: "live" | "awaiting_key" | "planned";
  source: string;
  description: string;
}

type RuntimeStatus = "off" | "loading" | "active" | "error" | "awaiting_key";

interface Detail {
  kind: "site" | "aircraft" | "vessel" | "powerplant" | "train";
  title: string;
  subtitle: string;
  body: string;
  trailId?: string;      // archive id for the trail (aircraft icao24 / mmsi)
  trailKind?: "aircraft" | "vessels" | "trains";
  trailNote?: string;
  /** External profile/photo pages — LINK OUT only, never embedded
   *  (third-party photo copyright). */
  links?: { label: string; href: string }[];
}

const IMAGERY_TILES =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const IMAGERY_ATTRIB = "© Esri, Maxar, Earthstar Geographics";

const DEFAULT_ON: Record<string, boolean> = { imagery: true, aircraft: true, sites: true, insider: true, powerplants: true, trains: true, shadowstats: true, portdwell: true };

// Layer panel v2 (2026-07-04): with 7+ layers the flat list stopped scaling —
// collapsible groups keep the panel scannable as layers keep arriving.
const PANEL_GROUPS = [
  { id: "base", label: "Base" },
  { id: "live", label: "Live tracking" },
  { id: "facilities", label: "Facilities" },
  { id: "filings", label: "Filings & flows" },
  { id: "signals", label: "Signals — coming soon" },
] as const;
const LAYER_GROUP: Record<string, string> = {
  imagery: "base",
  aircraft: "live", vessels: "live", trains: "live",
  sites: "facilities", powerplants: "facilities",
  insider: "filings", shadowstats: "filings", portdwell: "filings",
};
const groupOf = (l: LayerMeta): string =>
  l.kind === "signal" || l.status === "planned" ? "signals" : LAYER_GROUP[l.id] || "live";

// altitude → tint for aircraft icons (SDF icon-color)
const ALT_COLOR: any = ["case",
  ["get", "ground"], "#6680a0",
  ["<", ["coalesce", ["get", "alt"], 99999], 3000], "#fbb24c",
  "#4d9fff"];

const VESSEL_COLOR: Record<string, string> = {
  tanker: "#fbb24c", cargo: "#4ade80", passenger: "#c084fc",
  fishing: "#7cc4ff", tug: "#b3c2d8", other: "#4ade80",
};

export default function DataMapPage() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<any>(null);
  const glRef = useRef<any>(null);
  const sinceRef = useRef<Record<string, string>>({});
  const [layers, setLayers] = useState<LayerMeta[]>([]);
  const [enabled, setEnabled] = useState<Record<string, boolean>>(DEFAULT_ON);
  const [runtime, setRuntime] = useState<Record<string, { status: RuntimeStatus; count?: number; note?: string }>>({});
  const [mapReady, setMapReady] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);
  const [panelOpen, setPanelOpen] = useState<boolean>(() =>
    typeof window !== "undefined" ? window.innerWidth >= 768 : true);
  const [showRawInfo, setShowRawInfo] = useState(false);
  const [detail, setDetail] = useState<Detail | null>(null);
  // Full filings view (#/data/filings) — overlay on top of the map page so
  // the map stays mounted; hash-driven so it deep-links and back-buttons.
  const [filingsOpen, setFilingsOpen] = useState(() => window.location.hash === "#/data/filings");
  // v2.3: groups beyond the first fold start collapsed — the panel stays
  // scannable and everything below is one visible tap away.
  const [groupCollapsed, setGroupCollapsed] = useState<Record<string, boolean>>({
    facilities: true, filings: true, signals: true,
  });
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
  const [descOpen, setDescOpen] = useState<Record<string, boolean>>({});
  useEffect(() => {
    const onHash = () => setFilingsOpen(window.location.hash === "#/data/filings");
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const setStatus = useCallback((id: string, status: RuntimeStatus, count?: number, note?: string) => {
    setRuntime(s => ({ ...s, [id]: { status, count, note } }));
  }, []);

  // Layer registry (datacore boundary)
  useEffect(() => {
    fetch("/api/data/layers")
      .then(r => r.json())
      .then(d => setLayers(Array.isArray(d.layers) ? d.layers : []))
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
        const map = new maplibregl.Map({
          container: mapContainer.current,
          style: {
            version: 8,
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
        map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
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

  const showTrail = async (kind: "aircraft" | "vessels" | "trains", id: string) => {
    const map = mapRef.current;
    if (!map) return "";
    try {
      const r = await fetch(`/api/data/track/${kind}/${encodeURIComponent(id)}`);
      const d = await r.json();
      const pts = (d.points || []).map((p: any) => [p.lo, p.la]);
      clearTrail();
      if (pts.length >= 2) {
        map.addSource("trail", { type: "geojson", data: {
          type: "Feature", geometry: { type: "LineString", coordinates: pts }, properties: {},
        } as any });
        map.addLayer({
          id: "trail-line", type: "line", source: "trail",
          paint: { "line-color": "#7cc4ff", "line-width": 2, "line-opacity": 0.8, "line-dasharray": [1, 1.5] },
        });
      }
      return d.note || (pts.length ? `${pts.length} archived positions (our own feed history)` : "");
    } catch { return "trail unavailable"; }
  };

  // ── imagery toggle ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    try { map.setLayoutProperty("imagery", "visibility", enabled.imagery ? "visible" : "none"); } catch {}
    // IMAGERY METADATA honesty (DESIGN.md 2026-07-04): never imply
    // freshness — Esri base tiles expose no capture date, so say so.
    if (enabled.imagery) setStatus("imagery", "active", undefined, "capture date unavailable (Esri base tiles)");
    else setStatus("imagery", "off");
  }, [enabled.imagery, mapReady, setStatus]);

  // ── generic live-points wiring (aircraft + vessels share the machinery) ──
  const wireLivePoints = useCallback((opts: {
    id: "aircraft" | "vessels";
    intervalMs: number;
    toFeatures: (d: any) => any[];
    toVectors?: (d: any) => any[];
    onClick: (props: any, lngLat: any) => void;
    iconLayout: any;
    iconPaint: any;
  }) => {
    const map = mapRef.current;
    const { id } = opts;
    let stop = false;
    const srcId = id, layerId = `${id}-sym`, vecSrc = `${id}-vec`, vecLayer = `${id}-veclines`;

    const teardown = () => {
      stop = true;
      try {
        if (map.getLayer(layerId)) map.removeLayer(layerId);
        if (map.getLayer(vecLayer)) map.removeLayer(vecLayer);
        if (map.getSource(srcId)) map.removeSource(srcId);
        if (map.getSource(vecSrc)) map.removeSource(vecSrc);
      } catch {}
    };

    const load = async () => {
      try {
        const b = map.getBounds();
        const since = sinceRef.current[id] || "";
        const q = `lamin=${b.getSouth().toFixed(2)}&lamax=${b.getNorth().toFixed(2)}&lomin=${b.getWest().toFixed(2)}&lomax=${b.getEast().toFixed(2)}${since ? `&since=${since}` : ""}`;
        const r = await fetch(`/api/data/${id}?${q}`);
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (stop) return;
        if (d.enabled === false) { setStatus(id, "awaiting_key"); return; }
        if (d.unchanged) { return; } // delta: nothing new to draw
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
          map.addLayer({ id: layerId, type: "symbol", source: srcId, layout: opts.iconLayout, paint: opts.iconPaint });
          map.on("click", layerId, (e: any) => {
            const f = e.features?.[0];
            if (f) opts.onClick(f.properties, e.lngLat);
          });
          map.on("mouseenter", layerId, () => { map.getCanvas().style.cursor = "pointer"; });
          map.on("mouseleave", layerId, () => { map.getCanvas().style.cursor = ""; });
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
    const onMove = () => load();
    map.on("moveend", onMove);
    return () => {
      teardown();
      window.clearInterval(iv);
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
        if (map.getLayer("aircraft-veclines")) map.removeLayer("aircraft-veclines");
        if (map.getSource("aircraft")) map.removeSource("aircraft");
        if (map.getSource("aircraft-vec")) map.removeSource("aircraft-vec");
      } catch {}
      setStatus("aircraft", "off");
      return;
    }
    setStatus("aircraft", "loading");
    return wireLivePoints({
      id: "aircraft",
      intervalMs: 15_000,
      toFeatures: (d) => (d.aircraft || []).map((a: any) => {
        const cls = classifyAircraft(a.type, a.category);
        return {
          type: "Feature",
          geometry: { type: "Point", coordinates: [a.lon, a.lat] },
          properties: {
            icon: AIRCRAFT_ICON[cls], cls,
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
        const note = await showTrail("aircraft", p.icao24);
        setDetail(prev => prev && prev.trailId === p.icao24 ? { ...prev, trailNote: note } : prev);
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
        const note = await showTrail("vessels", p.mmsi);
        setDetail(prev => prev && prev.trailId === p.mmsi ? { ...prev, trailNote: note } : prev);
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
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch("/api/data/sites");
        const d = await r.json();
        if (cancelled || !d.sites || map.getSource("sites")) return;
        const colors: Record<string, string> = {};
        const catLabels: Record<string, string> = {};
        Object.entries(d.categories || {}).forEach(([k, v]: any) => { colors[k] = v.color; catLabels[k] = v.label; });
        map.addSource("sites", { type: "geojson", data: {
          type: "FeatureCollection",
          features: d.sites.map((s: any) => ({
            type: "Feature",
            geometry: { type: "Point", coordinates: [s.lon, s.lat] },
            properties: {
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
        map.on("click", "sites-icons", (e: any) => {
          const f = e.features?.[0];
          if (!f) return;
          setDetail({
            kind: "site",
            title: f.properties.name,
            subtitle: `${f.properties.category} · ${f.properties.operator}`,
            body: f.properties.relevance,
          });
        });
        map.on("mouseenter", "sites-icons", () => { map.getCanvas().style.cursor = "pointer"; });
        map.on("mouseleave", "sites-icons", () => { map.getCanvas().style.cursor = ""; });
        setStatus("sites", "active", d.sites.length);
      } catch {
        if (!cancelled) setStatus("sites", "error");
      }
    })();
    return () => { cancelled = true; };
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
    setStatus("powerplants", "loading");
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch("/api/data/powerplants");
        const d = await r.json();
        if (cancelled || !d.plants || map.getSource("powerplants")) return;
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
        map.on("click", "pp-clusters", (e: any) => {
          const f = e.features?.[0];
          if (!f) return;
          const src: any = map.getSource("powerplants");
          src.getClusterExpansionZoom(f.properties.cluster_id, (err: any, zoom: number) => {
            if (!err) map.easeTo({ center: f.geometry.coordinates, zoom: zoom + 0.3 });
          });
        });
        map.on("click", "pp-points", (e: any) => {
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
        for (const l of ["pp-clusters", "pp-points"]) {
          map.on("mouseenter", l, () => { map.getCanvas().style.cursor = "pointer"; });
          map.on("mouseleave", l, () => { map.getCanvas().style.cursor = ""; });
        }
        setStatus("powerplants", "active", d.count ?? d.plants.length,
          `top ${d.verified_count ?? 100} by MW imagery-verified · rest approximate`);
      } catch {
        if (!cancelled) setStatus("powerplants", "error");
      }
    })();
    return () => { cancelled = true; };
  }, [enabled.powerplants, mapReady, setStatus]);

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
    setStatus("trains", "loading");
    let stop = false;
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
          map.on("click", "trains-icons", async (e: any) => {
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
            const note = await showTrail("trains", p.id);
            setDetail(prev => prev && prev.trailId === p.id ? { ...prev, trailNote: note } : prev);
          });
          map.on("mouseenter", "trains-icons", () => { map.getCanvas().style.cursor = "pointer"; });
          map.on("mouseleave", "trains-icons", () => { map.getCanvas().style.cursor = ""; });
        }
        const per = (d.sources || []).map((s: any) => `${s.country} ${s.status === "ok" ? s.count : s.status}`).join(" · ");
        setStatus("trains", "active", d.count, per || undefined);
      } catch {
        if (!stop) setStatus("trains", "error");
      }
    };
    load();
    const iv = window.setInterval(load, 30_000);
    return () => { stop = true; window.clearInterval(iv); };
  }, [enabled.trains, mapReady, setStatus]);

  // ── dark-ship RAW statistics (non-geospatial; derived from our own AIS
  // archive — counts only, per-vessel claims stay ladder-gated) ──
  const [shadowStats, setShadowStats] = useState<any>(null);
  useEffect(() => {
    if (!enabled.shadowstats) { setStatus("shadowstats", "off"); setShadowStats(null); return; }
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
  }, [enabled.shadowstats, setStatus]);

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
  }, [enabled.portdwell, mapReady, setStatus]);

  // ── SEC EDGAR Form 4 insider transactions (RAW; non-geospatial — no
  // markers, an inline list inside the layer panel instead) ──
  useEffect(() => {
    if (!enabled.insider) { setStatus("insider", "off"); return; }
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
    const iv = window.setInterval(load, 60_000);
    return () => { stop = true; window.clearInterval(iv); };
  }, [enabled.insider, setStatus]);

  // ── panel helpers ──
  const layerIcon = (id: string) =>
    id === "imagery" ? <Satellite size={15} /> :
    id === "aircraft" ? <Plane size={15} /> :
    id === "vessels" ? <Ship size={15} /> :
    id === "sites" ? <MapPin size={15} /> :
    id === "powerplants" ? <Zap size={15} /> :
    id === "trains" ? <TrainFront size={15} /> :
    id === "insider" ? <FileText size={15} /> : <LayersIcon size={15} />;

  const statusFor = (l: LayerMeta): { dot: string; text: string; note?: string } => {
    const rt = runtime[l.id];
    if (l.status === "planned") return { dot: "var(--text-tertiary)", text: "coming soon" };
    if (l.status === "awaiting_key" || rt?.status === "awaiting_key") return { dot: "var(--accent-orange)", text: "awaiting API key" };
    if (rt?.status === "error") return { dot: "var(--accent-red)", text: rt.note || "feed error — retrying" };
    if (rt?.status === "loading") return { dot: "var(--accent-orange)", text: "loading…" };
    if (rt?.status === "active") {
      const c = rt.count;
      const unit = l.id === "sites" ? "sites" : l.id === "insider" ? "filings" : l.id === "powerplants" ? "plants" : l.id === "trains" ? "trains" : l.id === "shadowstats" ? "gap events" : l.id === "portdwell" ? "port calls" : l.id;
      return { dot: "var(--accent-green)", text: c != null ? `${c.toLocaleString()} ${unit}` : "active", note: rt.note };
    }
    return { dot: "var(--text-tertiary)", text: "off" };
  };

  const toggleable = (l: LayerMeta) => l.status === "live";

  const renderLayerRow = (l: LayerMeta) => {
    const st = statusFor(l);
    const on = !!enabled[l.id] && toggleable(l);
    const descIsOpen = !!descOpen[l.id];
    return (
      <div key={l.id}>
        <div className={`vt-layer-row${toggleable(l) ? "" : " vt-layer-row-disabled"}`} data-vt-layer={l.id}>
          <span className="vt-layer-ic">{layerIcon(l.id)}</span>
          <span className="vt-layer-name">
            <button className="vt-layer-namebtn" aria-expanded={descIsOpen}
                    aria-label={`About ${l.name}`}
                    onClick={() => setDescOpen((s) => ({ ...s, [l.id]: !s[l.id] }))}>
              {l.name} <Info size={11} aria-hidden style={{ opacity: 0.55 }} />
            </button>
            <span className={`vt-kind-badge ${l.kind}`}>{l.kind === "raw" ? "RAW" : "SIGNAL"}</span>
            <span className="vt-layer-status">
              <i style={{ background: st.dot }} /> {st.text}
            </span>
            {st.note && <span className="vt-layer-covnote">{st.note}</span>}
          </span>
          <button
            role="switch"
            aria-checked={on}
            aria-label={`Toggle ${l.name}`}
            disabled={!toggleable(l)}
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
      </div>
    );
  };

  return (
    <div className="vt-map-page" data-vt-map>
      {filingsOpen && (
        <FilingsView onBack={() => { window.location.hash = "#/data"; setFilingsOpen(false); }} />
      )}

      {/* v2.3 fullscreen: hide the site nav for a full-viewport map */}
      <button className="vt-map-fs-btn" data-vt-fullscreen
              aria-label={fullscreen ? "Exit fullscreen map" : "Fullscreen map"}
              aria-pressed={fullscreen}
              onClick={() => setFullscreen((v) => !v)}>
        {fullscreen ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
      </button>
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
              <span><LayersIcon size={14} /> Layers</span>
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
            {PANEL_GROUPS.map((g) => {
              const members = layers.filter((l) => groupOf(l) === g.id);
              if (members.length === 0) return null;
              const onCount = members.filter((l) => !!enabled[l.id] && toggleable(l)).length;
              const isCollapsed = !!groupCollapsed[g.id];
              return (
                <div key={g.id} className="vt-layer-group">
                  <button className="vt-layer-group-head" aria-expanded={!isCollapsed}
                          onClick={() => setGroupCollapsed((s) => ({ ...s, [g.id]: !s[g.id] }))}>
                    <span className={`vt-layer-group-chev${isCollapsed ? " closed" : ""}`}>▾</span>
                    <span>{g.label}</span>
                    <span className="vt-layer-group-count">{onCount}/{members.length} on</span>
                  </button>
                  {!isCollapsed && members.map((l) => renderLayerRow(l))}
                </div>
              );
            })}
            <div className="vt-legend">
              <span>
                <svg width="13" height="13" viewBox="0 0 40 40" style={{ color: "#4ade80" }} aria-hidden>
                  <g stroke="currentColor" strokeWidth="3.5" fill="none">
                    <circle cx="20" cy="9" r="3.5" /><path d="M20 12.5V32" /><path d="M12 18h16" />
                    <path d="M11 21a9.5 9.5 0 0 0 18 0" fill="none" />
                  </g>
                </svg> ports
              </span>
              <span>
                <svg width="13" height="13" viewBox="0 0 40 40" style={{ color: "#fbb24c" }} aria-hidden>
                  <g fill="currentColor">
                    <circle cx="13" cy="26" r="6.5" /><circle cx="27" cy="26" r="6.5" /><circle cx="20" cy="13" r="6.5" />
                  </g>
                </svg> tank farms
              </span>
              <span>
                <svg width="13" height="13" viewBox="0 0 40 40" style={{ color: "#ff5a6e" }} aria-hidden>
                  <g fill="currentColor">
                    <rect x="8" y="22" width="24" height="12" />
                    <path d="M8 22v-7l8 7v-7l8 7v-7l8 7z" />
                    <rect x="26" y="6" width="4.5" height="12" />
                  </g>
                </svg> steel mills
              </span>
              <span><i style={{ background: "#4d9fff" }} /> jet/cruise</span>
              <span><i style={{ background: "#fbb24c" }} /> low alt · tanker</span>
              <span><i style={{ background: "#6680a0" }} /> ground</span>
              <span><i style={{ background: "#4ade80" }} /> cargo</span>
              <span><i style={{ background: "#c084fc" }} /> passenger</span>
              <span><i style={{ background: "#2dd4bf" }} /> trains</span>
              <span style={{ flexBasis: "100%", height: 0 }} aria-hidden />
              <span style={{ color: "var(--text-tertiary)" }}>plants:</span>
              <span><i style={{ background: "#c084fc" }} /> nuclear</span>
              <span><i style={{ background: "#94a3b8" }} /> coal</span>
              <span><i style={{ background: "#fbb24c" }} /> gas</span>
              <span><i style={{ background: "#ff8a5c" }} /> oil</span>
              <span><i style={{ background: "#4d9fff" }} /> hydro</span>
              <span><i style={{ background: "#7cc4ff" }} /> wind</span>
              <span><i style={{ background: "#fde047" }} /> solar</span>
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
            <p className="vt-site-card-trail">Trail: {detail.trailNote}</p>
          )}
        </div>
      )}
    </div>
  );
}
