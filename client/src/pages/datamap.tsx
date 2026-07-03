import { useCallback, useEffect, useRef, useState } from "react";
import { Layers as LayersIcon, Info, X, Plane, Ship, MapPin, Satellite } from "lucide-react";
// Static CSS import: without maplibre's stylesheet loaded BEFORE the map
// constructs, maplibre mis-measures the container (300px fallback canvas) and
// its controls render unpositioned. The JS stays dynamically imported below —
// only this ~15KB stylesheet rides the main bundle.
import "maplibre-gl/dist/maplibre-gl.css";

/**
 * /data — the data-intelligence map.
 *
 * DESIGN.md-governed: full-viewport below the nav at every width, collapsible
 * layer controls (collapsed by default on phone), alive on first load
 * (aircraft + strategic sites default ON), designed loading/error/awaiting-key
 * states, theme tokens only. Verified by `npm run visual` at 390/768/1440.
 *
 * SPINOUT-READY rules: all overlay data flows through /api/data/* (datacore
 * boundary); base imagery tiles are the documented scoped exception. Every
 * layer is labeled RAW DATA or SIGNAL (gate-2 rule).
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

interface SiteDetail {
  name: string;
  category: string;
  operator: string;
  relevance: string;
}

const IMAGERY_TILES =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const IMAGERY_ATTRIB = "© Esri, Maxar, Earthstar Geographics";

const DEFAULT_ON: Record<string, boolean> = { imagery: true, aircraft: true, sites: true };

export default function DataMapPage() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<any>(null);
  const glRef = useRef<any>(null); // maplibregl module (loaded lazily)
  const [layers, setLayers] = useState<LayerMeta[]>([]);
  const [enabled, setEnabled] = useState<Record<string, boolean>>(DEFAULT_ON);
  const [runtime, setRuntime] = useState<Record<string, { status: RuntimeStatus; count?: number }>>({});
  const [mapReady, setMapReady] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);
  const [panelOpen, setPanelOpen] = useState<boolean>(() =>
    typeof window !== "undefined" ? window.innerWidth >= 768 : true);
  const [showRawInfo, setShowRawInfo] = useState(false);
  const [site, setSite] = useState<SiteDetail | null>(null);

  const setStatus = useCallback((id: string, status: RuntimeStatus, count?: number) => {
    setRuntime(s => ({ ...s, [id]: { status, count } }));
  }, []);

  // Layer registry (datacore boundary)
  useEffect(() => {
    fetch("/api/data/layers")
      .then(r => r.json())
      .then(d => setLayers(Array.isArray(d.layers) ? d.layers : []))
      .catch(() => setLayers([]));
  }, []);

  // Map bootstrap (maplibre lazy-loaded to keep the main bundle lean)
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
            // glyphs are required for symbol/text layers (site labels)
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
        // Ready = style usable, NOT all tiles fetched: a slow or blocked tile
        // CDN must never hold the page hostage on the skeleton. Overlays come
        // from our own /api/data/* and mount as soon as the style is live.
        let readyFired = false;
        const ready = () => {
          if (cancelled || readyFired) return;
          readyFired = true;
          window.clearInterval(stylePoll);
          try { map.resize(); } catch {}  // self-heal any container measure race
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
          // tile errors are routine; only surface fatal init failures
          if (readyFired) return;
          if (e?.error?.message && /style/i.test(e.error.message)) {
            setMapError(e.error.message);
          }
        });
      } catch (e: any) {
        setMapError(e?.message || "Map failed to load");
      }
    })();
    return () => {
      cancelled = true;
      try { mapRef.current?.remove(); mapRef.current = null; } catch {}
    };
  }, []);

  // Escape closes panel / detail card / tooltip (DESIGN.md keyboard rule)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      setSite(null);
      setShowRawInfo(false);
      if (window.innerWidth < 768) setPanelOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ── imagery toggle ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    try { map.setLayoutProperty("imagery", "visibility", enabled.imagery ? "visible" : "none"); } catch {}
    setStatus("imagery", enabled.imagery ? "active" : "off");
  }, [enabled.imagery, mapReady, setStatus]);

  // ── live aircraft (RAW — via /api/data/aircraft) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.aircraft) {
      try {
        if (map.getLayer("aircraft-dots")) map.removeLayer("aircraft-dots");
        if (map.getSource("aircraft")) map.removeSource("aircraft");
      } catch {}
      setStatus("aircraft", "off");
      return;
    }
    setStatus("aircraft", "loading");
    let stop = false;
    const load = async () => {
      try {
        const b = map.getBounds();
        const q = `lamin=${b.getSouth().toFixed(2)}&lamax=${b.getNorth().toFixed(2)}&lomin=${b.getWest().toFixed(2)}&lomax=${b.getEast().toFixed(2)}`;
        const r = await fetch(`/api/data/aircraft?${q}`);
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (stop || !d.aircraft) return;
        const geojson = {
          type: "FeatureCollection",
          features: d.aircraft.map((a: any) => ({
            type: "Feature",
            geometry: { type: "Point", coordinates: [a.lon, a.lat] },
            properties: {
              callsign: a.callsign || a.icao24,
              country: a.origin_country,
              alt: a.altitude_m == null ? -1 : Math.round(a.altitude_m),
              kts: a.velocity_ms == null ? null : Math.round(a.velocity_ms * 1.944),
              ground: !!a.on_ground,
            },
          })),
        };
        const src: any = map.getSource("aircraft");
        if (src) {
          src.setData(geojson);
        } else if (!map.getLayer("aircraft-dots")) {
          map.addSource("aircraft", { type: "geojson", data: geojson as any });
          map.addLayer({
            id: "aircraft-dots",
            type: "circle",
            source: "aircraft",
            paint: {
              "circle-color": ["case", ["get", "ground"], "#6680a0", ["<", ["get", "alt"], 3000], "#fbb24c", "#4d9fff"],
              "circle-radius": ["interpolate", ["linear"], ["zoom"], 3, 2.5, 8, 5],
              "circle-stroke-width": 1,
              "circle-stroke-color": "rgba(0,0,0,0.55)",
            },
          });
          map.on("click", "aircraft-dots", (e: any) => {
            const f = e.features?.[0];
            if (!f || !glRef.current) return;
            const p = f.properties;
            new glRef.current.Popup({ closeButton: true, className: "vt-popup", maxWidth: "240px" })
              .setLngLat(e.lngLat)
              .setHTML(
                `<div class="vt-popup-body"><div class="vt-popup-title">✈ ${p.callsign}</div>` +
                `<div class="vt-popup-line">${p.country || ""}</div>` +
                `<div class="vt-popup-line">${p.ground === true || p.ground === "true" ? "on ground" : `${p.alt} m`}${p.kts ? ` · ${p.kts} kts` : ""}</div></div>`
              )
              .addTo(map);
          });
          map.on("mouseenter", "aircraft-dots", () => { map.getCanvas().style.cursor = "pointer"; });
          map.on("mouseleave", "aircraft-dots", () => { map.getCanvas().style.cursor = ""; });
        }
        setStatus("aircraft", "active", d.count ?? geojson.features.length);
      } catch {
        if (!stop) setStatus("aircraft", "error");
      }
    };
    load();
    const iv = window.setInterval(load, 15_000);
    const onMove = () => load();
    map.on("moveend", onMove);
    return () => {
      stop = true;
      window.clearInterval(iv);
      try { map.off("moveend", onMove); } catch {}
    };
  }, [enabled.aircraft, mapReady, setStatus]);

  // ── live vessels (RAW — key-gated; registry only marks live when key set) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    const meta = layers.find(l => l.id === "vessels");
    if (meta && meta.status === "awaiting_key") setStatus("vessels", "awaiting_key");
    if (!enabled.vessels) {
      try {
        if (map.getLayer("vessel-dots")) map.removeLayer("vessel-dots");
        if (map.getSource("vessels")) map.removeSource("vessels");
      } catch {}
      if (!meta || meta.status !== "awaiting_key") setStatus("vessels", "off");
      return;
    }
    setStatus("vessels", "loading");
    let stop = false;
    const load = async () => {
      try {
        const b = map.getBounds();
        const q = `lamin=${b.getSouth().toFixed(2)}&lamax=${b.getNorth().toFixed(2)}&lomin=${b.getWest().toFixed(2)}&lomax=${b.getEast().toFixed(2)}`;
        const r = await fetch(`/api/data/vessels?${q}`);
        const d = await r.json();
        if (stop) return;
        if (!d.enabled) { setStatus("vessels", "awaiting_key"); return; }
        const geojson = {
          type: "FeatureCollection",
          features: (d.vessels || []).map((v: any) => ({
            type: "Feature",
            geometry: { type: "Point", coordinates: [v.lon, v.lat] },
            properties: { name: v.name, mmsi: v.mmsi, kts: v.sog == null ? null : Math.round(v.sog) },
          })),
        };
        const src: any = map.getSource("vessels");
        if (src) {
          src.setData(geojson);
        } else if (!map.getLayer("vessel-dots")) {
          map.addSource("vessels", { type: "geojson", data: geojson as any });
          map.addLayer({
            id: "vessel-dots",
            type: "circle",
            source: "vessels",
            paint: {
              "circle-color": "#4ade80",
              "circle-radius": ["interpolate", ["linear"], ["zoom"], 3, 2.5, 8, 5],
              "circle-stroke-width": 1,
              "circle-stroke-color": "rgba(0,0,0,0.55)",
            },
          });
          map.on("click", "vessel-dots", (e: any) => {
            const f = e.features?.[0];
            if (!f || !glRef.current) return;
            const p = f.properties;
            new glRef.current.Popup({ closeButton: true, className: "vt-popup", maxWidth: "240px" })
              .setLngLat(e.lngLat)
              .setHTML(
                `<div class="vt-popup-body"><div class="vt-popup-title">⚓ ${p.name}</div>` +
                `<div class="vt-popup-line">MMSI ${p.mmsi}${p.kts != null ? ` · ${p.kts} kts` : ""}</div></div>`
              )
              .addTo(map);
          });
          map.on("mouseenter", "vessel-dots", () => { map.getCanvas().style.cursor = "pointer"; });
          map.on("mouseleave", "vessel-dots", () => { map.getCanvas().style.cursor = ""; });
        }
        setStatus("vessels", "active", d.count ?? geojson.features.length);
      } catch {
        if (!stop) setStatus("vessels", "error");
      }
    };
    load();
    const iv = window.setInterval(load, 20_000);
    return () => { stop = true; window.clearInterval(iv); };
  }, [enabled.vessels, mapReady, layers, setStatus]);

  // ── strategic sites (RAW — datacore reference data; opens the detail card) ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    if (!enabled.sites) {
      try {
        if (map.getLayer("sites-dots")) map.removeLayer("sites-dots");
        if (map.getSource("sites")) map.removeSource("sites");
      } catch {}
      setStatus("sites", "off");
      setSite(null);
      return;
    }
    setStatus("sites", "loading");
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch("/api/data/sites");
        const d = await r.json();
        if (cancelled || !d.sites) return;
        const colors: Record<string, string> = {};
        Object.entries(d.categories || {}).forEach(([k, v]: any) => { colors[k] = v.color; });
        const catLabels: Record<string, string> = {};
        Object.entries(d.categories || {}).forEach(([k, v]: any) => { catLabels[k] = v.label; });
        const geojson = {
          type: "FeatureCollection",
          features: d.sites.map((s: any) => ({
            type: "Feature",
            geometry: { type: "Point", coordinates: [s.lon, s.lat] },
            properties: {
              name: s.name,
              category: catLabels[s.category] || s.category,
              operator: s.operator,
              relevance: s.relevance,
              color: colors[s.category] || "#4d9fff",
            },
          })),
        };
        if (!map.getSource("sites")) {
          map.addSource("sites", { type: "geojson", data: geojson as any });
          map.addLayer({
            id: "sites-dots",
            type: "circle",
            source: "sites",
            paint: {
              "circle-color": ["get", "color"],
              "circle-radius": ["interpolate", ["linear"], ["zoom"], 3, 5, 8, 8],
              "circle-opacity": 0.9,
              "circle-stroke-width": 1.5,
              "circle-stroke-color": "rgba(238,243,251,0.9)",
            },
          });
          map.on("click", "sites-dots", (e: any) => {
            const f = e.features?.[0];
            if (!f) return;
            setSite({
              name: f.properties.name,
              category: f.properties.category,
              operator: f.properties.operator,
              relevance: f.properties.relevance,
            });
          });
          map.on("mouseenter", "sites-dots", () => { map.getCanvas().style.cursor = "pointer"; });
          map.on("mouseleave", "sites-dots", () => { map.getCanvas().style.cursor = ""; });
        }
        setStatus("sites", "active", d.sites.length);
      } catch {
        if (!cancelled) setStatus("sites", "error");
      }
    })();
    return () => { cancelled = true; };
  }, [enabled.sites, mapReady, setStatus]);

  // ── panel helpers ──
  const layerIcon = (id: string) =>
    id === "imagery" ? <Satellite size={15} /> :
    id === "aircraft" ? <Plane size={15} /> :
    id === "vessels" ? <Ship size={15} /> :
    id === "sites" ? <MapPin size={15} /> : <LayersIcon size={15} />;

  const statusFor = (l: LayerMeta): { dot: string; text: string } => {
    const rt = runtime[l.id]?.status;
    if (l.status === "planned") return { dot: "var(--text-tertiary)", text: "coming soon" };
    if (l.status === "awaiting_key" || rt === "awaiting_key") return { dot: "var(--accent-orange)", text: "awaiting API key" };
    if (rt === "error") return { dot: "var(--accent-red)", text: "feed error — retrying" };
    if (rt === "loading") return { dot: "var(--accent-orange)", text: "loading…" };
    if (rt === "active") {
      const c = runtime[l.id]?.count;
      return { dot: "var(--accent-green)", text: c != null ? `${c.toLocaleString()} ${l.id === "sites" ? "sites" : l.id}` : "active" };
    }
    return { dot: "var(--text-tertiary)", text: "off" };
  };

  const toggleable = (l: LayerMeta) => l.status === "live";

  return (
    <div className="vt-map-page" data-vt-map>
      <div ref={mapContainer} className="vt-map-canvas" />

      {/* loading skeleton until the map style is ready */}
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

      {/* Layers control — top-right, collapsible; collapsed default on phone */}
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
                statistical validation (ladder gate 2).
              </div>
            )}
            {layers.map(l => {
              const st = statusFor(l);
              const on = !!enabled[l.id] && toggleable(l);
              return (
                <div key={l.id} className={`vt-layer-row${toggleable(l) ? "" : " vt-layer-row-disabled"}`}>
                  <span className="vt-layer-ic">{layerIcon(l.id)}</span>
                  <span className="vt-layer-name">
                    {l.name}
                    <span className={`vt-kind-badge ${l.kind}`}>{l.kind === "raw" ? "RAW" : "SIGNAL"}</span>
                    <span className="vt-layer-status">
                      <i style={{ background: st.dot }} /> {st.text}
                    </span>
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
              );
            })}
            <div className="vt-legend">
              <span><i style={{ background: "#4d9fff" }} /> cruise</span>
              <span><i style={{ background: "#fbb24c" }} /> low alt / tanks</span>
              <span><i style={{ background: "#6680a0" }} /> ground</span>
              <span><i style={{ background: "#ff5a6e" }} /> mills</span>
              <span><i style={{ background: "#4ade80" }} /> ports / vessels</span>
            </div>
          </div>
        )}
      </div>

      {/* Site detail card — side card on desktop, bottom sheet on phone */}
      {site && (
        <div className="vt-site-card" role="dialog" aria-label={site.name}>
          <div className="vt-site-card-head">
            <div>
              <div className="vt-site-card-title">{site.name}</div>
              <div className="vt-site-card-cat">{site.category} · {site.operator}</div>
            </div>
            <button className="vt-icon-btn" aria-label="Close details" onClick={() => setSite(null)}>
              <X size={17} />
            </button>
          </div>
          <p className="vt-site-card-body">{site.relevance}</p>
        </div>
      )}
    </div>
  );
}
