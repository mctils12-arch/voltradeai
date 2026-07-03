import { useEffect, useRef, useState } from "react";
import { Globe, Layers as LayersIcon, Info } from "lucide-react";

/**
 * /data — the data-intelligence map (v1, slice 1: map + imagery + layer UI).
 *
 * SPINOUT-READY rules (CLAUDE.md): overlay DATA comes only from /api/data/*
 * (the datacore boundary) — this page never calls external data APIs
 * directly. Base imagery tiles are the one scoped exception (static tiles,
 * attribution on-map). Every layer is labeled RAW DATA or SIGNAL; SIGNAL
 * layers cannot surface before ladder gate 2.
 */

interface LayerMeta {
  id: string;
  name: string;
  kind: "raw" | "signal";
  status: "live" | "awaiting_key" | "planned";
  source: string;
  description: string;
}

// Esri World Imagery — free raster tiles with required attribution.
const IMAGERY_TILES =
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const IMAGERY_ATTRIB =
  "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community";

export default function DataMapPage() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<any>(null);
  const [layers, setLayers] = useState<LayerMeta[]>([]);
  const [enabled, setEnabled] = useState<Record<string, boolean>>({ imagery: true });
  const [mapReady, setMapReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Layer registry from the datacore boundary
  useEffect(() => {
    fetch("/api/data/layers")
      .then(r => r.json())
      .then(d => setLayers(Array.isArray(d.layers) ? d.layers : []))
      .catch(() => setLayers([]));
  }, []);

  // Lazy-load maplibre so the main bundle stays lean
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const maplibregl = (await import("maplibre-gl")).default;
        await import("maplibre-gl/dist/maplibre-gl.css");
        if (cancelled || !mapContainer.current || mapRef.current) return;
        const map = new maplibregl.Map({
          container: mapContainer.current,
          style: {
            version: 8,
            sources: {
              imagery: {
                type: "raster",
                tiles: [IMAGERY_TILES],
                tileSize: 256,
                attribution: IMAGERY_ATTRIB,
              },
            },
            layers: [
              { id: "bg", type: "background", paint: { "background-color": "#05070d" } },
              { id: "imagery", type: "raster", source: "imagery", paint: { "raster-opacity": 1 } },
            ],
          },
          center: [-96.77, 35.98], // Cushing, OK — the first strategic site
          zoom: 4,
          attributionControl: { compact: true } as any,
        });
        map.addControl(new maplibregl.NavigationControl({ showCompass: true }), "top-right");
        map.addControl(new maplibregl.ScaleControl({ unit: "imperial" }), "bottom-left");
        mapRef.current = map;
        map.on("load", () => setMapReady(true));
      } catch (e: any) {
        setError(e?.message || "Map failed to load");
      }
    })();
    return () => {
      cancelled = true;
      try { mapRef.current?.remove(); mapRef.current = null; } catch {}
    };
  }, []);

  // Imagery toggle (other layers arrive in later slices and hook in here)
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    try {
      map.setLayoutProperty("imagery", "visibility", enabled.imagery ? "visible" : "none");
    } catch {}
  }, [enabled.imagery, mapReady]);

  const kindBadge = (kind: string) => (
    <span style={{
      fontSize: 9, fontWeight: 700, letterSpacing: "0.06em", padding: "2px 6px",
      borderRadius: 4, textTransform: "uppercase",
      background: kind === "raw" ? "rgba(77,159,255,0.15)" : "rgba(245,166,35,0.15)",
      color: kind === "raw" ? "#4d9fff" : "#f5a623",
      border: `1px solid ${kind === "raw" ? "rgba(77,159,255,0.4)" : "rgba(245,166,35,0.4)"}`,
    }}>
      {kind === "raw" ? "RAW DATA" : "SIGNAL"}
    </span>
  );

  return (
    <div style={{ position: "relative", height: "calc(100vh - 120px)", minHeight: 480, borderRadius: 10, overflow: "hidden", border: "1px solid rgba(255,255,255,0.08)" }}>
      <div ref={mapContainer} style={{ position: "absolute", inset: 0 }} />

      {!mapReady && !error && (
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "#7e8ca0", fontSize: 13, background: "#05070d" }}>
          <Globe size={16} style={{ marginRight: 8 }} /> Loading map…
        </div>
      )}
      {error && (
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "#ff6b5a", fontSize: 13, background: "#05070d" }}>
          {error}
        </div>
      )}

      {/* Layer panel */}
      <div style={{ position: "absolute", top: 14, left: 14, width: 262, background: "rgba(5,10,19,0.92)", backdropFilter: "blur(6px)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 10, padding: "12px 14px", color: "#eef3fb" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 12, fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase", color: "#b3c2d8", marginBottom: 10 }}>
          <LayersIcon size={14} /> Layers
        </div>
        {layers.map(l => {
          const isLive = l.status === "live";
          return (
            <div key={l.id} style={{ display: "flex", alignItems: "flex-start", gap: 8, padding: "6px 0", opacity: isLive ? 1 : 0.55 }}>
              <input
                type="checkbox"
                checked={!!enabled[l.id] && isLive}
                disabled={!isLive}
                onChange={e => setEnabled(s => ({ ...s, [l.id]: e.target.checked }))}
                style={{ marginTop: 2, accentColor: "#4d9fff" }}
              />
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12.5 }}>
                  {l.name} {kindBadge(l.kind)}
                </div>
                <div style={{ fontSize: 10.5, color: "#6b7c93", marginTop: 2 }}>
                  {isLive ? l.source : l.status === "awaiting_key" ? "Awaiting API key (see wishlist)" : "Coming soon"}
                </div>
              </div>
            </div>
          );
        })}
        <div style={{ display: "flex", gap: 6, marginTop: 10, paddingTop: 10, borderTop: "1px solid rgba(255,255,255,0.07)", fontSize: 10, color: "#5d6b80", lineHeight: 1.45 }}>
          <Info size={12} style={{ flexShrink: 0, marginTop: 1 }} />
          <span>
            RAW DATA layers display sources as-is with attribution — no predictive claim.
            SIGNAL layers appear only after statistical validation (ladder gate 2).
          </span>
        </div>
      </div>
    </div>
  );
}
