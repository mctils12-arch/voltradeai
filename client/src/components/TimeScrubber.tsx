import { useEffect, useRef, useState } from "react";
import { Play, Pause, X, Clock } from "lucide-react";
import type maplibregl from "maplibre-gl";
// EARTH TWIN E1: this panel IS the global time axis's UI — every committed
// scrub publishes the instant so dated layers (GIBS imagery) follow the same
// moment the archive replay shows. Closing the panel returns the world to
// LIVE. (lib/timeAxis; datamap subscribes.)
import { setTimeAxis } from "@/lib/timeAxis";

/**
 * TimeScrubber — ANALYST CONSOLE W3: "pick a window, scrub, watch the world
 * move" over the archives we already record. Lazy-loaded from datamap.tsx
 * (zero-cost-when-off, same pattern as AnalystPane) — a closed panel loads
 * no code and issues no requests. Pure archive readout (GET /api/data/
 * snapshot, server/queryEngine.ts's querySnapshot): zero new data cost, RAW
 * overlay (no ladder gate), and honestly labeled as historical replay so it
 * is never mistaken for a live layer.
 *
 * The map instance is owned by the PARENT (datamap.tsx); this component only
 * adds/updates/removes its OWN geojson source+layer ("time-scrubber-*") on
 * that instance, mirroring the existing trail-line pattern — never touches
 * any live layer's state.
 */

const SNAPSHOT_SOURCE = "time-scrubber-snapshot";
const SNAPSHOT_LAYER = "time-scrubber-points";
const PLAY_INTERVAL_MS = 900;
const DEFAULT_HOURS_BACK = 24;
const FALLBACK_MAX_HOURS = 7 * 24; // reconciled with the server's stated window once known

const LAYERS: Array<{ value: string; label: string }> = [
  { value: "aircraft", label: "Aircraft" },
  { value: "vessels", label: "Vessels" },
  { value: "trains", label: "Trains" },
  { value: "fires", label: "Fire detections" },
  { value: "alerts", label: "NWS alerts" },
  { value: "gauges", label: "River gauges" },
];

interface SnapshotPoint { id: string | null; lat: number; lon: number; label?: string; severity?: string | null; value?: number | null }
interface SnapshotEnvelope {
  layer: string; mode: "position" | "event"; bucket_at: string;
  data: SnapshotPoint[]; count: number; count_before_viewport: number; count_dropped_offscreen: number;
  viewport_filtered: boolean; capped: boolean; freshness: string | null; provenance: string;
  window: { min_iso: string; max_iso: string; days: number };
  note: string; error?: string;
}

function fmtUtc(iso: string): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toISOString().slice(0, 16).replace("T", " ") + " UTC";
}

export default function TimeScrubber({ map, onClose }: {
  map: maplibregl.Map | null;
  onClose: () => void;
}) {
  const [layer, setLayer] = useState("aircraft");
  const [hoursBack, setHoursBack] = useState(DEFAULT_HOURS_BACK);
  const [maxHours, setMaxHours] = useState(FALLBACK_MAX_HOURS);
  const [playing, setPlaying] = useState(false);
  const [snap, setSnap] = useState<SnapshotEnvelope | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef(false);
  const nowRef = useRef(Date.now()); // stable "now" for one panel session — a scrub session should not silently redefine hour 0 mid-drag

  const clearMapLayer = () => {
    const m = map;
    if (!m) return;
    try {
      if (m.getLayer(SNAPSHOT_LAYER)) m.removeLayer(SNAPSHOT_LAYER);
      if (m.getSource(SNAPSHOT_SOURCE)) m.removeSource(SNAPSHOT_SOURCE);
    } catch { /* style not ready / already gone — non-fatal */ }
  };

  const paint = (points: SnapshotPoint[]) => {
    const m = map;
    if (!m) return;
    const fc = {
      type: "FeatureCollection",
      features: points.map((p) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [p.lon, p.lat] },
        properties: { label: p.label || "", severity: p.severity || "" },
      })),
    } as any;
    try {
      const existing = m.getSource(SNAPSHOT_SOURCE) as any;
      if (existing) {
        existing.setData(fc);
      } else {
        m.addSource(SNAPSHOT_SOURCE, { type: "geojson", data: fc });
        m.addLayer({
          id: SNAPSHOT_LAYER, type: "circle", source: SNAPSHOT_SOURCE,
          paint: {
            "circle-radius": 5,
            "circle-color": "#f5a524",
            "circle-opacity": 0.85,
            "circle-stroke-width": 1.5,
            "circle-stroke-color": "#3a1d00",
          },
        });
      }
    } catch { /* style not ready yet — next fetch retries */ }
  };

  const fetchSnapshot = async (hb: number, lyr: string) => {
    if (!map || inFlight.current) return;
    inFlight.current = true;
    setLoading(true);
    setError(null);
    // E1 global time axis: every committed scrub position moves the WORLD's
    // clock, not just the replay dots — dated layers follow via lib/timeAxis.
    setTimeAxis(hb === 0 ? { mode: "live" } : { mode: "historical", atMs: nowRef.current - hb * 3600_000 });
    try {
      const at = new Date(nowRef.current - hb * 3600_000).toISOString();
      const b = map.getBounds();
      const bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].join(",");
      const r = await fetch(`/api/data/snapshot?layer=${encodeURIComponent(lyr)}&at=${encodeURIComponent(at)}&bbox=${encodeURIComponent(bbox)}`);
      const d: SnapshotEnvelope = await r.json();
      if (!r.ok) { setError(d.error || `request failed (${r.status})`); setSnap(null); paint([]); return; }
      setSnap(d);
      setMaxHours(d.window.days * 24);
      paint(d.data);
    } catch (e: any) {
      setError(e?.message || "network error");
      setSnap(null);
    } finally {
      setLoading(false);
      inFlight.current = false;
    }
  };

  // Initial fetch on open + whenever the layer or the COMMITTED hour changes.
  useEffect(() => { fetchSnapshot(hoursBack, layer); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [layer]);

  // Playback: step toward "now" (hoursBack -> 0), one fetch per tick, never
  // overlapping a fetch still in flight.
  useEffect(() => {
    if (!playing) return;
    const iv = setInterval(() => {
      if (inFlight.current) return;
      setHoursBack((prev) => {
        const next = Math.max(0, prev - 1);
        fetchSnapshot(next, layer);
        if (next === 0) setPlaying(false);
        return next;
      });
    }, PLAY_INTERVAL_MS);
    return () => clearInterval(iv);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, layer]);

  // Cleanup the map layer when the panel closes (unmounts) — and return the
  // global time axis to LIVE: the panel is the axis's only UI, so a closed
  // panel must never leave the world silently stuck in the past.
  useEffect(() => () => { clearMapLayer(); setTimeAxis({ mode: "live" }); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const onSliderCommit = (hb: number) => {
    setHoursBack(hb);
    setPlaying(false);
    fetchSnapshot(hb, layer);
  };

  const atIso = new Date(nowRef.current - hoursBack * 3600_000).toISOString();

  return (
    <div className="vt-timescrub-panel" data-vt-timescrub-panel role="dialog" aria-label="Time scrubber">
      <div className="vt-timescrub-header">
        <span className="vt-timescrub-title"><Clock size={15} /> Time Machine</span>
        <button aria-label="Close time machine" onClick={() => { setPlaying(false); clearMapLayer(); onClose(); }}>
          <X size={16} />
        </button>
      </div>

      <select className="vt-timescrub-layer" data-vt-timescrub-layer value={layer}
              onChange={(e) => { setPlaying(false); setLayer(e.target.value); }}>
        {LAYERS.map((l) => <option key={l.value} value={l.value}>{l.label}</option>)}
      </select>

      <div className="vt-timescrub-date">{hoursBack === 0 ? "Now" : fmtUtc(atIso)}</div>

      <div className="vt-timescrub-controls">
        <button className="vt-timescrub-play" data-vt-timescrub-play
                aria-label={playing ? "Pause playback" : "Play playback"}
                aria-pressed={playing}
                disabled={hoursBack === 0 && !playing}
                onClick={() => setPlaying((v) => !v)}>
          {playing ? <Pause size={16} /> : <Play size={16} />}
        </button>
        <input type="range" data-vt-timescrub-slider
               min={0} max={maxHours} step={1}
               value={hoursBack}
               aria-label="Hours back from now"
               onChange={(e) => setHoursBack(Number(e.target.value))}
               onMouseUp={(e) => onSliderCommit(Number((e.target as HTMLInputElement).value))}
               onTouchEnd={(e) => onSliderCommit(Number((e.target as HTMLInputElement).value))}
               onKeyUp={(e) => onSliderCommit(Number((e.target as HTMLInputElement).value))} />
      </div>

      <div className="vt-timescrub-status" role="status" aria-live="polite">
        {loading && "Loading…"}
        {!loading && error && <span className="vt-timescrub-error">{error}</span>}
        {!loading && !error && snap && (
          <>
            {snap.count} point{snap.count === 1 ? "" : "s"}
            {snap.capped && " (capped)"}
            {snap.viewport_filtered && snap.count_dropped_offscreen > 0 && ` · ${snap.count_dropped_offscreen} off-screen`}
            {" · "}{snap.provenance}
          </>
        )}
      </div>
      <div className="vt-timescrub-note">
        Historical replay from our own archive — not live. Window: last {Math.round(maxHours / 24)} days.
        Dated imagery layers (night lights, NDVI, soil moisture…) follow this clock to their nearest available day.
      </div>
    </div>
  );
}
