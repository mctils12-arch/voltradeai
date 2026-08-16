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
 * no code and issues no requests.
 *
 * Two modes, chosen by the selected layer:
 * - AIRCRAFT / VESSELS (TIME MACHINE v2 T-2 + T-4, earth_twin_program.md,
 *   human directive 2026-08-11): fetches one hex-multiplexed WINDOW
 *   (GET /api/data/aircraft/window?kind=aircraft|vessels,
 *   server/aircraftWindow.ts T-1, shipped v1.0.672; kind param wired T-4)
 *   covering the selected window length (1h/6h/24h/7d/30d) at the selected
 *   decimation step (1min/5min/15min/1h — the human's "mins or hours
 *   different thing to choose from"), then scrubs a CURSOR through the
 *   already-loaded per-hex tracks locally — no network round-trip per drag
 *   tick, unlike the old per-instant fetch model. Each hex draws at its
 *   last known position at-or-before the cursor. Vessels have no altitude
 *   (paths, not curtains — same as aircraft look today). T-3 (a batched
 *   MultiTrackLayer curtain fleet) is the next slice, not this one — this
 *   panel still paints flat points, matching the pre-T-2 visual for every
 *   other layer.
 * - EVERYTHING ELSE (trains/fires/alerts/gauges): unchanged single-instant
 *   snapshot model (GET /api/data/snapshot, server/queryEngine.ts's
 *   querySnapshot) — these layers have no window-read backend yet.
 *
 * Both modes are RAW overlays (no ladder gate) and honestly labeled as
 * historical replay so neither is mistaken for a live layer.
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

// TIME MACHINE v2 T-2/T-4: layers with a window-read backend. Both share the
// same server route (kind= query param) and client rendering path — vessels
// draw as flat points same as aircraft (no altitude, so no curtain either
// way until T-3 ships).
const WINDOW_KINDS = new Set(["aircraft", "vessels"]);

const WINDOW_OPTIONS_SEC: Array<{ value: number; label: string }> = [
  { value: 3600, label: "1h" },
  { value: 6 * 3600, label: "6h" },
  { value: 24 * 3600, label: "24h" },
  { value: 7 * 24 * 3600, label: "7d" },
  { value: 30 * 24 * 3600, label: "30d" },
];
const STEP_OPTIONS_SEC: Array<{ value: number; label: string }> = [
  { value: 60, label: "1 min" },
  { value: 300, label: "5 min" },
  { value: 900, label: "15 min" },
  { value: 3600, label: "1 hour" },
];
const DEFAULT_WINDOW_SEC = 24 * 3600;
const DEFAULT_STEP_SEC = 300;

interface SnapshotPoint { id: string | null; lat: number; lon: number; label?: string; severity?: string | null; value?: number | null }
interface SnapshotEnvelope {
  layer: string; mode: "position" | "event"; bucket_at: string;
  data: SnapshotPoint[]; count: number; count_before_viewport: number; count_dropped_offscreen: number;
  viewport_filtered: boolean; capped: boolean; freshness: string | null; provenance: string;
  window: { min_iso: string; max_iso: string; days: number };
  note: string; error?: string;
}

interface WindowHex {
  i: string; rg?: string; c?: string; ty?: string;
  points: Array<[number, number, number, number | null]>;
  raw_count: number; truncated: boolean;
}
interface WindowResult {
  kind: string; from: number; to: number; zoom: number; step_sec: number;
  hexes: WindowHex[]; hexes_seen: number; total_points: number;
  coverage: { requested_from: number; scanned_from: number; complete: boolean; files_scanned: number };
  note?: string; error?: string;
}

function fmtUtc(iso: string): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toISOString().slice(0, 16).replace("T", " ") + " UTC";
}

/** Reconstruct each hex's position at-or-before `cursorSec` — a local
 *  "snapshot at time T" over an already-loaded window, no network call. */
function positionsAtCursor(w: WindowResult, cursorSec: number): SnapshotPoint[] {
  const out: SnapshotPoint[] = [];
  for (const h of w.hexes) {
    let best: [number, number, number, number | null] | null = null;
    for (const p of h.points) {
      if (p[0] > cursorSec) break; // points are time-ascending
      best = p;
    }
    if (!best) continue;
    out.push({ id: h.i, lat: best[1], lon: best[2], label: h.c || h.rg || h.i, severity: null, value: best[3] });
  }
  return out;
}

export default function TimeScrubber({ map, onClose }: {
  map: maplibregl.Map | null;
  onClose: () => void;
}) {
  const [layer, setLayer] = useState("aircraft");
  const isWindowMode = WINDOW_KINDS.has(layer);

  // Snapshot-mode state (vessels/trains/fires/alerts/gauges) — unchanged.
  const [hoursBack, setHoursBack] = useState(DEFAULT_HOURS_BACK);
  const [maxHours, setMaxHours] = useState(FALLBACK_MAX_HOURS);
  const [snap, setSnap] = useState<SnapshotEnvelope | null>(null);

  // Window-mode state (aircraft, T-2) — a preloaded window scrubbed locally.
  const [windowSec, setWindowSec] = useState(DEFAULT_WINDOW_SEC);
  const [stepSec, setStepSec] = useState(DEFAULT_STEP_SEC);
  const [windowData, setWindowData] = useState<WindowResult | null>(null);
  const [cursorSec, setCursorSec] = useState<number | null>(null); // display-only until committed
  const [windowError, setWindowError] = useState<string | null>(null);

  const [playing, setPlaying] = useState(false);
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

  // T-2: fetch one window (bbox + zoom taken at fetch time, not tracked
  // live — no map-event subscription here, matching the snapshot mode's
  // existing pattern and the Rendering & Motion Law's "no visual state from
  // moveend/zoomend" rule). Cursor starts at the window's end (live/"now").
  const fetchWindow = async () => {
    if (!map || inFlight.current) return;
    inFlight.current = true;
    setLoading(true);
    setWindowError(null);
    try {
      const to = Math.floor(nowRef.current / 1000);
      const from = to - windowSec;
      const b = map.getBounds();
      const bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].join(",");
      const zoom = map.getZoom();
      // T-4: layer IS the kind for both window-mode layers today (aircraft,
      // vessels) — no separate mapping needed unless a third window kind
      // (trains?) ever joins with a different layer id.
      const r = await fetch(`/api/data/aircraft/window?bbox=${encodeURIComponent(bbox)}&from=${from}&to=${to}&zoom=${zoom}&step=${stepSec}&kind=${encodeURIComponent(layer)}`);
      const d: WindowResult = await r.json();
      if (!r.ok) { setWindowError(d.error || `request failed (${r.status})`); setWindowData(null); paint([]); return; }
      setWindowData(d);
      setCursorSec(d.to);
      setTimeAxis({ mode: "live" });
      paint(positionsAtCursor(d, d.to));
    } catch (e: any) {
      setWindowError(e?.message || "network error");
      setWindowData(null);
    } finally {
      setLoading(false);
      inFlight.current = false;
    }
  };

  // Initial fetch on open + whenever the layer, window, or step changes.
  useEffect(() => {
    setPlaying(false);
    if (isWindowMode) fetchWindow(); else fetchSnapshot(hoursBack, layer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layer, isWindowMode ? windowSec : null, isWindowMode ? stepSec : null]);

  // Snapshot-mode playback: step toward "now" (hoursBack -> 0), one fetch
  // per tick, never overlapping a fetch still in flight.
  useEffect(() => {
    if (!playing || isWindowMode) return;
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
  }, [playing, isWindowMode, layer]);

  // Window-mode playback: advance the cursor through the ALREADY-LOADED
  // window locally — no fetch per tick. Tick size scales with the window so
  // a full sweep takes ~60 ticks regardless of window length, floored at
  // the selected step (never advances finer than the data was decimated).
  useEffect(() => {
    if (!playing || !isWindowMode || !windowData) return;
    const tick = Math.max(stepSec, Math.round(windowSec / 60));
    const iv = setInterval(() => {
      setCursorSec((prev) => {
        const cur = prev ?? windowData.from;
        const next = Math.min(windowData.to, cur + tick);
        paint(positionsAtCursor(windowData, next));
        setTimeAxis(next >= windowData.to ? { mode: "live" } : { mode: "historical", atMs: next * 1000 });
        if (next >= windowData.to) setPlaying(false);
        return next;
      });
    }, PLAY_INTERVAL_MS);
    return () => clearInterval(iv);
  }, [playing, isWindowMode, windowData, stepSec, windowSec]);

  // Cleanup the map layer when the panel closes (unmounts) — and return the
  // global time axis to LIVE: the panel is the axis's only UI, so a closed
  // panel must never leave the world silently stuck in the past.
  useEffect(() => () => { clearMapLayer(); setTimeAxis({ mode: "live" }); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const onSliderCommit = (hb: number) => {
    setHoursBack(hb);
    setPlaying(false);
    fetchSnapshot(hb, layer);
  };

  const onCursorCommit = (cur: number) => {
    setCursorSec(cur);
    setPlaying(false);
    if (!windowData) return;
    paint(positionsAtCursor(windowData, cur));
    setTimeAxis(cur >= windowData.to ? { mode: "live" } : { mode: "historical", atMs: cur * 1000 });
  };

  const atIso = new Date(nowRef.current - hoursBack * 3600_000).toISOString();
  const cursorIso = cursorSec != null ? new Date(cursorSec * 1000).toISOString() : null;
  const atLive = isWindowMode
    ? (windowData != null && cursorSec != null && cursorSec >= windowData.to)
    : hoursBack === 0;

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

      {isWindowMode && (
        <div className="vt-timescrub-window-row" data-vt-timescrub-window-row>
          <select className="vt-timescrub-layer" data-vt-timescrub-window value={windowSec}
                  aria-label="Replay window length"
                  onChange={(e) => setWindowSec(Number(e.target.value))}>
            {WINDOW_OPTIONS_SEC.map((o) => <option key={o.value} value={o.value}>{o.label} window</option>)}
          </select>
          <select className="vt-timescrub-layer" data-vt-timescrub-step value={stepSec}
                  aria-label="Replay time step"
                  onChange={(e) => setStepSec(Number(e.target.value))}>
            {STEP_OPTIONS_SEC.map((o) => <option key={o.value} value={o.value}>{o.label} step</option>)}
          </select>
        </div>
      )}

      <div className="vt-timescrub-date">
        {atLive ? "Now" : isWindowMode ? (cursorIso ? fmtUtc(cursorIso) : "…") : fmtUtc(atIso)}
      </div>

      <div className="vt-timescrub-controls">
        <button className="vt-timescrub-play" data-vt-timescrub-play
                aria-label={playing ? "Pause playback" : "Play playback"}
                aria-pressed={playing}
                disabled={atLive && !playing}
                onClick={() => setPlaying((v) => !v)}>
          {playing ? <Pause size={16} /> : <Play size={16} />}
        </button>
        {isWindowMode ? (
          <input type="range" data-vt-timescrub-slider
                 min={windowData?.from ?? 0} max={windowData?.to ?? 1} step={1}
                 value={cursorSec ?? windowData?.to ?? 0}
                 disabled={!windowData}
                 aria-label="Cursor position within the loaded window"
                 onChange={(e) => setCursorSec(Number(e.target.value))}
                 onMouseUp={(e) => onCursorCommit(Number((e.target as HTMLInputElement).value))}
                 onTouchEnd={(e) => onCursorCommit(Number((e.target as HTMLInputElement).value))}
                 onKeyUp={(e) => onCursorCommit(Number((e.target as HTMLInputElement).value))} />
        ) : (
          <input type="range" data-vt-timescrub-slider
                 min={0} max={maxHours} step={1}
                 value={hoursBack}
                 aria-label="Hours back from now"
                 onChange={(e) => setHoursBack(Number(e.target.value))}
                 onMouseUp={(e) => onSliderCommit(Number((e.target as HTMLInputElement).value))}
                 onTouchEnd={(e) => onSliderCommit(Number((e.target as HTMLInputElement).value))}
                 onKeyUp={(e) => onSliderCommit(Number((e.target as HTMLInputElement).value))} />
        )}
      </div>

      <div className="vt-timescrub-status" role="status" aria-live="polite">
        {loading && "Loading…"}
        {!loading && isWindowMode && windowError && <span className="vt-timescrub-error">{windowError}</span>}
        {!loading && isWindowMode && !windowError && windowData && (
          <>
            {windowData.hexes.length} {windowData.kind === "vessels" ? "vessels" : "aircraft"}{windowData.hexes_seen > windowData.hexes.length && ` (of ${windowData.hexes_seen})`}
            {" · "}{windowData.step_sec === 0 ? "full fidelity" : `${windowData.step_sec / 60}min step`}
            {windowData.note && ` · ${windowData.note}`}
          </>
        )}
        {!loading && !isWindowMode && error && <span className="vt-timescrub-error">{error}</span>}
        {!loading && !isWindowMode && !error && snap && (
          <>
            {snap.count} point{snap.count === 1 ? "" : "s"}
            {snap.capped && " (capped)"}
            {snap.viewport_filtered && snap.count_dropped_offscreen > 0 && ` · ${snap.count_dropped_offscreen} off-screen`}
            {" · "}{snap.provenance}
          </>
        )}
      </div>
      <div className="vt-timescrub-note">
        {isWindowMode
          ? "Historical replay from our own archive — not live. Drag scrubs a preloaded window locally (no per-drag network call); the window and step selectors control how much history loads and at what resolution. Dated imagery layers (night lights, NDVI, soil moisture…) follow this clock to their nearest available day."
          : <>Historical replay from our own archive — not live. Window: last {Math.round(maxHours / 24)} days.
             Dated imagery layers (night lights, NDVI, soil moisture…) follow this clock to their nearest available day.</>}
      </div>
    </div>
  );
}
