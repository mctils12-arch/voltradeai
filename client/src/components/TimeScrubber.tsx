import { useEffect, useRef, useState } from "react";
import { Play, Pause, X, Clock, AlertTriangle, ChevronDown, ChevronUp } from "lucide-react";
import type maplibregl from "maplibre-gl";
// EARTH TWIN E1: this panel IS the global time axis's UI — every committed
// scrub publishes the instant so dated layers (GIBS imagery) follow the same
// moment the archive replay shows. Closing the panel returns the world to
// LIVE. (lib/timeAxis; datamap subscribes.)
import { setTimeAxis } from "@/lib/timeAxis";
import { subscribeUnits } from "@/lib/units";
import {
  FleetReplayController, type ReplayState, type ReplayKind,
} from "@/lib/air/fleetReplayController";
import {
  fmtSeparation, fmtUtcTime, sideLabel, approachKey, confidenceText,
  type CloseApproachEntry,
} from "@/lib/air/closeApproachView";

/**
 * TimeScrubber — ANALYST CONSOLE W3: "pick a window, scrub, watch the world
 * move" over the archives we already record. Lazy-loaded from datamap.tsx
 * (zero-cost-when-off, same pattern as AnalystPane) — a closed panel loads
 * no code and issues no requests.
 *
 * Two modes, chosen by the selected layer:
 * - AIRCRAFT / VESSELS (TIME MACHINE v2 T-2/T-3/T-4 + FLIGHT PROGRAM replay,
 *   earth_twin_program.md; human directive 2026-09-28 "rewind and see ALL
 *   planes in my field of view with tracks and curtains at any pan/tilt/
 *   zoom; if two planes came close you can see it"): a FleetReplayController
 *   (lib/air/fleetReplayController.ts) owns the replay — the window read
 *   FOLLOWS THE CAMERA TARGET (pitched/horizon-clamped footprint + margin,
 *   re-read only when the view leaves what was fetched, abortable, old data
 *   drawn until new lands), every plane's curtain/line/head is drawn by ONE
 *   batched GL layer with device-tier LOD + hysteresis, and the playhead is
 *   a lerped target driven by the frameCore loop (no per-tick fetch, no
 *   per-tick jump). Aircraft windows also list CLOSE APPROACHES (server/
 *   closeApproach.ts: < 5 nm / 1,000 ft per our recorded ADS-B data — never
 *   an official report); clicking one frames the pair and highlights them.
 *   Vessels have no altitude (paths + heads, no curtains, no approaches).
 * - EVERYTHING ELSE (trains/fires/alerts/gauges): unchanged single-instant
 *   snapshot model (GET /api/data/snapshot, server/queryEngine.ts's
 *   querySnapshot) — these layers have no window-read backend yet.
 *
 * Both modes are RAW overlays (no ladder gate) and honestly labeled as
 * historical replay so neither is mistaken for a live layer.
 *
 * The map instance is owned by the PARENT (datamap.tsx); this component only
 * adds/removes its OWN sources/layers ("time-scrubber-*", "fleet-replay-3d")
 * on that instance — never touches any live layer's state.
 */

const SNAPSHOT_SOURCE = "time-scrubber-snapshot";
const SNAPSHOT_LAYER = "time-scrubber-points";
const PLAY_INTERVAL_MS = 900;
const DEFAULT_HOURS_BACK = 24;
const FALLBACK_MAX_HOURS = 7 * 24; // reconciled with the server's stated window once known
/** the global time axis follows the replay at this granularity (dated
 *  imagery is day-granular; per-frame publishes would thrash followers) */
const AXIS_PUBLISH_SEC = 60;
/** …and at most this often in wall-clock time while the playhead moves */
const AXIS_PUBLISH_MIN_MS = 1000;

const LAYERS: Array<{ value: string; label: string }> = [
  { value: "aircraft", label: "Aircraft" },
  { value: "vessels", label: "Vessels" },
  { value: "trains", label: "Trains" },
  { value: "fires", label: "Fire detections" },
  { value: "alerts", label: "NWS alerts" },
  { value: "gauges", label: "River gauges" },
];

// TIME MACHINE v2 T-2/T-4: layers with a window-read backend (same server
// route, kind= query param). Both replay through the fleet controller.
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

function fmtUtc(iso: string): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toISOString().slice(0, 16).replace("T", " ") + " UTC";
}

/** Law V: how old the drawn window read is */
function ageText(fetchedAtMs: number | null, nowMs: number): string {
  if (fetchedAtMs == null) return "";
  const s = Math.max(0, Math.round((nowMs - fetchedAtMs) / 1000));
  return s < 5 ? "read just now" : s < 120 ? `read ${s}s ago` : `read ${Math.round(s / 60)} min ago`;
}

export default function TimeScrubber({ map, onClose }: {
  map: maplibregl.Map | null;
  onClose: () => void;
}) {
  const [layer, setLayer] = useState("aircraft");
  const isWindowMode = WINDOW_KINDS.has(layer);

  // Snapshot-mode state (trains/fires/alerts/gauges) — unchanged.
  const [hoursBack, setHoursBack] = useState(DEFAULT_HOURS_BACK);
  const [maxHours, setMaxHours] = useState(FALLBACK_MAX_HOURS);
  const [snap, setSnap] = useState<SnapshotEnvelope | null>(null);

  // Window-mode state — owned by the fleet replay controller.
  const [windowSec, setWindowSec] = useState(DEFAULT_WINDOW_SEC);
  const [stepSec, setStepSec] = useState(DEFAULT_STEP_SEC);
  const [replay, setReplay] = useState<ReplayState | null>(null);
  const [playhead, setPlayhead] = useState<{ value: number; target: number; playing: boolean } | null>(null);
  const ctrlRef = useRef<FleetReplayController | null>(null);
  const axisRef = useRef<number | null>(null);
  const axisPubAtRef = useRef(0);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const [, setUnitsTick] = useState(0);
  const [caOpen, setCaOpen] = useState(false);
  const [nowTick, setNowTick] = useState(Date.now());

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

  // ── window mode: the fleet replay controller's lifetime ─────────────────
  useEffect(() => {
    if (!map || !isWindowMode) return;
    const ctrl = new FleetReplayController({
      map,
      getObstruction: () => panelRef.current?.getBoundingClientRect() ?? null,
      onState: (s) => setReplay(s),
      onPlayhead: (value, target, isPlaying) => {
        setPlayhead({ value, target, playing: isPlaying });
        // the world clock follows the replay: at most once a second while
        // the playhead moves (every axis change re-renders the map page's
        // HISTORICAL badge + dated-layer dates), and once more where it
        // settles — the old per-commit/per-tick cadence, never per frame
        const last = axisRef.current;
        const settled = !isPlaying && Math.abs(value - target) < 0.5;
        const wall = Date.now();
        const due = wall - axisPubAtRef.current >= AXIS_PUBLISH_MIN_MS && (last == null || Math.abs(value - last) >= AXIS_PUBLISH_SEC);
        if (last == null || due || (settled && last !== value)) {
          axisRef.current = value;
          axisPubAtRef.current = wall;
          const toSec = Math.floor(nowRef.current / 1000);
          setTimeAxis(value >= toSec - 1 ? { mode: "live" } : { mode: "historical", atMs: Math.round(value * 1000) });
        }
      },
    });
    ctrlRef.current = ctrl;
    return () => {
      ctrl.dispose();
      if (ctrlRef.current === ctrl) ctrlRef.current = null;
      setReplay(null);
      setPlayhead(null);
    };
  }, [map, isWindowMode]);

  // window/step/kind → the controller's query (it re-reads from the CURRENT
  // camera target; the cursor starts at the window's end, i.e. "now")
  useEffect(() => {
    if (!isWindowMode) return;
    const to = Math.floor(nowRef.current / 1000);
    ctrlRef.current?.setQuery({ kind: layer as ReplayKind, fromSec: to - windowSec, toSec: to, stepSec });
    setTimeAxis({ mode: "live" });
    axisRef.current = null;
  }, [layer, isWindowMode, windowSec, stepSec, map]);

  // units preference re-renders the separation readouts
  useEffect(() => subscribeUnits(() => setUnitsTick((v) => v + 1)), []);
  // Law V age readout refresh
  useEffect(() => {
    if (!isWindowMode) return;
    const iv = setInterval(() => setNowTick(Date.now()), 5000);
    return () => clearInterval(iv);
  }, [isWindowMode]);

  // Snapshot mode: initial fetch on open + whenever the layer changes.
  useEffect(() => {
    setPlaying(false);
    if (!isWindowMode) fetchSnapshot(hoursBack, layer);
    else clearMapLayer();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layer, isWindowMode]);

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

  // Cleanup the map layer when the panel closes (unmounts) — and return the
  // global time axis to LIVE: the panel is the axis's only UI, so a closed
  // panel must never leave the world silently stuck in the past.
  useEffect(() => () => { clearMapLayer(); setTimeAxis({ mode: "live" }); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const onSliderCommit = (hb: number) => {
    setHoursBack(hb);
    setPlaying(false);
    fetchSnapshot(hb, layer);
  };

  const onCursor = (cur: number) => {
    const c = ctrlRef.current;
    if (!c) return;
    if (c.isPlaying()) c.setPlaying(false);
    c.setPlayheadTarget(cur);
  };

  const data = replay?.data ?? null;
  const toSec = Math.floor(nowRef.current / 1000);
  const fromSec = toSec - windowSec;
  const winPlaying = !!playhead?.playing;
  const atIso = new Date(nowRef.current - hoursBack * 3600_000).toISOString();
  const cursorValue = playhead?.value ?? toSec;
  const cursorIso = new Date(cursorValue * 1000).toISOString();
  const atLive = isWindowMode ? cursorValue >= toSec - 1 && !winPlaying : hoursBack === 0;
  const approaches: CloseApproachEntry[] = (layer === "aircraft" && data?.closeApproaches) || [];
  const caMeta = layer === "aircraft" ? data?.closeApproachesMeta : undefined;
  const focusKey = replay?.focusKey ?? null;
  const lod = replay?.lod;
  // collapsed by default (the panel shares the left edge with the Legend
  // card); a focused approach keeps it open
  const caExpanded = caOpen || !!focusKey;

  return (
    <div className="vt-timescrub-panel" data-vt-timescrub-panel role="dialog" aria-label="Time scrubber" ref={panelRef}>
      <div className="vt-timescrub-header">
        <span className="vt-timescrub-title"><Clock size={15} /> Time Machine</span>
        <button aria-label="Close time machine" onClick={() => { setPlaying(false); ctrlRef.current?.setPlaying(false); clearMapLayer(); onClose(); }}>
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
        {atLive ? "Now" : isWindowMode ? fmtUtc(cursorIso) : fmtUtc(atIso)}
      </div>

      <div className="vt-timescrub-controls">
        <button className="vt-timescrub-play" data-vt-timescrub-play
                aria-label={(isWindowMode ? winPlaying : playing) ? "Pause playback" : "Play playback"}
                aria-pressed={isWindowMode ? winPlaying : playing}
                disabled={isWindowMode ? !data : (hoursBack === 0 && !playing)}
                title={isWindowMode && atLive ? "Play the loaded window from its start" : undefined}
                onClick={() => {
                  if (isWindowMode) ctrlRef.current?.setPlaying(!winPlaying);
                  else setPlaying((v) => !v);
                }}>
          {(isWindowMode ? winPlaying : playing) ? <Pause size={16} /> : <Play size={16} />}
        </button>
        {isWindowMode ? (
          <input type="range" data-vt-timescrub-slider
                 min={fromSec} max={toSec} step={1}
                 value={Math.round(playhead?.target ?? toSec)}
                 disabled={!data}
                 aria-label="Replay time within the loaded window"
                 onChange={(e) => onCursor(Number(e.target.value))} />
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
        {isWindowMode && replay?.loading && !data && "Loading…"}
        {isWindowMode && replay?.error && <span className="vt-timescrub-error">{replay.error}{data ? " — showing the previous read" : ""}</span>}
        {isWindowMode && !replay?.error && data && (
          <>
            {data.hexes.length} {data.kind === "vessels" ? "vessels" : "aircraft"}{data.hexes_seen > data.hexes.length && ` (of ${data.hexes_seen})`} in view
            {" · "}{data.step_sec === 0 ? "full fidelity" : `${data.step_sec / 60}min step`}
            {lod && lod.tracks > 0 && ` · ${lod.full} curtains, ${lod.thin} lines, ${lod.headOnly} heads-only`}
            {" · "}{ageText(replay?.fetchedAtMs ?? null, nowTick)}{replay?.loading && " · updating view…"}
            {data.note && ` · ${data.note}`}
          </>
        )}
        {!isWindowMode && loading && "Loading…"}
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

      {isWindowMode && layer === "aircraft" && data && (
        <div className="vt-timescrub-ca" data-vt-timescrub-ca>
          <button className="vt-timescrub-ca-head" data-vt-timescrub-ca-toggle
                  aria-expanded={caExpanded}
                  onClick={() => setCaOpen((v) => !v)}>
            <span><AlertTriangle size={13} /> Close approaches</span>
            <span className="vt-timescrub-ca-count">
              {caMeta ? `${caMeta.found}${caMeta.capped ? ` (top ${caMeta.returned})` : ""}` : approaches.length}
              {caExpanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            </span>
          </button>
          {!caExpanded ? null : approaches.length === 0 ? (
            <div className="vt-timescrub-ca-empty">
              {caMeta
                ? `None below 5 nm / 1,000 ft among ${caMeta.evaluated_hexes} aircraft in this window and view${caMeta.partial_scan ? " (partial scan — see note)" : ""}.`
                : "Close-approach scan unavailable for this read."}
            </div>
          ) : (
            <ul className="vt-timescrub-ca-list" data-vt-timescrub-ca-list>
              {approaches.map((ca) => {
                const k = approachKey(ca);
                const on = focusKey === k;
                return (
                  <li key={k}>
                    <button className="vt-timescrub-ca-row" aria-pressed={on}
                            title={`${confidenceText(ca.confidence)}. ${ca.basis}`}
                            onClick={() => ctrlRef.current?.focusApproach(on ? null : ca)}>
                      <span className="vt-ca-time">{fmtUtcTime(ca.t)}</span>
                      <span className="vt-ca-pair">{sideLabel(ca.a, ca.ca)} ↔ {sideLabel(ca.b, ca.cb)}</span>
                      <span className="vt-ca-sep">{fmtSeparation(ca.horizNm, ca.vertFt)}</span>
                      <span className={`vt-ca-conf vt-ca-conf-${ca.confidence}`}>{ca.confidence}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
          {caExpanded && focusKey && (
            <div className="vt-timescrub-ca-focus">
              <span>
                {replay?.focusFullRes === "loading" && "Loading the pair's full-fidelity archived tracks…"}
                {replay?.focusFullRes === "ok" && "Pair shown at full archived fidelity; other aircraft dimmed."}
                {replay?.focusFullRes === "failed" && "Full-fidelity tracks unavailable — showing the window's decimated tracks."}
              </span>
              <button onClick={() => ctrlRef.current?.focusApproach(null)}>Show all</button>
            </div>
          )}
          {caExpanded && (
            <div className="vt-timescrub-ca-basis">
              Separation below 5 nm / 1,000 ft per our recorded ADS-B data — not an official loss-of-separation report.
              En-route minima; terminal, parallel-approach, formation and same-airport traffic is routinely closer by design.
              Map readouts marked ≈ are live interpolations; the listed value is the minimum on archived fixes.
            </div>
          )}
        </div>
      )}

      <div className="vt-timescrub-note">
        {isWindowMode
          ? "Historical replay from our own archive — not live. The replay follows your view (pan, tilt, zoom) and re-reads only when you leave the loaded area; heads move linearly between real recorded fixes and never across a gap. Dated imagery layers (night lights, NDVI, soil moisture…) follow this clock to their nearest available day."
          : <>Historical replay from our own archive — not live. Window: last {Math.round(maxHours / 24)} days.
             Dated imagery layers (night lights, NDVI, soil moisture…) follow this clock to their nearest available day.</>}
      </div>
    </div>
  );
}
