// ALTITUDE / TIME PROFILE — the handoff's bottom bar (design_handoff_
// flight_track_3d §3, installed 2026-07-20): the 2D twin of the 3D
// curtain. Terrain profile under the track (green fill), the altitude
// line (blue), the AGL band between them, 1px white playhead + dot riding
// the altitude line, play/pause + live UTC clock, and pointer scrub.
//
// LIVE-SITE ADAPTATION (per the handoff's own note): "now" = the latest
// ADS-B point — the panel opens pinned to live and new fixes extend the
// chart; the scrubber replays history (replay pauses live-pinning, play
// advances the replay, reaching the end snaps back to live). The 3D
// marker, the flight-card readouts and this playhead all read ONE clock
// (timeRef) so they can never disagree.
//
// HONESTY: the chart plots REAL recorded values — archived fixes densified
// linearly (no invented curves), altitude gaps break the line and the AGL
// band (the terrain floor continues — position history is real), terrain
// heights are the DEM reads under the track, unexaggerated. Grid lines are
// unit-aware via lib/units.ts (imperial 5,000 ft steps / metric 2,000 m).
//
// PERF: path geometry rebuilds only when the track changes (15–30s
// cadence); the playhead/clock are DOM-ref writes on a rAF that runs ONLY
// while replaying (live mode updates on data ticks) — zero idle frames.

import { useEffect, useMemo, useRef, useState } from "react";
import { getUnits, subscribeUnits } from "@/lib/units";
import { sampleAt, type TrackSample } from "@/lib/air/trackModel";
import { applyPanelPos, applyPanelScale, clampScale, clearPanelPos, getPanelPrefs, panelDragProps, savePanelPrefs, stepPanelScale } from "@/lib/panelLayout";

export interface FlightClock {
  /** epoch seconds the marker/card/profile display right now. */
  t: number;
  /** true = pinned to the newest fix (live); false = replaying history. */
  live: boolean;
  playing: boolean;
}

export interface FlightProfilePanelProps {
  samples: TrackSample[];
  /** REAL terrain meters under each sample (unexaggerated), same length. */
  groundM: Float32Array | null;
  altMin: number;
  altMax: number;
  /** shared playback clock — parent reads it every glide tick. */
  clockRef: React.MutableRefObject<FlightClock>;
  /** notify parent on user interaction (scrub/play) so the 3D marker
   *  repaints immediately instead of waiting for the next glide tick. */
  onClockChange?: () => void;
  /** phone: expanding the chart should minimize the card bottom-sheet. */
  onPhoneExpand?: () => void;
  /** track source note for the title row (honesty label). */
  sourceNote?: string;
}

const CW = 1000;
const CH = 140;
const PAD_T = 8;
const PAD_B = 14;
const FT = 0.3048;

/** replay speed: a full pass over the archived window takes ~80s (the
 *  prototype's 14× over its 1150s demo track ≈ the same feel). */
export function replaySpeed(spanSec: number): number {
  return Math.max(1, spanSec / 80);
}

const utc = (tSec: number) => new Date(tSec * 1000).toISOString().slice(11, 19) + "Z";

export default function FlightProfilePanel({
  samples,
  groundM,
  altMin,
  altMax,
  clockRef,
  onClockChange,
  onPhoneExpand,
  sourceNote,
}: FlightProfilePanelProps) {
  const [playing, setPlaying] = useState(false);
  const [live, setLive] = useState(true);
  // desktop remembers collapsed/expanded (layout memory, human 2026-07-20);
  // phones stay collapsed by default
  const [expanded, setExpanded] = useState<boolean>(() => {
    if (typeof window === "undefined") return true;
    if (window.innerWidth < 768) return false;
    const saved = getPanelPrefs("flight-profile").min;
    return saved == null ? true : !saved;
  });
  const [, setUnitsTick] = useState(0);
  useEffect(() => subscribeUnits(() => setUnitsTick((v) => v + 1)), []);

  // drag/lock/remembered placement (panelLayout lib — same chrome as the
  // cards and the nav cluster)
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [locked, setLocked] = useState<boolean>(() => !!getPanelPrefs("flight-profile").locked);
  const lockedRef = useRef(locked);
  lockedRef.current = locked;
  const drag = useMemo(
    () => panelDragProps("flight-profile", () => rootRef.current, () => lockedRef.current,
      { defaultOrigin: "bottom left" }),
    [],
  );
  const toggleLock = () =>
    setLocked((v) => { const n = !v; savePanelPrefs("flight-profile", { locked: n }); return n; });
  // panel SCALE (human 2026-07-20: "scale them up or down to fit your
  // screen") — remembered CSS transform; the bottom bar grows upward from
  // its bottom-left anchor so it never sinks below the viewport.
  const [pScale, setPScale] = useState<number>(() => clampScale(getPanelPrefs("flight-profile").scale));
  const bumpScale = (dir: number) => setPScale(stepPanelScale("flight-profile", dir));
  useEffect(() => {
    const el = rootRef.current;
    if (el && !applyPanelPos(el, "flight-profile")) clearPanelPos(el);
    applyPanelScale(el, "flight-profile", "bottom left");
  }, [expanded, pScale]);

  const wrapRef = useRef<HTMLDivElement | null>(null);
  const playheadRef = useRef<SVGGElement | null>(null);
  const phDotRef = useRef<SVGCircleElement | null>(null);
  const clockElRef = useRef<HTMLSpanElement | null>(null);
  const rafRef = useRef<number | null>(null);

  const t0 = samples.length ? samples[0].t : 0;
  const t1 = samples.length ? samples[samples.length - 1].t : 0;
  const span = Math.max(1, t1 - t0);

  // LIVE EDGE (human 2026-07-20: the banner "should continuously update …
  // as it gets it from adsb"): while pinned live the time axis runs to NOW,
  // ticking every second — the sample paths (built once in the t0..t1
  // domain) compress horizontally via a group transform, and a DASHED tail
  // extends from the last real fix to the right edge at the held altitude
  // (the 3D glide tail's honesty model: position dead-reckoned, altitude
  // held at the last broadcast, drawn dashed — never passed off as data).
  const dataGRef = useRef<SVGGElement | null>(null);
  const tailRef = useRef<SVGLineElement | null>(null);
  const axisEndRef = useRef<HTMLSpanElement | null>(null);
  const liveEdgeRef = useRef(t1);
  const extOf = () => Math.max(span, liveEdgeRef.current - t0);
  /** time → x under the DISPLAY domain (t0 → live edge). */
  const XT = (t: number) => ((t - t0) / extOf()) * CW;

  // y domain: REAL meters, headroom like the prototype (alt·1.08 → grid step)
  const yMaxM = useMemo(() => {
    const top = Math.max(altMax, 1) * 1.08;
    return Math.ceil(top / 500) * 500;
  }, [altMax]);

  const X = (t: number) => ((t - t0) / span) * CW;
  const Y = (m: number) => PAD_T + (1 - m / yMaxM) * (CH - PAD_T - PAD_B);

  // path geometry — rebuilt only when the track data changes
  const paths = useMemo(() => {
    if (samples.length < 2) return null;
    const g = (i: number) => (groundM && i < groundM.length ? groundM[i] : 0);
    let ter = `M0 ${CH}`;
    for (let i = 0; i < samples.length; i++) {
      ter += ` L${X(samples[i].t).toFixed(1)} ${Y(g(i)).toFixed(1)}`;
    }
    ter += ` L${CW} ${CH} Z`;
    // altitude line + AGL band per contiguous non-gap run (honest breaks)
    let alt = "";
    let band = "";
    let run: number[] = [];
    const flush = () => {
      if (run.length < 2) { run = []; return; }
      alt += run.map((i, k) =>
        `${k ? "L" : "M"}${X(samples[i].t).toFixed(1)} ${Y(samples[i].altM).toFixed(1)}`).join(" ") + " ";
      band += run.map((i, k) =>
        `${k ? "L" : "M"}${X(samples[i].t).toFixed(1)} ${Y(samples[i].altM).toFixed(1)}`).join(" ")
        + " " + [...run].reverse().map((i) =>
          `L${X(samples[i].t).toFixed(1)} ${Y(g(i)).toFixed(1)}`).join(" ") + " Z ";
      run = [];
    };
    for (let i = 0; i < samples.length; i++) {
      if (samples[i].gap) flush();
      else run.push(i);
    }
    flush();
    return { ter, alt: alt.trim(), band: band.trim() };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [samples, groundM, yMaxM, t0, span]);

  // unit-aware gridlines: 5,000 ft (imperial) / 2,000 m (metric)
  const grid = useMemo(() => {
    const imperial = getUnits() === "imperial";
    const stepM = imperial ? 5000 * FT : 2000;
    const out: { y: number; label: string }[] = [];
    for (let m = stepM; m < yMaxM; m += stepM) {
      out.push({
        y: Y(m),
        label: imperial ? `${Math.round(m / FT / 1000)}k ft` : `${(m / 1000).toFixed(0)} km`,
      });
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [yMaxM, getUnits()]);

  // ── the shared clock ───────────────────────────────────────────────────
  const paintHead = (tSec: number) => {
    const ph = playheadRef.current;
    const dot = phDotRef.current;
    const ck = clockElRef.current;
    if (ph) ph.setAttribute("transform", `translate(${XT(tSec)},0)`);
    if (dot) {
      const s = sampleAt(samples, tSec);
      dot.setAttribute("cy", s && !s.gap ? String(Y(s.altM)) : String(Y(0)));
      dot.style.display = s && !s.gap ? "" : "none";
    }
    if (ck) ck.textContent = utc(tSec);
  };

  /** One live-edge tick: extend the axis to NOW, compress the sample
   *  paths, place the dashed tail + playhead at the edge, tick the clock. */
  const paintLiveEdge = () => {
    if (samples.length < 2) return;
    const nowSec = Date.now() / 1000;
    liveEdgeRef.current = Math.max(t1, nowSec);
    const sx = span / extOf();
    dataGRef.current?.setAttribute("transform", `scale(${sx} 1)`);
    const lastS = samples[samples.length - 1];
    const tail = tailRef.current;
    if (tail) {
      if (!lastS.gap && liveEdgeRef.current > t1 + 1) {
        const y = Y(lastS.altM).toFixed(1);
        tail.setAttribute("x1", XT(t1).toFixed(1));
        tail.setAttribute("x2", String(CW));
        tail.setAttribute("y1", y);
        tail.setAttribute("y2", y);
        tail.style.display = "";
      } else {
        tail.style.display = "none";
      }
    }
    const ph = playheadRef.current, dot = phDotRef.current, ck = clockElRef.current;
    if (ph) ph.setAttribute("transform", `translate(${CW},0)`);
    if (dot) {
      dot.setAttribute("cy", String(Y(lastS.gap ? 0 : lastS.altM)));
      dot.style.display = lastS.gap ? "none" : "";
    }
    if (ck) ck.textContent = utc(nowSec);
    if (axisEndRef.current) axisEndRef.current.textContent = utc(nowSec);
  };

  // live mode: the edge ticks every second (clock even while collapsed —
  // the SVG refs are simply absent then); scrub/replay stops the ticking
  useEffect(() => {
    if (!live) return;
    paintLiveEdge();
    const iv = window.setInterval(() => { if (!document.hidden) paintLiveEdge(); }, 1000);
    return () => window.clearInterval(iv);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live, expanded, t1, paths]);

  // pin to the newest fix whenever the track extends
  useEffect(() => {
    const c = clockRef.current;
    if (c.live) {
      c.t = t1;
      paintLiveEdge();
      onClockChange?.();
    } else {
      paintHead(c.t); // re-place after a geometry rebuild
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [t1, paths]);

  // replay loop (rAF only while playing)
  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const speed = replaySpeed(span);
    const frame = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const c = clockRef.current;
      c.t += dt * speed;
      if (c.t >= t1) {
        // reached "now" → snap back to live (the handoff's live-site rule)
        c.t = t1;
        c.live = true;
        c.playing = false;
        setPlaying(false);
        setLive(true);
      }
      paintHead(c.t);
      onClockChange?.();
      if (clockRef.current.playing) rafRef.current = requestAnimationFrame(frame);
    };
    rafRef.current = requestAnimationFrame(frame);
    return () => { if (rafRef.current != null) cancelAnimationFrame(rafRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, span, t1]);

  const togglePlay = () => {
    const c = clockRef.current;
    if (playing) {
      c.playing = false;
      setPlaying(false);
    } else {
      if (c.live) c.t = t0; // play from the start when pinned to live
      c.live = false;
      c.playing = true;
      setLive(false);
      setPlaying(true);
    }
    onClockChange?.();
  };

  // Space = play/pause (prototype binding; profile owns it while mounted)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== " " && e.code !== "Space") return;
      const t = e.target;
      if (t instanceof HTMLElement && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      e.preventDefault();
      togglePlay();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, t0, t1]);

  // scrub: pointer down/drag sets time and pauses (prototype-exact)
  const scrubbing = useRef(false);
  const scrubTo = (e: React.PointerEvent) => {
    const el = wrapRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const u = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const c = clockRef.current;
    // pointer maps over the DISPLAY domain (t0 → live edge); the dashed
    // tail region carries no data, so times past the last fix clamp to it
    c.t = Math.min(t1, t0 + u * extOf());
    c.live = false;
    c.playing = false;
    if (playing) setPlaying(false);
    if (live) setLive(false);
    paintHead(c.t);
    onClockChange?.();
  };
  const onScrubDown = (e: React.PointerEvent) => {
    scrubbing.current = true;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    scrubTo(e);
  };
  const onScrubMove = (e: React.PointerEvent) => { if (scrubbing.current) scrubTo(e); };
  const onScrubUp = () => { scrubbing.current = false; };

  const backToLive = () => {
    const c = clockRef.current;
    c.live = true;
    c.playing = false;
    c.t = t1;
    setPlaying(false);
    setLive(true);
    paintHead(t1);
    onClockChange?.();
  };

  if (samples.length < 2 || !paths) return null;

  return (
    <div ref={rootRef} className={`vt-flight-profile${expanded ? "" : " vt-flight-profile-min"}`} data-vt-flight-profile>
      <div className="vt-flight-profile-top" {...drag}
           style={{ cursor: locked ? undefined : "grab", touchAction: "none" }}
           title={locked ? "Position locked" : "Drag to move · double-click to reset · spot is remembered"}>
        <button className="vt-flight-play" data-vt-flight-play onClick={togglePlay}
                aria-label={playing ? "Pause replay" : "Replay track"} title="Play / pause — Space">
          {playing ? (
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z" /></svg>
          ) : (
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>
          )}
        </button>
        <div className="vt-flight-profile-title">
          ALTITUDE / TIME <span className="vt-flight-clock" ref={clockElRef}>{utc(clockRef.current.live ? t1 : clockRef.current.t)}</span>
          <span className="vt-flight-profile-src"> · {sourceNote || "ADS-B track (our archive + live)"}</span>
          {!live && (
            <button className="vt-flight-live-btn" onClick={backToLive} title="Snap back to the latest position">
              ● LIVE
            </button>
          )}
        </div>
        <div className="vt-flight-legend">
          <span><i className="alt" />ALTITUDE</span>
          <span><i className="terr" />TERRAIN</span>
          <span><b />AGL BAND</span>
        </div>
        <button className="vt-flight-profile-toggle" data-vt-scale-down aria-label="Shrink panel"
                title="Smaller (size is remembered)" onClick={() => bumpScale(-1)}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="10" cy="10" r="6" /><path d="M14.5 14.5 20 20M7.5 10h5" /></svg>
        </button>
        <button className="vt-flight-profile-toggle" data-vt-scale-up aria-label="Enlarge panel"
                title="Bigger (size is remembered)" onClick={() => bumpScale(1)}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="10" cy="10" r="6" /><path d="M14.5 14.5 20 20M10 7.5v5M7.5 10h5" /></svg>
        </button>
        <button className={`vt-flight-profile-toggle vt-lock-btn${locked ? " on" : ""}`} aria-pressed={locked}
                aria-label={locked ? "Unlock panel position" : "Lock panel position"}
                title={locked ? "Position locked — click to unlock" : "Lock position"}
                onClick={toggleLock}>
          {locked ? (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M8 11V7a4 4 0 0 1 8 0v4" /><rect x="5" y="11" width="14" height="9" rx="1.5" /></svg>
          ) : (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M8 11V7a4 4 0 0 1 7.6-1.7" /><rect x="5" y="11" width="14" height="9" rx="1.5" /></svg>
          )}
        </button>
        <button className="vt-flight-profile-toggle" aria-expanded={expanded}
                aria-label={expanded ? "Collapse altitude profile" : "Expand altitude profile"}
                onClick={() => {
                  const v = !expanded;
                  setExpanded(v);
                  savePanelPrefs("flight-profile", { min: !v });
                  if (v && window.innerWidth < 768) onPhoneExpand?.();
                }}>
          {expanded ? "▾" : "▴"}
        </button>
      </div>
      {expanded && (
        <>
          <div className="vt-flight-chartwrap" ref={wrapRef}
               onPointerDown={onScrubDown} onPointerMove={onScrubMove}
               onPointerUp={onScrubUp} onPointerCancel={onScrubUp}>
            <svg className="vt-flight-chart" viewBox={`0 0 ${CW} ${CH}`} preserveAspectRatio="none">
              {grid.map((g) => (
                <g key={g.y}>
                  <line x1="0" y1={g.y} x2={CW} y2={g.y} stroke="rgba(130,170,230,.12)" strokeDasharray="3 5" />
                  <text x="6" y={g.y - 3} fontSize="8.5" fill="#8fa3bf" fontFamily="var(--font-mono)">{g.label}</text>
                </g>
              ))}
              {/* sample paths live in the t0..t1 domain; the group transform
                  compresses them as the live edge extends the axis to NOW
                  (non-scaling strokes keep line widths true) */}
              <g ref={dataGRef}>
                <path d={paths.band} fill="rgba(77,163,255,.16)" />
                <path d={paths.ter} fill="rgba(56,84,52,.55)" stroke="#6b8f5e" strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
                <path d={paths.alt} fill="none" stroke="#4da3ff" strokeWidth="2" vectorEffect="non-scaling-stroke" />
              </g>
              {/* dashed dead-reckoned tail: last real fix → now, altitude
                  held at the last broadcast (never drawn as solid data) */}
              <line ref={tailRef} stroke="#4da3ff" strokeWidth="2" strokeDasharray="4 5"
                    opacity="0.55" style={{ display: "none" }} />
              <g ref={playheadRef}>
                <line x1="0" y1="0" x2="0" y2={CH} stroke="rgba(223,232,245,.85)" strokeWidth="1" />
                <circle ref={phDotRef} cx="0" cy="0" r="4" fill="#fff" stroke="#4da3ff" strokeWidth="2" />
              </g>
            </svg>
          </div>
          <div className="vt-flight-axis">
            <span>{utc(t0)}</span>
            {/* right edge = the live clock while pinned (ticks via ref);
                a paused/replaying chart shows the last real fix time */}
            <span ref={axisEndRef}>{utc(t1)}</span>
          </div>
        </>
      )}
    </div>
  );
}
