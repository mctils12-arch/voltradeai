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
  const [expanded, setExpanded] = useState<boolean>(() =>
    typeof window !== "undefined" ? window.innerWidth >= 768 : true);
  const [, setUnitsTick] = useState(0);
  useEffect(() => subscribeUnits(() => setUnitsTick((v) => v + 1)), []);

  const wrapRef = useRef<HTMLDivElement | null>(null);
  const playheadRef = useRef<SVGGElement | null>(null);
  const phDotRef = useRef<SVGCircleElement | null>(null);
  const clockElRef = useRef<HTMLSpanElement | null>(null);
  const rafRef = useRef<number | null>(null);

  const t0 = samples.length ? samples[0].t : 0;
  const t1 = samples.length ? samples[samples.length - 1].t : 0;
  const span = Math.max(1, t1 - t0);

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
    if (ph) ph.setAttribute("transform", `translate(${X(tSec)},0)`);
    if (dot) {
      const s = sampleAt(samples, tSec);
      dot.setAttribute("cy", s && !s.gap ? String(Y(s.altM)) : String(Y(0)));
      dot.style.display = s && !s.gap ? "" : "none";
    }
    if (ck) ck.textContent = utc(tSec);
  };

  // live mode: pin to the newest fix whenever the track extends
  useEffect(() => {
    const c = clockRef.current;
    if (c.live) {
      c.t = t1;
      paintHead(t1);
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
    c.t = t0 + u * span;
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
    <div className={`vt-flight-profile${expanded ? "" : " vt-flight-profile-min"}`} data-vt-flight-profile>
      <div className="vt-flight-profile-top">
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
        <button className="vt-flight-profile-toggle" aria-expanded={expanded}
                aria-label={expanded ? "Collapse altitude profile" : "Expand altitude profile"}
                onClick={() => {
                  const v = !expanded;
                  setExpanded(v);
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
              <path d={paths.band} fill="rgba(77,163,255,.16)" />
              <path d={paths.ter} fill="rgba(56,84,52,.55)" stroke="#6b8f5e" strokeWidth="1.2" />
              <path d={paths.alt} fill="none" stroke="#4da3ff" strokeWidth="2" />
              <g ref={playheadRef}>
                <line x1="0" y1="0" x2="0" y2={CH} stroke="rgba(223,232,245,.85)" strokeWidth="1" />
                <circle ref={phDotRef} cx="0" cy="0" r="4" fill="#fff" stroke="#4da3ff" strokeWidth="2" />
              </g>
            </svg>
          </div>
          <div className="vt-flight-axis">
            <span>{utc(t0)}</span>
            <span>{utc(t1)}</span>
          </div>
        </>
      )}
    </div>
  );
}
