// MAP NAVIGATION CLUSTER — the handoff-approved button-driven camera
// (design_handoff_flight_track_3d, installed 2026-07-20). Fixed to the
// right edge of every 3D map view: compass dial (drag = rotate 360°,
// click = animate to north), rotate/tilt/zoom hold-buttons, pan D-pad,
// RESET VIEW — so the map can be moved 360° without mouse-dragging.
//
// The camera is the damped orbit rig from lib/cameraRig.ts: every channel
// {bearing, pitch, zoom, center} carries a goal; one rAF loop steps the
// values toward the goals (k = 1 − e^(−7·dt)) and drives map.jumpTo. The
// loop runs ONLY while something is moving (holds, keys, unsettled goals,
// follow) — an idle cluster costs zero frames (perf-gate requirement).
//
// ARBITRATION with the map's other camera writers (satellite follow's
// per-frame jumpTo, the space frame, MapLibre's own touch gestures and
// inertia): the rig is PASSIVE unless it has active input or unsettled
// goals — while passive it re-seeds cur+goal from the live camera on every
// loop start, so it never fights an external move.
//
// MOUSE SCHEME IS PLANE-VIEW ONLY (human 2026-07-20: "i still want to be
// able to move around the map with mouse controls"): the base map keeps
// MapLibre's NATIVE mouse gestures — left-drag pans, wheel zooms,
// right-drag rotates — exactly as before the handoff. Only while the
// caller sets dragScheme (an open flight card) are mouse gestures
// intercepted at capture BEFORE MapLibre's handlers (the zoom-seam
// precedent in datamap): left-drag rotates/orbits, right/shift-drag pans,
// wheel zooms exponentially, double-click recenters — the exact prototype
// scheme, in the context it was designed for. TOUCH pointers always pass
// through untouched (phones keep standard one-finger pan / pinch).
//
// Keyboard (prototype-exact): Q/E rotate, R/F tilt, arrows pan, +/− zoom.
// Space is owned by the flight profile panel (play/pause), not here.
//
// EMPIRICAL SIGN NOTE (probe vs the approved prototype, 2026-07-20): the
// prototype's N/S pan buttons moved opposite their own "Pan forward"
// tooltip and the handoff README ("up = away from camera") while E/W
// matched — a prototype sign slip. This implements the documented intent:
// ↑ pans away from the camera. All other signs (dial, drag, tilt, zoom,
// rotate) were probe-verified and match the prototype exactly.

import { useEffect, useMemo, useRef, useState } from "react";
import type maplibregl from "maplibre-gl";
import { useIsMobile } from "@/hooks/use-mobile";
import { applyPanelPos, clearPanelPos, getPanelPrefs, panelDragProps, savePanelPrefs } from "@/lib/panelLayout";
import {
  RIG_DAMPING_PER_S,
  RIG_DRAG_ROTATE_DEG_PX,
  RIG_DRAG_TILT_DEG_PX,
  RIG_PITCH_MAX,
  RIG_PITCH_MIN,
  RIG_ROTATE_DEG_S,
  RIG_TILT_DEG_S,
  RIG_WHEEL_ZOOM_PER_DY,
  RIG_ZOOM_LEVELS_S,
  bearingDeltaToNorth,
  clampLat,
  clampPitch,
  dialBearing,
  makeHoldRegistry,
  panDelta,
  stepRig,
  zoomGoalStep,
  type Rig,
} from "@/lib/cameraRig";

export interface MapNavClusterProps {
  map: maplibregl.Map | null;
  /** true once the map fired ready (the cluster stays inert before). */
  mapReady: boolean;
  /** the space frame owns the camera — cluster hides & rig stays passive. */
  suspended?: boolean;
  /** zoom-out pressed at the map's zoom floor — the continuous-zoom seam
   *  (space-frame entry). Return true when the seam consumed the step. */
  onZoomOutAtFloor?: () => boolean;
  /** any user pan/recenter — the flight Follow toggle auto-disables. */
  onUserPan?: () => void;
  /** per-frame follow target (lng, lat) or null — the Follow-aircraft
   *  camera lock: the rig's center goal tracks it while set; heading/
   *  tilt/zoom stay free (handoff flight-card spec). */
  followTarget?: () => { lng: number; lat: number } | null;
  /** double-click recenter fires this with the clicked point. */
  onRecenter?: (lngLat: { lng: number; lat: number }) => void;
  /** true = the prototype ORBIT mouse scheme owns the canvas (left-drag
   *  rotate, right-drag pan, wheel rig-zoom) — the flight view. false
   *  (default) = MapLibre's native gestures untouched: left-drag PANS,
   *  the standard map feel (human directive 2026-07-20). */
  dragScheme?: boolean;
  /** suspended (space-frame) mode: the rig is inert but zoom buttons stay —
   *  each press forwards one seam step to the space camera (out = true). */
  onSuspendedZoom?: (out: boolean) => void;
  /** suspended mode FLY HOME — a continuous flight back through the seam
   *  (the space frame's Escape behavior, given a button so the controls
   *  never "go away when you zoom out" — live report 2026-07-20). */
  onSuspendedReset?: () => void;
}

type HoldFn = (dt: number) => void;

/** RESET VIEW home — the /data map's initial camera (datamap map init). */
export const NAV_HOME = { lng: -96.77, lat: 37.5, zoom: 3.6, bearing: 0, pitch: 0 };

export default function MapNavCluster({
  map,
  mapReady,
  suspended,
  onZoomOutAtFloor,
  onUserPan,
  followTarget,
  onRecenter,
  onSuspendedZoom,
  onSuspendedReset,
  dragScheme = false,
}: MapNavClusterProps) {
  const ringRef = useRef<SVGGElement | null>(null);
  const rigRef = useRef<Rig>({
    cur: { ...NAV_HOME },
    goal: { ...NAV_HOME },
  } as unknown as Rig);
  // held buttons keyed by BUTTON NAME, not closure identity — the page
  // re-renders every data tick, re-creating the handlers; a release that
  // removes by function identity misses after a re-render and the camera
  // spins forever (live latch report 2026-07-20, round 3).
  const heldRef = useRef(makeHoldRegistry());
  const keysRef = useRef<Set<string>>(new Set());
  const rafRef = useRef<number | null>(null);
  const activeRef = useRef(false);
  const suspendedRef = useRef(!!suspended);
  suspendedRef.current = !!suspended;
  const followRef = useRef<MapNavClusterProps["followTarget"]>(followTarget);
  followRef.current = followTarget;
  // phone: collapsed by default behind a compass FAB (DESIGN.md rule 2 —
  // controls collapse on phone; desktop always shows the cluster).
  const [openOnPhone, setOpenOnPhone] = useState(false);
  // layout memory (human 2026-07-20): the cluster is draggable by its grip,
  // minimizable to a chip, lockable, and the placement/state is remembered.
  // Desktop only — the phone keeps its FAB collapse pattern.
  const isPhone = useIsMobile();
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [navLocked, setNavLocked] = useState<boolean>(() => !!getPanelPrefs("nav-cluster").locked);
  const navLockedRef = useRef(navLocked);
  navLockedRef.current = navLocked;
  const [navMin, setNavMin] = useState<boolean>(() => !!getPanelPrefs("nav-cluster").min);
  const navDrag = useMemo(
    () => panelDragProps("nav-cluster", () => rootRef.current, () => navLockedRef.current),
    [],
  );
  const toggleNavLock = () =>
    setNavLocked((v) => { const n = !v; savePanelPrefs("nav-cluster", { locked: n }); return n; });
  const setNavMinimized = (v: boolean) => { setNavMin(v); savePanelPrefs("nav-cluster", { min: v }); };
  const minChipActive = navMin && !isPhone;
  // re-apply the remembered spot whenever the rendered variant swaps (the
  // suspended stack, the mini chip and the full cluster are separate nodes)
  useEffect(() => {
    const el = rootRef.current;
    if (el && !applyPanelPos(el, "nav-cluster")) clearPanelPos(el);
  }, [suspended, minChipActive, mapReady]);

  // compass ring rotation follows the LIVE camera (also when other systems
  // move it): cheap DOM write on the map's own move events.
  useEffect(() => {
    if (!map || !mapReady) return;
    const paint = () => {
      const el = ringRef.current;
      if (el) el.setAttribute("transform", `rotate(${-map.getBearing()} 46 46)`);
    };
    paint();
    map.on("move", paint);
    return () => { try { map.off("move", paint); } catch {} };
  }, [map, mapReady]);

  // ── the rig loop ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!map || !mapReady) return;
    const rig = rigRef.current;

    const syncFromMap = () => {
      const c = map.getCenter();
      const v = { bearing: map.getBearing(), pitch: map.getPitch(), zoom: map.getZoom(), lng: c.lng, lat: c.lat };
      rig.cur = { ...v };
      rig.goal = { ...v };
    };

    let last = performance.now();
    const frame = (now: number) => {
      rafRef.current = null;
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (suspendedRef.current) { activeRef.current = false; return; }

      const held = heldRef.current;
      const keys = keysRef.current;
      held.tick(dt);
      keys.forEach((k) => keyFns[k]?.(dt));

      // Follow-aircraft: the center goal tracks the live/replayed craft
      const ft = followRef.current?.();
      if (ft) { rig.goal.lng = ft.lng; rig.goal.lat = clampLat(ft.lat); }

      const moving = stepRig(rig, dt);
      try {
        map.jumpTo({
          center: [rig.cur.lng, rig.cur.lat],
          bearing: rig.cur.bearing,
          pitch: rig.cur.pitch,
          zoom: rig.cur.zoom,
        });
      } catch { /* map torn down mid-frame */ }

      if (moving || held.size() > 0 || keys.size > 0 || ft) {
        rafRef.current = requestAnimationFrame(frame);
      } else {
        activeRef.current = false;
      }
    };

    /** Wake the rig: re-seed from the live camera if it was passive (so an
     *  external move — touch, satellite follow, easing — is adopted, never
     *  fought), then start the loop. */
    const wake = () => {
      if (activeRef.current) return;
      syncFromMap();
      activeRef.current = true;
      last = performance.now();
      rafRef.current = requestAnimationFrame(frame);
    };
    wakeRef.current = wake;

    // keyboard — prototype bindings; never captures typing targets, never
    // fights the space frame or the zoom-floor seam handler (+/− at the
    // floor is the seam's job; we skip so it can act).
    const isTyping = (t: EventTarget | null) =>
      t instanceof HTMLElement && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
    const onKeyDown = (e: KeyboardEvent) => {
      if (suspendedRef.current || isTyping(e.target)) return;
      const k = e.key.toLowerCase();
      if (!(k in keyFns)) return;
      if ((k === "-" || k === "_") && atZoomFloor()) return; // seam owns zoom-out at the floor
      keysRef.current.add(k);
      wake();
      e.preventDefault();
    };
    const onKeyUp = (e: KeyboardEvent) => keysRef.current.delete(e.key.toLowerCase());
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);

    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      activeRef.current = false;
      wakeRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, mapReady]);

  const wakeRef = useRef<(() => void) | null>(null);
  const wake = () => wakeRef.current?.();
  const rig = () => rigRef.current;
  const atZoomFloor = () => {
    try { return !!map && map.getZoom() <= map.getMinZoom() + 0.05; } catch { return false; }
  };
  const viewportH = () => {
    try { return map?.getCanvas().clientHeight || 600; } catch { return 600; }
  };

  // hold-button verbs (prototype rates; goals only — the loop damps)
  const keyFns: Record<string, HoldFn> = useMemo(() => {
    const pan = (fx: number, fz: number): HoldFn => (dt) => {
      const r = rig();
      const d = panDelta(r.goal.bearing, fx, fz, r.cur.zoom, r.cur.lat, viewportH(), dt);
      r.goal.lng += d.dLng;
      r.goal.lat = clampLat(r.goal.lat + d.dLat);
      onUserPan?.();
    };
    const fns: Record<string, HoldFn> = {
      q: (dt) => { rig().goal.bearing += RIG_ROTATE_DEG_S * dt; },
      e: (dt) => { rig().goal.bearing -= RIG_ROTATE_DEG_S * dt; },
      // R = flatter (toward the grazing tilt), F = toward top-down —
      // probe-verified prototype directions
      r: (dt) => { rig().goal.pitch = clampPitch(rig().goal.pitch + RIG_TILT_DEG_S * dt); },
      f: (dt) => { rig().goal.pitch = clampPitch(rig().goal.pitch - RIG_TILT_DEG_S * dt); },
      arrowup: pan(0, -1),
      arrowdown: pan(0, 1),
      arrowleft: pan(-1, 0),
      arrowright: pan(1, 0),
      "=": (dt) => { zoomBy(RIG_ZOOM_LEVELS_S * dt); },
      "+": (dt) => { zoomBy(RIG_ZOOM_LEVELS_S * dt); },
      "-": (dt) => { zoomBy(-RIG_ZOOM_LEVELS_S * dt); },
      _: (dt) => { zoomBy(-RIG_ZOOM_LEVELS_S * dt); },
    };
    return fns;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map]);

  /** all button/key/wheel zoom goes through the runaway-proof step:
   *  map bounds + imagery ceiling + goal-ahead cap (live fix 2026-07-20:
   *  hold-zoom flew uncontrollably into blank placeholder tiles). */
  const zoomBy = (delta: number) => {
    const r = rig();
    let minZ = 0, maxZ = 22;
    try { if (map) { minZ = map.getMinZoom(); maxZ = map.getMaxZoom(); } } catch {}
    r.goal.zoom = zoomGoalStep(r.cur.zoom, r.goal.zoom, delta, minZ, maxZ);
  };

  // hold-to-repeat wiring (pointer capture; touch-friendly). Press/release
  // are keyed by BUTTON — the page re-renders every data tick, and closure-
  // identity releases missed after a re-render, latching the camera into a
  // spin (live report 2026-07-20). Keys survive re-renders by construction.
  const holdProps = (key: string, fn: HoldFn, opts: { zoomOut?: boolean } = {}) => ({
    onPointerDown: (e: React.PointerEvent) => {
      e.preventDefault();
      // the continuous-zoom seam: a zoom-out step at the floor enters the
      // space frame instead of no-oping (the NavigationControl precedent)
      if (opts.zoomOut && atZoomFloor() && onZoomOutAtFloor?.()) return;
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      heldRef.current.press(key, fn);
      wake();
    },
    onPointerUp: () => heldRef.current.release(key),
    onPointerCancel: () => heldRef.current.release(key),
    onLostPointerCapture: () => heldRef.current.release(key),
  });

  // space-mode hold-repeat (interval-based — the rig loop is off in space;
  // the space camera takes discrete seam impulses)
  const spaceIvRef = useRef<number | null>(null);
  const stopSpaceHold = () => {
    if (spaceIvRef.current != null) {
      window.clearInterval(spaceIvRef.current);
      spaceIvRef.current = null;
    }
  };
  useEffect(() => () => stopSpaceHold(), []);

  // GLOBAL RELEASE FAILSAFES (latch bug, round 3): any pointer release
  // anywhere, a lost tab, or a window blur stops every held button — a
  // missed per-button pointerup can never leave the camera spinning. Blur/
  // hidden also drop held KEYS (their keyup would never arrive).
  useEffect(() => {
    const stopAll = () => { heldRef.current.clear(); stopSpaceHold(); };
    const stopAllAndKeys = () => { stopAll(); keysRef.current.clear(); };
    const onVis = () => { if (document.hidden) stopAllAndKeys(); };
    window.addEventListener("pointerup", stopAll, { capture: true });
    window.addEventListener("pointercancel", stopAll, { capture: true });
    window.addEventListener("blur", stopAllAndKeys);
    document.addEventListener("visibilitychange", onVis);
    return () => {
      window.removeEventListener("pointerup", stopAll, { capture: true } as any);
      window.removeEventListener("pointercancel", stopAll, { capture: true } as any);
      window.removeEventListener("blur", stopAllAndKeys);
      document.removeEventListener("visibilitychange", onVis);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── canvas mouse scheme (prototype-exact; mouse pointers only) ────────
  // PLANE VIEW ONLY (dragScheme): everywhere else this effect stays off and
  // MapLibre's native mouse handling owns the canvas — left-drag pans,
  // wheel zooms, right-drag rotates, exactly the pre-handoff map feel.
  useEffect(() => {
    if (!map || !mapReady || !dragScheme) return;
    const el = map.getCanvasContainer();
    let drag: { x: number; y: number; mode: "rot" | "pan"; moved: boolean } | null = null;

    const onMouseDown = (e: MouseEvent) => {
      if (suspendedRef.current) return;
      if (e.button !== 0 && e.button !== 2) return;
      drag = { x: e.clientX, y: e.clientY, mode: e.button === 2 || e.shiftKey ? "pan" : "rot", moved: false };
      wake();
      // MapLibre's own drag handlers never see mouse drags (capture phase);
      // touch events pass through untouched — native gestures keep working.
      e.stopPropagation();
      e.preventDefault();
    };
    const onMouseMove = (e: MouseEvent) => {
      if (!drag) return;
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      drag.x = e.clientX;
      drag.y = e.clientY;
      if (Math.abs(dx) + Math.abs(dy) > 1) drag.moved = true;
      const r = rig();
      if (drag.mode === "rot") {
        // probe-verified: drag-right → bearing decreases; drag-down → toward top-down
        r.goal.bearing -= dx * RIG_DRAG_ROTATE_DEG_PX;
        r.goal.pitch = clampPitch(r.goal.pitch - dy * RIG_DRAG_TILT_DEG_PX);
      } else {
        const d = panDelta(r.goal.bearing, -dx / 3, -dy / 3, r.cur.zoom, r.cur.lat, viewportH(), 1 / 60);
        // drag pan: content follows the pointer (prototype 'pan' drag mode)
        r.goal.lng += d.dLng;
        r.goal.lat = clampLat(r.goal.lat + d.dLat);
        onUserPan?.();
      }
      e.stopPropagation();
    };
    const onMouseUp = () => {
      // a rotate/pan drag must not become a feature click on release —
      // MapLibre normally suppresses click-after-drag itself; with its drag
      // handlers bypassed for mouse, we swallow the one synthetic click.
      if (drag?.moved) suppressNextClick = true;
      drag = null;
    };
    let suppressNextClick = false;
    const onClickCapture = (e: MouseEvent) => {
      if (suppressNextClick) {
        suppressNextClick = false;
        e.stopPropagation();
        e.preventDefault();
      }
    };
    const onWheel = (e: WheelEvent) => {
      if (suspendedRef.current) return;
      // zoom-out at the floor belongs to the continuous-zoom seam (it has
      // its own capture listener on this container) — stay out of its way
      if (e.deltaY > 0 && atZoomFloor()) return;
      wake();
      zoomBy(-e.deltaY * RIG_WHEEL_ZOOM_PER_DY);
      e.preventDefault();
      e.stopPropagation();
    };
    const onDblClick = (e: MouseEvent) => {
      if (suspendedRef.current) return;
      try {
        const rect = map.getCanvas().getBoundingClientRect();
        const ll = map.unproject([e.clientX - rect.left, e.clientY - rect.top]);
        const r = rig();
        wake();
        r.goal.lng = ll.lng;
        r.goal.lat = clampLat(ll.lat);
        onUserPan?.();
        onRecenter?.({ lng: ll.lng, lat: ll.lat });
      } catch {}
      e.stopPropagation();
      e.preventDefault();
    };
    const onCtx = (e: MouseEvent) => e.preventDefault();

    el.addEventListener("mousedown", onMouseDown, { capture: true });
    window.addEventListener("mousemove", onMouseMove, { capture: true });
    window.addEventListener("mouseup", onMouseUp, { capture: true });
    el.addEventListener("click", onClickCapture, { capture: true });
    el.addEventListener("wheel", onWheel, { capture: true, passive: false });
    el.addEventListener("dblclick", onDblClick, { capture: true });
    el.addEventListener("contextmenu", onCtx);
    return () => {
      el.removeEventListener("mousedown", onMouseDown, { capture: true } as any);
      window.removeEventListener("mousemove", onMouseMove, { capture: true } as any);
      window.removeEventListener("mouseup", onMouseUp, { capture: true } as any);
      el.removeEventListener("click", onClickCapture, { capture: true } as any);
      el.removeEventListener("wheel", onWheel, { capture: true } as any);
      el.removeEventListener("dblclick", onDblClick, { capture: true } as any);
      el.removeEventListener("contextmenu", onCtx);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, mapReady, dragScheme]);

  // compass dial drag/click
  const dialState = useRef<{ grab: number; startBearing: number; moved: boolean } | null>(null);
  const dialAngle = (e: React.PointerEvent) => {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    return Math.atan2(e.clientY - (r.top + r.height / 2), e.clientX - (r.left + r.width / 2));
  };
  const onDialDown = (e: React.PointerEvent) => {
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    dialState.current = { grab: dialAngle(e), startBearing: rig().goal.bearing, moved: false };
    wake();
  };
  const onDialMove = (e: React.PointerEvent) => {
    const d = dialState.current;
    if (!d) return;
    const a = dialAngle(e);
    if (Math.abs(a - d.grab) > 0.02) d.moved = true;
    rig().goal.bearing = dialBearing(d.startBearing, d.grab, a);
    wake();
  };
  const onDialUp = () => {
    const d = dialState.current;
    dialState.current = null;
    if (d && !d.moved) {
      // click → animate heading to north, shortest way (prototype-exact)
      const r = rig();
      wake();
      r.goal.bearing += bearingDeltaToNorth(r.goal.bearing);
    }
  };

  const resetView = () => {
    const r = rig();
    wake();
    r.goal = { ...NAV_HOME } as typeof r.goal;
    onUserPan?.();
  };

  // 30°-tick ring (90° ticks longer) — generated once, prototype geometry
  const ticks = useMemo(() => {
    const t: JSX.Element[] = [];
    for (let a = 0; a < 360; a += 30) {
      const r1 = a % 90 ? 34.5 : 32.5;
      const rad = (a * Math.PI) / 180;
      t.push(
        <line key={a}
          x1={46 + Math.sin(rad) * r1} y1={46 - Math.cos(rad) * r1}
          x2={46 + Math.sin(rad) * 38} y2={46 - Math.cos(rad) * 38}
          stroke="rgba(180,205,235,.5)" strokeWidth={a % 90 ? 1 : 1.6} />,
      );
    }
    return t;
  }, []);

  const Icon = ({ d, size = 16, sw = 2 }: { d: string; size?: number; sw?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={sw}>
      {d.split("|").map((p, i) => <path key={i} d={p} />)}
    </svg>
  );

  // shared move/lock/minimize grip (layout memory, human 2026-07-20) —
  // desktop only (the phone FAB pattern stands; CSS hides this <768px)
  const gripRow = (
    <div className="vt-nav-grip" {...navDrag}
         style={{ cursor: navLocked ? "default" : "grab", touchAction: "none" }}
         title={navLocked ? "Position locked" : "Drag to move · double-click to reset · spot is remembered"}>
      <span className="vt-card-grip" aria-hidden>⠿</span>
      <button className={`vt-nav-gripbtn${navLocked ? " on" : ""}`} aria-pressed={navLocked}
              aria-label={navLocked ? "Unlock controls position" : "Lock controls position"}
              title={navLocked ? "Position locked — click to unlock" : "Lock position"}
              onClick={toggleNavLock}>
        {navLocked
          ? <Icon d="M8 11V7a4 4 0 0 1 8 0v4|M5 11h14v9H5z" size={12} />
          : <Icon d="M8 11V7a4 4 0 0 1 7.6-1.7|M5 11h14v9H5z" size={12} />}
      </button>
      <button className="vt-nav-gripbtn" aria-label="Minimize map controls" title="Minimize controls"
              onClick={() => setNavMinimized(true)}>
        <Icon d="M5 12h14" size={12} />
      </button>
    </div>
  );

  if (minChipActive) {
    // minimized: one compass chip, still draggable/remembered; click restores
    return (
      <div ref={rootRef} className="vt-nav-cluster vt-nav-open" data-vt-nav data-vt-nav-min>
        <div className="vt-nav-card vt-nav-chiprow" {...navDrag}
             style={{ cursor: navLocked ? "default" : "grab", touchAction: "none" }}
             title={navLocked ? "Position locked" : "Drag to move · click the compass to restore"}>
          <span className="vt-card-grip" aria-hidden>⠿</span>
          <button className="vt-nav-btn" data-vt-nav-restore aria-label="Show map controls"
                  title="Show map controls" onClick={() => setNavMinimized(false)}>
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="9" />
              <path d="M15.5 8.5 13 13l-4.5 2.5L11 11z" fill="currentColor" stroke="none" />
            </svg>
          </button>
        </div>
      </div>
    );
  }

  if (suspended) {
    // space frame owns the camera — the rig stays inert, but button zoom
    // survives (the old NavigationControl kept working in space; the seam's
    // per-press nudge moves with the cluster). HOLD repeats (160ms seam
    // impulses), and FLY HOME gives the space view a reset — the controls
    // never "go away when you zoom out" (live report 2026-07-20).
    const spaceHoldProps = (out: boolean) => ({
      onPointerDown: (e: React.PointerEvent) => {
        e.preventDefault();
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        stopSpaceHold();
        onSuspendedZoom?.(out);
        spaceIvRef.current = window.setInterval(() => onSuspendedZoom?.(out), 160);
      },
      onPointerUp: stopSpaceHold,
      onPointerCancel: stopSpaceHold,
      onLostPointerCapture: stopSpaceHold,
    });
    return (
      <div ref={rootRef} className="vt-nav-cluster vt-nav-open" data-vt-nav role="group" aria-label="Space view navigation">
        {gripRow}
        <div className="vt-nav-lbl">ZOOM</div>
        <div className="vt-nav-card vt-nav-btncol">
          <div className="vt-nav-row">
            <button className="vt-nav-btn" data-vt-nav-zoomin title="Zoom in (hold)" aria-label="Zoom in"
                    {...spaceHoldProps(false)}>
              <Icon d="M12 5v14M5 12h14" size={15} sw={2.2} />
            </button>
            <button className="vt-nav-btn" data-vt-nav-zoomout title="Zoom out (hold)" aria-label="Zoom out"
                    {...spaceHoldProps(true)}>
              <Icon d="M5 12h14" size={15} sw={2.2} />
            </button>
          </div>
        </div>
        <button className="vt-nav-btn vt-nav-wide vt-nav-card" data-vt-nav-reset
                title="Fly home to the live map" onClick={() => onSuspendedReset?.()}>
          FLY HOME
        </button>
      </div>
    );
  }

  const holdRot = (sign: 1 | -1): HoldFn => (dt) => { rig().goal.bearing += sign * RIG_ROTATE_DEG_S * dt; };
  const holdTilt = (sign: 1 | -1): HoldFn => (dt) => { rig().goal.pitch = clampPitch(rig().goal.pitch + sign * RIG_TILT_DEG_S * dt); };
  const holdZoom = (sign: 1 | -1): HoldFn => (dt) => { zoomBy(sign * RIG_ZOOM_LEVELS_S * dt); };
  const holdPan = (fx: number, fz: number): HoldFn => (dt) => {
    const r = rig();
    const d = panDelta(r.goal.bearing, fx, fz, r.cur.zoom, r.cur.lat, viewportH(), dt);
    r.goal.lng += d.dLng;
    r.goal.lat = clampLat(r.goal.lat + d.dLat);
    onUserPan?.();
  };

  return (
    <>
      {/* phone FAB (collapsed-by-default per DESIGN.md; desktop hides it) */}
      <button className="vt-nav-fab" data-vt-nav-fab aria-label={openOnPhone ? "Hide map navigation" : "Show map navigation"}
              aria-expanded={openOnPhone} onClick={() => setOpenOnPhone((v) => !v)}>
        <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="9" />
          <path d="M15.5 8.5 13 13l-4.5 2.5L11 11z" fill="currentColor" stroke="none" />
        </svg>
      </button>
      <div ref={rootRef} className={`vt-nav-cluster${openOnPhone ? " vt-nav-open" : ""}`} data-vt-nav role="group" aria-label="Map navigation">
        {gripRow}
        <div className="vt-nav-compass" data-vt-nav-compass title="Drag to rotate · click N to reset"
             onPointerDown={onDialDown} onPointerMove={onDialMove}
             onPointerUp={onDialUp} onPointerCancel={onDialUp}>
          <svg viewBox="0 0 92 92">
            <g ref={ringRef}>
              <circle cx="46" cy="46" r="38" fill="none" stroke="rgba(130,170,230,.25)" strokeWidth="1" />
              {ticks}
              <text x="46" y="18" textAnchor="middle" fontSize="12" fontWeight="700" fill="#ff6b6b">N</text>
              <text x="46" y="82" textAnchor="middle" fontSize="9" fill="#8fa3bf">S</text>
              <text x="79" y="49.5" textAnchor="middle" fontSize="9" fill="#8fa3bf">E</text>
              <text x="13" y="49.5" textAnchor="middle" fontSize="9" fill="#8fa3bf">W</text>
            </g>
            <path d="M46 40 l4 9 -4 -2.4 -4 2.4 z" fill="#dfe8f5" />
          </svg>
        </div>
        <div className="vt-nav-lbl">ROTATE · TILT</div>
        <div className="vt-nav-card vt-nav-btncol">
          <div className="vt-nav-row">
            <button className="vt-nav-btn" data-vt-nav-rotl title="Rotate left (hold) — Q" aria-label="Rotate left"
                    {...holdProps("rotl", holdRot(1))}><Icon d="M3 12a9 9 0 1 0 3-6.7|M6 2v4h4" /></button>
            <button className="vt-nav-btn" data-vt-nav-rotr title="Rotate right (hold) — E" aria-label="Rotate right"
                    {...holdProps("rotr", holdRot(-1))}><Icon d="M21 12a9 9 0 1 1-3-6.7|M18 2v4h-4" /></button>
          </div>
          <div className="vt-nav-row">
            <button className="vt-nav-btn" data-vt-nav-tiltup title="Tilt up / flatter (hold) — R" aria-label="Tilt toward horizon"
                    {...holdProps("tiltup", holdTilt(1))}><Icon d="M4 17h16M6 12l6-6 6 6" /></button>
            <button className="vt-nav-btn" data-vt-nav-tiltdn title="Tilt down / top view (hold) — F" aria-label="Tilt toward top-down"
                    {...holdProps("tiltdn", holdTilt(-1))}><Icon d="M4 7h16M6 12l6 6 6-6" /></button>
          </div>
          <div className="vt-nav-row">
            <button className="vt-nav-btn" data-vt-nav-zoomin title="Zoom in (hold)" aria-label="Zoom in"
                    {...holdProps("zoomin", holdZoom(1))}><Icon d="M12 5v14M5 12h14" size={15} sw={2.2} /></button>
            <button className="vt-nav-btn" data-vt-nav-zoomout title="Zoom out (hold)" aria-label="Zoom out"
                    {...holdProps("zoomout", holdZoom(-1), { zoomOut: true })}><Icon d="M5 12h14" size={15} sw={2.2} /></button>
          </div>
        </div>
        <div className="vt-nav-lbl">PAN</div>
        <div className="vt-nav-card vt-nav-dpad">
          <span />
          <button className="vt-nav-btn" data-vt-nav-pann title="Pan forward (hold) — ↑" aria-label="Pan forward"
                  {...holdProps("pann", holdPan(0, -1))}><Icon d="M12 19V5M5 12l7-7 7 7" size={14} sw={2.4} /></button>
          <span />
          <button className="vt-nav-btn" data-vt-nav-panw title="Pan left (hold) — ←" aria-label="Pan left"
                  {...holdProps("panw", holdPan(-1, 0))}><Icon d="M19 12H5M12 5l-7 7 7 7" size={14} sw={2.4} /></button>
          <span className="vt-nav-btn vt-nav-ctr" aria-hidden />
          <button className="vt-nav-btn" data-vt-nav-pane title="Pan right (hold) — →" aria-label="Pan right"
                  {...holdProps("pane", holdPan(1, 0))}><Icon d="M5 12h14M12 5l7 7-7 7" size={14} sw={2.4} /></button>
          <span />
          <button className="vt-nav-btn" data-vt-nav-pans title="Pan back (hold) — ↓" aria-label="Pan back"
                  {...holdProps("pans", holdPan(0, 1))}><Icon d="M12 5v14M5 12l7 7 7-7" size={14} sw={2.4} /></button>
          <span />
        </div>
        <button className="vt-nav-btn vt-nav-wide vt-nav-card" data-vt-nav-reset title="Reset view" onClick={resetView}>
          RESET VIEW
        </button>
      </div>
    </>
  );
}
