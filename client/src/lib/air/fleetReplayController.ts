// FLEET REPLAY CONTROLLER — the Time Machine's per-frame driver (FLIGHT
// PROGRAM, human directive 2026-09-28: "rewind and see ALL planes in my
// field of view with tracks and curtains at any pan/tilt/zoom; if two planes
// came close you can see it").
//
// Owns, for one open Time Machine panel in window mode:
//  - the FIELD-OF-VIEW QUERY: every frame (frameCore STREAM priority) it
//    reads the camera, asks CameraTargetTracker for the TARGET pose (our
//    own flight's destination, else the pose once settled — never a
//    transient mid-animation view), computes the pitched/horizon-clamped
//    query box + margin ring (replayCamera.ts), and re-reads the window
//    only when the target's visible box leaves what was fetched. Requests
//    are abortable (a newer target aborts the older read); the previous
//    fleet stays drawn until the new one lands — never blank.
//  - the PLAYHEAD: a target set by the slider/play/jump, followed by a
//    frame-rate-independent lerp (frameCore.lerpTowards) — never a jump.
//  - LOD: 4× a second, assignLods (fleetModel.ts) with hysteresis over
//    device-tier budgets; batched geometry is rebuilt ONLY when class
//    membership changes (per-track vertex cache), never per camera event.
//  - HEADS: per frame, every track's head interpolated between REAL fixes
//    (headAt), packed into one instance buffer.
//  - CLOSE APPROACHES: the server's list (per our recorded data); focus
//    dims everything else, highlights the pair, loads the pair's
//    FULL-fidelity archived tracks, and flies the camera to their midpoint;
//    pairs inside the minima at the playhead get a connector + a live
//    separation readout (≈, from the displayed interpolation).
//
// Law I: NO map.on(...) subscriptions at all — the camera is polled inside
// the one rAF loop. Law IV: dispose() unregisters frame callbacks, aborts
// fetches, removes the layer + DOM labels, frees GL, and unregisters from
// the live layer registry.

import type maplibregl from "maplibre-gl";
import { frameCore, lerpTowards, PRIORITY, type FrameLoop } from "../../render/frameCore.ts";
import { registerLayer } from "../../render/layerContract.ts";
import { setGauge } from "../../render/perfMetrics.ts";
import {
  cameraQueryBBox, cameraEye, needsRefetch, bboxParam, metersPerPixel, CameraTargetTracker,
  type BBox, type CameraPose, type QueryBoxes,
} from "./replayCamera.ts";
import {
  prepareFleet, trackFromPoints, spliceOverride, headAt, assignLods, fixIndexAtOrBefore, eyeDistanceKm, trackScore,
  decimateIdx, buildFleetVerts, pointsPerTrack, fleetBudgetForTier, headColor, mercX, mercY,
  FLEET_FETCH_MAX, FLEET_TRAIL_SEC, HEAD_STRIDE, CURTAIN_BELOW_TERRAIN_M,
  LOD_FULL, LOD_THIN, LOD_HEAD, LOD_HIDDEN,
  type FleetTrack, type Head, type WindowHexIn, type FleetBudget,
} from "./fleetModel.ts";
import { FleetReplayLayer, FLEET_WARN_RGBA, FLEET_DIM, maxFeatures, vramBudget } from "./fleetReplayLayer.ts";
import {
  liveSeparation, withinMinima, fmtSeparation, approachKey, type CloseApproachEntry,
} from "./closeApproachView.ts";

export type ReplayKind = "aircraft" | "vessels";

export interface ReplayWindowResult {
  kind: string;
  from: number;
  to: number;
  zoom: number;
  step_sec: number;
  hexes: Array<WindowHexIn & { raw_count?: number; truncated?: boolean }>;
  hexes_seen: number;
  total_points: number;
  coverage: { requested_from: number; scanned_from: number; complete: boolean; files_scanned: number };
  note?: string;
  error?: string;
  closeApproaches?: CloseApproachEntry[];
  closeApproachesMeta?: {
    evaluated_hexes: number; found: number; returned: number; capped: boolean; partial_scan: boolean;
    excluded_non_icao?: number; method?: string;
  };
}

export interface ReplayState {
  loading: boolean;
  error: string | null;
  data: ReplayWindowResult | null;
  /** wall-clock ms the drawn window read landed (Law V data age) */
  fetchedAtMs: number | null;
  lod: { full: number; thin: number; headOnly: number; hidden: number; tracks: number };
  focusKey: string | null;
  focusFullRes: "loading" | "ok" | "failed" | null;
  playing: boolean;
}

export interface ReplayQuery {
  kind: ReplayKind;
  fromSec: number;
  toSec: number;
  stepSec: number;
}

/** replay speed: a full window sweep in ~54 s, never finer than the step
 *  (the pre-controller cadence: max(step, window/60) per 0.9 s tick) */
export function playRateSecPerSec(windowSec: number, stepSec: number): number {
  return Math.max(stepSec, windowSec / 60) / 0.9;
}

/** spring-shaped (critically damped) easing for our own camera flights */
export function springEase(t: number, k = 8): number {
  const f = (x: number) => 1 - (1 + k * x) * Math.exp(-k * x);
  return Math.min(1, Math.max(0, f(t) / f(1)));
}

const PLAYHEAD_HALF_LIFE_MS = 70;
const LOD_TICK_MS = 250;
const MIN_FETCH_INTERVAL_MS = 400;
const STATE_THROTTLE_MS = 120;
const FULLRES_PAD_SEC = 1800;
const CONN_WIDTH_PX = 2.5;
interface GeomCacheEntry { key: string; verts: Float32Array }

/** window.__vtFleetReplay — read-only diagnostics for the visual harness */
export interface FleetReplayDebug {
  counts(): { full: number; thin: number; conn: number; heads: number };
  failed(): boolean;
  draws(): { drawnFrames: number; failStreak: number };
  lod(): ReplayState["lod"];
}

/** soft failures the replay tolerates (teardown races, style reloads, a
 *  terrain query on an unloaded tile) — counted on the perf HUD and logged
 *  ONCE per site, never swallowed silently */
const softFailures = new Map<string, number>();
export function noteSoftFailure(site: string, e: unknown): number {
  const n = (softFailures.get(site) ?? 0) + 1;
  softFailures.set(site, n);
  setGauge(`fleetReplay.softFail.${site}`, n);
  // eslint-disable-next-line no-console
  if (n === 1) console.warn(`[fleetReplay] ${site} (tolerated; counted on the perf HUD):`, e);
  return n;
}

export class FleetReplayController {
  private readonly map: maplibregl.Map;
  private readonly loop: FrameLoop;
  private readonly fetchImpl: typeof fetch;
  private readonly onState: (s: ReplayState) => void;
  private readonly onPlayhead: (valueSec: number, targetSec: number, playing: boolean) => void;
  private readonly budget: FleetBudget;
  private readonly fetchMax: number;
  private readonly layer = new FleetReplayLayer({ id: "fleet-replay-3d" });
  private readonly getObstruction?: () => RectLike | null;
  private readonly tracker = new CameraTargetTracker();
  private unregs: Array<() => void> = [];

  private query: ReplayQuery | null = null;
  private forceFetch = false;
  private ac: AbortController | null = null;
  private fetchSeq = 0;
  private inFlightBox: BBox | null = null;
  private fetched: { query: BBox; capped: boolean } | null = null;
  private lastFetchAt = -Infinity;

  private data: ReplayWindowResult | null = null;
  private baseTracks: FleetTrack[] = [];
  private overrides = new Map<string, FleetTrack>();
  private tracks: FleetTrack[] = [];
  private idIndex = new Map<string, number>();
  private slotOf = new Map<string, number>();
  private nextSlot = 0;
  private lods: Uint8Array = new Uint8Array(0);
  private heads: Array<Head | null> = [];
  private geomCache = new Map<string, GeomCacheEntry>();
  private geometryDirty = true;
  private lastLodTick = -Infinity;
  private lastGeomSig = "";
  private headInst = new Float32Array(0);

  private playing = false;
  private playTarget = 0;
  private playValue = 0;
  private lastPlayValueDrawn = NaN;
  private lastPlayheadEmit = -Infinity;
  private lastEmittedValue = NaN;
  private lastEmittedTarget = NaN;

  private focus: CloseApproachEntry | null = null;
  private focusFullRes: ReplayState["focusFullRes"] = null;
  private focusAc: AbortController | null = null;
  private connShown = false;
  private labelRoot: HTMLDivElement | null = null;
  private labelPool: HTMLDivElement[] = [];

  private state: ReplayState = {
    loading: false, error: null, data: null, fetchedAtMs: null,
    lod: { full: 0, thin: 0, headOnly: 0, hidden: 0, tracks: 0 },
    focusKey: null, focusFullRes: null, playing: false,
  };
  private stateTimer: ReturnType<typeof setTimeout> | null = null;
  private disposed = false;
  private layerCheckAt = -Infinity;

  constructor(opts: {
    map: maplibregl.Map;
    onState: (s: ReplayState) => void;
    onPlayhead: (valueSec: number, targetSec: number, playing: boolean) => void;
    loop?: FrameLoop;
    fetchImpl?: typeof fetch;
    tier?: string | null;
    /** screen rect of UI covering the map (the panel) — framing avoids it */
    getObstruction?: () => RectLike | null;
  }) {
    this.map = opts.map;
    this.getObstruction = opts.getObstruction;
    this.onState = opts.onState;
    this.onPlayhead = opts.onPlayhead;
    this.loop = opts.loop ?? frameCore();
    this.fetchImpl = opts.fetchImpl ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
    const tier = opts.tier ?? (globalThis as { __vtDeviceTier?: { tier?: string } }).__vtDeviceTier?.tier ?? null;
    this.budget = fleetBudgetForTier(tier);
    this.fetchMax = FLEET_FETCH_MAX[tier === "full" || tier === "minimal" ? tier : "reduced"];
    this.ensureLayer();
    // harness/diagnostics hook (the visual harness asserts the GL programs
    // compiled and drew under SwiftShader — string tests cannot)
    const dbg: FleetReplayDebug = {
      counts: () => this.layer.getCounts(),
      failed: () => this.layer.getRenderFailed(),
      draws: () => this.layer.getDrawStats(),
      lod: () => this.state.lod,
    };
    (globalThis as { __vtFleetReplay?: FleetReplayDebug }).__vtFleetReplay = dbg;
    this.unregs.push(() => {
      const g = globalThis as { __vtFleetReplay?: FleetReplayDebug };
      if (g.__vtFleetReplay === dbg) delete g.__vtFleetReplay;
    });
    this.unregs.push(
      this.loop.register((dt, now) => this.streamTick(now), PRIORITY.STREAM, { label: "fleetReplay.stream" }),
      this.loop.register((dt, now) => this.simTick(dt, now), PRIORITY.SIM, { label: "fleetReplay.sim" }),
      this.loop.register((dt, now) => this.renderTick(now), PRIORITY.RENDER, { label: "fleetReplay.render" }),
      registerLayer({ id: this.layer.id, maxFeatures, vramBudget, dispose: () => this.layer.dispose() }),
    );
  }

  // ── public API ────────────────────────────────────────────────────────────

  /** new window/step/kind: clears the drawn fleet only when the replacement
   *  lands (never blank), re-reads from the CURRENT pose immediately */
  setQuery(q: ReplayQuery): void {
    const same = this.query && this.query.kind === q.kind && this.query.fromSec === q.fromSec &&
      this.query.toSec === q.toSec && this.query.stepSec === q.stepSec;
    if (same) return;
    const kindChanged = !this.query || this.query.kind !== q.kind;
    this.query = { ...q };
    this.fetched = null;
    this.forceFetch = true;
    if (kindChanged) this.clearFocus();
    this.playing = false;
    this.playTarget = this.playValue = q.toSec;
    this.emitPlayhead(true);
    this.patchState({ playing: false });
  }

  setPlaying(p: boolean): void {
    if (!this.query) return;
    if (p && this.playTarget >= this.query.toSec - 0.5) {
      // restart from the window start when play is pressed at the end
      this.playTarget = this.query.fromSec;
    }
    this.playing = p;
    this.patchState({ playing: p });
    this.emitPlayhead(true);
  }

  isPlaying(): boolean { return this.playing; }

  /** scrub target (seconds); the drawn playhead lerps to it per frame */
  setPlayheadTarget(tSec: number): void {
    if (!this.query || !Number.isFinite(tSec)) return;
    this.playTarget = Math.min(this.query.toSec, Math.max(this.query.fromSec, tSec));
    this.emitPlayhead(true);
  }

  getPlayhead(): { value: number; target: number } {
    return { value: this.playValue, target: this.playTarget };
  }

  /** focus a close approach: highlight + dim, playhead to t − 60 s, full-
   *  fidelity tracks for the pair, camera to their midpoint */
  focusApproach(ca: CloseApproachEntry | null): void {
    if (!ca) { this.clearFocus(); return; }
    this.focus = ca;
    this.playing = false;
    const tSec = ca.t / 1000;
    this.setPlayheadTarget(tSec - 60);
    this.patchState({ focusKey: approachKey(ca), playing: false });
    this.flyToPair(ca);
    this.loadFullRes(ca);
    this.applyHighlight();
    this.geometryDirty = true;
    this.lastLodTick = -Infinity;
    this.lastPlayValueDrawn = NaN; // re-color heads + connectors next frame
  }

  clearFocus(): void {
    this.focus = null;
    this.focusAc?.abort();
    this.focusAc = null;
    this.focusFullRes = null;
    if (this.overrides.size) {
      this.overrides.clear();
      this.rebuildFleet();
    }
    this.layer.setUniforms({ hlA: -1, hlB: -1 });
    this.patchState({ focusKey: null, focusFullRes: null });
    this.geometryDirty = true;
    this.lastLodTick = -Infinity;
    this.lastPlayValueDrawn = NaN;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const u of this.unregs) {
      try { u(); } catch (e) { noteSoftFailure("unregister", e); }
    }
    this.unregs = [];
    this.ac?.abort();
    this.focusAc?.abort();
    if (this.stateTimer) clearTimeout(this.stateTimer);
    try {
      if (this.map.getLayer(this.layer.id)) this.map.removeLayer(this.layer.id);
    } catch (e) { noteSoftFailure("removeLayer", e); }
    this.layer.dispose();
    this.labelRoot?.remove();
    this.labelRoot = null;
    this.labelPool = [];
    this.geomCache.clear();
    this.tracks = [];
    this.baseTracks = [];
    this.overrides.clear();
  }

  // ── frame callbacks ───────────────────────────────────────────────────────

  private readPose(): CameraPose {
    const c = this.map.getCenter();
    return { lon: c.lng, lat: c.lat, zoom: this.map.getZoom(), pitch: this.map.getPitch(), bearing: this.map.getBearing() };
  }

  private viewport(): { widthPx: number; heightPx: number } {
    const el = this.map.getContainer();
    return { widthPx: el.clientWidth || 1, heightPx: el.clientHeight || 1 };
  }

  /** STREAM: camera TARGET → query box → (maybe) one abortable window read */
  private streamTick(now: number): void {
    if (this.disposed || !this.query) return;
    const pose = this.readPose();
    let target = this.tracker.update(pose, now);
    if (this.forceFetch) target = target ?? pose;
    if (!target) return; // gesture in progress: destination unknown
    const boxes = cameraQueryBBox(target, this.viewport());
    const ref = this.inFlightBox ? { query: this.inFlightBox, capped: false } : this.fetched;
    if (!this.forceFetch && !needsRefetch(ref, boxes)) return;
    if (!this.forceFetch && now - this.lastFetchAt < MIN_FETCH_INTERVAL_MS) return;
    // a failing read backs off (1 s, 2 s, 4 s … 30 s) instead of re-asking
    // every MIN_FETCH_INTERVAL_MS; a new query (forceFetch) retries at once
    if (!this.forceFetch && now < this.retryAt) return;
    this.forceFetch = false;
    this.lastFetchAt = now;
    this.startFetch(boxes, target.zoom);
  }

  private startFetch(boxes: QueryBoxes, zoom: number): void {
    const q = this.query;
    if (!q) return;
    this.ac?.abort(); // a newer target supersedes the older read
    const ac = new AbortController();
    this.ac = ac;
    const seq = ++this.fetchSeq;
    this.inFlightBox = boxes.query;
    this.patchState({ loading: true });
    const url = `/api/data/aircraft/window?bbox=${encodeURIComponent(bboxParam(boxes.query))}` +
      `&from=${q.fromSec}&to=${q.toSec}&zoom=${zoom.toFixed(2)}&step=${q.stepSec}` +
      `&kind=${encodeURIComponent(q.kind)}&max=${this.fetchMax}`;
    this.fetchImpl(url, { signal: ac.signal })
      .then(async (r) => ({ ok: r.ok, status: r.status, d: (await r.json()) as ReplayWindowResult }))
      .then(({ ok, status, d }) => {
        if (this.disposed || seq !== this.fetchSeq) return;
        this.inFlightBox = null;
        if (!ok || !d || !Array.isArray(d.hexes)) {
          // keep the previous fleet drawn; say why; back off
          this.noteFetchFailure();
          this.patchState({ loading: false, error: (d && d.error) || `request failed (${status})` });
          return;
        }
        this.fetchFailures = 0;
        this.retryAt = -Infinity;
        this.fetched = { query: boxes.query, capped: d.hexes_seen > d.hexes.length };
        this.data = d;
        this.baseTracks = prepareFleet(d.hexes, d.step_sec);
        this.rebuildFleet();
        this.patchState({ loading: false, error: null, data: d, fetchedAtMs: Date.now() });
      })
      .catch((e: unknown) => {
        if ((e as { name?: string })?.name === "AbortError" || this.disposed || seq !== this.fetchSeq) return;
        this.inFlightBox = null;
        this.noteFetchFailure();
        this.patchState({ loading: false, error: (e as Error)?.message || "network error" });
      });
  }

  private fetchFailures = 0;
  private retryAt = -Infinity;

  private noteFetchFailure(): void {
    this.fetchFailures++;
    this.retryAt = this.loop.now() + Math.min(30_000, 1000 * 2 ** (this.fetchFailures - 1));
    setGauge("fleetReplay.fetchFailures", this.fetchFailures);
  }

  /** SIM: advance the playhead target while playing; lerp the drawn value */
  private simTick(dtMs: number, nowMs: number): void {
    if (this.disposed || !this.query) return;
    const q = this.query;
    if (this.playing) {
      this.playTarget += (dtMs / 1000) * playRateSecPerSec(q.toSec - q.fromSec, q.stepSec);
      if (this.playTarget >= q.toSec) {
        this.playTarget = q.toSec;
        this.playing = false;
        this.patchState({ playing: false });
      }
    }
    this.playValue = lerpTowards(this.playValue, this.playTarget, PLAYHEAD_HALF_LIFE_MS, dtMs);
    this.emitPlayhead(false, nowMs);
  }

  /** RENDER: heads, LOD (4 Hz), batched geometry on membership change,
   *  connectors + readouts, uniforms */
  private renderTick(now: number): void {
    if (this.disposed || !this.query) return;
    if (now - this.layerCheckAt > 1000) { this.layerCheckAt = now; this.ensureLayer(); }
    const n = this.tracks.length;
    const t = this.playValue;
    const moved = t !== this.lastPlayValueDrawn;
    if (moved || this.heads.length !== n) {
      if (this.heads.length !== n) this.heads = new Array(n).fill(null);
      for (let i = 0; i < n; i++) this.heads[i] = headAt(this.tracks[i], t);
    }
    let lodChanged = false;
    if (now - this.lastLodTick >= LOD_TICK_MS) {
      this.lastLodTick = now;
      lodChanged = this.lodTick(t);
    }
    if (this.geometryDirty) this.rebuildGeometry();
    const refresh = moved || lodChanged;
    if (refresh) {
      this.fillHeads();
      this.lastPlayValueDrawn = t;
      this.layer.setUniforms({ nowRel: t - this.query.fromSec });
    }
    // GL connectors only when the playhead/LOD moved; the DOM readouts
    // re-anchor every frame (they follow the camera like the flight tag)
    this.updateConnectors(t, refresh);
  }

  // ── fleet assembly ────────────────────────────────────────────────────────

  private rebuildFleet(): void {
    const prevLodById = new Map<string, number>();
    this.tracks.forEach((tr, i) => prevLodById.set(tr.id, this.lods[i] ?? LOD_HIDDEN));
    const merged: FleetTrack[] = [];
    const seen = new Set<string>();
    for (const tr of this.baseTracks) {
      const o = this.overrides.get(tr.id);
      merged.push(o ?? tr);
      seen.add(tr.id);
    }
    this.overrides.forEach((o, id) => { if (!seen.has(id)) merged.push(o); });
    this.tracks = merged.slice(0, maxFeatures);
    this.idIndex = new Map(this.tracks.map((tr, i) => [tr.id, i]));
    for (const tr of this.tracks) {
      if (!this.slotOf.has(tr.id)) this.slotOf.set(tr.id, this.nextSlot++);
    }
    this.lods = new Uint8Array(this.tracks.length).fill(LOD_HIDDEN);
    this.tracks.forEach((tr, i) => { this.lods[i] = (prevLodById.get(tr.id) ?? LOD_HIDDEN) as number; });
    this.heads = [];
    this.lastPlayValueDrawn = NaN;
    this.lastLodTick = -Infinity;
    this.geometryDirty = true;
    // drop cache entries for tracks no longer present
    const live = new Set(this.tracks.map((tr) => tr.id));
    this.geomCache.forEach((_v, k) => { if (!live.has(k.split("|")[0])) this.geomCache.delete(k); });
    setGauge("fleetReplay.tracks", this.tracks.length);
  }

  private isFocused(id: string): boolean {
    return !!this.focus && (this.focus.a === id || this.focus.b === id);
  }

  /** returns true when class membership changed */
  private lodTick(t: number): boolean {
    const n = this.tracks.length;
    if (n === 0) return false;
    const pose = this.readPose();
    const vp = this.viewport();
    const eye = cameraEye(pose, vp);
    const vis = cameraQueryBBox(pose, vp).visible;
    const scores = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const tr = this.tracks[i];
      if (this.isFocused(tr.id)) { scores[i] = 1e12; continue; }
      const h = this.heads[i];
      if (h) {
        scores[i] = trackScore(eyeDistanceKm(eye, h.lat, h.lon, h.altM), inBox(vis, h.lat, h.lon));
        continue;
      }
      // no head now: still worth a line while its trail is visible
      const k = fixIndexAtOrBefore(tr, t);
      if (k >= 0 && tr.t[k] >= t - FLEET_TRAIL_SEC) {
        scores[i] = trackScore(eyeDistanceKm(eye, tr.lat[k], tr.lon[k], tr.alt[k]), inBox(vis, tr.lat[k], tr.lon[k])) - 1e6;
      } else {
        scores[i] = -Infinity;
      }
    }
    const next = assignLods(scores, this.lods, this.budget);
    let membership = false;
    for (let i = 0; i < n; i++) {
      const a = this.lods[i], b = next[i];
      if (a !== b && (a <= LOD_THIN || b <= LOD_THIN)) { membership = true; break; }
    }
    const headsChanged = membership || next.some((v, i) => v !== this.lods[i]);
    this.lods = next;
    if (membership) this.geometryDirty = true;
    let full = 0, thin = 0, headOnly = 0, hidden = 0;
    for (let i = 0; i < n; i++) {
      if (next[i] === LOD_FULL) full++; else if (next[i] === LOD_THIN) thin++;
      else if (next[i] === LOD_HEAD) headOnly++; else hidden++;
    }
    const lod = { full, thin, headOnly, hidden, tracks: n };
    const prev = this.state.lod;
    if (prev.full !== full || prev.thin !== thin || prev.headOnly !== headOnly || prev.hidden !== hidden || prev.tracks !== n) {
      this.patchState({ lod });
    }
    setGauge("fleetReplay.full", full);
    setGauge("fleetReplay.thin", thin);
    setGauge("fleetReplay.headOnly", headOnly);
    return headsChanged;
  }

  private terrainState(): { on: boolean; altScale: number } {
    const terr = (this.map as unknown as { getTerrain?: () => { exaggeration?: number } | null }).getTerrain?.();
    const altScale = terr ? (Number.isFinite(terr.exaggeration) ? (terr.exaggeration as number) : 1) : 1;
    return { on: !!terr, altScale };
  }

  private rebuildGeometry(): void {
    this.geometryDirty = false;
    const q = this.query;
    if (!q) return;
    const fullIdx: number[] = [], thinIdx: number[] = [];
    this.lods.forEach((l, i) => { if (l === LOD_FULL) fullIdx.push(i); else if (l === LOD_THIN) thinIdx.push(i); });
    const { on: terrainOn, altScale } = this.terrainState();
    this.layer.setAltScale(altScale);
    const capFull = pointsPerTrack(this.budget.fullSegments, fullIdx.length);
    const capThin = pointsPerTrack(this.budget.thinSegments, thinIdx.length);
    const sig = `${fullIdx.join(",")}#${thinIdx.join(",")}#${capFull}#${capThin}#${altScale}#${terrainOn}#${this.tracks.map((x) => x.sig).join("")}`;
    if (sig === this.lastGeomSig) return;
    this.lastGeomSig = sig;
    const build = (i: number, mode: "full" | "thin", cap: number): Float32Array => {
      const tr = this.tracks[i];
      const key = `${tr.id}|${tr.sig}|${mode}|${cap}|${altScale}|${terrainOn}|${q.fromSec}`;
      const hit = this.geomCache.get(`${tr.id}|${mode}`);
      if (hit && hit.key === key) return hit.verts;
      const idx = decimateIdx(tr, cap);
      let groundZ: Float32Array | null = null;
      if (mode === "full" && terrainOn) {
        groundZ = new Float32Array(idx.length);
        const qte = (this.map as unknown as { queryTerrainElevation?: (ll: [number, number]) => number | null }).queryTerrainElevation;
        idx.forEach((k, j) => {
          let z: number | null = null;
          try { z = qte ? qte.call(this.map, [tr.lon[k], tr.lat[k]]) : null; } catch (e) { noteSoftFailure("terrainQuery", e); z = null; }
          groundZ![j] = Number.isFinite(z as number) ? (z as number) : 0;
        });
      }
      const verts = buildFleetVerts(tr, idx, mode, this.slotOf.get(tr.id) ?? 0, q.fromSec, altScale,
        groundZ, terrainOn ? CURTAIN_BELOW_TERRAIN_M : 0);
      this.geomCache.set(`${tr.id}|${mode}`, { key, verts });
      return verts;
    };
    const concat = (parts: Float32Array[]): Float32Array | null => {
      let len = 0;
      for (const p of parts) len += p.length;
      if (!len) return null;
      const out = new Float32Array(len);
      let o = 0;
      for (const p of parts) { out.set(p, o); o += p.length; }
      return out;
    };
    this.layer.setFull(concat(fullIdx.map((i) => build(i, "full", capFull))));
    this.layer.setThin(concat(thinIdx.map((i) => build(i, "thin", capThin))));
    this.applyHighlight();
  }

  private applyHighlight(): void {
    if (!this.focus) { this.layer.setUniforms({ hlA: -1, hlB: -1 }); return; }
    this.layer.setUniforms({
      hlA: this.slotOf.get(this.focus.a) ?? -1,
      hlB: this.slotOf.get(this.focus.b) ?? -1,
    });
  }

  private fillHeads(): void {
    const n = this.tracks.length;
    const need = Math.min(n, this.budget.heads) * HEAD_STRIDE;
    if (this.headInst.length < need) this.headInst = new Float32Array(Math.max(need, HEAD_STRIDE * 64));
    const { altScale } = this.terrainState();
    const lift = 30 * altScale;
    const hlOn = !!this.focus;
    let c = 0;
    for (let i = 0; i < n && c < this.budget.heads; i++) {
      if (this.lods[i] > LOD_HEAD) continue;
      const h = this.heads[i];
      if (!h) continue;
      const o = c * HEAD_STRIDE;
      const inst = this.headInst;
      inst[o] = mercX(h.lon);
      inst[o + 1] = mercY(h.lat);
      inst[o + 2] = Number.isFinite(h.altM) ? h.altM * altScale : lift;
      inst[o + 3] = (h.hdg * Math.PI) / 180;
      if (hlOn && this.isFocused(this.tracks[i].id)) {
        inst[o + 4] = FLEET_WARN_RGBA[0]; inst[o + 5] = FLEET_WARN_RGBA[1]; inst[o + 6] = FLEET_WARN_RGBA[2]; inst[o + 7] = 1;
      } else {
        const col = headColor(h.altM);
        inst[o + 4] = col[0]; inst[o + 5] = col[1]; inst[o + 6] = col[2];
        inst[o + 7] = hlOn ? FLEET_DIM + 0.1 : 0.95;
      }
      c++;
    }
    this.layer.setHeads(c ? this.headInst : null, c);
    setGauge("fleetReplay.heads", c);
  }

  // ── close approaches ──────────────────────────────────────────────────────

  private lastLabels: Array<{ mx: number; my: number; alt: number; text: string }> = [];

  private updateConnectors(t: number, rebuild: boolean): void {
    if (!rebuild) { this.placeLabels(this.lastLabels); return; }
    const list = this.data?.closeApproaches;
    if (!list || !list.length || this.query?.kind !== "aircraft") {
      if (this.connShown) { this.layer.setConnectors(null); this.connShown = false; }
      this.lastLabels = [];
      this.hideLabels(0);
      return;
    }
    const { altScale } = this.terrainState();
    const verts: number[] = [];
    const labels: Array<{ mx: number; my: number; alt: number; text: string }> = [];
    const nowRel = t - (this.query?.fromSec ?? 0);
    for (const ca of list) {
      const ia = this.idIndex.get(ca.a), ib = this.idIndex.get(ca.b);
      if (ia == null || ib == null) continue;
      const ha = this.heads[ia], hb = this.heads[ib];
      if (!ha || !hb) continue;
      const focused = !!this.focus && approachKey(this.focus) === approachKey(ca);
      // one line per pair even when the pair has several listed encounters
      if (!focused && this.focus && (this.focus.a === ca.a && this.focus.b === ca.b)) continue;
      const sep = liveSeparation(ha, hb);
      if (!focused && !withinMinima(sep)) continue;
      const ax = mercX(ha.lon), ay = mercY(ha.lat), bx = mercX(hb.lon), by = mercY(hb.lat);
      if (Math.abs(ax - bx) > 0.5) continue;
      const az = (Number.isFinite(ha.altM) ? ha.altM : 0) * altScale;
      const bz = (Number.isFinite(hb.altM) ? hb.altM : 0) * altScale;
      const slot = this.focus ? (this.slotOf.get(ca.a) ?? -1) : -1;
      const inside = withinMinima(sep);
      const col = inside ? FLEET_WARN_RGBA : [1, 0.85, 0.6, 0.9];
      pushRibbon(verts, ax, ay, az, bx, by, bz, CONN_WIDTH_PX, col, nowRel, focused ? slot : -1);
      labels.push({
        mx: (ax + bx) / 2, my: (ay + by) / 2,
        alt: (Number.isFinite(ha.altM) && Number.isFinite(hb.altM)) ? (ha.altM + hb.altM) / 2 : NaN,
        text: `≈ ${fmtSeparation(sep.horizNm, sep.vertFt)}`,
      });
      if (labels.length >= 24) break;
    }
    if (verts.length) {
      this.layer.setConnectors(new Float32Array(verts));
      this.connShown = true;
    } else if (this.connShown) {
      this.layer.setConnectors(null);
      this.connShown = false;
    }
    this.lastLabels = labels;
    this.placeLabels(labels);
  }

  private ensureLabelRoot(): HTMLDivElement | null {
    if (this.labelRoot) return this.labelRoot;
    const doc = (globalThis as { document?: Document }).document;
    if (!doc) return null;
    const root = doc.createElement("div");
    root.className = "vt-fleet-ca-labels";
    root.setAttribute("aria-hidden", "true");
    this.map.getContainer().appendChild(root);
    this.labelRoot = root;
    return root;
  }

  private placeLabels(labels: Array<{ mx: number; my: number; alt: number; text: string }>): void {
    if (!labels.length) { this.hideLabels(0); return; }
    const root = this.ensureLabelRoot();
    if (!root) return;
    const el = this.map.getContainer();
    const W = el.clientWidth, H = el.clientHeight;
    let used = 0;
    for (const L of labels) {
      const p = this.layer.projectToScreen(L.mx, L.my, L.alt, W, H);
      if (!p) continue;
      let node = this.labelPool[used];
      if (!node) {
        node = (root.ownerDocument as Document).createElement("div");
        node.className = "vt-fleet-ca-label";
        root.appendChild(node);
        this.labelPool.push(node);
      }
      if (node.textContent !== L.text) node.textContent = L.text;
      node.style.display = "";
      node.style.transform = `translate(${Math.round(p.x)}px, ${Math.round(p.y)}px) translate(-50%, -140%)`;
      used++;
    }
    this.hideLabels(used);
  }

  private hideLabels(from: number): void {
    for (let i = from; i < this.labelPool.length; i++) {
      if (this.labelPool[i].style.display !== "none") this.labelPool[i].style.display = "none";
    }
  }

  private flyToPair(ca: CloseApproachEntry): void {
    const tSec = ca.t / 1000;
    const ia = this.idIndex.get(ca.a), ib = this.idIndex.get(ca.b);
    const ha = ia != null ? headAt(this.tracks[ia], tSec) : null;
    const hb = ib != null ? headAt(this.tracks[ib], tSec) : null;
    let bearing = this.map.getBearing();
    if (ha && hb) {
      // look ACROSS the pair's separation line so both sit side by side
      const dLon = (hb.lon - ha.lon) * Math.cos((ca.lat * Math.PI) / 180);
      const dLat = hb.lat - ha.lat;
      if (dLon !== 0 || dLat !== 0) bearing = ((Math.atan2(dLon, dLat) * 180) / Math.PI + 90 + 360) % 360;
    }
    const vp = this.viewport();
    // the playhead lands at t − 60 s, when a converging pair is still up to
    // ~30 km apart: frame BOTH positions at that instant (and at t), with room
    const a60 = ia != null ? headAt(this.tracks[ia], tSec - 60) : null;
    const b60 = ib != null ? headAt(this.tracks[ib], tSec - 60) : null;
    const sep60M = a60 && b60 ? liveSeparation(a60, b60).horizNm * 1852 : 0;
    const spanM = Math.max(30_000, 1.8 * Math.max(sep60M, ca.horizNm * 1852));
    // zoom whose viewport width spans spanM: mpp(z) = mpp(0) / 2^z
    const zoom = Math.max(5, Math.min(13, Math.log2((metersPerPixel(ca.lat, 0) * vp.widthPx) / spanM)));
    const pitch = 55;
    const dest: CameraPose = { lon: ca.lon, lat: ca.lat, zoom, pitch, bearing };
    const duration = 1400;
    this.tracker.setExplicitTarget(dest, this.loop.now(), duration + 200);
    // frame the pair in the part of the map the panel does NOT cover (the
    // phone bottom sheet hides the container center) — easeTo's offset is
    // per-animation, never a persistent map padding
    const offset = framingOffset(this.map.getContainer(), this.getObstruction?.() ?? null);
    try {
      this.map.easeTo({ center: [ca.lon, ca.lat], zoom, pitch, bearing, duration, offset, easing: (x: number) => springEase(x) });
    } catch (e) { noteSoftFailure("easeTo", e); }
  }

  private loadFullRes(ca: CloseApproachEntry): void {
    const q = this.query;
    if (!q || q.kind !== "aircraft") return;
    this.focusAc?.abort();
    const ac = new AbortController();
    this.focusAc = ac;
    this.focusFullRes = "loading";
    this.patchState({ focusFullRes: "loading" });
    const tSec = Math.floor(ca.t / 1000);
    const from = Math.max(q.fromSec, tSec - FULLRES_PAD_SEC);
    const to = Math.min(q.toSec, tSec + FULLRES_PAD_SEC);
    const one = (hex: string, label: string | undefined) =>
      this.fetchImpl(`/api/data/track/aircraft/${encodeURIComponent(hex)}?from=${from}&to=${to}`, { signal: ac.signal })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`track ${r.status}`))))
        .then((d: { points?: Array<{ t: number; la: number; lo: number; al?: number | null }> }) =>
          trackFromPoints(hex, label || hex, Array.isArray(d.points) ? d.points : []));
    Promise.all([one(ca.a, ca.ca), one(ca.b, ca.cb)])
      .then(([ta, tb]) => {
        if (this.disposed || ac.signal.aborted || !this.focus || approachKey(this.focus) !== approachKey(ca)) return;
        // accept full-fidelity fixes only where they actually cover the
        // encounter (t − 60 s and t); spliced into the window track, never a
        // wholesale replacement
        const cover = [tSec - 60, tSec];
        const baseOf = (id: string) => this.baseTracks.find((tr) => tr.id === id);
        const sa = ta ? spliceOverride(baseOf(ca.a), ta, cover) : null;
        const sb = tb ? spliceOverride(baseOf(ca.b), tb, cover) : null;
        this.overrides.clear();
        if (sa) this.overrides.set(ca.a, sa);
        if (sb) this.overrides.set(ca.b, sb);
        this.rebuildFleet();
        this.applyHighlight();
        this.focusFullRes = this.overrides.size === 2 ? "ok" : "failed";
        this.patchState({ focusFullRes: this.focusFullRes });
      })
      .catch((e: unknown) => {
        if ((e as { name?: string })?.name === "AbortError" || this.disposed) return;
        this.focusFullRes = "failed";
        this.patchState({ focusFullRes: "failed" });
      });
  }

  // ── plumbing ──────────────────────────────────────────────────────────────

  private ensureLayer(): void {
    try {
      if (this.map.getLayer(this.layer.id)) return;
      const styleOk = (this.map as unknown as { isStyleLoaded?: () => boolean }).isStyleLoaded?.() ?? true;
      if (!styleOk) return;
      this.map.addLayer(this.layer as unknown as maplibregl.CustomLayerInterface);
    } catch (e) {
      // style mid-reload: the next 1 s check re-adds the layer
      noteSoftFailure("addLayer", e);
    }
  }

  /** React-facing playhead report, throttled to 10 Hz on the FRAME clock
   *  (the slider/date display is not map-visual state; the map reads the
   *  value per frame) */
  private emitPlayhead(force: boolean, frameNowMs?: number): void {
    const now = frameNowMs ?? this.loop.now();
    if (!force && now - this.lastPlayheadEmit < 100) return;
    if (!force && this.playValue === this.lastEmittedValue && this.playTarget === this.lastEmittedTarget) return;
    this.lastPlayheadEmit = now;
    this.lastEmittedValue = this.playValue;
    this.lastEmittedTarget = this.playTarget;
    this.onPlayhead(this.playValue, this.playTarget, this.playing);
  }

  private patchState(p: Partial<ReplayState>): void {
    this.state = { ...this.state, ...p };
    if (this.stateTimer) return;
    this.stateTimer = setTimeout(() => {
      this.stateTimer = null;
      if (!this.disposed) this.onState(this.state);
    }, STATE_THROTTLE_MS);
  }
}

export interface RectLike { left: number; top: number; right: number; bottom: number }

/**
 * Pixel offset (easeTo `offset`) that centers a target in the largest part of
 * the map container NOT covered by `obstruction` (the Time Machine panel):
 * a panel spanning most of the width (phone bottom sheet) → the free band
 * above/below it; a side panel → the free band beside it. [0, 0] when there
 * is no overlap.
 */
export function framingOffset(container: { getBoundingClientRect?: () => RectLike } | null, obstruction: RectLike | null): [number, number] {
  const cr = container?.getBoundingClientRect?.();
  if (!cr || !obstruction) return [0, 0];
  const x0 = Math.max(obstruction.left, cr.left), x1 = Math.min(obstruction.right, cr.right);
  const y0 = Math.max(obstruction.top, cr.top), y1 = Math.min(obstruction.bottom, cr.bottom);
  if (!(x1 > x0 && y1 > y0)) return [0, 0];
  const w = cr.right - cr.left, h = cr.bottom - cr.top;
  const cx = cr.left + w / 2, cy = cr.top + h / 2;
  if (x1 - x0 > 0.6 * w) {
    const above = y0 - cr.top, below = cr.bottom - y1;
    const mid = above >= below ? cr.top + above / 2 : y1 + below / 2;
    return [0, Math.round(mid - cy)];
  }
  const left = x0 - cr.left, right = cr.right - x1;
  const mid = right >= left ? x1 + right / 2 : cr.left + left / 2;
  return [Math.round(mid - cx), 0];
}

function inBox(b: BBox, lat: number, lon: number): boolean {
  if (lat < b.s || lat > b.n) return false;
  return b.w <= b.e ? lon >= b.w && lon <= b.e : lon >= b.w || lon <= b.e;
}

/** one screen-extruded ribbon quad in the fleet layout */
function pushRibbon(out: number[], ax: number, ay: number, az: number, bx: number, by: number, bz: number,
  w: number, c: ArrayLike<number>, t: number, slot: number): void {
  const v = (x: number, y: number, z: number, ox: number, oy: number, oz: number, side: number, dir: number) =>
    out.push(x, y, z, ox, oy, oz, side, dir, w, c[0], c[1], c[2], c[3], t, slot);
  v(ax, ay, az, bx, by, bz, -1, +1);
  v(ax, ay, az, bx, by, bz, +1, +1);
  v(bx, by, bz, ax, ay, az, -1, -1);
  v(bx, by, bz, ax, ay, az, +1, -1);
}
