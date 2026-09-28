// PLANNED-ROUTE CONTROLLER — the imperative half of the gray planned-route
// curtain for the ONE selected aircraft: owns the abortable plan fetch
// (select → now, then every PLAN_REFRESH_MS, and immediately when the plane
// is >10 nm off the drawn plan), builds the plan geometry in the live
// curtain's display datum, drives PlanCurtainLayer and the destination
// label, and publishes the card-row state through a tiny external store
// (so a 60 s refresh re-renders only the row, never the 16k-line map page).
//
// Everything the controller touches is injected (map, fetch, frame loop,
// document, timers) — the test drives it headless with fakes, including the
// teardown contract: stop() aborts the in-flight request, clears every
// timer, removes the layer, disposes it and removes the label element.
//
// DATUM (why it is not simply datamap's displayAltReal): the live curtain
// reads our own DEM tiles (lib/elevation, z9, 40-tile LRU) for its ground.
// A trans-oceanic plan would touch hundreds of z9 tiles and evict the live
// track's tiles on every build. So the plan reads the DEM only within
// PLAN_DEM_RADIUS_M of the plane (the stretch the user is actually looking
// at when following, and where the seam must match the live curtain
// exactly), the published field elevation near the two airports, and the
// rendered terrain MESH (no fetch) where terrain is on. Elsewhere the ground
// is unknown and treated as 0 — a cruise-altitude curtain is visually
// unaffected, and the geometry refreshes as the plane advances.

import {
  PLAN_REFRESH_MS,
  fetchFlightPlan,
  planDestLabel,
  planDrawable,
  planGeometryKey,
  shouldDeviationRefetch,
  type FlightPlan,
  type PlanQuery,
} from './flightPlan.js';
import {
  PlanCurtainLayer,
  reportPlanError,
  buildOriginalLineVertices,
  buildPlanGeometry,
  densifyPlan,
  type DensePlan,
  type PlanLocation,
  type PlanSeam,
} from './planCurtainLayer.js';
import { resolveGroundDisplayZ } from './groundDatum.js';
import { CURTAIN_BELOW_TERRAIN_M, distMeters } from './trackModel.js';
import { lonLatToMercator, mercatorToLonLat } from '../orbital/satBuffer.js';
import type { FrameLoop } from '../../render/frameCore.js';

export const PLAN_LAYER_ID = 'flight-plan-curtain';
/** the live curtain's layer id — the plan draws just beneath it. */
const LIVE_LAYER_ID = 'flight-track-3d';
/** our own DEM is read only this close to the plane (see header). */
export const PLAN_DEM_RADIUS_M = 90_000;
/** airport field elevation stands in for unknown ground this close to an
 *  airport (published data, not a guess). */
export const PLAN_AIRPORT_RADIUS_M = 20_000;
/** bounded DEM fill-in rebuilds (the live track's elevRetry precedent). */
export const PLAN_DEM_RETRIES = 3;
export const PLAN_DEM_RETRY_MS = 2500;

// ── the card-row store ──────────────────────────────────────────────────────

export type PlanRouteStatus = 'idle' | 'loading' | 'ok' | 'none' | 'error';

export interface PlanRouteState {
  status: PlanRouteStatus;
  plan: FlightPlan | null;
  /** client epoch ms the current plan was received (age math). */
  receivedAtMs: number;
  /** the latest refresh failed while an older plan stays drawn (Law V). */
  refreshFailed: boolean;
}

export interface PlanRouteStore {
  get(): PlanRouteState;
  set(patch: Partial<PlanRouteState>): void;
  reset(): void;
  subscribe(fn: () => void): () => void;
}

const INITIAL: PlanRouteState = { status: 'idle', plan: null, receivedAtMs: 0, refreshFailed: false };

export function createPlanRouteStore(): PlanRouteStore {
  let s: PlanRouteState = INITIAL;
  const ls = new Set<() => void>();
  const emit = () => ls.forEach((f) => { try { f(); } catch (e) { reportPlanError('store-listener', e); } });
  return {
    get: () => s,
    set(patch) { s = { ...s, ...patch }; emit(); },
    reset() { if (s !== INITIAL) { s = INITIAL; emit(); } },
    subscribe(fn) { ls.add(fn); return () => { ls.delete(fn); }; },
  };
}

// ── datum (pure given its readers) ──────────────────────────────────────────

export interface PlanDatumReaders {
  terrainOn: boolean;
  /** terrain exaggeration (1 when off). */
  exag: number;
  /** rendered-mesh ground in the display datum, 0 = not loaded (no fetch). */
  meshGround: (lon: number, lat: number) => number;
  /** our own DEM, meters MSL, null = not resident/pending (MAY fetch). */
  demGround: (lon: number, lat: number) => number | null;
}

export interface PlanDatum {
  altDisp: Float32Array;
  groundZ: Float32Array;
  drapeBelowM: number;
  /** DEM reads near the plane that are still in flight (retry trigger). */
  pending: number;
}

/**
 * Display-datum altitude + ground for every dense plan vertex — the live
 * curtain's rule (terrain ON: MSL clamped above the displayed ground, base
 * rides the mesh; terrain OFF: height above the flat plane).
 *
 * Ground is KNOWN from our own DEM within PLAN_DEM_RADIUS_M of the plane
 * (exactly the live curtain's reader there, so the seam cannot step) and
 * from the published field elevation near the two airports. Between known
 * stretches it is INTERPOLATED along the route (held flat past the last
 * known value): the ground itself is never drawn with terrain off, it only
 * converts the curtain top to the flat-plane datum — a hard switch from
 * "DEM-corrected" to "uncorrected" at the radius edge would put a
 * kilometre-high step in the curtain over high terrain. With terrain on the
 * rendered mesh wins wherever it is loaded (resolveGroundDisplayZ).
 */
export function computePlanDatum(
  dense: DensePlan,
  r: PlanDatumReaders,
  plane: { lon: number; lat: number } | null,
  airports: { lat: number; lon: number; elevM: number | null }[] = [],
): PlanDatum {
  const n = dense.n;
  const altDisp = new Float32Array(n);
  const groundZ = new Float32Array(n);
  let pending = 0;
  const exag = r.terrainOn && r.exag > 0 ? r.exag : 1;
  // pass 1: known ground (REAL meters MSL), NaN = unknown
  const known = new Float64Array(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    const lon = dense.lon[i], lat = dense.lat[i];
    let g: number | null = null;
    if (plane && distMeters(plane.lat, plane.lon, lat, lon) <= PLAN_DEM_RADIUS_M) {
      g = r.demGround(lon, lat);
      if (g == null) pending++;
    }
    if (g == null) {
      for (const a of airports) {
        if (a.elevM != null && distMeters(a.lat, a.lon, lat, lon) <= PLAN_AIRPORT_RADIUS_M) { g = a.elevM; break; }
      }
    }
    if (g != null && Number.isFinite(g)) known[i] = g;
  }
  // pass 2: fill unknown ground along the route (continuous — no steps)
  const est = fillAlongRoute(known, dense.alongM);
  for (let i = 0; i < n; i++) {
    const alt = dense.altM[i];
    const gEst = est[i];
    if (r.terrainOn) {
      const g = resolveGroundDisplayZ(r.meshGround(dense.lon[i], dense.lat[i]), gEst, exag).g;
      groundZ[i] = g;
      altDisp[i] = Number.isNaN(alt) ? NaN : Math.max(alt, g);
    } else {
      groundZ[i] = 0;
      altDisp[i] = Number.isNaN(alt) ? NaN : Math.max(0, alt - gEst);
    }
  }
  return {
    altDisp, groundZ, pending,
    drapeBelowM: r.terrainOn ? CURTAIN_BELOW_TERRAIN_M * exag : 0,
  };
}

/** Pure: NaN gaps in `v` filled by linear interpolation over `along`
 *  between the nearest known neighbours, held flat beyond the first/last
 *  known value; all-unknown → 0 (sea level, the flat map's own datum). */
export function fillAlongRoute(v: Float64Array, along: Float64Array): Float64Array {
  const n = v.length;
  const out = new Float64Array(n);
  let prev = -1;
  for (let i = 0; i < n; i++) {
    if (!Number.isNaN(v[i])) { out[i] = v[i]; prev = i; continue; }
    let next = i + 1;
    while (next < n && Number.isNaN(v[next])) next++;
    if (prev < 0 && next >= n) { for (let k = i; k < n; k++) out[k] = 0; break; }
    for (let k = i; k < next && k < n; k++) {
      if (prev < 0) out[k] = v[next];
      else if (next >= n) out[k] = v[prev];
      else {
        const span = along[next] - along[prev];
        const u = span > 0 ? (along[k] - along[prev]) / span : 0;
        out[k] = v[prev] + (v[next] - v[prev]) * u;
      }
    }
    i = next - 1;
  }
  return out;
}

// ── the controller ──────────────────────────────────────────────────────────

/** The subset of maplibre-gl's Map the controller uses. */
export interface PlanMapLike {
  getLayer(id: string): unknown;
  addLayer(layer: unknown, beforeId?: string): unknown;
  removeLayer(id: string): unknown;
  getTerrain?(): { exaggeration?: number } | null | undefined;
  queryTerrainElevation?(lngLat: [number, number]): number | null | undefined;
  getContainer?(): { appendChild(el: unknown): unknown };
  triggerRepaint?(): void;
}

export interface PlanLive {
  lon: number;
  lat: number;
  altM: number | null;
  trkDeg: number | null;
  callsign: string | null;
}

export interface PlanRouteDeps {
  map: PlanMapLike;
  hex: string;
  store: PlanRouteStore;
  /** the plane's latest real fix (query params + DEM radius centre). */
  getLive: () => PlanLive | null;
  /** where the live curtain currently ends (the seam). */
  getSeam: () => PlanSeam | null;
  fetchImpl?: typeof fetch;
  loop?: FrameLoop | null;
  /** document for the label + hidden-tab gate; null = no label (tests). */
  doc?: Document | null;
  /** context-restore registry (datamap's customLayerRegistryRef). */
  registry?: Map<string, unknown> | null;
  demGround?: (lon: number, lat: number) => number | null;
  now?: () => number;
  setInterval?: (fn: () => void, ms: number) => unknown;
  clearInterval?: (h: unknown) => void;
  setTimeout?: (fn: () => void, ms: number) => unknown;
  clearTimeout?: (h: unknown) => void;
}

export interface PlanRouteHandle {
  layer: PlanCurtainLayer;
  /** force a fetch now (tests / manual). */
  refetch(reason?: string): void;
  stop(): void;
  isStopped(): boolean;
}

/** Build the destination label element (inline theme tokens — no new CSS
 *  file, no hardcoded palette hex). pointer-events none: it annotates the
 *  map, it never covers a control. */
export function createPlanLabelEl(doc: Document): HTMLElement {
  const el = doc.createElement('div');
  el.setAttribute('data-vt-plan-dest', '');
  el.setAttribute('aria-hidden', 'true');
  el.style.cssText = 'position:absolute;left:0;top:0;width:0;height:0;pointer-events:none;'
    + 'z-index:2;display:none;will-change:transform;';
  const pin = doc.createElement('span');
  pin.style.cssText = 'position:absolute;left:-5px;top:-5px;width:8px;height:8px;transform:rotate(45deg);'
    + 'border:1.5px solid var(--text-secondary);background:var(--bg-primary);box-sizing:border-box;';
  const chip = doc.createElement('span');
  chip.setAttribute('data-vt-plan-dest-text', '');
  // the in-scene flight tag's handoff tokens (--flight-panel/-border/-ink)
  chip.style.cssText = 'position:absolute;left:0;bottom:9px;transform:translateX(-50%);white-space:nowrap;'
    + 'font:600 10.5px/1.35 var(--font-mono);letter-spacing:.04em;color:var(--flight-ink);'
    + 'background:var(--flight-panel);border:1px solid var(--flight-panel-border);border-radius:6px;'
    + 'padding:2px 7px;';
  el.appendChild(pin);
  el.appendChild(chip);
  return el;
}

export function startPlanRoute(deps: PlanRouteDeps): PlanRouteHandle {
  const { map, hex, store } = deps;
  const now = deps.now ?? (() => Date.now());
  const setIv = deps.setInterval ?? ((fn: () => void, ms: number) => setInterval(fn, ms));
  const clearIv = deps.clearInterval ?? ((h: unknown) => clearInterval(h as ReturnType<typeof setInterval>));
  const setTo = deps.setTimeout ?? ((fn: () => void, ms: number) => setTimeout(fn, ms));
  const clearTo = deps.clearTimeout ?? ((h: unknown) => clearTimeout(h as ReturnType<typeof setTimeout>));
  const demGround = deps.demGround ?? (() => null);
  const doc = deps.doc ?? null;

  const layer = new PlanCurtainLayer({ id: PLAN_LAYER_ID, loop: deps.loop ?? undefined });
  let stopped = false;
  let plan: FlightPlan | null = null;
  let geomKey = '';
  let datumKey = '';
  let inFlight: AbortController | null = null;
  let lastFetchStart = -Infinity;
  let demRetries = 0;
  let demTimer: unknown = null;

  const labelEl = doc ? createPlanLabelEl(doc) : null;
  if (labelEl) {
    try { map.getContainer?.().appendChild(labelEl); } catch (e) { reportPlanError('label-mount', e); }
  }

  const readDatumKey = (): string => {
    try {
      const t = map.getTerrain?.();
      return t ? `on|${t.exaggeration ?? 1}` : 'off';
    } catch (e) { reportPlanError('terrain-read', e); return 'off'; }
  };

  const ensureLayer = () => {
    try {
      if (!map.getLayer(PLAN_LAYER_ID)) {
        map.addLayer(layer, map.getLayer(LIVE_LAYER_ID) ? LIVE_LAYER_ID : undefined);
      }
    } catch (e) {
      reportPlanError('add-layer', e); // style mid-load — the next rebuild retries
    }
    deps.registry?.set(PLAN_LAYER_ID, layer);
  };

  // the display arrays of the geometry currently installed (DEM-refinement
  // rebuilds that change nothing must not touch the GPU)
  let shownAlt: Float32Array | null = null;
  let shownGround: Float32Array | null = null;
  const sameArrays = (a: Float32Array | null, b: Float32Array): boolean => {
    if (!a || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      const x = a[i], y = b[i];
      if (Number.isNaN(x) !== Number.isNaN(y)) return false;
      if (!Number.isNaN(x) && Math.abs(x - y) > 0.5) return false;
    }
    return true;
  };

  /** reason: 'plan' (a new plan — crossfade), 'datum' (terrain toggled /
   *  exaggeration — crossfade), 'dem' (late DEM tiles for the same plan —
   *  swap in place, or nothing at all when the numbers did not move). */
  const rebuild = (reason: 'plan' | 'datum' | 'dem' = 'plan') => {
    if (stopped) return;
    if (demTimer != null) { clearTo(demTimer); demTimer = null; }
    datumKey = readDatumKey();
    if (!planDrawable(plan)) {
      layer.setPlan(null);
      layer.setOriginal(null);
      layer.setLabel(null);
      shownAlt = shownGround = null;
      return;
    }
    const p = plan;
    const t = (() => { try { return map.getTerrain?.() ?? null; } catch (e) { reportPlanError('terrain-read', e); return null; } })();
    const readers: PlanDatumReaders = {
      terrainOn: !!t,
      exag: t?.exaggeration ?? 1,
      meshGround: (lon, lat) => {
        try { return map.queryTerrainElevation?.([lon, lat]) ?? 0; } catch (e) { reportPlanError('mesh-query', e); return 0; }
      },
      demGround,
    };
    const live = deps.getLive();
    // DEM-radius centre: the live fix, else the live curtain's drawn end
    let plane: { lon: number; lat: number } | null = live ? { lon: live.lon, lat: live.lat } : null;
    if (!plane) {
      const sm = deps.getSeam();
      if (sm && Number.isFinite(sm.mercX) && Number.isFinite(sm.mercY)) {
        const ll = mercatorToLonLat(sm.mercX, sm.mercY);
        plane = { lon: ll.lonDeg, lat: ll.latDeg };
      }
    }
    const airports = [p.origin, p.destination].filter((a): a is NonNullable<typeof a> => !!a);
    const dense = densifyPlan(p.points);
    const d = computePlanDatum(dense, readers, plane, airports);
    ensureLayer();
    const unchanged = reason === 'dem' && sameArrays(shownAlt, d.altDisp) && sameArrays(shownGround, d.groundZ);
    if (!unchanged) {
      layer.setPlan(
        buildPlanGeometry({ dense, altDisp: d.altDisp, groundZ: d.groundZ, drapeBelowM: d.drapeBelowM }),
        { crossfade: reason !== 'dem' },
      );
      shownAlt = d.altDisp;
      shownGround = d.groundZ;
    }
    let pending = d.pending;
    if (p.originalPoints) {
      const od = densifyPlan(p.originalPoints);
      const odat = computePlanDatum(od, readers, plane, airports);
      pending += odat.pending;
      layer.setOriginal(buildOriginalLineVertices(od, odat.altDisp, odat.groundZ));
    } else {
      layer.setOriginal(null);
    }
    const text = planDestLabel(p);
    if (labelEl && text && dense.n >= 2) {
      const chip = labelEl.querySelector('[data-vt-plan-dest-text]');
      if (chip) chip.textContent = text;
      const last = dense.n - 1;
      const m = lonLatToMercator(dense.lon[last], dense.lat[last]);
      layer.setLabel({ el: labelEl, mercX: m.x, mercY: m.y, z: d.groundZ[last] });
    } else {
      layer.setLabel(null);
    }
    // DEM tiles near the plane still in flight → bounded refinement
    if (pending > 0 && demRetries < PLAN_DEM_RETRIES) {
      demRetries++;
      demTimer = setTo(() => { demTimer = null; rebuild('dem'); }, PLAN_DEM_RETRY_MS);
    }
  };

  const apply = (next: FlightPlan) => {
    plan = next;
    store.set({
      status: planDrawable(next) ? 'ok' : 'none',
      plan: next,
      receivedAtMs: now(),
      refreshFailed: false,
    });
    const k = planGeometryKey(next);
    if (k === geomKey) return; // unchanged plan: no rebuild, no crossfade
    geomKey = k;
    demRetries = 0;
    rebuild('plan');
  };

  const doFetch = (_reason: string) => {
    if (stopped) return;
    inFlight?.abort();
    const ac = new AbortController();
    inFlight = ac;
    lastFetchStart = now();
    if (!plan) store.set({ status: 'loading' });
    const live = deps.getLive();
    const q: PlanQuery = live
      ? { callsign: live.callsign, lat: live.lat, lon: live.lon, altM: live.altM, trkDeg: live.trkDeg }
      : {};
    fetchFlightPlan(hex, q, ac.signal, deps.fetchImpl).then(
      (p) => {
        if (stopped || ac.signal.aborted) return;
        inFlight = null;
        apply(p);
      },
      (err: unknown) => {
        if (stopped || ac.signal.aborted) return;
        inFlight = null;
        reportPlanError('fetch', err); // counted + logged once, never silent
        // Law V: a failed refresh keeps the last plan drawn and says so
        if (plan) store.set({ refreshFailed: true });
        else store.set({ status: 'error' });
      },
    );
  };

  layer.setSeamSource(deps.getSeam);
  layer.setOnFrame(() => {
    if (stopped || !plan) return;
    // terrain toggled / exaggeration moved: re-datum the SAME plan (the live
    // curtain's repaintTrail3d counterpart), detected in the frame loop
    if (readDatumKey() !== datumKey) { demRetries = 0; rebuild('datum'); }
    // >10 nm off the DRAWN plan → re-fetch now (rate-limited; evaluated
    // every frame so a plane frozen at the glide cap still triggers it)
    const loc: PlanLocation | null = layer.getLocation();
    if (loc && shouldDeviationRefetch(loc.crossTrackM, lastFetchStart, now(), inFlight != null)) doFetch('deviation');
  });

  const iv = setIv(() => {
    if (doc && (doc as { hidden?: boolean }).hidden) return; // backgrounded: no polling
    doFetch('periodic');
  }, PLAN_REFRESH_MS);
  // first fetch on the next task: the page's other selection effects (which
  // seed the live fix the query carries) run in the same commit, after ours
  let kick: unknown = setTo(() => { kick = null; doFetch('select'); }, 0);

  return {
    layer,
    refetch: (reason = 'manual') => doFetch(reason),
    isStopped: () => stopped,
    stop() {
      if (stopped) return;
      stopped = true;
      inFlight?.abort();
      inFlight = null;
      clearIv(iv);
      if (kick != null) { clearTo(kick); kick = null; }
      if (demTimer != null) { clearTo(demTimer); demTimer = null; }
      deps.registry?.delete(PLAN_LAYER_ID);
      try { if (map.getLayer(PLAN_LAYER_ID)) map.removeLayer(PLAN_LAYER_ID); } catch (e) { reportPlanError('remove-layer', e); }
      layer.dispose();
      try { labelEl?.remove(); } catch (e) { reportPlanError('label-remove', e); }
      store.reset();
    },
  };
}
