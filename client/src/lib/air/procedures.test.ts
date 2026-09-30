// Procedures client contract: URL building, wire validation, the path ->
// map-source split (invalid coordinates dropped, Law IV cap), fix symbols by
// kind (SYMBOLS NOT DOTS), the plate-overlay gate, crop math, and the
// ProcedureLayer lifecycle against a fake map (fade-in targets, plate
// ready-gate + opacity, style-reload re-install, full teardown on dispose).
// Run: npx tsx --test client/src/lib/air/procedures.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  cropPixelRect, cycleBadge, fetchFlightProcedures, fetchProcedurePath, filedLabel, fixSymbol, flightProceduresUrl,
  overlayScale, pathToLayers, plateCornersUsable, plateGeorefUrl, procedurePathUrl, windText,
  type FiledProc, type PlateGeoref, type ProcedurePath,
} from "./procedures.js";
import {
  FADE_MS, LYR_CASING, LYR_FIXES, LYR_LEGS, LYR_PLATE, ProcedureLayer, SRC_LEGS, SRC_PLATE, maxFeatures, vramBudget, PLATE_MAX_PX,
} from "./procedureLayer.js";
import { FrameLoop, type FrameHost } from "../../render/frameCore.js";

const PATH: ProcedurePath = {
  type: "FeatureCollection",
  features: [
    { type: "Feature", geometry: { type: "LineString", coordinates: [[-97.66, 30.45], [-97.66, 30.4]] }, properties: { kind: "leg", approx: false, missed: false, selected: true } },
    { type: "Feature", geometry: { type: "LineString", coordinates: [[-97.66, 30.2], [Number.NaN, 30.1]] }, properties: { kind: "leg", approx: true, missed: true, selected: true } },
    { type: "Feature", geometry: { type: "Point", coordinates: [-97.66, 30.45] }, properties: { kind: "fix", ident: "DOFFS", label: "DOFFS 5000A", role: "IAF", fixKind: "waypoint", missed: false } },
    { type: "Feature", geometry: { type: "Point", coordinates: [-97.53, 30.38] }, properties: { kind: "fix", ident: "CWK", label: "CWK", role: null, fixKind: "navaid", missed: false } },
    { type: "Feature", geometry: { type: "Point", coordinates: [-97.659, 30.259] }, properties: { kind: "fix", ident: "DDTOO", label: "DDTOO 1600 (GS)", role: "FAF", fixKind: "waypoint", missed: false } },
  ],
  meta: { airport: "KAUS", procedure: "I18L", kind: "IAP", name: "ILS RWY 18L", selectedTransition: null, legs: 3, approxLegs: 1, unresolvedLegs: 0, truncated: false, bbox: null },
};

test("URLs: callsign/airport validated, transitions encoded, georef keyed by the CIFP procedure", () => {
  assert.equal(flightProceduresUrl("swa1234", "KDFW", "KAUS"), "/api/data/procedures/flight/SWA1234?dep=KDFW&arr=KAUS");
  assert.equal(flightProceduresUrl("UAL9", null, "bad icao"), "/api/data/procedures/flight/UAL9");
  assert.equal(flightProceduresUrl("", "KDFW", "KAUS"), null);
  assert.equal(procedurePathUrl("KAUS", "BLEWE5", "ACT"), "/api/data/procedures/KAUS/BLEWE5/path?transition=ACT");
  assert.equal(plateGeorefUrl({ name: "x", code: "IAP", pdf: "00556IL18L.PDF", url: "/api/data/plates/KAUS/00556IL18L.PDF", amdt: null, amdtDate: null }, "I18L"),
    "/api/data/plates/KAUS/00556IL18L.PDF/georef?proc=I18L");
});

test("fetch helpers: abortable, server error text surfaced, malformed wire refused", async () => {
  const ok = (async () => new Response(JSON.stringify(PATH), { status: 200 })) as typeof fetch;
  assert.equal((await fetchProcedurePath("/p", new AbortController().signal, ok)).meta.procedure, "I18L");
  const err = (async () => new Response(JSON.stringify({ error: "I99 not found at KAUS" }), { status: 404 })) as typeof fetch;
  await assert.rejects(fetchProcedurePath("/p", new AbortController().signal, err), /I99 not found/);
  const junk = (async () => new Response(JSON.stringify({ hello: 1 }), { status: 200 })) as typeof fetch;
  await assert.rejects(fetchFlightProcedures("/f", new AbortController().signal, junk), /malformed/);
  const ac = new AbortController();
  const slow = ((_u: string, init?: RequestInit) => new Promise<Response>((_, rej) => init?.signal?.addEventListener("abort", () => rej(new DOMException("aborted", "AbortError"))))) as typeof fetch;
  const p = fetchProcedurePath("/p", ac.signal, slow);
  ac.abort();
  await assert.rejects(p, /aborted/);
});

test("pathToLayers: legs vs fixes, invalid coordinates dropped (never guessed), cap enforced", () => {
  const L = pathToLayers(PATH);
  assert.equal(L.legs.features.length, 1, "the NaN leg keeps 1 valid point -> dropped");
  assert.equal(L.fixes.features.length, 3);
  assert.equal(L.dropped, 1);
  const capped = pathToLayers(PATH, 2);
  assert.equal(capped.legs.features.length + capped.fixes.features.length, 2);
  assert.equal(pathToLayers(null).legs.features.length, 0);
  assert.equal(maxFeatures, 1500);
  assert.ok(vramBudget >= Math.ceil((PLATE_MAX_PX * PLATE_MAX_PX * 4) / 1048576), "declared VRAM covers the capped plate raster");
});

test("fix symbols encode the kind (SYMBOLS NOT DOTS)", () => {
  assert.equal(fixSymbol({ role: "FAF", fixKind: "waypoint" }), "vt-fix-faf");
  assert.equal(fixSymbol({ role: null, fixKind: "navaid" }), "vt-fix-nav");
  assert.equal(fixSymbol({ role: "IAF", fixKind: "waypoint" }), "vt-fix-wpt");
});

const GEO: PlateGeoref = {
  georeferenced: true, reason: "3 fix symbols fit", rmsNm: 0.028, maxResidualNm: 0.039, controlPoints: [{ fix: "CWK", residualNm: 0.021 }],
  scaleNmPerInch: 6.765, page: { width: 387.36, height: 594, rotate: 0 }, planView: { x0: 18.21, y0: 208.31, x1: 369.15, y1: 456.36 },
  corners: [[-97.977297, 30.521679], [-97.341454, 30.516539], [-97.345657, 30.128092], [-97.9815, 30.133232]],
  embeddedGeoPdf: true, embeddedAgreementNm: 0.33, method: "geopdf-seeded", cycle: "2609", pdf: "00556IL18L.PDF", chart: "ILS OR LOC RWY 18L", proc: "I18L",
  pdfUrl: "/api/data/plates/KAUS/00556IL18L.PDF",
};

test("plate overlay gate: only a passed georeference with sane corners is placed", () => {
  assert.equal(plateCornersUsable(GEO), true);
  assert.equal(plateCornersUsable({ ...GEO, georeferenced: false }), false);
  assert.equal(plateCornersUsable({ ...GEO, corners: null }), false);
  assert.equal(plateCornersUsable({ ...GEO, corners: [[0, 0], [10, 0], [10, 10], [0, 10]] }), false, "a 10° plate is not a plan view");
  assert.equal(plateCornersUsable({ ...GEO, corners: [[Number.NaN, 0], [1, 0], [1, 1], [0, 1]] }), false);
  assert.equal(plateCornersUsable(null), false);
});

test("crop math: PDF points (origin bottom-left) -> canvas pixels (origin top-left), bounded scale", () => {
  const pv = GEO.planView!;
  const s = overlayScale(pv, 1536);
  assert.ok(Math.abs(Math.max(pv.x1 - pv.x0, pv.y1 - pv.y0) * s - 1536) < 1e-6);
  const r = cropPixelRect(pv, 594, 2);
  assert.deepEqual(r, { x: 36, y: 275, w: 702, h: 496 });
  assert.equal(overlayScale({ x0: 0, y0: 0, x1: 2000, y1: 100 }, 1536), 1, "never below 1 px/pt");
});

test("display strings: cycle badge, filed label with transition/version, wind", () => {
  assert.equal(cycleBadge({ ident: "2609", expires: "2026-10-01T09:01:00.000Z" }), "CIFP 2609 · valid to Oct 1");
  assert.equal(cycleBadge(null), "cycle unknown");
  const f = { id: "BLEWE5", name: "BLEWE FIVE", transition: "ACT", filedAs: "BLEWE4", versionMismatch: true } as FiledProc;
  assert.equal(filedLabel("STAR", f), "STAR BLEWE FIVE · ACT transition (filed BLEWE4)");
  assert.equal(windText({ dirDeg: 160, speedKt: 9, gustKt: null, obsTime: null, raw: null }), "wind 160@9 kt");
  assert.equal(windText({ dirDeg: null, speedKt: 3, gustKt: 12, obsTime: null, raw: null }), "wind VRB@3G12 kt");
});

// ── ProcedureLayer against a fake map ───────────────────────────────────────
function fakeMap() {
  const sources = new Map<string, { spec: Record<string, unknown>; data?: unknown; setData(d: unknown): void }>();
  const layers = new Map<string, { spec: Record<string, unknown>; before?: string }>();
  const paints: Array<[string, string, unknown]> = [];
  return {
    sources, layers, paints,
    getSource: (id: string) => sources.get(id),
    addSource: (id: string, spec: Record<string, unknown>) => {
      const s = { spec, data: spec.data, setData(d: unknown) { s.data = d; } };
      sources.set(id, s);
    },
    removeSource: (id: string) => { sources.delete(id); },
    getLayer: (id: string) => layers.get(id),
    addLayer: (spec: Record<string, unknown>, before?: string) => { layers.set(String(spec.id), { spec, before }); },
    removeLayer: (id: string) => { layers.delete(id); },
    setPaintProperty: (l: string, p: string, v: unknown) => { paints.push([l, p, v]); },
  };
}

test("ProcedureLayer: layers added transparent then eased to targets; symbols per kind; dispose frees everything", () => {
  const m = fakeMap();
  const revoked: string[] = [];
  const layer = new ProcedureLayer(m, { color: (t) => `token(${t})`, revoke: (u) => revoked.push(u) });
  layer.setPath(PATH);
  for (const id of [LYR_CASING, LYR_LEGS, LYR_FIXES]) assert.ok(m.layers.get(id), id);
  const legsPaint = m.layers.get(LYR_LEGS)!.spec.paint as Record<string, unknown>;
  assert.equal(legsPaint["line-opacity"], 0, "added transparent");
  assert.deepEqual(legsPaint["line-opacity-transition"], { duration: FADE_MS, delay: 0 });
  assert.equal(legsPaint["line-color"], "token(--accent-purple)", "theme token, never a hardcoded colour");
  assert.ok(m.paints.some(([l, p, v]) => l === LYR_LEGS && p === "line-opacity" && Array.isArray(v)), "eased to its target");
  const fixes = m.sources.get("vt-proc-fixes")!.data as { features: Array<{ properties: { icon: string } }> };
  assert.deepEqual(fixes.features.map((f) => f.properties.icon), ["vt-fix-wpt", "vt-fix-nav", "vt-fix-faf"]);
  // plate: added under the path, faded from 0 to the user's opacity
  layer.setPlate({ url: "blob:plate-1", corners: GEO.corners! });
  const plateL = m.layers.get(LYR_PLATE)!;
  assert.equal(plateL.before, LYR_CASING);
  assert.equal((plateL.spec.paint as Record<string, unknown>)["raster-opacity"], 0);
  assert.deepEqual(m.paints.at(-1), [LYR_PLATE, "raster-opacity", 0.7]);
  assert.deepEqual((m.sources.get(SRC_PLATE)!.spec as { coordinates: unknown }).coordinates, GEO.corners);
  layer.setOpacity(0.3);
  assert.deepEqual(m.paints.at(-1), [LYR_PLATE, "raster-opacity", 0.3]);
  // replacing the plate revokes the old bitmap
  layer.setPlate({ url: "blob:plate-2", corners: GEO.corners! });
  assert.deepEqual(revoked, ["blob:plate-1"]);
  layer.dispose();
  assert.equal(m.layers.size, 0);
  assert.equal(m.sources.size, 0);
  assert.deepEqual(revoked, ["blob:plate-1", "blob:plate-2"]);
  layer.setPath(PATH); // no-op after dispose
  assert.equal(m.layers.size, 0);
  assert.equal(layer.isDisposed(), true);
});

test("ProcedureLayer: a style reload that drops the sources is repaired by the frame-loop check; unregistered on dispose", () => {
  let cb: ((t: number) => void) | null = null;
  let clock = 0;
  const host: FrameHost = {
    now: () => clock,
    scheduler: { request: (fn: (t: number) => void) => { cb = fn; return 1; }, cancel: () => { cb = null; } },
    visibility: { isHidden: () => false, subscribe: () => () => {} },
  };
  const loop = new FrameLoop(() => host);
  const m = fakeMap();
  const layer = new ProcedureLayer(m, { loop, color: () => "c" });
  layer.setPath(PATH);
  // basemap switch: every native source/layer is gone
  m.layers.clear(); m.sources.clear();
  for (let i = 0; i < 40 && cb; i++) { clock += 16; const f: (t: number) => void = cb; cb = null; f(clock); }
  assert.ok(m.layers.get(LYR_LEGS) && m.sources.get(SRC_LEGS), "re-installed from the frame loop");
  const legs = m.sources.get(SRC_LEGS)!.data as { features: unknown[] };
  assert.equal(legs.features.length, 1, "with the current path's data");
  layer.dispose();
  const regsAfter = (loop as unknown as { regs: unknown[] }).regs.length;
  assert.equal(regsAfter, 0, "frame check unregistered");
});
