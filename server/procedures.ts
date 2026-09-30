// PROCEDURES + PLATES API — "say they filed the ILS: pull up that plate and
// show it on the map with the path on the plate" (human request 2026-09-30).
//
// REALITY THE API IS BUILT AROUND (and says in every response):
//  * Approaches are ASSIGNED by ATC, not filed. A flight plan names the
//    departure procedure (DP/SID) and the arrival (STAR) — so those are
//    matched from the filed route text; approaches are only SUGGESTED, from
//    the destination's runways and the current METAR wind.
//  * The procedure PATH comes from FAA CIFP (ARINC 424) — the authoritative
//    vector. The plate image is a backdrop, overlaid on the map only when its
//    plan view georeferences (server/plateGeoref.ts); otherwise it is shown
//    in a side viewer with the reason.
//  * NOT FOR NAVIGATION — display/situational awareness only.
//
// Endpoints (registered by ONE line in routes.ts):
//   GET /api/data/procedures/status
//   GET /api/data/procedures/flight/:callsign?dep=&arr=
//   GET /api/data/procedures/:airport
//   GET /api/data/procedures/:airport/:procId/path?transition=
//   GET /api/data/plates/:airport/:pdf
//   GET /api/data/plates/:airport/:pdf/georef?proc=

import type { Express, Request, Response } from "express";
import {
  CifpStore, approachName, parseApproachIdent, procedureFixes, procedurePath, segmentRole,
  type AirportData, type CifpIndex, type CifpProcedure, type AiracCycle, type ProcKind, type LL,
} from "./cifp";
import { DtppStore, PlateCache, chartsForProcedure, type DtppAirport, type DtppChart, type DtppIndex } from "./dtpp";
import { georeferencePlate, type GeorefResult } from "./plateGeoref";
import { getFlightPlanContext } from "./flightPlans";
import { lookupFix } from "./navFixes";

export const NOT_FOR_NAVIGATION = "NOT FOR NAVIGATION — FAA CIFP/d-TPP data drawn for situational awareness only.";
export const APPROACH_HONESTY =
  "Approaches are assigned by ATC, not filed: these are SUGGESTIONS from the destination's runways and the latest METAR wind — the aircraft may fly any of them, or a visual.";

// ── pure helpers (exported for tests) ───────────────────────────────────────

export const normAirport = (raw: string): string | null => {
  const s = String(raw || "").trim().toUpperCase();
  return /^[A-Z0-9]{3,4}$/.test(s) ? s : null;
};

/** CIFP is keyed by ICAO; accept "AUS" for a US airport too. */
export function cifpAirport(idx: CifpIndex, id: string): AirportData | null {
  return idx.airport(id) ?? (id.length === 3 ? idx.airport(`K${id}`) : null);
}
export function dtppAirport(d: DtppIndex | null, ap: AirportData | null, id: string): DtppAirport | null {
  if (!d) return null;
  return d.airports.get(id) ?? (ap ? d.airports.get(ap.icao) ?? d.airports.get(ap.icao.replace(/^K/, "")) : null) ?? null;
}

export interface ChartRef { name: string; code: string; pdf: string; url: string; amdt: string | null; amdtDate: string | null }
const chartRef = (icao: string, c: DtppChart): ChartRef => ({
  name: c.name, code: c.code, pdf: c.pdf, url: `/api/data/plates/${icao}/${c.pdf}`, amdt: c.amdt, amdtDate: c.amdtDate,
});

export interface ProcSummary {
  id: string; kind: ProcKind; name: string;
  runway: string | null; typeName: string | null;
  transitions: Array<{ id: string; role: string }>;
  runways: string[];
  charts: ChartRef[];
}

export function summarize(ap: AirportData, p: CifpProcedure, dap: DtppAirport | null, navaidName: (id: string) => string | null): ProcSummary {
  const a = p.kind === "IAP" ? parseApproachIdent(p.id) : null;
  const charts = dap ? chartsForProcedure(p.kind, p.id, dap.charts, navaidName).map((c) => chartRef(ap.icao, c)) : [];
  const trs = p.transitions.map((t) => ({ id: t.id, role: segmentRole(p.kind, t.routeType) }));
  const runways = trs.filter((t) => t.role === "runway transition" && /^RW/.test(t.id)).map((t) => t.id.slice(2));
  return {
    id: p.id, kind: p.kind,
    // approaches keep the CIFP-coded name (ILS and LOC are separate coded
    // procedures that share one "ILS OR LOC" chart); routes take the chart's
    name: p.kind === "IAP" ? approachName(p.id) : (charts[0]?.name.replace(/,\s*CONT\.\s*\d+/, "") ?? p.id),
    runway: a?.runway ?? null, typeName: a?.typeName ?? null,
    transitions: trs.filter((t) => t.id),
    runways,
    charts,
  };
}

export interface FiledMatch {
  id: string;
  transition: string | null;
  /** filed token when it differs from the CIFP ident (older/newer version) */
  filedAs: string | null;
  versionMismatch: boolean;
}

/**
 * The DP and STAR named in a filed route string ("KAUS.AUS7.CWK J25 …
 * ACT.BLEWE5.KAUS"). Exact ident first; a same-name different-version token
 * (BLEWE4 filed, CIFP now BLEWE5) is reported as a version mismatch rather
 * than silently promoted. The SID's enroute transition is the token after
 * it; the STAR's is the token before it.
 */
export function findFiledProcedures(routeText: string | null, dep: AirportData | null, arr: AirportData | null): { dp: FiledMatch | null; star: FiledMatch | null } {
  const toks = String(routeText || "").toUpperCase().split(/[\s.+/]+/).filter(Boolean);
  const find = (ap: AirportData | null, kind: ProcKind): FiledMatch | null => {
    if (!ap) return null;
    const procs = ap.procedures.filter((p) => p.kind === kind);
    const transOf = (p: CifpProcedure) => new Set(p.transitions.filter((t) => {
      const r = segmentRole(kind, t.routeType);
      return r === "enroute transition";
    }).map((t) => t.id));
    for (let i = 0; i < toks.length; i++) {
      const tk = toks[i];
      let p = procs.find((x) => x.id === tk);
      let mismatch = false;
      if (!p) {
        const m = /^([A-Z]{3,5})(\d)$/.exec(tk);
        if (m) { p = procs.find((x) => x.id.replace(/\d$/, "") === m[1]); mismatch = !!p; }
      }
      if (!p) continue;
      const tr = transOf(p);
      const nb = kind === "SID" ? toks[i + 1] : toks[i - 1];
      return { id: p.id, transition: nb && tr.has(nb) ? nb : null, filedAs: mismatch ? tk : null, versionMismatch: mismatch };
    }
    return null;
  };
  return { dp: find(dep, "SID"), star: find(arr, "STAR") };
}

export interface Wind { dirDeg: number | null; speedKt: number; gustKt: number | null; obsTime: string | null; raw: string | null }
export interface RunwayWind { runway: string; headingTrue: number; headwindKt: number | null; crosswindKt: number | null }

/** True runway heading: geometric bearing between opposite thresholds when
 *  both are coded, else magnetic bearing + variation. */
export function runwayHeadings(ap: AirportData): Array<{ runway: string; headingTrue: number }> {
  const out: Array<{ runway: string; headingTrue: number }> = [];
  for (const r of Array.from(ap.runways.values())) {
    const id = r.id.replace(/^RW/, "");
    const m = /^(\d{2})([LRC]?)$/.exec(id);
    if (!m) continue;
    const opp = `RW${String(((Number(m[1]) + 17) % 36) + 1).padStart(2, "0")}${m[2] === "L" ? "R" : m[2] === "R" ? "L" : m[2]}`;
    const o = ap.runways.get(opp);
    let hdg: number | null = null;
    if (o) {
      const la1 = r.ll.lat * Math.PI / 180, la2 = o.ll.lat * Math.PI / 180, dlo = (o.ll.lon - r.ll.lon) * Math.PI / 180;
      hdg = (Math.atan2(Math.sin(dlo) * Math.cos(la2), Math.cos(la1) * Math.sin(la2) - Math.sin(la1) * Math.cos(la2) * Math.cos(dlo)) * 180 / Math.PI + 360) % 360;
    } else if (r.bearingMag != null) hdg = (r.bearingMag + ap.magVar + 360) % 360;
    if (hdg != null) out.push({ runway: id, headingTrue: Math.round(hdg * 10) / 10 });
  }
  return out;
}

export const LIGHT_WIND_KT = 5;
export function runwayWinds(ap: AirportData, wind: Wind | null): RunwayWind[] {
  return runwayHeadings(ap).map(({ runway, headingTrue }) => {
    if (!wind || wind.dirDeg == null) return { runway, headingTrue, headwindKt: null, crosswindKt: null };
    const d = (wind.dirDeg - headingTrue) * Math.PI / 180;
    return { runway, headingTrue, headwindKt: Math.round(wind.speedKt * Math.cos(d) * 10) / 10, crosswindKt: Math.round(Math.abs(wind.speedKt * Math.sin(d)) * 10) / 10 };
  });
}

const TYPE_RANK: Record<string, number> = { I: 0, R: 1, L: 2, H: 3, X: 4, D: 5, V: 5, S: 5, N: 6, Q: 6 };

export interface SuggestedApproach { id: string; name: string; runway: string | null; reason: string; rank: number; charts: ChartRef[] }

/**
 * Suggest approaches: runways facing the wind first (headwind > 0 when the
 * wind is at least LIGHT_WIND_KT), ILS before RNAV before LOC…; circling
 * approaches last. With calm/variable/unknown wind every runway is eligible
 * and the reason says so. Never claims to know the assignment.
 */
export function suggestApproaches(ap: AirportData, wind: Wind | null, summaries: ProcSummary[], max = 6): SuggestedApproach[] {
  const rw = new Map(runwayWinds(ap, wind).map((r) => [r.runway, r]));
  const windUsable = !!wind && wind.dirDeg != null && wind.speedKt >= LIGHT_WIND_KT;
  const out: SuggestedApproach[] = [];
  const iaps = summaries.filter((x) => x.kind === "IAP");
  const ilsRunways = new Set(iaps.filter((x) => parseApproachIdent(x.id).typeCode === "I").map((x) => x.runway));
  for (const s of iaps) {
    // the LOC-only procedure is the ILS's own fallback: listed, not re-suggested
    if (parseApproachIdent(s.id).typeCode === "L" && ilsRunways.has(s.runway)) continue;
    const w = s.runway ? rw.get(s.runway.replace(/B$/, "")) : null;
    let reason: string;
    let rank = TYPE_RANK[parseApproachIdent(s.id).typeCode] ?? 7;
    if (!s.runway) { reason = "circling approach"; rank += 20; } else if (windUsable && w && w.headwindKt != null) {
      if (w.headwindKt <= 0) continue; // tailwind runway: not a suggestion
      reason = `RWY ${s.runway}: ${w.headwindKt.toFixed(0)} kt headwind, ${w.crosswindKt?.toFixed(0) ?? "?"} kt crosswind`;
      rank -= w.headwindKt / 100;
    } else {
      reason = wind ? `RWY ${s.runway}: wind light/variable — any runway may be in use` : `RWY ${s.runway}: no current wind — runway in use unknown`;
      rank += 10;
    }
    out.push({ id: s.id, name: s.name, runway: s.runway, reason, rank, charts: s.charts });
  }
  return out.sort((a, b) => a.rank - b.rank).slice(0, max);
}

/** aviationweather.gov METAR JSON (one airport) -> Wind */
export function parseMetarWind(json: unknown): Wind | null {
  const row = Array.isArray(json) ? json[0] : null;
  if (!row || typeof row !== "object") return null;
  const r = row as Record<string, unknown>;
  const spd = typeof r.wspd === "number" ? r.wspd : null;
  if (spd == null) return null;
  const dir = typeof r.wdir === "number" ? r.wdir : null; // "VRB" arrives as a string
  const obs = typeof r.obsTime === "number" ? new Date(r.obsTime * 1000).toISOString() : typeof r.reportTime === "string" ? r.reportTime : null;
  return { dirDeg: dir, speedKt: spd, gustKt: typeof r.wgst === "number" ? r.wgst : null, obsTime: obs, raw: typeof r.rawOb === "string" ? r.rawOb : null };
}

// ── context ────────────────────────────────────────────────────────────────

export interface FiledPlanLite { departure: string | null; arrival: string | null; routeText: string | null; updatedAt: number }
export interface ProceduresContext {
  cifp: CifpStore;
  dtpp: DtppStore;
  plates: PlateCache;
  /** filed SWIM plan for a callsign (null when none / SWIM off) */
  filedPlan: (callsign: string) => FiledPlanLite | null;
  metar: (icao: string) => Promise<Wind | null>;
  now: () => number;
}

export const METAR_TTL_MS = 10 * 60_000;
export function metarFetcher(fetchImpl: typeof fetch = fetch, now: () => number = () => Date.now()): (icao: string) => Promise<Wind | null> {
  const cache = new Map<string, { at: number; w: Wind | null }>();
  return async (icao) => {
    const hit = cache.get(icao);
    if (hit && now() - hit.at < METAR_TTL_MS) return hit.w;
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), 4000);
    try {
      const r = await fetchImpl(`https://aviationweather.gov/api/data/metar?ids=${encodeURIComponent(icao)}&format=json`, { signal: ac.signal });
      const w = r.ok ? parseMetarWind(await r.json()) : null;
      cache.set(icao, { at: now(), w });
      if (cache.size > 500) cache.delete(cache.keys().next().value as string);
      return w;
    } catch (e: unknown) {
      console.warn(`[procedures] METAR ${icao}:`, e instanceof Error ? e.message : e);
      return null;
    } finally {
      clearTimeout(timer);
    }
  };
}

const GEOREF_CACHE_MAX = 300;

// ── routes ─────────────────────────────────────────────────────────────────

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));
const cycleOut = (c: AiracCycle) => ({ ident: c.ident, effective: c.effective, expires: c.expires });

export function registerProcedureRoutes(app: Express, ctxIn?: Partial<ProceduresContext>): void {
  let lazy: ProceduresContext | null = null;
  const ctx = (): ProceduresContext => {
    if (lazy) return lazy;
    lazy = {
      // NASR fix/navaid gazetteer (server/navFixes.ts, #1214): fallback for
      // an unresolved CIFP reference + an independent cross-check of each fix
      cifp: ctxIn?.cifp ?? new CifpStore({ external: (id) => lookupFix(id) }),
      dtpp: ctxIn?.dtpp ?? new DtppStore(),
      plates: ctxIn?.plates ?? new PlateCache(),
      filedPlan: ctxIn?.filedPlan ?? defaultFiledPlan,
      metar: ctxIn?.metar ?? metarFetcher(),
      now: ctxIn?.now ?? (() => Date.now()),
    };
    return lazy;
  };
  const georefCache = new Map<string, GeorefResult>();

  /** CIFP index (required) + d-TPP index (optional — charts degrade to none) */
  const load = async (res: Response) => {
    let cifp: { idx: CifpIndex; cycle: AiracCycle };
    try { cifp = await ctx().cifp.get(); } catch (e: unknown) {
      res.status(503).json({ error: `FAA CIFP unavailable: ${errMsg(e)}`, notForNavigation: NOT_FOR_NAVIGATION });
      return null;
    }
    let dtpp: { idx: DtppIndex; cycle: AiracCycle } | null = null;
    let dtppError: string | null = null;
    try { dtpp = await ctx().dtpp.get(); } catch (e: unknown) { dtppError = errMsg(e); }
    return { cifp, dtpp, dtppError };
  };

  app.get("/api/data/procedures/status", (_req: Request, res: Response) => {
    const c = ctx();
    res.json({ cifp: c.cifp.status(), dtpp: c.dtpp.status(), plates: { backend: c.plates.backend, ...c.plates.counters }, georefCached: georefCache.size });
  });

  app.get("/api/data/procedures/flight/:callsign", async (req: Request, res: Response) => {
    const cs = String(req.params.callsign || "").trim().toUpperCase();
    if (!/^[A-Z0-9]{2,8}$/.test(cs)) return res.status(400).json({ error: "callsign required (2-8 letters/digits)" });
    const L = await load(res);
    if (!L) return;
    const filed = ctx().filedPlan(cs);
    const depId = normAirport(String(filed?.departure || req.query.dep || ""));
    const arrId = normAirport(String(filed?.arrival || req.query.arr || ""));
    const dep = depId ? cifpAirport(L.cifp.idx, depId) : null;
    const arr = arrId ? cifpAirport(L.cifp.idx, arrId) : null;
    const navName = (id: string) => L.cifp.idx.navaidName(id);
    const m = findFiledProcedures(filed?.routeText ?? null, dep, arr);
    const withSummary = (ap: AirportData | null, f: FiledMatch | null) => {
      if (!ap || !f) return null;
      const p = ap.procedures.find((x) => x.id === f.id);
      return p ? { ...f, ...summarize(ap, p, dtppAirport(L.dtpp?.idx ?? null, ap, ap.icao), navName), airport: ap.icao } : null;
    };
    let wind: Wind | null = null;
    let suggestions: SuggestedApproach[] = [];
    if (arr) {
      wind = await ctx().metar(arr.icao);
      const dap = dtppAirport(L.dtpp?.idx ?? null, arr, arr.icao);
      suggestions = suggestApproaches(arr, wind, arr.procedures.filter((p) => p.kind === "IAP").map((p) => summarize(arr, p, dap, navName)));
    }
    res.json({
      callsign: cs,
      planSource: filed ? "FILED_FAA" : depId || arrId ? "CLIENT_PLAN" : "NONE",
      departure: dep ? { icao: dep.icao, name: dep.name } : depId ? { icao: depId, name: null } : null,
      arrival: arr ? { icao: arr.icao, name: arr.name } : arrId ? { icao: arrId, name: null } : null,
      routeText: filed?.routeText ?? null,
      filed: {
        dp: withSummary(dep, m.dp),
        star: withSummary(arr, m.star),
        note: filed
          ? (filed.routeText ? "DP/STAR matched from the FAA-filed route text (SWIM SFDPS)." : "The filed plan carried no route text — no DP/STAR to match.")
          : "No FAA-filed plan for this callsign (SWIM) — DP/STAR cannot be known; airports come from the predicted route.",
      },
      suggestedApproaches: suggestions,
      wind,
      runwayWinds: arr ? runwayWinds(arr, wind) : [],
      cycle: cycleOut(L.cifp.cycle),
      dtppCycle: L.dtpp ? { ident: L.dtpp.cycle.ident, from: L.dtpp.idx.from, to: L.dtpp.idx.to } : null,
      dtppError: L.dtppError,
      approachHonesty: APPROACH_HONESTY,
      notForNavigation: NOT_FOR_NAVIGATION,
    });
  });

  app.get("/api/data/procedures/:airport", async (req: Request, res: Response) => {
    const id = normAirport(String(req.params.airport));
    if (!id) return res.status(400).json({ error: "airport ident required (3-4 letters/digits)" });
    const L = await load(res);
    if (!L) return;
    const ap = cifpAirport(L.cifp.idx, id);
    if (!ap) return res.status(404).json({ error: `${id} has no procedures in FAA CIFP cycle ${L.cifp.cycle.ident}`, notForNavigation: NOT_FOR_NAVIGATION });
    const dap = dtppAirport(L.dtpp?.idx ?? null, ap, id);
    const navName = (x: string) => L.cifp.idx.navaidName(x);
    const all = ap.procedures.map((p) => summarize(ap, p, dap, navName));
    const byName = (a: ProcSummary, b: ProcSummary) => a.name.localeCompare(b.name);
    const apd = dap?.charts.find((c) => c.code === "APD") ?? null;
    res.json({
      airport: { icao: ap.icao, faa: dap?.faa ?? null, name: ap.name, lat: ap.ll.lat, lon: ap.ll.lon, elevFt: ap.elevFt, magVar: ap.magVar },
      cycle: cycleOut(L.cifp.cycle),
      dtppCycle: L.dtpp ? { ident: L.dtpp.cycle.ident, from: L.dtpp.idx.from, to: L.dtpp.idx.to } : null,
      dtppError: L.dtppError,
      sids: all.filter((p) => p.kind === "SID").sort(byName),
      stars: all.filter((p) => p.kind === "STAR").sort(byName),
      approaches: all.filter((p) => p.kind === "IAP").sort((a, b) => (a.runway ?? "").localeCompare(b.runway ?? "") || byName(a, b)),
      charts: {
        airportDiagram: apd ? chartRef(ap.icao, apd) : null,
        other: (dap?.charts ?? []).filter((c) => c.code === "MIN" || c.code === "HOT" || c.code === "LAH").map((c) => chartRef(ap.icao, c)),
      },
      approachHonesty: APPROACH_HONESTY,
      notForNavigation: NOT_FOR_NAVIGATION,
    });
  });

  app.get("/api/data/procedures/:airport/:procId/path", async (req: Request, res: Response) => {
    const id = normAirport(String(req.params.airport));
    const procId = String(req.params.procId || "").trim().toUpperCase();
    if (!id || !/^[A-Z0-9-]{2,6}$/.test(procId)) return res.status(400).json({ error: "airport and procedure ident required" });
    const tr = req.query.transition ? String(req.query.transition).trim().toUpperCase() : null;
    if (tr && !/^[A-Z0-9]{1,5}$/.test(tr)) return res.status(400).json({ error: "bad transition" });
    const L = await load(res);
    if (!L) return;
    const ap = cifpAirport(L.cifp.idx, id);
    const proc = ap?.procedures.find((p) => p.id === procId) ?? null;
    if (!ap || !proc) return res.status(404).json({ error: `${procId} not found at ${id} in CIFP cycle ${L.cifp.cycle.ident}` });
    const path = procedurePath(L.cifp.idx, ap, proc, tr);
    res.setHeader("Cache-Control", "public, max-age=3600");
    res.json({ ...path, cycle: cycleOut(L.cifp.cycle), source: "FAA CIFP (ARINC 424)", notForNavigation: NOT_FOR_NAVIGATION });
  });

  /** the plate must be a chart the current d-TPP lists for this airport */
  const plateFor = async (req: Request, res: Response) => {
    const id = normAirport(String(req.params.airport));
    const pdf = String(req.params.pdf || "").toUpperCase();
    if (!id || !/^[A-Z0-9_]{2,40}\.PDF$/.test(pdf)) { res.status(400).json({ error: "airport and plate PDF name required" }); return null; }
    let d: { idx: DtppIndex; cycle: AiracCycle };
    try { d = await ctx().dtpp.get(); } catch (e: unknown) { res.status(503).json({ error: `FAA d-TPP unavailable: ${errMsg(e)}` }); return null; }
    const dap = d.idx.airports.get(id) ?? d.idx.airports.get(id.replace(/^K/, ""));
    const chart = dap?.charts.find((c) => c.pdf.toUpperCase() === pdf);
    if (!chart) { res.status(404).json({ error: `${pdf} is not a ${id} chart in d-TPP cycle ${d.cycle.ident}` }); return null; }
    return { id, pdf, chart, cycle: d.cycle };
  };

  app.get("/api/data/plates/:airport/:pdf", async (req: Request, res: Response) => {
    const p = await plateFor(req, res);
    if (!p) return;
    try {
      const got = await ctx().plates.get(p.cycle.ident, p.pdf);
      res.setHeader("Content-Type", "application/pdf");
      res.setHeader("Cache-Control", "public, max-age=86400");
      res.setHeader("X-Plate-Cycle", p.cycle.ident);
      res.setHeader("X-Plate-Source", got.source);
      res.setHeader("Content-Disposition", `inline; filename="${p.pdf}"`);
      res.end(got.body);
    } catch (e: unknown) {
      res.status(502).json({ error: `plate fetch failed: ${errMsg(e)}` });
    }
  });

  app.get("/api/data/plates/:airport/:pdf/georef", async (req: Request, res: Response) => {
    const p = await plateFor(req, res);
    if (!p) return;
    const procId = String(req.query.proc || "").trim().toUpperCase();
    if (!/^[A-Z0-9-]{2,6}$/.test(procId)) return res.status(400).json({ error: "proc (CIFP procedure ident) required" });
    const key = `${p.cycle.ident}|${p.pdf}|${procId}`;
    let g = georefCache.get(key);
    if (!g) {
      const L = await load(res);
      if (!L) return;
      const ap = cifpAirport(L.cifp.idx, p.id);
      const proc = ap?.procedures.find((x) => x.id === procId) ?? null;
      if (!ap || !proc) return res.status(404).json({ error: `${procId} not found at ${p.id}` });
      try {
        const got = await ctx().plates.get(p.cycle.ident, p.pdf);
        g = georeferencePlate(got.body, procedureFixes(L.cifp.idx, ap, proc), ap.ll as LL);
      } catch (e: unknown) {
        return res.status(502).json({ error: `plate fetch failed: ${errMsg(e)}` });
      }
      georefCache.set(key, g);
      while (georefCache.size > GEOREF_CACHE_MAX) georefCache.delete(georefCache.keys().next().value as string);
    }
    res.setHeader("Cache-Control", "public, max-age=3600");
    res.json({ ...g, cycle: p.cycle.ident, pdf: p.pdf, chart: p.chart.name, proc: procId, pdfUrl: `/api/data/plates/${p.id}/${p.pdf}`, notForNavigation: NOT_FOR_NAVIGATION });
  });
}

/** SWIM SFDPS store lookup — the same singleton store flightPlans.ts fills
 *  (getFlightPlanContext never opens a connection; registerFlightPlanRoutes does). */
function defaultFiledPlan(callsign: string): FiledPlanLite | null {
  const p = getFlightPlanContext().swim.lookup(callsign, Date.now());
  return p ? { departure: p.departure, arrival: p.arrival, routeText: p.routeText, updatedAt: p.updatedAt } : null;
}
