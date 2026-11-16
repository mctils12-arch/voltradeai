// aircraftEstimate.ts — GET /api/data/aircraft/estimate/:hex
// (owner 2026-11-16: "keep the model moving by dead reckoning between updates
// ... where there is no data, show an estimated path along the filed route,
// drawn dashed so it's clearly estimated").
//
// Returns the aircraft's last REAL fix and, beside it, an ESTIMATE of where
// it is now and the path it is estimated to be on (shared/aircraftEstimate.ts:
// straight ahead for a short while, then along the route it was on). The two
// are never mixed: `fix` is observed, `estimate` is not and says so in every
// field that matters (method, the fix it started from, its age, assumptions).
//
// The basis fix:
//   1. the worldwide snapshot's row for the hex (held up to 10 minutes after
//      its fix), with its broadcast ground speed and track;
//   2. else the newest position in our own archive for the hex, with ground
//      speed and track worked out from its last two positions (flagged
//      `motionDerived`). This is the case the owner asked for: no data now,
//      e.g. out of receiver range over water.
// The route: whatever GET /api/data/aircraft/plan/:hex would answer for that
// fix (resolveFlightPlan: a filed FAA plan, or a predicted one, each labelled;
// FAA-sourced plan data is already withheld there for LADD aircraft, so this
// route adds no FAA/SWIM data of its own). The plan is looked up only once a
// fix is older than the short dead-reckoning window, and cached per hex.
//
// Cost: no upstream request of its own. One archive tail read (cached 30 s by
// recentTrackCached) and one plan resolution (cached here ESTIMATE_PLAN_TTL_MS)
// per hex at most.

import type { Express } from "express";
import {
  estimateAircraft, motionBetween, DEAD_RECKONING_SHORT_SEC, ESTIMATE_MAX_AGE_SEC,
  type AircraftEstimate, type EstimateFix, type EstimateRoute,
} from "../shared/aircraftEstimate";
import type { LatLon } from "../shared/flightPlanGeometry";
import { withinBudget } from "./aircraftFind";

export const ESTIMATE_PLAN_TTL_MS = 60_000;
export const ESTIMATE_PLAN_CACHE_MAX = 500;
/** The archive's recent-track read (two days of hour files) may take this
 *  long inside one answer (aircraftFind.ts FIND_ARCHIVE_BUDGET_MS has the
 *  measurement: about 45 s cold). Past it the answer says no fix was read in
 *  time; the read keeps running and fills its cache for the next request. */
export const ESTIMATE_ARCHIVE_BUDGET_MS = 2_500;
const HEX_RE = /^[0-9a-f]{6}$/;
const FT_PER_M = 3.28084;

/** the snapshot row fields this route reads (server/globalSnapshot.ts SnapRow) */
export interface EstimateLiveRow {
  lat: number; lon: number; altFt: number | null; gsKt: number | null; trk: number | null;
  gnd: boolean; callsign: string | null; seenAt: number; src?: string | null;
}
/** the plan fields this route reads (server/flightPlans.ts FlightPlanResponse) */
export interface EstimatePlan {
  source: string; label?: string | null; privacy?: string | null;
  points: LatLon[];
}
export interface EstimatePlanRequest {
  hex: string; callsign: string | null; lat: number | null; lon: number | null;
  altFt: number | null; trkDeg: number | null; fixT: number | null;
}
export interface EstimateDeps {
  liveRow: (hex: string) => EstimateLiveRow | null | undefined;
  /** the archive's recent positions for the hex, oldest first */
  archiveTail: (hex: string) => Promise<Array<{ t: number; la: number; lo: number; al?: number | null }>>;
  plan: (q: EstimatePlanRequest) => Promise<EstimatePlan>;
  now?: () => number;
  /** time allowed for archiveTail inside one answer (default ESTIMATE_ARCHIVE_BUDGET_MS) */
  archiveBudgetMs?: number;
}

export const ESTIMATE_HONESTY =
  "`fix` is the last position actually received for this aircraft. `estimate` is NOT an observed position: it is "
  + "worked out from that fix (straight ahead at its last ground speed and track for a short while, then along the "
  + "route it was on), assumes no turn off the route and no speed change, and holds the last reported altitude. "
  + "Draw it as estimated. `withinNm` is a circle around the last fix the aircraft is inside unless it sped up by "
  + "more than 15%; it is not an error bar around the estimate.";

export interface EstimateBody {
  at: number; hex: string; kind: "estimate";
  fix: null | (EstimateFix & { ageSec: number; source: "snapshot" | "archive"; provider: string | null; callsign: string | null });
  estimate: AircraftEstimate | null;
  /** why there is no fix to estimate from (fix null) */
  reason: string | null;
  plan: null | { source: string; label: string | null; privacy: string | null; used: boolean; note: string | null };
  honesty: string;
}

export function createEstimator(deps: EstimateDeps) {
  const now = deps.now ?? Date.now;
  const plans = new Map<string, { at: number; fixMs: number; plan: EstimatePlan | null; error: string | null }>();

  /** set by basisFix when the archive read did not finish inside its budget */
  let archiveTimedOut = false;
  async function basisFix(hex: string, t: number): Promise<EstimateBody["fix"]> {
    archiveTimedOut = false;
    const row = deps.liveRow(hex);
    if (row && Number.isFinite(row.lat) && Number.isFinite(row.lon) && Number.isFinite(row.seenAt)) {
      return {
        lat: row.lat, lon: row.lon, altFt: row.altFt, gsKt: row.gsKt, trkDeg: row.trk, onGround: !!row.gnd,
        fixMs: row.seenAt, ageSec: Math.round(Math.max(0, t - row.seenAt) / 100) / 10,
        source: "snapshot", provider: row.src ?? null, callsign: row.callsign ?? null,
      };
    }
    const got = await withinBudget(deps.archiveTail(hex), deps.archiveBudgetMs ?? ESTIMATE_ARCHIVE_BUDGET_MS, null);
    if (got.timedOut || !got.value) { archiveTimedOut = got.timedOut; return null; }
    const tail = got.value;
    const last = tail[tail.length - 1];
    if (!last || !Number.isFinite(last.t) || !Number.isFinite(last.la) || !Number.isFinite(last.lo)) return null;
    const fixMs = last.t * 1000;
    const prev = tail.length > 1 ? tail[tail.length - 2] : null;
    const m = prev ? motionBetween({ lat: prev.la, lon: prev.lo, tMs: prev.t * 1000 }, { lat: last.la, lon: last.lo, tMs: fixMs }) : null;
    return {
      lat: last.la, lon: last.lo, altFt: last.al == null ? null : Math.round(last.al * FT_PER_M),
      gsKt: m ? m.gsKt : null, trkDeg: m ? m.trkDeg : null,
      // the archive's recent-track read does not carry the on-ground flag; an aircraft with no
      // altitude and no usable motion yields no estimate anyway
      onGround: false, motionDerived: true,
      fixMs, ageSec: Math.round(Math.max(0, t - fixMs) / 100) / 10, source: "archive", provider: null, callsign: null,
    };
  }

  async function planFor(hex: string, fix: NonNullable<EstimateBody["fix"]>, t: number) {
    const hit = plans.get(hex);
    if (hit && t - hit.at < ESTIMATE_PLAN_TTL_MS && hit.fixMs === fix.fixMs) return hit;
    let plan: EstimatePlan | null = null;
    let error: string | null = null;
    try {
      plan = await deps.plan({ hex, callsign: fix.callsign, lat: fix.lat, lon: fix.lon, altFt: fix.altFt, trkDeg: fix.trkDeg, fixT: fix.fixMs });
    } catch (e: unknown) {
      error = String(e instanceof Error ? e.message : e).slice(0, 160);
    }
    const rec = { at: t, fixMs: fix.fixMs, plan, error };
    plans.delete(hex);
    plans.set(hex, rec);
    if (plans.size > ESTIMATE_PLAN_CACHE_MAX) {
      const oldest = plans.keys().next().value;
      if (oldest !== undefined) plans.delete(oldest);
    }
    return rec;
  }

  return {
    async estimate(hexIn: string): Promise<EstimateBody> {
      const hex = String(hexIn || "").toLowerCase();
      if (!HEX_RE.test(hex)) throw new Error("icao24 hex required");
      const t = now();
      const fix = await basisFix(hex, t);
      const body: EstimateBody = { at: t, hex, kind: "estimate", fix, estimate: null, reason: null, plan: null, honesty: ESTIMATE_HONESTY };
      if (!fix) {
        body.reason = archiveTimedOut
          ? "not in the worldwide snapshot (last 10 minutes), and the archive's recent track was not read in time: ask again in a few seconds"
          : "no position held for this aircraft: not in the worldwide snapshot (last 10 minutes) and not in the archive's recent track";
        return body;
      }
      const cached = plans.get(hex);
      const cachedUsable = cached && t - cached.at < ESTIMATE_PLAN_TTL_MS && cached.fixMs === fix.fixMs ? cached : null;
      if (fix.ageSec <= DEAD_RECKONING_SHORT_SEC && !cachedUsable) {
        // young fix: straight ahead needs no plan, so none is looked up (this route is polled)
        const est = estimateAircraft(fix, t, null);
        if (est.method === "dead_reckoning") est.reason = `fix ${Math.round(fix.ageSec)} s old: straight ahead (the route is looked up once the fix is older than ${DEAD_RECKONING_SHORT_SEC} s)`;
        body.estimate = est;
        return body;
      }
      if (fix.ageSec > ESTIMATE_MAX_AGE_SEC) {
        body.estimate = estimateAircraft(fix, t, null);
        return body;
      }
      const rec = cachedUsable ?? await planFor(hex, fix, t);
      const p = rec.plan;
      const usable = !!p && p.source !== "NONE" && Array.isArray(p.points) && p.points.length >= 2;
      const route: EstimateRoute | null = usable && p ? { points: p.points.map((x) => ({ lat: x.lat, lon: x.lon })), source: p.source } : null;
      const est = estimateAircraft(fix, t, route);
      body.estimate = est;
      body.plan = p
        ? { source: p.source, label: p.label ?? null, privacy: p.privacy ?? null, used: est.method === "along_route", note: null }
        : { source: "NONE", label: null, privacy: null, used: false, note: rec.error ? `the route lookup failed: ${rec.error}` : "no route" };
      return body;
    },
    cacheSize: () => plans.size,
  };
}

export function registerAircraftEstimateRoute(app: Express, deps: EstimateDeps): ReturnType<typeof createEstimator> {
  const est = createEstimator(deps);
  app.get("/api/data/aircraft/estimate/:hex", async (req, res) => {
    const hex = String(req.params.hex || "").toLowerCase();
    if (!HEX_RE.test(hex)) return res.status(400).json({ error: "icao24 hex required" });
    try {
      const body = await est.estimate(hex);
      res.set("Cache-Control", "no-store");
      res.json(body);
    } catch (e: unknown) {
      res.status(500).json({ error: "estimate failed", detail: String(e instanceof Error ? e.message : e).slice(0, 200) });
    }
  });
  return est;
}
