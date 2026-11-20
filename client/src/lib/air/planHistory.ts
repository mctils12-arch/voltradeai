// PLAN CHANGES — the web flight card's read of GET
// /api/data/aircraft/plan-history/:hex (server/flightPlanHistory.ts; the
// mobile app's contract, docs/aircraft_app_data_contract.md section 9). The
// card showed the CURRENT plan and one word ("re-planned"); this module turns
// the route's versions, amendments, restatements and off-route stretches into
// the few lines the card can carry ([PRODUCT] 2026-11-20, UI coverage batch B).
//
// Pure: no DOM, no fetch of its own beyond fetchPlanHistory. Pinned by
// planHistory.test.ts. Read tolerantly: a body of another shape is null, and
// the card then shows nothing rather than a guess. Nothing is derived here
// that the server did not state: the words quote the server's change kinds,
// fix names, altitudes and its own reasons. Distances stay in the server's
// nautical miles until the component formats them through lib/units.

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);
const obj = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);
const names = (v: unknown): string[] => (Array.isArray(v) ? v.map(str).filter((x): x is string => !!x) : []);

export const PLAN_HISTORY_REFRESH_MS = 120_000;
export const HEX6_RE = /^[0-9a-f]{6}$/;

export type PlanChangeKind = "REROUTE" | "FIXES_ADDED" | "FIXES_REMOVED" | "SHORTCUT" | "DIRECT_TO" | "DESTINATION_CHANGE" | "ALTITUDE_CHANGE" | "ROUTE_TEXT_ONLY";
const KINDS: readonly PlanChangeKind[] = ["REROUTE", "FIXES_ADDED", "FIXES_REMOVED", "SHORTCUT", "DIRECT_TO", "DESTINATION_CHANGE", "ALTITUDE_CHANGE", "ROUTE_TEXT_ONLY"];

export interface PlanChangeView {
  key: string; version: number; fromVersion: number;
  /** when the change took effect (epoch ms) and which clock that is */
  atMs: number; atBasis: "message" | "received";
  kinds: PlanChangeKind[];
  /** one plain line: what changed */
  words: string;
  /** the largest sideways move of the route, in nautical miles; null when not measured or unchanged */
  lateralNm: number | null; lateralEstimated: boolean;
  /** where the aircraft was, when a real or interpolated fix exists for that moment */
  position: { method: string; lat: number; lon: number } | null;
  /** the server's reason when no position is given */
  positionReason: string | null;
  /** versions between the two were never held: the change may be several changes */
  spansMissing: boolean;
}
export interface OffRouteView { key: string; fromMs: number | null; toMs: number | null; maxCrossTrackNm: number | null; classification: string | null }
export interface PlanHistoryView {
  /** changes: more than one version or something to list · single: one version, nothing changed ·
   *  none: no FAA-filed plan (no versions exist) · withheld: the aircraft is on the FAA's LADD list */
  state: "changes" | "single" | "none" | "withheld";
  callsign: string | null; label: string | null; honesty: string | null;
  departure: string | null; arrival: string | null;
  versions: number; versionsMissing: number;
  /** true = version 1 came from the filing itself; false = it is the first version we saw; null = not stated */
  asFiled: boolean | null;
  /** newest first */
  changes: PlanChangeView[];
  restatements: number;
  offRoute: { assessed: boolean; reason: string | null; fixesJudged: number | null; episodes: OffRouteView[] } | null;
  generatedAt: number | null;
}

const fl = (ft: number): string => (ft >= 18000 ? `FL${Math.round(ft / 100)}` : `${Math.round(ft).toLocaleString("en-US")} ft`);
const list = (xs: string[], max = 4): string => (xs.length <= max ? xs.join(", ") : `${xs.slice(0, max).join(", ")} +${xs.length - max} more`);

/** One amendment's `change` block in words, in the order a reader asks: where to, how high, which way. */
export function changeWords(kinds: readonly PlanChangeKind[], change: Record<string, unknown> | null, detail: string | null): string {
  const parts: string[] = [];
  const dest = obj(change?.destination), alt = obj(change?.cruiseAltFt), direct = obj(change?.directTo);
  const added = names(change?.fixesAdded), removed = names(change?.fixesRemoved);
  if (kinds.includes("DESTINATION_CHANGE")) parts.push(dest && str(dest.from) && str(dest.to) ? `destination ${str(dest.from)} → ${str(dest.to)}` : "destination changed");
  if (kinds.includes("ALTITUDE_CHANGE")) parts.push(alt && fin(alt.from) && fin(alt.to) ? `cruise ${fl(alt.from)} → ${fl(alt.to)}` : "cruise altitude changed");
  if (kinds.includes("DIRECT_TO")) parts.push(direct && str(direct.fix) ? `direct to ${str(direct.fix)}` : "direct from its position");
  else if (kinds.includes("SHORTCUT")) parts.push("shortcut");
  else if (kinds.includes("REROUTE")) parts.push("rerouted");
  if (added.length) parts.push(`added ${list(added)}`);
  if (removed.length) parts.push(`dropped ${list(removed)}`);
  if (!parts.length && kinds.includes("ROUTE_TEXT_ONLY")) parts.push("route text reworded, same path");
  if (!parts.length) parts.push(detail ?? "changed");
  return parts.join(" · ");
}

/** GET /api/data/aircraft/plan-history/:hex → the card's view; null when the body is not that route's shape. */
export function parsePlanHistory(raw: unknown): PlanHistoryView | null {
  const o = obj(raw);
  const source = o ? str(o.source) : null;
  if (!o || (source !== "FILED_FAA" && source !== "NONE")) return null;
  const plan = obj(o.plan), current = obj(o.current), original = obj(o.original), off = obj(o.offRoute);
  const versions = Array.isArray(o.versions) ? o.versions.length : 0;
  const changes: PlanChangeView[] = [];
  for (const a of Array.isArray(o.amendments) ? o.amendments : []) {
    const x = obj(a);
    if (!x || !fin(x.version) || !fin(x.fromVersion) || !fin(x.at)) continue;
    const change = obj(x.change), geo = obj(change?.geometry), pos = obj(x.position);
    const kinds = (Array.isArray(x.kinds) ? x.kinds : []).filter((k): k is PlanChangeKind => KINDS.includes(k as PlanChangeKind));
    const lateral = geo && geo.changed === true && fin(geo.maxLateralNm) && geo.maxLateralNm > 0 ? geo.maxLateralNm : null;
    changes.push({
      key: `v${x.version}`, version: x.version, fromVersion: x.fromVersion, atMs: x.at, atBasis: x.atBasis === "SWIM_MESSAGE_TIME" ? "message" : "received",
      kinds, words: changeWords(kinds, change, str(x.detail)), lateralNm: lateral, lateralEstimated: geo?.lateralEstimated === true,
      position: pos && fin(pos.lat) && fin(pos.lon) ? { method: str(pos.method) ?? "UNKNOWN", lat: pos.lat, lon: pos.lon } : null,
      positionReason: str(x.positionReason), spansMissing: x.spansMissingVersions === true,
    });
  }
  changes.sort((a, b) => b.atMs - a.atMs);
  const episodes: OffRouteView[] = [];
  (Array.isArray(off?.episodes) ? off.episodes : []).forEach((e, i) => {
    const x = obj(e);
    if (!x) return;
    episodes.push({
      key: `e${i}`, fromMs: fin(x.startMs) ? x.startMs : null, toMs: fin(x.endMs) ? x.endMs : null,
      maxCrossTrackNm: fin(x.maxCrossTrackNm) ? x.maxCrossTrackNm : null, classification: str(x.classification),
    });
  });
  const restatements = Array.isArray(o.restatements) ? o.restatements.length : 0;
  const state: PlanHistoryView["state"] = o.privacy === "LADD" ? "withheld" : source === "NONE" || versions === 0 ? "none" : versions > 1 || changes.length || restatements || episodes.length ? "changes" : "single";
  return {
    state, callsign: str(o.callsign), label: str(o.label), honesty: str(o.honesty),
    departure: str(current?.departure) ?? str(original?.departure), arrival: str(current?.arrival) ?? str(original?.arrival),
    versions, versionsMissing: fin(plan?.versionsMissing) ? plan.versionsMissing : 0,
    asFiled: typeof plan?.originalAsFiled === "boolean" ? plan.originalAsFiled : null,
    changes, restatements,
    offRoute: off ? { assessed: off.assessed === true, reason: str(off.reason), fixesJudged: fin(off.fixesJudged) ? off.fixesJudged : null, episodes } : null,
    generatedAt: fin(o.generatedAt) ? o.generatedAt : null,
  };
}

/** "21:07Z" — aviation time for a change; the title carries the basis. */
export function zulu(ms: number): string {
  const d = new Date(ms);
  return Number.isNaN(d.getTime()) ? "—" : `${d.toISOString().slice(11, 16)}Z`;
}
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** The row's headline: how many versions, how many real changes, and the newest one. */
export function planChangesHeadline(v: PlanHistoryView): string {
  const head = `${plural(v.versions, "plan version")}${v.versionsMissing ? ` (+${v.versionsMissing} not held)` : ""}`;
  if (!v.changes.length) return `${head} · no change to the route ahead${v.restatements ? ` · ${plural(v.restatements, "restatement")}` : ""}`;
  const newest = v.changes[0];
  return `${head} · ${plural(v.changes.length, "change")} · latest ${zulu(newest.atMs)}: ${newest.words}`;
}
/** What "version 1" is, said once under the list. */
export function originNote(v: PlanHistoryView): string {
  if (v.asFiled === true) return "Version 1 is the plan as filed.";
  if (v.asFiled === false) return "Version 1 is the plan as we first saw it, not necessarily as filed.";
  return "Whether version 1 is the plan as filed is not stated.";
}
/** Off-route stretches in one line; the server's reason when the route could not be judged. */
export function offRouteLine(v: PlanHistoryView, fmtNm: (nm: number) => string): string | null {
  const o = v.offRoute;
  if (!o) return null;
  if (!o.assessed) return `Off-route stretches not judged${o.reason ? `: ${o.reason}` : ""}.`;
  if (!o.episodes.length) return o.fixesJudged !== null && o.fixesJudged > 0 ? `No off-route stretch in the ${plural(o.fixesJudged, "position")} we hold for this flight.` : "No position of this flight could be judged against its route.";
  const worst = o.episodes.reduce<number | null>((m, e) => (e.maxCrossTrackNm !== null && (m === null || e.maxCrossTrackNm > m) ? e.maxCrossTrackNm : m), null);
  return `${plural(o.episodes.length, "off-route stretch", "off-route stretches")}${worst !== null ? `, up to ${fmtNm(worst)} from the route` : ""}. No cause is given.`;
}
const METHOD_WORDS: Readonly<Record<string, string>> = { EXACT: "at a recorded position", NEAREST: "nearest recorded position", INTERPOLATED: "between two recorded positions (interpolated)" };
/** Where the aircraft was at a change, or why that is not known. */
export function positionWords(c: PlanChangeView): string {
  if (c.position) return `aircraft position: ${METHOD_WORDS[c.position.method] ?? c.position.method.toLowerCase()}`;
  return "aircraft position at that moment not known";
}

export type FetchLike = (input: string, init?: { signal?: AbortSignal }) => Promise<{ ok: boolean; status: number; json: () => Promise<unknown> }>;
/** One abortable read. A non-OK answer or another shape is null: the card then shows nothing. */
export async function fetchPlanHistory(hex: string, callsign: string | null, fetchImpl: FetchLike, signal?: AbortSignal): Promise<PlanHistoryView | null> {
  const h = hex.toLowerCase();
  if (!HEX6_RE.test(h)) return null;
  const cs = callsign && /^[A-Za-z0-9]{2,8}$/.test(callsign.trim()) ? `?callsign=${encodeURIComponent(callsign.trim().toUpperCase())}` : "";
  const r = await fetchImpl(`/api/data/aircraft/plan-history/${h}${cs}`, { signal });
  if (!r.ok) return null;
  return parsePlanHistory(await r.json());
}
