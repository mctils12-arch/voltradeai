// AIRPORT TRAFFIC MANAGEMENT ([PRODUCT] 2026-11-20, UI coverage batch B, part
// 3): what the FAA's traffic-management feed says about ONE airport, for the
// FAA airport card. Data: GET /api/data/swim/tfms/active (server/swimTfms.ts):
// the programs and restrictions the feed lists inside their published times,
// and the runway configuration and hourly rates each tower last reported.
//
// RAW, as received: nothing here is inferred. A program's kind and a
// restriction's category are the feed's own fields, worded from the FAA's
// standard abbreviations (MIT, MINIT, APREQ, TBM, STOP; GS, GDP, AFP, CTOP);
// an abbreviation this table does not hold is shown as the feed sent it. The
// feed's per-restriction action code is NOT decoded by the server, so a
// restriction cancelled early can still be listed until its published end:
// the card says so. No flight-level data is in this answer.
// Pure; pinned by airportTraffic.test.ts.

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);
const obj = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);

export const AIRPORT_TRAFFIC_REFRESH_MS = 60_000;
export const AIRPORT_CODE_RE = /^[A-Z0-9]{3,4}$/;

export type ProgramKind = "GS" | "GDP" | "AFP" | "CTOP";
export const PROGRAM_WORDS: Readonly<Record<ProgramKind, string>> = {
  GS: "Ground stop", GDP: "Ground delay program", AFP: "Airspace flow program", CTOP: "Trajectory options program",
};
/** the FAA's restriction categories (order 7210.3, traffic management initiatives) */
export const RESTRICTION_WORDS: Readonly<Record<string, string>> = {
  MIT: "Miles-in-trail", MINIT: "Minutes-in-trail", APREQ: "Call for release", TBM: "Time-based metering", STOP: "Stop",
};
export const FLOW_WORDS: Readonly<Record<string, string>> = { ARRIVAL: "arrivals", DEPARTURE: "departures", ENROUTE: "en route" };
const REASON_PREFIX: Readonly<Record<string, string>> = { VOL: "volume", WX: "weather", RWY: "runway", EQ: "equipment", OTHER: "other" };

export interface TrafficProgram {
  key: string; kind: ProgramKind;
  /** true = the feed's program message; false = known only from an advisory's title */
  fromProgram: boolean;
  startMs: number | null; endMs: number | null;
  reason: string | null;
  /** the advisory's own title, when one names this program */
  advisory: string | null;
}
export interface TrafficRestriction {
  key: string;
  /** "Miles-in-trail, arrivals", or the feed's text when the category is not in the table */
  words: string;
  elements: string[]; facility: string | null; reason: string | null;
  startMs: number | null; endMs: number | null;
}
export interface RunwayConfig {
  arriving: string | null; departing: string | null;
  /** aircraft an hour; null when the tower sent none (a 0 is "not given") */
  arrivalRate: number | null; departureRate: number | null;
  weather: string | null; atMs: number | null;
}
export interface AirportTrafficView {
  code: string;
  programs: TrafficProgram[]; restrictions: TrafficRestriction[]; config: RunwayConfig | null;
  /** the server's clock on the answer */
  atMs: number | null;
  /** how many programs and restrictions the whole answer held (what was searched) */
  searched: number;
  source: string | null;
  /** the server keeps its list across a restart (route field `kept`, 2026-11-21); null = it does not, or does not say */
  kept: TrafficKept | null;
}
/** What the server put back at its last start. Counts and times as the route gives them. */
export interface TrafficKept {
  /** programs and restrictions read back from the file at the last start; null = nothing was read back */
  carried: number | null;
  /** when the previous process wrote the file that was read back */
  savedAtMs: number | null;
  /** when this process started and read it */
  restoredAtMs: number | null;
}

/** a feed reason in plain lower case: "VOL:COMPACTED DEMAND" -> "volume: compacted demand"; NONE / empty -> null */
export function reasonWords(raw: unknown): string | null {
  const s = str(raw);
  if (!s || /^none$/i.test(s)) return null;
  const parts = s.split(":").map((p) => p.trim()).filter(Boolean);
  const head = REASON_PREFIX[parts[0].toUpperCase()] ?? parts[0].toLowerCase();
  const rest = parts.slice(1).join(": ").toLowerCase();
  return rest && rest !== head ? `${head}: ${rest}` : head;
}
/** the airport codes a program names ("MIA/MCO/FLL", "BUR/VNY", "ORD") */
const namesAirport = (field: unknown, code: string): boolean => (str(field) ?? "").toUpperCase().split(/[\s,/]+/).includes(code);

function restrictionWords(title: string | null): string {
  const t = (title ?? "").trim().toUpperCase().split(/\s+/).filter(Boolean);
  if (!t.length) return "Restriction";
  const cat = RESTRICTION_WORDS[t[0]], flow = t[1] ? FLOW_WORDS[t[1]] : null;
  if (!cat) return t.join(" ");
  return t.length === 1 ? cat : flow && t.length === 2 ? `${cat}, ${flow}` : `${cat}, ${t.slice(1).join(" ").toLowerCase()}`;
}

/** GET /api/data/swim/tfms/active → what it holds for `code`; null when the body is not the route's shape. */
export function parseAirportTraffic(raw: unknown, codeIn: string): AirportTrafficView | null {
  const o = obj(raw);
  const code = String(codeIn || "").trim().toUpperCase().replace(/^K(?=[A-Z0-9]{3}$)/, "");
  if (!o || !Array.isArray(o.programs) || !AIRPORT_CODE_RE.test(code)) return null;
  // per kind: the feed's own program message (times, reason) and the newest advisory that names the program
  type Seen = { startMs: number | null; endMs: number | null; reason: string | null; title: string | null; updatedAt: number };
  const messages = new Map<ProgramKind, Seen>(), advisories = new Map<ProgramKind, Seen>();
  const restrictions = new Map<string, TrafficRestriction & { updatedAt: number }>();
  for (const x of o.programs) {
    const p = obj(x);
    if (!p || !namesAirport(p.airport, code)) continue;
    const kind = str(p.kind), status = str(p.status);
    const startMs = fin(p.start) ? p.start : null, endMs = fin(p.end) ? p.end : null;
    const updatedAt = fin(p.updatedAt) ? p.updatedAt : 0;
    if (kind === "RESTRICTION") {
      const elements = Array.isArray(p.elements) ? p.elements.map(str).filter((e): e is string => !!e) : [];
      const facility = str(p.facility);
      // the feed re-sends a restriction under a new id when it is amended: one line per distinct restriction
      const key = [str(p.title) ?? "", facility ?? "", elements.join(" "), startMs ?? "", endMs ?? ""].join("|");
      const had = restrictions.get(key);
      if (!had || updatedAt > had.updatedAt) restrictions.set(key, { key, words: restrictionWords(str(p.title)), elements, facility, reason: reasonWords(p.reason), startMs, endMs, updatedAt });
      continue;
    }
    const pk = (kind === "ADVISORY" ? status : kind) as ProgramKind | null;
    if (!pk || !(pk in PROGRAM_WORDS)) continue;
    const into = kind === "ADVISORY" ? advisories : messages;
    const had = into.get(pk);
    if (!had || updatedAt >= had.updatedAt) into.set(pk, { startMs, endMs, reason: kind === "ADVISORY" ? null : reasonWords(p.reason), title: kind === "ADVISORY" ? str(p.title) : null, updatedAt });
  }
  const programOf = (k: ProgramKind): TrafficProgram | null => {
    const m = messages.get(k), a = advisories.get(k);
    if (!m && !a) return null;
    const t = m ?? a!;
    return { key: k, kind: k, fromProgram: !!m, startMs: t.startMs, endMs: t.endMs, reason: m?.reason ?? null, advisory: a?.title ?? null };
  };
  let config: RunwayConfig | null = null;
  for (const x of Array.isArray(o.airportConfigs) ? o.airportConfigs : []) {
    const c = obj(x);
    if (!c || (str(c.airport) ?? "").toUpperCase() !== code) continue;
    const rate = (v: unknown) => (fin(v) && v > 0 ? v : null);
    config = { arriving: str(c.arrRunways), departing: str(c.depRunways), arrivalRate: rate(c.aar), departureRate: rate(c.adr), weather: str(c.weather), atMs: fin(c.at) ? c.at : null };
  }
  const ORDER: readonly ProgramKind[] = ["GS", "GDP", "AFP", "CTOP"];
  // `kept` is an object only on a server that saves the list; a restore that read nothing (first boot, an old or
  // unreadable file) carries no count
  const k = obj(o.kept), kr = obj(k?.restored);
  const kept: TrafficKept | null = k && fin(k.saved_every_s)
    ? { carried: kr && kr.cause === "ok" && fin(kr.programs) && fin(kr.saved_at) ? kr.programs : null, savedAtMs: kr && fin(kr.saved_at) ? kr.saved_at : null, restoredAtMs: kr && fin(kr.at) ? kr.at : null }
    : null;
  return {
    code, config, atMs: fin(o.at) ? o.at : null, searched: o.programs.length, source: str(o.source), kept,
    programs: ORDER.map(programOf).filter((p): p is TrafficProgram => !!p),
    restrictions: Array.from(restrictions.values()).map(({ updatedAt: _u, ...r }) => r)
      .sort((a, b) => (a.startMs ?? 0) - (b.startMs ?? 0) || a.words.localeCompare(b.words) || a.key.localeCompare(b.key)),
  };
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** "22:45Z", or "Nov 21 04:59Z" when it is not the UTC day of `nowMs` */
export function zuluAt(ms: number, nowMs: number): string {
  const d = new Date(ms), n = new Date(nowMs);
  const hm = `${String(d.getUTCHours()).padStart(2, "0")}:${String(d.getUTCMinutes()).padStart(2, "0")}Z`;
  return d.getUTCFullYear() === n.getUTCFullYear() && d.getUTCMonth() === n.getUTCMonth() && d.getUTCDate() === n.getUTCDate() ? hm : `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()} ${hm}`;
}
/** "until 22:45Z" / "from 23:10Z to Nov 21 04:59Z" / "" when the feed gave no times */
export function periodWords(startMs: number | null, endMs: number | null, nowMs: number): string {
  if (startMs !== null && startMs > nowMs) return `from ${zuluAt(startMs, nowMs)}${endMs !== null ? ` to ${zuluAt(endMs, nowMs)}` : ""}`;
  return endMs !== null ? `until ${zuluAt(endMs, nowMs)}` : "";
}
/** "Ground delay program · volume · until Nov 21 04:59Z" */
export function programWords(p: TrafficProgram, nowMs: number): string {
  return [PROGRAM_WORDS[p.kind], p.reason, periodWords(p.startMs, p.endMs, nowMs)].filter(Boolean).join(" · ");
}
/** "Miles-in-trail, arrivals · via COATE · set by PHL · volume: compacted demand · until 21:35Z" */
export function restrictionLine(r: TrafficRestriction, nowMs: number): string {
  return [r.words, r.elements.length ? `via ${r.elements.join(", ")}` : null, r.facility ? `set by ${r.facility}` : null, r.reason, periodWords(r.startMs, r.endMs, nowMs)].filter(Boolean).join(" · ");
}
/** The row's first line: the programs, then how many restrictions; a designed sentence when there are none. */
export function trafficHeadline(v: AirportTrafficView, nowMs: number): string {
  const n = v.restrictions.length;
  const rest = n ? `${n} restriction${n === 1 ? "" : "s"}` : null;
  if (v.programs.length) return [...v.programs.map((p) => programWords(p, nowMs)), rest].filter(Boolean).join(" · ");
  // THE LIST RESTARTS WITH THE SERVER (seen on production 2026-11-20, two minutes after a deploy: an airport under a
  // ground stop read "nothing in the FAA feed" because the server held 48 rows, none sent again yet). "Nothing" is
  // therefore said of what the server HOLDS, with the number, never of the FAA's system. Since 2026-11-21 the server
  // saves its list each minute and reads it back at a start (route field `kept`); the bracketed warning is for a
  // server that does not, and the opened list's source line says what a restart can still lose (restartWords).
  const held = `${v.searched} program${v.searched === 1 ? "" : "s"} and restriction${v.searched === 1 ? "" : "s"} held from the FAA feed`;
  // with restrictions to list, the same caveat is in the opened list's source line (the control opens it)
  return rest ? `No ground stop or delay program · ${rest}` : `Nothing for ${v.code} among the ${held}${v.kept ? "" : " (the list restarts with the server)"}`;
}
/** The source line's sentence on what a server restart does to the list. Three cases: the server does not keep the
 *  list; it keeps it and read rows back at its last start; it keeps it and had nothing to read back. */
export function restartWords(v: AirportTrafficView, nowMs: number): string {
  const k = v.kept;
  if (!k) return "The list starts again when the server restarts: a program sent before then is missing until the FAA sends it again.";
  const gap = "What the FAA sent while the server was restarting is missing until it is sent again, and a program cancelled then stays listed until its end.";
  if (k.carried === null || k.savedAtMs === null) {
    return "The list is saved each minute to be kept across a server restart, but nothing was read back at the last one: a program sent before it is missing until the FAA sends it again.";
  }
  const at = k.restoredAtMs !== null ? `, read back ${zuluAt(k.restoredAtMs, nowMs)}` : "";
  return `The list is kept across a server restart: ${k.carried} program${k.carried === 1 ? "" : "s"} and restriction${k.carried === 1 ? "" : "s"} carried over the last one (saved ${zuluAt(k.savedAtMs, nowMs)}${at}). ${gap}`;
}
const WEATHER_WORDS: Readonly<Record<string, string>> = { VMC: "visual conditions (VMC)", IMC: "instrument conditions (IMC)", LOW_VMC: "low visual conditions", LOW_IMC: "low instrument conditions" };
/** "Arriving 27/32 · departing 33L · 44 arrivals / 36 departures an hour · visual conditions (VMC) · reported 21:01Z"; null when the tower's report is not in the feed */
export function configLine(v: AirportTrafficView, nowMs: number): string | null {
  const c = v.config;
  if (!c) return null;
  const rates = [c.arrivalRate !== null ? `${c.arrivalRate} arrivals` : null, c.departureRate !== null ? `${c.departureRate} departures` : null].filter(Boolean).join(" / ");
  const parts = [c.arriving ? `Arriving ${c.arriving}` : null, c.departing ? `${c.arriving ? "departing" : "Departing"} ${c.departing}` : null, rates ? `${rates} an hour` : null,
    c.weather ? WEATHER_WORDS[c.weather.toUpperCase()] ?? c.weather : null, c.atMs !== null ? `reported ${zuluAt(c.atMs, nowMs)}` : null].filter(Boolean);
  return parts.length ? parts.join(" · ") : null;
}

/** One abortable read of the feed's active set. */
export async function fetchAirportTraffic(code: string, fetchImpl: (url: string, init?: { signal?: AbortSignal }) => Promise<{ ok: boolean; status: number; json: () => Promise<unknown> }>, signal?: AbortSignal): Promise<AirportTrafficView | null> {
  const r = await fetchImpl("/api/data/swim/tfms/active", { signal });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return parseAirportTraffic(await r.json(), code);
}
