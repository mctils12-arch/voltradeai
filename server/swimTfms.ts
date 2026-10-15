// swimTfms.ts — FAA SWIM TFMS parser (ops item 1, follows the capture PR
// #1342). Written against REAL messages captured from our queue on
// 2026-10-14 (shape profile + diag-token samples), not an assumed schema.
//
// What it keeps (all public-release per the SCDS NDRB banner; the per-flight
// records are served only through the same LADD gate as FAA plans):
//   - PER FLIGHT (keyed by callsign): route amendments — ERAM/HCS amendments
//     (fltdMessage flightPlanAmendmentInformation: new route string, assigned
//     altitude) and TMI reroute amendments (fiMessage TMI_UPDATE rrAmendment:
//     the reroute program id, e.g. "TRZID", and the amended route) — plus TMI
//     participation (TMI_FLIGHT_LIST: the FCA/program the flight is in).
//     These are what the deviation classifier needs to say "ATC reroute"
//     instead of "off the filed route".
//   - PROGRAMS: ground delay programs, ground stops, restrictions (MIT/APREQ),
//     ATCSCC advisories and flow-constrained areas, each with its window.
//   - AIRPORT CONFIG: live arrival/departure runway configuration and
//     arrival/departure rates (APTC) — runway prediction and the ATC-stress
//     product read these.
// Everything else (track, sector, times, schedule…) is counted, not parsed:
// SFDPS already supplies positions and plans.

import { parseXmlLite, attr, findAll, findFirst, type XNode } from "./swimSfdps";

export const TFMS_AMENDMENT_TTL_MS = 12 * 3600_000;
export const TFMS_MAX_FLIGHTS = 20_000;
export const TFMS_AMENDMENTS_PER_FLIGHT = 8;
/** a program without an end time is kept this long after its last update */
export const TFMS_PROGRAM_DEFAULT_TTL_MS = 6 * 3600_000;
export const TFMS_MAX_PROGRAMS = 2_000;

/** msgTypes worth a full parse; everything else is only counted */
const PARSE_RE = /msgType="(RSTR|GDP|GS|AFP|CTOP|GADV|FXA|APTC|TMI_UPDATE|TMI_FLIGHT_LIST|flightPlanAmendmentInformation)"/;

export interface TfmsAmendment {
  acid: string;
  gufi: string | null;
  dep: string | null;
  arr: string | null;
  /** the amended route as filed in legacy (ICAO-ish) text */
  route: string | null;
  /** assigned altitude, hundreds of ft (e.g. 240 = FL240), when amended */
  assignedFl: number | null;
  /** HCS/ERAM amendment, or a TMI reroute with its program id */
  kind: "ERAM_AMENDMENT" | "TMI_REROUTE";
  tmiId: string | null;
  status: string | null;
  facility: string | null;
  at: number;
}

export interface TfmsProgram {
  id: string;
  kind: "GDP" | "GS" | "AFP" | "CTOP" | "RESTRICTION" | "ADVISORY" | "FCA";
  status: string | null;
  airport: string | null;
  facility: string | null;
  start: number | null;
  end: number | null;
  title: string | null;
  text: string | null;
  reason: string | null;
  elements: string[];
  /** FCA line/polygon points [lat, lon] */
  geometry: Array<[number, number]> | null;
  floorFl: number | null;
  ceilingFl: number | null;
  updatedAt: number;
}

export interface AirportConfig {
  airport: string;
  arrRunways: string | null;
  depRunways: string | null;
  aar: number | null;
  adr: number | null;
  weather: string | null;
  facility: string | null;
  at: number;
}

export interface FlightTmi { acid: string; fcaId: string; fcaName: string | null; entryAt: number | null; at: number }

export interface TfmsParsed {
  amendments: TfmsAmendment[];
  programs: TfmsProgram[];
  configs: AirportConfig[];
  tmis: FlightTmi[];
  /** msgTypes seen in the payload (counted by the store) */
  types: string[];
}

const txt = (n: XNode | null | undefined): string | null => {
  const t = n?.text?.trim();
  return t ? t : null;
};
const first = (n: XNode | null | undefined, name: string): string | null => txt(findFirst(n, name));
const num = (s: string | null): number | null => {
  if (s == null || s === "") return null;
  const v = Number(s);
  return Number.isFinite(v) ? v : null;
};
const time = (s: string | null): number | null => {
  if (!s) return null;
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : null;
};

const ADVISORY_KIND: Array<[RegExp, TfmsProgram["kind"]]> = [
  [/GROUND STOP/i, "GS"], [/GROUND DELAY/i, "GDP"], [/AIRSPACE FLOW|\bAFP\b/i, "AFP"], [/\bCTOP\b/i, "CTOP"],
];

function parseFi(fi: XNode, typ: string, now: number): Partial<TfmsParsed> {
  const ts = time(attr(fi, "sourceTimeStamp")) ?? now;
  const facility = attr(fi, "sourceFacility");
  const out: Partial<TfmsParsed> = { amendments: [], programs: [], configs: [], tmis: [] };
  if (typ === "RSTR") {
    const r = findFirst(fi, "restrictionMessage");
    if (!r) return out;
    const id = first(r, "restrictionId");
    out.programs!.push({
      id: `RSTR:${id ?? `${first(r, "facility")}:${first(r, "restrictedNasElements")}`}`, kind: "RESTRICTION",
      status: first(r, "action"), airport: first(r, "airports"), facility: first(r, "facility") ?? facility,
      start: time(first(r, "startTime")), end: time(first(r, "stopTime")),
      title: [first(r, "restrictionCategory"), first(r, "restrictionType")].filter(Boolean).join(" ") || null,
      text: null, reason: first(r, "reasonText"),
      elements: (first(r, "restrictedNasElements") ?? "").split(/[\s,]+/).filter(Boolean),
      geometry: null, floorFl: null, ceilingFl: null, updatedAt: ts,
    });
  } else if (typ === "GDP" || typ === "GS" || typ === "AFP" || typ === "CTOP") {
    const body = fi.children[0];
    if (!body) return out;
    const airport = first(body, "airportId") ?? first(body, "ctlElement") ?? first(body, "airport");
    const vp = findFirst(body, "advisoryValidPeriod") ?? findFirst(body, "effectivePeriod") ?? body;
    out.programs!.push({
      id: `${typ}:${airport ?? first(body, "fcaId") ?? body.name}`, kind: typ as TfmsProgram["kind"],
      status: first(body, "tmiStatus") ?? body.name, airport, facility: first(body, "center") ?? facility,
      start: time(first(vp, "startTime")), end: time(first(vp, "endTime")) ?? time(first(body, "pgmExpTime")),
      title: body.name, text: null, reason: first(body, "impactingCondition") ?? first(body, "cause"),
      elements: [], geometry: null, floorFl: null, ceilingFl: null, updatedAt: time(first(body, "updateTime")) ?? ts,
    });
  } else if (typ === "GADV") {
    const a = findFirst(fi, "generalAdvisory") ?? fi.children[0];
    if (!a) return out;
    const title = first(a, "advisoryTitle");
    const textBody = first(a, "advisoryText");
    const kindHit = ADVISORY_KIND.find(([re]) => re.test(title ?? ""));
    const ep = findFirst(a, "effectivePeriod");
    const ctl = /CTL ELEMENT:\s*([A-Z0-9]{3,4})/.exec(textBody ?? "");
    out.programs!.push({
      id: `ADV:${first(a, "origin") ?? ""}:${first(a, "advisoryNumber") ?? ""}:${(first(a, "dateSent") ?? "").slice(0, 10)}`,
      kind: "ADVISORY", status: kindHit ? kindHit[1] : null, airport: ctl ? ctl[1] : null, facility,
      start: time(first(ep, "startTime")), end: time(first(ep, "endTime")),
      title, text: textBody ? textBody.slice(0, 4000) : null, reason: null, elements: [],
      geometry: null, floorFl: null, ceilingFl: null, updatedAt: time(first(a, "dateSent")) ?? ts,
    });
  } else if (typ === "FXA") {
    const f = findFirst(fi, "feaFca");
    if (!f) return out;
    const pts: Array<[number, number]> = [];
    for (const p of findAll(f, "point")) {
      const la = num(first(p, "latitude")), lo = num(first(p, "longitude"));
      if (la != null && lo != null && Math.abs(la) <= 90 && Math.abs(lo) <= 180) pts.push([la, lo]);
    }
    out.programs!.push({
      id: `FCA:${first(f, "fcaId") ?? first(f, "fcaName")}`, kind: "FCA", status: first(f, "tmiStatus"),
      airport: null, facility, start: time(first(f, "startTime")), end: time(first(f, "endTime")),
      title: first(f, "fcaName"), text: null, reason: first(f, "fcaReason"), elements: [],
      geometry: pts.length ? pts : null, floorFl: num(first(f, "floor")), ceilingFl: num(first(f, "ceiling")),
      updatedAt: time(first(f, "lastUpdate")) ?? ts,
    });
  } else if (typ === "APTC") {
    const c = findFirst(fi, "airportConfigMessage");
    const ap = first(c, "airport");
    if (!c || !ap) return out;
    out.configs!.push({
      airport: ap, arrRunways: first(c, "arrRunwayConf"), depRunways: first(c, "depRunwayConf"),
      aar: num(first(c, "arrRate")), adr: num(first(c, "depRate")), weather: first(c, "weather"),
      facility: first(c, "facility") ?? facility, at: time(first(c, "updateTime")) ?? time(first(c, "eventTime")) ?? ts,
    });
  } else if (typ === "TMI_UPDATE") {
    for (const rr of findAll(fi, "rrAmendment")) {
      const st = findFirst(rr, "flightAmendmentStatus");
      const acid = first(rr, "acid");
      if (!acid) continue;
      out.amendments!.push({
        acid, gufi: first(rr, "gufi"), dep: null, arr: null, route: first(st, "routeAmendment"), assignedFl: null,
        kind: "TMI_REROUTE", tmiId: first(st, "tmiId"), status: first(st, "amendmentStatus"),
        facility: first(rr, "controllingSector"), at: time(first(st, "amendmentTime")) ?? ts,
      });
    }
  } else if (typ === "TMI_FLIGHT_LIST") {
    for (const fd of findAll(fi, "flightData")) {
      const acid = first(fd, "aircraftId");
      if (!acid) continue;
      for (const fx of findAll(fd, "fxaFlight")) {
        const id = first(fx, "fcaId");
        if (!id) continue;
        out.tmis!.push({ acid, fcaId: id, fcaName: first(fx, "fcaName"), entryAt: time(first(fx, "entryTm")), at: ts });
      }
    }
  }
  return out;
}

function parseFltdAmendment(m: XNode, now: number): TfmsAmendment | null {
  const acid = attr(m, "acid");
  if (!acid) return null;
  const a = findFirst(m, "flightPlanAmendmentInformation");
  const route = findFirst(a, "newRouteOfFlight");
  const alt = first(findFirst(a, "newAltitude"), "simpleAltitude");
  return {
    acid, gufi: first(a, "gufi"), dep: attr(m, "depArpt"), arr: attr(m, "arrArpt"),
    route: route ? (attr(route, "legacyFormat") ?? txt(route)) : null,
    assignedFl: num(alt), kind: "ERAM_AMENDMENT", tmiId: null, status: attr(m, "fdTrigger"),
    facility: attr(m, "sourceFacility"), at: time(attr(m, "sourceTimeStamp")) ?? now,
  };
}

/** Pure: one TFMS payload → the records worth keeping. */
export function parseTfmsPayload(xml: string, now: number = Date.now()): TfmsParsed {
  const out: TfmsParsed = { amendments: [], programs: [], configs: [], tmis: [], types: [] };
  if (typeof xml !== "string" || !xml) return out;
  const typeRe = /msgType="([^"]{1,48})"/g;
  let m: RegExpExecArray | null;
  while ((m = typeRe.exec(xml))) out.types.push(m[1]);
  if (!PARSE_RE.test(xml)) return out; // counted only — never parsed
  const doc = parseXmlLite(xml);
  if (!doc) return out;
  for (const fi of findAll(doc, "fiMessage")) {
    const r = parseFi(fi, attr(fi, "msgType") ?? "", now);
    out.amendments.push(...(r.amendments ?? []));
    out.programs.push(...(r.programs ?? []));
    out.configs.push(...(r.configs ?? []));
    out.tmis.push(...(r.tmis ?? []));
  }
  for (const fm of findAll(doc, "fltdMessage")) {
    if (attr(fm, "msgType") !== "flightPlanAmendmentInformation") continue;
    const a = parseFltdAmendment(fm, now);
    if (a) out.amendments.push(a);
  }
  return out;
}

/** In-memory TFMS state for the deviation classifier, the card and E1. */
export class TfmsStore {
  private readonly flights = new Map<string, { amendments: TfmsAmendment[]; tmis: FlightTmi[]; lastAt: number }>();
  private readonly programs = new Map<string, TfmsProgram>();
  private readonly configs = new Map<string, AirportConfig>();
  readonly typeCounts: Record<string, number> = {};
  parsedMessages = 0;
  parseErrors = 0;

  ingest(xml: string, now: number = Date.now()): void {
    let p: TfmsParsed;
    try { p = parseTfmsPayload(xml, now); } catch { this.parseErrors++; return; }
    for (const t of p.types) this.typeCounts[t] = (this.typeCounts[t] || 0) + 1;
    if (p.amendments.length || p.programs.length || p.configs.length || p.tmis.length) this.parsedMessages++;
    for (const a of p.amendments) {
      const f = this.flight(a.acid, now);
      f.amendments.push(a);
      f.amendments.sort((x, y) => x.at - y.at);
      if (f.amendments.length > TFMS_AMENDMENTS_PER_FLIGHT) f.amendments.splice(0, f.amendments.length - TFMS_AMENDMENTS_PER_FLIGHT);
    }
    for (const t of p.tmis) {
      const f = this.flight(t.acid, now);
      if (!f.tmis.some((x) => x.fcaId === t.fcaId)) f.tmis.push(t);
      if (f.tmis.length > 8) f.tmis.shift();
    }
    for (const pr of p.programs) {
      const prev = this.programs.get(pr.id);
      if (!prev || pr.updatedAt >= prev.updatedAt) this.programs.set(pr.id, pr);
    }
    for (const c of p.configs) {
      const prev = this.configs.get(c.airport);
      if (!prev || c.at >= prev.at) this.configs.set(c.airport, c);
    }
    if (this.programs.size > TFMS_MAX_PROGRAMS) this.evict(now);
  }

  private flight(acid: string, now: number) {
    const k = acid.trim().toUpperCase();
    let f = this.flights.get(k);
    if (!f) {
      f = { amendments: [], tmis: [], lastAt: now };
      this.flights.set(k, f);
      if (this.flights.size > TFMS_MAX_FLIGHTS) {
        const oldest = this.flights.keys().next().value;
        if (oldest !== undefined) this.flights.delete(oldest);
      }
    }
    f.lastAt = now;
    return f;
  }

  /** a program is active from its start (or now) until its end (or a
   *  default TTL past its last update); purged/cancelled ones are not */
  isActive(p: TfmsProgram, now: number): boolean {
    if (/PURGED|CANCEL|CNX|DELETED/i.test(`${p.status ?? ""} ${p.title ?? ""}`)) return false;
    const end = p.end ?? p.updatedAt + TFMS_PROGRAM_DEFAULT_TTL_MS;
    return end >= now && (p.start == null || p.start <= now + 6 * 3600_000);
  }

  evict(now: number = Date.now()): void {
    for (const [k, f] of Array.from(this.flights)) if (now - f.lastAt > TFMS_AMENDMENT_TTL_MS) this.flights.delete(k);
    for (const [k, p] of Array.from(this.programs)) {
      const end = p.end ?? p.updatedAt + TFMS_PROGRAM_DEFAULT_TTL_MS;
      if (end < now - 3600_000) this.programs.delete(k);
    }
  }

  forFlight(callsign: string, now: number = Date.now()) {
    const f = this.flights.get(String(callsign || "").trim().toUpperCase());
    if (!f || now - f.lastAt > TFMS_AMENDMENT_TTL_MS) return { amendments: [], tmis: [] };
    return { amendments: f.amendments.slice(), tmis: f.tmis.slice() };
  }

  active(now: number = Date.now()) {
    const programs = Array.from(this.programs.values()).filter((p) => this.isActive(p, now))
      .sort((a, b) => (a.kind === b.kind ? (a.airport ?? "").localeCompare(b.airport ?? "") : a.kind.localeCompare(b.kind)));
    return { programs, airportConfigs: Array.from(this.configs.values()).sort((a, b) => a.airport.localeCompare(b.airport)) };
  }

  airportConfig(airport: string): AirportConfig | null {
    const k = String(airport || "").trim().toUpperCase().replace(/^K(?=[A-Z0-9]{3}$)/, "");
    return this.configs.get(k) ?? null;
  }

  stats() {
    return {
      flights: this.flights.size, programs: this.programs.size, airportConfigs: this.configs.size,
      parsedMessages: this.parsedMessages, parseErrors: this.parseErrors, typeCounts: { ...this.typeCounts },
    };
  }
}
