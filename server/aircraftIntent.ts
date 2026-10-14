// aircraftIntent.ts — the ONE module that parses and normalizes what an
// aircraft says about its own intent and state (AIRCRAFT INTENT A,
// 2026-10-14): autopilot mode bits, the selected (MCP/FCU) and FMS altitude,
// selected heading, altimeter setting, plus the derived Mode-S EHS/ADS-B
// state fields (wind, temperature, airspeeds, roll, heading, both altitude
// and rate sources, ACAS RA).
//
// Every later intent feature (status line, vectored inference, ghost
// cleared-altitude line, go-around, safety flags, the E data products) reads
// these values through here — never by re-reading raw provider fields.
//
// HONESTY RULES (enforced here, tested in aircraftIntent.test.ts):
//   - a field the aircraft did not send is null — NEVER 0. A reported 0
//     (wind calm, wings level, 0 fpm) is a real value and survives.
//   - out-of-range / wrong-type values are rejected to null AND counted
//     (`rejected` in the coverage block) — never silently swallowed.
//   - nav_modes [] (mode bits sent, none engaged) is different from null
//     (no mode bits sent) and both are kept distinct.
//   - a value is FRESH only if it was observed within INTENT_FRESH_MS; a
//     stale value is never served as current (the view lists it under
//     `stale` with its age instead).
//   - per-aircraft CAPABILITY ("does it ever send mode bits?") is tracked,
//     so a later feature can tell "autopilot off" from "never says".
//
// No extra provider calls: values arrive on the existing fix bus (viewport
// discs, global sweep, scopes, tracked, gap-fill, fast lane, OpenSky). The
// parsed intent rides on each mapped row under a symbol key, so the
// /api/data/aircraft payload (which serializes those rows) is unchanged.

import type { FixBatch } from "./aircraftFixBus";

/** A value is current only if observed within this window (30 s). */
export const INTENT_FRESH_MS = 30_000;
/** Aircraft unseen this long drop out of the store. */
export const INTENT_TTL_MS = 10 * 60_000;
/** Hard bound on tracked aircraft (oldest-seen evicted first): ~0.9 KB each
 *  measured, so <= ~28 MB worst case (typical live load ~15k = ~14 MB). */
export const INTENT_MAX_HEXES = 30_000;
/** Coverage counts aircraft seen within this window. */
export const INTENT_COVERAGE_WINDOW_MS = 5 * 60_000;

/** Units live in ONE place. Storage keeps these; only display converts. */
export const INTENT_UNITS = {
  nav_modes: "set", nav_heading: "deg", nav_altitude_mcp: "ft", nav_altitude_fms: "ft",
  nav_qnh: "hPa", wd: "deg", ws: "kt", oat: "degC", tat: "degC", ias: "kt", tas: "kt",
  mach: "mach", roll: "deg", track_rate: "deg/s", mag_heading: "deg", true_heading: "deg",
  alt_geom: "ft", alt_baro: "ft", geom_rate: "ft/min", baro_rate: "ft/min",
  acas_ra: "object", squawk: "octal", emergency: "enum",
} as const;

export type IntentField = keyof typeof INTENT_UNITS;
export const INTENT_FIELDS = Object.keys(INTENT_UNITS) as IntentField[];
const FIELD_BIT: Record<string, number> = Object.fromEntries(INTENT_FIELDS.map((f, i) => [f, 1 << i]));

/** The autopilot mode bits readsb decodes from ADS-B target-state reports
 *  (and BDS 4,0 for the MCP/FMS altitude source). Anything else is dropped
 *  and counted, never invented into a known mode. */
export const NAV_MODES = ["autopilot", "vnav", "althold", "approach", "lnav", "tcas"] as const;
export type NavMode = typeof NAV_MODES[number];
const NAV_MODE_SET = new Set<string>(NAV_MODES);

/** readsb emergency enum (DO-260B emergency/priority status). */
const EMERGENCY_SET = new Set(["none", "general", "lifeguard", "minfuel", "nordo", "unlawful", "downed", "reserved"]);

export interface AcasRa {
  /** decoded advisory text when the feed sends one (e.g. "Climb") */
  advisory: string | null;
  advisory_complement: string | null;
  /** RA terminated / multiple-threat flags as sent */
  rat: boolean | null;
  mte: boolean | null;
  threat_hex: string | null;
  /** the RA's own time when sent (ms) */
  at_ms: number | null;
}

export interface IntentValues {
  nav_modes: NavMode[] | null;
  nav_heading: number | null;
  nav_altitude_mcp: number | null;
  nav_altitude_fms: number | null;
  nav_qnh: number | null;
  wd: number | null;
  ws: number | null;
  oat: number | null;
  tat: number | null;
  ias: number | null;
  tas: number | null;
  mach: number | null;
  roll: number | null;
  track_rate: number | null;
  mag_heading: number | null;
  true_heading: number | null;
  alt_geom: number | null;
  alt_baro: number | null;
  geom_rate: number | null;
  baro_rate: number | null;
  acas_ra: AcasRa | null;
  squawk: string | null;
  emergency: string | null;
}

/** One parsed observation: the values, the fields that arrived but were
 *  rejected as garbage, and how old the message was at the feed (s). */
export interface IntentParse {
  v: IntentValues;
  /** fields present upstream but rejected (wrong type / out of range) */
  rejected: IntentField[];
  /** readsb `seen` (s since the feed's last message of any kind) */
  seen_s: number | null;
  on_ground: boolean;
}


const isNum = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);

// Hot path (every mapped row of every batch): one fixed-shape literal per
// parse, rejections collected lazily — no per-row arrays when nothing fails.
let rej: IntentField[] | null = null;
const NO_REJECTS: IntentField[] = [];
function bad(f: IntentField): null {
  (rej ??= []).push(f);
  return null;
}
function num(x: unknown, lo: number, hi: number, f: IntentField): number | null {
  if (x === undefined || x === null) return null; // absent: stays null
  return typeof x === "number" && x >= lo && x <= hi ? x : bad(f); // NaN fails the range test
}
function takeRejects(): IntentField[] {
  const r = rej ?? NO_REJECTS;
  rej = null;
  return r;
}

function emptyValues(): IntentValues {
  return {
    nav_modes: null, nav_heading: null, nav_altitude_mcp: null, nav_altitude_fms: null, nav_qnh: null,
    wd: null, ws: null, oat: null, tat: null, ias: null, tas: null, mach: null, roll: null, track_rate: null,
    mag_heading: null, true_heading: null, alt_geom: null, alt_baro: null, geom_rate: null, baro_rate: null,
    acas_ra: null, squawk: null, emergency: null,
  };
}

const shortStr = (x: unknown, max = 64): string | null =>
  typeof x === "string" && x.trim() ? x.trim().slice(0, max) : null;
const flag = (x: unknown): boolean | null =>
  x === true || x === 1 || x === "1" ? true : x === false || x === 0 || x === "0" ? false : null;

/** readsb `acas_ra` → compact typed RA. Field names follow readsb's JSON
 *  output (advisory, advisory_complement, RAT, MTE, threat_id_hex,
 *  unix_timestamp); anything not an object is rejected. */
export function parseAcasRa(x: unknown): AcasRa | null {
  if (!x || typeof x !== "object" || Array.isArray(x)) return null;
  const o = x as Record<string, unknown>;
  const ts = isNum(o.unix_timestamp) ? o.unix_timestamp : null;
  const ra: AcasRa = {
    advisory: shortStr(o.advisory),
    advisory_complement: shortStr(o.advisory_complement),
    rat: flag(o.RAT),
    mte: flag(o.MTE),
    threat_hex: typeof o.threat_id_hex === "string" && /^[0-9a-f]{6}$/i.test(o.threat_id_hex.trim())
      ? o.threat_id_hex.trim().toLowerCase() : null,
    at_ms: ts == null ? null : Math.round(ts < 1e11 ? ts * 1000 : ts),
  };
  return ra;
}

/** readsb / tar1090 v2 aircraft record (adsb.lol, airplanes.live, adsb.fi)
 *  → normalized intent. Units are already ft / kt / deg / hPa / °C / fpm in
 *  readsb; this only validates and nulls. */
export function parseReadsbIntent(a: unknown): IntentParse {
  if (!a || typeof a !== "object") return { v: emptyValues(), rejected: NO_REJECTS, seen_s: null, on_ground: false };
  const r = a as Record<string, unknown>;
  rej = null;
  let navModes: NavMode[] | null = null;
  if (r.nav_modes !== undefined && r.nav_modes !== null) {
    if (Array.isArray(r.nav_modes)) {
      const known: NavMode[] = [];
      let unknown = 0;
      for (const m0 of r.nav_modes) {
        const m = typeof m0 === "string" ? m0.trim().toLowerCase() : "";
        if (NAV_MODE_SET.has(m)) { if (!known.includes(m as NavMode)) known.push(m as NavMode); } else unknown++;
      }
      if (unknown) bad("nav_modes"); // an unknown mode string arrived
      // [] is real: mode bits sent, nothing engaged
      if (known.length || r.nav_modes.length === 0) navModes = known;
    } else bad("nav_modes");
  }
  const onGround = r.alt_baro === "ground"; // on the ground: no baro altitude
  let acas: AcasRa | null = null;
  if (r.acas_ra !== undefined && r.acas_ra !== null) acas = parseAcasRa(r.acas_ra) ?? bad("acas_ra");
  let squawk: string | null = null;
  if (r.squawk !== undefined && r.squawk !== null) {
    const sq = typeof r.squawk === "string" ? r.squawk.trim() : "";
    squawk = /^[0-7]{4}$/.test(sq) ? sq : bad("squawk");
  }
  let emergency: string | null = null;
  if (r.emergency !== undefined && r.emergency !== null) {
    const em = typeof r.emergency === "string" ? r.emergency.trim().toLowerCase() : "";
    emergency = EMERGENCY_SET.has(em) ? em : bad("emergency");
  }
  // numeric fields: [lo, hi] are physical plausibility bounds (inclusive) —
  // generous, they reject decode garbage (e.g. a -215 °C OAT seen live on
  // 2026-10-14), not unusual aircraft
  const v: IntentValues = {
    nav_modes: navModes,
    nav_heading: num(r.nav_heading, 0, 360, "nav_heading"),
    nav_altitude_mcp: num(r.nav_altitude_mcp, -2_000, 100_000, "nav_altitude_mcp"),
    nav_altitude_fms: num(r.nav_altitude_fms, -2_000, 100_000, "nav_altitude_fms"),
    nav_qnh: num(r.nav_qnh, 800, 1_100, "nav_qnh"),
    wd: num(r.wd, 0, 360, "wd"),
    ws: num(r.ws, 0, 400, "ws"),
    oat: num(r.oat, -100, 60, "oat"),
    tat: num(r.tat, -100, 100, "tat"),
    ias: num(r.ias, 0, 1_000, "ias"),
    tas: num(r.tas, 0, 2_500, "tas"),
    mach: num(r.mach, 0, 4, "mach"),
    roll: num(r.roll, -180, 180, "roll"),
    track_rate: num(r.track_rate, -90, 90, "track_rate"),
    mag_heading: num(r.mag_heading, 0, 360, "mag_heading"),
    true_heading: num(r.true_heading, 0, 360, "true_heading"),
    alt_geom: num(r.alt_geom, -2_000, 120_000, "alt_geom"),
    alt_baro: onGround ? null : num(r.alt_baro, -2_000, 120_000, "alt_baro"),
    geom_rate: num(r.geom_rate, -60_000, 60_000, "geom_rate"),
    baro_rate: num(r.baro_rate, -60_000, 60_000, "baro_rate"),
    acas_ra: acas,
    squawk,
    emergency,
  };
  return { v, rejected: takeRejects(), seen_s: isNum(r.seen) && r.seen >= 0 ? r.seen : null, on_ground: onGround };
}

/** OpenSky state vector → the intent fields it carries: baro and geometric
 *  altitude (meters → ft) and squawk. OpenSky sends no mode bits, selected
 *  altitude, wind, temperature or airspeeds — those stay null. */
export function parseOpenSkyIntent(s: unknown): IntentParse {
  const v = emptyValues();
  if (!Array.isArray(s)) return { v, rejected: NO_REJECTS, seen_s: null, on_ground: false };
  rej = null;
  const onGround = s[8] === true;
  const ft = (m: unknown) => (isNum(m) ? Math.round(m / 0.3048) : m);
  if (!onGround) v.alt_baro = num(ft(s[7]), -2_000, 120_000, "alt_baro");
  v.alt_geom = num(ft(s[13]), -2_000, 120_000, "alt_geom");
  if (s[14] !== undefined && s[14] !== null) {
    const sq = typeof s[14] === "string" ? s[14].trim() : "";
    v.squawk = /^[0-7]{4}$/.test(sq) ? sq : bad("squawk");
  }
  return { v, rejected: takeRejects(), seen_s: null, on_ground: onGround };
}

/** Which fields carry a value in this parse (bitmask over INTENT_FIELDS). */
export function presentMask(v: IntentValues): number {
  let m = 0;
  for (const f of INTENT_FIELDS) if ((v as unknown as Record<string, unknown>)[f] !== null) m |= FIELD_BIT[f];
  return m;
}

// The parsed intent rides on each mapped row under a SYMBOL key: invisible
// to JSON.stringify, Object.keys and `in "intent"`, so every serialized
// aircraft payload stays byte-identical, while the bus subscriber reads it
// from the very object the bus carries (spread copies keep it too).
export const INTENT_KEY: unique symbol = Symbol("aircraftIntent");
export function attachIntent(row: object, p: IntentParse): void {
  (row as { [INTENT_KEY]?: IntentParse })[INTENT_KEY] = p;
}
export function intentOf(row: object): IntentParse | undefined {
  return (row as { [INTENT_KEY]?: IntentParse })[INTENT_KEY];
}

// ── Per-aircraft store ───────────────────────────────────────────────────

interface HexState {
  lastSeen: number;
  provider: string | null;
  onGround: boolean;
  /** fields present in the latest observation */
  mask: number;
  /** fields ever present (capability) */
  everMask: number;
  samples: number;
  firstSeen: number;
  vals: unknown[];
  ats: number[];
  provs: (string | null)[];
}

export interface IntentView {
  hex: string;
  at: number;
  last_seen_ms: number;
  provider: string | null;
  on_ground: boolean;
  /** FRESH values only (observed within fresh_s) */
  values: Partial<IntentValues>;
  /** age (s) of every field held, fresh or stale */
  age_s: Partial<Record<IntentField, number>>;
  /** which upstream supplied each held field */
  source: Partial<Record<IntentField, string | null>>;
  /** held but older than fresh_s — never served as current */
  stale: IntentField[];
  capability: {
    /** has this aircraft EVER sent autopilot mode bits (nav_modes)? */
    mode_bits: boolean;
    fields_ever: IntentField[];
    samples: number;
    first_seen_ms: number;
  };
  fresh_s: number;
  units: typeof INTENT_UNITS;
}

export interface IntentCoverage {
  at: number;
  window_s: number;
  /** airborne aircraft seen in the window */
  aircraft: number;
  /** % of those whose LATEST fix carried each field (what the feeds send,
   *  independent of our polling cadence) */
  latest_pct: Record<IntentField, number>;
  /** % that have ever sent each field since we started tracking them */
  ever_pct: Record<IntentField, number>;
  /** % that have ever sent mode bits */
  mode_bit_capable_pct: number;
  by_provider: Record<string, { aircraft: number; latest_pct: Partial<Record<IntentField, number>> }>;
  /** values that arrived but failed validation (cumulative) */
  rejected: Partial<Record<IntentField, number>>;
  tracked: number;
  rows_ingested: number;
  note: string;
}

const pct = (n: number, d: number) => (d ? Math.round((n / d) * 1000) / 10 : 0);

export class IntentStore {
  private readonly hexes = new Map<string, HexState>();
  private readonly rejectedCounts: Partial<Record<IntentField, number>> = {};
  private rows = 0;
  readonly freshMs: number;
  readonly ttlMs: number;
  readonly maxHexes: number;

  constructor(o: { freshMs?: number; ttlMs?: number; maxHexes?: number } = {}) {
    this.freshMs = o.freshMs ?? INTENT_FRESH_MS;
    this.ttlMs = o.ttlMs ?? INTENT_TTL_MS;
    this.maxHexes = o.maxHexes ?? INTENT_MAX_HEXES;
  }

  get size(): number { return this.hexes.size; }

  /** Observe one parsed row at time `t` (the fix's own time). Per field the
   *  freshest observation wins, so a late batch never overwrites newer data. */
  observe(hexIn: string, p: IntentParse, t: number, provider: string | null): void {
    const hex = String(hexIn || "").toLowerCase();
    if (!hex || !Number.isFinite(t)) return;
    this.rows++;
    for (const f of p.rejected) this.rejectedCounts[f] = (this.rejectedCounts[f] || 0) + 1;
    let s = this.hexes.get(hex);
    if (!s) {
      s = {
        lastSeen: t, provider, onGround: p.on_ground, mask: 0, everMask: 0, samples: 0, firstSeen: t,
        vals: new Array(INTENT_FIELDS.length).fill(null),
        ats: new Array(INTENT_FIELDS.length).fill(0),
        provs: new Array(INTENT_FIELDS.length).fill(null),
      };
    } else this.hexes.delete(hex); // re-insert below: Map order = LRU by last observe
    this.hexes.set(hex, s);
    s.samples++;
    const mask = presentMask(p.v);
    if (t >= s.lastSeen) {
      s.lastSeen = t; s.provider = provider; s.onGround = p.on_ground; s.mask = mask;
    }
    s.everMask |= mask;
    const vals = p.v as unknown as Record<string, unknown>;
    for (let i = 0; i < INTENT_FIELDS.length; i++) {
      const val = vals[INTENT_FIELDS[i]];
      if (val === null || t < s.ats[i]) continue;
      s.vals[i] = val; s.ats[i] = t; s.provs[i] = provider;
    }
    if (this.hexes.size > this.maxHexes) {
      const oldest = this.hexes.keys().next().value;
      if (oldest !== undefined) this.hexes.delete(oldest);
    }
  }

  /** Fix-bus subscriber: every mapped row carrying an attached parse. The
   *  observation time is the fix's own time when the row has one (OpenSky
   *  seen_at_ms), else receive time minus the feed's `seen` age. */
  ingest(b: FixBatch, now: number = Date.now()): number {
    let n = 0;
    for (const row of b.aircraft) {
      const p = intentOf(row);
      if (!p) continue;
      const own = (row as { seen_at_ms?: number | null }).seen_at_ms;
      const base = Number.isFinite(b.fetchedAt) ? b.fetchedAt : now;
      const t = isNum(own) ? own : base - (p.seen_s ?? 0) * 1000;
      this.observe(row.icao24, p, Math.min(t, now), row.provider ?? b.provider ?? null);
      n++;
    }
    return n;
  }

  evict(now: number = Date.now()): number {
    let n = 0;
    for (const [hex, s] of Array.from(this.hexes)) {
      if (now - s.lastSeen > this.ttlMs) { this.hexes.delete(hex); n++; }
    }
    return n;
  }

  /** The aircraft's intent as of `now`: fresh values only, stale fields
   *  listed with their age, capability alongside. null if never seen. */
  view(hexIn: string, now: number = Date.now()): IntentView | null {
    const hex = String(hexIn || "").toLowerCase();
    const s = this.hexes.get(hex);
    if (!s) return null;
    const values: Record<string, unknown> = {};
    const age: Record<string, number> = {};
    const source: Record<string, string | null> = {};
    const stale: IntentField[] = [];
    for (let i = 0; i < INTENT_FIELDS.length; i++) {
      if (!s.ats[i] || s.vals[i] === null) continue;
      const f = INTENT_FIELDS[i];
      const a = Math.max(0, now - s.ats[i]);
      age[f] = Math.round(a / 100) / 10;
      source[f] = s.provs[i];
      if (a <= this.freshMs) values[f] = s.vals[i];
      else stale.push(f);
    }
    return {
      hex, at: now, last_seen_ms: s.lastSeen, provider: s.provider, on_ground: s.onGround,
      values: values as Partial<IntentValues>, age_s: age, source, stale,
      capability: {
        mode_bits: (s.everMask & FIELD_BIT.nav_modes) !== 0,
        fields_ever: INTENT_FIELDS.filter((f) => (s.everMask & FIELD_BIT[f]) !== 0),
        samples: s.samples,
        first_seen_ms: s.firstSeen,
      },
      fresh_s: this.freshMs / 1000,
      units: INTENT_UNITS,
    };
  }

  /** Field coverage across airborne aircraft seen in the window. */
  coverage(now: number = Date.now(), windowMs: number = INTENT_COVERAGE_WINDOW_MS): IntentCoverage {
    const latest = new Array(INTENT_FIELDS.length).fill(0);
    const ever = new Array(INTENT_FIELDS.length).fill(0);
    const byProv = new Map<string, { n: number; c: number[] }>();
    let n = 0;
    let modeCapable = 0;
    for (const s of Array.from(this.hexes.values())) {
      if (now - s.lastSeen > windowMs || s.onGround) continue;
      n++;
      if (s.everMask & FIELD_BIT.nav_modes) modeCapable++;
      const pk = s.provider || "unknown";
      let bp = byProv.get(pk);
      if (!bp) { bp = { n: 0, c: new Array(INTENT_FIELDS.length).fill(0) }; byProv.set(pk, bp); }
      bp.n++;
      for (let i = 0; i < INTENT_FIELDS.length; i++) {
        const bit = 1 << i;
        if (s.mask & bit) { latest[i]++; bp.c[i]++; }
        if (s.everMask & bit) ever[i]++;
      }
    }
    const rec = (arr: number[], d: number) =>
      Object.fromEntries(INTENT_FIELDS.map((f, i) => [f, pct(arr[i], d)])) as Record<IntentField, number>;
    const by_provider: IntentCoverage["by_provider"] = {};
    for (const [k, bp] of Array.from(byProv)) by_provider[k] = { aircraft: bp.n, latest_pct: rec(bp.c, bp.n) };
    return {
      at: now, window_s: windowMs / 1000, aircraft: n,
      latest_pct: rec(latest, n), ever_pct: rec(ever, n),
      mode_bit_capable_pct: pct(modeCapable, n),
      by_provider, rejected: { ...this.rejectedCounts }, tracked: this.hexes.size, rows_ingested: this.rows,
      note: "Airborne aircraft seen in the window. latest_pct = share whose newest fix carried the field "
        + "(what the feeds send); ever_pct = share that sent it at least once since tracked. Fields an "
        + "aircraft never sends are null, never 0. A value is current only within fresh_s of its observation.",
    };
  }
}
