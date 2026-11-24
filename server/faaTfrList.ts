// faaTfrList.ts — the FAA's own public list of flight restrictions
// (tfr.faa.gov), read on a timer, so the /data map's "Flight restrictions"
// layer holds the STANDING US restrictions the notice feed never re-sends.
//
// WHY ([PRODUCT] 2026-11-22; research/open_questions.md 2026-11-21 (u3) and
// 2026-11-22 (w3)). The layer was built on the FAA notice feed (SWIM FNS
// NOTAM, swimNotam.ts). A feed of changes cannot list what was issued before
// we listened: on 2026-11-22 the FAA listed 61 restrictions and the feed
// store held 1 of them. LADDER GATE 1 (DATA) for COVERAGE therefore failed,
// and the layer said "not the FAA's complete list" everywhere.
//
// BUILD-FIRST: the raw material is free and public, no key.
//   list    https://tfr.faa.gov/tfrapi/exportTfrList            every restriction the FAA lists (number, type, facility, state)
//   detail  https://tfr.faa.gov/download/detail_<y>_<nnnn>.xml  one notice: the text as issued, its dates, and for each of
//                                                              its areas the FAA's OWN outline (abdMergedArea) and limits
// A detail file is read once per notice number (a changed restriction is a
// new notice with a new number), so a steady day is one list request each
// 30 minutes and two to four detail files.
//
// WHAT IT SERVES. One row per listed notice the FAA publishes an outline
// for, shaped like a feed row (swimNotam.TfrRecord) with two differences that
// are never hidden: `source: "faa_list"` and `geometry.from: "faa_list"`. The
// outline is the FAA's drawing, copied; nothing is read out of words here
// (swimNotamText.ts does that for feed-only notices). A listed notice whose
// detail file carries no outline (the nationwide security instructions, the
// Washington text notices) is COUNTED and NAMED, never drawn by guess.
//
// THE COVERAGE CHECK, every pass: FAA-listed notices against what this
// process holds from the list and, separately, from the feed. `coverage()`
// is the continuous form of gate 1 for coverage: it says how many listed
// notices are drawn, how many have no published outline, how many detail
// files could not be read, and how many the feed alone would have shown.
//
// WHAT THIS CANNOT KNOW. Only US restrictions the FAA lists publicly; other
// countries' notices still come from the feed alone. Between two reads a new
// restriction is known only if the feed sent it. If tfr.faa.gov cannot be
// reached the last read is served with its age for at most 24 hours, then
// nothing from the list is served and the route says so. An area the FAA
// schedules for certain hours of the day is listed as in force for its whole
// period (`scheduled: true` says which).
//
// RAW overlay. No trading path reads this module.
import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";
import { beat } from "./jobHeartbeat";
import { parseXmlLite, findAll, findFirst, type XNode } from "./swimSfdps";
import { swimMissingEnv } from "./swimConnector";
import { faaTfrArchiveEnabled, loadFaaTfrArchive, recordFaaTfrRead } from "./faaTfrArchive";

export const FAA_TFR_LIST_URL = "https://tfr.faa.gov/tfrapi/exportTfrList";
export const faaTfrDetailUrl = (noticeId: string) => `https://tfr.faa.gov/download/detail_${noticeId.replace("/", "_")}.xml`;
export const FAA_TFR_TICK_MS = 30 * 60_000;
export const FAA_TFR_FIRST_TICK_MS = 90_000;
/** between two detail-file requests */
export const FAA_TFR_DETAIL_SPACING_MS = 1_500;
/** detail files asked for in one pass; the rest wait for the next pass */
export const FAA_TFR_MAX_DETAILS_PER_PASS = 80;
export const FAA_TFR_TIMEOUT_MS = 20_000;
/** a body over this is not parsed (largest detail file seen 2026-11-22: 199 kB) */
export const FAA_TFR_MAX_BODY_BYTES = 2 * 1024 * 1024;
/** the FAA lists 50 to 70; a list longer than this is cut and the cut is counted */
export const FAA_TFR_MAX_LISTED = 500;
export const FAA_TFR_MAX_AREAS = 32;
/** corners kept per outline; a longer one is thinned evenly (the client's own cap) */
export const FAA_TFR_MAX_RING_POINTS = 512;
/** a list read older than this is no longer served */
export const FAA_TFR_STALE_MS = 24 * 3600_000;
export const FAA_TFR_KEEP_VERSION = 1;
export const FAA_TFR_KEEP_MAX_BYTES = 8 * 1024 * 1024;
const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };

/** On where the FAA notice feed is configured (the layer these rows join),
 *  unless VOLTRADE_FAA_TFR_LIST=off|0|false|no. A process with no feed (a
 *  test, a local run) makes no request. */
export function faaTfrListEnabled(env: Record<string, string | undefined> = process.env): boolean {
  const v = String(env.VOLTRADE_FAA_TFR_LIST ?? "").trim().toLowerCase();
  if (v === "off" || v === "0" || v === "false" || v === "no") return false;
  return swimMissingEnv("SWIM_NOTAM", env as NodeJS.ProcessEnv).length === 0;
}

/** one height limit as the FAA writes it: 4000 FT ALT (above mean sea level), 3000 FT HEI (above the ground), 430 FL */
export interface FaaLimit { value: number; uom: "FT" | "FL"; ref: "ALT" | "HEI" | "STD" | null }
export interface FaaArea { name: string | null; lower: FaaLimit | null; upper: FaaLimit | null; scheduled: boolean }
export interface FaaListEntry { notice: string; type: string | null; facility: string | null; state: string | null; description: string | null }

export interface FaaTfrRow {
  /** "FAA:6/9306" */
  id: string;
  source: "faa_list";
  /** "FDC 6/9306" */
  citation: string;
  classification: "FDC";
  series: null; number: string; year: null; type: null; qcode: null; scenario: null;
  /** the air traffic facility the FAA files it under ("ZDC") */
  location: string | null;
  issued: number | null; start: number | null; end: number | null;
  minFl: null; maxFl: null; center: null; radiusNm: null;
  text: string | null;
  geometry: { kind: "polygon"; rings: Array<Array<[number, number]>>; from: "faa_list" } | null;
  faa: {
    /** the FAA list's own type word ("SECURITY", "HAZARDS", "VIP", ...) */
    type: string | null;
    /** the rule the detail file names ("91.137(a)(2)", "99.7") */
    rule: string | null;
    place: string | null;
    /** one per area the detail file defines, in its order; an area without an outline is here too */
    areas: FaaArea[];
    /** areas the file defines that carried no usable outline */
    areas_without_outline: number;
  };
}

const NOTICE_RE = /^\d\/\d{4}$/;
const clean = (v: unknown, max = 200): string | null => (typeof v === "string" && v.trim() ? v.trim().slice(0, max) : null);

/** The FAA's list answer -> entries, each notice once, malformed rows counted. Pure. */
export function parseFaaList(raw: unknown): { entries: FaaListEntry[]; malformed: number; cut: number } {
  const out: FaaListEntry[] = [];
  const seen = new Set<string>();
  let malformed = 0;
  for (const v of Array.isArray(raw) ? raw : []) {
    const o = v as Record<string, unknown> | null;
    const notice = clean(o?.notam_id, 12);
    if (!o || !notice || !NOTICE_RE.test(notice)) { malformed++; continue; }
    if (seen.has(notice)) continue;
    seen.add(notice);
    out.push({ notice, type: clean(o.type, 60), facility: clean(o.facility, 8), state: clean(o.state, 8), description: clean(o.description, 300) });
  }
  const cut = Math.max(0, out.length - FAA_TFR_MAX_LISTED);
  if (cut) out.length = FAA_TFR_MAX_LISTED;
  return { entries: out, malformed, cut };
}

const unescapeXml = (s: string) => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&amp;/g, "&");
const tag = (xml: string, name: string): string | null => {
  const m = new RegExp(`<${name}>([\\s\\S]*?)</${name}>`).exec(xml);
  return m && m[1].trim() ? unescapeXml(m[1].trim()) : null;
};
/** "2026-11-22T12:00:00" (UTC, as the detail file writes it) -> ms */
const isoUtc = (s: string | null): number | null => {
  if (!s || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$/.test(s)) return null;
  const t = Date.parse(`${s}Z`);
  return Number.isFinite(t) ? t : null;
};
/** "2611221200" -> ms (UTC; the notice's own effective group) */
const group10 = (s: string): number | null => {
  const t = Date.UTC(2000 + +s.slice(0, 2), +s.slice(2, 4) - 1, +s.slice(4, 6), +s.slice(6, 8), +s.slice(8, 10));
  return Number.isFinite(t) && +s.slice(2, 4) >= 1 && +s.slice(2, 4) <= 12 && +s.slice(4, 6) >= 1 && +s.slice(4, 6) <= 31 && +s.slice(6, 8) <= 23 && +s.slice(8, 10) <= 59 ? t : null;
};
/** "37.86666667N" / "079.875W" -> signed degrees; null when not that form or out of range */
export function faaCoord(s: string | null, kind: "lat" | "lon"): number | null {
  const m = /^(\d{1,3}(?:\.\d+)?)([NSEW])$/.exec((s ?? "").trim());
  if (!m) return null;
  const isLat = m[2] === "N" || m[2] === "S";
  if (isLat !== (kind === "lat")) return null;
  const v = Number(m[1]) * (m[2] === "S" || m[2] === "W" ? -1 : 1);
  return Math.abs(v) <= (kind === "lat" ? 90 : 180) ? v : null;
}
const nodeText = (n: XNode | null | undefined, name: string): string | null => {
  const t = findFirst(n, name)?.text?.trim();
  return t ? t : null;
};
function limitOf(area: XNode, side: "Upper" | "Lower"): FaaLimit | null {
  const v = Number(nodeText(area, `valDistVer${side}`) ?? "");
  const uom = nodeText(area, `uomDistVer${side}`);
  const ref = nodeText(area, `codeDistVer${side}`);
  if (!Number.isFinite(v) || v < 0 || (uom !== "FT" && uom !== "FL")) return null;
  return { value: v, uom, ref: ref === "ALT" || ref === "HEI" || ref === "STD" ? ref : null };
}
/** the FAA's merged outline of one area -> closed ring [lat, lon]; null when any corner is not a plain point */
function mergedRing(merged: XNode | null): Array<[number, number]> | null {
  if (!merged) return null;
  let ring: Array<[number, number]> = [];
  for (const avx of merged.children) {
    if (avx.name !== "Avx") continue;
    // GRC = a corner joined to the next by a great-circle line. The merged
    // outline is written in these only; any other code would be an arc or
    // circle definition this reader does not draw.
    if (nodeText(avx, "codeType") !== "GRC") return null;
    const la = faaCoord(nodeText(avx, "geoLat"), "lat"), lo = faaCoord(nodeText(avx, "geoLong"), "lon");
    if (la === null || lo === null) return null;
    ring.push([Math.round(la * 1e6) / 1e6, Math.round(lo * 1e6) / 1e6]);
  }
  if (ring.length > 1 && ring[0][0] === ring[ring.length - 1][0] && ring[0][1] === ring[ring.length - 1][1]) ring.pop();
  if (ring.length < 3) return null;
  if (ring.length > FAA_TFR_MAX_RING_POINTS) {
    const step = ring.length / FAA_TFR_MAX_RING_POINTS, src = ring;
    ring = Array.from({ length: FAA_TFR_MAX_RING_POINTS }, (_, i) => src[Math.floor(i * step)]);
  }
  ring.push(ring[0]);
  return ring;
}

/** One FAA detail file -> the row for that notice; null when the file is not a notice. Pure. */
export function parseFaaDetail(xmlIn: string, entry: FaaListEntry): FaaTfrRow | null {
  if (typeof xmlIn !== "string" || xmlIn.length > FAA_TFR_MAX_BODY_BYTES) return null;
  const xml = xmlIn.replace(/^﻿/, "");
  // the file must be the notice asked for: a wrong or empty file is not a row.
  // The number is the last digit of the index year and the sequence number
  // (txtLocalName is a free label: "6/9306" on one file, "261121-22
  // Middleburg VIP" on another; read 2026-11-22).
  const year = tag(xml, "dateIndexYear"), seq = tag(xml, "noSeqNo");
  if (!year || !seq || !/^\d{4}$/.test(year) || !/^\d{1,4}$/.test(seq) || `${year.slice(3)}/${seq.padStart(4, "0")}` !== entry.notice) return null;
  const head = xml.slice(0, Math.max(0, xml.indexOf("<TfrNot>")) || xml.length);
  const text = tag(xml, "txtDescrUSNS");
  // the notice's own effective group is UTC by rule; the file's fields are the fallback
  const grp = /\b(\d{10})-(\d{10}|PERM)\b/.exec(text ?? "");
  const start = (grp ? group10(grp[1]) : null) ?? isoUtc(tag(head, "dateEffective"));
  const end = grp ? (grp[2] === "PERM" ? null : group10(grp[2])) : isoUtc(tag(head, "dateExpire"));
  const doc = parseXmlLite(xml);
  const areas: FaaArea[] = [];
  const rings: Array<Array<[number, number]>> = [];
  let without = 0;
  for (const g of findAll(doc, "TFRAreaGroup")) {
    if (areas.length >= FAA_TFR_MAX_AREAS) { without++; continue; }
    const a = findFirst(g, "aseTFRArea");
    if (!a) continue;
    areas.push({ name: clean(nodeText(a, "txtName"), 60), lower: limitOf(a, "Lower"), upper: limitOf(a, "Upper"), scheduled: nodeText(a, "isScheduledTfrArea") === "TRUE" });
    const ring = mergedRing(findFirst(g, "abdMergedArea"));
    if (ring) rings.push(ring); else without++;
  }
  const city = tag(head, "txtNameCity"), state = tag(head, "txtNameUSState");
  const tfrNot = /<TfrNot>\s*<codeType>([^<]{1,40})<\/codeType>/.exec(xml);
  return {
    id: `FAA:${entry.notice}`, source: "faa_list", citation: `FDC ${entry.notice}`, classification: "FDC",
    series: null, number: entry.notice, year: null, type: null, qcode: null, scenario: null,
    location: entry.facility ?? clean(tag(head, "codeFacility"), 8),
    issued: isoUtc(tag(head, "dateIssued")), start, end,
    minFl: null, maxFl: null, center: null, radiusNm: null,
    text: text ? text.slice(0, 4000) : null,
    geometry: rings.length ? { kind: "polygon", rings, from: "faa_list" } : null,
    faa: {
      type: entry.type, rule: tfrNot ? tfrNot[1].trim() : null,
      place: [city, state].filter(Boolean).join(", ").slice(0, 120) || entry.description?.split(",")[0] || null,
      areas, areas_without_outline: without,
    },
  };
}

/** why the last pass did not read the list; a closed list, never free text */
export type FaaListCause = "ok" | "http_error" | "timeout" | "network" | "not_a_list" | "too_large";
export type FaaListStatus = "off" | "never" | "ok" | "stale" | "unreachable";

export interface FaaCoverage {
  status: FaaListStatus;
  /** when the FAA's list was last read; null = never by this server */
  read_at: number | null;
  last_attempt_at: number | null;
  last_cause: FaaListCause | null;
  /** notices on the FAA's list at that read */
  listed: number;
  /** of those, held here with the FAA's outline */
  with_outline: number;
  /** of those, read and found to carry no outline in the FAA's file: listed, never drawn */
  without_outline: number;
  /** of those, detail file not yet read (asked next pass) */
  detail_pending: number;
  /** of the listed notices, how many the notice feed's own store also holds (by citation) */
  also_in_feed: number | null;
  /** listed notices without an outline, as the FAA names them (at most 40) */
  not_drawn: Array<{ citation: string; type: string | null; description: string | null }>;
  restored: boolean;
}

interface KeepDoc { v: number; read_at: number; entries: FaaListEntry[]; rows: FaaTfrRow[] }

/** The list as held: entries from the last read, one parsed row per notice ever read while listed. */
export class FaaTfrListStore {
  private entries: FaaListEntry[] = [];
  private readonly rows = new Map<string, FaaTfrRow>();
  readAt: number | null = null;
  lastAttemptAt: number | null = null;
  lastCause: FaaListCause | null = null;
  restored = false;
  passes = 0;
  detailFailures = 0;
  /** notices the last list answer held beyond FAA_TFR_MAX_LISTED (0 on every read so far) */
  lastListCut = 0;

  /** a successful list read: rows of notices no longer listed are dropped (the FAA ended or cancelled them) */
  setList(entries: FaaListEntry[], now: number): void {
    this.entries = entries;
    this.readAt = now;
    const listed = new Set(entries.map((e) => e.notice));
    for (const k of Array.from(this.rows.keys())) if (!listed.has(k)) this.rows.delete(k);
  }
  setRow(row: FaaTfrRow): void { this.rows.set(row.number, row); }
  /** the notices of the last list read, in the FAA's order */
  listed(): readonly FaaListEntry[] { return this.entries; }
  /** the detail row read for a listed notice ("6/9306"), with or without an outline */
  rowOf(notice: string): FaaTfrRow | undefined { return this.rows.get(notice); }
  /** every listed notice's row that carries the FAA's outline, in force or not; none once the read is stale */
  outlined(now: number): FaaTfrRow[] {
    if (!this.fresh(now)) return [];
    return this.entries.map((e) => this.rows.get(e.notice)).filter((r): r is FaaTfrRow => !!r && !!r.geometry);
  }
  /** listed notices whose detail file has not been read */
  pending(): FaaListEntry[] { return this.entries.filter((e) => !this.rows.has(e.notice)); }
  fresh(now: number): boolean { return this.readAt !== null && now - this.readAt <= FAA_TFR_STALE_MS; }

  /** Rows with an outline that are in force or start within `aheadMs`. None once the read is stale. */
  active(now: number, aheadMs = 6 * 3600_000): FaaTfrRow[] {
    if (!this.fresh(now)) return [];
    return this.entries.map((e) => this.rows.get(e.notice))
      .filter((r): r is FaaTfrRow => !!r && !!r.geometry && (r.end === null || r.end >= now) && (r.start === null || r.start <= now + aheadMs))
      .sort((a, b) => (a.start ?? 0) - (b.start ?? 0));
  }
  /** every citation the list holds a drawable row for, in force or not (the route's de-duplication key) */
  citations(now: number): Set<string> {
    const out = new Set<string>();
    if (this.fresh(now)) for (const e of this.entries) { const r = this.rows.get(e.notice); if (r?.geometry) out.add(r.citation); }
    return out;
  }

  coverage(now: number, enabled: boolean, feedCitations: ReadonlySet<string> | null): FaaCoverage {
    let withO = 0, withoutO = 0, pending = 0, inFeed = 0;
    const notDrawn: FaaCoverage["not_drawn"] = [];
    for (const e of this.entries) {
      const r = this.rows.get(e.notice);
      if (feedCitations?.has(`FDC ${e.notice}`)) inFeed++;
      if (!r) pending++;
      else if (r.geometry) withO++;
      else { withoutO++; if (notDrawn.length < 40) notDrawn.push({ citation: r.citation, type: e.type, description: e.description }); }
    }
    const status: FaaListStatus = !enabled ? "off" : this.readAt === null ? (this.lastAttemptAt === null ? "never" : "unreachable") : this.fresh(now) ? "ok" : "stale";
    return {
      status, read_at: this.readAt, last_attempt_at: this.lastAttemptAt, last_cause: this.lastCause,
      listed: this.entries.length, with_outline: withO, without_outline: withoutO, detail_pending: pending,
      also_in_feed: feedCitations ? inFeed : null, not_drawn: notDrawn, restored: this.restored,
    };
  }

  snapshot(): KeepDoc | null {
    return this.readAt === null ? null : { v: FAA_TFR_KEEP_VERSION, read_at: this.readAt, entries: this.entries, rows: Array.from(this.rows.values()) };
  }
  /** Put a saved read back (boot). Malformed input restores nothing. Returns the rows put back. */
  restore(raw: unknown, now: number): number {
    const d = raw as KeepDoc | null;
    if (!d || d.v !== FAA_TFR_KEEP_VERSION || !Number.isFinite(d.read_at) || d.read_at > now + 60_000 || !Array.isArray(d.entries) || !Array.isArray(d.rows)) return 0;
    const { entries } = parseFaaList(d.entries.map((e) => ({ notam_id: e?.notice, type: e?.type, facility: e?.facility, state: e?.state, description: e?.description })));
    if (!entries.length) return 0;
    this.entries = entries;
    this.readAt = d.read_at;
    const listed = new Set(entries.map((e) => e.notice));
    let n = 0;
    for (const r of d.rows) {
      if (!r || typeof r.number !== "string" || !listed.has(r.number) || r.source !== "faa_list" || r.id !== `FAA:${r.number}`) continue;
      if (r.geometry && (r.geometry.kind !== "polygon" || !Array.isArray(r.geometry.rings) || !r.geometry.rings.length)) continue;
      if ([r.start, r.end, r.issued].some((v) => v != null && !Number.isFinite(v))) continue;
      this.rows.set(r.number, r);
      n++;
    }
    this.restored = true;
    return n;
  }
}

export interface FaaFetch { (url: string): Promise<{ ok: boolean; status: number; text: string }> }
const realFetch: FaaFetch = async (url) => {
  const res = await fetch(url, { headers: UA, signal: AbortSignal.timeout(FAA_TFR_TIMEOUT_MS) });
  const len = Number(res.headers.get("content-length") ?? "0");
  if (len > FAA_TFR_MAX_BODY_BYTES) return { ok: false, status: 413, text: "" };
  return { ok: res.ok, status: res.status, text: res.ok ? await res.text() : "" };
};
const causeOf = (e: unknown): FaaListCause => ((e as { name?: string } | null)?.name === "TimeoutError" || (e as { name?: string } | null)?.name === "AbortError" ? "timeout" : "network");

export interface FaaPassResult { cause: FaaListCause; listed: number; details_read: number; details_failed: number; changed: boolean; cause_of_last_detail_failure?: FaaListCause }

/** One pass: the list, then the detail file of each listed notice not yet read. Never throws. */
export async function runFaaTfrPass(store: FaaTfrListStore, o: { fetch?: FaaFetch; now?: () => number; sleep?: (ms: number) => Promise<void> } = {}): Promise<FaaPassResult> {
  const get = o.fetch ?? realFetch, clock = o.now ?? Date.now;
  const sleep = o.sleep ?? ((ms: number) => new Promise<void>((r) => { setTimeout(r, ms).unref?.(); }));
  const res: FaaPassResult = { cause: "ok", listed: 0, details_read: 0, details_failed: 0, changed: false };
  store.passes++;
  store.lastAttemptAt = clock();
  try {
    const r = await get(FAA_TFR_LIST_URL);
    if (!r.ok) res.cause = r.status === 413 ? "too_large" : "http_error";
    else if (r.text.length > FAA_TFR_MAX_BODY_BYTES) res.cause = "too_large";
    else {
      let parsed: unknown = null;
      try { parsed = JSON.parse(r.text.replace(/^﻿/, "")); } catch { res.cause = "not_a_list"; }
      // an answer that is not an array is not "no restrictions": the FAA always lists the standing ones
      if (res.cause === "ok" && !Array.isArray(parsed)) res.cause = "not_a_list";
      if (res.cause === "ok") {
        const { entries, cut } = parseFaaList(parsed);
        if (!entries.length) res.cause = "not_a_list";
        else { store.setList(entries, clock()); store.lastListCut = cut; res.listed = entries.length; res.changed = true; }
      }
    }
  } catch (e) { res.cause = causeOf(e); }
  store.lastCause = res.cause;
  if (res.cause !== "ok") return res;
  const todo = store.pending().slice(0, FAA_TFR_MAX_DETAILS_PER_PASS);
  for (let i = 0; i < todo.length; i++) {
    if (i) await sleep(FAA_TFR_DETAIL_SPACING_MS);
    let row: FaaTfrRow | null = null;
    try {
      const d = await get(faaTfrDetailUrl(todo[i].notice));
      if (d.ok) row = parseFaaDetail(d.text, todo[i]);
    } catch (e) { res.cause_of_last_detail_failure = causeOf(e); }
    if (row) { store.setRow(row); res.details_read++; } else { res.details_failed++; store.detailFailures++; }
  }
  return res;
}

// ── the one process-wide store, its keep-file and its timer ────────────────
let shared: FaaTfrListStore | null = null;
export function sharedFaaTfrList(): FaaTfrListStore { return (shared ??= new FaaTfrListStore()); }
/** beside the feed stores' keep-file: a cache of live state, replaced after a read, never history */
export function faaTfrKeepFile(baseDir?: string): string { return path.join(baseDir ?? path.dirname(archiveBaseDir()), "faa_tfr_list.json"); }

export function restoreFaaTfrList(store: FaaTfrListStore, file = faaTfrKeepFile(), now = Date.now()): number {
  if (!fs.existsSync(file)) return 0;
  try {
    return store.restore(JSON.parse(fs.readFileSync(file, "utf8")), now);
  } catch (e) {
    console.error(`[faaTfrList] saved list not read back: ${(e as NodeJS.ErrnoException)?.code ?? "unparseable"}`);
    return 0;
  }
}
export async function saveFaaTfrList(store: FaaTfrListStore, file = faaTfrKeepFile()): Promise<boolean> {
  const doc = store.snapshot();
  if (!doc) return false;
  const body = JSON.stringify(doc);
  if (body.length > FAA_TFR_KEEP_MAX_BYTES) return false;
  try {
    const tmp = `${file}.${process.pid}.tmp`;
    await fs.promises.mkdir(path.dirname(file), { recursive: true });
    await fs.promises.writeFile(tmp, body);
    await fs.promises.rename(tmp, file);
    return true;
  } catch (e) {
    console.error(`[faaTfrList] save failed: ${(e as NodeJS.ErrnoException)?.code ?? "error"}`);
    return false;
  }
}

let timers: Array<ReturnType<typeof setTimeout>> = [];
let running = false;
/** Start the reads: the saved list is put back now, the first pass runs 90 s after boot, then every 30 minutes. */
export function bootFaaTfrList(env: NodeJS.ProcessEnv = process.env, o: { file?: string; fetch?: FaaFetch; archiveBaseDir?: string; afterPass?: () => string | null | Promise<string | null> } = {}): boolean {
  if (timers.length || !faaTfrListEnabled(env)) return false;
  const store = sharedFaaTfrList();
  const file = o.file ?? faaTfrKeepFile();
  const back = restoreFaaTfrList(store, file);
  if (back) console.log(`[faaTfrList] ${back} listed restrictions put back from the last read`);
  // the record of what appeared on and left the list (faaTfrArchive.ts)
  const recording = faaTfrArchiveEnabled(env);
  if (recording) void loadFaaTfrArchive(o.archiveBaseDir);
  const tick = async () => {
    if (running) return;
    running = true;
    beat("faa_tfr_list", "start");
    try {
      const r = await runFaaTfrPass(store, { fetch: o.fetch });
      if (r.changed) await saveFaaTfrList(store, file);
      // only a list read that succeeded is recorded: an unreachable FAA is a gap, not an ending
      const rec = recording && r.cause === "ok" && store.readAt !== null
        ? await recordFaaTfrRead({ entries: store.listed(), rowOf: (n) => store.rowOf(n), now: store.readAt, cut: store.lastListCut, baseDir: o.archiveBaseDir })
        : null;
      const recBad = !!rec && rec.status !== "recorded";
      // what the caller does with a fresh list (the text reader's running check, swimCapture.ts); its
      // failure is said on the job's line and never fails the list read
      let after: string | null = null;
      if (o.afterPass && r.cause === "ok") {
        try { after = await o.afterPass(); } catch (e) { after = `after-pass step failed: ${e instanceof Error ? e.message : String(e)}`; }
      }
      const detail = `${r.cause}: ${r.listed} listed, ${r.details_read} detail files read, ${r.details_failed} failed`
        + (rec ? `; record ${rec.status}${rec.listed || rec.left ? ` (${rec.listed} appeared, ${rec.left} left)` : ""}` : "")
        + (after ? `; ${after}` : "");
      beat("faa_tfr_list", "finish", r.cause === "ok" && !recBad ? "ok" : "error", detail);
      if (r.cause !== "ok" || r.details_failed || recBad) console.error(`[faaTfrList] pass ${detail}`);
    } catch (e) {
      beat("faa_tfr_list", "finish", "error", e instanceof Error ? e.message : String(e));
    } finally {
      running = false;
    }
  };
  const first = setTimeout(() => { void tick(); }, FAA_TFR_FIRST_TICK_MS);
  first.unref?.();
  const every = setInterval(() => { void tick(); }, FAA_TFR_TICK_MS);
  every.unref?.();
  timers = [first, every];
  return true;
}
export function _resetFaaTfrListForTests(): void {
  for (const t of timers) { clearTimeout(t); clearInterval(t); }
  timers = []; shared = null; running = false;
}
