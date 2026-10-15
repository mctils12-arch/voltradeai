// swimNotam.ts — FAA SWIM NOTAM (FNS digital NOTAM, AIXM 5.1) parser, ops
// item 1: TFRs with geometry for the deviation classifier. Field names come
// from REAL messages captured on our queue 2026-10-14 (capture PR #1342):
//   AIXMBasicMessage > hasMember > Event > timeSlice > EventTimeSlice >
//   textNOTAM > NOTAM { series, number, year, type, issued, affectedFIR,
//   selectionCode (Q-code), minimumFL, maximumFL, coordinates (DDMMNDDDMMW),
//   radius (NM), location, effectiveStart/End (YYYYMMDDHHMM), text }
//   + extension EventExtension { classification (DOM/INTL/FDC/…),
//   icaoLocation, xovernotamID }.
//
// TFR = an FDC NOTAM whose text cites the TFR rules (14 CFR 91.137–91.145,
// 99.7) or says FLIGHT RESTRICTION(S), or an ICAO Q-code QRT… (temporary
// restricted area). GEOMETRY, honestly labeled: an exact polygon when the
// message carries airspace geometry (gml:posList inside an Airspace member);
// otherwise the NOTAM's own Q-line circle (center + radius), labeled
// "approximate" — the Q-line radius encloses the area, it is not the area.
// No TFR has reached the sample buffer yet; capture now keeps TFR samples in
// their own bucket so the first real one verifies this parser.

import { parseXmlLite, findAll, findFirst, type XNode } from "./swimSfdps";

export const NOTAM_MAX_TFRS = 2_000;
/** a TFR with no end time is kept this long after issue */
export const NOTAM_TFR_DEFAULT_TTL_MS = 7 * 86_400_000;

export const TFR_TEXT_RE = /FLIGHT RESTRICTION|\b91\.1(?:37|38|39|41|43|45)\b|\b99\.7\b|\bTFR\b/i;

export interface NotamRecord {
  id: string;
  series: string | null;
  number: string | null;
  year: string | null;
  type: string | null;
  classification: string | null;
  location: string | null;
  qcode: string | null;
  issued: number | null;
  start: number | null;
  end: number | null;
  minFl: number | null;
  maxFl: number | null;
  center: [number, number] | null;
  radiusNm: number | null;
  text: string | null;
  scenario: string | null;
}

export interface TfrRecord extends NotamRecord {
  geometry: { kind: "polygon"; rings: Array<Array<[number, number]>> } | { kind: "circle"; center: [number, number]; radiusNm: number; approximate: true } | null;
}

const txt = (n: XNode | null | undefined): string | null => {
  const t = n?.text?.trim();
  return t ? t : null;
};
const first = (n: XNode | null | undefined, name: string) => txt(findFirst(n, name));
const num = (s: string | null) => (s != null && s !== "" && Number.isFinite(Number(s)) ? Number(s) : null);

/** "202610151530" → ms (UTC); "PERM"/garbage → null */
export function notamTime(s: string | null): number | null {
  const m = /^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})/.exec(s ?? "");
  if (!m) return null;
  const t = Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]);
  return Number.isFinite(t) ? t : null;
}

/** "3841N12135W" → [38.683, -121.583] (degrees + minutes) */
export function qlineCenter(s: string | null): [number, number] | null {
  const m = /^(\d{2})(\d{2})([NS])(\d{3})(\d{2})([EW])$/.exec((s ?? "").trim());
  if (!m) return null;
  const la = (+m[1] + +m[2] / 60) * (m[3] === "S" ? -1 : 1);
  const lo = (+m[4] + +m[5] / 60) * (m[6] === "W" ? -1 : 1);
  return Math.abs(la) <= 90 && Math.abs(lo) <= 180 ? [Math.round(la * 1e4) / 1e4, Math.round(lo * 1e4) / 1e4] : null;
}

export function isTfr(n: Pick<NotamRecord, "classification" | "qcode" | "text">): boolean {
  if (n.qcode && /^QRT/i.test(n.qcode)) return true;
  return (n.classification ?? "").toUpperCase() === "FDC" && TFR_TEXT_RE.test(n.text ?? "");
}

/** EPSG:4326 posList "lat lon lat lon …" → ring (needs ≥ 4 points) */
function ringOf(pl: string | null): Array<[number, number]> | null {
  const v = (pl ?? "").trim().split(/\s+/).map(Number);
  if (v.length < 8 || v.length % 2 || v.some((x) => !Number.isFinite(x))) return null;
  const ring: Array<[number, number]> = [];
  for (let i = 0; i < v.length; i += 2) {
    if (Math.abs(v[i]) > 90 || Math.abs(v[i + 1]) > 180) return null;
    ring.push([v[i], v[i + 1]]);
  }
  return ring;
}

/** Pure: one FNS payload → its text NOTAMs (TFRs carry geometry). */
export function parseNotamPayload(xml: string): TfrRecord[] {
  if (typeof xml !== "string" || xml.indexOf("textNOTAM") < 0) return [];
  const doc = parseXmlLite(xml);
  if (!doc) return [];
  // exact geometry only from airspace members (runway/taxiway posLists are
  // the affected feature's own shape, never a restriction area)
  const rings: Array<Array<[number, number]>> = [];
  for (const as of findAll(doc, "Airspace")) {
    for (const pl of findAll(as, "posList")) { const r = ringOf(txt(pl)); if (r) rings.push(r); }
  }
  const out: TfrRecord[] = [];
  for (const ets of findAll(doc, "EventTimeSlice")) {
    const n = findFirst(ets, "NOTAM");
    if (!n) continue;
    const ext = findFirst(ets, "EventExtension");
    const rec: NotamRecord = {
      id: "", series: first(n, "series"), number: first(n, "number"), year: first(n, "year"), type: first(n, "type"),
      classification: first(ext, "classification"), location: first(ext, "icaoLocation") ?? first(n, "location"),
      qcode: first(n, "selectionCode"), issued: Date.parse(first(n, "issued") ?? "") || null,
      start: notamTime(first(n, "effectiveStart")), end: notamTime(first(n, "effectiveEnd")),
      minFl: num(first(n, "minimumFL")), maxFl: num(first(n, "maximumFL")),
      center: qlineCenter(first(n, "coordinates")), radiusNm: num(first(n, "radius")),
      text: (first(n, "text") ?? "").slice(0, 4000) || null, scenario: first(ets, "scenario"),
    };
    // FNS numbers are per location: the location keeps "20/2026" at KCNY and
    // "20/2026" elsewhere apart when no cross-over id is sent
    rec.id = first(ext, "xovernotamID")
      ? `${rec.classification ?? ""}:${first(ext, "xovernotamID")}`
      : `${rec.location ?? ""}:${rec.series ?? ""}${rec.number ?? ""}/${rec.year ?? ""}`;
    const tfr = isTfr(rec);
    out.push({
      ...rec,
      geometry: !tfr ? null
        : rings.length ? { kind: "polygon", rings }
        : rec.center && rec.radiusNm != null ? { kind: "circle", center: rec.center, radiusNm: rec.radiusNm, approximate: true }
        : null,
    });
  }
  return out;
}

/** Active TFRs for the deviation classifier + counts of everything else. */
export class NotamStore {
  private readonly tfrs = new Map<string, TfrRecord>();
  readonly byClass: Record<string, number> = {};
  notams = 0;
  parseErrors = 0;

  ingest(xml: string, now: number = Date.now()): void {
    let recs: TfrRecord[];
    try { recs = parseNotamPayload(xml); } catch { this.parseErrors++; return; }
    for (const r of recs) {
      this.notams++;
      const k = r.classification ?? "UNKNOWN";
      this.byClass[k] = (this.byClass[k] || 0) + 1;
      if (!r.geometry) continue;
      if ((r.type ?? "").toUpperCase() === "C") { this.tfrs.delete(r.id); continue; } // cancellation
      this.tfrs.set(r.id, r);
    }
    if (this.tfrs.size > NOTAM_MAX_TFRS) this.evict(now);
  }

  private endOf(t: TfrRecord): number {
    return t.end ?? (t.issued ?? t.start ?? 0) + NOTAM_TFR_DEFAULT_TTL_MS;
  }

  evict(now: number = Date.now()): void {
    for (const [k, t] of Array.from(this.tfrs)) if (this.endOf(t) < now) this.tfrs.delete(k);
  }

  /** TFRs in force now or starting within `aheadMs` */
  active(now: number = Date.now(), aheadMs = 6 * 3600_000): TfrRecord[] {
    return Array.from(this.tfrs.values())
      .filter((t) => this.endOf(t) >= now && (t.start == null || t.start <= now + aheadMs))
      .sort((a, b) => (a.start ?? 0) - (b.start ?? 0));
  }

  stats() {
    return { notams: this.notams, tfrs: this.tfrs.size, byClassification: { ...this.byClass }, parseErrors: this.parseErrors };
  }
}
