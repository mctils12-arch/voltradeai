// ATC FREQUENCY rows on the aircraft card (2026-10-14, human request: "is
// there a way to get the ATC frequencies, the ARTCC, or anything"). The data
// rides the runway-status response (server/atcFrequencies.ts via
// /api/data/procedures/runway): the PUBLISHED FAA NASR frequencies that apply
// to the plane's position, altitude and inferred phase — tower for the runway
// it is on, ground/clearance/ATIS at the gate, the approach/departure sector
// containing it, or its ARTCC and nearest center transmitters. Which frequency
// the crew is actually tuned to is not public; the rows say so (tag + title).

import { fmtMeters, getUnits, type UnitSystem } from "@/lib/units";

export interface AtcEntryData {
  role: string;
  call: string;
  mhz: number;
  note: string | null;
  likely: boolean;
}

export interface AtcContextData {
  phase: "RUNWAY" | "SURFACE" | "TERMINAL" | "ENROUTE" | null;
  airport: string | null;
  artcc: { id: string; name: string; stratum: "LOW" | "HIGH" | "OCEANIC" } | null;
  entries: AtcEntryData[];
  cycle: string | null;
  honesty: string;
}

/** most frequencies shown on the card; geometry-singled-out entries first */
export const ATC_MAX_ROWS = 4;
const FT_TO_M = 0.3048;

/** Pure: "118.1", "128.325", "120.05" (as published; trailing zeros past one decimal dropped). */
export function fmtMhz(mhz: number): string {
  return mhz.toFixed(3).replace(/(\.\d)0+$|(\.\d*[1-9])0+$/, "$1$2");
}

const title = (s: string) => s.toLowerCase().replace(/\b([a-z])/g, (m) => m.toUpperCase());

/** Pure: the published sector text with its altitudes in the user's units
 *  ("8000 FT & BLW" -> "8000 ft & BLW" | "2438 m & BLW"). */
export function noteInUnits(note: string, system: UnitSystem = getUnits()): string {
  return note.replace(/(\d{3,5}) FT\b/g, (_m, n: string) => fmtMeters(Number(n) * FT_TO_M, 0, system));
}

/** Pure: the ARTCC line ("Atlanta Center (ZTL) · high"), or null. */
export function artccLine(atc: AtcContextData | null | undefined): string | null {
  const a = atc?.artcc;
  if (!a) return null;
  return a.stratum === "OCEANIC" ? `${title(a.name)} (${a.id})` : `${title(a.name)} Center (${a.id}) · ${a.stratum.toLowerCase()}`;
}

/** Pure: the rows to show, likely ones first, capped. */
export function atcRows(atc: AtcContextData | null | undefined, system: UnitSystem = getUnits()): Array<{ text: string; likely: boolean; key: string }> {
  if (!atc?.entries?.length) return [];
  const ordered = [...atc.entries.filter((e) => e.likely), ...atc.entries.filter((e) => !e.likely)].slice(0, ATC_MAX_ROWS);
  return ordered.map((e) => ({
    key: `${e.role}-${e.mhz}-${e.note ?? ""}`,
    likely: e.likely,
    text: `${e.call} ${fmtMhz(e.mhz)}${e.note ? ` · ${noteInUnits(e.note, system)}` : ""}`,
  }));
}

export function AtcFrequencyRows({ atc, airport }: { atc: AtcContextData | null | undefined; airport?: string | null }): JSX.Element | null {
  const rows = atcRows(atc);
  const center = artccLine(atc);
  if (!rows.length && !center) return null;
  const listenAt = atc?.airport ?? airport ?? null;
  return (
    <div className="vt-atc" data-vt-atc title={atc?.honesty}>
      <p className="vt-atc-head">
        <span className="vt-atc-label">ATC</span>
        {center && <span className="vt-atc-center">{center}</span>}
        <span className="vt-runway-tag">published · inferred</span>
      </p>
      {rows.map((r) => (
        <p key={r.key} className={`vt-atc-row${r.likely ? " vt-atc-row-likely" : ""}`}>{r.text}</p>
      ))}
      {listenAt && /^[A-Z0-9]{3,4}$/.test(listenAt) && (
        <a className="vt-atc-listen" href={`https://www.liveatc.net/search/?icao=${listenAt}`} target="_blank" rel="noopener noreferrer">
          Search LiveATC for {listenAt} feeds
        </a>
      )}
    </div>
  );
}
