// RUNWAY STATUS row on the aircraft card (2026-10-14, human request: "know
// what runway they are landing on based on their heading and position ...
// activate the plane for a takeoff when it's on the runway"). Asks
// /api/data/procedures/runway (server/runwayDetect.ts) with the selected
// plane's live fix while its card is open: "Lined up on runway 36", "Taking
// off — runway 36", "On final — runway 19L", "Departed runway 36", or the
// trip's departure runway. Every line is an INFERENCE from ADS-B position +
// track against FAA CIFP runway geometry, and says so — never an ATC
// assignment. Zero cost when no aircraft card is open (no fetch, no timer).
// The same response carries the published ATC frequencies for the fix
// (`atcRow`, components/AtcFrequencies.tsx) — one poll serves both rows.

import { useEffect, useState } from "react";
import { HEX_RE } from "@/lib/air/flightPlan";
import { pollWhenVisible, timeoutSignal } from "@/lib/tabResume";
import { AtcFrequencyRows, type AtcContextData } from "@/components/AtcFrequencies";

/** Re-check cadence while a card is open: runway states change in seconds. */
export const RUNWAY_POLL_MS = 8_000;

export interface RunwayLiveFix {
  lat: number;
  lon: number;
  altFt: number | null;
  gsKt: number | null;
  trk: number | null;
  onGround: boolean;
  vsFpm: number | null;
}

export interface RunwayStatusData {
  airport: { icao: string; name: string } | null;
  current: { state: string; runway: string; label: string } | null;
  departure: { airport: string; runway: string; label: string } | null;
  /** published ATC frequencies for the same fix (components/AtcFrequencies.tsx) */
  atc?: AtcContextData;
  honesty: string;
}

/** Pure: the request URL for a live fix (null fields are omitted). */
export function runwayStatusUrl(hex: string, f: RunwayLiveFix): string {
  const q = new URLSearchParams({ hex, lat: f.lat.toFixed(6), lon: f.lon.toFixed(6), gnd: f.onGround ? "1" : "0" });
  if (f.altFt != null && Number.isFinite(f.altFt)) q.set("altFt", String(Math.round(f.altFt)));
  if (f.gsKt != null && Number.isFinite(f.gsKt)) q.set("gsKt", String(Math.round(f.gsKt)));
  if (f.trk != null && Number.isFinite(f.trk)) q.set("trk", f.trk.toFixed(1));
  if (f.vsFpm != null && Number.isFinite(f.vsFpm)) q.set("vs", String(Math.round(f.vsFpm)));
  return `/api/data/procedures/runway?${q.toString()}`;
}

/** Pure: the one line the card shows (current state first, else the trip's
 *  departure runway), or null when there is nothing honest to say. */
export function runwayLine(d: RunwayStatusData | null): string | null {
  if (!d) return null;
  if (d.current) return d.airport ? `${d.current.label} · ${d.airport.icao}` : d.current.label;
  if (d.departure) return d.departure.label;
  return null;
}

let errCount = 0;

export function useRunwayStatus(opts: {
  hex: string | null | undefined;
  suppressed?: boolean;
  getLive: () => RunwayLiveFix | null;
}): { row: JSX.Element | null; atcRow: JSX.Element | null } {
  const hex = opts.hex && HEX_RE.test(opts.hex) && !opts.suppressed ? opts.hex.toLowerCase() : null;
  const [data, setData] = useState<RunwayStatusData | null>(null);
  const getLiveRef = opts.getLive;

  useEffect(() => {
    setData(null);
    if (!hex) return;
    let stop = false;
    const load = async () => {
      const f = getLiveRef();
      if (!f) return;
      try {
        const r = await fetch(runwayStatusUrl(hex, f), { signal: timeoutSignal(10_000) });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = (await r.json()) as RunwayStatusData;
        if (!stop) setData(d);
      } catch (e: unknown) {
        errCount++;
        if (errCount <= 3) console.warn("[runway] status:", e instanceof Error ? e.message : e);
      }
    };
    void load();
    const iv = pollWhenVisible(load, RUNWAY_POLL_MS);
    return () => { stop = true; iv(); };
    // getLive reads refs; re-subscribing on every render would refetch
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hex]);

  const line = runwayLine(data);
  const atcRow = hex && data?.atc ? <AtcFrequencyRows atc={data.atc} airport={data.airport?.icao} /> : null;
  if (!hex || !line) return { row: null, atcRow };
  const active = !!data?.current;
  return {
    atcRow,
    row: (
      <p className={`vt-runway-row${active ? " vt-runway-row-active" : ""}`} data-vt-runway title={data?.honesty}>
        <span className="vt-runway-label">{line}</span>
        <span className="vt-runway-tag">inferred</span>
      </p>
    ),
  };
}
