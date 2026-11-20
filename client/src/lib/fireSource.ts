// fireSource — which satellite the active-fire detections came from, in
// words. server/nasaFirms.ts asks NASA FIRMS for one VIIRS satellite at a
// time (NOAA-20, then NOAA-21, then Suomi NPP) and sends its last poll with
// the fires route; these functions turn that poll into the layer's status
// line. [REPAIR 2026-11-20] the layer read "warming up — first poll can take
// a few minutes" for 18 days while the one satellite it asked for had no rows.

export interface FirePollSource { source: string; rows: number | null; error: string | null }
export interface FirePoll {
  day_start?: boolean;
  last_attempt_at: string | null;
  last_rows_at: string | null;
  source_used: string | null;
  sources: FirePollSource[];
  polls_without_rows: number;
}

const SOURCE_NAME: Record<string, string> = {
  VIIRS_NOAA20_NRT: "NOAA-20",
  VIIRS_NOAA21_NRT: "NOAA-21",
  VIIRS_SNPP_NRT: "Suomi NPP",
};
/** The `satellite` column of a FIRMS row (NASA's codes). An unknown code is shown as it is. */
const SATELLITE_NAME: Record<string, string> = { N: "Suomi NPP", N20: "NOAA-20", N21: "NOAA-21" };

export function fireSourceName(source: string | null | undefined): string {
  if (!source) return "";
  return SOURCE_NAME[source] ?? source;
}

export function fireSatelliteName(code: string | null | undefined): string {
  const c = String(code ?? "").trim();
  if (!c || c === "unknown") return "satellite not stated";
  return SATELLITE_NAME[c] ?? c;
}

function ago(iso: string | null | undefined, nowMs: number): string {
  const t = iso ? Date.parse(iso) : NaN;
  if (!Number.isFinite(t)) return "";
  const min = Math.max(0, Math.round((nowMs - t) / 60000));
  if (min < 1) return "under a minute ago";
  if (min < 60) return `${min} min ago`;
  const h = Math.floor(min / 60);
  return h < 48 ? `${h} h ${min % 60} min ago` : `${Math.floor(h / 24)} d ago`;
}

/** "NOAA-20 no detections · NOAA-21 request failed · Suomi NPP no detections" */
function perSource(poll: FirePoll): string {
  return poll.sources
    .map((s) => `${fireSourceName(s.source)} ${s.error ? "request failed" : s.rows ? `${s.rows.toLocaleString("en-US")} detections` : "no detections"}`)
    .join(" · ");
}

/** The status line when the route has nothing to draw. Before the first poll
 *  that is a warm-up; after a poll with no rows it names what was asked. */
export function firesEmptyNote(d: { warming_up?: boolean; no_detections?: boolean; poll?: FirePoll | null }, nowMs: number): string {
  const poll = d.poll;
  if (!poll || !poll.last_attempt_at || !d.no_detections) return "warming up — first poll can take a few minutes";
  const when = ago(poll.last_attempt_at, nowMs);
  const failed = poll.sources.length > 0 && poll.sources.every((s) => s.error);
  const head = failed ? "NASA FIRMS did not answer" : "NASA FIRMS returned no detections from any satellite";
  return `${head}${when ? ` (asked ${when})` : ""}${poll.sources.length ? `: ${perSource(poll)}` : ""}`;
}

/** Appended to the live layer's note: which satellite, and whether the lead one was skipped. */
export function firesSourceNote(poll: FirePoll | null | undefined): string {
  const name = fireSourceName(poll?.source_used);
  if (!poll || !name) return "";
  const skipped = poll.sources.filter((s) => s.source !== poll.source_used);
  if (!skipped.length) return `from ${name}`;
  const why = skipped.map((s) => `${fireSourceName(s.source)} ${s.error ? "did not answer" : "had none"}`).join(", ");
  return `from ${name} (${why})`;
}
