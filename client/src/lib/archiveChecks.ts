// archiveChecks — the Data quality panel's sentences about an archive checked
// against its source's own record. First reader: the fire-detection archive's
// daily completeness pass (server/firesCompleteness.ts, 2026-11-20), which
// asks NASA for a finished day by date, adds the rows the live poll missed and
// refills days that have no file. Pure functions with tests; the page lays
// them out.

export interface FiresCheckDay {
  day: string;
  status: "complete" | "refilled" | "no_rows" | "failed";
  upstream_rows: number;
  held_before: number;
  added: number;
  attempts: number;
}
export interface FiresCheck {
  enabled?: boolean;
  last_run_at?: string | null;
  refilled_days?: number;
  no_rows_days?: number;
  failed_days?: number;
  live_poll_missed?: { days: number; upstream_rows: number; added: number; share: number | null };
  days?: FiresCheckDay[];
}

const n = (x: number) => x.toLocaleString("en-US");
const plural = (x: number, one: string, many = `${one}s`) => `${n(x)} ${x === 1 ? one : many}`;

function ago(iso: string, nowMs: number): string {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "at an unreadable time";
  const min = Math.max(0, Math.round((nowMs - t) / 60000));
  if (min < 1) return "under a minute ago";
  if (min < 60) return `${min} min ago`;
  const h = Math.floor(min / 60);
  return h < 48 ? `${h} h ${min % 60} min ago` : `${Math.floor(h / 24)} d ago`;
}

/** "2026-11-03 to 2026-11-19", or the one day. Input in any order. */
function span(days: string[]): string {
  const s = [...days].sort();
  return s.length === 1 ? s[0] : `${s[0]} to ${s[s.length - 1]}`;
}

/** A small share is shown with the digits it needs: 0.37%, 12.0%. */
export function sharePct(share: number | null | undefined): string {
  if (typeof share !== "number" || !Number.isFinite(share)) return "—";
  const p = share * 100;
  return `${p < 1 ? p.toFixed(2) : p.toFixed(1)}%`;
}

/** The fire-detection archive against NASA's dated record, one sentence per fact. */
export function firesCheckLines(s: FiresCheck | null | undefined, nowMs: number): string[] {
  if (!s) return [];
  if (s.enabled === false) return ["The daily check against NASA's record by date is off on this server."];
  if (!s.last_run_at) return ["The daily check against NASA's record by date has not run yet (it runs once a day from 06:00 UTC)."];
  const days = s.days ?? [];
  const of = (st: FiresCheckDay["status"]) => days.filter((d) => d.status === st);
  const out: string[] = [`Checked against NASA's record by date ${ago(s.last_run_at, nowMs)}.`];
  const m = s.live_poll_missed;
  if (m && m.days > 0) {
    out.push(m.added > 0
      ? `Over ${plural(m.days, "finished day")} the live recording had missed ${n(m.added)} of the ${n(m.upstream_rows)} detections NASA holds for those dates (${sharePct(m.share)}); they were added.`
      : `Over ${plural(m.days, "finished day")} the live recording held every one of the ${n(m.upstream_rows)} detections NASA holds for those dates.`);
  } else {
    out.push("No finished day has been compared yet.");
  }
  const refilled = of("refilled");
  if (refilled.length) {
    const rows = refilled.reduce((a, d) => a + d.added, 0);
    out.push(`${plural(refilled.length, "day")} with no recording (${span(refilled.map((d) => d.day))}) ${refilled.length === 1 ? "was" : "were"} refilled afterwards from NASA's answer for the date: ${n(rows)} detections. They are NASA's record as of the refill, not a live recording.`);
  }
  const none = of("no_rows");
  if (none.length) out.push(`${plural(none.length, "day")} (${span(none.map((d) => d.day))}) ${none.length === 1 ? "stays" : "stay"} empty: NASA returned nothing for the date from any satellite.`);
  const failed = of("failed");
  if (failed.length) out.push(`${plural(failed.length, "day")} (${span(failed.map((d) => d.day))}) could not be read from NASA: the request failed.`);
  return out;
}
