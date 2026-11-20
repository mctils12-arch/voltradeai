// historySeries — the number-shaping behind the /data history charts
// (crop conditions, App Store rankings, GitHub activity; UI coverage batch
// C, 2026-11-20). Pure functions, no React: the chart component draws what
// these return and nothing else.
//
// Rules held here, each pinned by a test:
//   - a missing reading (null) is a GAP in the line, never joined across and
//     never drawn as zero;
//   - a single reading between two gaps is drawn as a dot, so it is not lost;
//   - fewer than two readings is "not enough history", not a flat line;
//   - a sum of parts is null when any part is missing (no partial sums).

export interface SeriesPoint {
  /** The reading's own date or week label, verbatim from the archive. */
  label: string;
  value: number | null;
}

export interface SeriesShape {
  /** One SVG path per unbroken run of two or more readings. */
  segments: string[];
  /** Readings with a gap on both sides. */
  dots: Array<{ x: number; y: number }>;
  lo: number;
  hi: number;
  first: { label: string; value: number };
  last: { label: string; value: number };
  /** Readings present / absent in the window. */
  present: number;
  missing: number;
}

/** Lays a series out in a width x height box. Returns null when fewer than
 *  two readings exist (the caller says "not enough history yet"). */
export function shapeSeries(points: SeriesPoint[], width: number, height: number, pad = 3): SeriesShape | null {
  const have = points
    .map((p, i) => ({ i, label: p.label, value: p.value }))
    .filter((p): p is { i: number; label: string; value: number } => typeof p.value === "number" && Number.isFinite(p.value));
  if (have.length < 2) return null;
  let lo = Infinity, hi = -Infinity;
  for (const p of have) { if (p.value < lo) lo = p.value; if (p.value > hi) hi = p.value; }
  const span = hi - lo;
  const n = points.length;
  const x = (i: number) => pad + (n <= 1 ? 0 : (i / (n - 1)) * (width - 2 * pad));
  // A flat series sits on the middle line rather than the floor.
  const y = (v: number) => (span === 0 ? height / 2 : height - pad - ((v - lo) / span) * (height - 2 * pad));

  const segments: string[] = [];
  const dots: Array<{ x: number; y: number }> = [];
  let run: Array<{ i: number; value: number }> = [];
  const flush = () => {
    if (run.length === 1) dots.push({ x: +x(run[0].i).toFixed(1), y: +y(run[0].value).toFixed(1) });
    else if (run.length > 1) {
      segments.push(run.map((p, k) => `${k === 0 ? "M" : "L"}${x(p.i).toFixed(1)},${y(p.value).toFixed(1)}`).join(" "));
    }
    run = [];
  };
  let prev = -2;
  for (const p of have) {
    if (p.i !== prev + 1) flush();
    run.push(p);
    prev = p.i;
  }
  flush();

  const f = have[0], l = have[have.length - 1];
  return {
    segments, dots, lo, hi,
    first: { label: f.label, value: f.value },
    last: { label: l.label, value: l.value },
    present: have.length,
    missing: n - have.length,
  };
}

/** Sum of parts, null when ANY part is missing or not a number. */
export function sumOrNull(parts: Array<number | null | undefined>): number | null {
  let s = 0;
  for (const p of parts) {
    if (typeof p !== "number" || !Number.isFinite(p)) return null;
    s += p;
  }
  return parts.length ? s : null;
}

// ── Crop conditions: /api/data/crop-conditions/history `trend` ──────────────
export interface CropTrendWeek {
  week_ending: string;
  corn?: Record<string, number | null>;
  soybeans?: Record<string, number | null>;
}

/** Good + excellent share per week. A week whose commodity carries no
 *  ratings (out of season: NASS stops publishing) is a gap. */
export function goodExcellentSeries(trend: CropTrendWeek[], commodity: "corn" | "soybeans"): SeriesPoint[] {
  return trend.map((w) => {
    const c = w[commodity] ?? {};
    return { label: w.week_ending, value: sumOrNull([c["GOOD"], c["EXCELLENT"]]) };
  });
}

// ── App Store: /api/data/appstore-rankings/history `trend` ──────────────────
export interface AppStoreTrendDay {
  date: string;
  ranked_slots: number | null;
  total_slots: number | null;
  total_rating_count: number | null;
}

/** Share of the day's fetched chart slots in which a watchlist app ranked,
 *  in percent. A day with no fetched slots is a gap, not 0%. */
export function rankedShareSeries(trend: AppStoreTrendDay[]): SeriesPoint[] {
  return trend.map((d) => ({
    label: d.date,
    value: typeof d.ranked_slots === "number" && typeof d.total_slots === "number" && d.total_slots > 0
      ? (d.ranked_slots / d.total_slots) * 100
      : null,
  }));
}

export function ratingCountSeries(trend: AppStoreTrendDay[]): SeriesPoint[] {
  return trend.map((d) => ({ label: d.date, value: typeof d.total_rating_count === "number" ? d.total_rating_count : null }));
}

// ── GitHub: /api/data/github-activity/history `trend` ───────────────────────
export interface GithubTrendWeek {
  weekStart: string;
  weekEnd?: string;
  orgs_reporting: number | null;
  total_merged_prs: number | null;
  total_commits: number | null;
}

export function githubSeries(trend: GithubTrendWeek[], field: "total_merged_prs" | "total_commits"): SeriesPoint[] {
  return trend.map((w) => ({ label: w.weekStart, value: typeof w[field] === "number" ? (w[field] as number) : null }));
}

/** The weeks whose org count differs from the newest week's: totals of those
 *  weeks are sums over a different set of orgs, and the chart says so. */
export function weeksWithDifferentCoverage(trend: GithubTrendWeek[]): number {
  if (!trend.length) return 0;
  const ref = trend[trend.length - 1].orgs_reporting;
  return trend.filter((w) => w.orgs_reporting !== ref).length;
}
