// HistoryChart — the one line chart the /data history panels share (crop
// conditions, App Store rankings, GitHub activity; UI coverage batch C,
// 2026-11-20). Every decision about what is drawn lives in
// client/src/lib/historySeries.ts with its tests; this file only renders.
//
// What it always states, in words, under the line: the first and last
// reading with their own dates, the low and the high, how many readings the
// window holds and how many are missing (drawn as gaps). No colour carries a
// "good" or "bad" reading: these are RAW archives with no validated
// directional meaning.
import { useEffect, useState, type ReactNode } from "react";
import { shapeSeries, type SeriesPoint } from "@/lib/historySeries";

/** Reads a history route's `trend` array. `state` is what the panel says
 *  while there is no chart: a failed read says so, it is never an empty
 *  chart. */
export function useTrend<T>(url: string): { trend: T[] | null; state: "loading" | "ok" | "error" | "off"; reason?: string } {
  const [out, setOut] = useState<{ trend: T[] | null; state: "loading" | "ok" | "error" | "off"; reason?: string }>({ trend: null, state: "loading" });
  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch(url);
        if (!r.ok) throw new Error(String(r.status));
        const d = await r.json();
        if (stop) return;
        if (d?.enabled === false) setOut({ trend: null, state: "off", reason: typeof d.reason === "string" ? d.reason : undefined });
        else if (Array.isArray(d?.trend)) setOut({ trend: d.trend as T[], state: "ok" });
        else setOut({ trend: null, state: "error" });
      } catch {
        if (!stop) setOut({ trend: null, state: "error" });
      }
    })();
    return () => { stop = true; };
  }, [url]);
  return out;
}

/** The section that holds a panel's history charts, with its own loading,
 *  failed and switched-off sentences. */
export function HistorySection({ heading, state, reason, children }: {
  heading: string; state: "loading" | "ok" | "error" | "off"; reason?: string; children: ReactNode;
}) {
  return (
    <div className="vt-hist-section" data-vt-hist-section data-state={state}>
      <div className="vt-hist-heading">{heading}</div>
      {state === "loading" && <div className="vt-hist-skel" aria-busy="true" aria-label="Loading history" />}
      {state === "error" && <div className="vt-shortvol-spark-empty">The history did not load. The latest reading above is unaffected; a refresh asks again.</div>}
      {state === "off" && <div className="vt-shortvol-spark-empty">{reason ?? "This archive is switched off."}</div>}
      {state === "ok" && <div className="vt-hist-grid">{children}</div>}
    </div>
  );
}

const W = 320;
const H = 64;

export default function HistoryChart({
  title, points, format, unit, footnote,
}: {
  title: string;
  points: SeriesPoint[];
  /** Formats a value for the labels (the chart never shows a bare number). */
  format: (v: number) => string;
  /** What one point is: "week" or "day". */
  unit: "week" | "day";
  footnote?: string;
}) {
  const s = shapeSeries(points, W, H);
  return (
    <div className="vt-hist" data-vt-hist>
      <div className="vt-hist-head">
        <span className="vt-hist-title">{title}</span>
        {s && <span className="vt-hist-last" data-vt-hist-last>{format(s.last.value)} · {s.last.label}</span>}
      </div>
      {!s ? (
        <div className="vt-shortvol-spark-empty" data-vt-hist-empty>
          Not enough history yet: the archive holds fewer than two {unit}s for this series.
        </div>
      ) : (
        <>
          <svg className="vt-hist-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img"
               aria-label={`${title}: ${format(s.first.value)} on ${s.first.label} to ${format(s.last.value)} on ${s.last.label}; low ${format(s.lo)}, high ${format(s.hi)}`}>
            {s.segments.map((d, i) => (
              <path key={i} d={d} fill="none" stroke="var(--accent-bright)" strokeWidth={1.6} vectorEffect="non-scaling-stroke" />
            ))}
            {s.dots.map((p, i) => (
              <circle key={i} cx={p.x} cy={p.y} r={1.8} fill="var(--accent-bright)" />
            ))}
          </svg>
          <div className="vt-hist-foot">
            <span>{s.first.label}: {format(s.first.value)}</span>
            <span>low {format(s.lo)} · high {format(s.hi)}</span>
            <span data-vt-hist-count>
              {s.present} {unit}{s.present === 1 ? "" : "s"}
              {s.missing > 0 ? ` · ${s.missing} with no reading, left as gaps` : ""}
            </span>
          </div>
        </>
      )}
      {footnote && <div className="vt-hist-note">{footnote}</div>}
    </div>
  );
}
