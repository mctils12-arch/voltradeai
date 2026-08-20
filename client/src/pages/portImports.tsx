// PortImportsView — monthly US port-level import values, #/data/imports.
// server/censusImports.ts (BUILD ORDER 3 #4, 2026-07-05) shipped the route
// with no client view: the last of the shipped-data-no-UI gaps the
// 2026-08-20 air-quality session's own NEXT note named. RAW display only —
// the module's hypothesis (import value/weight deltas lead retail inventory
// cycles, and pair with our port-dwell analytics for a two-sided port view)
// stays gate-locked in research/open_questions.md; nothing on this page
// reflects it. Reuses the generic .vt-filings-*/.vt-shortvol-* CSS.
//
// Every number-shaping decision lives in client/src/lib/portImports.ts with
// its own tests — the national aggregate row, the undefined month-over-month
// delta, the published-0-vs-missing-column distinction, and the live
// reconciliation. See that file's header for the evidence behind each.
import { useEffect, useMemo, useState, useSyncExternalStore, type CSSProperties } from "react";
import { ArrowLeft, ExternalLink, Ship } from "lucide-react";
import {
  splitNational, monthsOf, joinMonths, reconcile, sortRows, filterRows,
  type ImportObs, type SortKey,
} from "@/lib/portImports";
import { fmtKilograms, getUnits, subscribeUnits } from "@/lib/units";

interface ImportsResponse {
  kind?: string;
  enabled?: boolean;
  reason?: string;
  warming_up?: boolean;
  source?: string;
  attribution?: string;
  time?: number;
  count?: number;
  note?: string;
  imports?: ImportObs[];
}

// USD, not unit-switched (currency is not a measurement system) — compact,
// because a port column spans $600K to $317B.
const usd = (n: number | null | undefined): string => {
  if (n == null) return "—";
  const a = Math.abs(n);
  if (a >= 1e9) return `$${(n / 1e9).toFixed(1)}B`;
  if (a >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
  return `$${n.toFixed(0)}`;
};

const exact = (n: number | null | undefined): string =>
  n == null ? "no figure published" : `$${n.toLocaleString("en-US")}`;

// Signed, uncoloured on purpose: a rise in imports is not "good" and a fall is
// not "bad" — painting this column green/red would assert a directional read
// this page has no validated basis for. The sign carries the direction.
const pct = (d: number | null): string => (d == null ? "—" : `${d >= 0 ? "+" : ""}${(d * 100).toFixed(1)}%`);

const monthLabel = (m: string): string => {
  const [y, mo] = m.split("-");
  const names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const name = names[Number(mo) - 1];
  return name ? `${name} ${y}` : m;
};

const DEFAULT_LIMIT = 50;

// Native <select>/<input> inherit the page font and colour globally
// (index.css) but keep the UA's own background, which reads as light-on-light
// in some browsers against this dark surface. Themed here with the existing
// surface/border tokens rather than by adding global CSS for three controls.
const CONTROL: CSSProperties = {
  background: "var(--surface-2)",
  border: "1px solid var(--border)",
  borderRadius: 5,
  color: "var(--text-primary)",
  padding: "3px 6px",
};

export default function PortImportsView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<ImportsResponse | null>(null);
  const [error, setError] = useState(false);
  const [month, setMonth] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("gen_val");
  const [query, setQuery] = useState("");
  const [showAll, setShowAll] = useState(false);
  // The unit preference is a live store, so the weight column re-renders when
  // the user flips imperial/metric in the layers panel without a reload.
  const units = useSyncExternalStore(subscribeUnits, getUnits, getUnits);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/imports");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const { ports, national } = useMemo(() => splitNational(data?.imports), [data]);
  const months = useMemo(() => monthsOf(data?.imports), [data]);
  const selected = month && months.includes(month) ? month : months[0] ?? null;
  const prevMonth = selected ? months[months.indexOf(selected) + 1] ?? null : null;

  const rows = useMemo(
    () => (selected ? joinMonths(ports, selected, prevMonth) : []),
    [ports, selected, prevMonth],
  );
  const check = useMemo(
    () => (selected ? reconcile(rows, national, selected) : null),
    [rows, national, selected],
  );
  const shown = useMemo(() => sortRows(filterRows(rows, query), sortKey), [rows, query, sortKey]);
  const visible = showAll ? shown : shown.slice(0, DEFAULT_LIMIT);

  const nat = selected ? national.find((r) => r.month === selected) ?? null : null;
  const natPrev = prevMonth ? national.find((r) => r.month === prevMonth) ?? null : null;
  const natDelta =
    nat?.gen_val != null && natPrev?.gen_val ? (nat.gen_val - natPrev.gen_val) / natPrev.gen_val : null;

  const inactive = data ? data.enabled === false : false;
  const weightUnitNote = units === "imperial" ? "US short tons" : "metric tonnes";

  return (
    <div className="vt-filings-page" role="region" aria-label="US port imports">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Ship size={16} />
        <div>
          <div className="vt-filings-title">US port imports — monthly</div>
          <div className="vt-filings-sub">
            general &amp; containerized import value by port of entry (RAW, not a signal) ·{" "}
            {data && !data.warming_up && !inactive ? `${ports.length} port-months over ${months.length} months` : ""} ·{" "}
            {data?.attribution ?? "U.S. Census Bureau (USA Trade Online / FT920)"} ·{" "}
            <a href="https://www.census.gov/foreign-trade/reference/products/catalog/ft920.html" target="_blank" rel="noreferrer">
              FT920 <ExternalLink size={11} />
            </a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}
      {!error && inactive && (
        <div className="vt-filings-state">CENSUS_API_KEY not set — the import feed is inactive.</div>
      )}
      {!error && data && !inactive && data.warming_up && (
        <div className="vt-filings-state">Warming up — the first poll is still in flight.</div>
      )}
      {!error && data && !inactive && !data.warming_up && !selected && (
        <div className="vt-filings-state">No months in the current polling window.</div>
      )}

      {!error && data && !inactive && !data.warming_up && selected && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            Census publishes FT920 on a monthly schedule roughly <strong>45 days</strong> after the month
            closes, so the newest month here is deliberately not the current one. Values are as published
            for the month of entry; revisions arrive as new vintages in the archive rather than overwriting.
            This is a raw statistical release — no model, ranking weight, or forecast is applied below.
          </div>
          <div className="vt-filings-sub">
            hypothesis under research (gate-locked, not reflected here): import value and containerized
            weight deltas at individual ports lead retail inventory cycles, and pair with our port-dwell
            analytics for a two-sided view of the same port — one side flow, the other side friction.
          </div>

          <div className="vt-filings-sub" style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <label>
              Month{" "}
              <select style={CONTROL} value={selected} onChange={(e) => { setMonth(e.target.value); setShowAll(false); }}>
                {months.map((m) => <option key={m} value={m}>{monthLabel(m)}</option>)}
              </select>
            </label>
            <label>
              Sort by{" "}
              <select style={CONTROL} value={sortKey} onChange={(e) => setSortKey(e.target.value as SortKey)}>
                <option value="gen_val">General import value</option>
                <option value="cnt_val">Containerized value</option>
                <option value="cnt_wgt">Containerized weight</option>
                <option value="delta">Month-over-month change</option>
                <option value="port_name">Port name</option>
              </select>
            </label>
            <label>
              Find{" "}
              <input
                type="search"
                style={CONTROL}
                value={query}
                placeholder="port name or code"
                onChange={(e) => { setQuery(e.target.value); setShowAll(false); }}
              />
            </label>
          </div>

          {nat && (
            <div className="vt-filings-sub">
              <strong>All ports, {monthLabel(selected)}:</strong>{" "}
              <span title={exact(nat.gen_val)}>{usd(nat.gen_val)}</span> general imports
              {natDelta != null && <span> ({pct(natDelta)} vs {monthLabel(prevMonth!)})</span>}
              {" · "}
              <span title={exact(nat.cnt_val)}>{usd(nat.cnt_val)}</span> containerized
              {" · "}
              <span title={nat.cnt_wgt == null ? "no figure published" : `${nat.cnt_wgt.toLocaleString("en-US")} kg`}>
                {fmtKilograms(nat.cnt_wgt, units)}
              </span>{" "}
              containerized weight. This is Census's own published national total, shown separately and
              excluded from the table below so it is never counted twice.
            </div>
          )}

          {check && (
            <div className="vt-filings-sub">
              {check.published == null
                ? "Integrity check unavailable — Census published no all-ports total for this month, so the per-port rows cannot be reconciled against anything."
                : check.exact
                  ? `Integrity check: the ${rows.length} per-port rows sum to ${exact(check.sum)}, matching Census's published all-ports total for ${monthLabel(selected)} exactly. Computed on this page from the rows you are reading, not asserted.`
                  : `Integrity check: the ${rows.length} per-port rows sum to ${exact(check.sum)}, which differs from Census's published all-ports total (${exact(check.published)}) by ${exact(check.diff)}. Shown rather than hidden — treat the port breakdown as incomplete for this month.`}
            </div>
          )}

          <div className="vt-filings-sub">
            A <strong>0</strong> in the containerized columns is a figure Census published for that
            port-month — inland, air and land-border ports move no containerized vessel cargo — while
            <strong> —</strong> means the value was absent from the response entirely. Change is blank
            in two cases, both real and both common here: the port has no prior-month row at all, or its
            prior-month value was $0, where a rise has no finite percentage. Ports Census names are
            named; one that it does not is listed by its Schedule D code alone. Weights render in{" "}
            {weightUnitNote} per your unit setting; the archive keeps the source's kilograms.
          </div>

          {shown.length === 0 ? (
            <div className="vt-filings-state">
              {query ? `No port matches "${query}" in ${monthLabel(selected)}.` : "No port rows in this month."}
            </div>
          ) : (
            <>
              <div className="vt-filings-tablewrap">
                <table className="vt-filings-table">
                  <thead>
                    <tr>
                      <th>Port</th>
                      <th>Code</th>
                      <th className="num">General imports</th>
                      <th className="num">Containerized</th>
                      <th className="num">Container weight</th>
                      <th className="num">vs {prevMonth ? monthLabel(prevMonth) : "prior month"}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visible.map((r) => (
                      <tr key={r.port}>
                        <td data-l="Port">{r.port_name ?? <em>unnamed in source</em>}</td>
                        <td data-l="Code">{r.port}</td>
                        <td data-l="General imports" className="num" title={exact(r.gen_val)}>{usd(r.gen_val)}</td>
                        <td data-l="Containerized" className="num" title={exact(r.cnt_val)}>{usd(r.cnt_val)}</td>
                        <td
                          data-l="Container weight"
                          className="num"
                          title={r.cnt_wgt == null ? "no figure published" : `${r.cnt_wgt.toLocaleString("en-US")} kg`}
                        >
                          {r.cnt_wgt == null ? "—" : fmtKilograms(r.cnt_wgt, units)}
                        </td>
                        <td data-l="Change" className="num">{pct(r.delta)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {shown.length > visible.length && (
                <button type="button" className="vt-streams-launch" onClick={() => setShowAll(true)}>
                  Show all {shown.length} ports
                  <span className="vt-streams-launch-sub">
                    showing the top {visible.length} by the current sort
                  </span>
                </button>
              )}
            </>
          )}

          {data.note && <div className="vt-filings-sub">{data.note}</div>}
          <div className="vt-filings-sub">{data.source}</div>
        </div>
      )}
    </div>
  );
}
