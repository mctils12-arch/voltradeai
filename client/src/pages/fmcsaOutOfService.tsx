// FmcsaOosView — FMCSA Out-of-Service (OOS) orders, #/data/fmcsa-oos.
// server/fmcsaOutOfService.ts (EDGE DOCTRINE #1, built + archived
// 2026-09-28, RAW per CLAUDE.md's RAW OVERLAYS vs SIGNALS rule — an
// enforcement-action log, no predictive claim) shipped API + archive only,
// deferring the client view to "not this session" (its own filed NEXT(3),
// research/open_questions.md) — this is that session. Reuses the generic
// .vt-filings-*/.vt-shortvol-* CSS every other RAW table view already uses.
//
// Not a spatial layer: this dataset carries no lat/lon (dot_number,
// legal_name, dates, reason, status only — no carrier address field in the
// upstream Socrata dataset), so it launches as a page-wide dashboard from
// the streams panel top, same as dtcc-swaps/un-comtrade, not a map layer.
import { useMemo, useState, useEffect } from "react";
import { ArrowLeft, ExternalLink, Truck } from "lucide-react";

interface OosRow {
  dot_number: string;
  legal_name: string;
  oos_date: string;
  oos_reason: string | null;
  status: string | null;
  rescind_date: string | null;
}
interface OosPayload {
  kind?: string;
  warming_up?: boolean;
  source?: string;
  attribution?: string;
  time?: number;
  count?: number;
  note?: string;
  orders?: OosRow[];
}

type StatusFilter = "ACTIVE" | "ALL";
const DEFAULT_LIMIT = 50;

export default function FmcsaOosView({ onBack }: { onBack: () => void }) {
  const [data, setData] = useState<OosPayload | null>(null);
  const [error, setError] = useState(false);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("ACTIVE");
  const [query, setQuery] = useState("");
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    let stop = false;
    (async () => {
      try {
        const r = await fetch("/api/data/fmcsa-oos");
        const d = await r.json();
        if (!stop) setData(d);
      } catch {
        if (!stop) setError(true);
      }
    })();
    return () => { stop = true; };
  }, []);

  const orders = data?.orders ?? [];
  const activeCount = useMemo(() => orders.filter((o) => o.status === "ACTIVE").length, [orders]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return orders
      .filter((o) => statusFilter === "ALL" || o.status === "ACTIVE")
      .filter((o) => !q || o.legal_name.toLowerCase().includes(q) || o.dot_number.includes(q))
      // most-recent first; the API returns oos_date ASC for its own
      // window-cutoff logic, not for display order.
      .slice().sort((a, b) => (a.oos_date < b.oos_date ? 1 : a.oos_date > b.oos_date ? -1 : 0));
  }, [orders, statusFilter, query]);

  const visible = showAll ? filtered : filtered.slice(0, DEFAULT_LIMIT);

  return (
    <div className="vt-filings-page" role="region" aria-label="FMCSA out-of-service orders">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Truck size={16} />
        <div>
          <div className="vt-filings-title">Out-of-service orders — motor carriers (FMCSA)</div>
          <div className="vt-filings-sub">
            small/non-public carrier enforcement actions, trailing 45 days · RAW, not a signal ·{" "}
            <a href="https://ai.fmcsa.dot.gov/SMS/" target="_blank" rel="noreferrer">FMCSA <ExternalLink size={11} /></a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && !data && <div className="vt-filings-state">Loading…</div>}
      {!error && data?.warming_up && <div className="vt-filings-state">Warming up — first poll not yet complete.</div>}
      {!error && data && !data.warming_up && orders.length === 0 && (
        <div className="vt-filings-state">No out-of-service orders in the trailing 45-day window.</div>
      )}

      {!error && data && !data.warming_up && orders.length > 0 && (
        <div className="vt-shortvol-body">
          <div className="vt-filings-sub">
            {orders.length.toLocaleString()} orders in window · {activeCount.toLocaleString()} currently ACTIVE ·{" "}
            {data.attribution ?? "FMCSA Out-of-Service Orders"}
          </div>

          <div className="vt-filings-sub" style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            <select
              aria-label="Status filter"
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as StatusFilter)}
              style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text-primary)", padding: "3px 6px" }}
            >
              <option value="ACTIVE">ACTIVE only</option>
              <option value="ALL">All statuses</option>
            </select>
            <input
              type="text"
              placeholder="Search carrier name or DOT #"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              style={{ background: "var(--surface-2)", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text-primary)", padding: "3px 6px", flex: "1 1 200px" }}
            />
          </div>

          <div className="vt-filings-tablewrap">
            <table className="vt-filings-table">
              <thead>
                <tr>
                  <th>Carrier</th>
                  <th>DOT #</th>
                  <th>Status</th>
                  <th>Reason</th>
                  <th>OOS date</th>
                  <th>Rescinded</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((row) => (
                  <tr key={`${row.dot_number}-${row.oos_date}`}>
                    <td data-l="Carrier">{row.legal_name || "—"}</td>
                    <td data-l="DOT #">{row.dot_number}</td>
                    <td data-l="Status">{row.status ?? "—"}</td>
                    <td data-l="Reason">{row.oos_reason ?? "—"}</td>
                    <td data-l="OOS date">{row.oos_date}</td>
                    <td data-l="Rescinded">{row.rescind_date ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {!showAll && filtered.length > DEFAULT_LIMIT && (
            <button type="button" className="vt-icon-btn" style={{ width: "auto", padding: "4px 10px" }} onClick={() => setShowAll(true)}>
              Show all {filtered.length.toLocaleString()} matching orders
            </button>
          )}

          <div className="vt-filings-sub vt-streams-foot">
            enforcement-action log, not a predictive reading — a carrier's presence here is a compliance event, not
            a trading signal (see research/open_questions.md's separate, unattempted GATE 2 hypothesis) ·{" "}
            {data.note}
          </div>
        </div>
      )}
    </div>
  );
}
