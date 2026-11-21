// Institutional13FView — SEC 13F-HR institutional holdings (#/data/filings13f).
// server/edgar13f.ts (GATE 1 DATA passed — parser hand-checked against two
// real filed filings, see edgar13f.test.ts) has archived 13F-HR filings
// since 2026-07-05 with /api/data/filings13f + /api/data/filings13f/history
// live and no client view until now — this closes that gap, the same shape
// as the Form-4 filings precedent (both are SEC EDGAR as-filed feeds). RAW
// display only (kind:"raw"): every value is the filing's own reported
// number; GATE 2 (new-position clustering vs forward returns) has not been
// attempted, no predictive claim is made here. Reuses vt-filings-* CSS only
// — no new styles needed.
//
// BY FILED DAY (2026-11-21, open_questions 2026-11-19 item (x)). The page asks
// for the last 45 FILED days (`?filed_from=`), not archive days: since
// 2026-11-19 a nightly pass recovers the 13F filings the 15-minute poll
// missed and stores each under the night it was recovered, so an archive-day
// list opened with months-old filings in recovery order (read live
// 2026-11-20: 600 of 640 "newest" filings were the 2026-11-16 deadline's).
// Each filed day is a line that opens and closes, with the count the archive
// holds and whether that is every 13F-HR on EDGAR's daily index ("complete"),
// some of them ("partial") or the poll's unchecked sample; an open day lists
// its filings a page at a time (lib/filingDayGroups.ts, the Form 4 page's
// rules). A deadline day is one server page (1,712 filings on 2026-11-16), so
// nothing is cut at a row cap. A server that answers by archive day (an older
// build) is labelled as that and keeps the flat capped list.
import { Fragment, useCallback, useMemo, useState } from "react";
import { ArrowLeft, ChevronDown, ChevronRight, ExternalLink, Landmark } from "lucide-react";
import { coverageWords, filedDayFirstQuery, filedDayState, filedDayWords, filedDaysSummary, loadOlderLabel } from "@/lib/filingCoverage";
import { dayFilingsWords, defaultOpenDay, groupByFiledDay, isDayOpen, rowsShown, showMoreLabel } from "@/lib/filingDayGroups";
import { useFilingHistory } from "@/lib/useFilingHistory";

interface Holding13F {
  issuer: string | null;
  titleOfClass: string | null;
  cusip: string | null;
  value: number | null;
  shares: number | null;
  sharesType: string | null;
  putCall: string | null;
  discretion: string | null;
}
interface Filing13F {
  accession: string;
  filedAt: string | null;
  cik: string;
  indexUrl: string;
  managerCik: string | null;
  managerName: string | null;
  periodOfReport: string | null;
  submissionType: string | null;
  entryTotal: number | null;
  valueTotal: number | null;
  otherManagersCount: number | null;
  holdings: Holding13F[] | null;
  holdingsOmitted: boolean;
}
interface HistoryPayload {
  kind: string;
  warming_up?: boolean;
  source?: string;
  days?: number | null;
  count?: number;
  focused_cap?: number;
  archive_start?: string;
  window?: { from: string; to: string };
  covered?: { from: string; to: string };
  truncated?: boolean;
  next?: string | null;
  filings: Filing13F[];
}

const WINDOW_DAYS = 45;
/** archive-day pages only (no day groups to open): the flat list's cap */
const MAX_ROWS = 1200;
const filedDayOf = (f: Filing13F): string => (typeof f.filedAt === "string" ? f.filedAt.slice(0, 10) : "");

function fmtUSD(v: number | null): string {
  if (v == null) return "—";
  const abs = Math.abs(v);
  if (abs >= 1e9) return `$${(v / 1e9).toFixed(2)}b`;
  if (abs >= 1e6) return `$${(v / 1e6).toFixed(1)}m`;
  if (abs >= 1e3) return `$${(v / 1e3).toFixed(0)}k`;
  return `$${v.toLocaleString()}`;
}

export default function Institutional13FView({ onBack }: { onBack: () => void }) {
  const [selected, setSelected] = useState<string | null>(null);
  // the header states the archive days actually loaded and the footer walks to
  // the older ones (the route pages by whole days under a filing cap)
  // fixed for the life of the view: a re-render after UTC midnight must not refetch
  const [firstQuery] = useState(() => filedDayFirstQuery(Date.now(), WINDOW_DAYS));
  const { first: data, filings, coverage, filedDays, error, loadingOlder, olderError, loadOlder } =
    useFilingHistory<Filing13F, HistoryPayload>("/api/data/filings13f/history", firstQuery);
  const covered = coverageWords(coverage);
  const olderLabel = loadOlderLabel(coverage, loadingOlder);
  const byFiledDay = coverage?.by === "filed_day";
  const daysSummary = byFiledDay ? filedDaysSummary(filedDays) : "";
  // filed-day page: every loaded day is a group; archive-day page: one flat capped list
  const groups = useMemo(
    () => (byFiledDay ? groupByFiledDay(filings, filedDayOf, filedDays) : [{ day: "", rows: filings.slice(0, MAX_ROWS), status: null }]),
    [byFiledDay, filings, filedDays],
  );
  const defaultDay = useMemo(() => (byFiledDay ? defaultOpenDay(groups) : null), [byFiledDay, groups]);
  // the visitor's own open/closed choices, and how many times "show more" was used per day
  const [chosen, setChosen] = useState<ReadonlyMap<string, boolean>>(() => new Map());
  const [moreSteps, setMoreSteps] = useState<ReadonlyMap<string, number>>(() => new Map());
  const toggleDay = useCallback((day: string, open: boolean) => setChosen((m) => new Map(m).set(day, open)), []);
  const showMore = useCallback((day: string) => setMoreSteps((m) => new Map(m).set(day, (m.get(day) || 0) + 1)), []);

  const active = useMemo(
    () => filings.find((f) => f.accession === selected) ?? null,
    [filings, selected],
  );
  const holdings = useMemo(
    () => (active?.holdings ?? []).slice().sort((a, b) => (b.value ?? 0) - (a.value ?? 0)),
    [active],
  );

  return (
    <div className="vt-filings-page" role="region" aria-label="Institutional holdings (13F-HR)">
      <div className="vt-filings-head">
        <button className="vt-icon-btn" aria-label="Back to map" onClick={onBack}><ArrowLeft size={17} /></button>
        <Landmark size={16} />
        <div>
          <div className="vt-filings-title">Institutional holdings — SEC 13F-HR</div>
          <div className="vt-filings-sub">
            as filed · {filings.length.toLocaleString()} filings
            {covered ? <> · <span data-vt-coverage>{covered}</span> ({byFiledDay ? "days filed" : "UTC archive days"})</> : null}
            {daysSummary ? <> · <span data-vt-filed-summary>{daysSummary}</span></> : null}
            {" "}· archive began {data?.archive_start || "2026-07-05"} · gate 1 (data) passed, gate 2 (new-position clustering) not attempted ·
            filers over {(data?.focused_cap ?? 250).toLocaleString()} positions archive as summary-only ·{" "}
            <a href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=13F-HR" target="_blank" rel="noreferrer">
              SEC EDGAR <ExternalLink size={11} />
            </a>
          </div>
        </div>
      </div>

      {error && <div className="vt-filings-state">Feed error — the archive may still answer on refresh.</div>}
      {!error && data?.warming_up && (
        <div className="vt-filings-state">First poll still in progress — this can take a few minutes on a cold start.</div>
      )}
      {!error && !data?.warming_up && data && filings.length === 0 && (
        <div className="vt-filings-state">
          No filings archived yet — the poll runs every 15 minutes; 13F-HR filings cluster around the quarterly
          deadline (Feb/May/Aug/Nov 14).
        </div>
      )}

      {filings.length > 0 && (
        <>
          <div className="vt-filings-tablewrap om-sb">
            <table className="vt-filings-table">
              <thead>
                <tr>
                  <th>Manager</th><th>Period</th><th>Filed</th>
                  <th className="num">Positions</th><th className="num">Value</th>
                  <th aria-label="SEC link" />
                </tr>
              </thead>
              <tbody>
                {groups.map((g, gi) => {
                  const st = g.status;
                  const open = !byFiledDay || isDayOpen(chosen, g.day, defaultDay);
                  const shownN = byFiledDay ? rowsShown(g.rows.length, moreSteps.get(g.day) || 0) : g.rows.length;
                  const more = byFiledDay && open ? showMoreLabel(shownN, g.rows.length) : null;
                  return (
                    <Fragment key={g.day || `group-${gi}`}>
                      {byFiledDay && (
                        <tr className="vt-filings-dayrow" data-vt-filed-day={g.day} data-state={st ? filedDayState(st) : "sample"} data-open={open ? "1" : "0"}>
                          <th colSpan={6} scope="colgroup">
                            <button type="button" className="vt-filings-daybtn" aria-expanded={open} disabled={g.rows.length === 0}
                                    onClick={() => toggleDay(g.day, !open)}>
                              {g.rows.length > 0 && (open ? <ChevronDown size={13} aria-hidden /> : <ChevronRight size={13} aria-hidden />)}
                              <span>
                                <span className="vt-filings-dayrow-day">Filed {g.day || "on an unstated day"}</span>
                                <span data-vt-day-rows> · {dayFilingsWords(st ? st.filings : null, g.rows.length)}</span>
                                <span className="vt-filings-dayrow-state"> · {st ? filedDayWords(st) : "sample · not yet checked against EDGAR's index"}</span>
                              </span>
                            </button>
                          </th>
                        </tr>
                      )}
                      {open && g.rows.slice(0, shownN).map((f) => (
                        <tr
                          key={f.accession}
                          onClick={() => setSelected(f.accession === selected ? null : f.accession)}
                          style={{ cursor: "pointer" }}
                          aria-selected={f.accession === selected}
                        >
                          <td data-l="Manager">
                            <span className="vt-filings-ticker">{f.managerName || "—"}</span>
                            {f.otherManagersCount ? (
                              <span className="vt-filings-issuername">+{f.otherManagersCount} other filer(s)</span>
                            ) : null}
                          </td>
                          <td data-l="Period">{f.periodOfReport || "—"}</td>
                          <td data-l="Filed">{f.filedAt ? f.filedAt.slice(0, 10) : "—"}</td>
                          <td data-l="Positions" className="num">
                            {f.entryTotal != null ? f.entryTotal.toLocaleString() : "—"}
                            {f.holdingsOmitted ? " (summary only)" : ""}
                          </td>
                          <td data-l="Value" className="num">{fmtUSD(f.valueTotal)}</td>
                          <td>{f.indexUrl && (
                            <a href={f.indexUrl} target="_blank" rel="noreferrer" aria-label="Open SEC filing"
                               className="vt-filings-seclink" onClick={(e) => e.stopPropagation()}>
                              <ExternalLink size={13} />
                            </a>
                          )}</td>
                        </tr>
                      ))}
                      {more && (
                        <tr className="vt-filings-morerow">
                          <td colSpan={6} data-l="">
                            <button type="button" className="vt-filings-filter" data-vt-day-more={g.day} onClick={() => showMore(g.day)}>{more}</button>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
            {!byFiledDay && filings.length > MAX_ROWS && <div className="vt-filings-state">showing first {MAX_ROWS.toLocaleString()} of {filings.length.toLocaleString()} filings</div>}
          </div>
          {!error && olderLabel && (
            <div className="vt-filings-older">
              <button className="vt-filings-filter" onClick={loadOlder} disabled={loadingOlder} data-vt-load-older>
                {olderLabel}
              </button>
              {olderError && <span role="status">Could not load the older days — the loaded rows are unchanged. Try again.</span>}
            </div>
          )}

          <div className="vt-filings-sub">
            {active
              ? `Top holdings by value — ${active.managerName || "selected manager"}, period ${active.periodOfReport || "—"}`
              : "Click a filing above to see its top holdings by value."}
          </div>

          {active && active.holdingsOmitted && (
            <div className="vt-filings-state">
              This manager reported more than {(data?.focused_cap ?? 250).toLocaleString()} positions — archived as
              summary-only (holdings table omitted) per the focused-manager cap; mega-manager tables are
              index-hugging noise for a capacity-constrained-manager signal (EDGE DOCTRINE #2).
            </div>
          )}
          {active && !active.holdingsOmitted && holdings.length === 0 && (
            <div className="vt-filings-state">No holdings parsed for this filing.</div>
          )}
          {active && !active.holdingsOmitted && holdings.length > 0 && (
            <div className="vt-filings-tablewrap">
              <table className="vt-filings-table">
                <thead>
                  <tr>
                    <th>Issuer</th><th>Class</th>
                    <th className="num">Value</th><th className="num">Shares</th><th>Discretion</th>
                  </tr>
                </thead>
                <tbody>
                  {holdings.map((h, i) => (
                    <tr key={`${h.cusip}-${i}`}>
                      <td data-l="Issuer">{h.issuer || "—"}{h.putCall ? ` (${h.putCall})` : ""}</td>
                      <td data-l="Class">{h.titleOfClass || "—"}</td>
                      <td data-l="Value" className="num">{fmtUSD(h.value)}</td>
                      <td data-l="Shares" className="num">{h.shares != null ? h.shares.toLocaleString() : "—"}{h.sharesType ? ` ${h.sharesType}` : ""}</td>
                      <td data-l="Discretion">{h.discretion || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
