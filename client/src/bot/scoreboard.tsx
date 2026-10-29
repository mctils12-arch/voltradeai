// Bot page — the daily bot-vs-SPY scoreboard (human-ordered 2026-10-28).
// One row per trading day from GET /api/bot/scoreboard (server/scoreboard.ts);
// every number shows where it came from and how old it is, and absence is
// drawn as a designed state, never as zeros.
import { useState } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import type { Scoreboard, ScoreRow } from "./useBotData";
import { fmtPct, fmtSigned, fmtClockET } from "./format";
import { visibleRows, fmtDay, beatLine, cashFlowNote, SCORE_VISIBLE_DAYS } from "./scoreView";
import { Help } from "./help";
import { St, NotRecorded } from "./panels";

const tone = (v: number) => (v > 0 ? "vt-bot-g" : v < 0 ? "vt-bot-r" : "vt-bot-t2");

function Skeleton() {
  return (
    <div className="vt-score" aria-busy="true" aria-label="Loading the daily scoreboard">
      <div className="vt-score-sum"><i className="vt-score-skel" style={{ width: 120, height: 26 }} /><i className="vt-score-skel" style={{ width: "60%", height: 12 }} /><i className="vt-score-skel" style={{ width: "45%", height: 12 }} /></div>
      <div className="vt-score-rows">{Array.from({ length: SCORE_VISIBLE_DAYS }, (_, k) => <i key={k} className="vt-score-skel vt-score-skelrow" />)}</div>
    </div>
  );
}

function Row({ r, year }: { r: ScoreRow; year: number }) {
  const gap = r.sessions > 1;
  return (
    <tr data-gap={gap ? "true" : undefined} data-provisional={r.provisional ? "true" : undefined}
      title={`${r.prevDate} close → ${r.date} close${gap ? ` · spans ${r.sessions} sessions (the equity snapshot missed ${r.sessions - 1})` : ""}${r.provisional ? " · today: the equity snapshot is still being rewritten" : ""}`}>
      <td className="d">{fmtDay(r.date, year)}{gap ? <span className="vt-score-flag">·{r.sessions}d</span> : null}{r.provisional ? <span className="vt-score-flag vt-bot-o">·live</span> : null}</td>
      <td>{fmtPct(r.botPct)}</td>
      <td>{fmtPct(r.spyPct)}</td>
      <td className={tone(r.diffPp)}>{fmtSigned(r.diffPp)}</td>
      <td className="cum">{fmtPct(r.botCumPct, 1)}</td>
      <td className="cum">{fmtPct(r.spyCumPct, 1)}</td>
      <td className={tone(r.cumDiffPp)}>{fmtSigned(r.cumDiffPp, 1)}</td>
    </tr>
  );
}

export function ScoreBody({ q, nowMs }: { q: UseQueryResult<Scoreboard>; nowMs: number }) {
  const [showAll, setShowAll] = useState(false);
  const data = q.data;
  if (!data) {
    if (q.isError) {
      return (
        <div className="vt-bot-empty" role="status">
          <St word="scoreboard unavailable" cls="vt-bot-o" />
          <span>{String(q.error instanceof Error ? q.error.message : q.error).slice(0, 160)} · the page asks again every 10 min</span>
        </div>
      );
    }
    return <Skeleton />;
  }
  const s = data.summary;
  if (!s || !data.rows.length) {
    return (
      <div className="vt-bot-empty" role="status">
        {data.error ? <St word="SPY series or equity curve unavailable" cls="vt-bot-o" /> : <NotRecorded what="not enough history yet" />}
        <span>{data.error ? `${data.error} · nothing is shown rather than a guess` : "a row needs two trading days with both an equity snapshot and a SPY close"}</span>
      </div>
    );
  }
  const year = new Date(nowMs).getUTCFullYear();
  const rows = visibleRows(data.rows, showAll);
  const cf = cashFlowNote(data.cashFlows);
  const asOf = data.fetchedAt ? `${fmtDay(new Date(data.fetchedAt).toLocaleDateString("en-CA", { timeZone: "America/New_York" }), year)} ${fmtClockET(data.fetchedAt).slice(0, 5)} ET` : "—";
  return (
    <div className="vt-score">
      <div className="vt-score-sum">
        <div className="vt-score-hero">
          <span className={`vt-bot-m ${tone(s.cumDiffPp)}`}>{fmtSigned(s.cumDiffPp, 2, " pt")}</span>
          <span className="vt-bot-t3">vs SPY since {fmtDay(s.from, year)}</span>
        </div>
        <div className="vt-score-kv vt-bot-m">
          <span>bot <b className={tone(s.botCumPct)}>{fmtPct(s.botCumPct)}</b> · SPY <b className={tone(s.spyCumPct)}>{fmtPct(s.spyCumPct)}</b></span>
          <span title="trading days the account's return beat SPY's (strictly), over the rows compared">beat SPY <b>{beatLine(s)}</b></span>
          <span title="worst peak-to-trough on the daily after-close points">max drawdown bot <b className="vt-bot-r">{fmtPct(s.botMaxDdPct, 1)}</b> · SPY <b className="vt-bot-r">{fmtPct(s.spyMaxDdPct, 1)}</b></span>
        </div>
        {q.isError ? <div className="vt-score-note vt-bot-o" role="status">refresh failed · the table below is as of {asOf}</div> : null}
      </div>
      <table className="vt-score-t">
        <thead>
          <tr><th className="d" scope="col">Day</th><th scope="col">Bot</th><th scope="col">SPY</th><th scope="col">Diff pt</th><th className="cum" scope="col">Bot cum</th><th className="cum" scope="col">SPY cum</th><th scope="col">Cum pt</th></tr>
        </thead>
        <tbody>{rows.map((r) => <Row key={r.date} r={r} year={year} />)}</tbody>
      </table>
      {data.rows.length > SCORE_VISIBLE_DAYS ? (
        <button type="button" className="vt-bot-wbtn vt-score-more" aria-expanded={showAll} onClick={() => setShowAll((v) => !v)}>
          {showAll ? `show the last ${SCORE_VISIBLE_DAYS} days` : `show all ${data.rows.length} days since ${fmtDay(s.from, year)}`}
        </button>
      ) : null}
      <div className="vt-score-src">
        <span>Bot: daily after-close equity snapshot of the Alpaca paper account · SPY: dividend-adjusted daily closes (Alpaca) through {fmtDay(data.barsThrough, year)} · fetched {asOf}<Help k="scoreboard" /></span>
        <span className={cf.tone === "warn" ? "vt-bot-o" : undefined}>{cf.text}</span>
        {s.gapRows ? <span>{s.gapRows} row{s.gapRows === 1 ? "" : "s"} span a missed snapshot (marked ·Nd) — both returns cover the same sessions</span> : null}
      </div>
    </div>
  );
}
