// Bot page — the Expectancy tile's number split by sleeve ([PRODUCT]
// 2026-11-03, human-approved item B2), rendered under the Book's P&L block as
// one mono table (brief §6: tables as monitors — tabular figures, signed
// colour on the cell, every number with formula, source and age on hover).
// Rows: options, stocks, their total (= the tile's headline), the ETF floor,
// the VRP leg, every lot. Columns: n, win rate, average win, average loss,
// payoff, breakeven win rate, expectancy. The numbers come from the broker
// block of /api/bot/performance (server/edgeLedger.ts bySleeve, computed from
// the one sleeve cache); nothing is recomputed here. Absence is a designed
// state: the cold ledger says it is warming up, a server that predates the
// split says so, a sleeve without lots prints dashes with the reason on hover.
// The wrapper scrolls sideways when the panel is narrower than the eight
// columns (the phone, the 340 px desktop column) with the sleeve column
// pinned — no cell is ever cut, the page never overflows (DESIGN.md).
import { Help } from "./help";
import { St, NotRecorded } from "./panels";
import type { BrokerEdge } from "./useBotData";
import { TABLE_ORDER, TOTAL_GROUPS, GROUP_LABEL, GROUP_NOTE, COLS, cell, flagged, rowTitle, expectancyCaption } from "./expectancy";

export function ExpectancyBySleeve({ broker, reason }: { broker: BrokerEdge | null | undefined; reason: string | null | undefined }) {
  const head = <div className="vt-bot-eb vt-exp-eb">Expectancy by sleeve<Help k="expectancyBySleeve" /> · % per closed lot · broker</div>;
  if (!broker) {
    return <div className="vt-exp" data-state="warming">{head}<div className="vt-exp-src"><St word={reason ?? "broker ledger warming up"} /></div></div>;
  }
  const x = broker.bySleeve;
  if (!x) {
    return <div className="vt-exp" data-state="absent">{head}<div className="vt-exp-src"><NotRecorded what="not in this payload · the server predates the split; the next deploy carries it" /></div></div>;
  }
  const cap = expectancyCaption(broker);
  return (
    <div className="vt-exp" data-state="ready">
      {head}
      <div className="vt-exp-wrap">
        <table className="vt-exp-t">
          <thead>
            <tr>
              <th className="s" scope="col">sleeve</th>
              <th scope="col">expectancy</th>
              <th scope="col">n</th>
              <th scope="col">win</th>
              <th scope="col">avg win</th>
              <th scope="col">avg loss</th>
              <th scope="col">payoff<Help k="payoffRatio" /></th>
              <th scope="col">breakeven<Help k="breakevenWinRate" /></th>
            </tr>
          </thead>
          <tbody>
            {TABLE_ORDER.map((g) => {
              const s = x[g];
              const total = TOTAL_GROUPS.has(g);
              const flag = !total && flagged(s);
              return (
                <tr key={g} data-group={g} data-total={total ? "true" : undefined} data-flag={flag ? "neg" : undefined} title={rowTitle(g, s)}>
                  <td className="s">{GROUP_LABEL[g]}{GROUP_NOTE[g] ? <i>{GROUP_NOTE[g]}</i> : null}</td>
                  {COLS.map((c) => { const k = cell(s, c); return <td key={c} className={k.tone} title={k.title}>{k.text}</td>; })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="vt-exp-src">{cap.map((p, i) => <span key={i} className={p.tone === "warn" ? "vt-bot-o" : undefined}>{i ? " · " : ""}{p.text}</span>)}</div>
    </div>
  );
}
