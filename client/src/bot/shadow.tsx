// ENTRY SHADOW MODE on the Bot page (2026-10-28): the bot logs would-be stock
// and options entries instead of ordering them; the floor basket and every
// exit stay live. Two pieces, both absent when nothing is shadowed:
//  - ShadowChip: an inline chip (the Now tile's eyebrow, the Orders header) —
//    same line height as the text it sits in, so it never shifts the layout;
//    a click opens the Orders panel zoomed, where the full block lives.
//  - ShadowBlock: the sentence, where the switch was set, and today's
//    would-be entries — full (zoomed / window) or folded (phone + tablet stack).
import type { BotStatus } from "./useBotData";
import { shadowHeadline, shadowSourceNote, shadowCountWord, shadowRowWhat, shadowRowTime } from "./shadowView";

export function ShadowChip({ status, onOpen, label }: { status: BotStatus | undefined; onOpen?: () => void; label: "short" | "count" }) {
  const headline = shadowHeadline(status?.entry_shadow);
  if (!headline) return null;
  const count = status?.shadow_entries_today?.count ?? 0;
  // "entries:" drops at phone width (CSS) so the Now eyebrow stays one line
  const text = label === "count" ? <>shadow · {count} would-be</> : <><span className="vt-shadow-pre">entries:{"\u00a0"}</span>shadow</>;
  const title = `${headline} ${shadowCountWord(count)}.`;
  return onOpen
    ? <button type="button" className="vt-shadow-chip" data-shadow-chip={label} onClick={onOpen} title={`${title} Click for the list.`} aria-label={`${title} Open the list`}>{text}<span aria-hidden="true"> →</span></button>
    : <span className="vt-shadow-chip" data-shadow-chip={label} title={title}>{text}</span>;
}

export function ShadowBlock({ status, mode }: { status: BotStatus | undefined; mode: "full" | "fold" }) {
  const es = status?.entry_shadow;
  const headline = shadowHeadline(es);
  if (!headline) return null;
  const today = status?.shadow_entries_today;
  const count = today?.count ?? 0;
  const rows = today?.entries ?? [];
  const list = rows.length ? (
    <div className="vt-bot-list vt-shadow-list">
      {rows.map((r, k) => (
        <div key={`${r.sleeve}|${r.ticker}|${k}`} className="vt-bot-trow" title={`${r.sleeve} · via ${r.source} — logged, no order placed`}>
          <span className="vt-bot-m vt-bot-t3 vt-shadow-t">{shadowRowTime(r)}</span>
          <span className="sym">{r.ticker}</span>
          <span className="vt-bot-m vt-bot-t2 vt-shadow-what">{shadowRowWhat(r)}</span>
          <span className="vt-bot-m vt-bot-t3 vt-shadow-sl">{r.sleeve}{mode === "full" ? ` · ${r.source}` : ""}</span>
        </div>
      ))}
      {count > rows.length ? <div className="vt-bot-t3" style={{ fontSize: 11 }}>newest {rows.length} of {count} shown</div> : null}
    </div>
  ) : null;
  return (
    <div className="vt-shadow" data-shadow="on" role="status">
      <div className="vt-shadow-msg"><span className="vt-bot-st vt-bot-o"><span className="vt-bot-dot" />shadow</span><span>{headline}</span></div>
      <div className="vt-shadow-src">{shadowSourceNote(es)} · would-be entries are recorded in the shadow log and measured, never sent to the broker</div>
      {!rows.length ? <div className="vt-shadow-empty">{shadowCountWord(0)}</div>
        : mode === "fold" ? <details className="vt-shadow-fold"><summary>{shadowCountWord(count)}</summary>{list}</details>
        : <><div className="vt-shadow-count">{shadowCountWord(count)}</div>{list}</>}
    </div>
  );
}

