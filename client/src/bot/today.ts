// Bot page — the Today tile's words (B1, human-approved 2026-11-03), pure and
// pinned by today.test.ts: the server's TodayTile (server/todayTile.ts —
// equity now vs the prior close by the daily-loss halt's own evaluateDailyPnl,
// SPY's move on the same ET day from the benchmark's cached bars) becomes the
// lines the tile prints and the hover sentence behind them. A side the server
// could not read prints its reason; nothing prints NaN, undefined or a zero
// that was not measured.
//
// BETWEEN SESSIONS ([REPAIR] 2026-11-06): the server names the session the
// numbers belong to (phase "between", day = the last session that ran, picked
// by the broker clock and the NYSE calendar — never guessed here). The tile
// then says so — "Thu Nov 5 session" in place of "Today" — and its lines say
// what the change runs between: the close before that session, and either that
// session's own close (Alpaca's snapshot) or equity now. SPY's note says
// "close" once its bar was fetched after the session closed.
import { fmtMoney, fmtPct, fmtSigned, fmtClockET, MINUS } from "./format";
import type { TodayTile } from "./useBotData";

/** SPY's number is older than this: the server's 30 min bar cache plus the poll that refreshes it has lapsed */
export const SPY_STALE_MS = 45 * 60_000;

export interface TodayView {
  /** the eyebrow's words before " · vs SPY": "Today" during a session; between sessions the session the numbers
   *  belong to, "Thu Nov 5", with eyebrowWord " session" after it (the phone and the folded strip drop that word) */
  eyebrow: string; eyebrowWord: string;
  /** the headline: the account's move minus SPY's, in percentage points; a dash without both sides */
  headline: string; headlineCls: string;
  /** the account's change against the prior close: "+$312 · +0.31%", or "acct — · <why>" */
  change: string; changeCls: string;
  /** "SPY +0.25%" with its own as-of clock ("14:02 ET", or "close" once the bar is the session's final close), or
   *  "SPY —" with the reason in a few words (the sentence is in the title) */
  spy: string; spyCls: string; spyNote: string; spyStale: boolean;
  /** the prior close and what the change runs to: "prior $99,860 → $100,172" during a session; between sessions
   *  "Wed $93,623 → Thu $89,889" (close to close) or "Wed $93,623 → now $89,800". pathValue is the numbers alone
   *  ("$99,860 → $100,172"); pathFrom is the word the phone and the folded strip drop, pathTo the one they keep (it says
   *  whether the number is the session's close or equity now: "$93,623 → Thu $89,889" is 22 mono characters, the tile
   *  holds about 25) */
  path: string; pathValue: string; pathFrom: string; pathTo: string;
  /** the two amounts of pathValue ([prior close, what the change runs to]); empty before the tile has numbers */
  pathSeg: string[];
  /** the whole provenance sentence (the tile's hover) */
  title: string;
}

/** a signed dollar amount with a true minus: +$312 · −$251 · $0 */
export function fmtSignedMoney(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const body = "$" + Math.round(Math.abs(v)).toLocaleString("en-US");
  return v < 0 ? MINUS + body : v > 0 ? "+" + body : body;
}

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const ymd = (day: string | null | undefined): [number, number, number] | null => {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(day ?? "");
  if (!m) return null;
  const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
  return mo >= 1 && mo <= 12 && d >= 1 && d <= 31 ? [y, mo, d] : null;
};
/** "Thu" — the weekday of an ET session date (YYYY-MM-DD); "" when unreadable */
export function weekdayOf(day: string | null | undefined): string {
  const p = ymd(day);
  return p ? WEEKDAYS[new Date(Date.UTC(p[0], p[1] - 1, p[2])).getUTCDay()] : "";
}
/** "Thu Nov 5" — an ET session date (YYYY-MM-DD) as the tile names it; "—" when unreadable */
export function sessionDayWords(day: string | null | undefined): string {
  const p = ymd(day);
  return p ? `${weekdayOf(day)} ${MONTHS[p[1] - 1]} ${p[2]}` : "—";
}

/** the server's reason sentence as the few words the phone tile has room for (the hover keeps the sentence):
 *  159 px of tile holds ~23 mono characters, and an ellipsis would hide the fact with no hover on touch */
export function shortReason(reason: string | null | undefined): string {
  if (!reason) return "not read";
  if (/^no SPY bars fetched/.test(reason)) return "no bars yet";
  if (/^no SPY bar for today/.test(reason)) return "no bar today";
  if (/^no SPY bar for \d{4}-\d{2}-\d{2}/.test(reason)) return "no bar yet";
  if (/prior close missing/.test(reason)) return "no prior close";
  if (/^account read invalid/.test(reason)) return "read invalid";
  if (/^the close before the .* is not recorded/.test(reason)) return "no daily line";
  return reason;
}
const cls = (v: number | null): string => (v === null ? "vt-bot-t3" : v < 0 ? "vt-bot-r" : v > 0 ? "vt-bot-g" : "vt-bot-t1");
const hhmm = (iso: string | null): string => (iso ? fmtClockET(iso).slice(0, 5) : "—");

export function todayView(t: TodayTile | null | undefined, nowMs: number, loaded: boolean): TodayView {
  if (!t) {
    const word = loaded ? "today not reported by this server" : "account not loaded";
    return { eyebrow: "Today", eyebrowWord: "", headline: "—", headlineCls: "vt-bot-t3", change: word, changeCls: "vt-bot-t3", spy: "SPY —", spyCls: "vt-bot-t3", spyNote: "", spyStale: false, path: "prior close —", pathValue: "close —", pathFrom: "prior", pathTo: "", pathSeg: [], title: word };
  }
  // a server that predates the phase field described the ET calendar day of the read, i.e. a session
  const between = t.phase === "between";
  const usd = t.pnl_usd, pct = t.pnl_pct, spy = t.spy_pct_today, ex = t.excess_pct;
  const acctOk = usd !== null && pct !== null && Number.isFinite(usd) && Number.isFinite(pct);
  const spyOk = spy !== null && Number.isFinite(spy);
  const spyFinal = between && spyOk && t.spy_final === true;
  const spyAge = t.spy_asof ? nowMs - Date.parse(t.spy_asof) : NaN;
  // a final close does not go stale: the bar cannot change after the session
  const spyStale = spyOk && !spyFinal && Number.isFinite(spyAge) && spyAge > SPY_STALE_MS;
  const change = acctOk ? `${fmtSignedMoney(usd)} · ${fmtPct(pct)}` : `acct — · ${shortReason(t.pnl_reason)}`;
  const changeLong = acctOk ? change : `acct — · ${t.pnl_reason ?? "not read"}`;
  const spyWord = spyOk ? `SPY ${fmtPct(spy)}` : "SPY —";
  const spyNote = spyOk ? (spyFinal ? "close" : `${hhmm(t.spy_asof)} ET`) : shortReason(t.spy_reason);
  const spyLong = t.spy_reason ?? "not read";
  const exOk = ex !== null && Number.isFinite(ex);
  // the change runs from the prior close to equity now, or (between sessions, once Alpaca has rolled last_equity) to the
  // session's own close; during a session equity_end is equity now (an older server sends no equity_end)
  const end = between ? t.equity_end ?? null : t.equity_end ?? t.equity_now;
  const pathSeg = [fmtMoney(t.equity_prev_close), fmtMoney(end)];
  const pathValue = `${pathSeg[0]} → ${pathSeg[1]}`;
  const prevWd = weekdayOf(t.prev_day);
  const pathFrom = between && prevWd ? prevWd : "prior";
  const pathTo = !between ? "" : t.end_basis === "close" ? weekdayOf(t.day) || "close" : t.end_basis === "now" ? "now" : "";
  const path = `${pathFrom} ${pathSeg[0]} → ${pathTo ? `${pathTo} ` : ""}${pathSeg[1]}`;
  const label = between ? `${sessionDayWords(t.day)} session` : "Today";
  // which session, and what decided it (a server that predates the field says nothing here)
  const sessionWords = t.phase_source ? `session: ${t.when ?? "—"}, by the ${t.phase_source}` : null;
  let account: string;
  if (!between) {
    account = `Today (${t.day} ET): equity ${fmtMoney(t.equity_now, true)} now vs ${fmtMoney(t.equity_prev_close, true)} at the prior close — ${changeLong} (Alpaca paper account equity − last_equity, the prior session's 16:00 ET value; read ${fmtClockET(t.asof)} ET)`;
  } else {
    const prevName = prevWd ? `the ${prevWd} close` : "the prior close";
    const prevSrc = t.prev_close_source === "daily_line" ? `as the ${t.day} daily line recorded Alpaca's last_equity after that session` : "Alpaca last_equity, which it still holds";
    const endWords = t.end_basis === "close"
      ? `at the ${weekdayOf(t.day) || "session's"} close (Alpaca's last_equity: its snapshot of that close, rolled to it overnight)`
      : t.end_basis === "now"
        ? `now (read ${fmtClockET(t.asof)} ET; after-hours marks included until Alpaca rolls last_equity to the session's close overnight)`
        : "—";
    const sinceClose = t.end_basis === "close" && t.equity_now !== null && end !== null && Number.isFinite(t.equity_now) && Number.isFinite(end)
      ? `; equity now ${fmtMoney(t.equity_now, true)}, ${fmtSignedMoney(t.equity_now - end)} since that close (after-hours and pre-market marks)`
      : "";
    account = `${label} (${t.day} ET, shown ${t.when ?? "between sessions"}): equity ${fmtMoney(end, true)} ${endWords} vs ${fmtMoney(t.equity_prev_close, true)} at ${prevName} (${prevSrc}) — ${changeLong}${sinceClose}`;
  }
  const spySentence = !spyOk
    ? `SPY — ${spyLong} (daily bars, price only, no dividends; SPY's side stays empty until ${between ? "that session's" : "today's"} bar exists, never an older day's move)`
    : between
      ? `SPY ${fmtPct(spy)} on ${t.day}: its daily bar against the prior close, price only, no dividends (Alpaca bars, feed iex; fetched ${hhmm(t.spy_asof)} ET, ${spyFinal ? "after the close: the final bar" : `before the close: not the final bar${spyStale ? ", older than the 30 min cache — stale" : ""}`})`
      // SPY is price only since 2026-11-05 (server/spyBenchmark.ts SPY_BENCH_ADJUSTMENT): the paper account is credited no dividends
      : `SPY ${fmtPct(spy)} today: its daily bar against the prior close, price only, no dividends (Alpaca bars, feed iex; fetched ${hhmm(t.spy_asof)} ET${spyStale ? ", older than the 30 min cache — stale" : ""})`;
  const title = [
    account,
    spySentence,
    exOk ? `headline ${fmtSigned(ex, 2, " pp")} = account − SPY in percentage points` : "headline — needs both sides",
    sessionWords,
  ].filter((x): x is string => !!x).join(" · ");
  return {
    eyebrow: between ? sessionDayWords(t.day) : "Today", eyebrowWord: between ? " session" : "",
    headline: exOk ? fmtSigned(ex, 2, " pp") : "—", headlineCls: cls(exOk ? ex : null),
    change, changeCls: cls(acctOk ? usd : null),
    spy: spyWord, spyCls: cls(spyOk ? spy : null), spyNote, spyStale,
    path, pathValue, pathFrom, pathTo, pathSeg,
    title,
  };
}
