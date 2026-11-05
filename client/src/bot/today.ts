// Bot page — the Today tile's words (B1, human-approved 2026-11-03), pure and
// pinned by today.test.ts: the server's TodayTile (server/todayTile.ts —
// equity now vs the prior close by the daily-loss halt's own evaluateDailyPnl,
// SPY's move on the same ET day from the benchmark's cached bars) becomes the
// lines the tile prints and the hover sentence behind them. A side the server
// could not read prints its reason; nothing prints NaN, undefined or a zero
// that was not measured.
import { fmtMoney, fmtPct, fmtSigned, fmtClockET, MINUS } from "./format";
import type { TodayTile } from "./useBotData";

/** SPY's number is older than this: the server's 30 min bar cache plus the poll that refreshes it has lapsed */
export const SPY_STALE_MS = 45 * 60_000;

export interface TodayView {
  /** the headline: the account's move minus SPY's, in percentage points; a dash without both sides */
  headline: string; headlineCls: string;
  /** the account's change against the prior close: "+$312 · +0.31%", or "acct — · <why>" */
  change: string; changeCls: string;
  /** "SPY +0.25%" with its own as-of clock ("14:02 ET"), or "SPY —" with the reason in a few words (the sentence is in the title) */
  spy: string; spyCls: string; spyNote: string; spyStale: boolean;
  /** the prior close and equity now: "prior $99,860 → $100,172"; pathValue is the "$99,860 → $100,172" part (the phone
   *  drops the word — 159 px of tile holds 18 mono characters, not 24) */
  path: string; pathValue: string;
  /** the whole provenance sentence (the tile's hover) */
  title: string;
}

/** a signed dollar amount with a true minus: +$312 · −$251 · $0 */
export function fmtSignedMoney(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const body = "$" + Math.round(Math.abs(v)).toLocaleString("en-US");
  return v < 0 ? MINUS + body : v > 0 ? "+" + body : body;
}

/** the server's reason sentence as the few words the phone tile has room for (the hover keeps the sentence):
 *  159 px of tile holds ~23 mono characters, and an ellipsis would hide the fact with no hover on touch */
export function shortReason(reason: string | null | undefined): string {
  if (!reason) return "not read";
  if (/^no SPY bars fetched/.test(reason)) return "no bars yet";
  if (/^no SPY bar for today/.test(reason)) return "no bar today";
  if (/prior close missing/.test(reason)) return "no prior close";
  if (/^account read invalid/.test(reason)) return "read invalid";
  return reason;
}
const cls = (v: number | null): string => (v === null ? "vt-bot-t3" : v < 0 ? "vt-bot-r" : v > 0 ? "vt-bot-g" : "vt-bot-t1");
const hhmm = (iso: string | null): string => (iso ? fmtClockET(iso).slice(0, 5) : "—");

export function todayView(t: TodayTile | null | undefined, nowMs: number, loaded: boolean): TodayView {
  if (!t) {
    const word = loaded ? "today not reported by this server" : "account not loaded";
    return { headline: "—", headlineCls: "vt-bot-t3", change: word, changeCls: "vt-bot-t3", spy: "SPY —", spyCls: "vt-bot-t3", spyNote: "", spyStale: false, path: "prior close —", pathValue: "close —", title: word };
  }
  const usd = t.pnl_usd, pct = t.pnl_pct, spy = t.spy_pct_today, ex = t.excess_pct;
  const acctOk = usd !== null && pct !== null && Number.isFinite(usd) && Number.isFinite(pct);
  const spyOk = spy !== null && Number.isFinite(spy);
  const spyAge = t.spy_asof ? nowMs - Date.parse(t.spy_asof) : NaN;
  const spyStale = spyOk && Number.isFinite(spyAge) && spyAge > SPY_STALE_MS;
  const change = acctOk ? `${fmtSignedMoney(usd)} · ${fmtPct(pct)}` : `acct — · ${shortReason(t.pnl_reason)}`;
  const changeLong = acctOk ? change : `acct — · ${t.pnl_reason ?? "not read"}`;
  const spyWord = spyOk ? `SPY ${fmtPct(spy)}` : "SPY —";
  const spyNote = spyOk ? `${hhmm(t.spy_asof)} ET` : shortReason(t.spy_reason);
  const spyLong = t.spy_reason ?? "not read";
  const exOk = ex !== null && Number.isFinite(ex);
  const pathValue = `${fmtMoney(t.equity_prev_close)} → ${fmtMoney(t.equity_now)}`;
  const title = [
    `Today (${t.day} ET): equity ${fmtMoney(t.equity_now, true)} now vs ${fmtMoney(t.equity_prev_close, true)} at the prior close — ${changeLong} (Alpaca paper account equity − last_equity, the prior session's 16:00 ET value; read ${fmtClockET(t.asof)} ET)`,
    // SPY is price only since 2026-11-05 (server/spyBenchmark.ts SPY_BENCH_ADJUSTMENT): the paper account is credited no dividends
    spyOk ? `SPY ${fmtPct(spy)} today: its daily bar against the prior close, price only, no dividends (Alpaca bars, feed iex; fetched ${hhmm(t.spy_asof)} ET${spyStale ? ", older than the 30 min cache — stale" : ""})` : `SPY — ${spyLong} (daily bars, price only, no dividends; SPY's side stays empty until today's bar exists, never an older day's move)`,
    exOk ? `headline ${fmtSigned(ex, 2, " pp")} = account − SPY in percentage points` : "headline — needs both sides",
  ].join(" · ");
  return {
    headline: exOk ? fmtSigned(ex, 2, " pp") : "—", headlineCls: cls(exOk ? ex : null),
    change, changeCls: cls(acctOk ? usd : null),
    spy: spyWord, spyCls: cls(spyOk ? spy : null), spyNote, spyStale,
    path: `prior ${pathValue}`, pathValue,
    title,
  };
}
