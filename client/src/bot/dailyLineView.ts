// Bot page OVERVIEW — the daily line card ([PRODUCT] 2026-11-06), pure and pinned by dailyLineView.test.ts.
// GET /api/bot/daily-line serves the lines server/dailyLine.ts writes once per trading day after the close (the
// human's 2026-11-05 daily line: account vs SPY on one basis, floor exposure in SPY terms, option P&L split into
// fills vs direction, option exits held by the 09:30–09:45 window), exactly as written; this turns them into one row
// per session, newest first, and says how old the newest is and where today's line stands. Every number is a field
// of a line; a field the line could not read stays a dash with its reason, never a zero. Parsed tolerantly: a server
// that predates the route answers 404 (or the SPA page) — a designed state, never a crash.
import { fmtPct, fmtSigned, fmtClockET, fmtMonDay } from "./format";
import { dayWords, signedMoney, unavailableWords } from "./whyView";

/** One session's line, as the card prints it. */
export interface DailyRow {
  day: string;
  /** "Thu Nov 5" */
  label: string;
  /** account and SPY day returns on the same basis (percent), their gap (percentage points) and why one is missing */
  acct: number | null; spy: number | null; gap: number | null; dayReason: string | null;
  /** what the account's day ends at ([RULE-REVIEW] 2026-11-06): "close" = its equity at the close (the last read before
   *  the bell); "write_time" = the equity read when the line was written, after-hours marks included (no usable close
   *  reading — and every line before 2026-11-06, which carry no source); null = no account day */
  acctBasis: "close" | "write_time" | null;
  /** when that equity was read; why a line since 2026-11-06 has no close reading */
  acctAt: string | null; closeReason: string | null;
  /** this line's check of the PREVIOUS session's line against Alpaca's own 16:00 ET close of it (last_equity, rolled
   *  overnight): that line's account day, Alpaca's, and the difference (pp) */
  check: { day: string; label: string; linePct: number; alpacaPct: number; errorPp: number } | null;
  /** the floor's exposure: "0.85×" in SPY terms (vol-managed; SSO counted at 2×), "0.40× eq" of equity in basket mode */
  floor: string; floorX: number | null; floorTarget: number | null; floorReason: string | null;
  /** option P&L over the line's window: total = fills (vs the quoted mid) + direction (the rest); null = not computable */
  optTotal: number | null; optFills: number | null; optDirection: number | null;
  /** "2/5 priced": fills joined to a quoted mid / filled option orders in the window */
  priced: string; optReason: string | null;
  /** the window spans more than the day's session (a missed line before it): where it starts */
  optSince: string | null;
  /** option exits the opening window held back that day; the split */
  blocks: number | null; blocksDetail: string; blocksReason: string | null;
  /** inputs the line could not read (it was written partial), by name */
  partial: string[];
  writtenAt: string | null;
}

export type DailyTone = "ok" | "wait" | "warn";
export interface DailyLineView {
  state: "loading" | "error" | "empty" | "ready";
  rows: DailyRow[];
  newest: DailyRow | null;
  /** where today's line stands (written, due later, waiting, failed, no session): one sentence and its tone */
  today: { text: string; tone: DailyTone } | null;
  /** lines stored on the server (the card shows the newest few) */
  stored: number;
  /** the newest line's own basis sentence */
  basis: string | null;
  /** a row shown ends its account day at the equity read when written (its cell is marked "*") */
  writeTime: boolean;
  /** why there is nothing to show */
  note: string | null;
}

const rec = (x: unknown): Record<string, unknown> => (x && typeof x === "object" && !Array.isArray(x) ? (x as Record<string, unknown>) : {});
const num = (x: unknown): number | null => (typeof x === "number" && Number.isFinite(x) ? x : null);
const str = (x: unknown): string | null => (typeof x === "string" && x.trim() ? x.trim() : null);
const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/** "Thu Nov 5" — an ET session date (YYYY-MM-DD); "—" when unreadable */
export function sessionLabel(day: string | null | undefined): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(day ?? "");
  if (!m) return "—";
  const wd = WEEKDAYS[new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]))).getUTCDay()];
  return `${wd} ${dayWords(day)}`;
}

/** One stored line -> its row; null when it is not a line. */
export function dailyRow(raw: unknown): DailyRow | null {
  const l = rec(raw);
  const day = str(l.day);
  if (!day || !/^\d{4}-\d{2}-\d{2}$/.test(day)) return null;
  const vs = rec(rec(l.account_vs_spy).day);
  const fl = rec(l.floor);
  const op = rec(l.options);
  const fills = rec(op.fills);
  const win = rec(op.window);
  const ow = rec(l.opening_window);
  const floorX = num(fl.exposure_x);
  const floorTarget = num(fl.target_x);
  const mode = str(fl.mode);
  const fillsN = num(fills.fills);
  const attributed = num(fills.attributed);
  const blocks = num(ow.total);
  const fromLine = str(win.from_line);
  const acct = num(vs.account_pct);
  // a line without the source predates the close reading: its day ended at the equity read when it was written
  const acctBasis = acct === null ? null : vs.account_source === "close" ? "close" : "write_time";
  const ck = rec(l.close_check);
  const ckDay = str(ck.day);
  const ckLine = num(ck.line_pct), ckAlpaca = num(ck.alpaca_pct), ckErr = num(ck.error_pp);
  return {
    day, label: sessionLabel(day),
    acct, spy: num(vs.spy_pct), gap: num(vs.gap_pp), dayReason: str(vs.reason),
    acctBasis, acctAt: acctBasis === null ? null : str(vs.equity_end_at) ?? (acctBasis === "write_time" ? str(l.written_at) : null),
    closeReason: acctBasis === "write_time" ? str(vs.close_reason) : null,
    check: ckDay && ckLine !== null && ckAlpaca !== null && ckErr !== null ? { day: ckDay, label: sessionLabel(ckDay), linePct: ckLine, alpacaPct: ckAlpaca, errorPp: ckErr } : null,
    floor: floorX === null ? "—" : mode === "vol_managed_spy" ? `${floorX.toFixed(2)}×` : `${floorX.toFixed(2)}× eq`,
    floorX, floorTarget: mode === "vol_managed_spy" ? floorTarget : null, floorReason: str(fl.reason),
    optTotal: num(op.total), optFills: num(fills.pnl), optDirection: num(op.direction),
    priced: fillsN === null ? "—" : `${attributed ?? 0}/${fillsN} priced`,
    optReason: str(op.reason),
    optSince: fromLine && win.single_session !== true ? sessionLabel(fromLine) : null,
    blocks, blocksDetail: blocks === null ? "" : `manager ${(num(ow.manager_single_leg) ?? 0) + (num(ow.manager_group) ?? 0)} · POS-KILL ${num(ow.pos_kill) ?? 0}`,
    blocksReason: str(ow.reason),
    partial: Array.isArray(l.missing) ? l.missing.map((m) => String(m).split(":")[0].trim()).filter(Boolean) : [],
    writtenAt: str(l.written_at),
  };
}

const hhmm = (iso: string | null): string => (iso ? fmtClockET(iso).slice(0, 5) : "—");

/** Where today's line stands, from the route's schedule (the NYSE calendar) and the job's last attempt. */
export function todayLine(body: Record<string, unknown>, newest: DailyRow | null, nowMs: number): { text: string; tone: DailyTone } | null {
  const sch = rec(body.schedule);
  const day = str(sch.day);
  if (!day) return null;
  if (newest && newest.day === day) return { text: `Today's line was written at ${hhmm(newest.writtenAt)} ET`, tone: "ok" };
  if (sch.session !== true) return { text: "No session today: the next line follows the next session's close", tone: "ok" };
  const due = str(sch.due_at);
  const dueMs = due ? Date.parse(due) : NaN;
  if (Number.isFinite(dueMs) && nowMs < dueMs) return { text: `Today's line is written after the close, from ${hhmm(due)} ET`, tone: "ok" };
  const at = rec(body.lastAttempt);
  const detail = str(at.detail);
  if (str(at.day) === day && at.status === "waiting") return { text: `Today's line is waiting for its inputs${detail ? ` (${detail})` : ""}; it is written partial if one stays unreadable`, tone: "wait" };
  if (str(at.day) === day && at.status === "error") return { text: `Today's line failed${detail ? `: ${detail}` : ""}; the job retries`, tone: "warn" };
  return { text: `Today's line was due at ${hhmm(due)} ET and is not written yet`, tone: "warn" };
}

/** The card's model: the route's body (or its failure) -> rows, newest first, and the states around them. */
export function dailyLineView(raw: unknown, q: { loading: boolean; error: unknown }, nowMs: number, max = 10): DailyLineView {
  const none = { rows: [], newest: null, today: null, stored: 0, basis: null, writeTime: false };
  // nothing fetched and nothing failed yet (pending, or not yet enabled): the skeleton, never a shape error
  if (raw === undefined && !q.error) return { state: "loading", ...none, note: null };
  const body = rec(raw);
  const recent = Array.isArray(body.recent) ? body.recent : null;
  if (!recent) {
    return { state: "error", ...none, note: q.error ? unavailableWords(q.error, "the daily line") : "the daily line answered in a shape this page does not read" };
  }
  // a failed refresh keeps the lines fetched before it on screen, saying so (DESIGN.md: last-known state, never a blank)
  const refreshNote = q.error ? `the last refresh failed (${String(q.error instanceof Error ? q.error.message : q.error).slice(0, 120)}) — these are the lines fetched before it` : null;
  const rows = recent.map(dailyRow).filter((r): r is DailyRow => r !== null).sort((a, b) => (a.day < b.day ? 1 : a.day > b.day ? -1 : 0)).slice(0, max);
  const newest = rows[0] ?? null;
  const stored = num(body.count) ?? rows.length;
  const today = todayLine(body, newest, nowMs);
  const fileError = str(body.error);
  if (!rows.length) {
    return { state: fileError ? "error" : "empty", rows: [], newest: null, today, stored, basis: null, writeTime: false,
      note: [fileError ?? "No daily line yet: the job writes one after each session's close (+20 min)", refreshNote].filter(Boolean).join(" · ") };
  }
  const newestRaw = recent.find((l) => str(rec(l).day) === newest.day);
  return { state: "ready", rows, newest, today, stored, basis: str(rec(rec(newestRaw).account_vs_spy).basis),
    writeTime: rows.some((r) => r.acctBasis === "write_time"), note: [fileError, refreshNote].filter(Boolean).join(" · ") || null };
}

// ── words ───────────────────────────────────────────────────────────────────

/** "+0.27 pp" / "—" */
export const gapWords = (v: number | null): string => (v === null ? "—" : fmtSigned(v, 2, " pp"));
/** "−4.07%" / "—" */
export const pctWords = (v: number | null): string => (v === null ? "—" : fmtPct(v));
/** the account cell: "−3.99%" at the close; "−4.07%*" when the day ends at the equity read when the line was written */
export const acctWords = (r: DailyRow): string => `${pctWords(r.acct)}${r.acctBasis === "write_time" ? "*" : ""}`;
/** The card's account-basis sentence ([RULE-REVIEW] 2026-11-06); the "*" clause only when a row shown carries it. */
export function acctBasisWords(writeTime: boolean): string {
  return "Account = equity at the close (the last read before the bell) vs the prior close (Alpaca last_equity)"
    + (writeTime ? " · * = equity when the line was written (no close reading; after-hours marks can move it)" : "");
}
/** "the Thu Nov 5 line read −4.07%, Alpaca's close −3.99% (−0.08 pp)" */
export function checkWords(c: NonNullable<DailyRow["check"]>): string {
  return `the ${c.label} line read ${pctWords(c.linePct)}, Alpaca's close ${pctWords(c.alpacaPct)} (${gapWords(c.errorPp)})`;
}
/** the option cell: the total when the line could compute it, else "n/a" */
export const optWords = (r: DailyRow): string => (r.optTotal === null ? "n/a" : signedMoney(Math.round(r.optTotal)));
/** the option split's two parts: "fills −$40 · direction −$80" (the fills alone when the total is not computable) and
 *  "(2/5 priced)" — the card keeps the second unbroken, so a narrow cell wraps before it, never inside it */
export function optSplitParts(r: DailyRow): { main: string; priced: string } {
  const fills = r.optFills === null ? "fills —" : `fills ${signedMoney(Math.round(r.optFills))}`;
  return { main: r.optDirection === null ? fills : `${fills} · direction ${signedMoney(Math.round(r.optDirection))}`, priced: `(${r.priced})` };
}
/** the option split as one phrase: "fills −$40 · direction −$80 (2/5 priced)" */
export function optSplitWords(r: DailyRow): string {
  const p = optSplitParts(r);
  return `${p.main} ${p.priced}`;
}
/** a row's whole story for its hover: every field with its caveat */
export function rowTitle(r: DailyRow): string {
  // the exception says so; the default (the close) is the card's own sentence
  const basis = r.acctBasis === "write_time"
    ? ` (equity when the line was written${r.acctAt ? `, ${hhmm(r.acctAt)} ET` : ""}: after-hours marks included${r.closeReason ? `; ${r.closeReason}` : ""})`
    : "";
  const parts = [
    `${r.label}: account ${acctWords(r)}${basis} vs SPY ${pctWords(r.spy)} (${gapWords(r.gap)})${r.dayReason ? ` — ${r.dayReason}` : ""}`,
    r.floorX === null ? `floor — ${r.floorReason ?? "not read"}` : `floor ${r.floor}${r.floorTarget !== null ? ` (target ${r.floorTarget.toFixed(2)}×)` : ""}`,
    r.optTotal === null ? `options n/a — ${r.optReason ?? "not computable"}; ${optSplitWords(r)}` : `options ${optWords(r)} = ${optSplitWords(r)}${r.optSince ? `, since the ${r.optSince} line` : ""}`,
    r.blocks === null ? "opening-window blocks —" : `opening-window blocks ${r.blocks} (${r.blocksDetail})${r.blocksReason ? ` — ${r.blocksReason}` : ""}`,
    r.partial.length ? `written partial: ${r.partial.join(", ")} unreadable` : null,
    r.writtenAt ? `written ${fmtMonDay(r.writtenAt)} ${hhmm(r.writtenAt)} ET` : null,
    r.check ? `close check: ${checkWords(r.check)}` : null,
  ];
  return parts.filter((x): x is string => !!x).join(" · ");
}

/** The newest row's missing pieces, each with its reason — on the card itself, since a touch screen has no hover. */
export function rowNotes(r: DailyRow): string[] {
  const out: string[] = [];
  if (r.dayReason) out.push(`${r.label}: account vs SPY — ${r.dayReason}`);
  // a line since 2026-11-06 that had to fall back says why (the older lines' "*" is the card's own footnote)
  if (r.acctBasis === "write_time" && r.closeReason) out.push(`${r.label}: account = equity when the line was written${r.acctAt ? ` (${hhmm(r.acctAt)} ET)` : ""} — ${r.closeReason}`);
  if (r.floorX === null && r.floorReason) out.push(`${r.label}: floor — ${r.floorReason}`);
  if (r.optTotal === null && r.optReason) out.push(`${r.label}: options total n/a — ${r.optReason}`);
  if (r.optSince) out.push(`${r.label}: the option window opens at the ${r.optSince} line (a line in between was missed)`);
  if (r.blocksReason) out.push(`${r.label}: 09:30 holds — ${r.blocksReason}`);
  if (r.partial.length) out.push(`${r.label}: written partial — ${r.partial.join(", ")} unreadable`);
  return out;
}
