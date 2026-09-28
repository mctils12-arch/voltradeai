/**
 * killSwitchAutoResume.ts — logic-based AUTO-RESUME of the max-drawdown kill
 * latch, PAPER ACCOUNT ONLY (human-directed 2026-09-28, KNOWN BROKEN #42/#43).
 *
 * WHY: the -10%-from-peak kill (drawdownGuard.ts evaluateDrawdown, tripped in
 * server/bot.ts) latched on 2026-09-10T03:12Z on a reading (~-18%, ~$12K)
 * that three independent lines of evidence say was a broker/price-data
 * anomaly, not a loss: summed unrealized P&L across the book ~ -$92..-$279,
 * only small round-trip fills, and a close-to-close reconstruction of the
 * held book at -$414.82 vs -$12,059.74 reported (research/open_questions.md
 * #42/#43, research/wishlist.md "ACTIVE LIVE CONCERN"). The persisted latch
 * (R16, 2026-07-07) had exactly one clear path — the owner toggle — so the
 * paper loop sat dark for 18 days. The human, exercising sovereignty over
 * the "only the owner toggle clears it" rule: "there was no drop — implement
 * it in a way that automatically turns the bot back on, it's only paper
 * trading anyways, with logic."
 *
 * WHAT IS NOT CHANGED: the trip itself (evaluateDrawdown, the -10% threshold,
 * the order cancel, the -25% liquidate-on-kill mercy rule) and the owner
 * toggle. This module only decides WHEN a drawdown-kill latch may clear.
 *
 * THE RULES (all must hold, on 3 consecutive tier-1 evaluations):
 *   1. every Alpaca base URL the process can trade through is the PAPER host;
 *   2. the latch reason is a drawdown kill — an owner-set manual kill, or a
 *      latch whose reason is unknown, is NEVER auto-cleared;
 *   3. the market is open (tier-1 cadence) and every read this evaluation is
 *      valid (equity credible, account not blocked, positions + fills read);
 *   4. latched for >= COOLDOWN_MINUTES;
 *   5. one resume BASIS holds:
 *      RECOVERED    — validated drawdown back above the kill threshold by a
 *                     hysteresis margin (peak kept: it is real);
 *      DATA_ANOMALY — positions + fills explain < ANOMALY_EXPLAINED_RATIO of
 *                     the reported drop (peak RE-BASELINED to current equity,
 *                     prior peak logged, so a bogus stale peak cannot
 *                     instantly re-trip). Disabled when the latch began
 *                     within ANTI_FLAP_HOURS of the previous auto-resume;
 *      PAPER_REBASE — the loss is explained (or unprovable) and the latch has
 *                     lasted PAPER_REBASE_AFTER_TRADING_DAYS market days: the
 *                     peak re-baselines to current equity and trading resumes,
 *                     loudly audited.
 *
 * HONEST LIMITS (stated where they bite, not hidden):
 *   - "explained loss" = -(summed unrealized P&L + realized P&L of fills in a
 *     bounded window). Unrealized P&L is measured vs COST BASIS, not vs the
 *     peak date, so gains a position already carried at the peak are not
 *     counted as lost. That UNDERSTATES since-peak losses on long-held
 *     winners and biases toward DATA_ANOMALY. Accepted for PAPER only: the
 *     fast path re-baselines the peak, so the kill stays armed at -10% from
 *     the new baseline — the bias moves resume timing, never removes the halt.
 *   - Incomplete reconciliation (truncated fill history, a closing fill whose
 *     cost basis predates the window, an unpriced position) can never prove
 *     an anomaly: it classifies UNRECONCILED and only RECOVERED/PAPER_REBASE
 *     can resume.
 *   - Market days are counted with liveness.ts's marketHoursBetween (NYSE
 *     weekday sessions, holidays NOT excluded — market_calendar.py stays the
 *     one holiday source): a holiday can count as a session, so the rebase
 *     can arrive up to one session early.
 *
 * PURE (node:test safe — no fs, no network, no wall clock).
 */
import { evaluateDrawdown } from "./drawdownGuard";
import { marketHoursBetween } from "./liveness";
import { isOptionPosition } from "./reconstructPnl";

// Tunables. 2026-09-28 (human-directed auto-resume, KNOWN BROKEN #42/#43).
// Each value's reason is next to it; expected effect: a drawdown-kill latch
// on the paper account clears itself within ~2-3 tier-1 cycles once a basis
// holds, instead of waiting indefinitely for the owner toggle.
export const AUTO_RESUME = Object.freeze({
  // Minimum time latched before any resume: long enough that a transient
  // bad read cannot trip-and-clear inside one flicker, short enough that a
  // clearly bogus reading does not cost a trading session.
  COOLDOWN_MINUTES: 30,
  // Debounce: the resume condition must hold on this many consecutive tier-1
  // evaluations (~45s apart). Any invalid read or failed condition resets it.
  CONSECUTIVE_OK_REQUIRED: 3,
  // DATA_ANOMALY when positions + fills explain less than this share of the
  // reported peak-to-equity drop. 0.25 = "three quarters of the drop has no
  // trace anywhere in the book" (2026-09-09 incident: 414.82 / 12,059.74 =
  // 3.4% explained).
  ANOMALY_EXPLAINED_RATIO: 0.25,
  // RECOVERED needs drawdown ABOVE the kill threshold by this many percentage
  // points (-10% kill -> resume above -8%) so the latch cannot flap at the line.
  RECOVERY_HYSTERESIS_PP: 2,
  // A real (explained) loss stays halted this many market days, then the
  // paper peak re-baselines to current equity and trading resumes.
  PAPER_REBASE_AFTER_TRADING_DAYS: 2,
  MARKET_HOURS_PER_TRADING_DAY: 6.5,
  // Realized-P&L window: fills from this many calendar days before the latch
  // through now (bounded — the fills endpoint is paged, see FILLS_MAX_PAGES).
  FILLS_LOOKBACK_DAYS: 10,
  FILLS_PAGE_SIZE: 100,
  FILLS_MAX_PAGES: 5,
  // A new latch that begins within this many hours of the previous
  // auto-resume may not use the DATA_ANOMALY fast path (a persistently bad
  // feed must not trip/clear/trip every 30 minutes).
  ANTI_FLAP_HOURS: 24,
});

export const PAPER_ALPACA_HOST = "paper-api.alpaca.markets";

/** True only for an https URL whose host is exactly Alpaca's paper API. */
export function isPaperAlpacaUrl(url: string | null | undefined): boolean {
  if (typeof url !== "string" || url.length === 0) return false;
  try {
    const u = new URL(url);
    return u.protocol === "https:" && u.hostname === PAPER_ALPACA_HOST;
  } catch {
    return false;
  }
}

/** Every EFFECTIVE base URL the process can trade through must be paper.
 *  Callers pass effective URLs (env value or that code path's own default),
 *  never raw possibly-unset env. An empty list is not verifiably paper. */
export function allPaperAlpacaUrls(urls: ReadonlyArray<string | null | undefined>): boolean {
  return urls.length > 0 && urls.every((u) => isPaperAlpacaUrl(u));
}

export type LatchKind = "DRAWDOWN_KILL" | "OWNER_MANUAL" | "UNKNOWN";

/** Reasons written by server/bot.ts saveKillSwitch(): "drawdown-kill
 *  (tier1)", "drawdown-kill (account route)", "owner toggle ON". Anything
 *  else (including a missing reason from a pre-R16 file) is UNKNOWN and is
 *  treated exactly like an owner kill: never auto-cleared. */
export function classifyLatchReason(reason: string | null | undefined): LatchKind {
  if (typeof reason !== "string") return "UNKNOWN";
  const r = reason.trim().toLowerCase();
  if (r.startsWith("drawdown-kill")) return "DRAWDOWN_KILL";
  if (r.startsWith("owner")) return "OWNER_MANUAL";
  return "UNKNOWN";
}

// ── Persisted latch record (backward-compatible file format) ───────────────

export interface KillSwitchRecord {
  killSwitch: boolean;
  reason: string | null;
  savedAtMs: number | null;
  lastAutoResumeAtMs: number | null;
}

function msFrom(v: unknown): number | null {
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  if (typeof v === "string" && v.length > 0) {
    const t = Date.parse(v);
    return Number.isFinite(t) ? t : null;
  }
  return null;
}

/** Parses voltrade_kill_switch.json. Accepts every historical shape:
 *  {killSwitch} (pre-R16), {killSwitch, reason, savedAt} (R16), and the
 *  current shape that adds lastAutoResumeAt. Null when `killSwitch` is not a
 *  boolean (same acceptance rule loadKillSwitch() has always used). */
export function parseKillSwitchRecord(raw: unknown): KillSwitchRecord | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  if (typeof o.killSwitch !== "boolean") return null;
  return {
    killSwitch: o.killSwitch,
    reason: typeof o.reason === "string" ? o.reason : null,
    savedAtMs: msFrom(o.savedAt),
    lastAutoResumeAtMs: msFrom(o.lastAutoResumeAt),
  };
}

// ── Broker payload parsing ─────────────────────────────────────────────────

export interface PositionRow {
  symbol: string;
  qty: number;              // signed: short < 0
  avgEntry: number | null;
  unrealizedPl: number | null;
  option: boolean;
}

export interface FillRow {
  symbol: string;
  dir: 1 | -1;              // buy = +1, sell / sell_short = -1
  qty: number;              // > 0
  price: number;            // > 0, per share / per contract-share
  t: number;                // epoch ms
}

function num(v: unknown): number {
  return typeof v === "number" ? v : parseFloat(String(v ?? ""));
}

/** /v2/positions -> rows. Null when the payload is not an array (a failed or
 *  garbage read — never treated as "no positions"). */
export function parsePositions(raw: unknown): PositionRow[] | null {
  if (!Array.isArray(raw)) return null;
  const rows: PositionRow[] = [];
  for (const p of raw) {
    if (!p || typeof p !== "object") continue;
    const o = p as Record<string, unknown>;
    const symbol = String(o.symbol ?? "");
    const q = num(o.qty);
    if (!symbol || !Number.isFinite(q) || q === 0) continue;
    const qty = o.side === "short" ? -Math.abs(q) : o.side === "long" ? Math.abs(q) : q;
    const avg = num(o.avg_entry_price);
    const upl = num(o.unrealized_pl);
    rows.push({
      symbol,
      qty,
      avgEntry: Number.isFinite(avg) && avg > 0 ? avg : null,
      unrealizedPl: Number.isFinite(upl) ? upl : null,
      option: isOptionPosition(symbol, typeof o.asset_class === "string" ? o.asset_class : undefined),
    });
  }
  return rows;
}

/** /v2/account/activities/FILL -> rows (malformed rows counted, skipped).
 *  Null when the payload is not an array. */
export function parseFills(raw: unknown): { rows: FillRow[]; malformed: number } | null {
  if (!Array.isArray(raw)) return null;
  const rows: FillRow[] = [];
  let malformed = 0;
  for (const f of raw) {
    const o = (f && typeof f === "object" ? f : {}) as Record<string, unknown>;
    const symbol = String(o.symbol ?? "");
    const side = String(o.side ?? "").toLowerCase();
    const qty = Math.abs(num(o.qty));
    const price = num(o.price);
    const t = msFrom(o.transaction_time);
    const dir = side === "buy" ? 1 : side === "sell" || side === "sell_short" ? -1 : 0;
    if (!symbol || dir === 0 || !(qty > 0) || !(price > 0) || t === null) {
      malformed++;
      continue;
    }
    rows.push({ symbol, dir: dir as 1 | -1, qty, price, t });
  }
  rows.sort((a, b) => a.t - b.t);
  return { rows, malformed };
}

// ── Realized P&L of the fill window ────────────────────────────────────────

export interface RealizedResult {
  realized: number;
  /** closing quantity matched against a cost basis we do not know (the
   *  position predates the window and is no longer held) — its P&L is NOT
   *  in `realized`, and the reconciliation is incomplete when > 0 */
  unpricedCloses: number;
  symbols: number;
}

/**
 * Average-cost realized P&L over a fill window. Each symbol's position at
 * the window start is inferred as (current qty - net filled qty in window);
 * its starting cost basis is the CURRENT avg_entry_price when the position is
 * still held on the same side (an approximation — stated), otherwise unknown,
 * and closes against an unknown basis are counted in `unpricedCloses`
 * instead of being guessed. Options use the standard x100 multiplier.
 */
export function realizedPnlFromFills(fills: readonly FillRow[], positions: readonly PositionRow[]): RealizedResult {
  const bySymbol = new Map<string, FillRow[]>();
  for (const f of fills) {
    const arr = bySymbol.get(f.symbol);
    if (arr) arr.push(f); else bySymbol.set(f.symbol, [f]);
  }
  const held = new Map(positions.map((p) => [p.symbol, p] as const));
  let realized = 0;
  let unpricedCloses = 0;
  for (const [symbol, list] of bySymbol) {
    const mult = isOptionPosition(symbol) ? 100 : 1;
    const cur = held.get(symbol);
    const netFilled = list.reduce((s, f) => s + f.dir * f.qty, 0);
    let pos = (cur ? cur.qty : 0) - netFilled;
    if (Math.abs(pos) < 1e-9) pos = 0;
    let avg: number | null =
      pos !== 0 && cur && cur.avgEntry !== null && Math.sign(cur.qty) === Math.sign(pos) ? cur.avgEntry : null;
    for (const f of list) {
      if (pos === 0 || Math.sign(pos) === f.dir) {
        // opening / adding: blend the average (unknown stays unknown when adding
        // to a pre-window position whose basis we do not have)
        const absPos = Math.abs(pos);
        avg = pos === 0 ? f.price : avg === null ? null : (absPos * avg + f.qty * f.price) / (absPos + f.qty);
        pos += f.dir * f.qty;
        continue;
      }
      const closing = Math.min(f.qty, Math.abs(pos));
      if (avg === null) unpricedCloses += closing;
      else realized += closing * (f.price - avg) * Math.sign(pos) * mult;
      pos += f.dir * closing;
      if (Math.abs(pos) < 1e-9) pos = 0;
      const rest = f.qty - closing;
      if (rest > 1e-9) { pos = f.dir * rest; avg = f.price; }
      else if (pos === 0) avg = null;
    }
  }
  return { realized: Math.round(realized * 100) / 100, unpricedCloses, symbols: bySymbol.size };
}

// ── Reconciliation: is the reported drop real? ─────────────────────────────

export type Classification = "NO_DRAWDOWN" | "DATA_ANOMALY" | "REAL_LOSS" | "UNRECONCILED";

export interface Reconciliation {
  classification: Classification;
  reportedDrop: number;         // peak - equity (dollars)
  unrealizedPl: number;
  realizedPl: number;
  explainedLoss: number;        // max(0, -(unrealized + realized))
  explainedRatio: number | null;
  incompleteBecause: string[];
}

export function reconcileDrawdown(args: {
  equity: number;
  peak: number;
  positions: readonly PositionRow[];
  fills: readonly FillRow[];
  fillsTruncated: boolean;
  malformedFills: number;
}): Reconciliation {
  const reportedDrop = Math.max(0, args.peak - args.equity);
  let unrealizedPl = 0;
  let unpricedPositions = 0;
  for (const p of args.positions) {
    if (p.unrealizedPl === null) unpricedPositions++;
    else unrealizedPl += p.unrealizedPl;
  }
  const r = realizedPnlFromFills(args.fills, args.positions);
  const explainedLoss = Math.max(0, -(unrealizedPl + r.realized));
  const incompleteBecause: string[] = [];
  if (args.fillsTruncated) incompleteBecause.push("fill history truncated at the page cap");
  if (args.malformedFills > 0) incompleteBecause.push(`${args.malformedFills} malformed fill row(s)`);
  if (r.unpricedCloses > 0) incompleteBecause.push(`${r.unpricedCloses} closed qty with a pre-window cost basis`);
  if (unpricedPositions > 0) incompleteBecause.push(`${unpricedPositions} position(s) without unrealized P&L`);
  const explainedRatio = reportedDrop > 0 ? explainedLoss / reportedDrop : null;
  let classification: Classification;
  if (reportedDrop <= 0) classification = "NO_DRAWDOWN";
  else if (incompleteBecause.length > 0) classification = "UNRECONCILED";
  else classification = explainedRatio! < AUTO_RESUME.ANOMALY_EXPLAINED_RATIO ? "DATA_ANOMALY" : "REAL_LOSS";
  return {
    classification,
    reportedDrop: Math.round(reportedDrop * 100) / 100,
    unrealizedPl: Math.round(unrealizedPl * 100) / 100,
    realizedPl: r.realized,
    explainedLoss: Math.round(explainedLoss * 100) / 100,
    explainedRatio: explainedRatio === null ? null : Math.round(explainedRatio * 10000) / 10000,
    incompleteBecause,
  };
}

// ── The decision ───────────────────────────────────────────────────────────

export type ResumeBasis = "RECOVERED" | "DATA_ANOMALY" | "PAPER_REBASE";

export type EvalOutcome =
  | "NOT_LATCHED" | "INELIGIBLE" | "MARKET_CLOSED" | "INVALID_READ" | "EVALUATED";

export interface AutoResumeInput {
  nowMs: number;
  enabled?: boolean;                 // autoResumeEnabled(process.env); default true
  paper: boolean;                    // allPaperAlpacaUrls(...) of the live process
  killSwitch: boolean;
  latchReason: string | null;
  latchedAtMs: number | null;        // kill-file savedAt
  firstObservedAtMs: number;         // when THIS process first saw the latch (fallback anchor)
  lastAutoResumeAtMs: number | null;
  marketOpen: boolean | null;        // null = clock read failed
  account: unknown;                  // raw /v2/account (null = fetch failed)
  positions: unknown;                // raw /v2/positions (null = fetch failed)
  fills: { rows: unknown; truncated: boolean } | null;
  equityPeak: number;
  maxDrawdownPct: number;            // e.g. -10
}

export interface AutoResumeNumbers {
  equity: number;
  peak: number;
  drawdownPct: number;
  resumeAbovePct: number;
  latchedMinutes: number;
  marketHoursSinceLatch: number;
  rebaseAfterMarketHours: number;
  reportedDrop: number;
  unrealizedPl: number;
  realizedPl: number;
  explainedLoss: number;
  explainedRatio: number | null;
}

export interface AutoResumeDecision {
  outcome: EvalOutcome;
  eligible: boolean;
  classification: Classification | null;
  basis: ResumeBasis | null;
  conditionOk: boolean;
  consecutiveOk: number;
  resume: boolean;
  /** set only on a resume whose basis re-baselines the peak */
  rebaselinePeakTo: number | null;
  reason: string;
  numbers: AutoResumeNumbers | null;
}

/** Rollback lever, no deploy needed: VOLTRADE_KILL_AUTO_RESUME=off (or
 *  0/false/disabled/no) restores the pre-2026-09-28 owner-only clear
 *  semantics. Unset = enabled (the human's directive). */
export const AUTO_RESUME_ENV = "VOLTRADE_KILL_AUTO_RESUME";
export function autoResumeEnabled(env: Record<string, string | undefined>): boolean {
  const v = (env[AUTO_RESUME_ENV] ?? "").trim().toLowerCase();
  return !["off", "0", "false", "disabled", "no"].includes(v);
}

/** The network-free part of eligibility, so the caller can skip broker reads
 *  entirely for a latch that can never auto-clear. */
export function autoResumeEligibility(a: { paper: boolean; killSwitch: boolean; latchReason: string | null; enabled?: boolean }):
  { eligible: boolean; outcome: EvalOutcome; reason: string } {
  if (!a.killSwitch) return { eligible: false, outcome: "NOT_LATCHED", reason: "kill switch is off" };
  if (a.enabled === false) {
    return { eligible: false, outcome: "INELIGIBLE", reason: `auto-resume disabled via ${AUTO_RESUME_ENV} — only the owner /api/bot/kill toggle clears the latch` };
  }
  if (!a.paper) {
    return { eligible: false, outcome: "INELIGIBLE", reason: "Alpaca base URL is not verifiably the PAPER endpoint — auto-resume never applies" };
  }
  const kind = classifyLatchReason(a.latchReason);
  if (kind === "OWNER_MANUAL") {
    return { eligible: false, outcome: "INELIGIBLE", reason: "owner-set manual kill — only the owner /api/bot/kill toggle clears it" };
  }
  if (kind !== "DRAWDOWN_KILL") {
    return { eligible: false, outcome: "INELIGIBLE", reason: `latch reason ${JSON.stringify(a.latchReason)} is not a drawdown kill — treated like a manual kill, never auto-cleared` };
  }
  return { eligible: true, outcome: "EVALUATED", reason: "drawdown-kill latch on the paper account" };
}

function fmtPct(n: number): string { return `${n.toFixed(2)}%`; }

export function evaluateAutoResume(input: AutoResumeInput, prev: { consecutiveOk: number }): AutoResumeDecision {
  const base = { classification: null, basis: null, conditionOk: false, consecutiveOk: 0, resume: false, rebaselinePeakTo: null, numbers: null };
  const el = autoResumeEligibility(input);
  if (!el.eligible) return { ...base, outcome: el.outcome, eligible: false, reason: el.reason };

  if (input.marketOpen === null) {
    return { ...base, outcome: "INVALID_READ", eligible: true, reason: "clock read failed — evaluation does not count" };
  }
  if (!input.marketOpen) {
    return { ...base, outcome: "MARKET_CLOSED", eligible: true, reason: "market closed — evaluations count only on tier-1 (market-hours) cycles" };
  }

  const acct = (input.account && typeof input.account === "object" ? input.account : null) as Record<string, unknown> | null;
  if (!acct) return { ...base, outcome: "INVALID_READ", eligible: true, reason: "account read failed — evaluation does not count" };
  if (acct.trading_blocked === true || acct.account_blocked === true ||
      (typeof acct.status === "string" && acct.status !== "ACTIVE")) {
    return { ...base, outcome: "INVALID_READ", eligible: true, reason: `account not tradable (status=${String(acct.status)}, trading_blocked=${String(acct.trading_blocked)}) — evaluation does not count` };
  }
  const dd = evaluateDrawdown(acct.equity, input.equityPeak, input.maxDrawdownPct);
  if (!dd.valid) {
    return { ...base, outcome: "INVALID_READ", eligible: true, reason: `equity read not credible (${JSON.stringify(acct.equity)}) — evaluation does not count` };
  }
  const positions = parsePositions(input.positions);
  if (!positions) return { ...base, outcome: "INVALID_READ", eligible: true, reason: "positions read failed — evaluation does not count" };
  const fills = input.fills ? parseFills(input.fills.rows) : null;
  if (!input.fills || !fills) return { ...base, outcome: "INVALID_READ", eligible: true, reason: "fills read failed — evaluation does not count" };

  const equity = dd.equity!;
  const peak = dd.newPeak;
  const drawdownPct = dd.drawdownPct!;
  const rec = reconcileDrawdown({
    equity, peak, positions, fills: fills.rows,
    fillsTruncated: input.fills.truncated, malformedFills: fills.malformed,
  });

  const latchedAt = input.latchedAtMs ?? input.firstObservedAtMs;
  const latchedMinutes = Math.max(0, (input.nowMs - latchedAt) / 60_000);
  const marketHoursSinceLatch = marketHoursBetween(latchedAt, input.nowMs);
  const rebaseAfterMarketHours = AUTO_RESUME.PAPER_REBASE_AFTER_TRADING_DAYS * AUTO_RESUME.MARKET_HOURS_PER_TRADING_DAY;
  const resumeAbovePct = input.maxDrawdownPct + AUTO_RESUME.RECOVERY_HYSTERESIS_PP;
  const antiFlap =
    input.lastAutoResumeAtMs !== null &&
    latchedAt >= input.lastAutoResumeAtMs &&
    latchedAt - input.lastAutoResumeAtMs < AUTO_RESUME.ANTI_FLAP_HOURS * 3_600_000;

  const numbers: AutoResumeNumbers = {
    equity, peak,
    drawdownPct: Math.round(drawdownPct * 100) / 100,
    resumeAbovePct,
    latchedMinutes: Math.round(latchedMinutes * 10) / 10,
    marketHoursSinceLatch: Math.round(marketHoursSinceLatch * 10) / 10,
    rebaseAfterMarketHours,
    reportedDrop: rec.reportedDrop, unrealizedPl: rec.unrealizedPl, realizedPl: rec.realizedPl,
    explainedLoss: rec.explainedLoss, explainedRatio: rec.explainedRatio,
  };

  let basis: ResumeBasis | null = null;
  let why: string;
  if (drawdownPct > resumeAbovePct) {
    basis = "RECOVERED";
    why = `drawdown ${fmtPct(drawdownPct)} is back above ${fmtPct(resumeAbovePct)} (kill ${fmtPct(input.maxDrawdownPct)} + ${AUTO_RESUME.RECOVERY_HYSTERESIS_PP}pp hysteresis)`;
  } else if (rec.classification === "DATA_ANOMALY" && !antiFlap) {
    basis = "DATA_ANOMALY";
    why = `positions + fills explain ${fmtPct((rec.explainedRatio ?? 0) * 100)} of the reported drop (< ${fmtPct(AUTO_RESUME.ANOMALY_EXPLAINED_RATIO * 100)})`;
  } else if (marketHoursSinceLatch >= rebaseAfterMarketHours) {
    basis = "PAPER_REBASE";
    why = `latched ${numbers.marketHoursSinceLatch} market hours (>= ${AUTO_RESUME.PAPER_REBASE_AFTER_TRADING_DAYS} market days) with a ${rec.classification} reading — paper re-baseline`;
  } else {
    why = rec.classification === "DATA_ANOMALY" && antiFlap
      ? `anomaly fast path disabled: this latch began < ${AUTO_RESUME.ANTI_FLAP_HOURS}h after the previous auto-resume; waiting for recovery or the ${AUTO_RESUME.PAPER_REBASE_AFTER_TRADING_DAYS}-market-day paper re-baseline`
      : `${rec.classification}${rec.incompleteBecause.length ? ` (${rec.incompleteBecause.join("; ")})` : ""}: drawdown ${fmtPct(drawdownPct)} not above ${fmtPct(resumeAbovePct)}; paper re-baseline after ${rebaseAfterMarketHours} market hours (${numbers.marketHoursSinceLatch} so far)`;
  }

  const cooldownOk = latchedMinutes >= AUTO_RESUME.COOLDOWN_MINUTES;
  const conditionOk = basis !== null && cooldownOk;
  const consecutiveOk = conditionOk ? prev.consecutiveOk + 1 : 0;
  const resume = consecutiveOk >= AUTO_RESUME.CONSECUTIVE_OK_REQUIRED;
  const reason = basis !== null && !cooldownOk
    ? `${why}; cooldown ${numbers.latchedMinutes}/${AUTO_RESUME.COOLDOWN_MINUTES} min`
    : basis !== null
      ? `${why}; ${consecutiveOk}/${AUTO_RESUME.CONSECUTIVE_OK_REQUIRED} consecutive evaluations`
      : why;

  return {
    outcome: "EVALUATED",
    eligible: true,
    classification: rec.classification,
    basis,
    conditionOk,
    consecutiveOk,
    resume,
    rebaselinePeakTo: resume && (basis === "DATA_ANOMALY" || basis === "PAPER_REBASE") ? equity : null,
    reason,
    numbers,
  };
}

/** One audit-line summary of paper_resume_sync.rebaseline_python_halts()'s
 *  report (or of its failure). Never throws on an unexpected shape — a
 *  malformed report is itself worth stating in the audit line. */
export function summarizePythonSync(ok: boolean, result: unknown, via: string): string {
  const r = (result && typeof result === "object" ? result : null) as Record<string, unknown> | null;
  if (!ok || !r || (typeof r.error === "string" && !("bot_engine_dd" in r))) {
    const err = r && typeof r.error === "string" ? r.error.slice(0, 160) : "no report";
    return `python halts: sync FAILED (via ${via}: ${err}) — bot_engine's DD halt / risk_kill_switch may still block entries; watch for DD-HALT / TIER-KILL audit lines`;
  }
  const obj = (v: unknown) => (v && typeof v === "object" ? (v as Record<string, unknown>) : {});
  const dd = obj(r.bot_engine_dd);
  const before = obj(dd.before);
  const after = obj(dd.after);
  const ks = obj(r.risk_kill_switch);
  const shared = obj(r.shared_peak);
  const parts = [
    `python halts (${r.applied === true ? "applied" : "report-only"}):`,
    `bot_engine DD halted=${String(before.halted)} peak=${String(before.peak_equity)}` +
      (dd.rebaselined === true ? ` -> ${String(after.peak_equity)} (re-baselined, halt cleared)` : ""),
    `risk_kill_switch killed=${String(ks.killed)}${ks.reset === true ? " -> reset via reset_kill_state()" : ""}` +
      (ks.manual_kill_file === true ? " (MANUAL kill file present — untouched)" : ""),
    `shared tiered peak=${String(shared.peak_equity)} (never lowered; dd vs equity ${String(shared.drawdown_pct_vs_equity)}%)`,
  ];
  const notes = Array.isArray(r.notes) ? r.notes.filter((n): n is string => typeof n === "string") : [];
  if (notes.length) parts.push(`notes: ${notes.join(" | ").slice(0, 400)}`);
  return parts.join(" ");
}

/** Bounded fill-history window start: FILLS_LOOKBACK_DAYS before the latch. */
export function fillsWindowStartMs(latchedAtMs: number): number {
  return latchedAtMs - AUTO_RESUME.FILLS_LOOKBACK_DAYS * 86_400_000;
}
