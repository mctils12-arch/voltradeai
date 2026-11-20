// Bot page — the evidence views ([PRODUCT] 2026-11-20, UI coverage batch A;
// CLAUDE.md UI INTEGRATION AND ONE ORDER PATH). Six reads the sessions had
// behind the diag token and the page did not show: the closed-lot ledger, the
// order history (the price ladder's steps), the base holding against its own
// rule, what each blocking rule prevented, what the regime engine changed
// against the old rule, and which option quote feed the account may read.
//
// Pure; pinned by evidenceView.test.ts. Every route is read tolerantly, like
// whyView.ts: a server that predates a route, or answers in a shape this page
// does not recognise, gets a designed sentence that says so — never a blank, a
// zero or a guess. Nothing is re-derived that the server already states: the
// sentences quote the server's own fields; the only arithmetic here is adding
// the regime report's daily COUNTS into a period total and a held-time span.
import { fmtMoney, fmtMonDay, fmtOcc, fmtPct, fmtSigned } from "./format";

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);
const numOf = (v: unknown): number | null => (fin(v) ? v : typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v)) ? Number(v) : null);
const obj = (v: unknown): Record<string, unknown> | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null);
const ms = (iso: string | null): number | null => { if (!iso) return null; const t = Date.parse(iso); return Number.isFinite(t) ? t : null; };
/** "Nov 19" for a calendar date as written ("2026-11-19"), never shifted a day by a clock */
const day = (v: string | null | undefined): string => (typeof v === "string" && /^\d{4}-\d{2}-\d{2}/.test(v) ? fmtMonDay(`${v.slice(0, 10)}T12:00:00Z`) : "—");
const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;

// ── closed lots ─────────────────────────────────────────────────────────────
export interface LotView {
  key: string; symbol: string; label: string; option: boolean; side: "long" | "short";
  qty: number | null; openedAt: string | null; closedAt: string; basis: number | null; pnl: number | null; returnPct: number | null;
  entryPrice: number | null; exitPrice: number | null;
  /** how long the lot was held, in ms; null when its opening fill predates the fetched history */
  heldMs: number | null;
}
export interface LotsView {
  state: "ready" | "pending" | "empty";
  /** newest first */
  rows: LotView[];
  /** lots in the whole ledger (the page holds the newest `rows.length`) */
  count: number; truncated: boolean; since: string | null; fetchedAt: string | null; lastError: string | null;
}
/** GET /api/bot/lots → newest first; null when the body is not the ledger's shape. */
export function parseLots(raw: unknown): LotsView | null {
  const o = obj(raw);
  if (!o || !Array.isArray(o.rows)) return null;
  const rows: LotView[] = [];
  o.rows.forEach((r, i) => {
    const x = obj(r);
    const symbol = x ? str(x.symbol) : null;
    const closedAt = x ? str(x.closedAt) : null;
    if (!x || !symbol || !closedAt || ms(closedAt) === null) return;
    const openedAt = str(x.openedAt);
    const o0 = ms(openedAt), c0 = ms(closedAt);
    rows.push({
      key: `${symbol}|${closedAt}|${i}`, symbol, label: fmtOcc(symbol), option: x.option === true, side: x.side === "short" ? "short" : "long",
      qty: numOf(x.qty), openedAt, closedAt, basis: numOf(x.basis), pnl: numOf(x.pnl), returnPct: numOf(x.returnPct),
      entryPrice: numOf(x.entryPrice), exitPrice: numOf(x.exitPrice),
      heldMs: o0 !== null && c0 !== null && c0 >= o0 ? c0 - o0 : null,
    });
  });
  rows.sort((a, b) => Date.parse(b.closedAt) - Date.parse(a.closedAt));
  const count = numOf(o.count) ?? rows.length;
  return {
    state: o.pending === true && !rows.length ? "pending" : rows.length ? "ready" : "empty",
    rows, count, truncated: o.truncated === true || count > rows.length, since: str(o.since), fetchedAt: str(o.fetchedAt), lastError: str(o.lastError),
  };
}
/** "42m" · "3h" · "12d": how long a lot was held; "—" when the open predates the history. */
export function heldWords(heldMs: number | null): string {
  if (heldMs === null || !Number.isFinite(heldMs) || heldMs < 0) return "—";
  const m = Math.round(heldMs / 60_000);
  if (m < 60) return `${Math.max(1, m)}m`;
  const h = Math.round(m / 60);
  if (h < 48) return `${h}h`;
  return `${Math.round(h / 24)}d`;
}
/** "sold 3 → bought back" for a short, "bought 82 → sold" for a long: what the lot did, in the order it happened. */
export function lotSideWords(l: Pick<LotView, "side" | "qty" | "option">): string {
  const q = l.qty === null ? "" : ` ${l.qty}${l.option ? " ct" : ""}`;
  return l.side === "short" ? `sold${q}, bought back` : `bought${q}, sold`;
}
/** The caption under the table: how many are shown of how many, the window, the age. */
export function lotsCaption(v: LotsView, shown: number): string {
  const of = v.count > shown ? `the newest ${shown.toLocaleString("en-US")} of ${v.count.toLocaleString("en-US")} closed lots` : plural(v.count, "closed lot");
  return `${of}${v.since ? ` since ${day(v.since)}` : ""} · Alpaca fills, average cost · a sold option's cost is its credit`;
}
const csvCell = (v: string | number | null): string => {
  if (v === null) return "";
  const s = String(v);
  // a cell that a spreadsheet would run as a formula is written as text
  const safe = /^[=+\-@\t\r]/.test(s) && typeof v === "string" ? `'${s}` : s;
  return /[",\n\r]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
};
export const LOTS_CSV_HEADER = ["closed_at", "opened_at", "symbol", "option", "side", "qty", "entry_price", "exit_price", "basis_usd", "pnl_usd", "return_pct"] as const;
/** The loaded lots as CSV (RFC 4180 quoting, CRLF rows), newest first: the ledger's own fields, nothing rounded. */
export function lotsCsv(rows: readonly LotView[]): string {
  const lines = [LOTS_CSV_HEADER.join(",")];
  for (const l of rows) lines.push([l.closedAt, l.openedAt, l.symbol, l.option ? "yes" : "no", l.side, l.qty, l.entryPrice, l.exitPrice, l.basis, l.pnl, l.returnPct].map(csvCell).join(","));
  return `${lines.join("\r\n")}\r\n`;
}

// ── order history ───────────────────────────────────────────────────────────
export interface OrderView {
  id: string; symbol: string; label: string; option: boolean; side: string; type: string;
  qty: number | null; filledQty: number | null; limit: number | null; fillPrice: number | null;
  status: string; submittedAt: string; endedAt: string | null;
  /** this order's place among the orders for the same contract and side on the same ET day, oldest = 1 */
  step: number; steps: number;
  /** the limit moved against the step before it: +0.01 = a cent more was offered / asked; null on step 1 */
  moved: number | null;
}
const etDay = (iso: string): string => new Date(iso).toLocaleDateString("en-CA", { timeZone: "America/New_York" });
/** GET /api/bot/order-history → newest first, each order with its step in that day's attempts; null on an unknown shape. */
export function parseOrderHistory(raw: unknown): OrderView[] | null {
  const o = obj(raw);
  const list = Array.isArray(raw) ? raw : o && Array.isArray(o.orders) ? o.orders : null;
  if (!list) return null;
  const rows: OrderView[] = [];
  for (const r of list) {
    const x = obj(r);
    const symbol = x ? str(x.symbol) : null;
    const submittedAt = x ? str(x.submitted_at) : null;
    if (!x || !symbol || !submittedAt || ms(submittedAt) === null) continue;
    rows.push({
      id: str(x.id) ?? `${symbol}|${submittedAt}`, symbol, label: fmtOcc(symbol), option: (str(x.asset_class) ?? "").includes("option") || fmtOcc(symbol) !== symbol,
      side: (str(x.side) ?? "").toLowerCase(), type: (str(x.type) ?? "").toLowerCase(),
      qty: numOf(x.qty), filledQty: numOf(x.filled_qty), limit: numOf(x.limit_price), fillPrice: numOf(x.filled_avg_price),
      status: (str(x.status) ?? "unknown").toLowerCase(), submittedAt, endedAt: str(x.filled_at) ?? str(x.canceled_at), step: 1, steps: 1, moved: null,
    });
  }
  const groups = new Map<string, OrderView[]>();
  for (const r of rows) { const k = `${r.symbol}|${r.side}|${etDay(r.submittedAt)}`; const g = groups.get(k); if (g) g.push(r); else groups.set(k, [r]); }
  for (const g of groups.values()) {
    g.sort((a, b) => Date.parse(a.submittedAt) - Date.parse(b.submittedAt));
    g.forEach((r, i) => { r.step = i + 1; r.steps = g.length; const prev = i ? g[i - 1].limit : null; r.moved = prev !== null && r.limit !== null ? Math.round((r.limit - prev) * 10000) / 10000 : null; });
  }
  return rows.sort((a, b) => Date.parse(b.submittedAt) - Date.parse(a.submittedAt));
}
/** "filled at $0.38" · "cancelled after 12m, unfilled" · "working": how the order ended, in words. */
export function orderOutcome(o: OrderView): { text: string; tone: "ok" | "wait" | "none" } {
  const partial = o.filledQty !== null && o.filledQty > 0 && o.qty !== null && o.filledQty < o.qty;
  if (o.status === "filled" || (o.filledQty !== null && o.qty !== null && o.filledQty >= o.qty && o.qty > 0)) return { text: o.fillPrice !== null ? `filled at $${o.fillPrice.toFixed(2)}` : "filled", tone: "ok" };
  const span = o.endedAt && ms(o.endedAt) !== null ? heldWords(Date.parse(o.endedAt) - Date.parse(o.submittedAt)) : null;
  if (o.status === "canceled" || o.status === "cancelled" || o.status === "expired" || o.status === "replaced") {
    const word = o.status === "expired" ? "expired" : o.status === "replaced" ? "replaced" : "cancelled";
    return { text: `${word}${span && span !== "—" ? ` after ${span}` : ""}, ${partial ? `${o.filledQty} of ${o.qty} filled` : "unfilled"}`, tone: "none" };
  }
  if (o.status === "rejected") return { text: "rejected by the broker", tone: "none" };
  if (o.status === "partially_filled" || partial) return { text: `working · ${o.filledQty ?? 0} of ${o.qty ?? "?"} filled`, tone: "wait" };
  return { text: o.status === "new" || o.status === "accepted" || o.status === "pending_new" ? "working" : o.status.replace(/_/g, " "), tone: "wait" };
}
/** "sell 1 · limit $0.30" */
export function orderAsk(o: OrderView): string {
  const q = o.qty === null ? "" : ` ${o.qty}${o.option ? " ct" : ""}`;
  return `${o.side || "order"}${q} · ${o.type || "order"}${o.limit !== null ? ` $${o.limit.toFixed(2)}` : ""}`;
}
/** "step 2 of 3 · +$0.01" for a contract tried more than once that day; null for a single attempt. */
export function orderStepWords(o: OrderView): string | null {
  if (o.steps < 2) return null;
  const mv = o.moved === null || o.moved === 0 ? (o.step > 1 && o.moved === 0 ? " · same price" : "") : ` · ${o.moved > 0 ? "+" : "−"}$${Math.abs(o.moved).toFixed(2)}`;
  return `step ${o.step} of ${o.steps}${mv}`;
}
/** One line for the panel's summary: how many of the loaded orders filled. */
export function orderHistorySummary(rows: readonly OrderView[]): string {
  if (!rows.length) return "no orders in the broker's recent history";
  const filled = rows.filter((r) => orderOutcome(r).tone === "ok").length;
  const oldest = rows[rows.length - 1];
  return `${plural(rows.length, "order")} since ${fmtMonDay(oldest.submittedAt)} · ${filled} filled · ${rows.length - filled} not`;
}

// ── the base holding against its own rule (floor tracking) ──────────────────
export interface FloorSessionView { day: string; entry: boolean; spyPct: number | null; target: number | null; held: number | null; drift: number | null; rulePct: number | null; floorPct: number | null; trackingPp: number | null; rebalance: boolean; reason: string | null }
export interface FloorView {
  ok: boolean; reason: string | null; mode: string | null; base: string | null; levered: string | null; cashFund: string | null; band: number | null; capitalPct: number | null;
  generatedAt: string | null; latest: FloorSessionView | null; sessions: FloorSessionView[];
  rolling: { sessions: number; windowSessions: number | null; floorPct: number | null; rulePct: number | null; trackingPp: number | null; from: string | null; to: string | null; complete: boolean } | null;
  entry: { day: string; shortfallPp: number | null; note: string | null } | null;
  /** status "met" = the pre-registered rollback trigger is met; tracking is its measurable leg (met / not_met / not_enough_history / not_evaluable) */
  trigger: { text: string | null; status: string | null; tracking: string | null; summary: string | null; limitPp: number | null } | null;
  turnover: { sessions: number | null; orders: number | null; filled: number | null; dollars: number | null; rebalanceSessions: number | null; outsideBandNoTrade: number | null } | null;
  definitions: Array<[string, string]>;
}
function floorSession(v: unknown): FloorSessionView | null {
  const x = obj(v);
  const d = x ? str(x.day) : null;
  if (!x || !d) return null;
  return { day: d, entry: x.entry === true, spyPct: numOf(x.spy_pct), target: numOf(x.target_exposure), held: numOf(x.held_exposure), drift: numOf(x.drift), rulePct: numOf(x.rule_pct), floorPct: numOf(x.floor_pct), trackingPp: numOf(x.tracking_pp), rebalance: x.rebalance === true, reason: str(x.reason) };
}
/** GET /api/bot/floor-tracking → the view; null when the body is not the report's shape. */
export function parseFloorTracking(raw: unknown): FloorView | null {
  const o = obj(raw);
  if (!o || (!str(o.status) && !obj(o.latest) && !Array.isArray(o.sessions))) return null;
  const cfg = obj(o.config), roll = obj(o.rolling), ent = obj(o.entry), trig = obj(o.rollback_trigger), turn = obj(obj(o.turnover)?.window), core = obj(turn?.core), defs = obj(o.definitions);
  const trk = obj(trig?.tracking);
  return {
    ok: str(o.status) === "ok", reason: str(o.reason), mode: str(o.mode), base: str(cfg?.base), levered: str(cfg?.levered), cashFund: str(cfg?.cash_fund), band: numOf(cfg?.band), capitalPct: numOf(cfg?.capital_pct),
    generatedAt: str(o.generated_at), latest: floorSession(o.latest),
    sessions: Array.isArray(o.sessions) ? o.sessions.map(floorSession).filter((s): s is FloorSessionView => !!s) : [],
    rolling: roll && numOf(roll.sessions) !== null ? { sessions: numOf(roll.sessions) as number, windowSessions: numOf(roll.window_sessions), floorPct: numOf(roll.floor_pct), rulePct: numOf(roll.rule_pct), trackingPp: numOf(roll.tracking_pp), from: str(roll.from), to: str(roll.to), complete: roll.complete === true } : null,
    entry: ent && str(ent.day) ? { day: str(ent.day) as string, shortfallPp: numOf(ent.shortfall_pp), note: str(ent.note) } : null,
    trigger: trig ? { text: str(trig.text), status: str(trig.status), tracking: str(trk?.status), summary: str(trig.summary), limitPp: numOf(trk?.limit_pp) } : null,
    turnover: turn ? { sessions: numOf(turn.sessions), orders: numOf(core?.orders), filled: numOf(core?.filled), dollars: numOf(core?.dollars), rebalanceSessions: numOf(turn.rebalance_sessions), outsideBandNoTrade: numOf(turn.outside_band_no_trade) } : null,
    definitions: defs ? Object.entries(defs).map(([k, v]) => [k.replace(/_/g, " "), str(v) ?? ""] as [string, string]).filter(([, v]) => v) : [],
  };
}
const x2 = (v: number | null) => (v === null ? "—" : `${v.toFixed(2)}×`);
/** The section's one sentence: the last session against the rule, the exposure held against the target, the period. */
export function floorSentence(f: FloorView): string {
  if (!f.ok && !f.latest) return f.reason ? `No tracking report: ${f.reason}.` : "No tracking report yet.";
  const l = f.latest;
  if (!l) return "No measured session yet.";
  const name = f.base ?? "the base";
  const head = `${day(l.day)}: the ${name} holding ${fmtPct(l.floorPct)} against ${fmtPct(l.rulePct)} for its rule (${x2(l.target)} of ${name} ${fmtPct(l.spyPct)})`;
  // the server's own drift (|target − held|) against FLOOR_VM_BAND, and whether a core order filled that session
  const inBand = l.drift !== null && f.band !== null ? l.drift <= f.band + 1e-9 : null;
  const drift = l.held !== null && l.target !== null ? ` It went in holding ${x2(l.held)} against a target of ${x2(l.target)}${inBand === null ? "" : inBand ? `: inside the ${f.band!.toFixed(2)}× band, where the rule does not trade` : `: outside the ${f.band!.toFixed(2)}× band`}${l.rebalance ? "; it rebalanced that session" : inBand === false ? "; no rebalance filled that session" : ""}.` : "";
  const r = f.rolling;
  const roll = r && r.trackingPp !== null ? ` Over the last ${plural(r.sessions, "session")}: ${fmtSigned(r.trackingPp, 2, " pt")} against the rule.` : "";
  return `${head}.${drift}${roll}`;
}
/** The pre-registered rollback line and where it stands, in words. */
export function floorTriggerWords(f: FloorView): { text: string; tone: "ok" | "wait" | "warn" } | null {
  const t = f.trigger;
  if (!t) return null;
  const tone = t.status === "met" || t.tracking === "met" ? "warn" : t.tracking === "not_met" ? "ok" : "wait";
  const lead = tone === "warn" ? "Rollback trigger MET" : tone === "ok" ? "Tracking inside the rollback line" : "Rollback line not yet testable";
  return { text: `${lead}${t.summary ? `: ${t.summary}` : ""}`, tone };
}

// ── what each blocking rule prevented (gate counterfactuals) ────────────────
export const RULE_NAMES: Readonly<Record<string, string>> = { exposure_ceiling: "Exposure cap", daily_loss: "Daily loss limit", drawdown: "Drawdown kill (tier engine)", cash_reserve: "Cash reserve" };
export interface RuleGroupView {
  key: string; gate: string; name: string; tier: string; action: string; nameDays: number; first: string | null; last: string | null;
  /** the +5 trading-day reading: names judged so far, their mean return, the same days' scanned names */
  labeled: number; pending: number; meanPct: number | null; winRatePct: number | null; baseN: number | null; baseMeanPct: number | null;
  /** the dollar ESTIMATE of what the rule prevented (positive = it stopped a loss); null when no blocked name carried a size */
  preventionUsd: number | null; sized: number;
  verdict: "helped" | "cost" | "even" | "thin";
}
export interface RulesView { ok: boolean; error: string | null; minN: number; generatedAt: string | null; first: string | null; last: string | null; rules: Array<{ gate: string; name: string; what: string | null; nameDays: number; names: number }>; groups: RuleGroupView[]; limits: string[]; basis: string | null; horizon: string }
const HORIZON = "+5d";
/** GET /api/bot/rule-counterfactuals → one row per rule, tier and action, as the report groups them; null on an unknown shape. */
export function parseRuleCounterfactuals(raw: unknown): RulesView | null {
  const o = obj(raw);
  if (!o) return null;
  const err = str(o.error);
  const gates = obj(o.gates);
  if (!gates && !err) return null;
  const minN = numOf(o.min_n) ?? 5;
  const log = obj(o.log);
  const rules: RulesView["rules"] = [];
  const groups: RuleGroupView[] = [];
  for (const [gate, gv] of Object.entries(gates ?? {})) {
    const g = obj(gv);
    if (!g) continue;
    const name = RULE_NAMES[gate] ?? gate.replace(/_/g, " ");
    rules.push({ gate, name, what: str(g.what), nameDays: numOf(g.name_days) ?? 0, names: numOf(g.names) ?? 0 });
    for (const grp of Array.isArray(g.groups) ? g.groups : []) {
      const x = obj(grp);
      const h = obj(x?.[HORIZON]);
      if (!x || !h) continue;
      const ret = obj(h.return_pct), base = obj(h.base_scanned_same_days), usd = obj(h.dollars);
      const labeled = numOf(h.labeled) ?? 0;
      const meanPct = numOf(ret?.mean), baseMeanPct = numOf(base?.mean_return_pct);
      const sized = numOf(usd?.sized_positions) ?? 0;
      const verdict: RuleGroupView["verdict"] = labeled < minN || meanPct === null || baseMeanPct === null ? "thin" : meanPct < baseMeanPct - 0.25 ? "helped" : meanPct > baseMeanPct + 0.25 ? "cost" : "even";
      groups.push({
        key: `${gate}|${str(x.tier) ?? "?"}|${str(x.action) ?? "?"}`, gate, name, tier: str(x.tier) ?? "?", action: str(x.action) ?? "?", nameDays: numOf(x.name_days) ?? 0, first: str(x.first_day), last: str(x.last_day),
        labeled, pending: numOf(h.pending) ?? 0, meanPct, winRatePct: numOf(h.win_rate_pct), baseN: numOf(base?.n), baseMeanPct,
        preventionUsd: sized > 0 ? numOf(usd?.prevention_pnl_dollars) : null, sized, verdict,
      });
    }
  }
  groups.sort((a, b) => b.labeled - a.labeled || b.nameDays - a.nameDays);
  return {
    ok: !err, error: err, minN, generatedAt: str(o.generated_at), first: str(log?.first_day), last: str(log?.last_day), rules, groups,
    limits: Array.isArray(o.limits) ? o.limits.map(str).filter((x): x is string => !!x) : [], basis: str(o.dollar_basis), horizon: "5 trading days",
  };
}
const TIER_WORDS: Readonly<Record<string, string>> = { "1": "tier 1 puts", "2": "tier 2", "3": "tier 3", scanner: "scanner picks", unrecorded: "tier puts (tier not recorded)" };
/** "tier 1 puts, dropped" */
export function ruleGroupWords(g: Pick<RuleGroupView, "tier" | "action">): string {
  return `${TIER_WORDS[g.tier] ?? `tier ${g.tier}`}, ${g.action === "scaled" ? "cut in size" : g.action}`;
}
/** the cell under the column "blocked names did": against everything scanned on the same days */
export const RULE_VERDICT_WORDS: Readonly<Record<RuleGroupView["verdict"], string>> = { helped: "worse than the rest", cost: "better than the rest", even: "same as the rest", thin: "too few to judge" };
/** The section's one sentence: the best-measured rule, its blocked names against everything scanned the same days. */
export function rulesSentence(v: RulesView): string {
  if (!v.ok) return `The rule report could not be computed${v.error ? `: ${v.error}` : ""}.`;
  const total = v.rules.reduce((a, r) => a + r.nameDays, 0);
  if (!total) return "No rule has blocked a trade in the logged period.";
  const g = v.groups.find((x) => x.verdict !== "thin");
  const lead = `${plural(v.rules.filter((r) => r.nameDays > 0).length, "rule")} blocked ${plural(total, "name-day")}${v.first ? ` since ${day(v.first)}` : ""}.`;
  if (!g) return `${lead} None has ${v.minN} judged outcomes yet, so none can be called.`;
  const cmp = g.verdict === "helped" ? "worse than" : g.verdict === "cost" ? "better than" : "about the same as";
  return `${lead} Best measured: ${g.name.toLowerCase()} (${ruleGroupWords(g)}): the ${g.labeled} judged so far returned ${fmtPct(g.meanPct, 1)} on average over ${v.horizon}, ${cmp} the ${fmtPct(g.baseMeanPct, 1)} of everything scanned those days.`;
}

// ── what the regime engine changed against the old rule ─────────────────────
export interface RegimeCfView {
  days: number | null; since: string | null; rows: number;
  /** selection + options decisions compared, and how many came out differently */
  decisions: number; differs: number;
  /** entries the engine blocked, by guard, largest first */
  blocks: Array<{ guard: string; n: number }>; blocked: number;
  sizeCuts: number; relabelled: number;
  /** tier-1 sizing on the newest day that has it: the old rule's multiplier and the engine's */
  tier1: { date: string; legacy: number; engine: number; n: number } | null;
  note: string | null;
}
export const GUARD_WORDS: Readonly<Record<string, string>> = { rs: "weak against the market", knife: "falling knife", vrp: "option premium too thin" };
/** GET /api/bot/regime-counterfactual → period totals of its daily counts; null on an unknown shape. */
export function parseRegimeCounterfactual(raw: unknown): RegimeCfView | null {
  const o = obj(raw);
  if (!o || !Array.isArray(o.by_day)) return null;
  const guards = new Map<string, number>();
  let decisions = 0, differs = 0, blocked = 0, sizeCuts = 0, relabelled = 0;
  let tier1: RegimeCfView["tier1"] = null;
  for (const d of o.by_day) {
    const x = obj(d);
    if (!x) continue;
    for (const k of ["selection", "options"] as const) {
      const s = obj(x[k]);
      if (!s) continue;
      decisions += numOf(s.n) ?? 0; differs += numOf(s.differs) ?? 0; sizeCuts += numOf(s.size_cut) ?? 0;
      blocked += numOf(obj(s.decisions)?.block) ?? 0;
      for (const [g, n] of Object.entries(obj(s.blocks_by_guard) ?? {})) guards.set(g, (guards.get(g) ?? 0) + (numOf(n) ?? 0));
    }
    relabelled += numOf(obj(x.label)?.differs) ?? 0;
    const t1 = obj(obj(x.sizing)?.tier1);
    const date = str(x.date);
    if (t1 && date && numOf(t1.mean_legacy_multiplier) !== null && numOf(t1.mean_engine_multiplier) !== null && (!tier1 || date > tier1.date)) {
      tier1 = { date, legacy: numOf(t1.mean_legacy_multiplier) as number, engine: numOf(t1.mean_engine_multiplier) as number, n: numOf(t1.n) ?? 0 };
    }
  }
  return {
    days: numOf(o.days), since: str(o.since), rows: numOf(o.rows) ?? 0, decisions, differs, blocked, sizeCuts, relabelled, tier1,
    blocks: [...guards.entries()].map(([guard, n]) => ({ guard, n })).filter((b) => b.n > 0).sort((a, b) => b.n - a.n), note: str(o.note),
  };
}
/** One line for the regime section: counts only — the report carries no P&L, and the line says so. */
export function regimeCfSentence(v: RegimeCfView): string {
  const span = v.days !== null ? `the last ${v.days} days` : v.since ? `since ${day(v.since)}` : "the logged period";
  if (!v.decisions && !v.relabelled && !v.tier1) return `Against the old rule, ${span}: no decision point logged yet.`;
  const parts: string[] = [];
  if (v.decisions) parts.push(`${v.differs.toLocaleString("en-US")} of ${plural(v.decisions, "entry decision")} came out differently`);
  if (v.blocked) parts.push(`${plural(v.blocked, "entry", "entries")} blocked${v.blocks.length ? ` (${v.blocks.slice(0, 3).map((b) => `${GUARD_WORDS[b.guard] ?? b.guard} ${b.n}`).join(", ")}; one entry can trip several)` : ""}`);
  if (v.sizeCuts) parts.push(`${plural(v.sizeCuts, "option entry", "option entries")} cut in size`);
  if (v.tier1) parts.push(`tier 1 sized at ${v.tier1.engine.toFixed(2)}× against ${v.tier1.legacy.toFixed(2)}× under the old rule on ${day(v.tier1.date)}`);
  return `Against the old rule, ${span}: ${parts.join("; ")}. Counts only: what the blocked names went on to do is not in this report.`;
}

// ── which option quote feed the account may read ────────────────────────────
export interface OptionsFeedWords { feed: "opra" | "indicative" | "none" | "unknown"; text: string; tone: "ok" | "warn" | "none"; checkedAt: string | null }
/** GET /api/bot/options-feed → the line under the Book's options; null on an unknown shape. */
export function parseOptionsFeed(raw: unknown): OptionsFeedWords | null {
  const o = obj(raw);
  const feed = o ? str(o.feed) : null;
  if (!o || !feed) return null;
  const checkedAt = str(o.checked_at);
  if (feed === "opra") return { feed, text: "option quotes: the exchanges' own feed (OPRA)", tone: "ok", checkedAt };
  if (feed === "indicative") return { feed, text: "option quotes are INDICATIVE: Alpaca's indicative feed, not the exchanges' quotes (the account is not entitled to OPRA). An option's last price, spread and P&L here can differ from what an order would get", tone: "warn", checkedAt };
  if (feed === "none") return { feed, text: "no option quote feed answers for this account: option prices here are the broker's position marks only", tone: "warn", checkedAt };
  return { feed: "unknown", text: "which option quote feed this account may read could not be checked just now", tone: "none", checkedAt };
}

/** A money figure with its sign, for the rules table ("+$1,240" = the rule stopped that loss). */
export const signedUsd = (v: number): string => (v > 0 ? `+${fmtMoney(v)}` : fmtMoney(v));
