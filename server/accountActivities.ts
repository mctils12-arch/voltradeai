// accountActivities.ts — the account's NON-TRADE cash flows and the
// reconciliation of the fill ledger to the equity curve ([RULE-REVIEW]
// measurement, 2026-10-20). The sleeve ledger (sleeveLedger.ts) summed to
// −$12,034 of realized + open P&L since 2026-05-05 while equity moved −$7,370
// over the same window: the +$4.7k between them is whatever the FILL ledger
// cannot see — dividends, interest, fees, journals, option expiration /
// assignment / exercise, corporate actions, transfers. bot.ts walks
// /v2/account/activities (EVERY type, one page walk) per sleeve refresh:
// FILL rows go to the sleeve ledger, the rest come here. Pure.
//
// HONESTY: amounts are Alpaca's net_amount summed as given. A row that moves
// shares but no cash (a stock journal, a split) is COUNTED in `nonCashRows`,
// never valued. The reconciliation's residual is what NO ledger explains,
// under the window's own approximations (positions that predate the window
// carry their pre-window P&L in `unrealized`; the equity curve is a daily
// snapshot while open P&L is live) — stated, not hidden.

export type ActivityGroup = "dividends" | "interest" | "fees" | "journals" | "optionEvents" | "corporate" | "transfers" | "other";
export const ACTIVITY_GROUP_ORDER: readonly ActivityGroup[] = ["dividends", "interest", "fees", "journals", "optionEvents", "corporate", "transfers", "other"];

/** Alpaca's documented non-trade activity types → group
 *  (docs.alpaca.markets/docs/account-activities, read 2026-10-20). An
 *  undocumented type lands in "other" and keeps its own name in `types`. */
export const ACTIVITY_GROUPS: Readonly<Record<string, ActivityGroup>> = {
  DIV: "dividends", DIVCGL: "dividends", DIVCGS: "dividends", DIVNRA: "dividends", DIVROC: "dividends", DIVTXEX: "dividends", CGD: "dividends",
  INT: "interest", INTNRA: "interest", INTTW: "interest",
  FEE: "fees", DIVFEE: "fees", CFEE: "fees", PTC: "fees", PTR: "fees",
  JNL: "journals", JNLC: "journals", JNLS: "journals",
  OPASN: "optionEvents", OPEXP: "optionEvents", OPXRC: "optionEvents", OPEXC: "optionEvents",
  MA: "corporate", NC: "corporate", SC: "corporate", REORG: "corporate", SSO: "corporate", SSP: "corporate",
  TRANS: "transfers", CSD: "transfers", CSW: "transfers", ACATC: "transfers", ACATS: "transfers", FOPT: "transfers",
  MISC: "other",
};
export const groupOf = (type: string): ActivityGroup => ACTIVITY_GROUPS[type] ?? "other";

export interface ActivityRow {
  type: string;
  t: number;
  /** net_amount as given; 0 when the row carries none (then `cash` is false) */
  net: number;
  cash: boolean;
  symbol: string | null;
  qty: number | null;
}

const num = (v: unknown): number => (typeof v === "number" ? v : parseFloat(String(v ?? "")));
const r2 = (x: number) => Math.round(x * 100) / 100;
const month = (t: number) => new Date(t).toISOString().slice(0, 7);

/** /v2/account/activities rows (any type) → the non-trade rows, oldest
 *  first. FILL rows belong to the sleeve ledger and are skipped (counted);
 *  a row without a type or a parseable date is malformed (counted, never
 *  guessed). Null when the payload is not an array. */
export function parseNonTradeActivities(raw: unknown): { rows: ActivityRow[]; fills: number; malformed: number } | null {
  if (!Array.isArray(raw)) return null;
  const rows: ActivityRow[] = [];
  let fills = 0;
  let malformed = 0;
  for (const a of raw) {
    const o = (a && typeof a === "object" ? a : {}) as Record<string, unknown>;
    const type = typeof o.activity_type === "string" ? o.activity_type.trim().toUpperCase() : "";
    if (type === "FILL") { fills++; continue; }
    const t = Date.parse(String(o.date ?? o.transaction_time ?? ""));
    if (!type || !Number.isFinite(t)) { malformed++; continue; }
    const net = num(o.net_amount);
    const qty = num(o.qty);
    rows.push({
      type, t, net: Number.isFinite(net) ? net : 0, cash: Number.isFinite(net),
      symbol: typeof o.symbol === "string" && o.symbol ? o.symbol : null,
      qty: Number.isFinite(qty) ? qty : null,
    });
  }
  rows.sort((a, b) => a.t - b.t);
  return { rows, fills, malformed };
}

export interface ActivityTypeRow { type: string; group: ActivityGroup; count: number; net: number; byMonth: Record<string, number> }
export interface ActivityLedger {
  /** all non-trade cash in the window */
  total: number;
  count: number;
  /** rows that moved shares, not cash — counted, not valued */
  nonCashRows: number;
  groups: Record<ActivityGroup, number>;
  byMonth: Record<string, number>;
  /** per Alpaca type, largest |net| first */
  types: ActivityTypeRow[];
  first: string | null;
  last: string | null;
}

export function activityLedger(rows: readonly ActivityRow[]): ActivityLedger {
  const sorted = [...rows].sort((a, b) => a.t - b.t);
  const types = new Map<string, ActivityTypeRow>();
  const groups: Record<ActivityGroup, number> = { dividends: 0, interest: 0, fees: 0, journals: 0, optionEvents: 0, corporate: 0, transfers: 0, other: 0 };
  const byMonth: Record<string, number> = {};
  let total = 0;
  let nonCashRows = 0;
  for (const r of sorted) {
    let row = types.get(r.type);
    if (!row) { row = { type: r.type, group: groupOf(r.type), count: 0, net: 0, byMonth: {} }; types.set(r.type, row); }
    row.count++;
    if (!r.cash) { nonCashRows++; continue; }
    const m = month(r.t);
    row.net += r.net;
    row.byMonth[m] = (row.byMonth[m] ?? 0) + r.net;
    groups[row.group] += r.net;
    byMonth[m] = (byMonth[m] ?? 0) + r.net;
    total += r.net;
  }
  const roundMonths = (o: Record<string, number>) => Object.fromEntries(Object.entries(o).sort().map(([k, v]) => [k, r2(v)]));
  return {
    total: r2(total), count: sorted.length, nonCashRows,
    groups: Object.fromEntries(ACTIVITY_GROUP_ORDER.map((g) => [g, r2(groups[g])])) as Record<ActivityGroup, number>,
    byMonth: roundMonths(byMonth),
    types: [...types.values()]
      .map((t) => ({ ...t, net: r2(t.net), byMonth: roundMonths(t.byMonth) }))
      .sort((a, b) => Math.abs(b.net) - Math.abs(a.net) || a.type.localeCompare(b.type)),
    first: sorted.length ? new Date(sorted[0].t).toISOString().slice(0, 10) : null,
    last: sorted.length ? new Date(sorted[sorted.length - 1].t).toISOString().slice(0, 10) : null,
  };
}

export interface EquityPoint { date: string; value: number }
export interface Reconciliation {
  equityFrom: string;
  equityTo: string;
  equityStart: number;
  equityEnd: number;
  equityChange: number;
  realized: number;
  unrealized: number | null;
  nonTrade: number;
  /** realized + unrealized + nonTrade — null while any open position lacks an unrealized P&L */
  explained: number | null;
  /** equityChange − explained: what NO ledger accounts for (+ means equity did better than the ledgers say) */
  residual: number | null;
}

/** `curve` is bot.ts's equity curve, NEWEST FIRST. Null without a curve. */
export function reconcile(curve: readonly EquityPoint[], realized: number, unrealized: number | null, nonTrade: number): Reconciliation | null {
  if (!curve.length) return null;
  const end = curve[0];
  const start = curve[curve.length - 1];
  const equityChange = r2(end.value - start.value);
  const explained = unrealized === null ? null : r2(realized + unrealized + nonTrade);
  return {
    equityFrom: start.date, equityTo: end.date, equityStart: start.value, equityEnd: end.value, equityChange,
    realized, unrealized, nonTrade, explained,
    residual: explained === null ? null : r2(equityChange - explained),
  };
}
