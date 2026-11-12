// FLOOR AUDIT (2026-11-05). The scan returns the volatility-managed floor's
// pass as scan_market's `spy_floor` (bot_engine._run_floor_step), but bot.ts
// never read it, so the floor's trades showed up only in the broker's order
// history. And while Node's -20% kill switch is tripped, the floor may only
// reduce exposure (human decision 2026-11-05) and every action it takes there
// must be logged. This turns one floor pass into audit rows: a summary row
// plus one row per order-level action. Passes that only deferred (outside
// 09:45-15:54 ET, or the broker clock said closed) or waited on a working
// order write nothing in a scan; the kill pass says once per day that it ran.
// 2026-11-06: when the institutional flow model cut the target
// (flows_model.floor_multiplier), the summary shows the rule's target and the
// multiplier, and one "flows ESTIMATE:" row gives the reason.

export interface FloorAuditRow { type: "FLOOR-VM" | "FLOOR-KILL-REDUCE" | "FLOOR-KILL-REDUCE-ERROR"; message: string }

interface FloorAction {
  type?: string; ticker?: string; symbol?: string; qty?: number; limit?: number;
  ok?: boolean; order_id?: string | null; error?: string | null; reason?: string;
}

/** action types that put (or tried to put) an order at the broker */
const ORDER_ACTIONS: Record<string, string> = {
  floor_vm_buy: "BUY",
  floor_vm_sell: "SELL",
  floor_vm_legacy_sell: "SELL legacy",
  cc_unwind: "BUY TO CLOSE covered call",
  // 2026-11-06: the floor's idle cash in the T-bill fund (bot_engine._manage_floor_cash)
  floor_vm_cash_buy: "BUY T-bill fund (idle floor cash)",
  floor_vm_cash_sell: "SELL T-bill fund",
};
/** the core's own order types (the others are the cash parking) */
const CORE_ACTIONS = new Set(["floor_vm_buy", "floor_vm_sell", "floor_vm_legacy_sell", "cc_unwind"]);
/** action types that explain why something did not trade (logged only with orders, or in the kill pass) */
const NOTE_ACTIONS = new Set(["cc_unwind_blocked", "floor_vm_no_quote", "floor_vm_cash_wide_spread", "floor_vm_cash_blocked"]);

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const pct = (v: unknown) => (num(v) === null ? "?" : `${(num(v)! * 100).toFixed(1)}%`);
const times = (v: unknown) => (num(v) === null ? "?" : `${num(v)!.toFixed(2)}x`);

function summary(r: Record<string, unknown>): string {
  // INSTITUTIONAL FLOWS (2026-11-06): a flows cut shows next to the study's rule
  const m = num(r.exposure_multiplier);
  const target = m !== null && m < 1
    ? `target ${times(r.target_exposure)} (rule ${times(r.rule_target_exposure)} x flows ${m.toFixed(2)})`
    : `target ${times(r.target_exposure)}`;
  const core = `sigma20 ${pct(r.sigma)}, ${target}, now ${times(r.current_exposure)} of capital $${num(r.capital)?.toFixed(0) ?? "?"}`;
  // FLOOR CASH PARKING (2026-11-06): the idle cash and its T-bill fund, when the step ran
  const c = (r.cash_parking && typeof r.cash_parking === "object" ? r.cash_parking : {}) as Record<string, unknown>;
  const cashTarget = num(c.target);
  if (cashTarget === null) return core;
  const dollars = (v: unknown) => (num(v) === null ? "?" : `$${num(v)!.toFixed(0)}`);
  return `${core}; idle cash ${dollars(c.idle)}, ${String(c.ticker || "fund")} target ${dollars(cashTarget)}, held ${dollars(c.held)}`;
}

/** the flows cut's plain-English reason, logged once with any pass that trades */
function flowsLine(r: Record<string, unknown>): string | null {
  const f = (r.flows && typeof r.flows === "object" ? r.flows : {}) as Record<string, unknown>;
  return f.applied === true && typeof f.reason === "string" ? `flows ESTIMATE: ${f.reason.slice(0, 400)}` : null;
}

function actionLine(a: FloorAction): string | null {
  const t = String(a.type || "");
  const who = String(a.ticker || a.symbol || "?").toUpperCase();
  if (t in ORDER_ACTIONS) {
    const sym = t === "cc_unwind" ? String(a.symbol || who) : who;
    const px = num(a.limit) === null ? "" : ` @ ${num(a.limit)!.toFixed(2)} limit`;
    const res = a.ok === false || a.error ? `REJECTED: ${String(a.error || "").slice(0, 160)}` : a.order_id ? `order ${a.order_id}` : "sent";
    const why = t.startsWith("floor_vm_cash_") && a.reason ? ` (${String(a.reason).slice(0, 160)})` : "";
    return `${ORDER_ACTIONS[t]} ${a.qty ?? "?"} ${sym}${px} — ${res}${why}`;
  }
  if (t === "floor_vm_cash_wide_spread") return `${who}: quote too wide, not traded (${String(a.reason || "").slice(0, 160)})`;
  // KILL LATCH 2026-11-12 (KNOWN BROKEN #51): a fund buy the latch stopped at the order
  if (t === "floor_vm_cash_blocked") return `${who}: T-bill fund buy NOT placed, Node kill latch set (${String(a.reason || "").slice(0, 160)})`;
  if (NOTE_ACTIONS.has(t)) return `${who}: ${t === "floor_vm_no_quote" ? "no quote, not traded" : `covered-call unwind blocked (${String(a.reason || "").slice(0, 160)})`}`;
  return null;
}

/**
 * One floor pass -> audit rows. `context` "scan" is the regular Tier 2 scan;
 * "kill" is the reduce-only pass while the kill switch is tripped.
 * `idleLoggedToday` lets the kill pass say once per day that it ran without
 * acting; the caller owns that flag (returned rows tell it whether to set it).
 */
export function floorAuditRows(result: unknown, context: "scan" | "kill", idleLoggedToday = false): FloorAuditRow[] {
  const r = (result && typeof result === "object" ? result : {}) as Record<string, unknown>;
  const type: FloorAuditRow["type"] = context === "kill" ? "FLOOR-KILL-REDUCE" : "FLOOR-VM";
  const status = String(r.status ?? "");
  if (status.startsWith("error")) {
    return [{ type: context === "kill" ? "FLOOR-KILL-REDUCE-ERROR" : "FLOOR-VM", message: `floor pass failed: ${status.slice(0, 200)}` }];
  }
  if (r.mode !== "vol_managed_spy") {
    // the basket floor (rollback mode) keeps its own behaviour; the kill pass says it held
    return context === "kill" && !idleLoggedToday
      ? [{ type, message: `kill switch tripped: ${status || "the basket floor holds (no reduce-only pass)"}` }]
      : [];
  }
  const actions = Array.isArray(r.actions) ? (r.actions as FloorAction[]) : [];
  const lines = actions.map(actionLine).filter((l): l is string => l !== null);
  const traded = actions.some((a) => String(a.type || "") in ORDER_ACTIONS);
  const coreTraded = actions.some((a) => CORE_ACTIONS.has(String(a.type || "")));
  const head = context === "kill" ? "kill switch tripped, reduce-only (sells only: SSO, then SPY, then legacy)"
    : coreTraded ? "rebalance" : "idle cash in the T-bill fund (paper cash earns 0; the fund accrues in price)";
  if (traded || (context === "kill" && lines.length)) {
    const flows = flowsLine(r);
    return [{ type, message: `${head} — ${summary(r)}` }, ...(flows ? [{ type, message: flows }] : []),
      ...lines.map((message) => ({ type, message }))];
  }
  if (context === "kill" && !idleLoggedToday) {
    const why = status === "deferred"
      ? `deferred (${String(actions.find((a) => a.type === "floor_vm_deferred")?.reason || "outside the trade window")})`
      : r.rebalance === false ? "nothing to reduce (within the band of its target)" : status || "no action";
    return [{ type, message: `${head}: pass ran, ${why} — ${summary(r)}` }];
  }
  return [];
}
