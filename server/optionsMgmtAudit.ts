// REPAIR 2026-10-20: options_manager.manage_options_positions() closes and
// rolls options positions every scan (bot_engine step 8b) and returns what it
// did as scan_market's `options_management` — but bot.ts never read it, so
// every options exit was invisible in the audit log and on the Bot page
// (live 2026-10-19/20: three iron condors market-closed 4-10 minutes after
// opening with no row saying why; the reason had to be reconstructed from
// code). This formats one audit row per close / roll, plus a manager error.
// WARNING actions (advisory, nothing changes) are left out: they repeat on
// every scan for the same position and would crowd the 2000-row log.

export interface OptionsMgmtAction {
  action?: string; ticker?: string; strategy?: string; occ_symbol?: string; legs?: string[];
  reason?: string; type?: string; combined_pnl?: number; dte?: number; roll_detail?: string;
  order?: { status?: string; detail?: string; method?: string; leg_results?: Array<{ status?: string }> } | null;
}
export interface OptionsMgmtRow { type: "OPTIONS-EXIT" | "OPTIONS-ROLL" | "OPTIONS-MGMT-ERROR"; message: string }

/** a scan closing more positions than this is summarised in one extra row */
export const OPTIONS_MGMT_AUDIT_CAP = 10;

const money = (v: number) => `${v < 0 ? "−" : "+"}$${Math.abs(v).toFixed(2)}`;

function describe(a: OptionsMgmtAction): string {
  const what = a.strategy && a.strategy !== "unknown" ? a.strategy : a.occ_symbol || "option";
  const parts = [`${String(a.ticker || "?").toUpperCase()} ${what}: ${String(a.reason || a.type || "no reason given").slice(0, 200)}`];
  if (typeof a.combined_pnl === "number" && Number.isFinite(a.combined_pnl)) parts.push(`unrealized ${money(a.combined_pnl)} at decision`);
  if (a.legs?.length) parts.push(`${a.legs.length} legs`);
  const o = a.order;
  if (o && typeof o === "object") {
    const how = o.method ? ` (${o.method})` : "";
    parts.push(o.status === "error" || o.status === "failed" ? `order FAILED${how}: ${String(o.detail || "").slice(0, 120)}` : `order ${o.status || "?"}${how}`);
    // the per-leg fallback says "submitted" even when some legs were rejected
    const legs = Array.isArray(o.leg_results) ? o.leg_results : [];
    const bad = legs.filter((l) => l && l.status === "error").length;
    if (bad) parts.push(`${bad}/${legs.length} legs FAILED`);
  }
  if (a.roll_detail) parts.push(String(a.roll_detail).slice(0, 120));
  return parts.join(" · ");
}

/** Pure: scan_market's `options_management` → audit rows. */
export function formatOptionsMgmt(result: unknown, cap: number = OPTIONS_MGMT_AUDIT_CAP): OptionsMgmtRow[] {
  if (!result || typeof result !== "object") return [];
  const r = result as { actions?: unknown; error?: unknown };
  const out: OptionsMgmtRow[] = [];
  if (typeof r.error === "string" && r.error) out.push({ type: "OPTIONS-MGMT-ERROR", message: `options manager failed this scan: ${r.error.slice(0, 200)}` });
  const acts = (Array.isArray(r.actions) ? r.actions : [])
    .filter((a): a is OptionsMgmtAction => !!a && typeof a === "object")
    .filter((a) => a.action === "CLOSE" || a.action === "CLOSE_STRATEGY" || a.action === "ROLL");
  for (const a of acts.slice(0, cap)) out.push({ type: a.action === "ROLL" ? "OPTIONS-ROLL" : "OPTIONS-EXIT", message: describe(a) });
  if (acts.length > cap) out.push({ type: "OPTIONS-EXIT", message: `… ${acts.length - cap} more options exits/rolls this scan` });
  return out;
}
