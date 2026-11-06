// REPAIR 2026-10-20: options_manager.manage_options_positions() closes and
// rolls options positions every scan (bot_engine step 8b) and returns what it
// did as scan_market's `options_management` — but bot.ts never read it, so
// every options exit was invisible in the audit log and on the Bot page
// (live 2026-10-19/20: three iron condors market-closed 4-10 minutes after
// opening with no row saying why; the reason had to be reconstructed from
// code). This formats one audit row per close / roll, plus a manager error.
// WARNING actions (advisory, nothing changes) are left out: they repeat on
// every scan for the same position and would crowd the 2000-row log.
// 2026-11-05 EARNINGS CLOSE: two earnings notices do not repeat, so they get
// an OPTIONS-EARNINGS row each: the first pass's schedule of every sold option
// ("earnings_backfill", once per process start) and a failed report-date
// lookup ("earnings_lookup_failed", once per underlying per ET day).

export interface OptionsMgmtAction {
  action?: string; ticker?: string; strategy?: string; occ_symbol?: string; legs?: string[];
  reason?: string; type?: string; combined_pnl?: number; dte?: number; roll_detail?: string;
  order?: { status?: string; detail?: string; method?: string; leg_results?: Array<{ status?: string }>; ladder?: { step?: number; of?: number; price?: number | null } | null } | null;
}
// 2026-11-05 MID-FIRST EXIT LADDER (order_ladder.py): a laddered close says
// which step its order is on, and each later step / the fill (with the step it
// filled at) is one OPTIONS-LADDER row — not an exit type, so the Bot page does
// not count a ladder's steps as extra exits. OPTIONS-EARNINGS: the earnings-close
// notices (2026-11-05).
export interface OptionsMgmtRow { type: "OPTIONS-EXIT" | "OPTIONS-ROLL" | "OPTIONS-LADDER" | "OPTIONS-MGMT-ERROR" | "OPTIONS-EARNINGS"; message: string }

/** a scan closing more positions than this is summarised in one extra row */
export const OPTIONS_MGMT_AUDIT_CAP = 10;
/** options_manager's earnings notices (EARNINGS_BACKFILL_TYPE / EARNINGS_LOOKUP_FAILED_TYPE) */
export const OPTIONS_EARNINGS_NOTICE_TYPES: readonly string[] = ["earnings_backfill", "earnings_lookup_failed"];
/** an earnings notice lists every sold option; longer text is cut here */
export const OPTIONS_EARNINGS_NOTICE_MAX_CHARS = 1200;

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
    const ladder = o.ladder;
    if (ladder && typeof ladder === "object" && Number.isFinite(ladder.step) && Number.isFinite(ladder.of)) {
      const px = typeof ladder.price === "number" && Number.isFinite(ladder.price) ? ` @ $${ladder.price.toFixed(2)}` : "";
      parts.push(`exit ladder step ${ladder.step}/${ladder.of}${px}`);
    }
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
  const all = (Array.isArray(r.actions) ? r.actions : []).filter((a): a is OptionsMgmtAction => !!a && typeof a === "object");
  for (const a of all.filter((a) => OPTIONS_EARNINGS_NOTICE_TYPES.includes(String(a.type)))) {
    out.push({ type: "OPTIONS-EARNINGS", message: String(a.reason || a.type).slice(0, OPTIONS_EARNINGS_NOTICE_MAX_CHARS) });
  }
  const acts = all.filter((a) => a.action === "CLOSE" || a.action === "CLOSE_STRATEGY" || a.action === "ROLL" || a.action === "LADDER");
  for (const a of acts.slice(0, cap)) out.push({ type: a.action === "ROLL" ? "OPTIONS-ROLL" : a.action === "LADDER" ? "OPTIONS-LADDER" : "OPTIONS-EXIT", message: describe(a) });
  if (acts.length > cap) out.push({ type: "OPTIONS-EXIT", message: `… ${acts.length - cap} more options exits/rolls this scan` });
  return out;
}
