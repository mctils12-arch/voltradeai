// THIRD-LEG AUDIT (2026-11-06, rule call "SVXY 10% stop / 24h cooldown",
// decided KEEP under the human's "fix all of this"). The scan returns the SVXY
// VRP leg's pass as scan_market's `third_leg` (bot_engine._run_third_leg), but
// bot.ts never read it: SVXY buys, its vol-spike / stop-loss / recovery exits
// and every re-entry the cooldown blocked showed up nowhere but the broker's
// order history, so neither threshold could ever be judged. This turns one
// pass into audit rows — one per action; a pass with no action writes nothing
// (the leg runs every scan). Rows carry the prices the rule review needs:
// the stop's rollback trigger (SVXY +5% within 5 sessions of a protective
// exit) and the cooldown's prevention P&L (SVXY after a blocked buy).

export type Leg3AuditType = "LEG3-BUY" | "LEG3-EXIT" | "LEG3-COOLDOWN-BLOCK" | "LEG3-RESERVE" | "LEG3-ERROR";
export interface Leg3AuditRow { type: Leg3AuditType; message: string }

interface Leg3Action {
  type?: string; symbol?: string; shares?: number; qty?: string | number; price?: number | null;
  reason?: string; regime?: string; order_id?: string | null; ok?: boolean | null;
  trigger?: string; pnl_pct?: number; vxx_ratio?: number; cooldown_until?: number | null;
  stage?: string; error?: string;
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const money = (v: unknown) => (num(v) === null ? "?" : `$${num(v)!.toFixed(2)}`);
const broker = (ok: unknown) => (ok === true ? "accepted" : ok === false ? "REJECTED by the broker" : "broker answer unknown");

const TRIGGER_WORDS: Record<string, string> = {
  vol_spike: "protective: vol spike, re-entry cooldown starts",
  stop_loss: "protective: stop loss, re-entry cooldown starts",
  regime_recovery: "take profit: regime recovered",
};

export function leg3AuditRows(thirdLeg: unknown): Leg3AuditRow[] {
  if (!thirdLeg || typeof thirdLeg !== "object") return [];
  const r = thirdLeg as { actions?: unknown; status?: unknown; error?: unknown };
  const rows: Leg3AuditRow[] = [];
  for (const raw of Array.isArray(r.actions) ? r.actions : []) {
    if (!raw || typeof raw !== "object") continue;
    const a = raw as Leg3Action;
    const sym = a.symbol || "SVXY";
    switch (a.type) {
      case "vrp_harvest":
        rows.push({ type: "LEG3-BUY", message: `${sym} BUY ${a.shares ?? "?"} @ ~${money(a.price)} limit — ${a.reason || "VRP harvest"} · regime ${a.regime || "?"} · order ${a.order_id || "?"} ${broker(a.ok)}` });
        break;
      case "third_leg_exit":
        rows.push({ type: "LEG3-EXIT", message: `${sym} SELL ${a.qty ?? "?"} @ ~${money(a.price)} market · ${a.trigger || "?"} (${TRIGGER_WORDS[a.trigger || ""] || "exit"}) — ${a.reason || ""} · P&L ${num(a.pnl_pct) === null ? "?" : `${num(a.pnl_pct)!.toFixed(2)}%`} · ${broker(a.ok)}` });
        break;
      case "leg3_cooldown_block": {
        const until = num(a.cooldown_until);
        rows.push({ type: "LEG3-COOLDOWN-BLOCK", message: `${sym} buy NOT placed: re-entry cooldown${until === null ? "" : ` until ${new Date(until * 1000).toISOString()}`} — would have bought at ~${money(a.price)} · VXX ratio ${num(a.vxx_ratio) === null ? "?" : num(a.vxx_ratio)!.toFixed(3)} · ${a.reason || ""}` });
        break;
      }
      case "cash_reserve_block":
      case "cash_reserve_trim":
        rows.push({ type: "LEG3-RESERVE", message: `${sym} buy ${a.type === "cash_reserve_block" ? "blocked" : "cut"} by the regime cash reserve — ${a.reason || ""}` });
        break;
      case "third_leg_error":
        rows.push({ type: "LEG3-ERROR", message: `${sym} ${a.stage || "order"} failed: ${a.error || "unknown error"}` });
        break;
      default:
        break; // shadow_entry rows are audited by the entry-shadow path
    }
  }
  if (r.status === "error") rows.push({ type: "LEG3-ERROR", message: `third leg pass failed: ${typeof r.error === "string" && r.error ? r.error : "unknown error"}` });
  return rows;
}
