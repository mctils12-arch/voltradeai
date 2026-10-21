// Bot page chart — which levels a subject gets drawn with, and their words.
// Presentation only: every number comes from shared/positionLevels via
// /api/bot/positions (the engine's own formulas), never recomputed here.
// Pure; pinned by levels.test.ts.
import type { PositionLevels, OptionLegLevels } from "../../../../shared/positionLevels";
import { optionLegLevels, pricePct } from "../../../../shared/positionLevels";
import type { DrawLevel, DrawZone, Theme } from "./draw";
import { MINUS } from "../format";

/** A time-anchored mark (the entry fill); BotChart maps timeS to a bar. */
export interface OverlayMarker { timeS: number; price: number; label: string; shape: "up" | "down"; color: string }
export interface Overlay {
  levels: DrawLevel[];
  zones: DrawZone[];
  markers: OverlayMarker[];
  /** label → value pairs for the legend row under the chart */
  legend: Array<[string, string]>;
  /** honest footnotes: what is not recorded, what is suppressed */
  notes: string[];
}
export const EMPTY_OVERLAY: Overlay = { levels: [], zones: [], markers: [], legend: [], notes: [] };

/** The subset of /api/bot/positions a chart overlay reads. */
export interface OverlaySubject {
  ticker: string;
  qty: number;
  entryPrice: number | null;
  currentPrice: number | null;
  isOption?: boolean;
  optionMeta?: { underlyingTicker?: string; strike?: number; optionType?: string; expiry?: string; daysToExpiry?: number; strategy?: string | null } | null;
  levels?: PositionLevels | null;
}

const f2 = (v: number | null | undefined, d = 2) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(d));
const signed = (v: number | null | undefined, d = 1, suffix = "") => (v === null || v === undefined || !Number.isFinite(v) ? "—" : `${v < 0 ? MINUS : "+"}${Math.abs(v).toFixed(d)}${suffix}`);

const STOP_WORDS: Record<NonNullable<PositionLevels["stopFrom"]>, string> = {
  entry: "stop · from entry",
  peak_pnl: "trailing stop · from peak P&L",
  peak_price: "trailing stop · from the high",
  breakeven: "breakeven stop",
};

export function positionOverlay(p: OverlaySubject, theme: Theme, nowMs: number = Date.now()): Overlay {
  const levels: DrawLevel[] = [];
  const zones: DrawZone[] = [];
  const markers: OverlayMarker[] = [];
  const legend: Array<[string, string]> = [];
  const notes: string[] = [];
  const entry = p.entryPrice;
  const lv = p.levels ?? null;
  if (entry !== null && Number.isFinite(entry)) levels.push({ price: entry, color: theme.accent, dash: "dash", label: "entry" });
  else notes.push("entry price not recorded");
  if (lv?.heldMinutes != null && entry !== null && Number.isFinite(entry)) markers.push({ timeS: (nowMs - lv.heldMinutes * 60_000) / 1000, price: entry, label: "in", shape: "up", color: theme.accent });
  if (!lv || !lv.recorded) {
    notes.push("stop state not recorded yet — the engine writes it on its first manage tick after entry");
    return { levels, zones, markers, legend, notes };
  }
  const long = lv.side === "long";
  // stop
  if (lv.stopPrice !== null) {
    const sup = lv.stopSuppressed;
    levels.push({ price: lv.stopPrice, color: theme.red, dash: "dot", label: sup ? `${STOP_WORDS[lv.stopFrom ?? "entry"]} · waits` : STOP_WORDS[lv.stopFrom ?? "entry"], muted: !!sup });
    if (sup && entry !== null) {
      levels.push({ price: pricePct(entry, -sup.overridePct, lv.side), color: theme.red, dash: "solid", label: `exit now only below ${MINUS}${sup.overridePct}%`, fit: false });
      notes.push(sup.until === "60min" ? `stop waits ${Math.ceil(sup.minutesLeft ?? 0)} min more (60-minute minimum hold); only a ${MINUS}${sup.overridePct}% loss closes it before then` : `stop waits ${sup.daysLeft ?? "?"} more day(s) (short-hold grace to day 4); only a ${MINUS}${sup.overridePct}% loss closes it before then`);
    }
    if (entry !== null) {
      const locked = long ? lv.stopPrice > entry : lv.stopPrice < entry;
      zones.push(locked
        ? { lo: Math.min(entry, lv.stopPrice), hi: Math.max(entry, lv.stopPrice), color: theme.green, alpha: 0.07, label: "locked in" }
        : { lo: Math.min(entry, lv.stopPrice), hi: Math.max(entry, lv.stopPrice), color: theme.red, alpha: 0.06, label: "risk to the stop" });
    }
  }
  if (lv.hardFloorPrice !== null && lv.hardFloorPct !== null) levels.push({ price: lv.hardFloorPrice, color: theme.red, dash: "dash", label: `hard floor ${MINUS}${lv.hardFloorPct}%`, muted: true, fit: false });
  // scale-outs (the "phase out" areas)
  for (const so of lv.scaleOuts) {
    if (!so.applicable) continue;
    levels.push({ price: so.price, color: theme.orange, dash: "dot", label: so.done ? `⅓ sold · +${f2(so.rMultiple, 1)}R` : `sell ⅓ · +${f2(so.rMultiple, 1)}R`, muted: so.done });
  }
  if (lv.scaleOuts.length && !lv.scaleOuts[0].applicable) notes.push("lot too small to scale out (the engine needs more than one share)");
  if (lv.scaleOuts.length >= 1 && lv.scaleOuts[0].applicable && entry !== null && !lv.scaleOuts[0].done) {
    const first = lv.scaleOuts[0].price;
    zones.push({ lo: Math.min(entry, first), hi: Math.max(entry, first), color: theme.green, alpha: 0.045, label: "to the first ⅓" });
  }
  // take profit
  if (lv.takeProfitPrice !== null && lv.takeProfitPct !== null) {
    levels.push({ price: lv.takeProfitPrice, color: theme.green, dash: "dash", label: lv.takeProfitArmed ? `target +${f2(lv.takeProfitPct, 1)}%` : `target +${f2(lv.takeProfitPct, 1)}% (after ⅔ out)`, muted: !lv.takeProfitArmed });
  } else if (lv.phase >= 3) notes.push("no fixed target in phase 3+ — the trailing stop takes the rest");
  // legend
  legend.push(["phase", `${lv.phase} of 4`]);
  legend.push(["R", `${signed(lv.rMultiple, 2, "R")} · peak ${signed(lv.peakR, 2, "R")}`]);
  if (lv.riskPct !== null) legend.push(["1R", `${f2(lv.riskPct, 2)}%${lv.atrPct !== null ? ` · ATR ${f2(lv.atrPct, 2)}%` : ""}${lv.stopPct !== null ? ` · stop ${f2(lv.stopPct, 2)}%` : ""}`]);
  const done = lv.scaleOuts.filter((s) => s.done).length;
  const next = lv.scaleOuts.find((s) => !s.done && s.applicable);
  legend.push(["scale-outs", `${done} of 2${next ? ` · next at +${f2(next.rMultiple, 1)}R` : ""}${lv.breakevenActive ? " · breakeven stop on" : ""}`]);
  if (lv.originalQty !== null && lv.remainingQty !== null && lv.remainingQty !== lv.originalQty) legend.push(["shares", `${lv.remainingQty} of ${lv.originalQty} left`]);
  legend.push(["held", `${lv.daysHeld ?? "?"}d · time stop d${lv.timeStop.day} if ±${lv.timeStop.withinPct}%${lv.timeStop.armed ? " · armed" : ""}`]);
  if (lv.regime) legend.push(["regime", `${lv.regime.toLowerCase()} · ⅔ at ${f2(lv.scaleOut2R, 1)}R · trail ${f2(lv.trailAtrMult, 1)}×ATR`]);
  return { levels, zones, markers, legend, notes };
}

export function optionOverlay(p: OverlaySubject, theme: Theme): Overlay {
  const levels: DrawLevel[] = [];
  const zones: DrawZone[] = [];
  const legend: Array<[string, string]> = [];
  const notes: string[] = [];
  const m = p.optionMeta ?? null;
  const type = m?.optionType === "PUT" || m?.optionType === "CALL" ? m.optionType : null;
  if (!m || !type || !Number.isFinite(m.strike ?? NaN) || p.entryPrice === null) {
    notes.push("option leg details not recorded (strike / type / premium) — showing the underlying only");
    return { levels, zones, markers: [], legend, notes };
  }
  const lg: OptionLegLevels | null = optionLegLevels({ optionType: type, strike: m.strike as number, qty: p.qty, premium: p.entryPrice });
  if (!lg) { notes.push("option leg details not recorded"); return { levels, zones, markers: [], legend, notes }; }
  levels.push({ price: lg.strike, color: theme.text, dash: "dash", label: `strike · ${lg.side} ${type.toLowerCase()}` });
  levels.push({ price: lg.breakevenAtExpiry, color: theme.orange, dash: "dot", label: "breakeven at expiry" });
  if (lg.maxProfitZone === "above_strike") zones.push({ lo: lg.strike, hi: Number.POSITIVE_INFINITY, color: theme.green, alpha: 0.05, label: "keeps the whole premium at expiry" });
  if (lg.maxProfitZone === "below_strike") zones.push({ lo: Number.NEGATIVE_INFINITY, hi: lg.strike, color: theme.green, alpha: 0.05, label: "keeps the whole premium at expiry" });
  if (lg.side === "short") zones.push(lg.maxProfitZone === "above_strike"
    ? { lo: Number.NEGATIVE_INFINITY, hi: lg.breakevenAtExpiry, color: theme.red, alpha: 0.05, label: "loses at expiry" }
    : { lo: lg.breakevenAtExpiry, hi: Number.POSITIVE_INFINITY, color: theme.red, alpha: 0.05, label: "loses at expiry" });
  legend.push(["leg", `${lg.side} ${type.toLowerCase()} ${f2(lg.strike, Number.isInteger(lg.strike) ? 0 : 2)} × ${Math.abs(p.qty)}`]);
  legend.push(["expiry", `${m.expiry ?? "—"}${Number.isFinite(m.daysToExpiry ?? NaN) ? ` · ${m.daysToExpiry}d` : ""}`]);
  legend.push(["premium", `$${f2(p.entryPrice)} / share`]);
  legend.push(["manager target", `closes at ${Math.round(lg.profitTargetFrac * 100)}% of max profit`]);
  legend.push(["strategy", m.strategy ? String(m.strategy) : "not recorded"]);
  legend.push(["underlying", m.underlyingTicker ?? "—"]);
  return { levels, zones, markers: [], legend, notes };
}

export function overlayFor(p: OverlaySubject, theme: Theme, nowMs: number = Date.now()): Overlay {
  return p.isOption ? optionOverlay(p, theme) : positionOverlay(p, theme, nowMs);
}

/** A scan candidate (client/src/bot/watch.ts) — not a position: the engine
 *  sizes a stop and a target from that day's ATR only AT ENTRY, so nothing
 *  of the kind is drawn; the scan's own price and score are. */
export interface CandidateSubject { ticker: string; rank: number | null; score: number | null; price: number | null; action: string; reasons: string[]; sector: string | null; admitted: boolean | null; changePct: number | null }
export function candidateOverlay(c: CandidateSubject, theme: Theme): Overlay {
  const levels: DrawLevel[] = [];
  const legend: Array<[string, string]> = [];
  if (c.price !== null && Number.isFinite(c.price) && c.price > 0) levels.push({ price: c.price, color: theme.textDim, dash: "dot", label: "price at scan", muted: true });
  legend.push(["score", `${c.score === null ? "—" : c.score.toFixed(1)} · ${c.admitted === null ? "threshold not recorded" : c.admitted ? "admitted" : "below threshold"}`]);
  legend.push(["action", c.action.toLowerCase()]);
  if (c.rank !== null) legend.push(["rank", `#${c.rank}`]);
  if (c.sector) legend.push(["sector", c.sector]);
  if (c.changePct !== null) legend.push(["day", signed(c.changePct, 2, "%")]);
  if (c.reasons.length) legend.push(["why", c.reasons.join(" · ")]);
  return { levels, zones: [], markers: [], legend, notes: ["a candidate has no stop or target yet — the engine sizes both from the ATR at entry; the chart shows the scan's price and score only"] };
}

/** The symbol whose bars a subject charts: itself, or an option leg's
 *  underlying (from its metadata, else the OCC root). */
export function underlyingOf(p: { ticker: string; isOption?: boolean; optionMeta?: { underlyingTicker?: string } | null }): string | null {
  if (!p.isOption) return p.ticker || null;
  const u = p.optionMeta?.underlyingTicker;
  if (u) return u;
  const m = /^([A-Z]{1,6})\d{6}[CP]\d{8}$/.exec((p.ticker || "").toUpperCase());
  return m ? m[1] : null;
}
