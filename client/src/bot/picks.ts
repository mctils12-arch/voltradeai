// Bot page OVERVIEW — the pick cards (human refocus spec 2026-11-05 §8: "per pick
// a plain-English card: why bought, why alternatives were rejected (incl. tax or
// instrument reasons), current P&L"). Pure; pinned by picks.test.ts.
//
// The reasons come from the decision journal (GET /api/bot/decisions: one row per
// buy / sell / skip / rotate / scale decision, every reason in plain English).
// A position the journal has no row for — opened before it began logging, or by
// a path that does not write to it yet — says exactly that and falls back to what
// the page already knows for certain: the position's own fields (what the
// instrument is, what was paid, what it is worth) and the engine's exit plan
// (shared/positionLevels via exitPlan.ts). Nothing is inferred about WHY.
import type { BotPosition } from "./useBotData";
import { optionExitPlan, optionManagement } from "../../../shared/positionLevels";
import { optionExitSegments, stockExitSegments, strategyWords, type PlanSegment } from "./exitPlan";
import { fmtOcc, fmtMoney, fmtMonDay } from "./format";

const DAY_MS = 86_400_000;
export const DECISION_ACTIONS = ["buy", "sell", "skip", "rotate", "scale_in", "scale_out"] as const;
/** the decision pipeline's stages, in the order the journal's `stages` object names them (spec §3) */
export const STAGE_ORDER = ["regime", "calendar", "flows", "sector", "selection", "options", "tax"] as const;
export const STAGE_LABEL: Record<(typeof STAGE_ORDER)[number], string> = {
  regime: "Market regime", calendar: "Calendar", flows: "Institutional flows", sector: "Sector",
  selection: "Stock selection", options: "Options positioning", tax: "Tax",
};
/** how far back the cards look for a closed pick, and for a candidate the bot passed on */
export const CLOSED_PICK_DAYS = 14;
export const PASSED_ON_HOURS = 48;

export interface DecisionRow {
  ts: string; t: number; ticker: string; action: string;
  instrument: string | null; chosen: string | null;
  rejected: Array<{ instrument: string; reason: string }>;
  stages: Array<{ key: string; label: string; text: string }>;
  sizing: string | null; summary: string | null;
  /** the ordered symbol (the OCC contract for an option; the journal's `ticker` is then its underlying) */
  symbol: string | null;
  /** written by the one-time backfill from the broker's fills — no reason was recorded at the time */
  backfilled: boolean;
  /** an exit's result, percent (the journal's pnl_pct) */
  pnlPct: number | null;
}
export interface DecisionsParse { ok: boolean; rows: DecisionRow[]; reason: string | null }

const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);
const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** A route this server does not serve (404, or the SPA's HTML answering where JSON was asked for). */
export function routeMissing(err: unknown): boolean {
  const m = err instanceof Error ? err.message : String(err ?? "");
  return /^404\b/.test(m) || /Unexpected token|not valid JSON|JSON\.parse|in JSON at position/i.test(m);
}

/** A stage's value as one plain line: a sentence as given, an object by its summary / reason / text field,
 *  else its plain fields as "key value" pairs; nothing printable → null (the stage is left out). */
export function plainLine(v: unknown): string | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "string") return str(v);
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  if (Array.isArray(v)) { const parts = v.map(plainLine).filter((x): x is string => !!x); return parts.length ? parts.join("; ") : null; }
  if (typeof v === "object") {
    const o = v as Record<string, unknown>;
    for (const k of ["summary", "reason", "text", "why", "decision", "note", "reasoning"]) { const s = str(o[k]); if (s) return s; }
    // the journal's selection stage: the score against its bar, the term that led it, the scanner's own reasons
    if (fin(o.score) && ("drivers" in o || "dominant" in o || "reasons" in o)) return selectionLine(o);
    for (const k of ["label", "name"]) { const s = str(o[k]); if (s) return s; }
    const parts = Object.entries(o).filter(([, x]) => typeof x === "string" || typeof x === "number" || typeof x === "boolean").map(([k, x]) => `${k.replace(/_/g, " ")} ${typeof x === "number" ? (Number.isInteger(x) ? x : x.toFixed(2)) : x}`);
    return parts.length ? parts.join(" · ") : null;
  }
  return null;
}

/** "score 75.2 vs the 65 bar · led by momentum (+9.0 points) · Momentum 1m +9.8%; VRP +3.2" */
export function selectionLine(o: Record<string, unknown>): string {
  const score = fin(o.score) ? o.score : null;
  const bar = fin(o.threshold) ? o.threshold : null;
  const dom = o.dominant && typeof o.dominant === "object" ? (o.dominant as Record<string, unknown>) : null;
  const domWords = dom && str(dom.label) && fin(dom.points) ? `led by ${str(dom.label)} (${dom.points > 0 ? "+" : dom.points < 0 ? "−" : ""}${Math.abs(dom.points).toFixed(1)} points)` : null;
  const reasons = Array.isArray(o.reasons) ? o.reasons.map((r) => str(r)).filter((r): r is string => !!r).slice(0, 3) : [];
  return [score !== null ? `score ${score.toFixed(1)}${bar !== null ? ` vs the ${bar} bar` : ""}` : null, domWords, reasons.length ? reasons.join("; ") : null].filter(Boolean).join(" · ");
}

/** The journal's sizing block as words: "61 at $741.32 = $45,220 (6.0% of equity) — <its reasoning>"; a plain
 *  string as given; a bare number as dollars. */
export function sizingWords(v: unknown): string | null {
  if (fin(v)) return fmtMoney(v);
  if (!v || typeof v !== "object" || Array.isArray(v)) return plainLine(v);
  const o = v as Record<string, unknown>;
  const qty = fin(o.qty) ? o.qty : null, price = fin(o.price) ? o.price : null, notional = fin(o.notional) ? o.notional : null, pct = fin(o.pct_equity) ? o.pct_equity : null;
  const why = str(o.reasoning) ?? str(o.reason) ?? str(o.summary);
  const head = [qty !== null && price !== null ? `${qty.toLocaleString("en-US")} at $${price.toFixed(2)}` : qty !== null ? `${qty.toLocaleString("en-US")}` : null, notional !== null ? `${qty !== null ? (price !== null ? "= " : "for ") : ""}${fmtMoney(notional)}` : null].filter(Boolean).join(" ");
  const line = `${head}${pct !== null ? `${head ? " " : ""}(${pct.toFixed(1)}% of equity)` : ""}`;
  return line && why ? `${line} — ${why}` : line || why || plainLine(o);
}

/** The journal's answer → rows, newest first. Not an array (an older server, the SPA's fallback page, an error)
 *  → ok false with the reason the cards print. */
export function parseDecisions(raw: unknown, err?: unknown): DecisionsParse {
  if (err) return { ok: false, rows: [], reason: routeMissing(err) ? "the decision journal is not on this server yet" : `decision journal unavailable · ${String(err instanceof Error ? err.message : err).slice(0, 120)}` };
  const list = Array.isArray(raw) ? raw : raw && typeof raw === "object" && Array.isArray((raw as { decisions?: unknown }).decisions) ? (raw as { decisions: unknown[] }).decisions : null;
  if (!list) return { ok: false, rows: [], reason: raw === undefined ? null : "the decision journal is not on this server yet" };
  const rows: DecisionRow[] = [];
  for (const r of list) {
    if (!r || typeof r !== "object") continue;
    const o = r as Record<string, unknown>;
    const ticker = str(o.ticker)?.toUpperCase();
    const ts = str(o.ts);
    const t = ts ? Date.parse(ts) : NaN;
    if (!ticker || !ts || !Number.isFinite(t)) continue;
    const rejected = Array.isArray(o.rejected) ? o.rejected.map((x) => {
      if (typeof x === "string") return { instrument: "", reason: x };
      const xo = (x ?? {}) as Record<string, unknown>;
      return { instrument: str(xo.instrument) ?? str(xo.name) ?? "", reason: plainLine(xo.reason ?? xo.why) ?? "" };
    }).filter((x) => x.instrument || x.reason) : [];
    const st = o.stages && typeof o.stages === "object" && !Array.isArray(o.stages) ? (o.stages as Record<string, unknown>) : {};
    const keys = [...STAGE_ORDER.filter((k) => k in st), ...Object.keys(st).filter((k) => !(STAGE_ORDER as readonly string[]).includes(k))];
    const stages = keys.map((k) => ({ key: k, label: (STAGE_LABEL as Record<string, string>)[k] ?? k.replace(/_/g, " "), text: plainLine(st[k]) })).filter((s): s is { key: string; label: string; text: string } => !!s.text);
    rows.push({
      ts, t, ticker, action: (str(o.action) ?? "").toLowerCase(),
      // a skip's chosen is the word "none" in the journal: nothing was chosen
      instrument: str(o.instrument), chosen: str(o.chosen) === "none" ? null : str(o.chosen), rejected, stages,
      sizing: sizingWords(o.sizing), summary: str(o.summary),
      symbol: str(o.symbol)?.toUpperCase() ?? null, backfilled: o.backfilled === true, pnlPct: fin(o.pnl_pct) ? o.pnl_pct : null,
    });
  }
  rows.sort((a, b) => b.t - a.t);
  return { ok: true, rows, reason: null };
}

// ── matching a decision to a held position ─────────────────────────────────
const isOpt = (p: BotPosition) => !!p.isOption || (p.asset_class ?? "").toLowerCase().includes("option");
export function underlyingOfPosition(p: BotPosition): string {
  if (!isOpt(p)) return p.ticker.toUpperCase();
  const u = p.optionMeta?.underlyingTicker;
  if (u) return u.toUpperCase();
  const m = /^([A-Z]{1,6})\d{6}[CP]\d{8}$/.exec(p.ticker.toUpperCase());
  return m ? m[1] : p.ticker.toUpperCase();
}
const OCC_SYMBOL = /^[A-Z]{1,6}\d{6}[CP]\d{8}$/;
const OPTIONY = /\b(option|options|put|puts|call|calls|spread|condor|straddle|strangle|csp|covered|premium|leg)\b/i;
const STOCKY = /\b(stock|stocks|share|shares|etf|equity|equities)\b/i;
/** Whether a decision row is about this position: the exact OCC symbol, or the underlying when the row's
 *  instrument words do not say it was the other kind (an option row never explains a share lot). */
export function decisionMatches(d: DecisionRow, p: BotPosition): boolean {
  const sym = p.ticker.toUpperCase();
  const opt = isOpt(p);
  const chosen = (d.chosen ?? "").toUpperCase();
  // an exact OCC symbol IS the instrument (a share lot's ticker is also its options' underlying, so it is not)
  if (opt && (d.ticker === sym || chosen === sym || d.symbol === sym)) return true;
  if (d.ticker !== underlyingOfPosition(p)) return false;
  if (OCC_SYMBOL.test(chosen)) return opt && chosen === sym;
  // the ordered symbol names the instrument outright: an OCC contract is that option, the bare ticker its shares
  if (d.symbol) return opt ? d.symbol === sym : !OCC_SYMBOL.test(d.symbol);
  const words = `${d.instrument ?? ""} ${d.chosen ?? ""}`;
  return opt ? !(STOCKY.test(words) && !OPTIONY.test(words)) : !(OPTIONY.test(words) && !STOCKY.test(words));
}
const ENTRY = new Set(["buy", "rotate"]);
const LATER = new Set(["scale_in", "scale_out", "rotate"]);

export interface PickCard {
  key: string;
  /** "QQQ" · "CRWV 55P 20-Nov" · "META bull put credit spread" */
  title: string;
  /** "61 shares · held 3 days" · "sold put · 30 days to expiry" · "2 legs · 18 days to expiry" */
  sub: string;
  pnlUsd: number | null;
  /** null for a multi-leg unit (a percent of a net premium reads wrong); its legs carry their own */
  pnlPct: number | null;
  legs: BotPosition[];
  /** what the instrument IS, from the position's own fields (always known) */
  what: string;
  /** WHY it was bought: the journal's summary, or the sentence saying no reason was recorded */
  why: string;
  source: "journal" | "none";
  /** the matched row came from the one-time backfill (the broker's fills, no reason recorded at the time) */
  backfilled: boolean;
  decidedAt: string | null;
  chosen: string | null;
  rejected: Array<{ instrument: string; reason: string }>;
  stages: Array<{ key: string; label: string; text: string }>;
  sizing: string | null;
  /** scale-ins, scale-outs and rotations since the entry, newest first */
  later: Array<{ ts: string; action: string; text: string }>;
  /** WHEN the engine sells — the Book's exit line, from shared/positionLevels */
  exit: PlanSegment[];
}

export const NO_JOURNAL_WHY = "No recorded reason — this position was opened before the decision journal began, or by a path that does not write to it yet.";

/** "Sold 1 put on CRWV: keeps the $247 premium if CRWV stays above $55 through Nov 20; below about $52.53 it loses money." */
export function optionWords(p: BotPosition): string {
  const m = p.optionMeta;
  const u = underlyingOfPosition(p);
  const n = Math.abs(Number(p.qty) || 0);
  const entry = fin(p.entryPrice) ? p.entryPrice : null;
  const k = m && fin(m.strike) ? m.strike : null;
  const type = m?.optionType === "PUT" ? "put" : m?.optionType === "CALL" ? "call" : "option";
  const dated = m?.expiry ? fmtMonDay(`${m.expiry}T12:00:00Z`) : null;
  const exp = dated ?? "expiry";
  const prem = entry !== null ? fmtMoney(entry * 100 * n) : "the premium";
  const plural = n === 1 ? "" : "s";
  const sold = Number(p.qty) < 0;
  const head = `${sold ? "Sold" : "Bought"} ${n} ${type}${plural} on ${u}`;
  if (k === null || entry === null || type === "option") return `${head}${k !== null ? ` (strike $${k})` : ""}${dated ? `, expiring ${dated}` : ""}.`;
  // the breakeven at expiry: strike minus the premium for a put, plus it for a call (either side of the trade)
  const be = (type === "put" ? k - entry : k + entry).toFixed(2);
  if (sold) return type === "put"
    ? `${head}: keeps the ${prem} premium if ${u} stays above $${k} through ${exp}; below about $${be} it loses money.`
    : `${head}: keeps the ${prem} premium if ${u} stays below $${k} through ${exp}; above about $${be} it loses money.`;
  return type === "call"
    ? `${head}: pays off if ${u} is above about $${be} by ${exp}; the most it can lose is the ${prem} paid.`
    : `${head}: pays off if ${u} is below about $${be} by ${exp}; the most it can lose is the ${prem} paid.`;
}

function stockWords(p: BotPosition): string {
  const q = Math.abs(Number(p.qty) || 0);
  const days = fin(p.daysHeld) ? Math.round(p.daysHeld) : null;
  const side = Number(p.qty) < 0 || p.side === "short" ? "Shorted" : "Bought";
  return `${side} ${q.toLocaleString("en-US")} share${q === 1 ? "" : "s"} of ${p.ticker} at ${fin(p.entryPrice) ? `$${p.entryPrice.toFixed(2)}` : "a price not recorded"}${days !== null ? `, ${days === 0 ? "today" : `${days} day${days === 1 ? "" : "s"} ago`}` : ""}.`;
}

/** The decision that opened this position (the latest buy / rotation into it inside its holding period), else
 *  the latest scale-in; null when the journal has none. */
export function entryDecision(p: BotPosition, rows: readonly DecisionRow[], nowMs: number): DecisionRow | null {
  const held = fin(p.daysHeld) ? p.daysHeld : null;
  const since = held !== null ? nowMs - (held + 3) * DAY_MS : -Infinity;
  const mine = rows.filter((d) => d.t <= nowMs + 60_000 && d.t >= since && decisionMatches(d, p));
  return mine.find((d) => ENTRY.has(d.action)) ?? mine.find((d) => d.action === "scale_in") ?? null;
}

/** One card per held position — the legs of one multi-leg strategy on one underlying are ONE card (the unit
 *  the options manager closes) — in the Book's order: shares first, then options. */
export function pickCards(positions: readonly BotPosition[] | undefined, rows: readonly DecisionRow[], nowMs: number): PickCard[] {
  if (!positions) return [];
  const ordered = [...positions.filter((p) => !isOpt(p)), ...positions.filter(isOpt)];
  const groups = new Map<string, BotPosition[]>();
  for (const p of ordered) {
    const grouped = isOpt(p) && optionManagement(p.optionMeta?.strategy) === "group";
    const key = grouped ? `grp:${underlyingOfPosition(p)}:${p.optionMeta?.strategy ?? ""}` : p.ticker;
    const g = groups.get(key);
    if (g) g.push(p); else groups.set(key, [p]);
  }
  const cards: PickCard[] = [];
  for (const [key, legs] of groups) {
    const p = legs[0];
    const opt = isOpt(p);
    const multi = key.startsWith("grp:");
    const d = entryDecision(p, rows, nowMs);
    const u = underlyingOfPosition(p);
    const dte = p.optionMeta && fin(p.optionMeta.daysToExpiry) ? Math.round(p.optionMeta.daysToExpiry) : null;
    const dteWords = dte === null ? "expiry not recorded" : dte <= 0 ? "expires today" : `${dte} day${dte === 1 ? "" : "s"} to expiry`;
    const pnlUsd = legs.every((l) => fin(l.pnl)) ? legs.reduce((s, l) => s + l.pnl, 0) : null;
    const later = d ? rows.filter((x) => x.t > d.t && x !== d && LATER.has(x.action) && decisionMatches(x, p)).map((x) => ({ ts: x.ts, action: x.action, text: x.summary ?? (x.rejected.map((r) => r.reason).filter(Boolean).join("; ") || "no reason recorded") })) : [];
    const exit = opt
      ? optionExitSegments(optionExitPlan({ qty: Number(p.qty) || 0, entryPrice: fin(p.entryPrice) ? p.entryPrice : null, currentPrice: fin(p.currentPrice) ? p.currentPrice : null, daysToExpiry: dte, strategy: p.optionMeta?.strategy, legsInGroup: legs.length }))
      : stockExitSegments(p.levels);
    const heldDays = fin(p.daysHeld) ? Math.round(p.daysHeld) : null;
    cards.push({
      key,
      title: multi ? `${u} ${strategyWords(p.optionMeta?.strategy)}` : opt ? fmtOcc(p.ticker) : p.ticker,
      sub: multi ? `${legs.length} legs · ${dteWords}` : opt ? `${Number(p.qty) < 0 ? "sold" : "bought"} ${p.optionMeta?.optionType === "PUT" ? "put" : p.optionMeta?.optionType === "CALL" ? "call" : "option"} · ${dteWords}` : `${Math.abs(Number(p.qty) || 0).toLocaleString("en-US")} shares${heldDays !== null ? ` · held ${heldDays}d` : ""}`,
      pnlUsd, pnlPct: multi ? null : fin(p.pnlPct) ? p.pnlPct : null,
      legs,
      what: multi ? `${strategyWords(p.optionMeta?.strategy)} on ${u}: ${legs.length} legs the manager opens and closes as one unit.` : opt ? optionWords(p) : stockWords(p),
      why: d ? (d.summary ?? `${d.action.replace(/_/g, " ")} — the journal recorded no summary`) : NO_JOURNAL_WHY,
      source: d ? "journal" : "none", backfilled: d?.backfilled ?? false,
      decidedAt: d?.ts ?? null, chosen: d?.chosen ?? d?.instrument ?? null,
      rejected: d?.rejected ?? [], stages: d?.stages ?? [], sizing: d?.sizing ?? null,
      later, exit,
    });
  }
  return cards;
}

const heldSymbols = (positions: readonly BotPosition[] | undefined) => {
  const s = new Set<string>();
  for (const p of positions ?? []) { s.add(p.ticker.toUpperCase()); s.add(underlyingOfPosition(p)); }
  return s;
};

export interface MiniPick { key: string; ticker: string; ts: string; action: string; text: string; rejected: Array<{ instrument: string; reason: string }>; pnlPct: number | null }

/** Exits of names no longer held, inside CLOSED_PICK_DAYS, newest first, one per ticker. */
export function closedPicks(rows: readonly DecisionRow[], positions: readonly BotPosition[] | undefined, nowMs: number, days = CLOSED_PICK_DAYS): MiniPick[] {
  const held = heldSymbols(positions);
  const seen = new Set<string>();
  const out: MiniPick[] = [];
  for (const d of rows) {
    if (d.action !== "sell" || nowMs - d.t > days * DAY_MS || held.has(d.ticker) || seen.has(d.ticker)) continue;
    seen.add(d.ticker);
    out.push({ key: `c-${d.ticker}-${d.ts}`, ticker: d.ticker, ts: d.ts, action: d.action, text: d.summary ?? "sold — no reason recorded", rejected: d.rejected, pnlPct: d.pnlPct });
  }
  return out.slice(0, 8);
}

/** Candidates the bot looked at and skipped inside PASSED_ON_HOURS, newest first, one per ticker. */
export function passedOn(rows: readonly DecisionRow[], positions: readonly BotPosition[] | undefined, nowMs: number, hours = PASSED_ON_HOURS): MiniPick[] {
  const held = heldSymbols(positions);
  const seen = new Set<string>();
  const out: MiniPick[] = [];
  for (const d of rows) {
    if (d.action !== "skip" || nowMs - d.t > hours * 3_600_000 || held.has(d.ticker) || seen.has(d.ticker)) continue;
    seen.add(d.ticker);
    out.push({ key: `s-${d.ticker}-${d.ts}`, ticker: d.ticker, ts: d.ts, action: d.action, text: d.summary ?? (d.rejected.map((r) => r.reason).filter(Boolean).join("; ") || "skipped — no reason recorded"), rejected: d.rejected, pnlPct: null });
  }
  return out.slice(0, 8);
}
