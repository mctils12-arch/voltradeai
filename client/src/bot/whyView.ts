// Bot page OVERVIEW — "why are we losing money" (human refocus spec 2026-11-05 §8:
// "a clear 'why are we losing money' panel: loss attribution by signal, sector,
// instrument, timing"), with the money-flow outlook, the tax filter and which
// signals work as compact sections beside it. Pure; pinned by whyView.test.ts.
//
// Every route here is read tolerantly: the shapes are the contract the backend
// agents build to (attribution / flows / tax from the decision journal and its
// neighbours, signals from the ML audit), but a server that predates a route, or
// answers in a shape this page does not recognise, gets a designed sentence that
// says so — never a blank, a zero or a guess. When the attribution route is
// missing the panel falls back to what the broker's fill ledger already measures
// (GET /api/bot/sleeves: realized P&L by sleeve, by options underlying and by
// hold time), labelled as that.
import type { Sleeves } from "./useBotData";
import { fmtMoney, fmtMonDay, MINUS } from "./format";
import { routeMissing, plainLine } from "./picks";

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);
const numOf = (v: unknown): number | null => (fin(v) ? v : typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v)) ? Number(v) : null);
const notesOf = (v: unknown): string[] => (Array.isArray(v) ? v.map(plainLine).filter((x): x is string => !!x) : str(v) ? [str(v) as string] : []);
const asOfOf = (o: Record<string, unknown>): string | null => str(o.asOf) ?? str(o.generatedAt) ?? str(o.generated_at) ?? str(o.computedAt) ?? null;

/** "May 7" for a calendar date ("2026-05-07") or a UTC-midnight timestamp of one — the date as written, never
 *  shifted a day by the ET clock. */
export function dayWords(v: string | null | undefined): string {
  const d = typeof v === "string" ? v.slice(0, 10) : "";
  return /^\d{4}-\d{2}-\d{2}$/.test(d) ? fmtMonDay(`${d}T12:00:00Z`) : "—";
}

/** "+$1,240" · "−$4,210" · "$0" — signed money, a true minus. */
export function signedMoney(v: number): string {
  return v > 0 ? `+${fmtMoney(v)}` : fmtMoney(v);
}

/** Why a route has nothing to show, in a few words. */
export function unavailableWords(err: unknown, what: string): string {
  if (!err) return `${what} is not on this server yet`;
  return routeMissing(err) ? `${what} is not on this server yet` : `${what} unavailable · ${String(err instanceof Error ? err.message : err).slice(0, 120)}`;
}

// ── attribution ─────────────────────────────────────────────────────────────
export type AttrKey = "signal" | "sector" | "instrument" | "timing" | "sleeve" | "underlying" | "hold";
export const ATTR_GROUPS: ReadonlyArray<{ key: AttrKey; field: string; title: string }> = [
  { key: "signal", field: "bySignal", title: "By signal" },
  { key: "sector", field: "bySector", title: "By sector" },
  { key: "instrument", field: "byInstrument", title: "By instrument" },
  { key: "timing", field: "byTiming", title: "By timing" },
];
export interface AttrRow { label: string; pnl: number; n: number | null; reason: string | null }
export interface AttrGroup { key: AttrKey; title: string; rows: AttrRow[] }
export interface AttributionView {
  source: "journal" | "ledger";
  /** "the last 30 days" · "since May 7 (realized)" */
  windowLabel: string;
  total: number | null;
  /** each group worst first */
  groups: AttrGroup[];
  notes: string[];
  asOf: string | null;
}

function windowLabelOf(w: unknown, days: number): string {
  if (str(w)) return str(w) as string;
  if (w && typeof w === "object") {
    const o = w as Record<string, unknown>;
    const d = numOf(o.days);
    if (d !== null) return `the last ${d} day${d === 1 ? "" : "s"}`;
    if (str(o.from)) return `since ${dayWords(str(o.from))}`;
  }
  return `the last ${days} day${days === 1 ? "" : "s"}`;
}

/** The attribution route's answer (the contract: {window, total, bySignal, bySector, byInstrument, byTiming,
 *  notes}) → groups worst first; null when it is not that shape (no group has a priced row). */
export function parseAttribution(raw: unknown, days: number): AttributionView | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const o = raw as Record<string, unknown>;
  const groups: AttrGroup[] = [];
  for (const g of ATTR_GROUPS) {
    const list = o[g.field];
    if (!Array.isArray(list)) continue;
    const rows: AttrRow[] = [];
    for (const r of list) {
      if (!r || typeof r !== "object") continue;
      const ro = r as Record<string, unknown>;
      const pnl = numOf(ro.pnl);
      const label = str(ro[g.key]) ?? str(ro.label) ?? str(ro.name) ?? str(ro.bucket) ?? str(ro.key);
      if (pnl === null || !label) continue;
      rows.push({ label, pnl, n: numOf(ro.n), reason: plainLine(ro.reason) });
    }
    if (rows.length) groups.push({ key: g.key, title: g.title, rows: rows.sort((a, b) => a.pnl - b.pnl) });
  }
  if (!groups.length) return null;
  const t = o.total;
  const total = numOf(t) ?? (t && typeof t === "object" ? numOf((t as Record<string, unknown>).pnl) : null);
  return { source: "journal", windowLabel: windowLabelOf(o.window, days), total, groups, notes: notesOf(o.notes), asOf: asOfOf(o) };
}

const SLEEVE_WORDS: Record<string, [string, string]> = {
  options: ["options", "sold premium, spreads and bought options"],
  floor: ["ETF floor", "the passive index core"],
  stocks: ["single stocks", "share lots the scanner bought"],
};
const HOLD_WORDS: Record<string, string> = {
  "<1h": "closed within an hour of opening", "<1d": "closed the same day", "1–7d": "held one to seven days",
  "8–30d": "held eight to thirty days", ">30d": "held over thirty days", unknown: "opened before the fetched fill history",
};

/** The fallback when the attribution route is missing: what the broker's fill ledger already measures, labelled
 *  as realized P&L since its window start. Null while the ledger is computing or has no rows. */
export function attributionFromSleeves(s: Sleeves | undefined): AttributionView | null {
  if (!s || s.pending || !Array.isArray(s.rows) || !s.rows.length) return null;
  const groups: AttrGroup[] = [];
  groups.push({ key: "sleeve", title: "By sleeve", rows: s.rows.filter((r) => fin(r.realized)).map((r) => {
    const [label, what] = SLEEVE_WORDS[r.sleeve] ?? [r.sleeve, ""];
    return { label, pnl: r.realized, n: null, reason: `${what}${what ? " — " : ""}${r.fills.toLocaleString("en-US")} fills${fin(r.unrealized) ? `; open ${signedMoney(r.unrealized)} not counted` : ""}` };
  }).sort((a, b) => a.pnl - b.pnl) });
  const und = (s.optionsByUnderlying ?? []).filter((u) => fin(u.realized) && u.realized !== 0).sort((a, b) => a.realized - b.realized);
  if (und.length) {
    const worst = und.slice(0, 4), best = und.filter((u) => u.realized > 0).slice(-1);
    groups.push({ key: "underlying", title: "Options by underlying", rows: [...worst, ...best.filter((b) => !worst.includes(b))].map((u) => ({ label: u.root, pnl: u.realized, n: null, reason: `${u.fills} fills` })) });
  }
  const hold = (s.optionsByHold ?? []).filter((h) => h.events > 0 && fin(h.realized));
  if (hold.length) groups.push({ key: "hold", title: "Options by hold time", rows: hold.map((h) => ({ label: h.bucket, pnl: h.realized, n: h.events, reason: HOLD_WORDS[h.bucket] ?? null })).sort((a, b) => a.pnl - b.pnl) });
  return {
    source: "ledger", windowLabel: `since ${dayWords(s.since ?? s.windowStart ?? null)} (realized)`,
    total: fin(s.realizedTotal) ? s.realizedTotal : null, groups,
    notes: ["from the broker's fill ledger: realized P&L per closed lot, average cost — attribution by signal, sector and timing comes with the decision journal"],
    asOf: s.fetchedAt ?? null,
  };
}

/** The panel's title for an attribution total: the human's own words while the window lost money (or the total is
 *  unknown), never "losing" over a window that made money. */
export function whyTitle(total: number | null | undefined): string {
  return typeof total === "number" && Number.isFinite(total) && total > 0 ? "Where the money came from" : "Why we're losing money";
}

/** The panel's lead sentence: what was lost (or made) over the window, the single biggest drag with its reason,
 *  and the best contributor. */
export function whyHeadline(a: AttributionView): string {
  const t = a.total;
  const span = a.source === "ledger" ? `in realized P&L ${a.windowLabel.replace(/ \(realized\)$/, "")}` : `over ${a.windowLabel}`;
  const lead = t === null ? `P&L ${span}, by where it came from.` : t < 0 ? `The bot lost ${fmtMoney(-t)} ${span}.` : t > 0 ? `The bot made ${fmtMoney(t)} ${span}.` : `The bot broke even ${span}.`;
  const all = a.groups.flatMap((g) => g.rows.map((r) => ({ ...r, group: g.title.replace(/^By /, "").toLowerCase() })));
  const worst = all.filter((r) => r.pnl < 0).sort((x, y) => x.pnl - y.pnl)[0];
  const best = all.filter((r) => r.pnl > 0).sort((x, y) => y.pnl - x.pnl)[0];
  const drag = worst ? ` Biggest drag: ${worst.label} (${worst.group}) ${fmtMoney(worst.pnl)}${worst.n !== null ? ` over ${worst.n} trade${worst.n === 1 ? "" : "s"}` : ""}${worst.reason ? ` — ${worst.reason}` : ""}.` : "";
  const help = best ? ` Best: ${best.label} (${best.group}) ${signedMoney(best.pnl)}.` : "";
  return `${lead}${drag}${help}`;
}

// ── institutional flows ─────────────────────────────────────────────────────
export type FlowDir = "buy" | "sell" | "neutral";
export interface FlowDriver { name: string; value: string | null; reason: string | null }
export interface FlowsView {
  asOf: string | null; horizonDays: number | null;
  /** the route's own "no estimate yet" body (asOf and computedAt null): its note is the sentence, never a neutral reading */
  computed: boolean;
  equity: { direction: FlowDir; pressure: number | null; confidence: number | null; drivers: FlowDriver[] } | null;
  sectors: Array<{ sector: string; etf: string | null; pressure: number | null; direction: FlowDir; reason: string | null }>;
  calendar: Array<{ date: string; event: string; effect: string | null }>;
  notes: string[];
  /** "ESTIMATE" — the model's own label */
  label: string | null;
  ageDays: number | null; stale: boolean;
  /** what the estimate did to the floor: its multiplier on the floor's target and whether it was applied */
  floor: { multiplier: number; applied: boolean; reason: string | null } | null;
  /** the expected pressure per session over the horizon */
  sessions: Array<{ date: string; pressure: number }>;
}
const dirOf = (v: unknown, pressure: number | null): FlowDir => {
  const s = (str(v) ?? "").toLowerCase();
  if (s === "buy" || s === "sell" || s === "neutral") return s;
  return pressure === null || pressure === 0 ? "neutral" : pressure > 0 ? "buy" : "sell";
};
const driversOf = (v: unknown): FlowDriver[] => (Array.isArray(v) ? v.map((d) => {
  const o = (d ?? {}) as Record<string, unknown>;
  const val = o.value;
  return { name: str(o.name) ?? "", value: fin(val) ? (Math.abs(val) < 10 ? val.toFixed(2) : Math.round(val).toLocaleString("en-US")) : str(val), reason: plainLine(o.reason) };
}).filter((d) => d.name || d.reason) : []);
/** confidence as a percent: a 0–1 fraction or a 0–100 number */
const pctOf = (v: unknown): number | null => { const n = numOf(v); return n === null ? null : n <= 1 ? Math.round(n * 100) : Math.round(n); };

export function parseFlows(raw: unknown): FlowsView | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const o = raw as Record<string, unknown>;
  const eq = o.equity && typeof o.equity === "object" ? (o.equity as Record<string, unknown>) : null;
  const sectors = Array.isArray(o.sectors) ? o.sectors.map((s) => {
    const so = (s ?? {}) as Record<string, unknown>;
    const pressure = numOf(so.pressure);
    return { sector: str(so.sector) ?? str(so.etf) ?? "", etf: str(so.etf), pressure, direction: dirOf(so.direction, pressure), reason: driversOf(so.drivers)[0]?.reason ?? plainLine(so.reason) };
  }).filter((s) => s.sector) : [];
  const calendar = Array.isArray(o.calendar) ? o.calendar.map((c) => {
    const co = (c ?? {}) as Record<string, unknown>;
    return { date: str(co.date) ?? "", event: str(co.event) ?? "", effect: plainLine(co.effect) };
  }).filter((c) => c.date && c.event) : [];
  if (!eq && !sectors.length && !calendar.length) return null;
  const pressure = eq ? numOf(eq.pressure) : null;
  const fl = o.floor && typeof o.floor === "object" && !Array.isArray(o.floor) ? (o.floor as Record<string, unknown>) : null;
  const mult = fl ? numOf(fl.multiplier) : null;
  return {
    asOf: str(o.asOf) ?? asOfOf(o), horizonDays: numOf(o.horizonDays),
    computed: !(o.asOf === null && (o.computedAt === null || o.computedAt === undefined)),
    equity: eq ? { direction: dirOf(eq.direction, pressure), pressure, confidence: pctOf(eq.confidence), drivers: driversOf(eq.drivers) } : null,
    sectors, calendar, notes: notesOf(o.notes),
    label: str(o.label), ageDays: numOf(o.ageDays), stale: o.stale === true,
    floor: fl && mult !== null ? { multiplier: mult, applied: fl.applied === true, reason: plainLine(fl.reason) } : null,
    sessions: Array.isArray(o.sessions) ? o.sessions.map((x) => { const so = (x ?? {}) as Record<string, unknown>; return { date: str(so.date) ?? "", pressure: numOf(so.pressure) }; }).filter((x): x is { date: string; pressure: number } => !!x.date && x.pressure !== null) : [],
  };
}

const signed2 = (v: number) => `${v < 0 ? MINUS : v > 0 ? "+" : ""}${Math.abs(v).toFixed(2)}`;
/** "Next 10 days: big funds are expected to be net sellers of stocks (pressure −0.42, confidence 60%) — <driver>.
 *  Flowing in: XLU, XLP; out: XLK." */
export function flowsSentence(f: FlowsView): string {
  if (!f.computed) return f.notes[0] ?? "No flows estimate yet.";
  const span = f.horizonDays !== null ? `Next ${f.horizonDays} day${f.horizonDays === 1 ? "" : "s"}` : "Ahead";
  const e = f.equity;
  let lead = `${span}: no equity-wide flow estimate.`;
  if (e) {
    const what = e.direction === "sell" ? "net sellers of stocks" : e.direction === "buy" ? "net buyers of stocks" : "neither net buyers nor sellers of stocks";
    const nums = [e.pressure !== null ? `pressure ${signed2(e.pressure)}` : null, e.confidence !== null ? `confidence ${e.confidence}%` : null].filter(Boolean).join(", ");
    const top = e.drivers.find((d) => d.reason) ?? e.drivers[0];
    lead = `${span}: big funds are expected to be ${what}${nums ? ` (${nums})` : ""}${top ? ` — ${top.reason ?? top.name}` : ""}.`;
  }
  const into = f.sectors.filter((s) => s.direction === "buy").sort((a, b) => (b.pressure ?? 0) - (a.pressure ?? 0)).slice(0, 3).map((s) => s.etf ?? s.sector);
  const out = f.sectors.filter((s) => s.direction === "sell").sort((a, b) => (a.pressure ?? 0) - (b.pressure ?? 0)).slice(0, 3).map((s) => s.etf ?? s.sector);
  const rot = into.length || out.length ? ` Flowing in: ${into.length ? into.join(", ") : "none"}; out: ${out.length ? out.join(", ") : "none"}.` : "";
  // what it did: the floor's cut ahead of expected selling (applied, or only computed)
  const fl = f.floor && f.floor.multiplier < 1 ? ` ${f.floor.applied ? "The floor runs at" : "The floor would run at"} ×${f.floor.multiplier.toFixed(2)} of its target because of it.` : "";
  const stale = f.stale ? ` Stale: ${f.ageDays ?? "?"} days old.` : "";
  return `${lead}${rot}${fl}${stale}`;
}

// ── tax filter ──────────────────────────────────────────────────────────────
export interface TaxDecisionView { ts: string; ticker: string; side: string; allowed: boolean | null; expectedPct: number | null; afterTaxPct: number | null; spyAfterTaxPct: number | null; holdingDays: number | null; reason: string | null;
  /** the verdict before the off switch (the route's wouldAllow): false = the filter would skip it */
  wouldAllow: boolean | null }
export interface TaxFilterView {
  rates: { shortTerm: number; longTerm: number } | null;
  decisions: TaxDecisionView[];
  harvest: Array<{ ticker: string; unrealized: number | null; replacement: string | null; washSaleUntil: string | null; reason: string | null }>;
  ltApproaching: Array<{ ticker: string; daysToLongTerm: number | null; gain: number | null; reason: string | null }>;
  /** false: the off switch is set — every row is logged and allowed (wouldAllow carries the verdict); null: not reported */
  enforced: boolean | null;
  notes: string[];
}
const rateOf = (v: unknown): number | null => { const n = numOf(v); return n === null ? null : n > 1 ? n / 100 : n; };

export function parseTax(raw: unknown): TaxFilterView | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const o = raw as Record<string, unknown>;
  if (!Array.isArray(o.decisions) && !Array.isArray(o.harvest) && !Array.isArray(o.ltApproaching)) return null;
  const r = o.rates && typeof o.rates === "object" ? (o.rates as Record<string, unknown>) : null;
  const st = r ? rateOf(r.shortTerm) : null, lt = r ? rateOf(r.longTerm) : null;
  const decisions = (Array.isArray(o.decisions) ? o.decisions : []).map((d) => {
    const x = (d ?? {}) as Record<string, unknown>;
    return { ts: str(x.ts) ?? "", ticker: (str(x.ticker) ?? "").toUpperCase(), side: str(x.side) ?? "", allowed: typeof x.allowed === "boolean" ? x.allowed : null, expectedPct: numOf(x.expectedPct), afterTaxPct: numOf(x.afterTaxPct), spyAfterTaxPct: numOf(x.spyAfterTaxPct), holdingDays: numOf(x.holdingDays), reason: plainLine(x.reason), wouldAllow: typeof x.wouldAllow === "boolean" ? x.wouldAllow : null };
  }).filter((d) => d.ticker).sort((a, b) => (Date.parse(b.ts) || 0) - (Date.parse(a.ts) || 0));
  const harvest = (Array.isArray(o.harvest) ? o.harvest : []).map((h) => {
    const x = (h ?? {}) as Record<string, unknown>;
    return { ticker: (str(x.ticker) ?? "").toUpperCase(), unrealized: numOf(x.unrealized), replacement: str(x.replacement), washSaleUntil: str(x.washSaleUntil), reason: plainLine(x.reason) };
  }).filter((h) => h.ticker);
  const ltApproaching = (Array.isArray(o.ltApproaching) ? o.ltApproaching : []).map((h) => {
    const x = (h ?? {}) as Record<string, unknown>;
    return { ticker: (str(x.ticker) ?? "").toUpperCase(), daysToLongTerm: numOf(x.daysToLongTerm), gain: numOf(x.gain), reason: plainLine(x.reason) };
  }).filter((h) => h.ticker);
  const sum = o.summary && typeof o.summary === "object" && !Array.isArray(o.summary) ? (o.summary as Record<string, unknown>) : null;
  return { rates: st !== null && lt !== null ? { shortTerm: st, longTerm: lt } : null, decisions, harvest, ltApproaching,
    enforced: sum && typeof sum.enforced === "boolean" ? sum.enforced : null, notes: notesOf(o.notes) };
}

/** "Tax filter: 3 of 8 trades checked in the last 7 days were skipped — their edge did not survive short-term
 *  tax (24%) against holding SPY. 2 losses could be harvested; 1 winner turns long-term within 30 days." */
export function taxSentence(t: TaxFilterView, nowMs: number, days = 7): string {
  const recent = t.decisions.filter((d) => { const ms = Date.parse(d.ts); return Number.isFinite(ms) && nowMs - ms <= days * 86_400_000; });
  // with the off switch set every row is allowed: the verdict that counts is wouldAllow, and the line says so
  const off = t.enforced === false;
  const blocked = recent.filter((d) => (off ? d.wouldAllow === false : d.allowed === false)).length;
  const rate = t.rates ? ` (${Math.round(t.rates.shortTerm * 100)}%)` : "";
  const lead = off && recent.length
    ? `Tax filter (logging only — its off switch is set): ${blocked} of ${recent.length} trade${recent.length === 1 ? "" : "s"} checked in the last ${days} days would have been skipped — the edge did not survive short-term tax${rate} against holding SPY.`
    : !t.decisions.length
    ? "Tax filter: no trade checked yet."
    : !recent.length
    ? `Tax filter: no trade checked in the last ${days} days (${t.decisions.length} earlier).`
    : blocked
    ? `Tax filter: ${blocked} of ${recent.length} trade${recent.length === 1 ? "" : "s"} checked in the last ${days} days ${blocked === 1 ? "was" : "were"} skipped — the edge did not survive short-term tax${rate} against holding SPY.`
    : `Tax filter: all ${recent.length} trade${recent.length === 1 ? "" : "s"} checked in the last ${days} days kept an edge over SPY after short-term tax${rate}.`;
  const h = t.harvest.length ? ` ${t.harvest.length} loss${t.harvest.length === 1 ? "" : "es"} could be harvested.` : "";
  const l = t.ltApproaching.length ? ` ${t.ltApproaching.length} winner${t.ltApproaching.length === 1 ? "" : "s"} near${t.ltApproaching.length === 1 ? "s" : ""} the long-term rate.` : "";
  return `${lead}${h}${l}`;
}

// ── market regime: the risk budget ──────────────────────────────────────────
export type RiskBand = "full" | "reduced" | "defensive" | "minimal";
export interface RiskBudgetView {
  budget: number | null; band: RiskBand | null;
  /** the server's own plain-English line ("Risk budget 0.56 (reduced): VIX term structure inverted (1.04) −0.21; …") */
  summary: string | null;
  drivers: Array<{ label: string; cut: number | null; reason: string | null }>;
  missing: string[];
  asOf: string | null; ageMarketDays: number | null; stale: boolean;
  enabled: boolean | null; applied: boolean | null; consumers: string[];
  history: Array<{ asOf: string; budget: number; band: string | null }>;
  note: string | null;
}
const BANDS: readonly string[] = ["full", "reduced", "defensive", "minimal"];

/** /api/bot/regime's body → its risk_budget block; null when the route has none (a server that predates it). */
export function parseRiskBudget(raw: unknown): RiskBudgetView | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const rb = (raw as Record<string, unknown>).risk_budget;
  if (!rb || typeof rb !== "object" || Array.isArray(rb)) return null;
  const o = rb as Record<string, unknown>;
  const band = (str(o.band) ?? "").toLowerCase();
  const list = (v: unknown): string[] => (Array.isArray(v) ? v.map((x) => (typeof x === "string" ? x : plainLine(x))).filter((x): x is string => !!x) : []);
  return {
    budget: numOf(o.budget), band: BANDS.includes(band) ? (band as RiskBand) : null, summary: str(o.summary),
    drivers: Array.isArray(o.drivers) ? o.drivers.map((d) => { const x = (d ?? {}) as Record<string, unknown>; return { label: str(x.label) ?? str(x.name) ?? "", cut: numOf(x.cut), reason: plainLine(x.reason) }; }).filter((d) => d.label) : [],
    missing: list(o.missing), asOf: str(o.as_of), ageMarketDays: numOf(o.age_market_days), stale: o.stale === true,
    enabled: typeof o.enabled === "boolean" ? o.enabled : null, applied: typeof o.applied === "boolean" ? o.applied : null, consumers: list(o.consumers),
    history: Array.isArray(o.history) ? o.history.map((h) => { const x = (h ?? {}) as Record<string, unknown>; return { asOf: str(x.as_of) ?? "", budget: numOf(x.budget), band: str(x.band) }; }).filter((h): h is { asOf: string; budget: number; band: string | null } => !!h.asOf && h.budget !== null) : [],
    note: str(o.note),
  };
}

/** The server's sentence, with what the budget is doing: applied, only computed (the switch is off), or stale. */
const CONSUMER_WORDS: Readonly<Record<string, string>> = { tier_exposure_cap: "the tier engine's exposure cap" };
/** a consumer's code name in words: the known ones by name, any other with its underscores as spaces */
export function consumerWords(c: string): string {
  return CONSUMER_WORDS[c] ?? c.replace(/_/g, " ");
}
export function riskSentence(r: RiskBudgetView): string {
  if (r.budget === null && !r.summary) return r.note ?? "No risk budget computed yet.";
  const lead = r.summary ?? `Risk budget ${(r.budget ?? 0).toFixed(2)}${r.band ? ` (${r.band})` : ""}.`;
  const state = r.enabled === false ? " It is computed but switched off: nothing is sized by it." : r.applied === false ? (r.stale ? " Not applied while the regime state is stale." : " Computed, not applied yet.") : r.applied && r.consumers.length ? ` Applied to: ${r.consumers.map(consumerWords).join(", ")}.` : "";
  const stale = r.stale ? ` Stale: ${r.ageMarketDays ?? "?"} market days old.` : "";
  return `${lead}${/[.!]$/.test(lead) ? "" : "."}${state}${stale}`;
}

// ── which signals work ──────────────────────────────────────────────────────
export type SignalVerdict = "works" | "noise" | "hurts" | "unclear";
export interface SignalRowView { name: string; verdict: SignalVerdict; metricLabel: string | null; metric: number | null; n: number | null; reason: string | null }
export type SignalsView = { kind: "report"; rows: SignalRowView[]; asOf: string | null; notes: string[] } | { kind: "legacy" };
const METRIC_KEYS: Array<[string, string]> = [["ic", "IC"], ["rank_ic", "rank IC"], ["rankIc", "rank IC"], ["spearman", "Spearman"], ["edgePct", "edge %"], ["edge", "edge"], ["pnl", "P&L"], ["hitRate", "hit rate"], ["hit_rate", "hit rate"], ["winRate", "win rate"], ["win_rate", "win rate"], ["meanReturnPct", "mean %"], ["mean_return_pct", "mean %"]];
export function verdictOf(v: unknown): SignalVerdict {
  if (typeof v === "boolean") return v ? "works" : "noise";
  const s = (str(v) ?? "").toLowerCase();
  // the negatives first: "no signal" / "not significant" must never read as a signal that works
  if (/hurt|harm|negative|anti|inverse/.test(s)) return "hurts";
  if (/\bno\b|\bnot\b|noise|cut|drop|remove|useless|fail|insignificant|none|flat|dead/.test(s)) return "noise";
  // "predictive" (signal_attribution.py) after the negatives: "anti_predictive" already read as hurts
  if (/keep|work|\badds?\b|useful|positive|pass|valid|signal|edge|predictive/.test(s)) return "works";
  return "unclear";
}

/** The signals route → a report of rows, or "legacy" for the old generated-signal list (an array of ticker /
 *  action rows that says nothing about which signals work), or null for anything else. */
export function parseSignals(raw: unknown): SignalsView | null {
  const looksLegacy = (a: unknown[]) => a.length > 0 && a.every((x) => x && typeof x === "object" && "ticker" in (x as object) && "action" in (x as object));
  if (Array.isArray(raw)) return looksLegacy(raw) || raw.length === 0 ? { kind: "legacy" } : null;
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const list = [o.signals, o.features, o.rows, o.results].find((x) => Array.isArray(x)) as unknown[] | undefined;
  if (!list) return null;
  if (looksLegacy(list)) return { kind: "legacy" };
  const rows: SignalRowView[] = [];
  for (const r of list) {
    if (!r || typeof r !== "object") continue;
    const x = r as Record<string, unknown>;
    // signal_attribution.py (the ML audit) names a row by `label` / `id` and nests its numbers:
    // long_term.h5.ic (2017-26 point-in-time walk-forward, 5-day IC) and forward.n (deduplicated shadow log)
    const name = str(x.name) ?? str(x.signal) ?? str(x.feature) ?? str(x.key) ?? str(x.label) ?? str(x.id);
    if (!name) continue;
    const mk = METRIC_KEYS.find(([k]) => numOf(x[k]) !== null);
    const lt = x.long_term && typeof x.long_term === "object" ? (x.long_term as Record<string, unknown>) : null;
    const h5 = lt && lt.h5 && typeof lt.h5 === "object" ? (lt.h5 as Record<string, unknown>) : null;
    const fw = x.forward && typeof x.forward === "object" ? (x.forward as Record<string, unknown>) : null;
    const nestedIc = !mk && h5 ? numOf(h5.ic) : null;
    rows.push({ name, verdict: verdictOf(x.verdict ?? x.status ?? x.decision ?? x.keep ?? x.useful),
      metricLabel: mk ? mk[1] : nestedIc !== null ? "5d IC" : null, metric: mk ? numOf(x[mk[0]]) : nestedIc,
      n: numOf(x.n) ?? numOf(x.count) ?? numOf(x.samples) ?? (fw ? numOf(fw.n) : null), reason: plainLine(x.reason ?? x.summary ?? x.note ?? x.why ?? x.action_text) });
  }
  if (!rows.length) return null;
  const order: Record<SignalVerdict, number> = { works: 0, hurts: 1, noise: 2, unclear: 3 };
  rows.sort((a, b) => order[a.verdict] - order[b.verdict] || a.name.localeCompare(b.name));
  const headline = plainLine(o.headline);
  return { kind: "report", rows, asOf: asOfOf(o) ?? str(o.generated_at), notes: headline ? [headline, ...notesOf(o.notes)] : notesOf(o.notes) };
}

export function signalsSentence(s: SignalsView): string {
  if (s.kind === "legacy") return "This server's signal list is the old per-ticker feed — the audit of which signals actually add return is not on it yet.";
  const c = (v: SignalVerdict) => s.rows.filter((r) => r.verdict === v).length;
  const parts = [`${c("works")} add return`, `${c("noise")} look like noise`, c("hurts") ? `${c("hurts")} hurt` : null, c("unclear") ? `${c("unclear")} unclear` : null].filter(Boolean);
  return `Of ${s.rows.length} signal${s.rows.length === 1 ? "" : "s"} measured: ${parts.join(", ")}.`;
}
