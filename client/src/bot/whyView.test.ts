// whyView — "why are we losing money": the attribution contract read tolerantly (worst first, a plain lead
// sentence), the fill-ledger fallback, the flow outlook, the tax filter and the signal audit, each with its
// designed "not on this server yet" state. Run: npx tsx --test client/src/bot/whyView.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseAttribution, attributionFromSleeves, whyHeadline, whyTitle, parseFlows, flowsSentence, parseTax, taxSentence, parseSignals, signalsSentence, verdictOf, unavailableWords, signedMoney, dayWords, parseRiskBudget, riskSentence } from "./whyView";
import type { Sleeves } from "./useBotData";

const NOW = Date.parse("2026-11-06T15:00:00Z");
const ago = (d: number) => new Date(NOW - d * 86_400_000).toISOString();

const attribution = {
  window: { days: 30, from: "2026-10-07", to: "2026-11-06" }, total: -4210.5,
  bySignal: [{ signal: "mean reversion", pnl: 310, n: 9, reason: "bought dips that bounced" }, { signal: "VRP (sold premium)", pnl: -3120, n: 22, reason: "sold puts into a falling market; three were stopped at twice the credit" }, { signal: "momentum", pnl: -1400.5, n: 6, reason: "late entries after the move" }],
  bySector: [{ sector: "Technology", pnl: -2600, n: 14, reason: "CRWV and TEM puts" }, { sector: "Energy", pnl: 120, n: 3 }],
  byInstrument: [{ instrument: "sold puts", pnl: -3010, n: 18, reason: "the losses came from the credit side" }, { instrument: "shares", pnl: -1200.5, n: 11, reason: null }],
  byTiming: [{ timing: "first 15 minutes", pnl: -1900, n: 7, reason: "entries in the opening auction filled at the worst prices" }],
  notes: ["realized + open P&L, attributed by the decision journal's signal tag"],
  asOf: "2026-11-06T14:55:00Z",
};

test("attribution: four groups, each worst first, labels from each group's own key, total and window words", () => {
  const a = parseAttribution(attribution, 30)!;
  assert.equal(a.source, "journal");
  assert.equal(a.windowLabel, "the last 30 days");
  assert.equal(a.total, -4210.5);
  assert.deepEqual(a.groups.map((g) => g.key), ["signal", "sector", "instrument", "timing"]);
  assert.deepEqual(a.groups[0].rows.map((r) => r.label), ["VRP (sold premium)", "momentum", "mean reversion"]);
  assert.equal(a.groups[1].rows[1].reason, null);
  assert.equal(a.asOf, "2026-11-06T14:55:00Z");
  assert.deepEqual(a.notes, ["realized + open P&L, attributed by the decision journal's signal tag"]);
  assert.equal(whyHeadline(a), "The bot lost $4,211 over the last 30 days. Biggest drag: VRP (sold premium) (signal) −$3,120 over 22 trades — sold puts into a falling market; three were stopped at twice the credit. Best: mean reversion (signal) +$310.");
});

test("attribution: a shape this page does not know is null (the panel falls back), never an empty panel of zeros", () => {
  assert.equal(parseAttribution({}, 30), null);
  assert.equal(parseAttribution([], 30), null);
  assert.equal(parseAttribution({ bySignal: [{ signal: "x" }] }, 30), null, "a row without a P&L is not drawn");
  const s = parseAttribution({ window: "Oct 7 – Nov 6", total: { pnl: 50 }, byTiming: [{ bucket: "after 15:30", pnl: "50", n: "2" }] }, 7)!;
  assert.equal(s.windowLabel, "Oct 7 – Nov 6");
  assert.equal(s.total, 50);
  assert.deepEqual(s.groups[0].rows[0], { label: "after 15:30", pnl: 50, n: 2, reason: null });
  assert.equal(parseAttribution({ byTiming: [{ timing: "t", pnl: 1 }] }, 7)!.windowLabel, "the last 7 days");
  assert.match(whyHeadline(s), /^The bot made \$50 over Oct 7 – Nov 6\. Best: after 15:30 \(timing\) \+\$50\.$/);
});

const sleeves: Sleeves = {
  since: "2026-05-07T00:00:00.000Z", realizedTotal: -2961.4, unrealizedTotal: -4408.9, fetchedAt: "2026-11-06T14:00:00Z",
  rows: [
    { sleeve: "options", fills: 412, symbols: 118, realized: -1532.1, unpricedCloses: 0, unrealized: -96, openPositions: 6, byMonth: {} },
    { sleeve: "floor", fills: 690, symbols: 7, realized: 93.2, unpricedCloses: 0, unrealized: -4398.2, openPositions: 5, byMonth: {} },
    { sleeve: "stocks", fills: 236, symbols: 61, realized: -1522.5, unpricedCloses: 3, unrealized: 85.3, openPositions: 2, byMonth: {} },
  ],
  optionsByUnderlying: [{ root: "CRWV", fills: 14, realized: -612.4, byMonth: {}, open: null, openPositions: 1 }, { root: "XLF", fills: 12, realized: 161.2, byMonth: {}, open: 0, openPositions: 0 }, { root: "ZERO", fills: 2, realized: 0, byMonth: {}, open: 0, openPositions: 0 }],
  optionsByHold: [{ bucket: "<1h", events: 96, realized: -1104.3 }, { bucket: "1–7d", events: 110, realized: 318.5 }, { bucket: ">30d", events: 0, realized: 0 }],
};

test("fallback: the fill ledger's realized P&L by sleeve, options underlying and hold time, labelled as that", () => {
  const a = attributionFromSleeves(sleeves)!;
  assert.equal(a.source, "ledger");
  assert.equal(a.windowLabel, "since May 7 (realized)");
  assert.equal(a.total, -2961.4);
  assert.deepEqual(a.groups.map((g) => g.title), ["By sleeve", "Options by underlying", "Options by hold time"]);
  assert.deepEqual(a.groups[0].rows.map((r) => r.label), ["options", "single stocks", "ETF floor"]);
  assert.match(a.groups[0].rows[0].reason ?? "", /412 fills; open −\$96 not counted/);
  assert.deepEqual(a.groups[1].rows.map((r) => r.label), ["CRWV", "XLF"], "a zero underlying is left out");
  assert.deepEqual(a.groups[2].rows.map((r) => [r.label, r.n, r.reason]), [["<1h", 96, "closed within an hour of opening"], ["1–7d", 110, "held one to seven days"]]);
  assert.match(a.notes[0], /fill ledger/);
  assert.equal(whyHeadline(a), "The bot lost $2,961 in realized P&L since May 7. Biggest drag: options (sleeve) −$1,532 — sold premium, spreads and bought options — 412 fills; open −$96 not counted. Best: 1–7d (options by hold time) +$319.");
  assert.equal(attributionFromSleeves({ pending: true }), null);
  assert.equal(attributionFromSleeves(undefined), null);
});

const flows = {
  asOf: "2026-11-06T13:00:00Z", horizonDays: 10,
  equity: { pressure: -0.42, direction: "sell", confidence: 0.6, drivers: [{ name: "pension rebalancing", value: -0.3, reason: "stocks ran ahead of bonds this quarter; target-weight funds sell the excess into month-end" }, { name: "fund tax-loss selling", value: -0.12, reason: "October fiscal year-ends" }] },
  sectors: [{ sector: "Technology", etf: "XLK", pressure: -0.5, direction: "sell", drivers: [{ name: "crowding", value: 1, reason: "most crowded sector" }] }, { sector: "Utilities", etf: "XLU", pressure: 0.3, direction: "buy", drivers: [] }, { sector: "Staples", etf: "XLP", pressure: 0.1, drivers: [] }],
  calendar: [{ date: "2026-11-20", event: "monthly options expiry", effect: "pinning" }, { date: "", event: "dropped" }],
  notes: ["a model estimate"],
};

test("flows: the contract read, directions from the field or the sign, confidence as a percent, and one plain sentence", () => {
  const f = parseFlows(flows)!;
  assert.equal(f.equity?.direction, "sell");
  assert.equal(f.equity?.confidence, 60);
  assert.equal(f.sectors[2].direction, "buy", "no direction field: the pressure's sign");
  assert.equal(f.calendar.length, 1);
  assert.equal(f.equity?.drivers[0].value, "-0.30");
  assert.equal(flowsSentence(f), "Next 10 days: big funds are expected to be net sellers of stocks (pressure −0.42, confidence 60%) — stocks ran ahead of bonds this quarter; target-weight funds sell the excess into month-end. Flowing in: XLU, XLP; out: XLK.");
  assert.equal(parseFlows({}), null);
  assert.equal(parseFlows([1, 2]), null);
  assert.equal(flowsSentence(parseFlows({ calendar: [{ date: "2026-12-18", event: "Nasdaq-100 reconstitution" }] })!), "Ahead: no equity-wide flow estimate.");
});

const tax = {
  rates: { shortTerm: 0.24, longTerm: 0.15 },
  decisions: [
    { ts: ago(1), ticker: "nvda", side: "buy", allowed: false, expectedPct: 0.5, afterTaxPct: 0.38, spyAfterTaxPct: 0.6, holdingDays: 10, reason: "0.38% after tax is below SPY's 0.60% over the same 10 days" },
    { ts: ago(2), ticker: "AMD", side: "buy", allowed: true, expectedPct: 3, afterTaxPct: 2.28, spyAfterTaxPct: 0.6, holdingDays: 20, reason: "keeps 2.28% after tax" },
    { ts: ago(20), ticker: "OLD", side: "buy", allowed: false, reason: "outside the week" },
  ],
  harvest: [{ ticker: "CRWV", unrealized: -420, replacement: "IGV", washSaleUntil: "2026-12-06", reason: "sell the loss, hold the sector through IGV" }],
  ltApproaching: [{ ticker: "GLD", daysToLongTerm: 21, gain: 900, reason: "hold 21 more days for the 15% rate" }],
};

test("tax: decisions newest first, rates as fractions, and the sentence counts the week's skips, harvests and long-term candidates", () => {
  const t = parseTax(tax)!;
  assert.deepEqual(t.rates, { shortTerm: 0.24, longTerm: 0.15 });
  assert.deepEqual(t.decisions.map((d) => d.ticker), ["NVDA", "AMD", "OLD"]);
  assert.equal(t.decisions[2].afterTaxPct, null);
  assert.equal(taxSentence(t, NOW), "Tax filter: 1 of 2 trades checked in the last 7 days was skipped — the edge did not survive short-term tax (24%) against holding SPY. 1 loss could be harvested. 1 winner nears the long-term rate.");
  assert.equal(taxSentence(parseTax({ decisions: [] })!, NOW), "Tax filter: no trade checked yet.");
  assert.equal(taxSentence(parseTax({ rates: { shortTerm: 24, longTerm: 15 }, decisions: [{ ts: ago(1), ticker: "A", allowed: true }] })!, NOW), "Tax filter: all 1 trade checked in the last 7 days kept an edge over SPY after short-term tax (24%).");
  assert.equal(parseTax({ rates: {} }), null);
});

test("signals: a report read under its common field names, verdicts sorted works first; the old per-ticker list is 'legacy', never a report", () => {
  const s = parseSignals({ asOf: "2026-11-06T12:00:00Z", signals: [{ name: "news_sentiment", verdict: "noise", ic: 0.004, n: 1900, reason: "IC indistinguishable from zero out of sample" }, { feature: "vrp", status: "keep", rank_ic: 0.061, n: 1900 }, { name: "deep_score", verdict: "hurts — anti-predictive", spearman: -0.08 }, { name: "x", verdict: "maybe" }] })!;
  assert.equal(s.kind, "report");
  if (s.kind !== "report") return;
  assert.deepEqual(s.rows.map((r) => [r.name, r.verdict]), [["vrp", "works"], ["deep_score", "hurts"], ["news_sentiment", "noise"], ["x", "unclear"]]);
  assert.deepEqual([s.rows[0].metricLabel, s.rows[0].metric], ["rank IC", 0.061]);
  assert.equal(signalsSentence(s), "Of 4 signals measured: 1 add return, 1 look like noise, 1 hurt, 1 unclear.");
  assert.deepEqual(parseSignals([{ ticker: "SPY", action: "HOLD", confidence: 50 }]), { kind: "legacy" });
  assert.deepEqual(parseSignals([]), { kind: "legacy" });
  assert.equal(parseSignals({}), null);
  assert.match(signalsSentence({ kind: "legacy" }), /old per-ticker feed/);
  assert.equal(verdictOf("no signal"), "noise");
  assert.equal(verdictOf("not significant"), "noise");
  assert.equal(verdictOf(true), "works");
  assert.equal(verdictOf("adds return"), "works");
});

test("words: a missing route, an error, signed money and calendar dates that never shift a day", () => {
  assert.equal(unavailableWords(null, "the flow model"), "the flow model is not on this server yet");
  assert.equal(unavailableWords(new Error("404: Not Found"), "the flow model"), "the flow model is not on this server yet");
  assert.match(unavailableWords(new Error("502: bad gateway"), "the flow model"), /^the flow model unavailable · 502/);
  assert.equal(signedMoney(1240), "+$1,240");
  assert.equal(signedMoney(-4210), "−$4,210");
  assert.equal(signedMoney(0), "$0");
  assert.equal(dayWords("2026-05-07T00:00:00.000Z"), "May 7");
  assert.equal(dayWords("2026-11-20"), "Nov 20");
  assert.equal(dayWords(null), "—");
});

test("flows, the finished route: its 'not computed yet' body reads as its own note (never a neutral estimate); stale, the floor's cut and the session path ride along", () => {
  const empty = parseFlows({ asOf: null, horizonDays: 5, equity: { pressure: 0, direction: "neutral", confidence: 0, drivers: [] }, sectors: [], calendar: [], label: "ESTIMATE", computedAt: null, ageDays: null, stale: true, floor: null, sessions: [], notes: ["No flows estimate: the daily estimate has not been computed yet. Tier 3 computes it once per completed session."] })!;
  assert.equal(empty.computed, false);
  assert.equal(flowsSentence(empty), "No flows estimate: the daily estimate has not been computed yet. Tier 3 computes it once per completed session.");
  const f = parseFlows({ ...flows, label: "ESTIMATE", computedAt: "2026-11-06T13:00:00Z", ageDays: 3, stale: true, floor: { multiplier: 0.94, applied: true, reason: "expected selling into month-end" }, sessions: [{ date: "2026-11-06", pressure: -0.4, phase: null }, { date: "2026-11-07", pressure: -0.2, phase: "month-end" }, { date: "", pressure: 1 }] })!;
  assert.equal(f.computed, true);
  assert.equal(f.label, "ESTIMATE");
  assert.deepEqual(f.floor, { multiplier: 0.94, applied: true, reason: "expected selling into month-end" });
  assert.deepEqual(f.sessions, [{ date: "2026-11-06", pressure: -0.4 }, { date: "2026-11-07", pressure: -0.2 }]);
  assert.match(flowsSentence(f), /Flowing in: XLU, XLP; out: XLK\. The floor runs at ×0\.94 of its target because of it\. Stale: 3 days old\.$/);
  const notApplied = parseFlows({ ...flows, floor: { multiplier: 0.9, applied: false, reason: "" } })!;
  assert.match(flowsSentence(notApplied), /The floor would run at ×0\.90 of its target/);
  assert.doesNotMatch(flowsSentence(parseFlows({ ...flows, floor: { multiplier: 1, applied: true } })!), /floor/, "no cut, no clause");
});

test("tax, the finished route: with its off switch set every row is allowed, so the line counts wouldAllow and says the filter is only logging", () => {
  const t = parseTax({ ...tax, summary: { rows: 3, allowed: 3, skipped: 0, wouldSkip: 1, notPriced: 0, since: null, enforced: false }, notes: ["TAX_FILTER.ENABLED is off: logging only"],
    decisions: tax.decisions.map((d, i) => ({ ...d, allowed: true, wouldAllow: i !== 0 })) })!;
  assert.equal(t.enforced, false);
  assert.deepEqual(t.notes, ["TAX_FILTER.ENABLED is off: logging only"]);
  assert.equal(taxSentence(t, NOW), "Tax filter (logging only — its off switch is set): 1 of 2 trades checked in the last 7 days would have been skipped — the edge did not survive short-term tax (24%) against holding SPY. 1 loss could be harvested. 1 winner nears the long-term rate.");
  assert.equal(parseTax(tax)!.enforced, null, "a route without the summary block: not reported");
});

test("regime risk budget: the server's plain line, what it is doing (applied, off, stale), drivers with their reasons, and nothing without the block", () => {
  const body = { composite: "RISK_OFF", risk_budget: { budget: 0.56, raw: 0.6, band: "reduced", summary: "Risk budget 0.56 (reduced): VIX term structure inverted (1.04) -0.21; breadth 38% -0.12",
    drivers: [{ name: "vix_ts", label: "VIX term structure", value: 1.04, score: 0.2, weight: 0.4, cut: -0.21, short: "inverted", reason: "VIX above VIX3M: hedging demand is high" }, { name: "breadth", label: "Breadth", value: 38, cut: -0.12, reason: "only 38% of S&P names above their 50-day" }, { name: "", label: "" }],
    missing: ["put/call"], as_of: "2026-11-05", age_market_days: 1, stale: false, enabled: true, applied: true, consumers: ["floor", "sizer"], history: [{ as_of: "2026-11-04", budget: 0.71, band: "full" }, { as_of: "bad" }], note: "an estimate" } };
  const r = parseRiskBudget(body)!;
  assert.equal(r.band, "reduced");
  assert.equal(r.budget, 0.56);
  assert.deepEqual(r.drivers.map((d) => [d.label, d.cut]), [["VIX term structure", -0.21], ["Breadth", -0.12]]);
  assert.deepEqual(r.history, [{ asOf: "2026-11-04", budget: 0.71, band: "full" }]);
  assert.deepEqual(r.missing, ["put/call"]);
  assert.equal(riskSentence(r), "Risk budget 0.56 (reduced): VIX term structure inverted (1.04) -0.21; breadth 38% -0.12. Applied to: floor, sizer.");
  assert.equal(riskSentence({ ...r, enabled: false }), "Risk budget 0.56 (reduced): VIX term structure inverted (1.04) -0.21; breadth 38% -0.12. It is computed but switched off: nothing is sized by it.");
  assert.equal(riskSentence({ ...r, applied: false, stale: true, ageMarketDays: 4 }), "Risk budget 0.56 (reduced): VIX term structure inverted (1.04) -0.21; breadth 38% -0.12. Not applied while the regime state is stale. Stale: 4 market days old.");
  assert.equal(riskSentence({ ...r, applied: false }), "Risk budget 0.56 (reduced): VIX term structure inverted (1.04) -0.21; breadth 38% -0.12. Computed, not applied yet.");
  // the merged route's shape (server/regimeReport.ts): a code-named consumer in words; no budget yet -> the server's own note
  assert.equal(riskSentence({ ...r, consumers: ["tier_exposure_cap", "new_thing"] }), "Risk budget 0.56 (reduced): VIX term structure inverted (1.04) -0.21; breadth 38% -0.12. Applied to: the tier engine's exposure cap, new thing.");
  const none = parseRiskBudget({ risk_budget: { budget: null, raw: null, band: null, summary: null, drivers: [], missing: [], as_of: null, age_market_days: null, stale: false, enabled: true, applied: false, consumers: ["tier_exposure_cap"], history: [], note: "no risk budget in the regime state yet (it is computed on the next scan)" } })!;
  assert.equal(riskSentence(none), "no risk budget in the regime state yet (it is computed on the next scan)");
  assert.equal(riskSentence({ ...r, summary: null }), "Risk budget 0.56 (reduced). Applied to: floor, sizer.");
  assert.equal(parseRiskBudget({ composite: "RISK_ON" }), null, "a regime route without the block (the server predates it)");
  assert.equal(parseRiskBudget({ risk_budget: { band: "weird" } })!.band, null);
});

test("the panel's title never says 'losing' over a window that made money", () => {
  assert.equal(whyTitle(-4210.5), "Why we're losing money");
  assert.equal(whyTitle(null), "Why we're losing money");
  assert.equal(whyTitle(0), "Why we're losing money");
  assert.equal(whyTitle(120), "Where the money came from");
  assert.equal(whyTitle(Number.NaN), "Why we're losing money");
});
