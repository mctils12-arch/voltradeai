import { test } from "node:test";
import assert from "node:assert/strict";
import {
  fmtPx, historyLine, insiderCluster, parseSymbol, symbolFromHash, bandGeometry, ladderGeometry, positionSize, decileFor, decileLine, coverageLine,
  ledgerRangeLine, ledgerPlanLine, directionHeadline, trackRecordLine, layerGaps, pushRecent, DOSSIER_RECENT_MAX,
  type BotTrack, type Plan, type Ledger, type Direction, type Dossier,
} from "./tickerDossier";

test("symbol box and hash: letters or an OCC code; anything else is refused", () => {
  assert.equal(parseSymbol(" rklb "), "RKLB");
  assert.equal(parseSymbol("RKLB261120C00030000"), "RKLB");
  assert.equal(parseSymbol("RK-LB"), null);
  assert.equal(symbolFromHash("#/dossier/rklb"), "RKLB");
  assert.equal(symbolFromHash("#/dossier"), null);
  assert.equal(symbolFromHash("#/bot"), null);
});

test("band geometry: ordered, inside 0–100, price outside the 90% band is flagged", () => {
  const b = { lo68: 95, hi68: 105, lo90: 92, hi90: 108, move68_pct: 5, move90_pct: 8 };
  const g = bandGeometry(b, 100, 101);
  assert.ok(g.lo90 < g.lo68 && g.lo68 < g.base && g.base < g.hi68 && g.hi68 < g.hi90);
  assert.ok(g.lo90 > 0 && g.hi90 < 100);
  assert.equal(g.priceOutside, null);
  const out = bandGeometry(b, 100, 120);
  assert.equal(out.price, 100);
  assert.equal(out.priceOutside, "above");
});

const plan: Plan = { entry_ref: 100, horizon_sessions: 5, stop: 90, target: 106, stretch_target: 110, risk_per_share: 10, reward_per_share: 6, reward_risk: 0.6, stretch_reward_risk: 1, stop_pct: -10, target_pct: 6, stretch_pct: 10,
  touch: { p_target_first: 0.5, p_stop_first: 0.2, p_neither: 0.3, paths: 4000 }, touch_stretch: { p_target_first: 0.2, p_stop_first: 0.2, p_neither: 0.6, paths: 4000 }, rules: {} };

test("ladder geometry keeps stop < entry < target < stretch", () => {
  const g = ladderGeometry(plan);
  assert.ok(g.stop < g.entry && g.entry < g.target && g.target < g.stretch);
});

test("position size = risk dollars / (entry − stop), floored; nonsense inputs refuse", () => {
  assert.deepEqual(positionSize(500, 100, 90), { shares: 50, notional: 5000, lossAtStop: 500 });
  assert.equal(positionSize(505, 100, 90)?.shares, 50);
  assert.equal(positionSize(5, 100, 90)?.shares, 0);
  assert.equal(positionSize(500, 90, 100), null);
  assert.equal(positionSize(-1, 100, 90), null);
  assert.equal(positionSize(Number.NaN, 100, 90), null);
});

const track: BotTrack = { n: 600, thin: false, base: { d5: { n: 600, pos_rate: 0.51, mean_pct: 0.1, rule_win_rate: 0.4 }, d20: { n: 0, pos_rate: null, mean_pct: null, rule_win_rate: null } },
  buckets: [
    { decile: 1, lo: 30, hi: 50, count: 300, d5: { n: 300, pos_rate: 0.48, mean_pct: -0.2, rule_win_rate: 0.3 }, d20: { n: 0, pos_rate: null, mean_pct: null, rule_win_rate: null }, decisions: {} },
    { decile: 2, lo: 50, hi: 90, count: 300, d5: { n: 12, pos_rate: 0.58, mean_pct: 0.6, rule_win_rate: 0.5 }, d20: { n: 0, pos_rate: null, mean_pct: null, rule_win_rate: null }, decisions: {} },
  ] };

test("decile match and its line always carry n and the base rate; thin is said", () => {
  assert.equal(decileFor(track, 70)?.decile, 2);
  assert.equal(decileFor(track, 10)?.decile, 1);
  assert.equal(decileFor(track, 95)?.decile, 2);
  assert.equal(decileFor(track, null), null);
  const line = decileLine(decileFor(track, 70), track.base);
  assert.match(line, /^n=12: up after 5 sessions 58% \(all scored candidates 51%\), mean \+0\.60%/);
  assert.match(line, /thin sample/);
  assert.match(decileLine(decileFor(track, 70), track.base, "d20"), /no settled \+20-session/);
  assert.equal(decileLine(null, undefined), "no bot score for this ticker yet");
});

test("coverage and ledger lines never show a rate without its count", () => {
  assert.match(coverageLine({ h: 1, n: 250, in68: 170, in90: 222, rate68: 0.68, rate90: 0.888, target68: 0.6827, target90: 0.9, overlapping: false }), /last 250 sessions: 68% band held 68% · 90% band held 89%/);
  const led: Ledger = { open: 3, min_n: 30, ticker: null,
    all: { range: { "1": { n: 40, in68: 29, in90: 37, rate68: 0.725, rate90: 0.925 }, "5": { n: 0, in68: 0, in90: 0, rate68: null, rate90: null } },
      plan: { n: 10, target_first: 5, stop_first: 3, neither: 2, ambiguous: 0, target_first_rate: 0.5, stop_first_rate: 0.3, neither_rate: 0.2, ambiguous_rate: 0 } } };
  assert.equal(ledgerRangeLine(led, "1"), "68% band contained the close 29/40 times (73%) · 90%: 37/40 (93%)");
  assert.match(ledgerRangeLine(led, "5"), /no scored bands yet/);
  assert.match(ledgerRangeLine(led, "1", "ticker"), /for this ticker/);
  assert.equal(ledgerPlanLine(led), "plans like this so far: target first 50%, stop first 30%, neither 20% · n=10 (thin)");
});

test("direction: no call says so; a call shows its probability and the model's record", () => {
  assert.deepEqual(directionHeadline({ status: "none", statement: "x" }), { word: "no validated direction", tone: "t" });
  const d: Direction = { status: "call", statement: "", call: { direction: "down", p_up: 0.41, base_date: "2026-10-23", horizon_end: "2026-10-30" },
    model_accuracy: { n: 45, thin: false, hits: 25, hit_rate: 0.5556, brier: 0.24, brier_skill_vs_coinflip: 0.04, in_range_rate: 0.7 } };
  assert.deepEqual(directionHeadline(d), { word: "down 59%", tone: "r" });
  assert.equal(trackRecordLine(d), "model record: 25/45 calls right (56%) · Brier skill vs coin flip +0.04");
  assert.match(trackRecordLine({ status: "none", statement: "" }), /untested/);
});

test("layer gaps list what could not run; recents dedupe and cap", () => {
  const d = { symbol: "X", generated_at: "", errors: {}, options: { status: "unavailable", source: "", as_of: null, reason: "no listed options" }, market: { status: "live", source: "", as_of: null, reason: null } } as Dossier;
  assert.deepEqual(layerGaps(d), ["options: no listed options"]);
  let r: string[] = [];
  for (const s of ["A", "B", "A", "C", "D", "E", "F", "G", "H", "I"]) r = pushRecent(r, s);
  assert.equal(r.length, DOSSIER_RECENT_MAX);
  assert.equal(r[0], "I");
  assert.equal(r.filter((x) => x === "A").length <= 1, true);
});

test("money reads as money: cents under $1,000, thousands separated above", () => {
  assert.equal(fmtPx(27.4), "$27.40");
  assert.equal(fmtPx(249.3), "$249.30");
  assert.equal(fmtPx(1673.2), "$1,673");
  assert.equal(fmtPx(null), "—");
});

const STUDY = { path: "research/event_reaction_study.md", date: "2026-10-29", window: "2016–2026" };

test("event history: measured day-0–1 average with n and t, labelled history; nothing says 'not priced in'", () => {
  const line = historyLine({ label: "8-K Item 4.02 (restatement)", n: 715, car01_pct: -1.51, t01: -3.4, drift: "no reliable drift after" }, STUDY);
  assert.equal(line, "Historically (2016–2026, n=715): after 8-K Item 4.02 (restatement), stocks moved -1.51% vs SPY on average over the filing day and the next (t -3.4); no reliable drift after. Source: research/event_reaction_study.md, 2026-10-29.");
  assert.match(historyLine({ label: "insider purchase clusters", n: 10090, car01_pct: 2.16, t01: 21.8, drift: "no reliable drift after" }, STUDY) ?? "", /n=10,090\): .* \+2\.16% .*\(t \+21\.8\)/);
  assert.equal(historyLine(null, STUDY), null);
  assert.equal(historyLine({ label: "x", n: 1, car01_pct: 1, t01: 1, drift: "d" }, null), null);
  assert.doesNotMatch(line ?? "", /priced in/i);
  assert.match(historyLine({ label: "sudden CEO/CFO exits", n: 3155, car01_pct: -1.16, t01: -7.2, drift: "no reliable drift after", conditional: "applies only if this 5.02 is a sudden CEO/CFO exit — not classified here; open the filing" }, STUDY) ?? "", /This applies only if this 5\.02 is a sudden CEO\/CFO exit/);
});

test("insider cluster: two distinct officer/director buyers within 10 days; 10%-only holders and sales do not count", () => {
  const row = (owner: string, date: string, kind = "buy", role = "CFO") => ({ filedAt: date, owner, role, kind, shares: 100, price: 10, date, url: "" });
  assert.equal(insiderCluster([row("A", "2026-10-01"), row("B", "2026-10-09")]), true);
  assert.equal(insiderCluster([row("A", "2026-10-01"), row("B", "2026-10-20")]), false);
  assert.equal(insiderCluster([row("A", "2026-10-01"), row("A", "2026-10-03")]), false);
  assert.equal(insiderCluster([row("A", "2026-10-01"), row("B", "2026-10-02", "sell")]), false);
  assert.equal(insiderCluster([row("A", "2026-10-01"), row("B", "2026-10-02", "buy", "10% owner")]), false);
  assert.equal(insiderCluster(null), false);
});
