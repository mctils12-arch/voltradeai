import { test } from "node:test";
import assert from "node:assert/strict";
import { watchModel, outcomeRows, parseTickerMessage, candidateKey, WATCH_OUTCOME_WINDOW_MS } from "./watch";
import type { AuditRow } from "./pageState";

const NOW = Date.parse("2026-10-21T14:00:00Z");
const ago = (min: number) => new Date(NOW - min * 60_000).toISOString();
const row = (min: number, type: string, message: string): AuditRow => ({ time: ago(min), type, message });

const scan = {
  timestamp: ago(1), scanned: 12138, filtered: 54, deep_analyzed: 5, slots_available: 0, options_slots_available: 1, partial: false,
  new_trades: [{ ticker: "KWEB", score: 75, side: "sell", action_label: "SELL COVERED CALL", trade_type: "options", reasons: ["388 shares held, no short call"], rules_score: 75, ml_score_raw: null, price: 24.54 }],
  top_10: [
    { ticker: "NVDA", score: 73.4, price: 184.2, change_pct: 2.1, side: "buy", action_label: "BUY", reasons: ["Momentum 1m +9.8%", "VRP +3.2"], rank: 2, sector: "Semiconductors", sector_clash: true, factors: { momentum: 86, mean_reversion: 30, vrp: 64, squeeze: 48, volume: 77 } },
    { ticker: "KWEB", score: 75, price: 24.54, change_pct: -1.2, side: "sell", action_label: "SELL COVERED CALL", reasons: ["388 shares held, no short call"], rank: 1, sector: "China internet", factors: { momentum: 42, mean_reversion: 71, vrp: 88, squeeze: 35, volume: 60 } },
    { ticker: "PFE", score: 58, price: 27.1, change_pct: 0.4, action_label: "WATCH", reasons: [], rank: 5 },
    { ticker: "", score: 70 }, null, "junk",
  ],
  sizing_blocks: [{ ticker: "AMD", reason: "Negative expectancy for strategy 'stocks' (WR=28%); refusing to size", score: 74.3, strategy: "stocks" }],
};

test("parseTickerMessage reads the ticker-first audit messages Node writes, with or without the [strategy] tag", () => {
  assert.deepEqual(parseTickerMessage("KWEB queued for market open (score 75)"), { ticker: "KWEB", rest: "queued for market open (score 75)" });
  assert.deepEqual(parseTickerMessage("AMD [stocks]: Negative expectancy for strategy 'stocks' (WR=28%); refusing to size (score 74.3)"), { ticker: "AMD", rest: "Negative expectancy for strategy 'stocks' (WR=28%); refusing to size (score 74.3)" });
  assert.deepEqual(parseTickerMessage("NVDA: score 60 < threshold 65"), { ticker: "NVDA", rest: "score 60 < threshold 65" });
  assert.deepEqual(parseTickerMessage("CRWV261120P00055000: options slots full (3/3)"), { ticker: "CRWV261120P00055000", rest: "options slots full (3/3)" });
  assert.equal(parseTickerMessage("Tier engine blocked: kill switch"), null); // not ticker-first: "Tier" is a word, no separator form
  assert.equal(parseTickerMessage(""), null);
});

test("outcomeRows keeps the newest row per ticker inside the window and lists refusals once per ticker and type", () => {
  const audit = [
    row(3, "SKIP", "NVDA: score 60 < threshold 65"),
    row(2, "CORR-CAP", "NVDA: sector Semiconductors at cap"),
    row(1, "QUEUE", "KWEB queued for market open (score 75)"),
    row(5, "SIZE-BLOCK", "AMD [stocks]: Negative expectancy (score 74.3)"),
    row(WATCH_OUTCOME_WINDOW_MS / 60_000 + 5, "SKIP", "OLD: score 10 < threshold 65"),
    row(4, "TIER2", "Scanned 12138 stocks, 2 trade candidates"),
  ];
  const { byTicker, refused } = outcomeRows(audit, NOW);
  assert.equal(byTicker.get("NVDA")?.type, "CORR-CAP"); // newest wins
  assert.equal(byTicker.get("NVDA")?.kind, "refused");
  assert.equal(byTicker.get("KWEB")?.kind, "queued");
  assert.equal(byTicker.get("AMD")?.kind, "blocked");
  assert.equal(byTicker.has("OLD"), false);
  assert.deepEqual(refused.map((r) => `${r.ticker}/${r.type}`), ["NVDA/CORR-CAP", "NVDA/SKIP", "AMD/SIZE-BLOCK"]);
  assert.equal(refused[1].score, 60);
});

test("watchModel ranks the candidates, marks admission vs the Node threshold, joins decisions and outcomes, and folds the engine's sizing blocks into the refusals", () => {
  const audit = [row(2, "CORR-CAP", "NVDA: sector Semiconductors at cap"), row(1, "QUEUE", "KWEB queued for market open (score 75)")];
  const m = watchModel({ lastScan: scan, audit, threshold: 65, nowMs: NOW });
  assert.equal(m.scanned, 12138);
  assert.equal(m.deep, 5);
  assert.equal(m.optionsSlots, 1);
  assert.deepEqual(m.candidates.map((c) => c.ticker), ["KWEB", "NVDA", "PFE"]);
  const kweb = m.candidates[0], nvda = m.candidates[1], pfe = m.candidates[2];
  assert.equal(kweb.decided, true);
  assert.equal(kweb.outcome?.kind, "queued");
  assert.equal(kweb.admitted, true);
  assert.equal(nvda.admitted, true);
  assert.equal(nvda.outcome?.type, "CORR-CAP");
  assert.equal(nvda.sectorClash, true);
  assert.equal(nvda.factors?.momentum, 86);
  assert.equal(pfe.admitted, false);
  assert.equal(pfe.factors, null);
  assert.equal(pfe.rank, 5);
  assert.equal(m.decided.length, 1);
  assert.equal(m.decided[0].kind, "options");
  assert.ok(m.refused.some((r) => r.ticker === "AMD" && r.type === "SIZE-BLOCK" && r.reason.startsWith("[stocks] ")));
  assert.ok(m.refused.some((r) => r.ticker === "NVDA" && r.type === "CORR-CAP"));
});

test("an empty or missing scan yields an honest empty model; an unknown threshold leaves admission null", () => {
  const m = watchModel({ lastScan: undefined, audit: [], threshold: null, nowMs: NOW });
  assert.equal(m.scanAt, null);
  assert.deepEqual(m.candidates, []);
  assert.deepEqual(m.refused, []);
  const m2 = watchModel({ lastScan: { top_10: [{ ticker: "X", score: 90 }] }, audit: [], threshold: null, nowMs: NOW });
  assert.equal(m2.candidates[0].admitted, null);
  assert.equal(m2.candidates[0].rank, 1);
  assert.equal(candidateKey("nvda"), "cand:NVDA");
});
