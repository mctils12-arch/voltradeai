import { test } from "node:test";
import assert from "node:assert/strict";
import { positionOverlay, optionOverlay, overlayFor, underlyingOf, candidateOverlay, type OverlaySubject } from "./levels";
import { FALLBACK_THEME } from "./draw";
import type { PositionLevels } from "../../../../shared/positionLevels";

const T = FALLBACK_THEME;
const lv = (over: Partial<PositionLevels> = {}): PositionLevels => ({
  recorded: true, side: "long", phase: 1, regime: "NEUTRAL", atrPct: 2.5, riskPct: 5, pnlPct: 0.4, rMultiple: 0.08, peakR: 0.1, highestPnl: 0.5, highestPrice: 100.5,
  stopPct: 5, stopPrice: 95, stopFrom: "entry", breakevenActive: false, stopSuppressed: null, hardFloorPct: 20, hardFloorPrice: 80,
  takeProfitPct: 7.5, takeProfitPrice: 107.5, takeProfitArmed: false,
  scaleOuts: [{ n: 1, rMultiple: 1, pct: 5, price: 105, done: false, applicable: true }, { n: 2, rMultiple: 2, pct: 10, price: 110, done: false, applicable: true }],
  trailAtrMult: 2, scaleOut2R: 2, originalQty: 30, remainingQty: 30, scalesCompleted: 0, daysHeld: 5, heldMinutes: 7200,
  timeStop: { day: 7, withinPct: 2, armed: false }, ...over,
});
const pos = (levels: PositionLevels | null, extra: Partial<OverlaySubject> = {}): OverlaySubject => ({ ticker: "QQQ", qty: 30, entryPrice: 100, currentPrice: 100.4, levels, ...extra });

test("a phase-1 long draws entry, stop, both thirds, the pending target and the hard floor (outside the fit), with the risk zone", () => {
  const o = positionOverlay(pos(lv()), T);
  const labels = o.levels.map((l) => l.label);
  assert.ok(labels.includes("entry"));
  assert.ok(labels.includes("stop · from entry"));
  assert.ok(labels.includes("sell ⅓ · +1.0R") && labels.includes("sell ⅓ · +2.0R"));
  assert.ok(labels.some((l) => l === "target +7.5% (after ⅔ out)"));
  const floor = o.levels.find((l) => l.label.startsWith("hard floor"));
  assert.ok(floor && floor.fit === false && floor.muted);
  assert.equal(o.zones[0].label, "risk to the stop");
  assert.ok(o.zones.some((z) => z.label === "to the first ⅓"));
  assert.ok(o.legend.some(([k, v]) => k === "phase" && v === "1 of 4"));
  assert.ok(o.legend.some(([k, v]) => k === "scale-outs" && v.startsWith("0 of 2 · next at +1.0R")));
  assert.deepEqual(o.notes, []);
});

test("a suppressed stop is muted, the override line is drawn and the note says how long it waits", () => {
  const o = positionOverlay(pos(lv({ stopSuppressed: { until: "day4", reason: "", overridePct: 15, minutesLeft: null, daysLeft: 2 }, daysHeld: 2 })), T);
  const stop = o.levels.find((l) => l.label.startsWith("stop · from entry"));
  assert.ok(stop && stop.muted && stop.label.endsWith("· waits"));
  const override = o.levels.find((l) => l.label.startsWith("exit now only below"));
  assert.ok(override && Math.abs(override.price - 85) < 1e-9);
  assert.ok(o.notes[0].includes("2 more day(s)"));
});

test("after the first third the done level is muted, the locked-in zone replaces the risk zone, and phase 3+ carries no fixed target", () => {
  const o = positionOverlay(pos(lv({ phase: 3, stopFrom: "peak_pnl", stopPrice: 106, takeProfitPct: null, takeProfitPrice: null, takeProfitArmed: false, breakevenActive: true, scalesCompleted: 1, scaleOuts: [{ n: 1, rMultiple: 1, pct: 5, price: 105, done: true, applicable: true }, { n: 2, rMultiple: 2, pct: 10, price: 110, done: false, applicable: true }] })), T);
  const first = o.levels.find((l) => l.label === "⅓ sold · +1.0R");
  assert.ok(first && first.muted);
  assert.equal(o.zones[0].label, "locked in");
  assert.ok(!o.levels.some((l) => l.label.startsWith("target")));
  assert.ok(o.notes.some((n) => n.startsWith("no fixed target in phase 3+")));
  assert.ok(o.legend.some(([k, v]) => k === "scale-outs" && v.includes("breakeven stop on")));
});

test("not recorded: only the entry line and the honest note; a one-share lot says it cannot scale out", () => {
  const o = positionOverlay(pos(null), T);
  assert.deepEqual(o.levels.map((l) => l.label), ["entry"]);
  assert.ok(o.notes[0].startsWith("stop state not recorded yet"));
  const tiny = positionOverlay(pos(lv({ scaleOuts: [{ n: 1, rMultiple: 1, pct: 5, price: 105, done: false, applicable: false }, { n: 2, rMultiple: 2, pct: 10, price: 110, done: false, applicable: false }] })), T);
  assert.ok(!tiny.levels.some((l) => l.label.includes("⅓")));
  assert.ok(tiny.notes.some((n) => n.startsWith("lot too small")));
});

test("an option leg draws its strike and expiry breakeven on the underlying with the keep-the-premium side shaded; missing meta falls back honestly", () => {
  const leg = overlayFor({ ticker: "CRWV261120P00055000", qty: -1, entryPrice: 2.47, currentPrice: 2.67, isOption: true, optionMeta: { underlyingTicker: "CRWV", strike: 55, optionType: "PUT", expiry: "2026-11-20", daysToExpiry: 30, strategy: "csp" } }, T);
  assert.ok(leg.levels.some((l) => l.label === "strike · short put" && l.price === 55));
  assert.ok(leg.levels.some((l) => l.label === "breakeven at expiry" && Math.abs(l.price - 52.53) < 1e-9));
  assert.ok(leg.zones.some((z) => z.label?.startsWith("keeps the whole premium") && z.lo === 55 && z.hi === Infinity));
  assert.ok(leg.zones.some((z) => z.label === "loses at expiry" && z.hi === 52.53));
  assert.ok(leg.legend.some(([k, v]) => k === "manager target" && v === "closes at 50% of max profit"));
  const bare = optionOverlay({ ticker: "X", qty: -1, entryPrice: 1, currentPrice: 1, isOption: true, optionMeta: null }, T);
  assert.equal(bare.levels.length, 0);
  assert.ok(bare.notes[0].startsWith("option leg details not recorded"));
});

test("underlyingOf: a stock charts itself, an option leg its underlying from the metadata or the OCC root", () => {
  assert.equal(underlyingOf({ ticker: "QQQ" }), "QQQ");
  assert.equal(underlyingOf({ ticker: "CRWV261120P00055000", isOption: true, optionMeta: { underlyingTicker: "CRWV" } }), "CRWV");
  assert.equal(underlyingOf({ ticker: "TEM261120P00070000", isOption: true, optionMeta: null }), "TEM");
  assert.equal(underlyingOf({ ticker: "", isOption: true }), null);
});

test("the entry marker is anchored at the fill time from heldMinutes; a candidate draws only the scan price with an honest note", () => {
  const now = Date.parse("2026-10-21T14:00:00Z");
  const o = positionOverlay(pos(lv({ heldMinutes: 90 })), T, now);
  assert.equal(o.markers.length, 1);
  assert.equal(o.markers[0].timeS, (now - 90 * 60_000) / 1000);
  assert.equal(o.markers[0].price, 100);
  assert.deepEqual(positionOverlay(pos(lv({ heldMinutes: null })), T, now).markers, []);
  const c = candidateOverlay({ ticker: "NVDA", rank: 2, score: 73.4, price: 184.2, action: "BUY", reasons: ["Momentum 1m +9.8%"], sector: "Semiconductors", admitted: true, changePct: 2.1 }, T);
  assert.deepEqual(c.levels.map((l) => [l.label, l.price, l.muted]), [["price at scan", 184.2, true]]);
  assert.ok(c.legend.some(([k, v]) => k === "score" && v === "73.4 · admitted"));
  assert.ok(c.notes[0].startsWith("a candidate has no stop or target yet"));
  assert.deepEqual(c.zones, []);
});
