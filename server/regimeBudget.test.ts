// Regime risk budget on the report / diag probe / health / audit (2026-11-06, refocus spec
// section 3 step 1). The Python engine computes it (regime_engine.risk_budget_series); this
// side only shapes what it persisted. Run: npx tsx --test server/regimeBudget.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  REGIME_FLAG_DEFAULTS, regimeFlags, regimeHealth, regimeReportBody, regimeRiskBudget, regimeBudgetAudit,
  riskBudgetEnabled, RISK_BUDGET_CONSUMERS, type RegimeStateFile,
} from "./regimeReport";

const drivers = [
  { name: "trend", label: "broad trend", value: 0.75, score: 0.75, weight: 0.3, cut: 0.075, short: "SPY trend mostly up (score 0.75)", reason: "trend score 0.75: ..." },
  { name: "term_structure", label: "VIX term structure", value: 1.04, score: 0.3, weight: 0.3, cut: 0.21, short: "VIX term structure inverted (1.04)", reason: "VIX/VIX3M 1.04: backwardation ..." },
  { name: "vix_level", label: "VIX level", value: 24.0, score: 0.7, weight: 0.25, cut: 0.075, short: "VIX 24.0 (1.4x its 10-year median 16.9)", reason: "..." },
  { name: "breadth", label: "breadth", value: 0.38, score: 0.45, weight: 0.15, cut: 0.0825, short: "breadth 38% (5 of 13 sector/size ETFs above their 200-day)", reason: "..." },
];
const SUMMARY = "Risk budget 0.56 (reduced): VIX term structure inverted (1.04) -0.21; breadth 38% (5 of 13 sector/size ETFs above their 200-day) -0.08; SPY trend mostly up (score 0.75) -0.07; VIX 24.0 (1.4x its 10-year median 16.9) -0.07.";

const state = (over: Partial<RegimeStateFile> = {}): RegimeStateFile => ({
  as_of: "2026-11-05", composite_regime: "VOLATILE_BULL", legacy_label: "CAUTION", exposure_multiplier: 0.75,
  risk_budget: 0.5575, budget_version: "2026-11-06.1",
  risk_budget_detail: { budget: 0.5575, raw: 0.5575, band: "reduced", summary: SUMMARY, drivers, missing: [] },
  ...over,
});
const NOW = new Date("2026-11-06T15:00:00Z");

test("the budget flag is mirrored and defaults on; an explicit off wins", () => {
  assert.equal(REGIME_FLAG_DEFAULTS.VOLTRADE_REGIME_BUDGET, "1");
  assert.equal(riskBudgetEnabled(regimeFlags({}), true), true);
  assert.equal(riskBudgetEnabled(regimeFlags({ VOLTRADE_REGIME_BUDGET: "off" }), true), false);
  assert.equal(riskBudgetEnabled(regimeFlags({}), false), false);        // engine off -> never applied
  assert.equal(riskBudgetEnabled(null, true), true);                      // no flags passed -> the default
  assert.deepEqual([...RISK_BUDGET_CONSUMERS], ["tier_exposure_cap"]);
});

test("block: budget, band, plain-English summary and drivers, applied when fresh and on", () => {
  const b = regimeRiskBudget(state(), [], NOW, true, regimeFlags({}));
  assert.equal(b.budget, 0.5575);
  assert.equal(b.band, "reduced");
  assert.equal(b.summary, SUMMARY);
  assert.match(b.summary ?? "", /^Risk budget 0\.56 \(reduced\): VIX term structure inverted \(1\.04\)/);
  assert.equal(b.drivers.length, 4);
  assert.deepEqual(b.drivers.map((d) => d.name), ["trend", "term_structure", "vix_level", "breadth"]);
  assert.equal(b.drivers[1].short, "VIX term structure inverted (1.04)");
  assert.equal(b.stale, false);
  assert.equal(b.enabled, true);
  assert.equal(b.applied, true);
  assert.match(b.note, /^applied: the tier engine's room above the passive floor/);
});

test("block: not applied when the flag is off, the state is stale, or there is no budget yet", () => {
  const off = regimeRiskBudget(state(), [], NOW, true, regimeFlags({ VOLTRADE_REGIME_BUDGET: "0" }));
  assert.equal(off.applied, false);
  assert.match(off.note, /VOLTRADE_REGIME_BUDGET is off/);
  const stale = regimeRiskBudget(state({ as_of: "2026-10-20" }), [], NOW, true, regimeFlags({}));
  assert.equal(stale.stale, true);
  assert.equal(stale.applied, false);
  assert.match(stale.note, /stale/);
  assert.equal(regimeRiskBudget(state({ error: "network" }), [], NOW, true, regimeFlags({})).applied, false);
  const none = regimeRiskBudget(state({ risk_budget: null, risk_budget_detail: null }), [], NOW, true, regimeFlags({}));
  assert.equal(none.budget, null);
  assert.equal(none.applied, false);
  assert.deepEqual(none.drivers, []);
  assert.match(none.note, /no risk budget/);
  assert.equal(regimeRiskBudget(null, [], NOW, true).budget, null);
});

test("history: one row per persisted day with a budget (pre-budget rows skipped)", () => {
  const hist = [
    state({ as_of: "2026-11-03", risk_budget: undefined, risk_budget_detail: undefined }),
    state({ as_of: "2026-11-04", risk_budget: 0.92, risk_budget_detail: { band: "full" } }),
    state({ as_of: "2026-11-05" }),
  ];
  const b = regimeRiskBudget(state(), hist, NOW, true, regimeFlags({}));
  assert.deepEqual(b.history, [{ as_of: "2026-11-04", budget: 0.92, band: "full" }, { as_of: "2026-11-05", budget: 0.5575, band: "reduced" }]);
});

test("report body, diag probe and health carry the budget; the existing note is unchanged", () => {
  const body = regimeReportBody(state(), [state()], NOW, true, null, regimeFlags({}));
  assert.equal(body.risk_budget.budget, 0.5575);
  assert.equal(body.risk_budget.applied, true);
  assert.equal(body.history[0].risk_budget, 0.5575);
  assert.match(body.note, /^live: VOLTRADE_REGIME_SELECTION, VOLTRADE_REGIME_OPTIONS, VOLTRADE_REGIME_SIZING, VOLTRADE_REGIME_LABELS act/);
  assert.equal(regimeHealth(state(), NOW, true).risk_budget, 0.5575);
  assert.equal(regimeHealth(state({ risk_budget: undefined }), NOW, true).risk_budget, null);
});

test("RISK-BUDGET audit: once per (as_of, band, applied); errors are loud; nothing without a budget", () => {
  const view = { budget: 0.5575, band: "reduced", summary: SUMMARY, as_of: "2026-11-05", applied: true, stale: false };
  const first = regimeBudgetAudit(view, "");
  assert.ok(first);
  assert.equal(first?.key, "2026-11-05|reduced|applied");
  assert.equal(first?.line, `${SUMMARY} [applied to the tier engine's satellite headroom; as_of 2026-11-05]`);
  assert.equal(regimeBudgetAudit(view, first?.key ?? ""), null);                       // same day, same band: silent
  assert.ok(regimeBudgetAudit({ ...view, band: "defensive" }, first?.key ?? ""));      // band change: a new line
  assert.ok(regimeBudgetAudit({ ...view, as_of: "2026-11-06" }, first?.key ?? ""));    // a new day: a new line
  const off = regimeBudgetAudit({ ...view, applied: false }, first?.key ?? "");
  assert.match(off?.line ?? "", /not applied: VOLTRADE_REGIME_BUDGET off/);
  assert.match(regimeBudgetAudit({ ...view, applied: false, stale: true }, "")?.line ?? "", /not applied: stale regime state/);
  assert.match(regimeBudgetAudit({ error: "boom" }, "")?.line ?? "", /^risk budget view failed: boom/);
  assert.equal(regimeBudgetAudit({ error: "boom" }, "error|boom"), null);
  assert.equal(regimeBudgetAudit(null, ""), null);
  assert.equal(regimeBudgetAudit({ budget: null }, ""), null);
});
