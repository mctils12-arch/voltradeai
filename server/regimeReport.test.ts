// regimeReport — read-only shaping of the regime engine's persisted state
// (docs/regime/FINDINGS.md step 3). Run: npx tsx --test server/regimeReport.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import {
  REGIME_STATE_FILE, REGIME_HISTORY_FILE, regimeEngineEnabled, readRegimeState, readRegimeHistory,
  marketDaysBetween, regimeHealth, regimeReportBody, regimeTransitionAudit, regimeTransitionAuditOnce, regimeTransitionKey,
  regimeTransitionKeyOfLine, readRegimeSizing, REGIME_SIZING_FILE,
  REGIME_FLAG_DEFAULTS, regimeFlags, regimeFlagRaw, regimeConsumersOn,
} from "./regimeReport";
import { SERVING_CHECKS } from "./healthGate";

const state = (over: Record<string, unknown> = {}) => ({
  as_of: "2026-10-26", gating_index: "SPY", composite_regime: "CALM_BULL", legacy_label: "BULL",
  trend_state: "UP", fast_state: "MIXED", vol_state: "LOW", breadth_state: "NARROW",
  exposure_multiplier: 1, core_multiplier: 1, confidence: 1, flips_12m: 3, composite_changes_12m: 3, alarm: false, ...over,
});

function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "regime-report-"));
}

test("flag parsing mirrors the Python gate (1/true/on/yes; default ON since 2026-10-28 when the env var is absent)", () => {
  // 2026-10-28 human directive: the default flipped (was false) — an ABSENT env var now reads the
  // system_config default (REGIME_ENGINE_ENABLED True); a PRESENT env var always wins.
  assert.equal(regimeEngineEnabled({}), true);
  assert.equal(regimeEngineEnabled({ VOLTRADE_REGIME_ENGINE: "0" }), false);
  assert.equal(regimeEngineEnabled({ VOLTRADE_REGIME_ENGINE: "1" }), true);
  assert.equal(regimeEngineEnabled({ VOLTRADE_REGIME_ENGINE: "TRUE" }), true);
});

test("an explicitly set env var always wins over the default (rollback without a deploy)", () => {
  for (const v of ["off", "0", "", "false", " OFF "]) {
    assert.equal(regimeEngineEnabled({ VOLTRADE_REGIME_ENGINE: v }), false, `engine env ${JSON.stringify(v)}`);
  }
  const all = regimeFlags({});
  assert.deepEqual(Object.keys(all).sort(), Object.keys(REGIME_FLAG_DEFAULTS).sort());
  for (const [name, f] of Object.entries(all)) {
    assert.equal(f.source, "config", name);
    assert.equal(f.value, REGIME_FLAG_DEFAULTS[name], name);
  }
  assert.deepEqual(regimeConsumersOn(all), ["VOLTRADE_REGIME_SELECTION", "VOLTRADE_REGIME_OPTIONS", "VOLTRADE_REGIME_SIZING", "VOLTRADE_REGIME_LABELS"]);
  const off = regimeFlags({ VOLTRADE_REGIME_SELECTION: "off", VOLTRADE_REGIME_OPTIONS: "", VOLTRADE_REGIME_SIZING: "0", VOLTRADE_REGIME_LABELS: "false" });
  assert.deepEqual(regimeConsumersOn(off), []);
  assert.deepEqual(off.VOLTRADE_REGIME_OPTIONS, { value: "", source: "env" });
  assert.deepEqual(regimeFlagRaw("VOLTRADE_REGIME_SIZING", { VOLTRADE_REGIME_SIZING: "stock" }), { value: "stock", source: "env" });
});

test("report body carries the effective flags and says live/shadow from them", () => {
  const now = new Date("2026-10-27T15:00:00Z");
  const live = regimeReportBody(state(), [], now, true, null, regimeFlags({}));
  assert.equal(live.flags?.VOLTRADE_REGIME_SIZING.value, "all");
  assert.match(live.note, /^live: VOLTRADE_REGIME_SELECTION, VOLTRADE_REGIME_OPTIONS, VOLTRADE_REGIME_SIZING, VOLTRADE_REGIME_LABELS act/);
  const allOff = regimeFlags({ VOLTRADE_REGIME_SELECTION: "off", VOLTRADE_REGIME_OPTIONS: "off", VOLTRADE_REGIME_SIZING: "off", VOLTRADE_REGIME_LABELS: "off" });
  assert.match(regimeReportBody(state(), [], now, true, null, allOff).note, /^shadow:/);
  assert.match(regimeReportBody(state(), [], now, false, null, regimeFlags({ VOLTRADE_REGIME_ENGINE: "off" })).note, /^shadow:/);
  assert.equal(regimeReportBody(state(), [], now, true).flags, null);
});

test("readers: missing files are null/[]; a torn last history line is skipped", () => {
  const dir = tmpDir();
  assert.equal(readRegimeState(dir), null);
  assert.deepEqual(readRegimeHistory(dir), []);
  fs.writeFileSync(path.join(dir, REGIME_STATE_FILE), JSON.stringify(state()));
  fs.writeFileSync(path.join(dir, REGIME_HISTORY_FILE),
    [JSON.stringify(state({ as_of: "2026-10-23" })), JSON.stringify(state({ as_of: "2026-10-24", composite_regime: "CHOPPY" })), '{"as_of": "2026-10-2'].join("\n"));
  assert.equal(readRegimeState(dir)?.composite_regime, "CALM_BULL");
  const hist = readRegimeHistory(dir);
  assert.equal(hist.length, 2);
  assert.equal(hist[1].composite_regime, "CHOPPY");
  assert.equal(readRegimeHistory(dir, 1).length, 1);
});

test("market days between: weekdays only, strictly after as_of", () => {
  assert.equal(marketDaysBetween("2026-10-23", new Date("2026-10-27T15:00:00Z")), 2);   // Fri -> Tue
  assert.equal(marketDaysBetween("2026-10-26", new Date("2026-10-26T23:00:00Z")), 0);
  assert.equal(marketDaysBetween(undefined, new Date()), null);
  assert.equal(marketDaysBetween("garbage", new Date()), null);
});

test("health: off without state, warning when enabled but empty, ok when fresh, warning when stale/alarm/error; never gates serving", () => {
  const now = new Date("2026-10-27T15:00:00Z");
  assert.equal(regimeHealth(null, now, false).status, "off");
  assert.equal(regimeHealth(null, now, true).status, "warning");
  const fresh = regimeHealth(state(), now, true);
  assert.equal(fresh.status, "ok");
  assert.equal(fresh.age_market_days, 1);
  assert.equal(fresh.gates_top_level_status, false);
  assert.equal(regimeHealth(state(), new Date("2026-11-05T15:00:00Z"), true).status, "warning");
  assert.match(regimeHealth(state({ alarm: true, flips_12m: 7 }), now, true).detail ?? "", /flip alarm/);
  assert.match(regimeHealth(state({ error: "network" }), now, true).detail ?? "", /last update failed/);
  assert.equal(regimeHealth(state(), now, false).status, "off");
  assert.ok(!(SERVING_CHECKS as readonly string[]).includes("regime"), "the regime block must never be a serving gate");
});

test("report: history rows, transitions and time-at-levels are derived from the history file", () => {
  const now = new Date("2026-10-27T15:00:00Z");
  const hist = [
    state({ as_of: "2026-10-20", exposure_multiplier: 1 }),
    state({ as_of: "2026-10-21", exposure_multiplier: 0.75, composite_regime: "VOLATILE_BULL" }),
    state({ as_of: "2026-10-22", exposure_multiplier: 0.75, composite_regime: "VOLATILE_BULL" }),
    state({ as_of: "2026-10-23", exposure_multiplier: 1, composite_regime: "CALM_BULL" }),
  ];
  const body = regimeReportBody(state(), hist, now, true);
  assert.equal(body.enabled_env, true);
  assert.equal(body.state_present, true);
  assert.equal(body.history_days, 4);
  assert.deepEqual(body.transitions.map((t) => `${t.from}>${t.to}`), ["CALM_BULL>VOLATILE_BULL", "VOLATILE_BULL>CALM_BULL"]);
  assert.deepEqual(body.time_at_levels, { "1.00": 2, "0.75": 2 });
  assert.equal(body.health.status, "ok");
  const empty = regimeReportBody(null, [], now, false);
  assert.equal(empty.state_present, false);
  assert.equal(empty.health.status, "off");
});

test("transition audit text: null without a transition, one line with it", () => {
  assert.equal(regimeTransitionAudit(state()), null);
  assert.equal(regimeTransitionAudit(null), null);
  const line = regimeTransitionAudit(state({ transition: { from: "CALM_BULL", to: "BEAR", to_as_of: "2026-10-26" }, exposure_multiplier: 0.25 }));
  assert.match(line ?? "", /^CALM_BULL -> BEAR \(as_of 2026-10-26; exposure 0.25/);
});

// KNOWN BROKEN #64 (2026-11-27). regime_engine.py persists `transition` with the state and
// returns the persisted state on every scan until the bar date advances. Production wrote
// "CALM_BULL -> CRISIS (as_of 2026-11-25; exposure 0, core 0.5, confidence 0.667)" 177 times
// between 2026-11-26T00:03Z and 2026-11-27T20:17Z: 8% of the audit log's rows in that span.
const crisis = () => state({
  as_of: "2026-11-25", composite_regime: "CRISIS", exposure_multiplier: 0, core_multiplier: 0.5, confidence: 0.667,
  transition: { from: "CALM_BULL", to: "CRISIS", from_as_of: "2026-11-24", to_as_of: "2026-11-25" },
});

test("#64: the same persisted transition on every scan is audited once, not once a scan", () => {
  let lastKey: string | null = "";            // the audit log was read and held no transition line
  const lines: string[] = [];
  for (let scan = 0; scan < 177; scan++) {
    const once = regimeTransitionAuditOnce(crisis(), lastKey);
    if (once) { lines.push(once.line); lastKey = once.key; }
  }
  assert.deepEqual(lines, ["CALM_BULL -> CRISIS (as_of 2026-11-25; exposure 0, core 0.5, confidence 0.667)"]);
});

test("#64: a restart reads the last audited transition back from its own audit line and stays silent", () => {
  const written = regimeTransitionAudit(crisis());
  const seeded = regimeTransitionKeyOfLine(written);
  assert.equal(seeded, "CALM_BULL|CRISIS|2026-11-25");
  assert.equal(seeded, regimeTransitionKey(crisis()), "the key read from the line is the key of the state that wrote it");
  assert.equal(regimeTransitionAuditOnce(crisis(), seeded), null);
});

test("#64: the next transition is still audited, and so is a first one on an empty or unreadable log", () => {
  const back = state({ as_of: "2026-11-30", transition: { from: "CRISIS", to: "CALM_BULL", to_as_of: "2026-11-30" } });
  assert.match(regimeTransitionAuditOnce(back, "CALM_BULL|CRISIS|2026-11-25")?.line ?? "", /^CRISIS -> CALM_BULL \(as_of 2026-11-30;/);
  // the same pair of regimes on a later bar date is a new transition
  const again = state({ as_of: "2026-12-09", composite_regime: "CRISIS", transition: { from: "CALM_BULL", to: "CRISIS", to_as_of: "2026-12-09" } });
  assert.ok(regimeTransitionAuditOnce(again, "CALM_BULL|CRISIS|2026-11-25"));
  assert.ok(regimeTransitionAuditOnce(crisis(), ""));
  assert.ok(regimeTransitionAuditOnce(crisis(), null));
  // no transition: nothing, whatever was audited before
  assert.equal(regimeTransitionAuditOnce(state(), ""), null);
  assert.equal(regimeTransitionAuditOnce(null, ""), null);
  assert.equal(regimeTransitionKey(state({ transition: null })), null);
  // a transition with no to_as_of is keyed by the state's own bar date, as the line prints it
  const bare = state({ as_of: "2026-12-01", transition: { from: "CALM_BULL", to: "BEAR" } });
  assert.equal(regimeTransitionKey(bare), regimeTransitionKeyOfLine(regimeTransitionAudit(bare)));
});

test("#64: text that is not a transition line gives no key", () => {
  for (const m of [undefined, null, 7, "", "regime engine update failed: x", "CALM_BULL to CRISIS"]) {
    assert.equal(regimeTransitionKeyOfLine(m), null);
  }
});

test("#64: bot.ts writes REGIME-TRANSITION only through the once-per-transition path, seeded from the audit log", () => {
  const bot = fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), "bot.ts"), "utf8");
  assert.equal(bot.split('audit("REGIME-TRANSITION"').length - 1, 1, "one write site");
  assert.match(bot, /regimeTransitionAuditOnce\(result\.regime, lastRegimeTransitionKey\);\s+if \(regimeTransition\) \{\s+audit\("REGIME-TRANSITION", regimeTransition\.line\);\s+lastRegimeTransitionKey = regimeTransition\.key;/);
  assert.match(bot, /regimeTransitionKeyOfLine\(\s*queryAuditLog\(db, \{ limit: 1, type: "REGIME-TRANSITION" \}\)\[0\]\?\.message\)/);
});

test("sizing counters (step 6): missing file is null; today's stale fallbacks warn in health and ride the report", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "regime-sizing-"));
  assert.equal(readRegimeSizing(dir), null);
  const now = new Date("2026-10-28T15:00:00Z");
  const sizing = { day: "2026-10-28", by_sleeve: { stock: { "0.75": 4 }, tier1: { "1.00": 2 } }, stale_fallbacks: 2, zero_blocks: 0, last: null };
  fs.writeFileSync(path.join(dir, REGIME_SIZING_FILE), JSON.stringify(sizing));
  assert.deepEqual(readRegimeSizing(dir)?.by_sleeve, sizing.by_sleeve);
  const fresh = state({ as_of: "2026-10-28" });
  const h = regimeHealth(fresh, now, true, sizing);
  assert.equal(h.status, "warning");
  assert.match(h.detail ?? "", /stale state 2x today/);
  assert.equal(h.gates_top_level_status, false);
  assert.equal(regimeHealth(fresh, now, true, { ...sizing, day: "2026-10-27" }).status, "ok");   // yesterday's counters do not warn today
  assert.equal(regimeHealth(fresh, now, true, { ...sizing, stale_fallbacks: 0 }).status, "ok");
  const body = regimeReportBody(fresh, [], now, true, sizing);
  assert.deepEqual(body.sizing, sizing);
  assert.equal(regimeReportBody(null, [], now, false).sizing, null);
});

