// selfCheckView.test — the bot's self-check and its alerts, parsed from the
// shapes diagnostics.py and server/bot.ts answer with, and from the shapes the
// page must survive (an error body, another shape, nulls).
import test from "node:test";
import assert from "node:assert/strict";
import { parseSelfCheck, selfCheckSentence, selfCheckSettings, checkItemWords, appliedNote, parseAlerts, alertsSentence, alertTone } from "./selfCheckView.ts";

const HEALTHY = {
  report: { timestamp: "2026-11-20T06:00:00.000000", overall_status: "healthy", problems: [], warnings: [], auto_fixes: [], recommendations: [] },
  auto_fix: { position_size_multiplier: 1.0, min_score_threshold: 65, should_pause: false, force_retrain: false, problems_summary: "All systems healthy" },
};
const DEGRADED = {
  report: {
    timestamp: "2026-11-20T06:00:00.000000", overall_status: "degraded",
    problems: [
      { system: "cache", severity: "high", message: "Critical data sources stale: ['macro', 'insider']", auto_fix: "reduce_position_size", fix_params: { multiplier: 0.7 } },
      { system: "ml_model", severity: "medium", message: "Low win rate: 36% over 58 trades", auto_fix: "increase_score_threshold", fix_params: { min_score: 75 } },
    ],
    warnings: [{ system: "disk", message: "Cache using 410MB of 500MB limit" }],
    auto_fixes: ["Removed 12 expired cache files"],
    recommendations: ["Check the macro feed"],
  },
  auto_fix: { position_size_multiplier: 0.7, min_score_threshold: 75, should_pause: false, force_retrain: false, problems_summary: "[HIGH] Critical data sources stale" },
};

test("a clear self-check says so, with the settings it leaves alone", () => {
  const v = parseSelfCheck(HEALTHY)!;
  assert.equal(v.ok, true);
  assert.deepEqual(selfCheckSentence(v), { text: "Self-check clear · normal position size · score bar 65", tone: "ok" });
  assert.equal(appliedNote(v, 65), null);
});

test("problems and warnings are counted, and what they set is said in the same line", () => {
  const v = parseSelfCheck(DEGRADED)!;
  assert.deepEqual([v.problems.length, v.warnings.length, v.sizeMultiplier, v.scoreBar], [2, 1, 0.7, 75]);
  assert.deepEqual(selfCheckSentence(v), { text: "Self-check: 2 problems, 1 warning · new positions at ×0.70 of normal size · score bar 75", tone: "bad" });
  assert.deepEqual(checkItemWords(v.problems[0]), { lead: "Data freshness", text: "Critical data sources stale: ['macro', 'insider']", fix: "new positions are sized smaller" });
  assert.deepEqual(checkItemWords(v.problems[1]), { lead: "Model", text: "Low win rate: 36% over 58 trades", fix: "only higher-scoring picks are taken" });
  assert.deepEqual(checkItemWords(v.warnings[0]), { lead: "Disk", text: "Cache using 410MB of 500MB limit", fix: null });
  assert.deepEqual(v.housekeeping, ["Removed 12 expired cache files"]);
  assert.deepEqual(v.recommendations, ["Check the macro feed"]);
  // the loop applies the check on its own cadence: a different applied bar is said, an equal one is not
  assert.equal(appliedNote(v, 65), "The loop is using a score bar of 65 until it next applies the check.");
  assert.equal(appliedNote(v, 75), null);
  assert.equal(appliedNote(v, null), null);
});

test("tone: a medium problem warns, a high one or a pause is bad, warnings alone are not a problem", () => {
  const med = parseSelfCheck({ report: { problems: [{ system: "ml_model", severity: "medium", message: "Low win rate" }], warnings: [] }, auto_fix: { position_size_multiplier: 1, min_score_threshold: 75 } })!;
  assert.equal(selfCheckSentence(med).tone, "warn");
  const pause = parseSelfCheck({ report: { problems: [{ system: "api", severity: "medium", message: "Multiple API sources down" }] }, auto_fix: { should_pause: true, force_retrain: true } })!;
  assert.equal(selfCheckSentence(pause).tone, "bad");
  assert.equal(selfCheckSettings(pause), "model retrain asked for · it asks the loop to PAUSE");
  const warn = parseSelfCheck({ report: { problems: [], warnings: [{ system: "ml_model", message: "ML model retrain needed" }] }, auto_fix: { position_size_multiplier: 1, min_score_threshold: 65 } })!;
  assert.deepEqual(selfCheckSentence(warn), { text: "Self-check: 1 warning · normal position size · score bar 65", tone: "ok" });
});

test("an error body is 'did not run', never 'clear'; another shape is null; unknown codes keep their own words", () => {
  const err = parseSelfCheck({ error: "Command failed: python3 -c ... timed out" })!;
  assert.equal(err.ok, false);
  assert.deepEqual(selfCheckSentence(err), { text: "Self-check did not run: Command failed: python3 -c ... timed out", tone: "warn" });
  assert.equal(parseSelfCheck(null), null);
  assert.equal(parseSelfCheck([]), null);
  assert.equal(parseSelfCheck({ hello: 1 }), null);
  // a report with no settings block says nothing about settings rather than inventing "normal size"
  const bare = parseSelfCheck({ report: { problems: [], warnings: [] } })!;
  assert.equal(selfCheckSentence(bare).text, "Self-check clear");
  const odd = parseSelfCheck({ report: { problems: [{ system: "new_thing", message: "x happened", auto_fix: "do_something_new" }, { message: "" }, "a bare line"] }, auto_fix: {} })!;
  assert.equal(odd.problems.length, 2);
  assert.deepEqual(checkItemWords(odd.problems[0]), { lead: "new thing", text: "x happened", fix: "do something new" });
  assert.deepEqual(checkItemWords(odd.problems[1]), { lead: "other", text: "a bare line", fix: null });
});

const NOW = Date.parse("2026-11-20T15:00:00Z");
const ALERTS = [
  { time: "2026-11-20T14:58:00Z", type: "trade", message: "OPTIONS: SELL 1x AAL $11.5 PUT (2026-12-31)", read: false },
  { time: "2026-11-20T14:40:00Z", type: "alert", message: "Daily loss: -3.1%. Trading halted.", read: false },
  { time: "2026-11-20T13:31:00Z", type: "system", message: "Bot activated — scanning for opportunities", read: true },
  { time: "2026-11-20T14:59:00Z", type: "critical", message: "x".repeat(140), read: false },
];

test("alerts: newest first, unread counted, the newest quoted (and cut at a card's width)", () => {
  const rows = parseAlerts(ALERTS)!;
  assert.deepEqual(rows.map((r) => r.kind), ["critical", "trade", "alert", "system"]);
  assert.equal(alertTone("critical"), "bad");
  assert.equal(alertTone("alert"), "warn");
  assert.equal(alertTone("trade"), "none");
  const say = alertsSentence(rows, NOW);
  assert.ok(say.startsWith("4 alerts · 3 new · newest 1m ago: "), say);
  assert.ok(say.endsWith("…") && say.length < 160, say);
  assert.equal(alertsSentence(rows.slice(1), NOW), "3 alerts · 2 new · newest 2m ago: OPTIONS: SELL 1x AAL $11.5 PUT (2026-12-31)");
  assert.equal(alertsSentence(rows.slice(3), NOW), "1 alert · newest 1h ago: Bot activated — scanning for opportunities");
});

test("alerts: an empty list is a sentence, a wrong shape is null, a row without a time or text is dropped", () => {
  assert.equal(alertsSentence(parseAlerts([])!, NOW), "No alerts since the server last restarted.");
  assert.equal(parseAlerts({ error: "x" }), null);
  assert.equal(parseAlerts(null), null);
  const rows = parseAlerts([null, { time: "nope", message: "x" }, { time: "2026-11-20T14:00:00Z" }, { time: "2026-11-20T14:00:00Z", type: "brand-new", message: "kept" }])!;
  assert.deepEqual(rows.map((r) => [r.kind, r.message, r.read]), [["other", "kept", false]]);
});
