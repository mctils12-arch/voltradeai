// Weekly forecast screen (2026-10-27, SHADOW) — the Node half: universe,
// scheduling, the exact python3 -c commands (run through server/pythonExec.ts's
// planPythonExec — what bot.ts's execPythonSerialized uses — so the 90 s kill
// cap is measured, not assumed), and the
// /api/forecast + daemon wiring. The model and store are test_forecast.py's.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { isTrivialPython, planPythonExec } from "./pythonExec";
import {
  occUnderlying, forecastUniverse, ivMapFrom, forecastPayloadPath, forecastGenerateCmd, forecastScoreCmd,
  shouldAttemptGenerate, shouldScore, FORECAST_REPORT_CMD, FORECAST_GENERATE_EVERY_MS, FORECAST_TIME_BUDGET_S,
  FORECAST_JOB_TIMEOUT_MS, TRIVIAL_PYTHON_MAX_CHARS, FORECAST_QUEUE_MAX_WAIT_MS, FORECAST_QUEUE_BUDGET_MS,
  vixFromCboe, forecastDailyPayloadPath, forecastGenerateDailyCmd, dailyAuditKind, dailyWantsVixRefresh,
} from "./forecast";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const daemon = fs.readFileSync(path.join(here, "..", "voltrade_daemon.py"), "utf8");

test("occUnderlying: OCC codes map to their root, stocks to null", () => {
  assert.equal(occUnderlying("AAPL261218C00200000"), "AAPL");
  assert.equal(occUnderlying("SPY261120P00550000"), "SPY");
  assert.equal(occUnderlying("AAPL"), null);
  assert.equal(occUnderlying(""), null);
});

test("forecastUniverse: SPY first, positions (options as underlyings), then capped candidates, deduped", () => {
  const u = forecastUniverse(
    [{ symbol: "XLE" }, { symbol: "NVDA261218C00200000" }, { symbol: "SPY" }, { symbol: "bad sym" }],
    [{ ticker: "NVDA" }, { ticker: "AMD" }, { ticker: "KWEB" }, { ticker: "PFE" }],
    2,
  );
  assert.deepEqual(u, [
    { symbol: "SPY", group: "index" }, { symbol: "XLE", group: "position" },
    { symbol: "NVDA", group: "position" }, { symbol: "AMD", group: "candidate" },
  ]);
  assert.deepEqual(forecastUniverse(null, undefined), [{ symbol: "SPY", group: "index" }]);
  const many = Array.from({ length: 50 }, (_, k) => ({ symbol: `Q${String.fromCharCode(65 + (k % 26))}${String.fromCharCode(65 + Math.floor(k / 26))}` }));
  assert.equal(forecastUniverse(many, []).length, 30);
});

test("ivMapFrom: keeps finite positive IV30 for wanted names only", () => {
  const m = ivMapFrom([{ u: "SPY", iv30: 0.14 }, { u: "AAPL", iv30: null }, { u: "QQQ", iv30: 0.2 }, { u: "XLE", iv30: 0 }], ["SPY", "AAPL", "XLE"]);
  assert.deepEqual(m, { SPY: 0.14 });
});

test("commands: both bar-fetching jobs take the mutex path (90 s cap), never the 10 s trivial kill", () => {
  const gen = forecastGenerateCmd(forecastPayloadPath(1_790_000_000_000));
  const score = forecastScoreCmd();
  for (const cmd of [gen, score]) {
    assert.ok(cmd.length > TRIVIAL_PYTHON_MAX_CHARS, `${cmd.length} chars: ${cmd}`);
    assert.equal(isTrivialPython(cmd, FORECAST_JOB_TIMEOUT_MS), false, cmd);
    const plan = planPythonExec(cmd, FORECAST_JOB_TIMEOUT_MS);
    assert.equal(plan.lane, "mutex", cmd);
    assert.equal(plan.timeoutMs, Math.min(FORECAST_JOB_TIMEOUT_MS, 90_000));
  }
  assert.ok(FORECAST_TIME_BUDGET_S * 1000 + 20_000 <= FORECAST_JOB_TIMEOUT_MS, "Python budget + fit/write must fit under the hard kill");
  assert.ok(score.includes(`time_budget_s=${FORECAST_TIME_BUDGET_S}`));
  assert.ok(gen.includes("from forecast import generate_weekly") && gen.includes("os.remove(p)"));
  assert.ok(FORECAST_REPORT_CMD.includes("from forecast import report"));
});

test("commands: the payload path is the only interpolated value and must be a safe temp path", () => {
  assert.equal(forecastPayloadPath(123.9), "/tmp/vt_forecast_123.json");
  assert.throws(() => forecastGenerateCmd("/tmp/x; rm -rf /"));
  assert.throws(() => forecastGenerateCmd("/etc/passwd"));
});

test("scheduling: generate at most hourly; score once per UTC day from 22:00", () => {
  const t = Date.UTC(2026, 9, 23, 21, 30);
  assert.equal(shouldAttemptGenerate(t, null), true);
  assert.equal(shouldAttemptGenerate(t, t - FORECAST_GENERATE_EVERY_MS + 1), false);
  assert.equal(shouldAttemptGenerate(t, t - FORECAST_GENERATE_EVERY_MS), true);
  assert.equal(shouldScore(Date.UTC(2026, 9, 23, 21, 59), null), false);
  assert.equal(shouldScore(Date.UTC(2026, 9, 23, 22, 0), null), true);
  assert.equal(shouldScore(Date.UTC(2026, 9, 23, 23, 0), "2026-10-23"), false);
  assert.equal(shouldScore(Date.UTC(2026, 9, 24, 22, 5), "2026-10-23"), true);
});

test("wiring: /api/forecast is owner-only, read-only, daemon-first with the subprocess fallback", () => {
  const i = bot.indexOf('app.get("/api/forecast", requireOwner,');
  assert.ok(i > 0, "GET /api/forecast must be registered with requireOwner (the calls name open positions)");
  const block = bot.slice(i, bot.indexOf("\n  });\n", i));
  assert.ok(block.includes('pythonCall("forecast_report", {}, FORECAST_REPORT_CMD'));
  assert.ok(!/app\.(post|put|delete)\("\/api\/forecast/.test(bot), "the forecast API is read-only");
});

test("wiring: the job runs on its own unconditional interval (not Tier 3, not market-hours gated)", () => {
  assert.ok(/setInterval\(\(\) => \{ checkForecasts\(\)\.catch\(/.test(bot));
  const fn = bot.slice(bot.indexOf("async function checkForecasts()"), bot.indexOf("// ── Tier 2: Intelligence"));
  assert.ok(fn.includes("execPythonSerialized(forecastGenerateCmd(tmp), { timeout: FORECAST_JOB_TIMEOUT_MS })"));
  assert.ok(fn.includes("execPythonSerialized(forecastScoreCmd(), { timeout: FORECAST_JOB_TIMEOUT_MS })"));
  assert.ok(fn.includes("FORECAST-GENERATE-ERROR") && fn.includes("FORECAST-SCORE-ERROR"), "failures are audited");
  assert.ok(!/state\.active|killSwitch|is_open/.test(fn), "shadow analytics must not depend on trading state");
  assert.ok(!/\/v2\/orders|submit_options_order/.test(fn), "the forecast job never touches order flow");
  const tier3 = bot.slice(bot.indexOf("async function tier3Strategic()"), bot.indexOf("async function tier3Strategic()") + 20_000);
  assert.ok(!tier3.includes("checkForecasts"));
});

test("wiring: the daemon routes the four entry points", () => {
  for (const [m, attr] of [["forecast_report", "report"], ["forecast_generate", "generate_weekly"], ["forecast_generate_daily", "generate_daily"], ["forecast_score", "score_matured"]]) {
    assert.ok(daemon.includes(`"${m}": ("forecast", "${attr}")`), `${m} missing from voltrade_daemon.py _routes`);
  }
});

// ── daily horizon (Part C, 2026-11-04) ──────────────────────────────────────
test("daily: the Cboe VIX close becomes {value, day}; anything else is null", () => {
  assert.deepEqual(vixFromCboe({ date: "2026-11-03", vix: 20.83 }), { value: 20.83, day: "2026-11-03" });
  assert.equal(vixFromCboe(null), null);
  assert.equal(vixFromCboe(undefined), null);
  assert.equal(vixFromCboe({ date: "2026-11-03", vix: null }), null);
  assert.equal(vixFromCboe({ date: "11/03/2026", vix: 20.8 }), null);
  assert.equal(vixFromCboe({ date: "2026-11-03", vix: 0 }), null);
  assert.equal(vixFromCboe({ date: "2026-11-03", vix: "20.8" }), null);
});

test("daily: its command takes the mutex path too, and only a safe temp path is interpolated", () => {
  const cmd = forecastGenerateDailyCmd(forecastDailyPayloadPath(1_790_000_000_000));
  assert.ok(cmd.length > TRIVIAL_PYTHON_MAX_CHARS, cmd);
  assert.equal(isTrivialPython(cmd, FORECAST_JOB_TIMEOUT_MS), false);
  assert.equal(planPythonExec(cmd, FORECAST_JOB_TIMEOUT_MS).lane, "mutex");
  assert.ok(cmd.includes("from forecast import generate_daily") && cmd.includes("os.remove(p)"));
  assert.equal(forecastDailyPayloadPath(5.7), "/tmp/vt_forecast_daily_5.json");
  assert.throws(() => forecastGenerateDailyCmd("/tmp/vt_forecast_1.json"), "the weekly payload path is not the daily one");
  assert.throws(() => forecastGenerateDailyCmd("/tmp/vt_forecast_daily_1.json; rm -rf /"));
});

test("daily: only runs that made calls and failures are audited; a VIX wait asks for one refresh", () => {
  assert.equal(dailyAuditKind({ status: "ok", new_calls: 9 }), "FORECAST-DAILY");
  assert.equal(dailyAuditKind({ status: "ok", new_calls: 0 }), null);          // the hourly live-band refresh
  assert.equal(dailyAuditKind({ status: "waiting", reason: "vix_not_published" }), null);
  assert.equal(dailyAuditKind({ status: "skipped", reason: "exists" }), null);
  assert.equal(dailyAuditKind({ status: "error", reason: "spy_base_bar_missing" }), "FORECAST-DAILY-ERROR");
  assert.equal(dailyAuditKind(null), "FORECAST-DAILY-ERROR");
  assert.equal(dailyWantsVixRefresh({ status: "waiting", reason: "vix_not_published" }), true);
  assert.equal(dailyWantsVixRefresh({ status: "waiting", reason: "options_capture_window" }), false);
  assert.equal(dailyWantsVixRefresh({ status: "ok", new_calls: 1 }), false);
});

test("wiring: the daily job shares the hourly slot, reuses the weekly payload, audits failures, never touches order flow", () => {
  const fn = bot.slice(bot.indexOf("async function checkForecasts()"), bot.indexOf("// ── Tier 2: Intelligence"));
  assert.ok(fn.includes("execPythonSerialized(forecastGenerateDailyCmd(tmpD), { timeout: FORECAST_JOB_TIMEOUT_MS })"));
  assert.ok(fn.indexOf("forecastGenerateCmd(tmp)") < fn.indexOf("forecastGenerateDailyCmd(tmpD)"), "the weekly attempt runs first");
  assert.ok(fn.includes("vix: vixFromCboe(latestCboeVix()?.latest)"), "SPY's implied vol comes from the Cboe archive");
  assert.ok(fn.includes("FORECAST-DAILY-ERROR") && fn.includes("dailyAuditKind(parsed)"));
  assert.ok(fn.includes("if (dailyWantsVixRefresh(parsed)) refreshCboeVix()"));
  assert.ok(!/\/v2\/orders|submit_options_order/.test(fn));
  assert.match(bot, /import \{ latestCboeVix, refreshCboeVix \} from "\.\/cboeVix";/);
});

test("wiring: the forecast subprocesses take the A6 heavy-job queue (never beside the ML retrain); a wait that runs out defers to the next poll", () => {
  const fn = bot.slice(bot.indexOf("async function checkForecasts()"), bot.indexOf("// ── Tier 2: Intelligence"));
  assert.ok(fn.includes('heavyPyJobs.run("forecast-generate", FORECAST_QUEUE_BUDGET_MS, () => generateForecasts(now), { maxWaitMs: FORECAST_QUEUE_MAX_WAIT_MS })'));
  assert.ok(fn.includes('heavyPyJobs.run("forecast-score", FORECAST_QUEUE_BUDGET_MS,'));
  assert.equal((fn.match(/maxWaitMs: FORECAST_QUEUE_MAX_WAIT_MS/g) || []).length, 2, "both runs bound their wait");
  assert.ok(fn.includes("_forecastLastGenAttempt = priorAttempt;"), "a generate that never started is retried, not counted as this hour's attempt");
  assert.ok(fn.indexOf("_forecastLastScoredDay = scoreDay;") > fn.indexOf('heavyPyJobs.run("forecast-score"'), "the score day is recorded only once the job actually runs");
  assert.equal((fn.match(/e instanceof PyJobWaitTimeout/g) || []).length, 2);
  assert.equal((fn.match(/PY-JOB-SKIPPED/g) || []).length, 2, "every deferral is audited");
  assert.ok(FORECAST_QUEUE_MAX_WAIT_MS <= 15 * 60_000, "never longer than the 15-minute poll");
  assert.ok(FORECAST_QUEUE_BUDGET_MS >= 2 * FORECAST_JOB_TIMEOUT_MS, "the budget covers both subprocesses' own kills");
});
