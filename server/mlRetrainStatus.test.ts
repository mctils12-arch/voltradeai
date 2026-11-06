// REPAIR 2026-10-16: pins that only a "trained" retrain is audited as
// complete. Every other status ml_model_v2.py / ml_retrain_safe.py can
// return (grep '"status":' in both files) must route to TIER3-ML-ERROR.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { retrainStatusIsSuccess, retrainFailureDetail, retrainHonestDetail } from "./mlRetrainStatus";

test("only 'trained' is a success", () => {
  assert.equal(retrainStatusIsSuccess("trained"), true);
  for (const s of ["skipped", "failed", "insufficient_data", "error", "fatal", "error: boom", "", undefined, null, "TRAINED "]) {
    assert.equal(retrainStatusIsSuccess(s), false, `status ${JSON.stringify(s)} must not audit as complete`);
  }
});

test("failure detail surfaces samples for insufficient_data and the wrapper's error/steps", () => {
  assert.equal(retrainFailureDetail({ status: "insufficient_data", samples: 412 }), "samples 412");
  assert.equal(retrainFailureDetail({ status: "error", error: "MemoryError", steps_completed: ["numpy_ok", "lightgbm_ok"] }), "MemoryError · steps numpy_ok>lightgbm_ok");
  assert.equal(retrainFailureDetail({ status: "failed" }), "");
});

test("wiring pinned: tier3Strategic routes every non-success status through retrainStatusIsSuccess", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(bot.includes('from "./mlRetrainStatus"'), "bot.ts must import the classifier");
  assert.ok(bot.includes('|| !retrainStatusIsSuccess(_statusStr)'), "the TIER3-ML-ERROR branch must fire for any non-success status");
  assert.ok(bot.includes("retrainFailureDetail(trainResult)"), "the audited message must carry the failure detail (samples / error / steps)");
});

// [RULE-REVIEW] 2026-11-06 (v2 training honesty): a trained result's honest
// out-of-sample record is audited beside the accuracy, each number with its bar.
const HONEST = {
  status: "trained", accuracy: 49.5,
  honest_cv: {
    pooled: { status: "OK", auc: 0.4963, accuracy: 0.494, majority_accuracy: 0.5802, log_loss: 0.7843, base_rate_log_loss: 0.6802 },
    auc_ci95_date_block: [0.4705, 0.5227], skill: false,
  },
  training_set: { shadow: { rows_labeled: 10642, rows_used: 210 } },
};

test("honest detail: AUC vs 0.5 with its CI, accuracy vs always-majority, log-loss vs the base rate, the shadow collapse", () => {
  assert.equal(retrainHonestDetail(HONEST),
    "out-of-sample AUC 0.496 (95% CI 0.470–0.523; 0.5 = no skill), pooled accuracy 49.4% vs 58.0% always-majority, " +
    "log-loss 0.784 vs 0.680 always-base-rate — no skill; shadow 10642 scan rows → 210 candidate-days");
  const skilled = { ...HONEST, honest_cv: { ...HONEST.honest_cv, skill: true } };
  assert.ok(retrainHonestDetail(skilled).includes("always-base-rate — skill;"));
});

test("honest detail is empty for a result without the record (switch off, older trainer) and never prints undefined/NaN", () => {
  for (const r of [{ status: "trained", accuracy: 62.2 }, null, undefined, "x", { honest_cv: { pooled: { status: "INSUFFICIENT", n: 3 } } }]) {
    assert.equal(retrainHonestDetail(r), "");
  }
  const partial = retrainHonestDetail({ honest_cv: { pooled: { status: "OK", auc: Number.NaN, accuracy: 0.5 } } });
  assert.ok(!/undefined|NaN|null/.test(partial), partial);
  const noCi = retrainHonestDetail({ honest_cv: { pooled: { status: "OK", auc: 0.51 }, auc_ci95_date_block: null } });
  assert.equal(noCi, "out-of-sample AUC 0.510 (0.5 = no skill) — no skill");
});

test("wiring pinned: the TIER3 success line appends the honest detail", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const i = bot.indexOf('audit("TIER3", `ML retrain complete');
  assert.ok(i > 0, "success audit line not found");
  const line = bot.slice(bot.lastIndexOf("\n", i), bot.indexOf("\n", i));
  assert.ok(line.includes("${_honest ? ` — ${_honest}` : \"\"}"), "the honest detail rides the success line");
  assert.ok(bot.slice(i - 400, i).includes("const _honest = retrainHonestDetail(trainResult);"));
});
