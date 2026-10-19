// REPAIR 2026-10-16: pins that only a "trained" retrain is audited as
// complete. Every other status ml_model_v2.py / ml_retrain_safe.py can
// return (grep '"status":' in both files) must route to TIER3-ML-ERROR.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { retrainStatusIsSuccess, retrainFailureDetail } from "./mlRetrainStatus";

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
