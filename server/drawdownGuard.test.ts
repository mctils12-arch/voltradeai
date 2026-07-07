// Regression battery for the 2026-07-07 market-open incident: the max-
// drawdown kill switch fired on a garbage equity read while the account
// sat at its peak. These tests pin BOTH properties: garbage never kills,
// and every credible catastrophic read STILL kills — the mechanism is
// preserved, only its input is validated.
import { test } from "node:test";
import assert from "node:assert/strict";
import { evaluateDrawdown } from "./drawdownGuard";

const PEAK = 109432.59;
const MAX_DD = -10;

test("the incident class: zero/garbage equity reads never kill and never move the peak", () => {
  for (const bad of ["0", 0, "", null, undefined, "NaN", NaN, -5, "-1", "garbage"]) {
    const r = evaluateDrawdown(bad, PEAK, MAX_DD);
    assert.equal(r.kill, false, `must not kill on ${JSON.stringify(bad)}`);
    assert.equal(r.valid, false);
    assert.equal(r.newPeak, PEAK, "peak untouched by invalid reads");
    assert.equal(r.drawdownPct, null);
  }
});

test("the mechanism is preserved: credible reads kill exactly at the threshold", () => {
  // -10.0% exactly -> kill (<= semantics unchanged from the old inline
  // code; clean decimal inputs — float noise on PEAK*0.9 sits just above
  // the threshold and is not what this pin is about)
  const atLimit = evaluateDrawdown(90_000, 100_000, MAX_DD);
  assert.equal(atLimit.kill, true);
  assert.equal(atLimit.drawdownPct, -10);
  // -9.99% -> no kill
  const justAbove = evaluateDrawdown(90_010, 100_000, MAX_DD);
  assert.equal(justAbove.kill, false);
  assert.ok(justAbove.drawdownPct! > -10);
  // catastrophic-but-possible low reads STILL kill — never filtered as
  // "bad data" (a real wipeout must halt the loop)
  const crash = evaluateDrawdown(50_000, PEAK, MAX_DD);
  assert.equal(crash.kill, true);
  const tiny = evaluateDrawdown(500, PEAK, MAX_DD);
  assert.equal(tiny.kill, true, "small positive equity is credible enough to kill on");
});

test("peak ratchet: rises on new highs, seeds from first credible read, string equity parses", () => {
  const rise = evaluateDrawdown("110000.50", PEAK, MAX_DD);
  assert.equal(rise.valid, true);
  assert.equal(rise.newPeak, 110000.5);
  assert.equal(rise.kill, false);
  const seed = evaluateDrawdown(100000, 0, MAX_DD);
  assert.equal(seed.newPeak, 100000);
  assert.equal(seed.drawdownPct, 0);
  assert.equal(seed.kill, false, "first read can never self-kill");
});
