// Regression battery for the 2026-07-07 market-open incident: the max-
// drawdown kill switch fired on a garbage equity read while the account
// sat at its peak. These tests pin BOTH properties: garbage never kills,
// and every credible catastrophic read STILL kills — the mechanism is
// preserved, only its input is validated.
import { test } from "node:test";
import assert from "node:assert/strict";
import { evaluateDrawdown, drawdownStatus } from "./drawdownGuard";

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

// Regression battery for the 2026-08-09 /api/monitoring/overview finding:
// the old inline code read a `current_dd_pct` field get_kill_switch_status()
// never returns, so `dd` was always the `|| 0` fallback and this section
// permanently reported "OK" at 100% proximity no matter the real drawdown.
// The proximity formula was separately inverted (0% drawdown -> 100%
// proximity). These pin the fixed, correctly-oriented replacement.
test("drawdownStatus: no drawdown reads OK at 0% proximity, not 100%", () => {
  const s = drawdownStatus(0, MAX_DD);
  assert.equal(s.status, "OK");
  assert.equal(s.proximity_pct, 0);
  assert.equal(s.current_pct, 0);
  assert.equal(s.kill_threshold_pct, MAX_DD);
});

test("drawdownStatus: at the kill threshold reads 100% proximity and CRITICAL", () => {
  const s = drawdownStatus(MAX_DD, MAX_DD); // -10% drawdown, -10% threshold
  assert.equal(s.proximity_pct, 100);
  assert.equal(s.status, "CRITICAL");
});

test("drawdownStatus: WARNING/CRITICAL bands scale off the real threshold, not a hardcoded -20/-15/-10", () => {
  // MAX_DD = -10: WARNING band starts at 50% of threshold (-5%), CRITICAL at 75% (-7.5%)
  assert.equal(drawdownStatus(-4, MAX_DD).status, "OK");
  assert.equal(drawdownStatus(-5, MAX_DD).status, "WARNING");
  assert.equal(drawdownStatus(-7.5, MAX_DD).status, "CRITICAL");
  // a smaller/looser configured threshold (-20) must not fire CRITICAL at -7.5
  assert.equal(drawdownStatus(-7.5, -20).status, "OK");
});

test("drawdownStatus: proximity is clamped to [0, 100] beyond the threshold", () => {
  const s = drawdownStatus(-30, MAX_DD); // far past a -10% threshold
  assert.equal(s.proximity_pct, 100);
  assert.equal(s.status, "CRITICAL");
});
