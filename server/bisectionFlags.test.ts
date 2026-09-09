import { test } from "node:test";
import assert from "node:assert/strict";
import { tier2Disabled, tier3Disabled } from "./bisectionFlags";

test("tier2Disabled: false when VOLTRADE_DISABLE_TIER2 is unset (default — zero behavior change)", () => {
  assert.equal(tier2Disabled({}), false);
});

test("tier2Disabled: false for any value other than the exact string \"1\"", () => {
  assert.equal(tier2Disabled({ VOLTRADE_DISABLE_TIER2: "true" }), false);
  assert.equal(tier2Disabled({ VOLTRADE_DISABLE_TIER2: "0" }), false);
  assert.equal(tier2Disabled({ VOLTRADE_DISABLE_TIER2: "" }), false);
});

test("tier2Disabled: true only when VOLTRADE_DISABLE_TIER2 is exactly \"1\"", () => {
  assert.equal(tier2Disabled({ VOLTRADE_DISABLE_TIER2: "1" }), true);
});

test("tier3Disabled: false when VOLTRADE_DISABLE_TIER3 is unset (default — zero behavior change)", () => {
  assert.equal(tier3Disabled({}), false);
});

test("tier3Disabled: true only when VOLTRADE_DISABLE_TIER3 is exactly \"1\"", () => {
  assert.equal(tier3Disabled({ VOLTRADE_DISABLE_TIER3: "1" }), true);
});

test("the two flags are independent — setting one does not disable the other", () => {
  assert.equal(tier2Disabled({ VOLTRADE_DISABLE_TIER3: "1" }), false);
  assert.equal(tier3Disabled({ VOLTRADE_DISABLE_TIER2: "1" }), false);
});
