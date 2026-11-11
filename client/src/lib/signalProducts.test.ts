// Hermetic tests for the Signals product hub helpers.
import { test } from "node:test";
import assert from "node:assert/strict";
import { relativeAge, licensePosture, curlFor } from "./signalProducts.ts";

const NOW = Date.parse("2026-11-11T00:00:00Z");

test("relativeAge: null/garbage never throw; seconds, minutes, hours, days", () => {
  assert.equal(relativeAge(null, NOW), "no reading yet");
  assert.equal(relativeAge(undefined, NOW), "no reading yet");
  assert.equal(relativeAge("not a date", NOW), "unknown age");
  assert.equal(relativeAge("2026-11-10T23:59:40Z", NOW), "just now");
  assert.equal(relativeAge("2026-11-10T23:48:00Z", NOW), "12m ago");
  assert.equal(relativeAge("2026-11-10T21:30:00Z", NOW), "2.5h ago");
  assert.equal(relativeAge("2026-11-10T04:00:00Z", NOW), "20h ago");
  assert.equal(relativeAge("2026-11-05T00:00:00Z", NOW), "6d ago");
  assert.equal(relativeAge("2026-11-12T00:00:00Z", NOW), "just now", "clock skew clamps to zero, never negative");
});

test("licensePosture maps the three LICENSE_MARKS values and falls back honestly", () => {
  assert.equal(licensePosture("ok").cls, "ok");
  assert.equal(licensePosture("share-alike").cls, "share");
  assert.equal(licensePosture("conditional").cls, "cond");
  assert.equal(licensePosture("unknown").cls, "unknown");
  assert.match(licensePosture("whatever").blurb, /do not redistribute/);
});

test("curlFor builds one pasteable line and tolerates a trailing slash on the base", () => {
  assert.equal(curlFor("https://voltradeai.com/", "/api/v1/data/x"), 'curl -H "x-api-key: $KEY" https://voltradeai.com/api/v1/data/x');
});
