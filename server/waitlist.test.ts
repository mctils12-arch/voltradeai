// Waitlist capture — pure-module tests (no express, no network).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { validEmail, addToWaitlist, _resetWaitlistSeen } from "./waitlist";

const here = path.dirname(fileURLToPath(import.meta.url));

test("email validation: format-gated, length-bounded", () => {
  assert.ok(validEmail("a@b.co"));
  assert.ok(!validEmail("not-an-email"));
  assert.ok(!validEmail("a@b"));
  assert.ok(!validEmail("a b@c.com"));
  assert.ok(!validEmail("x".repeat(250) + "@a.com"));
});

test("capture: adds once, dedups (case-insensitive), persists day-JSONL", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-wl-"));
  _resetWaitlistSeen();
  const NOW = Date.UTC(2026, 6, 4, 12, 0, 0);
  assert.equal(addToWaitlist("Dev@Example.com", "developers", base, NOW), "added");
  assert.equal(addToWaitlist("dev@example.com", "developers", base, NOW), "duplicate");
  assert.equal(addToWaitlist("nonsense", "developers", base, NOW), "invalid");
  const day = fs.readFileSync(path.join(base, "waitlist", "2026-07-04.jsonl"), "utf8").trim().split("\n");
  assert.equal(day.length, 1);
  const rec = JSON.parse(day[0]);
  assert.equal(rec.e, "dev@example.com");
  assert.equal(rec.s, "developers");
});

test("dedup survives restart: seeded from existing day files", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-wl2-"));
  _resetWaitlistSeen();
  const NOW = Date.UTC(2026, 6, 4, 12, 0, 0);
  assert.equal(addToWaitlist("x@y.co", "developers", base, NOW), "added");
  _resetWaitlistSeen(); // simulate process restart
  assert.equal(addToWaitlist("x@y.co", "developers", base, NOW), "duplicate");
});

test("wiring pinned: waitlist route + manifest exist; NO billing anywhere in the page or module", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes('"/api/waitlist"'), "waitlist route missing");
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "waitlist.json"), "utf8"));
  assert.ok(manifest.license.includes("NEVER exposed"), "PII handling must be manifested");
  const page = fs.readFileSync(path.join(here, "..", "client", "src", "pages", "developers.tsx"), "utf8");
  for (const banned of ["stripe", "Stripe", "checkout", "billing_enabled", "BILLING_ENABLED"]) {
    assert.ok(!page.includes(banned), `monetization tripwire: '${banned}' may not appear on /developers pre-approval`);
  }
  assert.ok(page.includes("not for sale yet"), "pricing must be explicitly marked not-enabled");
});

test("positioning copy pinned: honest not-a-basemap framing on /developers (atlas-parity Part 4)", () => {
  const page = fs.readFileSync(path.join(here, "..", "client", "src", "pages", "developers.tsx"), "utf8");
  assert.ok(page.includes("not a basemap competitor"), "positioning line missing");
  assert.ok(page.includes("no claim is made to proprietary imagery"), "the honest no-proprietary-imagery disclaimer must stay");
});
