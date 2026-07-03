// Monetization-tripwire runtime guard (CLAUDE.md KNOWN STATE 2026-07-03).
// Pure-function tests with injected env; plus source-level pins that keep
// the guard wired into /api/health and the aircraft path, and keep the
// non-commercial provider list in sync with the actual chain.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  billingActive, aircraftProviderCompliance, complianceAuditTick,
  setComplianceAuditWriter, NON_COMMERCIAL_AIRCRAFT_PROVIDERS,
} from "./providerCompliance";

const here = path.dirname(fileURLToPath(import.meta.url));

test("billing inactive by default -> compliance ok", () => {
  const env = {} as NodeJS.ProcessEnv;
  assert.equal(billingActive(env), false);
  assert.equal(aircraftProviderCompliance(env).status, "ok");
});

test("BILLING_ENABLED=true trips the violation while airplanes.live is in the chain", () => {
  const env = { BILLING_ENABLED: "true" } as NodeJS.ProcessEnv;
  const c = aircraftProviderCompliance(env);
  assert.equal(c.status, "violation");
  assert.ok(c.detail!.includes("airplaneslive"));
  assert.ok(c.detail!.includes("MONETIZATION TRIPWIRE"));
});

test("STRIPE_SECRET_KEY presence alone counts as billing active", () => {
  const env = { STRIPE_SECRET_KEY: "sk_test_x" } as NodeJS.ProcessEnv;
  assert.equal(billingActive(env), true);
  assert.equal(aircraftProviderCompliance(env).status, "violation");
});

test("complianceAuditTick throttles and writes through the injected audit writer", () => {
  const written: Array<{ type: string; message: string }> = [];
  setComplianceAuditWriter((type, message) => written.push({ type, message }));
  const bad = { BILLING_ENABLED: "true" } as NodeJS.ProcessEnv;
  const t0 = 1_000_000_000_000;
  assert.equal(complianceAuditTick(bad, 60_000, t0), true, "first tick fires");
  assert.equal(complianceAuditTick(bad, 60_000, t0 + 1_000), false, "inside window suppressed");
  assert.equal(complianceAuditTick(bad, 60_000, t0 + 61_000), true, "next window fires");
  assert.equal(complianceAuditTick({} as NodeJS.ProcessEnv, 60_000, t0 + 200_000), false, "ok env never fires");
  assert.equal(written.length, 2, "one audit row per fired window");
  assert.equal(written[0].type, "COMPLIANCE-WARNING");
  assert.ok(written[0].message.includes("airplaneslive"));
});

test("non-commercial provider list matches the live chain in routes.ts", () => {
  const src = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  for (const key of NON_COMMERCIAL_AIRCRAFT_PROVIDERS) {
    assert.ok(src.includes(`"${key}"`), `${key} listed as non-commercial but absent from the chain — update NON_COMMERCIAL_AIRCRAFT_PROVIDERS`);
  }
});

test("guard is wired: /api/health surfaces licensing, aircraft path ticks", () => {
  const botSrc = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(botSrc.includes("aircraftProviderCompliance"), "/api/health licensing check missing");
  assert.ok(botSrc.includes("checks.checks.licensing"), "licensing key missing from health payload");
  const routesSrc = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routesSrc.includes("complianceAuditTick"), "aircraft path does not tick the compliance guard");
  assert.ok(routesSrc.includes("setComplianceAuditWriter"), "audit writer not injected — violations would never reach the audit log");
});
