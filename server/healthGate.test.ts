// healthGate.ts — the /api/health HTTP-code contract (KNOWN BROKEN #41).
//
// The 2026-09-10 -> 2026-09-16 outage was not the crash loop: it was every
// fresh container answering Railway's deploy probe with 503 because the
// LIVENESS ALARM (a persisted, correct, loud alarm) flipped `status` to
// "degraded" and the handler mapped that to the HTTP code. These tests pin
// the split: alarms stay in the payload, only server+database gate the
// deploy. The first test is the exact production state that was rejected
// 30 deploys in a row.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { healthHttpCode, servingVerdict, SERVING_CHECKS, type HealthPayload } from "./healthGate";

const HERE = path.dirname(fileURLToPath(import.meta.url));

function payload(overrides: Record<string, { status?: string; [k: string]: unknown }> = {}, status = "degraded"): HealthPayload {
  return {
    status,
    checks: {
      server: { status: "ok", uptime_s: 12 },
      database: { status: "ok" },
      alpaca: { status: "ok" },
      python: { status: "ok" },
      bot: { status: "active", liveness: { dark: false } },
      scanner: { status: "ok", consecutiveFailures: 0 },
      feeds: { status: "ok", gates_top_level_status: false },
      licensing: { status: "ok" },
      ...overrides,
    },
  };
}

test("THE SEPT-10 CASE: kill switch latched + liveness dark must still answer 200", () => {
  // Exactly what a fresh container read on the volume from 2026-09-10T15:30Z
  // onward: killSwitch restored ON, liveness stamp 26 market hours old,
  // `status: "degraded"`. Railway needs 2xx; the loop being halted is not a
  // reason to refuse the deploy that would let a human un-halt it.
  const p = payload({
    bot: {
      status: "killed",
      liveness: { dark: true, marketHours: 26, wallHours: 143.1, detail: "LIVENESS ALARM: trading loop dark for 26.0 market hours" },
    },
  });
  assert.equal(healthHttpCode(p), 200);
  assert.equal(p.status, "degraded", "the alarm itself is untouched — Amendment 1's degraded state stays");
  const v = servingVerdict(p);
  assert.equal(v.ok, true);
  assert.deepEqual(v.failing, []);
});

test("every non-serving degradation reports without gating", () => {
  const cases: Array<[string, Record<string, { status?: string; [k: string]: unknown }>]> = [
    ["alpaca 401", { alpaca: { status: "error", detail: "Alpaca 401" } }],
    ["python probe timeout", { python: { status: "error", detail: "timeout" } }],
    ["scanner back-off", { scanner: { status: "degraded", consecutiveFailures: 12 } }],
    ["feed dead-air", { feeds: { status: "degraded", dead: ["vessels"], gates_top_level_status: false } }],
    ["licensing warning", { licensing: { status: "warning", detail: "non-commercial provider with billing on" } }],
    ["bot stopped", { bot: { status: "stopped", liveness: { dark: true, marketHours: 5, wallHours: 30, detail: "dark" } } }],
  ];
  for (const [label, over] of cases) {
    const p = payload(over);
    assert.equal(healthHttpCode(p), 200, `${label} must not fail the deploy gate`);
    assert.equal(servingVerdict(p).ok, true, label);
  }
  // all of them at once — the worst plausible payload — still serves
  const worst = payload(Object.assign({}, ...cases.map(([, o]) => o)));
  assert.equal(healthHttpCode(worst), 200);
});

test("a database that cannot answer SELECT 1 fails the gate", () => {
  const p = payload({ database: { status: "error", detail: "SQLITE_CANTOPEN" } });
  assert.equal(healthHttpCode(p), 503);
  assert.deepEqual(servingVerdict(p).failing, ["database"]);
});

test("a server block that is not ok fails the gate", () => {
  const p = payload({ server: { status: "error" } });
  assert.equal(healthHttpCode(p), 503);
  assert.deepEqual(servingVerdict(p).failing, ["server"]);
});

test("a missing serving block is a failure, not a pass", () => {
  // A refactor that forgets to populate `database` must not silently turn
  // the gate into a 200-always decoration.
  const p = payload();
  delete p.checks.database;
  assert.equal(healthHttpCode(p), 503);
});

test("the verdict names its gates and the contract, for readers of the payload", () => {
  const v = servingVerdict(payload());
  assert.deepEqual(v.gates, [...SERVING_CHECKS]);
  assert.deepEqual(v.gates, ["server", "database"], "adding a gate is a deploy-gate change — read the module header");
  assert.match(v.note, /HTTP code reflects only/);
});

test("the gate is minimal by construction: no alarm-class check is in it", () => {
  for (const alarm of ["alpaca", "python", "bot", "scanner", "feeds", "licensing"]) {
    assert.ok(!(SERVING_CHECKS as readonly string[]).includes(alarm), `${alarm} must never gate the deploy`);
  }
});

// ── source ratchet on the live handler (same pattern as feedDeadAir.test.ts /
// scannerHealth.test.ts): the handler must derive its HTTP code from this
// module and nowhere else, so the old `status !== "ok" -> 503` mapping can
// never be re-introduced by a well-meaning "consistency" edit.
test("bot.ts's /api/health handler derives its HTTP code from healthGate only", () => {
  const bot = fs.readFileSync(path.join(HERE, "bot.ts"), "utf8");
  const start = bot.indexOf('app.get("/api/health"');
  assert.ok(start > 0, "health handler not found");
  const end = bot.indexOf("app.get(", start + 10);
  const handler = bot.slice(start, end > 0 ? end : undefined);

  assert.match(handler, /healthHttpCode\(checks\)/, "HTTP code must come from healthHttpCode()");
  assert.match(handler, /servingVerdict\(checks\)/, "the payload must carry the serving verdict");
  assert.doesNotMatch(handler, /checks\.status\s*===\s*"ok"\s*\?\s*200\s*:\s*503/,
    "the pre-2026-09-16 mapping of any degraded status to 503 is what took the site down for 5 days");
  assert.doesNotMatch(handler, /res\.status\(503\)/, "no hand-rolled 503 inside the handler");
  // the alarms must still be raised in the payload — this change removes
  // their power over the deploy, not their voice
  assert.match(handler, /if \(lv\.dark\) checks\.status = "degraded"/, "liveness alarm still degrades `status`");
  assert.match(handler, /scannerDegraded\(tier2ConsecutiveFailures\)\) checks\.status = "degraded"/);
});
