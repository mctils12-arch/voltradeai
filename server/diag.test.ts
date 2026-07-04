// Token-gated read-only diagnostics (option (d), human-approved
// 2026-07-04) — the sanitizer pins are the approval's condition:
// responses may never contain key-like strings, emails, or env contents.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { diagEnabled, checkDiagToken, positionsSummary, sanitizeDiag, DIAG_PROBES } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
// Dummy token for tests only — NEVER the real DIAG_TOKEN value (that
// lives solely in Railway + the session env, per the approval).
const TOKEN = "test-only-dummy-token-0123456789abcdef";

test("closed by default: no token or short token = route disabled; wrong token = rejected", () => {
  assert.equal(diagEnabled({}), false);
  assert.equal(diagEnabled({ DIAG_TOKEN: "short" }), false, "short tokens refused (min 24 chars)");
  assert.equal(diagEnabled({ DIAG_TOKEN: TOKEN }), true);
  assert.equal(checkDiagToken({ DIAG_TOKEN: TOKEN }, TOKEN), true);
  assert.equal(checkDiagToken({ DIAG_TOKEN: TOKEN }, "wrong"), false);
  assert.equal(checkDiagToken({}, ""), false);
});

test("positions summary carries counts/exposure ONLY — no symbols, no per-position rows", () => {
  const s = positionsSummary([
    { symbol: "QQQ", asset_class: "us_equity", market_value: "10000" },
    { symbol: "SPY260501P00500000", asset_class: "us_option", market_value: "-500" },
  ]);
  assert.deepEqual(s, { count: 2, stocks: 1, options: 1, grossExposure: 10500, netExposure: 9500 });
  assert.ok(!JSON.stringify(s).includes("QQQ"), "symbols must never appear in the summary");
});

test("sanitizer: key-like strings, long hex, base64 blobs, and emails are redacted", () => {
  const dirty = {
    note: `alpaca key PKX9ABCDEFGHIJKLMNOP1234 leaked into an audit line`,
    hex: "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
    b64: "QWxhZGRpbjpvcGVuIHNlc2FtZUFsYWRkaW46b3BlbiBzZXNhbWU=",
    email: "mctils12@gmail.com asked about fills",
    fine: "CSP order filled at 2026-07-04T21:00:00Z qty 3",
  };
  const clean: any = sanitizeDiag(dirty);
  const flat = JSON.stringify(clean);
  assert.ok(!flat.includes("PKX9"), "key-like string must be redacted");
  assert.ok(!flat.includes("deadbeef"), "long hex must be redacted");
  assert.ok(!flat.includes("QWxhZGRpbj"), "long base64 must be redacted");
  assert.ok(!flat.includes("@gmail.com"), "emails must be redacted");
  assert.ok(clean.fine.includes("2026-07-04T21:00:00Z"), "normal timestamps/messages survive");
});

test("wiring pinned: /api/diag route exists in bot.ts, gated + sanitized, whitelist only, auth.ts untouched", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(bot.includes('"/api/diag/:probe"'), "diag route missing");
  assert.ok(bot.includes("diagEnabled"), "route must be closed-by-default on missing token");
  assert.ok(bot.includes("checkDiagToken"), "route must verify the token");
  assert.ok(bot.includes("sanitizeDiag"), "every diag response must pass the sanitizer");
  for (const probe of DIAG_PROBES) assert.ok(bot.includes(`case "${probe}"`), `whitelisted probe '${probe}' missing`);
  const auth = fs.readFileSync(path.join(here, "auth.ts"), "utf8");
  assert.ok(!auth.includes("DIAG_TOKEN"), "auth.ts (frozen) must remain untouched by the diag path");
});
