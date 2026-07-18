// Token-gated read-only diagnostics (option (d), human-approved
// 2026-07-04) — the sanitizer pins are the approval's condition:
// responses may never contain key-like strings, emails, or env contents.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { diagEnabled, checkDiagToken, positionsSummary, sanitizeDiag, DIAG_PROBES, orderRow, positionRow } from "./diag";

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

test("ml probe distinguishes seeded vs live feedback and surfaces live win-rate (KNOWN BROKEN #3/#4 verification, 2026-07-06)", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const mlProbeStart = bot.indexOf('case "ml":');
  const mlProbeEnd = bot.indexOf('case "daemon":');
  assert.ok(mlProbeStart > 0 && mlProbeEnd > mlProbeStart, "ml probe block not found");
  const mlProbe = bot.slice(mlProbeStart, mlProbeEnd);
  assert.ok(mlProbe.includes("check_model_health"), "ml probe must reuse diagnostics.check_model_health (not re-derive the seeded/corrupt filter)");
  assert.ok(mlProbe.includes("feedback_seeded_count"), "ml probe must report how many feedback records are backtest-seeded");
  assert.ok(mlProbe.includes("feedback_live_count"), "ml probe must report the live (non-seeded) feedback count");
  assert.ok(mlProbe.includes("live_performance"), "ml probe must surface check_model_health's live win-rate/degradation performance dict");
  assert.ok(mlProbe.includes("live_outcome_breakdown"), "ml probe must break down live (non-seeded) records by outcome (open/win/loss/flat/orphan_exit) to distinguish an entry/exit-matching bug from healthy-but-empty feedback");
});

// ---- 2026-07-07 whitelist widening (human-directed): orders + positions-detail ----

test("orderRow: whitelist shaping — trade fields survive, client_order_id and unknown fields never pass through", () => {
  const raw = {
    id: "61e69015-8549-4bfd-b9aa-4b73e4f4edcd",
    client_order_id: "eyJzb21lIjoib3BhcXVlLWJsb2IifQwhatever123456",
    symbol: "SMH", asset_class: "us_equity", side: "buy", type: "market",
    qty: "120", filled_qty: "120", filled_avg_price: "247.31",
    limit_price: null, notional: null, status: "filled",
    submitted_at: "2026-07-07T13:30:01.2Z", filled_at: "2026-07-07T13:30:01.9Z",
    canceled_at: null,
    legs: [{ secret: "should-never-appear" }],
  };
  const row: any = orderRow(raw);
  assert.equal(row.symbol, "SMH");
  assert.equal(row.side, "buy");
  assert.equal(row.qty, 120);
  assert.equal(row.filled_avg_price, 247.31);
  assert.equal(row.status, "filled");
  assert.equal(row.filled_at, "2026-07-07T13:30:01.9Z");
  assert.equal(row.canceled_at, null);
  const flat = JSON.stringify(row);
  assert.ok(!flat.includes("client_order_id") && !flat.includes("opaque"), "client_order_id must be dropped");
  assert.ok(!flat.includes("legs") && !flat.includes("should-never-appear"), "unknown fields must be dropped");
});

test("positionRow: per-position detail incl. lastday_price vs current_price (incident forensics readout)", () => {
  const row: any = positionRow({
    symbol: "SMH", asset_class: "us_equity", side: "long",
    qty: "50", avg_entry_price: "250.10", current_price: "49.90",
    lastday_price: "249.75", change_today: "-0.8002",
    market_value: "2495.00", cost_basis: "12505.00",
    unrealized_pl: "-10010.00", unrealized_plpc: "-0.8005",
    extra_internal_field: "never",
  });
  assert.equal(row.symbol, "SMH");
  assert.equal(row.current_price, 49.9);
  assert.equal(row.lastday_price, 249.75);
  assert.equal(row.market_value, 2495);
  assert.equal(row.unrealized_pl, -10010);
  assert.ok(!JSON.stringify(row).includes("never"), "unknown fields must be dropped");
});

test("orderRow/positionRow: garbage numerics become null, never NaN (NaN breaks JSON consumers)", () => {
  const o: any = orderRow({ qty: "garbage", filled_avg_price: "", notional: undefined });
  assert.equal(o.qty, null);
  assert.equal(o.filled_avg_price, null);
  assert.equal(o.notional, null);
  const p: any = positionRow({ qty: null, current_price: "NaN", market_value: {} });
  assert.equal(p.qty, null);
  assert.equal(p.current_price, null);
  assert.equal(p.market_value, null);
  assert.ok(!JSON.stringify({ o, p }).includes("NaN"));
});

test("timings probe (2026-07-18, KNOWN BROKEN #18 recurrence investigation): reads voltrade_scan_timings.json read-only, sanitized, both data-dir and /tmp fallback checked", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("timings"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const start = bot.indexOf('case "timings"');
  const end = bot.indexOf('default:', start);
  assert.ok(start > 0 && end > start, "timings probe block not found");
  const block = bot.slice(start, end);
  assert.ok(block.includes("/data/voltrade/voltrade_scan_timings.json"), "must check the Railway volume path first");
  assert.ok(block.includes("/tmp/voltrade_scan_timings.json"), "must fall back to /tmp for local/no-volume runs");
  assert.ok(block.includes("sanitizeDiag"), "timings probe must pass the sanitizer like every other probe");
  assert.ok(block.includes("found: false"), "must report found:false rather than erroring when no scan has run yet");
});

test("2026-07-07 widening is wired: orders + positions-detail probes exist, whitelisted, and the summary probe stays aggregate-only", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("orders"));
  assert.ok((DIAG_PROBES as readonly string[]).includes("positions-detail"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const ordersCase = bot.slice(bot.indexOf('case "orders"'), bot.indexOf('case "orders"') + 900);
  assert.ok(ordersCase.includes("orderRow"), "orders probe must shape through orderRow (whitelist), never return raw Alpaca orders");
  assert.ok(ordersCase.includes("sanitizeDiag"), "orders probe must pass the sanitizer");
  assert.ok(ordersCase.includes("Math.min(") && ordersCase.includes(", 200)"), "orders probe must cap limit at 200 (sanitizer array cap)");
  const pdCase = bot.slice(bot.indexOf('case "positions-detail"'), bot.indexOf('case "positions-detail"') + 700);
  assert.ok(pdCase.includes("positionRow"), "positions-detail probe must shape through positionRow");
  assert.ok(pdCase.includes("sanitizeDiag"), "positions-detail probe must pass the sanitizer");
  // The original aggregate-only summary probe is NOT weakened by the widening:
  const posCase = bot.slice(bot.indexOf('case "positions"'), bot.indexOf('case "positions-detail"'));
  assert.ok(!posCase.includes("positionRow"), "plain positions probe stays aggregate-only");
});
