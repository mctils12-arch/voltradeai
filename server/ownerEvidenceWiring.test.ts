// ownerEvidenceWiring.test.ts — how server/bot.ts serves the owner's page the
// six evidence reads ([PRODUCT] 2026-11-20, UI coverage batch A): owner-gated,
// read-only, and the SAME content as the diag probe each one mirrors. Source
// pins, like diag.test.ts and floorTrackingWiring.test.ts: three routes call
// the probe's own body function; the other three repeat the probe's few lines
// (those probe cases are pinned as written by older tests), so this file pins
// both copies to the same expressions — a change to one without the other fails.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function slice(start: string, end: string): string {
  const a = bot.indexOf(start);
  assert.ok(a >= 0, `anchor not found: ${start}`);
  const b = bot.indexOf(end, a + start.length);
  assert.ok(b > a, `end anchor not found after ${start}: ${end}`);
  return bot.slice(a, b);
}
const BLOCK = slice("// ── OWNER VIEWS OF THE EVIDENCE PROBES", "// ── Data Backup Endpoint");
const ROUTES = ["/api/bot/lots", "/api/bot/order-history", "/api/bot/floor-tracking", "/api/bot/rule-counterfactuals", "/api/bot/regime-counterfactual", "/api/bot/options-feed"] as const;
const route = (p: string) => { const a = BLOCK.indexOf(`app.get("${p}"`); assert.ok(a >= 0, `${p} missing`); const b = BLOCK.indexOf("\n  });", a); assert.ok(b > a); return BLOCK.slice(a, b); };
const probe = (name: string) => { const a = bot.indexOf(`case "${name}": {`); assert.ok(a > 0, `${name} probe missing`); return bot.slice(a, bot.indexOf("\n        case \"", a + 10)); };

test("six owner routes, each a GET behind requireOwner, and nothing in the block writes or sends an order", () => {
  for (const p of ROUTES) assert.ok(BLOCK.includes(`app.get("${p}", requireOwner,`), `${p} must be an owner-gated GET`);
  assert.equal((BLOCK.match(/app\.(get|post|put|patch|delete)\(/g) ?? []).length, ROUTES.length, "the block registers exactly these routes");
  assert.ok(!/app\.(post|put|patch|delete)\(/.test(BLOCK), "read-only: no mutating route");
  assert.ok(!/method:|writeFileSync|appendFileSync|audit\(|notify\(|submitOrder|\/v2\/orders["`]\s*,\s*\{/.test(BLOCK), "no order, file write, audit row or notification from an owner view");
  // the only broker call is the order list, a GET with a query string
  assert.deepEqual(BLOCK.match(/alpaca\([^)]*\)/g), ["alpaca(`/v2/orders?status=all&limit=${limit}&direction=desc`)"]);
});

test("floor tracking, the regime counterfactual and the feed serve the probe's own body function", () => {
  assert.ok(route("/api/bot/floor-tracking").includes("sanitizeDiag({ ...(await floorTrackingBody()) })"));
  assert.ok(probe("floor_tracking").includes("...(await floorTrackingBody())"));
  assert.ok(route("/api/bot/regime-counterfactual").includes("regimeCounterfactualBody(regimeDataDir(process.env), new Date(), req.query.days)"));
  assert.ok(probe("regime_counterfactual").includes("regimeCounterfactualBody(regimeDataDir(process.env), new Date(), req.query.days)"));
  const feed = route("/api/bot/options-feed");
  assert.ok(feed.includes("optionsFeedView(await cachedOptionsEntitlements({") && feed.includes('underlying: "SPY"') && feed.includes("{ fresh: false }"), "the feed line is derived from the entitlement report, never fetched fresh by a page");
  assert.ok(probe("options_entitlements").includes("await cachedOptionsEntitlements({"));
});

test("closed lots: the same expressions as the lots probe (sleeve cache, lotRow, explicit truncation), newest 200 by default", () => {
  const r = route("/api/bot/lots"), p = probe("lots");
  for (const expr of [
    "await sleevesBody();", "const events = sleeveCache?.taxInputs.events ?? [];", "const rows = events.slice(-limit).map((e) => sanitizeDiag(lotRow(e)));",
    "count: events.length, truncated: events.length > limit,", "since: sleeveCache ? new Date(sleeveCache.taxInputs.windowStartMs).toISOString() : null,",
    "fetchedAt: sleeveCache ? new Date(sleeveCache.at).toISOString() : null,", "pending: !sleeveCache, lastError: sleeveError, rows,",
  ]) { assert.ok(r.includes(expr), `route lacks: ${expr}`); assert.ok(p.includes(expr), `probe lacks: ${expr}`); }
  assert.ok(r.includes('parseInt(String(req.query.limit || "200"), 10) || 200, 1), 5000)'), "the page's default is the newest 200, capped at the probe's 5000");
  assert.ok(!/alpaca\(|fetch\(/.test(r), "the ledger is the sleeve cache: no second fetch");
});

test("order history and the rule report: the probe's shaping and the probe's command, each behind the short-lived memo", () => {
  const r = route("/api/bot/order-history"), p = probe("orders");
  assert.ok(r.includes("arr.map(orderRow)") && p.includes("arr.map(orderRow)"), "orders leave only through orderRow's whitelist, on both");
  assert.ok(r.includes("const arr = Array.isArray(orders) ? orders : [];") && p.includes("const arr = Array.isArray(orders) ? orders : [];"));
  assert.ok(r.includes("ownerView(`orders:${limit}`, 30_000,"), "one broker read per 30 s however many windows poll");
  assert.ok(r.includes('|| 60, 1), 200)'), "at most 200 orders, like the probe");
  const g = route("/api/bot/rule-counterfactuals"), gp = probe("gate_counterfactuals");
  const cmd = '`python3 -c "import json; from gate_counterfactuals_probe import gate_counterfactuals_report; print(json.dumps(gate_counterfactuals_report(), default=str))"`';
  for (const expr of ['"gate_counterfactuals",', cmd, "{ timeout: 60000 },", "45000,", '...(r.success ? r.result : { error: String(r.result?.error || "gate_counterfactuals_probe failed").slice(0, 300) }),']) {
    assert.ok(g.includes(expr), `route lacks: ${expr}`); assert.ok(gp.includes(expr), `probe lacks: ${expr}`);
  }
  assert.ok(g.includes('ownerView("gates", 10 * 60_000,'));
});

test("the memo: one computation per key per window and one in flight; a failure is not kept", async () => {
  // the helper as written, run on its own (no route, no broker)
  const src = slice("const ownerViewMemo = new Map", "const ownerViewError =");
  const js = src.replace(/<string, \{ at: number; body: unknown \}>|<string, Promise<unknown>>|<T>|: string|: number|: \(\) => Promise<T>|: Promise<T>| as T| as Promise<T>/g, "").replace(/\/\*\*[\s\S]*?\*\//g, "");
  // eslint-disable-next-line @typescript-eslint/no-implied-eval
  const ownerView = new Function(`${js}; return ownerView;`)() as <T>(k: string, ttl: number, fn: () => Promise<T>) => Promise<T>;
  let calls = 0;
  const slow = () => new Promise<number>((res) => setTimeout(() => res(++calls), 20));
  const [a, b] = await Promise.all([ownerView("k", 1000, slow), ownerView("k", 1000, slow)]);
  assert.deepEqual([a, b, calls], [1, 1, 1], "two callers share the one in flight");
  assert.equal(await ownerView("k", 1000, slow), 1, "inside the window: the kept body");
  assert.equal(await ownerView("k", 0, slow), 2, "past the window: computed again");
  assert.equal(await ownerView("other", 1000, slow), 3, "another key is its own computation");
  await assert.rejects(ownerView("bad", 1000, async () => { throw new Error("broker down"); }), /broker down/);
  assert.equal(await ownerView("bad", 1000, async () => 7), 7, "a failure is not kept: the next call computes");
});
