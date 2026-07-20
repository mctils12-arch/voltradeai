// Regression test for the 2026-07-20 dead-feed repair: feed=sip rejected by
// the subscription returned {message} bodies that parsed as empty snapshot
// maps, so scanner/sectors/watchlist served empty 200s forever. The proxy
// must (a) default to a feed the subscription permits and (b) recognize
// Alpaca error bodies instead of treating them as data.
import test from "node:test";
import assert from "node:assert/strict";
import { resolveAlpacaFeed, alpacaErrorBody } from "./alpacaFeed";

test("defaults to delayed_sip (subscription-permitted, full-tape volumes)", () => {
  assert.equal(resolveAlpacaFeed(undefined), "delayed_sip");
  assert.equal(resolveAlpacaFeed(""), "delayed_sip");
});

test("honors ALPACA_DATA_FEED override", () => {
  assert.equal(resolveAlpacaFeed("sip"), "sip");
  assert.equal(resolveAlpacaFeed("iex"), "iex");
});

test("recognizes the exact production failure body as an error", () => {
  // What data.alpaca.markets returned for feed=sip on this account —
  // previously iterated as if it were {TICKER: snapshot} and yielded zero rows.
  const msg = alpacaErrorBody(403, { message: "subscription does not permit querying recent SIP data" });
  assert.match(String(msg), /subscription does not permit/);
  // Same body with a 200 status must still be treated as an error.
  assert.ok(alpacaErrorBody(200, { message: "too many requests." }));
});

test("passes real snapshot data through", () => {
  const snap = { AAPL: { dailyBar: { c: 211.5, v: 1000000 } }, MSFT: { dailyBar: { c: 500.1, v: 900000 } } };
  assert.equal(alpacaErrorBody(200, snap), null);
});

test("non-2xx without message body is still an error", () => {
  assert.match(String(alpacaErrorBody(429, {})), /status 429/);
});
