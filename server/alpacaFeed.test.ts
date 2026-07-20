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

// Ratchet: no server code may hardcode feed=sip again — that exact string is
// what silently killed the market-data proxy AND the bot's overnight-gap
// guard (bot.ts one-liner parsed the error body as price 0, so the 5% gap
// check never fired). delayed_sip / the resolveAlpacaFeed default are the
// permitted forms; a genuine SIP plan opts in via ALPACA_DATA_FEED.
test("no hardcoded feed=sip anywhere in server code", async () => {
  const { readdirSync, readFileSync } = await import("node:fs");
  const offenders: string[] = [];
  for (const f of readdirSync("server")) {
    if (!f.endsWith(".ts") || f.endsWith(".test.ts")) continue;
    const src = readFileSync(`server/${f}`, "utf8");
    let idx = -1;
    while ((idx = src.indexOf("feed=sip", idx + 1)) !== -1) offenders.push(`${f}@${idx}`);
  }
  assert.deepEqual(offenders, []);
});
