// Regression test for the 2026-07-20 dead-feed repair: feed=sip rejected by
// the subscription returned {message} bodies that parsed as empty snapshot
// maps, so scanner/sectors/watchlist served empty 200s forever. The proxy
// must (a) default to a feed the subscription permits and (b) recognize
// Alpaca error bodies instead of treating them as data.
import test from "node:test";
import assert from "node:assert/strict";
import { resolveAlpacaFeed, resolveHistoricalBarsFeed, BARS_FALLBACK_FEED, alpacaErrorBody } from "./alpacaFeed";

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
// (2026-11-16: that holds for snapshots/quotes/trades ONLY. Historical bars
// reject delayed_sip — use resolveHistoricalBarsFeed; ratchet at the end.)
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

// ── 2026-11-16: the bars endpoint is not the snapshot endpoint ───────────────
// delayed_sip is right for snapshots/quotes/trades and a HTTP 400 on bars.
// The fact was learned live on 2026-07-18 (Python), 2026-07-20/27 (bot.ts) and
// again on 2026-11-16 (github_gate2_weekly's first registered run), because it
// lived in comments. It now lives in one function and this ratchet.
test("resolveHistoricalBarsFeed never returns delayed_sip; sip by default; an explicit other feed wins", () => {
  for (const v of [undefined, "", "  ", "delayed_sip", " delayed_sip "]) assert.equal(resolveHistoricalBarsFeed(v), "sip", JSON.stringify(v));
  assert.equal(resolveHistoricalBarsFeed("iex"), "iex");
  assert.equal(resolveHistoricalBarsFeed(" sip "), "sip");
  assert.equal(BARS_FALLBACK_FEED, "iex");
});

test("RATCHET: no server file that requests stock bars resolves its feed through resolveAlpacaFeed or writes delayed_sip beside a bars URL", async () => {
  const { readdirSync, readFileSync } = await import("node:fs");
  const barsUrl = /\/v2\/stocks\/(?:[^\s"'`?]*\/)?bars\b/;
  const offenders: string[] = [];
  let filesWithBars = 0;
  for (const f of readdirSync("server")) {
    if (!f.endsWith(".ts") || f.endsWith(".test.ts")) continue;
    const code = readFileSync(`server/${f}`, "utf8").split("\n").map((l) => (/^\s*(\/\/|\*|\/\*)/.test(l) ? "" : l));
    if (!code.some((l) => barsUrl.test(l))) continue;
    filesWithBars++;
    code.forEach((l, i) => {
      if (/\bresolveAlpacaFeed\s*\(/.test(l)) offenders.push(`${f}:${i + 1} resolveAlpacaFeed() in a file that requests stock bars — use resolveHistoricalBarsFeed`);
      if (barsUrl.test(l) && l.includes("delayed_sip")) offenders.push(`${f}:${i + 1} delayed_sip on a bars URL`);
    });
  }
  assert.ok(filesWithBars >= 4, `the scan must see the bars callers (githubGate2, pricePanel, barsCoverage, bot): saw ${filesWithBars}`);
  assert.deepEqual(offenders, []);
  // and the gate job really takes its feed from the bars resolver
  const g = readFileSync("server/githubGate2.ts", "utf8");
  assert.ok(g.includes("resolveHistoricalBarsFeed(deps.env.ALPACA_DATA_FEED)"));
});
