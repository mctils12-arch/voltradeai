// swimFeeds.test — the Streams inventory's FAA feed status, on the shape the
// server answers with (trimmed from production, 2026-11-20) and on the shapes
// the page must survive.
import test from "node:test";
import assert from "node:assert/strict";
import { parseSwimFeeds, feedCounts, feedAge, feedsSentence, FEED_QUIET_MS } from "./swimFeeds.ts";

const NOW = 1_795_210_306_219;
const feed = (over: Record<string, unknown>) => ({ env: { configured: true, missing: [] }, consumer: "x", connected: true, messagesReceived: 0, messagesProcessed: 0, processErrors: 0, droppedOverflow: 0, reconnects: 0, stallRestarts: 0, lastMessageAt: null, lastError: null, ...over });
const LIVE = { at: NOW, note: "Env readiness is reported by variable NAME only.", products: {
  SFDPS: feed({ consumer: "flight plans (swimSfdps.ts)", messagesReceived: 18113, messagesProcessed: 17506, droppedOverflow: 607, lastMessageAt: NOW - 53 }),
  TFMS: feed({ consumer: "capture (shape profile + samples)", messagesReceived: 21624, messagesProcessed: 21624, lastMessageAt: NOW - 66 }),
  STDDS: feed({ consumer: "configured-only", connected: false }),
  NOTAM: feed({ consumer: "capture (shape profile + samples)", messagesReceived: 32, messagesProcessed: 32, lastMessageAt: NOW - 4668 }),
  ITWS: feed({ env: { configured: false, missing: ["A", "B", "C"] }, consumer: "configured-only", connected: false }),
} };

test("the five feeds as production reports them: three receiving, one deliberately not read, one not set up", () => {
  const v = parseSwimFeeds(LIVE, NOW)!;
  assert.deepEqual(v.feeds.map((f) => [f.id, f.name, f.state]), [
    ["SFDPS", "Flight plans", "receiving"], ["TFMS", "Traffic management", "receiving"], ["STDDS", "Terminal data", "not_consumed"],
    ["NOTAM", "Notices to airmen", "receiving"], ["ITWS", "Terminal weather", "not_set_up"],
  ]);
  assert.equal(feedsSentence(v), "3 of 3 FAA feeds the server reads are receiving · 2 more not read");
  // a dropped message is said, with the number
  assert.equal(feedCounts(v.feeds[0]), "18,113 messages since the server started · 17,506 handled · 607 dropped under load");
  assert.equal(feedCounts(v.feeds[1]), "21,624 messages since the server started");
  assert.equal(feedCounts(v.feeds[2]), "set up on the server, deliberately not read yet");
  // the server names the missing settings; the page prints only how many
  assert.equal(feedCounts(v.feeds[4]), "3 settings missing on the server");
  assert.ok(!JSON.stringify(v).includes('"A"'));
  assert.equal(feedAge(v.feeds[0].lastMessageAt, NOW), "0 s ago");
  assert.equal(feedAge(v.feeds[3].lastMessageAt, NOW), "5 s ago");
  assert.equal(feedAge(null, NOW), "never");
  assert.equal(feedAge(NOW - 10 * 60_000, NOW), "10 min ago");
  assert.equal(feedAge(NOW - 3 * 3600_000, NOW), "3.0 h ago");
});

test("connected with no recent message is quiet, not receiving; a dropped connection is said with its error count", () => {
  const v = parseSwimFeeds({ at: NOW, products: {
    SFDPS: feed({ messagesReceived: 900, messagesProcessed: 900, lastMessageAt: NOW - FEED_QUIET_MS - 1000 }),
    TFMS: feed({ connected: false, messagesReceived: 40, messagesProcessed: 38, processErrors: 2, reconnects: 3, lastMessageAt: NOW - 3600_000, lastError: "socket closed" }),
    NOTAM: feed({ messagesReceived: 0, lastMessageAt: null }),
  } }, NOW)!;
  assert.deepEqual(v.feeds.map((f) => f.state), ["quiet", "disconnected", "quiet"]);
  assert.equal(feedsSentence(v), "0 of 3 FAA feeds the server reads are receiving · flight plans quiet, traffic management disconnected, notices to airmen quiet · 0 more not read");
  assert.equal(feedCounts(v.feeds[1]), "40 messages since the server started · 38 handled · 2 failed to parse · 3 reconnects");
  assert.equal(v.feeds[1].lastError, "socket closed");
});

test("another shape is null; an unknown feed keeps its own id; a malformed product is skipped", () => {
  for (const bad of [null, undefined, [], {}, { products: [] }, "<!doctype html>", { error: "x" }]) assert.equal(parseSwimFeeds(bad, NOW), null);
  const v = parseSwimFeeds({ products: { NEWFEED: feed({ lastMessageAt: NOW }), BROKEN: null } }, NOW)!;
  assert.deepEqual(v.feeds.map((f) => [f.id, f.name, f.use]), [["NEWFEED", "NEWFEED", null]]);
  assert.equal(v.at, null);
});
