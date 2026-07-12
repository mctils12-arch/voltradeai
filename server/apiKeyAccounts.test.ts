// Self-serve API key accounts — DB injected as in-memory better-sqlite3
// (never imports auth.ts's live singleton; see apiKeyAccounts.ts docstring
// on why that would hang the test runner).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import Database from "better-sqlite3";
import { createApiKeyStore, SELF_SERVE_MAX_KEYS, SELF_SERVE_TIER, MAX_USAGE_HISTORY_DAYS } from "./apiKeyAccounts";
import { keyId, meterUsage } from "./apiProduct";

function freshStore(baseDir?: string) {
  const db = new Database(":memory:");
  return { db, store: createApiKeyStore(db, baseDir) };
}

test("createApiKey issues a raw key once, previews never expose it, and the raw value authenticates", () => {
  const { store } = freshStore();
  const issued = store.createApiKey(7, "my laptop");
  assert.ok(issued.rawKey.startsWith("vt_live_"));
  assert.equal(issued.label, "my laptop");

  const list = store.listApiKeys(7);
  assert.equal(list.length, 1);
  assert.ok(!list[0].keyPreview.includes(issued.rawKey.slice(8)), "full key must never appear in a listing");
  assert.equal(list[0].revokedAt, null);

  const resolved = store.resolveApiKeyForAuth(issued.rawKey);
  assert.deepEqual(resolved, { userId: 7, tier: SELF_SERVE_TIER, label: "my laptop" });
});

test("unknown or revoked keys never authenticate", () => {
  const { store } = freshStore();
  assert.equal(store.resolveApiKeyForAuth("vt_live_totallymadeup"), null);
  const issued = store.createApiKey(1, "");
  const ok = store.revokeApiKey(1, issued.id);
  assert.equal(ok, true);
  assert.equal(store.resolveApiKeyForAuth(issued.rawKey), null, "revoked key must stop authenticating immediately");
  assert.equal(store.listApiKeys(1)[0].revokedAt !== null, true);
});

test("empty label falls back to 'unlabeled'; labels are truncated to 60 chars", () => {
  const { store } = freshStore();
  const a = store.createApiKey(2, "   ");
  assert.equal(a.label, "unlabeled");
  const longLabel = "x".repeat(200);
  const b = store.createApiKey(2, longLabel);
  assert.equal(b.label.length, 60);
});

test("a user is capped at SELF_SERVE_MAX_KEYS active keys; revoking frees a slot", () => {
  const { store } = freshStore();
  for (let i = 0; i < SELF_SERVE_MAX_KEYS; i++) store.createApiKey(3, `key ${i}`);
  assert.throws(() => store.createApiKey(3, "one too many"), /limited to \d+ active keys/);
  const first = store.listApiKeys(3)[0];
  store.revokeApiKey(3, first.id);
  assert.doesNotThrow(() => store.createApiKey(3, "replacement"));
});

test("revoke only affects the owning user's own key, not another user's key with the same id shape", () => {
  const { store } = freshStore();
  const mine = store.createApiKey(10, "mine");
  const ok = store.revokeApiKey(999, mine.id); // wrong user
  assert.equal(ok, false, "cross-account revoke must be refused");
  assert.equal(store.resolveApiKeyForAuth(mine.rawKey) !== null, true, "key must remain live");
});

test("resolveApiKeyForAuth stamps last_used_at on successful auth", () => {
  const { store } = freshStore();
  const issued = store.createApiKey(4, "watch");
  assert.equal(store.listApiKeys(4)[0].lastUsedAt, null);
  store.resolveApiKeyForAuth(issued.rawKey);
  assert.ok((store.listApiKeys(4)[0].lastUsedAt as number) > 0);
});

test("usageToday reads the same apiusage archive meterUsage() writes, correlated by keyId", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-apikeys-"));
  const { store } = freshStore(base);
  const issued = store.createApiKey(5, "usage test");
  const now = Date.now();
  meterUsage({ key: issued.rawKey, endpoint: "/api/v1/stats/archive", status: 200, tier: SELF_SERVE_TIER }, base, now);
  meterUsage({ key: issued.rawKey, endpoint: "/api/v1/stats/archive", status: 200, tier: SELF_SERVE_TIER }, base, now);
  meterUsage({ key: "some-other-key-entirely", endpoint: "/api/v1/stats/archive", status: 200, tier: SELF_SERVE_TIER }, base, now);
  const summary = store.listApiKeys(5)[0];
  assert.equal(summary.usageToday, 2, "only this key's own requests count");
  assert.equal(summary.keyPreview.includes(keyId(issued.rawKey).slice(0, 8)), true,
    "kid slicing used for correlation must match apiProduct.ts's own keyId()");
});

test("usageToday is 0 when no usage archive exists yet (fresh preview key)", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-apikeys-empty-"));
  const { store } = freshStore(base);
  store.createApiKey(6, "brand new");
  assert.equal(store.listApiKeys(6)[0].usageToday, 0);
});

test("keyUsageHistory returns a day-by-day breakdown oldest-first, correlated the same way usageToday is", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-apikeys-history-"));
  const { store } = freshStore(base);
  const issued = store.createApiKey(8, "history test");
  const now = Date.now();
  const yesterday = now - 86_400_000;
  meterUsage({ key: issued.rawKey, endpoint: "/api/v1/stats/archive", status: 200, tier: SELF_SERVE_TIER }, base, yesterday);
  meterUsage({ key: issued.rawKey, endpoint: "/api/v1/stats/archive", status: 200, tier: SELF_SERVE_TIER }, base, now);
  meterUsage({ key: issued.rawKey, endpoint: "/api/v1/stats/archive", status: 200, tier: SELF_SERVE_TIER }, base, now);
  meterUsage({ key: "some-other-key-entirely", endpoint: "/api/v1/stats/archive", status: 200, tier: SELF_SERVE_TIER }, base, now);

  const usage = store.keyUsageHistory(8, issued.id, 3);
  assert.ok(usage);
  assert.equal(usage!.days, 3);
  assert.equal(usage!.history.length, 3);
  assert.equal(usage!.history[2].count, 2, "today (last entry) has 2 hits from this key");
  assert.equal(usage!.history[1].count, 1, "yesterday has 1 hit from this key");
  assert.equal(usage!.history[0].count, 0, "the day before yesterday has no traffic");
  assert.equal(usage!.total, 3, "total sums the whole window, not just today");
  assert.equal(usage!.label, "history test");
});

test("keyUsageHistory refuses another user's key id and clamps days to MAX_USAGE_HISTORY_DAYS", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-apikeys-history-clamp-"));
  const { store } = freshStore(base);
  const issued = store.createApiKey(9, "mine");
  assert.equal(store.keyUsageHistory(999, issued.id, 14), null, "wrong user must not see this key's usage");
  const usage = store.keyUsageHistory(9, issued.id, 9999);
  assert.equal(usage!.days, MAX_USAGE_HISTORY_DAYS, "days is clamped to MAX_USAGE_HISTORY_DAYS, not passed through raw");
});
