/**
 * apiKeyAccounts.ts — self-serve preview API keys bound to a logged-in
 * account (PLATFORM INTEGRATION PROGRAM P3, research/platform_program.md).
 *
 * Pre-revenue by design: every self-serve key is issued at the fixed "dev"
 * tier regardless of the account's trading-side billing plan. Real paid
 * tiers, and free-vs-paid issuance rules, wait for the human to approve the
 * MONETIZATION READINESS CHECKLIST (wishlist.md) — nothing here charges,
 * gates payment, or gives an account more than the same public preview
 * limits anyone could get by joining the /developers waitlist. This module
 * only replaces "email us and wait" with "click a button," instantly.
 *
 * Factory pattern (db injected, not imported as a module-level singleton):
 * apiProduct.ts's own docstring warns that importing auth.ts's `db` hangs
 * the test runner (its module load has side effects: bcrypt, table
 * creation, an owner-account seed). Accepting a `Database` instance here
 * lets production wire the real auth.ts `db` while tests pass a throwaway
 * in-memory database — no auth.ts import, no hang, full coverage.
 */
import type { Database as DatabaseType } from "better-sqlite3";
import crypto from "crypto";
import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";

export const SELF_SERVE_TIER = "dev" as const;
export const SELF_SERVE_MAX_KEYS = 3;
/** Usage-history window cap (PLATFORM P4) — bounds how many day-files a
 *  single request reads; 30 covers "this month" without inviting an
 *  unbounded full-archive scan. */
export const MAX_USAGE_HISTORY_DAYS = 30;

export interface IssuedApiKey { id: number; rawKey: string; label: string; createdAt: number }
export interface ApiKeySummary {
  id: number;
  label: string;
  keyPreview: string;
  createdAt: number;
  lastUsedAt: number | null;
  revokedAt: number | null;
  usageToday: number;
}
export interface UsageDay { date: string; count: number }
export interface KeyUsageHistory {
  label: string;
  keyPreview: string;
  days: number;
  history: UsageDay[];
  total: number;
}

function sha256Hex(raw: string): string {
  return crypto.createHash("sha256").update(raw).digest("hex");
}

/** vt_live_ prefix keeps self-serve preview keys visually distinct from
 *  env-seeded operator keys in logs and support conversations. */
function generateRawKey(): string {
  return `vt_live_${crypto.randomBytes(24).toString("hex")}`;
}

/** Counts one UTC day's usage lines for a key by its correlation id. `kid`
 *  is the first 12 hex chars of sha256(rawKey) — by construction identical
 *  to apiProduct.ts's own `keyId()`, so it lines up with what meterUsage()
 *  already wrote for this key without importing that module (which would
 *  pull express-adjacent code into this DB-facing file). */
function countUsageForDay(kid: string, baseDir: string, day: string): number {
  const file = path.join(baseDir, "apiusage", `${day}.jsonl`);
  if (!fs.existsSync(file)) return 0;
  let count = 0;
  for (const line of fs.readFileSync(file, "utf8").split("\n")) {
    if (!line.trim()) continue;
    try { if (JSON.parse(line).k === kid) count++; } catch { /* skip malformed line */ }
  }
  return count;
}

function usageTodayForKid(kid: string, baseDir: string, nowMs = Date.now()): number {
  return countUsageForDay(kid, baseDir, new Date(nowMs).toISOString().slice(0, 10));
}

/** Oldest-first daily breakdown for the trailing `days` UTC days
 *  (inclusive of today) — the PLATFORM P4 usage-history view. Missing
 *  day-files (no traffic that day) count as 0, never as "unknown". */
function usageHistoryForKid(kid: string, baseDir: string, days: number, nowMs = Date.now()): UsageDay[] {
  const clamped = Math.min(Math.max(1, Math.floor(days) || 1), MAX_USAGE_HISTORY_DAYS);
  const out: UsageDay[] = [];
  for (let i = clamped - 1; i >= 0; i--) {
    const date = new Date(nowMs - i * 86_400_000).toISOString().slice(0, 10);
    out.push({ date, count: countUsageForDay(kid, baseDir, date) });
  }
  return out;
}

function ensureSchema(db: DatabaseType): void {
  db.prepare(`
    CREATE TABLE IF NOT EXISTS api_keys (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      key_hash TEXT NOT NULL UNIQUE,
      label TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      last_used_at INTEGER,
      revoked_at INTEGER
    )
  `).run();
}

export function createApiKeyStore(db: DatabaseType, baseDir?: string) {
  ensureSchema(db);

  function createApiKey(userId: number, label: string): IssuedApiKey {
    const activeCount = (db.prepare(
      "SELECT COUNT(*) as n FROM api_keys WHERE user_id = ? AND revoked_at IS NULL"
    ).get(userId) as { n: number }).n;
    if (activeCount >= SELF_SERVE_MAX_KEYS) {
      throw new Error(`preview accounts are limited to ${SELF_SERVE_MAX_KEYS} active keys — revoke one first`);
    }
    const rawKey = generateRawKey();
    const keyHash = sha256Hex(rawKey);
    const createdAt = Date.now();
    const cleanLabel = String(label || "").trim().slice(0, 60) || "unlabeled";
    const info = db.prepare(
      "INSERT INTO api_keys (user_id, key_hash, label, created_at) VALUES (?, ?, ?, ?)"
    ).run(userId, keyHash, cleanLabel, createdAt);
    return { id: Number(info.lastInsertRowid), rawKey, label: cleanLabel, createdAt };
  }

  function listApiKeys(userId: number): ApiKeySummary[] {
    const rows = db.prepare(
      "SELECT id, key_hash, label, created_at, last_used_at, revoked_at FROM api_keys WHERE user_id = ? ORDER BY created_at DESC"
    ).all(userId) as Array<{ id: number; key_hash: string; label: string; created_at: number; last_used_at: number | null; revoked_at: number | null }>;
    const base = baseDir || archiveBaseDir();
    return rows.map((r) => ({
      id: r.id,
      label: r.label,
      keyPreview: `vt_live_${r.key_hash.slice(0, 8)}…`,
      createdAt: r.created_at,
      lastUsedAt: r.last_used_at ?? null,
      revokedAt: r.revoked_at ?? null,
      usageToday: usageTodayForKid(r.key_hash.slice(0, 12), base),
    }));
  }

  /** PLATFORM P4: trailing-days usage breakdown for one key, ownership-
   *  checked the same way revokeApiKey is (id + userId, never id alone —
   *  a bare id is guessable and must not leak another account's usage). */
  function keyUsageHistory(userId: number, id: number, days: number): KeyUsageHistory | null {
    const row = db.prepare(
      "SELECT key_hash, label FROM api_keys WHERE id = ? AND user_id = ?"
    ).get(id, userId) as { key_hash: string; label: string } | undefined;
    if (!row) return null;
    const base = baseDir || archiveBaseDir();
    const history = usageHistoryForKid(row.key_hash.slice(0, 12), base, days);
    return {
      label: row.label,
      keyPreview: `vt_live_${row.key_hash.slice(0, 8)}…`,
      days: history.length,
      history,
      total: history.reduce((sum, d) => sum + d.count, 0),
    };
  }

  function revokeApiKey(userId: number, id: number): boolean {
    const result = db.prepare(
      "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND user_id = ? AND revoked_at IS NULL"
    ).run(Date.now(), id, userId);
    return result.changes > 0;
  }

  /** Called on every /api/v1/* request when the key isn't one of the
   *  env-seeded operator keys — resolves a raw header value to its owning
   *  account and stamps last_used_at. Self-serve keys are always "dev"
   *  tier; no request-time DB write beyond the timestamp touch. */
  function resolveApiKeyForAuth(rawKey: string): { userId: number; tier: typeof SELF_SERVE_TIER; label: string } | null {
    if (!rawKey) return null;
    const keyHash = sha256Hex(rawKey);
    const row = db.prepare(
      "SELECT id, user_id, label FROM api_keys WHERE key_hash = ? AND revoked_at IS NULL"
    ).get(keyHash) as { id: number; user_id: number; label: string } | undefined;
    if (!row) return null;
    db.prepare("UPDATE api_keys SET last_used_at = ? WHERE id = ?").run(Date.now(), row.id);
    return { userId: row.user_id, tier: SELF_SERVE_TIER, label: row.label };
  }

  return { createApiKey, listApiKeys, revokeApiKey, resolveApiKeyForAuth, keyUsageHistory };
}

export type ApiKeyStore = ReturnType<typeof createApiKeyStore>;
