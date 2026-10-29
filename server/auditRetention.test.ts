// [RULE-REVIEW] 2026-10-27 — audit-log retention (audit §7.2.9, KB #46).
// Before: persistAudit kept the newest 2000 rows (~3 market hours) by running
// a NOT IN (... LIMIT 2000) delete after every insert. After: 30-day window,
// 300k-row hard cap, periodic prune, (time) and (type, id) indexes, and a
// ?since= filter on the diag audit probe.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import Database from "better-sqlite3";
import {
  AUDIT_MAX_ROWS, AUDIT_QUERY_MAX_LIMIT, AUDIT_RETENTION_DAYS,
  ensureAuditSchema, parseAuditSince, pruneAuditLog, queryAuditLog,
} from "./auditRetention";

const NOW = Date.parse("2026-10-27T20:00:00Z");
const H = 3_600_000;

function seed(db: Database.Database, rows: Array<[number, string, string]>) {
  const ins = db.prepare("INSERT INTO audit_log (time, type, message) VALUES (?, ?, ?)");
  for (const [ms, type, msg] of rows) ins.run(new Date(ms).toISOString(), type, msg);
}

test("retention keeps far more than 2000 rows: a full day of activity survives a prune", () => {
  const db = new Database(":memory:");
  ensureAuditSchema(db);
  // 10k rows over the last 24 h — the old rule would have kept only 2000.
  const rows: Array<[number, string, string]> = [];
  for (let i = 0; i < 10_000; i++) rows.push([NOW - 24 * H + i * 8640, i % 50 ? "TIER2" : "TRADE", `row ${i}`]);
  seed(db, rows);
  const r = pruneAuditLog(db, NOW);
  assert.deepEqual(r, { byAge: 0, byCap: 0 });
  assert.equal((db.prepare("SELECT COUNT(*) AS n FROM audit_log").get() as { n: number }).n, 10_000);
});

test("rows older than the retention window are removed, newer ones kept", () => {
  const db = new Database(":memory:");
  ensureAuditSchema(db);
  seed(db, [
    [NOW - (AUDIT_RETENTION_DAYS + 1) * 24 * H, "OLD", "a"],
    [NOW - (AUDIT_RETENTION_DAYS - 1) * 24 * H, "KEEP", "b"],
    [NOW - H, "KEEP", "c"],
  ]);
  const r = pruneAuditLog(db, NOW);
  assert.equal(r.byAge, 1);
  const types = (db.prepare("SELECT type FROM audit_log").all() as Array<{ type: string }>).map((x) => x.type);
  assert.deepEqual(types, ["KEEP", "KEEP"]);
});

test("the row cap keeps exactly the newest maxRows", () => {
  const db = new Database(":memory:");
  ensureAuditSchema(db);
  const rows: Array<[number, string, string]> = [];
  for (let i = 0; i < 50; i++) rows.push([NOW - 50_000 + i * 1000, "T", `m${i}`]);
  seed(db, rows);
  const r = pruneAuditLog(db, NOW, { maxRows: 20 });
  assert.equal(r.byCap, 30);
  const msgs = (db.prepare("SELECT message FROM audit_log ORDER BY id").all() as Array<{ message: string }>).map((x) => x.message);
  assert.equal(msgs.length, 20);
  assert.equal(msgs[0], "m30");
  assert.equal(msgs[19], "m49");
  assert.ok(AUDIT_MAX_ROWS >= 100_000, "the production cap must be well above the old 2000");
});

test("schema creates the time and (type,id) indexes", () => {
  const db = new Database(":memory:");
  ensureAuditSchema(db);
  ensureAuditSchema(db); // idempotent on an existing table
  const idx = (db.prepare("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='audit_log'").all() as Array<{ name: string }>).map((x) => x.name);
  assert.ok(idx.includes("idx_audit_log_time"));
  assert.ok(idx.includes("idx_audit_log_type_id"));
  const agePlan = JSON.stringify(db.prepare("EXPLAIN QUERY PLAN SELECT id FROM audit_log WHERE time < ?").all("2026-10-27"));
  assert.ok(/idx_audit_log_time/.test(agePlan), `the age prune must use the time index: ${agePlan}`);
  const typePlan = JSON.stringify(db.prepare(
    "EXPLAIN QUERY PLAN SELECT time, type, message FROM audit_log WHERE type = ? ORDER BY id DESC LIMIT 5").all("TRADE"));
  assert.ok(/idx_audit_log_type_id/.test(typePlan), `the ?type= probe must use the (type,id) index: ${typePlan}`);
});

test("queryAuditLog filters by since and type, newest first, bounded limit", () => {
  const db = new Database(":memory:");
  ensureAuditSchema(db);
  seed(db, [
    [NOW - 48 * H, "TRADE", "two days ago"],
    [NOW - 20 * H, "TRADE", "yesterday"],
    [NOW - 19 * H, "TIER2", "scan"],
    [NOW - H, "TRADE", "today"],
  ]);
  const since = parseAuditSince(new Date(NOW - 24 * H).toISOString())!;
  assert.deepEqual(queryAuditLog(db, { limit: 50, since }).map((r) => r.message), ["today", "scan", "yesterday"]);
  assert.deepEqual(queryAuditLog(db, { limit: 50, since, type: "TRADE" }).map((r) => r.message), ["today", "yesterday"]);
  assert.deepEqual(queryAuditLog(db, { limit: 1 }).map((r) => r.message), ["today"]);
  assert.equal(queryAuditLog(db, { limit: 10_000_000 }).length, 4);
  assert.ok(AUDIT_QUERY_MAX_LIMIT <= 5000);
});

test("parseAuditSince normalizes valid ISO input and rejects everything else", () => {
  assert.equal(parseAuditSince("2026-10-26"), "2026-10-26T00:00:00.000Z");
  assert.equal(parseAuditSince("2026-10-26T13:30:00Z"), "2026-10-26T13:30:00.000Z");
  assert.equal(parseAuditSince(""), null);
  assert.equal(parseAuditSince("yesterday"), null);
  assert.equal(parseAuditSince("1; DROP TABLE audit_log"), null);
  assert.equal(parseAuditSince(undefined), null);
});

// ── bot.ts wiring (static) ──────────────────────────────────────────────────
const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

test("persistAudit no longer truncates to the newest 2000 rows on every insert", () => {
  assert.ok(!/LIMIT 2000\)/.test(bot), "the per-insert NOT IN (... LIMIT 2000) delete must be gone");
  const start = bot.indexOf("function persistAudit(");
  const fn = bot.slice(start, bot.indexOf("function getPersistedAuditLog(", start));
  assert.ok(/pruneAuditLog\(db, Date\.now\(\)\)/.test(fn), "persistAudit prunes through auditRetention");
  assert.ok(/AUDIT_PRUNE_EVERY_INSERTS/.test(fn), "pruning is periodic, not per insert");
  assert.ok(/ensureAuditSchema\(db\)/.test(bot), "the indexes are created at boot");
});

test("the diag audit probe accepts ?since= and routes through queryAuditLog", () => {
  const start = bot.indexOf('case "audit": {');
  const probe = bot.slice(start, bot.indexOf('case "ml": {', start));
  assert.ok(/parseAuditSince\(req\.query\.since\)/.test(probe));
  assert.ok(/queryAuditLog\(db, \{/.test(probe));
});
