// See server/dbWriteTiming.ts header for the full KNOWN BROKEN #18 evidence
// trail this diagnostic tests (tmpCleanup.ts's fix confirmed refuted
// 2026-07-13; synchronous better-sqlite3 writes to the /data persistent
// volume is the new, previously-uninvestigated candidate).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  isSlowDbWrite,
  formatSlowWriteMessage,
  DB_SLOW_WRITE_THRESHOLD_MS,
} from "./dbWriteTiming";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

test("isSlowDbWrite: below the threshold is not slow", () => {
  assert.equal(isSlowDbWrite(DB_SLOW_WRITE_THRESHOLD_MS - 1), false);
});

test("isSlowDbWrite: at or above the threshold is slow", () => {
  assert.equal(isSlowDbWrite(DB_SLOW_WRITE_THRESHOLD_MS), true);
  assert.equal(isSlowDbWrite(DB_SLOW_WRITE_THRESHOLD_MS + 60000), true);
});

test("isSlowDbWrite: a fast write (typical case) is not slow", () => {
  assert.equal(isSlowDbWrite(5), false);
});

test("formatSlowWriteMessage: reports both statement durations and the threshold", () => {
  const msg = formatSlowWriteMessage(12, 65000);
  assert.match(msg, /INSERT 12ms/);
  assert.match(msg, /DELETE 65000ms/);
  assert.match(msg, new RegExp(`threshold ${DB_SLOW_WRITE_THRESHOLD_MS}ms`));
});

test("wiring pinned: bot.ts imports the slow-write helpers from dbWriteTiming.ts", () => {
  assert.ok(
    /import\s*\{[^}]*isSlowDbWrite[^}]*\}\s*from\s*"\.\/dbWriteTiming"/.test(bot),
    "bot.ts must import isSlowDbWrite (and siblings) from ./dbWriteTiming rather than reimplementing the check inline",
  );
});

test("wiring pinned: bot.ts switches the shared db connection to WAL journal mode without editing the frozen auth.ts", () => {
  assert.ok(
    /db\.pragma\(\s*"journal_mode\s*=\s*WAL"/.test(bot),
    "bot.ts must set journal_mode=WAL on the db connection imported from auth.ts",
  );
});

test("wiring pinned: persistAudit times the INSERT and DELETE independently and gates DB-SLOW-WRITE on isSlowDbWrite", () => {
  const fnIdx = bot.indexOf("function persistAudit");
  assert.ok(fnIdx > 0, "persistAudit not found in bot.ts");
  const block = bot.slice(fnIdx, fnIdx + 1500);
  assert.ok(block.includes("insertStart"), "must time the INSERT statement");
  assert.ok(block.includes("deleteStart"), "must time the DELETE statement");
  assert.ok(
    /isSlowDbWrite\(insertMs\)\s*\|\|\s*isSlowDbWrite\(deleteMs\)/.test(block),
    "must gate the DB-SLOW-WRITE audit call on either statement crossing the threshold",
  );
  assert.ok(block.includes('audit("DB-SLOW-WRITE"'), "must log to the audit trail under a distinct DB-SLOW-WRITE type");
});

test("wiring pinned: persistAudit guards against re-entrant recursion when logging its own slow-write event", () => {
  const fnIdx = bot.indexOf("function persistAudit");
  const block = bot.slice(fnIdx, fnIdx + 1500);
  assert.ok(
    /type\s*!==\s*"DB-SLOW-WRITE"/.test(block),
    "must skip the DB-SLOW-WRITE check when already inside a DB-SLOW-WRITE call, or a slow diagnostic write recurses forever",
  );
});
