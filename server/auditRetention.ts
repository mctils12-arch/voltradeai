// ── Audit-log retention (2026-10-27, [RULE-REVIEW], audit §7.2.9 / KB #46) ──
// persistAudit (bot.ts) used to run
//   DELETE FROM audit_log WHERE id NOT IN (SELECT id ... ORDER BY id DESC LIMIT 2000)
// after EVERY insert. 2000 rows is ~3 market hours of this log, so "did X
// happen yesterday" could not be answered from it, and every counterfactual
// or incident review that leans on the audit trail lost its evidence the
// same afternoon.
//
// Now: rows are kept for AUDIT_RETENTION_DAYS, with AUDIT_MAX_ROWS as a hard
// ceiling (whichever bites first), pruned every AUDIT_PRUNE_EVERY_INSERTS
// inserts instead of on every insert. Two indexes keep the reads cheap at
// that size: (time) for the age prune and ?since= queries, (type, id) for the
// diag probe's ?type= filter.
//
// DISK ESTIMATE (Railway volume): 2000 rows ~= 3 market hours -> ~650 rows/h.
// The loop is active ~16 h/day (04:00-20:00 ET) -> ~10k rows/day, ~300k per
// 30 days. Row: ISO time 24 B + type ~12 B + message (capped 500, typically
// 100-200 B) + rowid/record overhead ~15 B ~= 200 B; the two indexes add
// ~70 B/row. 300k x ~270 B ~= 80 MB worst case at the cap. AUDIT_MAX_ROWS
// bounds it even if the log gets chattier: the cap wins before disk does.

export const AUDIT_RETENTION_DAYS = 30;
export const AUDIT_MAX_ROWS = 300_000;
export const AUDIT_PRUNE_EVERY_INSERTS = 200;
/** Hard ceiling on rows one read may return (diag probe, API). */
export const AUDIT_QUERY_MAX_LIMIT = 2000;

/** The slice of better-sqlite3's Database this module uses. */
export interface AuditDb {
  prepare(sql: string): {
    run(...params: unknown[]): { changes: number };
    all(...params: unknown[]): unknown[];
    get(...params: unknown[]): unknown;
  };
}

export interface AuditRow { time: string; type: string; message: string }

export function ensureAuditSchema(db: AuditDb): void {
  db.prepare(`CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    type TEXT NOT NULL,
    message TEXT NOT NULL
  )`).run();
  db.prepare("CREATE INDEX IF NOT EXISTS idx_audit_log_time ON audit_log(time)").run();
  db.prepare("CREATE INDEX IF NOT EXISTS idx_audit_log_type_id ON audit_log(type, id)").run();
}

/** Delete rows older than the retention window, then everything below the
 *  newest `maxRows`. Both deletes are index-driven (time index, rowid). */
export function pruneAuditLog(
  db: AuditDb, nowMs: number,
  opts: { days?: number; maxRows?: number } = {},
): { byAge: number; byCap: number } {
  const days = opts.days ?? AUDIT_RETENTION_DAYS;
  const maxRows = opts.maxRows ?? AUDIT_MAX_ROWS;
  const cutoff = new Date(nowMs - days * 86_400_000).toISOString();
  const byAge = db.prepare("DELETE FROM audit_log WHERE time < ?").run(cutoff).changes;
  // ids only grow and rows are only ever removed from the old end, so
  // "everything at or below MAX(id) - maxRows" is the newest-maxRows cut in
  // O(log n) (any gap would only make it keep fewer rows, never more).
  let byCap = 0;
  const top = db.prepare("SELECT MAX(id) AS m FROM audit_log").get() as { m?: number | null } | undefined;
  if (top && typeof top.m === "number" && top.m > maxRows) {
    byCap = db.prepare("DELETE FROM audit_log WHERE id <= ?").run(top.m - maxRows).changes;
  }
  return { byAge, byCap };
}

/** Normalize a ?since= value (ISO date or date-time) to the stored ISO
 *  format so the string comparison on `time` is exact; null if invalid. */
export function parseAuditSince(raw: unknown): string | null {
  const s = String(raw ?? "").trim();
  if (!s || !/^\d{4}-\d{2}-\d{2}/.test(s)) return null;
  const t = Date.parse(s);
  return Number.isFinite(t) ? new Date(t).toISOString() : null;
}

/** Longest ?prefix= the audit probe accepts. */
export const AUDIT_PREFIX_MAX = 80;

/** Normalize a ?prefix= value: the start of a row's message, compared
 *  exactly (no wildcard). null when absent, empty or too long. */
export function parseAuditPrefix(raw: unknown): string | null {
  if (typeof raw !== "string") return null;
  return raw.length > 0 && raw.length <= AUDIT_PREFIX_MAX ? raw : null;
}

type AuditFilter = { type?: string | null; since?: string | null; prefix?: string | null };

function auditWhere(q: AuditFilter): { where: string[]; params: unknown[] } {
  const where: string[] = [];
  const params: unknown[] = [];
  if (q.type) { where.push("type = ?"); params.push(q.type); }
  if (q.since) { where.push("time >= ?"); params.push(q.since); }
  if (q.prefix) { where.push("substr(message, 1, ?) = ?"); params.push(q.prefix.length, q.prefix); }
  return { where, params };
}

/** How many rows match, with no cap. 2026-11-24 (open_questions (af2)): the
 *  diag probe returns at most 200 rows whatever `limit` asks (sanitizeDiag
 *  cuts every array), so a reader counting rows could not tell "200 scans"
 *  from "200 of 600". The probe now serves this beside the rows. */
export function countAuditLog(db: AuditDb, q: AuditFilter): number {
  const { where, params } = auditWhere(q);
  const row = db.prepare(
    "SELECT COUNT(*) AS n FROM audit_log" + (where.length ? ` WHERE ${where.join(" AND ")}` : ""),
  ).get(...params) as { n?: number } | undefined;
  return typeof row?.n === "number" ? row.n : 0;
}

/** Newest-first rows, optionally filtered by exact type, time >= since
 *  and/or the start of the message. */
export function queryAuditLog(
  db: AuditDb,
  q: { limit: number } & AuditFilter,
): AuditRow[] {
  const { where, params } = auditWhere(q);
  const limit = Math.min(Math.max(Math.floor(q.limit) || 1, 1), AUDIT_QUERY_MAX_LIMIT);
  const sql = "SELECT time, type, message FROM audit_log" +
    (where.length ? ` WHERE ${where.join(" AND ")}` : "") +
    " ORDER BY id DESC LIMIT ?";
  return db.prepare(sql).all(...params, limit) as AuditRow[];
}
