// SQLITE WRITE TIMING (REPAIR 2026-07-13, KNOWN BROKEN #18 follow-up)
//
// research/open_questions.md item #18: after the tmpCleanup.ts fix (v1.0.291)
// shipped on the theory that a synchronous /tmp directory sweep was blocking
// the Node event loop on a ~600s cadence, live data (2026-07-13 pre-flight
// check) REFUTED it — EVENTLOOP-LAG entries continued on the same ~10-minute
// rhythm, magnitude GREW (86,000-97,800ms vs. the pre-fix 59,000-75,000ms),
// and zero TMP-CLEANUP entries fired in the same window (the sweep was never
// even scanning enough files to be the trigger). Per that item's own next
// step, this is a fresh re-grep of bot.ts's setIntervals for the next
// ~600000ms-period blocking candidate.
//
// READ BEFORE WRITE traced a DIFFERENT, previously-uninvestigated candidate:
// `db` (server/auth.ts, imported into bot.ts for the audit_log table) is a
// better-sqlite3 connection — a fully SYNCHRONOUS binding, every .run()/.get()
// call blocks the calling thread (the Node event loop, there is no worker
// thread involved) for the full duration of the underlying write syscall,
// including any journal-file fsync. That connection's DB_PATH resolves to
// `/data/voltrade.db` — the Railway PERSISTENT VOLUME, not local/ephemeral
// disk — and auth.ts never sets `PRAGMA journal_mode`, so it runs SQLite's
// default rollback-journal mode, which does a create-write-fsync-delete
// cycle on a separate journal file for every single transaction. audit()
// calls persistAudit() synchronously on every audit-worthy event across the
// bot (dozens to hundreds of times across any given 10-minute window), so if
// the persistent volume exhibits any periodic I/O latency (a known
// characteristic of network-attached block storage — background snapshot/
// housekeeping jobs are a common cause), whichever synchronous write happens
// to be in flight (or issued next) during that window blocks the entire
// process for as long as the write takes. This explains the observed
// ~600s-periodic, market-hours-independent cadence (an infra-side period,
// not an app-side one) better than the refuted tmp-file theory did, without
// requiring Tier 2's own interval (which varies 60s-1800s by time of day —
// inconsistent with the constant ~600s cadence actually observed) to line up.
//
// NOT YET DIRECTLY MEASURED at the moment of a live EVENTLOOP-LAG entry —
// this module exists to make that measurement possible instead of guessed
// at, exactly like eventLoopLag.ts did for the (now refuted) zombie-thread
// theory: it times persistAudit()'s two statements independently and audits
// a DB-SLOW-WRITE entry (with which statement was slow and for how long)
// whenever either one crosses the threshold. If DB-SLOW-WRITE entries land
// inside the next EVENTLOOP-LAG window, that confirms this theory; if
// EVENTLOOP-LAG keeps firing with zero coincident DB-SLOW-WRITE entries, it
// is refuted in turn and the search moves to the next candidate (per RULE
// REVIEW/RECURRENCE ESCALATES discipline: read the instrument before
// patching again). Pure measurement — no trading, sizing, or scheduling
// behavior changes.
//
// Alongside the measurement, this session also switched the shared `db`
// connection to WAL journal mode (wired from bot.ts, since server/auth.ts is
// a FROZEN PATH — calling `.pragma()` on the already-exported `db` instance
// does not edit that file). WAL avoids the create/fsync/delete overhead of a
// per-transaction rollback-journal file regardless of whether this specific
// stall theory is confirmed, so it stands on its own as a low-risk
// improvement to a known synchronous-write hot path — `synchronous` was left
// untouched (SQLite's FULL default) to avoid weakening durability for the
// users/sessions tables auth.ts also owns.

export const DB_SLOW_WRITE_THRESHOLD_MS = 500;

export function isSlowDbWrite(durationMs: number): boolean {
  return durationMs >= DB_SLOW_WRITE_THRESHOLD_MS;
}

export function formatSlowWriteMessage(insertMs: number, deleteMs: number): string {
  return `audit_log write stalled — INSERT ${insertMs}ms, DELETE ${deleteMs}ms ` +
    `(KNOWN BROKEN #18 diagnostic, threshold ${DB_SLOW_WRITE_THRESHOLD_MS}ms)`;
}
