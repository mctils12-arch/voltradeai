// ORPHANED-TEMP-FILE CLEANUP (REPAIR 2026-07-12, KNOWN BROKEN #18 follow-up)
//
// Extracted from server/bot.ts, where this ran as a setInterval(..., 600000)
// body using the synchronous fs API (readdirSync/statSync/unlinkSync) — all
// on the Node event loop's own thread. That interval's period (exactly 600000ms)
// matches the cadence observed in production between EVENTLOOP-LAG entries
// (59000-75000ms stalls) and their coincident STREAM-DISCONNECT entries
// (research/open_questions.md item #18): 2026-07-12 20:16:01Z carried both
// an EVENTLOOP-LAG entry (~59319ms) and a STREAM-DISCONNECT within 31ms of
// it, and the prior EVENTLOOP-LAG/STREAM-DISCONNECT pairs cluster on the
// same ~600s rhythm going back through the day. Synchronous fs calls in a
// 10-minute setInterval are a textbook event-loop-blocking pattern — if
// /tmp has accumulated enough fb_/fill_/opt_ files (e.g. from a burst of
// fire-and-forget Python subprocess calls that timed out before their own
// cleanup ran), readdirSync + a statSync/unlinkSync pair per file blocks
// the entire process for the full scan, during which nothing else (stop-
// loss monitoring, the WS ping/pong keepalive, other timers) can run.
//
// This does not yet PROVE the tmp-cleanup interval is the sole cause (per
// REASONING STANDARD #4/#10, that needs a direct measurement, not just a
// matching period) — but converting it to async fs.promises calls removes
// the hazard unconditionally: async fs work runs on libuv's thread pool,
// never blocks the event loop, regardless of how many files accumulate.
// The added `scanned` telemetry lets a future session see whether large
// counts were actually occurring (confirming the theory) even after the
// blocking risk itself is gone.
import fs from "node:fs/promises";

export const TMP_CLEANUP_INTERVAL_MS = 600000;
export const TMP_FILE_MAX_AGE_MS = 300000;
export const TMP_FILE_PREFIXES = ["fb_", "fill_", "opt_"];
// Large-scan threshold for the diagnostic audit line below — chosen well
// above any single Tier 2 scan's normal temp-file volume (dozens, not
// hundreds), so an audit entry here is itself evidence of an anomaly.
export const TMP_CLEANUP_AUDIT_THRESHOLD = 200;

export interface TmpCleanupResult {
  scanned: number;
  deleted: number;
}

export async function cleanupOrphanedTempFiles(tmpDir: string = "/tmp"): Promise<TmpCleanupResult> {
  let scanned = 0;
  let deleted = 0;
  let entries: string[];
  try {
    entries = await fs.readdir(tmpDir);
  } catch (_) {
    return { scanned: 0, deleted: 0 };
  }
  const targets = entries.filter((f) => TMP_FILE_PREFIXES.some((prefix) => f.startsWith(prefix)));
  scanned = targets.length;
  const now = Date.now();
  await Promise.all(
    targets.map(async (f) => {
      const full = `${tmpDir}/${f}`;
      try {
        const stat = await fs.stat(full);
        if (now - stat.mtimeMs > TMP_FILE_MAX_AGE_MS) {
          await fs.unlink(full);
          deleted++;
        }
      } catch (_) {
        // Gone already (raced with its own consumer's cleanup) or unreadable — skip.
      }
    }),
  );
  return { scanned, deleted };
}
