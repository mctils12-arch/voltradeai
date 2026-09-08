/**
 * crashSafeRefresh.ts — shared crash-loop guard for expensive, unconditional
 * "refresh once at boot, then every N minutes" cache-refresh jobs.
 *
 * WHY THIS EXISTS (2026-09-08 live-incident REPAIR, second PR on the same
 * incident — see research/experiments.md this date and KNOWN BROKEN #41 in
 * research/open_questions.md): production OOM-crash-looped roughly every
 * 90-130s during market hours. The first fix (server/portDwellCapture.ts)
 * targeted the Tier-3 in-process weekly-capture path and shipped as PR
 * #1030 — but the crash loop continued after that fix deployed. Re-tracing
 * live: `server/routes.ts`'s `refreshShadowStats()`/`refreshPortDwell()`
 * are called UNCONDITIONALLY the instant each is registered (i.e. at every
 * process boot, with no delay) and again every 10 minutes, each folding the
 * full growing AIS vessel archive (`computeShadowStatsAsync` /
 * `computePortDwellAsync`, the same class of archive-scan documented
 * elsewhere in this repo as "grows monotonically with the archive" —
 * see shadowFleet.ts's own KNOWN BROKEN #18 comment). Neither function
 * persisted any record of an in-flight attempt, so the exact same failure
 * shape as the Tier-3 case applies: if a fold OOM-kills the whole process
 * before finishing, the crash erases the only evidence an attempt was
 * made, and the very next boot retries the identical fold immediately.
 *
 * This module generalizes the fix `portDwellCapture.ts` shipped first
 * (durably record the attempt BEFORE the risky work runs, so a crash still
 * leaves the fact on disk) into a reusable form for jobs that have no
 * natural backlog/per-item state to hang a marker off of — just "did the
 * last attempt at this job ever finish."
 */
import fs from "node:fs";
import path from "node:path";

export function refreshStateDir(env: NodeJS.ProcessEnv = process.env): string {
  return env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp");
}

interface AttemptMarker { startedAt: number; completedAt?: number }

function attemptFile(dir: string, name: string): string {
  return path.join(dir, `voltrade_refresh_attempt_${name}.json`);
}

function loadMarker(dir: string, name: string): AttemptMarker | null {
  try {
    const raw = JSON.parse(fs.readFileSync(attemptFile(dir, name), "utf8"));
    return typeof raw?.startedAt === "number" ? raw : null;
  } catch { return null; }
}

function writeMarker(dir: string, name: string, marker: AttemptMarker): void {
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(attemptFile(dir, name), JSON.stringify(marker));
}

export interface GuardedRefreshResult {
  ran: boolean;
  reason?: "cooldown";
  detail?: string;
}

/** Wraps an expensive, unconditional refresh job with the crash-loop guard.
 *  A marker is written IMMEDIATELY BEFORE `fn` runs (durable, on disk) and
 *  updated with `completedAt` in a `finally` once `fn` returns or throws an
 *  ORDINARY (catchable) error — so a normal failure still resolves the
 *  marker and the job retries on its regular cadence. Only a hard process
 *  crash (which never reaches the `finally`) leaves the marker unresolved,
 *  and only THAT state triggers the cooldown on the next call — an ordinary
 *  caught error is never mistaken for a suspected crash. */
export async function guardedRefresh(name: string, cooldownMs: number, fn: () => Promise<void>,
                                      dir = refreshStateDir(), nowMs = Date.now()): Promise<GuardedRefreshResult> {
  const marker = loadMarker(dir, name);
  const unresolved = marker != null && marker.completedAt == null;
  if (unresolved && nowMs - marker!.startedAt < cooldownMs) {
    return {
      ran: false, reason: "cooldown",
      detail: `last attempt at ${new Date(marker!.startedAt).toISOString()} did not complete (suspected crash) — cooling down until ${new Date(marker!.startedAt + cooldownMs).toISOString()}`,
    };
  }
  writeMarker(dir, name, { startedAt: nowMs });
  try {
    await fn();
  } finally {
    writeMarker(dir, name, { startedAt: nowMs, completedAt: Date.now() });
  }
  return { ran: true };
}
