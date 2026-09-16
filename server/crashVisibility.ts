/**
 * crashVisibility.ts — a process that dies must say why, somewhere we can
 * read. Pure (node:test safe): the process object and the sinks are
 * injected.
 *
 * WHY (KNOWN BROKEN #41, the 2026-09-08 "crash loop"): the Node process
 * restarted every ~90-130s for hours and three sessions could only watch
 * `/api/health`'s uptime reset. They called it an OOM because rss was
 * climbing; on 2026-09-16 the live ceilings came back as a 6.2GB V8 heap cap
 * inside a 22.9GB cgroup — a process at ~1GB was never near either wall.
 * Whatever ended it wrote its reason to stderr, which only Railway's log
 * viewer shows, and this repo's sessions cannot open that. The server had
 * no `unhandledRejection` or `uncaughtException` handler at all, so on
 * Node >= 15 any rejected promise nobody awaited (fifty-five eager pollers
 * fire at boot, each a chain of fetch/parse/fold) terminates the process,
 * `run_with_daemon.sh` `exec`s Node so the container goes with it, and
 * Railway restarts it — with zero trace in the audit log.
 *
 * WHAT THIS DOES:
 *  - unhandledRejection: write a FATAL-REJECTION audit line (durable: the
 *    audit sink is a synchronous SQLite insert) with the reason and stack,
 *    and KEEP RUNNING. Registering the handler is what changes Node's
 *    default from "throw -> exit" to "reported"; that is deliberate
 *    (Priority 1: a rejected archive fetch must not take the trading loop
 *    down) and it is loud, not silent: the line is in the audit log,
 *    `crashStats()` puts the count and the last reason on /api/health, and
 *    the DAILY routine reads both.
 *  - uncaughtException: write a FATAL-EXCEPTION audit line, then exit(1)
 *    after a short grace so stdout flushes. Node's own docs are clear that
 *    continuing after an uncaught exception is unsafe; the change here is
 *    only that the reason is now on disk before the process goes.
 *
 * Neither handler can throw: a crash handler that crashes is the one
 * failure mode worse than none.
 */

export interface ProcessLike {
  on(event: "unhandledRejection", listener: (reason: unknown, promise: Promise<unknown>) => void): unknown;
  on(event: "uncaughtException", listener: (err: Error, origin: string) => void): unknown;
}

export type AuditSink = (action: string, detail: string) => void;

export interface CrashStats {
  unhandledRejections: number;
  uncaughtExceptions: number;
  /** times the crash handler ITSELF hit an error (audit sink threw, exit
   *  could not be scheduled) — nonzero means the trace is incomplete */
  handlerFaults: number;
  lastRejection: { at: string; detail: string } | null;
  lastException: { at: string; detail: string } | null;
}

export const FATAL_REJECTION = "FATAL-REJECTION";
export const FATAL_EXCEPTION = "FATAL-EXCEPTION";
/** ms between the audit write and process.exit on an uncaught exception —
 *  enough for stdout/stderr to flush, short enough that Railway's restart
 *  is not delayed in any way a human would notice. */
export const EXIT_GRACE_MS = 250;
const DETAIL_MAX = 1500;

const stats: CrashStats = { unhandledRejections: 0, uncaughtExceptions: 0, handlerFaults: 0, lastRejection: null, lastException: null };

/** One line a human can act on: message first, then the top of the stack,
 *  bounded. Non-Error reasons (strings, objects, undefined) are stringified
 *  rather than dropped — `throw "oops"` and `reject()` with no reason are
 *  exactly the shapes that produce useless logs elsewhere. */
export function describeReason(reason: unknown): string {
  let out: string;
  if (reason instanceof Error) {
    const stack = (reason.stack || "").split("\n").slice(1, 5).map((l) => l.trim()).join(" | ");
    out = `${reason.name}: ${reason.message}${stack ? ` @ ${stack}` : ""}`;
  } else if (reason === undefined) {
    out = "undefined (rejected with no reason)";
  } else if (typeof reason === "string") {
    out = reason;
  } else {
    try { out = JSON.stringify(reason); } catch { out = String(reason); }
    if (out === undefined) out = String(reason);
  }
  return out.length > DETAIL_MAX ? out.slice(0, DETAIL_MAX) + "…" : out;
}

export function crashStats(): CrashStats {
  return { ...stats, lastRejection: stats.lastRejection && { ...stats.lastRejection }, lastException: stats.lastException && { ...stats.lastException } };
}

export function _resetCrashStats(): void {
  stats.unhandledRejections = 0; stats.uncaughtExceptions = 0; stats.handlerFaults = 0; stats.lastRejection = null; stats.lastException = null;
}

export interface InstallOpts {
  audit: AuditSink;
  /** injected for tests; defaults to process.exit */
  exit?: (code: number) => void;
  /** injected for tests; defaults to setTimeout */
  schedule?: (fn: () => void, ms: number) => void;
  now?: () => Date;
}

/** Idempotent per process object: installing twice would double-log. */
const installed = new WeakSet<object>();

export function installCrashHandlers(proc: ProcessLike, opts: InstallOpts): boolean {
  if (installed.has(proc as object)) return false;
  installed.add(proc as object);
  const exit = opts.exit ?? ((code: number) => process.exit(code));
  const schedule = opts.schedule ?? ((fn, ms) => setTimeout(fn, ms));
  const now = opts.now ?? (() => new Date());
  const safeAudit = (action: string, detail: string) => {
    try {
      opts.audit(action, detail);
    } catch (e) {
      stats.handlerFaults += 1;
      try {
        console.error(`[crashVisibility] audit sink threw while recording ${action}:`, e);
      } catch (e2) {
        stats.handlerFaults += 1;
      }
    }
  };

  // describeReason() is total (every branch guarded) and safeAudit() cannot
  // throw, so neither handler needs an outer try: a throw here would itself
  // become an uncaughtException and be recorded by the handler below.
  proc.on("unhandledRejection", (reason) => {
    const detail = describeReason(reason);
    stats.unhandledRejections += 1;
    stats.lastRejection = { at: now().toISOString(), detail };
    safeAudit(FATAL_REJECTION, `unhandled promise rejection #${stats.unhandledRejections} (process kept alive; before 2026-09-16 this exited Node with no audit trace): ${detail}`);
  });

  proc.on("uncaughtException", (err, origin) => {
    const detail = describeReason(err);
    stats.uncaughtExceptions += 1;
    stats.lastException = { at: now().toISOString(), detail };
    safeAudit(FATAL_EXCEPTION, `uncaught exception (${origin}) — exiting 1 in ${EXIT_GRACE_MS}ms so Railway restarts a clean process: ${detail}`);
    try {
      schedule(() => exit(1), EXIT_GRACE_MS);
    } catch (e) {
      stats.handlerFaults += 1;
      exit(1);
    }
  });
  return true;
}
