// REPAIR 2026-10-27 (trading-engine audit §1.2 / defect D1): how bot.ts runs a
// Python subprocess — which lane, which kill timer, and how the kill lands.
//
// WHAT WAS WRONG. execPythonSerialized() called a command "trivial" from its
// TEXT alone (<= 150 chars, no literal `import numpy|pandas|...`), then ran
// every trivial command with no mutex and `Math.min(timeout, 10000)` +
// SIGKILL — silently overriding whatever timeout the caller had asked for.
// `python3 ml_retrain_safe.py` is 26 chars, so the hourly ML retrain that asks
// for 900 s was SIGKILLed at 10 s (live: TIER3-ML-ERROR "(code=? signal=SIGKILL)"
// 10 s after "triggering retrain"); so were the nightly shadow backfill
// (asks 120 s), /api/bot/insight (40 s), the SEC Form 4 update (120 s), the EOD
// intraday-short cover (30 s) and others. And the kill went to the `/bin/sh -c`
// child only: a Python started under it (any command the shell did not
// exec-in-place, e.g. one with a redirection or a pipeline) survived as an
// orphan, still holding its memory, with nothing tracking it.
//
// THE FIX, as pure, tested functions:
//   * isTrivialPython(cmd, requestedTimeoutMs) — trivial now also requires the
//     caller NOT to have asked for more than the trivial cap. A caller asking
//     for 15 s / 120 s / 900 s has declared the job non-trivial; that
//     declaration is honoured instead of being overridden by a text heuristic.
//   * planPythonExec(cmd, requestedTimeoutMs) — the lane (trivial: no mutex;
//     mutex: serialized behind bot.ts's global Python mutex) and the kill timer,
//     which is now the caller's own timeout, bounded by the lane's ceiling. The
//     mutex ceiling stays 90 s, exactly as before, so the worst-case time one
//     holder can block every other mutex caller is unchanged.
//   * Jobs that legitimately need more than that ceiling (the ML retrain) do
//     not go through the mutex at all: bot.ts runs them in a separate
//     single-flight long-job lane with an explicit budget (ML_RETRAIN_TIMEOUT_MS)
//     so they never hold the mutex — and never block the scan loop — for
//     minutes.
//   * execGroupKill() — exec() semantics (shell, env, maxBuffer, utf8 stdout,
//     the same error shape: .code/.signal/.killed/.stdout/.stderr) but the
//     child is started as its own process-group leader (detached) and every
//     kill (timeout, maxBuffer) goes to the WHOLE group, so no grandchild
//     survives its parent's timeout.
import { spawn, type ExecOptions } from "child_process";

/** Kill timer for a trivial (no-mutex) command. Unchanged from before. */
export const PY_TRIVIAL_TIMEOUT_MS = 10_000;
/** Mutex-lane kill timer when the caller names none. Unchanged from before. */
export const PY_MUTEX_DEFAULT_TIMEOUT_MS = 90_000;
/**
 * Ceiling on how long one command may hold the global Python mutex. Unchanged
 * (90 s): raising it would let one holder starve every other mutex caller for
 * longer. Anything that needs more belongs in the long-job lane.
 */
export const PY_MUTEX_MAX_TIMEOUT_MS = 90_000;
/** How long a mutex-lane caller waits for the lock when it names no timeout. Unchanged. */
export const PY_MUTEX_DEFAULT_WAIT_MS = 30_000;

/**
 * ML retrain budget (`python3 ml_retrain_safe.py` → ml_model_v2.train_model(fast_mode=True)),
 * estimated from the code, worst case: Alpaca daily bars for <= 100 tickers in
 * 10 batches at a 20 s HTTP timeout each (<= ~200 s), the earnings-surprise
 * fetch for 30 tickers, feature build over ~2 years of bars, then <= 10k rows ×
 * 34 features through 5 walk-forward folds + 4 regime models at 300 boosting
 * rounds on 2 threads (well under a minute each). Typical total: a few minutes;
 * pessimistic: ~10 min. 20 min leaves 2x headroom over the pessimistic case
 * while still bounding a hung run (the old code comments asked for 900 s; the
 * 10 s cap meant nothing ever measured it — the audit line now records the
 * elapsed seconds so the budget can be tightened from evidence).
 */
export const ML_RETRAIN_TIMEOUT_MS = 20 * 60_000;

const HEAVY_IMPORT_RE = /import\s+(numpy|pandas|sklearn|lightgbm|scipy|torch)/;

/** A requested timeout, or undefined when the caller named none (0/negative/NaN = none, as before). */
function normalizeTimeout(requestedTimeoutMs?: number): number | undefined {
  return typeof requestedTimeoutMs === "number" && Number.isFinite(requestedTimeoutMs) && requestedTimeoutMs > 0
    ? requestedTimeoutMs : undefined;
}

/**
 * Trivial = may skip the global Python mutex under a 10 s kill. The old
 * text test still applies (short, no literal heavy import) AND the caller must
 * not have asked for longer than the trivial cap — a caller that asks for more
 * has declared the job non-trivial. With no timeout argument this is exactly
 * the old text-only classification.
 */
export function isTrivialPython(cmd: string, requestedTimeoutMs?: number): boolean {
  if (cmd.length > 150) return false;
  if (HEAVY_IMPORT_RE.test(cmd)) return false;
  const req = normalizeTimeout(requestedTimeoutMs);
  return req === undefined || req <= PY_TRIVIAL_TIMEOUT_MS;
}

export interface PythonExecPlan {
  lane: "trivial" | "mutex";
  /** The kill timer handed to the subprocess. */
  timeoutMs: number;
  /** Mutex lane only: how long to wait for the lock before giving up. */
  mutexWaitMs: number;
}

/** Lane + kill timer for one execPythonSerialized call. */
export function planPythonExec(cmd: string, requestedTimeoutMs?: number): PythonExecPlan {
  const req = normalizeTimeout(requestedTimeoutMs);
  if (isTrivialPython(cmd, req)) {
    return { lane: "trivial", timeoutMs: req ?? PY_TRIVIAL_TIMEOUT_MS, mutexWaitMs: 0 };
  }
  return {
    lane: "mutex",
    timeoutMs: Math.min(req ?? PY_MUTEX_DEFAULT_TIMEOUT_MS, PY_MUTEX_MAX_TIMEOUT_MS),
    mutexWaitMs: req ?? PY_MUTEX_DEFAULT_WAIT_MS,
  };
}

/** The error execGroupKill rejects with — the same fields child_process.exec's error carries. */
export interface GroupExecError extends Error {
  cmd: string;
  code: number | string | null;
  signal: NodeJS.Signals | null;
  killed: boolean;
  stdout: string;
  stderr: string;
}

/** Signal every process in the group led by `pid`; false if none was there to signal. */
export function killProcessGroup(pid: number, signal: NodeJS.Signals): boolean {
  try {
    process.kill(-pid, signal);
    return true;
  } catch (e: unknown) {
    // ESRCH: the group is already gone. Anything else (EPERM) is unexpected
    // for our own child — surface it rather than swallow it.
    if ((e as NodeJS.ErrnoException)?.code !== "ESRCH") console.error("[execGroupKill] group kill failed:", e);
    return false;
  }
}

/**
 * child_process.exec() semantics, process-group kill. Same shell (`/bin/sh -c`
 * unless opts.shell names one), env, cwd, timeout, killSignal and maxBuffer
 * options; stdout/stderr decoded as utf8; resolves {stdout, stderr} on exit 0,
 * otherwise rejects with exec's error shape. The child leads its own process
 * group (detached), so a timeout or maxBuffer kill takes every descendant
 * with it. The child is NOT unref'd: Node still waits for it like exec does.
 */
export function execGroupKill(
  cmd: string, opts: ExecOptions & { encoding?: BufferEncoding } = {},
): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const shell = typeof opts.shell === "string" ? opts.shell : "/bin/sh";
    const maxBuffer = opts.maxBuffer ?? 1024 * 1024;
    const killSignal = (opts.killSignal ?? "SIGTERM") as NodeJS.Signals;
    const out: Buffer[] = [];
    const err: Buffer[] = [];
    let outLen = 0;
    let errLen = 0;
    let killed = false;
    let overflow = false;
    let settled = false;

    const child = spawn(shell, ["-c", cmd], {
      cwd: opts.cwd, env: opts.env, detached: true, stdio: ["ignore", "pipe", "pipe"],
    });

    const killAll = () => {
      killed = true;
      if (child.pid === undefined || !killProcessGroup(child.pid, killSignal)) child.kill(killSignal);
    };
    const timer = opts.timeout && opts.timeout > 0 ? setTimeout(killAll, opts.timeout) : null;

    const collect = (bucket: Buffer[], chunk: Buffer, isOut: boolean) => {
      if (overflow) return;
      const len = (isOut ? outLen : errLen) + chunk.length;
      if (len > maxBuffer) {
        overflow = true;
        killAll();
        return;
      }
      bucket.push(chunk);
      if (isOut) outLen = len; else errLen = len;
    };
    child.stdout?.on("data", (c: Buffer) => collect(out, c, true));
    child.stderr?.on("data", (c: Buffer) => collect(err, c, false));

    const fail = (message: string, code: number | string | null, signal: NodeJS.Signals | null) => {
      const e = new Error(message) as GroupExecError;
      e.cmd = cmd;
      e.code = code;
      e.signal = signal;
      e.killed = killed;
      e.stdout = Buffer.concat(out).toString("utf8");
      e.stderr = Buffer.concat(err).toString("utf8");
      reject(e);
    };

    child.on("error", (spawnErr: NodeJS.ErrnoException) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      fail(spawnErr.message, spawnErr.code ?? null, null);
    });

    child.on("close", (code: number | null, signal: NodeJS.Signals | null) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      const stdout = Buffer.concat(out).toString("utf8");
      const stderr = Buffer.concat(err).toString("utf8");
      if (overflow) {
        fail("stdout maxBuffer length exceeded", "ERR_CHILD_PROCESS_STDIO_MAXBUFFER", signal);
      } else if (code === 0 && !signal && !killed) {
        resolve({ stdout, stderr });
      } else {
        fail(`Command failed: ${cmd}\n${stderr}`, code, signal);
      }
    });
  });
}
