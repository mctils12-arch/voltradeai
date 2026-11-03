// A6 HEAVY PYTHON JOB QUEUE (2026-11-03, memory audit fix A6, second half;
// territory T-BOT).
//
// WHAT WAS WRONG. bot.ts ran its heavy Python subprocesses — each a whole
// numpy/pandas/LightGBM import, 250-600 MB apiece on Railway — in two lanes
// that never looked at each other:
//   * the global Python mutex (execPythonSerialized, 90 s ceiling): the
//     overnight research phases (20:00-04:00 ET: evening scan + extreme-mover
//     analyses, backtest, earnings analyses, pre-market scan) run here, one
//     subprocess at a time;
//   * the long-job lane (runLongPythonJob: the ML retrain, 20 min budget),
//     single-flight PER KEY and deliberately outside the mutex.
// So the 04:00 ET retrain — or the hourly Tier-3 retrain whenever the model is
// stale — could start while a research scan, backtest or analysis was still
// running: two heavy interpreters beside the daemon and Node in one cgroup.
//
// THE FIX: one FIFO queue for the HEAVY subprocess jobs — the retrain and each
// overnight research phase — so at most one of them runs at a time. Tier 1 and
// the Tier-2 scans are NOT in it: they must never wait behind research. Every
// job declares the budget its own SIGKILL timer already enforces; a holder
// still there past budget + grace is force-released (the same contract as the
// Python mutex watchdog), so a leaked promise can never wedge the retrain. A
// caller may bound its wait; giving up never lets a successor overtake the
// job that is actually running. Every wait is reported (bot.ts audits
// PY-JOB-QUEUED <name> waited_ms=<n>). Rollback: VOLTRADE_SERIALIZE_PY_JOBS=0
// makes run() call the job directly, exactly as before. The two timers here
// (stale release, bounded wait) are ordinary referenced timers, like
// execPythonSerialized's kill timers: a release that is due must fire even if
// nothing else is keeping the process alive.

export const SERIALIZE_PY_JOBS_ENV = "VOLTRADE_SERIALIZE_PY_JOBS";

/** Grace past a job's own budget before the queue stops waiting for it (mirrors the Python mutex's stale threshold). */
export const PY_JOB_STALE_GRACE_MS = 10_000;

/** "0"/"false"/"off"/"no" -> off; unset or anything else -> on (the default). */
export function serializePyJobs(env: Record<string, string | undefined> = process.env): boolean {
  const raw = env[SERIALIZE_PY_JOBS_ENV];
  if (raw === undefined) return true;
  const v = raw.trim().toLowerCase();
  return !(v === "0" || v === "false" || v === "off" || v === "no");
}

/** Thrown by run() when a bounded wait (maxWaitMs) expires before the job could start. The job never ran. */
export class PyJobWaitTimeout extends Error {
  constructor(public readonly job: string, public readonly waitedMs: number, public readonly behind: string) {
    super(`${job}: waited ${waitedMs} ms behind ${behind} without getting the heavy Python job queue`);
    this.name = "PyJobWaitTimeout";
  }
}

export interface PyJobQueueHooks {
  /** A job had to wait for a predecessor: called once, just before it starts. */
  onWait?: (name: string, waitedMs: number, behind: string) => void;
  /** A holder was still there past budget + grace and the queue stopped waiting for it. */
  onStaleRelease?: (name: string, heldMs: number, budgetMs: number) => void;
  /** Clock, for tests. */
  now?: () => number;
  /** Stale grace, for tests (default PY_JOB_STALE_GRACE_MS). */
  staleGraceMs?: number;
}

export interface PyJobRunOptions {
  /** Give up (PyJobWaitTimeout) if the job cannot start within this many ms. Unbounded when absent. */
  maxWaitMs?: number;
}

export interface PyJobOutcome<T> {
  result: T;
  /** How long the job waited for the queue before starting (0 when it ran at once or the queue is off). */
  waitedMs: number;
}

interface Holder {
  name: string;
  startedAt: number;
  budgetMs: number;
}

export class PyJobQueue {
  private tail: Promise<void> = Promise.resolve();
  private holder: Holder | null = null;
  private waitingCount = 0;

  constructor(private readonly serialize: boolean, private readonly hooks: PyJobQueueHooks = {}) {}

  /** The job running right now, if any. */
  get running(): string | null {
    return this.holder?.name ?? null;
  }

  /** Jobs waiting for their turn. */
  get waiting(): number {
    return this.waitingCount;
  }

  /** The queue is active (false = VOLTRADE_SERIALIZE_PY_JOBS off: jobs run directly). */
  get enabled(): boolean {
    return this.serialize;
  }

  /**
   * Run `fn` as the only heavy job: FIFO behind whatever is running or waiting.
   * `budgetMs` is the job's own kill timer; the queue stops waiting for a holder
   * at budgetMs + grace. Rejections from `fn` propagate after the queue is
   * released. With `maxWaitMs`, a job that cannot start in time throws
   * PyJobWaitTimeout without ever running — and without letting the jobs
   * queued behind it overtake the one that is actually running.
   */
  async run<T>(name: string, budgetMs: number, fn: () => Promise<T>, opts: PyJobRunOptions = {}): Promise<PyJobOutcome<T>> {
    if (!this.serialize) return { result: await fn(), waitedMs: 0 };
    const now = this.hooks.now ?? Date.now;
    const arrived = now();
    const behind = this.holder ? this.holder.name : this.waitingCount > 0 ? "queued jobs" : null;
    const prev = this.tail;
    let release: () => void = () => {};
    const mine = new Promise<void>((resolve) => { release = resolve; });
    this.tail = mine;
    this.waitingCount++;
    try {
      if (opts.maxWaitMs !== undefined && behind !== null) {
        let giveUpTimer: NodeJS.Timeout | undefined;
        const gaveUp = await Promise.race([
          prev.then(() => false),
          new Promise<boolean>((resolve) => {
            giveUpTimer = setTimeout(() => resolve(true), opts.maxWaitMs);
          }),
        ]);
        if (giveUpTimer) clearTimeout(giveUpTimer);
        if (gaveUp) {
          // Keep the chain intact: whoever queued behind this job still waits
          // for the job that is running, never overtaking it.
          void prev.then(release);
          throw new PyJobWaitTimeout(name, now() - arrived, behind);
        }
      } else {
        await prev;
      }
    } finally {
      this.waitingCount--;
    }
    const waitedMs = now() - arrived;
    if (behind !== null) this.hooks.onWait?.(name, waitedMs, behind);

    const me: Holder = { name, startedAt: now(), budgetMs };
    this.holder = me;
    const grace = this.hooks.staleGraceMs ?? PY_JOB_STALE_GRACE_MS;
    const staleTimer = setTimeout(() => {
      if (this.holder !== me) return;
      this.hooks.onStaleRelease?.(name, now() - me.startedAt, budgetMs);
      this.holder = null;
      release();
    }, budgetMs + grace);
    try {
      return { result: await fn(), waitedMs };
    } finally {
      clearTimeout(staleTimer);
      if (this.holder === me) this.holder = null;
      release();
    }
  }
}
