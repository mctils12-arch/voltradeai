/**
 * exitRecorder.ts — a process that is KILLED must leave something behind.
 * The recorder itself is pure (node:test safe): the clock, the file system,
 * the audit sink and the memory sample are injected; only exitRecorderFile()
 * looks at the real disk, to find the volume.
 *
 * WHY (2026-11-18, 09:51 ET, market hours): the server booted eight times in
 * 56 seconds (14:51:17Z -> 14:52:13Z, v1.0.1463) and then stayed up. Nothing
 * asked it to stop: no SHUTDOWN row (a deploy sends SIGTERM and writes one),
 * no FATAL-EXCEPTION / FATAL-REJECTION row (crashVisibility.ts writes those),
 * the last merge was an hour earlier, the container held 2 GB of a 22.9 GB
 * cgroup and the volume was 57% full. A kill signal or a native abort runs no
 * JavaScript, so the dying process could not say anything, and the audit log
 * of the minute is eight "Server boot" rows with nothing between them. The
 * cause is in Railway's deploy log, which no session can open. Seven of those
 * boots died within ~3 s, and railway.json stops restarting after 10 failures
 * in a row: two more and the trading loop would have stayed down.
 *
 * WHAT THIS DOES — it cannot name the killer; it records what the victim
 * looked like and how often it happens:
 *  - every ALIVE_INTERVAL_MS the live process writes one small file: when it
 *    was last alive, how long it had been up, its boot stage and its memory;
 *  - a clean end (SIGTERM / SIGINT, or an uncaught exception the crash
 *    handler already recorded) marks that file first;
 *  - at boot, a previous process whose file carries no end mark was killed:
 *    one UNCLEAN-EXIT audit row says when it was last alive, how long it had
 *    lived, the stage it had reached and its memory then;
 *  - boots are counted: BURST_BOOTS or more inside BURST_WINDOW_MS writes one
 *    RESTART-BURST row per boot of the burst;
 *  - status() puts all of it on /api/health as `checks.restarts` (never a
 *    serving gate, never the top-level status), and
 *    scripts/session_health_check.py reads it.
 *
 * Nothing here can throw into the caller: a recorder that breaks the boot is
 * worse than the silence it replaces.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export const EXIT_RECORDER_FILE = "voltrade_exit_recorder.json";

/** Where the record lives: the VOLUME (DATA_DIR, else /data/voltrade on
 *  Railway). It has to outlive the container that was killed — a file in the
 *  container's own tmp dies with it, which is the failure this module exists
 *  to end. Local runs without a volume fall back to the OS tmp dir. */
export function exitRecorderFile(env: NodeJS.ProcessEnv = process.env, exists: (p: string) => boolean = fs.existsSync): string {
  const dir = env.DATA_DIR || (exists("/data") ? "/data/voltrade" : os.tmpdir());
  return path.join(dir, EXIT_RECORDER_FILE);
}
export const UNCLEAN_EXIT = "UNCLEAN-EXIT";
export const RESTART_BURST = "RESTART-BURST";
/** how often the live process says "still here" — the resolution of "last
 *  seen alive". One ~1 KB write; the volume takes thousands of larger ones a
 *  day. */
export const ALIVE_INTERVAL_MS = 10_000;
export const BURST_WINDOW_MS = 10 * 60_000;
export const BURST_BOOTS = 3;
/** how long a boot / an unclean exit stays in the file and on /api/health */
export const KEEP_MS = 24 * 3600_000;
export const BOOTS_MAX = 200;
export const UNCLEAN_MAX = 20;
/** railway.json deploy.restartPolicyMaxRetries (a frozen file; quoted in the
 *  rows so the reader knows how close a burst came) */
export const RAILWAY_MAX_RETRIES = 10;

export interface MemorySample {
  heapUsedMB: number | null;
  rssMB: number | null;
  cgroupUsageMB: number | null;
}

export interface BootRecord {
  at: string;
  version: string;
  pid: number;
  stage: string;
  stageAt: string;
  lastAliveAt: string;
  sample: MemorySample | null;
  /** null while the process lives; set first thing on a clean end */
  end: { kind: string; at: string } | null;
}

export interface UncleanExit {
  bootAt: string;
  version: string;
  lastAliveAt: string;
  /** seconds between its boot and its last "still here" */
  livedS: number;
  /** seconds between its last "still here" and the boot that found it */
  goneS: number;
  stage: string;
  sample: MemorySample | null;
  detectedAt: string;
}

export interface RecorderState {
  v: 1;
  boot: BootRecord | null;
  boots: string[];
  unclean: UncleanExit[];
}

export interface RecorderFs {
  readFileSync(path: string, enc: "utf8"): string;
  writeFileSync(path: string, data: string): void;
  renameSync(from: string, to: string): void;
}

export interface RecorderOpts {
  file: string;
  fs: RecorderFs;
  now: () => number;
  audit: (action: string, detail: string) => void;
  sample: () => MemorySample;
  version: string;
  pid: number;
}

export interface RestartStatus {
  status: "ok" | "warning";
  detail: string | null;
  boots_10m: number;
  boots_24h: number;
  burst: boolean;
  last_boot_at: string | null;
  unclean_24h: number;
  last_unclean: UncleanExit | null;
  /** times the recorder could not read or write its file since this boot */
  recorder_faults: number;
  gates_top_level_status: false;
}

const iso = (ms: number) => new Date(ms).toISOString();
const ms = (s: unknown): number => {
  const t = typeof s === "string" ? Date.parse(s) : NaN;
  return Number.isFinite(t) ? t : NaN;
};
const mb = (v: number | null | undefined) => (typeof v === "number" && Number.isFinite(v) ? `${Math.round(v)} MB` : "unknown");

function emptyState(): RecorderState {
  return { v: 1, boot: null, boots: [], unclean: [] };
}

/** Tolerant parse: anything that is not this file's shape is "no record". */
export function parseState(raw: string | null): RecorderState {
  if (!raw) return emptyState();
  let j: unknown;
  try { j = JSON.parse(raw); } catch { return emptyState(); }
  if (!j || typeof j !== "object") return emptyState();
  const o = j as Record<string, unknown>;
  const b = o.boot as Record<string, unknown> | null | undefined;
  const boot: BootRecord | null = b && typeof b === "object" && typeof b.at === "string" && Number.isFinite(ms(b.at))
    ? {
        at: b.at,
        version: String(b.version ?? "unknown"),
        pid: Number(b.pid) || 0,
        stage: String(b.stage ?? "boot"),
        stageAt: typeof b.stageAt === "string" ? b.stageAt : b.at,
        lastAliveAt: typeof b.lastAliveAt === "string" && Number.isFinite(ms(b.lastAliveAt)) ? b.lastAliveAt : b.at,
        sample: b.sample && typeof b.sample === "object" ? (b.sample as MemorySample) : null,
        end: b.end && typeof b.end === "object" && typeof (b.end as { kind?: unknown }).kind === "string"
          ? { kind: String((b.end as { kind: string }).kind), at: String((b.end as { at?: unknown }).at ?? "") }
          : null,
      }
    : null;
  const boots = Array.isArray(o.boots) ? o.boots.filter((x): x is string => typeof x === "string" && Number.isFinite(ms(x))) : [];
  const unclean = Array.isArray(o.unclean)
    ? o.unclean.filter((x): x is UncleanExit => !!x && typeof x === "object" && Number.isFinite(ms((x as UncleanExit).detectedAt)))
    : [];
  return { v: 1, boot, boots, unclean };
}

/** The sentence a human reads. No guess at the cause: the file cannot know it. */
export function describeUnclean(u: UncleanExit): string {
  const s = u.sample;
  return `the previous process (v${u.version}, booted ${u.bootAt}) ended with no shutdown signal and no recorded exception. `
    + `Last seen alive ${u.lastAliveAt}, ${u.livedS}s after its boot, at stage "${u.stage}"; this boot is ${u.goneS}s later. `
    + `Memory then: heap ${mb(s?.heapUsedMB)}, rss ${mb(s?.rssMB)}, container ${mb(s?.cgroupUsageMB)}. `
    + `A kill signal or a native crash runs no code, so nothing here can name the cause; Railway's deploy log for that minute has it.`;
}

export interface ExitRecorder {
  /** call once, as early in the boot as the audit sink exists */
  begin(): { unclean: UncleanExit | null; boots10m: number; burst: boolean };
  /** a boot milestone ("routes", "bot_initialized", "listening") */
  stage(name: string): void;
  /** the periodic "still here" */
  tick(): void;
  /** a clean end is about to happen (signal name, or "uncaught exception") */
  end(kind: string): void;
  status(): RestartStatus;
}

/** The process's one recorder, set by the orchestrator when it creates it.
 *  server/index.ts marks its boot milestones through markBootStage() here and
 *  never imports the orchestrator (test_spinout_boundary.py: only the reviewed
 *  HTTP-wiring files may import server/bot.ts). */
let active: ExitRecorder | null = null;
export function setActiveRecorder(r: ExitRecorder | null): void {
  active = r;
}
/** A boot milestone ("routes_registered", "listening"). A no-op until the
 *  orchestrator has set the recorder; never throws. */
export function markBootStage(name: string): void {
  if (active) active.stage(name);
}

export function createExitRecorder(opts: RecorderOpts): ExitRecorder {
  let state: RecorderState = emptyState();
  let faults = 0;

  const safeSample = (): MemorySample | null => {
    try { return opts.sample(); } catch { faults += 1; return null; }
  };
  const save = () => {
    try {
      const tmp = `${opts.file}.tmp`;
      opts.fs.writeFileSync(tmp, JSON.stringify(state));
      opts.fs.renameSync(tmp, opts.file);
    } catch {
      faults += 1;   // counted and shown on /api/health (recorder_faults); never thrown
    }
  };
  const safeAudit = (action: string, detail: string) => {
    try { opts.audit(action, detail); } catch { faults += 1; }
  };
  const recent = (list: string[], now: number, window: number) => list.filter((t) => now - ms(t) <= window && ms(t) <= now + 60_000);

  return {
    begin() {
      const now = opts.now();
      let prior: RecorderState;
      try {
        prior = parseState(opts.fs.readFileSync(opts.file, "utf8"));
      } catch {
        prior = emptyState();   // first boot on this volume, or an unreadable file: no record, not a fault
      }
      let found: UncleanExit | null = null;
      const pb = prior.boot;
      if (pb && !pb.end) {
        const bootMs = ms(pb.at), aliveMs = ms(pb.lastAliveAt);
        found = {
          bootAt: pb.at, version: pb.version, lastAliveAt: pb.lastAliveAt,
          livedS: Math.max(0, Math.round((aliveMs - bootMs) / 1000)),
          goneS: Math.max(0, Math.round((now - aliveMs) / 1000)),
          stage: pb.stage, sample: pb.sample, detectedAt: iso(now),
        };
      }
      const boots = [...recent(prior.boots, now, KEEP_MS), iso(now)].slice(-BOOTS_MAX);
      const unclean = [...prior.unclean.filter((u) => now - ms(u.detectedAt) <= KEEP_MS), ...(found ? [found] : [])].slice(-UNCLEAN_MAX);
      state = {
        v: 1,
        boot: { at: iso(now), version: opts.version, pid: opts.pid, stage: "boot", stageAt: iso(now), lastAliveAt: iso(now), sample: safeSample(), end: null },
        boots, unclean,
      };
      save();
      const boots10m = recent(boots, now, BURST_WINDOW_MS).length;
      const burst = boots10m >= BURST_BOOTS;
      if (found) safeAudit(UNCLEAN_EXIT, describeUnclean(found));
      if (burst) {
        safeAudit(RESTART_BURST, `${boots10m} boots in the last ${BURST_WINDOW_MS / 60_000} minutes `
          + `(${unclean.filter((u) => now - ms(u.detectedAt) <= BURST_WINDOW_MS).length} of them after a process that was killed). `
          + `Railway stops restarting after ${RAILWAY_MAX_RETRIES} failures in a row (railway.json restartPolicyMaxRetries); past that the trading loop stays down until a deploy.`);
      }
      return { unclean: found, boots10m, burst };
    },
    stage(name: string) {
      if (!state.boot) return;
      const now = opts.now();
      state.boot.stage = String(name).slice(0, 60);
      state.boot.stageAt = iso(now);
      state.boot.lastAliveAt = iso(now);
      state.boot.sample = safeSample();
      save();
    },
    tick() {
      if (!state.boot || state.boot.end) return;
      state.boot.lastAliveAt = iso(opts.now());
      state.boot.sample = safeSample();
      save();
    },
    end(kind: string) {
      if (!state.boot || state.boot.end) return;
      state.boot.end = { kind: String(kind).slice(0, 60), at: iso(opts.now()) };
      save();
    },
    status() {
      const now = opts.now();
      const boots10m = recent(state.boots, now, BURST_WINDOW_MS).length;
      const boots24h = recent(state.boots, now, KEEP_MS).length;
      const unclean24h = state.unclean.filter((u) => now - ms(u.detectedAt) <= KEEP_MS);
      const burst = boots10m >= BURST_BOOTS;
      const last = unclean24h.length ? unclean24h[unclean24h.length - 1] : null;
      let detail: string | null = null;
      if (burst) detail = `${boots10m} boots in the last ${BURST_WINDOW_MS / 60_000} minutes (Railway stops restarting after ${RAILWAY_MAX_RETRIES} failures in a row)`;
      else if (last) detail = `${unclean24h.length} process(es) killed without a shutdown signal in the last 24h; the last was seen alive ${last.lastAliveAt}, ${last.livedS}s after its boot`;
      return {
        status: burst || last ? "warning" : "ok",
        detail,
        boots_10m: boots10m,
        boots_24h: boots24h,
        burst,
        last_boot_at: state.boot?.at ?? null,
        unclean_24h: unclean24h.length,
        last_unclean: last,
        recorder_faults: faults,
        gates_top_level_status: false,
      };
    },
  };
}
