/**
 * fatalReport.ts — when Node itself aborts, keep its own account of why.
 *
 * WHY (2026-11-18, 09:51 ET): eight boots in 56 seconds and no row saying
 * why. exitRecorder.ts now records WHAT a killed process looked like; it
 * cannot say WHO ended it. One whole class of endings does leave evidence if
 * asked: a fatal error inside Node/V8 (the heap limit reached, a failed
 * allocation, an internal check) makes the runtime write a diagnostic report
 * before it aborts — the event, the native stack, the heap's size and limit,
 * the process's memory. The report goes to the process's working directory
 * by default, i.e. into a container that is about to be thrown away, and is
 * off unless requested. This turns it on and points it at the VOLUME.
 *
 * WHAT THE NEXT INCIDENT THEN SHOWS:
 *  - a report was written  -> the process ended itself: FATAL-REPORT names the
 *    event (e.g. "Allocation failed - JavaScript heap out of memory");
 *  - no report, and exitRecorder's UNCLEAN-EXIT -> it was killed from outside
 *    (the container's memory limit, the platform), with its memory 10 s before.
 * It cannot recover the cause of 2026-11-18: nothing was recording then.
 *
 * SECRETS: on Node 20 a report includes the process's environment variables
 * (the option that leaves them out, `excludeEnv`, exists only from 22.13; it
 * is set when present). So the raw file is never kept: the next boot reads
 * it, writes a short summary with NO environment in it to the audit log, and
 * deletes the file. Until that boot the raw file sits on the private volume.
 *
 * Nothing here can throw into the caller.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export const FATAL_REPORT = "FATAL-REPORT";
export const FATAL_REPORT_SUBDIR = "fatal_reports";
/** a report is ~10-200 KB; anything larger is not read into memory */
export const FATAL_REPORT_MAX_BYTES = 8 * 1024 * 1024;
/** reports summarised per boot (newest first); older ones are deleted unread */
export const FATAL_REPORT_MAX_PER_BOOT = 5;
const NATIVE_FRAMES = 6;
const SUMMARY_MAX = 1400;

/** On the volume (DATA_DIR, else /data/voltrade): it must outlive the container. */
export function fatalReportDir(env: NodeJS.ProcessEnv = process.env, exists: (p: string) => boolean = fs.existsSync): string {
  const base = env.DATA_DIR || (exists("/data") ? "/data/voltrade" : os.tmpdir());
  return path.join(base, FATAL_REPORT_SUBDIR);
}

export interface ReportApi {
  directory: string;
  reportOnFatalError: boolean;
  compact?: boolean;
  excludeEnv?: boolean;
}

export interface FatalFs {
  mkdirSync(p: string, o: { recursive: true }): unknown;
  readdirSync(p: string): string[];
  statSync(p: string): { size: number; mtimeMs: number };
  readFileSync(p: string, enc: "utf8"): string;
  unlinkSync(p: string): void;
}

export interface FatalSummary {
  file: string;
  at: string | null;
  event: string;
  trigger: string | null;
  heapUsedMB: number | null;
  heapLimitMB: number | null;
  rssMB: number | null;
  native: string[];
  line: string;
}

export interface FatalReportStatus {
  status: "ok" | "warning" | "off";
  detail: string | null;
  enabled: boolean;
  env_excluded: boolean | null;
  dir: string;
  found_at_boot: number;
  last: Omit<FatalSummary, "line"> | null;
  faults: number;
  gates_top_level_status: false;
}

const mb = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) && v >= 0 ? Math.round(v / (1024 * 1024)) : null);
const str = (v: unknown): string | null => (typeof v === "string" && v ? v : null);

/** Pure: a report's JSON text -> the few facts worth keeping. Never includes
 *  `environmentVariables`, the command line or the working directory. */
export function summarizeReport(file: string, raw: string): FatalSummary {
  let d: Record<string, unknown> = {};
  try {
    const j: unknown = JSON.parse(raw);
    if (j && typeof j === "object") d = j as Record<string, unknown>;
  } catch {
    d = {};   // unreadable: reported below as such, with the file's name
  }
  const header = (d.header && typeof d.header === "object" ? d.header : {}) as Record<string, unknown>;
  const heap = (d.javascriptHeap && typeof d.javascriptHeap === "object" ? d.javascriptHeap : {}) as Record<string, unknown>;
  const ru = (d.resourceUsage && typeof d.resourceUsage === "object" ? d.resourceUsage : {}) as Record<string, unknown>;
  const native = (Array.isArray(d.nativeStack) ? d.nativeStack : [])
    .map((f) => str((f as { symbol?: unknown } | null)?.symbol)?.trim() ?? "")
    // the report writer's own frames say nothing about the cause
    .filter((s) => s && !/TriggerNodeReport|node::report::|^\[node\]$/.test(s))
    .slice(0, NATIVE_FRAMES)
    .map((s) => s.slice(0, 110));
  const event = str(header.event) ?? "unreadable report";
  const s: Omit<FatalSummary, "line"> = {
    file: path.basename(file),
    at: str(header.dumpEventTime),
    event,
    trigger: str(header.trigger),
    heapUsedMB: mb(heap.usedMemory),
    heapLimitMB: mb(heap.memoryLimit),
    rssMB: mb(ru.rss) ?? mb(ru.maxRss),
    native,
  };
  const n = (v: number | null) => (v == null ? "unknown" : `${v} MB`);
  const line = (`the previous process ended ITSELF with a fatal error${s.at ? ` at ${s.at}` : ""}: ${s.event}`
    + `${s.trigger ? ` (${s.trigger})` : ""}. JS heap ${n(s.heapUsedMB)} of a ${n(s.heapLimitMB)} limit, process memory ${n(s.rssMB)}.`
    + `${native.length ? ` Native stack: ${native.join(" | ")}.` : ""}`
    + ` Raw report ${s.file} deleted after this summary (it can hold environment variables).`).slice(0, SUMMARY_MAX);
  return { ...s, line };
}

export interface FatalReports {
  /** call at boot, BEFORE enable(): summarise and delete what the last process left */
  collect(): FatalSummary[];
  /** ask Node to write a report on a fatal error, onto the volume */
  enable(report: ReportApi | null | undefined): boolean;
  status(): FatalReportStatus;
}

export function createFatalReports(opts: {
  dir: string;
  fs: FatalFs;
  audit: (action: string, detail: string) => void;
}): FatalReports {
  let enabled = false;
  let envExcluded: boolean | null = null;
  let found: FatalSummary[] = [];
  let faults = 0;
  const safeAudit = (a: string, d: string) => {
    try { opts.audit(a, d); } catch { faults += 1; }
  };
  return {
    collect() {
      let names: string[];
      try {
        names = opts.fs.readdirSync(opts.dir).filter((f) => /^report\..*\.json$/.test(f));
      } catch {
        return [];   // no directory yet: nothing was ever written
      }
      const files = names.map((f) => {
        const p = path.join(opts.dir, f);
        try { const st = opts.fs.statSync(p); return { p, size: st.size, m: st.mtimeMs }; } catch { faults += 1; return { p, size: -1, m: 0 }; }
      }).sort((a, b) => b.m - a.m);
      const out: FatalSummary[] = [];
      files.forEach((f, i) => {
        if (i < FATAL_REPORT_MAX_PER_BOOT) {
          let s: FatalSummary;
          if (f.size < 0 || f.size > FATAL_REPORT_MAX_BYTES) {
            s = summarizeReport(f.p, "");
            s = { ...s, event: `report not read (${f.size} bytes)`, line: `a fatal-error report ${s.file} was left by the previous process but was not read (${f.size} bytes); deleted.` };
          } else {
            try { s = summarizeReport(f.p, opts.fs.readFileSync(f.p, "utf8")); } catch { faults += 1; s = summarizeReport(f.p, ""); }
          }
          out.push(s);
          safeAudit(FATAL_REPORT, s.line);
        }
        try { opts.fs.unlinkSync(f.p); } catch { faults += 1; }
      });
      if (files.length > FATAL_REPORT_MAX_PER_BOOT) {
        safeAudit(FATAL_REPORT, `${files.length - FATAL_REPORT_MAX_PER_BOOT} older fatal-error report(s) deleted unread (only the newest ${FATAL_REPORT_MAX_PER_BOOT} are summarised per boot).`);
      }
      found = out;
      return out;
    },
    enable(report) {
      if (!report || typeof report !== "object") return false;
      try {
        opts.fs.mkdirSync(opts.dir, { recursive: true });
        report.directory = opts.dir;
        report.compact = true;
        if ("excludeEnv" in report) { report.excludeEnv = true; envExcluded = true; } else { envExcluded = false; }
        report.reportOnFatalError = true;
        enabled = true;
      } catch {
        faults += 1;
        enabled = false;
      }
      return enabled;
    },
    status() {
      const last = found.length ? found[0] : null;
      return {
        status: !enabled ? "off" : last ? "warning" : "ok",
        detail: last ? `${found.length} fatal-error report(s) found at boot; the newest: ${last.event}${last.at ? ` at ${last.at}` : ""}` : null,
        enabled,
        env_excluded: envExcluded,
        dir: opts.dir,
        found_at_boot: found.length,
        last: last ? { file: last.file, at: last.at, event: last.event, trigger: last.trigger, heapUsedMB: last.heapUsedMB, heapLimitMB: last.heapLimitMB, rssMB: last.rssMB, native: last.native } : null,
        faults,
        gates_top_level_status: false,
      };
    },
  };
}
