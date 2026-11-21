// jobsView — the Pipeline health panel's "Scheduled jobs" section: every job
// the server runs on a schedule, read against its own schedule
// (server/jobHeartbeat.ts publicJobsView, carried in
// /api/data/pipeline-health-dashboard as `jobs`). Pure functions with tests;
// the page lays them out. The server sends no free text from a failed run,
// only a closed cause, so every word a reader sees is written here.

export type JobVerdict = "ok" | "late" | "missing" | "errored" | "not_due";
export type JobErrorClass = "timeout" | "memory_budget" | "unparseable_output" | "nonzero_exit" | "interrupted" | "other";

export interface PublicJob {
  id: string;
  label: string;
  status: JobVerdict;
  schedule: string;
  last_start_at: string | null;
  last_finish_at: string | null;
  last_status: "ok" | "error" | "skipped" | "running" | null;
  running: boolean;
  disabled: boolean;
  recent_errors: number;
  error_class: JobErrorClass | null;
}
export interface PublicJobsView {
  evaluated_at: string;
  verdict: "ok" | "degraded";
  counts: Record<JobVerdict, number>;
  jobs: PublicJob[];
}

export type JobTone = "bad" | "ok" | "idle";
export interface JobRow {
  id: string;
  label: string;
  tone: JobTone;
  /** the chip: one or two words */
  state: string;
  schedule: string;
  /** when it last ran and how that run ended, in words */
  last: string;
  /** why it is failing, in words; null when it is not */
  cause: string | null;
}

const CAUSE: Record<JobErrorClass, string> = {
  timeout: "It ran past its time limit and was stopped.",
  memory_budget: "It ran past its memory limit and was stopped.",
  unparseable_output: "It finished, but its result could not be read.",
  nonzero_exit: "The program exited with an error.",
  interrupted: "A server restart cut it off before it finished.",
  other: "The cause is not classified.",
};

const STATE: Record<JobVerdict, string> = {
  ok: "on time", late: "late", missing: "never ran", errored: "last run failed", not_due: "not due now",
};

export function ago(iso: string | null | undefined, nowMs: number): string | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return null;
  const s = Math.max(0, Math.round((nowMs - t) / 1000));
  if (s < 90) return `${s} s ago`;
  const min = Math.round(s / 60);
  if (min < 90) return `${min} min ago`;
  const h = Math.round(min / 60);
  if (h < 48) return `${h} h ago`;
  return `${Math.round(h / 24)} d ago`;
}

function lastRun(j: PublicJob, nowMs: number): string {
  const started = ago(j.last_start_at, nowMs);
  if (!started) return "No run is recorded.";
  if (j.running) return `Running now, started ${started}.`;
  const how = j.last_status === "error" ? "it failed"
    : j.last_status === "skipped" ? "it had nothing to do"
    : j.last_status === "ok" ? "it finished"
    : "how it ended is not recorded";
  return `Last started ${started}; ${how}.`;
}

function cause(j: PublicJob): string | null {
  if (j.status === "errored") return CAUSE[j.error_class ?? "other"] ?? CAUSE.other;
  if (j.status === "late") return j.running ? "The run has been going longer than this job is allowed." : "No run started inside its schedule and allowance.";
  if (j.status === "missing") return "It has never run on this server, and its time has passed.";
  return null;
}

/** failing first, then on time, then not due, then switched off; registry order inside each */
function rank(j: PublicJob): number {
  if (j.status === "errored" || j.status === "late" || j.status === "missing") return 0;
  if (j.status === "ok") return 1;
  return j.disabled ? 3 : 2;
}

export function jobRows(view: PublicJobsView | null | undefined, nowMs: number): JobRow[] {
  const jobs = Array.isArray(view?.jobs) ? view!.jobs : [];
  return jobs
    .map((j, i) => ({ j, i }))
    .sort((a, b) => rank(a.j) - rank(b.j) || a.i - b.i)
    .map(({ j }) => {
      const failing = rank(j) === 0;
      return {
        id: j.id,
        label: j.label,
        tone: failing ? "bad" : j.status === "ok" ? "ok" : "idle",
        state: j.disabled ? "switched off" : j.running && !failing ? "running" : STATE[j.status] ?? j.status,
        schedule: j.schedule,
        last: lastRun(j, nowMs),
        cause: cause(j),
      } satisfies JobRow;
    });
}

/** "25 jobs: 16 on time, 8 not due now, 1 failing." Counted from the rows, not from the server's totals. */
export function jobsSummary(view: PublicJobsView | null | undefined): string {
  const jobs = Array.isArray(view?.jobs) ? view!.jobs : [];
  if (jobs.length === 0) return "no job is registered";
  const failing = jobs.filter((j) => rank(j) === 0).length;
  const onTime = jobs.filter((j) => j.status === "ok").length;
  const off = jobs.filter((j) => j.status === "not_due" && j.disabled).length;
  const idle = jobs.length - failing - onTime - off;
  const parts = [`${onTime} on time`];
  if (idle > 0) parts.push(`${idle} not due now`);
  if (off > 0) parts.push(`${off} switched off`);
  parts.push(failing === 0 ? "none failing" : `${failing} failing`);
  return `${jobs.length} ${jobs.length === 1 ? "job" : "jobs"}: ${parts.join(", ")}`;
}
