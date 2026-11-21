import { test } from "node:test";
import assert from "node:assert/strict";
import { ago, jobRows, jobsSummary, type PublicJob, type PublicJobsView } from "./jobsView";

const NOW = Date.parse("2026-11-21T00:10:00Z");
const job = (id: string, over: Partial<PublicJob> = {}): PublicJob => ({
  id, label: `${id} label`, status: "ok", schedule: "every 15m (always), grace 20m",
  last_start_at: "2026-11-21T00:00:00Z", last_finish_at: "2026-11-21T00:01:00Z", last_status: "ok",
  running: false, disabled: false, recent_errors: 0, error_class: null, ...over,
});
const view = (jobs: PublicJob[]): PublicJobsView => ({
  evaluated_at: "2026-11-21T00:10:00Z", verdict: "ok", counts: { ok: 0, late: 0, missing: 0, errored: 0, not_due: 0 }, jobs,
});

test("ago: seconds, minutes, hours, days; a missing or unreadable time is null, never '0 s ago'", () => {
  assert.equal(ago("2026-11-21T00:09:30Z", NOW), "30 s ago");
  assert.equal(ago("2026-11-21T00:00:00Z", NOW), "10 min ago");
  assert.equal(ago("2026-11-20T09:10:00Z", NOW), "15 h ago");
  assert.equal(ago("2026-11-15T23:26:00Z", NOW), "5 d ago");
  assert.equal(ago(null, NOW), null);
  assert.equal(ago("not a time", NOW), null);
});

test("failing jobs come first, then on time, then not due, then switched off; registry order is kept inside each", () => {
  const rows = jobRows(view([
    job("a_ok"),
    job("b_idle", { status: "not_due" }),
    job("c_off", { status: "not_due", disabled: true }),
    job("d_err", { status: "errored", last_status: "error", error_class: "nonzero_exit" }),
    job("e_ok"),
    job("f_late", { status: "late" }),
    job("g_idle", { status: "not_due" }),
  ]), NOW);
  assert.deepEqual(rows.map((r) => r.id), ["d_err", "f_late", "a_ok", "e_ok", "b_idle", "g_idle", "c_off"]);
  assert.deepEqual(rows.map((r) => r.tone), ["bad", "bad", "ok", "ok", "idle", "idle", "idle"]);
  assert.deepEqual(rows.map((r) => r.state), ["last run failed", "late", "on time", "on time", "not due now", "not due now", "switched off"]);
});

test("an errored job says its cause in words, and an unknown or missing cause says it is not classified", () => {
  const [a, b, c] = jobRows(view([
    job("x", { status: "errored", last_status: "error", error_class: "nonzero_exit", last_start_at: "2026-11-15T23:26:00Z" }),
    job("y", { status: "errored", last_status: "error", error_class: null }),
    job("z", { status: "errored", last_status: "error", error_class: "something_new" as never }),
  ]), NOW);
  assert.equal(a.cause, "The program exited with an error.");
  assert.equal(a.last, "Last started 5 d ago; it failed.");
  assert.equal(b.cause, "The cause is not classified.");
  assert.equal(c.cause, "The cause is not classified.");
});

test("late with a run still open reads differently from late with no run; never ran says so; a healthy job has no cause", () => {
  const [stuck, absent, never, fine] = jobRows(view([
    job("stuck", { status: "late", running: true, last_status: "running", last_finish_at: null }),
    job("absent", { status: "late" }),
    job("never", { status: "missing", last_start_at: null, last_finish_at: null, last_status: null }),
    job("fine"),
  ]), NOW);
  assert.equal(stuck.cause, "The run has been going longer than this job is allowed.");
  assert.equal(stuck.state, "late");
  assert.equal(stuck.last, "Running now, started 10 min ago.");
  assert.equal(absent.cause, "No run started inside its schedule and allowance.");
  assert.equal(never.cause, "It has never run on this server, and its time has passed.");
  assert.equal(never.last, "No run is recorded.");
  assert.equal(fine.cause, null);
  assert.equal(fine.last, "Last started 10 min ago; it finished.");
});

test("a run inside its schedule shows 'running'; a tick that had nothing to do says so", () => {
  const [r, s] = jobRows(view([
    job("r", { running: true, last_status: "running", last_finish_at: null }),
    job("s", { status: "not_due", last_status: "skipped" }),
  ]), NOW);
  assert.equal(r.state, "running");
  assert.equal(r.tone, "ok");
  assert.equal(s.last, "Last started 10 min ago; it had nothing to do.");
});

test("the summary is counted from the rows and its parts add up to the total", () => {
  assert.equal(jobsSummary(view([
    job("a"), job("b"), job("c", { status: "not_due" }), job("d", { status: "not_due", disabled: true }),
    job("e", { status: "errored", error_class: "timeout" }),
  ])), "5 jobs: 2 on time, 1 not due now, 1 switched off, 1 failing");
  assert.equal(jobsSummary(view([job("a")])), "1 job: 1 on time, none failing");
  assert.equal(jobsSummary(view([])), "no job is registered");
  assert.equal(jobsSummary(null), "no job is registered");
});

test("no payload and a payload with no list draw nothing and do not throw", () => {
  assert.deepEqual(jobRows(null, NOW), []);
  assert.deepEqual(jobRows({ evaluated_at: "", verdict: "ok", counts: {} as never, jobs: undefined as never }, NOW), []);
});
