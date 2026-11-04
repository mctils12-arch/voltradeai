// CI CHECKOUT BUDGET (2026-11-04, human-approved; research/wishlist.md "CI
// CHECKOUT BUDGET"). The `changes` job decides which heavy jobs run, so every
// PR and every main push passes through it. It checks out with
// `fetch-depth: 0` — the full history of every branch (~960 refs, a ~1.26 GiB
// pack) — under a 5-minute cap. On 2026-11-03 GitHub's server-side object
// enumeration alone took ~126 s and 6 of 7 runs were cancelled inside
// actions/checkout before any test ran: nothing could merge or deploy.
//
// The fix gave the job headroom (15 min). This ratchet keeps the budget and the
// fetch depth from drifting apart again: a full-history checkout must carry at
// least FULL_FETCH_MIN_TIMEOUT minutes; a shallower checkout may use less.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const FULL_FETCH_MIN_TIMEOUT = 15;

interface Job { timeout: number | null; fullHistoryCheckout: boolean }

/** The workflow's jobs, read structurally from the text (no YAML dependency:
 *  the repo declares none). A job is a 2-space-indented key under `jobs:`;
 *  comments are dropped first so prose about a setting never counts as one. */
function ciJobs(): Record<string, Job> {
  const lines = fs.readFileSync(path.join(repoRoot, ".github/workflows/ci.yml"), "utf8")
    .split("\n").map((l) => l.replace(/\s+#.*$/, "")).filter((l) => !/^\s*#/.test(l));
  const start = lines.findIndex((l) => /^jobs:\s*$/.test(l));
  assert.ok(start >= 0, "ci.yml must define jobs");
  const jobs: Record<string, Job> = {};
  let name: string | null = null;
  let block: string[] = [];
  const flush = (): void => {
    if (!name) return;
    const text = block.join("\n");
    const t = /^ {4}timeout-minutes:\s*(\d+)\s*$/m.exec(text);
    // a checkout step whose `with:` sets fetch-depth: 0 (the step's own indentation, before the next step)
    const full = /-\s+uses:\s*actions\/checkout@[^\n]*\n(?:[ \t]+[^-\s][^\n]*\n)*?[ \t]+fetch-depth:\s*0\b/.test(text + "\n");
    jobs[name] = { timeout: t ? Number(t[1]) : null, fullHistoryCheckout: full };
  };
  for (const l of lines.slice(start + 1)) {
    if (/^\S/.test(l)) break; // next top-level key
    const m = /^ {2}([A-Za-z0-9_-]+):\s*$/.exec(l);
    if (m) { flush(); name = m[1]; block = []; continue; }
    block.push(l);
  }
  flush();
  return jobs;
}

test("the changes gate exists and still does a full-history checkout (what the budget is sized for)", () => {
  const jobs = ciJobs();
  assert.ok(jobs.changes, "the `changes` path-filter job must exist");
  assert.ok(jobs.changes.fullHistoryCheckout, "the `changes` job checks out with fetch-depth: 0");
});

test("every full-history checkout job has at least a 15-minute budget", () => {
  for (const [name, job] of Object.entries(ciJobs())) {
    if (!job.fullHistoryCheckout) continue;
    const t = job.timeout;
    assert.ok(typeof t === "number" && t >= FULL_FETCH_MIN_TIMEOUT,
      `job '${name}' fetches every branch's history but its timeout is ${t} min (< ${FULL_FETCH_MIN_TIMEOUT}); on 2026-11-03 the fetch alone exceeded 5 min`);
  }
});
