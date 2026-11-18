// EDGAR poll health (server/edgarPollHealth.ts): the three EDGAR pollers'
// failures are counted and served, and the archive's last write is read from
// its day files. Regression: Monday 2026-11-16, when every Form 4 poll threw
// for 11.6 h and nothing a session could read said so.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import {
  EDGAR_POLL_STREAMS, _resetEdgarPollHealthForTests, edgarPollHealthStatus, lastArchiveWriteMs,
  notePollFailed, notePollOk, type EdgarPollHealthRow,
} from "./edgarPollHealth";
import { refreshForm4Cache, _resetForm4CacheForTests } from "./edgarForm4";
import { DIAG_PROBES, sanitizeDiag } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
const T0 = Date.parse("2026-11-16T12:40:00Z");

function tmp(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "edgar-poll-health-"));
}
function row(status: Record<string, unknown>, stream: string): EdgarPollHealthRow {
  const r = (status.streams as EdgarPollHealthRow[]).find((s) => s.stream === stream);
  assert.ok(r, `${stream} in the probe`);
  return r;
}

test("counters: failures accumulate, a success clears the run and keeps the last error", () => {
  _resetEdgarPollHealthForTests(T0);
  const dir = tmp();
  notePollOk("filings", 25, T0);
  for (let i = 1; i <= 9; i++) notePollFailed("filings", new Error("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&count=75&output=atom -> 400"), T0 + i * 900_000);
  let r = row(edgarPollHealthStatus(dir, T0 + 10 * 900_000), "filings");
  assert.equal(r.polls_ok, 1);
  assert.equal(r.polls_failed, 9);
  assert.equal(r.consecutive_failures, 9);
  assert.equal(r.rows_archived, 25);
  assert.equal(r.last_ok_at, "2026-11-16T12:40:00.000Z");
  assert.equal(r.last_failure_at, "2026-11-16T14:55:00.000Z");
  assert.match(String(r.last_error), /count=75.* -> 400$/);
  notePollOk("filings", 0, T0 + 11 * 900_000);
  r = row(edgarPollHealthStatus(dir, T0 + 11 * 900_000), "filings");
  assert.equal(r.consecutive_failures, 0);
  assert.equal(r.polls_failed, 9);
  assert.match(String(r.last_error), /-> 400$/, "the last error stays readable after recovery");
  assert.equal(r.rows_archived, 25, "a poll that archived nothing adds nothing");
});

test("streams are independent, all three are always listed, and odd inputs never throw", () => {
  _resetEdgarPollHealthForTests(T0);
  notePollFailed("filings13f", "plain string", T0);
  notePollFailed("earnings8k", undefined, T0);
  notePollFailed("earnings8k", new Error("x".repeat(1000)), T0);
  notePollOk("filings", Number.NaN, T0);
  const status = edgarPollHealthStatus(tmp(), T0 + 60_000);
  assert.deepEqual((status.streams as EdgarPollHealthRow[]).map((s) => s.stream), [...EDGAR_POLL_STREAMS]);
  assert.equal(row(status, "filings").consecutive_failures, 0);
  assert.equal(row(status, "filings").rows_archived, 0);
  assert.equal(row(status, "filings13f").last_error, "plain string");
  assert.equal(row(status, "earnings8k").consecutive_failures, 2);
  assert.ok(String(row(status, "earnings8k").last_error).length <= 241, "a long error is truncated");
  assert.equal(status.uptime_s, 60);
  assert.equal(status.booted_at, "2026-11-16T12:40:00.000Z");
});

test("last archive write: newest plain day file; gz files, other names and a missing directory are not writes", () => {
  const dir = tmp();
  assert.equal(lastArchiveWriteMs("filings", dir), null, "no directory");
  const d = path.join(dir, "filings");
  fs.mkdirSync(d);
  assert.equal(lastArchiveWriteMs("filings", dir), null, "empty directory");
  const at = (name: string, iso: string) => {
    fs.writeFileSync(path.join(d, name), "{}\n");
    const t = new Date(iso);
    fs.utimesSync(path.join(d, name), t, t);
  };
  at("2026-11-13.jsonl.gz", "2026-11-16T23:00:00Z"); // written by the compactor, later than any poll
  at("notes.txt", "2026-11-16T23:30:00Z");
  assert.equal(lastArchiveWriteMs("filings", dir), null, "only a gz and a stray file");
  at("2026-11-15.jsonl", "2026-11-15T21:00:00Z");
  at("2026-11-16.jsonl", "2026-11-16T12:40:00Z");
  assert.equal(lastArchiveWriteMs("filings", dir), Date.parse("2026-11-16T12:40:00Z"));
  assert.equal(lastArchiveWriteMs("filings13f", dir), null, "another stream's directory is not read");
  const r = row(edgarPollHealthStatus(dir, Date.parse("2026-11-17T00:19:00Z")), "filings");
  assert.equal(r.last_archive_write_at, "2026-11-16T12:40:00.000Z");
});

test("2026-11-16 replayed: a Form 4 poll whose feed answers 400 is counted with its error, and a good poll is counted with its rows", async () => {
  _resetEdgarPollHealthForTests(T0);
  _resetForm4CacheForTests();
  const refuse = (async (url: string) => ({ ok: false, status: 400, text: async () => "", url })) as any;
  const errors: unknown[][] = [];
  const orig = console.error;
  console.error = (...a: unknown[]) => { errors.push(a); };
  try {
    await refreshForm4Cache(25, refuse);
    await refreshForm4Cache(25, refuse);
  } finally {
    console.error = orig;
  }
  const r = row(edgarPollHealthStatus(tmp()), "filings");
  assert.equal(r.polls_failed, 2);
  assert.equal(r.consecutive_failures, 2);
  assert.equal(r.polls_ok, 0);
  assert.match(String(r.last_error), /sec\.gov.* -> 400$/);
  assert.equal(errors.length, 2, "the console line is still written");
  // an empty feed page is a poll that worked
  const emptyFeed = (async () => ({ ok: true, status: 200, text: async () => "<feed></feed>" })) as any;
  await refreshForm4Cache(25, emptyFeed);
  const after = row(edgarPollHealthStatus(tmp()), "filings");
  assert.equal(after.polls_ok, 1);
  assert.equal(after.consecutive_failures, 0);
});

test("the probe is registered and served, passes the diag scrub unchanged, and carries no filing identity", () => {
  _resetEdgarPollHealthForTests(T0);
  notePollFailed("filings", new Error("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&company=&dateb=&owner=include&count=75&output=atom -> 400"), T0);
  const status = edgarPollHealthStatus(tmp(), T0);
  assert.equal(status.probe, "edgar_poll");
  assert.deepEqual(sanitizeDiag(status), status);
  assert.ok(!/\d{10}-\d{2}-\d{6}/.test(JSON.stringify(status)), "no accession in the probe");
  assert.ok((DIAG_PROBES as readonly string[]).includes("edgar_poll"));
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(bot.includes('case "edgar_poll"'));
  for (const [file, stream] of [["edgarForm4.ts", "filings"], ["edgar13f.ts", "filings13f"], ["sec8kEarnings.ts", "earnings8k"]] as const) {
    const src = fs.readFileSync(path.join(here, file), "utf8");
    assert.ok(src.includes(`notePollOk("${stream}", archived)`), `${file} counts its good polls`);
    assert.ok(src.includes(`notePollFailed("${stream}", e)`), `${file} counts its failed polls`);
  }
  const hc = fs.readFileSync(path.join(here, "..", "scripts", "session_health_check.py"), "utf8");
  assert.ok(hc.includes("/api/diag/edgar_poll"), "the session health check reads the probe");
});
