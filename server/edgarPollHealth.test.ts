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

// ── [PIPELINE 2026-11-24, open_questions.md (ai3)] document counts ──────────
import { describeStageError, notePollDocuments, pollDocuments, pollUnreadError, type EdgarPollDocuments } from "./edgarPollHealth";
import { fetchLatestForm4Filings } from "./edgarForm4";

test("describeStageError: the stage and the status; no URL, accession or file name survives", () => {
  const u = "https://www.sec.gov/Archives/edgar/data/1762716/000176271626000003/13F_ATMOS_6_30_26.xml";
  assert.equal(describeStageError("information table", new Error(`${u} -> 404`)), "information table -> 404");
  assert.equal(describeStageError("exhibit", new Error(`${u} -> 503`)), "exhibit -> 503");
  const odd = describeStageError("index.json", new Error(`Unexpected token < in ${u} for 0001762716-26-000003`));
  assert.equal(odd, "index.json: Unexpected token < in <url> for <accession>");
  assert.equal(describeStageError("index.json", "The operation was aborted due to timeout"), "index.json: The operation was aborted due to timeout");
  assert.ok(describeStageError("exhibit", new Error("x".repeat(500))).length <= 120);
});

test("pollUnreadError: only a poll with failures and nothing read", () => {
  const base: EdgarPollDocuments = { at: "", candidates: 0, returned: 0, skipped: 0, failed: 0, first_error: null };
  assert.equal(pollUnreadError("13F filings", null), null);
  assert.equal(pollUnreadError("13F filings", base), null, "an empty page");
  assert.equal(pollUnreadError("13F filings", { ...base, candidates: 3, skipped: 3 }), null, "all skipped");
  assert.equal(pollUnreadError("13F filings", { ...base, candidates: 40, returned: 37, failed: 3, first_error: "index.json -> 404" }), null, "read in part");
  assert.equal(pollUnreadError("13F filings", { ...base, candidates: 40, failed: 40, first_error: "index.json -> 404" })?.message,
    "0 of 40 13F filings could be read: index.json -> 404");
  assert.match(String(pollUnreadError("earnings 8-Ks", { ...base, candidates: 2, failed: 2 })?.message), /no error recorded$/);
});

test("the probe serves each stream's latest document counts, null before a first poll, and they pass the sanitizer whole", () => {
  _resetEdgarPollHealthForTests(T0);
  const dir = tmp();
  assert.equal(row(edgarPollHealthStatus(dir, T0), "filings13f").documents, null);
  const docs: EdgarPollDocuments = { at: "2026-11-24T13:00:00.000Z", candidates: 40, returned: 37, skipped: 0, failed: 3, first_error: "index.json -> 404" };
  notePollDocuments("filings13f", docs);
  docs.failed = 99; // the module keeps its own copy
  assert.equal(pollDocuments("filings13f")?.failed, 3);
  const served = sanitizeDiag(edgarPollHealthStatus(dir, T0)) as Record<string, unknown>;
  assert.deepEqual(row(served, "filings13f").documents, { ...docs, failed: 3 });
  assert.equal(row(served, "earnings8k").documents, null);
  assert.match(String(served.note), /documents is the latest poll alone/);
});

test("Form 4 reports the same counts in the shared shape", async () => {
  _resetEdgarPollHealthForTests(T0);
  const feed = `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
<title>4 - Example Co (0001234567) (Issuer)</title>
<link rel="alternate" type="text/html" href="https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/0001234567-26-000001-index.htm"/>
<summary type="html"> &lt;b&gt;Filed:&lt;/b&gt; 2026-11-24 &lt;b&gt;AccNo:&lt;/b&gt; 0001234567-26-000001 &lt;b&gt;Size:&lt;/b&gt; 5 KB</summary>
<category scheme="https://www.sec.gov/" label="form type" term="4"/>
</entry></feed>`;
  // the listing is refused (403): thrown, not asked a second way, counted failed
  const impl = async (url: string) => url.includes("output=atom")
    ? { ok: true, status: 200, text: async () => feed }
    : { ok: false, status: 403, text: async () => "" };
  assert.deepEqual(await fetchLatestForm4Filings(25, impl as any, 0), []);
  assert.deepEqual({ ...pollDocuments("filings"), at: null },
    { at: null, candidates: 1, returned: 0, skipped: 0, failed: 1, first_error: "index.json -> 403" });
});

// ── [PRODUCT 2026-11-24, open_questions.md (aj2)] the public view ───────────
import { publicEdgarPollView, publicFirstError } from "./edgarPollHealth";

test("publicFirstError: a recorded stage and status pass; any other text is reduced to the stage and 'timed out' or 'failed'", () => {
  assert.equal(publicFirstError(null), null);
  assert.equal(publicFirstError("index.json -> 404"), "index.json -> 404");
  assert.equal(publicFirstError("information table -> 503"), "information table -> 503");
  assert.equal(publicFirstError("<accession>.txt -> 404"), "<accession>.txt -> 404");
  assert.equal(publicFirstError("exhibit: The operation was aborted due to timeout"), "exhibit timed out");
  assert.equal(publicFirstError("index.json: Unexpected token < in <url> for <accession>"), "index.json failed");
  // Form 4 keeps an error with no status as the bare message (no stage): nothing of it is passed on
  assert.equal(publicFirstError("The operation was aborted due to timeout"), "a request failed");
  assert.equal(publicFirstError("fetch failed"), "a request failed");
  assert.equal(publicFirstError("something a remote server wrote, with a name in it"), "a request failed");
});

test("the public view carries counts and times, never last_error, and is wired into the Pipeline health route", () => {
  _resetEdgarPollHealthForTests(T0);
  const dir = tmp();
  notePollFailed("filings13f", new Error("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=13F-HR -> 503"), T0);
  notePollOk("filings", 7, T0);
  notePollDocuments("earnings8k", { at: "2026-11-24T13:00:00.000Z", candidates: 2, returned: 1, skipped: 0, failed: 1, first_error: "exhibit: socket hang up at some.host" });
  const v = publicEdgarPollView(dir, T0 + 60_000);
  assert.deepEqual(v.streams.map((s) => s.stream), [...EDGAR_POLL_STREAMS]);
  assert.equal(v.poll_interval_min, 15);
  assert.equal(v.streams[1].polls_failed, 1);
  assert.equal(v.streams[1].consecutive_failures, 1);
  assert.equal(v.streams[2].documents?.first_error, "exhibit failed");
  const text = JSON.stringify(v);
  assert.ok(!text.includes("sec.gov") && !text.includes("last_error") && !text.includes("some.host"));
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.match(routes, /edgar = publicEdgarPollView\(\)/);
  assert.match(routes, /jobs,\n\s+edgar,/);
});
