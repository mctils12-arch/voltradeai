import { test } from "node:test";
import assert from "node:assert/strict";
import { edgarPollRows, edgarPollSummary, type PublicEdgarPollStream, type PublicEdgarPollView } from "./edgarPollView";

const NOW = Date.parse("2026-11-24T13:10:00Z");
const stream = (id: string, over: Partial<PublicEdgarPollStream> = {}): PublicEdgarPollStream => ({
  stream: id, polls_ok: 38, polls_failed: 0, consecutive_failures: 0,
  last_archive_write_at: "2026-11-24T12:52:00Z",
  documents: { at: "2026-11-24T13:06:00Z", candidates: 25, returned: 25, skipped: 0, failed: 0, first_error: null },
  ...over,
});
const view = (streams: PublicEdgarPollStream[]): PublicEdgarPollView => ({
  read_at: "2026-11-24T13:10:00Z", booted_at: "2026-11-24T03:37:00Z", poll_interval_min: 15, streams,
});

test("a poller that read its whole page: the counts, the archive's last row and the polls since the start, no cause", () => {
  const [r] = edgarPollRows(view([stream("filings")]), NOW);
  assert.equal(r.label, "Form 4: insider trades");
  assert.equal(r.tone, "ok");
  assert.equal(r.state, "reading");
  assert.equal(r.cause, null);
  assert.equal(r.meta, "Latest poll 4 min ago: 25 filings asked for, 25 read. Last new filing archived 18 min ago. Since the server started 10 h ago: 38 polls read, 0 failed.");
});

test("what a poll could not read is said with the first failure; read in part is not a failure, read nothing is", () => {
  const part = stream("filings13f", { documents: { at: "2026-11-24T13:06:00Z", candidates: 40, returned: 37, skipped: 0, failed: 3, first_error: "index.json -> 404" } });
  const none = stream("earnings8k", { polls_failed: 1, consecutive_failures: 1,
    documents: { at: "2026-11-24T13:06:00Z", candidates: 1, returned: 0, skipped: 0, failed: 1, first_error: "exhibit timed out" } });
  const rows = edgarPollRows(view([stream("filings"), part, none]), NOW);
  assert.deepEqual(rows.map((r) => [r.id, r.tone, r.state]), [
    ["earnings8k", "bad", "read nothing"], ["filings13f", "idle", "read in part"], ["filings", "ok", "reading"],
  ]);
  assert.equal(rows[0].cause, "The latest poll could not read 1 of 1 filing. First failure: exhibit timed out.");
  assert.equal(rows[1].cause, "The latest poll could not read 3 of 40 filings. First failure: index.json -> 404.");
  assert.equal(edgarPollSummary(view([stream("filings"), part, none]), NOW), "3 pollers, every 15 min: 1 read nothing, 1 read in part, 1 reading");
});

test("two failed polls in a row is failing, whether the list or the documents failed", () => {
  const list = stream("filings", { polls_ok: 3, polls_failed: 9, consecutive_failures: 9 });
  const [r] = edgarPollRows(view([list]), NOW);
  assert.equal(r.state, "failing");
  assert.equal(r.tone, "bad");
  assert.equal(r.cause, "The last 9 polls could not read EDGAR's list of new filings.");
  assert.match(r.meta, /3 polls read, 9 failed\.$/);
  const docs = stream("filings13f", { polls_failed: 2, consecutive_failures: 2,
    documents: { at: "2026-11-24T13:06:00Z", candidates: 40, returned: 0, skipped: 0, failed: 40, first_error: "index.json -> 404" } });
  assert.equal(edgarPollRows(view([docs]), NOW)[0].cause, "The latest poll could not read 40 of 40 filings. First failure: index.json -> 404.");
  assert.equal(edgarPollRows(view([stream("filings", { polls_failed: 1, consecutive_failures: 1 })]), NOW)[0].state, "reading", "one failed poll is noise");
});

test("skipped filings are said in the stream's own words; an empty list and a server with no poll yet read plainly", () => {
  const skip = stream("earnings8k", { documents: { at: "2026-11-24T13:06:00Z", candidates: 15, returned: 9, skipped: 6, failed: 0, first_error: null } });
  assert.match(edgarPollRows(view([skip]), NOW)[0].meta, /^Latest poll 4 min ago: 15 filings asked for, 9 read, 6 had no press-release exhibit\./);
  const empty = stream("filings13f", { last_archive_write_at: null, documents: { at: "2026-11-24T13:06:00Z", candidates: 0, returned: 0, skipped: 0, failed: 0, first_error: null } });
  assert.equal(edgarPollRows(view([empty]), NOW)[0].meta,
    "Latest poll 4 min ago: no filing of this kind on EDGAR's list. No filing archived in the last two days. Since the server started 10 h ago: 38 polls read, 0 failed.");
  const cold = stream("filings", { polls_ok: 0, polls_failed: 0, documents: null });
  const [r] = edgarPollRows(view([cold]), NOW);
  assert.equal(r.state, "no poll yet");
  assert.equal(r.tone, "idle");
  assert.match(r.meta, /^No poll has finished reading its list since the server started\./);
});

test("odd input never throws and never invents a row", () => {
  assert.deepEqual(edgarPollRows(null, NOW), []);
  assert.deepEqual(edgarPollRows(undefined, NOW), []);
  assert.deepEqual(edgarPollRows({ streams: "no" } as unknown as PublicEdgarPollView, NOW), []);
  assert.equal(edgarPollSummary(null, NOW), "no poller reported");
  const odd = { read_at: "x", booted_at: "x", poll_interval_min: 0, streams: [{ stream: "other", polls_ok: NaN, polls_failed: -1, consecutive_failures: 0, last_archive_write_at: "x", documents: { at: "x", candidates: 2, returned: 2, skipped: 0, failed: 0, first_error: null } }] } as unknown as PublicEdgarPollView;
  const [r] = edgarPollRows(odd, NOW);
  assert.equal(r.label, "other");
  assert.equal(r.state, "no poll yet");
  assert.equal(r.meta, "Latest poll: 2 filings asked for, 2 read. No filing archived in the last two days. Since the server started: 0 polls read, 0 failed.");
  assert.equal(edgarPollSummary(odd, NOW), "1 poller: 1 no poll yet");
});
