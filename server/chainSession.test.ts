// OPTIONS-ARCHIVE SESSION MAPPING (2026-11-06, rule call "options-archive
// day-shift relabel 07-06 → 10-27"). Before the 2026-10-28 capture fix every
// day-file was captured ~00:40 ET and labelled with the capture's UTC date, so
// labels 07-06 … 10-27 hold the PREVIOUS session's quotes (prod: first records
// at 00:01–00:51 ET on the label date; 10-28 on at 16:30–16:34 ET). These
// tests pin the read-time mapping (server/chainSession.ts): which session a
// capture holds, which file a session reads, that nothing on disk changes,
// and that unknown provenance is never re-dated.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import {
  sessionOfCapture, previousTradingDay, isTradingDay, headRecordT, buildSessionIndex, sessionSummary, sessionDerivedDir,
  SESSION_OPEN_ET_MINUTES,
} from "./chainSession";
import { listChainDays, readChainDay, iv30ForDays, derivedDir as ivDerivedDir } from "./chainIv";
import { derivedDir as cspDerivedDir } from "./chainCspEntry";
import { derivedDir as vsDerivedDir } from "./chainVs";
import type { ChainRecord } from "./optionsChainArchive";

const HOL = new Set(["2026-07-03", "2026-09-07", "2026-11-26"]); // the 2026 closures these cases cross
const Z = (iso: string) => Date.parse(iso);

test("sessionOfCapture: a capture before the open (or on a closed day) holds the previous session; after the open, that day", () => {
  assert.equal(sessionOfCapture(Z("2026-07-06T04:51:00Z"), HOL), "2026-07-02", "Mon 00:51 ET after the July 3 closure -> Thu");
  assert.equal(sessionOfCapture(Z("2026-09-08T04:01:00Z"), HOL), "2026-09-04", "Tue 00:01 ET after Labor Day -> Fri");
  assert.equal(sessionOfCapture(Z("2026-10-27T04:37:00Z"), HOL), "2026-10-26", "Tue 00:37 ET (the last pre-fix label) -> Mon");
  assert.equal(sessionOfCapture(Z("2026-10-28T20:33:00Z"), HOL), "2026-10-28", "Wed 16:33 EDT post-close (fixed capture) -> same day");
  assert.equal(sessionOfCapture(Z("2026-11-05T21:32:00Z"), HOL), "2026-11-05", "Thu 16:32 EST -> same day");
  assert.equal(sessionOfCapture(Z("2026-10-31T14:00:00Z"), HOL), "2026-10-30", "Saturday -> Friday");
  assert.equal(sessionOfCapture(Z("2026-11-02T14:29:00Z"), HOL), "2026-10-30", "Mon 09:29 EST, before the open -> Fri");
  assert.equal(sessionOfCapture(Z("2026-11-02T14:30:00Z"), HOL), "2026-11-02", "Mon 09:30 EST, the open -> Mon");
  assert.equal(sessionOfCapture(NaN, HOL), null);
  assert.equal(SESSION_OPEN_ET_MINUTES, 570);
});

test("previousTradingDay / isTradingDay skip weekends and the listed closures", () => {
  assert.equal(previousTradingDay("2026-07-06", HOL), "2026-07-02");
  assert.equal(previousTradingDay("2026-11-27", HOL), "2026-11-25");
  assert.equal(previousTradingDay("2026-10-27", HOL), "2026-10-26");
  assert.equal(isTradingDay("2026-09-07", HOL), false);
  assert.equal(isTradingDay("2026-10-31", HOL), false);
  assert.equal(isTradingDay("2026-10-30", HOL), true);
});

// ── a temp archive shaped like prod around the 2026-10-28 fix ───────────────
const occ = (u: string, exp: string, type: "C" | "P", strike: number) =>
  `${u}${exp.slice(2, 4)}${exp.slice(5, 7)}${exp.slice(8, 10)}${type}${String(Math.round(strike * 1000)).padStart(8, "0")}`;
const rec = (u: string, t: number, exp: string, iv: number): ChainRecord =>
  ({ s: occ(u, exp, "C", 100), u, t, bid: 1, ask: 1.1, bsz: 1, asz: 1, iv, delta: 0.5, oi: null, feed: "indicative" });
const lines = (rs: ChainRecord[]) => rs.map((r) => JSON.stringify(r)).join("\n") + "\n";

function prodLikeArchive(): { base: string; dir: string } {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "chainsession-"));
  const dir = path.join(base, "optionchains");
  fs.mkdirSync(path.join(dir, "_captured"), { recursive: true });
  fs.mkdirSync(path.join(dir, "_superseded"), { recursive: true });
  // pre-fix midnight captures (no sidecar), gzipped like the 2-day lifecycle leaves them
  fs.writeFileSync(path.join(dir, "2026-10-26.jsonl.gz"), zlib.gzipSync(lines([rec("FRI", Z("2026-10-26T04:36:00Z"), "2026-11-22", 0.41)])));
  fs.writeFileSync(path.join(dir, "2026-10-27.jsonl.gz"), zlib.gzipSync(lines([rec("MON", Z("2026-10-27T04:37:00Z"), "2026-11-25", 0.42)])));
  // the fix day: the 16:33 capture is published with its sidecar; the 00:37 file was moved to _superseded/
  fs.writeFileSync(path.join(dir, "2026-10-28.jsonl.gz"), zlib.gzipSync(lines([rec("WED", Z("2026-10-28T20:33:00Z"), "2026-11-27", 0.44)])));
  fs.writeFileSync(path.join(dir, "_captured", "2026-10-28.json"), "{}");
  fs.writeFileSync(path.join(dir, "_superseded", "2026-10-28.jsonl"), lines([rec("TUE", Z("2026-10-28T04:37:00Z"), "2026-11-26", 0.43)]));
  fs.writeFileSync(path.join(dir, "2026-10-29.jsonl"), lines([rec("THU", Z("2026-10-29T20:34:00Z"), "2026-11-28", 0.45)]));
  fs.writeFileSync(path.join(dir, "_captured", "2026-10-29.json"), "{}");
  return { base, dir };
}

const snapshot = (dir: string): Record<string, string> => {
  const out: Record<string, string> = {};
  const walk = (d: string) => {
    for (const f of fs.readdirSync(d)) {
      const p = path.join(d, f);
      if (fs.statSync(p).isDirectory()) walk(p);
      else out[path.relative(dir, p)] = fs.readFileSync(p).toString("base64");
    }
  };
  walk(dir);
  return out;
};

test("pre-fix labels read as the previous session; the 10-27 session comes from _superseded/; post-fix labels unchanged", async () => {
  const { base } = prodLikeArchive();
  assert.deepEqual(listChainDays(base), ["2026-10-23", "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29"]);
  const u = async (day: string) => (await readChainDay(day, base)).map((r) => r.u);
  assert.deepEqual(await u("2026-10-23"), ["FRI"], "label 10-26 (Mon 00:36 ET) holds Friday's close");
  assert.deepEqual(await u("2026-10-26"), ["MON"], "label 10-27 holds Monday's close");
  assert.deepEqual(await u("2026-10-27"), ["TUE"], "the 10-27 session lives only in _superseded/2026-10-28");
  assert.deepEqual(await u("2026-10-28"), ["WED"], "post-fix: label == session");
  assert.deepEqual(await u("2026-10-29"), ["THU"]);
  assert.deepEqual(await readChainDay("2026-10-30", base), []);
  const ix = buildSessionIndex(base);
  assert.equal(ix.sessions.get("2026-10-26")?.basis, "prefix_capture");
  assert.equal(ix.sessions.get("2026-10-27")?.basis, "superseded");
  assert.equal(ix.sessions.get("2026-10-28")?.basis, "captured");
  assert.deepEqual(sessionSummary(base), { sessions: 5, redated: 3, shadowed: 0, first: "2026-10-23", last: "2026-10-29" });
});

test("read time only: indexing and reading never modify, move or delete an archive file", async () => {
  const { base, dir } = prodLikeArchive();
  const before = snapshot(dir);
  listChainDays(base);
  for (const d of listChainDays(base)) await readChainDay(d, base);
  assert.deepEqual(snapshot(dir), before);
});

test("unknown provenance keeps its label: a capture from another UTC date, or no t, is never re-dated", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "chainsession-"));
  const dir = path.join(base, "optionchains");
  fs.mkdirSync(dir, { recursive: true });
  // the existing chain tests' shape: one fixed t (10-01 16:00 ET) under three labels
  for (const d of ["2026-10-01", "2026-10-02", "2026-10-03"]) fs.writeFileSync(path.join(dir, `${d}.jsonl`), lines([rec("AAA", Z("2026-10-01T20:00:00Z"), "2026-10-31", 0.4)]));
  fs.writeFileSync(path.join(dir, "2026-10-05.jsonl"), '{"s":"AAA261105C00100000","u":"AAA"}\n'); // no t
  fs.writeFileSync(path.join(dir, "2026-10-06.jsonl"), "not json\n");
  assert.deepEqual(listChainDays(base), ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-05", "2026-10-06"]);
  const ix = buildSessionIndex(base);
  assert.equal(ix.sessions.get("2026-10-02")?.basis, "label");
  assert.equal(ix.sessions.get("2026-10-05")?.basis, "label");
  assert.equal(ix.sessions.get("2026-10-01")?.basis, "prefix_capture", "same UTC date, after the open: the label is its own session");
});

test("one session claimed twice: a post-fix capture beats a superseded file; among pre-fix files the latest capture wins and the other is reported", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "chainsession-"));
  const dir = path.join(base, "optionchains");
  fs.mkdirSync(path.join(dir, "_captured"), { recursive: true });
  fs.mkdirSync(path.join(dir, "_superseded"), { recursive: true });
  // Labor Day: a holiday-labelled midnight file and Tuesday's midnight file both hold Friday 09-04
  fs.writeFileSync(path.join(dir, "2026-09-07.jsonl.gz"), zlib.gzipSync(lines([rec("HOL", Z("2026-09-07T04:20:00Z"), "2026-10-02", 0.3)])));
  fs.writeFileSync(path.join(dir, "2026-09-08.jsonl.gz"), zlib.gzipSync(lines([rec("TUE", Z("2026-09-08T04:01:00Z"), "2026-10-02", 0.3)])));
  // a post-fix capture and an (odd) superseded evening capture for the same session
  fs.writeFileSync(path.join(dir, "2026-11-02.jsonl"), lines([rec("NEW", Z("2026-11-02T21:31:00Z"), "2026-12-02", 0.3)]));
  fs.writeFileSync(path.join(dir, "_captured", "2026-11-02.json"), "{}");
  fs.writeFileSync(path.join(dir, "_superseded", "2026-11-02.jsonl.gz.1762000000000"), zlib.gzipSync(lines([rec("OLD", Z("2026-11-02T21:00:00Z"), "2026-12-02", 0.3)])));
  const ix = buildSessionIndex(base, new Set(["2026-09-07"]));
  assert.deepEqual([...ix.sessions.keys()].sort(), ["2026-09-04", "2026-11-02"]);
  assert.equal(ix.sessions.get("2026-09-04")?.label, "2026-09-08", "latest capture wins");
  assert.equal(ix.sessions.get("2026-11-02")?.basis, "captured");
  assert.deepEqual(ix.shadowed.map((s) => s.label).sort(), ["2026-09-07", "2026-11-02"]);
});

test("headRecordT reads the first record of plain, gzipped and timestamp-suffixed superseded files", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "chainsession-"));
  const t = Z("2026-10-28T04:37:00Z");
  const big = lines(Array.from({ length: 3000 }, () => rec("BIG", t, "2026-11-26", 0.4))); // > the 64 KiB head
  const plain = path.join(base, "a.jsonl"), gz = path.join(base, "b.jsonl.gz"), sup = path.join(base, "c.jsonl.gz.1762000000000");
  fs.writeFileSync(plain, big);
  fs.writeFileSync(gz, zlib.gzipSync(big));
  fs.writeFileSync(sup, zlib.gzipSync(big));
  assert.ok(fs.statSync(plain).size > 64 * 1024);
  for (const f of [plain, gz, sup]) assert.equal(headRecordT(f), t, f);
  assert.equal(headRecordT(path.join(base, "missing.jsonl")), null);
});

test("derived readers: DTE is counted from the session (the measurable before/after) and caches live in derived/session/<kind>/; old label-keyed caches are not read or touched", async () => {
  const { base, dir } = prodLikeArchive();
  // label 10-27 holds Monday 10-26's quotes; its contract expires 2026-11-25 = 30 days after the SESSION (29 after the label)
  const old = path.join(dir, "derived", "iv30", "2026-10-26.json");
  fs.mkdirSync(path.dirname(old), { recursive: true });
  fs.writeFileSync(old, JSON.stringify([{ day: "2026-10-26", u: "STALE", iv30: 9, method: "interp", dteLo: 29, dteHi: 29, n: 1, expiries: 1 }]));
  const r = await iv30ForDays(["2026-10-26"], base, Z("2026-11-06T15:00:00Z"));
  assert.equal(r.rows.length, 1);
  assert.equal(r.rows[0].u, "MON");
  assert.equal(r.rows[0].dteLo, 30, "DTE from the session date (the label would give 29)");
  assert.equal(r.rows[0].iv30, 0.42);
  assert.equal(JSON.parse(fs.readFileSync(old, "utf8"))[0].u, "STALE", "the old cache is untouched");
  assert.ok(fs.existsSync(path.join(dir, "derived", "session", "iv30", "2026-10-26.json")));
  assert.equal(ivDerivedDir(base), sessionDerivedDir("iv30", base));
  assert.equal(cspDerivedDir(base), sessionDerivedDir("csp_entry", base));
  assert.equal(vsDerivedDir(base), sessionDerivedDir("vs", base));
  assert.ok(sessionDerivedDir("vs", base).endsWith(path.join("optionchains", "derived", "session", "vs")));
});

test("the index follows the directory: a new day appears without a restart", () => {
  const { base, dir } = prodLikeArchive();
  assert.equal(listChainDays(base).length, 5);
  fs.writeFileSync(path.join(dir, "2026-10-30.jsonl"), lines([rec("FRI2", Z("2026-10-30T20:31:00Z"), "2026-11-29", 0.4)]));
  fs.writeFileSync(path.join(dir, "_captured", "2026-10-30.json"), "{}");
  assert.deepEqual(listChainDays(base).slice(-1), ["2026-10-30"]);
});
