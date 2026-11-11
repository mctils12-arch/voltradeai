// peakCatchUp.test.ts — the ONE-TIME shared-peak catch-up (2026-11-11, KNOWN
// BROKEN #49, the human's wishlist option (a)): when it runs, what it runs, how
// its report becomes an audit line, the marker, and the boot wiring.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import {
  PEAK_CATCHUP_MARKER, decidePeakCatchUp, peakCatchUpCmd, peakCatchUpFlag, peakCatchUpMarkerPath,
  readPeakCatchUpMarker, summarizePeakCatchUp, writePeakCatchUpMarker, type PeakCatchUpInput,
} from "./peakCatchUp";

const LIVE: PeakCatchUpInput = { enabled: true, paper: true, marker: false, nodeKillSwitch: false, nodePeak: 99962.18, equity: 91229.62 };

test("the live case runs once: paper, no marker, Node's kill off, equity below Node's peak", () => {
  const d = decidePeakCatchUp(LIVE);
  assert.equal(d.run, true);
  assert.match(d.reason, /99962\.18/);
});

test("it never runs when switched off, already done, off paper, or while Node's own kill is latched", () => {
  for (const [patch, re] of [
    [{ enabled: false }, /VOLTRADE_PEAK_CATCHUP off/],
    [{ marker: true }, /already done/],
    [{ paper: false }, /paper-only/],
    [{ nodeKillSwitch: true }, /auto-resume path owns/],
  ] as const) {
    const d = decidePeakCatchUp({ ...LIVE, ...patch });
    assert.equal(d.run, false);
    assert.equal(d.retry, false, "nothing to retry");
    assert.match(d.reason, re);
  }
});

test("missing inputs wait for a later attempt; equity above Node's peak is not a catch-up", () => {
  assert.deepEqual([decidePeakCatchUp({ ...LIVE, nodePeak: 0 }).run, decidePeakCatchUp({ ...LIVE, nodePeak: 0 }).retry], [false, true]);
  assert.deepEqual([decidePeakCatchUp({ ...LIVE, equity: null }).run, decidePeakCatchUp({ ...LIVE, equity: null }).retry], [false, true]);
  const up = decidePeakCatchUp({ ...LIVE, equity: 100500 });
  assert.equal(up.run, false);
  assert.equal(up.retry, false);
});

test("the flag: absent = on; a present env var wins", () => {
  assert.equal(peakCatchUpFlag({}).on, true);
  assert.equal(peakCatchUpFlag({}).source, "default");
  for (const v of ["0", "off", "false", "no", ""]) assert.equal(peakCatchUpFlag({ VOLTRADE_PEAK_CATCHUP: v }).on, false, v);
  assert.equal(peakCatchUpFlag({ VOLTRADE_PEAK_CATCHUP: "1" }).on, true);
});

test("the fallback command carries two plain amounts and refuses anything else", () => {
  assert.equal(
    peakCatchUpCmd(99962.18, 91229.62),
    `python3 -c "import json; from paper_resume_sync import catch_up_shared_peak; print(json.dumps(catch_up_shared_peak(target_peak=99962.18, equity=91229.62)))"`,
  );
  for (const bad of [NaN, Infinity, -1, 0]) assert.throws(() => peakCatchUpCmd(bad, 91229.62));
  assert.throws(() => peakCatchUpCmd(99962.18, NaN));
});

test("a successful report becomes one audit line naming the old and new peak and the new trip level", () => {
  const report = {
    ok: true, applied: true,
    shared_peak: { before: 111736.69, after: 99962.18, changed: true },
    risk_kill_switch: { killed: true, kill_reason: "Portfolio DD -20.0% <= -20%", reset: true },
    levels: { peak: 99962.18, kill_equity: 79969.74, drawdown_pct: -8.74 }, notes: [],
  };
  const s = summarizePeakCatchUp(true, report, "daemon");
  assert.equal(s.ok, true);
  assert.match(s.line, /111736\.69 -> 99962\.18 \(written down\)/);
  assert.match(s.line, /was latched, RESET/);
  assert.match(s.line, /trips at 79969\.74, drawdown -8\.74%/);
});

test("a report that is not ok, or a failed call, is never treated as done", () => {
  const notOk = summarizePeakCatchUp(true, { ok: false, shared_peak: { before: 111736.69, after: 111736.69, changed: false },
    risk_kill_switch: { killed: true, reset: false }, levels: {}, notes: ["refused: x"] }, "daemon");
  assert.equal(notOk.ok, false);
  assert.match(notOk.line, /^NOT DONE: .*NOT reset.*refused: x/);
  const failed = summarizePeakCatchUp(false, { error: "boom" }, "subprocess");
  assert.equal(failed.ok, false);
  assert.match(failed.line, /python call failed \(via subprocess\)/);
});

test("the marker round-trips, atomically, at <DATA_DIR>/voltrade_shared_peak_catchup.json", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "peakcu-"));
  const fp = peakCatchUpMarkerPath({ DATA_DIR: dir });
  assert.equal(fp, path.join(dir, PEAK_CATCHUP_MARKER));
  assert.equal(readPeakCatchUpMarker(fp), null);
  writePeakCatchUpMarker(fp, { at: "2026-11-11T21:30:00Z", nodePeak: 99962.18 });
  assert.deepEqual(readPeakCatchUpMarker(fp), { at: "2026-11-11T21:30:00Z", nodePeak: 99962.18 });
  assert.deepEqual(fs.readdirSync(dir), [PEAK_CATCHUP_MARKER], "no tmp file left behind");
});

test("bot.ts wiring: one delayed boot attempt, bounded retries, audit types, marker only on ok", () => {
  const src = fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), "bot.ts"), "utf8");
  assert.match(src, /from "\.\/peakCatchUp"/);
  assert.match(src, /setTimeout\(\(\) => \{ void peakCatchUpLoop\(\); \}, PEAK_CATCHUP_DELAY_MS\)/);
  assert.match(src, /peakCatchUpAttempts < PEAK_CATCHUP_MAX_ATTEMPTS/);
  assert.match(src, /"paper_resume_catch_up"/);
  assert.match(src, /audit\(s\.ok \? "PEAK-CATCH-UP" : "PEAK-CATCH-UP-FAILED"/);
  assert.match(src, /if \(s\.ok\) \{\s*writePeakCatchUpMarker/);
  assert.match(src, /nodeKillSwitch: state\.killSwitch, nodePeak: state\.equityPeak, equity/);
});
