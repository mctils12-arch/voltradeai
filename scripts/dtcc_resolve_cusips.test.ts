import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { cusipForRow, readArchivedCusips } from "./dtcc_resolve_cusips";

test("cusipForRow: CUSIP-sourced rows pass the identifier through unchanged", () => {
  assert.equal(cusipForRow({ underlierId: "037833100", underlierIdSource: "CUSIP" }), "037833100");
});

test("cusipForRow: ISIN-sourced rows derive the embedded CUSIP", () => {
  assert.equal(cusipForRow({ underlierId: "US0378331005", underlierIdSource: "ISIN" }), "037833100");
});

test("cusipForRow: an unknown source yields null (never a guessed identifier)", () => {
  assert.equal(cusipForRow({ underlierId: "X", underlierIdSource: "OTHER" }), null);
});

test("readArchivedCusips: dedupes across multiple gz files, normalizes CUSIP+ISIN to one set, skips malformed lines", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dtcc-archive-"));
  const day1 = [
    JSON.stringify({ underlierId: "037833100", underlierIdSource: "CUSIP" }),
    JSON.stringify({ underlierId: "US0378331005", underlierIdSource: "ISIN" }), // same underlier as above via ISIN
    "not json {{{",
  ].join("\n");
  const day2 = [
    JSON.stringify({ underlierId: "594918104", underlierIdSource: "CUSIP" }),
  ].join("\n");
  fs.writeFileSync(path.join(dir, "2026-08-21.jsonl.gz"), zlib.gzipSync(day1));
  fs.writeFileSync(path.join(dir, "2026-08-22.jsonl"), day2);
  const cusips = readArchivedCusips(dir).sort();
  assert.deepEqual(cusips, ["037833100", "594918104"]);
});

test("readArchivedCusips: a missing directory returns an empty list, not a throw", () => {
  assert.deepEqual(readArchivedCusips("/nonexistent/path/xyz"), []);
});
