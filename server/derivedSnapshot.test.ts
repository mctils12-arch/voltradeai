import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { saveSnapshot, loadSnapshot, SNAPSHOT_MAX_AGE_MS } from "./derivedSnapshot";

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), "derivedsnap-"));

test("round-trips and keeps the ORIGINAL timestamp (never re-stamped fresh)", () => {
  const d = tmp();
  assert.equal(saveSnapshot("x", { at: 1_000_000, data: { kind: "raw", n: 3 } }, d), true);
  assert.deepEqual(loadSnapshot("x", d, 1_000_000 + 60_000), { at: 1_000_000, data: { kind: "raw", n: 3 } });
});

test("absent, corrupt, malformed, too-old and future snapshots all load as null", () => {
  const d = tmp();
  assert.equal(loadSnapshot("none", d), null);
  fs.writeFileSync(path.join(d, "voltrade_derived_snapshot_bad.json"), "{not json");
  assert.equal(loadSnapshot("bad", d), null);
  fs.writeFileSync(path.join(d, "voltrade_derived_snapshot_shape.json"), JSON.stringify({ at: "x", data: {} }));
  assert.equal(loadSnapshot("shape", d), null);
  saveSnapshot("old", { at: 1000, data: { a: 1 } }, d);
  assert.equal(loadSnapshot("old", d, 1000 + SNAPSHOT_MAX_AGE_MS + 1), null);
  saveSnapshot("fut", { at: 10_000_000, data: { a: 1 } }, d);
  assert.equal(loadSnapshot("fut", d, 1000), null);
});

test("save never throws on an unwritable dir and leaves no tmp files on success", () => {
  assert.equal(saveSnapshot("x", { at: 1, data: {} }, "/etc/hostname/nope"), false);
  const d = tmp();
  saveSnapshot("y", { at: 5, data: { z: 1 } }, d);
  assert.deepEqual(fs.readdirSync(d), ["voltrade_derived_snapshot_y.json"]);
});
