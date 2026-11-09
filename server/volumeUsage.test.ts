// Run: npx tsx --test server/volumeUsage.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { volumeUsage } from "./volumeUsage";
import { DIAG_PROBES } from "./diag";

function tree() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "volusage-"));
  fs.mkdirSync(path.join(root, "a", "b"), { recursive: true });
  fs.mkdirSync(path.join(root, "c"));
  fs.writeFileSync(path.join(root, "a", "b", "x"), Buffer.alloc(300));
  fs.writeFileSync(path.join(root, "a", "y"), Buffer.alloc(100));
  fs.writeFileSync(path.join(root, "c", "z"), Buffer.alloc(50));
  fs.writeFileSync(path.join(root, "top.json"), Buffer.alloc(7));
  return root;
}

test("sums bytes per ancestor directory, sorted largest first", () => {
  const r = volumeUsage(tree());
  assert.equal(r.totalBytes, 457);
  assert.equal(r.totalFiles, 4);
  assert.equal(r.truncated, false);
  const by = Object.fromEntries(r.top.map(n => [n.path, n.bytes]));
  assert.equal(by["a"], 400);
  assert.equal(by["a/b"], 300);
  assert.equal(by["c"], 50);
  assert.equal(by["(root files)"], 7);
  assert.equal(r.top[0].path, "a");
});

test("entry cap truncates and says so", () => {
  const r = volumeUsage(tree(), { maxEntries: 2 });
  assert.equal(r.truncated, true);
});

test("symlinks are not followed", () => {
  const root = tree();
  fs.symlinkSync("/etc", path.join(root, "link"));
  const r = volumeUsage(root);
  assert.equal(r.totalFiles, 4);
});

test("unreadable root yields empty usage, not a throw", () => {
  const r = volumeUsage("/nonexistent-volusage-root");
  assert.equal(r.totalBytes, 0);
});

test("volume probe is whitelisted", () => {
  assert.ok((DIAG_PROBES as readonly string[]).includes("volume"));
});
