import { test } from "node:test";
import assert from "node:assert/strict";
import { addWatch, removeWatch, isWatched, normalizeHex, WATCHLIST_CAP, type WatchedPlane } from "./watchlist";

const w = (hex: string, extra: Partial<WatchedPlane> = {}): WatchedPlane =>
  ({ hex, addedAt: 1, ...extra });

test("normalizeHex: lowercase 6-hex only — junk is rejected, never stored", () => {
  assert.equal(normalizeHex("ABE872"), "abe872");
  assert.equal(normalizeHex(" abe872 "), "abe872");
  assert.equal(normalizeHex("xyz"), null);
  assert.equal(normalizeHex("abe8721"), null);
  assert.equal(normalizeHex(""), null);
  assert.equal(normalizeHex(null), null);
});

test("addWatch: newest first, idempotent by hex with metadata refresh", () => {
  let l = addWatch([], w("abe872", { reg: "N8667D" }));
  l = addWatch(l, w("c01234"));
  assert.deepEqual(l.map((x) => x.hex), ["c01234", "abe872"]);
  // re-adding the same hex refreshes metadata and moves it to the front
  l = addWatch(l, w("ABE872", { reg: "N8667D", callsign: "SWA762" }));
  assert.equal(l.length, 2);
  assert.equal(l[0].hex, "abe872");
  assert.equal(l[0].callsign, "SWA762");
});

test("addWatch: invalid hex is a no-op; cap holds at WATCHLIST_CAP", () => {
  assert.equal(addWatch([], w("nope")).length, 0);
  let l: WatchedPlane[] = [];
  for (let i = 0; i < WATCHLIST_CAP + 10; i++) {
    l = addWatch(l, w((0xa00000 + i).toString(16)));
  }
  assert.equal(l.length, WATCHLIST_CAP);
});

test("removeWatch + isWatched round-trip, case-insensitive", () => {
  let l = addWatch([], w("abe872"));
  assert.equal(isWatched(l, "ABE872"), true);
  l = removeWatch(l, "ABE872");
  assert.equal(isWatched(l, "abe872"), false);
  assert.equal(l.length, 0);
});
