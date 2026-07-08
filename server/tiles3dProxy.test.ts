// 3D-tiles proxy allowlist — the SSRF/open-proxy guard must be airtight.
import { test } from "node:test";
import assert from "node:assert/strict";
import { is3dTilesUrl, withKey, proxyPathFor, ROOT_URL } from "./tiles3dProxy.ts";

test("is3dTilesUrl accepts only real Google 3D-tiles URLs", () => {
  assert.equal(is3dTilesUrl(ROOT_URL), true);
  assert.equal(is3dTilesUrl("https://tile.googleapis.com/v1/3dtiles/datasets/foo/files/bar.glb"), true);
});

test("is3dTilesUrl rejects everything else (SSRF / open-proxy guard)", () => {
  const bad = [
    "http://tile.googleapis.com/v1/3dtiles/root.json",          // not https
    "https://evil.com/v1/3dtiles/root.json",                    // wrong host
    "https://tile.googleapis.com.evil.com/v1/3dtiles/x",        // host suffix trick
    "https://tile.googleapis.com/v1/other/root.json",           // wrong path
    "https://tile.googleapis.com/../secret",                    // path escape
    "https://metadata.google.internal/",                        // SSRF target
    "http://169.254.169.254/latest/meta-data/",                 // cloud metadata
    "file:///etc/passwd",
    "not a url",
    "",
  ];
  for (const u of bad) assert.equal(is3dTilesUrl(u), false, `must reject: ${u}`);
});

test("withKey appends/replaces the key", () => {
  assert.match(withKey(ROOT_URL, "K"), /[?&]key=K$/);
  assert.match(withKey("https://tile.googleapis.com/v1/3dtiles/x?key=old&a=1", "NEW"), /key=NEW/);
  assert.doesNotMatch(withKey("https://tile.googleapis.com/v1/3dtiles/x?key=old", "NEW"), /key=old/);
});

test("proxyPathFor encodes the google url as a single param", () => {
  const p = proxyPathFor("https://tile.googleapis.com/v1/3dtiles/a?session=s&v=1");
  assert.ok(p.startsWith("/api/data/3dtiles/proxy?u="));
  // the whole google url (incl. its own query) is percent-encoded into one param
  assert.ok(p.includes(encodeURIComponent("https://tile.googleapis.com/v1/3dtiles/a?session=s&v=1")));
});
