// clientVersion — the release version travels in index.html, not in the
// hashed JS, so a deploy that changes no client file renames no asset.
// Run: npx tsx --test client/src/lib/clientVersion.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readClientVersion, isVersionSkew } from "./clientVersion.ts";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const metaDoc = (content: string | null) => ({
  querySelector: (sel: string) => (sel === 'meta[name="vt-client-version"]' && content !== null ? { getAttribute: (n: string) => (n === "content" ? content : null) } : null),
});

test("readClientVersion: the meta tag's content, trimmed; '' when the page states none", () => {
  assert.equal(readClientVersion(metaDoc("1.0.1472")), "1.0.1472");
  assert.equal(readClientVersion(metaDoc(" 1.0.1473\n")), "1.0.1473");
  assert.equal(readClientVersion(metaDoc(null)), "", "no tag — unknown, never a guess");
  assert.equal(readClientVersion(metaDoc("")), "");
  assert.equal(readClientVersion({ querySelector() { throw new Error("detached document"); } }), "", "a throwing document reads as unknown");
  assert.equal(readClientVersion(), "", "no DOM (node) — unknown");
});

test("isVersionSkew: only two KNOWN, different versions are a skew", () => {
  assert.equal(isVersionSkew("1.0.1473", "1.0.1472"), true, "the server moved on: say so");
  assert.equal(isVersionSkew("1.0.1472", "1.0.1472"), false);
  assert.equal(isVersionSkew(null, "1.0.1472"), false, "a registry without a version proves nothing");
  assert.equal(isVersionSkew("", "1.0.1472"), false);
  assert.equal(isVersionSkew("1.0.1473", ""), false, "an unknown client version is not evidence of an update — no false 'site updated' banner");
});

test("the version is not imported into the bundle: datamap reads lib/clientVersion, the build writes the meta tag", () => {
  const page = readFileSync(path.join(root, "client", "src", "pages", "datamap.tsx"), "utf8");
  assert.doesNotMatch(page, /from\s+["'][^"']*package\.json["']/,
    "importing package.json bakes the version into the entry chunk and renames it (607 KB on the wire) on every deploy");
  assert.match(page, /import \{ CLIENT_VERSION, isVersionSkew \} from "@\/lib\/clientVersion";/);
  // no other client module may reintroduce the import either
  const offenders: string[] = [];
  const walk = (dir: string) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (/\.(ts|tsx)$/.test(e.name) && !/\.test\.ts$/.test(e.name) && /from\s+["'][^"']*package\.json["']/.test(readFileSync(p, "utf8"))) offenders.push(path.relative(root, p));
    }
  };
  walk(path.join(root, "client", "src"));
  assert.deepEqual(offenders, [], "a client module imports package.json");
  const cfg = readFileSync(path.join(root, "vite.config.ts"), "utf8");
  assert.match(cfg, /plugins: \[react\(\), clientVersionMeta\(\)\]/);
  assert.match(cfg, /tag: "meta", attrs: \{ name: "vt-client-version", content: String\(version\) \}, injectTo: "head"/);
});

test("a built client states its version in index.html and in no hashed script", () => {
  const dist = path.join(root, "dist", "public");
  if (!existsSync(path.join(dist, "index.html"))) return; // no build in this checkout — the rules above still hold
  const html = readFileSync(path.join(dist, "index.html"), "utf8");
  const m = html.match(/<meta name="vt-client-version" content="([^"]+)">/);
  if (!m) return; // a build made before this change
  assert.match(m[1], /^\d+\.\d+\.\d+$/);
  const assets = path.join(dist, "assets");
  const carrying = readdirSync(assets).filter((f) => f.endsWith(".js") && readFileSync(path.join(assets, f), "utf8").includes(`"${m[1]}"`));
  assert.deepEqual(carrying, [], "the release version is inside a hashed script: every deploy will rename it");
});
