// chunkStability — the rule the visual harness applies to the real build
// (scripts/chunkStability.mjs). Lives in the client suite so CI runs it.
// Run: npx tsx --test client/src/lib/chunkStability.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { entryChunkNames, staticImports, vendorChunkViolations, STABLE_VENDOR_CHUNKS } from "../../../scripts/chunkStability.mjs";

const HTML = `<!doctype html><html><head>
<script type="module" crossorigin src="/assets/index-AAAA1111.js"></script>
<link rel="modulepreload" crossorigin href="/assets/cjs-interop-CqkleIqs.js">
<link rel="stylesheet" crossorigin href="/assets/index-BlvMtqig.css"></head><body></body></html>`;

test("entryChunkNames: the module scripts index.html boots, not its preloads or stylesheets", () => {
  assert.deepEqual(entryChunkNames(HTML), ["index-AAAA1111.js"]);
  assert.deepEqual(entryChunkNames("<html></html>"), []);
});

test("staticImports: minified and spaced forms, side-effect imports, each file once", () => {
  assert.deepEqual(staticImports('import{g as B_}from"./index-AAAA1111.js";function O_(){}'), ["index-AAAA1111.js"]);
  assert.deepEqual(staticImports("import { a } from './a-1.js';\nimport './b-2.js';\nimport { c } from './a-1.js';").sort(), ["a-1.js", "b-2.js"]);
  assert.deepEqual(staticImports("const x = 1; // no imports here"), []);
});

test("a vendor chunk importing the entry chunk is a violation (the measured defect)", () => {
  const v = vendorChunkViolations({ indexHtml: HTML, chunks: [
    { name: "index-AAAA1111.js", code: "export function g(e){return e}" },
    { name: "maplibre-gl-SEKs8A_a.js", code: 'import{g as B_}from"./index-AAAA1111.js";var maplibre=1;' },
  ] });
  assert.equal(v.length, 1);
  assert.match(v[0], /maplibre-gl-SEKs8A_a\.js imports the entry chunk \(index-AAAA1111\.js\)/);
});

test("a vendor chunk importing only a helper chunk passes; another index-* chunk is not the entry", () => {
  assert.deepEqual(vendorChunkViolations({ indexHtml: HTML, chunks: [
    { name: "index-AAAA1111.js", code: 'import{g}from"./cjs-interop-CqkleIqs.js";' },
    { name: "cjs-interop-CqkleIqs.js", code: "function e(t){return t}export{e as g};" },
    // pmtiles' own chunk is also called index-<hash>.js: only the chunk index.html boots is the entry
    { name: "maplibre-gl-DXl5kUcc.js", code: 'import{g as B_}from"./cjs-interop-CqkleIqs.js";import"./index-CrSO6uNF.js";' },
  ] }), []);
});

test("the rule cannot pass by default: a missing vendor chunk or a missing entry is reported", () => {
  assert.match(vendorChunkViolations({ indexHtml: HTML, chunks: [{ name: "index-AAAA1111.js", code: "" }] })[0], /no built chunk named maplibre-gl-\*\.js/);
  assert.match(vendorChunkViolations({ indexHtml: "<html></html>", chunks: [] })[0], /no module script found/);
  assert.ok(STABLE_VENDOR_CHUNKS.includes("maplibre-gl"));
});

test("vite.config.ts keeps the CommonJS helper out of the entry chunk; a built client obeys the rule", () => {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
  const cfg = readFileSync(path.join(root, "vite.config.ts"), "utf8");
  assert.match(cfg, /manualChunks\(id\)\s*\{\s*if \(id\.includes\("commonjsHelpers"\)\) return "cjs-interop";/,
    "without this the helper lands in the entry chunk and maplibre-gl's chunk is renamed by every deploy");
  // when a build is present (the harness and CI build before testing), check the real files too
  const assets = path.join(root, "dist", "public", "assets");
  if (!existsSync(assets)) return;
  const chunks = readdirSync(assets).filter((f: string) => f.endsWith(".js")).map((name: string) => ({ name, code: readFileSync(path.join(assets, name), "utf8") }));
  assert.deepEqual(vendorChunkViolations({ indexHtml: readFileSync(path.join(root, "dist", "public", "index.html"), "utf8"), chunks }), []);
});
