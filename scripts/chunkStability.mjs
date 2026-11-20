// chunkStability.mjs — DEPLOY-STABLE VENDOR CHUNKS (map speed, 2026-11-19).
//
// A hashed asset is cached for a year (server/static.ts: immutable), so a
// returning visitor only re-downloads what CHANGED NAME. The entry chunk
// changes name on every deploy (it embeds package.json's version). A lazily
// loaded vendor chunk that statically imports the entry chunk therefore
// changes name on every deploy too, although the library inside it is
// byte-identical — the import specifier is part of its content.
//
// Measured before the fix: maplibre-gl's chunk (1.05 MB, 283 KB on the wire,
// 0.40 s to fetch from production, requested on the way to the first map
// frame) began `import{g}from"./index-<hash>.js"` for one 120-byte CommonJS
// helper, and was re-downloaded after each of ~14 deploys a day.
//
// Pure helpers, shared by the visual harness (which checks the real build)
// and scripts/chunkStability.test.mjs.

/** Vendor chunks that must keep their file name across a deploy that does
 *  not change the library: the chunk's name prefix (Rollup names a lazily
 *  imported package's chunk after the package). */
export const STABLE_VENDOR_CHUNKS = ["maplibre-gl"];

/** File names of the entry chunks index.html boots (its module scripts). */
export function entryChunkNames(indexHtml) {
  const out = [];
  const re = /<script\b[^>]*\btype=["']module["'][^>]*\bsrc=["']([^"']+)["']/g;
  let m;
  while ((m = re.exec(String(indexHtml)))) out.push(m[1].split("/").pop());
  return out;
}

/** File names a built chunk statically imports from its own directory. */
export function staticImports(code) {
  const out = new Set();
  const re = /(?:\bfrom|\bimport)\s*["']\.\/([^"']+)["']/g;
  let m;
  while ((m = re.exec(String(code)))) out.add(m[1]);
  return [...out];
}

/**
 * One line per violation: a vendor chunk that would be renamed by a deploy.
 * `chunks` = [{ name, code }] for the built .js assets.
 */
export function vendorChunkViolations({ indexHtml, chunks, vendors = STABLE_VENDOR_CHUNKS }) {
  const entries = new Set(entryChunkNames(indexHtml));
  const out = [];
  if (!entries.size) return ["chunk-stability: no module script found in index.html — cannot tell which chunk is the entry"];
  for (const vendor of vendors) {
    const mine = chunks.filter((c) => c.name.startsWith(`${vendor}-`) && c.name.endsWith(".js"));
    if (!mine.length) { out.push(`chunk-stability: no built chunk named ${vendor}-*.js — the vendor is no longer split out, or the rule needs its new name`); continue; }
    for (const c of mine) {
      const bad = staticImports(c.code).filter((f) => entries.has(f));
      if (bad.length) out.push(`chunk-stability: ${c.name} imports the entry chunk (${bad.join(", ")}) — its name changes on every deploy, so every returning visitor re-downloads the library (vite.config.ts manualChunks keeps shared helpers out of the entry)`);
    }
  }
  return out;
}
