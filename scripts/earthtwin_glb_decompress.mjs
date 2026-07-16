// EARTH TWIN — offline pre-step for scripts/earthtwin_real_mesh.mjs.
//
// Most current NASA 3D Resources spacecraft GLBs (the Eyes on the Solar
// System exports: TDRS, Aqua, Aura, Landsat, …) ship with
// KHR_draco_mesh_compression geometry and EXT_texture_webp textures.
// earthtwin_real_mesh.mjs deliberately implements plain glTF 2.0 only, so
// this tool converts such a GLB into one it can read:
//
//   1. decode Draco geometry (lossless w.r.t. the quantized source
//      attributes) and write plain uncompressed accessors;
//   2. restore the spec PNG fallback `source` on every webp-extended
//      texture — the NASA exports pair each webp image with a PNG twin of
//      the same name inside the same GLB (the EXT_texture_webp fallback
//      pattern); gltf-transform drops the pairing on write, so it is
//      re-attached here by image name. Nothing is added or recolored:
//      the PNG bytes are NASA's own.
//
// REGENERATION RECIPE (manual, output .vtm is committed — this repo does
// NOT depend on these packages; install them ad hoc OUTSIDE the repo):
//   mkdir -p /tmp/glbtool && cd /tmp/glbtool && npm init -y \
//     && npm install @gltf-transform/core @gltf-transform/extensions draco3dgltf
//   node /path/to/repo/scripts/earthtwin_glb_decompress.mjs in.glb plain.glb
//   node /path/to/repo/scripts/earthtwin_real_mesh.mjs plain.glb out.vtm <grid>
// (run the decompress step from /tmp/glbtool so its node_modules resolve)

import fs from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";

// The ad-hoc deps are resolved from the CURRENT WORKING DIRECTORY (the
// scratch dir of the recipe above), not from this repo — the repo stays
// dependency-free for this manual tool.
let NodeIO, VertexLayout, ALL_EXTENSIONS, draco3d;
try {
  const requireFromCwd = createRequire(join(process.cwd(), "noop.js"));
  ({ NodeIO, VertexLayout } = requireFromCwd("@gltf-transform/core"));
  ({ ALL_EXTENSIONS } = requireFromCwd("@gltf-transform/extensions"));
  draco3d = requireFromCwd("draco3dgltf");
} catch {
  console.error(
    "missing ad-hoc deps — run from a scratch dir with @gltf-transform/core," +
      " @gltf-transform/extensions and draco3dgltf installed (see header recipe)",
  );
  process.exit(1);
}

const [, , inPath, outPath] = process.argv;
if (!inPath || !outPath) {
  console.error("usage: node scripts/earthtwin_glb_decompress.mjs <in.glb> <out.glb>");
  process.exit(1);
}

// ---- 1. draco decode, plain non-interleaved write --------------------------
const io = new NodeIO().setVertexLayout(VertexLayout.SEPARATE).registerExtensions(ALL_EXTENSIONS).registerDependencies({
  "draco3d.decoder": await draco3d.createDecoderModule(),
});
const doc = await io.read(inPath);
for (const ext of doc.getRoot().listExtensionsUsed()) {
  if (ext.extensionName === "KHR_draco_mesh_compression") ext.dispose();
}
await io.write(outPath, doc);

// ---- 2. restore PNG fallback `source` on EXT_texture_webp textures ---------
const glb = fs.readFileSync(outPath);
const jsonLen = glb.readUInt32LE(12);
const gltf = JSON.parse(glb.toString("utf8", 20, 20 + jsonLen));
let patched = 0;
for (const tex of gltf.textures || []) {
  const webpIdx = tex.extensions?.EXT_texture_webp?.source;
  if (webpIdx == null || tex.source != null) continue;
  const name = gltf.images[webpIdx].name;
  const pngIdx = (gltf.images || []).findIndex((im) => im.mimeType === "image/png" && im.name === name);
  if (pngIdx < 0) throw new Error(`no PNG fallback twin for texture image "${name}"`);
  tex.source = pngIdx;
  patched++;
}
if (patched) {
  let json = Buffer.from(JSON.stringify(gltf), "utf8");
  const pad = (4 - (json.length % 4)) % 4;
  json = Buffer.concat([json, Buffer.alloc(pad, 0x20)]); // spaces per GLB spec
  const rest = glb.subarray(20 + jsonLen); // BIN chunk (header + data), untouched
  const head = Buffer.alloc(12);
  head.write("glTF", 0, "ascii");
  head.writeUInt32LE(2, 4);
  head.writeUInt32LE(12 + 8 + json.length + rest.length, 8);
  const jsonHdr = Buffer.alloc(8);
  jsonHdr.writeUInt32LE(json.length, 0);
  jsonHdr.writeUInt32LE(0x4e4f534a, 4); // "JSON"
  fs.writeFileSync(outPath, Buffer.concat([head, jsonHdr, json, rest]));
}
console.log(JSON.stringify({ wrote: outPath, pngFallbacksRestored: patched }));
