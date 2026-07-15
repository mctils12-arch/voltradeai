// EARTH TWIN O5-3b — NASA spacecraft GLB → compact .vtm mesh asset.
//
// Regeneration tool (run manually, output is committed):
//   node scripts/earthtwin_real_mesh.mjs <model.glb> <out.vtm> [gridRes]
//
// Source asset: NASA's official ISS model (public domain, "ISS_stationary.glb",
// 42MB, 247k tris) — https://science.nasa.gov/resource/international-space-station-3d-model/
// This tool turns it into a small committed binary the globe renderer can
// draw at focused-object size (~72-150px): bake the node hierarchy, replace
// each texture with its true average color (sampled from the real PNG, not
// invented), vertex-cluster decimate, quantize, write indexed .vtm.
//
// .vtm v1 layout (little-endian):
//   0  magic "VTM1"
//   4  u32 vertexCount (<= 65535 — u16 indices)
//   8  u32 triCount
//   12 f32x3 bbox min, f32x3 bbox max   (model space, fits +-1.2 half-extent)
//   36 u16x3 * V   positions (normalized 0..65535 within bbox)
//   .. i8x3  * V   normals   (unit * 127)
//   .. u8x3  * V   colors    (sRGB-ish 0..255)
//   .. u16x3 * T   indices
// Offsets stay 2-byte aligned for every u16 view: 36 + 6V + 3V + 3V = 36+12V.
//
// Decimation is VERTEX CLUSTERING on a uniform grid. Cluster key includes the
// quantized material color AND the normal octant, so differently-colored parts
// never blend and the two faces of a thin solar panel never collapse into one
// (which back-face culling would then hide from one side).

import fs from "node:fs";
import zlib from "node:zlib";
import pngjs from "pngjs";
const { PNG } = pngjs;

const [, , glbPath, outPath, gridResArg] = process.argv;
if (!glbPath || !outPath) {
  console.error("usage: node scripts/earthtwin_real_mesh.mjs <in.glb> <out.vtm> [gridRes=176]");
  process.exit(1);
}
const GRID_RES = Number(gridResArg) || 176;
const HALF_EXTENT = 1.2; // model3d.ts contract: forms fit ~[-1.2, 1.2]

// ---- GLB parse (glTF 2.0, no extensions — verified for this asset) --------
const glb = fs.readFileSync(glbPath);
if (glb.toString("ascii", 0, 4) !== "glTF") throw new Error("not a GLB");
const jsonLen = glb.readUInt32LE(12);
const gltf = JSON.parse(glb.toString("utf8", 20, 20 + jsonLen));
if (gltf.extensionsRequired?.length) {
  throw new Error("GLB needs extensions this tool does not implement: " + gltf.extensionsRequired);
}
const binOff = 20 + jsonLen + 8;
const bin = glb.subarray(binOff, binOff + glb.readUInt32LE(20 + jsonLen));

function accessorArray(idx) {
  const acc = gltf.accessors[idx];
  const bv = gltf.bufferViews[acc.bufferView];
  const compCount = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[acc.type];
  const byteOff = (bv.byteOffset || 0) + (acc.byteOffset || 0);
  const Ctor = { 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array }[acc.componentType];
  if (bv.byteStride && bv.byteStride !== compCount * Ctor.BYTES_PER_ELEMENT) {
    throw new Error("interleaved bufferView not supported");
  }
  // copy so alignment is guaranteed
  const bytes = bin.subarray(byteOff, byteOff + acc.count * compCount * Ctor.BYTES_PER_ELEMENT);
  return new Ctor(new Uint8Array(bytes).buffer, 0, acc.count * compCount);
}

// ---- material -> honest color, sampled from the REAL textures --------------
// Per-vertex color = baseColorFactor * the actual texel at that vertex's UV
// (nearest texel, REPEAT wrap). This keeps the true spatial coloring of the
// source asset (dark solar cells, white modules, gold foil, baked AO) instead
// of washing every part to one average — every value still comes straight
// from NASA's textures, nothing invented.
const texPixelCache = new Map();
function texturePixels(texIdx) {
  let t = texPixelCache.get(texIdx);
  if (!t) {
    const img = gltf.images[gltf.textures[texIdx].source];
    const bv = gltf.bufferViews[img.bufferView];
    const png = PNG.sync.read(Buffer.from(bin.subarray(bv.byteOffset || 0, (bv.byteOffset || 0) + bv.byteLength)));
    t = { w: png.width, h: png.height, data: png.data };
    texPixelCache.set(texIdx, t);
  }
  return t;
}
const wrap = (v) => v - Math.floor(v); // REPEAT

/** Returns a per-vertex color sampler for the primitive. */
function primitiveSampler(prim) {
  const m = prim.material != null ? gltf.materials[prim.material] : null;
  const pbr = (m && m.pbrMetallicRoughness) || {};
  const factor = pbr.baseColorFactor || [1, 1, 1, 1];
  const flat = [Math.min(1, factor[0]), Math.min(1, factor[1]), Math.min(1, factor[2])];
  if (!pbr.baseColorTexture || prim.attributes[`TEXCOORD_${pbr.baseColorTexture.texCoord || 0}`] == null) {
    return { uv: null, sample: () => (m ? flat : [0.6, 0.6, 0.6]) };
  }
  const tex = texturePixels(pbr.baseColorTexture.index);
  const uv = accessorArray(prim.attributes[`TEXCOORD_${pbr.baseColorTexture.texCoord || 0}`]);
  return {
    uv,
    sample: (vi) => {
      const x = Math.min(tex.w - 1, Math.floor(wrap(uv[vi * 2]) * tex.w));
      const y = Math.min(tex.h - 1, Math.floor(wrap(uv[vi * 2 + 1]) * tex.h));
      const i = (y * tex.w + x) * 4;
      return [flat[0] * tex.data[i] / 255, flat[1] * tex.data[i + 1] / 255, flat[2] * tex.data[i + 2] / 255];
    },
  };
}

// ---- node hierarchy -> world transforms (translation + rotation only) -----
function quatToMat3(q) {
  const [x, y, z, w] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
    2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
    2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y),
  ]; // column-major
}
const IDENT = [1, 0, 0, 0, 1, 0, 0, 0, 1];
function mulMat3(a, b) {
  const o = new Array(9);
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) {
    o[c * 3 + r] = a[r] * b[c * 3] + a[3 + r] * b[c * 3 + 1] + a[6 + r] * b[c * 3 + 2];
  }
  return o;
}
function applyMat3(m, x, y, z) {
  return [m[0] * x + m[3] * y + m[6] * z, m[1] * x + m[4] * y + m[7] * z, m[2] * x + m[5] * y + m[8] * z];
}

// world transform per node: {rot, trans}
const world = new Map();
function walk(nodeIdx, parentRot, parentNor, parentTrans) {
  const n = gltf.nodes[nodeIdx];
  if (n.matrix) throw new Error("node matrix not supported (verified absent in the supported assets)");
  const localRot = n.rotation ? quatToMat3(n.rotation) : IDENT;
  const sc = n.scale || [1, 1, 1];
  // local linear part M = R*S; normal matrix (M^-1)^T = R*diag(1/s)
  const localM = mulMat3(localRot, [sc[0], 0, 0, 0, sc[1], 0, 0, 0, sc[2]]);
  const localN = mulMat3(localRot, [1 / sc[0], 0, 0, 0, 1 / sc[1], 0, 0, 0, 1 / sc[2]]);
  const t = n.translation || [0, 0, 0];
  const rt = applyMat3(parentRot, t[0], t[1], t[2]);
  const rot = mulMat3(parentRot, localM);
  const nor = mulMat3(parentNor, localN);
  const trans = [parentTrans[0] + rt[0], parentTrans[1] + rt[1], parentTrans[2] + rt[2]];
  world.set(nodeIdx, { rot, nor, trans });
  for (const c of n.children || []) walk(c, rot, nor, trans);
}
for (const root of gltf.scenes[gltf.scene ?? 0].nodes) walk(root, IDENT, IDENT, [0, 0, 0]);

// ---- gather world-space triangles ------------------------------------------
let srcTris = 0;
const groups = []; // {pos: Float32Array-ish arrays, idx, color, rot}
for (const [nodeIdx, xf] of world) {
  const n = gltf.nodes[nodeIdx];
  if (n.mesh == null) continue;
  for (const prim of gltf.meshes[n.mesh].primitives) {
    if ((prim.mode ?? 4) !== 4 || prim.indices == null) continue;
    const pos = accessorArray(prim.attributes.POSITION);
    const nor = prim.attributes.NORMAL != null ? accessorArray(prim.attributes.NORMAL) : null;
    const idx = accessorArray(prim.indices);
    srcTris += idx.length / 3;
    groups.push({ pos, nor, idx, sampler: primitiveSampler(prim), matKey: prim.material ?? -1, xf });
  }
}

// pass 1: world bbox
const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
for (const g of groups) {
  for (let i = 0; i < g.pos.length; i += 3) {
    const p = applyMat3(g.xf.rot, g.pos[i], g.pos[i + 1], g.pos[i + 2]);
    for (let k = 0; k < 3; k++) {
      const v = p[k] + g.xf.trans[k];
      if (v < mn[k]) mn[k] = v;
      if (v > mx[k]) mx[k] = v;
    }
  }
}
const center = [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2];
const span = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]);
const scale = (2 * HALF_EXTENT) / span; // uniform: aspect preserved

// pass 2: cluster
const cell = (2 * HALF_EXTENT) / GRID_RES;
const clusters = new Map(); // key -> {sx,sy,sz,n, nx,ny,nz, color, id}
const cornerIds = []; // per group: cluster id per source vertex per triangle corner
for (const g of groups) {
  const vertCluster = new Int32Array(g.pos.length / 3).fill(-1);
  const norOf = (vi) => {
    if (!g.nor) return [0, 0, 1];
    return applyMat3(g.xf.nor, g.nor[vi * 3], g.nor[vi * 3 + 1], g.nor[vi * 3 + 2]);
  };
  for (let vi = 0; vi < g.pos.length / 3; vi++) {
    const pw = applyMat3(g.xf.rot, g.pos[vi * 3], g.pos[vi * 3 + 1], g.pos[vi * 3 + 2]);
    const x = (pw[0] + g.xf.trans[0] - center[0]) * scale;
    const y = (pw[1] + g.xf.trans[1] - center[1]) * scale;
    const z = (pw[2] + g.xf.trans[2] - center[2]) * scale;
    const nrm = norOf(vi);
    const oct = (nrm[0] >= 0 ? 4 : 0) | (nrm[1] >= 0 ? 2 : 0) | (nrm[2] >= 0 ? 1 : 0);
    const gx = Math.min(GRID_RES - 1, Math.max(0, Math.floor((x + HALF_EXTENT) / cell)));
    const gy = Math.min(GRID_RES - 1, Math.max(0, Math.floor((y + HALF_EXTENT) / cell)));
    const gz = Math.min(GRID_RES - 1, Math.max(0, Math.floor((z + HALF_EXTENT) / cell)));
    const key = `${gx},${gy},${gz},${g.matKey},${oct}`;
    let c = clusters.get(key);
    if (!c) {
      c = { sx: 0, sy: 0, sz: 0, n: 0, nx: 0, ny: 0, nz: 0, cr: 0, cg: 0, cb: 0, id: clusters.size };
      clusters.set(key, c);
    }
    const col = g.sampler.sample(vi);
    c.sx += x; c.sy += y; c.sz += z; c.n++;
    c.nx += nrm[0]; c.ny += nrm[1]; c.nz += nrm[2];
    c.cr += col[0]; c.cg += col[1]; c.cb += col[2];
    vertCluster[vi] = c.id;
  }
  cornerIds.push(vertCluster);
}

// pass 3: triangles between distinct clusters, deduped
const triSet = new Set();
const tris = [];
groups.forEach((g, gi) => {
  const vc = cornerIds[gi];
  for (let t = 0; t < g.idx.length; t += 3) {
    const a = vc[g.idx[t]], b = vc[g.idx[t + 1]], c = vc[g.idx[t + 2]];
    if (a === b || b === c || a === c) continue;
    // dedupe on the unordered set, keep the original winding
    const key = a < b ? (b < c ? `${a},${b},${c}` : a < c ? `${a},${c},${b}` : `${c},${a},${b}`)
      : (a < c ? `${b},${a},${c}` : b < c ? `${b},${c},${a}` : `${c},${b},${a}`);
    if (triSet.has(key)) continue;
    triSet.add(key);
    tris.push(a, b, c);
  }
});

const V = clusters.size, T = tris.length / 3;
if (V > 65535) throw new Error(`vertexCount ${V} exceeds u16 indices — lower gridRes`);

// ---- quantize + write ------------------------------------------------------
const verts = new Array(V);
for (const c of clusters.values()) verts[c.id] = c;
const bmn = [Infinity, Infinity, Infinity], bmx = [-Infinity, -Infinity, -Infinity];
for (const c of verts) {
  const p = [c.sx / c.n, c.sy / c.n, c.sz / c.n];
  c.p = p;
  for (let k = 0; k < 3; k++) { if (p[k] < bmn[k]) bmn[k] = p[k]; if (p[k] > bmx[k]) bmx[k] = p[k]; }
}
const buf = Buffer.alloc(36 + 12 * V + 6 * T);
buf.write("VTM1", 0, "ascii");
buf.writeUInt32LE(V, 4);
buf.writeUInt32LE(T, 8);
for (let k = 0; k < 3; k++) buf.writeFloatLE(bmn[k], 12 + k * 4);
for (let k = 0; k < 3; k++) buf.writeFloatLE(bmx[k], 24 + k * 4);
let o = 36;
for (const c of verts) {
  for (let k = 0; k < 3; k++) {
    const t = (c.p[k] - bmn[k]) / ((bmx[k] - bmn[k]) || 1);
    buf.writeUInt16LE(Math.round(t * 65535), o); o += 2;
  }
}
for (const c of verts) {
  let len = Math.hypot(c.nx, c.ny, c.nz);
  let n = len > 1e-6 ? [c.nx / len, c.ny / len, c.nz / len] : [0, 0, 1];
  for (let k = 0; k < 3; k++) { buf.writeInt8(Math.round(n[k] * 127), o); o += 1; }
}
for (const c of verts) {
  for (const v of [c.cr, c.cg, c.cb]) { buf.writeUInt8(Math.min(255, Math.round((v / c.n) * 255)), o); o += 1; }
}
for (const i of tris) { buf.writeUInt16LE(i, o); o += 2; }
fs.writeFileSync(outPath, buf);

const gz = zlib.gzipSync(buf).length;
console.log(JSON.stringify({
  source: { tris: srcTris, bytes: glb.length },
  out: { verts: V, tris: T, bytes: buf.length, gzipBytes: gz, gridRes: GRID_RES },
  spanMeters: span,
}, null, 2));
