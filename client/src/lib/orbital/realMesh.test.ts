// Real spacecraft models — EARTH TWIN O5-3b. Pins the .vtm decode contract,
// the registry honesty, and the INTEGRITY of the committed ISS asset against
// its committed provenance meta (the asset ships in client/public/models/).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { REAL_MODELS, decodeVtm, loadRealModel, realModelLabel } from './realMesh.js';

const here = dirname(fileURLToPath(import.meta.url));
const modelsDir = join(here, '..', '..', '..', 'public', 'models');

/** Build a minimal valid .vtm: 3 verts, 1 triangle, known values. */
function tinyVtm(): ArrayBuffer {
  const V = 3, T = 1;
  const buf = Buffer.alloc(36 + 12 * V + 6 * T);
  buf.write('VTM1', 0, 'ascii');
  buf.writeUInt32LE(V, 4);
  buf.writeUInt32LE(T, 8);
  for (const [i, v] of [-1, -1, -1, 1, 1, 1].entries()) buf.writeFloatLE(v, 12 + i * 4);
  // vert 0 at bbox min, vert 1 at bbox max, vert 2 mid
  const q = [0, 0, 0, 65535, 65535, 65535, 32768, 32768, 32768];
  q.forEach((v, i) => buf.writeUInt16LE(v, 36 + i * 2));
  let o = 36 + 6 * V;
  for (const n of [127, 0, 0, 0, 127, 0, 0, 0, 127]) { buf.writeInt8(n, o); o += 1; }
  for (const c of [255, 0, 0, 0, 255, 0, 0, 0, 255]) { buf.writeUInt8(c, o); o += 1; }
  for (const i of [0, 1, 2]) { buf.writeUInt16LE(i, o); o += 2; }
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length);
}

test('decodeVtm: dequantizes into the model3d Mesh contract (triangle soup)', () => {
  const m = decodeVtm(tinyVtm());
  assert.equal(m.vertexCount, 3, 'T*3 soup vertices');
  assert.equal(m.positions.length, 9);
  assert.deepEqual([...m.positions.slice(0, 3)], [-1, -1, -1], 'bbox min corner exact');
  assert.deepEqual([...m.positions.slice(3, 6)], [1, 1, 1], 'bbox max corner exact');
  for (const v of m.positions.slice(6, 9)) assert.ok(Math.abs(v) < 0.001, 'midpoint ~0');
  assert.equal(m.normals[0], 1, 'i8 127 -> 1.0');
  assert.equal(m.colors[0], 1, 'u8 255 -> 1.0');
  assert.equal(m.colors[1], 0);
});

test('decodeVtm: malformed input throws, never returns junk', () => {
  assert.throws(() => decodeVtm(new ArrayBuffer(10)), /truncated header/);
  const bad = tinyVtm();
  new Uint8Array(bad)[0] = 88; // corrupt magic
  assert.throws(() => decodeVtm(bad), /bad magic/);
  assert.throws(() => decodeVtm(tinyVtm().slice(0, 40)), /truncated body/);
  const oob = tinyVtm();
  new DataView(oob).setUint16(36 + 12 * 3, 9, true); // index 9 of 3 verts
  assert.throws(() => decodeVtm(oob), /index out of range/);
});

test('registry: ISS present, honest label, unknown ids resolve to null', async () => {
  assert.ok(REAL_MODELS[25544], 'ISS (NORAD 25544) registered');
  assert.ok(REAL_MODELS[25544].url.startsWith('/models/'), 'same-origin committed asset');
  const label = realModelLabel(25544)!;
  assert.ok(/NASA/.test(label), 'label names the source');
  assert.ok(/public domain/i.test(label), 'label carries the license');
  assert.ok(/simplified/i.test(label), 'label admits the decimation — no false fidelity claim');
  assert.equal(realModelLabel(99999), null);
  assert.equal(await loadRealModel(99999), null, 'unregistered id: null without any fetch');
});

test('committed ISS asset decodes and matches its committed provenance meta', () => {
  const meta = JSON.parse(readFileSync(join(modelsDir, 'iss-25544.json'), 'utf8'));
  const raw = readFileSync(join(modelsDir, 'iss-25544.vtm'));
  const mesh = decodeVtm(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.length));
  assert.equal(mesh.vertexCount, meta.processing.tris * 3, 'triangle count matches the meta');
  // model3d contract: forms fit ~[-1.2, 1.2] (modelLayer sizes from this)
  let maxAbs = 0;
  for (const v of mesh.positions) maxAbs = Math.max(maxAbs, Math.abs(v));
  assert.ok(maxAbs <= 1.2 + 1e-3, `fits the MODEL_HALF_EXTENT contract (got ${maxAbs.toFixed(3)})`);
  assert.ok(maxAbs > 1.1, 'actually spans the extent (not accidentally tiny)');
  // colors are real texture samples: expect true spread, not one flat gray
  const seen = new Set<number>();
  for (let i = 0; i < mesh.colors.length; i += 3) {
    seen.add((Math.round(mesh.colors[i] * 7) << 6) | (Math.round(mesh.colors[i + 1] * 7) << 3) | Math.round(mesh.colors[i + 2] * 7));
  }
  assert.ok(seen.size >= 8, `sampled palette has real spread (got ${seen.size} coarse buckets)`);
  assert.equal(meta.source.license.toLowerCase().includes('public domain'), true, 'provenance pinned');
});
