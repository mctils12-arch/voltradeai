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

test('TDRS: the Boeing-601 fleet model covers exactly the gen-2/3 units, never gen-1', async () => {
  // Second/third generation (Boeing BSS-601/601HP, springback antennas):
  // the SAME committed asset for all six on-orbit units of that design.
  const gen23 = [26388, 27389, 27566, 39070, 39504, 42915]; // TDRS 8..13
  const urls = new Set(gen23.map((id) => REAL_MODELS[id]?.url));
  assert.deepEqual([...urls], ['/models/tdrs-boeing601.vtm'], 'all six share the one fleet-design asset');
  for (const id of gen23) {
    const label = realModelLabel(id)!;
    assert.ok(/second\/third-generation TDRS design/.test(label),
      `${id}: label claims the DESIGN, not a unit's own imagery`);
    assert.ok(/NASA/.test(label) && /public domain/i.test(label) && /simplified/i.test(label));
  }
  // First-generation TDRS (TRW: hexagonal bus, umbrella antennas) look
  // different — registering them to this model would be dishonest. Pin it.
  for (const gen1 of [13969, 19548, 19883, 21639, 22314, 23613]) {
    assert.equal(realModelLabel(gen1), null, `TDRS gen-1 NORAD ${gen1} must NOT get the Boeing-601 model`);
  }
  // ids sharing an asset share one load promise (cache keyed by URL)
  const a = loadRealModel(26388);
  const b = loadRealModel(42915);
  assert.equal(a, b, 'same asset -> same in-flight promise');
  assert.equal(await a, null, 'no network in the test env: resolves null, never throws');
});

test('Aqua (NORAD 27424) registered with its own single-unit asset', () => {
  assert.equal(REAL_MODELS[27424]?.url, '/models/aqua-27424.vtm');
  const label = realModelLabel(27424)!;
  assert.ok(/Aqua/.test(label) && /NASA/.test(label) && /public domain/i.test(label) && /simplified/i.test(label));
});

test('every committed real-model asset decodes and matches its committed provenance meta', () => {
  const assets = Object.entries(REAL_MODELS);
  assert.ok(assets.length >= 2, 'ISS + Hubble at minimum');
  for (const [norad, entry] of assets) {
    const base = entry.url.split('/').pop()!.replace(/\.vtm$/, '');
    const meta = JSON.parse(readFileSync(join(modelsDir, `${base}.json`), 'utf8'));
    // meta.norad is a number for single-unit assets, an array for a
    // fleet-design asset shared by several units (Boeing-601 TDRS) — every
    // registered id must still be named by its meta.
    assert.ok(
      [meta.norad].flat().map(String).includes(norad),
      `${base}: meta names the registry NORAD id ${norad}`,
    );
    const raw = readFileSync(join(modelsDir, `${base}.vtm`));
    const mesh = decodeVtm(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.length));
    assert.equal(mesh.vertexCount, meta.processing.tris * 3, `${base}: triangle count matches the meta`);
    // model3d contract: forms fit ~[-1.2, 1.2] (modelLayer sizes from this)
    let maxAbs = 0;
    for (const v of mesh.positions) maxAbs = Math.max(maxAbs, Math.abs(v));
    assert.ok(maxAbs <= 1.2 + 1e-3, `${base}: fits the MODEL_HALF_EXTENT contract (got ${maxAbs.toFixed(3)})`);
    assert.ok(maxAbs > 1.1, `${base}: actually spans the extent (not accidentally tiny)`);
    // colors are real texture samples: expect true spread, not one flat gray
    const seen = new Set<number>();
    for (let i = 0; i < mesh.colors.length; i += 3) {
      seen.add((Math.round(mesh.colors[i] * 7) << 6) | (Math.round(mesh.colors[i + 1] * 7) << 3) | Math.round(mesh.colors[i + 2] * 7));
    }
    assert.ok(seen.size >= 8, `${base}: sampled palette has real spread (got ${seen.size} coarse buckets)`);
    assert.equal(meta.source.license.toLowerCase().includes('public domain'), true, `${base}: provenance pinned`);
    assert.ok(/NASA/.test(entry.label) && /public domain/i.test(entry.label) && /simplified/i.test(entry.label),
      `${base}: label names source + license + admits decimation`);
  }
});
