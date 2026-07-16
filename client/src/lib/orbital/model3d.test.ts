// Class-representative 3D forms — ORBITAL O5 slice 2. Pins the honesty
// mapping (catalog class → form, unknowns stay fragments), the geometry
// invariants, and the deterministic builds.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { classForm, classFormNamed, formLabel, buildFormMesh, rotationMat3 } from './model3d.js';

test('classForm: catalog class maps honestly; unknowns are NEVER dressed as spacecraft', () => {
  assert.equal(classForm('PAYLOAD', 'SMALL'), 'cubesat');
  assert.equal(classForm('PAYLOAD', 'MEDIUM'), 'smallsat');
  assert.equal(classForm('PAYLOAD', null), 'smallsat', 'un-bucketed payload takes the middle form');
  assert.equal(classForm('PAYLOAD', 'LARGE'), 'bus');
  assert.equal(classForm('ROCKET BODY', 'LARGE'), 'rocket-body');
  assert.equal(classForm('DEBRIS', 'SMALL'), 'fragment');
  assert.equal(classForm('UNKNOWN', 'LARGE'), 'fragment', 'unknown type = neutral fragment, never a guessed spacecraft');
  assert.equal(classForm(null, null), 'fragment');
});

test('formLabel: every label states derivation AND not-imagery (the honesty caption)', () => {
  for (const k of ['cubesat', 'smallsat', 'bus', 'rocket-body', 'fragment'] as const) {
    const l = formLabel(k);
    assert.ok(/derived/i.test(l), `${k}: label must say derived`);
    assert.ok(/not imagery of this unit/i.test(l), `${k}: label must say it is not imagery of this unit`);
  }
});

test('buildFormMesh: valid triangle soup with unit normals, per-form geometry, deterministic', () => {
  for (const k of ['cubesat', 'smallsat', 'bus', 'rocket-body', 'fragment'] as const) {
    const m = buildFormMesh(k);
    assert.ok(m.vertexCount >= 24, `${k}: has real geometry (${m.vertexCount} verts)`);
    assert.equal(m.vertexCount % 3, 0, `${k}: whole triangles`);
    assert.equal(m.positions.length, m.vertexCount * 3);
    assert.equal(m.normals.length, m.vertexCount * 3);
    assert.equal(m.colors.length, m.vertexCount * 3);
    for (let i = 0; i < m.vertexCount; i++) {
      const nx = m.normals[i * 3], ny = m.normals[i * 3 + 1], nz = m.normals[i * 3 + 2];
      assert.ok(Math.abs(Math.hypot(nx, ny, nz) - 1) < 1e-4, `${k}: unit normal at ${i}`);
      for (const c of [0, 1, 2]) {
        assert.ok(Math.abs(m.positions[i * 3 + c]) <= 2.5, `${k}: vertex inside the view volume`);
      }
    }
    // deterministic: two builds are byte-identical (no randomness allowed)
    const m2 = buildFormMesh(k);
    assert.deepEqual(Array.from(m2.positions), Array.from(m.positions), `${k}: deterministic rebuild`);
  }
});

test('rotationMat3: rotates unit vectors correctly (column-major, Rx·Ry)', () => {
  const apply = (m: Float32Array, v: [number, number, number]): [number, number, number] => [
    m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
    m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
    m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
  ];
  const ident = rotationMat3(0, 0);
  assert.deepEqual(apply(ident, [1, 2, 3]), [1, 2, 3]);
  // yaw 90°: +x → -z (right-handed Y rotation)
  const yaw = rotationMat3(Math.PI / 2, 0);
  const r = apply(yaw, [1, 0, 0]);
  assert.ok(Math.abs(r[0]) < 1e-6 && Math.abs(r[1]) < 1e-6 && Math.abs(r[2] + 1) < 1e-6);
  // pitch 90°: +y → +z
  const pitch = rotationMat3(0, Math.PI / 2);
  const p = apply(pitch, [0, 1, 0]);
  assert.ok(Math.abs(p[0]) < 1e-6 && Math.abs(p[1]) < 1e-6 && Math.abs(p[2] - 1) < 1e-6);
});

test('O6-6 starlink documented-design form: name-keyed, payload-only, honest label, valid mesh', () => {
  assert.equal(classFormNamed('PAYLOAD', 'MEDIUM', 'STARLINK-34089'), 'starlink');
  assert.equal(classFormNamed('PAYLOAD', 'LARGE', 'starlink-1'), 'starlink', 'case-insensitive');
  assert.equal(classFormNamed('DEBRIS', 'SMALL', 'STARLINK-1 DEB'), 'fragment',
    'a name never upgrades a non-payload — debris stays debris');
  assert.equal(classFormNamed('PAYLOAD', 'MEDIUM', 'ONEWEB-0012'), 'smallsat', 'others fall through to class');
  assert.equal(classFormNamed('PAYLOAD', 'MEDIUM', null), 'smallsat');
  const label = formLabel('starlink');
  assert.ok(/[Dd]ocumented/.test(label) && /public specifications/.test(label) && /not imagery/.test(label),
    'label names the derivation and disclaims imagery');
  const mesh = buildFormMesh('starlink');
  assert.ok(mesh.vertexCount >= 36, 'a real mesh');
  assert.equal(mesh.positions.length, mesh.vertexCount * 3);
  let maxAbs = 0;
  for (const v of mesh.positions) maxAbs = Math.max(maxAbs, Math.abs(v));
  assert.ok(maxAbs <= 1.21, 'fits the form envelope');
  // deterministic (rebuild identical)
  assert.deepEqual([...buildFormMesh('starlink').positions], [...mesh.positions]);
});
