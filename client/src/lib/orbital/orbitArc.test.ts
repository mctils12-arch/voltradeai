// Orbit arc — EARTH TWIN O6-1. Pins the arc sampler's honesty (real SGP4,
// gaps never bridged), the segment builder's split rules, and the arc
// shader contract (same discipline as satLayer/modelLayer/airLayer tests).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sampleOrbitArc, raanColor, ARC_SAMPLES, ARC_GAP } from './orbitArc.js';
import { buildArcVertices, ARC_VERT_SRC, ARC_VERT_STRIDE, ArcLayer } from './arcLayer.js';
import { OCCLUSION_RADIUS } from './occlusion.js';
import type { GpRecord } from './tle.js';

// ISS-like LEO elements (valid, near-earth, ~92.9 min period)
const ISS: GpRecord = {
  noradId: 25544, name: 'ISS (ZARYA)', epoch: '2026-07-15T12:00:00',
  meanMotion: 15.5, ecc: 0.0003, inclination: 51.64, raan: 120.5,
  argp: 30.2, meanAnomaly: 200.1, bstar: 0.00003,
} as any;

test('sampleOrbitArc: full-period real track — plausible LEO altitudes, no invented points', () => {
  const arc = sampleOrbitArc(ISS, Date.parse('2026-07-15T13:00:00Z'));
  assert.ok(arc, 'LEO object samples');
  assert.equal(arc!.length, ARC_SAMPLES * 3);
  let good = 0;
  for (let k = 0; k < ARC_SAMPLES; k++) {
    const alt = arc![k * 3 + 2];
    if (alt === ARC_GAP) continue;
    good++;
    assert.ok(alt > 200_000 && alt < 800_000, `sample ${k}: LEO-plausible altitude (${alt})`);
    assert.ok(arc![k * 3] >= 0 && arc![k * 3] <= 1, 'mercX in range');
    assert.ok(arc![k * 3 + 1] >= 0 && arc![k * 3 + 1] <= 1, 'mercY in range');
  }
  assert.ok(good >= ARC_SAMPLES * 0.75, 'mostly sampleable');
  // the track MOVES (an arc, not a stuck point)
  assert.ok(Math.abs(arc![0] - arc![Math.floor(ARC_SAMPLES / 4) * 3]) > 0.01, 'ground track advances');
});

test('sampleOrbitArc: refuses what SGP4 refuses', () => {
  assert.equal(sampleOrbitArc({ ...ISS, meanMotion: null } as any, Date.now()), null, 'no elements = no arc');
  assert.equal(sampleOrbitArc({ ...ISS, meanMotion: 1.0 } as any, Date.parse('2026-07-15T13:00:00Z')), null,
    'deep-space (needs SDP4) = no arc, never faked');
});

test('buildArcVertices: GL_LINES pairs; gaps and antimeridian jumps break the line', () => {
  const pts = new Float32Array([
    0.10, 0.5, 400_000,
    0.11, 0.5, 401_000,
    0.12, 0.5, ARC_GAP,     // gap — both adjacent segments dropped
    0.13, 0.5, 402_000,
    0.95, 0.5, 403_000,     // 0.13→0.95 jumps the antimeridian — dropped
    0.96, 0.5, 404_000,
  ]);
  const v = buildArcVertices([{ pts, color: [1, 0.8, 0.4, 0.9] }]);
  // surviving segments: (0→1) and (4→5) = 2 segments = 4 vertices
  assert.equal(v.length, 4 * ARC_VERT_STRIDE);
  assert.ok(Math.abs(v[0] - 0.10) < 1e-6, 'first segment start (float32 storage)');
  assert.equal(v[3], 1, 'color r packed per vertex');
  assert.ok(Math.abs(v[2 * ARC_VERT_STRIDE] - 0.95) < 1e-6, 'second surviving segment starts after the jump');
});

test('shader contract: true-altitude projection, GLOBE-guarded fragment-discard cull', () => {
  const src = ARC_VERT_SRC('/* prelude stub */', '#define GLOBE');
  assert.ok(src.includes('projectTileFor3D(a_pos.xy, a_pos.z)'), 'arc vertices at REAL altitude');
  const ifdef = src.indexOf('#ifdef GLOBE');
  const endif = src.indexOf('#endif');
  const outside = src.slice(0, ifdef) + src.slice(endif);
  for (const sym of ['u_projection_clipping_plane', 'u_projection_transition', 'projectToSphere', 'GLOBE_RADIUS']) {
    assert.ok(src.slice(ifdef, endif).includes(sym) && !outside.includes(sym), `${sym} inside the GLOBE guard`);
  }
  assert.ok(src.includes((OCCLUSION_RADIUS * OCCLUSION_RADIUS).toFixed(6)), 'cull radius² matches ./occlusion');
  assert.ok(src.includes('v_cull = 1.0'), 'far side flags the fragment stage (discard), never snaps a vertex');
});

test('raanColor: stable plane buckets, distinct neighbors', () => {
  const c0 = raanColor(10), c1 = raanColor(100), c2 = raanColor(10);
  assert.deepEqual(c0, c2, 'deterministic');
  assert.notDeepEqual(c0, c1, 'different planes read apart');
  assert.deepEqual(raanColor(null), raanColor(0.0001).map((v, i) => (i < 4 ? raanColor(0.0001)[i] : v)), 'null → first bucket, never crashes');
});

test('API smoke (no GL): empty arcs render is a no-op', () => {
  const layer = new ArcLayer();
  assert.equal(layer.getVertexCount(), 0);
  const explodingGl = new Proxy({}, { get() { throw new Error('gl touched with no arcs'); } });
  layer.render(explodingGl as any, {} as any);
  assert.equal(layer.getRenderFailed(), false);
  layer.setArcs([{ pts: new Float32Array([0.1, 0.5, 400_000, 0.11, 0.5, 400_000]), color: [1, 1, 1, 1] }]);
  assert.equal(layer.getVertexCount(), 2, 'one segment packed');
  layer.setArcs(null);
  assert.equal(layer.getVertexCount(), 0);
});
