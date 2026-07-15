// Real spacecraft models — EARTH TWIN O5-3b (the charter's "real
// photo-derived models … replace the representative forms ONLY where
// verifiable public assets exist", model3d.ts header).
//
// The registry below maps a NORAD id to a committed .vtm asset produced by
// scripts/earthtwin_iss_mesh.mjs from an official public-domain source model
// (provenance JSON sits next to each asset in client/public/models/). The
// mesh is lazy-fetched ONLY when that satellite is followed; a fetch failure
// falls back to the class-representative form — the honest tier below.
//
// .vtm v1 (little-endian; writer: scripts/earthtwin_iss_mesh.mjs):
//   0  "VTM1"
//   4  u32 vertexCount V (<= 65535)   8  u32 triCount T
//   12 f32x3 bbox min   24 f32x3 bbox max      (model space, +-1.2 extent)
//   36 u16x3*V positions (bbox-normalized) · i8x3*V normals · u8x3*V colors
//   .. u16x3*T indices
// Every u16 view lands on a 2-byte boundary (36 + 12V stays even).

import type { Mesh } from './model3d.js';

export interface RealModelEntry {
  /** asset path under client/public (same-origin fetch). */
  url: string;
  /** honest caption — names the source; shown in place of the
   *  "representative form" disclaimer because this IS the real design. */
  label: string;
}

export const REAL_MODELS: Record<number, RealModelEntry> = {
  25544: {
    url: '/models/iss-25544.vtm',
    label:
      'NASA model of the International Space Station (public domain source, simplified for display)',
  },
};

export function realModelLabel(noradId: number): string | null {
  return REAL_MODELS[noradId]?.label ?? null;
}

/** Decode a .vtm buffer into the triangle-soup Mesh SatModelLayer draws.
 *  Throws on malformed input — callers treat that as "no real model". */
export function decodeVtm(buf: ArrayBuffer): Mesh {
  if (buf.byteLength < 36) throw new Error('vtm: truncated header');
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'VTM1') throw new Error('vtm: bad magic');
  const V = dv.getUint32(4, true);
  const T = dv.getUint32(8, true);
  if (V === 0 || V > 65535 || T === 0) throw new Error('vtm: bad counts');
  if (buf.byteLength < 36 + 12 * V + 6 * T) throw new Error('vtm: truncated body');
  const mn = [dv.getFloat32(12, true), dv.getFloat32(16, true), dv.getFloat32(20, true)];
  const mx = [dv.getFloat32(24, true), dv.getFloat32(28, true), dv.getFloat32(32, true)];
  const posQ = new Uint16Array(buf, 36, V * 3);
  const norQ = new Int8Array(buf, 36 + 6 * V, V * 3);
  const colQ = new Uint8Array(buf, 36 + 9 * V, V * 3);
  const idx = new Uint16Array(buf, 36 + 12 * V, T * 3);
  const positions = new Float32Array(T * 9);
  const normals = new Float32Array(T * 9);
  const colors = new Float32Array(T * 9);
  for (let c = 0; c < T * 3; c++) {
    const v = idx[c];
    if (v >= V) throw new Error('vtm: index out of range');
    for (let k = 0; k < 3; k++) {
      positions[c * 3 + k] = mn[k] + (posQ[v * 3 + k] / 65535) * (mx[k] - mn[k]);
      normals[c * 3 + k] = norQ[v * 3 + k] / 127;
      colors[c * 3 + k] = colQ[v * 3 + k] / 255;
    }
  }
  return { positions, normals, colors, vertexCount: T * 3 };
}

// One in-flight/settled promise per asset: repeated follows never refetch a
// success; a FAILURE clears the slot so the next follow retries the network.
const cache = new Map<number, Promise<Mesh | null>>();

export function loadRealModel(noradId: number): Promise<Mesh | null> {
  const entry = REAL_MODELS[noradId];
  if (!entry) return Promise.resolve(null);
  let p = cache.get(noradId);
  if (!p) {
    p = fetch(entry.url)
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.arrayBuffer();
      })
      .then(decodeVtm)
      .catch(() => {
        cache.delete(noradId); // retry on the next follow
        return null;
      });
    cache.set(noradId, p);
  }
  return p;
}
