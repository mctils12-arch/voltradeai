// Real spacecraft models — EARTH TWIN O5-3b (the charter's "real
// photo-derived models … replace the representative forms ONLY where
// verifiable public assets exist", model3d.ts header).
//
// The registry below maps a NORAD id to a committed .vtm asset produced by
// scripts/earthtwin_real_mesh.mjs from an official public-domain source model
// (Draco-compressed NASA GLBs are first decoded offline by
// scripts/earthtwin_glb_decompress.mjs; provenance JSON sits next to each
// asset in client/public/models/). The mesh is lazy-fetched ONLY when that
// satellite is followed; a fetch failure falls back to the
// class-representative form — the honest tier below.
//
// .vtm v1 (little-endian; writer: scripts/earthtwin_real_mesh.mjs):
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

// TDRS second/third generation — ONE NASA fleet-design model (Boeing
// BSS-601/601HP: two springback mesh single-access antennas) honestly covers
// the six on-orbit units built to that design, so all six NORAD ids share the
// asset. The first-generation TDRS 1–7 (TRW: hexagonal bus, umbrella
// antennas) look different and are deliberately NOT registered — they keep
// the class-representative form. Evidence chain (design match, generation
// membership, SATCAT rows) lives in client/public/models/tdrs-boeing601.json.
const TDRS_BOEING_601: RealModelEntry = {
  url: '/models/tdrs-boeing601.vtm',
  label:
    'NASA model of the second/third-generation TDRS design shared by TDRS 8–13 (public domain source, simplified for display)',
};

export const REAL_MODELS: Record<number, RealModelEntry> = {
  25544: {
    url: '/models/iss-25544.vtm',
    label:
      'NASA model of the International Space Station (public domain source, simplified for display)',
  },
  20580: {
    url: '/models/hubble-20580.vtm',
    label:
      'NASA model of the Hubble Space Telescope (public domain source, simplified for display)',
  },
  27424: {
    url: '/models/aqua-27424.vtm',
    label:
      'NASA model of the Aqua Earth-observing satellite (public domain source, simplified for display)',
  },
  26388: TDRS_BOEING_601, // TDRS 8  (2000-034A)
  27389: TDRS_BOEING_601, // TDRS 9  (2002-011A)
  27566: TDRS_BOEING_601, // TDRS 10 (2002-055A)
  39070: TDRS_BOEING_601, // TDRS 11 (2013-004A)
  39504: TDRS_BOEING_601, // TDRS 12 (2014-004A)
  42915: TDRS_BOEING_601, // TDRS 13 (2017-047A)
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
// Keyed by asset URL (not NORAD id) so ids sharing a fleet-design asset —
// the six Boeing-601 TDRS — share one fetch.
const cache = new Map<string, Promise<Mesh | null>>();

export function loadRealModel(noradId: number): Promise<Mesh | null> {
  const entry = REAL_MODELS[noradId];
  if (!entry) return Promise.resolve(null);
  let p = cache.get(entry.url);
  if (!p) {
    p = fetch(entry.url)
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.arrayBuffer();
      })
      .then(decodeVtm)
      .catch(() => {
        cache.delete(entry.url); // retry on the next follow
        return null;
      });
    cache.set(entry.url, p);
  }
  return p;
}
