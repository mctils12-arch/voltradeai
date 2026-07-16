// Class-representative satellite 3D forms — EARTH TWIN / ORBITAL O5 slice 2
// (human directive: zoom to a satellite → see a 3D rendering of it).
//
// HONESTY TIERS (orbital_program.md FIDELITY BY DESIGN CLASS): a real
// satellite is meters-sized — invisible at true scale from any map zoom —
// so the 3D rendering is a DETAIL-VIEW form. This module builds
// class-REPRESENTATIVE parametric meshes (CubeSat box + wings, small-sat
// bus, large bus + solar arrays, spent rocket stage, debris fragment)
// derived from the object's CATALOGUED class (SATCAT object type + radar-
// cross-section bucket). These are the charter's honest symbolic markers,
// in 3D: every label says "representative form … not imagery of this unit".
// Real photo-derived models (ISS via NASA's public-domain 3D resources,
// documented constellation designs) are the chartered O5-3 upgrade and
// replace these ONLY where verifiable public assets exist. NOTHING here
// ever claims to show a specific unit's actual appearance.
//
// Pure geometry + tiny mat3 math (no WebGL, no DOM) so it is testable;
// components/SatModelView.tsx owns the canvas.

import type { ObjectType, RcsSize } from './tle.js';

export type FormKind = 'cubesat' | 'smallsat' | 'bus' | 'rocket-body' | 'fragment' | 'starlink';

/** Catalog class → representative form. Unknowns stay a neutral fragment —
 *  never dressed up as a spacecraft we can't substantiate. */
export function classForm(objectType: ObjectType | null, rcsSize: RcsSize | null): FormKind {
  if (objectType === 'ROCKET BODY') return 'rocket-body';
  if (objectType === 'DEBRIS') return 'fragment';
  if (objectType === 'PAYLOAD') {
    if (rcsSize === 'SMALL') return 'cubesat';
    if (rcsSize === 'LARGE') return 'bus';
    return 'smallsat'; // MEDIUM or un-bucketed payload: the middle form
  }
  return 'fragment'; // UNKNOWN / uncatalogued type
}

/** Name-aware form: DOCUMENTED-DESIGN tier (charter O6-6) — a constellation
 *  whose bus design is publicly documented gets a design-specific form,
 *  keyed on the catalog NAME (a broadcastable fact). Currently: Starlink
 *  (flat bus + single deployed solar wing — SpaceX's published design).
 *  Everything else falls through to the class form; a name alone never
 *  upgrades a non-payload. */
export function classFormNamed(
  objectType: ObjectType | null,
  rcsSize: RcsSize | null,
  name: string | null | undefined,
): FormKind {
  if (objectType === 'PAYLOAD' && (name || '').toUpperCase().startsWith('STARLINK')) return 'starlink';
  return classForm(objectType, rcsSize);
}

/** Honest caption for the viewer — ALWAYS states derivation + not-imagery. */
export function formLabel(kind: FormKind): string {
  const base =
    kind === 'cubesat' ? 'Representative CubeSat-class form (catalog type + radar cross-section)'
    : kind === 'smallsat' ? 'Representative small-satellite form (catalog class)'
    : kind === 'bus' ? 'Representative large-satellite form — bus + solar arrays (catalog class)'
    : kind === 'rocket-body' ? 'Representative spent rocket stage (catalog type)'
    : kind === 'starlink' ? 'Documented Starlink design — flat bus + single solar wing (public specifications)'
    : 'Representative fragment form (catalog type)';
  return `${base} — derived, not imagery of this unit`;
}

export interface Mesh {
  /** interleaved triangles: xyz per vertex. */
  positions: Float32Array;
  /** per-vertex face normals (flat shading). */
  normals: Float32Array;
  /** per-vertex rgb. */
  colors: Float32Array;
  vertexCount: number;
}

// palette (generic spacecraft materials — deliberately neutral)
const GOLD: RGB = [0.82, 0.66, 0.28]; // MLI foil
const PANEL: RGB = [0.16, 0.25, 0.55]; // solar cell blue
const SILVER: RGB = [0.72, 0.74, 0.78];
const DARK: RGB = [0.28, 0.28, 0.3];
type RGB = [number, number, number];

class MeshBuilder {
  pos: number[] = [];
  nor: number[] = [];
  col: number[] = [];

  tri(a: number[], b: number[], c: number[], color: RGB): void {
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz) || 1;
    nx /= len; ny /= len; nz /= len;
    for (const p of [a, b, c]) {
      this.pos.push(p[0], p[1], p[2]);
      this.nor.push(nx, ny, nz);
      this.col.push(color[0], color[1], color[2]);
    }
  }

  quad(a: number[], b: number[], c: number[], d: number[], color: RGB): void {
    this.tri(a, b, c, color);
    this.tri(a, c, d, color);
  }

  /** axis-aligned box centered at (cx,cy,cz), full sizes (sx,sy,sz). */
  box(cx: number, cy: number, cz: number, sx: number, sy: number, sz: number, color: RGB): void {
    const x0 = cx - sx / 2, x1 = cx + sx / 2;
    const y0 = cy - sy / 2, y1 = cy + sy / 2;
    const z0 = cz - sz / 2, z1 = cz + sz / 2;
    this.quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], color); // +z
    this.quad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], color); // -z
    this.quad([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], color); // +x
    this.quad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], color); // -x
    this.quad([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], color); // +y
    this.quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], color); // -y
  }

  /** Y-axis cylinder/cone from y0..y1, radii rBottom→rTop (0 = cone tip). */
  cylinder(cy0: number, cy1: number, rBottom: number, rTop: number, color: RGB, segments = 14): void {
    for (let i = 0; i < segments; i++) {
      const a0 = (i / segments) * Math.PI * 2;
      const a1 = ((i + 1) / segments) * Math.PI * 2;
      const b0 = [Math.cos(a0) * rBottom, cy0, Math.sin(a0) * rBottom];
      const b1 = [Math.cos(a1) * rBottom, cy0, Math.sin(a1) * rBottom];
      const t0 = [Math.cos(a0) * rTop, cy1, Math.sin(a0) * rTop];
      const t1 = [Math.cos(a1) * rTop, cy1, Math.sin(a1) * rTop];
      if (rTop > 0) this.quad(b0, b1, t1, t0, color);
      else this.tri(b0, b1, [0, cy1, 0], color);
      if (rBottom > 0) this.tri(b1, b0, [0, cy0, 0], color); // bottom cap
      if (rTop > 0) this.tri(t0, t1, [0, cy1, 0], color);    // top cap
    }
  }

  build(): Mesh {
    return {
      positions: new Float32Array(this.pos),
      normals: new Float32Array(this.nor),
      colors: new Float32Array(this.col),
      vertexCount: this.pos.length / 3,
    };
  }
}

/** Build the representative mesh for a form. All forms fit ~[-1.2, 1.2]. */
export function buildFormMesh(kind: FormKind): Mesh {
  const m = new MeshBuilder();
  if (kind === 'cubesat') {
    m.box(0, 0, 0, 0.6, 0.6, 0.6, GOLD);                 // 1U-style body
    m.box(-0.85, 0, 0, 0.9, 0.5, 0.04, PANEL);           // deployed wings
    m.box(0.85, 0, 0, 0.9, 0.5, 0.04, PANEL);
    m.box(0, 0.42, 0, 0.03, 0.25, 0.03, SILVER);         // antenna
  } else if (kind === 'smallsat') {
    m.box(0, 0, 0, 0.7, 0.9, 0.7, GOLD);
    m.box(-1.0, 0, 0, 1.2, 0.55, 0.04, PANEL);
    m.box(1.0, 0, 0, 1.2, 0.55, 0.04, PANEL);
    m.cylinder(0.45, 0.75, 0.16, 0.16, SILVER, 10);      // payload dish stub
  } else if (kind === 'bus') {
    m.box(0, 0, 0, 0.8, 1.3, 0.8, GOLD);                 // big bus
    m.box(-1.35, 0, 0, 1.7, 0.7, 0.05, PANEL);           // large arrays
    m.box(1.35, 0, 0, 1.7, 0.7, 0.05, PANEL);
    m.cylinder(0.65, 1.0, 0.28, 0.28, SILVER, 12);       // antenna platform
  } else if (kind === 'starlink') {
    // documented design (public specs/renders): flat rectangular bus flying
    // "table-top", ONE long solar wing deployed upward from an edge hinge
    m.box(0, -0.55, 0, 1.05, 0.5, 0.07, DARK);            // flat bus panel (dark chassis)
    m.box(0, -0.55, 0.05, 0.95, 0.4, 0.02, SILVER);       // dish/antenna face
    m.box(0, -0.28, 0, 0.16, 0.06, 0.05, SILVER);         // wing hinge
    m.box(0, 0.475, 0, 0.6, 1.45, 0.03, PANEL);           // single solar wing, deployed up
    m.box(0, 0.475, 0.02, 0.05, 1.45, 0.015, SILVER);     // wing spine
  } else if (kind === 'rocket-body') {
    m.cylinder(-1.0, 0.8, 0.42, 0.42, SILVER, 14);       // stage
    m.cylinder(-1.35, -1.0, 0.28, 0.42, DARK, 14);       // nozzle skirt
    m.cylinder(0.8, 1.15, 0.42, 0.2, SILVER, 14);        // forward taper
  } else {
    // fragment: an irregular shard — deterministic vertex jitter (NEVER
    // random: rebuilds must be identical), clearly not a spacecraft.
    const v = [
      [0.9, 0.1, 0], [-0.6, 0.5, 0.35], [-0.4, -0.55, 0.3],
      [0.15, 0.05, -0.75], [-0.15, 0.7, -0.3], [0.3, -0.6, -0.25],
    ];
    m.tri(v[0], v[1], v[2], DARK); m.tri(v[0], v[2], v[5], SILVER);
    m.tri(v[0], v[5], v[3], DARK); m.tri(v[0], v[3], v[4], SILVER);
    m.tri(v[0], v[4], v[1], DARK); m.tri(v[1], v[4], v[2], SILVER);
    m.tri(v[2], v[4], v[3], DARK); m.tri(v[2], v[3], v[5], SILVER);
  }
  return m.build();
}

/** 3x3 rotation (column-major): yaw about Y then pitch about X. */
export function rotationMat3(yawRad: number, pitchRad: number): Float32Array {
  const cy = Math.cos(yawRad), sy = Math.sin(yawRad);
  const cx = Math.cos(pitchRad), sx = Math.sin(pitchRad);
  // Rx * Ry, column-major
  return new Float32Array([
    cy, sx * sy, -cx * sy,
    0, cx, sx,
    sy, -sx * cy, cx * cy,
  ]);
}
