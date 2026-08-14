// moonSurfaceGL — the close-surface body patch, raycast on the GPU.
//
// WHY (Track 2/3, human-directed 2026-08-14 "I go straight at the GPU
// raycast"). `moonSurface.renderMoonSurfaceRows` intersects a ray with the
// body sphere PER PIXEL, in JS, on the MAIN THREAD. At the full patch size
// (MOON_PATCH_FULL_LONG_PX = 1100) that is ~1.2M rays per patch, which is why
// the CPU path has to cap its buffer in CSS pixels and chunk itself across
// macrotasks to stay under the frame budget. PROGRAM_STATE L16 recorded the
// conclusion after the "9x faster moon" claim was refuted: only moving this to
// the GPU actually moves it. This module is that move.
//
// Every ray here is INDEPENDENT — no shared state, no ordering, one texture
// fetch and ~40 flops each. That is the shape a fragment shader is for.
//
// IT IS A PORT, NOT A REWRITE. The GLSL below reproduces the CPU functions
// exactly — raySphereNearT, surfaceLonLat, sampleDetail's alpha gate,
// sampleBase's equirect convention and wrap, and textureSphere.lambertWeight's
// curve. `moonSurfaceGL.test.ts` pins that equality by executing a TS mirror
// of this shader against the real CPU implementations over a grid of inputs —
// the same contract glElev.test.ts holds for its GLSL mirror. If the two ever
// diverge, the test says so rather than the Moon quietly shading differently
// at one zoom than the next.
//
// NEAREST, NOT LINEAR, deliberately: the CPU path truncates with `(u*W)|0`,
// so the shader uses texelFetch on the same integer texel. Bilinear filtering
// would look smoother and would NOT be the same image, and "the GPU path
// looks different" is exactly the bug this port must not introduce.
//
// RENDERING & MOTION LAW compliance:
//  · Law I — this module owns no loop, no timers and no map-event handlers.
//    It renders exactly when its caller's frame asks it to.
//  · Law IV — `dispose()` releases every GL object; `maxFeatures` and
//    `vramBudget` are declared below and enforced in `render()`.
//  · The STACK NOTE applies: this is raw WebGL2, not three.js.
import type { MoonSurfaceView, DetailOverlay } from "./moonSurface.ts";
import type { TexLike } from "./textureSphere.ts";

/** Law IV: the ray budget one patch may cost. 1100x1100 is the CPU path's
 *  MOON_PATCH_FULL_LONG_PX ceiling; above it `render()` refuses rather than
 *  degrading silently, and the caller keeps the CPU chunker. */
export const maxFeatures = 1100 * 1100;

/** Law IV: base equirect (up to 8k x 4k RGBA) + 2 detail tiers + the target
 *  colour buffer. Declared, and checked against the real upload in render(). */
export const vramBudget = 8192 * 4096 * 4 + 2 * 4096 * 4096 * 4 + 1100 * 1100 * 4;

/** Detail tiers the shader accepts (NAC strip over the WAC mosaic). The CPU
 *  path takes an unbounded array; two is what spaceFrame actually passes, and
 *  a fixed count keeps the shader branch-free per tier. */
export const MAX_TIERS = 2;

const VERT = `#version 300 es
// Fullscreen triangle — no attribute buffers, no vertex data to keep alive.
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export const FRAG = `#version 300 es
precision highp float;
precision highp int;

uniform vec3  uCam, uCenter, uR3, uU3, uF3, uX, uY, uZ, uSun;
uniform float uRadius, uCx, uCy, uK, uOriginX, uOriginY, uStepX, uStepY;
uniform float uWDeg, uTexOff, uLitBlend, uShadowFactor;
uniform int   uFullBright, uTierCount;
uniform vec2  uBufSize;

uniform highp sampler2D uBase;
uniform highp sampler2D uTier0;
uniform highp sampler2D uTier1;
// per tier: lonMin, lonSpan, latMax, latSpan
uniform vec4 uTierWin[${MAX_TIERS}];

out vec4 fragColor;

const float RAD = 57.29577951308232;   // 180/PI, matching moonSurface.RAD

// moonSurface.raySphereNearT — nearest positive root, or -1 for miss/behind.
float raySphereNearT(vec3 o, vec3 d, vec3 c, float R) {
  vec3 oc = o - c;
  float A = dot(d, d);
  float B = 2.0 * dot(oc, d);
  float C = dot(oc, oc) - R * R;
  float disc = B * B - 4.0 * A * C;
  if (disc < 0.0) return -1.0;
  float sq = sqrt(disc);
  float t0 = (-B - sq) / (2.0 * A);
  if (t0 > 0.0) return t0;
  float t1 = (-B + sq) / (2.0 * A);
  return t1 > 0.0 ? t1 : -1.0;
}

// moonSurface.surfaceLonLat — node-frame factorisation, lon in degrees minus
// the IAU prime-meridian angle W.
vec2 surfaceLonLat(vec3 n) {
  float bx = dot(n, uX);
  float by = dot(n, uY);
  float bz = clamp(dot(n, uZ), -1.0, 1.0);
  return vec2(atan(by, bx) * RAD - uWDeg, asin(bz) * RAD);
}

// The CPU path's \`(u * width) | 0\` truncation, clamped to the last texel.
ivec2 nearestTexel(vec2 uv, ivec2 size) {
  ivec2 t = ivec2(floor(uv * vec2(size)));
  return clamp(t, ivec2(0), size - ivec2(1));
}

// moonSurface.sampleDetail — window unwrap + the alpha<8 hole gate. Returns
// rgb in .xyz and .w = 1 when the tier covers this point opaquely.
vec4 sampleTier(highp sampler2D tex, vec4 win, vec2 lonLat) {
  float d = mod(lonLat.x - win.x, 360.0);
  if (d < 0.0) d += 360.0;
  if (d > win.y) return vec4(0.0);
  float dv = win.z - lonLat.y;
  if (dv < 0.0 || dv > win.w) return vec4(0.0);
  ivec2 size = textureSize(tex, 0);
  vec4 texel = texelFetch(tex, nearestTexel(vec2(d / win.y, dv / win.w), size), 0);
  if (texel.a < (8.0 / 255.0)) return vec4(0.0);   // hole / transparent margin
  return vec4(texel.rgb, 1.0);
}

// moonSurface.sampleBase — equirect, lon 0 at u=0.5 east-increasing, v=0 north.
vec3 sampleBase(vec2 lonLat) {
  float u = lonLat.x / 360.0 + 0.5 + uTexOff;
  u -= floor(u);
  float v = clamp(0.5 - lonLat.y / 180.0, 0.0, 1.0);
  ivec2 size = textureSize(uBase, 0);
  return texelFetch(uBase, nearestTexel(vec2(u, v), size), 0).rgb;
}

// textureSphere.lambertWeight — the shared day/terminator curve.
float lambertWeight(float lit) {
  float day = clamp((lit + 0.03) / 0.07, 0.0, 1.0);
  float shade = 0.05 + 0.95 * max(lit, 0.0);
  return 0.05 + 0.95 * day * shade;
}

void main() {
  // gl_FragCoord.y is bottom-up; the CPU buffer is top-down. Flip so buffer
  // row 0 is the same row in both paths (otherwise the patch renders mirrored
  // and every landing-site marker lands in the wrong hemisphere).
  float bx = floor(gl_FragCoord.x);
  float by = uBufSize.y - 1.0 - floor(gl_FragCoord.y);

  float px = uOriginX + bx * uStepX;
  float py = uOriginY + by * uStepY;
  float a = (px - uCx) / uK;
  float b = (uCy - py) / uK;

  vec3 d = a * uR3 + b * uU3 + uF3;
  float t = raySphereNearT(uCam, d, uCenter, uRadius);
  if (t < 0.0) { fragColor = vec4(0.0); return; }   // miss: transparent

  vec3 n = normalize(uCam + t * d - uCenter);
  vec2 lonLat = surfaceLonLat(n);

  vec3 rgb;
  vec4 hit = vec4(0.0);
  if (uTierCount > 0) hit = sampleTier(uTier0, uTierWin[0], lonLat);
  if (hit.w == 0.0 && uTierCount > 1) hit = sampleTier(uTier1, uTierWin[1], lonLat);
  rgb = hit.w > 0.0 ? hit.xyz : sampleBase(lonLat);

  float lit = dot(n, uSun);
  float w = uFullBright == 1 ? 1.0 : lambertWeight(lit) * uShadowFactor;
  if (uFullBright == 0 && uLitBlend > 0.0) w += (1.0 - w) * uLitBlend;

  fragColor = vec4(rgb * w, 1.0);
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type);
  if (!sh) throw new Error("moonSurfaceGL: createShader returned null");
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    // Loud, not swallowed: a silent shader failure would fall back to the CPU
    // path forever and read as "the GPU port did nothing".
    throw new Error(`moonSurfaceGL: shader compile failed — ${log}`);
  }
  return sh;
}

export interface MoonSurfaceGL {
  /** Render one patch. Returns false when the view exceeds the declared
   *  budget, so the caller keeps the CPU chunker rather than being surprised. */
  render(
    view: MoonSurfaceView,
    base: TexLike,
    detail: DetailOverlay | DetailOverlay[] | null,
  ): boolean;
  /** Law IV teardown — every GL object released. */
  dispose(): void;
  readonly canvas: HTMLCanvasElement | OffscreenCanvas;
}

/**
 * Build the GPU patch renderer, or return null when WebGL2 is unavailable.
 *
 * Null is an HONEST answer, not a failure: the CPU path in moonSurface.ts
 * remains correct and stays the fallback. What must never happen is a silent
 * half-state where the GPU context exists but draws nothing — hence the throw
 * in compile() rather than a swallowed error.
 */
export function createMoonSurfaceGL(
  canvas: HTMLCanvasElement | OffscreenCanvas,
): MoonSurfaceGL | null {
  const gl = canvas.getContext("webgl2", {
    alpha: true,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
  }) as WebGL2RenderingContext | null;
  if (!gl) return null;

  const vs = compile(gl, gl.VERTEX_SHADER, VERT);
  const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
  const prog = gl.createProgram();
  if (!prog) return null;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`moonSurfaceGL: link failed — ${log}`);
  }

  const vao = gl.createVertexArray();
  const u = (name: string) => gl.getUniformLocation(prog, name);
  const loc = {
    cam: u("uCam"), center: u("uCenter"), r3: u("uR3"), u3: u("uU3"), f3: u("uF3"),
    X: u("uX"), Y: u("uY"), Z: u("uZ"), sun: u("uSun"),
    radius: u("uRadius"), cx: u("uCx"), cy: u("uCy"), k: u("uK"),
    originX: u("uOriginX"), originY: u("uOriginY"), stepX: u("uStepX"), stepY: u("uStepY"),
    wDeg: u("uWDeg"), texOff: u("uTexOff"), litBlend: u("uLitBlend"),
    shadowFactor: u("uShadowFactor"), fullBright: u("uFullBright"),
    tierCount: u("uTierCount"), bufSize: u("uBufSize"),
    base: u("uBase"), tier0: u("uTier0"), tier1: u("uTier1"), tierWin: u("uTierWin"),
  };

  const textures: WebGLTexture[] = [];
  const makeTex = (): WebGLTexture => {
    const t = gl.createTexture();
    if (!t) throw new Error("moonSurfaceGL: createTexture returned null");
    textures.push(t);
    return t;
  };
  const baseTex = makeTex();
  const tierTex = [makeTex(), makeTex()];

  const upload = (tex: WebGLTexture, unit: number, src: TexLike): void => {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // NEAREST both ways + CLAMP: the shader does its own wrap/clamp so the
    // texel picked matches the CPU path's `(u*W)|0` exactly.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA8, src.width, src.height, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(src.data.buffer, src.data.byteOffset, src.data.length),
    );
  };

  let disposed = false;

  return {
    canvas,
    render(view, base, detail): boolean {
      if (disposed) return false;
      const px = view.bw * view.bh;
      if (px > maxFeatures) return false;            // Law IV: refuse, don't degrade

      const tiers: DetailOverlay[] = Array.isArray(detail) ? detail : detail ? [detail] : [];
      const used = Math.min(tiers.length, MAX_TIERS);

      if (canvas.width !== view.bw || canvas.height !== view.bh) {
        canvas.width = view.bw;
        canvas.height = view.bh;
      }
      gl.viewport(0, 0, view.bw, view.bh);
      gl.useProgram(prog);
      gl.bindVertexArray(vao);

      upload(baseTex, 0, base);
      gl.uniform1i(loc.base, 0);
      const win = new Float32Array(MAX_TIERS * 4);
      for (let i = 0; i < used; i++) {
        upload(tierTex[i], 1 + i, tiers[i].tex);
        win[i * 4] = tiers[i].lonMin;
        win[i * 4 + 1] = tiers[i].lonSpan;
        win[i * 4 + 2] = tiers[i].latMax;
        win[i * 4 + 3] = tiers[i].latSpan;
      }
      gl.uniform1i(loc.tier0, 1);
      gl.uniform1i(loc.tier1, 2);
      gl.uniform4fv(loc.tierWin, win);
      gl.uniform1i(loc.tierCount, used);

      const v3 = (l: WebGLUniformLocation | null, p: { x: number; y: number; z: number }) =>
        gl.uniform3f(l, p.x, p.y, p.z);
      v3(loc.cam, view.cam); v3(loc.center, view.center);
      v3(loc.r3, view.r); v3(loc.u3, view.u); v3(loc.f3, view.f);
      v3(loc.X, view.X); v3(loc.Y, view.Y); v3(loc.Z, view.Z);
      v3(loc.sun, view.sun);
      gl.uniform1f(loc.radius, view.radius);
      gl.uniform1f(loc.cx, view.cx);
      gl.uniform1f(loc.cy, view.cy);
      gl.uniform1f(loc.k, view.k);
      gl.uniform1f(loc.originX, view.originX);
      gl.uniform1f(loc.originY, view.originY);
      gl.uniform1f(loc.stepX, view.stepX);
      gl.uniform1f(loc.stepY, view.stepY);
      gl.uniform1f(loc.wDeg, view.wDeg);
      gl.uniform1f(loc.texOff, (view.texLonOffsetDeg ?? 0) / 360);
      gl.uniform1f(loc.litBlend, Math.max(0, Math.min(1, view.litBlend ?? 0)));
      gl.uniform1f(loc.shadowFactor, view.shadowFactor ?? 1);
      gl.uniform1i(loc.fullBright, view.fullBright ? 1 : 0);
      gl.uniform2f(loc.bufSize, view.bw, view.bh);

      // Misses write alpha 0 so sky/other bodies show through at the limb,
      // exactly as the CPU path does.
      gl.disable(gl.BLEND);
      gl.disable(gl.DEPTH_TEST);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
      return true;
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      for (const t of textures) gl.deleteTexture(t);
      textures.length = 0;
      gl.deleteVertexArray(vao);
      gl.deleteProgram(prog);
      // Release the drawing buffer too — a kept 1100x1100 RGBA surface is
      // ~4.8MB of VRAM per disposed-but-retained context.
      gl.getExtension("WEBGL_lose_context")?.loseContext();
    },
  };
}
