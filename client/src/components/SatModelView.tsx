// SatModelView — EARTH TWIN / ORBITAL O5 slice 2: the 3D rendering a clicked
// satellite resolves to (human directive: "zoom in on a satellite and see a
// 3D rendering of it"). Renders the class-REPRESENTATIVE form from
// lib/orbital/model3d in a small self-contained WebGL canvas inside the
// detail card — slowly tumbling, directionally lit. Zero dependencies (raw
// WebGL1, ~same footprint as the map's own custom layers); real
// photo-derived models for documented craft (ISS, constellation designs)
// are the chartered O5-3 upgrade and will replace the form ONLY where
// verifiable public assets exist.
//
// HONESTY: the caption ALWAYS states the form is derived from the catalog
// class and is not imagery of this unit (formLabel pins that in tests).
// DEGRADE: no WebGL → no canvas, caption only (never a broken box).

import { useEffect, useRef, useState } from "react";
import {
  classForm,
  formLabel,
  buildFormMesh,
  rotationMat3,
} from "@/lib/orbital/model3d";
import type { ObjectType, RcsSize } from "@/lib/orbital/tle";

const VERT = `
attribute vec3 a_pos;
attribute vec3 a_normal;
attribute vec3 a_color;
uniform mat3 u_rot;
uniform float u_aspect;
varying vec3 v_color;
varying vec3 v_normal;
void main() {
  vec3 p = u_rot * a_pos;
  vec3 n = u_rot * a_normal;
  float persp = 1.0 + p.z * 0.12;           // gentle perspective
  gl_Position = vec4(p.x * 0.55 / u_aspect, p.y * 0.55, p.z * 0.05, persp);
  v_color = a_color;
  v_normal = n;
}`;

const FRAG = `
precision mediump float;
varying vec3 v_color;
varying vec3 v_normal;
void main() {
  vec3 light = normalize(vec3(0.5, 0.7, 0.6));  // fixed key light
  float diff = max(dot(normalize(v_normal), light), 0.0);
  vec3 c = v_color * (0.35 + 0.75 * diff);
  gl_FragColor = vec4(c, 1.0);
}`;

export default function SatModelView({ objectType, rcsSize }: {
  objectType: ObjectType | null;
  rcsSize: RcsSize | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [glFailed, setGlFailed] = useState(false);
  const kind = classForm(objectType, rcsSize);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext("webgl", { antialias: true }) as WebGLRenderingContext | null;
    if (!gl) { setGlFailed(true); return; }

    const mkShader = (type: number, src: string): WebGLShader | null => {
      const sh = gl.createShader(type);
      if (!sh) return null;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      return gl.getShaderParameter(sh, gl.COMPILE_STATUS) ? sh : null;
    };
    const vs = mkShader(gl.VERTEX_SHADER, VERT);
    const fs = mkShader(gl.FRAGMENT_SHADER, FRAG);
    const prog = gl.createProgram();
    if (!vs || !fs || !prog) { setGlFailed(true); return; }
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { setGlFailed(true); return; }
    gl.useProgram(prog);

    const mesh = buildFormMesh(kind);
    const buf = (data: Float32Array, attr: string) => {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(prog, attr);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
      return b;
    };
    const buffers = [
      buf(mesh.positions, "a_pos"),
      buf(mesh.normals, "a_normal"),
      buf(mesh.colors, "a_color"),
    ];
    const uRot = gl.getUniformLocation(prog, "u_rot");
    const uAspect = gl.getUniformLocation(prog, "u_aspect");
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0.02, 0.04, 0.09, 1); // deep-space backdrop

    let raf = 0;
    const t0 = performance.now();
    const draw = (t: number) => {
      const yaw = ((t - t0) / 1000) * 0.5; // slow tumble
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.uniformMatrix3fv(uRot, false, rotationMat3(yaw, 0.4));
      gl.uniform1f(uAspect, canvas.width / canvas.height);
      gl.drawArrays(gl.TRIANGLES, 0, mesh.vertexCount);
      raf = requestAnimationFrame(draw); // rAF self-pauses in hidden tabs
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      try {
        for (const b of buffers) gl.deleteBuffer(b);
        gl.deleteProgram(prog);
        gl.deleteShader(vs);
        gl.deleteShader(fs);
      } catch { /* context may already be lost */ }
    };
  }, [kind]);

  return (
    <div data-vt-sat3d style={{ margin: "8px 0 4px" }}>
      {!glFailed && (
        <canvas
          ref={canvasRef}
          width={440}
          height={240}
          aria-label={`3D ${formLabel(kind)}`}
          style={{ width: "100%", height: "auto", borderRadius: 8, display: "block" }}
        />
      )}
      <p className="vt-site-card-trail" style={{ marginTop: 4 }}>{formLabel(kind)}</p>
    </div>
  );
}
